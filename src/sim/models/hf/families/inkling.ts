/**
 * Inkling config → logical IR (transformers modeling_inkling.py):
 * - local_layer_ids → SWA with swa_* heads/dims and sliding_window_size tokens; others global.
 * - d_rel / rel_extent → input-dependent relative bias; RoPE is absent.
 * - use_sconv / sconv_kernel_size → K/V and sublayer-output depthwise convolutions.
 * - dense_mlp_idx → number of initial dense layers of dense_intermediate_size elements.
 * - n_routed_experts / num_experts_per_tok / n_shared_experts → MoE; intermediate_size is EACH expert's width.
 * Embedding and head are always untied (source _tied_weights_keys = {}), regardless of missing HF tie flags.
 */
import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import type { MixerParams } from '../../../ir/builders';
import { blockParams } from '../../../ir/params';
import { modelSpecErrors } from '../../../ir/validateModel';
import { createReader } from '../reader';
import { mtpDraft, withDraft } from '../drafter';
import type { FamilyEntry } from '../types';

/** Family entry for the Inkling language decoder, excluding vision/audio and MTP. */
export const inklingFamily: FamilyEntry = {
  family: 'inkling',
  modelTypes: ['inkling_mm_model', 'inkling_text'],
  architectures: ['InklingForConditionalGeneration'],
  build(text, top, files) {
    const r = createReader(text, top === text ? '' : 'text_config');
    const t = createReader(top, '', r.errors);
    const hidden = r.int('hidden_size');
    const vocab = r.int('vocab_size');
    const n = r.int('num_hidden_layers');
    if (n === 0 || hidden === 0 || vocab === 0) r.errors.push('text_config: hidden_size, vocab_size and num_hidden_layers must be positive');
    const local = r.numArray('local_layer_ids');
    const denseCount = r.int('dense_mlp_idx');
    const denseWidth = r.int('dense_intermediate_size');
    const expertWidth = r.int('intermediate_size');
    const experts = r.int('n_routed_experts');
    const topK = r.int('num_experts_per_tok');
    const shared = r.int('n_shared_experts');
    const conv = r.bool('use_sconv', false) ? r.int('sconv_kernel_size') : undefined;
    const relDim = r.int('d_rel');
    const window = r.int('sliding_window_size');
    const extent = r.int('rel_extent');
    const heads = r.int('num_attention_heads');
    const kvHeads = r.int('num_key_value_heads');
    const headDim = r.int('head_dim');
    const swaHeads = r.int('swa_num_attention_heads');
    const swaKvHeads = r.int('swa_num_key_value_heads');
    const swaHeadDim = r.int('swa_head_dim');
    const qBias = r.bool('q_bias', false);
    const outBias = r.bool('o_bias', false);
    const mtp = t.optObj('mtp_config');
    const mtpCount = mtp?.optInt('num_nextn_predict_layers') ?? 0;
    const mtpLocal = mtp?.optNumArray('local_layer_ids') ?? [];
    if (local.some(i => !Number.isInteger(i) || i < 0 || i >= n)) r.errors.push('text_config.local_layer_ids: expected zero-indexed decoder layer IDs');
    if (denseCount > n) r.errors.push('text_config.dense_mlp_idx: exceeds num_hidden_layers');
    // Coarse excluded-tower estimate: the real HMLP folds patches through varying projection widths.
    // No HMLP or audio computation enters the language decoder; this estimate affects only paramCheck.
    const v = t.optObj('vision_config');
    let vision = 0;
    if (v) {
      const patch = v.int('patch_size') ** 2 * v.int('temporal_patch_size') * v.int('n_channels');
      vision = patch * hidden + v.int('n_layers') * 2 * hidden * hidden;
    }
    const audio = t.optObj('audio_config');
    if (audio) vision += 2 * audio.int('n_mel_bins') * audio.int('mel_vocab_size') * hidden;
    if (r.errors.length) return { kind: 'invalid_config', errors: r.errors };
    const mixer = (isLocal: boolean): MixerParams => ({
      kind: 'gqa', heads: isLocal ? swaHeads : heads, kvHeads: isLocal ? swaKvHeads : kvHeads,
      headDim: isLocal ? swaHeadDim : headDim, qkNorm: true, rope: false,
      qkvBias: qBias, outBias,
      relativeBias: { relDim, extentTokens: isLocal ? window : extent },
      ...(isLocal ? { slidingWindow: window } : {}), ...(conv !== undefined ? { kvConvKernel: conv } : {}),
    });
    const shortConv = conv !== undefined ? { kernel: conv } : undefined;
    const layers = repeatLayers(n, i => decoderLayer({ name: `layer.${i}`, hidden, mixer: mixer(local.includes(i)), shortConv,
      ffn: i < denseCount ? { kind: 'dense', intermediate: denseWidth }
        : { kind: 'moe', experts, topK, expertIntermediate: expertWidth, sharedExperts: shared, sharedIntermediate: expertWidth },
    }));
    // Header-confirmed MTP uses DENSE MLPs in every draft layer, never the backbone's huge routed MoE.
    const mtpModule = (i: number) => decoderLayer({ hidden, mixer: mixer(mtpLocal.includes(i)), shortConv,
      ffn: { kind: 'dense', intermediate: denseWidth },
    });
    const mtpParamsEstimate = repeatLayers(mtpCount, mtpModule)
      .reduce((sum, layer) => sum + layer.blocks.reduce((p, b) => p + blockParams(b), 0) + 2 * hidden * hidden + 2 * hidden, 0);
    // The MTP modules are kept as model.draft: one module per draft position (DeepSeek-V3 style), costed only with
    // speculative decoding on.
    const draft = mtpDraft(mtpModule, mtpCount, true);
    const model = decoderModel({ name: files.id, hidden, vocab, tieEmbeddings: false, layers, source: files.id });
    const errors = modelSpecErrors(model);
    if (errors.length) return { kind: 'invalid_config', errors };
    return { kind: 'ok', model: withDraft(model, draft),
      mtpParamsEstimate, visionParamsEstimate: vision,
      excludedParts: [...(mtpCount ? ['MTP'] : []), ...(v ? ['vision tower'] : []), ...(audio ? ['audio tower'] : [])],
      notes: [
        'Language decoder only: MTP, vision and audio towers excluded; their parameters are estimated from config.',
        'The shared-expert sink gate and E+S router rows, embedding norm, log-scaled queries and logit softcap are negligible approximations.',
        'Checkpoint-derived parameter counts can differ from rounded model-card totals.',
      ],
    };
  },
};
