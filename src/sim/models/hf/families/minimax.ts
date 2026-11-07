/**
 * MiniMax M3 config → logical IR (transformers modeling_minimax_m3_vl.py):
 * - sparse_attention_config.sparse_attention_freq[i] → indexed GQA at layer i.
 * - sparse_{num_index_heads,index_dim,topk_blocks,block_size} → indexer dimensions / selected tokens.
 * - moe_layer_freq[i] → routed MoE, otherwise dense_intermediate_size (NOT intermediate_size).
 * - Native transformers equivalents: layer_types ('minimax_m3_sparse' / 'full_attention'),
 *   mlp_layer_types ('sparse' / 'dense'), and index_{n_heads,head_dim,topk_blocks,block_size}.
 * - intermediate_size / shared_intermediate_size → per-expert widths in elements.
 * The local block wins a slot INSIDE sparse_topk_blocks (source lines 574–584), not an extra block.
 */
import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import { modelSpecErrors } from '../../../ir/validateModel';
import { createReader } from '../reader';
import type { FamilyEntry } from '../types';

/** Family entry for MiniMax-M3 and config-compatible siblings such as M3.1-Flash. */
export const minimaxFamily: FamilyEntry = {
  family: 'minimax',
  modelTypes: ['minimax_m3_vl', 'minimax_m3_vl_text'],
  architectures: ['MiniMaxM3SparseForConditionalGeneration', 'MiniMaxM3SparseForCausalLM'],
  build(text, top, files) {
    const r = createReader(text, top === text ? '' : 'text_config');
    const t = createReader(top, '', r.errors);
    const hidden = r.int('hidden_size');
    const vocab = r.int('vocab_size');
    const n = r.int('num_hidden_layers');
    if (n === 0 || hidden === 0 || vocab === 0) r.errors.push('text_config: hidden_size, vocab_size and num_hidden_layers must be positive');
    const heads = r.int('num_attention_heads');
    const kvHeads = r.int('num_key_value_heads');
    const headDim = r.int('head_dim');
    // Native transformers serializes named schedules; legacy remote-code configs use binary arrays.
    const mlpTypes = r.optStrArray('mlp_layer_types');
    if (mlpTypes?.some(type => type !== 'dense' && type !== 'sparse')) r.errors.push('text_config.mlp_layer_types: expected dense or sparse');
    const moeSchedule = r.has('moe_layer_freq') ? r.numArray('moe_layer_freq')
      : mlpTypes?.map(type => type === 'sparse' ? 1 : 0) ?? r.numArray('moe_layer_freq');
    const dense = r.int('dense_intermediate_size');
    const experts = r.int('num_local_experts');
    const topK = r.int('num_experts_per_tok');
    const expertIntermediate = r.int('intermediate_size');
    const sharedExperts = r.optInt('n_shared_experts') ?? 0;
    const sharedIntermediate = r.optInt('shared_intermediate_size') ?? 0;
    const qkNorm = r.bool('use_qk_norm', true);
    const outputGate = r.bool('attention_output_gate', false);
    const tieEmbeddings = r.bool('tie_word_embeddings', false);
    const s = r.optObj('sparse_attention_config');
    const layerTypes = r.optStrArray('layer_types');
    if (layerTypes?.some(type => type !== 'full_attention' && type !== 'minimax_m3_sparse')) r.errors.push('text_config.layer_types: expected full_attention or minimax_m3_sparse');
    const sparseSchedule = s?.optNumArray('sparse_attention_freq')
      ?? layerTypes?.map(type => type === 'minimax_m3_sparse' ? 1 : 0) ?? Array<number>(n).fill(0);
    const sparseEnabled = s?.bool('use_sparse_attention', true) ?? layerTypes !== undefined;
    const indexer = s ? { heads: s.int('sparse_num_index_heads'), headDim: s.int('sparse_index_dim'),
      topK: s.int('sparse_topk_blocks') * s.int('sparse_block_size'), headWeights: false }
      : sparseSchedule.some(Boolean) ? { heads: r.int('index_n_heads'), headDim: r.int('index_head_dim'),
        topK: r.int('index_topk_blocks') * r.int('index_block_size'), headWeights: false } : undefined;
    for (const [key, schedule] of [['moe_layer_freq', moeSchedule], ['sparse_attention_freq', sparseSchedule]] as const) {
      if (schedule.length !== n || schedule.some(v => v !== 0 && v !== 1)) r.errors.push(`text_config.${key}: expected num_hidden_layers binary entries`);
    }
    // ViT + two-layer patch-merge projector; vector parameters are sub-percent and approximated.
    const v = t.optObj('vision_config');
    let vision = 0;
    if (v) {
      const d = v.int('hidden_size');
      const f = v.int('intermediate_size');
      const depth = v.int('num_hidden_layers');
      const merged = d * (t.optObj('img_token_compression_config')?.optInt('spatial_merge_size') ?? 2) ** 2;
      const projector = t.optInt('projector_hidden_size') ?? hidden;
      vision = depth * (4 * d * d + 2 * d * f + 6 * d + f)
        + d * v.int('num_channels') * v.int('patch_size') ** 2 + merged * projector + projector * hidden;
    }
    if (r.errors.length) return { kind: 'invalid_config', errors: r.errors };
    const model = decoderModel({ name: files.id, hidden, vocab, tieEmbeddings, source: files.id,
      layers: repeatLayers(n, i => decoderLayer({ name: `layer.${i}`, hidden,
        mixer: { kind: 'gqa', heads, kvHeads, headDim, qkNorm, outputGate,
          ...(sparseEnabled && sparseSchedule[i] && indexer ? { sparse: { kind: 'indexer', indexer } } : {}),
        },
        ffn: moeSchedule[i] ? { kind: 'moe', experts, topK, expertIntermediate, sharedExperts, sharedIntermediate }
          : { kind: 'dense', intermediate: dense },
      })),
    });
    const errors = modelSpecErrors(model);
    if (errors.length) return { kind: 'invalid_config', errors };
    return { kind: 'ok', model, mtpParamsEstimate: 0, visionParamsEstimate: vision,
      excludedParts: [...(v ? ['vision tower'] : []), ...(r.has('num_nextn_predict_layers') ? ['MTP (config only, no checkpoint weights)'] : [])],
      notes: [
        'Language decoder only: vision tower excluded. MTP config fields have no MTP tensors in the released checkpoint; no MTP parameters added.',
        'swigluoai is costed as silu; partial RoPE, routing bias and Gemma norm offset are negligible approximations.',
        'MSA local blocks occupy slots inside top-k; block max-pooling and per-KV-group selections are approximated by token top-k with the same selected-token budget.',
      ],
    };
  },
};
