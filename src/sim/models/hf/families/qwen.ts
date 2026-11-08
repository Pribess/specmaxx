/**
 * Qwen hybrid text decoders (transformers modeling_qwen3_5.py / modeling_qwen4_exp.py).
 * Config → logical IR (dimensions in elements, counts unsharded):
 * - layer_types / full_attention_interval → GatedDeltaNet versus gated GQA.
 * - linear_* heads/dims/kernel → recurrent mixer; indexer_* → pooled QSA.
 * - num_experts / *_intermediate_size → routed and shared SwiGLU experts.
 * - hc_count / hc_lowrank → parallel residual streams and low-rank mixers.
 * - ple_layer_ids (ONE-indexed), ngram_* / ple_* → hashed lookup and projections.
 * Vision is excluded and estimated for checkpoint comparison only. The MTP module (`mtp_num_hidden_layers`, trained
 * for several draft positions) is kept as `model.draft`, costed only with speculative decoding on, and estimated.
 */
import type { Dtype } from '../../../ir/dtype';
import { decoderLayer, decoderModel, hyperCollapse, repeatLayers } from '../../../ir/builders';
import type { FfnParams, MixerParams, ResidualParams } from '../../../ir/builders';
import { blockParams } from '../../../ir/params';
import { modelSpecErrors } from '../../../ir/validateModel';
import { createReader } from '../reader';
import type { ConfigReader } from '../reader';
import { mtpDraft, withDraft } from '../drafter';
import type { FamilyEntry } from '../types';

/** Approximate Qwen ViT + patch merger parameters from its config; unit: logical parameters. */
function visionParams(v: ConfigReader | undefined, hidden: number): number {
  if (!v) return 0;
  const d = v.int('hidden_size');
  const f = v.int('intermediate_size');
  const depth = v.int('depth');
  const merged = d * v.int('spatial_merge_size') ** 2;
  return depth * (4 * d * d + 2 * d * f + 6 * d + f)
    + d * v.int('in_channels') * v.int('temporal_patch_size') * v.int('patch_size') ** 2
    + d * v.int('num_position_embeddings') + merged * merged + merged * hidden;
}

/** Family entry for config-driven Qwen3.5/3.8 hybrids and Qwen4Exp sparse MoE hybrids. */
export const qwenFamily: FamilyEntry = {
  family: 'qwen',
  modelTypes: ['qwen3_5', 'qwen3_5_text', 'qwen4_exp', 'qwen4_exp_text'],
  architectures: ['Qwen3_5ForConditionalGeneration', 'Qwen4ExpForConditionalGeneration'],
  build(text, top, files) {
    const r = createReader(text, top === text ? '' : 'text_config');
    const t = createReader(top, '', r.errors);
    const hidden = r.int('hidden_size');
    const vocab = r.int('vocab_size');
    const n = r.int('num_hidden_layers');
    if (n === 0 || hidden === 0 || vocab === 0) r.errors.push('text_config: hidden_size, vocab_size and num_hidden_layers must be positive');
    const streams = r.optInt('hc_count') ?? 1;
    const rank = streams > 1 ? r.int('hc_lowrank') : 0;
    const width = streams * hidden;
    const projections = [
      { name: 'down', inputDim: width, outputDim: rank },
      { name: 'up', inputDim: rank, outputDim: width },
    ];
    const pre = { normGamma: true, projections, sinkhornIters: 0, vectorParams: 0 };
    const residual: ResidualParams = streams > 1 ? {
      kind: 'hyper', streams, mixesStreams: false, subNorms: false,
      pre: { ...pre, projections: [...projections, { name: 'inject', inputDim: width, outputDim: streams }] },
    } : { kind: 'plain' };
    const ffn: FfnParams = r.has('num_experts') ? {
      kind: 'moe', experts: r.int('num_experts'), topK: r.int('num_experts_per_tok'),
      expertIntermediate: r.int('moe_intermediate_size'),
      sharedExperts: r.has('shared_expert_intermediate_size') ? 1 : 0,
      sharedIntermediate: r.optInt('shared_expert_intermediate_size') ?? 0,
    } : { kind: 'dense', intermediate: r.int('intermediate_size') };
    const schedule = r.optStrArray('layer_types');
    const interval = r.optInt('full_attention_interval') ?? 4;
    if (schedule && schedule.length !== n) r.errors.push('text_config.layer_types: length must equal num_hidden_layers');
    if (!schedule && interval === 0) r.errors.push('text_config.full_attention_interval: must be positive');
    const full: MixerParams = {
      kind: 'gqa', heads: r.int('num_attention_heads'), kvHeads: r.int('num_key_value_heads'),
      headDim: r.int('head_dim'), qkNorm: true, qkvBias: r.bool('attention_bias', false),
      // qwen4_exp always gates Q; presence of hc_count identifies its residual convention, not a repo name.
      outputGate: r.bool('attn_output_gate', streams > 1),
      ...(r.has('indexer_budget') ? { sparse: { kind: 'indexer' as const, indexer: {
        heads: r.int('indexer_n_heads'), headDim: r.int('indexer_head_dim'),
        topK: r.int('indexer_budget'), keyPoolTokens: r.int('indexer_compress_ratio'), headWeights: false,
      } } } : {}),
    };
    const hasLinear = !schedule || schedule.includes('linear_attention');
    const linear: MixerParams | undefined = hasLinear ? {
      kind: 'linear', variant: 'gated_delta_net', qkHeads: r.int('linear_num_key_heads'),
      vHeads: r.int('linear_num_value_heads'), keyDim: r.int('linear_key_head_dim'),
      valueDim: r.int('linear_value_head_dim'), convKernel: r.int('linear_conv_kernel_dim'), outputGate: true,
    } : undefined;
    const pleLayers = r.optNumArray('ple_layer_ids') ?? [];
    const ngramSize = pleLayers.length ? r.int('ngram_size') : 0;
    const rowsPerToken = pleLayers.length ? (ngramSize - 1) * r.int('heads_per_ngram') : 0;
    const embedDim = pleLayers.length ? r.int('ple_embed_dim') : 0;
    const multiple = pleLayers.length ? r.int('make_ngram_vocab_size_divisible_by') : 1;
    const rows = pleLayers.length ? Math.ceil(rowsPerToken * r.int('ngram_vocab_size_base') / multiple) * multiple : 0;
    const convKernel = pleLayers.length ? r.int('ple_conv_kernel_size') : 0;
    if (pleLayers.length && (rowsPerToken <= 0 || multiple <= 0 || embedDim % rowsPerToken !== 0)) r.errors.push('text_config: PLE needs positive row counts and an integral row dimension');
    if (pleLayers.some(id => !Number.isInteger(id) || id < 1 || id > n)) r.errors.push('text_config.ple_layer_ids: expected one-indexed decoder layer IDs');
    if (schedule?.some(s => !['linear_attention', 'full_attention', 'indexed_attention'].includes(s))) r.errors.push('text_config.layer_types: unsupported layer type');
    const mtpLayers = r.optInt('mtp_num_hidden_layers') ?? 0;
    const vision = visionParams(t.optObj('vision_config'), hidden);
    const tieEmbeddings = r.bool('tie_word_embeddings', t.bool('tie_word_embeddings', false));
    const state = r.optStr('mamba_ssm_dtype');
    const recurrentState: Dtype | undefined = state === 'float32' ? 'fp32' : state === 'float16' ? 'fp16' : state === 'bfloat16' ? 'bf16' : undefined;
    if (r.errors.length) return { kind: 'invalid_config', errors: r.errors };
    const layers = repeatLayers(n, i => decoderLayer({
      name: `layer.${i}`, hidden, residual, ffn,
      mixer: (schedule ? schedule[i] === 'linear_attention' : (i + 1) % interval !== 0) ? linear! : full,
      ...(pleLayers.includes(i + 1) ? { ngram: {
        tableRows: rows, rowDim: embedDim / rowsPerToken, rowsPerToken,
        projections: [{ name: 'key', inputDim: embedDim, outputDim: width }, { name: 'value', inputDim: embedDim, outputDim: hidden }],
        outputDim: width, conv: { kernel: convKernel, dilation: ngramSize }, vectorParams: 3 * width,
        // The model card keeps the PLE table in host RAM; this drives the UI hint only (the simulator's default
        // placement stays the device, design doc §2.10).
        offDeviceByDesign: true,
      } } : {}),
    }));
    // MTP has an attention layer plus a 2h→h fusion and two input norms; no duplicate embedding/head.
    const mtpLayer = decoderLayer({ hidden, mixer: full, ffn, residual });
    const mtpParamsEstimate = mtpLayers * (mtpLayer.blocks.reduce((sum, b) => sum + blockParams(b), 0) + 2 * hidden * hidden + 2 * hidden);
    const model = decoderModel({ name: files.id, hidden, vocab, tieEmbeddings, layers,
      residualStreams: streams, finalNorm: streams > 1 ? 'none' : 'rmsnorm', source: files.id,
      ...(streams > 1 ? { beforeHead: [hyperCollapse({ streams, dim: hidden, pre })] } : {}),
    });
    const errors = modelSpecErrors(model);
    if (errors.length) return { kind: 'invalid_config', errors };
    return {
      kind: 'ok', model: withDraft(model, mtpDraft(() => mtpLayer, mtpLayers, false)), mtpParamsEstimate, visionParamsEstimate: vision,
      excludedParts: [...(mtpLayers ? ['MTP'] : []), ...(vision ? ['vision tower'] : [])],
      notes: [
        'Language decoder only: MTP and vision tower excluded; their parameters are estimated from config.',
        'Partial RoPE and output-gate activation variants are approximated; shared-expert sigmoid gates are omitted.',
        ...(pleLayers.length ? ['PLE prime-sized hash slices approximated by padded ngram_vocab_size_base; QSA uses pooled indexer keys.'] : []),
      ],
      ...(recurrentState ? { quantHints: { recurrentState } } : {}),
    };
  },
};
