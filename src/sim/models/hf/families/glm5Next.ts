/**
 * Family `glm5_next`: GLM-5.3-Flash (`Glm5NextForConditionalGeneration`; language model under `text_config`).
 *
 * A hybrid stack with mHC hyper-connections (`hc_mult` 4 residual streams; `modular_glm5_next.py:1141`):
 * - `layer_types[i]` 'linear_attention' → Kimi Delta Attention (KDA) from `linear_attn_config` {`num_heads`,
 *   `head_dim`, `short_conv_kernel_size`}, with low-rank forget and output gates of rank `head_dim`.
 * - 'deepseek_sparse_attention' / 'indexed_attention' / 'full_attention' → NoPE MLA (`qk_rope_head_dim` 0) with a
 *   DSA indexer whose keys are pooled over `index_kpool` tokens with a learned gate (`index_kpool_compress`);
 *   `indexer_types[i]` 'shared' reuses the last full layer's top-k.
 * - Absent `layer_types`: KDA unless `i % 4 == 3` (transformers default).
 * FFN: dense for `mlp_layer_types` 'dense' (first 3), else MoE with one shared expert. Each sublayer keeps its
 * pre-norm after `hc_pre` (`input_layernorm`, `post_attention_layernorm`); the streams are averaged (`hc_head`,
 * no params; `modular_glm5_next.py:1281`) before the final RMSNorm.
 *
 * The MTP layer (`num_nextn_predict_layers`; a full-indexer MLA + MoE layer without mHC params, plus `eh_proj` and
 * 3 norms) is kept as `model.draft` (single-step MTP, costed only with speculative decoding on) and estimated for the
 * param check. Excluded: the vision tower (generic estimate from `vision_config`).
 */

import type { MixerParams, ResidualParams } from '../../../ir/builders';
import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import { estimateVisionParams } from '../paramCheck';
import { createReader, isJsonObject } from '../reader';
import { mtpDraft, withDraft } from '../drafter';
import type { FamilyBuildResult, FamilyEntry, JsonObject } from '../types';
import {
  dsaSelection,
  mhcCollapse,
  mhcResidual,
  mtpParamsEstimate,
  readActivation,
  readDenseFfn,
  readDsaIndexer,
  readIndexerSchedule,
  readMlaMixer,
  readMlpSchedule,
  readMoeFfn,
  structuralErrors,
} from './mlaCommon';

/** `layer_types` values that mean an MLA + DSA layer. */
const ATTENTION_LAYER_TYPES = ['deepseek_sparse_attention', 'indexed_attention', 'full_attention'];

/** Build the GLM-5.3-Flash IR from its text config. */
export function buildGlm5Next(
  text: JsonObject,
  top: JsonObject,
  repoId: string,
): FamilyBuildResult {
  const r = createReader(text, text === top ? '' : 'text_config');
  const hidden = r.int('hidden_size');
  const n = r.int('num_hidden_layers');
  const vocab = r.int('vocab_size');
  const tie = r.bool('tie_word_embeddings', false);
  const act = readActivation(r);
  const mla = readMlaMixer(r);
  const indexer = readDsaIndexer(r);
  const indexerModes = readIndexerSchedule(r, n);
  const mlpTypes = readMlpSchedule(r, n, 3);
  const moe = readMoeFfn(r, act);
  const dense = readDenseFfn(r, act);
  const mtpLayers = r.optInt('num_nextn_predict_layers') ?? 0;

  const lin = r.obj('linear_attn_config');
  const kdaHeads = lin.int('num_heads');
  const kdaHeadDim = lin.int('head_dim');
  const kda: MixerParams = {
    kind: 'linear',
    variant: 'kda',
    qkHeads: kdaHeads,
    vHeads: kdaHeads,
    keyDim: kdaHeadDim,
    valueDim: kdaHeadDim,
    convKernel: lin.optInt('short_conv_kernel_size') ?? 4,
    outputGate: true,
    // The forget gate (f_a, f_b) and output gate (g_a, g_b) are rank head_dim (`f_a_proj [128, h]`).
    gateRank: kdaHeadDim,
  };

  const layerTypes =
    r.optStrArray('layer_types') ??
    Array.from({ length: n }, (_, i) => (i % 4 === 3 ? 'indexed_attention' : 'linear_attention'));
  const at = r.path ? `${r.path}.layer_types` : 'layer_types';
  if (layerTypes.length < n) r.errors.push(`${at}: ${layerTypes.length} entries for ${n} layers`);
  layerTypes.slice(0, n).forEach((t, i) => {
    if (t !== 'linear_attention' && !ATTENTION_LAYER_TYPES.includes(t))
      r.errors.push(`${at}[${i}]: unknown type '${t}'`);
  });

  const useMhc = r.bool('mhc', true);
  const streams = useMhc ? r.int('hc_mult') : 1;
  const sinkhorn = useMhc ? r.int('hc_sinkhorn_iters') : 0;
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

  const residual: ResidualParams = useMhc
    ? mhcResidual(streams, hidden, sinkhorn)
    : { kind: 'plain' };
  const isKda = (i: number) => layerTypes[i] === 'linear_attention';
  const layers = repeatLayers(n, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden,
      mixer: isKda(i) ? kda : { ...mla, sparse: dsaSelection(indexerModes[i], indexer) },
      ffn: mlpTypes[i] === 'dense' ? dense : moe,
      residual,
    }),
  );
  const model = decoderModel({
    name: repoId,
    hidden,
    vocab,
    tieEmbeddings: tie,
    layers,
    ...(useMhc ? { residualStreams: streams, beforeHead: [mhcCollapse(streams, hidden)] } : {}),
    source: repoId,
  });
  const errors = structuralErrors(model);
  if (errors.length > 0) return { kind: 'invalid_config', errors };

  // MTP layer `layers.<n>.*`: MLA with its own indexer + MoE, plain residual (no hc_* tensors).
  const mtpLayer = decoderLayer({
    hidden,
    mixer: { ...mla, sparse: dsaSelection('full', indexer) },
    ffn: moe,
  });
  const vision = isJsonObject(top.vision_config)
    ? top.vision_config
    : isJsonObject(text.vision_config)
      ? text.vision_config
      : undefined;
  const visionParams = estimateVisionParams(vision, hidden);

  const kdaCount = layers.filter((_, i) => isKda(i)).length;
  const notes = [
    `hybrid: ${kdaCount} KDA layers, ${n - kdaCount} NoPE MLA + DSA layers (k-pool ${indexer.keyPoolTokens ?? 1})`,
    useMhc
      ? `mHC residual: ${streams} streams, ${sinkhorn} Sinkhorn iterations`
      : 'plain residual (mhc false)',
    'router bias (e_score_correction_bias) not counted (E params per MoE layer)',
  ];
  const excludedParts: string[] = [];
  if (mtpLayers > 0) excludedParts.push('MTP layer');
  if (visionParams > 0) excludedParts.push('vision tower');
  return {
    kind: 'ok',
    model: withDraft(
      model,
      mtpDraft(() => mtpLayer, mtpLayers, true),
    ),
    notes,
    mtpParamsEstimate: mtpParamsEstimate(mtpLayer, hidden, mtpLayers),
    visionParamsEstimate: visionParams,
    excludedParts,
  };
}

/** Family entry for GLM-5.3-Flash (KDA + NoPE MLA/DSA hybrid with mHC). */
export const glm5NextFamily: FamilyEntry = {
  family: 'glm5_next',
  modelTypes: ['glm5_next', 'glm5_next_text'],
  architectures: ['Glm5NextForConditionalGeneration'],
  build: (text, top, files) => buildGlm5Next(text, top, files.id),
};
