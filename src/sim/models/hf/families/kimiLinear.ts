/**
 * Family `kimi_linear`: Kimi K3 (`KimiK3ForConditionalGeneration`) and Kimi K3.1-Pro (`KimiK31ForConditionalGeneration`),
 * both wrapping the language model `KimiLinearForCausalLM` / `kimi_linear` under `text_config`. Facts from the
 * shipped `modeling_kimi_linear.py` / `configuration_kimi_k3.py` and the safetensors headers; the vision tower
 * (MoonViT-V2) and its projector are left out and estimated for the param check.
 *
 * - Layer kinds: `linear_attn_config.kda_layers` → Kimi Delta Attention, `full_attn_layers` → MLA. Both lists are
 *   1-BASED (`configuration_kimi_k3.py:152-156`: `(layer_idx + 1) in kda_layers`), unlike GLM-5.3-Flash's 0-based
 *   ones; {@link zeroBasedLayerLists} tells the two apart.
 * - KDA (`modeling_kimi_linear.py:477-541`): `num_heads × head_dim` q/k/v with convs of `short_conv_kernel_size`,
 *   a rank-`head_dim` forget gate (`f_a_proj`, `f_b_proj`), `b_proj`, and the output gate either full rank
 *   (`use_full_rank_gate`: `g_proj h → H·d`) or low rank (`g_a_proj`, `g_b_proj`). Matches the IR's `'kda'`.
 * - MLA (`modeling_kimi_linear.py:335-475`): DeepSeek-V3 shapes (`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
 *   `qk_rope_head_dim`, `v_head_dim`) plus a sigmoid output gate `g_proj h → H·dv` (`mla_use_output_gate`).
 *   `mla_use_nope`: the `qk_rope_head_dim` slice is NOT rotated but still projected by `kv_a_proj_with_mqa` and
 *   cached (latent `kv_lora_rank + qk_rope_head_dim`), so it is kept as `qkRopeDim`; only the rope elementwise op is
 *   an over-count (§9).
 * - K3.1-Pro adds a DSA indexer on the layers of `dsa_config.dsa_layers` (1-based too): `index_num_heads`,
 *   `index_head_dim`, `index_topk`, LayerNorm keys, head weights; `index_rope_dim` 0 (no indexer rope op anyway).
 * - FFN (`modeling_kimi_linear.py:877-917`): dense `intermediate_size` for `layer_idx < first_k_dense_replace` or
 *   `layer_idx % moe_layer_freq != 0`, else MoE: `num_experts` × `moe_intermediate_size`, top `num_experts_per_token`,
 *   `num_shared_experts` fused into ONE shared MLP of `moe_intermediate_size · num_shared_experts`, and the latent
 *   form when `routed_expert_hidden_size` is set (`routed_expert_{down,up}_proj`, `routed_expert_norm` with
 *   `latent_moe_use_norm`). The router bias `e_score_correction_bias` (E params per MoE layer) is not counted.
 * - Attention Residuals (`attn_res_block_size`, `modeling_kimi_linear.py:975-1047, 1075-1088`): `residual_mix`
 *   blocks before each sublayer and before the final norm (`output_attn_res_*`).
 * - `hidden_act` `situ`; `num_nextn_predict_layers` 0 for both repos (a non-zero value is noted, not modeled).
 * Not counted: `A_log`, stored as [128] (the config's 96 heads give 96): 32 params per KDA layer.
 */

import type { FfnParams, MixerParams, ResidualParams } from '../../../ir/builders';
import { attnResCollapse, decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import type { IndexerSpec } from '../../../ir/model';
import { estimateVisionParams } from '../paramCheck';
import type { ConfigReader } from '../reader';
import { createReader } from '../reader';
import type { FamilyBuildResult, FamilyEntry, JsonObject } from '../types';
import { layerListOffset } from '../layerLists';
import {
  dsaSelection,
  readActivation,
  readIndexerSchedule,
  readMlaMixer,
  structuralErrors,
} from './mlaCommon';

/**
 * The 0-based KDA and MLA layer sets from `kda_layers` and `full_attn_layers`, and the offset the lists use. Kimi
 * writes them 1-based (the lists then name layer `n` and never layer 0); a 0-based pair (GLM-5.3-Flash) names 0 and
 * never `n` ({@link layerListOffset}, shared with the quant probes). Together they must cover every layer exactly
 * once, else an error is recorded.
 */
function zeroBasedLayerLists(
  lin: ConfigReader,
  n: number,
): { kda: Set<number>; full: Set<number>; offset: 0 | 1 } {
  const kdaRaw = lin.numArray('kda_layers');
  const fullRaw = lin.numArray('full_attn_layers');
  const union = [...kdaRaw, ...fullRaw];
  // Formula: offset = union contains n ? 1 (1-based) : 0 (0-based); layer = listed − offset
  const offset = layerListOffset([kdaRaw, fullRaw], n);
  const kda = new Set(kdaRaw.map((i) => i - offset));
  const full = new Set(fullRaw.map((i) => i - offset));
  const covered = Array.from(
    { length: n },
    (_, i) => (kda.has(i) ? 1 : 0) + (full.has(i) ? 1 : 0),
  ).every((c) => c === 1);
  if (!covered || union.length !== n) {
    lin.errors.push(
      `${lin.path}.kda_layers + full_attn_layers must list every layer 0..${n - 1} (or 1..${n}) exactly once`,
    );
  }
  return { kda, full, offset };
}

/**
 * Parameters of the MoonViT-V2 vision tower and `mm_projector` from Kimi's `vision_config` (`vt_*` keys, which the
 * generic estimate does not read), for the param check only; exact against the safetensors headers (K3: 447,358,976).
 * Falls back to {@link estimateVisionParams} for a `vision_config` without `vt_hidden_size`. Unit: params.
 */
export function estimateKimiVisionParams(
  vision: JsonObject | undefined,
  textHidden: number,
): number {
  if (vision === undefined) return 0;
  const v = createReader(vision, 'vision_config');
  if (!v.has('vt_hidden_size')) return estimateVisionParams(vision, textHidden);
  const d = v.int('vt_hidden_size');
  const L = v.int('vt_num_hidden_layers');
  const qkv = v.optInt('qkv_hidden_size') ?? d;
  const fi = v.optInt('vt_intermediate_size') ?? 4 * d;
  const ps = v.optInt('patch_size') ?? 14;
  const posH = v.optInt('init_pos_emb_height') ?? 64;
  const posW = v.optInt('init_pos_emb_width') ?? 64;
  const merge = v.optNumArray('merge_kernel_size') ?? [2, 2];
  const mm = v.optInt('mm_hidden_size') ?? d;
  const out = v.optInt('text_hidden_size') ?? textHidden;
  if (v.errors.length > 0) return 0;
  // Formula: P_block = d·3·qkv (wqkv) + qkv·d (wo) + 2·d·fi (fc0, fc1) + 2·d (norm0, norm1)
  // Formula: P_tower = L·P_block + d (final_layernorm) + posH·posW·d (pos_emb) + d·3·ps² (patch proj)
  // Formula: m = mm·Π merge;  P_projector = m² (proj.0) + out·m (proj.2) + out (post_norm)
  const block = d * 3 * qkv + qkv * d + 2 * d * fi + 2 * d;
  const tower = L * block + d + posH * posW * d + d * 3 * ps * ps;
  const m = mm * merge.reduce((p, k) => p * k, 1);
  const projector = m * m + out * m + out;
  return tower + projector;
}

/** Build the Kimi K3 / K3.1-Pro IR from its text config. */
export function buildKimiLinear(
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

  // Token mixers.
  const lin = r.obj('linear_attn_config');
  const kdaHeads = lin.int('num_heads');
  const kdaHeadDim = lin.int('head_dim');
  const fullRankGate = lin.bool('use_full_rank_gate', false);
  const kda: MixerParams = {
    kind: 'linear',
    variant: 'kda',
    qkHeads: kdaHeads,
    vHeads: kdaHeads,
    keyDim: kdaHeadDim,
    valueDim: kdaHeadDim,
    convKernel: lin.optInt('short_conv_kernel_size') ?? 4,
    outputGate: true,
    // The forget gate (f_a, f_b) is rank head_dim (`f_a_proj [128, h]`); the output gate too unless full rank.
    gateRank: kdaHeadDim,
    ...(fullRankGate ? { fullRankOutputGate: true } : {}),
  };
  const mla: MixerParams = {
    ...readMlaMixer(r),
    ...(r.bool('mla_use_output_gate', false) ? { outputGate: true } : {}),
  };
  const { kda: kdaLayers, full: fullLayers, offset: listOffset } = zeroBasedLayerLists(lin, n);

  // K3.1-Pro's DSA indexer on `dsa_layers`; `index_topk_freq` / `index_skip_topk_offset` give full vs reused top-k.
  const dsa = r.optObj('dsa_config');
  let indexer: IndexerSpec | undefined;
  let dsaLayers = new Set<number>();
  let indexerModes: ReturnType<typeof readIndexerSchedule> = [];
  if (dsa !== undefined) {
    indexer = {
      heads: dsa.int('index_num_heads'),
      headDim: dsa.int('index_head_dim'),
      topK: dsa.int('index_topk'),
    };
    const listed = dsa.numArray('dsa_layers');
    // Formula: same numbering as kda_layers / full_attn_layers; layer = listed − listOffset
    dsaLayers = new Set(listed.map((i) => i - listOffset));
    indexerModes = readIndexerSchedule(dsa, n);
    for (const i of dsaLayers)
      if (!fullLayers.has(i))
        dsa.errors.push(`${dsa.path}.dsa_layers: layer ${i} is not an MLA layer`);
  }

  // Feed-forward blocks.
  const dense: FfnParams = {
    kind: 'dense',
    intermediate: r.int('intermediate_size'),
    activation: act,
    gated: true,
  };
  const experts = r.optInt('num_experts');
  const firstDense = r.optInt('first_k_dense_replace') ?? 0;
  const moeFreq = Math.max(r.optInt('moe_layer_freq') ?? 1, 1);
  let moe: FfnParams | undefined;
  if (experts !== undefined && experts > 0) {
    const fe = r.int('moe_intermediate_size');
    const shared = r.optInt('num_shared_experts') ?? 0;
    const latentDim = r.optInt('routed_expert_hidden_size');
    moe = {
      kind: 'moe',
      experts,
      topK: r.int('num_experts_per_token'),
      expertIntermediate: fe,
      // Formula: f_shared = moe_intermediate_size · num_shared_experts (one fused shared MLP, `KimiMLP`)
      sharedExperts: shared > 0 ? 1 : 0,
      sharedIntermediate: shared > 0 ? fe * shared : 0,
      activation: act,
      gated: true,
      ...(latentDim !== undefined
        ? { latentDim, ...(r.bool('latent_moe_use_norm', false) ? { latentNorm: true } : {}) }
        : {}),
    };
  }
  // Formula: MoE iff num_experts set and i ≥ first_k_dense_replace and i % moe_layer_freq == 0 (modeling_kimi_linear.py:892-896)
  const isMoe = (i: number) => moe !== undefined && i >= firstDense && i % moeFreq === 0;

  const attnResBlock = r.optInt('attn_res_block_size');
  const mtpLayers = r.optInt('num_nextn_predict_layers') ?? 0;
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

  const residualAt = (i: number): ResidualParams =>
    attnResBlock !== undefined && attnResBlock > 0
      ? { kind: 'attn_res', blockSize: attnResBlock, layerIndex: i }
      : { kind: 'plain' };
  const mixerAt = (i: number): MixerParams => {
    if (kdaLayers.has(i)) return kda;
    if (indexer !== undefined && dsaLayers.has(i))
      return { ...mla, sparse: dsaSelection(indexerModes[i], indexer) };
    return mla;
  };
  const layers = repeatLayers(n, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden,
      mixer: mixerAt(i),
      ffn: isMoe(i) && moe !== undefined ? moe : dense,
      residual: residualAt(i),
    }),
  );
  const model = decoderModel({
    name: repoId,
    hidden,
    vocab,
    tieEmbeddings: tie,
    layers,
    ...(attnResBlock !== undefined && attnResBlock > 0
      ? { beforeHead: [attnResCollapse({ blockSize: attnResBlock, layers: n, dim: hidden })] }
      : {}),
    source: repoId,
  });
  const errors = structuralErrors(model);
  if (errors.length > 0) return { kind: 'invalid_config', errors };

  const vision = top.vision_config ?? text.vision_config;
  const visionParams = estimateKimiVisionParams(
    typeof vision === 'object' && vision !== null ? (vision as JsonObject) : undefined,
    hidden,
  );
  const moeCount = layers.filter((_, i) => isMoe(i)).length;
  const notes = [
    `hybrid: ${kdaLayers.size} KDA layers (${fullRankGate ? 'full-rank' : 'low-rank'} output gate), ${fullLayers.size} MLA layers` +
      (indexer !== undefined ? ` with a DSA indexer on ${dsaLayers.size}` : '') +
      (mla.kind === 'mla' && mla.outputGate === true ? ', sigmoid output gate' : ''),
    `FFN: ${n - moeCount} dense, ${moeCount} MoE` +
      (moe?.kind === 'moe' && moe.latentDim !== undefined
        ? ` (latent MoE: routed experts at ${moe.latentDim})`
        : ''),
    ...(attnResBlock !== undefined && attnResBlock > 0
      ? [`Attention Residuals: block size ${attnResBlock}`]
      : []),
    'mla_use_nope: the RoPE slice is cached but not rotated; the rope op is a small over-count',
    'router bias (e_score_correction_bias) and the A_log padding (32 per KDA layer) not counted',
    ...(mtpLayers > 0
      ? [`num_nextn_predict_layers ${mtpLayers}: MTP layers not modeled (no shipped code)`]
      : []),
  ];
  return {
    kind: 'ok',
    model,
    notes,
    mtpParamsEstimate: 0,
    visionParamsEstimate: visionParams,
    excludedParts: visionParams > 0 ? ['vision tower'] : [],
  };
}

/** Family entry for Kimi K3 / K3.1-Pro (KDA + MLA hybrid with latent MoE and Attention Residuals). */
export const kimiLinearFamily: FamilyEntry = {
  family: 'kimi_linear',
  modelTypes: ['kimi_linear', 'kimi_k3', 'kimi_k31'],
  architectures: [
    'KimiLinearForCausalLM',
    'KimiK3ForConditionalGeneration',
    'KimiK31ForConditionalGeneration',
  ],
  build: (text, top, files) => buildKimiLinear(text, top, files.id),
};
