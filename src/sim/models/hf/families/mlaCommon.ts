/**
 * Config readers shared by the MLA / DSA families (DeepSeek-V4.1, GLM-5.3-Flash, GLM-5.3, Mistral Large 4):
 * the MLA mixer, the DSA indexer, the per-layer schedules (`indexer_types`, `mlp_layer_types`), the MoE and
 * dense FFNs, the mHC hyper-connection residual, and the MTP estimate.
 *
 * Traps these readers avoid on purpose:
 * - `head_dim` and `num_key_value_heads` are NOT MLA dims. GLM-5.3 has `head_dim` 192 and GLM-5.3-Flash 0;
 *   transformers overwrites `head_dim` with `qk_rope_head_dim`, and MLA has no separate KV heads (every head reads
 *   the shared latent). The mixer is read from `qk_nope_head_dim`, `qk_rope_head_dim` and `v_head_dim` only.
 * - A missing `q_lora_rank` (Mistral Large 4) means no q compression: `qLoraRank` 0.
 *
 * Every reader records path-qualified errors in the reader's shared list instead of throwing (see `reader.ts`).
 */

import type { FfnParams, MixerParams, ResidualParams } from '../../../ir/builders';
import { hyperCollapse } from '../../../ir/builders';
import type { HcPreBlock, IndexerSpec, LayerSpec, ModelSpec, SparseSelection } from '../../../ir/model';
import { blockParams } from '../../../ir/params';
import { modelSpecErrors } from '../../../ir/validateModel';
import type { ConfigReader } from '../reader';

/** The MLA mixer params of the builder API. */
export type MlaMixerParams = Extract<MixerParams, { kind: 'mla' }>;

/** Activation of an FFN, as the IR names it. */
export type FfnActivation = 'silu' | 'gelu' | 'relu';

/**
 * The FFN activation from `hidden_act` (default `'silu'` when absent). `silu`/`swish`/`swiglu` → silu, any
 * `gelu*` → gelu, `relu` → relu; anything else records an error. A `swiglu_limit` clamp is elementwise and
 * not modeled.
 */
export function readActivation(r: ConfigReader, key = 'hidden_act'): FfnActivation {
  const act = r.optStr(key);
  if (act === undefined || act === 'silu' || act === 'swish' || act === 'swiglu') return 'silu';
  if (act.startsWith('gelu')) return 'gelu';
  if (act === 'relu') return 'relu';
  r.errors.push(`${r.path ? `${r.path}.` : ''}${key}: unsupported activation '${act}'`);
  return 'silu';
}

/**
 * The MLA mixer of one layer (without `sparse` / `kvCacheFromLayer`, which the family adds):
 * `num_attention_heads`, `q_lora_rank` (absent or null → 0), `kv_lora_rank`, `qk_nope_head_dim`,
 * `qk_rope_head_dim`, `v_head_dim`.
 *
 * `useRope: false` makes a NoPE layer the way Mistral Large 4 does (`modeling_mistral_large4.py:342-345`): the
 * RoPE slice joins the non-RoPE part, so `qkNopeDim = qk_nope_head_dim + qk_rope_head_dim` and `qkRopeDim = 0`
 * (the q/k head dim stays the same; the cached latent loses its `dr` RoPE elements).
 */
export function readMlaMixer(r: ConfigReader, useRope = true): MlaMixerParams {
  const nope = r.int('qk_nope_head_dim');
  const rope = r.int('qk_rope_head_dim');
  return {
    kind: 'mla',
    heads: r.int('num_attention_heads'),
    qLoraRank: r.optInt('q_lora_rank') ?? 0,
    kvLoraRank: r.int('kv_lora_rank'),
    qkNopeDim: useRope ? nope : nope + rope,
    qkRopeDim: useRope ? rope : 0,
    vHeadDim: r.int('v_head_dim'),
  };
}

/**
 * The DSA indexer from `index_n_heads`, `index_head_dim`, `index_topk` (tokens the main attention reads per
 * query). GLM's k-pool: `index_kpool` p > 1 → `keyPoolTokens` p (one indexer key per p tokens), and
 * `index_kpool_compress` → `keyPoolGate` (the learned pooling gate and its APE). Keys come from the hidden state
 * and are owned by the layer; head weights (`weights_proj`) are on.
 */
export function readDsaIndexer(r: ConfigReader): IndexerSpec {
  const indexer: IndexerSpec = { heads: r.int('index_n_heads'), headDim: r.int('index_head_dim'), topK: r.int('index_topk') };
  const pool = r.optInt('index_kpool');
  if (pool !== undefined && pool > 1) {
    indexer.keyPoolTokens = pool;
    if (r.bool('index_kpool_compress', false)) indexer.keyPoolGate = true;
  }
  return indexer;
}

/** Per-layer indexer mode: `'full'` runs its own indexer, `'shared'` reuses the top-k of the last full layer. */
export type IndexerMode = 'full' | 'shared';

/**
 * Per-layer indexer modes for `n` layers, as transformers derives them (`configuration_glm_moe_dsa.py`
 * `__post_init__`): `indexer_types` when present; else `index_topk_pattern` (a string of `F`/`S` or a list);
 * else the schedule `full iff max(i − offset + 1, 0) mod freq == 0` with `freq = max(index_topk_freq ?? 1, 1)`
 * and `offset = index_skip_topk_offset ?? 2` (freq 1, the default, makes every layer full).
 */
export function readIndexerSchedule(r: ConfigReader, n: number): IndexerMode[] {
  const where = (key: string) => `${r.path ? `${r.path}.` : ''}${key}`;
  const explicit = r.optStrArray('indexer_types');
  const patternRaw = r.raw.index_topk_pattern;
  let modes: string[];
  if (explicit !== undefined) {
    modes = explicit;
  } else if (typeof patternRaw === 'string') {
    modes = [...patternRaw].map((c) => (c === 'F' ? 'full' : c === 'S' ? 'shared' : c));
  } else if (Array.isArray(patternRaw)) {
    modes = patternRaw.map(String);
  } else {
    const freq = Math.max(r.optInt('index_topk_freq') ?? 1, 1);
    const offset = r.optInt('index_skip_topk_offset') ?? 2;
    modes = Array.from({ length: n }, (_, i) => (Math.max(i - offset + 1, 0) % freq === 0 ? 'full' : 'shared'));
  }
  if (modes.length < n) r.errors.push(`${where('indexer_types')}: ${modes.length} entries for ${n} layers`);
  const out: IndexerMode[] = [];
  for (let i = 0; i < n; i++) {
    const m = modes[i] ?? 'full';
    if (m !== 'full' && m !== 'shared') r.errors.push(`${where('indexer_types')}[${i}]: unknown mode '${m}'`);
    out.push(m === 'shared' ? 'shared' : 'full');
  }
  return out;
}

/** Per-layer FFN kind: `'dense'` MLP or `'sparse'` MoE. */
export type MlpLayerType = 'dense' | 'sparse';

/**
 * Per-layer FFN kinds for `n` layers: `mlp_layer_types` when present, else the first `first_k_dense_replace`
 * layers dense (default `defaultDense` when that key is absent too) and the rest MoE.
 */
export function readMlpSchedule(r: ConfigReader, n: number, defaultDense: number): MlpLayerType[] {
  const where = `${r.path ? `${r.path}.` : ''}mlp_layer_types`;
  const explicit = r.optStrArray('mlp_layer_types');
  if (explicit === undefined) {
    const dense = Math.min(r.optInt('first_k_dense_replace') ?? defaultDense, n);
    return Array.from({ length: n }, (_, i) => (i < dense ? 'dense' : 'sparse'));
  }
  if (explicit.length < n) r.errors.push(`${where}: ${explicit.length} entries for ${n} layers`);
  const out: MlpLayerType[] = [];
  for (let i = 0; i < n; i++) {
    const t = explicit[i] ?? 'sparse';
    if (t !== 'dense' && t !== 'sparse') r.errors.push(`${where}[${i}]: unknown type '${t}'`);
    out.push(t === 'dense' ? 'dense' : 'sparse');
  }
  return out;
}

/**
 * The MoE FFN: `n_routed_experts` E, `num_experts_per_tok` k, `moe_intermediate_size` fe. The shared experts
 * (`n_shared_experts` S) are ONE fused gated MLP of width `fe · S` (transformers `shared_experts =
 * MLP(intermediate_size = moe_intermediate_size · n_shared_experts)`), so `sharedExperts` is 1 when S > 0.
 * The router bias (`e_score_correction_bias`, E params) is not modeled. `override.experts` / `override.topK` replace E and
 * k (DeepSeek's DSpark draft layers).
 */
export function readMoeFfn(
  r: ConfigReader,
  activation: FfnActivation,
  override: { experts?: number; topK?: number } = {},
): FfnParams {
  const fe = r.int('moe_intermediate_size');
  const shared = r.optInt('n_shared_experts') ?? 0;
  return {
    kind: 'moe',
    experts: override.experts ?? r.int('n_routed_experts'),
    topK: override.topK ?? r.int('num_experts_per_tok'),
    expertIntermediate: fe,
    sharedExperts: shared > 0 ? 1 : 0,
    // Formula: f_shared = moe_intermediate_size · n_shared_experts
    sharedIntermediate: shared > 0 ? fe * shared : 0,
    activation,
    gated: true,
  };
}

/** The dense gated FFN of the first layers: width `intermediate_size`. */
export function readDenseFfn(r: ConfigReader, activation: FfnActivation): FfnParams {
  return { kind: 'dense', intermediate: r.int('intermediate_size'), activation, gated: true };
}

/** The top-k selection of an MLA layer: its own indexer (`'full'`) or the reused top-k (`'shared'`). */
export function dsaSelection(mode: IndexerMode, indexer: IndexerSpec): SparseSelection {
  return mode === 'full' ? { kind: 'indexer', indexer } : { kind: 'reuse', topK: indexer.topK };
}

/**
 * The mHC (manifold-constrained hyper-connection) residual of DeepSeek-V4.1 and GLM-5.3-Flash, with N streams
 * (`hc_mult`) of width D. Each sublayer has its own `hc_pre`: an RMS statistic over N·D (no γ), ONE projection
 * `fn: N·D → (2+N)·N` (N pre weights, N post weights, the N×N comb matrix), `base` ((2+N)·N) and `scale` (3)
 * vectors, and `hc_sinkhorn_iters` Sinkhorn iterations on the comb matrix. Its `hc_post` mixes the streams with
 * comb; the sublayer keeps its own pre-norm (`subNorms`). Weights: `hc_attn_fn [24, 4·D]`, `hc_attn_base [24]`,
 * `hc_attn_scale [3]` and the same for `hc_ffn_*` (N = 4).
 */
export function mhcResidual(streams: number, hidden: number, sinkhornIters: number): ResidualParams {
  // Formula: mix = (2 + N)·N; vectorParams = mix (base) + 3 (scale)
  const mix = (2 + streams) * streams;
  return {
    kind: 'hyper',
    streams,
    pre: {
      normGamma: false,
      projections: [{ name: 'hc.fn', inputDim: streams * hidden, outputDim: mix }],
      sinkhornIters,
      vectorParams: mix + 3,
    },
    mixesStreams: true,
    subNorms: true,
  };
}

/**
 * The final mHC collapse before the LM head: N streams → one, with no projection and no params (DeepSeek
 * `hc_pre` with the last FFN's pre weights; GLM `hc_head`, a mean over the streams).
 */
export function mhcCollapse(streams: number, hidden: number): HcPreBlock {
  return hyperCollapse({ streams, dim: hidden, pre: { normGamma: false, projections: [], sinkhornIters: 0, vectorParams: 0 } });
}

/** All params of one layer (every block). Unit: params. */
export function layerParams(layer: LayerSpec): number {
  return layer.blocks.reduce((sum, b) => sum + blockParams(b), 0);
}

/**
 * Params of `count` DeepSeek-V3-style MTP (next-n) layers (`num_nextn_predict_layers`) for the param check: each
 * is a full decoder layer (`layer`, as the checkpoint has it at index `num_hidden_layers + j`) plus
 * `eh_proj` (2h → h) and the `enorm`, `hnorm`, `shared_head.norm` RMSNorms; embedding and head are shared with
 * the main model. Unit: params.
 */
export function mtpParamsEstimate(layer: LayerSpec, hidden: number, count: number): number {
  // Formula: P_mtp = count · (P_layer + 2·h·h (eh_proj) + 3·h (enorm, hnorm, shared_head.norm))
  return count * (layerParams(layer) + 2 * hidden * hidden + 3 * hidden);
}

/**
 * Structural errors of a built model (cross-layer references, residual widths; `ir/validateModel.ts`), each
 * prefixed so a config whose schedules point nowhere reads as an invalid config.
 */
export function structuralErrors(model: ModelSpec): string[] {
  return modelSpecErrors(model).map((e) => `model structure: ${e}`);
}
