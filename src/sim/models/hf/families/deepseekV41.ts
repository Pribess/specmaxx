/**
 * Family `deepseek_v41`: DeepSeek-V4.1 (`DeepseekV41ForCausalLM`; language model under `text_config`). Not in
 * transformers; the reference is the repo's `inference/model.py` (`Block`, `Attention`, `Compressor`, `Indexer`,
 * `Engram`, `Transformer.forward`).
 *
 * Every layer (`num_hidden_layers` 40): [Engram before layers `engram_layer_ids`], then two mHC sublayers
 * (`hc_mult` 4 streams; each with its own `hc_*_fn [(2+N)·N, N·D]`, base, scale, Sinkhorn, and its own pre-norm):
 * compressed sparse attention (CSA) and a MoE (all layers; one shared expert of width `moe_intermediate_size`).
 * After the last layer the streams collapse without a projection (`hc_pre` with the last FFN's pre weights),
 * then the final RMSNorm and the head.
 *
 * CSA layer roles come from the config lists, entry i of each (the first `num_hidden_layers` entries of
 * `compress_ratios`; the rest belong to the MTP stages):
 * - `compress_ratios[i]` r = 0 → sliding window only (no global branch).
 * - i ∈ `kv_source_layer_ids` → owns the compressed global KV (runs the compressor); otherwise it reads the cache
 *   of the latest source layer before i.
 * - i ∈ `index_source_layer_ids` → runs its own indexer (keys pooled r tokens per key, projected from the
 *   `head_dim`-wide compressed latent); it owns indexer keys iff it is also a KV source ("Full"), else it reads
 *   the source's keys ("Reindex"). Indexers after `candidate_source_layer_id` score at most
 *   `candidate_topk_blocks · candidate_block_size` candidates. Any other r > 0 layer reuses the top-k ("Reuse").
 * DeepSeek-V4.1-Flash: 2 SWA-only, 4 Full (2, 8, 14, 20), 4 Reindex (24, 28, 32, 36), 30 Reuse, Engram at 1, 14.
 * Prefill runs every layer on every token, as `model.py` does (no "Bounded Replay").
 *
 * Excluded: the DSpark MTP stages (`num_nextn_predict_layers`; estimated from the `dspark_*` fields) and the vision
 * tower (estimated from `vision_config`). Approximations: the attention sink (H per layer), the router biases
 * (`gate.bias`, `gate.bias_vl`: 2·E per layer) are not counted, and the indexer key norm is counted as a
 * LayerNorm (2·hd_I instead of hd_I); together 960 params per layer at most.
 */

import type { FfnParams, MixerParams } from '../../../ir/builders';
import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import type { CsaGlobalSpec, IndexerSpec, LayerSpec } from '../../../ir/model';
import type { ConfigReader } from '../reader';
import { createReader, isJsonObject } from '../reader';
import type { FamilyBuildResult, FamilyEntry, JsonObject } from '../types';
import { layerParams, mhcCollapse, mhcResidual, readActivation, readMoeFfn, structuralErrors } from './mlaCommon';

/** The CSA fields every layer shares (heads, dims, ranks, window). */
type CsaBase = Omit<Extract<MixerParams, { kind: 'csa' }>, 'global'>;

/** Role of a CSA layer, for notes and tests. */
export type CsaRole = 'swa' | 'full' | 'reindex' | 'reuse';

/** The CSA role of layer i (see the module header). */
export function csaRole(ratio: number, isKvSource: boolean, isIndexSource: boolean): CsaRole {
  if (ratio === 0) return 'swa';
  if (isIndexSource) return isKvSource ? 'full' : 'reindex';
  return 'reuse';
}

/** Everything the layer builder reads from the config. */
interface DeepseekConfig {
  hidden: number;
  csa: CsaBase;
  ratios: number[];
  kvSources: number[];
  indexSources: number[];
  candidateSource: number | undefined;
  maxScoredKeys: number | undefined;
  indexer: { heads: number; headDim: number; topK: number };
  streams: number;
  sinkhornIters: number;
  engramLayers: number[];
  engramRows: number[];
  engramRowDim: number;
  engramRowsPerToken: number;
}

/**
 * The global branch of layer i, or undefined for an SWA-only layer. Errors (a consumer with no earlier KV source)
 * go to `errors`.
 */
function csaGlobal(c: DeepseekConfig, i: number, errors: string[]): CsaGlobalSpec | undefined {
  const r = c.ratios[i];
  if (r === 0) return undefined;
  const ownsKv = c.kvSources.includes(i);
  const earlierSources = c.kvSources.filter((s) => s < i);
  // Formula: kvCacheFromLayer = max{ s ∈ kv_source_layer_ids : s < i } for a non-source layer
  const kvFrom = ownsKv || earlierSources.length === 0 ? undefined : Math.max(...earlierSources);
  if (!ownsKv && kvFrom === undefined) errors.push(`kv_source_layer_ids: layer ${i} has compress ratio ${r} but no earlier KV source`);
  let selection: CsaGlobalSpec['selection'];
  if (c.indexSources.includes(i)) {
    const indexer: IndexerSpec = {
      ...c.indexer,
      keyPoolTokens: r,
      // wk reads the compressed latent (`indexer.wk [index_head_dim, head_dim]`).
      keyInputDim: c.csa.headDim,
      ownsKeys: ownsKv,
      ...(c.candidateSource !== undefined && c.maxScoredKeys !== undefined && i > c.candidateSource ? { maxScoredKeys: c.maxScoredKeys } : {}),
    };
    selection = { kind: 'indexer', indexer };
  } else {
    selection = { kind: 'reuse', topK: c.indexer.topK };
  }
  return { compressRatio: r, ...(kvFrom !== undefined ? { kvCacheFromLayer: kvFrom } : {}), selection };
}

/** One decoder layer: [Engram], mHC + CSA, mHC + `ffn`. */
function deepseekLayer(c: DeepseekConfig, i: number, ratio: number, global: CsaGlobalSpec | undefined, ffn: FfnParams, name: string): LayerSpec {
  const N = c.streams;
  const D = c.hidden;
  const engramIndex = c.engramLayers.indexOf(i);
  const mixer: MixerParams = { ...c.csa, ...(ratio > 0 && global !== undefined ? { global } : {}) };
  return decoderLayer({
    name,
    hidden: D,
    mixer,
    ffn,
    residual: mhcResidual(N, D, c.sinkhornIters),
    ...(engramIndex >= 0
      ? {
          ngram: {
            name: 'engram',
            tableRows: c.engramRows[engramIndex],
            rowDim: c.engramRowDim,
            rowsPerToken: c.engramRowsPerToken,
            // Formula: wkv: rowsPerToken·rowDim → (N + 1)·D (one key per stream + a shared value)
            projections: [{ name: 'engram.wkv', inputDim: c.engramRowsPerToken * c.engramRowDim, outputDim: (N + 1) * D }],
            outputDim: N * D,
            // Formula: q_weight, k_weight: 2·N·D
            vectorParams: 2 * N * D,
          },
        }
      : {}),
  });
}

/** Read the config fields the layers need. */
function readDeepseekConfig(r: ConfigReader): DeepseekConfig {
  const hidden = r.int('hidden_size');
  const csa: CsaBase = {
    kind: 'csa',
    heads: r.int('num_attention_heads'),
    headDim: r.int('head_dim'),
    ropeDim: r.int('qk_rope_head_dim'),
    qLoraRank: r.int('q_lora_rank'),
    outGroups: r.int('o_groups'),
    outLoraRank: r.int('o_lora_rank'),
    windowTokens: r.int('sliding_window'),
  };
  const candidateBlocks = r.optInt('candidate_topk_blocks');
  const candidateBlockSize = r.optInt('candidate_block_size');
  const engramLayers = r.optNumArray('engram_layer_ids') ?? [];
  return {
    hidden,
    csa,
    ratios: r.numArray('compress_ratios'),
    kvSources: r.optNumArray('kv_source_layer_ids') ?? [],
    indexSources: r.optNumArray('index_source_layer_ids') ?? [],
    candidateSource: r.optInt('candidate_source_layer_id'),
    // Formula: maxScoredKeys = candidate_topk_blocks · candidate_block_size
    maxScoredKeys: candidateBlocks !== undefined && candidateBlockSize !== undefined ? candidateBlocks * candidateBlockSize : undefined,
    indexer: { heads: r.int('index_n_heads'), headDim: r.int('index_head_dim'), topK: r.int('index_topk') },
    streams: r.int('hc_mult'),
    sinkhornIters: r.optInt('hc_sinkhorn_iters') ?? 0,
    engramLayers,
    engramRows: engramLayers.length > 0 ? r.numArray('engram_num_embeddings') : [],
    engramRowDim: engramLayers.length > 0 ? r.int('engram_head_dim') : 0,
    // Formula: rowsPerToken = (engram_max_ngram_size − 1) · engram_n_heads
    engramRowsPerToken: engramLayers.length > 0 ? (r.int('engram_max_ngram_size') - 1) * r.int('engram_n_heads') : 0,
  };
}

/**
 * Params of the DSpark MTP stages (`mtp.*`), for the param check: `num_nextn_predict_layers` decoder layers
 * (CSA with ratio `compress_ratios[n + j]`, MoE with `dspark_n_routed_experts` / `dspark_num_experts_per_tok`
 * when given, mHC) + the first stage's `main_proj` (targets·h → h) and `main_norm` + the last stage's `norm`,
 * Markov head (embed and head, vocab × `dspark_markov_rank`) and confidence projection (h + rank → 1).
 * Embedding and head are shared with the main model. Unit: params.
 */
function dsparkParamsEstimate(r: ConfigReader, c: DeepseekConfig, moe: FfnParams, n: number, vocab: number): number {
  const stages = r.optInt('num_nextn_predict_layers') ?? 0;
  if (stages === 0) return 0;
  const experts = r.optInt('dspark_n_routed_experts');
  const topK = r.optInt('dspark_num_experts_per_tok');
  const draftMoe: FfnParams = moe.kind === 'moe' ? { ...moe, ...(experts ? { experts } : {}), ...(topK ? { topK } : {}) } : moe;
  let total = 0;
  for (let j = 0; j < stages; j++) {
    const ratio = c.ratios[n + j] ?? 0;
    // DSpark attention is window-only (`assert self.compress_ratio == 0`); a non-zero ratio is ignored.
    total += layerParams(deepseekLayer({ ...c, engramLayers: [] }, n + j, ratio, undefined, draftMoe, `mtp.${j}`));
  }
  const h = c.hidden;
  const targets = (r.optNumArray('dspark_target_layer_ids') ?? []).length;
  const rank = r.optInt('dspark_markov_rank') ?? 0;
  // Formula: P = Σ_j P_stage + targets·h·h + h (main_proj, main_norm) + h + 2·vocab·rank + (h + rank) (norm, Markov, confidence)
  return total + targets * h * h + h + h + 2 * vocab * rank + (h + rank);
}

/**
 * Vision tower + aligner of DeepSeek-V4.1 from `vision_config`, for the param check: `num_hidden_layers` blocks
 * of `4·d²` (wqkv, wo) + `3·d·ffn` (gated MLP; `w1` holds gate and up) and the aligner `w1: d·s² → h`,
 * `w2: h → h` with s = `downsample_ratio`. Biases, norms and patch embeddings are left out. Unit: params.
 */
function visionParamsEstimate(vision: JsonObject | undefined, hidden: number): number {
  if (vision === undefined) return 0;
  const num = (k: string) => (typeof vision[k] === 'number' ? vision[k] : 0);
  const depth = num('num_hidden_layers');
  const d = num('hidden_size');
  const ffn = num('intermediate_size');
  const s = num('downsample_ratio') || 1;
  // Formula: P_vision = depth·(4·d² + 3·d·ffn) + d·s²·h + h·h
  return depth * (4 * d * d + 3 * d * ffn) + d * s * s * hidden + hidden * hidden;
}

/** Build the DeepSeek-V4.1 IR from its text config. */
export function buildDeepseekV41(text: JsonObject, top: JsonObject, repoId: string): FamilyBuildResult {
  const r = createReader(text, text === top ? '' : 'text_config');
  const n = r.int('num_hidden_layers');
  const vocab = r.int('vocab_size');
  const tie = r.bool('tie_word_embeddings', false);
  const act = readActivation(r);
  const c = readDeepseekConfig(r);
  const moe = readMoeFfn(r, act);
  if (c.ratios.length < n) r.errors.push(`${r.path}.compress_ratios: ${c.ratios.length} entries for ${n} layers`);
  if (c.engramRows.length < c.engramLayers.length) r.errors.push(`${r.path}.engram_num_embeddings: one entry per engram layer needed`);
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

  const errors: string[] = [];
  const roles: CsaRole[] = [];
  const layers = repeatLayers(n, (i) => {
    const ratio = c.ratios[i];
    roles.push(csaRole(ratio, c.kvSources.includes(i), c.indexSources.includes(i)));
    return deepseekLayer(c, i, ratio, csaGlobal(c, i, errors), moe, `layer.${i}`);
  });
  if (errors.length > 0) return { kind: 'invalid_config', errors };
  const model = decoderModel({
    name: repoId,
    hidden: c.hidden,
    vocab,
    tieEmbeddings: tie,
    layers,
    residualStreams: c.streams,
    beforeHead: [mhcCollapse(c.streams, c.hidden)],
    source: repoId,
  });
  const structural = structuralErrors(model);
  if (structural.length > 0) return { kind: 'invalid_config', errors: structural };

  const vision = isJsonObject(top.vision_config) ? top.vision_config : isJsonObject(text.vision_config) ? text.vision_config : undefined;
  const visionParams = visionParamsEstimate(vision, c.hidden);
  const mtp = dsparkParamsEstimate(r, c, moe, n, vocab);
  const count = (role: CsaRole) => roles.filter((x) => x === role).length;
  const notes = [
    `CSA layers: ${count('swa')} window-only, ${count('full')} own KV + indexer, ${count('reindex')} own indexer on shared KV, ${count('reuse')} reuse KV and top-k`,
    `mHC residual: ${c.streams} streams; Engram tables before layers ${c.engramLayers.join(', ') || 'none'} (total params, not active)`,
    'prefill runs every layer on every prompt token (reference model.py; no Bounded Replay)',
    'attention sink, router biases not counted; indexer key norm counted as LayerNorm (< 1 k params per layer)',
  ];
  const excludedParts: string[] = [];
  if (mtp > 0) excludedParts.push('DSpark MTP stages');
  if (visionParams > 0) excludedParts.push('vision tower + aligner');
  return {
    kind: 'ok',
    model,
    notes,
    mtpParamsEstimate: mtp,
    visionParamsEstimate: visionParams,
    excludedParts,
    // Architecture-native cache formats (model.py; findings R2 §1): window KV fp8 with a ue8m0 scale per 32
    // (`act_quant(kv, fp8_block_size, …)`); compressed global KV fp4 e2m1 with an e4m3 scale per 16; indexer keys
    // fp4 with a ue8m0 scale per 32. They are defaults: a cache format the repo's quantization metadata states
    // explicitly wins. Weight-module formats are not hinted here: the quant readers derive them per repo (the native
    // checkpoint keeps the compressor and indexer wk / weights_proj in bf16; quant/moduleClass.ts marks them).
    quantHints: {
      kvCache: { dtype: 'mxfp8' },
      compressedKvCache: { dtype: 'nvfp4' },
      indexerKeyCache: { dtype: 'mxfp4' },
    },
  };
}

/** Family entry for DeepSeek-V4.1 (CSA + hyper-connections + Engram). */
export const deepseekV41Family: FamilyEntry = {
  family: 'deepseek_v41',
  modelTypes: ['deepseek_v41', 'deepseek_v41_text'],
  architectures: ['DeepseekV41ForCausalLM'],
  build: (text, top, files) => buildDeepseekV41(text, top, files.id),
};
