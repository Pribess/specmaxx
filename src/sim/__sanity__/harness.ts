/**
 * Sanity-check harness: run one community benchmark setup (model, weight quant, hardware, workload) through the
 * same `simulate()` the app uses and return the simulated theoretical-maximum prefill and decode rates, to compare
 * with the reported ones (docs/sanity-check.md). Pure TypeScript (no vite / vitest types), like the rest of src/sim.
 *
 * Adding a model that is not in the fixtures:
 * - Preferred: hand-write a `ModelSpec` with {@link simpleDecoder} (or the IR builders directly, like
 *   `REF_DENSE_8B` in `models/reference.ts`), copying the dims from the repo's config.json and citing it in
 *   `source`. The Hugging Face loader (`models/hf/`) only maps the shortlist families (qwen3_5 / qwen4_exp,
 *   MiniMax M3, Inkling, DeepSeek-V4.1, GLM-5.3, Mistral Large 4); plain `llama`, `qwen3`, `qwen3_moe`, `gpt_oss`,
 *   `gemma3`, `mistral` configs are `unsupported`, so a saved config.json of those would not load.
 * - For a model of a supported family: save the real files as `__sanity__/configs/<org>__<repo>.config.json`
 *   (and `.api.json`, trimmed as in `models/__fixtures__/README.md`) and load them with
 *   `sanityRepoFiles('<org>/<repo>')` from `configs.testutil.ts`, which falls back to `models/__fixtures__/`. Pass
 *   the result as `{ kind: 'hf', files }`.
 *
 * Workload convention (llama-bench style): prefill runs `promptLen` tokens from an EMPTY cache; decode generates
 * `genLen` tokens starting with `contextDepth` tokens already in the cache and is evaluated once, at the midpoint
 * context `contextDepth + genLen/2` (decode time is ~affine in the context length, so the midpoint rate is the
 * run's average rate up to a small convexity error). `pp512` → `promptLen: 512`; `tg128` → `genLen: 128,
 * contextDepth: 0`; `tg128 @ d4096` → `contextDepth: 4096`. Prefill at a depth is not modeled by `simulate`;
 * prefill from an empty cache is faster, so it stays an upper bound for a `pp @ dN` report.
 */

import { simulate } from '../engine/simulate';
import type { Bound, SimOptions, SimResult } from '../engine/types';
import { applyDeviceModifiers } from '../hw/modifiers';
import { DEVICE_PRESETS, TOPOLOGY_PRESETS, deviceUnlocks, singleDevice } from '../hw/presets/index';
import type { DeviceSpec, HardwareStack, TopologySpec } from '../hw/types';
import { resolveUnlocks, type UnlockSelection } from '../hw/unlocks';
import { decoderLayer, decoderModel, repeatLayers, type FfnParams, type MixerParams } from '../ir/builders';
import type { Dtype, TensorFormat } from '../ir/dtype';
import type { ModelSpec, ModuleQuant, ParallelConfig, QuantConfig, WeightModule } from '../ir/model';
import { mapHfModel } from '../models/hf/index';
import type { HfRepoFiles } from '../models/hf/types';

// ---------------------------------------------------------------------------------------------------------------
// Models

/** The model of a benchmark report. */
export type SanityModel =
  /** A hand-written architecture (e.g. `REF_DENSE_8B` or a {@link simpleDecoder}). */
  | { kind: 'spec'; spec: ModelSpec }
  /** Hugging Face repo files (from `sanityRepoFiles` / `fixtureRepoFiles`), mapped with `mapHfModel`. */
  | { kind: 'hf'; files: HfRepoFiles };

/**
 * A uniform decoder-only model: `layers` identical-shape layers (or per-index mixer / FFN via functions), pre-norm
 * RMSNorm, plain residual. Enough for Llama / Qwen3 / Mistral / Qwen3-MoE style models; anything more exotic uses
 * the IR builders directly. All dims are logical (unsharded).
 */
export function simpleDecoder(p: {
  /** Display name, e.g. `'Llama-3.1-8B'`. */
  name: string;
  /** Where the dims come from (config.json URL). */
  source: string;
  /** Hidden size d_model. Unit: elements. */
  hidden: number;
  /** Vocabulary size. Count. */
  vocab: number;
  /** LM head shares the embedding matrix. */
  tieEmbeddings: boolean;
  /** Decoder layers. Count. */
  layers: number;
  /** Token mixer of every layer, or of layer i. */
  mixer: MixerParams | ((i: number) => MixerParams);
  /** Feed-forward of every layer, or of layer i. */
  ffn: FfnParams | ((i: number) => FfnParams);
}): ModelSpec {
  const at = <T>(v: T | ((i: number) => T), i: number): T => (typeof v === 'function' ? (v as (i: number) => T)(i) : v);
  return decoderModel({
    name: p.name,
    hidden: p.hidden,
    vocab: p.vocab,
    tieEmbeddings: p.tieEmbeddings,
    source: p.source,
    layers: repeatLayers(p.layers, (i) =>
      decoderLayer({ name: `layer.${i}`, hidden: p.hidden, mixer: at(p.mixer, i), ffn: at(p.ffn, i) }),
    ),
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Weight quantization

/**
 * Named weight-quantization presets (see {@link QUANT_PRESETS} for the numbers):
 * - `bf16`, `fp16`: unquantized.
 * - `fp8`: W8A8 fp8 e4m3 (vLLM / SGLang FP8 checkpoints; the GEMM input is fp8 too).
 * - `fp8-wo`: fp8 e4m3 weight-only (fp8 storage, GEMM in the activation format: Marlin, Ampere, Apple).
 * - `nvfp4`: W4A4 NVFP4 (ModelOpt; the GEMM input is nvfp4 too).
 * - `mxfp4-wo`: mxfp4 weight-only on every linear layer (gpt-oss experts; attention BF16 needs a custom quant).
 * - `int4-g128`: GPTQ / AWQ int4 group 128, fp16 scale + 4-bit zero, weight-only.
 * - `mlx-4bit`, `mlx-8bit`: MLX affine group 64 (16-bit scale + 16-bit bias per group), weight-only, embeddings
 *   and LM head quantized too (mlx-lm `nn.quantize`).
 * - `q4_0`, `q4_k_m`, `q5_k_m`, `q6_k`, `q8_0`: llama.cpp GGUF types, weight-only, by average bits per weight.
 * - `native`: the repo's own quantization as the HF loader derives it (`kind: 'hf'` models only).
 */
export type QuantPresetId =
  | 'bf16'
  | 'fp16'
  | 'fp8'
  | 'fp8-wo'
  | 'nvfp4'
  | 'mxfp4-wo'
  | 'int4-g128'
  | 'mlx-4bit'
  | 'mlx-8bit'
  | 'q4_0'
  | 'q4_k_m'
  | 'q5_k_m'
  | 'q6_k'
  | 'q8_0'
  | 'native';

/** One weight-quant preset: the formats of the linear layers, the embedding table and the LM head. */
export interface QuantPreset {
  /** Linear-layer weight format (every weight module except hyper-connection mixers). */
  weights: TensorFormat;
  /**
   * GEMM input dtype for W8A8 / W4A4 schemes (the activations are quantized too); absent = weight-only (the
   * GEMM runs in the activation format, `resolveGemmMath`).
   */
  gemmInput?: Dtype;
  /** Embedding-table format; absent = the activation dtype (unquantized). */
  embeddings?: TensorFormat;
  /** LM-head format; absent = the embedding format (unquantized unless `embeddings` is set). */
  lmHead?: TensorFormat;
  /**
   * Per-module weight formats that differ from `weights` (e.g. gpt-oss GGUF: MXFP4 routed experts, Q8_0 attention);
   * the preset's `gemmInput` applies to them too.
   */
  modules?: Partial<Record<WeightModule, TensorFormat>>;
  /** Where the bits-per-weight figure comes from, and what it approximates. */
  note: string;
}

/**
 * A storage format with `bitsPerWeight` average bits per weight (scales and mins included), for formats the dtype
 * table has no exact dtype for (GGUF K-quants, MLX affine). The element dtype is `int4` below 8 bits and `int8`
 * from 8 bits, and the rest of the bits go in `extraBitsPerElem` (NEGATIVE below 4 bits, e.g. Q3_K: a storage
 * device only). Exact for bytes moved and resident; for compute it is honest only for weight-only schemes, where
 * the GEMM runs in the activation format anyway (`resolveGemmMath`: the wider operand wins).
 */
export function bpwFormat(bitsPerWeight: number): TensorFormat {
  const dtype: Dtype = bitsPerWeight >= 8 ? 'int8' : 'int4';
  const elementBits = dtype === 'int8' ? 8 : 4;
  // Formula: extra = bpw − elementBits
  const extraBitsPerElem = bitsPerWeight - elementBits;
  return extraBitsPerElem === 0 ? { dtype } : { dtype, extraBitsPerElem };
}

/** GGUF Q6_K: 210 bytes per 256-weight super-block (6-bit quants + 8-bit sub-block scales + fp16 d) = 6.5625 bpw. */
const GGUF_Q6_K_BPW = (210 * 8) / 256;
/** GGUF Q4_K: 144 bytes per 256-weight super-block (4-bit quants, 6-bit scales/mins, fp16 d/dmin) = 4.5 bpw. */
const GGUF_Q4_K_BPW = (144 * 8) / 256;

/**
 * The preset table (`native` is not here: it comes from the repo). GGUF numbers are llama.cpp block sizes
 * (ggml-common.h: block_q4_0 18 B / 32, block_q8_0 34 B / 32, block_q4_K 144 B / 256, block_q6_K 210 B / 256);
 * the `_K_M` mixes (some tensors one type up) are approximated by one average over the linear layers.
 */
export const QUANT_PRESETS: Record<Exclude<QuantPresetId, 'native'>, QuantPreset> = {
  bf16: { weights: { dtype: 'bf16' }, note: 'unquantized bf16' },
  fp16: { weights: { dtype: 'fp16' }, note: 'unquantized fp16' },
  fp8: {
    weights: { dtype: 'fp8_e4m3' },
    gemmInput: 'fp8_e4m3',
    note: 'W8A8 fp8 e4m3, per-channel or 128×128 block scales (≤ 0.002 bit/weight, left out); LM head unquantized',
  },
  'fp8-wo': { weights: { dtype: 'fp8_e4m3' }, note: 'fp8 e4m3 weight-only (dequantized to the activation format)' },
  nvfp4: {
    weights: { dtype: 'nvfp4' },
    gemmInput: 'nvfp4',
    note: 'W4A4 NVFP4 (e2m1 + e4m3 scale per 16 = 4.5 bpw); LM head unquantized',
  },
  'mxfp4-wo': { weights: { dtype: 'mxfp4' }, note: 'mxfp4 weight-only (e2m1 + e8m0 scale per 32 = 4.25 bpw); LM head unquantized' },
  'int4-g128': {
    weights: { dtype: 'int4', extraBitsPerElem: (16 + 4) / 128 },
    note: 'GPTQ/AWQ int4 g128: fp16 scale + 4-bit zero per 128 = 4.15625 bpw, weight-only; LM head unquantized',
  },
  'mlx-4bit': {
    weights: bpwFormat(4 + 32 / 64),
    embeddings: bpwFormat(4 + 32 / 64),
    note: 'MLX affine 4-bit g64: 16-bit scale + 16-bit bias per 64 = 4.5 bpw, embeddings and LM head too',
  },
  'mlx-8bit': {
    weights: bpwFormat(8 + 32 / 64),
    embeddings: bpwFormat(8 + 32 / 64),
    note: 'MLX affine 8-bit g64 = 8.5 bpw, embeddings and LM head too',
  },
  q4_0: {
    weights: bpwFormat(4.5),
    embeddings: bpwFormat(4.5),
    lmHead: bpwFormat(GGUF_Q6_K_BPW),
    note: 'GGUF Q4_0: 18 B per 32 = 4.5 bpw; output.weight Q6_K (llama.cpp default)',
  },
  q4_k_m: {
    weights: bpwFormat(4.85),
    embeddings: bpwFormat(GGUF_Q4_K_BPW),
    lmHead: bpwFormat(GGUF_Q6_K_BPW),
    note:
      'GGUF Q4_K_M: Q4_K (4.5 bpw) with attn_v / ffn_down of about half the layers in Q6_K → ≈ 4.85 bpw average ' +
      '(±0.1 by model); token_embd Q4_K, output.weight Q6_K',
  },
  q5_k_m: {
    weights: bpwFormat(5.7),
    embeddings: bpwFormat((176 * 8) / 256),
    lmHead: bpwFormat(GGUF_Q6_K_BPW),
    note: 'GGUF Q5_K_M: Q5_K (176 B / 256 = 5.5 bpw) with some Q6_K → ≈ 5.7 bpw; output.weight Q6_K',
  },
  q6_k: {
    weights: bpwFormat(GGUF_Q6_K_BPW),
    embeddings: bpwFormat(GGUF_Q6_K_BPW),
    note: 'GGUF Q6_K: 210 B per 256 = 6.5625 bpw everywhere',
  },
  q8_0: {
    weights: bpwFormat(8.5),
    embeddings: bpwFormat(8.5),
    note: 'GGUF Q8_0: 34 B per 32 (int8 + fp16 scale) = 8.5 bpw everywhere',
  },
};

/** Weight modules a preset with a `gemmInput` or `modules` sets explicitly (every class except the hyper-connection mixer). */
const QUANTIZED_MODULES: Record<Exclude<WeightModule, 'residual_mixer'>, true> = {
  attention: true,
  attention_kv_down: true,
  attention_kv_up: true,
  linear_attention: true,
  indexer: true,
  indexer_key: true,
  indexer_head_weights: true,
  dense_mlp: true,
  shared_experts: true,
  routed_experts: true,
  ngram: true,
};

/**
 * The weight quantization of a report: a preset id, an ad-hoc {@link QuantPreset} (resolved like a named one, e.g.
 * a GGUF file's measured bits per weight), or a full `QuantConfig` written by hand (used as is).
 */
export type SanityQuant =
  | QuantPresetId
  | { kind: 'preset'; preset: QuantPreset }
  | { kind: 'config'; config: QuantConfig };

/** Numeric formats around the weights; every field is optional. */
export interface SanityNumerics {
  /** Activation dtype (and the GEMM input of weight-only schemes). Default `'fp16'` (llama.cpp, MLX, vLLM fp16). */
  activations?: Dtype;
  /**
   * GEMM input override for weight-only presets, e.g. `'int8'` for llama.cpp CUDA MMQ, which quantizes activations
   * to q8_1 and runs int8 MMA. Default: the preset's.
   */
  gemmInput?: Dtype;
  /** KV-cache format. Default: the repo's (`kind: 'hf'`), else `{ dtype: activations }` (llama.cpp f16 cache). */
  kvCache?: TensorFormat;
  /** Requested float accumulator (fp16 doubles the GeForce fp16 tensor rate). Default `'fp32'`. */
  accumulate?: 'fp32' | 'fp16';
  /**
   * `native` only: drop the repo's activation quantization (`gemmInput`, `lmHeadGemmInput`), for engines that run a
   * W4A4 / W8A8 checkpoint weight-only (vLLM Marlin on Ampere, which has no fp8 / fp4 tensor cores). Default false.
   */
  weightOnly?: boolean;
}

/**
 * The `QuantConfig` of a preset (or a hand-written config, returned as is). Over an HF model, the preset replaces
 * the weight, embedding and LM-head formats and keeps the repo's special caches (compressed KV, indexer keys,
 * recurrent state, lookup tables); `native` keeps the repo's quantization whole.
 */
export function resolveQuant(quant: SanityQuant, numerics: SanityNumerics = {}, native?: QuantConfig): QuantConfig {
  if (typeof quant !== 'string' && quant.kind === 'config') return quant.config;
  const activations = numerics.activations ?? 'fp16';
  const accumulate = numerics.accumulate ?? 'fp32';
  if (quant === 'native') {
    if (native === undefined) throw new Error("quant 'native' needs a Hugging Face model (kind: 'hf')");
    const base: QuantConfig = { ...native, ...(numerics.kvCache ? { kvCache: numerics.kvCache } : {}), accumulate };
    return numerics.weightOnly ? stripActivationQuant(base) : base;
  }
  const p = typeof quant === 'string' ? QUANT_PRESETS[quant] : quant.preset;
  const gemmInput = numerics.gemmInput ?? p.gemmInput;
  const unquantized: TensorFormat = { dtype: activations };
  const embeddings = p.embeddings ?? unquantized;
  const keep: Partial<QuantConfig> =
    native === undefined
      ? {}
      : {
          ...(native.compressedKvCache ? { compressedKvCache: native.compressedKvCache } : {}),
          ...(native.indexerKeyCache ? { indexerKeyCache: native.indexerKeyCache } : {}),
          ...(native.recurrentState ? { recurrentState: native.recurrentState } : {}),
          ...(native.lookupTables ? { lookupTables: native.lookupTables } : {}),
        };
  const modules: Partial<Record<WeightModule, ModuleQuant>> | undefined =
    gemmInput === undefined && p.modules === undefined
      ? undefined
      : Object.fromEntries(
          (Object.keys(QUANTIZED_MODULES) as WeightModule[]).map((m) => {
            const weights = p.modules?.[m] ?? p.weights;
            return [m, gemmInput === undefined ? { weights } : { weights, gemmInput }];
          }),
        );
  return {
    ...keep,
    weights: p.weights,
    activations,
    kvCache: numerics.kvCache ?? native?.kvCache ?? unquantized,
    embeddings,
    lmHead: p.lmHead ?? embeddings,
    highPrecision: activations === 'fp16' ? 'fp16' : 'bf16',
    accumulate,
    ...(modules ? { modules } : {}),
  };
}

/** `q` without activation quantization: every module's `gemmInput` and the `lmHeadGemmInput` removed. */
function stripActivationQuant(q: QuantConfig): QuantConfig {
  const { lmHeadGemmInput: _dropped, modules, ...rest } = q;
  void _dropped;
  if (modules === undefined) return rest;
  const weightOnly = Object.fromEntries(
    Object.entries(modules).map(([m, mq]) => [m, { weights: (mq as ModuleQuant).weights }]),
  ) as Partial<Record<WeightModule, ModuleQuant>>;
  return { ...rest, modules: weightOnly };
}

// ---------------------------------------------------------------------------------------------------------------
// Hardware

/** The hardware of a report. */
export interface SanityHardware {
  /** Device preset id (`DeviceSpec.id`), e.g. `'rtx-4090'`, `'apple-m3-ultra-80c'`, `'dgx-spark'`. */
  deviceId: string;
  /**
   * Topology: a `TopologySpec` (e.g. `pcieBox(RTX_3090, 2, PCIE_4_X16)`), the id of a `TOPOLOGY_PRESETS` entry,
   * or absent for one device. It must describe `deviceId`'s devices (not checked: topologies carry no device).
   */
  topology?: TopologySpec | string;
  /** Unlock selection (axis id → option id) on the device's `DEVICE_UNLOCKS` axes, e.g. `{ 'pcie-p2p': 'direct' }`. */
  unlocks?: UnlockSelection;
  /** Memory capacity per device; must be one of the (unlocked) device's options. Default: the largest. Unit: GB (1e9 B). */
  memCapacityGb?: number;
}

/** The `HardwareStack` of a {@link SanityHardware}. Throws on an unknown device or topology id. */
export function resolveHardware(h: SanityHardware): HardwareStack {
  const device: DeviceSpec | undefined = DEVICE_PRESETS.find((d) => d.id === h.deviceId);
  if (device === undefined) throw new Error(`unknown device preset '${h.deviceId}'`);
  let topology: TopologySpec;
  if (h.topology === undefined) topology = singleDevice(device);
  else if (typeof h.topology === 'string') {
    const found = TOPOLOGY_PRESETS.find((t) => t.id === h.topology);
    if (found === undefined) throw new Error(`unknown topology preset '${h.topology}'`);
    topology = found;
  } else topology = h.topology;
  const unlocks = resolveUnlocks(deviceUnlocks(device), h.unlocks ?? {});
  const modified = applyDeviceModifiers(device, unlocks.deviceModifiers);
  const options = modified.memory.capacityOptionsBytes;
  const memCapacityBytes = h.memCapacityGb === undefined ? Math.max(...options) : h.memCapacityGb * 1e9;
  return {
    device,
    topology,
    ...(unlocks.deviceModifiers.length > 0 ? { deviceModifiers: unlocks.deviceModifiers } : {}),
    ...(unlocks.intraNodeLinkModifiers.length > 0 ? { linkModifiers: { intraNode: unlocks.intraNodeLinkModifiers } } : {}),
    memCapacityBytes,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Report input and result

/** A benchmark workload, llama-bench style (see the header comment). */
export interface SanityWorkload {
  /** Concurrent sequences (global). Default 1. Count. */
  batch?: number;
  /** Prompt tokens per sequence processed by prefill, from an empty cache (`pp512` → 512). Unit: tokens. */
  promptLen: number;
  /** Tokens generated per sequence (`tg128` → 128). Unit: tokens. */
  genLen: number;
  /** Tokens already in the cache when generation starts (`-d`, or the prompt of a pp+tg run). Default 0. Unit: tokens. */
  contextDepth?: number;
}

/** Everything one simulated report needs. */
export interface SanityInput {
  /** Model architecture. */
  model: SanityModel;
  /** Weight quantization. */
  quant: SanityQuant;
  /** Activation / KV / accumulate formats. */
  numerics?: SanityNumerics;
  /** Hardware. */
  hardware: SanityHardware;
  /** Parallelism; default tp = every device of the topology, pp = dp = ep = 1. */
  parallel?: Partial<ParallelConfig>;
  /** Workload. */
  workload: SanityWorkload;
  /** Simulator assumption switches (default `DEFAULT_SIM_OPTIONS`). */
  options?: Partial<SimOptions>;
}

/** Seconds of op time per bound over one phase (every stage, one microbatch). Unit: seconds. */
export type BoundSeconds = Record<Bound, number>;

/** The simulated rates of one report. Rates are theoretical maxima. */
export interface SanityResult {
  /** Aggregate prefill rate over all sequences and replicas. Unit: tokens/s. */
  prefillTokensPerSec: number;
  /** Prefill rate seen by one sequence (`promptLen / TTFT`). Unit: tokens/s. */
  prefillPerSequenceTokensPerSec: number;
  /** Aggregate decode rate (all sequences). Unit: tokens/s. */
  decodeTokensPerSec: number;
  /** Decode rate of one sequence (what llama-bench `tg` reports at batch 1). Unit: tokens/s. */
  decodePerSequenceTokensPerSec: number;
  /** Tokens already in the KV cache at the evaluated decode step: `contextDepth + ceil(genLen / 2)`. Unit: tokens. */
  decodeContextLen: number;
  /** The bound with the most op time in prefill, and the time per bound. */
  prefillBound: { dominant: Bound; seconds: BoundSeconds };
  /** The bound with the most op time in decode, and the time per bound. */
  decodeBound: { dominant: Bound; seconds: BoundSeconds };
  /** True if the model, cache and activations fit every device. */
  fits: boolean;
  /** The `QuantConfig` simulated. */
  quant: QuantConfig;
  /** The full simulator result (per-op detail, memory, warnings). */
  sim: SimResult;
}

/** Op seconds per bound over a phase's stages, and the dominant bound (ties: compute, memory, comm). */
function boundSummary(phase: SimResult['prefill']): { dominant: Bound; seconds: BoundSeconds } {
  const seconds: BoundSeconds = { compute: 0, memory: 0, comm: 0 };
  for (const stage of phase.stages) {
    for (const layer of stage.layers) for (const op of layer.ops) seconds[op.bound] += op.sec;
    if (stage.sendToNext) seconds.comm += stage.sendToNext.sec;
  }
  const order: Bound[] = ['compute', 'memory', 'comm'];
  const dominant = order.reduce((best, b) => (seconds[b] > seconds[best] ? b : best), order[0]);
  return { dominant, seconds };
}

/** The `ModelSpec` and the repo's derived quant (if any) of a {@link SanityModel}. Throws if the repo does not map. */
export function resolveModel(m: SanityModel): { spec: ModelSpec; native?: QuantConfig } {
  if (m.kind === 'spec') return { spec: m.spec };
  const r = mapHfModel(m.files);
  if (r.kind !== 'ok') throw new Error(`${m.files.id}: ${r.kind}${'reason' in r && r.reason ? `: ${r.reason}` : ''}`);
  return { spec: r.model, native: r.quant.quant };
}

/**
 * Simulate one benchmark report with `simulate()`. Throws on an unknown preset, an unmappable repo, or an input
 * `simulate` rejects.
 */
export function simulateReport(input: SanityInput): SanityResult {
  const { spec, native } = resolveModel(input.model);
  const quant = resolveQuant(input.quant, input.numerics, native);
  const hardware = resolveHardware(input.hardware);
  const devices = hardware.topology.devicesPerNode * hardware.topology.nodes;
  const parallel: ParallelConfig = { tp: devices, pp: 1, dp: 1, ep: 1, ...input.parallel };
  const w = input.workload;
  // Formula: decode context = depth + ceil(genLen / 2)   (midpoint of the generated run)
  const decodeContextLen = Math.max(1, (w.contextDepth ?? 0) + Math.ceil(w.genLen / 2));
  const sim = simulate({
    model: spec,
    quant,
    parallel,
    workload: { batch: w.batch ?? 1, promptLen: w.promptLen, decodeContextLen },
    hardware,
    ...(input.options ? { options: input.options } : {}),
  });
  return {
    prefillTokensPerSec: sim.prefill.tokensPerSec,
    prefillPerSequenceTokensPerSec: sim.prefill.perSequenceTokensPerSec,
    decodeTokensPerSec: sim.decode.tokensPerSec,
    decodePerSequenceTokensPerSec: sim.decode.perSequenceTokensPerSec,
    decodeContextLen,
    prefillBound: boundSummary(sim.prefill),
    decodeBound: boundSummary(sim.decode),
    fits: sim.memory.fits,
    quant,
    sim,
  };
}
