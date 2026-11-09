/**
 * Community benchmark reports for the simulator sanity check (docs/sanity-check.md), as typed data, each with the
 * harness input that matches it. The check: the simulator gives THEORETICAL MAXIMA, so for every asserted metric
 * `simulated ≥ 0.9 × reported` must hold; a report above that is a sim underestimate (a bug in a preset number or
 * an op cost) unless it is a listed exception with a reason.
 *
 * Sources (collected 2026-11-07; extended 2026-11-09):
 * - LocalMaxxing (`lm-NN`, NN = the row of the researchers' curated table): `https://localmaxxing.com/api/speed-tests`
 *   records; prefill there is promptTokens / CLIENT TTFT (HTTP, tokenization and the first decode step included)
 *   over short (~300-token) prompts, so prefill ratios are expected to be very low.
 * - Web (`web-NN`, NN = the row of the web table): llama.cpp GitHub discussions, r/LocalLLaMA (via a redlib mirror),
 *   mlx-lm BENCHMARKS.md, blogs and gists. llama-bench `ppN` / `tgN` are separate tests from an empty cache.
 *
 * - Fresh review (`F-LMNN`, `F-WNN`: rows of the 2026-11-09 collection; `-ppN` / `-pN` distinguish
 *   separately reported prompt lengths): LocalMaxxing run URLs, Apple MLX distributed-inference tables,
 *   NVIDIA blogs, llama.cpp discussions and cited community benchmarks. Short-client-TTFT prefill is generally
 *   printed but excluded; unknown-length prefill rates are not encoded. Unknown decode context is explicitly a
 *   favourable short-context scenario, not a measured depth. See docs/review/practicality-review.md §2.
 *
 * Simulation settings, the most favourable HONEST upper bound (never tuned to fit a report):
 * - `options: UPPER_BOUND_OPTIONS`: ideal elementwise fusion (elementwise ops cost no activation traffic; real
 *   engines fuse most of them), ideal comm/compute overlap, best all-reduce algorithm (the simulator defaults).
 * - GEMM input: int8 only where the engine's real compute path is int8 (llama.cpp GGUF MMQ: activations quantized to
 *   q8_1 and int8 MMA / dot products; {@link LLAMACPP_MMQ}); otherwise the activation dtype (weight-only
 *   dequantization: Metal, MLX, vLLM Marlin / AWQ / GPTQ), or the checkpoint's own W8A8 / W4A4 format.
 * - Accumulator: fp32 (the default; no engine here asks for fp16-accumulate GEMMs).
 * - Prefill: `perSequence` (prompt / time to first token of one sequence) for batch-1 reports; aggregate only for
 *   batched reports. Decode: per stream at batch 1, aggregate when the report gives an aggregate.
 * - Unknown details resolve in the simulator's favour (the PCIe P2P driver when a report does not say, the faster of
 *   the stated link widths, the lower end of a bits-per-weight range, the largest memory option).
 * - Memory fit: not asserted, at the simulator defaults (GPU-usable share of unified memory, 8192-token prefill chunk,
 *   no runtime reserve; docs/simulation-model.md §8). They change no rate, so no exception depends on them; lm-27 and
 *   F-LM49 exceed the default GPU share (their runs raised the limit), see docs/sanity-check.md "Settings".
 */

import { ACCEPTANCE_PRESETS, expectedTokens, mtpTailAcceptance } from '../engine/specdec';
import type { OffloadConfig, SimOptions, SpecDecodeConfig } from '../engine/types';
import { ARC_PRO_B65 } from '../hw/presets/intel';
import { PCIE_4_X16, PCIE_4_X16_DIRECT, PCIE_4_X8, PCIE_5_X16_DIRECT, SPARK_CONNECTX_7, hostPcieLink } from '../hw/presets/links';
import {
  APPLE_SSD_M5_PRO_MAX,
  DGX_SPARK_SSD_4TB,
  DDR4_3200_8CH,
  DDR5_4800_12CH,
  DDR5_5600_2CH,
  DDR5_6000_2CH,
  DDR5_6400_2CH,
} from '../hw/presets/storage';
import { RADEON_AI_PRO_R9700 } from '../hw/presets/amd';
import { DGX_SPARK } from '../hw/presets/nvidia-datacenter';
import { RTX_3090, RTX_4090 } from '../hw/presets/nvidia';
import {
  DGX_SPARK_PAIR,
  DGX_SPARK_4,
  DGX_SPARK_4_RING,
  M5_ULTRA_TB5_CLUSTERS,
  RTX_5090_PCIE_2,
  pcieBox,
  socCluster,
} from '../hw/presets/topologies';
import type { ModelSpec } from '../ir/model';
import { REF_DENSE_8B } from '../models/reference';
import { bpwFormat, type QuantPreset, type SanityInput, type SanityNumerics } from './harness';
import {
  GLM_4_5_AIR,
  GPT_OSS_120B,
  GPT_OSS_20B,
  LLAMA3_70B,
  QWEN3_235B_A22B,
  QWEN3_30B_A3B,
  QWEN3_NEXT_80B_A3B,
  QWQ_32B,
} from './models';

// ---------------------------------------------------------------------------------------------------------------
// Types

/** The model a report is simulated with. */
export type ReportModel =
  /** A hand-written spec (`models.ts`, `models/reference.ts`). */
  | { kind: 'spec'; spec: ModelSpec }
  /** A Hugging Face fixture, loaded by the test with `sanityRepoFiles(repoId, { dir })` (`dir: 'quant'` for quantized repos). */
  | { kind: 'hf'; repoId: string; dir?: 'quant' };

/**
 * The block drafter of a speculative-decoding report: a separate drafter repo whose saved config.json the test
 * parses (`sanityDrafterSpec(repoId)`, `models/hf/__fixtures__/`), or the target checkpoint's own block drafter.
 */
export type ReportDrafter = { kind: 'repo'; repoId: string } | { kind: 'checkpoint' };

/**
 * The deployment-scenario knobs a report ran with (docs/simulation-model.md §12 offload, §13 speculative
 * decoding), as `DeploymentConfig` with a {@link ReportDrafter}. Their parameters (acceptance, hit ratio, resident
 * fraction) come from the report wherever it states them; the result is the bound given those parameters.
 */
export interface ReportDeployment {
  /** Weight offloading, as `DeploymentConfig.offload`. */
  offload?: OffloadConfig;
  /** Speculative decoding, as `DeploymentConfig.speculative` with a {@link ReportDrafter}. */
  speculative?: Omit<SpecDecodeConfig, 'drafter'> & { drafter?: ReportDrafter };
}

/** What a reported rate measures. */
export type RateKind =
  /** One sequence's rate (prefill: prompt / its TTFT; decode: its inter-token rate). Unit: tokens/s. */
  | 'perSequence'
  /** Summed over all concurrent sequences. Unit: tokens/s. */
  | 'aggregate';

/**
 * How one metric of a report is checked:
 * - `assert`: `simulated ≥ 0.9 × reported` must hold.
 * - `exception`: a KNOWN failure (`simulated < 0.9 × reported`) with its `reason`; the test asserts that it still fails,
 *   so a fix or a changed report shows up and the list stays honest.
 * - `excluded`: not compared (a report whose setup the simulator cannot express); the ratio is still printed.
 *   Speculative-decoding and offload reports are simulated with the same knobs on (`input.deployment`) and asserted.
 */
export type CheckMode = { mode: 'assert' } | { mode: 'exception'; reason: string } | { mode: 'excluded'; reason: string };

/** One reported rate. */
export interface ReportedRate {
  /** The reported rate. Unit: tokens/s. */
  tokensPerSec: number;
  /** Per sequence or aggregate. */
  kind: RateKind;
  /** How it is checked; default `{ mode: 'assert' }`. */
  check?: CheckMode;
}

/** One community benchmark report. */
export interface BenchmarkReport {
  /** Stable id: `lm-NN` / `web-NN` (original tables), or `F-LMNN` / `F-WNN` (2026-11-09 collection). */
  id: string;
  /** Link to the report (run page, thread, post, file). */
  sourceUrl: string;
  /** Date of the report or the run (ISO date). */
  date: string;
  /** LocalMaxxing curation tier (A verified clean, B clean unverified, C usable with a caveat); web rows have none. */
  tier?: 'A' | 'B' | 'C';
  /** Model as reported (HF id of the checkpoint actually run). */
  model: string;
  /** What is simulated instead, when not the exact model (sibling / shape / fixture), and how it relates. */
  simulatedModel: string;
  /** Quantization as reported (format, bits per weight, KV-cache format). */
  quant: string;
  /** Hardware as reported, including unlocks, power limits, links and memory size. */
  hardware: string;
  /** Engine, version and backend. */
  engine: string;
  /** Parallelism as reported (TP / layer split / EP / DP). */
  parallelism: string;
  /** Concurrent sequences. Count. */
  batch: number;
  /** Prompt (prefill) length per sequence. Unit: tokens. */
  promptLen: number;
  /** Generated tokens per sequence. Unit: tokens. */
  genLen: number;
  /**
   * Tokens in the cache when decode starts: the prompt (plus any prior depth) wherever generation follows the prompt
   * (servers, vLLM, `mlx_lm.benchmark` / `generate`, llama-batched-bench); only `-d` for llama-bench, whose `tg` test
   * starts from an empty cache. Must equal `input.workload.contextDepth` (checked by the test). Unit: tokens.
   */
  contextDepth: number;
  /** Reported prefill rate; absent when the report has none. */
  prefill?: ReportedRate;
  /** Reported decode rate; absent when the report has none. */
  decode?: ReportedRate;
  /**
   * Size of the weight file, when the report gives it, to cross-check the simulated resident weight bytes.
   * Unit: bytes.
   */
  weightFileBytes?: number;
  /** Caveats of the report (data quality, engine limits, what was assumed to map it). */
  caveats: string;
  /**
   * The harness input that matches the report (its `model` is a {@link ReportModel}, its `deployment` a
   * {@link ReportDeployment}).
   */
  input: Omit<SanityInput, 'model' | 'deployment'> & { model: ReportModel; deployment?: ReportDeployment };
}

// ---------------------------------------------------------------------------------------------------------------
// Shared settings

/**
 * Simulator options of every report: ideal elementwise fusion (the favourable bound: real engines fuse norms,
 * activations and residual adds into GEMM epilogues); overlap and all-reduce keep their (already ideal) defaults.
 * See docs/simulation-model.md §7 "Per-op time".
 */
export const UPPER_BOUND_OPTIONS: Partial<SimOptions> = { elementwiseFusion: 'ideal' };

/**
 * llama.cpp GGUF on CUDA / HIP / Vulkan / SYCL: the quantized matmul path (MMQ, and the Vulkan / SYCL integer-dot
 * shaders) quantizes activations to q8_1 and runs int8 dot products, so the GEMM peak is the device's int8 rate.
 * On every device here the int8 peak is ≥ its fp16 peak, so int8 also bounds the dequantize-to-fp16 path a backend
 * may pick instead. Activations and the flash-attention math are fp16.
 */
export const LLAMACPP_MMQ: SanityNumerics = { activations: 'fp16', gemmInput: 'int8' };

/** llama.cpp / Ollama on Metal: quantized weights are dequantized to fp16 inside simdgroup matmuls. */
export const LLAMACPP_METAL: SanityNumerics = { activations: 'fp16' };

/** MLX: weight-only quantized matmuls in the model's bf16 activations. */
export const MLX: SanityNumerics = { activations: 'bf16' };

/** vLLM / SGLang: bf16 activations; the checkpoint's own W8A8 / W4A4 formats when its quant config sets them. */
export const VLLM: SanityNumerics = { activations: 'bf16' };

/** A GGUF file with `bpw` average bits per linear weight (and the same for embeddings and LM head unless given). */
function ggufBpw(bpw: number, note: string, head?: number): QuantPreset {
  return { weights: bpwFormat(bpw), embeddings: bpwFormat(bpw), lmHead: bpwFormat(head ?? bpw), note };
}

/**
 * gpt-oss GGUF (ggml-org/gpt-oss-*-GGUF "MXFP4"): routed experts MXFP4 (4.25 bpw), every other tensor ≈ Q8_0
 * (8.5 bpw). Cross-check: 20b = 19.1 G expert params × 4.25/8 + 1.8 G other × 8.5/8 = 12.06 GB vs the 11.27 GiB =
 * 12.10 GB file.
 */
const GPT_OSS_GGUF: QuantPreset = {
  weights: bpwFormat(8.5),
  modules: { routed_experts: { dtype: 'mxfp4' } },
  embeddings: bpwFormat(8.5),
  note: 'gpt-oss GGUF: MXFP4 routed experts, Q8_0-sized everything else (matches the file sizes)',
};

/** GiB → bytes (×2³⁰). */
const GIB = 2 ** 30;

/**
 * ubergarm Qwen3-235B-A22B mix-IQ3_K (ik_llama.cpp): attention IQ6_K (6.625 bpw), routed experts gate/up IQ3_K
 * (3.4375 bpw) and down IQ4_K (4.5 bpw), averaged over the three matrices to (2·3.4375 + 4.5)/3 = 3.7917 bpw;
 * token_embd and output Q8_0 (8.5). Cross-check: 227.0 G expert params × 3.7917/8 + 6.70 G attention × 6.625/8 +
 * 1.245 G embedding/head × 8.5/8 = 114.5 GB vs the 106.830 GiB = 114.7 GB file.
 */
const UBERGARM_MIX_IQ3_K: QuantPreset = {
  weights: bpwFormat(6.625),
  modules: { routed_experts: bpwFormat((2 * 3.4375 + 4.5) / 3) },
  embeddings: bpwFormat(8.5),
  note: 'mix-IQ3_K: IQ6_K attention, IQ3_K/IQ4_K experts (3.79 bpw average), Q8_0 embedding and head',
};

/**
 * The constant per-position acceptance α that reproduces a reported mean accepted length `tau` (E[τ], bonus token
 * included) at draft length `k` under the i.i.d. formula E[τ] = (1 − α^(k+1)) / (1 − α) (design doc §5
 * conventions: α_iid is always derived). Bisection on [0, 1); E[τ] grows monotonically in α. Needs 1 ≤ tau < k + 1.
 * Dimensionless.
 */
export function iidAlphaForExpectedTokens(tau: number, k: number): number {
  if (!(tau >= 1 && tau < k + 1)) throw new Error(`mean accepted length ${tau} outside [1, ${k + 1}) at k ${k}`);
  let lo = 0;
  let hi = 1 - 1e-12;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (expectedTokens({ kind: 'iid', alpha: mid }, k) < tau) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** KV cache in GGUF q4_0 (4.5 bpw) and q8_0 (8.5 bpw). */
const KV_Q4_0 = bpwFormat(4.5);
const KV_Q8_0 = bpwFormat(8.5);

/** Shortlist models used by several reports (HF fixtures in `models/__fixtures__/`). */
const QWEN27 = { kind: 'hf', repoId: 'Qwen/Qwen3.8-27B' } as const;
const FLASH_NEXT = { kind: 'hf', repoId: 'Qwen/Qwen3.8-Flash-Next' } as const;
const GLM53_FLASH = { kind: 'hf', repoId: 'zai-org/GLM-5.3-Flash' } as const;

// ---------------------------------------------------------------------------------------------------------------
// Reports

/** Every report, LocalMaxxing first, then web; see the header for the settings. */
export const REPORTS: BenchmarkReport[] = [
  // ----------------------------------------------------------------------------------------- LocalMaxxing
  {
    id: 'lm-01',
    sourceUrl: 'https://localmaxxing.com/en/models/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF?run=cmw13j1dv0aeioa01vc2babep',
    date: '2026-11-03',
    tier: 'A',
    model: 'JonathanColetti/Qwen3.8-27B-Uncensored-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture; same architecture fine-tune)',
    quant: 'Q4_K_M (≈16.5 GB), KV q4_0',
    hardware: '1× RTX 3090 24 GB, Zagitta gpu-unlock label, 420 W limit, PCIe Gen3 x16',
    engine: 'llama.cpp b12030 (CUDA), llama-server -fa on',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 2048,
    genLen: 512,
    contextDepth: 2048,
    prefill: { tokensPerSec: 1284.3, kind: 'perSequence' },
    decode: { tokensPerSec: 43.2, kind: 'perSequence' },
    caveats: 'Verified. Prefill = prompt / client TTFT. The gpu-unlock tensor-rate axis is applied (stated in the run notes).',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q4_0 },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { promptLen: 2048, genLen: 512, contextDepth: 2048 },
    },
  },
  {
    id: 'lm-02',
    sourceUrl: 'https://localmaxxing.com/en/models/RedHatAI/Qwen3.8-27B-INT4?run=cmvu9h7cs0hbqs6017yhvimid',
    date: '2026-10-30',
    tier: 'A',
    model: 'RedHatAI/Qwen3.8-27B-INT4',
    simulatedModel: 'RedHatAI/Qwen3.8-27B-INT4 (quant fixture, native W4A16 g128)',
    quant: 'INT4 W4A16 g128, KV bf16 (auto)',
    hardware: '1× RTX 3090 24 GB, Zagitta gpu-unlock label, 420 W limit',
    engine: 'vLLM 0.32.0 (CUDA, Marlin)',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 296,
    genLen: 512,
    contextDepth: 296,
    prefill: { tokensPerSec: 1938.1, kind: 'perSequence' },
    decode: { tokensPerSec: 48.6, kind: 'perSequence' },
    caveats: 'Verified. Short canonical prompt (TTFT-dominated prefill). Marlin is weight-only fp16 MMA with fp32 accumulate.',
    input: {
      model: { kind: 'hf', repoId: 'RedHatAI/Qwen3.8-27B-INT4', dir: 'quant' },
      quant: 'native',
      numerics: { ...VLLM, kvCache: { dtype: 'bf16' } },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { promptLen: 296, genLen: 512, contextDepth: 296 },
    },
  },
  {
    id: 'lm-04',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmszmr3yy0e75ms01z8t64io4',
    date: '2026-08-19',
    tier: 'B',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'UD-Q4_K_XL (≈4.9–5.3 bpw), KV q8_0',
    hardware: '1× RTX 4090 24 GB (stock)',
    engine: 'llama.cpp build 8101 (CUDA), llama-bench -fa 1 -b 128 -ub 64',
    parallelism: 'split-mode none',
    batch: 1,
    promptLen: 8192,
    genLen: 512,
    contextDepth: 0,
    prefill: { tokensPerSec: 1559.4, kind: 'perSequence' },
    decode: { tokensPerSec: 45.74, kind: 'perSequence' },
    caveats: 'llama-bench pp8192 and tg512 are separate tests; ubatch 64 depresses prefill.',
    input: {
      model: QWEN27,
      quant: { kind: 'preset', preset: ggufBpw(4.9, 'UD-Q4_K_XL, lower end of 4.9–5.3 bpw') },
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q8_0 },
      hardware: { deviceId: 'rtx-4090' },
      workload: { promptLen: 8192, genLen: 512 },
    },
  },
  {
    id: 'lm-05',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmsxftgtn0b99ms01ubcci7my',
    date: '2026-08-17',
    tier: 'B',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'UD-Q4_K_XL (≈4.9–5.3 bpw)',
    hardware: '1× RTX 5090 32 GB',
    engine: 'llama.cpp b10470 (CUDA, Windows), llama-bench -p 512 -n 128',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 3097.2, kind: 'perSequence' },
    decode: { tokensPerSec: 62.81, kind: 'perSequence' },
    caveats: 'llama-bench defaults (no -fa); unverified.',
    input: {
      model: QWEN27,
      quant: { kind: 'preset', preset: ggufBpw(4.9, 'UD-Q4_K_XL, lower end of 4.9–5.3 bpw') },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-5090' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  {
    id: 'lm-06',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-NVFP4?run=cmsxzbo3t0cnbms016n4fdk9x',
    date: '2026-08-18',
    tier: 'B',
    model: 'unsloth/Qwen3.8-27B-NVFP4',
    simulatedModel: 'unsloth/Qwen3.8-27B-NVFP4 (quant fixture, native NVFP4 W4A4 MLP + FP8 attention)',
    quant: 'NVFP4, KV fp8',
    hardware: '1× RTX 5090 32 GB, ~423 W',
    engine: 'vLLM 0.27.1 (CUDA, WSL2)',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 1037,
    genLen: 1024,
    contextDepth: 1037,
    prefill: { tokensPerSec: 9144.9, kind: 'perSequence' },
    decode: { tokensPerSec: 71.11, kind: 'perSequence' },
    caveats: 'MTP off per notes; prefix cache on (one request, so no hit is assumed); client TTFT prefill.',
    input: {
      model: { kind: 'hf', repoId: 'unsloth/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: { ...VLLM, kvCache: { dtype: 'fp8_e4m3' } },
      hardware: { deviceId: 'rtx-5090' },
      workload: { promptLen: 1037, genLen: 1024, contextDepth: 1037 },
    },
  },
  {
    id: 'lm-07',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B-FP8?run=cmtpbv6tt02hin701q3p40kpp',
    date: '2026-09-06',
    tier: 'A',
    model: 'Qwen/Qwen3.8-27B-FP8',
    simulatedModel: 'Qwen/Qwen3.8-27B-FP8 (quant fixture, native W8A8 fp8 block 128)',
    quant: 'FP8 block-quantized',
    hardware: '1× B200 (rented, 179 GB visible)',
    engine: 'vLLM 0.28.0 (CUDA, full CUDA graphs)',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 347,
    genLen: 1024,
    contextDepth: 347,
    prefill: { tokensPerSec: 9352.3, kind: 'perSequence' },
    decode: { tokensPerSec: 106.2, kind: 'perSequence' },
    caveats: 'Verified. Decode is kernel-launch-bound per the submitter (far below the HBM roofline).',
    input: {
      model: { kind: 'hf', repoId: 'Qwen/Qwen3.8-27B-FP8', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'b200' },
      workload: { promptLen: 347, genLen: 1024, contextDepth: 347 },
    },
  },
  {
    id: 'lm-08',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmtpbzgly02htn701f9arg2ui',
    date: '2026-09-06',
    tier: 'A',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'UD-Q4_K_M (≈4.8 bpw, 16.5 GB)',
    hardware: '1× B200',
    engine: 'llama.cpp 74a7c89 (CUDA sm_100), llama-server -fa on',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 347,
    genLen: 1024,
    contextDepth: 347,
    prefill: { tokensPerSec: 548.7, kind: 'perSequence' },
    decode: { tokensPerSec: 86.8, kind: 'perSequence' },
    caveats: 'Verified. Kernel-launch-bound decode; tiny prompt.',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'b200' },
      workload: { promptLen: 347, genLen: 1024, contextDepth: 347 },
    },
  },
  {
    id: 'lm-09',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B-FP8?run=cmtrfjhhv00dcpa013a8uyn0j',
    date: '2026-09-07',
    tier: 'C',
    model: 'Qwen/Qwen3.8-27B-FP8',
    simulatedModel: 'Qwen/Qwen3.8-27B-FP8 (quant fixture)',
    quant: 'FP8',
    hardware: '1× H200 SXM 141 GB',
    engine: 'vLLM 0.24.1.dev (custom Pareton image, MTP head with 8 draft tokens auto-enabled)',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 345,
    genLen: 512,
    contextDepth: 345,
    prefill: { tokensPerSec: 7747.6, kind: 'perSequence' },
    decode: { tokensPerSec: 274.9, kind: 'perSequence' },
    caveats:
      'MTP speculative decoding (8 draft tokens) is on and simulated (k 8). The record gives no acceptance; the ' +
      'multi-step MTP prose preset (α 0.75, Qwen3.8-27B card k=7 MT-Bench) is used (the canonical prompt is assumed chat prose).',
    input: {
      model: { kind: 'hf', repoId: 'Qwen/Qwen3.8-27B-FP8', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'h200-sxm' },
      workload: { promptLen: 345, genLen: 512, contextDepth: 345 },
      deployment: {
        speculative: { method: 'mtp', draftLength: 8, acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha } },
      },
    },
  },
  {
    id: 'lm-10',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmtm14eqs00wgoe01mmqnbok9',
    date: '2026-09-03',
    tier: 'B',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'Q4_0 (4.5 bpw, 14.94 GiB)',
    hardware: '1× Radeon AI Pro R9700 32 GB',
    engine: 'llama.cpp b10711 (Vulkan, Windows), llama-bench -fa 1 -p 512 -n 128',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 932.1, kind: 'perSequence' },
    decode: { tokensPerSec: 34.91, kind: 'perSequence' },
    weightFileBytes: 14.94 * GIB,
    caveats: 'Paired with a UD-Q4_K_XL run on the same binary (30.5 tok/s); other R9700 records vary 1.2–52.7 tok/s.',
    input: {
      model: QWEN27,
      quant: 'q4_0',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'radeon-ai-pro-r9700' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  {
    id: 'lm-11',
    sourceUrl: 'https://localmaxxing.com/en/models/amd/Qwen3.8-27B-Quark-AWQ-MXFP4?run=cmtx6ay860ai3ps011oye4gtx',
    date: '2026-09-11',
    tier: 'B',
    model: 'amd/Qwen3.8-27B-Quark-AWQ-MXFP4',
    simulatedModel: 'amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (quant fixture, native MXFP4)',
    quant: 'MXFP4 run as W4A8 (RADIANCE_MXFP4_W4A8=1), KV fp8',
    hardware: '2× Radeon AI Pro R9700 32 GB, PCIe (P2P not stated)',
    engine: 'vllm-radiance 1.0.16 (vLLM 0.28.0, ROCm 7.14)',
    parallelism: 'TP2',
    batch: 1,
    promptLen: 573,
    genLen: 512,
    contextDepth: 573,
    prefill: { tokensPerSec: 3487.3, kind: 'perSequence' },
    decode: { tokensPerSec: 45.2, kind: 'perSequence' },
    caveats:
      'The fixture says W4A4 mxfp4; the R9700 has no fp4 peak, so the lookup falls back to its fp8 peak = the W4A8 rate ' +
      'the engine runs. PCIe 5.0 x16 direct P2P assumed (favourable).',
    input: {
      model: { kind: 'hf', repoId: 'amd/Qwen3.8-27B-Quark-AWQ-MXFP4', dir: 'quant' },
      quant: 'native',
      numerics: { ...VLLM, kvCache: { dtype: 'fp8_e4m3' } },
      hardware: { deviceId: 'radeon-ai-pro-r9700', topology: pcieBox(RADEON_AI_PRO_R9700, 2, PCIE_5_X16_DIRECT) },
      workload: { promptLen: 573, genLen: 512, contextDepth: 573 },
    },
  },
  {
    id: 'lm-13',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmsxtoccy0cjlms01lnbw0pzu',
    date: '2026-08-17',
    tier: 'B',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'Q6_K (6.56 bpw, ≈22 GB)',
    hardware: '1× Intel Arc Pro B60 24 GB',
    engine: 'llama.cpp (SYCL or Vulkan, not stated), llama-bench p512 / n128',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 643.1, kind: 'perSequence' },
    decode: { tokensPerSec: 15.99, kind: 'perSequence' },
    caveats: 'Script-driven, no engine version.',
    input: {
      model: QWEN27,
      quant: 'q6_k',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'arc-pro-b60' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  {
    id: 'lm-15',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B?run=cmvrc4jz300j0s601ppkoyo08',
    date: '2026-10-27',
    tier: 'A',
    model: 'Qwen/Qwen3.8-27B (GGUF Q8_0)',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'Q8_0 (8.5 bpw, ≈28.7 GB)',
    hardware: '1× Intel Arc Pro B65 32 GB (PCIe passthrough into a VM)',
    engine: 'llama.cpp b11913 (SYCL/Vulkan image), llama-server -fa on',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 343,
    genLen: 256,
    contextDepth: 343,
    prefill: { tokensPerSec: 419.4, kind: 'perSequence' },
    decode: { tokensPerSec: 15.9, kind: 'perSequence' },
    caveats: 'Verified. Decode is flat across Q4–Q8 on this rig (engine-bound, not bandwidth-bound).',
    input: {
      model: QWEN27,
      quant: 'q8_0',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'arc-pro-b65' },
      workload: { promptLen: 343, genLen: 256, contextDepth: 343 },
    },
  },
  {
    id: 'lm-18',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmsxndz0o0bxsms01v7pv9o2d',
    date: '2026-08-17',
    tier: 'B',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'Q6_K (6.56 bpw)',
    hardware: '1× Intel Arc Pro B70 32 GB',
    engine: 'llama.cpp (SYCL or Vulkan, not stated), llama-bench p512 / n128',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 905.3, kind: 'perSequence' },
    decode: { tokensPerSec: 20.74, kind: 'perSequence' },
    caveats: 'Q6_K vs Q8_0 on the same box scales with bytes (20.74 vs 16.05): a clean bandwidth-bound pair (~75% of 608 GB/s).',
    input: {
      model: QWEN27,
      quant: 'q6_k',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'arc-pro-b70' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  {
    id: 'lm-20',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmvpm4oa7028vng011l44ex55',
    date: '2026-10-26',
    tier: 'A',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'UD-Q4_K_M (≈4.75 bpw, 16.5 GB)',
    hardware: 'AMD Ryzen AI Max+ 395 (Radeon 8060S), 96 GB unified record',
    engine: 'llama.cpp b11901 (Vulkan), llama-server -fa on',
    parallelism: 'single APU',
    batch: 1,
    promptLen: 349,
    genLen: 256,
    contextDepth: 349,
    prefill: { tokensPerSec: 261.7, kind: 'perSequence' },
    decode: { tokensPerSec: 12.9, kind: 'perSequence' },
    caveats: 'Verified. 96 GB is not a preset option; the 128 GB option is used (bandwidth is the same 256 GB/s).',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { promptLen: 349, genLen: 256, contextDepth: 349 },
    },
  },
  {
    id: 'lm-22',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmtmy9ugn004ppb01hljz9pcq',
    date: '2026-09-04',
    tier: 'C',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'Q4_K_M',
    hardware: 'Apple M5 Max 128 GB (40-core GPU inferred from 128 GB)',
    engine: 'Ollama (Metal)',
    parallelism: 'single SoC',
    batch: 1,
    promptLen: 75,
    genLen: 256,
    contextDepth: 75,
    prefill: { tokensPerSec: 882.3, kind: 'perSequence' },
    decode: { tokensPerSec: 24.3, kind: 'perSequence' },
    caveats: '75-token prompt: prefill is TTFT-dominated. Metal dequantizes to fp16 (no int8 path).',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m5-max-40c' },
      workload: { promptLen: 75, genLen: 256, contextDepth: 75 },
    },
  },
  {
    id: 'lm-23',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-NVFP4?run=cmt5s1z0103m7pu01km9fczpa',
    date: '2026-08-23',
    tier: 'C',
    model: 'unsloth/Qwen3.8-27B-NVFP4',
    simulatedModel: 'unsloth/Qwen3.8-27B-NVFP4 (quant fixture)',
    quant: 'NVFP4',
    hardware: '1× DGX Spark (GB10) 128 GB',
    engine: 'vLLM (version not stated)',
    parallelism: 'single device',
    batch: 1,
    promptLen: 74,
    genLen: 256,
    contextDepth: 74,
    prefill: { tokensPerSec: 373.7, kind: 'perSequence' },
    decode: { tokensPerSec: 10.1, kind: 'perSequence' },
    caveats: 'Early GB10 record, 74-token prompt, no notes.',
    input: {
      model: { kind: 'hf', repoId: 'unsloth/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark' },
      workload: { promptLen: 74, genLen: 256, contextDepth: 74 },
    },
  },
  {
    id: 'lm-24',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-Flash-Next?run=cmw1704i40ajcoa01fwx8w8m9',
    date: '2026-11-03',
    tier: 'B',
    model: 'Qwen/Qwen3.8-Flash-Next (GGUF UD-IQ3_XXS)',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (fixture)',
    quant: 'UD-IQ3_XXS (≈3.1–3.3 bpw), KV f16',
    hardware: '3× Radeon AI Pro R9700 32 GB, PCIe (host Ryzen 9 5950X)',
    engine: 'llama.cpp rdna-boosts fork r206 (ROCm 7.14), -ub 1024',
    parallelism: 'layer split -ts 1/1/0.8',
    batch: 1,
    promptLen: 573,
    genLen: 256,
    contextDepth: 573,
    prefill: { tokensPerSec: 1012.5, kind: 'perSequence' },
    decode: { tokensPerSec: 47.51, kind: 'perSequence' },
    caveats:
      'Sub-4-bpw GGUF: represented as int4 with a NEGATIVE extraBitsPerElem (3.1 bpw). Layer split = pp 3 at batch 1 ' +
      '(one GPU at a time). Even 1 : 1 : 1 split simulated instead of 1 : 1 : 0.8.',
    input: {
      model: FLASH_NEXT,
      quant: { kind: 'preset', preset: ggufBpw(3.1, 'UD-IQ3_XXS, lower end of 3.1–3.3 bpw') },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'radeon-ai-pro-r9700', topology: pcieBox(RADEON_AI_PRO_R9700, 3, PCIE_4_X16_DIRECT) },
      parallel: { tp: 1, pp: 3 },
      workload: { promptLen: 573, genLen: 256, contextDepth: 573 },
    },
  },
  {
    id: 'lm-27',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Qwen3.8-Flash-Next-GGUF?run=cmvpm4rw10292ng01k4wnsipr',
    date: '2026-10-26',
    tier: 'A',
    model: 'unsloth/Qwen3.8-Flash-Next-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (fixture)',
    quant: 'UD-Q2_K_XL (≈2.9–3 bpw)',
    hardware: 'AMD Ryzen AI Max+ 395, 96 GB unified record',
    engine: 'llama.cpp b11901 (Vulkan), llama-server -fa on',
    parallelism: 'single APU',
    batch: 1,
    promptLen: 345,
    genLen: 256,
    contextDepth: 345,
    prefill: { tokensPerSec: 298.3, kind: 'perSequence' },
    decode: { tokensPerSec: 32, kind: 'perSequence' },
    caveats: 'Verified. Sub-4-bpw (2.9) GGUF, negative extraBitsPerElem; 128 GB memory option used.',
    input: {
      model: FLASH_NEXT,
      quant: { kind: 'preset', preset: ggufBpw(2.9, 'UD-Q2_K_XL, lower end of 2.9–3 bpw') },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { promptLen: 345, genLen: 256, contextDepth: 345 },
    },
  },
  {
    id: 'lm-28',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-Flash-Next?run=cmtexccjm00bgn0011lvaqdyi',
    date: '2026-08-29',
    tier: 'C',
    model: 'Qwen3.8-Flash-Next-MLX-4bit',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (fixture)',
    quant: 'MLX 4-bit affine group 32 (4 + 32/32 = 5 bpw)',
    hardware: 'Apple M3 Ultra 512 GB (80-core GPU inferred from 512 GB)',
    engine: 'oMLX v0.6.3 (MLX)',
    parallelism: 'single SoC',
    batch: 1,
    promptLen: 619,
    genLen: 256,
    contextDepth: 619,
    prefill: { tokensPerSec: 309.6, kind: 'perSequence' },
    decode: { tokensPerSec: 14.8, kind: 'perSequence' },
    caveats: 'Decode is far below the bandwidth bound of a ~7 B-active MoE (engine / MoE gather bound).',
    input: {
      model: FLASH_NEXT,
      quant: { kind: 'preset', preset: { ...ggufBpw(5, 'MLX affine 4-bit g32: 16-bit scale + bias per 32'), note: 'MLX 4-bit g32 = 5 bpw' } },
      numerics: MLX,
      hardware: { deviceId: 'apple-m3-ultra-80c' },
      workload: { promptLen: 619, genLen: 256, contextDepth: 619 },
    },
  },
  {
    id: 'lm-32',
    sourceUrl: 'https://localmaxxing.com/en/models/nvidia/GLM-5.3-Flash-FP4-Local?run=cmvfggz3n014srf01ulz9qf41',
    date: '2026-10-19',
    tier: 'A',
    model: 'nvidia/GLM-5.3-Flash-FP4-Local',
    simulatedModel: 'nvidia/GLM-5.3-Flash-FP4-Local (quant fixture), run weight-only',
    quant: 'NVFP4 routed experts + FP8/MXFP8 rest, KV fp8',
    hardware: '8× RTX 3090 24 GB, ~240 W each, PCIe (no NVLink; P2P not stated), Threadripper PRO 7965WX',
    engine: 'vLLM 0.28.1rc1 + local SM86 adapter (Marlin repack)',
    parallelism: 'TP8 / EP8',
    batch: 1,
    promptLen: 278,
    genLen: 966,
    contextDepth: 278,
    prefill: { tokensPerSec: 720.5, kind: 'perSequence' },
    decode: { tokensPerSec: 81.28, kind: 'perSequence' },
    caveats:
      'Verified. Ampere has no fp8 / fp4 tensor cores: Marlin runs the checkpoint weight-only in fp16 (weightOnly). ' +
      'PCIe 4.0 x16 with the P2P driver assumed (favourable; not stated).',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/GLM-5.3-Flash-FP4-Local', dir: 'quant' },
      quant: 'native',
      numerics: { ...VLLM, kvCache: { dtype: 'fp8_e4m3' }, weightOnly: true },
      hardware: { deviceId: 'rtx-3090', topology: pcieBox(RTX_3090, 8, PCIE_4_X16), unlocks: { 'pcie-p2p': 'direct' } },
      parallel: { tp: 8, ep: 8 },
      workload: { promptLen: 278, genLen: 966, contextDepth: 278 },
    },
  },
  {
    id: 'lm-34',
    sourceUrl: 'https://localmaxxing.com/en/models/zai-org/GLM-5.3-Flash?run=cmtipc711066gp401ddtg1b3p',
    date: '2026-09-01',
    tier: 'C',
    model: 'GLM-5.3-Flash-MLX-4bit',
    simulatedModel: 'zai-org/GLM-5.3-Flash (fixture)',
    quant: 'MLX 4-bit (≈4.5–5 bpw)',
    hardware: 'Apple M3 Ultra 512 GB (80c inferred)',
    engine: 'oMLX v0.6.4 (MLX)',
    parallelism: 'single SoC',
    batch: 1,
    promptLen: 574,
    genLen: 256,
    contextDepth: 574,
    prefill: { tokensPerSec: 344.5, kind: 'perSequence' },
    decode: { tokensPerSec: 32.9, kind: 'perSequence' },
    caveats: '"Mild GPU contention" (machine shared with another fleet), so the rates are lower bounds.',
    input: {
      model: GLM53_FLASH,
      quant: 'mlx-4bit',
      numerics: MLX,
      hardware: { deviceId: 'apple-m3-ultra-80c' },
      workload: { promptLen: 574, genLen: 256, contextDepth: 574 },
    },
  },
  {
    id: 'lm-35',
    sourceUrl: 'https://localmaxxing.com/en/models/zai-org/GLM-5.3-Flash?run=cmum3byy10j02lq01b6821y8t',
    date: '2026-09-29',
    tier: 'B',
    model: 'GLM-5.3-Flash NVFP4 (/models/glm-5.3-flash-nvfp4)',
    simulatedModel: 'nvidia/GLM-5.3-Flash-NVFP4 (quant fixture; the closest published NVFP4 checkpoint, layout unverified)',
    quant: 'NVFP4, KV fp8_e4m3',
    hardware: '4× DGX Spark (GB10), ConnectX-7 200 GbE RoCE (record mislabeled as one 128 GB device)',
    engine: 'vLLM 0.30.1rc1 (CUDA), Ray, flashinfer_cutlass MoE',
    parallelism: 'TP4 (no expert parallel flag)',
    batch: 1,
    promptLen: 253,
    genLen: 768,
    contextDepth: 253,
    decode: {
      tokensPerSec: 74.6,
      kind: 'perSequence',
      check: {
        mode: 'exception',
        reason:
          'UNRESOLVED: the run used a local checkpoint (/models/glm-5.3-flash-nvfp4) whose module layout is not ' +
          'published; the record says only "NVFP4 4-bit weights". The closest published repo, nvidia/GLM-5.3-Flash-NVFP4, ' +
          'keeps attention, KDA and shared experts in BF16 (≈ 22 GB read per token over 4 devices), which simulates to ' +
          '46.6 tok/s < 0.9 × 74.6. An all-linear NVFP4 layout would simulate to 88.8 tok/s (ratio 0.84), but nothing ' +
          'in the record shows that layout, so it is not used. Either the checkpoint quantizes more than the published ' +
          'one, or the simulator underestimates this setup; not decided.',
      },
    },
    caveats:
      'Verified; prefill not reported. The only multi-node Spark row with plain decode. Checkpoint layout unknown ' +
      '(see the exception reason).',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/GLM-5.3-Flash-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: { ...VLLM, kvCache: { dtype: 'fp8_e4m3' } },
      hardware: { deviceId: 'dgx-spark', topology: socCluster(DGX_SPARK, 4, SPARK_CONNECTX_7) },
      parallel: { tp: 4 },
      workload: { promptLen: 253, genLen: 768, contextDepth: 253 },
    },
  },
  {
    id: 'lm-36',
    sourceUrl: 'https://localmaxxing.com/en/models/deepseek-ai/DeepSeek-V4.1-Flash?run=cmuhen2200drdlq01fa2qr70h',
    date: '2026-09-25',
    tier: 'C',
    model: 'deepseek-ai/DeepSeek-V4.1-Flash',
    simulatedModel: 'deepseek-ai/DeepSeek-V4.1-Flash (fixture, native FP8 + MXFP4 experts)',
    quant: 'FP8 attention / dense + MXFP4 experts, KV fp8',
    hardware: '4× DGX Spark (GB10), 200 Gb/s RoCE',
    engine: 'SGLang 0.0.0.dev0 (dsv4 attention backend)',
    parallelism: 'TP4, ep-size 1',
    batch: 1,
    promptLen: 274,
    genLen: 1024,
    contextDepth: 274,
    prefill: { tokensPerSec: 887.6, kind: 'perSequence' },
    decode: { tokensPerSec: 76, kind: 'perSequence' },
    caveats:
      'Verified. The only DeepSeek-V4.1-Flash data; every record has spec decoding on. DSpark (the in-checkpoint ' +
      'drafter, k 5 = its block 6 − 1) is simulated at the reported mean accepted length 2.85 (E[τ], bonus token included).',
    input: {
      model: { kind: 'hf', repoId: 'deepseek-ai/DeepSeek-V4.1-Flash' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: socCluster(DGX_SPARK, 4, SPARK_CONNECTX_7) },
      parallel: { tp: 4 },
      workload: { promptLen: 274, genLen: 1024, contextDepth: 274 },
      deployment: {
        speculative: {
          method: 'dspark',
          drafter: { kind: 'checkpoint' },
          draftLength: 5,
          acceptance: { kind: 'iid', alpha: iidAlphaForExpectedTokens(2.85, 5) },
        },
      },
    },
  },
  {
    id: 'lm-38',
    sourceUrl: 'https://localmaxxing.com/en/models/unsloth/Inkling-Small-GGUF?run=cms90mh5r0045pm01lf9n6cre',
    date: '2026-07-31',
    tier: 'B',
    model: 'unsloth/Inkling-Small-GGUF',
    simulatedModel: 'thinkingmachines/Inkling-Small (fixture)',
    quant: 'UD-IQ4_XS (≈4.25 bpw), KV f16',
    hardware: '6× RTX 3090 24 GB, PCIe, Threadripper PRO 7965WX',
    engine: 'llama.cpp inkling-exp branch (CUDA)',
    parallelism: 'layer split -ts 8,7,7,7,7,7',
    batch: 1,
    promptLen: 367,
    genLen: 508,
    contextDepth: 367,
    prefill: { tokensPerSec: 148.1, kind: 'perSequence' },
    decode: { tokensPerSec: 47.36, kind: 'perSequence' },
    caveats: 'Layer split = pp 6 at batch 1 (one GPU at a time). Even split simulated.',
    input: {
      model: { kind: 'hf', repoId: 'thinkingmachines/Inkling-Small' },
      quant: { kind: 'preset', preset: ggufBpw(4.25, 'IQ4_XS = 4.25 bpw') },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', topology: pcieBox(RTX_3090, 6, PCIE_4_X16) },
      parallel: { tp: 1, pp: 6 },
      workload: { promptLen: 367, genLen: 508, contextDepth: 367 },
    },
  },
  {
    id: 'lm-39',
    sourceUrl: 'https://localmaxxing.com/en/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead?run=cmtx753t70aiups01lmudzstu',
    date: '2026-09-11',
    tier: 'C',
    model: 'RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead',
    simulatedModel: 'nvidia/Qwen3.8-27B-NVFP4 (quant fixture; same NVFP4 MLP + FP8 attention layout)',
    quant: 'NVFP4 (BF16 LM head)',
    hardware: '2× RTX 5090 32 GB, PCIe, patched driver with forced P2P',
    engine: 'SGLang main@0149f56e (flashinfer 0.6.17), custom one-shot all-reduce',
    parallelism: 'TP2',
    batch: 1,
    promptLen: 512,
    genLen: 1024,
    contextDepth: 512,
    prefill: { tokensPerSec: 17562.7, kind: 'perSequence' },
    decode: { tokensPerSec: 227.3, kind: 'perSequence' },
    caveats:
      'Prefill may include the draft path. Fixture LM head is nvfp4, the report says BF16 (prefill hardly changes). ' +
      'DFlash2 speculative decoding is on and simulated with incoai/Qwen3.8-27B-DFlash2 (k 7, block 8); the record ' +
      'gives no acceptance, so the DFlash 2 prose preset (α 0.79) is used.',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'rtx-5090', topology: RTX_5090_PCIE_2, unlocks: { 'pcie-p2p': 'direct' } },
      parallel: { tp: 2 },
      workload: { promptLen: 512, genLen: 1024, contextDepth: 512 },
      deployment: {
        speculative: {
          method: 'dflash2',
          drafter: { kind: 'repo', repoId: 'incoai/Qwen3.8-27B-DFlash2' },
          draftLength: 7,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.dflash2.prose.alpha },
        },
      },
    },
  },
  {
    id: 'lm-42',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B?run=cmvjl3vpc03vcrf01vc9hojva',
    date: '2026-10-22',
    tier: 'C',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (fixture)',
    quant: 'BF16',
    hardware: '4× Intel Arc Pro B65 32 GB, EPYC 7343 host (PCIe 4.0)',
    engine: 'vLLM 0.32.0rc1 (XPU)',
    parallelism: 'TP4',
    batch: 16,
    promptLen: 345,
    genLen: 256,
    contextDepth: 345,
    prefill: { tokensPerSec: 238.6, kind: 'perSequence' },
    decode: { tokensPerSec: 371.6, kind: 'aggregate' },
    caveats:
      'Concurrency 16: decode is the aggregate; prefill is one request\'s prompt / TTFT under load (compared with the ' +
      'per-sequence rate of a 16-sequence prefill). PCIe 4.0 x16 direct P2P assumed (host is Gen4).',
    input: {
      model: QWEN27,
      quant: 'bf16',
      numerics: VLLM,
      hardware: { deviceId: 'arc-pro-b65', topology: pcieBox(ARC_PRO_B65, 4, PCIE_4_X16_DIRECT) },
      parallel: { tp: 4 },
      workload: { batch: 16, promptLen: 345, genLen: 256, contextDepth: 345 },
    },
  },

  // ----------------------------------------------------------------------------------------- Web
  {
    id: 'web-01',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md',
    date: '2026-02-05',
    model: 'ggml-org/gpt-oss-120b-GGUF',
    simulatedModel: 'GPT_OSS_120B (hand-written spec)',
    quant: 'MXFP4 experts (59.02 GiB file)',
    hardware: '1× DGX Spark (GB10)',
    engine: 'llama.cpp build 7941 (CUDA 13), llama-bench -fa 1 -ub 2048',
    parallelism: 'single device',
    batch: 1,
    promptLen: 2048,
    genLen: 32,
    contextDepth: 0,
    prefill: { tokensPerSec: 2443.91, kind: 'perSequence' },
    decode: { tokensPerSec: 58.72, kind: 'perSequence' },
    weightFileBytes: 59.02 * GIB,
    caveats: 'Official llama.cpp Spark benchmark file. tg32 is a short window.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'dgx-spark' },
      workload: { promptLen: 2048, genLen: 32 },
    },
  },
  {
    id: 'web-02',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md',
    date: '2026-02-05',
    model: 'ggml-org/gpt-oss-120b-GGUF',
    simulatedModel: 'GPT_OSS_120B (hand-written spec)',
    quant: 'MXFP4 experts',
    hardware: '1× DGX Spark (GB10)',
    engine: 'llama.cpp build 7941 (CUDA 13), llama-bench pp2048 @ d32768, tg32 @ d32768',
    parallelism: 'single device',
    batch: 1,
    promptLen: 2048,
    genLen: 32,
    contextDepth: 32768,
    prefill: { tokensPerSec: 1567.08, kind: 'perSequence' },
    decode: { tokensPerSec: 42.76, kind: 'perSequence' },
    caveats: 'Prefill at depth 32768 is not modeled: the depth-0 prefill is simulated (a higher, still valid bound).',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'dgx-spark' },
      workload: { promptLen: 2048, genLen: 32, contextDepth: 32768 },
    },
  },
  {
    id: 'web-04',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md',
    date: '2026-02-05',
    model: 'ggml-org/gpt-oss-120b-GGUF',
    simulatedModel: 'GPT_OSS_120B (hand-written spec)',
    quant: 'MXFP4 experts',
    hardware: '1× DGX Spark (GB10)',
    engine: 'llama.cpp build 7941, llama-batched-bench B=32 PP=4096 TG=32',
    parallelism: 'single device',
    batch: 32,
    promptLen: 4096,
    genLen: 32,
    contextDepth: 4096,
    prefill: { tokensPerSec: 2411.23, kind: 'aggregate' },
    decode: {
      tokensPerSec: 262.2,
      kind: 'aggregate',
      check: {
        mode: 'exception',
        reason:
          'MoE routing at batch > 1: the sim routes "balanced" (docs §5 "MoE modes A/B"), so 32 × top-4 assignments ' +
          'touch min(E, T·k) = all 128 experts per layer and decode reads every weight (63 GB/step → 129 tok/s). ' +
          'Real routing touches fewer distinct experts (uniform random: 128·(1 − (1 − 4/128)³²) ≈ 82; skewed real ' +
          'routing fewer still): 262 tok/s needs ≤ 33 GB/step at 273 GB/s. A modelling limit, not a preset error.',
      },
    },
    caveats:
      'S_PP / S_TG columns: aggregate over 32 sequences (8.2 tok/s per stream). batched-bench feeds random tokens ' +
      '(since llama.cpp #15398, Aug 2025), so the sequences do not share their routing.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'dgx-spark' },
      workload: { batch: 32, promptLen: 4096, genLen: 32, contextDepth: 4096 },
    },
  },
  {
    id: 'web-07',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1o7k7zz/dgx_spark_compiled_llamacpp_benchmarks_compared/',
    date: '2025-10-15',
    model: 'zai-org/GLM-4.5-Air (GGUF Q4_K)',
    simulatedModel: 'GLM_4_5_AIR (hand-written spec, MTP layer excluded)',
    quant: 'Q4_K (67.85 GiB → 5.45 bpw average over 106.9 G params)',
    hardware: '1× DGX Spark (GB10)',
    engine: 'llama.cpp CUDA, early Spark build (Oct 2025), llama-bench',
    parallelism: 'single device',
    batch: 1,
    promptLen: 2048,
    genLen: 32,
    contextDepth: 0,
    prefill: { tokensPerSec: 841.44, kind: 'perSequence' },
    decode: { tokensPerSec: 22.59, kind: 'perSequence' },
    weightFileBytes: 67.85 * GIB,
    caveats: 'Early build (lower than the refreshed official file); read through a redlib mirror.',
    input: {
      model: { kind: 'spec', spec: GLM_4_5_AIR },
      quant: { kind: 'preset', preset: ggufBpw(5.45, 'file size 67.85 GiB / 106.85 G params') },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'dgx-spark' },
      workload: { promptLen: 2048, genLen: 32 },
    },
  },
  {
    id: 'web-12',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1o7k7zz/dgx_spark_compiled_llamacpp_benchmarks_compared/',
    date: '2025-10-15',
    model: 'ggml-org/gpt-oss-120b-GGUF',
    simulatedModel: 'GPT_OSS_120B (hand-written spec)',
    quant: 'MXFP4 experts',
    hardware: 'Apple M4 Max (cores / RAM not stated; ≥ 64 GB needed for the 63 GB file → 40-core GPU)',
    engine: 'llama.cpp Metal (second-hand numbers), llama-bench',
    parallelism: 'single SoC',
    batch: 1,
    promptLen: 2048,
    genLen: 32,
    contextDepth: 0,
    prefill: { tokensPerSec: 871.16, kind: 'perSequence' },
    decode: { tokensPerSec: 62.85, kind: 'perSequence' },
    caveats: 'Second-hand (nitter post quoted on reddit); noisy (± 3.84 tok/s).',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m4-max-40c' },
      workload: { promptLen: 2048, genLen: 32 },
    },
  },
  {
    id: 'web-14',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/discussions/15396',
    date: '2025-08-18',
    model: 'ggml-org/gpt-oss-20b-GGUF',
    simulatedModel: 'GPT_OSS_20B (hand-written spec)',
    quant: 'MXFP4 experts (11.27 GiB)',
    hardware: 'Apple M3 Ultra 512 GB, 80-core GPU',
    engine: 'llama.cpp build 6310 (Metal), llama-bench -fa 1 -ub 2048',
    parallelism: 'single SoC',
    batch: 1,
    promptLen: 2048,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 2816.47, kind: 'perSequence' },
    decode: { tokensPerSec: 115.52, kind: 'perSequence' },
    weightFileBytes: 11.27 * GIB,
    caveats: 'Official llama.cpp gpt-oss guide table.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_20B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m3-ultra-80c' },
      workload: { promptLen: 2048, genLen: 128 },
    },
  },
  {
    id: 'web-26',
    sourceUrl: 'https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md',
    date: '2025-10-08',
    model: 'mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit',
    simulatedModel: 'QWEN3_30B_A3B (hand-written spec)',
    quant: 'MLX 4-bit (g64, 4.5 bpw)',
    hardware: 'Apple M4 Max 64 GB (64 GB is only sold with the 40-core GPU)',
    engine: 'mlx 0.29.2.dev / mlx-lm 0.28.2, mlx_lm.benchmark -p 2048 -g 128',
    parallelism: 'single SoC',
    batch: 1,
    promptLen: 2048,
    genLen: 128,
    contextDepth: 2048,
    prefill: { tokensPerSec: 1753.9, kind: 'perSequence' },
    decode: { tokensPerSec: 113.33, kind: 'perSequence' },
    caveats:
      'Official MLX table; peak memory 18.2 GB. mlx_lm.benchmark (v0.28.2) generates from the cache of its 2048-token ' +
      'prompt, so decode runs at 2048 + the generated tokens (unlike llama-bench, whose tg test starts empty).',
    input: {
      model: { kind: 'spec', spec: QWEN3_30B_A3B },
      quant: 'mlx-4bit',
      numerics: MLX,
      hardware: { deviceId: 'apple-m4-max-40c', memCapacityGb: 64 },
      workload: { promptLen: 2048, genLen: 128, contextDepth: 2048 },
    },
  },
  {
    id: 'web-29',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1m6b151/updated_strix_halo_ryzen_ai_max_395_llm_benchmark/',
    date: '2025-07-22',
    model: 'unsloth/Qwen3-30B-A3B-GGUF',
    simulatedModel: 'QWEN3_30B_A3B (hand-written spec)',
    quant: 'UD-Q4_K_XL (≈4.5–5 bpw; 4.85 assumed as Q4_K_M)',
    hardware: 'Framework Desktop, Ryzen AI Max+ 395, 128 GB LPDDR5x-8000 (poster measures ~215 GB/s)',
    engine: 'llama.cpp ~b5863 (Vulkan, fa=1), llama-bench pp512 / tg128',
    parallelism: 'single APU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 604.8, kind: 'perSequence' },
    decode: { tokensPerSec: 66.3, kind: 'perSequence' },
    caveats: 'Pre-production unit; prefill rose from 119 to 605 tok/s between May and July 2025 builds.',
    input: {
      model: { kind: 'spec', spec: QWEN3_30B_A3B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  {
    id: 'web-31',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1m6b151/updated_strix_halo_ryzen_ai_max_395_llm_benchmark/',
    date: '2025-07-22',
    model: 'shisa-ai Shisa V2 70B (Llama 3.x 70B), i1-Q4_K_M GGUF',
    simulatedModel: 'LLAMA3_70B (hand-written spec)',
    quant: 'i1-Q4_K_M',
    hardware: 'Framework Desktop, Ryzen AI Max+ 395, 128 GB',
    engine: 'llama.cpp ~b5863, llama-bench pp512 / tg128: pp from HIP rocWMMA, tg (best) backend',
    parallelism: 'single APU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 94.7, kind: 'perSequence' },
    decode: { tokensPerSec: 4.5, kind: 'perSequence' },
    caveats: 'pp and tg come from different backends (Vulkan fa=1: pp 26.4 / tg 5.0).',
    input: {
      model: { kind: 'spec', spec: LLAMA3_70B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  {
    id: 'web-36',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1x023z4/qwen38flashnext_on_6x3090_6x4090_without_nvlink/',
    date: '2026-10-07',
    model: 'unsloth/Qwen3.8-Flash-Next-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (fixture)',
    quant: 'UD-Q4_K_XL (~4.5 bpw per poster), KV q8_0',
    hardware: '6× RTX 4090 (rented, PCIe)',
    engine: 'llama.cpp upstream df03399b8 (stock, CUDA), llama-server',
    parallelism: 'layer split, no P2P',
    batch: 1,
    promptLen: 50000,
    genLen: 128,
    contextDepth: 50000,
    prefill: { tokensPerSec: 1728, kind: 'perSequence' },
    decode: { tokensPerSec: 46.2, kind: 'perSequence' },
    caveats:
      'Prefill of a 50k-token session; decode at 50k depth (generated length not stated; 128 assumed). The poster\'s ' +
      'patched fork reaches 7050 / 57.5.',
    input: {
      model: FLASH_NEXT,
      quant: { kind: 'preset', preset: ggufBpw(4.5, 'UD-Q4_K_XL ~4.5 bpw per the poster') },
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q8_0 },
      hardware: { deviceId: 'rtx-4090', topology: pcieBox(RTX_4090, 6, PCIE_4_X16) },
      parallel: { tp: 1, pp: 6 },
      workload: { promptLen: 50000, genLen: 128, contextDepth: 50000 },
    },
  },
  {
    id: 'web-41',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1r66jyp/vllm_maximum_performance_on_multi3090/',
    date: '2026-02-16',
    model: 'Qwen/Qwen3-Coder-Next-FP8',
    simulatedModel: 'QWEN3_NEXT_80B_A3B (hand-written spec)',
    quant: 'FP8 (Marlin on Ampere: weight-only)',
    hardware: '4× RTX 3090, PCIe ≥ Gen3 x8 / Gen4 x4 each, patched aikitoria P2P driver',
    engine: 'vLLM (version not stated), is_fully_connected patched',
    parallelism: 'TP4',
    batch: 1,
    promptLen: 512,
    genLen: 512,
    contextDepth: 512,
    decode: { tokensPerSec: 144.6, kind: 'perSequence' },
    caveats: 'One vLLM log line ("Avg generation throughput"), not an average. Prompt length not stated (512 assumed).',
    input: {
      model: { kind: 'spec', spec: QWEN3_NEXT_80B_A3B },
      quant: 'fp8-wo',
      numerics: VLLM,
      hardware: { deviceId: 'rtx-3090', topology: pcieBox(RTX_3090, 4, PCIE_4_X8), unlocks: { 'pcie-p2p': 'direct' } },
      parallel: { tp: 4 },
      workload: { promptLen: 512, genLen: 512, contextDepth: 512 },
    },
  },
  {
    id: 'web-42',
    sourceUrl: 'https://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html',
    date: '2025-03-10',
    model: 'Qwen/QwQ-32B (--quantization fp8)',
    simulatedModel: 'QWQ_32B (hand-written spec)',
    quant: 'FP8 dynamic (Marlin on Ampere: weight-only)',
    hardware: '4× RTX 3090, PCIe Gen4 x8 each, 300 W limit, NCCL_P2P_DISABLE=1',
    engine: 'vLLM (version not stated), benchmark_serving.py random dataset',
    parallelism: 'TP4, no P2P',
    batch: 1,
    promptLen: 1024,
    genLen: 128,
    contextDepth: 1024,
    decode: { tokensPerSec: 44, kind: 'perSequence' },
    caveats: '--max-num-seqs 1. Prompt / output lengths not stated: benchmark_serving random defaults (1024 / 128) assumed.',
    input: {
      model: { kind: 'spec', spec: QWQ_32B },
      quant: 'fp8-wo',
      numerics: VLLM,
      hardware: { deviceId: 'rtx-3090', topology: pcieBox(RTX_3090, 4, PCIE_4_X8) },
      parallel: { tp: 4 },
      workload: { promptLen: 1024, genLen: 128, contextDepth: 1024 },
    },
  },
  {
    id: 'web-44',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1jobe0u/benchmark_dualgpu_boosts_speed_despire_all_common/',
    date: '2025-03-31',
    model: 'Qwen/QwQ-32B-AWQ',
    simulatedModel: 'QWQ_32B (hand-written spec)',
    quant: 'AWQ 4-bit g128',
    hardware: '1× H100 80 GB HBM3 (SXM)',
    engine: 'vLLM (version not stated)',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 512,
    contextDepth: 512,
    decode: { tokensPerSec: 78, kind: 'perSequence' },
    caveats: 'Prompt / output lengths not stated (512 / 512 assumed; TTFT < 0.1 s).',
    input: {
      model: { kind: 'spec', spec: QWQ_32B },
      quant: 'int4-g128',
      numerics: VLLM,
      hardware: { deviceId: 'h100-sxm' },
      workload: { promptLen: 512, genLen: 512, contextDepth: 512 },
    },
  },
  {
    id: 'web-45',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1jobe0u/benchmark_dualgpu_boosts_speed_despire_all_common/',
    date: '2025-03-31',
    model: 'Qwen/QwQ-32B-AWQ',
    simulatedModel: 'QWQ_32B (hand-written spec)',
    quant: 'AWQ 4-bit g128',
    hardware: '1× RTX 5090',
    engine: 'vLLM (custom build for 5090)',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 512,
    contextDepth: 512,
    decode: { tokensPerSec: 65, kind: 'perSequence' },
    caveats: 'Prompt / output lengths not stated (512 / 512 assumed).',
    input: {
      model: { kind: 'spec', spec: QWQ_32B },
      quant: 'int4-g128',
      numerics: VLLM,
      hardware: { deviceId: 'rtx-5090' },
      workload: { promptLen: 512, genLen: 512, contextDepth: 512 },
    },
  },
  {
    id: 'web-48',
    sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/1jobe0u/benchmark_dualgpu_boosts_speed_despire_all_common/',
    date: '2025-03-31',
    model: 'Qwen/QwQ-32B-AWQ',
    simulatedModel: 'QWQ_32B (hand-written spec)',
    quant: 'AWQ 4-bit g128',
    hardware: '1× RTX 4090',
    engine: 'vLLM (version not stated)',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 512,
    contextDepth: 512,
    decode: { tokensPerSec: 43, kind: 'perSequence' },
    caveats: 'Prompt / output lengths not stated (512 / 512 assumed).',
    input: {
      model: { kind: 'spec', spec: QWQ_32B },
      quant: 'int4-g128',
      numerics: VLLM,
      hardware: { deviceId: 'rtx-4090' },
      workload: { promptLen: 512, genLen: 512, contextDepth: 512 },
    },
  },
  {
    id: 'web-50',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/discussions/14640',
    date: '2025-07-11',
    model: 'unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF',
    simulatedModel: 'REF_DENSE_8B (Qwen3-8B shape = the same config)',
    quant: 'Q4_K_M (4.68 GiB)',
    hardware: '1× AMD Instinct MI300X VF (AMD Developer Cloud)',
    engine: 'llama.cpp builds 5873–5898 (ROCm / HIP), llama-bench -ngl 999',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 5839.67, kind: 'perSequence' },
    decode: { tokensPerSec: 161.47, kind: 'perSequence' },
    weightFileBytes: 4.68 * GIB,
    caveats: 'llama.cpp ROCm kernels are not vendor-tuned for CDNA3.',
    input: {
      model: { kind: 'spec', spec: REF_DENSE_8B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'instinct-mi300x' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  {
    id: 'web-51',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/discussions/14640',
    date: '2025-07-11',
    model: 'unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF',
    simulatedModel: 'LLAMA3_70B (hand-written spec)',
    quant: 'Q4_K_M (39.59 GiB)',
    hardware: '1× AMD Instinct MI300X VF',
    engine: 'llama.cpp builds 5873–5898 (ROCm / HIP), llama-bench -ngl 999',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 949.85, kind: 'perSequence' },
    decode: { tokensPerSec: 36.31, kind: 'perSequence' },
    weightFileBytes: 39.59 * GIB,
    caveats: 'llama.cpp ROCm kernels are not vendor-tuned for CDNA3.',
    input: {
      model: { kind: 'spec', spec: LLAMA3_70B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'instinct-mi300x' },
      workload: { promptLen: 512, genLen: 128 },
    },
  },
  // ------------------------------------------------------------------------------- offload (design doc §8.2 S-1)
  {
    id: 'web-52',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/pull/29887',
    date: '2026-10-03',
    model: 'Qwen3.8-Flash-Next GGUF Q4_0 (93.7 GiB, 65.4 GiB of routed experts)',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (fixture) at the q4_0 preset',
    quant: 'Q4_0',
    hardware: '1× RTX 5090 32 GB on PCIe 4.0 x16, EPYC 7742 host (16 cores on the GPU\'s NUMA node)',
    engine: 'llama.cpp PR 29887 llama-server -fa on -cmoe --moe-cache-mib 19000 (GPU LRU cache of host experts), no MTP',
    parallelism: 'single GPU',
    batch: 1,
    promptLen: 1,
    genLen: 512,
    contextDepth: 1,
    decode: { tokensPerSec: 67.8, kind: 'perSequence' },
    caveats:
      'Every routed expert lives in host RAM (-cmoe); a 19000 MiB GPU cache holds f = 19000 MiB / 65.4 GiB = 0.284 of ' +
      'them, at a measured 89% hit rate (custom h 0.89). Misses cross PCIe 4.0 x16 (28.8 GB/s payload, the platform link, ' +
      'not the card\'s Gen5). The EPYC 7742 host RAM (8× DDR4-3200) has no preset; DDR5-6000 2ch (96 GB/s) stands in, ' +
      'which gives the same result since the link binds, and its 96 GB capacity makes the host tier not fit (not ' +
      'checked here). The n-gram (PLE) table sits in host RAM as llama.cpp keeps it. SPEED-Bench qualitative prompts ' +
      '(--osl 512, temp 0) have no stated length: decode is simulated from an empty cache, the favourable end.',
    input: {
      model: FLASH_NEXT,
      quant: 'q4_0',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-5090', host: { link: hostPcieLink(4, 16), ram: DDR5_6000_2CH } },
      workload: { promptLen: 1, genLen: 512, contextDepth: 1 },
      deployment: {
        offload: {
          routedExperts: { tier: 'host_ram', residentFraction: 19000 / (65.4 * 1024), hitRatio: { kind: 'custom', h: 0.89 } },
          lookupTables: { tier: 'host_ram' },
        },
      },
    },
  },
  {
    id: 'web-53',
    sourceUrl: 'https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF',
    date: '2025-05',
    model: 'ubergarm/Qwen3-235B-A22B-GGUF mix-IQ3_K (106.830 GiB, 3.903 bpw)',
    simulatedModel: 'QWEN3_235B_A22B (hand-written spec)',
    quant: 'mix-IQ3_K: attention IQ6_K, routed gate/up IQ3_K, down IQ4_K, token_embd / output Q8_0',
    hardware: '1× RTX 3090 Ti FE 24 GB, Ryzen 9 9950X, 96 GB DDR5-6400 (2×48 GB)',
    engine: 'ik_llama.cpp llama-sweep-bench -c 32768 -b 4096 -ub 4096 -rtr, -ot ffn of layers 10–93 = CPU',
    parallelism: 'single GPU + CPU experts',
    batch: 1,
    promptLen: 4096,
    genLen: 1024,
    contextDepth: 0,
    prefill: { tokensPerSec: 140, kind: 'perSequence' },
    decode: {
      tokensPerSec: 10,
      kind: 'perSequence',
      check: {
        mode: 'exception',
        reason:
          'CPU-compute offload: ik_llama.cpp runs the experts of layers 10–93 on the CPU, reading them from DDR5 at ' +
          '102.4 GB/s; the simulator\'s transfer mode streams the missed experts over the PCIe 4.0 x16 link (28.8 GB/s payload) ' +
          'instead (cpu_compute is a hook validation rejects, design doc §2.8), so this decode is not bounded by it',
      },
    },
    caveats:
      'The page quotes "up to 140 tok/sec PP and 10 tok/sec TG" (sweep-bench at N_KV 0, -ub 4096: pp 4096, tg 1024 ' +
      'tokens). Simulated on the RTX 3090 preset (no 3090 Ti preset; 936 vs 1008 GB/s, prefill is link-bound anyway). ' +
      'Experts of layers 10–93 in host RAM with none resident (f 0); prefill streams them once per 4096-token ubatch, ' +
      'which bounds both a GPU-offloaded and a CPU-computed prefill from above here. The host tier (96 GB) is ' +
      'smaller than the offloaded experts (the run mmaps the file; not checked). IQ_K bits per weight: IQ3_K 3.4375, ' +
      'IQ4_K 4.5, IQ6_K 6.625 (ik_llama.cpp block sizes).',
    input: {
      model: { kind: 'spec', spec: QWEN3_235B_A22B },
      quant: { kind: 'preset', preset: UBERGARM_MIX_IQ3_K },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR5_6400_2CH } },
      workload: { promptLen: 4096, genLen: 1024, contextDepth: 0 },
      deployment: {
        offload: {
          routedExperts: { tier: 'host_ram', layers: { from: 10, to: 94 }, residentFraction: 0, hitRatio: { kind: 'uniform' } },
        },
      },
    },
  },
  // ---------------------------------------------------------------------- Fresh review collection (2026-11-09)
  // single GPU; collection row F-LM01
  {
    id: 'F-LM01',
    sourceUrl:
      'https://localmaxxing.com/en/models/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF?run=cmw165mx20aisoa01hs07mfwe',
    date: '2026-11-03',
    model: '0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'Q4_K_M',
    hardware:
      '1x RTX 3090, VRAM 24 GB each; CPU Intel Core i7-9700K (8C); host RAM 32 GB; OS Ubuntu 24.04.5; class DISCRETE_GPU',
    engine: 'llama.cpp b12030',
    parallelism: 'tensorParallel=1, gpuLayers=999',
    batch: 1,
    promptLen: 231,
    genLen: 512,
    contextDepth: 231,
    prefill: {
      tokensPerSec: 713.9,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 43.8, kind: 'perSequence' },
    caveats:
      'Abliterated fine-tune with the same base-model shapes. Report states gpu-unlock, 420 W, Q4_0 KV. Decode follows the prompt.',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q4_0 },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { batch: 1, promptLen: 231, genLen: 512, contextDepth: 231 },
    },
  },
  // single GPU; collection row F-LM02
  {
    id: 'F-LM02',
    sourceUrl:
      'https://localmaxxing.com/en/models/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF?run=cmw1732ul0ajloa01fxuehrcn',
    date: '2026-11-03',
    model: '0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'Q4_K_M',
    hardware:
      '1x RTX 3090, VRAM 24 GB each; CPU Intel Core i7-9700K (8C); host RAM 32 GB; OS Ubuntu 24.04.5; class DISCRETE_GPU',
    engine: 'llama.cpp b12030',
    parallelism: 'tensorParallel=1, gpuLayers=999',
    batch: 1,
    promptLen: 8192,
    genLen: 512,
    contextDepth: 8192,
    prefill: { tokensPerSec: 1490.3, kind: 'perSequence' },
    decode: { tokensPerSec: 41.7, kind: 'perSequence' },
    caveats:
      'Abliterated fine-tune with the same base-model shapes. Report states gpu-unlock, 420 W, Q4_0 KV. Decode follows the prompt.',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q4_0 },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { batch: 1, promptLen: 8192, genLen: 512, contextDepth: 8192 },
    },
  },
  // single GPU; collection row F-LM03
  {
    id: 'F-LM03',
    sourceUrl:
      'https://localmaxxing.com/en/models/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF?run=cmw17990y0ajvoa01v0c2hzci',
    date: '2026-11-03',
    model: '0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'Q4_K_M',
    hardware:
      '1x RTX 3090, VRAM 24 GB each; CPU Intel Core i7-9700K (8C); host RAM 32 GB; OS Ubuntu 24.04.5; class DISCRETE_GPU',
    engine: 'llama.cpp b12030',
    parallelism: 'tensorParallel=1, gpuLayers=999',
    batch: 1,
    promptLen: 32763,
    genLen: 512,
    contextDepth: 32763,
    prefill: { tokensPerSec: 1398.7, kind: 'perSequence' },
    decode: { tokensPerSec: 36.5, kind: 'perSequence' },
    caveats:
      'Abliterated fine-tune with the same base-model shapes. Report states gpu-unlock, 420 W, Q4_0 KV. Decode follows the prompt.',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q4_0 },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { batch: 1, promptLen: 32763, genLen: 512, contextDepth: 32763 },
    },
  },
  // single GPU; collection row F-LM04
  {
    id: 'F-LM04',
    sourceUrl:
      'https://localmaxxing.com/en/models/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF?run=cmw179bz90ajyoa01k39k8dmf',
    date: '2026-11-03',
    model: '0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'Q4_K_M',
    hardware:
      '1x RTX 3090, VRAM 24 GB each; CPU Intel Core i7-9700K (8C); host RAM 32 GB; OS Ubuntu 24.04.5; class DISCRETE_GPU',
    engine: 'llama.cpp b12030',
    parallelism: 'tensorParallel=1, gpuLayers=999',
    batch: 1,
    promptLen: 261095,
    genLen: 512,
    contextDepth: 261095,
    prefill: { tokensPerSec: 705.8, kind: 'perSequence' },
    decode: { tokensPerSec: 16.5, kind: 'perSequence' },
    caveats:
      'Abliterated fine-tune with the same base-model shapes. Report states gpu-unlock, 420 W, Q4_0 KV. Decode follows the prompt.',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q4_0 },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { batch: 1, promptLen: 261095, genLen: 512, contextDepth: 261095 },
    },
  },
  // single GPU; collection row F-LM05
  {
    id: 'F-LM05',
    sourceUrl:
      'https://localmaxxing.com/en/models/openai/gpt-oss-20b?run=cmvubed2h0hl9s601wwfuy0kf',
    date: '2026-10-30',
    model: 'openai/gpt-oss-20b',
    simulatedModel: 'GPT_OSS_20B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware:
      '1x RTX 3090, VRAM 24 GB each; CPU Intel Core i7-9700K (8C); host RAM 32 GB; OS Ubuntu 24.04.5; class DISCRETE_GPU',
    engine: 'vllm 0.32.0',
    parallelism: 'tensorParallel=1',
    batch: 1,
    promptLen: 266,
    genLen: 512,
    contextDepth: 266,
    prefill: {
      tokensPerSec: 4195,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 151.5, kind: 'perSequence' },
    caveats:
      'Native MXFP4 routed experts; non-expert tensors BF16 in vLLM (not GGUF Q8_0). int4_per_token_head KV approximated at 4 bpw in LM05 (scale overhead omitted, favourable). gpu-unlock stated.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_20B },
      quant: {
        kind: 'preset',
        preset: { ...GPT_OSS_GGUF, weights: { dtype: 'bf16' }, embeddings: { dtype: 'bf16' } },
      },
      numerics: { ...VLLM, kvCache: bpwFormat(4) },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { batch: 1, promptLen: 266, genLen: 512, contextDepth: 266 },
    },
  },
  // single GPU; collection row F-LM06
  {
    id: 'F-LM06',
    sourceUrl:
      'https://localmaxxing.com/en/models/openai/gpt-oss-20b?run=cmvu9dd7f0hbks6014go6jvy5',
    date: '2026-10-30',
    model: 'openai/gpt-oss-20b',
    simulatedModel: 'GPT_OSS_20B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware:
      '1x RTX 3090, VRAM 24 GB each; CPU Intel Core i7-9700K (8C); host RAM 32 GB; OS Ubuntu 24.04.5; class DISCRETE_GPU',
    engine: 'vllm 0.32.0',
    parallelism: 'tensorParallel=1',
    batch: 1,
    promptLen: 266,
    genLen: 512,
    contextDepth: 266,
    prefill: {
      tokensPerSec: 6816.5,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 177.8, kind: 'perSequence' },
    caveats:
      'Native MXFP4 routed experts; non-expert tensors BF16 in vLLM (not GGUF Q8_0). int4_per_token_head KV approximated at 4 bpw in LM05 (scale overhead omitted, favourable). gpu-unlock stated.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_20B },
      quant: {
        kind: 'preset',
        preset: { ...GPT_OSS_GGUF, weights: { dtype: 'bf16' }, embeddings: { dtype: 'bf16' } },
      },
      numerics: { ...VLLM, kvCache: { dtype: 'bf16' } },
      hardware: { deviceId: 'rtx-3090', unlocks: { 'tensor-rate': 'full-rate' } },
      workload: { batch: 1, promptLen: 266, genLen: 512, contextDepth: 266 },
    },
  },
  // multi-GPU PCIe; collection row F-LM10
  {
    id: 'F-LM10',
    sourceUrl:
      'https://localmaxxing.com/en/models/ggml-org/gpt-oss-20b-GGUF?run=cmvqkqw4e0xumng01racygmd3',
    date: '2026-10-27',
    model: 'ggml-org/gpt-oss-20b-GGUF',
    simulatedModel: 'GPT_OSS_20B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware:
      '2x RTX 3090, VRAM 24 GB each; CPU Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz; host RAM None GB; OS linux; class DISCRETE_GPU',
    engine: 'llama.cpp None',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 334,
    genLen: 512,
    contextDepth: 334,
    prefill: {
      tokensPerSec: 2573.1,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 195.3, kind: 'perSequence' },
    caveats:
      'Two GPUs reported; split unspecified. Favourable TP2 and PCIe4 x16 direct P2P, not an assertion that the engine used them. Short client TTFT excluded.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_20B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', topology: pcieBox(RTX_3090, 2, PCIE_4_X16_DIRECT) },
      workload: { batch: 1, promptLen: 334, genLen: 512, contextDepth: 334 },
      parallel: { tp: 2 },
    },
  },
  // spec-dec; collection row F-LM11
  {
    id: 'F-LM11',
    sourceUrl:
      'https://localmaxxing.com/en/models/Frozenlock/Qwen3.8-27B-int4-AutoRound?run=cmu9j9vce08xnlq01c9e1ba6l',
    date: '2026-09-20',
    model: 'Frozenlock/Qwen3.8-27B-int4-AutoRound',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'INT4_W4A16',
    hardware:
      '2x RTX 3090, VRAM 24 GB each; CPU AMD Ryzen 5 3600X 6-Core Processor; host RAM 128 GB; OS Ubuntu 26.04.1 LTS; class DISCRETE_GPU',
    engine: 'vllm 0.29.0',
    parallelism: 'tensorParallel=2, pipelineParallel=1',
    batch: 1,
    promptLen: 304,
    genLen: 512,
    contextDepth: 304,
    decode: { tokensPerSec: 188.7763, kind: 'perSequence' },
    caveats:
      'AutoRound W4A16 approximated by int4 g128; same base-model shapes. DFlash2 k7 with the existing block-8 drafter fixture; acceptance not reported, prose default 0.79. LM12 explicitly PCIe4 x8/x8, no P2P, gpu-unlock; LM11 link unspecified so direct x16 assumed.',
    input: {
      model: QWEN27,
      quant: 'int4-g128',
      numerics: { ...VLLM, kvCache: { dtype: 'fp8_e4m3' } },
      hardware: { deviceId: 'rtx-3090', topology: pcieBox(RTX_3090, 2, PCIE_4_X16_DIRECT) },
      workload: { batch: 1, promptLen: 304, genLen: 512, contextDepth: 304 },
      parallel: { tp: 2 },
      deployment: {
        speculative: {
          method: 'dflash2',
          drafter: { kind: 'repo', repoId: 'incoai/Qwen3.8-27B-DFlash2' },
          draftLength: 7,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.dflash2.prose.alpha },
        },
      },
    },
  },
  // spec-dec; collection row F-LM12
  {
    id: 'F-LM12',
    sourceUrl:
      'https://localmaxxing.com/en/models/noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound?run=cmvzr3fg609ytoa0153s1v25m',
    date: '2026-11-02',
    model: 'noon-at-cgn/Qwen3.8-27B-Uncensored-W4A16-AutoRound',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'W4A16',
    hardware:
      '2x RTX 3090, VRAM 24 GB each; CPU AMD Ryzen 7 9700X; host RAM 64 GB; OS CachyOS; class DISCRETE_GPU',
    engine: 'vllm 0.31.0',
    parallelism: 'tensorParallel=2',
    batch: 1,
    promptLen: 17201,
    genLen: 512,
    contextDepth: 17201,
    prefill: { tokensPerSec: 2920, kind: 'perSequence' },
    decode: { tokensPerSec: 194.3, kind: 'perSequence' },
    caveats:
      'AutoRound W4A16 approximated by int4 g128; same base-model shapes. DFlash2 k7 with the existing block-8 drafter fixture; acceptance not reported, prose default 0.79. LM12 explicitly PCIe4 x8/x8, no P2P, gpu-unlock; LM11 link unspecified so direct x16 assumed.',
    input: {
      model: QWEN27,
      quant: 'int4-g128',
      numerics: { ...VLLM, kvCache: { dtype: 'fp8_e4m3' } },
      hardware: {
        deviceId: 'rtx-3090',
        topology: pcieBox(RTX_3090, 2, PCIE_4_X8),
        unlocks: { 'tensor-rate': 'full-rate' },
      },
      workload: { batch: 1, promptLen: 17201, genLen: 512, contextDepth: 17201 },
      parallel: { tp: 2 },
      deployment: {
        speculative: {
          method: 'dflash2',
          drafter: { kind: 'repo', repoId: 'incoai/Qwen3.8-27B-DFlash2' },
          draftLength: 7,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.dflash2.prose.alpha },
        },
      },
    },
  },
  // spec-dec; collection row F-LM13
  {
    id: 'F-LM13',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B?run=cmu2x8yi3068xlq01l9esdp49',
    date: '2026-09-15',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'Q8_0',
    hardware:
      '3x Radeon AI Pro R9700, VRAM 32 GB each; CPU AMD Ryzen 9 5950X; host RAM 64 GB; OS Arch Linux; class DISCRETE_GPU',
    engine: 'llama.cpp 790cf51aa + stew675 rdna-boosts v16-790cf51aa-r5 (16 blocks), ROCm 7.14',
    parallelism: 'gpuLayers=99',
    batch: 1,
    promptLen: 30,
    genLen: 400,
    contextDepth: 30,
    decode: { tokensPerSec: 90.561, kind: 'perSequence' },
    caveats:
      'Three cards installed, but notes explicitly say only TWO compute cards; desktop GPU excluded. Decode follows its 30-token prompt. The independently measured pp2520 prefill is recorded in F-LM13-pp2520, not compared with this short-prompt workload.',
    input: {
      model: QWEN27,
      quant: 'q8_0',
      numerics: LLAMACPP_MMQ,
      hardware: {
        deviceId: 'radeon-ai-pro-r9700',
        topology: pcieBox(RADEON_AI_PRO_R9700, 2, PCIE_4_X16_DIRECT),
      },
      workload: { batch: 1, promptLen: 30, genLen: 400, contextDepth: 30 },
      parallel: { tp: 2 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 6,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha },
        },
      },
    },
  },
  // spec-dec; collection row F-LM13 (separate pp2520)
  {
    id: 'F-LM13-pp2520',
    sourceUrl: 'https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B?run=cmu2x8yi3068xlq01l9esdp49',
    date: '2026-09-15',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'Q8_0',
    hardware:
      '3x Radeon AI Pro R9700, VRAM 32 GB each; CPU AMD Ryzen 9 5950X; host RAM 64 GB; OS Arch Linux; class DISCRETE_GPU',
    engine: 'llama.cpp 790cf51aa + stew675 rdna-boosts v16-790cf51aa-r5 (16 blocks), ROCm 7.14',
    parallelism: 'gpuLayers=99',
    batch: 1,
    promptLen: 2520,
    genLen: 400,
    contextDepth: 2520,
    prefill: { tokensPerSec: 1407.21, kind: 'perSequence' },
    caveats:
      'Separate llama-benchy pp2520 measurement from the same two-compute-card R9700 server. Prefill only; decode is recorded once in F-LM13 at its actual 30-token prompt. MTP deployment remains enabled as in the source.',
    input: {
      model: QWEN27,
      quant: 'q8_0',
      numerics: LLAMACPP_MMQ,
      hardware: {
        deviceId: 'radeon-ai-pro-r9700',
        topology: pcieBox(RADEON_AI_PRO_R9700, 2, PCIE_4_X16_DIRECT),
      },
      workload: { batch: 1, promptLen: 2520, genLen: 400, contextDepth: 2520 },
      parallel: { tp: 2 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 6,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha },
        },
      },
    },
  },
  // spec-dec; collection row F-LM17
  {
    id: 'F-LM17',
    sourceUrl:
      'https://localmaxxing.com/en/models/devan-carlin/Qwen3.8-Flash-Next-W4A16?run=cmvdrt2la005vrf018ecajmtx',
    date: '2026-10-18',
    model: 'devan-carlin/Qwen3.8-Flash-Next-W4A16',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (HF fixture; same base-model shapes)',
    quant: 'INT4_W4A16',
    hardware:
      '4x Intel Arc Pro B65, VRAM 32 GB each; CPU AMD EPYC 7343 16-Core Processor; host RAM 499 GB; OS Ubuntu 26.04.1 LTS; class DISCRETE_GPU',
    engine: 'vllm 0.30.0',
    parallelism: 'tensorParallel=4',
    batch: 1,
    promptLen: 348,
    genLen: 256,
    contextDepth: 348,
    prefill: {
      tokensPerSec: 1992.4,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 122.5, kind: 'perSequence' },
    caveats:
      'W4A16 AutoRound g128 surrogate; same base shapes. PCIe4 x16 direct P2P favourable assumption on EPYC host; generic existing pcieBox topology. Native MTP k3 in LM17, no acceptance supplied, multi-step prose default.',
    input: {
      model: FLASH_NEXT,
      quant: 'int4-g128',
      numerics: VLLM,
      hardware: { deviceId: 'arc-pro-b65', topology: pcieBox(ARC_PRO_B65, 4, PCIE_4_X16_DIRECT) },
      workload: { batch: 1, promptLen: 348, genLen: 256, contextDepth: 348 },
      parallel: { tp: 4 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 3,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha },
        },
      },
    },
  },
  // multi-GPU PCIe; collection row F-LM18
  {
    id: 'F-LM18',
    sourceUrl:
      'https://localmaxxing.com/en/models/devan-carlin/Qwen3.8-Flash-Next-W4A16?run=cmvdrszqa005srf01xy6l4fty',
    date: '2026-10-18',
    model: 'devan-carlin/Qwen3.8-Flash-Next-W4A16',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (HF fixture; same base-model shapes)',
    quant: 'INT4_W4A16',
    hardware:
      '4x Intel Arc Pro B65, VRAM 32 GB each; CPU AMD EPYC 7343 16-Core Processor; host RAM 499 GB; OS Ubuntu 26.04.1 LTS; class DISCRETE_GPU',
    engine: 'vllm 0.30.0',
    parallelism: 'tensorParallel=4',
    batch: 1,
    promptLen: 344,
    genLen: 256,
    contextDepth: 344,
    prefill: {
      tokensPerSec: 1977.7,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 44.5, kind: 'perSequence' },
    caveats:
      'W4A16 AutoRound g128 surrogate; same base shapes. PCIe4 x16 direct P2P favourable assumption on EPYC host; generic existing pcieBox topology. Native MTP k3 in LM17, no acceptance supplied, multi-step prose default.',
    input: {
      model: FLASH_NEXT,
      quant: 'int4-g128',
      numerics: VLLM,
      hardware: { deviceId: 'arc-pro-b65', topology: pcieBox(ARC_PRO_B65, 4, PCIE_4_X16_DIRECT) },
      workload: { batch: 1, promptLen: 344, genLen: 256, contextDepth: 344 },
      parallel: { tp: 4 },
    },
  },
  // single Mac; collection row F-LM21
  {
    id: 'F-LM21',
    sourceUrl:
      'https://localmaxxing.com/en/models/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF?run=cmtmy9uzd004wpb01x1p14j7x',
    date: '2026-09-04',
    model: 'Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'Q8_0',
    hardware:
      '1x None (M5 Max, 128 GB unified); CPU Apple M5 Max; host RAM None GB; OS darwin; class UNIFIED',
    engine: 'llama.cpp None',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 378,
    genLen: 256,
    contextDepth: 378,
    prefill: {
      tokensPerSec: 1154.4,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 17.3, kind: 'perSequence' },
    caveats:
      'Batch 1; source context and quantization mapped without fitting. Client TTFT prefill is diagnostic only on short prompts. ',
    input: {
      model: QWEN27,
      quant: 'q8_0',
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m5-max-40c' },
      workload: { batch: 1, promptLen: 378, genLen: 256, contextDepth: 378 },
    },
  },
  // single Mac; collection row F-LM25
  {
    id: 'F-LM25',
    sourceUrl:
      'https://localmaxxing.com/en/models/unsloth/Qwen3-Coder-Next?run=cmofnviki000ll704ogjm0ei2',
    date: '2026-04-26',
    model: 'unsloth/Qwen3-Coder-Next',
    simulatedModel: 'QWEN3_NEXT_80B_A3B (existing hand-written spec)',
    quant: 'Q5_K_XL',
    hardware:
      '1x None (M3 Ultra, 512 GB unified); CPU None; host RAM None GB; OS macOS 15.4; class UNIFIED',
    engine: 'llama.cpp b8920',
    parallelism: 'gpuLayers=99',
    batch: 1,
    promptLen: 59,
    genLen: 256,
    contextDepth: 59,
    decode: { tokensPerSec: 57.14, kind: 'perSequence' },
    caveats:
      'Qwen3-Coder-Next mapped to existing Qwen3-Next-80B-A3B sibling spec. Q5_K_XL approximated by Q5_K_M; module mix not published. Decode only.',
    input: {
      model: { kind: 'spec', spec: QWEN3_NEXT_80B_A3B },
      quant: 'q5_k_m',
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m3-ultra-80c' },
      workload: { batch: 1, promptLen: 59, genLen: 256, contextDepth: 59 },
    },
  },
  // single Mac; collection row F-LM26
  {
    id: 'F-LM26',
    sourceUrl:
      'https://localmaxxing.com/en/models/Qwen/Qwen3-30B-A3B?run=cmosdr72m0005l50423g2a02c',
    date: '2026-05-05',
    model: 'Qwen/Qwen3-30B-A3B',
    simulatedModel: 'QWEN3_30B_A3B (existing hand-written spec)',
    quant: 'Q4_K_M',
    hardware:
      '1x None (M3 Ultra, 512 GB unified); CPU None; host RAM None GB; OS macOS 15.4; class UNIFIED',
    engine: 'llama.cpp None',
    parallelism: 'gpuLayers=99',
    batch: 1,
    promptLen: 50,
    genLen: 184,
    contextDepth: 50,
    decode: { tokensPerSec: 101.95, kind: 'perSequence' },
    caveats:
      'Batch 1; source context and quantization mapped without fitting. Client TTFT prefill is diagnostic only on short prompts. ',
    input: {
      model: { kind: 'spec', spec: QWEN3_30B_A3B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m3-ultra-80c' },
      workload: { batch: 1, promptLen: 50, genLen: 184, contextDepth: 50 },
    },
  },
  // spec-dec; collection row F-LM27
  {
    id: 'F-LM27',
    sourceUrl:
      'https://localmaxxing.com/en/models/grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit?run=cmvp6afl2024ung01yh36dlcd',
    date: '2026-10-26',
    model: 'grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (HF fixture; same base-model shapes)',
    quant: 'oQ4e',
    hardware:
      '1x None (M5 Ultra, 256 GB unified); CPU Apple M5 Ultra (30-core CPU, 64-core GPU); host RAM None GB; OS macOS 27.0.1; class UNIFIED',
    engine: 'mlx oMLX 0.7.2',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 304,
    genLen: 1024,
    contextDepth: 304,
    decode: { tokensPerSec: 156.37, kind: 'perSequence' },
    caveats:
      'Actual 64-core M5 Ultra preset, not the collector-suggested 80-core variant. oQ4e approximated by MLX 4-bit g64. Native MTP depth up to 3; accepted draft counts alone do not identify per-position acceptance, so method prose default used.',
    input: {
      model: FLASH_NEXT,
      quant: 'mlx-4bit',
      numerics: MLX,
      hardware: { deviceId: 'apple-m5-ultra-64c' },
      workload: { batch: 1, promptLen: 304, genLen: 1024, contextDepth: 304 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 3,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha },
        },
      },
    },
  },
  // spec-dec; collection row F-LM28
  {
    id: 'F-LM28',
    sourceUrl:
      'https://localmaxxing.com/en/models/grant-ai/GLM-5.3-Flash-Abliterated-MLX-4bit?run=cmvp6busl024zng01cgvtzh42',
    date: '2026-10-26',
    model: 'grant-ai/GLM-5.3-Flash-Abliterated-MLX-4bit',
    simulatedModel: 'zai-org/GLM-5.3-Flash (HF fixture; same base-model shapes)',
    quant: 'oQ4e',
    hardware:
      '1x None (M5 Ultra, 256 GB unified); CPU Apple M5 Ultra (30-core CPU, 64-core GPU); host RAM None GB; OS macOS 27.0.1; class UNIFIED',
    engine: 'mlx oMLX 0.7.2',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 283,
    genLen: 864,
    contextDepth: 283,
    decode: { tokensPerSec: 78.35, kind: 'perSequence' },
    caveats:
      'Actual 64-core M5 Ultra preset, not the collector-suggested 80-core variant. oQ4e approximated by MLX 4-bit g64. Native MTP depth up to 3; accepted draft counts alone do not identify per-position acceptance, so the default single-step acceptance uses alpha 0.85 at position 1 and the collapsing tail [0.30, 0.10] at positions 2 and 3.',
    input: {
      model: GLM53_FLASH,
      quant: 'mlx-4bit',
      numerics: MLX,
      hardware: { deviceId: 'apple-m5-ultra-64c' },
      workload: { batch: 1, promptLen: 283, genLen: 864, contextDepth: 283 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 3,
          acceptance: mtpTailAcceptance(ACCEPTANCE_PRESETS.mtp_single_step.prose.alpha, 1),
        },
      },
    },
  },
  // Spark; collection row F-LM31
  {
    id: 'F-LM31',
    sourceUrl:
      'https://localmaxxing.com/en/models/RadixArk/Qwen3.8-27B-NVFP4?run=cmtlqp41100s7oe0109hzfmvk',
    date: '2026-09-03',
    model: 'RadixArk/Qwen3.8-27B-NVFP4',
    simulatedModel: 'nvidia/Qwen3.8-27B-NVFP4 (existing HF fixture)',
    quant: 'NVFP4',
    hardware:
      '1x None (GB10 Grace Blackwell, 128 GB unified); CPU None; host RAM None GB; OS Ubuntu 24.04; class UNIFIED',
    engine: 'sglang 0.0.0.dev1+g5f55db35e',
    parallelism: 'tensorParallel=1',
    batch: 1,
    promptLen: 1,
    genLen: 512,
    contextDepth: 1,
    decode: { tokensPerSec: 12.37, kind: 'perSequence' },
    caveats:
      'RadixArk checkpoint mapped to existing NVIDIA NVFP4 quant fixture. Prompt length recorded as zero/missing; use one token (favourable minimum), not configured 131072 window. Decode only.',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: { ...VLLM, kvCache: { dtype: 'fp8_e4m3' } },
      hardware: { deviceId: 'dgx-spark' },
      workload: { batch: 1, promptLen: 1, genLen: 512, contextDepth: 1 },
    },
  },
  // Strix Halo; collection row F-LM35
  {
    id: 'F-LM35',
    sourceUrl:
      'https://localmaxxing.com/en/models/ggml-org/gpt-oss-120b-GGUF?run=cmw18ryrz0an1oa01bw8k6c5u',
    date: '2026-11-03',
    model: 'ggml-org/gpt-oss-120b-GGUF',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware:
      '1x None (Ryzen AI Max 395+, 96 GB unified); CPU AMD Ryzen AI Max+ 395 w/ Radeon 8060S; host RAM None GB; OS Ubuntu 24.04; class UNIFIED',
    engine: 'llama.cpp b11901',
    parallelism: 'gpuLayers=99',
    batch: 1,
    promptLen: 335,
    genLen: 256,
    contextDepth: 335,
    prefill: {
      tokensPerSec: 425,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 51.7, kind: 'perSequence' },
    caveats:
      'Vulkan on Ryzen AI Max+ 395. Largest existing memory option used; reported 96 GB allocation is a fit detail, not a changed bandwidth. MXFP4 experts/Q8_0 non-experts.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { batch: 1, promptLen: 335, genLen: 256, contextDepth: 335 },
    },
  },
  // Strix Halo; collection row F-LM36
  {
    id: 'F-LM36',
    sourceUrl:
      'https://localmaxxing.com/en/models/ggml-org/gpt-oss-20b-GGUF?run=cmw18rzv60an8oa01cv0mbhlu',
    date: '2026-11-03',
    model: 'ggml-org/gpt-oss-20b-GGUF',
    simulatedModel: 'GPT_OSS_20B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware:
      '1x None (Ryzen AI Max 395+, 96 GB unified); CPU AMD Ryzen AI Max+ 395 w/ Radeon 8060S; host RAM None GB; OS Ubuntu 24.04; class UNIFIED',
    engine: 'llama.cpp b11901',
    parallelism: 'gpuLayers=99',
    batch: 1,
    promptLen: 334,
    genLen: 256,
    contextDepth: 334,
    prefill: {
      tokensPerSec: 1193.1,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 72.3, kind: 'perSequence' },
    caveats:
      'Vulkan on Ryzen AI Max+ 395. Largest existing memory option used; reported 96 GB allocation is a fit detail, not a changed bandwidth. MXFP4 experts/Q8_0 non-experts.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_20B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { batch: 1, promptLen: 334, genLen: 256, contextDepth: 334 },
    },
  },
  // Strix Halo; collection row F-LM37
  {
    id: 'F-LM37',
    sourceUrl:
      'https://localmaxxing.com/en/models/unsloth/Qwen3-30B-A3B-GGUF?run=cmsph6us000r3mp01z0gvxd76',
    date: '2026-08-12',
    model: 'unsloth/Qwen3-30B-A3B-GGUF',
    simulatedModel: 'QWEN3_30B_A3B (existing hand-written spec)',
    quant: 'Q4_K_XL',
    hardware:
      '1x None (Ryzen AI Max 395+, 64 GB unified); CPU AMD Ryzen AI Max+ 395 w/ Radeon 8060S; host RAM None GB; OS Ubuntu 24.04; class UNIFIED',
    engine: 'llama.cpp b10347-e6f10a799',
    parallelism: 'gpuLayers=999',
    batch: 1,
    promptLen: 1177,
    genLen: 512,
    contextDepth: 1177,
    prefill: {
      tokensPerSec: 1416.6,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 88.5, kind: 'perSequence' },
    caveats:
      'UD-Q4_K_XL approximated by Q4_K_M as in existing rows; Vulkan quantized dot path, Q8_0 KV. Plain autoregressive decoding.',
    input: {
      model: { kind: 'spec', spec: QWEN3_30B_A3B },
      quant: 'q4_k_m',
      numerics: { ...LLAMACPP_MMQ, kvCache: KV_Q8_0 },
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { batch: 1, promptLen: 1177, genLen: 512, contextDepth: 1177 },
    },
  },
  // spec-dec; collection row F-LM39
  {
    id: 'F-LM39',
    sourceUrl:
      'https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmw18v6om0antoa01dmno4a3w',
    date: '2026-11-03',
    model: 'unsloth/Qwen3.8-27B-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'UD-Q4_K_M',
    hardware:
      '1x None (Ryzen AI Max 395+, 96 GB unified); CPU AMD Ryzen AI Max+ 395 w/ Radeon 8060S; host RAM None GB; OS Ubuntu 24.04; class UNIFIED',
    engine: 'custom Loculus Core 0baf389',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 349,
    genLen: 256,
    contextDepth: 349,
    prefill: {
      tokensPerSec: 304.1,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 28.4, kind: 'perSequence' },
    caveats:
      'Loculus Vulkan MTP; draft depth not stated, k3 method scenario default used. Accepted-draft total 247 alone is insufficient to derive per-position acceptance; multi-step prose default 0.75. UD-Q4_K_M mapped to Q4_K_M.',
    input: {
      model: QWEN27,
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { batch: 1, promptLen: 349, genLen: 256, contextDepth: 349 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 3,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha },
        },
      },
    },
  },
  // spec-dec; collection row F-LM53
  {
    id: 'F-LM53',
    sourceUrl:
      'https://localmaxxing.com/en/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp?run=cmtk5446l03ogp701qhjuheiu',
    date: '2026-09-02',
    model: 'Jundot/Qwen3.8-Flash-Next-oQ4e-mtp',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (HF fixture; same base-model shapes)',
    quant: 'MLX-oQ4e',
    hardware:
      '1x None (M3 Ultra, 96 GB unified); CPU Apple M3 Ultra (28-core CPU); host RAM None GB; OS macOS 26.4; class UNIFIED',
    engine:
      'mlx oMLX 0.6.4 (tag 1d7826185c5b5b69b38b27cbe57d7597b7551fd7; official wheel f13d92900bf6c7e925e9a6d5525b4465c615c404ee796d328ffc5d4d379ddb0b)',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 618,
    genLen: 512,
    contextDepth: 618,
    prefill: {
      tokensPerSec: 477.75,
      kind: 'perSequence',
      check: {
        mode: 'excluded',
        reason:
          'Short-prompt client TTFT includes HTTP, tokenization and first decode; not an op-level prefill measurement.',
      },
    },
    decode: { tokensPerSec: 38.1, kind: 'perSequence' },
    caveats:
      '28-core CPU M3 Ultra identifies the 60-core GPU preset (not collector-suggested 80). oQ4e approximated by MLX4 g64; MTP k3, acceptance unspecified, prose default. LM54 tokSOut interpreted as aggregate at batch 8; client TTFT is excluded.',
    input: {
      model: FLASH_NEXT,
      quant: 'mlx-4bit',
      numerics: MLX,
      hardware: { deviceId: 'apple-m3-ultra-60c' },
      workload: { batch: 1, promptLen: 618, genLen: 512, contextDepth: 618 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 3,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha },
        },
      },
    },
  },
  // spec-dec; collection row F-LM54
  {
    id: 'F-LM54',
    sourceUrl:
      'https://localmaxxing.com/en/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp?run=cmtk587pj03opp7010pfdm2p5',
    date: '2026-09-02',
    model: 'Jundot/Qwen3.8-Flash-Next-oQ4e-mtp',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (HF fixture; same base-model shapes)',
    quant: 'MLX-oQ4e',
    hardware:
      '1x None (M3 Ultra, 96 GB unified); CPU Apple M3 Ultra (28-core CPU); host RAM None GB; OS macOS 26.4; class UNIFIED',
    engine:
      'mlx oMLX 0.6.4 (tag 1d7826185c5b5b69b38b27cbe57d7597b7551fd7; official wheel f13d92900bf6c7e925e9a6d5525b4465c615c404ee796d328ffc5d4d379ddb0b)',
    parallelism: 'n/a',
    batch: 8,
    promptLen: 620,
    genLen: 512,
    contextDepth: 620,
    prefill: { tokensPerSec: 683.99, kind: 'aggregate' },
    decode: { tokensPerSec: 91.1, kind: 'aggregate' },
    caveats:
      '28-core CPU M3 Ultra identifies the 60-core GPU preset (not collector-suggested 80). oQ4e approximated by MLX4 g64; MTP k3, acceptance unspecified, prose default. Both rates interpreted as aggregate at batch 8. Prefill 683.99 is near 8 × 620/6.71967≈738 tok/s, not 620/6.71967≈92.3 tok/s per sequence; raw timing payload is absent, so aggregation is inferred. Short TTFT can lower the aggregate prefill, preserving the upper-bound comparison; it cannot explain a per-sequence rate above the simulator.',
    input: {
      model: FLASH_NEXT,
      quant: 'mlx-4bit',
      numerics: MLX,
      hardware: { deviceId: 'apple-m3-ultra-60c' },
      workload: { batch: 8, promptLen: 620, genLen: 512, contextDepth: 620 },
      deployment: {
        speculative: {
          method: 'mtp',
          draftLength: 3,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W01
  {
    id: 'F-W01-pp2048',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/discussions/15396',
    date: '2026-01-20',
    model: 'gpt-oss-120b (ggml-org MXFP4 GGUF, 59.02 GiB)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 experts (native)',
    hardware: '1x RTX 3090 24GB; Intel (Core Ultra 7) 265K; 64 GB DDR5-5200 (channels not stated)',
    engine: 'llama.cpp 388ce82 (7760), CUDA, llama-bench -n-cpu-moe 25',
    parallelism: 'single GPU; experts of 25 layers on CPU',
    batch: 1,
    promptLen: 2048,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 364.8, kind: 'perSequence' },
    decode: {
      tokensPerSec: 26.9,
      kind: 'perSequence',
      check: {
        mode: 'exception',
        reason:
          'Sim underestimate (missing CPU-side expert compute): llama.cpp computes offloaded experts in DDR5, whereas the sim streams weights over PCIe; see docs/review/practicality-review.md §3 F3.',
      },
    },
    caveats:
      'DDR5-5200 has no exact preset; existing DDR5-5600 dual channel is the favourable surrogate. CPU executes first 25 layers of experts, but sim can only stream their weights to GPU; default prefetch 0.8. Decode listed once to avoid replicating the same tg128 test across pp lengths. Larger prefill microbatches than real ub4096 are favourable.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR5_5600_2CH } },
      workload: { batch: 1, promptLen: 2048, genLen: 128, contextDepth: 0 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 25 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W01
  {
    id: 'F-W01-pp8192',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/discussions/15396',
    date: '2026-01-20',
    model: 'gpt-oss-120b (ggml-org MXFP4 GGUF, 59.02 GiB)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 experts (native)',
    hardware: '1x RTX 3090 24GB; Intel (Core Ultra 7) 265K; 64 GB DDR5-5200 (channels not stated)',
    engine: 'llama.cpp 388ce82 (7760), CUDA, llama-bench -n-cpu-moe 25',
    parallelism: 'single GPU; experts of 25 layers on CPU',
    batch: 1,
    promptLen: 8192,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 682.19, kind: 'perSequence' },
    caveats:
      'DDR5-5200 has no exact preset; existing DDR5-5600 dual channel is the favourable surrogate. CPU executes first 25 layers of experts, but sim can only stream their weights to GPU; default prefetch 0.8. Decode listed once to avoid replicating the same tg128 test across pp lengths. Larger prefill microbatches than real ub4096 are favourable.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR5_5600_2CH } },
      workload: { batch: 1, promptLen: 8192, genLen: 128, contextDepth: 0 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 25 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W01
  {
    id: 'F-W01-pp16384',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/discussions/15396',
    date: '2026-01-20',
    model: 'gpt-oss-120b (ggml-org MXFP4 GGUF, 59.02 GiB)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 experts (native)',
    hardware: '1x RTX 3090 24GB; Intel (Core Ultra 7) 265K; 64 GB DDR5-5200 (channels not stated)',
    engine: 'llama.cpp 388ce82 (7760), CUDA, llama-bench -n-cpu-moe 25',
    parallelism: 'single GPU; experts of 25 layers on CPU',
    batch: 1,
    promptLen: 16384,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 704.37, kind: 'perSequence' },
    caveats:
      'DDR5-5200 has no exact preset; existing DDR5-5600 dual channel is the favourable surrogate. CPU executes first 25 layers of experts, but sim can only stream their weights to GPU; default prefetch 0.8. Decode listed once to avoid replicating the same tg128 test across pp lengths. Larger prefill microbatches than real ub4096 are favourable.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR5_5600_2CH } },
      workload: { batch: 1, promptLen: 16384, genLen: 128, contextDepth: 0 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 25 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W01
  {
    id: 'F-W01-pp32768',
    sourceUrl: 'https://github.com/ggml-org/llama.cpp/discussions/15396',
    date: '2026-01-20',
    model: 'gpt-oss-120b (ggml-org MXFP4 GGUF, 59.02 GiB)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 experts (native)',
    hardware: '1x RTX 3090 24GB; Intel (Core Ultra 7) 265K; 64 GB DDR5-5200 (channels not stated)',
    engine: 'llama.cpp 388ce82 (7760), CUDA, llama-bench -n-cpu-moe 25',
    parallelism: 'single GPU; experts of 25 layers on CPU',
    batch: 1,
    promptLen: 32768,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 672.56, kind: 'perSequence' },
    caveats:
      'DDR5-5200 has no exact preset; existing DDR5-5600 dual channel is the favourable surrogate. CPU executes first 25 layers of experts, but sim can only stream their weights to GPU; default prefetch 0.8. Decode listed once to avoid replicating the same tg128 test across pp lengths. Larger prefill microbatches than real ub4096 are favourable.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR5_5600_2CH } },
      workload: { batch: 1, promptLen: 32768, genLen: 128, contextDepth: 0 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 25 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W07
  {
    id: 'F-W07-p85',
    sourceUrl: 'https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/',
    date: '2025-11-10',
    model: 'gpt-oss-120b (gpt-oss-120b-F16.gguf per command; MXFP4 native)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 (native)',
    hardware: '1x RTX 3090 24GB; AMD EPYC 7343; 64 GB DDR4-3200 (channels not stated; EPYC has 8)',
    engine: 'llama.cpp llama-server latest (Ubuntu 24.04), Open WebUI',
    parallelism: 'single GPU, --n-gpu-layers 99, 27 MoE layers on CPU',
    batch: 1,
    promptLen: 85,
    genLen: 128,
    contextDepth: 85,
    prefill: {
      tokensPerSec: 48.49,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'Short-prompt client TTFT is not op-level prefill.' },
    },
    decode: { tokensPerSec: 1.81, kind: 'perSequence' },
    caveats:
      'Low-reliability secondary benchmark, unusually slow. EPYC 7343 has eight channels but populated count unknown: favourable eight-channel preset. CPU executes first 27 expert layers; sim only supports transfer. Generated length not stated: standard 128 scenario assumed; actual prompt depth retained.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR4_3200_8CH } },
      workload: { batch: 1, promptLen: 85, genLen: 128, contextDepth: 85 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 27 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W07
  {
    id: 'F-W07-p500',
    sourceUrl: 'https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/',
    date: '2025-11-10',
    model: 'gpt-oss-120b (gpt-oss-120b-F16.gguf per command; MXFP4 native)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 (native)',
    hardware: '1x RTX 3090 24GB; AMD EPYC 7343; 64 GB DDR4-3200 (channels not stated; EPYC has 8)',
    engine: 'llama.cpp llama-server latest (Ubuntu 24.04), Open WebUI',
    parallelism: 'single GPU, --n-gpu-layers 99, 27 MoE layers on CPU',
    batch: 1,
    promptLen: 500,
    genLen: 128,
    contextDepth: 500,
    prefill: {
      tokensPerSec: 173.6,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'Short-prompt client TTFT is not op-level prefill.' },
    },
    decode: { tokensPerSec: 1.57, kind: 'perSequence' },
    caveats:
      'Low-reliability secondary benchmark, unusually slow. EPYC 7343 has eight channels but populated count unknown: favourable eight-channel preset. CPU executes first 27 expert layers; sim only supports transfer. Generated length not stated: standard 128 scenario assumed; actual prompt depth retained.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR4_3200_8CH } },
      workload: { batch: 1, promptLen: 500, genLen: 128, contextDepth: 500 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 27 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W07
  {
    id: 'F-W07-p1400',
    sourceUrl: 'https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/',
    date: '2025-11-10',
    model: 'gpt-oss-120b (gpt-oss-120b-F16.gguf per command; MXFP4 native)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 (native)',
    hardware: '1x RTX 3090 24GB; AMD EPYC 7343; 64 GB DDR4-3200 (channels not stated; EPYC has 8)',
    engine: 'llama.cpp llama-server latest (Ubuntu 24.04), Open WebUI',
    parallelism: 'single GPU, --n-gpu-layers 99, 27 MoE layers on CPU',
    batch: 1,
    promptLen: 1400,
    genLen: 128,
    contextDepth: 1400,
    prefill: {
      tokensPerSec: 229.07,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'Short-prompt client TTFT is not op-level prefill.' },
    },
    decode: { tokensPerSec: 1.61, kind: 'perSequence' },
    caveats:
      'Low-reliability secondary benchmark, unusually slow. EPYC 7343 has eight channels but populated count unknown: favourable eight-channel preset. CPU executes first 27 expert layers; sim only supports transfer. Generated length not stated: standard 128 scenario assumed; actual prompt depth retained.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR4_3200_8CH } },
      workload: { batch: 1, promptLen: 1400, genLen: 128, contextDepth: 1400 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 27 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W07
  {
    id: 'F-W07-p5500',
    sourceUrl: 'https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/',
    date: '2025-11-10',
    model: 'gpt-oss-120b (gpt-oss-120b-F16.gguf per command; MXFP4 native)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 (native)',
    hardware: '1x RTX 3090 24GB; AMD EPYC 7343; 64 GB DDR4-3200 (channels not stated; EPYC has 8)',
    engine: 'llama.cpp llama-server latest (Ubuntu 24.04), Open WebUI',
    parallelism: 'single GPU, --n-gpu-layers 99, 27 MoE layers on CPU',
    batch: 1,
    promptLen: 5500,
    genLen: 128,
    contextDepth: 5500,
    prefill: { tokensPerSec: 246.18, kind: 'perSequence' },
    decode: { tokensPerSec: 1.41, kind: 'perSequence' },
    caveats:
      'Low-reliability secondary benchmark, unusually slow. EPYC 7343 has eight channels but populated count unknown: favourable eight-channel preset. CPU executes first 27 expert layers; sim only supports transfer. Generated length not stated: standard 128 scenario assumed; actual prompt depth retained.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR4_3200_8CH } },
      workload: { batch: 1, promptLen: 5500, genLen: 128, contextDepth: 5500 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 27 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W08
  {
    id: 'F-W08',
    sourceUrl: 'https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/',
    date: '2025-11-10',
    model: 'gpt-oss-120b (same)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware: 'same box as F-W07',
    engine: 'llama.cpp llama-server',
    parallelism: 'simple layer split: only 13 of 32 layers on GPU, no -ncmoe',
    batch: 1,
    promptLen: 71,
    genLen: 128,
    contextDepth: 71,
    prefill: {
      tokensPerSec: 42.5,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'Short-prompt client TTFT is not op-level prefill.' },
    },
    decode: {
      tokensPerSec: 0.9,
      kind: 'perSequence',
      check: {
        mode: 'exception',
        reason:
          'Sim underestimate (missing CPU-side whole-layer compute): CPU layers execute in DDR4, whereas the sim streams every whole-layer weight over PCIe. Source also claims 32 layers for a 36-layer model (possible mislabeled report); see docs/review/practicality-review.md §2/§3 F3.',
      },
    },
    caveats:
      'Source claims 13/32 layers on GPU, but GPT-OSS-120B fixture has 36 layers: use stated 13 GPU layers, remaining 23 CPU. Possible mislabeled layer-count report. Simulator streams whole layers instead of CPU compute. Generated length unknown, standard 128 assumed; EPYC eight-channel DDR4 assumed.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-3090', host: { link: hostPcieLink(4, 16), ram: DDR4_3200_8CH } },
      workload: { batch: 1, promptLen: 71, genLen: 128, contextDepth: 71 },
      deployment: { offload: { denseLayers: { tier: 'host_ram', layers: { from: 0, to: 23 } } } },
    },
  },
  // DDR expert offload; collection row F-W09
  {
    id: 'F-W09-p87',
    sourceUrl: 'https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/',
    date: '2025-11-10',
    model: 'gpt-oss-120b (gpt-oss-120b-F16.gguf)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 (native)',
    hardware: '1x RTX 5090 32GB; host CPU/RAM NOT stated',
    engine: 'llama.cpp llama-server',
    parallelism:
      'single GPU, --n-cpu-moe 21 --n-gpu-layers 99 --ctx-size 8192 --flash-attn --jinja',
    batch: 1,
    promptLen: 87,
    genLen: 128,
    contextDepth: 87,
    prefill: {
      tokensPerSec: 84.91,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'Short-prompt client TTFT is not op-level prefill.' },
    },
    decode: { tokensPerSec: 9.6, kind: 'perSequence' },
    caveats:
      'Host RAM/CPU not reported: default DDR5-6000 dual-channel scenario, not a hardware-matched measurement. First 21 expert layers execute on CPU; simulator only streams weights. Generated length unknown, standard 128 scenario. Secondary source.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-5090', host: { ram: DDR5_6000_2CH } },
      workload: { batch: 1, promptLen: 87, genLen: 128, contextDepth: 87 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 21 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W09
  {
    id: 'F-W09-p5650',
    sourceUrl: 'https://www.hardware-corner.net/gpt-oss-offloading-moe-layers/',
    date: '2025-11-10',
    model: 'gpt-oss-120b (gpt-oss-120b-F16.gguf)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4 (native)',
    hardware: '1x RTX 5090 32GB; host CPU/RAM NOT stated',
    engine: 'llama.cpp llama-server',
    parallelism:
      'single GPU, --n-cpu-moe 21 --n-gpu-layers 99 --ctx-size 8192 --flash-attn --jinja',
    batch: 1,
    promptLen: 5650,
    genLen: 128,
    contextDepth: 5650,
    prefill: { tokensPerSec: 448.15, kind: 'perSequence' },
    decode: { tokensPerSec: 8.14, kind: 'perSequence' },
    caveats:
      'Host RAM/CPU not reported: default DDR5-6000 dual-channel scenario, not a hardware-matched measurement. First 21 expert layers execute on CPU; simulator only streams weights. Generated length unknown, standard 128 scenario. Secondary source.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: { kind: 'preset', preset: GPT_OSS_GGUF },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-5090', host: { ram: DDR5_6000_2CH } },
      workload: { batch: 1, promptLen: 5650, genLen: 128, contextDepth: 5650 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            layers: { from: 0, to: 21 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // DDR expert offload; collection row F-W10
  {
    id: 'F-W10',
    sourceUrl:
      'https://tinker-ai.com/guides/epyc-ddr5-hybrid-cpu-gpu-ktransformers-ik-llama-cpp-deepseek-r1-671b-qwen3-235b-benchmark/',
    date: '2026-11-04',
    model: 'Qwen3-235B-A22B (GGUF Q4_K_M, 143 GB)',
    simulatedModel: 'QWEN3_235B_A22B (existing hand-written spec)',
    quant: 'Q4_K_M',
    hardware:
      '1x RTX 5090 32GB; AMD EPYC 9374F (32C) 12-channel DDR5-4800 576 GB (~460 GB/s stated)',
    engine: 'ik_llama.cpp 9c4e21a, llama-bench pp2048/tg128, -ot exps=CPU',
    parallelism: 'single GPU; experts on CPU; numactl --interleave=all, -t 32',
    batch: 1,
    promptLen: 2048,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 510, kind: 'perSequence' },
    decode: {
      tokensPerSec: 26.4,
      kind: 'perSequence',
      check: {
        mode: 'exception',
        reason:
          'Sim underestimate (missing CPU-side expert compute): EPYC 12-channel DDR5 computes experts locally rather than transferring them to GPU. Low-reliability secondary report, not verified; see docs/review/practicality-review.md §3 F3.',
      },
    },
    caveats:
      'LOW RELIABILITY secondary article with stale build references. All routed experts computed on EPYC 9374F CPU, 12-channel DDR5-4800; simulator instead transfers weights over native PCIe5. Not an independent confirmation of its numerical accuracy.',
    input: {
      model: { kind: 'spec', spec: QWEN3_235B_A22B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'rtx-5090', host: { ram: DDR5_4800_12CH } },
      workload: { batch: 1, promptLen: 2048, genLen: 128, contextDepth: 0 },
      deployment: {
        offload: {
          routedExperts: { tier: 'host_ram', residentFraction: 0, hitRatio: { kind: 'uniform' } },
        },
      },
    },
  },
  // SSD offload; collection row F-W12
  {
    id: 'F-W12',
    sourceUrl: 'https://argonautlabs.ai/research/apple-silicon-results.html',
    date: '2026-09-27',
    model: 'DeepSeek-V4.1-Flash (page: "552B, 518 GB at 4-bit"), ds4 Q4 build',
    simulatedModel: 'deepseek-ai/DeepSeek-V4.1-Flash (existing HF fixture)',
    quant: 'Q4 (routed experts ~4-bit)',
    hardware: 'Apple M5 Max MacBook Pro 128 GB, one internal SSD',
    engine: 'antirez/ds4 upstream binary bd66c402',
    parallelism: 'single device; experts streamed from NVMe',
    batch: 1,
    promptLen: 512,
    genLen: 200,
    contextDepth: 512,
    prefill: { tokensPerSec: 17.96, kind: 'perSequence' },
    decode: {
      tokensPerSec: 10.18,
      kind: 'perSequence',
      check: {
        mode: 'exception',
        reason:
          'Possible sim underestimate: uniform cache-hit default misses real expert locality; measured hit rate and exact ds4 Q4 module layout are unknown. Cache size is reported, not fitted; see docs/review/practicality-review.md §2/§3.',
      },
    },
    caveats:
      "40 layers ×384 experts in fixture; 3688 cached expert slots means f=3688/15360. Hit rate not reported: uniform default, prefetch default 0.8; NOT fitted to observed throughput. Q4 storage represented at 4.5 bpw. Steady decode excludes first step; engine model uses midpoint across 200 generated tokens. Page's 552B/518GB label is not a verified weight-file size.",
    input: {
      model: { kind: 'hf', repoId: 'deepseek-ai/DeepSeek-V4.1-Flash' },
      quant: {
        kind: 'preset',
        preset: ggufBpw(4.5, 'Q4 block-storage surrogate; exact ds4 module mix unpublished'),
      },
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m5-max-40c', host: { ssd: APPLE_SSD_M5_PRO_MAX } },
      workload: { batch: 1, promptLen: 512, genLen: 200, contextDepth: 512 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'nvme',
            residentFraction: 3688 / (40 * 384),
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // SSD offload; collection row F-W13
  {
    id: 'F-W13',
    sourceUrl: 'https://argonautlabs.ai/research/apple-silicon-results.html',
    date: '2026-09-27',
    model: 'DeepSeek-V4.1-Flash ds4 fork (same file)',
    simulatedModel: 'deepseek-ai/DeepSeek-V4.1-Flash (existing HF fixture)',
    quant: 'Q4',
    hardware: 'M5 Max 128 GB, one internal SSD',
    engine: 'argodrive fork, profile v41-stack-20260927, 4,600 cached experts',
    parallelism: 'same',
    batch: 1,
    promptLen: 512,
    genLen: 200,
    contextDepth: 512,
    prefill: { tokensPerSec: 30.98, kind: 'perSequence' },
    decode: {
      tokensPerSec: 18.66,
      kind: 'perSequence',
      check: {
        mode: 'exception',
        reason:
          'Possible sim underestimate: uniform cache-hit default misses real expert locality/read-pattern improvements; exact ds4 Q4 module layout and measured hit rate are unknown. No acceptance or cache parameter fitted; see docs/review/practicality-review.md §2/§3.',
      },
    },
    caveats:
      "40 layers ×384 experts in fixture; 4600 cached expert slots means f=4600/15360. Hit rate not reported: uniform default, prefetch default 0.8; NOT fitted to observed throughput. Q4 storage represented at 4.5 bpw. Steady decode excludes first step; engine model uses midpoint across 200 generated tokens. Page's 552B/518GB label is not a verified weight-file size.",
    input: {
      model: { kind: 'hf', repoId: 'deepseek-ai/DeepSeek-V4.1-Flash' },
      quant: {
        kind: 'preset',
        preset: ggufBpw(4.5, 'Q4 block-storage surrogate; exact ds4 module mix unpublished'),
      },
      numerics: LLAMACPP_METAL,
      hardware: { deviceId: 'apple-m5-max-40c', host: { ssd: APPLE_SSD_M5_PRO_MAX } },
      workload: { batch: 1, promptLen: 512, genLen: 200, contextDepth: 512 },
      deployment: {
        offload: {
          routedExperts: {
            tier: 'nvme',
            residentFraction: 4600 / (40 * 384),
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    },
  },
  // single Mac; collection row F-W19
  {
    id: 'F-W19',
    sourceUrl: 'https://machinelearning.apple.com/research/mlx-distributed-inference',
    date: '2026-10-15',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'BF16 (native)',
    hardware: '1x Mac Studio M3 Ultra (512 GB; 819 GB/s)',
    engine:
      'MLX (>= 0.30.1) mlx_lm.benchmark, jaccl backend over RDMA/Thunderbolt 5, MLX_METAL_FAST_SYNCH=1',
    parallelism: 'single device',
    batch: 1,
    promptLen: 8192,
    genLen: 128,
    contextDepth: 8192,
    prefill: { tokensPerSec: 450.1098901098901, kind: 'perSequence' },
    decode: { tokensPerSec: 12.46, kind: 'perSequence' },
    caveats:
      'Apple MLX distributed-inference table. Prefill=8192/TTFT (long prompt), decode follows prompt, not empty-cache tg. Existing TB5 mesh preset represents link; report two cables per pair, preset bandwidth convention unchanged. No link tuning.',
    input: {
      model: QWEN27,
      quant: 'bf16',
      numerics: MLX,
      hardware: { deviceId: 'apple-m3-ultra-80c' },
      workload: { batch: 1, promptLen: 8192, genLen: 128, contextDepth: 8192 },
    },
  },
  // single Mac; collection row F-W20
  {
    id: 'F-W20',
    sourceUrl: 'https://machinelearning.apple.com/research/mlx-distributed-inference',
    date: '2026-10-15',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'BF16 (native)',
    hardware: '1x Mac Studio M5 Ultra (512 GB; 1.2 TB/s)',
    engine:
      'MLX (>= 0.30.1) mlx_lm.benchmark, jaccl backend over RDMA/Thunderbolt 5, MLX_METAL_FAST_SYNCH=1',
    parallelism: 'single device',
    batch: 1,
    promptLen: 8192,
    genLen: 128,
    contextDepth: 8192,
    prefill: { tokensPerSec: 1998.048780487805, kind: 'perSequence' },
    decode: { tokensPerSec: 19.69, kind: 'perSequence' },
    caveats:
      'Apple MLX distributed-inference table. Prefill=8192/TTFT (long prompt), decode follows prompt, not empty-cache tg. Existing TB5 mesh preset represents link; report two cables per pair, preset bandwidth convention unchanged. No link tuning.',
    input: {
      model: QWEN27,
      quant: 'bf16',
      numerics: MLX,
      hardware: { deviceId: 'apple-m5-ultra-80c' },
      workload: { batch: 1, promptLen: 8192, genLen: 128, contextDepth: 8192 },
    },
  },
  // Mac TB5 cluster; collection row F-W21
  {
    id: 'F-W21',
    sourceUrl: 'https://machinelearning.apple.com/research/mlx-distributed-inference',
    date: '2026-10-15',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'BF16 (native)',
    hardware: '2x Mac Studio M5 Ultra, two TB5 cables, RDMA',
    engine:
      'MLX (>= 0.30.1) mlx_lm.benchmark, jaccl backend over RDMA/Thunderbolt 5, MLX_METAL_FAST_SYNCH=1',
    parallelism: 'pipeline parallel (--pipeline), 2 nodes',
    batch: 1,
    promptLen: 8192,
    genLen: 128,
    contextDepth: 8192,
    prefill: { tokensPerSec: 1998.048780487805, kind: 'perSequence' },
    decode: { tokensPerSec: 19.44, kind: 'perSequence' },
    caveats:
      'Apple MLX distributed-inference table. Prefill=8192/TTFT (long prompt), decode follows prompt, not empty-cache tg. Existing TB5 mesh preset represents link; report two cables per pair, preset bandwidth convention unchanged. No link tuning.',
    input: {
      model: QWEN27,
      quant: 'bf16',
      numerics: MLX,
      hardware: { deviceId: 'apple-m5-ultra-80c', topology: M5_ULTRA_TB5_CLUSTERS[0] },
      workload: { batch: 1, promptLen: 8192, genLen: 128, contextDepth: 8192 },
      parallel: { tp: 1, pp: 2 },
    },
  },
  // Mac TB5 cluster; collection row F-W22
  {
    id: 'F-W22',
    sourceUrl: 'https://machinelearning.apple.com/research/mlx-distributed-inference',
    date: '2026-10-15',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'BF16 (native)',
    hardware: '2x Mac Studio M5 Ultra, mesh with 2 TB5 cables per link, RDMA',
    engine:
      'MLX (>= 0.30.1) mlx_lm.benchmark, jaccl backend over RDMA/Thunderbolt 5, MLX_METAL_FAST_SYNCH=1',
    parallelism: 'tensor parallel, 2 nodes',
    batch: 1,
    promptLen: 8192,
    genLen: 128,
    contextDepth: 8192,
    prefill: { tokensPerSec: 2642.5806451612902, kind: 'perSequence' },
    decode: { tokensPerSec: 33.27, kind: 'perSequence' },
    caveats:
      'Apple MLX distributed-inference table. Prefill=8192/TTFT (long prompt), decode follows prompt, not empty-cache tg. Existing TB5 mesh preset represents link; report two cables per pair, preset bandwidth convention unchanged. No link tuning.',
    input: {
      model: QWEN27,
      quant: 'bf16',
      numerics: MLX,
      hardware: { deviceId: 'apple-m5-ultra-80c', topology: M5_ULTRA_TB5_CLUSTERS[0] },
      workload: { batch: 1, promptLen: 8192, genLen: 128, contextDepth: 8192 },
      parallel: { tp: 2 },
    },
  },
  // Mac TB5 cluster; collection row F-W23
  {
    id: 'F-W23',
    sourceUrl: 'https://machinelearning.apple.com/research/mlx-distributed-inference',
    date: '2026-10-15',
    model: 'Qwen/Qwen3.8-27B',
    simulatedModel: 'Qwen/Qwen3.8-27B (HF fixture; same base-model shapes)',
    quant: 'BF16 (native)',
    hardware: '4x Mac Studio M5 Ultra, full mesh, 2 TB5 cables per pair, RDMA',
    engine:
      'MLX (>= 0.30.1) mlx_lm.benchmark, jaccl backend over RDMA/Thunderbolt 5, MLX_METAL_FAST_SYNCH=1',
    parallelism: 'tensor parallel, 4 nodes',
    batch: 1,
    promptLen: 8192,
    genLen: 128,
    contextDepth: 8192,
    prefill: { tokensPerSec: 4818.823529411765, kind: 'perSequence' },
    decode: { tokensPerSec: 50.89, kind: 'perSequence' },
    caveats:
      'Apple MLX distributed-inference table. Prefill=8192/TTFT (long prompt), decode follows prompt, not empty-cache tg. Existing TB5 mesh preset represents link; report two cables per pair, preset bandwidth convention unchanged. No link tuning.',
    input: {
      model: QWEN27,
      quant: 'bf16',
      numerics: MLX,
      hardware: { deviceId: 'apple-m5-ultra-80c', topology: M5_ULTRA_TB5_CLUSTERS[2] },
      workload: { batch: 1, promptLen: 8192, genLen: 128, contextDepth: 8192 },
      parallel: { tp: 4 },
    },
  },
  // Spark; collection row F-W30
  {
    id: 'F-W30',
    sourceUrl:
      'https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/',
    date: '2026-11-09',
    model: 'Qwen/Qwen3.8-27B (NVFP4)',
    simulatedModel: 'nvidia/Qwen3.8-27B-NVFP4 (existing HF fixture)',
    quant: 'NVFP4',
    hardware: '1x DGX Spark (GB10, 128 GB LPDDR5X 273 GB/s), ConnectX-7 200 GbE, topology: —',
    engine: 'NVIDIA stack (vLLM/FlashInfer); no spec-dec/MTP',
    parallelism: 'tensor parallel over 1 node(s)',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 11.86, kind: 'perSequence' },
    caveats:
      'Native NVFP4 fixture. Source omits workload lengths; decode simulated at one-token prompt +64 generated midpoint (128-token conventional window), NOT configured maximum context. No prefill metric encoded because prompt length is unknown. Date only 2026-11: date field is collection date.',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark' },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 1 },
    },
  },
  // Spark; collection row F-W31
  {
    id: 'F-W31',
    sourceUrl:
      'https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/',
    date: '2026-11-09',
    model: 'Qwen/Qwen3.8-27B (NVFP4)',
    simulatedModel: 'nvidia/Qwen3.8-27B-NVFP4 (existing HF fixture)',
    quant: 'NVFP4',
    hardware:
      '2x DGX Spark (GB10, 128 GB LPDDR5X 273 GB/s), ConnectX-7 200 GbE, topology: direct QSFP',
    engine: 'NVIDIA stack (vLLM/FlashInfer); no spec-dec/MTP',
    parallelism: 'tensor parallel over 2 node(s)',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 21.71, kind: 'perSequence' },
    caveats:
      'Native NVFP4 fixture. Source omits workload lengths; decode simulated at one-token prompt +64 generated midpoint (128-token conventional window), NOT configured maximum context. No prefill metric encoded because prompt length is unknown. Date only 2026-11: date field is collection date.',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_PAIR },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 2 },
    },
  },
  // Spark; collection row F-W32
  {
    id: 'F-W32',
    sourceUrl:
      'https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/',
    date: '2026-11-09',
    model: 'Qwen/Qwen3.8-27B (NVFP4)',
    simulatedModel: 'nvidia/Qwen3.8-27B-NVFP4 (existing HF fixture)',
    quant: 'NVFP4',
    hardware:
      '4x DGX Spark (GB10, 128 GB LPDDR5X 273 GB/s), ConnectX-7 200 GbE, topology: physical ring',
    engine: 'NVIDIA stack (vLLM/FlashInfer); no spec-dec/MTP',
    parallelism: 'tensor parallel over 4 node(s)',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 32.82, kind: 'perSequence' },
    caveats:
      'Native NVFP4 fixture. Source omits workload lengths; decode simulated at one-token prompt +64 generated midpoint (128-token conventional window), NOT configured maximum context. No prefill metric encoded because prompt length is unknown. Date only 2026-11: date field is collection date.',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_4_RING },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 4 },
    },
  },
  // Spark; collection row F-W33
  {
    id: 'F-W33',
    sourceUrl:
      'https://developer.nvidia.com/blog/running-frontier-class-ai-models-across-8-nvidia-dgx-spark-nodes/',
    date: '2026-11-09',
    model: 'Qwen/Qwen3.8-27B (NVFP4)',
    simulatedModel: 'nvidia/Qwen3.8-27B-NVFP4 (existing HF fixture)',
    quant: 'NVFP4',
    hardware:
      '4x DGX Spark (GB10, 128 GB LPDDR5X 273 GB/s), ConnectX-7 200 GbE, topology: switch + RoCEnante all-reduce',
    engine: 'NVIDIA stack (vLLM/FlashInfer); no spec-dec/MTP',
    parallelism: 'tensor parallel over 4 node(s)',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 36.34, kind: 'perSequence' },
    caveats:
      'Native NVFP4 fixture. Source omits workload lengths; decode simulated at one-token prompt +64 generated midpoint (128-token conventional window), NOT configured maximum context. No prefill metric encoded because prompt length is unknown. Date only 2026-11: date field is collection date.',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_4 },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 4 },
    },
  },
  // Spark; collection row F-W37
  {
    id: 'F-W37',
    sourceUrl: 'https://developer.nvidia.com/blog/?p=114188',
    date: '2026-03-16',
    model: 'Llama 3.3 70B (NVFP4)',
    simulatedModel: 'LLAMA3_70B (existing hand-written spec)',
    quant: 'NVFP4',
    hardware: '1x DGX Spark, ConnectX-7 RoCE (4-node with 200 GbE switch)',
    engine: 'TensorRT-LLM',
    parallelism: 'TP1',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 3.717472118959108, kind: 'perSequence' },
    caveats:
      'Llama3.3-70B uses existing Llama3-70B shape. Decode=1000/TPOTms. Prompt/output length not reported; one-token prompt,128 generated standard scenario is a favourable short-context bound. TTFT cannot convert to prefill without prompt length; no rate invented.',
    input: {
      model: { kind: 'spec', spec: LLAMA3_70B },
      quant: 'nvfp4',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark' },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 1 },
    },
  },
  // Spark; collection row F-W38
  {
    id: 'F-W38',
    sourceUrl: 'https://developer.nvidia.com/blog/?p=114188',
    date: '2026-03-16',
    model: 'Llama 3.3 70B (NVFP4)',
    simulatedModel: 'LLAMA3_70B (existing hand-written spec)',
    quant: 'NVFP4',
    hardware: '2x DGX Spark, ConnectX-7 RoCE (4-node with 200 GbE switch)',
    engine: 'TensorRT-LLM',
    parallelism: 'TP2',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 7.518796992481203, kind: 'perSequence' },
    caveats:
      'Llama3.3-70B uses existing Llama3-70B shape. Decode=1000/TPOTms. Prompt/output length not reported; one-token prompt,128 generated standard scenario is a favourable short-context bound. TTFT cannot convert to prefill without prompt length; no rate invented.',
    input: {
      model: { kind: 'spec', spec: LLAMA3_70B },
      quant: 'nvfp4',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_PAIR },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 2 },
    },
  },
  // Spark; collection row F-W39
  {
    id: 'F-W39',
    sourceUrl: 'https://developer.nvidia.com/blog/?p=114188',
    date: '2026-03-16',
    model: 'Llama 3.3 70B (NVFP4)',
    simulatedModel: 'LLAMA3_70B (existing hand-written spec)',
    quant: 'NVFP4',
    hardware: '4x DGX Spark, ConnectX-7 RoCE (4-node with 200 GbE switch)',
    engine: 'TensorRT-LLM',
    parallelism: 'TP4',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 13.88888888888889, kind: 'perSequence' },
    caveats:
      'Llama3.3-70B uses existing Llama3-70B shape. Decode=1000/TPOTms. Prompt/output length not reported; one-token prompt,128 generated standard scenario is a favourable short-context bound. TTFT cannot convert to prefill without prompt length; no rate invented.',
    input: {
      model: { kind: 'spec', spec: LLAMA3_70B },
      quant: 'nvfp4',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_4 },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 4 },
    },
  },
  // Spark; collection row F-W40
  {
    id: 'F-W40',
    sourceUrl: 'https://developer.nvidia.com/blog/?p=114188',
    date: '2026-03-16',
    model: 'Qwen3 Coder Next 80B (FP8) [sibling Qwen3-Next-80B-A3B spec]',
    simulatedModel: 'QWEN3_NEXT_80B_A3B (existing hand-written spec)',
    quant: 'FP8',
    hardware: '1x DGX Spark',
    engine: 'vLLM',
    parallelism: 'single device',
    batch: 1,
    promptLen: 131072,
    genLen: 1024,
    contextDepth: 131072,
    prefill: { tokensPerSec: 2390, kind: 'perSequence' },
    decode: { tokensPerSec: 28.95, kind: 'perSequence' },
    caveats:
      'Qwen3-Coder-Next is mapped to Qwen3-Next-80B-A3B sibling spec;128K/1K interpreted as 131072/1024. FP8 W8A8 compute. Model-family surrogate, not an exact checkpoint.',
    input: {
      model: { kind: 'spec', spec: QWEN3_NEXT_80B_A3B },
      quant: 'fp8',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark' },
      workload: { batch: 1, promptLen: 131072, genLen: 1024, contextDepth: 131072 },
    },
  },
  // Spark; collection row F-W41
  {
    id: 'F-W41',
    sourceUrl: 'https://www.gdep.co.jp/tech_report/tech_blog/dgx-spark-llm_01_20260904/',
    date: '2026-09-29',
    model: 'openai/gpt-oss-120b (MXFP4, ~60 GB)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware: '1x DGX Spark',
    engine: 'vLLM (vllm bench serve, ShareGPT, num-prompts=1)',
    parallelism: 'single device',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 32.28, kind: 'perSequence' },
    caveats:
      'ShareGPT np1 request, per-sequence. Exact prompt/output lengths absent: one-token prompt/128-token conventional output favourable scenario, not actual context. Native MXFP4 experts, BF16 remaining modules. np10/100 rates not used: sequential num-prompts does not establish concurrent batch size. TTFT alone not converted to prefill.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: {
        kind: 'preset',
        preset: { ...GPT_OSS_GGUF, weights: { dtype: 'bf16' }, embeddings: { dtype: 'bf16' } },
      },
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark' },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 1 },
    },
  },
  // Spark; collection row F-W42
  {
    id: 'F-W42',
    sourceUrl: 'https://www.gdep.co.jp/tech_report/tech_blog/dgx-spark-llm_01_20260904/',
    date: '2026-09-29',
    model: 'openai/gpt-oss-120b (MXFP4)',
    simulatedModel: 'GPT_OSS_120B (existing hand-written spec)',
    quant: 'MXFP4',
    hardware: '2x DGX Spark, QSFP 200 Gbps direct (NCCL ~22 GB/s measured); vs RJ45 row',
    engine: 'vLLM multi-node TP',
    parallelism: 'TP2',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 44.36, kind: 'perSequence' },
    caveats:
      'ShareGPT np1 request, per-sequence. Exact prompt/output lengths absent: one-token prompt/128-token conventional output favourable scenario, not actual context. Native MXFP4 experts, BF16 remaining modules. np10/100 rates not used: sequential num-prompts does not establish concurrent batch size. TTFT alone not converted to prefill.',
    input: {
      model: { kind: 'spec', spec: GPT_OSS_120B },
      quant: {
        kind: 'preset',
        preset: { ...GPT_OSS_GGUF, weights: { dtype: 'bf16' }, embeddings: { dtype: 'bf16' } },
      },
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_PAIR },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
      parallel: { tp: 2 },
    },
  },
  // Strix Halo; collection row F-W45
  {
    id: 'F-W45',
    sourceUrl: 'https://github.com/visorcraft/strix-halo-llm-perf/blob/main/BENCHMARKS.md',
    date: '2026-02-13',
    model: 'Qwen3-30B-A3B MoE Q4_K_M',
    simulatedModel: 'QWEN3_30B_A3B (existing hand-written spec)',
    quant: 'Q4_K_M',
    hardware: 'GMKtec EVO-X2, Ryzen AI Max+ 395, 128 GB LPDDR5X-8000, 120 W TDP',
    engine: 'llama.cpp b8038 (05a6f0e89), Vulkan, llama-bench pp512/tg128',
    parallelism: 'single device',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 1141.78, kind: 'perSequence' },
    decode: { tokensPerSec: 86.07, kind: 'perSequence' },
    caveats:
      'GMKtec EVO-X2 Ryzen AI Max+ 395, 128 GB LPDDR5X-8000; separate empty-cache llama-bench pp512/tg128. W47 uses Qwen3-Next sibling shape for Coder-Next.',
    input: {
      model: { kind: 'spec', spec: QWEN3_30B_A3B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { batch: 1, promptLen: 512, genLen: 128, contextDepth: 0 },
    },
  },
  // Strix Halo; collection row F-W46
  {
    id: 'F-W46',
    sourceUrl: 'https://github.com/visorcraft/strix-halo-llm-perf/blob/main/BENCHMARKS.md',
    date: '2026-02-13',
    model: 'Llama 3.1 70B Q4_K_M',
    simulatedModel: 'LLAMA3_70B (existing hand-written spec)',
    quant: 'Q4_K_M',
    hardware: 'GMKtec EVO-X2, Ryzen AI Max+ 395, 128 GB LPDDR5X-8000, 120 W TDP',
    engine: 'llama.cpp b8038 (05a6f0e89), Vulkan, llama-bench pp512/tg128',
    parallelism: 'single device',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 81.63, kind: 'perSequence' },
    decode: { tokensPerSec: 5.1, kind: 'perSequence' },
    caveats:
      'GMKtec EVO-X2 Ryzen AI Max+ 395, 128 GB LPDDR5X-8000; separate empty-cache llama-bench pp512/tg128. W47 uses Qwen3-Next sibling shape for Coder-Next.',
    input: {
      model: { kind: 'spec', spec: LLAMA3_70B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { batch: 1, promptLen: 512, genLen: 128, contextDepth: 0 },
    },
  },
  // Strix Halo; collection row F-W47
  {
    id: 'F-W47',
    sourceUrl: 'https://github.com/visorcraft/strix-halo-llm-perf/blob/main/BENCHMARKS.md',
    date: '2026-02-13',
    model: 'Qwen3-Coder-Next 80B-A3B Q4_K_M',
    simulatedModel: 'QWEN3_NEXT_80B_A3B (existing hand-written spec)',
    quant: 'Q4_K_M',
    hardware: 'GMKtec EVO-X2, Ryzen AI Max+ 395, 128 GB LPDDR5X-8000, 120 W TDP',
    engine: 'llama.cpp b8038 (05a6f0e89), Vulkan, llama-bench pp512/tg128',
    parallelism: 'single device',
    batch: 1,
    promptLen: 512,
    genLen: 128,
    contextDepth: 0,
    prefill: { tokensPerSec: 531.48, kind: 'perSequence' },
    decode: { tokensPerSec: 42.7, kind: 'perSequence' },
    caveats:
      'GMKtec EVO-X2 Ryzen AI Max+ 395, 128 GB LPDDR5X-8000; separate empty-cache llama-bench pp512/tg128. W47 uses Qwen3-Next sibling shape for Coder-Next.',
    input: {
      model: { kind: 'spec', spec: QWEN3_NEXT_80B_A3B },
      quant: 'q4_k_m',
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'ryzen-ai-max-plus-395' },
      workload: { batch: 1, promptLen: 512, genLen: 128, contextDepth: 0 },
    },
  },
  // SSD offload; collection row F-LM07
  {
    id: 'F-LM07',
    sourceUrl:
      'https://localmaxxing.com/en/models/RadixArk/Qwen3.8-Flash-Next-NVFP4?run=cmtoqy58v00ymn701886t60sj',
    date: '2026-09-05',
    model: 'RadixArk/Qwen3.8-Flash-Next-NVFP4',
    simulatedModel:
      'nvidia/Qwen3.8-Flash-Next-NVFP4 (existing quant fixture; RadixArk layout surrogate)',
    quant: 'NVFP4',
    hardware:
      '1x RTX 5090, VRAM 32 GB each; CPU 13th Gen Intel(R) Core(TM) i9-13900K; host RAM 192 GB; OS NixOS 26.05; class DISCRETE_GPU',
    engine: 'sglang FreeToken af71ba4 (disk-PLE; submitted under sglang-compatible category)',
    parallelism: 'tensorParallel=1',
    batch: 1,
    promptLen: 262004,
    genLen: 127,
    contextDepth: 262004,
    prefill: { tokensPerSec: 3388.48, kind: 'perSequence' },
    decode: { tokensPerSec: 74.5315, kind: 'perSequence' },
    caveats:
      'FreeToken disk-PLE run: FP8 lookup table on NVMe, native NVFP4 experts. Existing NVIDIA quant fixture stands in for RadixArk; exact local layout unverified. SSD model/host RAM speed not reported: engine defaults (990Pro/DDR5-6000) used explicitly as scenario assumptions; not measured hardware. Scale layout/default prefetch unchanged; 262004-token cold prompt, 127 generated.',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-Flash-Next-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'rtx-5090' },
      deployment: { offload: { lookupTables: { tier: 'nvme' } } },
      workload: { batch: 1, promptLen: 262004, genLen: 127, contextDepth: 262004 },
    },
  },
  // SSD offload; collection row F-LM34
  {
    id: 'F-LM34',
    sourceUrl:
      'https://localmaxxing.com/en/models/unsloth/Qwen3.8-Flash-Next-GGUF?run=cmtipwk3a066sp401nqezeu6d',
    date: '2026-09-01',
    model: 'unsloth/Qwen3.8-Flash-Next-GGUF',
    simulatedModel: 'Qwen/Qwen3.8-Flash-Next (existing HF fixture)',
    quant: 'UD-IQ4_XS',
    hardware:
      '1x None (GB10 Grace Blackwell, 128 GB unified); CPU None; host RAM None GB; OS DGX OS (Ubuntu 24.04); class UNIFIED',
    engine: 'llama.cpp b10689',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 1,
    genLen: 128,
    contextDepth: 1,
    decode: { tokensPerSec: 27.3, kind: 'perSequence' },
    caveats:
      'Notes explicitly say single Spark with PLE mmap from NVMe. Stock 4 TB Spark SSD preset used (actual SKU not reported); GGUF dynamic IQ4_XS approximated at 4.25 bpw. Prompt/output lengths missing: favourable one-token prompt, conventional 128 generated, not configured 131072 context. Decode only.',
    input: {
      model: FLASH_NEXT,
      quant: { kind: 'preset', preset: ggufBpw(4.25, 'UD-IQ4_XS surrogate: 4.25 bpw') },
      numerics: LLAMACPP_MMQ,
      hardware: { deviceId: 'dgx-spark', host: { ssd: DGX_SPARK_SSD_4TB } },
      deployment: { offload: { lookupTables: { tier: 'nvme' } } },
      workload: { batch: 1, promptLen: 1, genLen: 128, contextDepth: 1 },
    },
  },
  // SSD offload; collection row F-LM49
  {
    id: 'F-LM49',
    sourceUrl:
      'https://localmaxxing.com/en/models/ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw?run=cmvx0vro606jmoa01cyaareex',
    date: '2026-10-31',
    model: 'ddalcu/DeepSeek-V4.1-Flash-1017-MLX-Serve-iQ-MLX-3.0bpw',
    simulatedModel:
      'deepseek-ai/DeepSeek-V4.1-Flash-1017 (existing HF fixture; uniform3bpw quant surrogate)',
    quant: 'iQ-MLX-3.0bpw',
    hardware:
      '1x None (M5 Ultra, 256 GB unified); CPU Apple M5 Ultra (30-core CPU, 64-core GPU); host RAM None GB; OS macOS 27.2; class UNIFIED',
    engine: 'mlx mlx-serve 662387ba (main 5615bb65 + PRs #1040, #1041)',
    parallelism: 'n/a',
    batch: 1,
    promptLen: 278,
    genLen: 1024,
    contextDepth: 278,
    decode: { tokensPerSec: 76.18, kind: 'perSequence' },
    caveats:
      'Actual 64-core GPU, exact 1017 architecture fixture. Engram SSD placement stated; drive model absent, engine 990 Pro default is an explicit scenario assumption. Quant uses advertised 3.0 bpw uniformly because module mix unpublished; not tuned. DSpark checkpoint block 5 => default k 4; source gives accepted draft count 687 but no acceptance/step count, so DSpark prose default. PLD assistance mentioned but not expressible.',
    input: {
      model: { kind: 'hf', repoId: 'deepseek-ai/DeepSeek-V4.1-Flash-1017' },
      quant: {
        kind: 'preset',
        preset: ggufBpw(
          3,
          'iQ-MLX advertised 3.0bpw; unpublished module mix approximated uniformly',
        ),
      },
      numerics: MLX,
      hardware: { deviceId: 'apple-m5-ultra-64c' },
      deployment: {
        offload: { lookupTables: { tier: 'nvme' } },
        speculative: {
          method: 'dspark',
          drafter: { kind: 'checkpoint' },
          draftLength: 4,
          acceptance: { kind: 'iid', alpha: ACCEPTANCE_PRESETS.dspark.prose.alpha },
        },
      },
      workload: { batch: 1, promptLen: 278, genLen: 1024, contextDepth: 278 },
    },
  },
];
