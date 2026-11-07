/**
 * Community benchmark reports for the simulator sanity check (docs/sanity-check.md), as typed data, each with the
 * harness input that matches it. The check: the simulator gives THEORETICAL MAXIMA, so for every asserted metric
 * `simulated ≥ 0.9 × reported` must hold; a report above that is a sim underestimate (a bug in a preset number or
 * an op cost) unless it is a listed exception with a reason.
 *
 * Sources (collected 2026-11-07):
 * - LocalMaxxing (`lm-NN`, NN = the row of the researchers' curated table): `https://localmaxxing.com/api/speed-tests`
 *   records; prefill there is promptTokens / CLIENT TTFT (HTTP, tokenization and the first decode step included)
 *   over short (~300-token) prompts, so prefill ratios are expected to be very low.
 * - Web (`web-NN`, NN = the row of the web table): llama.cpp GitHub discussions, r/LocalLLaMA (via a redlib mirror),
 *   mlx-lm BENCHMARKS.md, blogs and gists. llama-bench `ppN` / `tgN` are separate tests from an empty cache.
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
 */

import type { SimOptions } from '../engine/types';
import { ARC_PRO_B65 } from '../hw/presets/intel';
import { PCIE_4_X16, PCIE_4_X16_DIRECT, PCIE_4_X8, PCIE_5_X16_DIRECT, SPARK_CONNECTX_7 } from '../hw/presets/links';
import { RADEON_AI_PRO_R9700 } from '../hw/presets/amd';
import { DGX_SPARK } from '../hw/presets/nvidia-datacenter';
import { RTX_3090, RTX_4090 } from '../hw/presets/nvidia';
import { RTX_5090_PCIE_2, pcieBox, socCluster } from '../hw/presets/topologies';
import type { ModelSpec } from '../ir/model';
import { REF_DENSE_8B } from '../models/reference';
import { bpwFormat, type QuantPreset, type SanityInput, type SanityNumerics } from './harness';
import { GLM_4_5_AIR, GPT_OSS_120B, GPT_OSS_20B, LLAMA3_70B, QWEN3_30B_A3B, QWEN3_NEXT_80B_A3B, QWQ_32B } from './models';

// ---------------------------------------------------------------------------------------------------------------
// Types

/** The model a report is simulated with. */
export type ReportModel =
  /** A hand-written spec (`models.ts`, `models/reference.ts`). */
  | { kind: 'spec'; spec: ModelSpec }
  /** A Hugging Face fixture, loaded by the test with `sanityRepoFiles(repoId, { dir })` (`dir: 'quant'` for quantized repos). */
  | { kind: 'hf'; repoId: string; dir?: 'quant' };

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
 * - `excluded`: not compared (e.g. speculative decoding, whose decode can legitimately beat the autoregressive bound);
 *   the ratio is still printed.
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
  /** Stable id: `lm-NN` (LocalMaxxing table row) or `web-NN` (web table row). */
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
  /** The harness input that matches the report (its `model` is a {@link ReportModel}). */
  input: Omit<SanityInput, 'model'> & { model: ReportModel };
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
    decode: {
      tokensPerSec: 274.9,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'MTP speculative decoding (8 draft tokens) is on' },
    },
    caveats: 'Spec decode inflates decode; prefill is the usable number.',
    input: {
      model: { kind: 'hf', repoId: 'Qwen/Qwen3.8-27B-FP8', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'h200-sxm' },
      workload: { promptLen: 345, genLen: 512, contextDepth: 345 },
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
    decode: {
      tokensPerSec: 76,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'DSPARK speculative decoding always on (mean accepted length 2.85)' },
    },
    caveats: 'Verified. The only DeepSeek-V4.1-Flash data; every record has spec decoding on.',
    input: {
      model: { kind: 'hf', repoId: 'deepseek-ai/DeepSeek-V4.1-Flash' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'dgx-spark', topology: socCluster(DGX_SPARK, 4, SPARK_CONNECTX_7) },
      parallel: { tp: 4 },
      workload: { promptLen: 274, genLen: 1024, contextDepth: 274 },
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
    decode: {
      tokensPerSec: 227.3,
      kind: 'perSequence',
      check: { mode: 'excluded', reason: 'DFlash2 speculative decoding is on' },
    },
    caveats: 'Prefill may include the draft path. Fixture LM head is nvfp4, the report says BF16 (prefill hardly changes).',
    input: {
      model: { kind: 'hf', repoId: 'nvidia/Qwen3.8-27B-NVFP4', dir: 'quant' },
      quant: 'native',
      numerics: VLLM,
      hardware: { deviceId: 'rtx-5090', topology: RTX_5090_PCIE_2, unlocks: { 'pcie-p2p': 'direct' } },
      parallel: { tp: 2 },
      workload: { promptLen: 512, genLen: 1024, contextDepth: 512 },
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
];
