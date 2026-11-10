/**
 * AMD device presets: Radeon (RDNA4) cards, the Ryzen AI Max+ 395 "Strix Halo" APU (RDNA 3.5 iGPU) and
 * Instinct accelerators (CDNA3, CDNA4). Every number is cited in `sources[]` (with the page/table it comes
 * from) or listed in `estimated[]` with the reason. See docs/simulation-model.md §4 ("Hardware").
 *
 * How the peaks are written: AMD defines every peak as compute units × FLOP per clock per CU × the peak
 * (boost) engine clock (CDNA3 whitepaper footnote: "taking the engine clock from the highest DPM state and
 * multiplying it by xx CUs per GPU"). The presets use exactly that product ({@link cuPeakFlopsPerSec}), from
 * the per-CU rates of AMD's architecture documents, so a reviewer can check each number against the
 * product page (which rounds; the tests in amd.test.ts compare both). FLOP counts use 1 FMA = 2 FLOP.
 *
 * Conventions and traps shared by these presets:
 * - Sparsity: AMD calls its structured sparsity "4:2" (RDNA4 SWMMAC, CDNA SMFMAC: at most 2 non-zero values in
 *   each group of 4 of the A matrix). That is the same pattern as NVIDIA's 2:4, so it is recorded as
 *   `sparsity: '2:4'`. AMD's "with Structured Sparsity" figures are 2× dense and go only in the `'2:4'` entries.
 * - Accumulate: AMD matrix units run fp16/bf16 inputs at the same rate whether they accumulate in fp16 or
 *   fp32 (AMD matrix instruction calculator), and fp8 only accumulates in fp32. So only fp32-accumulate
 *   entries are listed; an fp16-accumulate request falls back to them (docs §4, peak lookup step 2).
 * - RDNA vector FP32: AMD's headline "FP32 Vector" counts DUAL issue (VOPD / wave64), which compilers rarely
 *   reach for FMA-heavy code (Chips and Cheese). `vectorFlopsPerSec.fp32` is the SINGLE-issue rate
 *   (stream processors × 2 FLOP × clock), i.e. half the headline; packed fp16 is 2× that and matches AMD's
 *   "FP16 Vector" figure. CDNA has no such caveat.
 * - Capacities are recorded as quoted in SI bytes (gb(32) = 32e9 B); HBM/GDDR is physically GiB, so SI is
 *   conservative. Cache sizes are binary (MiB). `l2Bytes` holds the LAST-LEVEL cache (the field's definition),
 *   which on these parts is the Infinity Cache (MALL), not the L2; the L2 is given in comments.
 * - Inputs AMD lists no rate for are omitted (e.g. int4 on CDNA, TF32 on CDNA4); the engine falls back to a
 *   wider input and warns. FP64 matrix rates are given in comments only: the peak table has no fp64
 *   accumulator and LLM inference does not use fp64.
 *
 * Not modeled (yet):
 * - Strix Halo XDNA2 NPU (AMD: "NPU TOPS Up to 50 TOPS"). Omitted rather than added as an `'npu'` engine:
 *   (1) the 50 TOPS precision is not stated (int8 is only the usual convention), and AMD gives no bf16 or
 *   vector rate, while an engine needs a vector fp32 peak as the end of its fallback chain, so we would invent
 *   one; (2) its memory-bandwidth cap (~50 GB/s, arXiv 2512.13282) was measured on Strix Point, not Strix
 *   Halo; (3) it can address under half of system DRAM (FastFlowLM docs), which a per-device capacity cannot
 *   express; (4) measured LLM decode on it (9–12 tok/s for 8B models) is far below the iGPU, so it is never the
 *   engine that sets the theoretical maximum. Revisit if AMD publishes an XDNA2 datasheet.
 * - Instinct MI455X (CDNA5) IS included, but only with AMD's published per-device figures (no per-CU derivation):
 *   its datasheet says "COMPUTE UNITS 256" while its product page says "Work Group Processors 256", so per-CU
 *   rates are ambiguous, and no CDNA5 ISA is public, so its tiling is a CDNA4 proxy (estimated).
 * - Instinct MI350X (same silicon as the MI355X at 2200 MHz, 1000 W) is not in this preset set.
 * - Interconnects (xGMI, USB4, Ethernet) are links, in links.ts / topologies.ts.
 */

import type { MathFormat } from '../../ir/dtype';
import type {
  ComputeEngine,
  DeviceSpec,
  EstimatedField,
  GemmTiling,
  MatmulPeak,
  SourceRef,
} from '../types';
import { gb, gbPerSec, tbPerSec, tflops } from '../units';
import { AMD_PEERDIRECT_SOURCE, NO_RDMA_SOURCE_ESTIMATE } from './gpudirect';
import { AMD_CDNA3_SFU, AMD_CDNA4_SFU, AMD_MI455X_SFU, AMD_RDNA_SFU, engineSfu } from './sfu';

/** Field path of every AMD preset's SFU rate (one engine each). */
const SFU_FIELD = 'engines.0.sfuOpsPerSec';

/** One mebibyte (cache sizes are binary). Unit: bytes. */
const MIB = 1024 * 1024;

/**
 * Peak rate of a matrix or vector unit = compute units × FLOP per clock per CU × clock.
 * Units: count × FLOP/(clk·CU) × Hz = FLOP/s (OP/s for integer inputs). Per device.
 */
export function cuPeakFlopsPerSec(
  computeUnits: number,
  flopPerClockPerCu: number,
  clockHz: number,
): number {
  return computeUnits * flopPerClockPerCu * clockHz;
}

/** A dense peak and, when `sparseFlopPerClockPerCu` is given, its 4:2 (= `'2:4'`) structured-sparse twin. */
function cuPeaks(
  input: MathFormat,
  computeUnits: number,
  clockHz: number,
  denseFlopPerClockPerCu: number,
  sparseFlopPerClockPerCu?: number,
): MatmulPeak[] {
  const accumulate = input === 'int8' || input === 'int4' ? 'int32' : 'fp32';
  const peaks: MatmulPeak[] = [
    {
      input,
      accumulate,
      sparsity: 'dense',
      flopsPerSec: cuPeakFlopsPerSec(computeUnits, denseFlopPerClockPerCu, clockHz),
    },
  ];
  if (sparseFlopPerClockPerCu !== undefined) {
    peaks.push({
      input,
      accumulate,
      sparsity: '2:4',
      flopsPerSec: cuPeakFlopsPerSec(computeUnits, sparseFlopPerClockPerCu, clockHz),
    });
  }
  return peaks;
}

/** Field paths (`engines.0.matmulPeaks.<i>`) of the engine-0 peaks whose input is in `inputs`. */
function peakPaths(peaks: MatmulPeak[], inputs: MathFormat[]): string[] {
  return peaks.flatMap((p, i) => (inputs.includes(p.input) ? [`engines.0.matmulPeaks.${i}`] : []));
}

/** Tiling with an `mmaM × mmaN × mmaK` instruction shape (elements) and the given CTA (workgroup) tiles. */
function tiling(
  mmaM: number,
  mmaN: number,
  mmaK: number,
  ctaTiles: { m: number; n: number }[],
): GemmTiling {
  return { mmaM, mmaN, mmaK, ctaTiles: ctaTiles.map((t) => ({ ...t })), ctasPerUnit: 1 };
}

/** `engines.0.gemmTiling.<key>.{mmaM,mmaN,mmaK}` for each key: the ISA-sourced part of a tiling. */
function mmaShapePaths(keys: string[]): string[] {
  return keys.flatMap((k) => ['mmaM', 'mmaN', 'mmaK'].map((f) => `engines.0.gemmTiling.${k}.${f}`));
}

/** The estimated part of each tiling (CTA tiles and residency), with a shared reason. */
function ctaEstimates(keys: string[], tileReason: string): EstimatedField[] {
  return keys.flatMap((k) => [
    { field: `engines.0.gemmTiling.${k}.ctaTiles`, reason: tileReason },
    {
      field: `engines.0.gemmTiling.${k}.ctasPerUnit`,
      reason:
        'One resident large workgroup per CU (register / LDS bound for these tile sizes); kernel-dependent, not a vendor spec.',
    },
  ]);
}

// ---------------------------------------------------------------------------------------------------------
// Shared sources
// ---------------------------------------------------------------------------------------------------------

/** AMD's matrix instruction calculator (ROCm GitHub): cycles and FLOP per instruction per architecture. */
const AMIC_URL = 'https://github.com/ROCm/amd_matrix_instruction_calculator';
/** GPUOpen: per-CU dense and sparse WMMA rates, RDNA4 vs RDNA3. */
const GPUOPEN_GENAI_URL =
  'https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/';
/** GPUOpen: WMMA on RDNA3 (per-CU rates and the 16×16×16 shape). */
const GPUOPEN_RDNA3_URL = 'https://gpuopen.com/learn/wmma_on_rdna3/';
/** RDNA4 ISA reference guide (§7.12 WMMA / SWMMAC). */
const RDNA4_ISA_URL = 'https://gpuopen.com/download/rdna4-instruction-set-architecture.pdf';
/** ROCm blog: FP16 GEMM on RDNA (peaks of the 9070 XT and 8060S; hipBLASLt measurements). */
const ROCM_FP16_GEMM_URL =
  'https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html';
/** ROCm blog: FP8 GEMM on RDNA4 (FP8 WMMA accumulates into FP32 only). */
const ROCM_FP8_GEMM_URL =
  'https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp8-gemm/README.html';
/** Chips and Cheese: microbenchmarking RDNA3 (dual-issue VOPD is rarely emitted for FMA). */
const CNC_RDNA3_URL =
  'https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture';
/** AMD CDNA 3 architecture whitepaper (Table 1 per-CU rates p. 7; MI300X spec table p. 24). */
const CDNA3_WP_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf';
/** AMD CDNA 4 architecture whitepaper (Table 1 per-CU rates p. 8; Table 2 MI350X / MI355X specs p. 18). */
const CDNA4_WP_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf';
/** AMD Instinct CDNA4 ISA reference (§7.1.2 Table 28 MFMA, Table 33 SMFMAC). */
const CDNA4_ISA_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf';

// ---------------------------------------------------------------------------------------------------------
// RDNA4: Radeon AI PRO R9700 and Radeon RX 9070 XT (both Navi 48, 64 CU)
// ---------------------------------------------------------------------------------------------------------

/**
 * RDNA4 dense WMMA rates per CU per clock (GPUOpen "Dense WMMA rates … RX 9070 XT FLOPS/clock/CU"):
 * FP16 1024, BF16 1024, FP8 2048, BF8 2048, IU8 2048, IU4 4096. Unit: FLOP (OP) / (clk · CU).
 */
const RDNA4_DENSE_FLOP_PER_CLK_PER_CU = {
  fp16: 1024,
  bf16: 1024,
  fp8: 2048,
  int8: 2048,
  int4: 4096,
} as const;
/** RDNA4 4:2-sparse (SWMMAC) rates: 2× dense ("effectively doubling the performance of standard dense WMMA"). */
const RDNA4_SPARSE_FLOP_PER_CLK_PER_CU = {
  fp16: 2048,
  bf16: 2048,
  fp8: 4096,
  int8: 4096,
  int4: 8192,
} as const;
/** RDNA lanes per CU: 64 stream processors (4096 SP / 64 CU on Navi 48). Single-issue FP32 FMA = 2 FLOP per lane. */
const RDNA_SP_PER_CU = 64;

/** Representative rocWMMA / hipBLASLt workgroup tiles for RDNA (elements, m × n); multiples of 16. Estimated. */
const RDNA_CTA_TILES = [
  { m: 256, n: 128 },
  { m: 128, n: 128 },
  { m: 128, n: 64 },
  { m: 64, n: 64 },
];
const RDNA_CTA_REASON =
  'Representative rocWMMA / hipBLASLt workgroup (macro) tiles for wave32 WMMA kernels; not a vendor spec.';

/** RDNA4 tiling keys: WMMA 16×16×16 for f16/bf16/fp8/iu8; iu4 reaches its full rate only as 16×16×32. */
const RDNA4_TILING_KEYS = ['default', 'fp16', 'bf16', 'fp8', 'int8', 'int4'];

/**
 * An RDNA4 matrix engine with `computeUnits` CUs at `boostClockHz`.
 * Matmul peak = CUs × per-CU WMMA rate × boost clock (dense and 4:2 sparse); vector fp32 = CUs × 64 SP × 2 FLOP
 * × boost (single issue), vector fp16 = 2× that (packed).
 */
function rdna4Engine(
  name: string,
  computeUnits: number,
  boostClockHz: number,
  sfuOpsPerSec: number,
): ComputeEngine {
  const d = RDNA4_DENSE_FLOP_PER_CLK_PER_CU;
  const s = RDNA4_SPARSE_FLOP_PER_CLK_PER_CU;
  const fp32SingleIssueFlopsPerSec = cuPeakFlopsPerSec(
    computeUnits,
    RDNA_SP_PER_CU * 2,
    boostClockHz,
  );
  return {
    id: 'gpu',
    name,
    kind: 'gpu',
    computeUnits,
    matmulPeaks: [
      ...cuPeaks('fp16', computeUnits, boostClockHz, d.fp16, s.fp16),
      ...cuPeaks('bf16', computeUnits, boostClockHz, d.bf16, s.bf16),
      ...cuPeaks('fp8', computeUnits, boostClockHz, d.fp8, s.fp8),
      ...cuPeaks('int8', computeUnits, boostClockHz, d.int8, s.int8),
      ...cuPeaks('int4', computeUnits, boostClockHz, d.int4, s.int4),
    ],
    vectorFlopsPerSec: {
      // Single issue; AMD's "FP32 Vector" headline is 2× this (dual-issue VOPD).
      fp32: fp32SingleIssueFlopsPerSec,
      // Packed FP16 (2 halves per lane) = 2× single-issue FP32; equals AMD's "FP16 Vector" figure.
      // No vendor bf16 vector rate: omitted (falls back to fp32).
      fp16: 2 * fp32SingleIssueFlopsPerSec,
    },
    sfuOpsPerSec,
    gemmTiling: {
      default: tiling(16, 16, 16, RDNA_CTA_TILES),
      fp16: tiling(16, 16, 16, RDNA_CTA_TILES),
      bf16: tiling(16, 16, 16, RDNA_CTA_TILES),
      fp8: tiling(16, 16, 16, RDNA_CTA_TILES),
      int8: tiling(16, 16, 16, RDNA_CTA_TILES),
      int4: tiling(16, 16, 32, RDNA_CTA_TILES),
    },
  };
}

/** RDNA4 sources shared by both cards: per-CU rates, accumulate rule, instruction shapes. */
const RDNA4_SHARED_SOURCES: SourceRef[] = [
  {
    url: GPUOPEN_GENAI_URL,
    title: 'GPUOpen: Accelerating generative AI on AMD Radeon GPUs (RDNA4 per-CU WMMA rates)',
    fields: ['engines.0.matmulPeaks'],
    note:
      '"Dense WMMA rates: … RX 9070 XT FLOPS/clock/CU FP16 1024, BF16 1024, FP8 2048, BF8 2048, IU8 2048, IU4 4096"; ' +
      '"Sparse WMMA rates: … FP16 2048 BF16 2048 FP8 4096 BF8 4096 IU8 4096 IU4 8192"; "hardware support for 4:2 ' +
      'structured sparsity, effectively doubling the performance of standard dense WMMA operations". ' +
      'Each peak = 64 CU × rate × boost clock.',
  },
  {
    url: AMIC_URL,
    title: 'AMD matrix instruction calculator (commit 2ef9189), -a RDNA4',
    fields: ['engines.0.matmulPeaks', ...mmaShapePaths(RDNA4_TILING_KEYS)],
    note:
      'v_wmma_f32_16x16x16_f16 and v_wmma_f16_16x16x16_f16 both 16 cycles, "FLOPs/WGP/cycle: 2048" (= 1024/CU): fp16 and ' +
      'fp32 accumulate run at the same rate, so only fp32-accumulate entries are listed. v_wmma_f32_16x16x16_fp8_fp8 ' +
      '8 cycles (4096/WGP); v_wmma_i32_16x16x16_iu8 8 cycles; v_wmma_i32_16x16x32_iu4 8 cycles (the full IU4 rate; ' +
      '16x16x16 iu4 is half). Sparse v_swmmac_f32_16x16x32_f16 16 cycles = 2× dense.',
  },
  {
    url: RDNA4_ISA_URL,
    title: 'AMD RDNA4 Instruction Set Architecture reference guide, §7.12 (WMMA / SWMMAC)',
    fields: mmaShapePaths(RDNA4_TILING_KEYS),
    note: 'Wave32 WMMA shapes: 16x16x16 for f16, bf16, fp8/bf8 and iu8; iu4 16x16x16 or 16x16x32; SWMMAC 16x16x32 / 16x16x64.',
  },
  {
    url: ROCM_FP8_GEMM_URL,
    title: 'ROCm blog: FP8 GEMM on RDNA4',
    fields: ['engines.0.matmulPeaks'],
    note: '"FP8 WMMA accumulates into FP32; there is no FP16 accumulator variant."',
  },
];

/** Estimated vector-fp32 entry for an RDNA card: the single-issue convention with its arithmetic. */
function rdnaVectorFp32Estimate(
  sp: number,
  boostGhz: number,
  headlineTflops: number,
): EstimatedField {
  return {
    field: 'engines.0.vectorFlopsPerSec.fp32',
    reason:
      `Single-issue FP32: ${sp} SP × 2 FLOP × ${boostGhz} GHz = ${((sp * 2 * boostGhz) / 1000).toFixed(2)} TFLOP/s. ` +
      `AMD's ${headlineTflops} TFLOP/s "FP32 Vector" headline counts dual issue (VOPD / wave64, 2×), which compilers ` +
      `rarely emit for FMA (Chips and Cheese, ${CNC_RDNA3_URL}); the single-issue rate is the reachable peak.`,
  };
}

/** Radeon AI PRO R9700 product page. */
const R9700_URL =
  'https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html';
/** R9700 boost clock: "Boost Frequency Up to 2920 MHz" (game clock 2350 MHz). Unit: Hz. */
const R9700_BOOST_CLOCK_HZ = 2.92e9;
/** R9700 SFU: 64 CU × 16 exp2/clk × 2.92 GHz (estimate, quarter-rate transcendental unit). */
const R9700_SFU = engineSfu(AMD_RDNA_SFU, 64, 'CU', R9700_BOOST_CLOCK_HZ, SFU_FIELD);
/** The R9700's RDNA4 engine: 64 CU at the 2920 MHz boost clock. */
const R9700_ENGINE = rdna4Engine(
  'Navi 48 RDNA4 AI Accelerators (WMMA)',
  64,
  R9700_BOOST_CLOCK_HZ,
  R9700_SFU.sfuOpsPerSec,
);

/**
 * AMD Radeon AI PRO R9700 (Navi 48, RDNA4, 64 CU, 32 GB GDDR6, 640 GB/s), all peaks at the 2920 MHz BOOST
 * clock (AMD's FP16 matrix 191 TFLOP/s = 64 × 1024 × 2.92 GHz, so the page assumes boost; the 2350 MHz game
 * clock would give 80% of that).
 *
 * Traps: the "383 TFLOPs" FP16 and "1531 TOPs" INT4 headlines are 4:2-sparse; dense FP16 is 191. AMD's
 * 47.8 TFLOP/s FP32 vector figure counts dual issue: the preset uses the single-issue 23.9. AMD lists no BF16
 * rate on the page; BF16 runs at the FP16 rate (GPUOpen per-CU table), marked estimated.
 */
export const RADEON_AI_PRO_R9700: DeviceSpec = {
  id: 'radeon-ai-pro-r9700',
  name: 'AMD Radeon AI PRO R9700',
  vendor: 'AMD',
  engines: [R9700_ENGINE],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6',
    // "32 GB" as quoted, recorded SI (32e9 B); physically 32 GiB, so SI is conservative.
    capacityOptionsBytes: [gb(32)],
    // 256-bit × 20 Gbps / 8 = 640 GB/s ("Peak Memory Bandwidth 640 GB/s").
    bandwidthBytesPerSec: gbPerSec(640),
  },
  // Infinity Cache 64 MB (last-level cache); the Navi 48 L2 is 8 MB (Hot Chips 2025).
  l2Bytes: 64 * MIB,
  // "PCIe® 5.0 x16" host interface (product page).
  hostInterface: { generation: 5, lanes: 16 },
  gpuDirectRdma: false,
  sources: [
    {
      url: R9700_URL,
      title: 'AMD Radeon AI PRO R9700 product page',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec.fp16',
        'memory',
        'l2Bytes',
      ],
      note:
        '"Compute Units 64", "Stream Processors 4096", "Boost Frequency Up to 2920 MHz"; FP16 Matrix 191 / with ' +
        'Structured Sparsity 383 TFLOPs; FP8 (E5M2, E4M3) 383 / 766 TFLOPs; INT8 383 / 766 TOPs; INT4 766 / 1531 TOPs ' +
        '(each = 64 CU × per-CU rate × 2.92 GHz, rounded); FP32 Vector 47.8 TFLOPs (dual issue); FP16 Vector 47.8 TFLOPs; ' +
        '32 GB GDDR6, 256-bit, 640 GB/s; Infinity Cache 64 MB. No BF16 rate is listed.',
    },
    ...RDNA4_SHARED_SOURCES,
    {
      url: R9700_URL,
      title: 'AMD Radeon AI PRO R9700 product page',
      fields: ['hostInterface'],
      note: 'Host interface "PCIe® 5.0 x16".',
    },
    ...R9700_SFU.sources,
  ],
  estimated: [
    NO_RDMA_SOURCE_ESTIMATE,
    ...peakPaths(R9700_ENGINE.matmulPeaks, ['bf16']).map((field) => ({
      field,
      reason:
        'Not on the R9700 page. 64 CU × 1024 (dense) or 2048 (4:2 sparse) FLOP/clk/CU (GPUOpen BF16 = FP16 rate) × ' +
        '2.92 GHz = 191.4 / 382.7 TFLOP/s.',
    })),
    rdnaVectorFp32Estimate(4096, 2.92, 47.8),
    ...ctaEstimates(RDNA4_TILING_KEYS, RDNA_CTA_REASON),
    ...R9700_SFU.estimated,
  ],
};

/** Radeon RX 9070 XT product page. */
const RX_9070_XT_URL =
  'https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html';
/** RX 9070 XT boost clock: "Boost Frequency Up to 2970 MHz" (game clock 2400 MHz). Unit: Hz. */
const RX_9070_XT_BOOST_CLOCK_HZ = 2.97e9;
/** RX 9070 XT SFU: 64 CU × 16 exp2/clk × 2.97 GHz (estimate, quarter-rate transcendental unit). */
const RX_9070_XT_SFU = engineSfu(AMD_RDNA_SFU, 64, 'CU', RX_9070_XT_BOOST_CLOCK_HZ, SFU_FIELD);
/** The RX 9070 XT's RDNA4 engine: 64 CU at the 2970 MHz boost clock. */
const RX_9070_XT_ENGINE = rdna4Engine(
  'Navi 48 RDNA4 AI Accelerators (WMMA)',
  64,
  RX_9070_XT_BOOST_CLOCK_HZ,
  RX_9070_XT_SFU.sfuOpsPerSec,
);

/**
 * AMD Radeon RX 9070 XT (Navi 48, RDNA4, 64 CU, 16 GB GDDR6, 640 GB/s), AMD's top consumer card as of
 * 2026-11, all peaks at the 2970 MHz BOOST clock (FP16 matrix 195 = 64 × 1024 × 2.97 GHz).
 *
 * Traps: as the R9700 (same die, 50 MHz higher boost, half the memory). "389 TFLOPs" FP16 and "1557 TOPs"
 * INT4 are 4:2-sparse; the 48.7 TFLOP/s FP32 vector headline counts dual issue (single-issue 24.3 is used).
 * Measured: hipBLASLt FP16 GEMM 136.6 TFLOP/s at 8192³ (70% of 195, ROCm FP16 GEMM blog).
 */
export const RADEON_RX_9070_XT: DeviceSpec = {
  id: 'radeon-rx-9070-xt',
  name: 'AMD Radeon RX 9070 XT',
  vendor: 'AMD',
  engines: [RX_9070_XT_ENGINE],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6',
    // "16 GB" as quoted, recorded SI (16e9 B).
    capacityOptionsBytes: [gb(16)],
    // 256-bit × 20 Gbps / 8 = 640 GB/s ("Memory Bandwidth Up to 640 GB/s").
    bandwidthBytesPerSec: gbPerSec(640),
  },
  // Infinity Cache 64 MB (last-level cache); L2 8 MB.
  l2Bytes: 64 * MIB,
  // "PCIe® 5.0 x16" host interface (product page).
  hostInterface: { generation: 5, lanes: 16 },
  gpuDirectRdma: false,
  sources: [
    {
      url: RX_9070_XT_URL,
      title: 'AMD Radeon RX 9070 XT product page',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec.fp16',
        'memory',
        'l2Bytes',
      ],
      note:
        '"Compute Units 64", "Stream Processors 4096", "Boost Frequency Up to 2970 MHz"; FP16 Matrix 195 / sparse 389 ' +
        'TFLOPs; FP8 389 / 779 TFLOPs; INT8 389 / 779 TOPs; INT4 779 / 1557 TOPs (each = 64 CU × per-CU rate × 2.97 GHz, ' +
        'rounded); FP32 Vector 48.7 TFLOPs (dual issue); FP16 Vector 48.7 TFLOPs; "Up to 20 Gbps", 16 GB, 256-bit, ' +
        '"Up to 640 GB/s"; Infinity Cache 64 MB. No BF16 rate is listed.',
    },
    ...RDNA4_SHARED_SOURCES,
    {
      url: RX_9070_XT_URL,
      title: 'AMD Radeon RX 9070 XT product page',
      fields: ['hostInterface'],
      note: 'Host interface "PCIe® 5.0 x16".',
    },
    {
      url: ROCM_FP16_GEMM_URL,
      title: 'ROCm blog: FP16 GEMM on RDNA',
      fields: ['engines.0.matmulPeaks'],
      note: '"The peak FP16 throughput is … 195 on the 9070 XT"; measured hipBLASLt 136.58 TFLOPS at M=N=K=8192.',
    },
    ...RX_9070_XT_SFU.sources,
  ],
  estimated: [
    NO_RDMA_SOURCE_ESTIMATE,
    ...peakPaths(RX_9070_XT_ENGINE.matmulPeaks, ['bf16']).map((field) => ({
      field,
      reason:
        'Not on the 9070 XT page. 64 CU × 1024 (dense) or 2048 (4:2 sparse) FLOP/clk/CU (GPUOpen BF16 = FP16 rate) × ' +
        '2.97 GHz = 194.6 / 389.3 TFLOP/s.',
    })),
    rdnaVectorFp32Estimate(4096, 2.97, 48.7),
    ...ctaEstimates(RDNA4_TILING_KEYS, RDNA_CTA_REASON),
    ...RX_9070_XT_SFU.estimated,
  ],
};

// ---------------------------------------------------------------------------------------------------------
// Strix Halo: Ryzen AI Max+ 395 (Radeon 8060S, RDNA 3.5, gfx1151)
// ---------------------------------------------------------------------------------------------------------

/** Ryzen AI Max+ 395 product page. */
const RYZEN_AI_MAX_395_URL =
  'https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html';
/** AMD blog: Variable Graphics Memory FAQ (256 GB/s; 96 GB VGM on Windows). */
const AMD_VGM_FAQ_URL =
  'https://www.amd.com/en/blogs/2025/faqs-amd-variable-graphics-memory-vram-ai-model-sizes-quantization-mcp-more.html';
/** AMD technical article: one-trillion-parameter LLM on a Strix Halo cluster (120 GB GTT on Linux). */
const AMD_TRILLION_URL =
  'https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html';
/** AMD ROCm documentation: Strix Halo system optimization (default GTT limit ≈ 50% of system RAM). */
const ROCM_STRIX_HALO_URL =
  'https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/strixhalo.html';
/** HP Z2 Mini G1a configurator (memory options shipped with the Ryzen AI Max+ PRO 395). */
const HP_Z2_MINI_G1A_CONFIG_URL = 'https://www.hp.com/us-en/workstations/z2-mini-a-configure.html';
/** Chips and Cheese: AMD's chiplet APU, an overview of Strix Halo (32 MB Infinity Cache). */
const CNC_STRIX_HALO_URL = 'https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix';

/** Radeon 8060S graphics clock: "Graphics Frequency 2900 MHz". Unit: Hz. */
const RADEON_8060S_CLOCK_HZ = 2.9e9;
/** Radeon 8060S compute units: "Graphics Core Count 40". */
const RADEON_8060S_CUS = 40;
/** Radeon 8060S SFU: 40 CU × 16 exp2/clk × 2.9 GHz (estimate, quarter-rate transcendental unit). */
const RADEON_8060S_SFU = engineSfu(
  AMD_RDNA_SFU,
  RADEON_8060S_CUS,
  'CU',
  RADEON_8060S_CLOCK_HZ,
  SFU_FIELD,
);
/**
 * RDNA3 / RDNA 3.5 WMMA rates per CU per clock (GPUOpen "RX 7900 XTX FLOPS/clock/CU FP16 512 BF16 512 IU8 512
 * IU4 1024"); gfx1151 uses the RDNA3 WMMA instructions (AMD calculator maps GFX1151 to RDNA3). Dense only:
 * RDNA3 has no sparse WMMA and no FP8. Unit: FLOP (OP) / (clk · CU).
 */
const RDNA3_DENSE_FLOP_PER_CLK_PER_CU = { fp16: 512, bf16: 512, int8: 512, int4: 1024 } as const;
/** Strix Halo tiling keys: RDNA3 WMMA is 16×16×16 for every input. */
const RDNA3_TILING_KEYS = ['default', 'fp16', 'bf16', 'int8', 'int4'];

/** The Radeon 8060S engine: 40 CU × per-CU WMMA rate × 2.9 GHz; vector fp32 single-issue, fp16 packed 2×. */
function radeon8060sEngine(): ComputeEngine {
  const cus = RADEON_8060S_CUS;
  const clk = RADEON_8060S_CLOCK_HZ;
  const d = RDNA3_DENSE_FLOP_PER_CLK_PER_CU;
  // 40 CU × 64 SP × 2 FLOP × 2.9 GHz = 14.85 TFLOP/s (single issue; dual-issue VOPD would be 29.7).
  const fp32SingleIssueFlopsPerSec = cuPeakFlopsPerSec(cus, RDNA_SP_PER_CU * 2, clk);
  return {
    id: 'gpu',
    name: 'Radeon 8060S (RDNA 3.5 WMMA)',
    kind: 'gpu',
    computeUnits: cus,
    matmulPeaks: [
      // 40 × 512 × 2.9 GHz = 59.39 TFLOP/s: AMD's "FP16 AI Compute Up to 60 TFLOPS".
      ...cuPeaks('fp16', cus, clk, d.fp16),
      ...cuPeaks('bf16', cus, clk, d.bf16),
      // 40 × 512 × 2.9 GHz = 59.39 TOP/s; INT4 40 × 1024 × 2.9 GHz = 118.78 TOP/s.
      ...cuPeaks('int8', cus, clk, d.int8),
      ...cuPeaks('int4', cus, clk, d.int4),
    ],
    vectorFlopsPerSec: {
      fp32: fp32SingleIssueFlopsPerSec,
      // Packed FP16 = 2× single-issue FP32 = 29.7 TFLOP/s (= WMMA fp16 / 2).
      fp16: 2 * fp32SingleIssueFlopsPerSec,
    },
    sfuOpsPerSec: RADEON_8060S_SFU.sfuOpsPerSec,
    gemmTiling: {
      default: tiling(16, 16, 16, RDNA_CTA_TILES),
      fp16: tiling(16, 16, 16, RDNA_CTA_TILES),
      bf16: tiling(16, 16, 16, RDNA_CTA_TILES),
      int8: tiling(16, 16, 16, RDNA_CTA_TILES),
      int4: tiling(16, 16, 16, RDNA_CTA_TILES),
    },
  };
}

const RADEON_8060S_ENGINE = radeon8060sEngine();

/**
 * AMD Ryzen AI Max+ 395 "Strix Halo" APU, modeled as its Radeon 8060S iGPU (40 CU RDNA 3.5 at 2900 MHz) on
 * 256-bit LPDDR5X-8000 unified memory (256 GB/s), peaks at the 2900 MHz graphics clock.
 *
 * Memory: `capacityOptionsBytes` are the SKU sizes (128 GB, 64 GB); the GPU can use only part of it. On
 * Windows, Variable Graphics Memory dedicates up to 96 GB (and the iGPU "can technically access a total
 * graphics memory size of 112GB" on 128 GB systems, AMD VGM FAQ); on Linux the TTM/GTT kernel parameters
 * (`ttm.pages_limit=30720000 amdgpu.gttsize=120000`) allow ~120 GB (AMD trillion-parameter article). The OS
 * and CPU use the rest. Measured GPU copy bandwidth 212 GB/s (≈83%, llm-tracker.info rocm_bandwidth_test).
 * `gpuUsableFraction` 0.5 is the Linux (ROCm) DEFAULT: "By default, the GTT limit is set to approximately 50% of
 * total system RAM" (AMD ROCm Strix Halo guide); the raised ~120 GB (0.94) or Windows VGM are the override.
 *
 * Traps: "Overall TOPS Up to 126" adds CPU, GPU and NPU, which never run one matmul together; "NPU TOPS Up to
 * 50" is the XDNA2 NPU, not modeled (see the file header). RDNA 3.5 has no sparse WMMA and no FP8: fp8 inputs
 * fall back to fp16. WMMA fp16 is 2× the packed-fp16 vector rate, not more. Measured hipBLASLt FP16 GEMM
 * 39.07 TFLOP/s at 8192³ (66% of 59.4). The GPU L2 is omitted: Chips and Cheese gives 2 MB in one article and
 * 4 MB in another, and AMD publishes neither. `l2Bytes` records the 32 MB Infinity Cache (last level).
 */
export const RYZEN_AI_MAX_PLUS_395: DeviceSpec = {
  id: 'ryzen-ai-max-plus-395',
  name: 'AMD Ryzen AI Max+ 395 (Radeon 8060S)',
  vendor: 'AMD',
  engines: [RADEON_8060S_ENGINE],
  memory: {
    kind: 'unified',
    technology: 'LPDDR5X',
    // SKU sizes as sold, recorded SI. GPU-allocatable: ≤ 96 GB (Windows VGM) / ~120 GB (Linux GTT) of 128 GB.
    capacityOptionsBytes: [gb(128), gb(64)],
    // 256-bit × 8000 MT/s / 8 = 256 GB/s ("doubles the bandwidth to 256 GB/s").
    bandwidthBytesPerSec: gbPerSec(256),
    // Linux default GTT limit ≈ 50% of system RAM (ROCm guide); raised (ttm.pages_limit) ~120 GB is an override.
    gpuUsableFraction: 0.5,
  },
  // Infinity Cache (MALL) 32 MB, the GPU's last-level cache; the L2 is ambiguous (2 or 4 MB) and omitted.
  l2Bytes: 32 * MIB,
  sources: [
    {
      url: RYZEN_AI_MAX_395_URL,
      title: 'AMD Ryzen AI Max+ 395 product page',
      fields: ['engines.0.computeUnits', 'engines.0.matmulPeaks', 'memory'],
      note:
        '"Graphics Model Radeon 8060S Graphics", "Graphics Core Count 40", "Graphics Frequency 2900 MHz"; "FP16 AI ' +
        'Compute Up to 60 TFLOPS" (= 40 × 512 × 2.9 GHz = 59.4); "System Memory Type 256-bit LPDDR5x", "Max. Memory ' +
        '128 GB", "Max Memory Speed LPDDR5x-8000" (256 × 8000 / 8 = 256 GB/s); "NPU TOPS Up to 50 TOPS".',
    },
    {
      url: ROCM_STRIX_HALO_URL,
      title: 'AMD ROCm documentation: AMD Strix Halo system optimization, "Memory settings"',
      fields: ['memory.gpuUsableFraction'],
      note:
        'GTT "is the memory pool used by applications such as PyTorch and other AI/compute workloads"; "By default, ' +
        'the GTT limit is set to approximately 50% of total system RAM." (Linux).',
    },
    {
      url: HP_Z2_MINI_G1A_CONFIG_URL,
      title: 'HP Z2 Mini G1a configurations (Ryzen AI Max+ PRO 395)',
      fields: ['memory.capacityOptionsBytes'],
      note: 'Memory size options "128 GB LPDDR5x" and "64 GB LPDDR5x" with the Ryzen AI Max+ PRO 395 (same silicon).',
    },
    {
      url: AMD_VGM_FAQ_URL,
      title: 'AMD blog: Variable Graphics Memory FAQ',
      fields: ['memory.bandwidthBytesPerSec'],
      note:
        '"doubles the bandwidth to 256 GB/s"; "the iGPU (set to 96GB VGM) can technically access a total graphics ' +
        'memory size of 112GB on AMD Ryzen AI MAX+ powered 128GB systems" (Windows).',
    },
    {
      url: AMD_TRILLION_URL,
      title: 'AMD: How to run a one-trillion-parameter LLM locally (Strix Halo cluster)',
      fields: ['memory.capacityOptionsBytes'],
      note: 'Linux: "increase our maximum VRAM allocation to 120GB per node" via ttm.pages_limit / amdgpu.gttsize.',
    },
    {
      url: GPUOPEN_RDNA3_URL,
      title: 'GPUOpen: How to accelerate AI applications on RDNA 3 using WMMA',
      fields: ['engines.0.matmulPeaks', ...mmaShapePaths(RDNA3_TILING_KEYS)],
      note:
        'Per-CU WMMA rates "RX 7900 XTX FLOPS/clock/CU FP16 512 BF16 512 IU8 512 IU4 1024"; "In the case of RDNA 3, ' +
        'only 16x16 tile sizes are supported" (16x16x16 for f16, bf16, iu8, iu4).',
    },
    {
      url: AMIC_URL,
      title: 'AMD matrix instruction calculator (commit 2ef9189), -a RDNA3 (covers GFX1150/1151)',
      fields: ['engines.0.matmulPeaks', ...mmaShapePaths(RDNA3_TILING_KEYS)],
      note:
        'v_wmma_f32_16x16x16_f16 and v_wmma_f16_16x16x16_f16 both 32 cycles, "FLOPs/WGP/cycle: 1024" (512/CU): fp16 ' +
        'and fp32 accumulate at the same rate; iu8 32 cycles, iu4 16 cycles.',
    },
    {
      url: ROCM_FP16_GEMM_URL,
      title: 'ROCm blog: FP16 GEMM on RDNA',
      fields: ['engines.0.matmulPeaks'],
      note:
        '"The peak FP16 throughput is … 59.4 on the 8060S"; "[the 8060S] uses the same WMMA instructions as RDNA 3"; ' +
        'measured hipBLASLt 39.07 TFLOPS at M=N=K=8192.',
    },
    {
      url: CNC_STRIX_HALO_URL,
      title: "Chips and Cheese: AMD's chiplet APU, an overview of Strix Halo",
      kind: 'review',
      fields: ['l2Bytes'],
      note: '"40 compute units, 32MB of Infinity Cache, and a boost clock of up to 2.9GHz".',
    },
    ...RADEON_8060S_SFU.sources,
  ],
  estimated: [
    ...peakPaths(RADEON_8060S_ENGINE.matmulPeaks, ['int8', 'int4']).map((field) => ({
      field,
      reason:
        'AMD quotes no GPU integer rate for the 8060S. 40 CU × 512 (IU8) or 1024 (IU4) OP/clk/CU (GPUOpen RDNA3 table) ' +
        '× 2.9 GHz = 59.4 / 118.8 TOP/s.',
    })),
    {
      field: 'engines.0.vectorFlopsPerSec.fp32',
      reason:
        'AMD publishes no FP32 figure for the 8060S. Single-issue: 40 CU × 64 SP × 2 FLOP × 2.9 GHz = 14.85 TFLOP/s ' +
        `(dual-issue VOPD would be 29.7, rarely reached: ${CNC_RDNA3_URL}).`,
    },
    {
      field: 'engines.0.vectorFlopsPerSec.fp16',
      reason:
        'Packed FP16 FMA (2 halves per lane) = 2 × 14.85 = 29.7 TFLOP/s; consistent with WMMA fp16 (512/CU) being 2× ' +
        'packed-FP16 vector (Chips and Cheese: WMMA gives "the same theoretical throughput as using dot product ' +
        'instructions"). No AMD figure.',
    },
    ...ctaEstimates(RDNA3_TILING_KEYS, RDNA_CTA_REASON),
    ...RADEON_8060S_SFU.estimated,
  ],
};

// ---------------------------------------------------------------------------------------------------------
// CDNA3: Instinct MI300X and MI325X (304 CU at 2100 MHz)
// ---------------------------------------------------------------------------------------------------------

/**
 * CDNA3 per-CU rates (CDNA3 whitepaper Table 1, AMD calculator -a CDNA3). Matrix: TF32 1024, FP16 2048,
 * BF16 2048, FP8 4096, INT8 4096, FP32 256 (FP64 256, not listed as a peak). Vector: FP64 128, FP32 256,
 * FP16 256 (CDNA4 whitepaper Table 1, MI300X column). Sparse (SMFMAC, 4:2 on A) = 2× dense.
 * Unit: FLOP (OP) / (clk · CU).
 */
const CDNA3_FLOP_PER_CLK_PER_CU = {
  tf32: 1024,
  fp16: 2048,
  bf16: 2048,
  fp8: 4096,
  int8: 4096,
  fp32Matrix: 256,
  fp64Vector: 128,
  fp32Vector: 256,
  fp16Vector: 256,
} as const;
/** MI300X / MI325X: "Compute Units 304". */
const CDNA3_MI300_CUS = 304;
/** MI300X / MI325X: "Peak Engine Clock 2100 MHz". Unit: Hz. */
const CDNA3_MI300_CLOCK_HZ = 2.1e9;
/** MI300X / MI325X SFU: 304 CU × 16 exp2/clk × 2.1 GHz (estimate, quarter-rate transcendentals). */
const CDNA3_MI300_SFU = engineSfu(
  AMD_CDNA3_SFU,
  CDNA3_MI300_CUS,
  'CU',
  CDNA3_MI300_CLOCK_HZ,
  SFU_FIELD,
);

/** Representative hipBLASLt / Tensile macro tiles for MFMA GEMMs (elements, m × n); multiples of 16. Estimated. */
const CDNA_CTA_TILES = [
  { m: 256, n: 256 },
  { m: 256, n: 128 },
  { m: 128, n: 128 },
  { m: 128, n: 64 },
];
const CDNA_CTA_REASON =
  'Representative hipBLASLt / Tensile macro tiles for MFMA GEMM kernels; not a vendor spec.';

/** CDNA3 tiling keys: MFMA 16×16×16 f16/bf16 (default), 16×16×32 fp8/i8, 16×16×8 xf32 (TF32). */
const CDNA3_TILING_KEYS = ['default', 'fp16', 'bf16', 'tf32', 'fp8', 'int8'];

/**
 * The MI300-family engine: 304 CU × per-CU rate × 2.1 GHz.
 * FP16 2048 → 1307.4; TF32 1024 → 653.7; FP8 / INT8 4096 → 2614.9; FP32 256 → 163.4 TFLOP/s; sparse 2×.
 */
function cdna3Engine(): ComputeEngine {
  const cus = CDNA3_MI300_CUS;
  const clk = CDNA3_MI300_CLOCK_HZ;
  const r = CDNA3_FLOP_PER_CLK_PER_CU;
  return {
    id: 'gpu',
    name: 'CDNA3 Matrix Cores (MFMA)',
    kind: 'gpu',
    computeUnits: cus,
    matmulPeaks: [
      ...cuPeaks('fp16', cus, clk, r.fp16, 2 * r.fp16),
      ...cuPeaks('bf16', cus, clk, r.bf16, 2 * r.bf16),
      // Sparse TF32 1307.4 is AMD's headline ("expected to be able to take advantage of fine-grained structure
      // sparsity … estimated 2x", footnote MI300-17); the CDNA3 SMFMAC list has f16/bf16/fp8/i8 variants only.
      ...cuPeaks('tf32', cus, clk, r.tf32, 2 * r.tf32),
      ...cuPeaks('fp8', cus, clk, r.fp8, 2 * r.fp8),
      ...cuPeaks('int8', cus, clk, r.int8, 2 * r.int8),
      ...cuPeaks('fp32', cus, clk, r.fp32Matrix),
      // FP64 matrix 304 × 256 × 2.1 GHz = 163.4 TFLOP/s: not listed (no fp64 accumulator key; unused by LLMs).
      // No int4 MFMA on CDNA3: int4 falls back to int8.
    ],
    vectorFlopsPerSec: {
      fp64: cuPeakFlopsPerSec(cus, r.fp64Vector, clk),
      fp32: cuPeakFlopsPerSec(cus, r.fp32Vector, clk),
      fp16: cuPeakFlopsPerSec(cus, r.fp16Vector, clk),
    },
    sfuOpsPerSec: CDNA3_MI300_SFU.sfuOpsPerSec,
    gemmTiling: {
      default: tiling(16, 16, 16, CDNA_CTA_TILES),
      fp16: tiling(16, 16, 16, CDNA_CTA_TILES),
      bf16: tiling(16, 16, 16, CDNA_CTA_TILES),
      tf32: tiling(16, 16, 8, CDNA_CTA_TILES),
      fp8: tiling(16, 16, 32, CDNA_CTA_TILES),
      int8: tiling(16, 16, 32, CDNA_CTA_TILES),
    },
  };
}

/** CDNA3 sources shared by the MI300X and MI325X (architecture and instruction shapes). */
const CDNA3_SHARED_SOURCES: SourceRef[] = [
  {
    url: CDNA3_WP_URL,
    title: 'AMD CDNA 3 architecture whitepaper, Table 1 (p. 7) and MI300X spec table (p. 24)',
    fields: [
      'engines.0.computeUnits',
      'engines.0.matmulPeaks',
      'engines.0.vectorFlopsPerSec.fp32',
      'engines.0.vectorFlopsPerSec.fp64',
    ],
    note:
      'Per CU per clock: FP64 matrix 256, FP64 vector 128, FP32 matrix 256, FP32 vector 256, TF32 1024, FP16 2048, BF16 2048, ' +
      'FP8 4096, INT8 4096; sparsity "doubling the computational throughput up to … 8K operations per clock for a CU". ' +
      'Footnote: peak = engine clock of the highest DPM state × CUs × FLOP/clk/CU. TYPO: Table 1 lists MI300X "Matrix ' +
      'TF32 … 490.3", which is the MI300A (228 CU) value; the spec table and datasheet give 653.7 (= 304 × 1024 × 2.1 GHz), ' +
      'which is used.',
  },
  {
    url: CDNA4_WP_URL,
    title: 'AMD CDNA 4 architecture whitepaper, Table 1 (MI300X column)',
    fields: ['engines.0.vectorFlopsPerSec.fp16'],
    note: '"Vector FP16 256 … 163.4 TF" for the MI300X (304 × 256 × 2.1 GHz).',
  },
  {
    url: AMIC_URL,
    title: 'AMD matrix instruction calculator (commit 2ef9189), -a CDNA3',
    fields: ['engines.0.matmulPeaks', ...mmaShapePaths(CDNA3_TILING_KEYS)],
    note:
      'v_mfma_f32_16x16x16_f16 16 cycles / 32x32x8_f16 32 cycles → 2048 FLOP/CU/clk; v_mfma_f32_16x16x32_fp8_fp8 and ' +
      'v_mfma_i32_16x16x32_i8 → 4096; v_mfma_f32_16x16x8_xf32 (TF32) → 1024; v_smfmac_f32_16x16x32_f16 → 4096; ' +
      'v_smfmac_{f32_16x16x64_fp8_fp8, i32_16x16x64_i8} → 8192. All MFMA accumulate in FP32 / I32. The 16x16 ' +
      'variants are used as the representative shapes.',
  },
];

/** Instinct MI300X data sheet. */
const MI300X_DS_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf';
/** Instinct MI300X product page. */
const MI300X_PAGE_URL = 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html';

/**
 * AMD Instinct MI300X (CDNA3, 304 CU, 192 GB HBM3, 5.3 TB/s), peaks at the 2100 MHz peak engine clock.
 *
 * Traps: "5.2 PFLOPs" FP8 and "2.6 PFLOPs" FP16 headlines are 4:2-sparse (dense 2614.9 / 1307.4). Sparse TF32
 * (1307.4) is AMD's estimate (see the engine comment). The CDNA3 whitepaper's Table 1 TF32 490.3 is a typo
 * (MI300A); 653.7 is used. Measured HBM ≈ 4.3 TB/s (81%, BabelStream, arXiv 2510.27583). L2 4 MB per XCD × 8.
 */
export const INSTINCT_MI300X: DeviceSpec = {
  id: 'instinct-mi300x',
  name: 'AMD Instinct MI300X',
  vendor: 'AMD',
  engines: [cdna3Engine()],
  memory: {
    kind: 'dedicated',
    technology: 'HBM3',
    // "Up to 192 GB HBM3", recorded SI.
    capacityOptionsBytes: [gb(192)],
    // 8192-bit × 5.2 GT/s / 8 = 5.32 TB/s, quoted "5.3 TB/s max. peak theoretical".
    bandwidthBytesPerSec: tbPerSec(5.3),
  },
  // 256 MB Infinity Cache (last-level cache).
  l2Bytes: 256 * MIB,
  gpuDirectRdma: true,
  sources: [
    AMD_PEERDIRECT_SOURCE,
    {
      url: MI300X_DS_URL,
      title: 'AMD Instinct MI300X data sheet',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec.fp32',
        'engines.0.vectorFlopsPerSec.fp64',
        'memory',
        'l2Bytes',
      ],
      note:
        '"AI PEAK THEORETICAL PERFORMANCE": TF32 / FP16 / BF16 / FP8 / INT8 653.7 / 1307.4 / 1307.4 / 2614.9 / 2614.9, with ' +
        'sparsity 1307.4 / 2614.9 / 2614.9 / 5229.8 / 5229.8 (footnote MI300-17, at 2,100 MHz); FP64 vector 81.7, FP32 vector ' +
        '163.4, FP64 matrix 163.4, FP32 matrix 163.4; "Up to 192 GB HBM3", 8192 bits, "Up to 5.2 GT/s", "5.3 TB/s max. peak ' +
        'theoretical"; 8 XCDs × 38 CUs, 4 MB L2 per XCD, 256 MB AMD Infinity Cache.',
    },
    {
      url: MI300X_PAGE_URL,
      title: 'AMD Instinct MI300X product page',
      fields: ['engines.0.computeUnits', 'engines.0.matmulPeaks'],
      note: '"Compute Units 304", "Matrix Cores 1216", "Peak Engine Clock 2100 MHz"; FP8 2.61 / sparse 5.22 PFLOPs; FP16 1.3 / 2.61 PFLOPs.',
    },
    ...CDNA3_SHARED_SOURCES,
    ...CDNA3_MI300_SFU.sources,
  ],
  estimated: [...ctaEstimates(CDNA3_TILING_KEYS, CDNA_CTA_REASON), ...CDNA3_MI300_SFU.estimated],
};

/** Instinct MI325X data sheet. */
const MI325X_DS_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf';
/** Instinct MI325X product page. */
const MI325X_PAGE_URL = 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html';

/**
 * AMD Instinct MI325X (CDNA3, 304 CU, 256 GB HBM3E, 6 TB/s): the MI300X compute die at the same 2100 MHz
 * (all peaks identical) with more and faster memory. Traps as the MI300X. TBP 1000 W.
 */
export const INSTINCT_MI325X: DeviceSpec = {
  id: 'instinct-mi325x',
  name: 'AMD Instinct MI325X',
  vendor: 'AMD',
  engines: [cdna3Engine()],
  memory: {
    kind: 'dedicated',
    technology: 'HBM3E',
    // "Dedicated Memory Size 256 GB", recorded SI.
    capacityOptionsBytes: [gb(256)],
    // 8192-bit × 6.0 GT/s / 8 = 6.14 TB/s, quoted "6 TB/s max. peak theoretical".
    bandwidthBytesPerSec: tbPerSec(6),
  },
  // "Last Level Cache (LLC) 256 MB".
  l2Bytes: 256 * MIB,
  gpuDirectRdma: true,
  sources: [
    AMD_PEERDIRECT_SOURCE,
    {
      url: MI325X_PAGE_URL,
      title: 'AMD Instinct MI325X product page',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec.fp32',
        'engines.0.vectorFlopsPerSec.fp64',
        'memory',
        'l2Bytes',
      ],
      note:
        '"Compute Units 304", "Peak Engine Clock 2100 MHz"; peak performance figures identical to the MI300X; ' +
        '"Dedicated Memory Size 256 GB", "HBM3E", "Memory Clock 6 GHz", "Peak Memory Bandwidth 6 TB/s", "Last Level Cache ' +
        '(LLC) 256 MB", "Typical Board Power (TBP) 1000W Peak".',
    },
    {
      url: MI325X_DS_URL,
      title: 'AMD Instinct MI325X data sheet',
      fields: ['memory'],
      note: '"6 TB/s max. peak theoretical", "Up to 6.0 GT/s".',
    },
    ...CDNA3_SHARED_SOURCES,
    ...CDNA3_MI300_SFU.sources,
  ],
  estimated: [...ctaEstimates(CDNA3_TILING_KEYS, CDNA_CTA_REASON), ...CDNA3_MI300_SFU.estimated],
};

// ---------------------------------------------------------------------------------------------------------
// CDNA4: Instinct MI355X (256 CU at 2400 MHz)
// ---------------------------------------------------------------------------------------------------------

/**
 * CDNA4 per-CU rates (CDNA4 whitepaper Table 1, MI355X column). Matrix: FP16 4096, BF16 4096, FP8 8192,
 * INT8 8192, MXFP6 / MXFP4 16384, FP32 256 (FP64 128, not listed as a peak). Vector: FP64 128, FP32 256, FP16 256.
 * Sparse (SMFMAC, 4:2 on A, f16/bf16/fp8/i8 only) = 2× dense. No TF32 hardware. Unit: FLOP (OP) / (clk · CU).
 *
 * TYPO: the whitepaper prints MXFP6 / MXFP4 as "16834"; 16384 is meant: 256 × 16384 × 2.4 GHz = 10.07 PFLOP/s
 * matches the "10 PF" in the same table and the product page's 10.1 PFLOPs (16834 would give 10.34).
 */
const CDNA4_FLOP_PER_CLK_PER_CU = {
  fp16: 4096,
  bf16: 4096,
  fp8: 8192,
  int8: 8192,
  fp6: 16384,
  fp4: 16384,
  fp32Matrix: 256,
  fp64Vector: 128,
  fp32Vector: 256,
  fp16Vector: 256,
} as const;
/** MI355X: "Compute Units 256". */
const MI355X_CUS = 256;
/** MI355X: "Peak Engine Clock 2400 MHz". Unit: Hz. */
const MI355X_CLOCK_HZ = 2.4e9;
/** MI355X SFU: 256 CU × 32 exp2/clk (CDNA4: 2× CDNA3) × 2.4 GHz. */
const MI355X_SFU = engineSfu(AMD_CDNA4_SFU, MI355X_CUS, 'CU', MI355X_CLOCK_HZ, SFU_FIELD);

/** CDNA4 tiling keys: MFMA 16×16×32 f16/bf16 (default), 16×16×64 i8, 16×16×128 f8f6f4 (fp8, fp6, fp4). */
const CDNA4_TILING_KEYS = ['default', 'fp16', 'bf16', 'fp8', 'fp6', 'fp4', 'int8'];

/** Instinct MI355X product page. */
const MI355X_PAGE_URL = 'https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html';

/**
 * The MI355X engine: 256 CU × per-CU rate (CDNA4 Table 1) × 2.4 GHz.
 * FP16 4096 → 2516.6; FP8 / INT8 8192 → 5033.2; FP6 / FP4 16384 → 10066.3; FP32 256 → 157.3 TFLOP/s; sparse 2×.
 */
function mi355xEngine(): ComputeEngine {
  const cus = MI355X_CUS;
  const clk = MI355X_CLOCK_HZ;
  const r = CDNA4_FLOP_PER_CLK_PER_CU;
  return {
    id: 'gpu',
    name: 'CDNA4 Matrix Cores (MFMA)',
    kind: 'gpu',
    computeUnits: cus,
    matmulPeaks: [
      // 256 × 4096 × 2.4 GHz = 2516.6 TFLOP/s ("2.5 PFLOPs"); sparse 5033.2 ("5 PFLOPs").
      ...cuPeaks('fp16', cus, clk, r.fp16, 2 * r.fp16),
      ...cuPeaks('bf16', cus, clk, r.bf16, 2 * r.bf16),
      // 256 × 8192 × 2.4 GHz = 5033.2 ("5 PFLOPs", OCP-FP8 and MXFP8); sparse 10066.3 ("10.1 PFLOPs").
      ...cuPeaks('fp8', cus, clk, r.fp8, 2 * r.fp8),
      ...cuPeaks('int8', cus, clk, r.int8, 2 * r.int8),
      // FP6 runs at the FP4 rate: 256 × 16384 × 2.4 GHz = 10066.3 ("10.1 PFLOPs"). No sparse FP6/FP4 (no SMFMAC variant).
      ...cuPeaks('fp6', cus, clk, r.fp6),
      ...cuPeaks('fp4', cus, clk, r.fp4),
      // 256 × 256 × 2.4 GHz = 157.3 TFLOP/s.
      ...cuPeaks('fp32', cus, clk, r.fp32Matrix),
      // No TF32 (moved to BF16 software emulation): tf32 requests fall back to fp32. No int4: falls back to int8.
      // FP64 matrix 256 × 128 × 2.4 GHz = 78.6 TFLOP/s: not listed (no fp64 accumulator key).
    ],
    vectorFlopsPerSec: {
      fp64: cuPeakFlopsPerSec(cus, r.fp64Vector, clk),
      fp32: cuPeakFlopsPerSec(cus, r.fp32Vector, clk),
      fp16: cuPeakFlopsPerSec(cus, r.fp16Vector, clk),
    },
    sfuOpsPerSec: MI355X_SFU.sfuOpsPerSec,
    gemmTiling: {
      default: tiling(16, 16, 32, CDNA_CTA_TILES),
      fp16: tiling(16, 16, 32, CDNA_CTA_TILES),
      bf16: tiling(16, 16, 32, CDNA_CTA_TILES),
      // The dedicated v_mfma_f32_16x16x32_fp8 runs at the FP16 rate; the full FP8 rate needs the f8f6f4 K=128 form.
      fp8: tiling(16, 16, 128, CDNA_CTA_TILES),
      fp6: tiling(16, 16, 128, CDNA_CTA_TILES),
      fp4: tiling(16, 16, 128, CDNA_CTA_TILES),
      int8: tiling(16, 16, 64, CDNA_CTA_TILES),
    },
  };
}

const MI355X_ENGINE = mi355xEngine();

/**
 * AMD Instinct MI355X (CDNA4, 256 CU, 288 GB HBM3E, 8 TB/s), peaks at the 2400 MHz peak engine clock
 * (liquid-cooled, 1400 W TBP).
 *
 * Traps: the "10.1 PFLOPs" FP8 and INT8 figures are 4:2-sparse (dense 5033); the 10.1 PFLOPs MXFP4 / MXFP6 figures
 * are DENSE (no sparse FP4/FP6). TF32 is not in hardware (software-emulated via BF16), so it is omitted and tf32
 * falls back to fp32 (157.3). FP64 matrix halved vs CDNA3 (78.6). Whitepaper per-CU MXFP4 "16834" is a typo
 * for 16384. L2 4 MB per XCD; `l2Bytes` = the 256 MB Infinity Cache.
 */
export const INSTINCT_MI355X: DeviceSpec = {
  id: 'instinct-mi355x',
  name: 'AMD Instinct MI355X',
  vendor: 'AMD',
  engines: [MI355X_ENGINE],
  memory: {
    kind: 'dedicated',
    technology: 'HBM3E',
    // "Dedicated Memory Size 288 GB", recorded SI.
    capacityOptionsBytes: [gb(288)],
    // 1024-bit × 8 stacks × 8 Gbps / 8 = 8.19 TB/s, quoted "8 TB/s".
    bandwidthBytesPerSec: tbPerSec(8),
  },
  // "Last Level Cache (LLC) 256 MB".
  l2Bytes: 256 * MIB,
  gpuDirectRdma: true,
  sources: [
    AMD_PEERDIRECT_SOURCE,
    {
      url: MI355X_PAGE_URL,
      title: 'AMD Instinct MI355X product page',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec',
        'memory',
        'l2Bytes',
      ],
      note:
        '"Compute Units 256", "Peak Engine Clock 2400 MHz"; MXFP4 10.1, MXFP6 10.1, MXFP8 5, OCP-FP8 5 / with Structured ' +
        'Sparsity 10.1, FP16 matrix 2.5 / 5, BF16 2.5 / 5 PFLOPs; INT8 5 / 10.1 POPs; FP16 (vector) 157.3, FP32 matrix and ' +
        'vector 157.3, FP64 matrix and vector 78.6 TFLOPs; "Dedicated Memory Size 288 GB", HBM3E, "Peak Memory Bandwidth ' +
        '8 TB/s", "Last Level Cache (LLC) 256 MB", "1400W (TBP)".',
    },
    {
      url: CDNA4_WP_URL,
      title: 'AMD CDNA 4 architecture whitepaper, Table 1 (p. 8) and Table 2 (p. 18)',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec',
        'memory',
        'l2Bytes',
      ],
      note:
        'Table 1 MI355X FLOPS/clock/CU: vector FP64 128, FP32 256, FP16 256; matrix FP64 128, FP32 256, FP16 4096, BF16 4096, ' +
        'FP8 8192, INT8 8192, MXFP6 and MXFP4 "16834" (TYPO for 16384: 256 × 16384 × 2.4 GHz = 10.07 PF = the table\'s 10 PF). ' +
        'TF32 "has been moved out of hardware and is supported through software emulation utilizing the BF16 datatype". ' +
        'Table 2: 256 CU, 2,400 MHz; 288 GB HBM3E, 1024-bits × 8 stacks, up to 8.0 TB/sec; L2 4 MB; Infinity Cache 256 MB.',
    },
    {
      url: CDNA4_ISA_URL,
      title: 'AMD Instinct CDNA4 ISA reference, §7.1.2 Table 28 (MFMA) and Table 33 (SMFMAC)',
      fields: ['engines.0.matmulPeaks', ...mmaShapePaths(CDNA4_TILING_KEYS)],
      note:
        'v_mfma_f32_16x16x32_{f16,bf16} 16 cycles; v_mfma_f32_16x16x32_fp8 16 cycles (FP16 rate); v_mfma_i32_16x16x64_i8 ' +
        '16 cycles; v_mfma_[scale_]f32_16x16x128_f8f6f4 "16 or 32" cycles ("Larger cycle count if either matrix A or B is ' +
        'F8"), i.e. FP8 at 2× and FP6/FP4 at 4× the FP16 rate. SMFMAC (4:2, "Only the A-matrix may be sparse"): ' +
        'v_smfmac_f32_16x16x64_{f16,bf16}, v_smfmac_i32_16x16x128_i8, v_smfmac_f32_16x16x128_{fp8,bf8}; no F8F6F4 variant.',
    },
    ...MI355X_SFU.sources,
  ],
  estimated: [...ctaEstimates(CDNA4_TILING_KEYS, CDNA_CTA_REASON), ...MI355X_SFU.estimated],
};

// ---------------------------------------------------------------------------------------------------------
// CDNA5: Instinct MI455X (published per-device figures only)
// ---------------------------------------------------------------------------------------------------------

/** Instinct MI455X brochure / datasheet ("AI PEAK THEORETICAL PERFORMANCE TFLOPS" table, specifications). */
const MI455X_DS_URL =
  'https://www.amd.com/content/dam/amd/en/documents/products/accelerators/instinct/amd-instinct-mi455x_brochure.pdf';
/** Instinct MI455X product page. */
const MI455X_PAGE_URL = 'https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html';

/** MI455X tiling keys; every shape is a CDNA4 proxy (no CDNA5 ISA is public). */
const MI455X_TILING_KEYS = ['default', 'fp16', 'bf16', 'fp8', 'fp6', 'fp4', 'int8'];
/** MI455X SFU: 256 units × 64 exp2/clk × 2.4 GHz (estimate, no source; the upper-bound rule). */
const MI455X_SFU = engineSfu(AMD_MI455X_SFU, 256, 'compute units', 2.4e9, SFU_FIELD);

/**
 * AMD Instinct MI455X (CDNA5, 432 GB HBM4 in 12 stacks, 23.3 TB/s), peaks as AMD's datasheet table prints them
 * (TFLOP/s, at the 2.4 GHz peak engine clock). Deliberately NOT derived from per-CU rates: the datasheet says
 * "COMPUTE UNITS 256" while the product page says "Work Group Processors 256".
 *
 * Traps: the 10,066 TFLOP/s FP16 / BF16 / INT8 figures are 4:2-sparse (dense 5,033). FP8, MXFP6, MXFP8 (20,133)
 * and MXFP4 (40,265) have NO published sparse figure, so they are dense only; note FP8 is 4× FP16 here, and
 * MXFP6 runs at the FP8 rate (not the FP4 rate as on CDNA4). FP64 is only 5 TFLOP/s (matrix and vector). No TF32
 * figure: tf32 falls back to fp32 (315). `l2Bytes` is omitted: the page lists "L2 Cache 192 MB" and no separate
 * last-level / Infinity Cache, so whether 192 MB is the last level is unclear.
 */
export const INSTINCT_MI455X: DeviceSpec = {
  id: 'instinct-mi455x',
  name: 'AMD Instinct MI455X',
  vendor: 'AMD',
  engines: [
    {
      id: 'gpu',
      name: 'CDNA5 Matrix Cores',
      kind: 'gpu',
      computeUnits: 256,
      matmulPeaks: [
        { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(5033) },
        { input: 'fp16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(10066) },
        { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(5033) },
        { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(10066) },
        // OCP FP8 and MXFP8 20,133; no sparse figure published.
        { input: 'fp8', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(20133) },
        // MXFP6 20,133 (= FP8 rate).
        { input: 'fp6', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(20133) },
        // MXFP4 40,265.
        { input: 'fp4', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(40265) },
        { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: tflops(5033) },
        { input: 'int8', accumulate: 'int32', sparsity: '2:4', flopsPerSec: tflops(10066) },
        // Matrix FP32 315. Matrix FP64 5 TFLOP/s is not listed (no fp64 accumulator key).
        { input: 'fp32', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(315) },
      ],
      vectorFlopsPerSec: { fp64: tflops(5), fp32: tflops(315), fp16: tflops(315) },
      sfuOpsPerSec: MI455X_SFU.sfuOpsPerSec,
      gemmTiling: {
        default: tiling(16, 16, 32, CDNA_CTA_TILES),
        fp16: tiling(16, 16, 32, CDNA_CTA_TILES),
        bf16: tiling(16, 16, 32, CDNA_CTA_TILES),
        fp8: tiling(16, 16, 128, CDNA_CTA_TILES),
        fp6: tiling(16, 16, 128, CDNA_CTA_TILES),
        fp4: tiling(16, 16, 128, CDNA_CTA_TILES),
        int8: tiling(16, 16, 64, CDNA_CTA_TILES),
      },
    },
  ],
  memory: {
    kind: 'dedicated',
    technology: 'HBM4',
    // "432 GB HBM4" (12 stacks), recorded SI.
    capacityOptionsBytes: [gb(432)],
    // "Peak Memory Bandwidth 23.3 TB/s".
    bandwidthBytesPerSec: tbPerSec(23.3),
  },
  gpuDirectRdma: true,
  sources: [
    AMD_PEERDIRECT_SOURCE,
    {
      url: MI455X_DS_URL,
      title:
        'AMD Instinct MI455X brochure (datasheet): AI peak theoretical performance table and specifications',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec', 'memory'],
      note:
        '"AI PEAK THEORETICAL PERFORMANCE TFLOPS" (dense, then "W/STRUCTURED SPARSITY"): OCP MXFP4 40265, OCP MXFP6 20133, ' +
        'OCP MXFP8 20133, OCP FP8 20133, MATRIX FP16 5033 / 10066, VECTOR FP16 315, MATRIX FP32 315, VECTOR FP32 315, ' +
        'MATRIX FP64 5, VECTOR FP64 5, MATRIX INT8 5033 / 10066, MATRIX BF16 5033 / 10066 (sparse column only for FP16, ' +
        'INT8, BF16). "COMPUTE UNITS 256", 8 XCD, 2 IOD, "PEAK ENGINE CLOCK 2.4 GHz", "432 GB HBM4", "23.3 TB/s".',
    },
    {
      url: MI455X_PAGE_URL,
      title: 'AMD Instinct MI455X product page',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec', 'memory'],
      note:
        '"Work Group Processors 256", "Peak Engine Clock 2400 MHz"; MXFP4 40.3, MXFP6 / MXFP8 / OCP FP8 20.1 PFLOPs; ' +
        'Matrix FP16 5 / sparse 10.1 PFLOPs; BF16 5 / 10.1; INT8 5 / 10.1 POPs; vector FP16, matrix and vector FP32 ' +
        '315 TFLOPs; FP64 5 TFLOPs; "Dedicated Memory Size 432 GB", "HBM4", "Stacks of Memory 12", "Peak Memory ' +
        'Bandwidth 23.3 TB/s", "L2 Cache 192 MB".',
    },
    ...MI455X_SFU.sources,
  ],
  estimated: [
    {
      field: 'engines.0.computeUnits',
      reason:
        'The datasheet says "COMPUTE UNITS 256", the product page "Work Group Processors 256" (a WGP would be 2 CUs, ' +
        'i.e. 512). 256 is used; the field sets the wave granularity of the shape-efficiency model, and only the SFU ' +
        'estimate (engines.0.sfuOpsPerSec, per unit) is derived from it.',
    },
    ...MI455X_TILING_KEYS.flatMap((k) =>
      ['mmaM', 'mmaN', 'mmaK'].map((f) => ({
        field: `engines.0.gemmTiling.${k}.${f}`,
        reason:
          'No CDNA5 ISA is public: CDNA4 MFMA shapes used as a proxy (16x16x32 f16/bf16, 16x16x64 i8, 16x16x128 ' +
          `f8f6f4; ${CDNA4_ISA_URL} Table 28).`,
      })),
    ),
    ...ctaEstimates(MI455X_TILING_KEYS, CDNA_CTA_REASON),
    ...MI455X_SFU.estimated,
  ],
};

/**
 * All AMD device presets, per-device peaks at each part's vendor boost / peak engine clock. Display order: client
 * parts (Radeon AI PRO R9700, RX 9070 XT, Strix Halo), then Instinct newest generation first.
 */
export const AMD_DEVICE_PRESETS: DeviceSpec[] = [
  RADEON_AI_PRO_R9700,
  RADEON_RX_9070_XT,
  RYZEN_AI_MAX_PLUS_395,
  INSTINCT_MI455X,
  INSTINCT_MI355X,
  INSTINCT_MI325X,
  INSTINCT_MI300X,
];
