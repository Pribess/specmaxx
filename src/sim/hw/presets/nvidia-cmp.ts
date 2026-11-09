/**
 * NVIDIA CMP 170HX (8 GB SKU, PCI 10de:20c2), the GA100 "cryptocurrency mining processor", and its community unlock
 * axes. NVIDIA publishes NO datasheet for the 170HX (its CMP page lists only the 30HX / 40HX / 50HX / 90HX), so every
 * number here rests on reviews (`kind: 'review'`) and community measurements and driver work (`kind: 'community'`),
 * and the peaks are DERIVED from the A100's published per-SM rates (vendor) × 70 SMs × 1410 MHz, divided by the
 * fused 1/32 issue-rate throttle where it applies. Each derivation is in `estimated[]`.
 * See docs/simulation-model.md §4 ("Preset library", "Unlock axes").
 */

import type { MathFormat } from '../../ir/dtype';
import type { DeviceModifier, ModifierProvenance } from '../modifiers';
import type { DeviceSpec, GemmTiling, MatmulPeak, SourceRef } from '../types';
import type { UnlockAxis } from '../unlocks';
import { gb } from '../units';
import { NVIDIA_SM_SFU, engineSfu } from './sfu';

// ---------------------------------------------------------------------------------------------------------------
// Sources

/** niconiconi: CMP 170HX review (deviceQuery, clpeak, lspci link status, FMA throttle). */
const NICONICONI_URL = 'https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/';
/** Consensus-Protocol cmp170hx wiki: complete specification, stock and unlocked, both SKUs. */
const CONSENSUS_OVERVIEW_URL =
  'https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/hardware/overview.md';
/** Consensus-Protocol cmp170hx wiki: the issue-rate fuses and the cmpunlocker override. */
const CONSENSUS_THROTTLE_URL =
  'https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/unlock/compute-throttle.md';
/** Consensus-Protocol cmp170hx wiki: physical mods (PCIe AC-coupling capacitors on lanes 4–15). */
const CONSENSUS_PHYSICAL_URL =
  'https://github.com/Consensus-Protocol/cmp170hx/blob/main/docs/hardware/physical-mods.md';
/** arabel1a/ml-on-cmp: instruction and mma.sync microbenchmarks, the DP2A patch, llama.cpp numbers. */
const ARABEL1A_URL = 'https://github.com/arabel1a/ml-on-cmp';
/** amoghmunikote/cmpunlocker: patched open kernel modules (compute, memory geometry, PCIe Gen2). */
const CMPUNLOCKER_URL = 'https://github.com/amoghmunikote/cmpunlocker';
/** PixelML club-170hx: vBIOS 67 vs 6D memory clocks, Gen2 x16 measurements, single- vs dual-rank capacity. */
const PIXELML_URL =
  'https://github.com/PixelML/club-170hx/blob/main/docs/CMP170HX-UNDERSTANDING.md';
/** polyuij42-del: flashing the signed 92.00.6D.00.0A vBIOS on 8 GB cards. */
const VBIOS_6D_FLASH_URL = 'https://github.com/polyuij42-del/cmp170hx-vbios-flash';
/** cachenetics/170tune: HBM PLL overclock reference matrices. */
const TUNE_170_URL = 'https://github.com/cachenetics/170tune/blob/main/docs/reference-matrices.md';
/** wanghar: 6D vBIOS in memory + HBM 1890 MHz. */
const WANGHAR_URL = 'https://github.com/wanghar/cmp170hx-6d-VBIOS-upgrade';
/** NVIDIA A100 Tensor Core GPU architecture whitepaper (Table 1 peak rates, 108 SMs at 1410 MHz). */
const A100_WHITEPAPER_URL =
  'https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf';
/** PTX ISA, warp-level matrix (mma.sync) shapes. */
const PTX_MMA_SHAPES_URL =
  'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma';

// ---------------------------------------------------------------------------------------------------------------
// Silicon and rates

/** SMs enabled on every 170HX (both SKUs): 5 GPCs / 35 TPCs / 70 SMs (Consensus overview). Count. */
const CMP_170HX_SMS = 70;
/** Boost / sustained SM clock at stock: 1410 MHz (Consensus overview "Sustained SM clock (stock) 1410 MHz"). Unit: Hz. */
const CMP_170HX_CLOCK_HZ = 1410e6;
/**
 * The fused issue-rate divider of the 170HX: FUSE_SS_FFMA, FMLA16, FMLA32 and IMLA0–4 all read 0x5 = divide by 32
 * (FUSE_SS_DP, a one-bit field, reads its maximum 1). Dimensionless. Applies to FP32 FMA, FP64 and every tensor-core
 * (mma.sync) instruction; NOT to FADD / FMUL, HFMA2 (half2), INT32, DP2A or the MUFU (SFU) pipe.
 */
const CMP_THROTTLE_DIVISOR = 32;

/**
 * Per-SM, per-clock rates of an unthrottled GA100 SM: the A100's Table 1 peaks ÷ (108 SMs × 1.41 GHz), rounded to the
 * power of two they are (e.g. 312e12 / (108 × 1.41e9) = 2049 → 2048). Unit: FLOP (OP for integers) per clock per SM,
 * dense (a 2:4 sparse peak is 2×).
 */
const GA100_PER_SM_PER_CLOCK = {
  /** FP16 / BF16 tensor, FP32 accumulate: 1024 FMA = 2048 FLOP (A100 312 TFLOP/s). */
  fp16Tensor: 2048,
  /** TF32 tensor: 1024 FLOP (A100 156 TFLOP/s). */
  tf32Tensor: 1024,
  /** INT8 tensor: 4096 OP (A100 624 TOP/s). */
  int8Tensor: 4096,
  /** INT4 tensor: 8192 OP (A100 1248 TOP/s). */
  int4Tensor: 8192,
  /** FP32 FMA: 64 lanes × 2 FLOP = 128 (A100 19.5 TFLOP/s). */
  fp32Fma: 128,
  /** FP32 FADD or FMUL alone: 64 lanes × 1 FLOP = 64 (half the FMA rate; never throttled on the 170HX). */
  fp32AddMul: 64,
  /** FP64 FMA: 64 FLOP (A100 9.7 TFLOP/s). */
  fp64Fma: 64,
  /** FP16 half2 HFMA2 on the CUDA cores: 512 FLOP (A100 non-tensor FP16 78 TFLOP/s). */
  fp16Hfma2: 512,
  /** BF16 bf16x2 FMA on the CUDA cores: 256 FLOP (A100 non-tensor BF16 39 TFLOP/s). */
  bf16Fma2: 256,
  /** DP2A: 64 lanes × (2 multiplies + 2 adds) = 256 OP, at the FADD issue rate (arabel1a timings). */
  int8Dp2a: 256,
} as const;

/** Device rate of a per-SM per-clock rate on the stock 170HX: rate = SMs × perSmPerClock × clock. Unit: FLOP/s (OP/s). */
function cmpRate(perSmPerClock: number): number {
  // Formula: R = 70 SM × r_SM × 1.41e9 Hz
  return CMP_170HX_SMS * perSmPerClock * CMP_170HX_CLOCK_HZ;
}

/** A throttled tensor peak at stock, dense and 2:4: full GA100 rate ÷ 32 (sparse = 2 × dense, dense-equivalent). */
function throttledPair(
  input: MathFormat,
  accumulate: MatmulPeak['accumulate'],
  perSmPerClock: number,
): MatmulPeak[] {
  // Formula: P_stock = 70 · r_SM · 1.41e9 / 32;  P_2:4 = 2 · P_stock
  const dense = cmpRate(perSmPerClock) / CMP_THROTTLE_DIVISOR;
  return [
    { input, accumulate, sparsity: 'dense', flopsPerSec: dense },
    { input, accumulate, sparsity: '2:4', flopsPerSec: 2 * dense },
  ];
}

/** sm_80 `mma.sync` tiling: m16 n8 k`mmaK` (PTX ISA), typical CUTLASS CTA tiles, one CTA per SM. Elements. */
function sm80Tiling(mmaK: number): GemmTiling {
  return {
    mmaM: 16,
    mmaN: 8,
    mmaK,
    ctaTiles: [
      { m: 128, n: 256 },
      { m: 128, n: 128 },
      { m: 128, n: 64 },
      { m: 64, n: 128 },
      { m: 64, n: 64 },
    ],
    ctasPerUnit: 1,
  };
}

/** Tiling keys of the engine (for the provenance paths). */
const CMP_TILING_KEYS: ('default' | MathFormat)[] = [
  'default',
  'fp16',
  'bf16',
  'tf32',
  'int8',
  'int4',
];

/**
 * SFU (MUFU exp2): 70 SMs × 16 exp2/clk × 1.41 GHz = 1.579e12 exp2/s. Not throttled: arabel1a times f32_ex2 at
 * 2.135 ns/op, the same ≈ 1.5× ratio to a 2080S as the unthrottled ops (throttled ones are 14–20×).
 */
const CMP_170HX_SFU = engineSfu(
  NVIDIA_SM_SFU,
  CMP_170HX_SMS,
  'SM',
  CMP_170HX_CLOCK_HZ,
  'engines.0.sfuOpsPerSec',
);

/** The A100 whitepaper citation of the per-SM rates (vendor), for the derived peaks. */
const A100_PER_SM_SOURCE: SourceRef = {
  url: A100_WHITEPAPER_URL,
  title:
    'NVIDIA A100 Tensor Core GPU Architecture (whitepaper), Table 1 and the A100 SM description',
  fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec'],
  note:
    'A100 (108 SMs, 1410 MHz boost): Peak FP64 9.7, FP32 19.5, FP16 (non-tensor) 78, BF16 (non-tensor) 39, TF32 ' +
    'Tensor 156/312, FP16 and BF16 Tensor 312/624, INT8 Tensor 624/1248, INT4 Tensor 1248/2496 (dense/sparse) ' +
    'TFLOPS/TOPS. Per SM per clock: ÷ (108 × 1.41e9) = 64 / 128 / 512 / 256 / 1024 / 2048 / 4096 / 8192. The 170HX ' +
    'is the same GA100 SM (sm_80), so its rates are these × 70 SMs × 1.41e9, ÷ 32 where fused (see estimated[]).',
};

// ---------------------------------------------------------------------------------------------------------------
// The device

/**
 * NVIDIA CMP 170HX, 8 GB SKU (GA100-105F-A1, sm_80, 70 SMs / 4480 FP32 lanes / 280 tensor cores, 8 GB HBM2e 4096-bit,
 * 32 MB L2, 250 W), AS SHIPPED (vBIOS 92.00.67.00.01, stock driver), at the stock 1410 MHz SM clock.
 *
 * Stock throttles (each a separate mechanism):
 * - Issue-rate fuses ÷32 on FP32 FMA, FP64 and every tensor-core instruction: FP32 FMA 12.63 → 0.395 TFLOP/s
 *   (niconiconi measured 0.39), FP16/BF16 tensor 202.1 → 6.32 (measured 6.40–6.52), TF32 3.16 (measured 2.96–3.21).
 *   INT8 tensor: 404.3 / 32 = 12.63 TOP/s. Consensus lists 1.60 TOP/s measured, but arabel1a's mma.sync timing pins
 *   EVERY shape to one issue rate (181.7 ns per instruction; m16n8k32 s8 does 2× the FLOP of m16n8k16 f16), which
 *   gives exactly 12.63, and the stock llama.cpp Q4_0 pp512 of 599.6 tok/s (MMQ on IMMA) needs ≈ 8.1 TOP/s, above
 *   1.60. So the higher, consistent value is taken (the simulation is an upper bound).
 *   Tensor FP16 with FP16 accumulate (FMLA16) and with FP32 accumulate (FMLA32) are both fused: 6.32 TFLOP/s each.
 *   Not throttled: FADD / FMUL (6.32 TFLOP/s when the compiler emits no FMA), HFMA2 half2 (50.5 theoretical,
 *   measured 41.9–47.4), INT32, DP2A, the SFU.
 * - Memory: NOT throttled. 1458 MHz × 2 × 4096 bit / 8 = 1492.99 GB/s (deviceQuery; clpeak 1355). Capacity is a
 *   strap-selected 8 GB geometry.
 * - PCIe: Gen1 (2.5 GT/s, fuse FUSE_PCIE_GEN23_DIS) × 4 lanes (lanes 4–15 lack their AC-coupling capacitors;
 *   LnkCap x16, LnkSta x4): ≈ 0.85 GB/s measured, 0.914 GB/s payload ceiling. NVLink fused off; no P2P.
 *
 * Unlock stages (`CMP_170HX_UNLOCKS`, all community work, numbers community-measured), listed in stage order
 * (firmware, driver, board), which is also the order their modifiers apply:
 * 1. vBIOS 6D (checkbox): NVIDIA's signed 92.00.6D.00.0A "OC" ROM: HBM 1728 MHz = 1769 GB/s, 300 W, SM sustained
 *    1470 MHz measured. Independent of the driver patch.
 * 2. Compute path (radio; each option is a complete, valid stage, so no option depends on another axis):
 *    `locked-workarounds` (stock driver, kernels built with `-fmad=false`, half2 and the DP2A patch: arabel1a,
 *    llama.cpp #24616); `cmpunlocker` (Linux driver patch on nvidia-open 610.43: full GA100 rates, memory geometry
 *    8 → 64 GB on dual-rank cards / 32 GB on single-rank, PCIe Gen2; re-applied at every driver load, vBIOS
 *    untouched); `cmpunlocker-hbm-1890` (cmpunlocker plus its PLM patch and the 170tune / wanghar HBM PLL overclock:
 *    1935 GB/s; the overclock needs the patched driver, so it exists only as this combined option, with cmpunlocker's
 *    modifiers applied once). Its absolute 1935 GB/s applies after vBIOS 6D's 1769, so 6D + overclock = 1935.
 *    `cmpunlocker-74sm` and `cmpunlocker-74sm-hbm-1890`: the same two plus cmpunlocker PR #58
 *    (https://github.com/amoghmunikote/cmpunlocker/pull/58), which re-enables 4 fused-off SMs (70 → 74; a
 *    `computeUnits` modifier, every compute rate × 74/70; reported FP32 12.25 → 12.97, BF16 175 → 183 TFLOPS, not a
 *    clean A/B). The SM patch is part of the patched driver, so it too exists only inside cmpunlocker options; each
 *    option applies cmpunlocker's modifiers and `computeUnits` exactly once.
 * 3. PCIe capacitor mod (checkbox): 24 × 0402 220 nF caps on lanes 4–15 → x16 (Gen1 x16 measured 2.88 GB/s, with
 *    cmpunlocker's Gen2 6.63–6.74 GB/s; Gen2 and x16 combine through two `hostInterface` modifiers).
 *
 * Not modelled (no option):
 * - PCIe Gen3: only on the 10 GB SKU (10de:2082) after a permanent RIR fuse burn
 *   (https://github.com/evanrodgers/cmp-170hx-rir-burn-util, https://github.com/lixiangwuxian/170hx_tips); on 8 GB
 *   cards OPT_DISABLE_GEN3_SPEED is set and forcing 8 GT/s drops the card off the bus (PixelML).
 * - PCIe x8 (12 of the 24 capacitors): a documented half mod; only x16 is offered.
 * - P2P: static-BAR1 patches give 5.30–5.79 GB/s one way on Gen2 x16 (PixelML); the topology links stay host bounce.
 * - The 10 GB SKU (5120-bit HBM2 at 1215 MHz = 1555 GB/s, 40 GB after unlock): a different board, not a preset.
 * - Power-cap tuning (100–250 W), VF offsets, the PWRBRK# (B30) tape fix: no measured tensor-clock effect at the
 *   stock cap (PixelML: large GEMMs within ±3 % across caps).
 *
 * Matmul-peak choice for FP16 / BF16 inference (docs/simulation-model.md §4 "Peak lookup and fallback"): GEMMs ask
 * for (fp16 | bf16, fp32 accumulate) by default, (fp16, fp16) when the user picks fp16 accumulation. The table lists
 * BOTH fp16 accumulators explicitly (GA100 runs them at one rate, so stock and unlocked they are equal), because the
 * locked-card half2 path accumulates in half: `locked-workarounds` raises only the (fp16, fp16) entries (dense ×8 to
 * the HFMA2 rate). An fp32-accumulating fp16 GEMM keeps 6.32 TFLOP/s: the throttled tensor rate, and also exactly
 * what an FP32-accumulating CUDA-core kernel reaches with separate FMUL + FADD (64 lanes × 1 FLOP × 70 × 1.41 GHz).
 * GGUF prefill (llama.cpp MMQ) uses the int8 entry, which the DP2A path raises ×2.
 */
export const CMP_170HX: DeviceSpec = {
  id: 'cmp-170hx',
  name: 'NVIDIA CMP 170HX 8GB',
  vendor: 'NVIDIA',
  engines: [
    {
      id: 'gpu',
      name: 'GA100 3rd-gen Tensor Cores (sm_80, issue-throttled)',
      kind: 'gpu',
      computeUnits: CMP_170HX_SMS,
      matmulPeaks: [
        ...throttledPair('fp16', 'fp16', GA100_PER_SM_PER_CLOCK.fp16Tensor), // 6.317 / 12.63 TFLOP/s (FMLA16)
        ...throttledPair('fp16', 'fp32', GA100_PER_SM_PER_CLOCK.fp16Tensor), // 6.317 / 12.63 (FMLA32)
        ...throttledPair('bf16', 'fp32', GA100_PER_SM_PER_CLOCK.fp16Tensor), // 6.317 / 12.63
        ...throttledPair('tf32', 'fp32', GA100_PER_SM_PER_CLOCK.tf32Tensor), // 3.158 / 6.317
        ...throttledPair('int8', 'int32', GA100_PER_SM_PER_CLOCK.int8Tensor), // 12.63 / 25.27 TOP/s
        ...throttledPair('int4', 'int32', GA100_PER_SM_PER_CLOCK.int4Tensor), // 25.27 / 50.53
      ],
      vectorFlopsPerSec: {
        // FP32 FMA throttled ÷32: 70 × 128 × 1.41e9 / 32 = 0.3948e12.
        fp32: cmpRate(GA100_PER_SM_PER_CLOCK.fp32Fma) / CMP_THROTTLE_DIVISOR,
        // FP64 throttled (FUSE_SS_DP): 70 × 64 × 1.41e9 / 32 = 0.1974e12 (measured 0.18–0.20).
        fp64: cmpRate(GA100_PER_SM_PER_CLOCK.fp64Fma) / CMP_THROTTLE_DIVISOR,
        // HFMA2 not throttled: 70 × 512 × 1.41e9 = 50.53e12.
        fp16: cmpRate(GA100_PER_SM_PER_CLOCK.fp16Hfma2),
        // bf16x2 FMA, assumed not throttled like HFMA2: 70 × 256 × 1.41e9 = 25.27e12.
        bf16: cmpRate(GA100_PER_SM_PER_CLOCK.bf16Fma2),
      },
      sfuOpsPerSec: CMP_170HX_SFU.sfuOpsPerSec,
      gemmTiling: {
        default: sm80Tiling(16),
        fp16: sm80Tiling(16),
        bf16: sm80Tiling(16),
        tf32: sm80Tiling(8),
        int8: sm80Tiling(32),
        int4: sm80Tiling(64),
      },
    },
  ],
  memory: {
    kind: 'dedicated',
    technology: 'HBM2e',
    // 8192 MiB as reported; recorded SI 8 GB like the other presets (conservative).
    capacityOptionsBytes: [gb(8)],
    // 1458e6 × 2 × 4096 / 8 = 1492.99e9 B/s (deviceQuery "1492.99 GB/sec").
    bandwidthBytesPerSec: (1458e6 * 2 * 4096) / 8,
  },
  // 32768 KB (binary KB) per deviceQuery and torch; TechPowerUp's 8 MB is wrong. Recorded only.
  l2Bytes: 32768 * 1024,
  // As shipped: Gen1 x4 (LnkSta "Speed 2.5GT/s, Width x4 (downgraded)").
  hostInterface: { generation: 1, lanes: 4 },
  gpuDirectRdma: false,
  sources: [
    {
      url: CONSENSUS_OVERVIEW_URL,
      title: 'Consensus-Protocol cmp170hx wiki: hardware overview (complete specification)',
      kind: 'community',
      fields: [
        'engines.0.computeUnits',
        'l2Bytes',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec',
        'memory.capacityOptionsBytes',
      ],
      note:
        '"70 SMs / 4480 CUDA cores at compute capability 8.0 on both SKUs"; "Sustained SM clock (stock) 1410 MHz"; ' +
        'L2 "32 MB (32768 KB)"; 8 GB SKU "8192 MiB", 4096-bit. Locked measurements: FP32 FFMA 0.30–0.41, FP32 non-FMA ' +
        '4.31–6.29 ("Never throttled"), FP64 ~0.20, TF32 tensor 2.96–3.21, FP16 tensor 6.52, BF16 tensor 6.40, FP16 ' +
        'scalar ~42–50 ("Never throttled"), INT8 tensor MMA 1.60, INT4 11.55 TFLOPS/TOPS.',
    },
    {
      url: CONSENSUS_THROTTLE_URL,
      title: 'Consensus-Protocol cmp170hx wiki: the compute throttle and how it is removed',
      kind: 'community',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec'],
      note:
        'Nine OPT_SM_SPEED_SELECT fuses: FUSE_SS_FFMA, FMLA16, FMLA32, IMLA0–4 read 0x5 ("the maximum divisor (divide by ' +
        '32)"), FUSE_SS_DP reads 1; 0 on every A100. "FP32 goes from 0.395 TFLOPS to roughly 12.7 to 12.9 TFLOPS (about ' +
        '32x), FP64 and every tensor path come back with it".',
    },
    {
      url: ARABEL1A_URL,
      title: 'arabel1a/ml-on-cmp: instruction and tensor-core microbenchmarks on the CMP 170HX',
      kind: 'community',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec'],
      note:
        'mma.sync m16n8k16 f16/bf16, m16n8k8 tf32 and m16n8k32 s8/u8 all 181.7 ns per instruction ("One fixed rate for ' +
        'every precision"): a cap on the issue rate, so INT8 = 2 × FP16 FLOP/s. f32_fma 5.681 ns/op (20× a 2080S), ' +
        'f32_add / f32_mul 0.401, f16x2_fma 0.744, dp4a 5.681, dp2a_lo/hi 0.401, prmt 0.400, f32_ex2 2.135.',
    },
    A100_PER_SM_SOURCE,
    {
      url: NICONICONI_URL,
      title: 'niconiconi: All GB/s without FLOPS - Nvidia CMP 170HX review',
      kind: 'review',
      fields: ['memory', 'hostInterface', 'engines.0.vectorFlopsPerSec.fp32'],
      note:
        'deviceQuery "Memory bandwidth: 1492.99 GB/sec" (729 MHz × 2 DDR × 4096 bit), clpeak float8 1355.40 GB/s, 8 GiB ' +
        'HBM2e; FP32 FMA "0.39 TFLOPS"; "LnkCap: Port #0, Speed 2.5GT/s, Width x16", "LnkSta: Speed 2.5GT/s, Width x4 ' +
        '(downgraded)", enqueueWriteBuffer 0.85 GB/s; "omitting the AC coupling capacitors on some PCIe lanes".',
    },
    {
      url: PTX_MMA_SHAPES_URL,
      title: 'PTX ISA mma.sync shapes for sm_80',
      fields: CMP_TILING_KEYS.flatMap((k) => [
        `engines.0.gemmTiling.${k}.mmaM`,
        `engines.0.gemmTiling.${k}.mmaN`,
        `engines.0.gemmTiling.${k}.mmaK`,
      ]),
      note: 'f16/bf16 m16n8k16; tf32 m16n8k8; s8 m16n8k32; s4 m16n8k64. Dense instruction shapes.',
    },
    ...CMP_170HX_SFU.sources,
  ],
  estimated: [
    {
      field: 'engines.0.matmulPeaks',
      reason:
        'No vendor figure. Derived: A100 per-SM rate × 70 SMs × 1.41e9 Hz ÷ 32 (the fused issue-rate divider): FP16 / ' +
        'BF16 2048 → 6.317, TF32 1024 → 3.158 TFLOP/s; INT8 4096 → 12.63, INT4 8192 → 25.27 TOP/s; 2:4 = 2 × dense ' +
        '(mma.sp unmeasured). Measured FP16 6.52, BF16 6.40, TF32 2.96–3.21 agree. INT8: the derived 12.63 is kept over ' +
        'the 1.60 TOP/s Consensus measurement (likely a library path: unlocked, the same split is 335 raw MMA vs 44 TOP/s ' +
        'library) because arabel1a times s8 MMA at the f16 instruction rate and stock llama.cpp MMQ prefill already ' +
        'needs ≈ 8.1 TOP/s.',
    },
    {
      field: 'engines.0.vectorFlopsPerSec',
      reason:
        'No vendor figure. Derived at 70 SMs × 1.41e9 Hz: FP32 FMA 128 FLOP/clk/SM ÷ 32 = 0.3948 TFLOP/s (measured ' +
        '0.39); FP64 64 ÷ 32 = 0.1974 (measured 0.18–0.20); FP16 HFMA2 512, not throttled = 50.53 (measured 41.9–47.4); ' +
        'BF16 256 = 25.27, ASSUMED not throttled like HFMA2 (no bf16x2 measurement found).',
    },
    {
      field: 'gpuDirectRdma',
      reason:
        'ESTIMATED false: a mining SKU, not one of the Tesla / Quadro lines NVIDIA lists for GPUDirect RDMA, and no ' +
        'community source shows NIC RDMA into its memory (the BAR1 P2P patches are GPU-to-GPU only).',
    },
    ...CMP_TILING_KEYS.flatMap((k) => [
      {
        field: `engines.0.gemmTiling.${k}.ctaTiles`,
        reason:
          'Representative CUTLASS sm_80 CTA tiles (https://github.com/NVIDIA/cutlass), not a hardware specification.',
      },
      {
        field: `engines.0.gemmTiling.${k}.ctasPerUnit`,
        reason:
          'One resident large CTA per SM, register/shared-memory constrained; kernel-dependent estimate.',
      },
    ]),
    ...CMP_170HX_SFU.estimated,
  ],
};

/** The NVIDIA CMP device presets (one: the 8 GB CMP 170HX). */
export const NVIDIA_CMP_DEVICE_PRESETS: DeviceSpec[] = [CMP_170HX];

// ---------------------------------------------------------------------------------------------------------------
// Unlock axes

/** cmpunlocker README: what the driver patch does and which features work. */
const CMPUNLOCKER_SOURCE: Omit<SourceRef, 'fields'> = {
  url: CMPUNLOCKER_URL,
  title: 'amoghmunikote/cmpunlocker: patched NVIDIA open kernel modules for the CMP 170HX',
  kind: 'community',
  note:
    'Feature table: compute unlock, "Memory geometry (64GB on 8GB cards, 40GB on 10GB cards) | Working", "PCIe Gen 2 ' +
    'speeds | Working", "Full BAR1 Size (64GB) | Working"; `install.sh --profile=8gb` ("8GB card → 64GB unlock"). ' +
    'Writes FEATURE_OVERRIDE_SM_SPEED_SELECT at every driver load; never touches the vBIOS.',
};

/** Consensus overview: the measured unlocked rates (community). */
const UNLOCKED_RATES_SOURCE: SourceRef = {
  url: CONSENSUS_OVERVIEW_URL,
  title: 'Consensus-Protocol cmp170hx wiki: compute throughput, measured (locked and unlocked)',
  kind: 'community',
  fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec'],
  note:
    'Unlocked at stock clocks: FP32 12.2–12.8 (theoretical 12.63 = 4480 × 2 × 1410 MHz), FP64 6.2–6.3, TF32 tensor ' +
    '79–94, FP16 tensor 158.7–190, BF16 tensor 164.4–192.7 ("Ceiling is 2048 × 70 × 1410 MHz = 202.1 TFLOPS"), INT8 ' +
    'tensor MMA 335.0–335.6, INT4 320.2 TFLOPS/TOPS. Every full-rate value is ≤ the derived ceiling except FP32 12.8, ' +
    'within 1.3 % (one card sustained 1425 MHz).',
};

/** The cmpunlocker modifiers shared by both cmpunlocker options (compute ×32, 64 GB, Gen2). */
function cmpunlockerModifiers(): DeviceModifier[] {
  const compute: ModifierProvenance = {
    note: 'cmpunlocker: issue-rate override, full GA100 rate (×32 on the fused paths)',
    sources: [
      { ...CMPUNLOCKER_SOURCE, fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec'] },
      UNLOCKED_RATES_SOURCE,
    ],
  };
  return [
    // Every tensor peak: P_full = P_stock · 32 (6.317 → 202.1 TFLOP/s FP16/BF16; 12.63 → 404.3 TOP/s INT8).
    { kind: 'peakScale', engineId: 'gpu', factor: CMP_THROTTLE_DIVISOR, ...compute },
    // FP32 FMA 0.3948 → 12.63 TFLOP/s; FP64 0.1974 → 6.317. FP16 / BF16 vector were never throttled.
    {
      kind: 'vectorScale',
      engineId: 'gpu',
      dtype: 'fp32',
      factor: CMP_THROTTLE_DIVISOR,
      ...compute,
    },
    {
      kind: 'vectorScale',
      engineId: 'gpu',
      dtype: 'fp64',
      factor: CMP_THROTTLE_DIVISOR,
      ...compute,
    },
    {
      kind: 'memoryCapacity',
      // Dual-rank cards reach 65536 MiB (recorded SI 64 GB); single-rank cards stop at 32 GB.
      capacityOptionsBytes: [gb(64), gb(32)],
      note: 'cmpunlocker memory geometry: 8 → 64 GB on dual-rank cards (32 GB on single-rank); bus width and bandwidth unchanged',
      sources: [
        { ...CMPUNLOCKER_SOURCE, fields: ['memory.capacityOptionsBytes'] },
        {
          url: PIXELML_URL,
          title: 'PixelML club-170hx: understanding the CMP 170HX',
          kind: 'community',
          fields: ['memory.capacityOptionsBytes'],
          note:
            '"8 GB stock; 64 GB after the memory unlock on dual-rank cards (single-rank cards stop at 32 GB)". The ' +
            'Consensus overview\'s same-card A/B shows bandwidth 1592 → 1599 GB/s ("Effect of the unlock on bandwidth: None").',
        },
      ],
    },
    {
      kind: 'hostInterface',
      generation: 2,
      note: 'cmpunlocker PCIe link patch: Gen1 → Gen2 (5 GT/s); width unchanged',
      sources: [
        { ...CMPUNLOCKER_SOURCE, fields: ['hostInterface.generation'] },
        {
          url: PIXELML_URL,
          title: 'PixelML club-170hx: PCIe speed with the cmpunlocker link patch',
          kind: 'community',
          fields: ['hostInterface.generation'],
          note:
            '"PCIe | Gen1 stock; Gen2 with the cmpunlocker link patch; Gen3 blocked by a fuse on every card we read"; ' +
            '"Gen2 x16 measured 6.68 GB/s device-to-host on one card" (~84% of the 8 GB/s line rate; payload ceiling ' +
            '7.31 GB/s).',
        },
      ],
    },
  ];
}

/**
 * The HBM PLL overclock to 1890 MHz (170tune / wanghar). Needs cmpunlocker plus its PLM-opening patch, so it is used
 * only inside the combined `cmpunlocker-hbm-1890` option. Absolute bandwidth: applied after the vBIOS 6D axis, so 6D +
 * overclock = 1935 GB/s.
 */
const HBM_1890_MODIFIER: DeviceModifier = {
  kind: 'memoryBandwidth',
  // HBM PLL NDIV 70 × 27 MHz = 1890 MHz: 1890e6 × 2 × 4096 / 8 = 1935.36e9 B/s.
  bandwidthBytesPerSec: (1890e6 * 2 * 4096) / 8,
  note: 'HBM PLL overclock NDIV 70 (1890 MHz): 1935 GB/s theoretical (stream read measured 1825–1859 GB/s)',
  sources: [
    {
      url: TUNE_170_URL,
      title: 'cachenetics/170tune: HBM overclock reference matrices',
      kind: 'community',
      fields: ['memory.bandwidthBytesPerSec'],
      note:
        'HBM clock = NDIV × 27 MHz; NDIV 70 = 1890 MHz, measured stream read 1825–1859 GB/s, the serving-safe ' +
        'ceiling (HBM ≤ 76 °C); NDIV 72 corrupts when hot. Needs cmpunlocker plus the PLM-opening patch ' +
        '(cmpunlocker PR #60).',
    },
    {
      url: WANGHAR_URL,
      title: 'wanghar/cmp170hx-6d-VBIOS-upgrade: 6D in memory + HBM 1890 MHz',
      kind: 'community',
      fields: ['memory.bandwidthBytesPerSec'],
      note: 'HBM 1890 MHz on 8 GB cards.',
    },
  ],
};

/**
 * cmpunlocker PR #58: re-enables 4 fused-off SMs (70 → 74), scaling every compute rate by 74/70 and setting the wave
 * granularity to 74 SMs. Part of the patched driver, so used only inside cmpunlocker options.
 */
const SM_74_MODIFIER: DeviceModifier = {
  kind: 'computeUnits',
  engineId: 'gpu',
  computeUnits: 74,
  note: 'cmpunlocker PR #58: re-enables 4 fused-off SMs, 70 → 74 (every compute rate × 74/70)',
  sources: [
    {
      url: 'https://github.com/amoghmunikote/cmpunlocker/pull/58',
      title: 'cmpunlocker PR #58: +4 SMs (70 → 74)',
      kind: 'community',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec',
        'engines.0.sfuOpsPerSec',
      ],
      note:
        'Reported FP32 12.25 → 12.97 TFLOPS, BF16 175 → 183 TFLOPS (not a clean A/B); 74/70 = +5.7 %. PixelML runs ' +
        'it as commit 6c442ee "+4 SMs (70 → 74)".',
    },
  ],
};
/**
 * Compute path: mutually exclusive, each option a complete stage (no option needs another axis): the locked card
 * with workaround kernels, or cmpunlocker alone or with what it enables (the HBM overclock, the +4 SM patch, both).
 */
const COMPUTE_PATH_AXIS: UnlockAxis = {
  id: 'compute-path',
  name: 'Compute throttle',
  kind: 'tensorRate',
  description:
    'The card ships with FP32 FMA, FP64 and every tensor-core instruction fused to 1/32 issue rate. ' +
    'Locked-card kernels avoid the fused paths (FADD + FMUL, half2, DP2A); cmpunlocker, a Linux driver patch, ' +
    'removes the throttle, unlocks 64 GB and PCIe Gen2, and allows the HBM overclock. Numbers are community-measured.',
  options: [
    {
      id: 'locked-workarounds',
      name: 'Locked card: no-FMA, half2 and DP2A kernels',
      deviceModifiers: [
        {
          kind: 'vectorScale',
          engineId: 'gpu',
          dtype: 'fp32',
          // FP32 as separate FADD + FMUL: 64 FLOP/clk/SM vs the throttled FMA 128 / 32 = 4 → ×16 (0.3948 → 6.317 TFLOP/s).
          factor:
            GA100_PER_SM_PER_CLOCK.fp32AddMul /
            (GA100_PER_SM_PER_CLOCK.fp32Fma / CMP_THROTTLE_DIVISOR),
          note: 'Locked card, kernels built with -fmad=false: FP32 as unthrottled FADD + FMUL, 6.32 TFLOP/s (measured 6.29)',
          sources: [
            {
              url: ARABEL1A_URL,
              title: 'arabel1a/ml-on-cmp: FMA throttled, FADD / FMUL not',
              kind: 'community',
              fields: ['engines.0.vectorFlopsPerSec.fp32'],
              note:
                '"Fused Multiply Accumulate (FMA) is thottled 8x, but can be replaced with FADD+FMUL on CMP170HX"; ' +
                'f32_add / f32_mul 0.401 ns/op vs f32_fma 5.681. Consensus: "FP32 non-FMA (control) 4.31 - 6.29 ' +
                'TFLOPS". 70 × 64 × 1.41e9 = 6.317 TFLOP/s.',
            },
          ],
        },
        {
          kind: 'peakScale',
          engineId: 'gpu',
          input: 'fp16',
          accumulate: 'fp16',
          sparsity: 'dense',
          // HFMA2 512 FLOP/clk/SM vs the throttled tensor 2048 / 32 = 64 → ×8 (6.317 → 50.53 TFLOP/s).
          factor:
            GA100_PER_SM_PER_CLOCK.fp16Hfma2 /
            (GA100_PER_SM_PER_CLOCK.fp16Tensor / CMP_THROTTLE_DIVISOR),
          note: 'Locked card: FP16-accumulate GEMMs on the CUDA cores as half2 HFMA2 (not throttled), 50.5 TFLOP/s',
          sources: [
            {
              url: CONSENSUS_OVERVIEW_URL,
              title: 'Consensus-Protocol cmp170hx wiki: FP16 scalar never throttled',
              kind: 'community',
              fields: ['engines.0.matmulPeaks'],
              note:
                '"FP16 scalar (non-tensor) | ~42 - 50 TFLOPS | unchanged | Never throttled, even on a locked card. ' +
                'GA100 runs 16-bit hfma at 4x its FP32 rate". 70 × 512 × 1.41e9 = 50.53 TFLOP/s. HFMA2 accumulates ' +
                'in half, so only the fp16-accumulate entry rises; fp32-accumulate stays 6.32 (the throttled tensor ' +
                'rate, equal to FMUL + FADD on the CUDA cores: 70 × 64 × 1.41e9).',
            },
          ],
        },
        {
          kind: 'peakScale',
          engineId: 'gpu',
          input: 'fp16',
          accumulate: 'fp16',
          sparsity: '2:4',
          // A 2:4 GEMM runs dense on HFMA2: 50.53 dense-equivalent = 12.63 × 4.
          factor:
            GA100_PER_SM_PER_CLOCK.fp16Hfma2 /
            ((2 * GA100_PER_SM_PER_CLOCK.fp16Tensor) / CMP_THROTTLE_DIVISOR),
          note: 'Locked card: a 2:4 FP16-accumulate GEMM runs dense on HFMA2 (no sparse half2 path), 50.5 TFLOP/s',
          sources: [],
          estimatedReason:
            'The dense HFMA2 rate (50.53 TFLOP/s) applied to sparse FP16 GEMMs; no sparse half2 instruction exists.',
        },
        {
          kind: 'peakScale',
          engineId: 'gpu',
          input: 'int8',
          sparsity: 'dense',
          // DP2A 256 OP/clk/SM vs the throttled IMMA 4096 / 32 = 128 → ×2 (12.63 → 25.27 TOP/s).
          factor:
            GA100_PER_SM_PER_CLOCK.int8Dp2a /
            (GA100_PER_SM_PER_CLOCK.int8Tensor / CMP_THROTTLE_DIVISOR),
          note: 'Locked card: INT8 dot products as unthrottled DP2A (2 × PRMT + 2 × DP2A per DP4A), 25.3 TOP/s ceiling',
          sources: [
            {
              url: ARABEL1A_URL,
              title: 'arabel1a/ml-on-cmp: the DP2A patch (llama.cpp MMVQ)',
              kind: 'community',
              fields: ['engines.0.matmulPeaks'],
              note:
                '"DP2A is unthrottled (0.31 ns)", dp2a 0.401 ns/op on the 170HX = f32_add, i.e. 64 lanes/clk/SM; ' +
                'DP4A 5.681 (20× throttled). Llama-2-7B Q4_0 tg128 61.7 → 143.6 with dp2a + --fmad=false. See also ' +
                'https://github.com/ggml-org/llama.cpp/issues/24616.',
            },
          ],
          estimatedReason:
            'Derived: 70 SMs × 64 DP2A/clk × 4 OP × 1.41e9 = 25.27 TOP/s, counting the DP2A issue rate only (the PRMT ' +
            'widening runs on another pipe and is not charged: an upper bound). The 2:4 int8 entry (25.27) is unchanged.',
        },
      ],
    },
    {
      id: 'cmpunlocker',
      name: 'cmpunlocker: full compute, 64 GB, PCIe Gen2',
      deviceModifiers: cmpunlockerModifiers(),
    },
    {
      id: 'cmpunlocker-hbm-1890',
      name: 'cmpunlocker + HBM overclock 1890 MHz',
      deviceModifiers: [...cmpunlockerModifiers(), HBM_1890_MODIFIER],
    },
    {
      id: 'cmpunlocker-74sm',
      name: 'cmpunlocker + 4 SMs (74 SMs)',
      deviceModifiers: [...cmpunlockerModifiers(), SM_74_MODIFIER],
    },
    {
      id: 'cmpunlocker-74sm-hbm-1890',
      name: 'cmpunlocker + 4 SMs + HBM overclock 1890 MHz',
      deviceModifiers: [...cmpunlockerModifiers(), SM_74_MODIFIER, HBM_1890_MODIFIER],
    },
  ],
};

/** PCIe capacitor mod: a board mod, independent of firmware and driver. */
const PCIE_CAPACITOR_AXIS: UnlockAxis = {
  id: 'pcie-capacitor-mod',
  name: 'PCIe capacitor mod',
  kind: 'interconnect',
  description:
    'Soldering the 24 missing AC-coupling capacitors (0402, 220 nF) on lanes 4–15 trains the link at x16. ' +
    'A hardware mod, independent of the driver: Gen1 x16 measured 2.88 GB/s, Gen2 x16 6.63–6.74 GB/s (community).',
  options: [
    {
      id: 'x16',
      name: 'x16 (24 capacitors)',
      deviceModifiers: [
        {
          kind: 'hostInterface',
          lanes: 16,
          note: 'PCIe capacitor mod: lanes 4–15 populated, x4 → x16 (generation unchanged)',
          sources: [
            {
              url: CONSENSUS_PHYSICAL_URL,
              title:
                'Consensus-Protocol cmp170hx wiki: physical mods (PCIe AC-coupling capacitors)',
              kind: 'community',
              fields: ['hostInterface.lanes'],
              note:
                '24 × 0402 220 nF X7R ≥ 6.3 V capacitors on lanes 4–15 → x16 (12 of 24 → x8). Measured Gen1 x16 ' +
                '2.88 GB/s, Gen2 x16 6.63–6.74 GB/s. The overview: "12 of the 16 lanes ship with their AC-coupling ' +
                'capacitors depopulated. The lane fuses are clean".',
            },
            {
              url: NICONICONI_URL,
              title: 'niconiconi: CMP 170HX review, PCIe lockdown',
              kind: 'review',
              fields: ['hostInterface.lanes'],
              note: '"upgradable to x16 by hardware modding, but still at PCIe 1.0\'s 4 GB/s"; LnkCap Width x16.',
            },
          ],
        },
      ],
    },
  ],
};

/** vBIOS 6D: firmware, independent of the driver patch. Applied first (stage order), so the HBM overclock overrides it. */
const VBIOS_6D_AXIS: UnlockAxis = {
  id: 'vbios-6d',
  name: 'vBIOS 6D',
  kind: 'memoryClock',
  description:
    'vBIOS 6D is NVIDIA\'s signed "OC" ROM for the 8 GB card: HBM 1728 MHz and a 300 W limit; flashing it is reversible. ' +
    'Bandwidth is the theoretical bus peak; the measured gain (+21 %) is a community report.',
  options: [
    {
      id: '6d',
      name: 'HBM 1728 MHz, 300 W',
      deviceModifiers: [
        {
          kind: 'memoryBandwidth',
          // 1728e6 × 2 × 4096 / 8 = 1769.47e9 B/s (stock 1492.99).
          bandwidthBytesPerSec: (1728e6 * 2 * 4096) / 8,
          note: 'vBIOS 92.00.6D.00.0A: HBM 1458 → 1728 MHz, 1769 GB/s theoretical (measured copy 1.32 → 1.59–1.60 TB/s, +21 %)',
          sources: [
            {
              url: PIXELML_URL,
              title: 'PixelML club-170hx: vBIOS 67 vs 6D HBM clocks',
              kind: 'community',
              fields: ['memory.bandwidthBytesPerSec'],
              note:
                '"HBM2e clock | VBIOS-dependent: 1,458 MHz (NDIV 54) on VBIOS 92.00.67.00.01, 1,728 MHz (NDIV 64) on ' +
                '92.00.6D.00.0A, same board part number, identical timings | measured". 250 W / 300 W variants.',
            },
            {
              url: VBIOS_6D_FLASH_URL,
              title: 'polyuij42-del/cmp170hx-vbios-flash: 6D vBIOS on 8 GB cards',
              kind: 'community',
              fields: ['memory.bandwidthBytesPerSec'],
              note: '8 GB SKU only (nvflash 5.867); 300 W cap; the 64 GB unlock is unaffected.',
            },
          ],
        },
        {
          kind: 'clockScale',
          engineId: 'gpu',
          // Sustained SM clock 1470 MHz at -pl 300 vs the 1410 MHz the base peaks assume: ×1.0426.
          factor: 1470 / 1410,
          note: 'vBIOS 6D at a 300 W limit: sustained SM 1410 → 1470 MHz (one card), compute ×1470/1410',
          sources: [
            {
              url: CONSENSUS_OVERVIEW_URL,
              title: 'Consensus-Protocol cmp170hx wiki: clocks',
              kind: 'community',
              fields: [
                'engines.0.matmulPeaks',
                'engines.0.vectorFlopsPerSec',
                'engines.0.sfuOpsPerSec',
              ],
              note:
                '"Sustained SM clock at -pl 300 | 1470 MHz | 8 GB card with the OC VBIOS"; "VBIOS table max graphics ' +
                'clock 1695 MHz". PixelML: large GEMMs within ±3 % at the same cap, 300 W +6 % bf16 over 250 W.',
            },
          ],
        },
      ],
    },
  ],
};

/**
 * Independent CMP 170HX community unlock axes, in stage order (firmware, driver, board), which is the order their
 * modifiers apply. Every combination validates and is physically possible (presets.test.ts, nvidia-cmp.test.ts).
 * Numbers are community measurements or derived from them; see {@link CMP_170HX} for the stages and what is not
 * modelled.
 */
export const CMP_170HX_UNLOCKS: UnlockAxis[] = [
  VBIOS_6D_AXIS,
  COMPUTE_PATH_AXIS,
  PCIE_CAPACITOR_AXIS,
];
