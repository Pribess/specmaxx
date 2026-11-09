/**
 * NVIDIA datacenter and DGX device presets: DGX Spark (GB10), H100 SXM, H200 SXM, B200 (HGX), B300 (HGX).
 * Every number is cited in `sources[]` (with the table / quote it comes from) or listed in `estimated[]` with
 * the derivation. See docs/simulation-model.md §4 ("Hardware").
 *
 * Conventions (as in hw/types.ts): per device (one GPU, not one 8-GPU board); SI decimal units; 2:4 entries are
 * dense-equivalent FLOP/s. Datacenter parts run fp16/fp8 matmuls at the same rate with fp16 or fp32 accumulate,
 * so only `fp32`-accumulate entries are listed (the peak lookup maps an fp16-accumulate request onto them).
 *
 * FOOTNOTE TRAPS: NVIDIA's headline tensor numbers are 2:4 SPARSE ("* With sparsity" on the H100/H200 pages,
 * "Dense is ½ sparse spec shown" on the HGX page and the Blackwell Ultra datasheet). Dense entries here are the
 * dense figure where NVIDIA prints one, else ½ of the sparse headline.
 *
 * Not included (yet):
 * - H100 PCIe: only preliminary 2022 tensor figures were found (Hopper architecture blog, "800/1600" FP16); the
 *   shipping datasheet value is not sourced, so the preset is left out.
 * - Rubin (HGX Rubin NVL8, 224 SMs, 288 GB HBM4, 22 TB/s): NVIDIA marks its specs preliminary. Future addition;
 *   sources https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/
 *   and https://www.nvidia.com/en-us/data-center/hgx/ (Rubin NVL8 tab).
 *
 * Interconnects (NVLink 4/5, ConnectX-7/8, InfiniBand) are links, not device fields: see `links.ts`.
 */

import type { MathFormat } from '../../ir/dtype';
import type { ComputeEngine, DeviceSpec, EstimatedField, GemmTiling, MatmulPeak, SourceRef } from '../types';
import { gb, gbPerSec, mb, tbPerSec, tflops } from '../units';
import { NVIDIA_DC_GPUDIRECT_SOURCE } from './gpudirect';

// ---------------------------------------------------------------------------------------------------------------
// Shared sources and helpers
// ---------------------------------------------------------------------------------------------------------------

/** PTX ISA (9.5), warp-level `mma.sync` shapes (sm_80+, the only MMA path on GB10 / sm_121). */
const PTX_MMA_SYNC_URL = 'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape';
/** PTX ISA (9.5), warpgroup `wgmma.mma_async` shapes (sm_90a: H100 / H200). */
const PTX_WGMMA_URL =
  'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape';
/** PTX ISA (9.5), `tcgen05.mma` shapes (sm_100a / sm_103a: B200 / B300), Table 48. */
const PTX_TCGEN05_URL = 'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape';
/** CUTLASS repository: the source of the "typical CTA tile" estimates. */
const CUTLASS_URL = 'https://github.com/NVIDIA/cutlass';

/**
 * Dotted paths `engines.0.matmulPeaks.<i>` of the peaks that `match` selects, so a source can cite exactly the
 * entries it states (e.g. only the sparse FP4 headline) without hand-written indices.
 */
function peakPaths(peaks: MatmulPeak[], match: (p: MatmulPeak) => boolean): string[] {
  return peaks.flatMap((p, i) => (match(p) ? [`engines.0.matmulPeaks.${i}`] : []));
}

/**
 * The tiling of an engine: one {@link GemmTiling} per listed key, all sharing the MMA M/N granularity, CTA tiles
 * and CTAs per SM, differing only in the MMA K (which depends on the input format's width).
 */
function tilingTable(
  mmaM: number,
  mmaN: number,
  ctaTiles: { m: number; n: number }[],
  ctasPerUnit: number,
  mmaKByKey: Partial<Record<'default' | MathFormat, number>> & { default: number },
): { default: GemmTiling } & Partial<Record<MathFormat, GemmTiling>> {
  const make = (mmaK: number): GemmTiling => ({ mmaM, mmaN, mmaK, ctaTiles: ctaTiles.map((t) => ({ ...t })), ctasPerUnit });
  const table: { default: GemmTiling } & Partial<Record<MathFormat, GemmTiling>> = { default: make(mmaKByKey.default) };
  for (const [key, k] of Object.entries(mmaKByKey)) {
    if (key !== 'default' && k !== undefined) table[key as MathFormat] = make(k);
  }
  return table;
}

/** The ISA source for the MMA shape fields (mmaM/N/K) of every listed tiling key. */
function tilingShapeSource(url: string, title: string, keys: string[], note: string): SourceRef {
  return {
    url,
    title,
    fields: keys.flatMap((k) => [`engines.0.gemmTiling.${k}.mmaM`, `engines.0.gemmTiling.${k}.mmaN`, `engines.0.gemmTiling.${k}.mmaK`]),
    note,
  };
}

/** `estimated[]` entries for the CTA tiles and CTAs per SM of every listed tiling key (no vendor publishes them). */
function tilingEstimates(keys: string[], tilesReason: string): EstimatedField[] {
  return keys.flatMap((k) => [
    { field: `engines.0.gemmTiling.${k}.ctaTiles`, reason: `${tilesReason} (${CUTLASS_URL}); not a vendor spec.` },
    {
      field: `engines.0.gemmTiling.${k}.ctasPerUnit`,
      reason: 'One resident CTA per SM for large tiles (register / shared-memory / TMEM bound); not a vendor spec.',
    },
  ]);
}

// ---------------------------------------------------------------------------------------------------------------
// DGX Spark (GB10)
// ---------------------------------------------------------------------------------------------------------------

/** NVIDIA GB10 Superchip White Paper WP-12718-001_v02 (April 2026). */
const GB10_WHITEPAPER_URL = 'https://www.nvidia.com/content/dam/en-zz/Solutions/dgx-spark/nvidia-gb10-superchip-whitepaper.pdf';
/** DGX Spark product page (spec table, footnotes). */
const DGX_SPARK_PRODUCT_URL = 'https://www.nvidia.com/en-us/products/workstations/dgx-spark/';
/** NVIDIA RTX Blackwell GPU Architecture whitepaper (Appendix C: GB205 / RTX 5070, the same SM generation and SM count). */
const RTX_BLACKWELL_WHITEPAPER_URL =
  'https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf';
/** NVIDIA developer forum: mmapeak measurements on DGX Spark. */
const GB10_MMAPEAK_FORUM_URL = 'https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993';

/**
 * GB10 dense tensor rates, estimated (NVIDIA publishes only the sparse FP4 headline):
 * - FP4 dense = 1 PFLOP sparse (product-page footnote "Theoretical FP4 TOPS using the sparsity feature") / 2.
 * - FP8 = FP4 / 2, FP16 / BF16 = FP8 / 2, TF32 = FP16 / 2: the per-SM ratios of the RTX Blackwell SM (RTX 5070,
 *   also 48 SMs: FP4 493.9, FP8 246.9, FP16 123.5 dense TFLOP/s) and John Carmack's public reading
 *   ("1 PF sparse FP4 = 125 TF dense BF16").
 * The 1 PF headline implies ≈ 2.54 GHz (500e12 / (48 SM × 4096 FLOP/clk/SM)), above the 2.4 GHz boost clock of
 * the whitepaper; these rates follow the vendor headline, not 2.4 GHz (which would give 471.9 TFLOP/s FP4).
 */
const GB10_FP4_DENSE_TFLOPS = 1000 / 2; // 1 PFLOP FP4 2:4 sparse → 500 TFLOP/s dense
const GB10_FP8_DENSE_TFLOPS = GB10_FP4_DENSE_TFLOPS / 2; // 250
const GB10_FP16_DENSE_TFLOPS = GB10_FP8_DENSE_TFLOPS / 2; // 125 (FP16 and BF16)
const GB10_TF32_DENSE_TFLOPS = GB10_FP16_DENSE_TFLOPS / 2; // 62.5

/**
 * GB10 vector FP32: 6,144 CUDA cores × 2 FLOP/clk (FMA) × 2.4 GHz boost = 29.49 TFLOP/s (whitepaper Table 4 core
 * count and clock; ServeTheHome's Hot Chips report quotes "31 TFLOPS", i.e. ≈ 2.52 GHz).
 */
const GB10_VECTOR_FP32_FLOPS = 6144 * 2 * 2.4e9;

/** GB10 matmul peaks (all fp32-accumulate; the fp32-acc = fp16-acc rate rests on a community measurement). */
const GB10_PEAKS: MatmulPeak[] = [
  { input: 'fp4', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(GB10_FP4_DENSE_TFLOPS) },
  { input: 'fp4', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(1000) }, // the "1 PFLOP FP4" headline
  { input: 'fp8', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(GB10_FP8_DENSE_TFLOPS) },
  { input: 'fp8', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(2 * GB10_FP8_DENSE_TFLOPS) },
  { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(GB10_FP16_DENSE_TFLOPS) },
  { input: 'fp16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(2 * GB10_FP16_DENSE_TFLOPS) },
  { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(GB10_FP16_DENSE_TFLOPS) },
  { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(2 * GB10_FP16_DENSE_TFLOPS) },
  { input: 'tf32', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(GB10_TF32_DENSE_TFLOPS) },
  { input: 'tf32', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(2 * GB10_TF32_DENSE_TFLOPS) },
  // INT8 = FP8 rate per SM on the RTX Blackwell SM (RTX 5070: INT8 246.9 TOPS = FP8 246.9 TFLOPS).
  { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: tflops(GB10_FP8_DENSE_TFLOPS) },
  { input: 'int8', accumulate: 'int32', sparsity: '2:4', flopsPerSec: tflops(2 * GB10_FP8_DENSE_TFLOPS) },
];

/** Tiling keys of the GB10 engine. */
const GB10_TILING_KEYS = ['default', 'fp16', 'bf16', 'tf32', 'fp8', 'int8', 'fp4'];

/**
 * NVIDIA DGX Spark: GB10 Grace Blackwell superchip, GPU = 48 consumer-Blackwell SMs (compute capability 12.1),
 * 128 GB unified LPDDR5X at 273 GB/s shared with the 20-core Grace CPU.
 *
 * Clocks: tensor peaks follow the vendor's "1 PFLOP FP4 with sparsity" headline (≈ 2.54 GHz implied); vector
 * peaks use the 2.4 GHz boost clock of the whitepaper. NVIDIA's own staff report the GPU draws at most ~120 W
 * (140 W SoC TDP), and users see about half the quoted rate in practice; this preset is the theoretical ceiling.
 *
 * Traps: the "1 PFLOP" headline is FP4 with 2:4 SPARSITY; dense FP4 is 500 TFLOP/s. All dense rates are
 * ESTIMATED (NVIDIA publishes no dense number for GB10). Unlike GeForce, fp32 accumulate runs at the fp16
 * accumulate rate (community mmapeak measurement), so only fp32-accumulate entries are listed. No wgmma or
 * tcgen05 on sm_121: GEMMs use warp-level `mma.sync`.
 */
export const DGX_SPARK: DeviceSpec = {
  id: 'dgx-spark',
  name: 'NVIDIA DGX Spark (GB10)',
  vendor: 'NVIDIA',
  engines: [
    {
      id: 'gpu',
      name: 'GB10 Blackwell GPU, 5th-gen Tensor Cores',
      kind: 'gpu',
      computeUnits: 48,
      matmulPeaks: GB10_PEAKS,
      vectorFlopsPerSec: {
        fp32: GB10_VECTOR_FP32_FLOPS,
        // Non-tensor FP16/BF16 at the FP32 rate (1:1), as on the RTX Blackwell SM.
        fp16: GB10_VECTOR_FP32_FLOPS,
        bf16: GB10_VECTOR_FP32_FLOPS,
        // FP64 = 1/64 of FP32 (Chips and Cheese): 29.49 / 64 = 0.461 TFLOP/s.
        fp64: GB10_VECTOR_FP32_FLOPS / 64,
      },
      gemmTiling: tilingTable(
        16,
        8,
        [
          { m: 128, n: 256 },
          { m: 128, n: 128 },
          { m: 128, n: 64 },
          { m: 64, n: 128 },
          { m: 64, n: 64 },
        ],
        1,
        // mma.sync K: f16/bf16 k16, tf32 k8, e4m3/e5m2 and s8 k32, block-scaled e2m1 (kind::mxf4nvf4) k64.
        { default: 16, fp16: 16, bf16: 16, tf32: 8, fp8: 32, int8: 32, fp4: 64 },
      ),
    },
  ],
  memory: {
    kind: 'unified',
    technology: 'LPDDR5X',
    // 128 GB (Founders Edition) and 64 GB (OEM-only SKU), as quoted (SI). Shared with the CPU and OS: the
    // whitepaper says the GPU can allocate "more than 100 GiB"; deviceQuery reports 119.7 GiB total.
    capacityOptionsBytes: [gb(128), gb(64)],
    // 256-bit × 8533 MT/s / 8 = 273.06 GB/s; recorded as NVIDIA quotes it, 273 GB/s (CPU + GPU total).
    bandwidthBytesPerSec: gbPerSec(273),
    // GPU-usable share: CUDA reports "Total global memory: 119.7 GiB" of the 128 GiB of LPDDR5X (the rest is the
    // kernel and firmware). Formula: f = 119.7 GiB / 128 GiB = 0.935; the 64 GB SKU is assumed to keep the same share.
    gpuUsableFraction: 119.7 / 128,
  },
  // "24 MiB" (binary) = 24 × 1024 × 1024 B. Recorded only; not used by the v1 model.
  l2Bytes: 24 * 1024 * 1024,
  sources: [
    {
      url: GB10_WHITEPAPER_URL,
      title: 'NVIDIA GB10 Superchip White Paper (WP-12718-001_v02), Table 4 "GB10 GPU Specifications" and Table 2',
      fields: ['engines.0.computeUnits', 'l2Bytes'],
      note:
        'Table 4: "Streaming multiprocessors 48", "NVIDIA CUDA cores 6,144 (128 per SM)", "GPU boost clock 2.4 GHz", ' +
        '"Global (L2) cache 24 MiB". Table 2: "AI TOPS performance 1 PFLOP NVFP4 Sparse". Table 1: "Memory ' +
        'interface speed 8,533 Mbps". Text: "allowing the GPU to allocate more than 100 GiB".',
    },
    {
      url: DGX_SPARK_PRODUCT_URL,
      title: 'NVIDIA DGX Spark product page, specifications',
      fields: [
        ...peakPaths(GB10_PEAKS, (p) => p.input === 'fp4' && p.sparsity === '2:4'),
        'memory.capacityOptionsBytes',
        'memory.bandwidthBytesPerSec',
      ],
      note:
        '"Tensor Performance¹ Up to 1 PFLOP FP4" with footnote "1 Theoretical FP4 TOPS using the sparsity feature."; ' +
        '"64 GB LPDDR5X* or 128 GB LPDDR5x, coherent unified system memory" ("*64 GB memory configuration is ' +
        'available exclusively through participating OEM partners"); "Memory Bandwidth 273 GB/s"; "GB10 TDP 140 W".',
    },
    {
      url: GB10_MMAPEAK_FORUM_URL,
      title: 'NVIDIA developer forum: "Detailed compute performance metrics for DGX Spark" (device query)',
      fields: ['memory.gpuUsableFraction'],
      kind: 'community',
      note:
        'Device query on DGX OS: "Device 0: NVIDIA GB10", "Total global memory: 119.7 GiB" of 128 GiB installed ' +
        '(119.7 / 128 = 0.935). The whitepaper only says the GPU can allocate "more than 100 GiB".',
    },
    {
      url: GB10_MMAPEAK_FORUM_URL,
      title: 'NVIDIA developer forum: "Detailed compute performance metrics for DGX Spark" (mmapeak)',
      fields: peakPaths(GB10_PEAKS, (p) => p.accumulate === 'fp32' && p.sparsity === 'dense'),
      kind: 'community',
      note:
        'Supports the RATIO fp32-accumulate = fp16-accumulate, not the absolute values: mma_f16f16f16_16_16_16 213.0 vs ' +
        'mma_f16f16f32_16_16_16 212.9, mma_bf16bf16f32 212.9, mma_f8f8f16_16_8_32 213.8 vs mma_f8f8f32 213.7 T(fl)ops. ' +
        'CAVEAT: the absolute numbers do not fit the per-SM derivation (FP16 213 > the 125 estimate while FP8 equals ' +
        'FP16, unlike RTX 5080 where FP8 = 2 × FP16); old mmapeak runs on all-zero inputs. Also measured: tf32 53.3, ' +
        's8 215.1, nvf4 427.3.',
    },
    {
      url: RTX_BLACKWELL_WHITEPAPER_URL,
      title: 'NVIDIA RTX Blackwell GPU Architecture (whitepaper), Appendix C: GB205 (RTX 5070, 48 SMs, 2512 MHz)',
      fields: ['engines.0.vectorFlopsPerSec.fp16', 'engines.0.vectorFlopsPerSec.bf16'],
      note:
        'Same SM generation as GB10. Dense/sparse TFLOPS: FP4 with FP32 acc 493.9/987.8; FP8 with FP16 acc ' +
        '246.9/493.9; FP16 with FP16 acc 123.5/246.9; INT8 TOPS 246.9/493.9; TF32 30.9/61.7 (GeForce half rate); ' +
        '"Peak FP16 TFLOPS (non-Tensor)" = FP32 = 30.9 → vector FP16 at the FP32 rate.',
    },
    {
      url: 'https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu',
      title: 'Chips and Cheese: Analyzing NVIDIA GB10\'s GPU',
      fields: ['engines.0.vectorFlopsPerSec.fp64'],
      kind: 'review',
      note: '"GB10 has a 1:64 FP64 to FP32 ratio" → 29.49 / 64 = 0.461 TFLOP/s.',
    },
    tilingShapeSource(
      PTX_MMA_SYNC_URL,
      'PTX ISA 9.5: warp-level matrix shapes (mma.sync; sm_121 has no wgmma / tcgen05)',
      GB10_TILING_KEYS,
      'f16/bf16 m16n8k16; tf32 m16n8k8; e4m3/e5m2 and s8 m16n8k32; block-scaled e2m1 (kind::mxf4nvf4, sm_120a/f) ' +
        'm16n8k64 (peak FP4 needs this form). wgmma "Requires sm_90a"; tcgen05 lists sm_100a/sm_100f/sm_110f only.',
    ),
  ],
  estimated: [
    ...peakPaths(GB10_PEAKS, (p) => !(p.input === 'fp4' && p.sparsity === '2:4')).map((field) => ({
      field,
      reason:
        'NVIDIA publishes no dense rate for GB10. FP4 dense = 1 PFLOP sparse / 2 = 500 TFLOP/s; FP8 = 250, FP16/BF16 ' +
        '= 125, TF32 = 62.5 (each ½ the previous: RTX Blackwell per-SM ratios, RTX 5070 whitepaper table, and ' +
        'Carmack\'s "1 PF sparse FP4 = 125 TF dense BF16"); INT8 = FP8 rate (RTX 5070: INT8 TOPS = FP8 TFLOPS); ' +
        'fp32 accumulate at the fp16-accumulate rate (forum mmapeak ratio); 2:4 = 2 × dense. TF32 at ½ FP16 is the ' +
        'RTX PRO / datacenter ratio (forum: TF32 tensor ≈ 2 × FP32 vector), not GeForce\'s ¼.',
    })),
    {
      field: 'engines.0.vectorFlopsPerSec.fp32',
      reason:
        'Derived: 6,144 CUDA cores × 2 FLOP/clk (FMA) × 2.4 GHz boost (whitepaper Table 4) = 29.49 TFLOP/s. ' +
        'ServeTheHome (Hot Chips 2025) quotes "31 TFLOPS of FP32" (≈ 2.52 GHz).',
    },
    {
      field: 'engines.0.vectorFlopsPerSec.fp16',
      reason: 'Vector FP16 = FP32 rate (RTX Blackwell SM, whitepaper App. C); GB10 publishes none.',
    },
    {
      field: 'engines.0.vectorFlopsPerSec.bf16',
      reason: 'Vector BF16 = FP32 rate (RTX Blackwell SM, whitepaper App. C); GB10 publishes none.',
    },
    ...tilingEstimates(GB10_TILING_KEYS, 'Typical CUTLASS CTA tile shapes for sm_120 mma.sync GEMMs'),
  ],
};

// ---------------------------------------------------------------------------------------------------------------
// Hopper: H100 SXM, H200 SXM
// ---------------------------------------------------------------------------------------------------------------

/** H100 product page (spec table, footnote "* With sparsity"). */
const H100_PRODUCT_URL = 'https://www.nvidia.com/en-us/data-center/h100/';
/** H200 product page (spec table, footnote "² With sparsity."). */
const H200_PRODUCT_URL = 'https://www.nvidia.com/en-us/data-center/h200/';
/** "NVIDIA Hopper Architecture In-Depth" (developer blog, 2022). */
const HOPPER_BLOG_URL = 'https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/';

/**
 * Hopper GH100 (H100 SXM5 / H200 SXM) matmul peaks. The product pages list SPARSE numbers only; dense = ½ of
 * each (the Hopper blog's table pairs every dense/sparse figure 1:2):
 * FP16/BF16 1,979 → 989.5; TF32 989 → 494.5; FP8 3,958 → 1,979; INT8 3,958 TOPS → 1,979. The tensor peak implies
 * ≈ 1.83 GHz (989.5e12 / (132 SM × 4096 FLOP/clk/SM)).
 */
const HOPPER_SXM_PEAKS: MatmulPeak[] = [
  { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(1979 / 2) },
  { input: 'fp16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(1979) },
  { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(1979 / 2) },
  { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(1979) },
  { input: 'tf32', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(989 / 2) },
  { input: 'tf32', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(989) },
  { input: 'fp8', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(3958 / 2) },
  { input: 'fp8', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(3958) },
  { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: tflops(3958 / 2) },
  { input: 'int8', accumulate: 'int32', sparsity: '2:4', flopsPerSec: tflops(3958) },
];

/** Tiling keys of the Hopper engines. */
const HOPPER_TILING_KEYS = ['default', 'fp16', 'bf16', 'tf32', 'fp8', 'int8'];

/** Hopper engine (shared by H100 SXM and H200 SXM: same GH100 configuration and the same table numbers). */
function hopperSxmEngine(): ComputeEngine {
  return {
    id: 'gpu',
    name: 'GH100 4th-gen Tensor Cores',
    kind: 'gpu',
    computeUnits: 132,
    matmulPeaks: HOPPER_SXM_PEAKS.map((p) => ({ ...p })),
    // FP32 67 TFLOP/s and FP64 34 TFLOP/s (non-tensor); the FP64 Tensor Core 67 TFLOP/s is not modeled.
    vectorFlopsPerSec: { fp32: tflops(67), fp64: tflops(34) },
    gemmTiling: tilingTable(
      64,
      8,
      [
        { m: 128, n: 256 },
        { m: 256, n: 128 },
        { m: 128, n: 128 },
        { m: 128, n: 64 },
        { m: 64, n: 128 },
      ],
      1,
      // wgmma K: f16/bf16 k16, tf32 k8, e4m3/e5m2 and s8 k32.
      { default: 16, fp16: 16, bf16: 16, tf32: 8, fp8: 32, int8: 32 },
    ),
  };
}

/** The wgmma tiling source (H100 / H200). */
const HOPPER_TILING_SOURCE = tilingShapeSource(
  PTX_WGMMA_URL,
  'PTX ISA 9.5: asynchronous warpgroup-level matrix shapes (wgmma.mma_async, sm_90a)',
  HOPPER_TILING_KEYS,
  'f16/bf16 m64nNk16; tf32 m64nNk8; e4m3/e5m2 m64nNk32; s8/u8 m64nNk32; N = 8..256 in steps of 8 → mmaM 64, ' +
    'mmaN 8. Sparse variants double K.',
);

/**
 * NVIDIA H100 SXM5 80 GB (Hopper GH100, 132 SMs, HBM3 3.35 TB/s, up to 700 W).
 *
 * Clock: the tensor peaks imply ≈ 1.83 GHz; the FP32 67 TFLOP/s peak a higher ≈ 1.98 GHz (16,896 cores × 2).
 * TRAP: the product page's tensor numbers (1,979 FP16 / 3,958 FP8) are 2:4 SPARSE ("* With sparsity"); dense is
 * half (989.5 / 1,979).
 */
export const H100_SXM: DeviceSpec = {
  id: 'h100-sxm',
  name: 'NVIDIA H100 SXM 80GB',
  vendor: 'NVIDIA',
  engines: [hopperSxmEngine()],
  memory: {
    kind: 'dedicated',
    technology: 'HBM3',
    // "80GB" as quoted (SI).
    capacityOptionsBytes: [gb(80)],
    bandwidthBytesPerSec: tbPerSec(3.35),
  },
  // "50 MB L2 cache" (blog), recorded as quoted (SI). Not used by the v1 model.
  l2Bytes: mb(50),
  gpuDirectRdma: true,
  sources: [
    NVIDIA_DC_GPUDIRECT_SOURCE,
    {
      url: H100_PRODUCT_URL,
      title: 'NVIDIA H100 product page, Product Specifications (H100 SXM column)',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec', 'memory'],
      note:
        '"TF32 Tensor Core* 989 teraFLOPS", "BFLOAT16 Tensor Core* 1,979", "FP16 Tensor Core* 1,979", "FP8 Tensor ' +
        'Core* 3,958", "INT8 Tensor Core* 3,958 TOPS", footnote "* With sparsity" → dense = ½ (989.5 / 494.5 / ' +
        '1,979). "FP64 34 teraFLOPS", "FP64 Tensor Core 67", "FP32 67", "GPU Memory 80GB", "GPU Memory Bandwidth ' +
        '3.35TB/s", "Up to 700W", "NVLink: 900GB/s, PCIe Gen5: 128GB/s".',
    },
    {
      url: HOPPER_BLOG_URL,
      title: 'NVIDIA Hopper Architecture In-Depth (developer blog)',
      fields: ['engines.0.computeUnits', 'l2Bytes', 'engines.0.matmulPeaks'],
      note:
        '"The H100 SXM5 GPU has 132 SMs, and the PCIe version has 114 SMs"; "50 MB L2 cache". Its (preliminary) ' +
        'table gives every tensor rate as dense/sparse 1:2 and equal with FP16 or FP32 accumulate ("Peak FP16 Tensor ' +
        'TFLOPS with FP16 Accumulate 1000/2000", "with FP32 Accumulate 1000/2000"), hence fp32-accumulate entries only.',
    },
    HOPPER_TILING_SOURCE,
  ],
  estimated: tilingEstimates(HOPPER_TILING_KEYS, 'Typical CUTLASS sm90 warp-specialized CTA tiles (m a multiple of 64)'),
};

/**
 * NVIDIA H200 SXM 141 GB (Hopper GH100 with HBM3e 4.8 TB/s, up to 700 W). Compute identical to {@link H100_SXM}.
 *
 * TRAP: tensor numbers on the product page are 2:4 SPARSE (footnote "² With sparsity."); the page also marks
 * the H200 column "Preliminary specifications. May be subject to change." SM count and L2 are not on the H200
 * page; they are taken from H100 SXM (same GH100 configuration, identical published TFLOPS).
 */
export const H200_SXM: DeviceSpec = {
  id: 'h200-sxm',
  name: 'NVIDIA H200 SXM 141GB',
  vendor: 'NVIDIA',
  engines: [hopperSxmEngine()],
  memory: {
    kind: 'dedicated',
    technology: 'HBM3e',
    // "141GB" as quoted (SI).
    capacityOptionsBytes: [gb(141)],
    bandwidthBytesPerSec: tbPerSec(4.8),
  },
  l2Bytes: mb(50),
  gpuDirectRdma: true,
  sources: [
    NVIDIA_DC_GPUDIRECT_SOURCE,
    {
      url: H200_PRODUCT_URL,
      title: 'NVIDIA H200 product page, Specifications (H200 SXM column)',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec', 'memory'],
      note:
        '"TF32 Tensor Core² 989 TFLOPS", "BFLOAT16 Tensor Core² 1,979", "FP16 Tensor Core² 1,979", "FP8 Tensor ' +
        'Core² 3,958", "INT8 Tensor Core² 3,958", footnote "² With sparsity." → dense = ½. "FP64 34 TFLOPS", "FP32 ' +
        '67 TFLOPS", "GPU Memory 141GB", "GPU Memory Bandwidth 4.8TB/s", "Up to 700W"; footnote "¹ Preliminary ' +
        'specifications. May be subject to change."',
    },
    HOPPER_TILING_SOURCE,
  ],
  estimated: [
    {
      field: 'engines.0.computeUnits',
      reason: '132 SMs as H100 SXM5 (Hopper blog): same GH100 die, identical published tensor and vector TFLOPS.',
    },
    { field: 'l2Bytes', reason: '50 MB as H100 SXM5 (Hopper blog): same GH100 configuration; the H200 page states no L2.' },
    ...tilingEstimates(HOPPER_TILING_KEYS, 'Typical CUTLASS sm90 warp-specialized CTA tiles (m a multiple of 64)'),
  ],
};

// ---------------------------------------------------------------------------------------------------------------
// Blackwell datacenter: B200, B300 (Blackwell Ultra)
// ---------------------------------------------------------------------------------------------------------------

/** NVIDIA Blackwell Architecture Technical Brief V2.1, Table 3 "System Specifications for HGX B300 and HGX B200". */
const BLACKWELL_BRIEF_URL = 'https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf';
/** NVIDIA HGX platform page (HGX B300 / B200 8-GPU totals; footnote "Dense is ½ sparse spec shown"). */
const HGX_PAGE_URL = 'https://www.nvidia.com/en-us/data-center/hgx/';
/** DGX B200 page (system memory totals). */
const DGX_B200_URL = 'https://www.nvidia.com/en-us/data-center/dgx-b200/';
/** NVIDIA Blackwell Ultra Datasheet (OCT25), HGX B300 column. */
const BLACKWELL_ULTRA_DATASHEET_URL = 'https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf';

/** Tiling keys of the Blackwell datacenter engines. */
const BLACKWELL_DC_TILING_KEYS = ['default', 'fp16', 'bf16', 'tf32', 'fp8', 'fp6', 'int8', 'fp4'];

/**
 * tcgen05 tiling (B200 / B300), cta_group::1 with M = 128: N in steps of 8, K per kind (f16 16, tf32 8,
 * f8f6f4 and i8 32, block-scaled mxf4nvf4 64). CTA tiles: m a multiple of 128.
 */
function blackwellDcTiling(): { default: GemmTiling } & Partial<Record<MathFormat, GemmTiling>> {
  return tilingTable(
    128,
    8,
    [
      { m: 128, n: 256 },
      { m: 256, n: 128 },
      { m: 128, n: 128 },
      { m: 128, n: 64 },
    ],
    1,
    { default: 16, fp16: 16, bf16: 16, tf32: 8, fp8: 32, fp6: 32, int8: 32, fp4: 64 },
  );
}

/** The tcgen05 tiling source (B200 / B300). */
const BLACKWELL_DC_TILING_SOURCE = tilingShapeSource(
  PTX_TCGEN05_URL,
  'PTX ISA 9.5: tcgen05.mma shapes (Table 48, sm_100a / sm_103a)',
  BLACKWELL_DC_TILING_KEYS,
  'kind::f16 cta_group::1: M = 64 or 128, "N = {8, 16, 24, … 256} steps of 8", K = 16; kind::tf32 K = 8; ' +
    'kind::f8f6f4 and kind::i8 K = 32; kind::mxf4nvf4 (block-scaled e2m1) K = 64. Sparse doubles K. mmaM 128 = ' +
    'the full-rate single-CTA shape.',
);

/**
 * B200 per-GPU matmul peaks (tech brief Table 3, HGX B200 column, "Dense/Sparse"): FP4 9/18 PF, FP8/FP6 4.5/9 PF,
 * INT8 4.5/9 POPS, FP16/BF16 2.2/4.5 PF, TF32 1.1/2.2 PF. FP16/BF16 dense = 4.5 / 2 = 2.25 PF (the HGX page:
 * 36 PF sparse per 8 GPUs, "Dense is ½ sparse"; the brief rounds it to 2.2).
 */
const B200_PEAKS: MatmulPeak[] = [
  { input: 'fp4', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(9000) },
  { input: 'fp4', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(18000) },
  { input: 'fp6', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(4500) },
  { input: 'fp6', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(9000) },
  { input: 'fp8', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(4500) },
  { input: 'fp8', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(9000) },
  { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(4500 / 2) },
  { input: 'fp16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(4500) },
  { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(4500 / 2) },
  { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(4500) },
  { input: 'tf32', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(1100) },
  { input: 'tf32', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(2200) },
  { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: tflops(4500) },
  { input: 'int8', accumulate: 'int32', sparsity: '2:4', flopsPerSec: tflops(9000) },
];

/**
 * NVIDIA B200 (HGX B200 / DGX B200 GPU; Blackwell, two dies, 148 SMs, 180 GB HBM3e, 7.7 TB/s, 1,000 W).
 *
 * Clock: NVIDIA publishes none; peaks are the vendor table's. TRAPS: the "144 PFLOPS" HGX headline is FP4
 * 2:4 sparse for 8 GPUs (18 PF sparse / 9 PF dense per GPU). Memory: the tech brief's per-GPU row says "Up to
 * 192 GB HBM3e | 7.7 TB/s"; the shipping HGX/DGX B200 systems carry 1.4 TB / 1,440 GB per 8 GPUs = 180 GB per
 * GPU, so 180 GB is used (192 GB is not an orderable SKU). Bandwidth: 7.7 TB/s from the brief; DGX B200 states
 * 64 TB/s / 8 = 8 TB/s (the conservative 7.7 is kept). The GB200 variant (1,200 W, FP4 10/20 PF) is a different
 * part. SM count is not published by NVIDIA (Chips and Cheese).
 */
export const B200: DeviceSpec = {
  id: 'b200',
  name: 'NVIDIA B200 (HGX)',
  vendor: 'NVIDIA',
  engines: [
    {
      id: 'gpu',
      name: 'Blackwell 5th-gen Tensor Cores (tcgen05)',
      kind: 'gpu',
      computeUnits: 148,
      matmulPeaks: B200_PEAKS,
      // FP32 75 TFLOP/s; FP64 (and FP64 Tensor Core) 37 TFLOP/s.
      vectorFlopsPerSec: { fp32: tflops(75), fp64: tflops(37) },
      gemmTiling: blackwellDcTiling(),
    },
  ],
  memory: {
    kind: 'dedicated',
    technology: 'HBM3e',
    // 1,440 GB / 8 GPUs = 180 GB per GPU (DGX B200; HGX page "Total Memory 1.4 TB"), SI.
    capacityOptionsBytes: [gb(1440 / 8)],
    bandwidthBytesPerSec: tbPerSec(7.7),
  },
  // "126 MB" as quoted (SI). Not used by the v1 model.
  l2Bytes: mb(126),
  gpuDirectRdma: true,
  sources: [
    NVIDIA_DC_GPUDIRECT_SOURCE,
    {
      url: BLACKWELL_BRIEF_URL,
      title: 'NVIDIA Blackwell Architecture Technical Brief V2.1, Table 3 (HGX B200 column, "Per GPU Specs")',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec', 'memory.bandwidthBytesPerSec'],
      note:
        '"FP4 Tensor Core Dense/Sparse 9/18 petaFLOPS", "FP8/FP6 Tensor Core Dense/Sparse 4.5/9", "INT8 Tensor Core ' +
        'Dense/Sparse 4.5/9 petaOPS", "FP16/BF16 Tensor Core Dense/Sparse 2.2/4.5", "TF32 Tensor Core Dense/Sparse ' +
        '1.1/2.2", "FP32 75 teraFLOPS", "FP64 Tensor Core | FP64 37 teraFLOPS", "GPU memory | Bandwidth Up to 192 ' +
        'GB HBM3e | 7.7 TB/s", "Max thermal design power (TDP) 1000W". FP16/BF16 dense recorded as 4.5 / 2 = 2.25.',
    },
    {
      url: HGX_PAGE_URL,
      title: 'NVIDIA HGX platform page, HGX B200 column (8 GPUs)',
      fields: ['engines.0.matmulPeaks', 'memory.capacityOptionsBytes'],
      note:
        '"FP4 Tensor Core¹ 144 PFLOPS | 72 PFLOPS" ("¹ Specification in Sparse | Dense"), "FP16/BF16 Tensor Core² ' +
        '36 PFLOPS" ("² Specification in Sparse. Dense is ½ sparse spec shown.") → 36 / 8 / 2 = 2.25 PF dense per ' +
        'GPU; "Total Memory 1.4 TB" → ≈ 180 GB per GPU.',
    },
    {
      url: DGX_B200_URL,
      title: 'NVIDIA DGX B200 page, specifications',
      fields: ['memory.capacityOptionsBytes'],
      note: '"GPU Memory 1,440 GB total, 64 TB/s HBM3e bandwidth" (8 GPUs) → 180 GB and 8 TB/s per GPU.',
    },
    {
      url: 'https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut',
      title: 'Chips and Cheese: NVIDIA\'s B200, keeping the CUDA juggernaut rolling',
      fields: ['engines.0.computeUnits', 'l2Bytes'],
      kind: 'review',
      note:
        '"B200 enables 74 SMs per die, giving 148 SMs across the GPU" (NVIDIA does not publish it; arXiv 2512.02189 ' +
        'agrees); measured "126 MB of total capacity" L2.',
    },
    {
      url: 'https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html',
      title: 'NVIDIA Blackwell Tuning Guide',
      fields: ['l2Bytes'],
      note: '"The NVIDIA GB200 GPU increases the L2 cache capacity to 126 MB" (same B200 die).',
    },
    BLACKWELL_DC_TILING_SOURCE,
  ],
  estimated: tilingEstimates(BLACKWELL_DC_TILING_KEYS, 'Typical CUTLASS sm100 CTA tiles (m a multiple of 128)'),
};

/**
 * B300 per-GPU matmul peaks (Blackwell Ultra datasheet, HGX B300 column): FP4 "18 PFLOPS | 14 PFLOPS"
 * (sparse | dense; NOT 2:1); the rest sparse with "Dense is ½ sparse spec shown": FP8/FP6 9 → 4.5,
 * INT8 307 TOPS → 153.5, FP16/BF16 4.5 → 2.25, TF32 2.2 → 1.1 PF.
 */
const B300_PEAKS: MatmulPeak[] = [
  { input: 'fp4', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(14000) },
  { input: 'fp4', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(18000) },
  { input: 'fp6', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(9000 / 2) },
  { input: 'fp6', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(9000) },
  { input: 'fp8', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(9000 / 2) },
  { input: 'fp8', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(9000) },
  { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(4500 / 2) },
  { input: 'fp16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(4500) },
  { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(4500 / 2) },
  { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(4500) },
  { input: 'tf32', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(2200 / 2) },
  { input: 'tf32', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(2200) },
  // INT8 cut ~30× vs B200 (Blackwell Ultra trades INT8 / FP64 for NVFP4).
  { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: tflops(307 / 2) },
  { input: 'int8', accumulate: 'int32', sparsity: '2:4', flopsPerSec: tflops(307) },
];

/**
 * NVIDIA B300 (HGX B300; Blackwell Ultra, two dies, 148 active SMs, 270 GB HBM3e, 7.7 TB/s, up to 1,100 W).
 *
 * Clock: NVIDIA publishes none; peaks are the datasheet's. TRAPS: dense FP4 is 14 PF but sparse FP4 is only
 * 18 PF (sparsity is not 2× here); the HGX page's 8-GPU "144 PFLOPS | 108 PFLOPS" gives 13.5 PF dense per GPU
 * (the datasheet's 14 is used). INT8 drops to 153.5 TOPS dense (0.15 POPS in the tech brief) and FP64 to
 * 1.2 TFLOP/s: INT8 models run far slower than on B200. SM count: NVIDIA says "up to 160 SMs ... varies by
 * SKU"; 148 is a measured count (arXiv). L2 size is not published, so `l2Bytes` is omitted. The GB300 variant
 * (279 GB, 8 TB/s, FP4 15/20 PF, 1,400 W) is a different part.
 */
export const B300: DeviceSpec = {
  id: 'b300',
  name: 'NVIDIA B300 (HGX, Blackwell Ultra)',
  vendor: 'NVIDIA',
  engines: [
    {
      id: 'gpu',
      name: 'Blackwell Ultra 5th-gen Tensor Cores (tcgen05)',
      kind: 'gpu',
      computeUnits: 148,
      matmulPeaks: B300_PEAKS,
      // FP32 75 TFLOP/s; FP64 (and FP64 Tensor Core) 1.2 TFLOP/s.
      vectorFlopsPerSec: { fp32: tflops(75), fp64: tflops(1.2) },
      gemmTiling: blackwellDcTiling(),
    },
  ],
  memory: {
    kind: 'dedicated',
    technology: 'HBM3e',
    // "270 GB" as quoted (SI).
    capacityOptionsBytes: [gb(270)],
    bandwidthBytesPerSec: tbPerSec(7.7),
  },
  gpuDirectRdma: true,
  sources: [
    NVIDIA_DC_GPUDIRECT_SOURCE,
    {
      url: BLACKWELL_ULTRA_DATASHEET_URL,
      title: 'NVIDIA Blackwell Ultra Datasheet (OCT25), HGX B300 column (per GPU)',
      fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec', 'memory'],
      note:
        '"FP4 Tensor Core¹ 18 PFLOPS | 14 PFLOPS" ("1. Specification in Sparse | Dense"); "FP8/FP6 Tensor Core² 9 ' +
        'PFLOPS", "INT8 Tensor Core² 307 TOPS", "FP16/BF16 Tensor Core² 4.5 PFLOPS", "TF32 Tensor Core² 2.2 PFLOPS" ' +
        '("2. Specification in sparse. Dense is ½ sparse spec shown."); "FP32 75 TFLOPS", "FP64/FP64 Tensor Core ' +
        '1.2 TFLOPS", "GPU Memory | Bandwidth 270 GB HBM3E | 7.7 TB/s", "Configurable up to 1,100 W".',
    },
    {
      url: BLACKWELL_BRIEF_URL,
      title: 'NVIDIA Blackwell Architecture Technical Brief V2.1, Table 3 (HGX B300 column, "Per GPU Specs")',
      fields: ['engines.0.matmulPeaks'],
      note:
        'Agrees with the datasheet: "FP4 14/18", "FP8/FP6 4.5/9", "INT8 Tensor Core Dense/Sparse 0.15/ 0.30 petaOPS", ' +
        '"FP16/BF16 2.2/4.5", "TF32 1.1/2.2", "270 GB HBM3e | 7.7 TB/s", "1100 W". The HGX page (8 GPUs) instead ' +
        'reads FP4 "144 PFLOPS | 108 PFLOPS" (13.5 dense per GPU) and INT8 "3 POPS" sparse (0.375 per GPU); the ' +
        'datasheet values are used.',
    },
    {
      url: 'https://arxiv.org/html/2609.15034',
      title: 'Isaacs et al., "FastPair: GPU-Optimized String Decoding" (arXiv 2609.15034), Table 1',
      fields: ['engines.0.computeUnits'],
      kind: 'review',
      note:
        'Measured device table: "B300 | Blackwell Ultra | 10.3 | HBM3e | 8.0 TB/s | 148 SMs". NVIDIA only says ' +
        '"Blackwell Ultra GPUs contain up to 160 SMs ... Available SM count and HBM capacity varies by SKU" ' +
        '(https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/).',
    },
    BLACKWELL_DC_TILING_SOURCE,
  ],
  estimated: tilingEstimates(BLACKWELL_DC_TILING_KEYS, 'Typical CUTLASS sm100 CTA tiles (m a multiple of 128)'),
};

/**
 * Every NVIDIA datacenter / DGX device preset, in display order (DGX Spark, then the datacenter parts newest
 * generation first and higher tier first). Each must pass `validateDevice`.
 */
export const NVIDIA_DATACENTER_DEVICE_PRESETS: DeviceSpec[] = [DGX_SPARK, B300, B200, H200_SXM, H100_SXM];
