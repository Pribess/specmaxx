/**
 * Intel Arc (Battlemage, Xe2-HPG) device presets. Every number is cited in `sources[]` or listed in
 * `estimated[]` with its derivation. See docs/simulation-model.md §4 ("Hardware").
 *
 * What Intel publishes, and what is derived:
 * - Intel quotes, per card, dense INT8 XMX TOPS and vector FP32 TFLOPS, both at the card's MAX clock
 *   (datasheet footnote: "As defined by maximum clock frequency"). Those two are vendor numbers here.
 * - Intel does NOT quote per-card FP16/BF16/INT4 XMX rates. They are derived from the Xe2 core's per-clock
 *   rates (Intel Tech Tour 2024 "Xe2 and Lunar Lake's GPU" deck, pp. 8–9, the "2nd Gen Xe-core" that
 *   Battlemage uses): XMX FP16/BF16 2048, INT8 4096, INT4 8192 OPS/clk per Xe core; vector FP32 256,
 *   FP16 512 FLOP/clk per Xe core. `peak = Xe cores × ops/clk/core × max clock`. The same formula with
 *   4096 reproduces each card's vendor INT8 TOPS to 3 significant digits (checked in intel.test.ts),
 *   which is what justifies the derivation. Derived entries are listed in `estimated[]`.
 *
 * Formats Xe2 XMX does NOT have (so they are omitted and the engine falls back to a wider format):
 * - No structured sparsity: no Xe2 document lists a sparse rate, and the TOPS footnote says "dense models".
 * - No FP8 / FP4: the deck lists only INT2/INT4/INT8/FP16/BF16; Chips and Cheese (Hot Chips 2026, Crescent
 *   Island) states FP8/FP4 XMX support arrives with Xe3P. FP8 weights are dequantized to FP16 on Xe2.
 * - TF32 is omitted: the SYCL joint_matrix spec lists a tf32 DPAS shape for BMG, but the Xe2 XMX dtype slide
 *   does not, and no rate is published (it may be emulated).
 * - FP64 vector is native (EM port) but its rate is unpublished: omitted.
 * - Accumulate: fp32 for fp16/bf16 (int32 for integers). joint_matrix also allows fp16/bf16 C/D types, with
 *   no published rate difference, so only fp32-accumulate entries are listed.
 *
 * Multi-GPU (the link specs live in `links.ts` / `topologies.ts`, not here): Arc Pro cards have no Xe Link
 * or bridge; GPU↔GPU traffic is PCIe peer-to-peer (Intel "Project Battlematrix", up to 8 × B60).
 * - Arc Pro B70 and B65: PCIe 5.0 x16 native.
 * - Arc Pro B60: x16 connector, PCIe 5.0 **x8 electrical** ("Gen 5.0 x16 (x8 Electrical)").
 * - Maxsun Arc Pro B60 Dual 48G Turbo: two full B60s on one board, each on PCIe 5.0 x8, with no onboard PCIe
 *   switch (the slot must bifurcate x8/x8). Model it as 2 × {@link ARC_PRO_B60} devices, not one 48 GB device.
 * - Arc B580: PCIe 4.0 x8.
 */

import type { ComputeEngine, DeviceSpec, EstimatedField, GemmTiling, MatmulPeak, SourceRef } from '../types';
import { gb, gbPerSec, tflops } from '../units';
import { engineSfu, INTEL_XE2_SFU, type EngineSfu } from './sfu';
import { NO_RDMA_SOURCE_ESTIMATE } from './gpudirect';

// ---------------------------------------------------------------------------------------------------------
// Source URLs
// ---------------------------------------------------------------------------------------------------------

/** Intel Tech Tour 2024, "Xe2 and Lunar Lake's GPU" deck (per-Xe-core XMX / vector ops per clock, pp. 6–9). */
const XE2_TECH_TOUR_URL =
  'https://cdrdv2-public.intel.com/824434/2024_Intel_Tech%20Tour%20TW_Xe2%20and%20Lunar%20Lakes%20GPU.pdf';
/** SYCL joint_matrix extension spec: "Intel XMX Supported Combinations" table (intel_gpu_bmg_g21 / bmg_g31). */
const SYCL_JOINT_MATRIX_URL =
  'https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc';
/** IGC vISA documentation of the DPAS (XMX) instruction: M = repeat count, N = exec size, K = depth × ops/chan. */
const IGC_DPAS_URL = 'https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md';
/** PCGH launch article for Arc Pro B70 / B65 with a spec table also covering Arc Pro B60 and Arc B580. */
const PCGH_B70_B65_URL =
  'https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/Specials/Arc-Pro-B70-B65-Preis-Gaming-Specs-Release-1523369/';

const B70_ARK_URL = 'https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html';
const B70_DATASHEET_URL =
  'https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf';
const B65_ARK_URL = 'https://www.intel.com/content/www/us/en/products/sku/245796/intel-arc-pro-b65-graphics/specifications.html';
const B60_ARK_URL = 'https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html';
const B60_DATASHEET_URL =
  'https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b60-gpu.pdf';
const B580_ARK_URL = 'https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html';
/** Intel oneAPI GPU optimization guide 2025-2, "Intel Xe GPU architecture" table (B580 column). */
const ONEAPI_XE_ARCH_URL =
  'https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/intel-xe-gpu-architecture.html';
/** Chips and Cheese, "Intel's Battlemage Architecture" (B580 measured at 2.85 GHz). */
const CNC_BATTLEMAGE_URL = 'https://chipsandcheese.com/p/intels-battlemage-architecture';

// ---------------------------------------------------------------------------------------------------------
// Xe2 per-core rates (Intel Tech Tour 2024 deck, p. 9 table). Unit: operations per clock per Xe core,
// counting a multiply-accumulate as 2 operations.
// ---------------------------------------------------------------------------------------------------------

/** XMX FP16/BF16: 8 XMX engines × SIMD16 × 2 MAC/lane × systolic depth 4 × 2 = 2048 FLOP/clk/core (deck p. 8). */
export const XE2_XMX_FP16_FLOP_PER_CLOCK_PER_CORE = 2048;
/** XMX INT8: 8 × 16 × 4 × 4 × 2 = 4096 OP/clk/core (deck p. 8: "INT8 4096 OPS/clock"). */
export const XE2_XMX_INT8_OP_PER_CLOCK_PER_CORE = 4096;
/** XMX INT4 (and INT2): 8 × 16 × 8 × 4 × 2 = 8192 OP/clk/core (deck p. 9: "XMX INT4 / INT2 … 8192"). */
export const XE2_XMX_INT4_OP_PER_CLOCK_PER_CORE = 8192;
/** Vector FP32: 8 XVE × SIMD16 × 1 FMA × 2 = 256 FLOP/clk/core (deck p. 9, row FP32). */
export const XE2_VECTOR_FP32_FLOP_PER_CLOCK_PER_CORE = 256;
/** Vector FP16: 512 FLOP/clk/core, 2 × FP32 (deck p. 9, row FP16). */
export const XE2_VECTOR_FP16_FLOP_PER_CLOCK_PER_CORE = 512;

// ---------------------------------------------------------------------------------------------------------
// Tiling
// ---------------------------------------------------------------------------------------------------------

/**
 * Candidate workgroup output tiles (elements, m × n) for Xe2 XMX GEMMs. ESTIMATED: representative of
 * oneDNN / XeTLA large-GEMM workgroup tiles, not a vendor spec. Every m is a multiple of mmaM 8, every n of
 * mmaN 16.
 */
const XE2_WORKGROUP_TILES: { m: number; n: number }[] = [
  { m: 256, n: 256 },
  { m: 128, n: 256 },
  { m: 128, n: 128 },
  { m: 64, n: 128 },
  { m: 32, n: 64 },
];

/**
 * Xe2 tiling for an input class whose DPAS shape is M 8 × N 16 × K `mmaK` (IGC DPAS doc; joint_matrix table
 * for bmg_g21/g31): fp16/bf16 8×16×16, int8 8×16×32, int4 8×16×64 (K = systolic depth 8 × ops per channel
 * 2 / 4 / 8). One workgroup per Xe core: a 256×256 tile fills the core's register file and SLM (estimate).
 */
function xe2Tiling(mmaK: number): GemmTiling {
  return { mmaM: 8, mmaN: 16, mmaK, ctaTiles: XE2_WORKGROUP_TILES.map((t) => ({ ...t })), ctasPerUnit: 1 };
}

/** The tiling keys each Xe2 engine lists (used to write the source/estimate field paths below). */
const XE2_TILING_KEYS = ['default', 'fp16', 'bf16', 'int8', 'int4'] as const;

// ---------------------------------------------------------------------------------------------------------
// Engine builder and shared provenance
// ---------------------------------------------------------------------------------------------------------

/**
 * Index of each peak in {@link xe2Engine}'s `matmulPeaks`, so field paths (`engines.0.matmulPeaks.<i>`) in
 * sources / estimates name the right entry.
 */
const PEAK_INDEX = { fp16: 0, bf16: 1, int8: 2, int4: 3 } as const;

/** Inputs of one Xe2 engine: the core count and max clock, plus the two rates Intel quotes for the card. */
interface Xe2EngineParams {
  /** Xe2 cores (each 8 XMX + 8 XVE). Count. */
  xeCores: number;
  /** Max dynamic graphics clock the vendor peaks assume. Unit: Hz. */
  maxClockHz: number;
  /** Vendor dense INT8 XMX peak ("GPU Peak TOPS (Int8)"), per device. Unit: OP/s. */
  int8DenseOpsPerSec: number;
  /** FP32 vector peak, per device (vendor where Intel quotes it). Unit: FLOP/s. */
  fp32VectorFlopsPerSec: number;
}

/**
 * One Battlemage GPU engine. INT8 and vector FP32 are the caller's (vendor) numbers; the rest is derived as
 * `xeCores × ops/clk/core × maxClockHz` from the Xe2 per-core rates. All peaks dense, per device.
 */
function xe2Engine(name: string, p: Xe2EngineParams): ComputeEngine {
  // SFU (extended math) exp2: xeCores × 32 × clock (estimate, sfu.ts INTEL_XE2_SFU).
  const sfu = xe2Sfu(p.xeCores, p.maxClockHz);
  // FP16/BF16 XMX, fp32 accumulate: xeCores × 2048 FLOP/clk × clock.
  const fp16XmxFlopsPerSec = p.xeCores * XE2_XMX_FP16_FLOP_PER_CLOCK_PER_CORE * p.maxClockHz;
  // INT4 XMX, int32 accumulate: xeCores × 8192 OP/clk × clock.
  const int4XmxOpsPerSec = p.xeCores * XE2_XMX_INT4_OP_PER_CLOCK_PER_CORE * p.maxClockHz;
  // Vector FP16: xeCores × 512 FLOP/clk × clock (2 × FP32).
  const fp16VectorFlopsPerSec = p.xeCores * XE2_VECTOR_FP16_FLOP_PER_CLOCK_PER_CORE * p.maxClockHz;
  const matmulPeaks: MatmulPeak[] = [
    { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: fp16XmxFlopsPerSec },
    { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: fp16XmxFlopsPerSec },
    { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: p.int8DenseOpsPerSec },
    { input: 'int4', accumulate: 'int32', sparsity: 'dense', flopsPerSec: int4XmxOpsPerSec },
  ];
  return {
    id: 'gpu',
    name,
    kind: 'gpu',
    computeUnits: p.xeCores,
    matmulPeaks,
    vectorFlopsPerSec: { fp32: p.fp32VectorFlopsPerSec, fp16: fp16VectorFlopsPerSec },
    sfuOpsPerSec: sfu.sfuOpsPerSec,
    gemmTiling: {
      default: xe2Tiling(16),
      fp16: xe2Tiling(16),
      bf16: xe2Tiling(16),
      int8: xe2Tiling(32),
      int4: xe2Tiling(64),
    },
  };
}

/**
 * The SFU rate of an Xe2 engine with `xeCores` cores at `maxClockHz` (the clock of its vector peak), with the
 * provenance of `engines.0.sfuOpsPerSec` (spread into each device's `sources` and `estimated`).
 */
function xe2Sfu(xeCores: number, maxClockHz: number): EngineSfu {
  return engineSfu(INTEL_XE2_SFU, xeCores, 'Xe cores', maxClockHz, 'engines.0.sfuOpsPerSec');
}

/** Citation of the DPAS instruction shapes (mmaM/N/K of every tiling key). */
const XE2_DPAS_SHAPE_SOURCE: SourceRef = {
  url: SYCL_JOINT_MATRIX_URL,
  title: 'SYCL joint_matrix spec, "Intel XMX Supported Combinations" (intel_gpu_bmg_g21 / bmg_g31)',
  fields: XE2_TILING_KEYS.flatMap((k) => [
    `engines.0.gemmTiling.${k}.mmaM`,
    `engines.0.gemmTiling.${k}.mmaN`,
    `engines.0.gemmTiling.${k}.mmaK`,
  ]),
  note:
    'BMG rows: fp16/bf16 M "<= 8", N 16, K 16; int8 M <= 8, N 16, K 32. Int4 K 64 follows the DPAS rule ' +
    `K = systolic depth 8 × OPS_PER_CHAN (8 for sub-8-bit) from ${IGC_DPAS_URL}.`,
};

/** Per-card numbers the shared estimates need: core count and clock, as named in the arithmetic. */
interface Xe2Derivation {
  /** Card name, for the reason text. */
  card: string;
  /** Xe2 cores. Count. */
  xeCores: number;
  /** Max clock the peaks assume. Unit: GHz (for readable reasons only). */
  maxClockGHz: number;
  /** Vendor INT8 TOPS the INT8 formula is checked against. Unit: TOP/s. */
  vendorInt8Tops: number;
}

/**
 * The `estimated[]` entries every Xe2 engine shares: the derived FP16/BF16/INT4 XMX peaks and vector FP16
 * (with the arithmetic and the INT8 cross-check), and the workgroup tiles.
 */
function xe2Estimates(d: Xe2Derivation): EstimatedField[] {
  const derivedFrom = (opsPerClock: number): string =>
    `${d.xeCores} Xe cores × ${opsPerClock} ops/clk/core × ${d.maxClockGHz} GHz = ` +
    `${((d.xeCores * opsPerClock * d.maxClockGHz) / 1000).toFixed(1)} T/s`;
  const check =
    `Cross-check: ${derivedFrom(XE2_XMX_INT8_OP_PER_CLOCK_PER_CORE)} reproduces Intel's ${d.vendorInt8Tops} INT8 TOPS ` +
    `for the ${d.card}. Per-core rates: ${XE2_TECH_TOUR_URL} (p. 9 table).`;
  return [
    {
      field: `engines.0.matmulPeaks.${PEAK_INDEX.fp16}`,
      reason: `Intel quotes no FP16 XMX rate for the ${d.card}. FP16 (fp32 acc): ${derivedFrom(XE2_XMX_FP16_FLOP_PER_CLOCK_PER_CORE)} (= INT8 / 2). ${check}`,
    },
    {
      field: `engines.0.matmulPeaks.${PEAK_INDEX.bf16}`,
      reason: `Intel quotes no BF16 XMX rate for the ${d.card}. BF16 runs at the FP16 rate (deck p. 9 "XMX FP16 / BF16 … 2048"): ${derivedFrom(XE2_XMX_FP16_FLOP_PER_CLOCK_PER_CORE)}. ${check}`,
    },
    {
      field: `engines.0.matmulPeaks.${PEAK_INDEX.int4}`,
      reason: `Intel quotes no INT4 XMX rate for the ${d.card}. INT4 (int32 acc): ${derivedFrom(XE2_XMX_INT4_OP_PER_CLOCK_PER_CORE)} (= INT8 × 2). ${check}`,
    },
    {
      field: 'engines.0.vectorFlopsPerSec.fp16',
      reason: `Intel quotes no vector FP16 rate for the ${d.card}. ${derivedFrom(XE2_VECTOR_FP16_FLOP_PER_CLOCK_PER_CORE)} (= 2 × FP32, deck p. 9 rows FP32 256 / FP16 512). Note: PCGH's B70 table lists "FP16-Leistung ALUs" equal to FP32 (22.9), which contradicts Intel's slide; the slide is followed here.`,
    },
    ...XE2_TILING_KEYS.flatMap((k) => [
      {
        field: `engines.0.gemmTiling.${k}.ctaTiles`,
        reason: 'Representative oneDNN / XeTLA workgroup tiles (multiples of the DPAS 8×16 shape); not a vendor spec.',
      },
      {
        field: `engines.0.gemmTiling.${k}.ctasPerUnit`,
        reason: 'One resident large workgroup per Xe core (register file / SLM bound); kernel-dependent estimate.',
      },
    ]),
  ];
}

/** PCGH spec table (review): L2 sizes and boost clocks of B70 / B65 / B60 / B580. */
function pcghSource(fields: string[], note: string): SourceRef {
  return {
    url: PCGH_B70_B65_URL,
    title: 'PCGH: Intel launcht Arc Pro B70 & B65 (spec table, also lists Arc Pro B60 and Arc B580)',
    kind: 'review',
    fields,
    note,
  };
}

// ---------------------------------------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------------------------------------

/** B70 max dynamic clock (ark "Graphics Max Dynamic Clock 2800 MHz"; base "Graphics Clock 2280 MHz"). Unit: Hz. */
const B70_MAX_CLOCK_HZ = 2.8e9;

/**
 * Intel Arc Pro B70 (BMG-G31 full die, 32 Xe2 cores, 32 GB GDDR6, 608 GB/s, 24 MiB L2), March 2026.
 * Peaks at the 2800 MHz max dynamic clock (Intel's TOPS/TFLOPS convention), not the 2280 MHz graphics clock.
 * The "367 AI TOPS" headline is dense INT8, the only XMX rate Intel quotes; FP16/BF16 (183.5 TFLOP/s) and INT4
 * (734 TOP/s) are derived. PCIe 5.0 x16 native.
 */
export const ARC_PRO_B70: DeviceSpec = {
  id: 'arc-pro-b70',
  name: 'Intel Arc Pro B70',
  vendor: 'Intel',
  engines: [
    xe2Engine('BMG-G31 Xe2 XMX engines (256)', {
      xeCores: 32,
      maxClockHz: B70_MAX_CLOCK_HZ,
      // Vendor: 367 TOPS dense INT8 (= 32 × 4096 × 2.8 GHz = 367.0).
      int8DenseOpsPerSec: tflops(367),
      // Vendor: 22.94 TFLOPS FP32 (= 32 × 256 × 2.8 GHz = 22.94).
      fp32VectorFlopsPerSec: tflops(22.94),
    }),
  ],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6',
    // "32 GB" as quoted, recorded SI (32e9 B); physically 32 GiB (PCGH "32 GiByte"), so SI is conservative.
    capacityOptionsBytes: [gb(32)],
    // 256-bit × 19 Gbps / 8 = 608 GB/s.
    bandwidthBytesPerSec: gbPerSec(608),
  },
  // 24 MiB (PCGH "Level-2-Cache (MiB) 24", confirmed by Intel per PCGH). Recorded only; not used by the v1 model.
  l2Bytes: 24 * 1024 * 1024,
  // ark "PCI Express 5.0 x16"; datasheet "PCIe Support Gen 5 x16 Native".
  hostInterface: { generation: 5, lanes: 16 },
  gpuDirectRdma: false,
  sources: [
    {
      url: B70_ARK_URL,
      title: 'Intel ark: Intel Arc Pro B70 Graphics specifications',
      fields: ['engines.0.computeUnits', `engines.0.matmulPeaks.${PEAK_INDEX.int8}`, 'engines.0.vectorFlopsPerSec.fp32', 'memory', 'hostInterface'],
      note:
        'Xe-cores 32; XMX Engines 256; Graphics Max Dynamic Clock 2800 MHz (Graphics Clock 2280 MHz); ' +
        'GPU Peak TOPS (Int8) 367; GPU TFLOPS (FP32) 22.94; 32 GB GDDR6, 256 bit, 608 GB/s, ECC; PCI Express 5.0 x16.',
    },
    {
      url: B70_DATASHEET_URL,
      title: 'Intel Arc Pro B70 datasheet (2026-03)',
      fields: [`engines.0.matmulPeaks.${PEAK_INDEX.int8}`, 'memory'],
      note:
        '"367 AI TOPS Int8 Dense"; footnote: "peak throughput when running XMX workloads with INT8 datatype and dense ' +
        'models"; "32 GB VRAM"; "608 GB/s"; "PCIe Support Gen 5 x16 Native"; TBP 230 W (Intel card), 160–290 W partner.',
    },
    pcghSource(['l2Bytes'], 'B70 column: BMG-G31 WKSTN, 368 mm², 27.7 B transistors; Level-2-Cache (MiB) 24; boost 2800 MHz; FP32 22.9 TFLOPS.'),
    XE2_DPAS_SHAPE_SOURCE,
    ...xe2Sfu(32, B70_MAX_CLOCK_HZ).sources,
  ],
  estimated: [
    NO_RDMA_SOURCE_ESTIMATE,
    ...xe2Estimates({ card: 'Arc Pro B70', xeCores: 32, maxClockGHz: 2.8, vendorInt8Tops: 367 }),
    ...xe2Sfu(32, B70_MAX_CLOCK_HZ).estimated,
  ],
};

/** B65 / B60 max clock (ark "Graphics Clock 2400 MHz"; datasheet: peaks "as defined by maximum clock frequency"). Unit: Hz. */
const B60_B65_MAX_CLOCK_HZ = 2.4e9;

/**
 * Intel Arc Pro B65 (BMG-G31 cut to 20 Xe2 cores, 32 GB GDDR6, 608 GB/s, 18 MiB L2), April 2026.
 * Same compute as the B60 (20 cores at 2400 MHz, 197 INT8 TOPS) with the B70's memory system and PCIe 5.0 x16.
 */
export const ARC_PRO_B65: DeviceSpec = {
  id: 'arc-pro-b65',
  name: 'Intel Arc Pro B65',
  vendor: 'Intel',
  engines: [
    xe2Engine('BMG-G31 Xe2 XMX engines (160)', {
      xeCores: 20,
      maxClockHz: B60_B65_MAX_CLOCK_HZ,
      // Vendor: 197 TOPS dense INT8 (= 20 × 4096 × 2.4 GHz = 196.6).
      int8DenseOpsPerSec: tflops(197),
      // Vendor: 12.28 TFLOPS FP32 (= 20 × 256 × 2.4 GHz = 12.29).
      fp32VectorFlopsPerSec: tflops(12.28),
    }),
  ],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6',
    // "32 GB" as quoted, recorded SI.
    capacityOptionsBytes: [gb(32)],
    // 256-bit GDDR6, 608 GB/s (ark).
    bandwidthBytesPerSec: gbPerSec(608),
  },
  // 18 MiB (PCGH B65 column). Recorded only.
  l2Bytes: 18 * 1024 * 1024,
  // ark "PCI Express 5.0 x16".
  hostInterface: { generation: 5, lanes: 16 },
  gpuDirectRdma: false,
  sources: [
    {
      url: B65_ARK_URL,
      title: 'Intel ark: Intel Arc Pro B65 Graphics specifications',
      fields: ['engines.0.computeUnits', `engines.0.matmulPeaks.${PEAK_INDEX.int8}`, 'engines.0.vectorFlopsPerSec.fp32', 'memory', 'hostInterface'],
      note: 'Xe-cores 20; XMX Engines 160; 2400 MHz; GPU Peak TOPS (Int8) 197; FP32 12.28 TFLOPS; 32 GB GDDR6, 256 bit, 608 GB/s; PCI Express 5.0 x16; TBP 200 W.',
    },
    pcghSource(['l2Bytes'], 'B65 column: BMG-G31 WKSTN (cut down), 2560 FP32 ALUs, 160 matrix units; Level-2-Cache (MiB) 18; boost 2400 MHz.'),
    XE2_DPAS_SHAPE_SOURCE,
    ...xe2Sfu(20, B60_B65_MAX_CLOCK_HZ).sources,
  ],
  estimated: [
    NO_RDMA_SOURCE_ESTIMATE,
    ...xe2Estimates({ card: 'Arc Pro B65', xeCores: 20, maxClockGHz: 2.4, vendorInt8Tops: 197 }),
    ...xe2Sfu(20, B60_B65_MAX_CLOCK_HZ).estimated,
  ],
};

/**
 * Intel Arc Pro B60 (BMG-G21, 20 Xe2 cores, 24 GB GDDR6, 456 GB/s, 18 MiB L2), 2025.
 * Peaks at the 2400 MHz clock of the standard SKU (the low-profile SKU runs 2000 MHz and is NOT modeled).
 * Footnote trap: x16 connector but PCIe 5.0 **x8 electrical**. The Maxsun "B60 Dual 48G" is two of these on
 * x8/x8 bifurcation: model it as 2 devices.
 */
export const ARC_PRO_B60: DeviceSpec = {
  id: 'arc-pro-b60',
  name: 'Intel Arc Pro B60',
  vendor: 'Intel',
  engines: [
    xe2Engine('BMG-G21 Xe2 XMX engines (160)', {
      xeCores: 20,
      maxClockHz: B60_B65_MAX_CLOCK_HZ,
      // Vendor: 197 TOPS dense INT8 (= 20 × 4096 × 2.4 GHz = 196.6).
      int8DenseOpsPerSec: tflops(197),
      // Vendor: "Up to 12.28 TFLOPS (Single Precision)" (= 20 × 256 × 2.4 GHz = 12.29).
      fp32VectorFlopsPerSec: tflops(12.28),
    }),
  ],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6',
    // "24 GB" as quoted, recorded SI.
    capacityOptionsBytes: [gb(24)],
    // 192-bit × 19 Gbps / 8 = 456 GB/s.
    bandwidthBytesPerSec: gbPerSec(456),
  },
  // 18 MiB (PCGH B60 column; same G21 die as the B580, for which Intel's guide lists 18 MB). Recorded only.
  l2Bytes: 18 * 1024 * 1024,
  // x16 connector, x8 ELECTRICAL: datasheet "Gen 5.0 x16 (x8 Electrical)", ark "PCI Express 5.0 x8".
  hostInterface: { generation: 5, lanes: 8 },
  gpuDirectRdma: false,
  sources: [
    {
      url: B60_DATASHEET_URL,
      title: 'Intel Arc Pro B60 datasheet (2026-03)',
      fields: ['engines.0.computeUnits', `engines.0.matmulPeaks.${PEAK_INDEX.int8}`, 'engines.0.vectorFlopsPerSec.fp32', 'memory', 'hostInterface'],
      note:
        '20 Xe2-HPG cores, 160 XMX engines; "197 AI TOPS Int 8 Dense" (footnote: "INT8 datatype and dense models"); ' +
        '"Up to 12.28 TFLOPS (Single Precision)" (footnote: "As defined by maximum clock frequency"); 24GB GDDR6; ' +
        '456 GB/s; "Gen 5.0 x16 (x8 Electrical)"; 120–200 W TBP.',
    },
    {
      url: B60_ARK_URL,
      title: 'Intel ark: Intel Arc Pro B60 Graphics specifications',
      fields: ['engines.0.computeUnits', 'memory', 'hostInterface'],
      note: 'Xe-cores 20; Graphics Clock 2400 MHz (LP 2000 MHz); 24 GB GDDR6, 192 bit, 19 Gbps, 456 GB/s; PCI Express 5.0 x8.',
    },
    pcghSource(['l2Bytes'], 'B60 column: BMG-G21 WKSTN, 272 mm²; Level-2-Cache (MiB) 18; boost 2400 MHz; FP32 12.3 TFLOPS.'),
    XE2_DPAS_SHAPE_SOURCE,
    ...xe2Sfu(20, B60_B65_MAX_CLOCK_HZ).sources,
  ],
  estimated: [
    NO_RDMA_SOURCE_ESTIMATE,
    ...xe2Estimates({ card: 'Arc Pro B60', xeCores: 20, maxClockGHz: 2.4, vendorInt8Tops: 197 }),
    ...xe2Sfu(20, B60_B65_MAX_CLOCK_HZ).estimated,
  ],
};

/**
 * B580 boost clock 2850 MHz (PCGH "Typ. GPU-Boost-Takt 2.850"; Chips and Cheese "at 2.85 GHz"). Intel's ark lists
 * only the 2670 MHz graphics clock, but its 233 INT8 TOPS matches 20 × 4096 × 2.85 GHz = 233.5 (at 2670 MHz it
 * would be 218.7), so the TOPS figure assumes 2850 MHz. Unit: Hz.
 */
const B580_MAX_CLOCK_HZ = 2.85e9;

/**
 * Intel Arc B580 (consumer, BMG-G21, 20 Xe2 cores, 12 GB GDDR6, 456 GB/s, 18 MB L2), December 2024.
 * Peaks at the 2850 MHz boost clock (a review number; see {@link B580_MAX_CLOCK_HZ}), not ark's 2670 MHz.
 * Intel quotes no FP32 TFLOPS on ark, so vector FP32 is derived too. PCIe 4.0 x8.
 */
export const ARC_B580: DeviceSpec = {
  id: 'arc-b580',
  name: 'Intel Arc B580',
  vendor: 'Intel',
  engines: [
    xe2Engine('BMG-G21 Xe2 XMX engines (160)', {
      xeCores: 20,
      maxClockHz: B580_MAX_CLOCK_HZ,
      // Vendor: 233 TOPS dense INT8 (= 20 × 4096 × 2.85 GHz = 233.5).
      int8DenseOpsPerSec: tflops(233),
      // Derived: 20 × 256 × 2.85 GHz = 14.59 TFLOP/s (PCGH quotes 14.6).
      fp32VectorFlopsPerSec: 20 * XE2_VECTOR_FP32_FLOP_PER_CLOCK_PER_CORE * B580_MAX_CLOCK_HZ,
    }),
  ],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6',
    // "12 GB" as quoted, recorded SI.
    capacityOptionsBytes: [gb(12)],
    // 192-bit × 19 Gbps / 8 = 456 GB/s.
    bandwidthBytesPerSec: gbPerSec(456),
  },
  // Intel: "L3 cache size 18 MB" (Intel's name for the GPU-wide last-level cache, which is the L2). Taken as
  // binary MB (18 MiB), matching PCGH's 18 MiB for the same G21 die. Recorded only.
  l2Bytes: 18 * 1024 * 1024,
  // ark "PCI Express 4.0 x8".
  hostInterface: { generation: 4, lanes: 8 },
  gpuDirectRdma: false,
  sources: [
    {
      url: B580_ARK_URL,
      title: 'Intel ark: Intel Arc B580 Graphics specifications',
      fields: ['engines.0.computeUnits', `engines.0.matmulPeaks.${PEAK_INDEX.int8}`, 'memory', 'hostInterface'],
      note: 'Xe-cores 20; XMX 160; Graphics Clock 2670 MHz; GPU Peak TOPS (Int8) 233; 12 GB GDDR6, 192 bit, 19 Gbps, 456 GB/s; PCI Express 4.0 x8; TBP 190 W.',
    },
    {
      url: ONEAPI_XE_ARCH_URL,
      title: 'Intel oneAPI GPU optimization guide 2025-2: Intel Xe GPU architecture table (B580 column)',
      fields: ['l2Bytes'],
      note: '"L3 cache size 18 MB" (Chips and Cheese: "The entire GPU shares an 18 MB L2 cache"); 256 KB L1 / 128 KB SLM per Xe core.',
    },
    XE2_DPAS_SHAPE_SOURCE,
    ...xe2Sfu(20, B580_MAX_CLOCK_HZ).sources,
  ],
  estimated: [
    NO_RDMA_SOURCE_ESTIMATE,
    {
      field: 'engines.0.vectorFlopsPerSec.fp32',
      reason:
        `Intel ark quotes no FP32 TFLOPS for the B580. 20 Xe cores × 256 FLOP/clk/core × 2.85 GHz = 14.59 TFLOP/s, ` +
        `at the boost clock from PCGH (${PCGH_B70_B65_URL}: "Typ. GPU-Boost-Takt 2.850", "FP32 14,6") and ` +
        `Chips and Cheese (${CNC_BATTLEMAGE_URL}: "at 2.85 GHz").`,
    },
    ...xe2Estimates({ card: 'Arc B580', xeCores: 20, maxClockGHz: 2.85, vendorInt8Tops: 233 }),
    ...xe2Sfu(20, B580_MAX_CLOCK_HZ).estimated,
  ],
};

/**
 * Every Intel device preset (Battlemage / Xe2), per-device peaks at Intel's max-clock convention. No unlock
 * axes: no community unlock with evidence exists for these cards.
 */
export const INTEL_DEVICE_PRESETS: DeviceSpec[] = [ARC_PRO_B70, ARC_PRO_B65, ARC_PRO_B60, ARC_B580];
