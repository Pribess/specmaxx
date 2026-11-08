/**
 * NVIDIA device presets. Every number is cited in `sources[]` (with the page/table it comes from) or listed in
 * `estimated[]` with the reason. See docs/simulation-model.md §4 ("Hardware").
 */

import type { MathFormat } from '../../ir/dtype';
import type { DeviceSpec, GemmTiling, MatmulPeak, SourceRef } from '../types';
import type { UnlockAxis } from '../unlocks';
import { gbPerSec, gb, tflops } from '../units';

/** NVIDIA RTX Blackwell GPU Architecture whitepaper (Appendix A, Table 3: GeForce RTX 5090 specs, pp. 46–48). */
const RTX_BLACKWELL_WHITEPAPER_URL =
  'https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf';
/** GeForce RTX 5090 product page. */
const RTX_5090_PRODUCT_URL = 'https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/';
/** PTX ISA, warp-level matrix (mma.sync) shapes. */
const PTX_MMA_SHAPES_URL =
  'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma';

/**
 * Typical CUTLASS CTA output tiles for GeForce sm_86/sm_89/sm_120 `mma.sync` GEMMs (elements, m × n).
 * See each `estimated` entry. Every m is a multiple of mmaM 16 and every n of mmaN 8.
 */
const SM120_CTA_TILES: { m: number; n: number }[] = [
  { m: 128, n: 256 },
  { m: 128, n: 128 },
  { m: 128, n: 64 },
  { m: 64, n: 128 },
  { m: 64, n: 64 },
];

/**
 * GeForce tiling for an input class whose `mma.sync` shape is m16 n8 k`mmaK` (PTX ISA):
 * f16/bf16 m16n8k16, tf32 m16n8k8, e4m3/e5m2 and s8 m16n8k32, block-scaled fp4 (mxf4nvf4) m16n8k64.
 * One CTA per SM (a 128×256 or 128×128 tile uses most of the SM's registers and shared memory).
 */
function geforceTiling(mmaK: number): GemmTiling {
  return {
    mmaM: 16,
    mmaN: 8,
    mmaK,
    ctaTiles: SM120_CTA_TILES.map((t) => ({ ...t })),
    ctasPerUnit: 1,
  };
}

/** The tiling keys the RTX 5090 engine lists (used to write the source/estimate field paths below). */
const RTX_5090_TILING_KEYS: ('default' | MathFormat)[] = ['default', 'fp16', 'bf16', 'tf32', 'fp8', 'int8', 'fp4'];

/** Whitepaper Appendix A, Table 3 citation (pages 46–48). */
const RTX_5090_WHITEPAPER: SourceRef = {
  url: RTX_BLACKWELL_WHITEPAPER_URL,
  title: 'NVIDIA RTX Blackwell GPU Architecture (whitepaper), Appendix A, Table 3: GeForce RTX 5090 specs',
  fields: [
    'engines.0.computeUnits',
    'engines.0.matmulPeaks',
    'engines.0.vectorFlopsPerSec.fp32',
    'engines.0.vectorFlopsPerSec.fp16',
    'engines.0.vectorFlopsPerSec.bf16',
    'memory.bandwidthBytesPerSec',
    'l2Bytes',
  ],
  note:
    'Table 3 (pp. 46–48), all peaks at the 2407 MHz boost clock; "a/b" entries are dense / 2:4 sparse. ' +
    'SMs 170; Peak FP32/FP16/BF16 TFLOPS (non-Tensor) 104.8; FP4 Tensor with FP32 accumulate 1676/3352; ' +
    'FP8 with FP16 acc 838/1676, with FP32 acc 419/838; FP16 with FP16 acc 419/838, with FP32 acc 209.5/419; ' +
    'BF16 with FP32 acc 209.5/419; TF32 104.8/209.5; INT8 Tensor TOPS 838/1676; memory 512-bit, 28 Gbps, ' +
    '1792 GB/sec; L2 98304 KB (binary KB: the full GB202 has 131,072 KB = 128 MiB). The whitepaper lists no ' +
    'FP6 or INT4 tensor rate, so those inputs fall back to FP8 / INT8 (conservative).',
};

/**
 * GeForce RTX 5090 (GB202, 170 SMs, 32 GB GDDR7, 1792 GB/s), base spec at the 2407 MHz boost clock.
 *
 * GeForce rule: fp16/fp8 matmuls with fp32 accumulate run at HALF the fp16-accumulate rate, and bf16 / tf32
 * only accumulate in fp32 (so bf16 runs at 209.5 TFLOP/s dense). The "3352 AI TOPS" headline is FP4 with 2:4
 * sparsity; dense FP4 is 1676.
 */
export const RTX_5090: DeviceSpec = {
  id: 'rtx-5090',
  name: 'NVIDIA GeForce RTX 5090',
  vendor: 'NVIDIA',
  engines: [
    {
      id: 'gpu',
      name: 'GB202 5th-gen Tensor Cores',
      kind: 'gpu',
      computeUnits: 170,
      matmulPeaks: [
        { input: 'fp16', accumulate: 'fp16', sparsity: 'dense', flopsPerSec: tflops(419) },
        { input: 'fp16', accumulate: 'fp16', sparsity: '2:4', flopsPerSec: tflops(838) },
        { input: 'fp16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(209.5) },
        { input: 'fp16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(419) },
        { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(209.5) },
        { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(419) },
        { input: 'tf32', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(104.8) },
        { input: 'tf32', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(209.5) },
        { input: 'fp8', accumulate: 'fp16', sparsity: 'dense', flopsPerSec: tflops(838) },
        { input: 'fp8', accumulate: 'fp16', sparsity: '2:4', flopsPerSec: tflops(1676) },
        { input: 'fp8', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(419) },
        { input: 'fp8', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(838) },
        { input: 'fp4', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: tflops(1676) },
        { input: 'fp4', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: tflops(3352) },
        { input: 'int8', accumulate: 'int32', sparsity: 'dense', flopsPerSec: tflops(838) },
        { input: 'int8', accumulate: 'int32', sparsity: '2:4', flopsPerSec: tflops(1676) },
      ],
      vectorFlopsPerSec: {
        fp32: tflops(104.8),
        fp16: tflops(104.8),
        bf16: tflops(104.8),
        // FP64 = 1/64 of the FP32 rate (whitepaper p. 14 note): 104.8 / 64 = 1.6375 TFLOP/s.
        fp64: tflops(104.8 / 64),
      },
      gemmTiling: {
        default: geforceTiling(16),
        fp16: geforceTiling(16),
        bf16: geforceTiling(16),
        tf32: geforceTiling(8),
        fp8: geforceTiling(32),
        int8: geforceTiling(32),
        fp4: geforceTiling(64),
      },
    },
  ],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR7',
    // "32 GB" as quoted, recorded SI (32e9 B). Physically 16 × 16 Gbit = 32 GiB = 34.36e9 B; SI is conservative.
    capacityOptionsBytes: [gb(32)],
    // 512-bit × 28 Gbps / 8 = 1792 GB/s (Table 3).
    bandwidthBytesPerSec: gbPerSec(1792),
  },
  // 98,304 KB with KB = 1024 B (Table 3) = 100,663,296 B. Recorded only; not used by the v1 model.
  l2Bytes: 98304 * 1024,
  sources: [
    RTX_5090_WHITEPAPER,
    {
      url: RTX_BLACKWELL_WHITEPAPER_URL,
      title: 'NVIDIA RTX Blackwell GPU Architecture (whitepaper), GB202 block diagram note (p. 14)',
      fields: ['engines.0.vectorFlopsPerSec.fp64'],
      note: '"The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations." → 104.8 / 64 = 1.6375 TFLOP/s.',
    },
    {
      url: RTX_5090_PRODUCT_URL,
      title: 'NVIDIA GeForce RTX 5090 product page',
      fields: ['memory.capacityOptionsBytes'],
      note: '32 GB GDDR7; 21,760 CUDA cores; "3352 AI TOPS" (= FP4 with 2:4 sparsity, cf. whitepaper Table 3).',
    },
    {
      url: PTX_MMA_SHAPES_URL,
      title: 'PTX ISA: warp-level matrix multiply-accumulate shapes (mma.sync)',
      fields: RTX_5090_TILING_KEYS.flatMap((k) => [
        `engines.0.gemmTiling.${k}.mmaM`,
        `engines.0.gemmTiling.${k}.mmaN`,
        `engines.0.gemmTiling.${k}.mmaK`,
      ]),
      note: 'f16/bf16 m16n8k16; tf32 m16n8k8; e4m3/e5m2 and s8 m16n8k32; block-scaled fp4 (kind::mxf4nvf4) m16n8k64.',
    },
  ],
  estimated: RTX_5090_TILING_KEYS.flatMap((k) => [
    {
      field: `engines.0.gemmTiling.${k}.ctaTiles`,
      reason: 'Typical CUTLASS CTA tile shapes (https://github.com/NVIDIA/cutlass); not a vendor spec.',
    },
    {
      field: `engines.0.gemmTiling.${k}.ctasPerUnit`,
      reason: 'One resident CTA per SM for large tiles (register/shared-memory bound); not a vendor spec.',
    },
  ]),
};

// Ampere / Ada tables retain NVIDIA's rounded rates, not later comparison-table rounding.
const AMPERE_WHITEPAPER_URL =
  'https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/' +
  'NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf';
const ADA_WHITEPAPER_URL =
  'https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf';

/**
 * Make explicitly quoted dense / sparse table entries.
 * Rates are per-device TFLOP/s for floats, TOP/s for integers; sparse counts dense-equivalent operations.
 */
function tensorPair(
  input: MatmulPeak['input'],
  accumulate: MatmulPeak['accumulate'],
  dense: number,
  sparse: number,
): MatmulPeak[] {
  return [
    { input, accumulate, sparsity: 'dense', flopsPerSec: tflops(dense) },
    { input, accumulate, sparsity: '2:4', flopsPerSec: tflops(sparse) },
  ];
}

/**
 * Tiling citations and estimates shared by Ampere / Ada.
 * ISA dimensions are in elements; CTA scheduling and residency are estimates, not hardware guarantees.
 */
function legacyTilingEvidence(keys: ('default' | MathFormat)[]): Pick<DeviceSpec, 'sources' | 'estimated'> {
  return {
    sources: [
      {
        url: PTX_MMA_SHAPES_URL,
        title: 'PTX ISA mma.sync shapes for sm_86 / sm_89',
        fields: keys.flatMap((k) => [
          `engines.0.gemmTiling.${k}.mmaM`,
          `engines.0.gemmTiling.${k}.mmaN`,
          `engines.0.gemmTiling.${k}.mmaK`,
        ]),
        note:
          'f16/bf16 m16n8k16; tf32 m16n8k8; s8 and Ada fp8 m16n8k32; s4 m16n8k64. ' +
          'Dense instruction shapes.',
      },
    ],
    estimated: keys.flatMap((k) => [
      {
        field: `engines.0.gemmTiling.${k}.ctaTiles`,
        reason:
          'Representative CUTLASS CTA tiles (https://github.com/NVIDIA/cutlass), ' +
          'not a hardware specification.',
      },
      {
        field: `engines.0.gemmTiling.${k}.ctasPerUnit`,
        reason: 'One resident large CTA per SM, register/shared-memory constrained; kernel-dependent estimate.',
      },
    ]),
  };
}

const AMPERE_TILING_EVIDENCE = legacyTilingEvidence(['default', 'fp16', 'bf16', 'tf32', 'int8', 'int4']);
const ADA_TILING_EVIDENCE = legacyTilingEvidence(['default', 'fp16', 'bf16', 'tf32', 'fp8', 'int8', 'int4']);

/**
 * GeForce RTX 3090 (GA102, sm_86), 1695 MHz boost. Appendix A Table 9's rounded rates are retained.
 * FP16 with FP32 accumulation is half FP16 accumulation; sparse rows are dense-equivalent 2:4 rates.
 */
export const RTX_3090: DeviceSpec = {
  id: 'rtx-3090',
  name: 'NVIDIA GeForce RTX 3090',
  vendor: 'NVIDIA',
  engines: [
    {
      id: 'gpu',
      name: 'GA102 3rd-gen Tensor Cores (sm_86)',
      kind: 'gpu',
      computeUnits: 82,
      matmulPeaks: [
        ...tensorPair('fp16', 'fp16', 142, 284),
        ...tensorPair('fp16', 'fp32', 71, 142),
        ...tensorPair('bf16', 'fp32', 71, 142),
        ...tensorPair('tf32', 'fp32', 35.6, 71),
        ...tensorPair('int8', 'int32', 284, 568),
        ...tensorPair('int4', 'int32', 568, 1136),
      ],
      vectorFlopsPerSec: {
        fp32: tflops(35.6),
        fp16: tflops(35.6),
        bf16: tflops(35.6),
        // Figure 2: FP64 = FP32 / 64; vector FP16/BF16 are not tensor peaks.
        fp64: tflops(35.6 / 64),
      },
      gemmTiling: {
        default: geforceTiling(16),
        fp16: geforceTiling(16),
        bf16: geforceTiling(16),
        tf32: geforceTiling(8),
        int8: geforceTiling(32),
        int4: geforceTiling(64),
      },
    },
  ],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6X',
    // 24576 MB (binary in table) = physical 24 GiB; record marketed 24 GB in SI, like other presets.
    capacityOptionsBytes: [gb(24)],
    // 384 bits × 19.5 Gbit/s / 8 = 936 GB/s.
    bandwidthBytesPerSec: gbPerSec(936),
  },
  l2Bytes: 6144 * 1024,
  sources: [
    {
      url: AMPERE_WHITEPAPER_URL,
      title: 'NVIDIA Ampere GA102 whitepaper, Appendix A Table 9 (pp. 44–45), Figure 2',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec',
        'memory',
        'l2Bytes',
      ],
      note:
        '1695 MHz; 82 SMs; non-tensor FP32/FP16/BF16 35.6 TFLOPS; FP64 1/64 FP32. ' +
        'Dense/sparse: FP16-acc 142/284, FP32-acc FP16/BF16 71/142, TF32 35.6/71, ' +
        'INT8 284/568, INT4 568/1136. 24576 MB GDDR6X (marketed 24 GB), ' +
        '384-bit 19.5 Gbps, 936 GB/s; L2 6144 binary KB.',
    },
    ...AMPERE_TILING_EVIDENCE.sources,
  ],
  estimated: AMPERE_TILING_EVIDENCE.estimated,
};

/**
 * GeForce RTX 4090 (AD102, sm_89), 2520 MHz boost. Ada Table 2 v2.02 includes corrected FP8→FP32 rates.
 * GeForce FP32 accumulation halves FP16/FP8 throughput. Table rounding is deliberately preserved.
 */
export const RTX_4090: DeviceSpec = {
  id: 'rtx-4090',
  name: 'NVIDIA GeForce RTX 4090',
  vendor: 'NVIDIA',
  engines: [
    {
      id: 'gpu',
      name: 'AD102 4th-gen Tensor Cores (sm_89)',
      kind: 'gpu',
      computeUnits: 128,
      matmulPeaks: [
        ...tensorPair('fp16', 'fp16', 330.3, 660.6),
        ...tensorPair('fp16', 'fp32', 165.2, 330.4),
        ...tensorPair('bf16', 'fp32', 165.2, 330.4),
        ...tensorPair('tf32', 'fp32', 82.6, 165.2),
        ...tensorPair('fp8', 'fp16', 660.6, 1321.2),
        ...tensorPair('fp8', 'fp32', 330.3, 660.6),
        ...tensorPair('int8', 'int32', 660.6, 1321.2),
        ...tensorPair('int4', 'int32', 1321.2, 2642.4),
      ],
      vectorFlopsPerSec: {
        fp32: tflops(82.6),
        fp16: tflops(82.6),
        bf16: tflops(82.6),
        // Figure 1: FP64 = FP32 / 64.
        fp64: tflops(82.6 / 64),
      },
      gemmTiling: {
        default: geforceTiling(16),
        fp16: geforceTiling(16),
        bf16: geforceTiling(16),
        tf32: geforceTiling(8),
        fp8: geforceTiling(32),
        int8: geforceTiling(32),
        int4: geforceTiling(64),
      },
    },
  ],
  memory: {
    kind: 'dedicated',
    technology: 'GDDR6X',
    capacityOptionsBytes: [gb(24)],
    // 384 bits × 21 Gbit/s / 8 = 1008 GB/s.
    bandwidthBytesPerSec: gbPerSec(1008),
  },
  l2Bytes: 73728 * 1024,
  sources: [
    {
      url: ADA_WHITEPAPER_URL,
      title: 'NVIDIA Ada Lovelace whitepaper, Appendix A Table 2 (pp. 29–31), Figure 1',
      fields: [
        'engines.0.computeUnits',
        'engines.0.matmulPeaks',
        'engines.0.vectorFlopsPerSec',
        'memory',
        'l2Bytes',
      ],
      note:
        '2520 MHz; 128 SMs; vector FP32/FP16/BF16 82.6 TFLOPS; FP64 1/64 FP32. ' +
        'Dense/sparse: FP16-acc 330.3/660.6, FP32-acc FP16/BF16 165.2/330.4, TF32 82.6/165.2, ' +
        'FP8 FP16-acc 660.6/1321.2, FP8 FP32-acc 330.3/660.6 (v2.02 corrected), ' +
        'INT8 660.6/1321.2, INT4 1321.2/2642.4. 24 GB GDDR6X, 384-bit 21 Gbps, 1008 GB/s; ' +
        'L2 73728 binary KB.',
    },
    ...ADA_TILING_EVIDENCE.sources,
  ],
  estimated: ADA_TILING_EVIDENCE.estimated,
};

/**
 * All stock NVIDIA consumer devices, per-device peaks at vendor boost clocks. Unlocks are separate axes. Display
 * order: newest generation first (Blackwell, Ada, Ampere).
 */
export const NVIDIA_CONSUMER_DEVICE_PRESETS: DeviceSpec[] = [RTX_5090, RTX_4090, RTX_3090];

/** Community project: Ampere stable, Ada experimental; Blackwell explicitly unsupported. */
const TENSOR_UNLOCK_SOURCE: SourceRef = {
  url: 'https://github.com/Zagitta/gpu-unlock',
  title: 'gpu-unlock compatibility and tensor issue-rate measurements',
  kind: 'community',
  fields: ['engines.0.matmulPeaks'],
  note:
    'RTX 3090 stable (100+ reports), BF16 75→125 TFLOPS at stock power; ' +
    'RTX 4090 experimental, BF16 peak MMA 165→329 TFLOPS. ' +
    'FP16/BF16/TF32 with FP32 accumulate issue rate ×2; FP8/integer unchanged. Blackwell unsupported.',
};

/**
 * Independent tensor-rate axis for supported generations, with dimensionless issue-rate factor ×2.
 * The flag labels Ada support experimental; this does not promise 2× sustained GEMM speed.
 */
function tensorUnlock(experimental: boolean): UnlockAxis {
  return {
    id: 'tensor-rate',
    name: 'Tensor-rate unlock',
    kind: 'tensorRate',
    description:
      'Zagitta gpu-unlock doubles FP16/BF16/TF32 FP32-acc tensor issue rate using community-measured settings. ' +
      (experimental ? 'RTX 4090 support is experimental. ' : 'RTX 3090 support is listed stable. ') +
      'Unsupported by NVIDIA; power limits can reduce sustained gains.',
    options: [
      {
        id: 'full-rate',
        name: 'Full FP32-acc tensor issue rate (×2)',
        deviceModifiers: (['fp16', 'bf16', 'tf32'] as const).map((input) => ({
          kind: 'peakScale',
          engineId: 'gpu',
          input,
          accumulate: 'fp32',
          factor: 2,
          note:
            `Community gpu-unlock: ${input} with FP32 accumulate ×2, dense and sparse; ` +
            'vendor-rounded rates retained, not corrected to unrounded clocks. ' +
            'Not a sustained-performance guarantee.',
          sources: [TENSOR_UNLOCK_SOURCE],
        })),
      },
    ],
  };
}

/**
 * Enable PCIe driver P2P, leaving bandwidth (B/s) and latency (seconds) on the selected topology.
 * Ampere selects the newer fork with explicit RTX 3090 support; copy benchmarks are host-specific.
 */
function p2pUnlock(ampere: boolean): UnlockAxis {
  const source: SourceRef = {
    url: ampere
      ? 'https://github.com/YangWang92/p2p-open-gpu-kernel-modules'
      : 'https://github.com/tinygrad/open-gpu-kernel-modules/tree/570.148.08-p2p',
    title: 'Community NVIDIA open kernel driver PCIe P2P patch',
    kind: 'community',
    fields: ['p2p'],
    note: ampere
      ? '610-series fork lists RTX 3090/4090/5090; 3090 Gen4 direct copies measured 23–26 GB/s. ' +
        'Full-framebuffer BAR1 and matching drivers required.'
      : '570.148.08-p2p lists RTX 4090/5090. Requires compatible host topology and large BAR mappings; ' +
        'enabled state does not guarantee any universal copy rate.',
  };
  return {
    id: 'pcie-p2p',
    name: 'Patched PCIe P2P driver',
    kind: 'interconnect',
    description:
      'YangWang92/tinygrad patched drivers enable direct PCIe peer copies, verified by community measurements. ' +
      'Requires full-framebuffer BAR1, compatible host/IOMMU settings and matching drivers; unsupported by NVIDIA. ' +
      'Keeps the selected link bandwidth and latency because copy rates are host-specific.',
    options: [
      {
        id: 'direct',
        name: 'Direct peer copies',
        deviceModifiers: [],
        intraNodeLinkModifiers: [
          {
            kind: 'p2p',
            p2p: 'direct',
            note:
              'Community driver patch removes host bouncing only; ' +
              'no host-specific copy benchmark is treated as a universal link peak.',
            sources: [source],
          },
        ],
      },
    ],
  };
}

/** Independent RTX 3090 community unlocks; vendor NVLink bridge is a separate topology, not an unlock. */
export const RTX_3090_UNLOCKS: UnlockAxis[] = [tensorUnlock(false), p2pUnlock(true)];

/** Independent RTX 4090 community/review axes; the VRAM option assumes a full AD102-300 donor, not 4090D. */
export const RTX_4090_UNLOCKS: UnlockAxis[] = [
  tensorUnlock(true),
  {
    id: 'vram-mod',
    name: '48 GB custom-PCB VRAM mod',
    kind: 'vram',
    description:
      'GPU Lab and teardown reports document a 48 GB custom-PCB mod using a full RTX 4090 donor, not a 4090D. ' +
      'Capacity is community-measured; reliability, warranty and memory clocks depend on the board. ' +
      'This option changes capacity only, since exact bandwidth is unverified.',
    options: [
      {
        id: '48gb',
        name: '48 GB (full RTX 4090 donor)',
        deviceModifiers: [
          {
            kind: 'memoryCapacity',
            capacityOptionsBytes: [gb(48)],
            note:
              '24×2 GB clamshell custom PCB, marketed 48 GB recorded SI. ' +
              'Capacity only: exact board memory clock/bandwidth not established.',
            sources: [
              {
                url: 'https://www.hardware-corner.net/48gb-rtx-4090-first-tests/',
                title: '48 GB RTX 4090 teardown report',
                kind: 'review',
                fields: ['memory.capacityOptionsBytes'],
                note:
                  '24×2 GB GDDR6X, 12 modules per side, 384-bit retained; ' +
                  'similarly clocked memory and ~1 TB/s (not an exact clock measurement).',
              },
              {
                url: 'https://gpulab.net/48gb-pcb',
                title: 'GPU Lab 48 GB custom PCB',
                kind: 'community',
                fields: ['memory.capacityOptionsBytes'],
                note:
                  'Gen3 PG139-derived PCB, full AD102-300-A1 donor, ' +
                  'transplant 12 original plus 12 additional modules. ' +
                  'Seller claims 900 GB/s for both stock and mod, insufficient evidence to alter bandwidth.',
              },
            ],
          },
        ],
      },
    ],
  },
  p2pUnlock(false),
  {
    id: 'memory-overclock',
    name: 'Sample memory overclock',
    kind: 'memoryClock',
    description:
      'TechPowerUp community/review measurements reached 1545 MHz physical memory clock on one RTX 4090 FE. ' +
      'The option derives a theoretical bus peak, not a measured bandwidth gain. ' +
      'Stability and error retries vary by sample, including on modified VRAM boards.',
    options: [
      {
        id: '1545mhz',
        name: '1545 MHz physical (24.72 Gbit/s)',
        deviceModifiers: [
          {
            kind: 'memoryBandwidth',
            // 1545 MHz × 16 transfers/clock × 384 bits / 8 = 1186.56 GB/s; stock 1008 GB/s.
            bandwidthBytesPerSec: gbPerSec(1545 * 16 / 1000 * 384 / 8),
            note:
              'Review-sample memory OC: 1545 MHz×16 = 24.72 Gbit/s; ' +
              '384-bit/8 gives 1186.56 GB/s theoretical bus peak, not measured throughput. ' +
              'Validate stability on your board.',
            sources: [
              {
                url: 'https://www.techpowerup.com/review/nvidia-geforce-rtx-4090-founders-edition/41.html',
                title: 'TechPowerUp RTX 4090 FE overclocking',
                kind: 'review',
                fields: ['memory.bandwidthBytesPerSec'],
                note:
                  'Maximum overclock table: 1545 MHz memory; ' +
                  '3018 MHz average GPU clock measured in graphics, not tensor load. ' +
                  'Only the memory clock is used here.',
              },
            ],
          },
        ],
      },
    ],
  },
];

/** Vendor boost baseline in MHz: RTX Blackwell whitepaper Appendix A Table 3 (RTX_5090_WHITEPAPER). */
const RTX_5090_BOOST_CLOCK_MHZ = 2407;
/** Community sustained SM clock in MHz at the 575 W stock limit on one Palit board, Unsloth SFT load. */
const RTX_5090_575W_SUSTAINED_CLOCK_MHZ = 2768;
/** Community sustained SM clock in MHz at a 400 W limit on the same board and Unsloth SFT load. */
const RTX_5090_400W_SUSTAINED_CLOCK_MHZ = 2465;

/** One-board sustained-clock measurements, not universal power-to-performance multipliers. */
const RTX_5090_SUSTAINED_CLOCK_SOURCE: SourceRef = {
  url: 'https://zenn.dev/holy_fox/articles/572083c130ce46?locale=en',
  title: 'RTX 5090 575 W versus 400 W AI-workload study',
  kind: 'community',
  fields: ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec'],
  note:
    'Palit GameRock repeated Unsloth SFT thermal test: average SM clock 2768 MHz at 575 W, ' +
    '2465 MHz at 400 W. Compute scaling uses each measured clock divided by the vendor Table 3 ' +
    '2407 MHz boost baseline; power fraction is not compute fraction.',
};

/** Independent RTX 5090 community axes. No tensor-rate or unverified large-VRAM unlock is advertised. */
export const RTX_5090_UNLOCKS: UnlockAxis[] = [
  p2pUnlock(false),
  {
    id: 'power-clock',
    name: 'Measured sustained clock',
    kind: 'powerClock',
    description:
      'Community clocks measured on one Palit board under an Unsloth SFT load, not guaranteed settings. ' +
      'Results are workload- and cooling-dependent; GPU Boost commonly runs above the rated boost clock. ' +
      'Both options scale compute from the vendor 2407 MHz baseline, leaving memory bandwidth unchanged.',
    options: [
      {
        id: '575w',
        name: '575 W (stock limit): sustained 2768 MHz',
        deviceModifiers: [
          {
            kind: 'clockScale',
            engineId: 'gpu',
            // FLOP/s at measured clock = vendor FLOP/s × measured MHz / vendor boost MHz.
            factor: RTX_5090_575W_SUSTAINED_CLOCK_MHZ / RTX_5090_BOOST_CLOCK_MHZ,
            note:
              'Community Unsloth SFT sustained clock: 2768/2407 ≈1.150 compute scaling at 575 W. ' +
              'Relative to vendor boost-based peaks; sample-specific, not guaranteed sustained performance.',
            sources: [RTX_5090_SUSTAINED_CLOCK_SOURCE],
          },
        ],
      },
      {
        id: '400w',
        name: '400 W limit: sustained 2465 MHz',
        deviceModifiers: [
          {
            kind: 'clockScale',
            engineId: 'gpu',
            // Use vendor boost, not the measured 575 W clock, as the base preset's denominator.
            factor: RTX_5090_400W_SUSTAINED_CLOCK_MHZ / RTX_5090_BOOST_CLOCK_MHZ,
            note:
              'Community Unsloth SFT sustained clock: 2465/2407 ≈1.024 compute scaling at 400 W. ' +
              'Relative to vendor boost-based peaks; sample-specific, not guaranteed sustained performance.',
            sources: [RTX_5090_SUSTAINED_CLOCK_SOURCE],
          },
        ],
      },
    ],
  },
];

// Rejected, not UI choices: 3090 48 GB has no verified capacity test (VBIOS/addressing remains unresolved):
// https://www.reddit.com/r/LocalLLaMA/comments/16lji25/3090_48gb/
// 4090 96 GB report explicitly says testing/claims, not demonstrated hardware:
// https://www.emsi.me/tech/ai-ml/nvidia-rtx-4090-modded-to-96gb-vram-for-ai-extreme-power-or-risky-experiment/
// 2025-03-06/223a41 (continuation of the URL above).
// 5090 64/96 GB listings lack independent full-capacity tests and contain conflicting memory specifications:
// https://www.guru3d.com/story/modified-geforce-rtx-5090-with-96gb-appears-at-but-listing-contains-conflicting-
// specifications/ (continuation of the URL above).
// 5090 128 GB is debunked, not a working-board proof:
// https://hardwarecanucks.com/forum/threads/modded-rtx-5090-with-128gb-of-vram-debunked.88884/
// Blackwell tensor unlock is explicitly unsupported: https://github.com/Zagitta/gpu-unlock
// No 3090/4090 numeric power axis: measured gaming clocks/FPS do not establish sustained tensor clocks.
