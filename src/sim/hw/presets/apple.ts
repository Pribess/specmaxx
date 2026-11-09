/**
 * Apple GPU presets, per SoC (both Ultra dies together). See docs/simulation-model.md §4.
 * Apple quotes memory bandwidth, not GPU clocks or FLOPS: all compute rates below are estimates derived
 * from community architecture measurements and measured/assumed clocks, not vendor performance claims.
 *
 * M5 GPU Neural Accelerators are Metal/MPP-accessible matrix units, NOT the separate Neural Engine (ANE).
 * ANE is excluded: Core ML-only/static-shape access and a restricted DRAM path make an uncapped engine
 * misleading. Measurements: https://arxiv.org/html/2606.22283 and
 * https://github.com/cadamcat/llms-on-apple-neural-engine (GPU faster for tested LLM prefill/decode).
 * No M4 Ultra exists. No sparse, FP8/FP4 or INT4 hardware peak is claimed; compressed weight storage is
 * not a faster arithmetic dtype. Unknown GPU cache sizes are omitted. Capacities use Apple's quoted GB
 * in SI bytes; the OS also occupies this shared pool. Engines are alternatives, never additive.
 */

import type { ComputeEngine, DeviceSpec, EstimatedField, GemmTiling, SourceRef } from '../types';
import { gb, gbPerSec, tbPerSec } from '../units';
import { APPLE_GPU_SFU, engineSfu, sfuOpsPerSec } from './sfu';

const STUDIO_2025_URL = 'https://support.apple.com/en-us/122211';
const STUDIO_2026_URL = 'https://support.apple.com/en-us/128107';
const MBP14_M3_URL = 'https://support.apple.com/en-us/117736';
const MBP16_M3_URL = 'https://support.apple.com/en-us/117737';
const MSL_URL = 'https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf';
const DVFS_URL = 'https://github.com/maderix/apple-gpu-dvfs';
const ARCH_URL = 'https://tzakharko.github.io/apple-neural-accelerators-benchmark/';
const SIMD_URL = 'https://stonesjtu.github.io/2026/09/09/apple-gpu-evolution/';
const M5_MATMUL_URL = 'https://gist.github.com/wanderingmeow/6562adbbecf6426ebfce8ab05a83b05f';
const METAL_TECH_TALK_URL = 'https://developer.apple.com/videos/play/tech-talks/10580/';
const MLX_WIRED_LIMIT_URL =
  'https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html';
const LLAMACPP_WIRED_LIMIT_URL = 'https://github.com/ggml-org/llama.cpp/discussions/2182';

/**
 * Default GPU-usable fraction of a Mac's unified memory (`MemorySpec.gpuUsableFraction`): macOS lets the GPU wire
 * about 3/4 of RAM on Macs with 36 GB or more (`recommendedMaxWorkingSetSize`; Apple: 48 GB of 64 GB) and about
 * 2/3 on smaller ones. Every Apple preset here has ≥ 36 GB, so 0.75 applies to all of them. Raisable with
 * `sudo sysctl iogpu.wired_limit_mb=<MB>` (the override, `HardwareStack.gpuUsableFraction`). Dimensionless.
 */
export const MACOS_GPU_USABLE_FRACTION = 0.75;

/** Citations of {@link MACOS_GPU_USABLE_FRACTION}, shared by every Apple preset. */
const MACOS_WIRED_LIMIT_SOURCES: SourceRef[] = [
  {
    url: METAL_TECH_TALK_URL,
    title: 'Apple tech talk: Metal Compute on MacBook Pro (recommendedMaxWorkingSetSize)',
    fields: ['memory.gpuUsableFraction'],
    note:
      'The GPU working set is limited to recommendedMaxWorkingSetSize: 21 GB of a 32 GB Mac and 48 GB of a 64 GB ' +
      'Mac (3/4).',
  },
  {
    url: MLX_WIRED_LIMIT_URL,
    title: 'MLX documentation: mlx.core.set_wired_limit',
    fields: ['memory.gpuUsableFraction'],
    note:
      '"Setting a wired limit larger than system wired limit is an error. You can increase the system wired limit ' +
      'with: sudo sysctl iogpu.wired_limit_mb=<size_in_megabytes>"; the system limit is reported as ' +
      '"max_recommended_working_set_size".',
  },
  {
    url: LLAMACPP_WIRED_LIMIT_URL,
    title: 'llama.cpp discussion #2182: Adjust VRAM/RAM split on Apple Silicon',
    fields: ['memory.gpuUsableFraction'],
    kind: 'community',
    note:
      'Reported defaults: about 2/3 of RAM up to 32 GB and 3/4 above; iogpu.wired_limit_mb raises it until reboot. ' +
      'Some recent macOS releases report a larger share on large pools; set it under Advanced if yours does.',
  },
];

// Community DVFS tables: M2 Max 1398 MHz, used as an ESTIMATE for the unmeasured M3 Max and M3 Ultra (same GPU generation).
const M3_GPU_CLOCK_HZ = 1.398e9;
const M3_ULTRA_GPU_CLOCK_HZ = M3_GPU_CLOCK_HZ;
// Community base-M4 device-tree P15 = 1578 MHz; extrapolated to M4 Max, not directly measured on Max.
const M4_MAX_GPU_CLOCK_HZ = 1.578e9;
// Community M5 Max powermetrics: uncapped 1620 MHz; DVFS_URL's GFLOPS column is explicitly invalid.
const M5_MAX_GPU_CLOCK_HZ = 1.62e9;
// No Ultra clock measurement: assume the Max clock (not guaranteed under sustained load).
const M5_ULTRA_GPU_CLOCK_HZ = M5_MAX_GPU_CLOCK_HZ;
// 4 partitions/core × 32 lanes × 2 FLOP/FMA (ARCH_URL, philipturner/metal-benchmarks).
const SIMD_FP32_FLOP_PER_CLOCK_PER_CORE = 4 * 32 * 2;
// Reverse-engineered Apple10: 512 FP16 FMA/core/clock = 1024 FLOP/core/clock (ARCH_URL).
const M5_NA_FP16_FLOP_PER_CLOCK_PER_CORE = 1024;
// Extrapolated INT8: 2 × FP16, consistent with A19 ~1900 vs nominal 2048 OP/core/clock and M5 Max 119 TOPS.
const M5_NA_INT8_OP_PER_CLOCK_PER_CORE = 2 * M5_NA_FP16_FLOP_PER_CLOCK_PER_CORE;
/**
 * GEMV path row limit (`ComputeEngine.gemvMaxRows`): GEMMs with m ≤ 1 token run as SIMD dot products, no 8×8 MMA.
 * Conservative: m = 1 is the vector path in both MLX and llama.cpp Metal for every weight format; larger limits are
 * format- and chip-specific (MLX affine 4/8-bit qmv up to m 3 on M3/M4 and 5 on M5; llama.cpp Q4_0 / Q8_0 m 1).
 */
const APPLE_GEMV_MAX_ROWS = 1;
/** MLX source of the vector (non-MMA) GEMV dispatch (main, commit fc344c98, 2026-11-06). */
const MLX_GEMV_URL = 'https://github.com/ml-explore/mlx/blob/fc344c98/mlx/backend/metal/matmul.cpp';
/** llama.cpp Metal mat-vec vs mat-mat dispatch (master, commit a0ad695, 2026-11-09). */
const LLAMACPP_MUL_MV_URL = 'https://github.com/ggml-org/llama.cpp/blob/a0ad695/ggml/src/ggml-metal/ggml-metal-common.cpp';

/** SIMD-group MMA dimensions in elements; CTA shapes and residency remain estimates. */
function simdTiling(): GemmTiling {
  return {
    mmaM: 8,
    mmaN: 8,
    mmaK: 8,
    ctaTiles: [
      {
        m: 64,
        n: 64,
      },
      {
        m: 32,
        n: 32,
      },
    ],
    ctasPerUnit: 1,
  };
}

/** NAX native instruction shape is unpublished: 8³ is a proxy, not a vendor instruction claim. */
function neuralTiling(): GemmTiling {
  return {
    mmaM: 8,
    mmaN: 8,
    mmaK: 8,
    // M5 Max MPP benchmark's best threadgroup tile, four SIMD groups (M5_MATMUL_URL).
    ctaTiles: [
      {
        m: 128,
        n: 64,
      },
    ],
    ctasPerUnit: 1,
  };
}

/** GPU ALU path. FLOP/s = GPU cores × FLOP/core/clock × assumed clock in Hz. */
function simdEngine(cores: number, clockHz: number, isM5: boolean): ComputeEngine {
  const fp32FlopsPerSec = cores * SIMD_FP32_FLOP_PER_CLOCK_PER_CORE * clockHz;
  // Apple7–9 has equal rates; Apple10 can issue two half-precision instructions concurrently.
  const halfFlopsPerSec = fp32FlopsPerSec * (isM5 ? 2 : 1);
  return {
    id: isM5 ? 'gpu-simd' : 'gpu',
    name: 'GPU SIMD ALUs (no tensor acceleration)',
    kind: 'gpu',
    computeUnits: cores,
    matmulPeaks: [
      {
        input: 'fp16',
        accumulate: 'fp32',
        sparsity: 'dense',
        flopsPerSec: halfFlopsPerSec,
      },
      {
        input: 'bf16',
        accumulate: 'fp32',
        sparsity: 'dense',
        flopsPerSec: halfFlopsPerSec,
      },
      {
        input: 'fp32',
        accumulate: 'fp32',
        sparsity: 'dense',
        flopsPerSec: fp32FlopsPerSec,
      },
    ],
    vectorFlopsPerSec: {
      fp32: fp32FlopsPerSec,
      fp16: halfFlopsPerSec,
      bf16: halfFlopsPerSec,
    },
    // exp2 on the SIMD ALUs: cores × 32 × clock (estimate, sfu.ts APPLE_GPU_SFU).
    sfuOpsPerSec: sfuOpsPerSec(cores, APPLE_GPU_SFU.exp2PerClockPerUnit, clockHz),
    gemvMaxRows: APPLE_GEMV_MAX_ROWS,
    gemmTiling: {
      default: simdTiling(),
    },
  };
}

/** M5 GPU matrix path; non-matrix ops still use this GPU's SIMD ALUs. */
function neuralEngine(cores: number, clockHz: number): ComputeEngine {
  const vectorEngine = simdEngine(cores, clockHz, true);
  // E.g. full M5 Max: 40 cores × 1024 FLOP/clock/core × 1.62e9 Hz = 66.3552e12 FLOP/s.
  const halfFlopsPerSec = cores * M5_NA_FP16_FLOP_PER_CLOCK_PER_CORE * clockHz;
  const int8OpsPerSec = cores * M5_NA_INT8_OP_PER_CLOCK_PER_CORE * clockHz;
  return {
    id: 'gpu-na',
    name: 'GPU Neural Accelerators (Metal tensor ops)',
    kind: 'gpu',
    computeUnits: cores,
    matmulPeaks: [
      {
        input: 'fp16',
        accumulate: 'fp32',
        sparsity: 'dense',
        flopsPerSec: halfFlopsPerSec,
      },
      {
        input: 'bf16',
        accumulate: 'fp32',
        sparsity: 'dense',
        flopsPerSec: halfFlopsPerSec,
      },
      {
        input: 'int8',
        accumulate: 'int32',
        sparsity: 'dense',
        flopsPerSec: int8OpsPerSec,
      },
      // FP32 has no NAX acceleration: peak lookup falls back to the vector FP32 rate.
    ],
    vectorFlopsPerSec: vectorEngine.vectorFlopsPerSec,
    // Softmax exp and GEMV (m ≤ 1) run on the same SIMD ALUs as the vector engine: NAX at m = 1 would pad to its
    // MMA shape (1024/8 FLOP/core/clk) below the SIMD FP16 512.
    sfuOpsPerSec: vectorEngine.sfuOpsPerSec,
    gemvMaxRows: APPLE_GEMV_MAX_ROWS,
    gemmTiling: {
      default: neuralTiling(),
      fp32: simdTiling(),
    },
  };
}

/** Named SKU facts. Physical rates are per SoC, including both dies on Ultra. */
interface AppleDeviceParams {
  /** Apple chip name (display text). */
  chip: string;
  /** Enabled GPU cores, count; one compute unit per core. */
  gpuCores: number;
  /** Assumed GPU clock in Hz, used for every derived compute peak. */
  gpuClockHz: number;
  /** Unified-memory configurations in Apple's quoted decimal GB; first is the default. */
  capacityOptionsGb: number[];
  /** Quoted shared DRAM bandwidth, read + write combined, in B/s. */
  memBandwidthBytesPerSec: number;
  /** Whether Metal-accessible GPU Neural Accelerators exist; unrelated to the separate ANE. */
  hasNeuralAccelerators: boolean;
  /** Explanation of the clock measurement or extrapolation, including uncertainty. */
  clockNote: string;
  /** Quoted vendor memory/core facts for this SKU. */
  memoryNote: string;
  /** Additional SKU-specific citations, with field paths relative to the complete device. */
  additionalSources?: SourceRef[];
  /**
   * Apple tech-specs page that quotes the memory options, bandwidth and GPU cores. Default: the Mac Studio page of
   * the chip's generation; the M3 Max exists only in the MacBook Pro, so it names that page.
   */
  specPage?: { url: string; title: string };
}

/** Assemble one SKU; SIMD and Neural Accelerator provenance remain separate and explicit. */
function appleDevice(params: AppleDeviceParams): DeviceSpec {
  const {
    chip,
    gpuCores,
    gpuClockHz,
    capacityOptionsGb,
    memBandwidthBytesPerSec,
    hasNeuralAccelerators,
    clockNote,
    memoryNote,
    additionalSources = [],
    specPage,
  } = params;
  const simd = simdEngine(gpuCores, gpuClockHz, hasNeuralAccelerators);
  const simdRoot = hasNeuralAccelerators ? 'engines.1' : 'engines.0';
  const simdTilingPath = `${simdRoot}.gemmTiling.default`;
  const simdComputeFields = [`${simdRoot}.matmulPeaks`, `${simdRoot}.vectorFlopsPerSec`];
  // Every engine (SIMD, and NAX on M5) runs softmax exp and the GEMV path on the SIMD ALUs.
  const engineRoots = hasNeuralAccelerators ? ['engines.0', 'engines.1'] : ['engines.0'];
  const gemvFields = engineRoots.map((root) => `${root}.gemvMaxRows`);
  const sfus = engineRoots.map((root) => engineSfu(APPLE_GPU_SFU, gpuCores, 'GPU cores', gpuClockHz, `${root}.sfuOpsPerSec`));
  const fp32Derivation =
    `${gpuCores} GPU cores × 256 FLOP/core/clock × ${gpuClockHz} Hz = SIMD FP32 FLOP/s. `;
  let halfDerivation = 'FP16/BF16 vector and matmul rate equals the FP32 SIMD rate. ';
  if (hasNeuralAccelerators) {
    halfDerivation = 'M5 FP16/BF16 SIMD rate assumed 2× FP32 (concurrent half-precision issue). ';
  }

  const simdSources: SourceRef[] = [
    {
      url: DVFS_URL,
      title: 'Community Apple GPU DVFS tables and powermetrics',
      kind: 'community',
      fields: simdComputeFields,
      note: clockNote,
    },
    {
      url: SIMD_URL,
      title: 'Community Apple GPU SIMD architecture/throughput analysis',
      kind: 'community',
      fields: simdComputeFields,
      note: '128 FMA lanes/core × 2 FLOP/FMA = 256 FP32 FLOP/core/clock. ' + halfDerivation,
    },
    {
      url: MSL_URL,
      title: 'Metal Shading Language specification, SIMD-group matrix types (§2.4)',
      fields: [
        `${simdTilingPath}.mmaM`,
        `${simdTilingPath}.mmaN`,
        `${simdTilingPath}.mmaK`,
      ],
      note: 'simdgroup_matrix<T,8,8>: 8×8 matrix multiply-accumulate has K=8; half/bfloat/float types.',
    },
    {
      url: MLX_GEMV_URL,
      title: 'MLX Metal backend: matmul.cpp GEMV dispatch (Apple ml-explore/mlx, commit fc344c98)',
      kind: 'community',
      fields: gemvFields,
      note:
        'Dense fp16/bf16 GEMMs with min(M, N) == 1 run the SIMD `gemv` kernel (matmul.cpp:1647), not the ' +
        'simdgroup-MMA GEMM; quantized decode uses `qmv` (quantized.cpp:648–650, vector path up to m 3 on M3/M4 ' +
        'affine 4/8-bit). Kernel code, not a hardware spec: m = 1 is the limit every format shares.',
    },
    {
      url: LLAMACPP_MUL_MV_URL,
      title: 'llama.cpp Metal: ggml_metal_mul_mv_mma_rows_min (commit a0ad695)',
      kind: 'community',
      fields: gemvFields,
      note: 'ggml-metal-common.cpp:121–140: pure mat-vec (no simdgroup MMA) for m = 1 on Q4_0 / Q8_0 / Q6_K, m ≤ 2 F16 / Q4_K / MXFP4, m ≤ 3 BF16.',
    },
    ...sfus.flatMap((s) => s.sources),
  ];
  const simdEstimated: EstimatedField[] = [
    {
      field: `${simdRoot}.matmulPeaks`,
      reason: fp32Derivation + halfDerivation + clockNote,
    },
    {
      field: `${simdRoot}.vectorFlopsPerSec`,
      reason: fp32Derivation + halfDerivation + clockNote,
    },
    {
      field: `${simdTilingPath}.ctaTiles`,
      reason: 'Assumed 64×64 and 32×32 threadgroup output tiles; not vendor-specified.',
    },
    {
      field: `${simdTilingPath}.ctasPerUnit`,
      reason: 'One resident GEMM threadgroup per GPU core assumed; register/shared-memory use affects occupancy.',
    },
    ...sfus.flatMap((s) => s.estimated),
  ];

  // Empty on M3/M4: no Neural Accelerator performance claim is attached to these devices.
  const neuralSources: SourceRef[] = [];
  const neuralEstimated: EstimatedField[] = [];
  const engines = [simd];
  if (hasNeuralAccelerators) {
    engines.unshift(neuralEngine(gpuCores, gpuClockHz));
    const neuralComputeFields = ['engines.0.matmulPeaks', 'engines.0.vectorFlopsPerSec'];
    const halfNeuralDerivation =
      `${gpuCores} GPU cores × 1024 FP16/BF16 FLOP/core/clock × ${gpuClockHz} Hz. `;
    const int8NeuralDerivation =
      'INT8 extrapolated: cores × 2048 OP/core/clock × clock Hz (2× FP16). ';
    neuralSources.push(
      {
        url: DVFS_URL,
        title: 'Community Apple GPU DVFS tables and powermetrics',
        kind: 'community',
        fields: neuralComputeFields,
        note: clockNote,
      },
      {
        url: ARCH_URL,
        title: 'Community Apple GPU Neural Accelerator architecture measurements',
        kind: 'community',
        fields: ['engines.0.matmulPeaks'],
        note: '512 FP16 FMA/core/clock × 2 FLOP/FMA = 1024 FLOP/core/clock. ' +
          'INT8 nominal 2048 OP/core/clock extrapolated from A19; BF16 matches FP16 on M5 Max.',
      },
      {
        ...simdSources[1],
        fields: ['engines.0.vectorFlopsPerSec'],
      },
      {
        ...simdSources[2],
        fields: [
          'engines.0.gemmTiling.fp32.mmaM',
          'engines.0.gemmTiling.fp32.mmaN',
          'engines.0.gemmTiling.fp32.mmaK',
        ],
      },
      {
        url: M5_MATMUL_URL,
        title: 'M5 Max MPP matmul2d community benchmark',
        kind: 'community',
        fields: ['engines.0.gemmTiling.default.ctaTiles'],
        note: 'Best tile 128×64, four SIMD groups; FP16→FP32 61.2 TFLOPS, BF16 61.0, INT8 119 TOPS. ' +
          'Measurement was power-limited.',
      },
    );
    neuralEstimated.push(
      {
        field: 'engines.0.matmulPeaks',
        reason: halfNeuralDerivation + int8NeuralDerivation + clockNote,
      },
      {
        field: 'engines.0.vectorFlopsPerSec',
        reason: fp32Derivation + halfDerivation + clockNote,
      },
      {
        field: 'engines.0.gemmTiling.default',
        reason: 'Native NAX MMA shape unpublished: 8×8×8 is a proxy, not a hardware fact. ' +
          'Measured 128×64 CTA extrapolated across M5 SKUs; one resident threadgroup/core assumed.',
      },
      {
        field: 'engines.0.gemmTiling.fp32.ctaTiles',
        reason: 'FP32 uses SIMD, with assumed 64×64 and 32×32 threadgroup output tiles.',
      },
      {
        field: 'engines.0.gemmTiling.fp32.ctasPerUnit',
        reason: 'FP32 uses SIMD, with one resident GEMM threadgroup/core assumed.',
      },
    );
  }

  const memorySource: SourceRef = {
    url: specPage?.url ?? (hasNeuralAccelerators ? STUDIO_2026_URL : STUDIO_2025_URL),
    title: specPage?.title ?? 'Apple Mac Studio technical specifications',
    fields: [
      'memory.capacityOptionsBytes',
      'memory.bandwidthBytesPerSec',
      ...engines.map((_, i) => `engines.${i}.computeUnits`),
    ],
    note: memoryNote,
  };
  return {
    id: `apple-${chip.toLowerCase().replaceAll(' ', '-')}-${gpuCores}c`,
    name: `Apple ${chip} (${gpuCores}-core GPU)`,
    vendor: 'Apple',
    engines,
    memory: {
      kind: 'unified',
      // The M3 generation (Max and Ultra) uses LPDDR5; M4 and M5 use LPDDR5X.
      technology: chip.startsWith('M3') ? 'LPDDR5' : 'LPDDR5X',
      capacityOptionsBytes: capacityOptionsGb.map(gb),
      bandwidthBytesPerSec: memBandwidthBytesPerSec,
      // macOS GPU wired limit (≈ 3/4 of RAM at ≥ 36 GB); the 2/3 rule of smaller Macs is not needed here.
      gpuUsableFraction: MACOS_GPU_USABLE_FRACTION,
    },
    sources: [memorySource, ...simdSources, ...neuralSources, ...MACOS_WIRED_LIMIT_SOURCES, ...additionalSources],
    estimated: [...simdEstimated, ...neuralEstimated],
  };
}

const M3_CLOCK_NOTE = 'M3 Ultra clock estimated at 1.398 GHz from M2 Max DVFS; no direct Ultra measurement.';
const M3_MAX_CLOCK_NOTE = 'M3 Max clock estimated at 1.398 GHz from M2 Max DVFS; no direct M3 Max measurement found.';
/** The MacBook Pro (Nov 2023) spec pages: the only machines with an M3 Max. Thunderbolt 4 only (no TB5, so no RDMA). */
const MBP_M3_SPEC_PAGE = { url: MBP14_M3_URL, title: 'Apple MacBook Pro (14-inch, Nov 2023) technical specifications' };
const MBP16_M3_SOURCE: SourceRef = {
  url: MBP16_M3_URL,
  title: 'Apple MacBook Pro (16-inch, Nov 2023) technical specifications',
  fields: ['memory.capacityOptionsBytes', 'memory.bandwidthBytesPerSec'],
  note: 'Same M3 Max options as the 14-inch: 30-core GPU with 36GB or 96GB at 300GB/s; 40-core GPU with 48GB, 64GB or ' +
    '128GB at 400GB/s. Three Thunderbolt 4 ports (40 Gb/s).',
};

/** M3 Max 30-core (14-core CPU): 36/96 GB, 300 GB/s; MacBook Pro only, Thunderbolt 4; SIMD compute estimated at 1.398 GHz. */
export const M3_MAX_30C = appleDevice({
  chip: 'M3 Max',
  gpuCores: 30,
  gpuClockHz: M3_GPU_CLOCK_HZ,
  capacityOptionsGb: [36, 96],
  memBandwidthBytesPerSec: gbPerSec(300),
  hasNeuralAccelerators: false,
  clockNote: M3_MAX_CLOCK_NOTE,
  memoryNote: 'M3 Max with 14-core CPU, 30-core GPU; 36GB or 96GB unified memory; 300GB/s memory bandwidth.',
  specPage: MBP_M3_SPEC_PAGE,
  additionalSources: [MBP16_M3_SOURCE],
});
/** M3 Max 40-core (16-core CPU): 48/64/128 GB, 400 GB/s; MacBook Pro only, Thunderbolt 4; SIMD compute estimated at 1.398 GHz. */
export const M3_MAX_40C = appleDevice({
  chip: 'M3 Max',
  gpuCores: 40,
  gpuClockHz: M3_GPU_CLOCK_HZ,
  capacityOptionsGb: [48, 64, 128],
  memBandwidthBytesPerSec: gbPerSec(400),
  hasNeuralAccelerators: false,
  clockNote: M3_MAX_CLOCK_NOTE,
  memoryNote: 'M3 Max with 16-core CPU, 40-core GPU; 48GB, 64GB or 128GB unified memory; 400GB/s memory bandwidth.',
  specPage: MBP_M3_SPEC_PAGE,
  additionalSources: [MBP16_M3_SOURCE],
});
const M4_CLOCK_NOTE = '1.578 GHz base-M4 device-tree P15 extrapolated to M4 Max, not an Apple clock guarantee.';
const M5_MAX_CLOCK_NOTE =
  '1.62 GHz community M5 Max uncapped powermetrics; workload/cooling dependent, not Apple-published.';
const M5_ULTRA_CLOCK_NOTE = '1.62 GHz assumed from M5 Max; no M5 Ultra GPU clock measurement found.';

/** M4 Max 32-core: 36 GB, 410 GB/s; SIMD compute estimated at 1.578 GHz (base-M4 clock). */
export const M4_MAX_32C = appleDevice({
  chip: 'M4 Max',
  gpuCores: 32,
  gpuClockHz: M4_MAX_GPU_CLOCK_HZ,
  capacityOptionsGb: [36],
  memBandwidthBytesPerSec: gbPerSec(410),
  hasNeuralAccelerators: false,
  clockNote: M4_CLOCK_NOTE,
  memoryNote: '32-core GPU; 36GB unified memory; 410GB/s memory bandwidth.',
});
/** M4 Max 40-core: 48/64/128 GB, 546 GB/s; SIMD compute estimated at 1.578 GHz. */
export const M4_MAX_40C = appleDevice({
  chip: 'M4 Max',
  gpuCores: 40,
  gpuClockHz: M4_MAX_GPU_CLOCK_HZ,
  capacityOptionsGb: [48, 64, 128],
  memBandwidthBytesPerSec: gbPerSec(546),
  hasNeuralAccelerators: false,
  clockNote: M4_CLOCK_NOTE,
  memoryNote: '40-core GPU; 48GB, 64GB or 128GB unified memory; 546GB/s memory bandwidth.',
});
/** M3 Ultra 60-core: 96/256 GB, 819 GB/s; 1.398 GHz assumed from M2 Max, not measured. */
export const M3_ULTRA_60C = appleDevice({
  chip: 'M3 Ultra',
  gpuCores: 60,
  gpuClockHz: M3_ULTRA_GPU_CLOCK_HZ,
  capacityOptionsGb: [96, 256],
  memBandwidthBytesPerSec: gbPerSec(819),
  hasNeuralAccelerators: false,
  clockNote: M3_CLOCK_NOTE,
  memoryNote: '60-core GPU; 96GB configurable to 256GB unified memory; 819GB/s memory bandwidth.',
});
/** M3 Ultra 80-core: 96/256/512 GB, 819 GB/s; 1.398 GHz assumed, no dedicated matrix unit. */
export const M3_ULTRA_80C = appleDevice({
  chip: 'M3 Ultra',
  gpuCores: 80,
  gpuClockHz: M3_ULTRA_GPU_CLOCK_HZ,
  capacityOptionsGb: [96, 256, 512],
  memBandwidthBytesPerSec: gbPerSec(819),
  hasNeuralAccelerators: false,
  clockNote: M3_CLOCK_NOTE,
  memoryNote: '80-core GPU; 96GB, 256GB or 512GB unified memory; 819GB/s. Apple launch release confirms up to 512GB.',
  additionalSources: [
    {
      url: 'https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/',
      title: 'Apple introduces Mac Studio with M3 Ultra',
      fields: ['memory.capacityOptionsBytes'],
      note: 'M3 Ultra starts with 96GB unified memory, configurable up to 512GB (80-core GPU SKU).',
    },
  ],
});

/** M5 Max 32-core: 36 GB, 460 GB/s; NAX default, 1.62 GHz community Max clock; INT8 extrapolated. */
export const M5_MAX_32C = appleDevice({
  chip: 'M5 Max',
  gpuCores: 32,
  gpuClockHz: M5_MAX_GPU_CLOCK_HZ,
  capacityOptionsGb: [36],
  memBandwidthBytesPerSec: gbPerSec(460),
  hasNeuralAccelerators: true,
  clockNote: M5_MAX_CLOCK_NOTE,
  memoryNote: '32-core GPU; 36GB unified memory; 460GB/s memory bandwidth.',
});
/** M5 Max 40-core: 48/64/128 GB, 614 GB/s; NAX default, 1.62 GHz community clock; INT8 extrapolated. */
export const M5_MAX_40C = appleDevice({
  chip: 'M5 Max',
  gpuCores: 40,
  gpuClockHz: M5_MAX_GPU_CLOCK_HZ,
  capacityOptionsGb: [48, 64, 128],
  memBandwidthBytesPerSec: gbPerSec(614),
  hasNeuralAccelerators: true,
  clockNote: M5_MAX_CLOCK_NOTE,
  memoryNote: '40-core GPU; 48GB, 64GB or 128GB unified memory; 614GB/s memory bandwidth.',
});
// Apple quotes 1.2 TB/s: do NOT substitute the estimated 1024-bit × 9600 MT/s / 8 = 1228.8 GB/s.
/** M5 Ultra 64-core: 96/256 GB, quoted 1.2 TB/s; 1.62 GHz assumed from Max; NAX/INT8 extrapolated. */
export const M5_ULTRA_64C = appleDevice({
  chip: 'M5 Ultra',
  gpuCores: 64,
  gpuClockHz: M5_ULTRA_GPU_CLOCK_HZ,
  capacityOptionsGb: [96, 256],
  memBandwidthBytesPerSec: tbPerSec(1.2),
  hasNeuralAccelerators: true,
  clockNote: M5_ULTRA_CLOCK_NOTE,
  memoryNote: '64-core GPU; 96GB or 256GB unified memory; 1.2TB/s memory bandwidth.',
});
/** M5 Ultra 80-core: 96/256/512 GB, quoted 1.2 TB/s; 1.62 GHz assumed; NAX/INT8 extrapolated. */
export const M5_ULTRA_80C = appleDevice({
  chip: 'M5 Ultra',
  gpuCores: 80,
  gpuClockHz: M5_ULTRA_GPU_CLOCK_HZ,
  capacityOptionsGb: [96, 256, 512],
  memBandwidthBytesPerSec: tbPerSec(1.2),
  hasNeuralAccelerators: true,
  clockNote: M5_ULTRA_CLOCK_NOTE,
  memoryNote: '80-core GPU; 96GB, 256GB or 512GB unified memory; 1.2TB/s memory bandwidth.',
});

/**
 * Apple per-SoC GPU SKU library; all rates per device, derived compute explicitly estimated. Display order: newest
 * generation first, then by tier (Ultra before Max), then more GPU cores first. The M3 Ultra (Mac Studio 2025) is
 * newer than the M3 Max (MacBook Pro 2023).
 */
export const APPLE_DEVICE_PRESETS: DeviceSpec[] = [
  M5_ULTRA_80C,
  M5_ULTRA_64C,
  M5_MAX_40C,
  M5_MAX_32C,
  M4_MAX_40C,
  M4_MAX_32C,
  M3_ULTRA_80C,
  M3_ULTRA_60C,
  M3_MAX_40C,
  M3_MAX_30C,
];
