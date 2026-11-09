/**
 * RTX PRO Blackwell workstation peaks, per device at vendor boost (docs/simulation-model.md §4).
 * Traps: full-rate FP32 accumulation unlike GeForce; only fp32-accumulate float entries are needed.
 * The 5000 / 4000 SFF per-format tables and full-rate accumulation are estimates, not vendor tables.
 * 4000 SFF uses PCIe Gen5 x8 electrically despite its full-length connector. No NVLink is offered.
 * FP6 support is mentioned by NVIDIA but its rate is unpublished: omitted, so lookup warns and uses FP8.
 */
import type { DeviceSpec, GemmTiling, MatmulPeak } from '../types';
import type { UnlockAxis } from '../unlocks';
import { gb, gbPerSec, tflops } from '../units';
import { NVIDIA_DC_GPUDIRECT_SOURCE } from './gpudirect';
import { NVIDIA_SM_SFU, nvidiaSfu } from './sfu';

const WHITEPAPER =
  'https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf';
const PRODUCT_ROOT =
  'https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/';
const PTX =
  'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma';
const POWER_SWEEP =
  'https://github.com/local-inference-lab/rtx6kpro/blob/master/hardware/blackwell-power-limit-sweep.md';

/** Dense and dense-equivalent 2:4 rates, in per-device TFLOP/s (integer: TOP/s). */
function pair(input: MatmulPeak['input'], dense: number, sparse: number): MatmulPeak[] {
  const accumulate = input === 'int8' ? 'int32' : 'fp32';
  return [
    { input, accumulate, sparsity: 'dense', flopsPerSec: tflops(dense) },
    { input, accumulate, sparsity: '2:4', flopsPerSec: tflops(sparse) },
  ];
}

/** SM120 warp-level mma.sync dimensions in elements; representative CUTLASS CTA scheduling. */
function tiling(mmaK: number): GemmTiling {
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

/** Explicit per-device tensor pairs, in TFLOP/s (integer: TOP/s); rounding retained from Table 4. */
function tensorTable(
  fp16: [number, number],
  tf32: [number, number],
  fp8: [number, number],
  fp4: [number, number],
): MatmulPeak[] {
  return [
    ...pair('fp16', ...fp16),
    ...pair('bf16', ...fp16),
    ...pair('tf32', ...tf32),
    ...pair('fp8', ...fp8),
    ...pair('int8', ...fp8),
    ...pair('fp4', ...fp4),
  ];
}

/** SKU facts; rates are per-device and memory uses marketed SI GB. */
interface ProSku {
  /** Stable device id. */
  id: string;
  /** Workstation SKU display name, without vendor prefix. */
  name: string;
  /** Number of SMs (128 FP32 lanes each). */
  sms: number;
  /** Non-tensor FP32 peak in TFLOP/s, per device. */
  vector: number;
  /** Marketed memory capacities in SI GB, same compute and bandwidth for all options. */
  capacities: number[];
  /** Memory bandwidth in SI GB/s. */
  bandwidth: number;
  /** Electrical PCIe Gen5 lane count, not physical connector width. */
  lanes: 8 | 16;
  /** Per-device dense and sparse tensor rates, in FLOP/s. */
  peaks: MatmulPeak[];
  /** Vendor datasheet URL. */
  datasheet: string;
  /** Exact quoted specification values and derivation. */
  note: string;
  /** Whether non-headline tensor rates / FP32 accumulation are inferred rather than vendor-tabulated. */
  inferred: boolean;
}

/** Build a dedicated-memory SKU with cited SM120 instruction shapes and explicitly estimated scheduling. */
function proDevice(s: ProSku): DeviceSpec {
  // 128 FP32 lanes × 2 FLOP/FMA = 256 FLOP/clock/SM: nvidiaSfu's clock derivation fits SM120.
  const sfu = nvidiaSfu(NVIDIA_SM_SFU, s.sms, tflops(s.vector));
  const gemmTiling = {
    default: tiling(16),
    fp16: tiling(16),
    bf16: tiling(16),
    tf32: tiling(8),
    fp8: tiling(32),
    int8: tiling(32),
    fp4: tiling(64),
  };
  const keys = Object.keys(gemmTiling);
  return {
    id: s.id,
    name: `NVIDIA ${s.name}`,
    vendor: 'NVIDIA',
    engines: [
      {
        id: 'gpu',
        name: 'Blackwell 5th-gen Tensor Cores (sm_120)',
        kind: 'gpu',
        computeUnits: s.sms,
        matmulPeaks: s.peaks,
        vectorFlopsPerSec: {
          fp32: tflops(s.vector),
          fp16: tflops(s.vector),
          bf16: tflops(s.vector),
          fp64: tflops(s.vector / 64),
        },
        sfuOpsPerSec: sfu.sfuOpsPerSec,
        gemmTiling,
      },
    ],
    memory: {
      kind: 'dedicated',
      technology: 'GDDR7',
      capacityOptionsBytes: s.capacities.map(gb),
      bandwidthBytesPerSec: gbPerSec(s.bandwidth),
    },
    ...(s.inferred ? {} : { l2Bytes: 131072 * 1024 }),
    hostInterface: { generation: 5, lanes: s.lanes },
    gpuDirectRdma: true,
    sources: [
      ...(s.lanes === 8
        ? [
            {
              url: 'https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4000-sff/',
              title: 'NVIDIA RTX PRO 4000 SFF product page, AI performance footnotes',
              fields: ['engines.0.matmulPeaks'],
              note: '770 TOPS; footnote 2: theoretical FP4 TOPS using sparsity. Dense and other dtypes are inferred as described in estimated[].',
            },
          ]
        : []),
      {
        url: s.datasheet,
        title: `${s.name} datasheet`,
        fields: [
          'memory',
          'hostInterface',
          'engines.0.computeUnits',
          'engines.0.vectorFlopsPerSec.fp32',
          ...(s.inferred ? ['engines.0.matmulPeaks'] : []),
        ],
        note:
          s.note +
          ' Memory is quoted in SI GB. SM count = CUDA cores / 128 (PRO whitepaper). All rates at GPU Boost Clock; AI TOPS means sparse FP4.',
      },
      {
        url: WHITEPAPER,
        title: 'NVIDIA RTX Blackwell PRO architecture, Table 4 pp. 45–47; p. 8 FP64 note',
        fields: s.inferred
          ? ['engines.0.vectorFlopsPerSec.fp64']
          : [
              'engines.0.computeUnits',
              'engines.0.matmulPeaks',
              'engines.0.vectorFlopsPerSec',
              'l2Bytes',
            ],
        note:
          '6000 WS / Max-Q: 188 SMs, 128 CUDA lanes/SM; vector FP32/FP16/BF16 126.0 / 109.7 TFLOP/s; FP64 = FP32/64. L2 131072 binary KB. Full-rate FP32 accumulation; FP16 accumulation has identical rates, so only FP32 entries are needed. ' +
          'WS dense/sparse: FP16/BF16 503.8/1007.6, TF32 251.9/503.8, FP8/INT8 1007.6/2015.2, FP4 2015.2/4030.4. Max-Q: 438.9/877.9, 219.5/438.9, 877.9/1755.7, 1755.7/3511.4. FP6 supported but no published rate; omit (fallback to FP8). No INT4 rate.',
      },
      {
        url: PTX,
        title: 'PTX ISA warp-level mma.sync shapes',
        fields: keys.flatMap((k) =>
          ['mmaM', 'mmaN', 'mmaK'].map((f) => `engines.0.gemmTiling.${k}.${f}`),
        ),
        note: 'f16/bf16 m16n8k16; tf32 m16n8k8; fp8/s8 m16n8k32; mxf4nvf4 m16n8k64.',
      },
      NVIDIA_DC_GPUDIRECT_SOURCE,
      ...sfu.sources,
    ],
    estimated: [
      ...keys.flatMap((k) => [
        {
          field: `engines.0.gemmTiling.${k}.ctaTiles`,
          reason:
            'Representative CUTLASS SM120 CTA tiles (https://github.com/NVIDIA/cutlass); not vendor hardware limits.',
        },
        {
          field: `engines.0.gemmTiling.${k}.ctasPerUnit`,
          reason:
            'One large CTA per SM, register/shared-memory constrained; kernel-dependent estimate.',
        },
      ]),
      ...(s.inferred
        ? [
            {
              field: 'engines.0.matmulPeaks',
              reason:
                'Only sparse FP4 is published: dense = sparse/2; FP8/INT8 = dense FP4/2; FP16/BF16 = FP4/4; TF32 = FP4/8, using PRO Table 4 ratios. Full-rate FP32 accumulation assumed for PRO firmware; unmeasured on 5000 and especially GB203-based 4000 SFF.',
            },
            {
              field: 'engines.0.vectorFlopsPerSec.fp16',
              reason: 'Same rate as FP32, inferred from the 6000 SM120 table.',
            },
            {
              field: 'engines.0.vectorFlopsPerSec.bf16',
              reason: 'Same rate as FP32, inferred from the 6000 SM120 table.',
            },
            ...(s.lanes === 8
              ? [
                  {
                    field: 'engines.0.vectorFlopsPerSec.fp64',
                    reason: 'FP32/64 inferred for GB203 from the GB202 PRO whitepaper p. 8.',
                  },
                ]
              : []),
          ]
        : []),
      ...sfu.estimated,
    ],
  };
}

/** RTX PRO 6000 Workstation, 96 GB, 1792 GB/s; per-device boost peaks (2617 MHz, 600 W). */
export const RTX_PRO_6000_WS = proDevice({
  id: 'rtx-pro-6000-ws',
  name: 'RTX PRO 6000 Blackwell Workstation Edition',
  sms: 188,
  vector: 126,
  capacities: [96],
  bandwidth: 1792,
  lanes: 16,
  inferred: false,
  peaks: tensorTable([503.8, 1007.6], [251.9, 503.8], [1007.6, 2015.2], [2015.2, 4030.4]),
  datasheet:
    'https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf',
  note: '24064 CUDA cores; 96 GB ECC GDDR7, 1792 GB/s, PCIe 5.0 x16. Use whitepaper 126.0 TFLOPS / 4030.4 sparse FP4 rather than rounded product headlines.',
});

/** RTX PRO 6000 Max-Q, own 300 W SKU; 96 GB and 1792 GB/s, lower 2280 MHz boost peaks. */
export const RTX_PRO_6000_MAX_Q = proDevice({
  id: 'rtx-pro-6000-max-q',
  name: 'RTX PRO 6000 Blackwell Max-Q Workstation Edition',
  sms: 188,
  vector: 109.7,
  capacities: [96],
  bandwidth: 1792,
  lanes: 16,
  inferred: false,
  peaks: tensorTable([438.9, 877.9], [219.5, 438.9], [877.9, 1755.7], [1755.7, 3511.4]),
  datasheet:
    PRODUCT_ROOT +
    'rtx-pro-6000-max-q/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-us-5349650-web.pdf',
  note: '24064 CUDA cores; 96 GB ECC GDDR7, 1792 GB/s, PCIe 5.0 x16. Whitepaper 109.7 vector TFLOPS / 3511.4 sparse FP4 retain table rounding.',
});

/** RTX PRO 5000: 48/72 GB capacity options share 1344 GB/s and identical per-device boost peaks. */
export const RTX_PRO_5000 = proDevice({
  id: 'rtx-pro-5000',
  name: 'RTX PRO 5000 Blackwell',
  sms: 110,
  vector: 65,
  capacities: [48, 72],
  bandwidth: 1344,
  lanes: 16,
  inferred: true,
  peaks: tensorTable([258, 516], [129, 258], [516, 1032], [1032, 2064]),
  datasheet:
    PRODUCT_ROOT +
    'rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-5488550-nvidia.pdf',
  note: '14080 CUDA cores / 128 = 110 SMs; 65 vector TFLOPS; 2064 sparse FP4 TOPS. Both 48/72 GB ECC GDDR7 columns share 1344 GB/s, 384-bit bus, PCIe 5.0 x16.',
});

/** RTX PRO 4000 SFF: 24 GB, 432 GB/s; PCIe Gen5 x8 electrical despite its full-length connector. */
export const RTX_PRO_4000_SFF = proDevice({
  id: 'rtx-pro-4000-sff',
  name: 'RTX PRO 4000 Blackwell SFF Edition',
  sms: 70,
  vector: 24,
  capacities: [24],
  bandwidth: 432,
  lanes: 8,
  inferred: true,
  peaks: tensorTable([96.25, 192.5], [48.125, 96.25], [192.5, 385], [385, 770]),
  datasheet:
    PRODUCT_ROOT +
    'rtx-pro-4000-sff/workstation-datasheet-blackwell-rtx-pro-4000-sff-nvidia-us-4016700.pdf',
  note: '8960 CUDA cores / 128 = 70 SMs; 24 vector TFLOPS; 770 sparse FP4 TOPS (product page footnote: theoretical FP4 using sparsity). 24 GB ECC GDDR7, 192-bit bus, 432 GB/s; PCIe 5.0 x8, footnote: uses full-length PCIe interface.',
});

/** RTX PRO picker order: Blackwell highest tier first, Workstation before its lower-power Max-Q SKU. */
export const NVIDIA_PRO_DEVICE_PRESETS: DeviceSpec[] = [
  RTX_PRO_6000_WS,
  RTX_PRO_6000_MAX_Q,
  RTX_PRO_5000,
  RTX_PRO_4000_SFF,
];

/** Optional measured compute-clock scaling; dimensionless factor relative to vendor boost, memory unchanged. */
function sustainedClock(measuredMHz: number, boostMHz: number, watts: number): UnlockAxis[] {
  return [
    {
      id: 'power-clock',
      name: 'Measured sustained clock',
      kind: 'powerClock',
      description:
        'Community tensor-burn measurement on one board at its stock power limit. Workload and cooling change the sustained clock; memory bandwidth is unchanged.',
      options: [
        {
          id: 'tensor-burn',
          name: `${watts} W: sustained ${measuredMHz} MHz`,
          deviceModifiers: [
            {
              kind: 'clockScale',
              engineId: 'gpu',
              factor: measuredMHz / boostMHz,
              note: 'Measured gpu_burn -tc clock divided by vendor boost MHz; not a guaranteed sustained rate.',
              sources: [
                {
                  url: POWER_SWEEP,
                  title: 'RTX PRO Blackwell community power-limit sweep',
                  kind: 'community',
                  fields: ['factor'],
                  note: `gpu_burn -tc: ${measuredMHz} MHz at ${watts} W; base boost ${boostMHz} MHz (PRO whitepaper Table 4).`,
                },
              ],
            },
          ],
        },
      ],
    },
  ];
}

/** 600 W Workstation sample: 1872/2617 compute-clock factor, opt-in; default remains vendor boost. */
export const RTX_PRO_6000_WS_UNLOCKS = sustainedClock(1872, 2617, 600);
/** 300 W Max-Q sample: 1417/2280 compute-clock factor, opt-in; default remains vendor boost. */
export const RTX_PRO_6000_MAX_Q_UNLOCKS = sustainedClock(1417, 2280, 300);
