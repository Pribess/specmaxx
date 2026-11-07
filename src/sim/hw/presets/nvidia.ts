/**
 * NVIDIA device presets. Every number is cited in `sources[]` (with the page/table it comes from) or listed in
 * `estimated[]` with the reason. See docs/simulation-model.md §4 ("Hardware").
 */

import type { MathFormat } from '../../ir/dtype';
import type { DeviceSpec, GemmTiling, SourceRef } from '../types';
import { gbPerSec, gb, tflops } from '../units';

/** NVIDIA RTX Blackwell GPU Architecture whitepaper (Appendix A, Table 3: GeForce RTX 5090 specs, pp. 46–48). */
const RTX_BLACKWELL_WHITEPAPER_URL =
  'https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf';
/** GeForce RTX 5090 product page. */
const RTX_5090_PRODUCT_URL = 'https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/';
/** PTX ISA, warp-level matrix (mma.sync) shapes. */
const PTX_MMA_SHAPES_URL =
  'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape';

/**
 * Typical CUTLASS CTA output tiles for sm_120 `mma.sync` GEMMs (elements, m × n). Not a vendor spec; see the
 * `estimated` entry. Every m is a multiple of mmaM 16 and every n of mmaN 8.
 */
const SM120_CTA_TILES: { m: number; n: number }[] = [
  { m: 128, n: 256 },
  { m: 128, n: 128 },
  { m: 128, n: 64 },
  { m: 64, n: 128 },
  { m: 64, n: 64 },
];

/**
 * sm_120 tiling for an input class whose `mma.sync` shape is m16 n8 k`mmaK` (PTX ISA):
 * f16/bf16 m16n8k16, tf32 m16n8k8, e4m3/e5m2 and s8 m16n8k32, block-scaled fp4 (mxf4nvf4) m16n8k64.
 * One CTA per SM (a 128×256 or 128×128 tile uses most of the SM's registers and shared memory).
 */
function sm120Tiling(mmaK: number): GemmTiling {
  return { mmaM: 16, mmaN: 8, mmaK, ctaTiles: SM120_CTA_TILES.map((t) => ({ ...t })), ctasPerUnit: 1 };
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
        default: sm120Tiling(16),
        fp16: sm120Tiling(16),
        bf16: sm120Tiling(16),
        tf32: sm120Tiling(8),
        fp8: sm120Tiling(32),
        int8: sm120Tiling(32),
        fp4: sm120Tiling(64),
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
