/**
 * Tensor element types, matmul input classes and accumulator types.
 *
 * Three related notions are kept apart:
 * - {@link Dtype}: how one element is STORED (decides bytes moved and bytes resident).
 * - {@link MathFormat}: the input class a matmul runs in (the first key of a device's peak table).
 * - {@link AccumulateType}: the matmul accumulator (the second key of a device's peak table).
 *
 * See docs/simulation-model.md §2 (dtype table, math and accumulate rules, quant overheads).
 */

/**
 * Storage dtype of one tensor element.
 *
 * Block-scaled ("microscaling") formats store a shared scale per block of elements in addition to the
 * element bits; that overhead is in {@link DtypeInfo.blockScaleBitsPerElem}:
 * - `mxfp8`, `mxfp6`, `mxfp4`: OCP MX formats, one e8m0 (8-bit) scale per 32 elements → 0.25 bit/element.
 * - `nvfp4`: NVIDIA FP4, e2m1 elements plus one e4m3 (8-bit) scale per 16 elements → 0.5 bit/element.
 *
 * `tf32` is stored as a full 32-bit word (it is an fp32 value that the tensor core reads with a 10-bit mantissa).
 * `int32` is an accumulator/index type only (e.g. top-k indices); it is never a matmul input.
 */
export type Dtype =
  | 'fp32'
  | 'tf32'
  | 'bf16'
  | 'fp16'
  | 'fp8_e4m3'
  | 'fp8_e5m2'
  | 'mxfp8'
  | 'fp6_e3m2'
  | 'fp6_e2m3'
  | 'mxfp6'
  | 'fp4_e2m1'
  | 'nvfp4'
  | 'mxfp4'
  | 'int8'
  | 'int4'
  | 'int32';

/**
 * Matmul input class: the first key of a device's matmul peak table (`MatmulPeak.input`).
 * Several storage dtypes share one class (e.g. `fp8_e4m3`, `fp8_e5m2` and `mxfp8` all run as `fp8`).
 */
export type MathFormat = 'fp64' | 'fp32' | 'tf32' | 'bf16' | 'fp16' | 'fp8' | 'fp6' | 'fp4' | 'int8' | 'int4';

/**
 * Matmul accumulator type: the second key of a device's matmul peak table (`MatmulPeak.accumulate`).
 * Consumer GPUs often run fp16-accumulate at twice the fp32-accumulate rate, hence the separate key.
 */
export type AccumulateType = 'fp16' | 'fp32' | 'int32';

/** Static facts about one storage {@link Dtype}. */
export interface DtypeInfo {
  /** Bits of one element's value, without any block scale (e.g. 4 for nvfp4, 32 for tf32). Unit: bits. */
  elementBits: number;
  /** Amortized bits per element of the format's built-in block scale (0 for unscaled formats). Unit: bits/element. */
  blockScaleBitsPerElem: number;
  /** Matmul input class this dtype runs as, or `null` if it is never a matmul input (`int32`). */
  math: MathFormat | null;
  /** True for integer formats (their matmuls accumulate in `int32` and their peaks are OP/s, not FLOP/s). */
  isInteger: boolean;
}

/**
 * The dtype table. Block-scale overheads: MX formats 8 bit / 32 elements = 0.25; nvfp4 8 bit / 16 elements = 0.5.
 * Sources: OCP Microscaling Formats (MX) v1.0 spec
 * (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and NVIDIA's NVFP4
 * description (https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
 */
export const DTYPE_INFO: Record<Dtype, DtypeInfo> = {
  fp32: { elementBits: 32, blockScaleBitsPerElem: 0, math: 'fp32', isInteger: false },
  tf32: { elementBits: 32, blockScaleBitsPerElem: 0, math: 'tf32', isInteger: false },
  bf16: { elementBits: 16, blockScaleBitsPerElem: 0, math: 'bf16', isInteger: false },
  fp16: { elementBits: 16, blockScaleBitsPerElem: 0, math: 'fp16', isInteger: false },
  fp8_e4m3: { elementBits: 8, blockScaleBitsPerElem: 0, math: 'fp8', isInteger: false },
  fp8_e5m2: { elementBits: 8, blockScaleBitsPerElem: 0, math: 'fp8', isInteger: false },
  mxfp8: { elementBits: 8, blockScaleBitsPerElem: 8 / 32, math: 'fp8', isInteger: false },
  fp6_e3m2: { elementBits: 6, blockScaleBitsPerElem: 0, math: 'fp6', isInteger: false },
  fp6_e2m3: { elementBits: 6, blockScaleBitsPerElem: 0, math: 'fp6', isInteger: false },
  mxfp6: { elementBits: 6, blockScaleBitsPerElem: 8 / 32, math: 'fp6', isInteger: false },
  fp4_e2m1: { elementBits: 4, blockScaleBitsPerElem: 0, math: 'fp4', isInteger: false },
  nvfp4: { elementBits: 4, blockScaleBitsPerElem: 8 / 16, math: 'fp4', isInteger: false },
  mxfp4: { elementBits: 4, blockScaleBitsPerElem: 8 / 32, math: 'fp4', isInteger: false },
  int8: { elementBits: 8, blockScaleBitsPerElem: 0, math: 'int8', isInteger: true },
  int4: { elementBits: 4, blockScaleBitsPerElem: 0, math: 'int4', isInteger: true },
  int32: { elementBits: 32, blockScaleBitsPerElem: 0, math: null, isInteger: true },
};

/**
 * A stored tensor format: a dtype plus optional group-quantization overhead outside the dtype itself.
 */
export interface TensorFormat {
  /** Element storage dtype. */
  dtype: Dtype;
  /**
   * Extra amortized bits per element for group-quantization scales/zeros that are not part of the dtype
   * (e.g. GPTQ int4 with group size 128, a 16-bit scale and a 4-bit zero: (16 + 4) / 128 = 0.15625).
   * Unit: bits/element. Default 0.
   */
  extraBitsPerElem?: number;
}

/**
 * Storage bits per element of a tensor format, including block scales and group-quant overhead.
 * Unit: bits/element.
 */
export function bitsPerElement(f: TensorFormat): number {
  const info = DTYPE_INFO[f.dtype];
  // Formula: bits = elementBits + blockScaleBitsPerElem + extraBitsPerElem
  // See docs/simulation-model.md §2
  const bits = info.elementBits + info.blockScaleBitsPerElem + (f.extraBitsPerElem ?? 0);
  return bits;
}

/** Storage bytes per element of a tensor format (`bitsPerElement / 8`; fractional for sub-byte formats). Unit: bytes/element. */
export function bytesPerElement(f: TensorFormat): number {
  // Formula: bpe = bits / 8
  return bitsPerElement(f) / 8;
}

/** Matmul input class of a storage dtype. Throws for `int32`, which is never a matmul input. */
export function mathFormatOf(d: Dtype): MathFormat {
  const math = DTYPE_INFO[d].math;
  if (math === null) throw new Error(`dtype ${d} is not a matmul input format`);
  return math;
}

/**
 * Input class of a GEMM with a `weight` and an `activation` operand: the operand with more element bits
 * wins (a weight-only quantized int4 × bf16 GEMM dequantizes to bf16 and runs at the bf16 rate); on a tie
 * the activation's format wins (bf16 weights × fp16 activations run as fp16). Block scales do not count.
 */
export function resolveGemmMath(weight: Dtype, activation: Dtype): MathFormat {
  const weightBits = DTYPE_INFO[weight].elementBits;
  const activationBits = DTYPE_INFO[activation].elementBits;
  return weightBits > activationBits ? mathFormatOf(weight) : mathFormatOf(activation);
}

/**
 * Accumulator of a matmul with input class `math`:
 * - integer inputs (int8, int4) → `int32`;
 * - bf16, tf32, fp32 and fp64 → `fp32` (hardware has no fp16 accumulate for these; fp64 has no own
 *   accumulate key and is keyed as `fp32`);
 * - other float inputs (fp16, fp8, fp6, fp4) → `requested` (the user's choice, default `fp32`).
 */
export function resolveAccumulate(math: MathFormat, requested: 'fp32' | 'fp16'): AccumulateType {
  switch (math) {
    case 'int8':
    case 'int4':
      return 'int32';
    case 'fp64':
    case 'fp32':
    case 'tf32':
    case 'bf16':
      return 'fp32';
    case 'fp16':
    case 'fp8':
    case 'fp6':
    case 'fp4':
      return requested;
    default: {
      const _never: never = math;
      throw new Error(`unknown math format ${String(_never)}`);
    }
  }
}
