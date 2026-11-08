/**
 * ExLlamaV3 (EXL3) weight format: QTIP-style trellis-coded weights at a fractional number of bits per weight.
 *
 * What a quantized linear layer stores (exllamav3 `LinearEXL3`, `exllamav3/modules/quant/exl3.py`; format notes
 * in docs/simulation-model.md §2 "EXL3"):
 * - `trellis`: int16 tensor `[in/16, out/16, 16·K]`. Every 16×16 weight tile takes `16·K` 16-bit words =
 *   `256·K` bits, so the tensor stores EXACTLY `K` bits per weight. `K` is an integer 1–8, or a half-integer
 *   (mul1 codebook only). A repo's "2.5 bpw" is the AVERAGE of `K` over its tensors (an optimizer gives some
 *   tensors more bits), so any real number in [1, 8] is a valid average.
 * - `suh` (`in` fp16 values) and `svh` (`out` fp16 values): the sign-flip / scale vectors of the input- and
 *   output-side Hadamard rotations. Per matrix: `16·(in + out)` bits, independent of `K`.
 * - `mul1` (or `mcg`): one 32-bit codebook scalar per tensor (negligible, not modelled).
 *
 * Compute: the kernels decode the trellis to fp16 inside the matmul (tensor-core `mma` fp16 inputs, fp32/fp16
 * accumulate) or, for many rows, reconstruct an fp16 weight and run a normal fp16 GEMM. So the GEMM runs
 * as an fp16 × fp16 matmul: {@link EXL3_COMPUTE_DTYPE} is the dequantized dtype, and `resolveGemmMath`
 * (higher element bits win, tie → activation dtype) then gives the activation dtype's peak for 16-bit
 * activations. The trellis decode and the Hadamard rotations are NOT charged as extra FLOPs: they are
 * integer / add-only work that overlaps the tensor-core GEMM, and the theoretical-maximum model counts
 * weight bytes and the GEMM FLOPs only (see docs §2 "EXL3").
 */

import type { TensorFormat } from './dtype';

/** Dtype EXL3 weights are dequantized to inside the matmul; the compute peak is that of this dtype (or of the activations, if wider). */
export const EXL3_COMPUTE_DTYPE = 'fp16' as const;

/** Bits of the fp16 `suh` / `svh` vectors, stored once per input feature and once per output feature of a matrix. Unit: bits. */
export const EXL3_SIDE_VECTOR_BITS = 16;

/** Smallest legal per-tensor `K` (and so the smallest average). Unit: bits/weight. */
export const EXL3_MIN_BPW = 1;

/** Largest per-tensor `K` (and so the largest quantized average); `head_bits` 16 means "unquantized". Unit: bits/weight. */
export const EXL3_MAX_BPW = 8;

/** exllamav3's default `--head_bits` (LM head). Unit: bits/weight. */
export const EXL3_DEFAULT_HEAD_BPW = 6;

/**
 * Storage format of EXL3 weights at `bpw` average bits per weight (trellis bits only; the side vectors are
 * `perDimBits`, charged per matrix by `matrixStoredBytes`). `bpw` ≥ 16 means unquantized fp16 (exllamav3
 * `head_bits` / `mtp_bits` / `vision_bits` of 16). Values below 1 are raised to 1, above 8 lowered to 8.
 * Unit of `bpw`: bits/weight.
 */
export function exl3Format(bpw: number): TensorFormat {
  if (bpw >= 16) return { dtype: EXL3_COMPUTE_DTYPE };
  // Formula: stored value bits = clamp(bpw, 1, 8) per weight; + 16·(in + out) bits per matrix
  // See docs/simulation-model.md §2 "EXL3"
  const valueBitsPerElem = Math.min(EXL3_MAX_BPW, Math.max(EXL3_MIN_BPW, bpw));
  return { dtype: EXL3_COMPUTE_DTYPE, valueBitsPerElem, perDimBits: EXL3_SIDE_VECTOR_BITS };
}
