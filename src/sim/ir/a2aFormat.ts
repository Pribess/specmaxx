/**
 * Wire formats of the expert-parallel (MoE mode B) dispatch and combine all-to-alls: what one element of a routed
 * token's hidden vector looks like on the link. Chosen by the deployment, not by the weights: DeepSeek-V3 dispatches
 * FP8 and combines BF16, SGLang/vLLM on GB200 dispatch NVFP4, AMD MoRI dispatches FP4 and combines FP8.
 *
 * Each format is a {@link TensorFormat}, so its bytes per element come from the same helper as every stored tensor
 * (`bytesPerElement`): element bits plus the amortized bits of its block scales. See docs/simulation-model.md §5
 * "Dispatch and combine precision".
 */

import { bytesPerElement, type Dtype, type TensorFormat } from './dtype';

/**
 * A dispatch or combine wire format. `'fp8_block128'` is FP8 (e4m3) with one fp32 scale per 128 values, the layout
 * DeepEP's FP8 dispatch and DeepSeek-V3's 1×128 activation quantization use; the others are the storage dtypes of
 * `dtype.ts` with their built-in block scales (`mxfp8`/`mxfp4`: one 8-bit scale per 32 values, `nvfp4`: one 8-bit scale
 * per 16). `'fp8_e4m3'` is FP8 without any scale (an upper bound of the saving; e.g. a combine whose scales stay on chip).
 */
export type A2aFormat = 'bf16' | 'fp8_e4m3' | 'fp8_block128' | 'mxfp8' | 'nvfp4' | 'mxfp4';

/** Every {@link A2aFormat}, in the order a picker lists them. */
export const A2A_FORMATS: readonly A2aFormat[] = [
  'bf16',
  'fp8_e4m3',
  'fp8_block128',
  'mxfp8',
  'nvfp4',
  'mxfp4',
];

/** Values per fp32 scale of `'fp8_block128'`. Count. */
export const FP8_SCALE_BLOCK = 128;

/** Bits of one fp32 scale. Unit: bits. */
const FP32_SCALE_BITS = 32;

/**
 * The storage format behind each {@link A2aFormat}. Bits per element, with block scales:
 * - `bf16` 16; `fp8_e4m3` 8;
 * - `fp8_block128` 8 + 32/128 = 8.25 (DeepEP: x in FP8 plus `x_scales` in fp32 per 128 channels, e.g. 7168 + 224 B
 *   per token at h = 7168: https://github.com/deepseek-ai/DeepEP);
 * - `mxfp8` 8 + 8/32 = 8.25; `mxfp4` 4 + 8/32 = 4.25; `nvfp4` 4 + 8/16 = 4.5 (OCP MX v1.0; NVIDIA NVFP4 blog, see
 *   `DTYPE_INFO`).
 */
export const A2A_TENSOR_FORMATS: Record<A2aFormat, TensorFormat> = {
  bf16: { dtype: 'bf16' },
  fp8_e4m3: { dtype: 'fp8_e4m3' },
  // Formula: extraBitsPerElem = 32/128 = 0.25   (one fp32 scale per 128 values)
  fp8_block128: { dtype: 'fp8_e4m3', extraBitsPerElem: FP32_SCALE_BITS / FP8_SCALE_BLOCK },
  mxfp8: { dtype: 'mxfp8' },
  nvfp4: { dtype: 'nvfp4' },
  mxfp4: { dtype: 'mxfp4' },
};

/** True when `value` is an {@link A2aFormat}. */
export function isA2aFormat(value: unknown): value is A2aFormat {
  return typeof value === 'string' && (A2A_FORMATS as readonly string[]).includes(value);
}

/** Bytes per element on the wire of a {@link A2aFormat}, block scales included. Unit: bytes/element. */
export function a2aBytesPerElement(format: A2aFormat): number {
  // Formula: bpe = (elementBits + blockScaleBits)/8   (bytesPerElement of the storage format)
  // See docs/simulation-model.md §5 "Dispatch and combine precision"
  return bytesPerElement(A2A_TENSOR_FORMATS[format]);
}

/** Element dtype tag a {@link A2aFormat} carries on the wire (`CollectiveOp.dtype`; informational). */
export function a2aWireDtype(format: A2aFormat): Dtype {
  return A2A_TENSOR_FORMATS[format].dtype;
}
