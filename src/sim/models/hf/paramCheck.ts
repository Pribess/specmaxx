/**
 * Check the IR's parameter count against the checkpoint's, and a generic vision-tower estimate.
 *
 * Expected = the Hub's `safetensors.total` (the sum of the safetensors header element counts; it includes
 * MTP layers and vision/audio towers). Actual = `countParams(model) + mtpParamsEstimate + visionParamsEstimate`.
 * Rules, in order:
 * 1. No `safetensors` in the api info (e.g. Mistral consolidated repos) → `unavailable`.
 * 2. Packed quantized storage → `packed`: the Hub counts each packed word (`U8`, `U32`, `I32`, …) as ONE
 *    element although it holds several weights, so the total is not a parameter count. Packed when
 *    `parameters` has a `U8`, `U16`, `U32`, `I32` or `I16` key, or an `I8` key while the repo's
 *    `quant_method` is not DeepSeek's native `fp8` (DeepSeek-V4.1 stores its fp4 experts as `I8` with one
 *    element per weight, and its total matches the IR).
 * 3. `|actual − expected| / expected ≤ PARAM_CHECK_TOLERANCE` (2%) → `ok`, else `mismatch` (a warning).
 */

import { countParams } from '../../ir/params';
import type { ModelSpec } from '../../ir/model';
import type { ExcludedParams, HfApiModelInfo, JsonObject, ParamCheck } from './types';
import { PARAM_CHECK_TOLERANCE } from './types';

/** Safetensors dtype keys that always mean packed sub-byte quantized words. */
const PACKED_KEYS = ['U8', 'U16', 'U32', 'I16', 'I32'];

/**
 * Compare `model` (+ excluded estimates) with `api.safetensors.total`.
 * `quantMethod` is the config's `quantization_config.quant_method`, used to tell packed `I8` storage
 * from DeepSeek's unpacked one.
 */
export function checkParams(
  model: ModelSpec,
  excluded: Pick<ExcludedParams, 'mtpParamsEstimate' | 'visionParamsEstimate'>,
  api: HfApiModelInfo | undefined,
  quantMethod?: string,
): ParamCheck {
  // Formula: actual = P_IR + P_mtp + P_vision
  const irParams = countParams(model);
  const actual = irParams + excluded.mtpParamsEstimate + excluded.visionParamsEstimate;

  const st = api?.safetensors;
  if (!st || !Number.isFinite(st.total) || st.total <= 0) {
    return { status: 'unavailable', actual, note: 'the repo publishes no safetensors parameter totals' };
  }
  const expected = st.total;
  // Formula: relError = (actual − expected) / expected
  const relError = (actual - expected) / expected;

  const keys = Object.keys(st.parameters ?? {});
  const packedKey = keys.find((k) => PACKED_KEYS.includes(k));
  const packedI8 = keys.includes('I8') && quantMethod !== undefined && quantMethod !== 'fp8';
  if (packedKey || packedI8) {
    const base = api?.cardData?.base_model;
    const baseList = base === undefined ? [] : Array.isArray(base) ? base : [base];
    const baseHint = baseList.length > 0 ? `compare against the base model (${baseList.join(', ')}) instead` : 'compare against the unquantized base model instead';
    return {
      status: 'packed',
      expected,
      actual,
      relError,
      note: `packed quantized storage (${packedKey ?? 'I8'}): the Hub counts packed words as elements, so safetensors.total is not a parameter count; ${baseHint}`,
    };
  }

  const pct = (relError * 100).toFixed(2);
  if (Math.abs(relError) <= PARAM_CHECK_TOLERANCE) {
    return { status: 'ok', expected, actual, relError, note: `IR + excluded parts match safetensors.total (${pct}%)` };
  }
  return {
    status: 'mismatch',
    expected,
    actual,
    relError,
    note: `IR + excluded parts differ from safetensors.total by ${pct}% (tolerance ±${PARAM_CHECK_TOLERANCE * 100}%)`,
  };
}

/** A positive finite number field of `o`, or undefined. */
function posNum(o: JsonObject, ...keys: string[]): number | undefined {
  for (const k of keys) {
    const v = o[k];
    if (typeof v === 'number' && Number.isFinite(v) && v > 0) return v;
  }
  return undefined;
}

/**
 * Generic vision-tower parameter estimate from `vision_config`, for the param check only. Unit: params.
 *
 * Per encoder layer: attention `4·d²` (q, k, v, o) plus MLP `2·d·ffn`, or `3·d·ffn` when `hidden_act` is a
 * gated `silu`/`swiglu`. Merger: with `spatial_merge_size` s, `m = d·s²` and `m² + m·out` (a 2-layer MLP onto
 * `out_hidden_size`), plus a gated `3·out·projection_intermediate_size` when present (GLM); otherwise one
 * projection `d·out` with out = `out_hidden_size` / `projection_dim` / `textHidden`. Biases, norms, patch
 * and position embeddings are ignored. Typically within ±20% of the real tower (towers are ≤ 1.7% of any
 * shortlist model); families that know their tower better may compute their own. Returns 0 when the config
 * lacks a width or a depth.
 */
export function estimateVisionParams(visionConfig: JsonObject | undefined, textHidden?: number): number {
  if (!visionConfig) return 0;
  const depth = posNum(visionConfig, 'depth', 'num_hidden_layers', 'n_layers');
  const d = posNum(visionConfig, 'hidden_size', 'embed_dim');
  if (depth === undefined || d === undefined) return 0;
  const ffn = posNum(visionConfig, 'intermediate_size') ?? 4 * d;
  const act = visionConfig.hidden_act;
  const gated = act === 'silu' || act === 'swiglu';
  const out = posNum(visionConfig, 'out_hidden_size', 'projection_dim') ?? textHidden ?? d;

  // Formula: P_layer = 4·d² + (gated ? 3 : 2)·d·ffn
  const perLayer = 4 * d * d + (gated ? 3 : 2) * d * ffn;
  const mergeSize = posNum(visionConfig, 'spatial_merge_size');
  let merger: number;
  if (mergeSize !== undefined) {
    // Formula: m = d·s²; P_merger = m² + m·out (+ 3·out·pi when projection_intermediate_size is set)
    const m = d * mergeSize * mergeSize;
    const pi = posNum(visionConfig, 'projection_intermediate_size');
    merger = m * m + m * out + (pi !== undefined ? 3 * out * pi : 0);
  } else {
    // Formula: P_merger = d·out
    merger = d * out;
  }
  // Formula: P_vision = depth·P_layer + P_merger
  return depth * perLayer + merger;
}
