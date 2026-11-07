/**
 * Quant overrides: the three selects of the quantization panel (weights, activations, KV cache) applied over
 * a model's default `QuantConfig` (from the checkpoint's quant metadata, `QuantDefaults`).
 */

import { DTYPE_INFO, type Dtype } from '../../sim/ir/dtype';
import type { QuantConfig } from '../../sim/ir/model';
import type { QuantDefaults } from '../../sim/models/hf/types';
import type { QuantOverrides } from './appState';

/** Dtypes offered by the selects: every storage dtype but `int32` (never a matmul input) and `tf32` (a math mode). */
export const QUANT_DTYPE_CHOICES: Dtype[] = (Object.keys(DTYPE_INFO) as Dtype[]).filter((d) => d !== 'int32' && d !== 'tf32');

/** True when `value` is one of {@link QUANT_DTYPE_CHOICES}. */
export function isQuantDtype(value: string): value is Dtype {
  return (QUANT_DTYPE_CHOICES as string[]).includes(value);
}

/** True when any override is set. */
export function hasQuantOverride(o: QuantOverrides): boolean {
  return o.weights !== undefined || o.activations !== undefined || o.kvCache !== undefined;
}

/**
 * The model's default formats with the user's overrides applied. Rules:
 * - **Weights W**: `weights = { dtype: W }` (no group-quant overhead) and `modules` is dropped, so every linear
 *   module (attention, MLP, experts, indexer, linear attention, n-gram projections) uses W; `residual_mixer`
 *   falls back to `highPrecision` as usual. Embeddings, the LM head and lookup tables keep their defaults.
 *   Per-module `gemmInput`s go with `modules`, so GEMM inputs follow `activations`.
 * - **Activations A**: `activations = A`. A per-module `gemmInput` from the checkpoint (W8A8, W4A4) stays unless
 *   the weights are overridden too.
 * - **KV cache K**: `kvCache = { dtype: K }`, and `compressedKvCache` / `indexerKeyCache` are removed so they
 *   follow it (their documented default).
 * Nothing else changes. Pure: returns a new object.
 */
export function applyQuantOverride(defaults: QuantConfig, o: QuantOverrides): QuantConfig {
  let q: QuantConfig = { ...defaults };
  if (o.weights !== undefined) {
    const { modules: _modules, ...rest } = q;
    void _modules;
    q = { ...rest, weights: { dtype: o.weights } };
  }
  if (o.activations !== undefined) q = { ...q, activations: o.activations };
  if (o.kvCache !== undefined) {
    const { compressedKvCache: _c, indexerKeyCache: _i, ...rest } = q;
    void _c;
    void _i;
    q = { ...rest, kvCache: { dtype: o.kvCache } };
  }
  return q;
}

/** Quant defaults of the built-in reference model: bf16 weights, activations and KV cache. */
export const REFERENCE_QUANT_DEFAULTS: QuantDefaults = {
  quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
  scheme: 'bf16 (unquantized)',
  source: 'none',
  notes: [],
  explicit: [],
};

/**
 * A one-line label of the formats in use: the checkpoint's scheme when nothing is overridden, else the
 * overrides followed by the scheme they apply to, e.g. `'weights fp8_e4m3, KV fp8_e4m3 (over bf16 (unquantized))'`.
 */
export function quantSummary(defaults: Pick<QuantDefaults, 'scheme'>, o: QuantOverrides): string {
  const parts: string[] = [];
  if (o.weights !== undefined) parts.push(`weights ${o.weights}`);
  if (o.activations !== undefined) parts.push(`activations ${o.activations}`);
  if (o.kvCache !== undefined) parts.push(`KV ${o.kvCache}`);
  return parts.length === 0 ? defaults.scheme : `${parts.join(', ')} (over ${defaults.scheme})`;
}
