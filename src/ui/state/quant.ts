/**
 * Quant overrides: the three selects of the quantization panel (weights, activations, KV cache) applied over
 * a model's default `QuantConfig` (from the checkpoint's quant metadata, `QuantDefaults`).
 */

import { DTYPE_INFO, type Dtype, type TensorFormat } from '../../sim/ir/dtype';
import { EXL3_DEFAULT_HEAD_BPW, EXL3_MAX_BPW, EXL3_MIN_BPW, exl3Format } from '../../sim/ir/exl3';
import type { QuantConfig } from '../../sim/ir/model';
import type { QuantDefaults } from '../../sim/models/hf/types';
import type { QuantOverrides } from './appState';

/** Dtypes offered by the selects: every storage dtype but `int32` (never a matmul input) and `tf32` (a math mode). */
export const QUANT_DTYPE_CHOICES: Dtype[] = (Object.keys(DTYPE_INFO) as Dtype[]).filter(
  (d) => d !== 'int32' && d !== 'tf32',
);

/** True when `value` is one of {@link QUANT_DTYPE_CHOICES}. */
export function isQuantDtype(value: string): value is Dtype {
  return (QUANT_DTYPE_CHOICES as string[]).includes(value);
}

/** KV-cache formats in the Quantization panel itself: bf16 and fp8 (the common `--kv-cache-dtype` choices). */
export const KV_MAIN_DTYPES: Dtype[] = ['bf16', 'fp8_e4m3'];

/**
 * KV-cache formats under the panel's "More KV cache formats": every storage dtype the URL accepts for `kv=` that is not
 * in {@link KV_MAIN_DTYPES}, so no dtype becomes unselectable.
 */
export const KV_ADVANCED_DTYPES: Dtype[] = QUANT_DTYPE_CHOICES.filter((d) => !KV_MAIN_DTYPES.includes(d));

/** True when `value` is one of {@link KV_ADVANCED_DTYPES}. */
export function isKvAdvancedDtype(value: string | undefined): value is Dtype {
  return value !== undefined && (KV_ADVANCED_DTYPES as string[]).includes(value);
}

/** Dtypes offered for the lookup tables: the storage formats a row gather can read (no fp32/tf32 math modes). */
export const LOOKUP_TABLE_DTYPES: Dtype[] = (
  ['bf16', 'fp16', 'fp8_e4m3', 'mxfp8', 'nvfp4', 'mxfp4', 'int8', 'int4'] as Dtype[]
).filter((d) => d in DTYPE_INFO);

/** True when `value` is one of {@link LOOKUP_TABLE_DTYPES}. */
export function isLookupTableDtype(value: string): value is Dtype {
  return (LOOKUP_TABLE_DTYPES as string[]).includes(value);
}

/** Default EXL3 average bits per weight when the checkpoint states none. Unit: bits/weight. */
export const DEFAULT_EXL3_BPW = 4;

/** The weight-format choices of the Weights picker: every dtype choice, then `'exl3'`. */
export const WEIGHT_FORMAT_CHOICES: (Dtype | 'exl3')[] = [...QUANT_DTYPE_CHOICES, 'exl3'];

/** True when `value` is one of {@link WEIGHT_FORMAT_CHOICES}. */
export function isWeightFormat(value: string): value is Dtype | 'exl3' {
  return (WEIGHT_FORMAT_CHOICES as string[]).includes(value);
}

/** True for a usable EXL3 average bpw (finite, within [1, 8]) — the range one tensor's `K` can take. */
export function isValidBpw(n: number): boolean {
  return Number.isFinite(n) && n >= EXL3_MIN_BPW && n <= EXL3_MAX_BPW;
}

/** True for a usable EXL3 head bpw: {@link isValidBpw} or 16 (unquantized head). */
export function isValidHeadBpw(n: number): boolean {
  return isValidBpw(n) || n === 16;
}

/** The bpw pair EXL3 weights are simulated with: the override, else the checkpoint's, else the defaults. Unit: bits/weight. */
export function effectiveBpw(
  o: QuantOverrides,
  exl3: QuantDefaults['exl3'],
): { bpw: number; headBpw: number } {
  return {
    bpw: o.bpw ?? exl3?.bpw ?? DEFAULT_EXL3_BPW,
    headBpw: o.headBpw ?? exl3?.headBpw ?? EXL3_DEFAULT_HEAD_BPW,
  };
}

/** True when any override is set. */
export function hasQuantOverride(o: QuantOverrides): boolean {
  return (
    o.weights !== undefined ||
    o.activations !== undefined ||
    o.kvCache !== undefined ||
    o.lookupTables !== undefined
  );
}

/**
 * The model's default formats with the user's overrides applied. Rules:
 * - **Weights W**: `weights = { dtype: W }` (no group-quant overhead) and `modules` is dropped, so every linear
 *   module (attention, MLP, experts, indexer, linear attention, n-gram projections) uses W; `residual_mixer`
 *   falls back to `highPrecision` as usual. Embeddings and the LM head keep their defaults.
 *   Per-module `gemmInput`s go with `modules`, so GEMM inputs follow `activations`.
 * - **Weights `exl3`**: linear modules get `exl3Format(bpw)` (fractional bits, fp16 compute) and the LM head
 *   `exl3Format(headBpw)`, both from {@link effectiveBpw}; activations become fp16 unless overridden (exllamav3
 *   runs in fp16).
 * - **Activations A**: `activations = A`. A per-module `gemmInput` from the checkpoint (W8A8, W4A4) stays unless
 *   the weights are overridden too.
 * - **KV cache K**: `kvCache = { dtype: K }`, and `compressedKvCache` / `indexerKeyCache` are removed so they
 *   follow it (their documented default).
 * - **Lookup tables L**: `lookupTables = { dtype: L }` (n-gram / Engram tables only; no effect on a model without).
 * Nothing else changes. Pure: returns a new object.
 */
export function applyQuantOverride(
  defaults: QuantConfig,
  o: QuantOverrides,
  exl3?: QuantDefaults['exl3'],
): QuantConfig {
  let q: QuantConfig = { ...defaults };
  if (o.weights !== undefined) {
    const { modules: _modules, ...rest } = q;
    void _modules;
    if (o.weights === 'exl3') {
      const { bpw, headBpw } = effectiveBpw(o, exl3);
      q = { ...rest, weights: exl3Format(bpw), lmHead: exl3Format(headBpw), activations: 'fp16' };
      delete q.lmHeadGemmInput;
    } else q = { ...rest, weights: { dtype: o.weights } };
  }
  if (o.activations !== undefined) q = { ...q, activations: o.activations };
  if (o.kvCache !== undefined) {
    const { compressedKvCache: _c, indexerKeyCache: _i, ...rest } = q;
    void _c;
    void _i;
    q = { ...rest, kvCache: { dtype: o.kvCache } };
  }
  if (o.lookupTables !== undefined) q = { ...q, lookupTables: { dtype: o.lookupTables } };
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
export function quantSummary(
  defaults: Pick<QuantDefaults, 'scheme' | 'exl3'>,
  o: QuantOverrides,
): string {
  const parts: string[] = [];
  if (o.weights === 'exl3') {
    const { bpw, headBpw } = effectiveBpw(o, defaults.exl3);
    parts.push(`weights exl3 ${bpw} bpw (head ${headBpw})`);
  } else if (o.weights !== undefined) parts.push(`weights ${o.weights}`);
  if (o.activations !== undefined) parts.push(`activations ${o.activations}`);
  if (o.kvCache !== undefined) parts.push(`KV ${o.kvCache}`);
  if (o.lookupTables !== undefined) parts.push(`lookup tables ${o.lookupTables}`);
  return parts.length === 0 ? defaults.scheme : `${parts.join(', ')} (over ${defaults.scheme})`;
}

/** Which quant field a default label describes. */
export type QuantFieldKey = 'weights' | 'activations' | 'kvCache' | 'lookupTables';

/** Display name of a tensor format: the dtype, or `exl3 <bpw>bpw` for a fractional-bit trellis format. */
function formatName(f: TensorFormat): string {
  return f.valueBitsPerElem !== undefined ? `exl3 ${f.valueBitsPerElem}bpw` : f.dtype;
}

/** Distinct entries in first-seen order. */
function distinct(xs: string[]): string[] {
  return [...new Set(xs)];
}

/**
 * What a field's "Model default" resolves to: `label` is short (one name, or `a + b` for a mixed default, with
 * `a + b + …` beyond two) and `detail` is the full split by tensor class, present only for a mixed default.
 */
export function resolvedDefault(
  q: QuantConfig,
  field: QuantFieldKey,
): { label: string; detail?: string } {
  const parts: { name: string; where: string }[] = [];
  if (field === 'weights') {
    parts.push({ name: formatName(q.weights), where: 'other linear layers' });
    // residual_mixer is not a Weights-picker target (it stays at high precision under an override): skip it.
    for (const [mod, m] of Object.entries(q.modules ?? {})) {
      if (mod !== 'residual_mixer') parts.push({ name: formatName(m.weights), where: mod });
    }
  } else if (field === 'activations') {
    parts.push({ name: q.activations, where: 'other activations' });
    for (const [mod, m] of Object.entries(q.modules ?? {})) {
      if (m.gemmInput !== undefined) parts.push({ name: m.gemmInput, where: `${mod} GEMM input` });
    }
  } else if (field === 'lookupTables') {
    // Mirrors the engine: `lookupTableFormat` = quant.lookupTables, else the embeddings format (high precision, bf16).
    parts.push({
      name: formatName(q.lookupTables ?? q.embeddings ?? { dtype: q.highPrecision ?? 'bf16' }),
      where: 'lookup tables',
    });
  } else {
    parts.push({ name: formatName(q.kvCache), where: 'KV cache' });
    if (q.compressedKvCache)
      parts.push({ name: formatName(q.compressedKvCache), where: 'compressed KV cache' });
    if (q.indexerKeyCache)
      parts.push({ name: formatName(q.indexerKeyCache), where: 'indexer key cache' });
  }
  const names = distinct(parts.map((p) => p.name));
  if (names.length === 1) return { label: names[0] };
  const label = names.length <= 2 ? names.join(' + ') : `${names.slice(0, 2).join(' + ')} + …`;
  const detail = names
    .map(
      (n) =>
        `${n}: ${parts
          .filter((p) => p.name === n)
          .map((p) => p.where)
          .join(', ')}`,
    )
    .join('; ');
  return { label, detail };
}

/** The EXL3 average bpw presets listed in the Weights picker. Unit: bits/weight. */
export const EXL3_BPW_PRESETS: number[] = [2, 2.5, 3, 3.5, 4, 5, 6, 8];

/** Weights-picker value of the `EXL3 custom…` entry (the bpw field is shown). */
export const EXL3_CUSTOM_KEY = 'exl3:custom';

/** Weights-picker value of the `EXL3 <bpw>bpw (model default)` entry for a checkpoint bpw that is not a preset. */
export const EXL3_MODEL_KEY = 'exl3:model';

/** Weights-picker value of the preset entry for `bpw`. */
export function exl3PresetKey(bpw: number): string {
  return `exl3:${bpw}`;
}

/**
 * The Weights-picker entry selected under EXL3 weights. `bpw` is the effective bpw, `checkpointBpw` the
 * checkpoint's stated bpw, and `custom` whether the user picked `EXL3 custom…`. A bpw that is neither a
 * preset nor the checkpoint's shows as custom.
 */
export function exl3PickerKey(
  bpw: number,
  checkpointBpw: number | undefined,
  custom: boolean,
): string {
  if (custom) return EXL3_CUSTOM_KEY;
  if (EXL3_BPW_PRESETS.includes(bpw)) return exl3PresetKey(bpw);
  if (checkpointBpw === bpw) return EXL3_MODEL_KEY;
  return EXL3_CUSTOM_KEY;
}
