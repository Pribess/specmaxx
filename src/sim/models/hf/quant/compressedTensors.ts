/**
 * compressed-tensors (`quant_method: "compressed-tensors"`, llm-compressor / AutoRound exports): `config_groups`
 * of `{ targets, weights, input_activations, format }`, an `ignore` list and `kv_cache_scheme`.
 *
 * Weight formats (`weights` of a group, with the group's or the config's `format`):
 * - int 4/8, `strategy: group`: `int4` / `int8` + (scaleBits + zeroPointBits) / group_size, where the scale is
 *   `scale_dtype` (default bf16, the module dtype) and the zero point exists only when `symmetric: false`
 *   (`zp_dtype`, default packed at `num_bits`). Sym g128: 16/128 = 0.125; asym g32 with int8 zp: 24/32 = 0.75.
 * - float 4: `nvfp4-pack-quantized` (g16, e4m3 scales) → `nvfp4`; `mxfp4-pack-quantized` (g32, uint8 e8m0
 *   scales) → `mxfp4`.
 * - float 8: `strategy: block` [r, c] → fp8 + scaleBits / (r·c); `channel` / `tensor` → fp8, ≈0 overhead;
 *   `mxfp8-quantized` (g32, uint8 scales) → `mxfp8`.
 * - `input_activations` present (dynamic or static) → the GEMM input is quantized too (W8A8, W4A4).
 * Matching: `ignore` wins; then the group whose target matches most specifically (an exact name or `re:` regex
 * before a class name other than `Linear`, before `Linear`), the first group on a tie; unmatched modules keep
 * the native format. See docs/simulation-model.md §2 (quant overheads).
 */

import type { Dtype, TensorFormat } from '../../../ir/dtype';
import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import {
  blockScaleBits,
  dtypeNameBits,
  groupScaleBits,
  intDtypeFor,
  numAt,
  numListAt,
  objAt,
  strAt,
  strListAt,
} from './common';
import { anyMatch, compilePattern, compilePatterns, patternMatches, type NamePattern } from './patterns';
import { isJsonObject } from '../reader';
import type { JsonObject, QuantDefaults } from '../types';

/** One config group, compiled. */
export interface CtGroup {
  /** Group key, e.g. `'group_0'`. */
  name: string;
  /** Compiled targets with their specificity (2 name/regex, 1 specific class, 0 `Linear`). */
  targets: { pattern: NamePattern; specificity: number }[];
  /** The format the group gives a module, or null when it quantizes nothing. */
  format: ModuleQuant | null;
  /** Short description for the label. */
  label: string;
}

/**
 * Weight storage format of a compressed-tensors `weights` spec (also the ModelOpt `config_groups` shape,
 * which has no `strategy` / `format`), or null. Adds notes for approximations.
 */
export function ctWeightFormat(
  w: JsonObject,
  format: string | undefined,
  scaleFallbackBits: number,
  notes: string[],
): TensorFormat | null {
  const bits = numAt(w, 'num_bits');
  if (bits === undefined) return null;
  const type = strAt(w, 'type') ?? 'int';
  const strategy = strAt(w, 'strategy');
  const group = numAt(w, 'group_size');
  const block = numListAt(w, 'block_structure');
  const scaleBits = dtypeNameBits(w.scale_dtype, scaleFallbackBits);
  const fmt = format ?? '';

  if (type === 'float' && bits === 4) {
    if (fmt.includes('mxfp4') || group === 32) return { dtype: 'mxfp4' };
    if (fmt.includes('nvfp4') || group === 16) return { dtype: 'nvfp4' };
    notes.push(`fp4 weights with group ${group ?? '?'}: stored as fp4_e2m1 + its group scales`);
    return { dtype: 'fp4_e2m1', extraBitsPerElem: group ? groupScaleBits(scaleBits, 0, group) : 0 };
  }
  if (type === 'float' && bits === 8) {
    if (fmt.includes('mxfp8') || (group === 32 && dtypeNameBits(w.scale_dtype, 0) === 8)) return { dtype: 'mxfp8' };
    if (block && block.length === 2) return { dtype: 'fp8_e4m3', extraBitsPerElem: blockScaleBits(scaleBits, block[0], block[1]) };
    if (strategy === 'group' && group) return { dtype: 'fp8_e4m3', extraBitsPerElem: groupScaleBits(scaleBits, 0, group) };
    if (strategy === 'channel') notes.push('fp8 per-channel weight scales: overhead ≈0');
    return { dtype: 'fp8_e4m3' };
  }
  if (type === 'int') {
    const { dtype, approx } = intDtypeFor(bits);
    if (approx) notes.push(`int${bits} weights stored as ${dtype} (no ${bits}-bit dtype)`);
    if (strategy === 'channel' || strategy === 'tensor' || (!group && !block)) {
      if (strategy === 'channel') notes.push(`int${bits} per-channel weight scales: overhead ≈0`);
      return { dtype };
    }
    if (block && block.length === 2) return { dtype, extraBitsPerElem: blockScaleBits(scaleBits, block[0], block[1]) };
    const zpBits = w.symmetric === false ? dtypeNameBits(w.zp_dtype, bits) : 0;
    return { dtype, extraBitsPerElem: groupScaleBits(scaleBits, zpBits, group ?? 0) };
  }
  notes.push(`unrecognised weight spec (type ${type}, ${bits} bits); kept native`);
  return null;
}

/** GEMM-input dtype of an `input_activations` spec, or undefined when activations are not quantized. */
export function ctInputDtype(a: unknown, format: string | undefined): Dtype | undefined {
  if (!isJsonObject(a)) return undefined;
  const bits = numAt(a, 'num_bits');
  const type = strAt(a, 'type') ?? 'int';
  const group = numAt(a, 'group_size');
  const fmt = format ?? '';
  if (type === 'float' && bits === 4) return fmt.includes('mxfp4') || group === 32 ? 'mxfp4' : 'nvfp4';
  if (type === 'float' && bits === 8) {
    return fmt.includes('mxfp8') || (group === 32 && dtypeNameBits(a.scale_dtype, 0) === 8) ? 'mxfp8' : 'fp8_e4m3';
  }
  if (type === 'int' && bits === 8) return 'int8';
  if (type === 'int' && bits === 4) return 'int4';
  return undefined;
}

/** Specificity of a compiled target. */
function specificity(p: NamePattern): number {
  if (p.kind !== 'class') return 2;
  return p.raw === 'Linear' ? 0 : 1;
}

/**
 * Compile the `config_groups` of a compressed-tensors (or ModelOpt) config. `exact`: plain-name targets match
 * whole module names only (compressed-tensors); ModelOpt targets also name parent modules (`…mlp.experts`).
 */
export function compileCtGroups(qc: JsonObject, ctx: SchemeContext, notes: string[], exact = true): CtGroup[] {
  const groups = objAt(qc, 'config_groups') ?? {};
  const topFormat = strAt(qc, 'format');
  const scaleFallback = ctx.modelDtype === 'fp32' ? 32 : 16;
  const out: CtGroup[] = [];
  for (const [name, g] of Object.entries(groups)) {
    if (!isJsonObject(g)) continue;
    const format = strAt(g, 'format') ?? topFormat;
    const w = objAt(g, 'weights');
    const weights = w ? ctWeightFormat(w, format, scaleFallback, notes) : null;
    const input = ctInputDtype(g.input_activations, format);
    const targets = strListAt(g, 'targets')
      .map((t) => compilePattern(t, { exact }))
      .filter((p): p is NamePattern => p !== null)
      .map((pattern) => ({ pattern, specificity: specificity(pattern) }));
    const formatOut: ModuleQuant | null = weights
      ? input === undefined
        ? { weights }
        : { weights, gemmInput: input }
      : null;
    const wDesc = weights ? `${weights.dtype}${input ? ` W+A(${input})` : ''}` : 'none';
    out.push({ name, targets, format: formatOut, label: `${name}: ${wDesc}` });
  }
  return out;
}

/** The group that quantizes a probe: most specific target first, then config order; null when none. */
export function matchCtGroup(groups: CtGroup[], p: Probe): CtGroup | null {
  let best: CtGroup | null = null;
  let bestSpec = -1;
  for (const g of groups) {
    for (const t of g.targets) {
      if (t.specificity > bestSpec && patternMatches(t.pattern, p)) {
        best = g;
        bestSpec = t.specificity;
      }
    }
  }
  return best;
}

/** KV-cache format of a `kv_cache_scheme`, or undefined. */
export function ctKvCache(scheme: unknown): TensorFormat | undefined {
  if (!isJsonObject(scheme)) return undefined;
  const bits = numAt(scheme, 'num_bits');
  const type = strAt(scheme, 'type') ?? 'int';
  if (bits === 8) return { dtype: type === 'float' ? 'fp8_e4m3' : 'int8' };
  if (bits === 4) return { dtype: type === 'float' ? 'nvfp4' : 'int4' };
  return undefined;
}

/** Read a compressed-tensors `quantization_config`. */
export function readCompressedTensors(
  qc: JsonObject,
  ctx: SchemeContext,
  source: QuantDefaults['source'] = 'quantization_config',
): SchemeReading {
  const notes: string[] = [];
  const groups = compileCtGroups(qc, ctx, notes);
  const ignoreRaw = strListAt(qc, 'ignore');
  const ignore = compilePatterns(ignoreRaw, { exact: true });
  const kvCache = ctKvCache(qc.kv_cache_scheme);
  const sparsity = objAt(qc, 'sparsity_config');
  if (sparsity && Object.keys(sparsity).length > 0) notes.push('sparsity_config present but not modeled (dense weights assumed)');

  const resolve = (p: Probe): ModuleQuant => {
    if (anyMatch(ignore, p)) return ctx.native;
    return matchCtGroup(groups, p)?.format ?? ctx.native;
  };
  const provider = strAt(qc, 'provider');
  const label =
    `compressed-tensors ${strAt(qc, 'format') ?? ''}`.trim() +
    ` (${groups.map((g) => g.label).join('; ')})` +
    (kvCache ? `, ${kvCache.dtype} KV cache` : '') +
    (provider ? `, by ${provider}` : '');
  return {
    label,
    source,
    resolve,
    ...(kvCache ? { kvCache } : {}),
    ignoreLists: [{ label: 'ignore', patterns: ignoreRaw }],
    notes,
  };
}
