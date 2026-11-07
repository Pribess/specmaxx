/**
 * AMD Quark (`quant_method: "quark"`): `global_quant_config`, per-module `layer_quant_config` (names or fnmatch
 * globs), per-type `layer_type_quant_config`, an `exclude` list and `kv_cache_quant_config`.
 *
 * Spec → format (`weight` of a spec; `input_tensors` present → the GEMM input is quantized too):
 * - `fp4` per_group 32 with `scale_format: e8m0` → `mxfp4` (OCP MX); group 16 → `nvfp4`.
 * - `fp8_e4m3` per_block [r, c] → fp8 + scaleBits / (r·c) (`scale_type`: `float32` 32 bits,
 *   `float8_e8m0fnu` 8): 32×32 e8m0 = 8/1024; 128×128 fp32 = 32/16384. per_group g → scaleBits / g.
 *   per_channel / per_tensor → ≈0.
 * - `int4` / `uint4` per_group g → int4 + (16 + zero bits) / g (a 4-bit zero point when asymmetric or uint).
 * Matching: `exclude` wins, then the first matching `layer_quant_config` key, then `layer_type_quant_config`
 * (`Linear`), then the global config for Linear modules; the rest stays native.
 * See docs/simulation-model.md §2 (quant overheads).
 */

import type { Dtype, TensorFormat } from '../../../ir/dtype';
import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import { blockScaleBits, describeFormat, dtypeNameBits, groupScaleBits, numAt, numListAt, objAt, strAt, strListAt } from './common';
import { anyMatch, compilePatterns, firstMatch } from './patterns';
import { isJsonObject } from '../reader';
import type { JsonObject } from '../types';

/** Weight storage format of one Quark tensor spec, or null when the spec is absent or unknown. */
export function quarkWeightFormat(spec: unknown, notes: string[]): TensorFormat | null {
  if (!isJsonObject(spec)) return null;
  const dtype = (strAt(spec, 'dtype') ?? '').toLowerCase();
  const qscheme = strAt(spec, 'qscheme');
  const group = numAt(spec, 'group_size');
  const block = numListAt(spec, 'block_size');
  const scaleFmt = strAt(spec, 'scale_format');
  const scaleBits = scaleFmt === 'e8m0' ? 8 : dtypeNameBits(spec.scale_type, 32);

  if (dtype === 'fp4' || dtype === 'mxfp4' || dtype === 'mx_fp4') {
    if (group === 16 && scaleFmt !== 'e8m0') return { dtype: 'nvfp4' };
    if (group === 32 || dtype !== 'fp4') return { dtype: 'mxfp4' };
    notes.push(`quark fp4 with group ${group ?? '?'}: stored as fp4_e2m1 + group scales`);
    return { dtype: 'fp4_e2m1', extraBitsPerElem: group ? groupScaleBits(scaleBits, 0, group) : 0 };
  }
  if (dtype === 'mxfp8' || dtype === 'mx_fp8') return { dtype: 'mxfp8' };
  if (dtype === 'mxfp6' || dtype.startsWith('mx_fp6') || dtype.startsWith('mxfp6')) return { dtype: 'mxfp6' };
  if (dtype === 'fp8_e4m3' || dtype === 'fp8_e5m2') {
    const d: Dtype = dtype === 'fp8_e5m2' ? 'fp8_e5m2' : 'fp8_e4m3';
    if (qscheme === 'per_block' && block && block.length === 2) {
      return { dtype: d, extraBitsPerElem: blockScaleBits(scaleBits, block[0], block[1]) };
    }
    if (qscheme === 'per_group' && group) return { dtype: d, extraBitsPerElem: groupScaleBits(scaleBits, 0, group) };
    if (qscheme === 'per_channel') notes.push('quark fp8 per-channel weight scales: overhead ≈0');
    return { dtype: d };
  }
  if (dtype === 'int4' || dtype === 'uint4') {
    const zp = dtype === 'uint4' || spec.symmetric === false ? 4 : 0;
    if (qscheme === 'per_group' && group) return { dtype: 'int4', extraBitsPerElem: groupScaleBits(16, zp, group) };
    return { dtype: 'int4' };
  }
  if (dtype === 'int8' || dtype === 'uint8') return { dtype: 'int8' };
  if (dtype) notes.push(`unrecognised quark dtype '${dtype}'; kept native`);
  return null;
}

/** GEMM-input dtype of a Quark `input_tensors` spec, or undefined. */
export function quarkInputDtype(spec: unknown): Dtype | undefined {
  if (!isJsonObject(spec)) return undefined;
  const dtype = (strAt(spec, 'dtype') ?? '').toLowerCase();
  const group = numAt(spec, 'group_size');
  if (dtype === 'fp4' || dtype === 'mxfp4' || dtype === 'mx_fp4') {
    return group === 16 && strAt(spec, 'scale_format') !== 'e8m0' ? 'nvfp4' : 'mxfp4';
  }
  if (dtype === 'mxfp8' || dtype === 'mx_fp8') return 'mxfp8';
  if (dtype === 'fp8_e4m3') return 'fp8_e4m3';
  if (dtype === 'fp8_e5m2') return 'fp8_e5m2';
  if (dtype === 'int8' || dtype === 'uint8') return 'int8';
  if (dtype === 'int4' || dtype === 'uint4') return 'int4';
  return undefined;
}

/** Module format of a Quark layer config (`{ weight, input_tensors }`), or null. */
function quarkEntryFormat(entry: unknown, notes: string[]): ModuleQuant | null {
  if (!isJsonObject(entry)) return null;
  const weights = quarkWeightFormat(entry.weight, notes);
  if (!weights) return null;
  const input = quarkInputDtype(entry.input_tensors);
  return input === undefined ? { weights } : { weights, gemmInput: input };
}

/** KV-cache format of `kv_cache_quant_config` (any entry with an fp8 / int8 spec), or undefined. */
function quarkKvCache(kv: unknown): TensorFormat | undefined {
  if (!isJsonObject(kv)) return undefined;
  for (const v of Object.values(kv)) {
    const spec = isJsonObject(v) ? (v.output_tensors ?? v) : undefined;
    const d = quarkInputDtype(spec);
    if (d) return { dtype: d };
  }
  return undefined;
}

/** Read a Quark `quantization_config`. */
export function readQuark(qc: JsonObject, ctx: SchemeContext): SchemeReading {
  const notes: string[] = [];
  const global = quarkEntryFormat(objAt(qc, 'global_quant_config'), notes);
  const layerCfg = objAt(qc, 'layer_quant_config') ?? {};
  const layerKeys = Object.keys(layerCfg);
  const layerList = compilePatterns(layerKeys);
  const layerFormats = new Map(layerKeys.map((k) => [k, quarkEntryFormat(layerCfg[k], notes)]));
  const typeCfg = objAt(qc, 'layer_type_quant_config') ?? {};
  const linearType = quarkEntryFormat(typeCfg.Linear ?? typeCfg['torch.nn.Linear'], notes);
  const excludeRaw = strListAt(qc, 'exclude');
  const exclude = compilePatterns(excludeRaw);
  const kvCache = quarkKvCache(qc.kv_cache_quant_config);

  const resolve = (p: Probe): ModuleQuant => {
    if (anyMatch(exclude, p)) return ctx.native;
    const i = firstMatch(layerList, p);
    if (i >= 0) return layerFormats.get(layerList.entries[i].raw) ?? ctx.native;
    if (p.kind !== 'linear') return ctx.native;
    return linearType ?? global ?? ctx.native;
  };

  const overrides = new Set<string>();
  for (const f of layerFormats.values()) if (f) overrides.add(describeFormat(f));
  const label =
    `quark: ${global ? describeFormat(global) : 'no global config'}` +
    (overrides.size > 0 ? `; per-layer ${[...overrides].join(' / ')}` : '') +
    (kvCache ? `, ${kvCache.dtype} KV cache` : '');
  return {
    label,
    source: 'quantization_config',
    resolve,
    ...(kvCache ? { kvCache } : {}),
    ignoreLists: [{ label: 'exclude', patterns: excludeRaw }],
    notes,
  };
}
