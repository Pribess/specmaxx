/**
 * Integer group quantization with packed zero points: GPTQ (`quant_method: "gptq"`, gptqmodel / auto-gptq),
 * AWQ (`"awq"`) and Intel AutoRound (`"auto-round"`).
 *
 * Storage: `int<bits>` weights, one fp16 scale and one `bits`-wide packed zero point (`qzeros`) per group of
 * `group_size` input elements. GPTQ / AutoRound in the `auto_gptq` layout store `qzeros` even when symmetric.
 * Overhead = (16 + bits) / group_size: g128 = 20/128 = 0.15625, g32 = 20/32 = 0.625 bit/elem.
 * `group_size: -1` (per channel) → ≈0. Weight-only (the GEMM runs on the dequantized weights), unless
 * AutoRound `act_bits` < 16.
 * Scope: Linear modules of the decoder layers (GPTQ / AWQ never touch the LM head unless `lm_head: true`);
 * AutoRound only under `block_name_to_quantize`, with `extra_config` overrides per module (exact names, or
 * regexes searched anywhere), and AWQ / GPTQ `modules_to_not_convert`. See docs/simulation-model.md §2.
 */

import type { Dtype } from '../../../ir/dtype';
import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import {
  describeFormat,
  groupScaleBits,
  intDtypeFor,
  moduleFormat,
  numAt,
  objAt,
  strAt,
  strListAt,
} from './common';
import { anyMatch, compilePatterns, dottedContains, firstMatch } from './patterns';
import { isJsonObject } from '../reader';
import type { JsonObject, QuantDefaults } from '../types';

/** Format of an integer / AutoRound data type at `bits` with groups of `group` (or ≤0 for per-channel). */
function groupFormat(
  dataType: string,
  bits: number,
  group: number,
  actBits: number | undefined,
  notes: string[],
): ModuleQuant {
  const dt = dataType.toLowerCase();
  const act: Dtype | undefined =
    actBits !== undefined && actBits < 16
      ? actBits <= 4
        ? dt.startsWith('nv')
          ? 'nvfp4'
          : 'mxfp4'
        : 'fp8_e4m3'
      : undefined;
  if (dt.startsWith('mx') && bits === 4) return moduleFormat('mxfp4', 0, act);
  if (dt.startsWith('nv') && bits === 4) return moduleFormat('nvfp4', 0, act);
  if ((dt === 'fp' || dt === 'float' || dt.startsWith('fp8')) && bits === 8)
    return moduleFormat('fp8_e4m3', 0, act);
  const { dtype, approx } = intDtypeFor(bits);
  if (approx) notes.push(`int${bits} weights stored as ${dtype} (no ${bits}-bit dtype)`);
  if (group <= 0) {
    notes.push(`int${bits} per-channel scales: overhead ≈0`);
    return moduleFormat(dtype, 0, act);
  }
  return moduleFormat(dtype, groupScaleBits(16, bits, group), act);
}

/** Read a GPTQ `quantization_config` (also `dynamic` per-module overrides of gptqmodel). */
export function readGptq(
  qc: JsonObject,
  ctx: SchemeContext,
  source: QuantDefaults['source'] = 'quantization_config',
): SchemeReading {
  const notes: string[] = [];
  const bits = numAt(qc, 'bits') ?? 4;
  const group = numAt(qc, 'group_size') ?? 128;
  const base = groupFormat('int', bits, group, undefined, notes);
  const lmHead = qc.lm_head === true;
  const skipRaw = strListAt(qc, 'modules_to_not_convert');
  const skip = compilePatterns(skipRaw);

  // gptqmodel `dynamic`: { "-:regex": {} } skips, { "+:regex" | "regex": { bits, group_size } } overrides.
  const dynamic = objAt(qc, 'dynamic') ?? {};
  const rules: { re: RegExp; skip: boolean; format: ModuleQuant }[] = [];
  for (const [key, v] of Object.entries(dynamic)) {
    const neg = key.startsWith('-:');
    const src = key.replace(/^[+-]:/, '');
    try {
      const o = isJsonObject(v) ? v : {};
      rules.push({
        re: new RegExp(src),
        skip: neg,
        format: groupFormat(
          'int',
          numAt(o, 'bits') ?? bits,
          numAt(o, 'group_size') ?? group,
          undefined,
          notes,
        ),
      });
    } catch {
      notes.push(`gptq dynamic key '${key}' is not a valid regex; ignored`);
    }
  }

  const resolve = (p: Probe): ModuleQuant => {
    if (p.kind !== 'linear' || anyMatch(skip, p)) return ctx.native;
    if (p.cls === 'lm_head') return lmHead ? base : ctx.native;
    if (p.layer === null) return ctx.native;
    const rule = rules.find((r) => r.re.test(p.name));
    if (rule) return rule.skip ? ctx.native : rule.format;
    return base;
  };
  notes.push(`gptq: fp16 scale + ${bits}-bit packed zero point per group of ${group}`);
  return {
    label: `gptq int${bits} g${group}${qc.sym === false ? ' asym' : ' sym'}${qc.desc_act === true ? ', desc_act' : ''}`,
    source,
    resolve,
    ignoreLists: [{ label: 'modules_to_not_convert', patterns: skipRaw }],
    notes,
  };
}

/** Read an AWQ `quantization_config` (`bits` / `w_bit`, `group_size` / `q_group_size`, `zero_point`). */
export function readAwq(qc: JsonObject, ctx: SchemeContext): SchemeReading {
  const notes: string[] = [];
  const bits = numAt(qc, 'bits') ?? numAt(qc, 'w_bit') ?? 4;
  const group = numAt(qc, 'group_size') ?? numAt(qc, 'q_group_size') ?? 128;
  const { dtype } = intDtypeFor(bits);
  const zp = qc.zero_point === false ? 0 : bits;
  const base = moduleFormat(dtype, group > 0 ? groupScaleBits(16, zp, group) : 0);
  const skipRaw = strListAt(qc, 'modules_to_not_convert');
  const skip = compilePatterns(skipRaw);
  const resolve = (p: Probe): ModuleQuant =>
    p.kind !== 'linear' || p.layer === null || anyMatch(skip, p) ? ctx.native : base;
  notes.push(`awq: fp16 scale${zp ? ` + ${bits}-bit zero point` : ''} per group of ${group}`);
  return {
    label: `awq int${bits} g${group}`,
    source: 'quantization_config',
    resolve,
    ignoreLists: [{ label: 'modules_to_not_convert', patterns: skipRaw }],
    notes,
  };
}

/** Read an Intel AutoRound `quantization_config`. */
export function readAutoRound(qc: JsonObject, ctx: SchemeContext): SchemeReading {
  const notes: string[] = [];
  const bits = numAt(qc, 'bits') ?? 4;
  const group = numAt(qc, 'group_size') ?? 128;
  const dataType = strAt(qc, 'data_type') ?? 'int';
  const actBits = numAt(qc, 'act_bits');
  const base = groupFormat(dataType, bits, group, actBits, notes);
  const blocksRaw = qc.block_name_to_quantize ?? qc.to_quant_block_names;
  const blocks = (
    Array.isArray(blocksRaw) ? blocksRaw : typeof blocksRaw === 'string' ? blocksRaw.split(',') : []
  )
    .filter((b): b is string => typeof b === 'string')
    .map((b) => b.trim())
    .filter((b) => b.length > 0);

  const extra = objAt(qc, 'extra_config') ?? {};
  const keys = Object.keys(extra);
  const list = compilePatterns(keys, { regexKeys: true });
  const formats = new Map<string, ModuleQuant | 'native'>();
  for (const k of keys) {
    const o = extra[k];
    if (!isJsonObject(o)) continue;
    const b = numAt(o, 'bits') ?? bits;
    formats.set(
      k,
      b >= 16
        ? 'native'
        : groupFormat(
            strAt(o, 'data_type') ?? dataType,
            b,
            numAt(o, 'group_size') ?? group,
            numAt(o, 'act_bits') ?? actBits,
            notes,
          ),
    );
  }

  const inBlocks = (p: Probe): boolean =>
    blocks.length === 0
      ? p.layer !== null
      : blocks.some((b) => p.name.startsWith(`${b}.`) || dottedContains(p.name, b));
  const resolve = (p: Probe): ModuleQuant => {
    if (p.kind !== 'linear' || !inBlocks(p)) return ctx.native;
    const i = firstMatch(list, p);
    if (i >= 0) {
      const f = formats.get(list.entries[i].raw);
      if (f !== undefined) return f === 'native' ? ctx.native : f;
    }
    return base;
  };
  const packing = strAt(qc, 'packing_format') ?? 'auto_round';
  if (base.weights.dtype === 'int4' || base.weights.dtype === 'int8') {
    notes.push(
      `auto-round (${packing}): fp16 scale + ${bits}-bit packed zero point per group of ${group}`,
    );
  }
  return {
    label: `auto-round ${describeFormat(base)} (${dataType} ${bits}-bit g${group}${qc.sym === false ? ' asym' : ' sym'})`,
    source: 'quantization_config',
    resolve,
    ignoreLists: [
      { label: 'extra_config', patterns: keys.filter((k) => formats.get(k) === 'native') },
    ],
    notes,
  };
}
