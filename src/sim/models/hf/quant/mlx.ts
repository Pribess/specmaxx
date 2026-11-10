/**
 * MLX (`mlx-community/*`): config key `quantization` (often mirrored as a `quantization_config` without
 * `quant_method`): `{ bits, group_size, mode?, recipe?, <module path>: false | { bits, group_size } }`.
 *
 * - `mode: affine` (default): `int<bits>` with a scale and a bias per group, both in the model dtype (16-bit):
 *   extra = (16 + 16) / group_size, e.g. 4-bit g64 = 32/64 = 0.5 bit/elem.
 * - `mode: mxfp4` / `nvfp4` / `mxfp8`: the MX / NV dtype (scales inside the dtype).
 * Weight-only (MLX dequantizes in the matmul kernel). Scope: every Linear and Embedding, the LM head and
 * embedding table included (mlx-lm `nn.quantize`), or only the routed experts with `recipe: experts_only`;
 * per-module entries override. See docs/simulation-model.md §2.
 */

import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import { describeFormat, groupScaleBits, intDtypeFor, moduleFormat, numAt, strAt } from './common';
import { compilePatterns, firstMatch } from './patterns';
import { isJsonObject } from '../reader';
import type { JsonObject } from '../types';

/** Keys of an MLX `quantization` object that are settings, not module paths. */
const SETTINGS = new Set(['bits', 'group_size', 'mode', 'recipe', 'quant_method']);

/** Format of one MLX setting set, or null for an unknown mode. */
function mlxFormat(bits: number, group: number, mode: string, notes: string[]): ModuleQuant | null {
  if (mode === 'mxfp4') return moduleFormat('mxfp4');
  if (mode === 'nvfp4') return moduleFormat('nvfp4');
  if (mode === 'mxfp8') return moduleFormat('mxfp8');
  if (mode !== 'affine') return null;
  const { dtype, approx } = intDtypeFor(bits);
  if (approx)
    notes.push(`MLX ${bits}-bit affine weights stored as ${dtype} (no ${bits}-bit dtype)`);
  // Formula: extra = (scaleBits + biasBits) / group_size, 16-bit scale and bias
  return moduleFormat(dtype, groupScaleBits(16, 16, group));
}

/** Read an MLX `quantization` object. */
export function readMlx(q: JsonObject, ctx: SchemeContext): SchemeReading {
  const notes: string[] = [];
  const bits = numAt(q, 'bits') ?? 4;
  const group = numAt(q, 'group_size') ?? 64;
  const mode = strAt(q, 'mode') ?? 'affine';
  const recipe = strAt(q, 'recipe');
  const base = mlxFormat(bits, group, mode, notes);
  if (!base) notes.push(`unrecognised MLX mode '${mode}'; using torch_dtype`);
  if (recipe !== undefined && recipe !== 'experts_only')
    notes.push(`unrecognised MLX recipe '${recipe}'; all layers assumed quantized`);
  if (mode === 'affine') notes.push(`MLX affine: 16-bit scale + 16-bit bias per group of ${group}`);

  const keys = Object.keys(q).filter((k) => !SETTINGS.has(k));
  const list = compilePatterns(keys);
  const overrides = new Map<string, ModuleQuant | null>();
  for (const k of keys) {
    const v = q[k];
    if (v === false) overrides.set(k, null);
    else if (isJsonObject(v)) {
      overrides.set(
        k,
        mlxFormat(
          numAt(v, 'bits') ?? bits,
          numAt(v, 'group_size') ?? group,
          strAt(v, 'mode') ?? mode,
          notes,
        ),
      );
    }
  }

  const resolve = (p: Probe): ModuleQuant => {
    const i = firstMatch(list, p);
    if (i >= 0 && overrides.has(list.entries[i].raw))
      return overrides.get(list.entries[i].raw) ?? ctx.native;
    if (!base || p.kind === 'other') return ctx.native;
    if (recipe === 'experts_only' && p.cls !== 'routed_experts') return ctx.native;
    return base;
  };
  return {
    label: `mlx ${base ? describeFormat(base) : mode} (${mode} ${bits}-bit g${group}${recipe ? `, ${recipe}` : ''})`,
    source: 'quantization (mlx)',
    resolve,
    ignoreLists: [{ label: 'mlx', patterns: keys.filter((k) => overrides.get(k) === null) }],
    notes,
  };
}
