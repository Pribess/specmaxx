/**
 * NVIDIA ModelOpt exports: `quant_method: "modelopt"` in config.json, or `hf_quant_config.json`
 * (`{ producer, quantization: { quant_algo, kv_cache_quant_algo, exclude_modules, quantized_layers } }`).
 * NVIDIA's DeepSeek-V4.1 NVFP4 says `quant_method: "fp8"` but is a ModelOpt export (`producer`, `quant_algo`,
 * `moe_quant_algo`, `quantized_layers`): it is read here, and the modules it leaves alone keep the native
 * fp8 format (the fp8 reader of the same config is the base).
 *
 * - `quant_algo: MIXED_PRECISION` with `quantized_layers`: only the listed modules are converted, each with its
 *   own `quant_algo`; everything else stays in the base format.
 * - a single `quant_algo` (e.g. `NVFP4`): every Linear except `ignore` / `exclude_modules` and the LM head
 *   (ModelOpt's default config disables `*lm_head*`).
 * - `MIXED_PRECISION` with only `config_groups`: read as compressed-tensors groups.
 * - `kv_cache_quant_algo: FP8` / `kv_cache_scheme {num_bits: 8, type: float}` → fp8_e4m3 KV cache.
 *
 * Algorithm → format (scales: NVFP4 e4m3 per 16 and MXFP8/MXFP4 e8m0 per 32 are inside the dtypes; the fp32
 * per-tensor `weight_scale_2` is ≈0): see {@link modeloptAlgoFormat}. See docs/simulation-model.md §2.
 */

import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import { blockScaleBits, groupScaleBits, moduleFormat, numAt, objAt, strAt, strListAt } from './common';
import { compileCtGroups, ctKvCache, matchCtGroup } from './compressedTensors';
import { anyMatch, compilePatterns, firstMatch } from './patterns';
import { isJsonObject } from '../reader';
import type { JsonObject, QuantDefaults } from '../types';

/**
 * Format of one ModelOpt `quant_algo`, or null when unknown. `entryGroup` is the `quantized_layers` entry's
 * `group_size`, `configGroup` the config's (elements). FP8 block algorithms use only the entry's (default 128):
 * the config-level `group_size` is the NVFP4 group (16), not the fp8 block.
 * - `NVFP4`, `NVFP4_AWQ`: nvfp4 W4A4. `W4A16_NVFP4`: nvfp4 weight-only.
 * - `MXFP8`: mxfp8 W8A8. `MXFP4`: mxfp4 W4A4. `W4A16_MXFP4`: mxfp4 weight-only.
 * - `FP8`: fp8 per-tensor W8A8 (≈0 overhead). `FP8_PER_CHANNEL_PER_TOKEN`: the same, per channel.
 * - `FP8_PB_WO`: fp8 128×128 blocks, fp32 scales (32/16384), weight-only. `FP8_BLOCK_SCALES`: same, W8A8.
 * - `INT4_AWQ` / `W4A16_AWQ`: int4 + one fp16 scale per group (16/g); `W4A8_AWQ`: same with fp8 input.
 * - `INT8_SQ` / `W8A8_SQ_PER_CHANNEL`: int8 W8A8.
 */
export function modeloptAlgoFormat(
  algo: string,
  entryGroup: number | undefined,
  configGroup?: number,
): ModuleQuant | null {
  const a = algo.toUpperCase();
  const block = entryGroup ?? 128;
  const groupSize = entryGroup ?? configGroup;
  // Formula: extra = 32 / block² (one fp32 scale per block × block tile)
  const fp8Block = blockScaleBits(32, block, block);
  switch (a) {
    case 'NVFP4':
    case 'NVFP4_AWQ':
      return moduleFormat('nvfp4', 0, 'nvfp4');
    case 'W4A16_NVFP4':
    case 'NVFP4_WO':
      return moduleFormat('nvfp4');
    case 'W4A8_NVFP4_FP8':
      return moduleFormat('nvfp4', 0, 'fp8_e4m3');
    case 'MXFP8':
      return moduleFormat('mxfp8', 0, 'mxfp8');
    case 'MXFP4':
      return moduleFormat('mxfp4', 0, 'mxfp4');
    case 'W4A16_MXFP4':
      return moduleFormat('mxfp4');
    case 'W4A8_MXFP4_FP8':
      return moduleFormat('mxfp4', 0, 'fp8_e4m3');
    case 'FP8':
    case 'FP8_PER_CHANNEL_PER_TOKEN':
      return moduleFormat('fp8_e4m3', 0, 'fp8_e4m3');
    case 'FP8_PB_WO':
      return moduleFormat('fp8_e4m3', fp8Block);
    case 'FP8_BLOCK_SCALES':
    case 'FP8_PB':
      return moduleFormat('fp8_e4m3', fp8Block, 'fp8_e4m3');
    case 'INT4_AWQ':
    case 'W4A16_AWQ':
      return moduleFormat('int4', groupScaleBits(16, 0, groupSize ?? 128));
    case 'W4A8_AWQ':
      return moduleFormat('int4', groupScaleBits(16, 0, groupSize ?? 128), 'fp8_e4m3');
    case 'INT8_SQ':
    case 'W8A8_SQ_PER_CHANNEL':
      return moduleFormat('int8', 0, 'int8');
    default:
      return null;
  }
}

/** True when a `quant_method: "fp8"` config is really a ModelOpt export. */
export function isModeloptFp8(qc: JsonObject): boolean {
  return (
    strAt(objAt(qc, 'producer'), 'name') === 'modelopt' ||
    qc.quant_algo !== undefined ||
    qc.quantized_layers !== undefined
  );
}

/** Options of {@link readModelopt}. */
export interface ModeloptOptions {
  /** Source label; default `'quantization_config'`. */
  source?: QuantDefaults['source'];
  /** Reading of the native checkpoint format, for modules the export leaves alone (default: unquantized). */
  base?: SchemeReading;
}

/**
 * Read a ModelOpt config: a config.json `quantization_config`, or the `quantization` object of
 * `hf_quant_config.json` (pass `hfQuantConfig.quantization`).
 */
export function readModelopt(q: JsonObject, ctx: SchemeContext, opts: ModeloptOptions = {}): SchemeReading {
  const notes: string[] = [];
  const algo = strAt(q, 'quant_algo') ?? 'NVFP4';
  const groupSize = numAt(q, 'group_size');
  const ignoreRaw = [...strListAt(q, 'ignore'), ...strListAt(q, 'exclude_modules')];
  const ignore = compilePatterns(ignoreRaw);
  const base = opts.base;
  const baseOf = (p: Probe): ModuleQuant => (base ? base.resolve(p) : ctx.native);

  const layersObj = objAt(q, 'quantized_layers');
  const algosUsed = new Set<string>();
  let resolve: (p: Probe) => ModuleQuant;

  if (layersObj) {
    const keys = Object.keys(layersObj);
    const list = compilePatterns(keys);
    const formats = keys.map((k) => {
      const e = layersObj[k];
      const a = strAt(e, 'quant_algo');
      if (!a) return null;
      const f = modeloptAlgoFormat(a, numAt(e, 'group_size'), groupSize);
      if (!f) notes.push(`unrecognised ModelOpt quant_algo '${a}' (${k}); kept native`);
      return f ? { algo: a, format: f } : null;
    });
    const byKey = new Map(keys.map((k, i) => [k, formats[i]]));
    resolve = (p) => {
      const i = firstMatch(list, p);
      return (i >= 0 ? byKey.get(list.entries[i].raw)?.format : undefined) ?? baseOf(p);
    };
    for (const f of formats) if (f) algosUsed.add(f.algo);
  } else if (algo.toUpperCase() === 'MIXED_PRECISION' && objAt(q, 'config_groups')) {
    const groups = compileCtGroups(q, ctx, notes, false);
    resolve = (p) => (anyMatch(ignore, p) ? baseOf(p) : (matchCtGroup(groups, p)?.format ?? baseOf(p)));
    for (const g of groups) algosUsed.add(g.label);
  } else {
    const f = modeloptAlgoFormat(algo, undefined, groupSize);
    if (!f) notes.push(`unrecognised ModelOpt quant_algo '${algo}'; kept native`);
    algosUsed.add(algo);
    resolve = (p) => {
      if (!f || p.kind !== 'linear' || p.cls === 'lm_head' || anyMatch(ignore, p)) return baseOf(p);
      return f;
    };
  }

  const kvAlgo = q.kv_cache_quant_algo;
  let kvCache = ctKvCache(q.kv_cache_scheme);
  if (typeof kvAlgo === 'string' && kvAlgo.toUpperCase() === 'FP8') kvCache = { dtype: 'fp8_e4m3' };
  if (typeof kvAlgo === 'string' && kvAlgo.toUpperCase() === 'NVFP4') kvCache = { dtype: 'nvfp4' };
  if (isJsonObject(q.modelopt_quant_config)) notes.push('hf_quant_config modelopt_quant_config.quant_cfg not read (exclude_modules used)');
  if (base) notes.push(`${base.label}: base format of the modules the ModelOpt export leaves alone`);

  const mixed = algo.toUpperCase() === 'MIXED_PRECISION' ? ` (${[...algosUsed].join(', ')})` : '';
  return {
    label: `modelopt ${algo}${mixed}${kvCache ? `, ${kvCache.dtype} KV cache` : ''}`,
    source: opts.source ?? 'quantization_config',
    resolve,
    ...(kvCache ? { kvCache } : {}),
    ignoreLists: [{ label: 'ignore', patterns: ignoreRaw }],
    notes: [...notes, ...(base?.notes ?? [])],
  };
}
