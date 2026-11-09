/**
 * Quant-default derivation: read a repo's quantization metadata and pick the formats to simulate with.
 *
 * Source, first found wins:
 * 1. `params.json` `quantization_config` (Mistral consolidated exports; the repo has no config.json of its own,
 *    so config.json's block would be the base model's);
 * 2. config.json `quantization_config` (top level, else `text_config` / `llm_config`), dispatched on
 *    `quant_method` (`modelopt` and `modelopt_mixed` are ModelOpt); one without `quant_method` but with `bits` +
 *    `group_size` is MLX;
 * 3. `hf_quant_config.json` (ModelOpt);
 * 4. `quantization_config.json`;
 * 5. the MLX `quantization` key;
 * 6. `params.json` `quantization` (native Mistral fp8);
 * 7. none: everything in `torch_dtype` / `dtype`.
 * Mistral `params.json` `dsa.fp8_kv_cache` / `dsa.fp8_indexer` set fp8 KV / indexer-key caches on top.
 *
 * Every reader answers per probe (`moduleClass.ts`); `assemble.ts` decides per class. The result is plain data
 * the UI may override. Never throws: unknown methods and fields become notes.
 */

import type { SchemeContext, SchemeReading } from './common';
import { modelDtypeOf, objAt, strAt } from './common';
import { readBitsAndBytes } from './bnb';
import { readExl3 } from './exl3';
import { readCompressedTensors } from './compressedTensors';
import { mistralCacheFlags, readFp8, readMistralParamsQuantization } from './fp8';
import { readAutoRound, readAwq, readGptq } from './gptq';
import { readMlx } from './mlx';
import { isModeloptFp8, readModelopt, readModeloptMixed } from './modelopt';
import { buildProbes } from './moduleClass';
import { readQuark } from './quark';
import { assemble } from './assemble';
import { isJsonObject } from '../reader';
import type { JsonObject, QuantDefaults } from '../types';

/** Inputs of {@link deriveQuantDefaults}: the configs and optional quant files of one repo. */
export interface QuantInputs {
  /** Top-level config.json. */
  top: JsonObject;
  /** Language-model config (`text_config` or the flat config). */
  text: JsonObject;
  /** Parsed `hf_quant_config.json` (NVIDIA ModelOpt), if present. */
  hfQuantConfig?: unknown;
  /** Parsed `params.json` (Mistral consolidated format), if present. */
  paramsJson?: unknown;
  /** Parsed `quantization_config.json`, if present. */
  quantizationConfigJson?: unknown;
  /** The MLX `quantization` key of config.json, if present. */
  mlx?: unknown;
}

/** True for an MLX-style quantization object (`bits` and `group_size`, no `quant_method`). */
function isMlxShape(q: JsonObject): boolean {
  return q.quant_method === undefined && typeof q.bits === 'number' && typeof q.group_size === 'number';
}

/** Reader for a `quantization_config`-shaped object, by `quant_method`; null for an unknown method. */
function readByMethod(
  qc: JsonObject,
  ctx: SchemeContext,
  source: QuantDefaults['source'],
  paramsJson: unknown,
): SchemeReading | null {
  const method = (strAt(qc, 'quant_method') ?? '').toLowerCase();
  const paramsQuantization = objAt(paramsJson, 'quantization');
  switch (method) {
    case 'fp8':
    case 'mxfp8': {
      const native = readFp8(qc, ctx, { source, ...(paramsQuantization ? { paramsQuantization } : {}) });
      return method === 'fp8' && isModeloptFp8(qc) ? readModelopt(qc, ctx, { source, base: native }) : native;
    }
    case 'modelopt':
      return readModelopt(qc, ctx, { source });
    case 'modelopt_mixed':
      return readModeloptMixed(qc, ctx, { source });
    case 'compressed-tensors':
      return readCompressedTensors(qc, ctx, source);
    case 'quark':
      return { ...readQuark(qc, ctx), source };
    case 'gptq':
      return readGptq(qc, ctx, source);
    case 'awq':
      return { ...readAwq(qc, ctx), source };
    case 'auto-round':
    case 'autoround':
    case 'intel/auto-round':
      return { ...readAutoRound(qc, ctx), source };
    case 'exl3':
      return { ...readExl3(qc, ctx), source };
    case 'bitsandbytes':
      return { ...readBitsAndBytes(qc, ctx), source };
    case 'mlx':
      return readMlx(qc, ctx);
    case '':
      return isMlxShape(qc) ? readMlx(qc, ctx) : null;
    default:
      return null;
  }
}

/** A reading that converts nothing. */
function unquantized(ctx: SchemeContext, notes: string[]): SchemeReading {
  return {
    label: `${ctx.modelDtype} (unquantized)`,
    source: 'none',
    resolve: () => ctx.native,
    ignoreLists: [],
    notes,
  };
}

/** Derive the default numeric formats of a repo from its quantization metadata. */
export function deriveQuantDefaults(input: QuantInputs): QuantDefaults {
  const { top, text } = input;
  const modelDtype = modelDtypeOf(text.dtype, text.torch_dtype, top.dtype, top.torch_dtype);
  const ctx: SchemeContext = { modelDtype, native: { weights: { dtype: modelDtype } }, text, top };
  const notes: string[] = [];

  const paramsQc = objAt(input.paramsJson, 'quantization_config');
  const configQc = objAt(top, 'quantization_config') ?? objAt(text, 'quantization_config');
  const hfQuant = objAt(input.hfQuantConfig, 'quantization');
  const qcJson = isJsonObject(input.quantizationConfigJson) ? input.quantizationConfigJson : undefined;
  const mlx = isJsonObject(input.mlx) ? input.mlx : undefined;
  const paramsQuant = objAt(input.paramsJson, 'quantization');

  /** Read a `quantization_config`-shaped object; an unknown method gives a note and no reading. */
  const read = (qc: JsonObject, source: QuantDefaults['source']): { reading: SchemeReading | null; meta: unknown } => {
    const reading = readByMethod(qc, ctx, source, input.paramsJson);
    if (!reading) {
      notes.push(`unrecognised quantization '${strAt(qc, 'quant_method') ?? '?'}' (${source}); using torch_dtype`);
    }
    return { reading, meta: qc };
  };

  if (configQc && Object.keys(configQc).length === 0) notes.push('quantization_config is empty; ignored');
  const usableConfig = configQc !== undefined && Object.keys(configQc).length > 0 ? configQc : undefined;

  // The first source found wins (see the module comment).
  let picked: { reading: SchemeReading | null; meta: unknown } = { reading: null, meta: undefined };
  if (paramsQc) picked = read(paramsQc, 'params.json');
  else if (usableConfig && mlx && isMlxShape(usableConfig)) picked = { reading: readMlx(mlx, ctx), meta: mlx };
  else if (usableConfig) picked = read(usableConfig, 'quantization_config');
  else if (hfQuant) {
    picked = { reading: readModelopt(hfQuant, ctx, { source: 'hf_quant_config.json' }), meta: input.hfQuantConfig };
  } else if (qcJson) picked = read(qcJson, 'quantization_config.json');
  else if (mlx) picked = { reading: readMlx(mlx, ctx), meta: mlx };
  else if (paramsQuant) picked = { reading: readMistralParamsQuantization(paramsQuant, ctx), meta: paramsQuant };
  const { reading, meta } = picked;

  let r: SchemeReading = reading ?? unquantized(ctx, []);
  r = { ...r, notes: [...notes, ...r.notes] };

  // Mistral DSA cache flags.
  const flags = mistralCacheFlags(input.paramsJson);
  if (flags.kv && !r.kvCache) {
    r = { ...r, kvCache: { dtype: 'fp8_e4m3' }, notes: [...r.notes, 'params.json dsa.fp8_kv_cache: fp8 KV cache'] };
  }
  if (flags.indexer && !r.indexerKeyCache) {
    r = {
      ...r,
      indexerKeyCache: { dtype: 'fp8_e4m3' },
      notes: [...r.notes, 'params.json dsa.fp8_indexer: fp8 indexer key cache'],
    };
  }

  const probes = buildProbes(top, text, meta === undefined ? '' : JSON.stringify(meta), input.paramsJson);
  const out = assemble(r, probes, ctx);
  return { ...out, notes: [...new Set(out.notes)] };
}
