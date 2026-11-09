/**
 * Native FP8 checkpoints: transformers `quant_method: "fp8"` (FineGrainedFP8: block-scaled e4m3 weights,
 * dynamic per-group activations) and `"mxfp8"` (MiniMax: `weight_block_size [1, 32]`, e8m0 scales), plus the
 * Mistral `params.json` `quantization` block.
 *
 * Formats:
 * - `weight_block_size [r, c]`: one scale per r×c block, fp32 (`weight_scale_inv`) or e8m0 with
 *   `scale_fmt: "ue8m0"` → extra = scaleBits / (r·c): 128×128 fp32 = 32/16384 ≈ 0.00195; 32×32 ue8m0 =
 *   8/1024 = 0.0078125 bit/elem.
 * - `[1, 32]` with e8m0 scales is OCP MXFP8 → dtype `mxfp8` (the scale is inside the dtype).
 * - no block size: per-tensor scales, ≈0 overhead.
 * - `activation_scheme` dynamic/static (default dynamic) → the GEMM input is fp8 (W8A8).
 * - DeepSeek `expert_dtype: "fp4"` (V4.1: inside `quantization_config`; V4: at the top level of config.json, next
 *   to it): routed experts are e2m1 with a ue8m0 scale per 32 → `mxfp4`, run with
 *   fp8 activations. MiniMax-M3.1 `moe_quant_algo: "NVFP4"`: routed experts `nvfp4` W4A4.
 * Unconverted: `modules_to_not_convert` / `ignored_layers`, the LM head and embeddings (transformers never
 * converts the output embedding), and raw parameters. `modules_to_convert` adds embedding-like modules (the
 * Qwen PLE n-gram table). See docs/simulation-model.md §2 (quant overheads).
 */

import type { Dtype } from '../../../ir/dtype';
import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import { blockScaleBits, moduleFormat, numListAt, objAt, strAt, strListAt } from './common';
import { anyMatch, compilePatterns } from './patterns';
import type { JsonObject, QuantDefaults } from '../types';

/** Options of {@link readFp8}. */
export interface Fp8Options {
  /** Source label of the reading; default `'quantization_config'`. */
  source?: QuantDefaults['source'];
  /** Mistral `params.json` `quantization` block, merged in for its scale format (`use_ue8m0_for_qscales`). */
  paramsQuantization?: JsonObject;
}

/** Read a native fp8 / mxfp8 `quantization_config`. */
export function readFp8(qc: JsonObject, ctx: SchemeContext, opts: Fp8Options = {}): SchemeReading {
  const notes: string[] = [];
  const method = strAt(qc, 'quant_method') ?? 'fp8';
  const block = numListAt(qc, 'weight_block_size');
  const paramsUe8m0 = opts.paramsQuantization?.use_ue8m0_for_qscales === true;
  const ue8m0 = strAt(qc, 'scale_fmt') === 'ue8m0' || paramsUe8m0;
  if (paramsUe8m0 && strAt(qc, 'scale_fmt') !== 'ue8m0') notes.push('fp8 scales are ue8m0 (from params.json)');
  const isMx = method === 'mxfp8' || (block?.[0] === 1 && block[1] === 32 && ue8m0);
  const fp8: Dtype = strAt(qc, 'fmt') === 'e5m2' ? 'fp8_e5m2' : 'fp8_e4m3';
  const actScheme = strAt(qc, 'activation_scheme') ?? 'dynamic';
  const actQuantized = actScheme === 'dynamic' || actScheme === 'static';

  let weight: ModuleQuant;
  let shape: string;
  if (isMx) {
    weight = moduleFormat('mxfp8', 0, actQuantized ? 'mxfp8' : undefined);
    shape = 'mxfp8 (block 1×32, e8m0 scales)';
  } else if (block && block.length === 2) {
    const scaleBits = ue8m0 ? 8 : 32;
    weight = moduleFormat(fp8, blockScaleBits(scaleBits, block[0], block[1]), actQuantized ? fp8 : undefined);
    shape = `${fp8.slice(4)} fp8, block ${block[0]}×${block[1]} ${ue8m0 ? 'ue8m0' : 'fp32'} scales`;
  } else {
    weight = moduleFormat(fp8, 0, actQuantized ? fp8 : undefined);
    shape = `${fp8.slice(4)} fp8, per-tensor scales`;
    notes.push('fp8 per-tensor scales: scale overhead ≈0');
  }

  // Routed-expert overrides.
  let experts: ModuleQuant | undefined;
  let expertsLabel = '';
  // DeepSeek-V4.1 puts expert_dtype inside quantization_config; DeepSeek-V4 next to it, at the top level.
  const expertDtype = strAt(qc, 'expert_dtype') ?? strAt(ctx.top, 'expert_dtype') ?? strAt(ctx.text, 'expert_dtype');
  if (expertDtype === 'fp4') {
    experts = moduleFormat('mxfp4', 0, actQuantized ? fp8 : undefined);
    expertsLabel = ', routed experts fp4 (mxfp4: e2m1 + ue8m0 per 32)';
    notes.push('routed experts fp4 with a ue8m0 scale per 32 elements → mxfp4, fp8 GEMM input');
  } else if (strAt(qc, 'moe_quant_algo') === 'NVFP4') {
    experts = moduleFormat('nvfp4', 0, 'nvfp4');
    expertsLabel = ', routed experts NVFP4 W4A4';
  }

  // Lookup tables: Qwen `modules_to_convert` (PLE n-gram table), DeepSeek Engram.
  const convert = compilePatterns(strListAt(qc, 'modules_to_convert'));
  let tables: ModuleQuant | undefined;
  const engramQuant = strAt(qc, 'engram_embed_quant');
  if (engramQuant === 'mxfp4') tables = moduleFormat('mxfp4');
  else if (engramQuant === 'mxfp8') tables = moduleFormat('mxfp8');
  else if (ue8m0 && numListAt(ctx.text, 'engram_layer_ids')) {
    tables = moduleFormat('mxfp8');
    notes.push('Engram table: fp8 with a ue8m0 scale per 32 elements (checkpoint layout) → mxfp8');
  }
  if (convert.entries.length > 0) {
    notes.push('modules_to_convert (lookup tables) converted with the linear-weight format and scale layout');
  }

  const notConvertRaw = [...strListAt(qc, 'modules_to_not_convert'), ...strListAt(qc, 'ignored_layers')];
  // params.json `quantize_kv_a: false` keeps the MLA kv_a projection native (both namings). These synthetic
  // entries are matched but not reported as an ignore list (only one naming exists in a given checkpoint).
  const kvA = opts.paramsQuantization?.quantize_kv_a === false ? ['wkv_a_with_mqa', 'kv_a_proj_with_mqa'] : [];
  const notConvert = compilePatterns([...notConvertRaw, ...kvA]);

  const resolve = (p: Probe): ModuleQuant => {
    if (anyMatch(notConvert, p)) return ctx.native;
    if (p.cls === 'lookup_tables') return tables ?? (anyMatch(convert, p) ? weight : ctx.native);
    if (p.kind === 'embedding' && anyMatch(convert, p)) return weight;
    if (p.kind !== 'linear' || p.cls === 'lm_head' || p.nativeHighPrecision) return ctx.native;
    if (p.cls === 'routed_experts' && experts) return experts;
    return weight;
  };

  const act = actQuantized ? `, ${actScheme} ${isMx ? 'mxfp8' : 'fp8'} activations` : '';
  return {
    label: `${method}: ${shape}${act}${expertsLabel}`,
    source: opts.source ?? 'quantization_config',
    resolve,
    ignoreLists: [{ label: 'modules_to_not_convert', patterns: notConvertRaw }],
    notes,
  };
}

/**
 * Read the Mistral `params.json` `quantization` block (consolidated checkpoints): `qformat_weight`,
 * `block_size_out` × `block_size_in`, `use_ue8m0_for_qscales`, `qdynamic_act`, `quantize_kv_a`.
 */
export function readMistralParamsQuantization(pq: JsonObject, ctx: SchemeContext): SchemeReading {
  const out = typeof pq.block_size_out === 'number' ? pq.block_size_out : undefined;
  const inn = typeof pq.block_size_in === 'number' ? pq.block_size_in : undefined;
  const qc: JsonObject = {
    quant_method: 'fp8',
    fmt: strAt(pq, 'qformat_weight') === 'fp8_e5m2' ? 'e5m2' : 'e4m3',
    activation_scheme: pq.qdynamic_act === false ? 'static' : 'dynamic',
    ...(out !== undefined && inn !== undefined ? { weight_block_size: [out, inn] } : {}),
  };
  return readFp8(qc, ctx, { source: 'params.json', paramsQuantization: pq });
}

/** `params.json` `dsa` cache flags (Mistral): `fp8_kv_cache`, `fp8_indexer`. */
export function mistralCacheFlags(paramsJson: unknown): { kv: boolean; indexer: boolean } {
  const dsa = objAt(paramsJson, 'dsa');
  return { kv: dsa?.fp8_kv_cache === true, indexer: dsa?.fp8_indexer === true };
}
