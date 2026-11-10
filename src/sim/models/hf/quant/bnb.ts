/**
 * bitsandbytes (`quant_method: "bitsandbytes"`): weight-only 4-bit (NF4 / FP4) or 8-bit (LLM.int8()).
 *
 * - 4-bit: one absmax scale per block of 64 weights. Plain: an fp32 absmax → 32/64 = 0.5 bit/elem. With
 *   `bnb_4bit_use_double_quant` the absmax is itself 8-bit, with one fp32 scale per 256 absmax values:
 *   8/64 + 32/(64·256) ≈ 0.127 bit/elem. NF4 / FP4 codes are 4-bit lookups, stored here as `int4`; the GEMM
 *   runs on weights dequantized to `bnb_4bit_compute_dtype` (the activations dtype).
 * - 8-bit: `int8` with one fp32 scale per output row (≈0); the outlier columns LLM.int8() keeps in fp16 are
 *   not modeled.
 * Scope: every Linear except `llm_int8_skip_modules` and the LM head. See docs/simulation-model.md §2.
 */

import type { ModuleQuant } from '../../../ir/model';
import type { Probe, SchemeContext, SchemeReading } from './common';
import { groupScaleBits, modelDtypeOf, moduleFormat, strAt, strListAt } from './common';
import { anyMatch, compilePatterns } from './patterns';
import type { JsonObject } from '../types';

/** Weights per bitsandbytes 4-bit absmax block. Unit: elements. */
export const BNB_BLOCK_SIZE = 64;
/** Absmax values per second-level (double-quant) block. Unit: elements. */
export const BNB_DOUBLE_QUANT_BLOCK = 256;

/** Read a bitsandbytes `quantization_config`. */
export function readBitsAndBytes(qc: JsonObject, ctx: SchemeContext): SchemeReading {
  const notes: string[] = [];
  const four = qc.load_in_4bit === true || qc._load_in_4bit === true;
  const type = strAt(qc, 'bnb_4bit_quant_type') ?? 'fp4';
  const double = qc.bnb_4bit_use_double_quant === true;
  let weight: ModuleQuant;
  if (four) {
    // Formula: extra = 8 / 64 + 32 / (64 · 256) with double quant, else 32 / 64
    // See docs/simulation-model.md §2 (quant overheads)
    const extra = double
      ? groupScaleBits(8, 0, BNB_BLOCK_SIZE) +
        groupScaleBits(32, 0, BNB_BLOCK_SIZE * BNB_DOUBLE_QUANT_BLOCK)
      : groupScaleBits(32, 0, BNB_BLOCK_SIZE);
    weight = moduleFormat('int4', extra);
    notes.push(
      `bitsandbytes ${type}: 4-bit codes stored as int4, dequantized for the GEMM (weight-only)`,
    );
  } else {
    weight = moduleFormat('int8');
    notes.push(
      'bitsandbytes 8-bit: per-row fp32 scales ≈0; LLM.int8() fp16 outlier columns not modeled',
    );
  }
  const compute = modelDtypeOf(qc.bnb_4bit_compute_dtype, ctx.modelDtype);
  const skipRaw = strListAt(qc, 'llm_int8_skip_modules');
  const skip = compilePatterns(skipRaw);
  const resolve = (p: Probe): ModuleQuant =>
    p.kind !== 'linear' || p.cls === 'lm_head' || anyMatch(skip, p) ? ctx.native : weight;
  return {
    label: four
      ? `bitsandbytes ${type}${double ? ' + double quant' : ''} (blocksize ${BNB_BLOCK_SIZE})`
      : 'bitsandbytes int8',
    source: 'quantization_config',
    resolve,
    activations: compute,
    ignoreLists: [{ label: 'llm_int8_skip_modules', patterns: skipRaw }],
    notes,
  };
}
