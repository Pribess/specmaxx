/**
 * Peak lookup: which FLOP/s rate an op of a given (input, accumulate, sparsity) runs at on an engine, with a
 * documented fallback when the engine does not list that exact key.
 * See docs/simulation-model.md §4 ("Peak lookup and fallback").
 */

import type { AccumulateType, Dtype, MathFormat } from '../ir/dtype';
import type { ComputeEngine, MatmulPeak } from './types';

/** The peak an op will run at, and which table key it came from (after any fallback). */
export interface PeakLookup {
  /** Peak rate, per device. Unit: FLOP/s (OP/s for integer inputs). Dense-equivalent for `'2:4'`. */
  flopsPerSec: number;
  /** Input class of the entry used (differs from the request after an input fallback). */
  usedInput: MathFormat;
  /** Accumulator of the entry used. */
  usedAccumulate: AccumulateType;
  /** Sparsity of the entry used (`'dense'` after a sparsity fallback). */
  usedSparsity: 'dense' | '2:4';
}

/**
 * Next input class to try when an engine lists no peak for `input` (one step of the fallback chain):
 * fp4→fp6→fp8→fp16, int4→int8→fp16, bf16↔fp16, tf32→fp32. `null` = end of chain (use the vector fp32 peak).
 * Every step moves to an input that is at least as wide, so the fallback never overstates the rate.
 */
export function nextFallbackInput(input: MathFormat): MathFormat | null {
  switch (input) {
    case 'fp4':
      return 'fp6';
    case 'fp6':
      return 'fp8';
    case 'fp8':
      return 'fp16';
    case 'int4':
      return 'int8';
    case 'int8':
      return 'fp16';
    case 'bf16':
      return 'fp16';
    case 'fp16':
      return 'bf16';
    case 'tf32':
      return 'fp32';
    case 'fp32':
    case 'fp64':
      return null;
    default: {
      const _never: never = input;
      throw new Error(`unknown math format ${String(_never)}`);
    }
  }
}

/** True for integer input classes (their peaks are OP/s and their accumulator is int32). */
function isIntegerInput(input: MathFormat): boolean {
  return input === 'int8' || input === 'int4';
}

/** The entry with exactly this key, if the engine lists it. */
function findPeak(
  engine: ComputeEngine,
  input: MathFormat,
  acc: AccumulateType,
  sparsity: 'dense' | '2:4',
): MatmulPeak | undefined {
  return engine.matmulPeaks.find(
    (p) => p.input === input && p.accumulate === acc && p.sparsity === sparsity,
  );
}

/**
 * Matmul peak of `engine` for an op keyed (input, acc, sparsity), per device, in FLOP/s (OP/s for integers).
 *
 * Lookup order (docs §4 "Peak lookup and fallback"):
 * 1. The exact (input, acc, sparsity) entry.
 * 2. If `acc = 'fp16'` is missing, the `'fp32'`-accumulate entry (datacenter parts list a single rate).
 * 3. If `'2:4'` is missing, the dense entry (the op runs dense; it still reads the compressed weights).
 * 4. Otherwise walk {@link nextFallbackInput} (visited set; when the chain switches between integer and float
 *    inputs the accumulator switches to `int32` / `fp32` to match) and repeat 1–3 for each input.
 * 5. At the end of the chain: the engine's vector (non-tensor) fp32 peak.
 *
 * Any lookup that is not an exact hit pushes ONE human-readable line to `warnings` naming the requested and
 * the used key. Throws if even the vector fp32 peak is missing.
 */
export function matmulPeakFlopsPerSec(
  engine: ComputeEngine,
  input: MathFormat,
  acc: AccumulateType,
  sparsity: 'dense' | '2:4',
  warnings: string[],
): PeakLookup {
  const requested = `${input}/${acc}/${sparsity}`;
  const visited = new Set<MathFormat>();
  let candidate: MathFormat | null = input;
  while (candidate !== null && !visited.has(candidate)) {
    visited.add(candidate);
    // The accumulator follows the input class when the chain crosses integer ↔ float.
    const candidateAcc: AccumulateType = isIntegerInput(candidate)
      ? 'int32'
      : acc === 'int32'
        ? 'fp32'
        : acc;
    const accOptions: AccumulateType[] =
      candidateAcc === 'fp16' ? ['fp16', 'fp32'] : [candidateAcc];
    const sparsityOptions: ('dense' | '2:4')[] = sparsity === '2:4' ? ['2:4', 'dense'] : ['dense'];
    for (const s of sparsityOptions) {
      for (const a of accOptions) {
        const hit = findPeak(engine, candidate, a, s);
        if (hit !== undefined) {
          const used = `${hit.input}/${hit.accumulate}/${hit.sparsity}`;
          if (used !== requested) {
            warnings.push(
              `peak: engine '${engine.id}' lists no ${requested} matmul peak; using ${used} ` +
                `(${hit.flopsPerSec / 1e12} TFLOP/s)` +
                (input === 'bf16' && hit.input === 'fp16'
                  ? '; assumes a cast to FP16 (e.g. --dtype=half); FP16 range/overflow is not modeled'
                  : ''),
            );
          }
          return {
            flopsPerSec: hit.flopsPerSec,
            usedInput: hit.input,
            usedAccumulate: hit.accumulate,
            usedSparsity: hit.sparsity,
          };
        }
      }
    }
    candidate = nextFallbackInput(candidate);
  }
  const vectorFp32 = engine.vectorFlopsPerSec.fp32;
  if (vectorFp32 === undefined) {
    throw new Error(
      `engine '${engine.id}' has no matmul peak for ${requested} and no vector fp32 peak to fall back to`,
    );
  }
  warnings.push(
    `peak: engine '${engine.id}' has no tensor peak for ${requested} or its fallbacks; ` +
      `using the vector fp32 peak (${vectorFp32 / 1e12} TFLOP/s)`,
  );
  return {
    flopsPerSec: vectorFp32,
    usedInput: 'fp32',
    usedAccumulate: 'fp32',
    usedSparsity: 'dense',
  };
}

/**
 * Vector (non-tensor, SIMT) peak of `engine` for elementwise work and recurrent steps on `dtype`, per device,
 * in FLOP/s: the bf16 / fp16 entry for those dtypes when listed, else the fp32 entry (narrower formats are
 * computed in fp32 registers). Throws if the engine lists no fp32 vector peak.
 */
export function vectorPeakFlopsPerSec(engine: ComputeEngine, dtype: Dtype): number {
  const v = engine.vectorFlopsPerSec;
  if (dtype === 'bf16' && v.bf16 !== undefined) return v.bf16;
  if (dtype === 'fp16' && v.fp16 !== undefined) return v.fp16;
  if (v.fp32 === undefined) {
    throw new Error(`engine '${engine.id}' lists no vector fp32 peak (needed for ${dtype})`);
  }
  return v.fp32;
}
