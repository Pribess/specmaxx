/**
 * Decode rate as a function of the batch size, for the batch-sweep chart, and the batch where speculative decoding
 * stops paying (docs/simulation-model.md §13 "Break-even batch").
 */

import { simulate } from '../../sim/engine/simulate';
import { breakEvenBracket } from '../../sim/engine/specdec';
import type { SimInput, SimResult } from '../../sim/engine/types';
import { memoryFits } from './memoryVerdict';

/** One point of the sweep. */
export interface SweepPoint {
  /** Tokens already cached per sequence at this decode step. Unit: tokens. */
  contextTokens: number;
  /** Global batch (sequences). Count. */
  batch: number;
  /** Aggregate decode rate. Unit: tokens/s. */
  decodeTokensPerSec: number;
  /** Decode rate per sequence. Unit: tokens/s. */
  perSeqTokensPerSec: number;
  /** True when every stage and off-device tier fits at this batch (the same verdict as the main result). */
  fits: boolean;
  /** Speculative decoding on only: the aggregate decode rate of the same input with it off. Unit: tokens/s. */
  offDecodeTokensPerSec?: number;
  /** Speculative decoding on only: `decodeTokensPerSec / offDecodeTokensPerSec`. Dimensionless. */
  speedup?: number;
}

/** Points of the sweep: batch = dp · 2^k for k = 0..9 (the batch must divide over DP). */
export const SWEEP_STEPS = 10;

/** `input` at another global batch. */
function atBatch(input: SimInput, batch: number): SimInput {
  return { ...input, workload: { ...input.workload, batch } };
}

/** The sweep point of one simulated batch; the spec-dec fields only when speculative decoding is on. */
function sweepPoint(input: SimInput, batch: number, r: SimResult): SweepPoint {
  const point: SweepPoint = {
    batch,
    contextTokens: input.workload.decodeContextLen,
    decodeTokensPerSec: r.decode.tokensPerSec,
    perSeqTokensPerSec: r.decode.perSequenceTokensPerSec,
    fits: memoryFits({ stages: r.memory.perStage, tiers: r.memory.tiers }),
  };
  const spec = r.decode.speculative;
  if (spec !== undefined) {
    point.offDecodeTokensPerSec = spec.offTokensPerSec;
    point.speedup = spec.speedup;
  }
  return point;
}

/**
 * Simulates `input` at batch = dp · 2^k, k = 0..{@link SWEEP_STEPS}−1, keeping everything else. A batch the
 * engine rejects is skipped. Costs ten `simulate` runs (≈ 2 ms each on the shortlist).
 */
export function batchSweep(input: SimInput): SweepPoint[] {
  const points: SweepPoint[] = [];
  for (let k = 0; k < SWEEP_STEPS; k++) {
    const batch = input.parallel.dp * 2 ** k;
    try {
      points.push(sweepPoint(input, batch, simulate(atBatch(input, batch))));
    } catch {
      // Invalid at this batch: no point.
    }
  }
  return points;
}

/**
 * The smallest global batch at which speculative decoding stops paying (speedup < 1), or undefined when the sweep
 * shows no crossing (spec-dec off, or it pays at every swept batch). Unit: sequences.
 *
 * Takes the first bracket `(lo, hi]` of the sweep (`specdec.ts: breakEvenBracket`) and bisects over the multiples
 * of dp inside it with `simulate` (at most log2((hi − lo)/dp) ≈ 9 extra runs). It assumes ONE crossing inside the
 * bracket: shape quantization can make the speedup wiggle, and then the first crossing the bisection finds is
 * returned. A batch the engine rejects counts as "still pays", which keeps the result inside the bracket.
 * See docs/simulation-model.md §13 "Break-even batch".
 */
export function speculativeBreakEvenBatch(
  input: SimInput,
  points: readonly SweepPoint[],
): number | undefined {
  if (input.deployment?.speculative === undefined) return undefined;
  const bracket = breakEvenBracket(
    points.flatMap((p) =>
      p.speedup === undefined ? [] : [{ batch: p.batch, speedup: p.speedup }],
    ),
  );
  if (bracket === undefined) return undefined;
  const dp = input.parallel.dp;
  // Bisect over n = batch/dp: speedup(lo) ≥ 1, speedup(hi) < 1.
  let lo = bracket[0] / dp;
  let hi = bracket[1] / dp;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    let pays = true;
    try {
      const speedup = simulate(atBatch(input, mid * dp)).decode.speculative?.speedup;
      pays = speedup === undefined || speedup >= 1;
    } catch {
      pays = true;
    }
    if (pays) lo = mid;
    else hi = mid;
  }
  return hi * dp;
}
