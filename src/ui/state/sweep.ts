/**
 * Decode rate as a function of the batch size, for the batch-sweep chart.
 */

import { simulate } from '../../sim/engine/simulate';
import type { SimInput } from '../../sim/engine/types';
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
}

/** Points of the sweep: batch = dp · 2^k for k = 0..9 (the batch must divide over DP). */
export const SWEEP_STEPS = 10;

/**
 * Simulates `input` at batch = dp · 2^k, k = 0..{@link SWEEP_STEPS}−1, keeping everything else. A batch the
 * engine rejects is skipped. Costs ten `simulate` runs (≈ 2 ms each on the shortlist).
 */
export function batchSweep(input: SimInput): SweepPoint[] {
  const points: SweepPoint[] = [];
  for (let k = 0; k < SWEEP_STEPS; k++) {
    const batch = input.parallel.dp * 2 ** k;
    try {
      const r = simulate({ ...input, workload: { ...input.workload, batch } });
      points.push({
        batch,
        contextTokens: input.workload.decodeContextLen,
        decodeTokensPerSec: r.decode.tokensPerSec,
        perSeqTokensPerSec: r.decode.perSequenceTokensPerSec,
        fits: memoryFits({ stages: r.memory.perStage, tiers: r.memory.tiers }),
      });
    } catch {
      // Invalid at this batch: no point.
    }
  }
  return points;
}
