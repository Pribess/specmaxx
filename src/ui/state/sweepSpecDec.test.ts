/**
 * The batch sweep with speculative decoding on: the spec-dec-off rate and speedup per point, and the break-even batch
 * where it stops paying (docs/simulation-model.md §13 "Break-even batch"). Bundled default model (Qwen3.8-27B, MTP
 * head) at the default hardware and context.
 */
import { describe, expect, it } from 'vitest';
import { simulate } from '../../sim/engine/simulate';
import { breakEvenBracket } from '../../sim/engine/specdec';
import type { SimInput } from '../../sim/engine/types';
import { DEFAULT_STATE, reduce } from './appState';
import { buildSimInput } from './buildInput';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import type { SpecDecChoice } from './deployment';
import { memoryFits } from './memoryVerdict';
import { batchSweep, speculativeBreakEvenBatch, SWEEP_STEPS } from './sweep';
import { poolInput, type DisaggregatedInput } from '../../sim/engine/disaggregation';
import { inlineRunner } from '../sim/runner';

function inputOf(specDec: SpecDecChoice | undefined): SimInput {
  const built = buildSimInput(
    reduce(DEFAULT_STATE, { type: 'setSpecDec', specDec }),
    BUNDLED_DEFAULT_MODEL,
  );
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  return built.input;
}

/** The single-pool input of the default state with these actions applied (spec-dec MTP k 7). */
function inputWith(actions: Parameters<typeof reduce>[1][]): SimInput {
  const state = [
    ...actions,
    { type: 'setSpecDec', specDec: { method: 'mtp', k: 7 } } as Parameters<typeof reduce>[1],
  ].reduce(reduce, DEFAULT_STATE);
  const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  return built.input;
}

const speedupAt = (input: SimInput, batch: number): number => {
  const s = simulate({ ...input, workload: { ...input.workload, batch } }).decode.speculative;
  if (s === undefined) throw new Error('no speculative summary');
  return s.speedup;
};

describe('sweep with MTP k 7 (the e2e crossover scenario, URL sd=mtp&sdk=7)', () => {
  const input = inputOf({ method: 'mtp', k: 7 });
  const points = batchSweep(input);
  const breakEven = speculativeBreakEvenBatch(input, points);
  it('every point carries the off rate and speedup = rate / off rate', () => {
    expect(points).toHaveLength(SWEEP_STEPS);
    for (const p of points) {
      expect(p.offDecodeTokensPerSec).toBeGreaterThan(0);
      expect(p.speedup).toBe(p.decodeTokensPerSec / (p.offDecodeTokensPerSec ?? NaN));
    }
  });
  it('spec-dec pays at batch dp (speedup > 1) and not at the last point (< 1)', () => {
    expect(points[0].speedup).toBeGreaterThan(1);
    expect(points.at(-1)?.speedup).toBeLessThan(1);
  });
  it('the sweep brackets the crossing in (32, 64] and the break-even batch is 49', () => {
    const bracket = breakEvenBracket(
      points.map((p) => ({ batch: p.batch, speedup: p.speedup ?? NaN })),
    );
    expect(bracket).toEqual([32, 64]);
    expect(breakEven).toBe(49);
  });
  it('speedup(49) < 1 ≤ speedup(48): the bisection found the first losing batch', () => {
    expect(speedupAt(input, 49)).toBeLessThan(1);
    expect(speedupAt(input, 48)).toBeGreaterThanOrEqual(1);
  });
  it('speedup = E[τ] / stepCostRatio at every point (rel 1e-12)', () => {
    for (const p of points) {
      const s = simulate({ ...input, workload: { ...input.workload, batch: p.batch } }).decode
        .speculative;
      if (s === undefined) throw new Error('no speculative summary');
      expect(s.stepCostRatio).toBeGreaterThan(0);
      const viaRatio = s.expectedTokensPerStep / s.stepCostRatio;
      expect(Math.abs(viaRatio - s.speedup)).toBeLessThanOrEqual(1e-12 * s.speedup);
    }
  });
});

describe('sweep with the default MTP settings (k 3)', () => {
  it('pays at every swept batch: no break-even batch', () => {
    const input = inputOf({ method: 'mtp' });
    const points = batchSweep(input);
    expect(points.every((p) => (p.speedup ?? 0) >= 1)).toBe(true);
    expect(speculativeBreakEvenBatch(input, points)).toBeUndefined();
  });
});

describe('sweep with speculative decoding off', () => {
  const input = inputOf(undefined);
  const points = batchSweep(input);
  it('no break-even batch', () => {
    expect(speculativeBreakEvenBatch(input, points)).toBeUndefined();
  });
  it('points are exactly the plain ones (no off rate, no speedup keys)', () => {
    const expected = Array.from({ length: SWEEP_STEPS }, (_, k) => {
      const batch = input.parallel.dp * 2 ** k;
      const r = simulate({ ...input, workload: { ...input.workload, batch } });
      return {
        batch,
        contextTokens: input.workload.decodeContextLen,
        decodeTokensPerSec: r.decode.tokensPerSec,
        perSeqTokensPerSec: r.decode.perSequenceTokensPerSec,
        fits: memoryFits({ stages: r.memory.perStage, tiers: r.memory.tiers }),
      };
    });
    expect(points).toStrictEqual(expected);
  });
});

describe('with prefill/decode disaggregation (docs §14): the break-even batch is the decode pool’s', () => {
  /** A disaggregated input whose prefill pool is `prefill` and decode pool is `decode` (both single-pool inputs). */
  const pdOf = (prefill: SimInput, decode: SimInput): DisaggregatedInput => {
    const shared: DisaggregatedInput['shared'] = { ...prefill };
    delete (shared as Partial<SimInput>).hardware;
    delete (shared as Partial<SimInput>).parallel;
    return {
      shared,
      prefillPool: { hardware: prefill.hardware, parallel: prefill.parallel, batch: 1 },
      decodePool: { hardware: decode.hardware, parallel: decode.parallel, batch: 1 },
      config: { outputLen: 1024, prefillInstances: 1, decodeInstances: 1 },
    };
  };
  it('one-GPU decode pool: the same 49 as the single-pool sweep; no single-pool break-even is computed', async () => {
    const one = inputWith([]);
    const run = await inlineRunner.run(one, [], { input: pdOf(one, one), decodeNotes: [] });
    expect(run.pd?.decodeBreakEvenBatch).toBe(49);
    expect(run.breakEvenBatch).toBeUndefined();
  });
  it('PP 2 decode pool on two GPUs: spec-dec runs, and the break-even search uses that pool’s own input', async () => {
    const one = inputWith([]);
    const pp2 = inputWith([
      { type: 'setCounts', devicesPerNode: 2 },
      { type: 'setParallel', parallel: { pp: 2 } },
    ] as Parameters<typeof reduce>[1][]);
    expect(pp2.parallel.pp).toBe(2);
    const pd = pdOf(one, pp2);
    const run = await inlineRunner.run(one, [], { input: pd, decodeNotes: [] });
    const decodeInput = poolInput(pd.shared, pd.decodePool, 'decode');
    const sweep = batchSweep(decodeInput);
    expect(run.pd?.decodeSweep).toEqual(sweep);
    expect(sweep.every((p) => p.speedup !== undefined)).toBe(true);
    expect(run.pd?.decodeBreakEvenBatch).toBe(speculativeBreakEvenBatch(decodeInput, sweep));
  });
});
