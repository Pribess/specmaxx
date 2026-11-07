import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_NVLINK } from '../__fixtures__/synthetic';
import type { QuantConfig } from '../ir/model';
import { REF_DENSE_8B, TINY_DENSE } from '../models/reference';
import type { TopologySpec } from '../hw/types';
import { singleDevice } from '../hw/presets';
import { microbatching, phaseResult, stageResult } from './pipeline';
import { simulate } from './simulate';
import type { SimInput } from './types';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

function input(over: Partial<SimInput>): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 512, decodeContextLen: 1024 },
    hardware: { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU) },
    ...over,
  };
}

describe('microbatching', () => {
  it('m = min(pp, B_r), b_mb = ceil(B_r / m)', () => {
    expect(microbatching(2, 2)).toEqual({ microbatches: 2, microbatchSize: 1 });
    expect(microbatching(4, 2)).toEqual({ microbatches: 2, microbatchSize: 1 });
    expect(microbatching(2, 5)).toEqual({ microbatches: 2, microbatchSize: 3 });
    expect(microbatching(1, 8)).toEqual({ microbatches: 1, microbatchSize: 8 });
  });
});

describe('T12 PP 2, batch 2 (REF_DENSE_8B on SYNTH_GPU × SYNTH_CLUSTER)', () => {
  const single = simulate(input({}));
  const pp2 = simulate(
    input({ parallel: { tp: 1, pp: 2, dp: 1, ep: 1 }, workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 }, hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER } }),
  );

  it('decode cycle = max(2·max t_s, Σ t_s + p2p), from the result’s own stages', () => {
    const [s0, s1] = pp2.decode.stages;
    expect(pp2.decode.microbatches).toBe(2);
    expect(pp2.decode.microbatchSize).toBe(1);
    expect(s0.sendToNext?.linkLevel).toBe('intra'); // ranks 0 → 1 share node 0
    expect(s1.sendToNext).toBeUndefined();
    const p2p = s0.sendToNext?.sec ?? NaN;
    expectClose(pp2.decode.cycleSec, Math.max(2 * Math.max(s0.sec, s1.sec), s0.sec + s1.sec + p2p));
    expectClose(pp2.decode.tokensPerSec, 2 / pp2.decode.cycleSec);
    expectClose(pp2.decode.perSequenceTokensPerSec, 1 / pp2.decode.cycleSec);
  });

  it('decode throughput vs single (batch 1) ∈ [1.75, 2.0]', () => {
    // Stage 1 carries the LM head (1.24 GB of 15.1 GB streamed), so the slower stage is ≈ 0.54 of the single
    // device: 2 / (2 · 0.54) ≈ 1.85.
    const ratio = pp2.decode.tokensPerSec / single.decode.tokensPerSec;
    expect(ratio).toBeGreaterThanOrEqual(1.75);
    expect(ratio).toBeLessThanOrEqual(2.0);
  });

  it('prefill: cycle = m·max t_s; TTFT = Σ t_s + p2p + (m−1)·max t_s', () => {
    const [s0, s1] = pp2.prefill.stages;
    const maxT = Math.max(s0.sec, s1.sec);
    expectClose(pp2.prefill.cycleSec, 2 * maxT);
    expectClose(pp2.prefill.latencySec, s0.sec + s1.sec + (s0.sendToNext?.sec ?? NaN) + maxT);
    expectClose(pp2.prefill.tokensPerSec, (2 * 512) / pp2.prefill.cycleSec);
    expectClose(pp2.prefill.perSequenceTokensPerSec, 512 / pp2.prefill.latencySec);
  });
});

describe('T13 DP 2, batch 2', () => {
  it('aggregate = 2 × single exactly; per-sequence rates equal', () => {
    const single = simulate(input({}));
    const dp2 = simulate(
      input({ parallel: { tp: 1, pp: 1, dp: 2, ep: 1 }, workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 }, hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER } }),
    );
    for (const phase of ['prefill', 'decode'] as const) {
      expectClose(dp2[phase].tokensPerSec, 2 * single[phase].tokensPerSec, 1e-12);
      expectClose(dp2[phase].perSequenceTokensPerSec, single[phase].perSequenceTokensPerSec, 1e-12);
    }
    expect(dp2.devicesUsed).toBe(2);
  });
});

describe('stageResult overlap rules', () => {
  it("'none' adds the PP send into t_s; 'ideal' leaves it to the latency path", () => {
    const r = simulate(
      input({ parallel: { tp: 1, pp: 2, dp: 1, ep: 1 }, workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 }, hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER }, options: { overlap: 'none' } }),
    );
    const s0 = r.decode.stages[0];
    const send = s0.sendToNext?.sec ?? NaN;
    expectClose(s0.sec, s0.computeSec + s0.commLatencySec + s0.commBandwidthSec + send);
    // Rebuilding with 'ideal' from the same layers drops the send from t_s.
    const ideal = stageResult(s0.index, s0.layers, s0.sendToNext, 'ideal');
    expectClose(ideal.sec, s0.computeSec); // tp = 1: no collectives inside the stage
    // Under 'none' the decode cycle does not add the sends a second time.
    const phase = phaseResult(r.decode.stages, {
      phase: 'decode',
      dp: 1,
      sequencesPerReplica: 2,
      microbatches: 2,
      microbatchSize: 1,
      promptLen: 512,
      overlap: 'none',
    });
    const [a, b] = r.decode.stages;
    expectClose(phase.cycleSec, Math.max(2 * Math.max(a.sec, b.sec), a.sec + b.sec));
  });
});

describe('PP link bandwidth bounds the cycle (review P1): TINY_DENSE pp 2, batch 2, prompt 128, link 1 B/s', () => {
  // Every per-stage time is microseconds; the 1 B/s link dominates by ~11 orders of magnitude, so the
  // expected values below are the link terms alone (rel 1e-6 tolerance covers the stage times and α = 1 µs).
  const slow: TopologySpec = { ...SYNTH_CLUSTER, intraNode: { ...SYNTH_NVLINK, bandwidthPerDirectionBytesPerSec: 1 } };
  const run = (overlap: 'ideal' | 'none') =>
    simulate(
      input({
        model: TINY_DENSE,
        parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
        workload: { batch: 2, promptLen: 128, decodeContextLen: 256 },
        hardware: { device: SYNTH_GPU, topology: slow },
        options: { overlap },
      }),
    );

  for (const overlap of ['ideal', 'none'] as const) {
    it(`${overlap}: prefill cycle ≥ 2 microbatches · 65,536 B / 1 B/s = 131,072 s; TTFT ≈ 131,072 s`, () => {
      const r = run(overlap);
      expect(r.prefill.microbatches).toBe(2); // m = min(pp 2, B_r 2), b_mb = 1
      // Send per microbatch: T·h·bpe = (1 seq · 128 tokens) · 256 · 2 B = 65,536 B → 65,536 s at 1 B/s.
      expect(r.prefill.stages[0].sendToNext?.op.kind === 'collective' && r.prefill.stages[0].sendToNext.op.messageBytes).toBe(65_536);
      // The link serves both microbatches each cycle: cycle ≥ 2 · 65,536 = 131,072 s.
      expect(r.prefill.cycleSec).toBeGreaterThanOrEqual(131_072);
      expectClose(r.prefill.cycleSec, 131_072, 1e-6);
      // tokens/s = B_r · promptLen / cycle = 2 · 128 / 131,072 = 1/512
      expectClose(r.prefill.tokensPerSec, 1 / 512, 1e-6);
      // TTFT: the first microbatch's send (65,536 s) + the second waits one link slot (65,536 s) = 131,072 s.
      expect(r.prefill.latencySec).toBeGreaterThanOrEqual(131_072);
      expectClose(r.prefill.latencySec, 131_072, 1e-6);
    });

    it(`${overlap}: decode cycle = 2 · 512 B / 1 B/s = 1,024 s`, () => {
      const r = run(overlap);
      // Send per microbatch: 1 seq · 1 token · 256 · 2 B = 512 B → 512 s; two microbatches per cycle.
      expect(r.decode.cycleSec).toBeGreaterThanOrEqual(1_024);
      expectClose(r.decode.cycleSec, 1_024, 1e-6);
      expectClose(r.decode.tokensPerSec, 2 / 1_024, 1e-6);
    });
  }
});
