/**
 * All-to-all achieved-bandwidth efficiency (`ParallelConfig.allToAllEfficiency` η, docs §6 "All-to-all efficiency")
 * and the dispatch / combine formats, end to end through `simulate` on a MoE mode B layout (tp 1, dp = ep = 8):
 * - identity: η absent or 1 gives the same SimResult as before (`toEqual`);
 * - η divides the all-to-all bandwidth term only; latency, other collectives and all compute are untouched;
 * - the formats scale the all-to-all bandwidth term with their bytes;
 * - validation of the new fields.
 */
import { describe, expect, it } from 'vitest';
import { B200, composeTopology } from '../hw/presets';
import type { ParallelConfig } from '../ir/model';
import type { Op } from '../ir/ops';
import { REF_MOE_30B_A3B } from '../models/reference';
import { simulate } from './simulate';
import type { OpResult, PhaseResult, SimInput, SimResult } from './types';
import { validateInput } from './validate';

function input(parallel: Partial<ParallelConfig>, nodes = 1): SimInput {
  return {
    model: REF_MOE_30B_A3B,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp: 1, pp: 1, dp: 8, ep: 8, ...parallel },
    workload: { batch: 64, promptLen: 4096, decodeContextLen: 4096 },
    hardware: {
      device: B200,
      topology: composeTopology({ deviceId: 'b200', devicesPerNode: 8, nodes }),
    },
  };
}

function run(i: SimInput): SimResult {
  expect(validateInput(i)).toEqual([]);
  return simulate(i);
}

const opResults = (p: PhaseResult): OpResult[] =>
  p.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
const isA2a = (o: Op) => o.kind === 'collective' && o.collective === 'all_to_all';
const a2aResults = (p: PhaseResult) => opResults(p).filter((r) => isA2a(r.op));

describe('identity at the defaults', () => {
  const base = run(input({}));
  it('η absent, η = 1 and absent formats give equal SimResults', () => {
    expect(run(input({ allToAllEfficiency: 1 }))).toEqual(base);
    expect(run(input({ dispatchFormat: undefined, combineFormat: undefined }))).toEqual(base);
  });
  it('the explicit formats equal to the defaults (bf16 activations: bf16 both ways) give equal results', () => {
    expect(run(input({ dispatchFormat: 'bf16', combineFormat: 'bf16' }))).toEqual(base);
  });
  it('mode A has no all-to-all: the new fields change nothing', () => {
    const a = run(input({ tp: 8, dp: 1, ep: 8 }));
    expect(
      run(input({ tp: 8, dp: 1, ep: 8, allToAllEfficiency: 0.5, dispatchFormat: 'nvfp4' })),
    ).toEqual(a);
  });
});

describe('efficiency η scales the all-to-all bandwidth term only', () => {
  const base = run(input({}));
  const half = run(input({ allToAllEfficiency: 0.5 }));

  it('every all-to-all: β′ = β/η, α unchanged, t = α + β′', () => {
    for (const phase of ['prefill', 'decode'] as const) {
      const before = a2aResults(base[phase]);
      const after = a2aResults(half[phase]);
      expect(after.length).toBe(before.length);
      expect(before.length).toBeGreaterThan(0);
      before.forEach((b, i) => {
        expect(after[i].commLatencySec).toBe(b.commLatencySec);
        expect(after[i].commBandwidthSec).toBeCloseTo(b.commBandwidthSec / 0.5, 15);
        expect(after[i].sec).toBeCloseTo(b.commLatencySec + b.commBandwidthSec / 0.5, 15);
        expect(after[i].bytes).toEqual(b.bytes);
      });
    }
  });

  it('every other op is bit-identical', () => {
    const others = (p: PhaseResult) => opResults(p).filter((r) => !isA2a(r.op));
    expect(others(half.decode)).toEqual(others(base.decode));
    expect(others(half.prefill)).toEqual(others(base.prefill));
  });

  it('a slower all-to-all never speeds a phase up; the prefill step grows when the all-to-all is on the critical path', () => {
    expect(half.decode.latencySec).toBeGreaterThanOrEqual(base.decode.latencySec);
    expect(half.prefill.latencySec).toBeGreaterThanOrEqual(base.prefill.latencySec);
    expect(half.decode.comm.bandwidthSec).toBeGreaterThan(base.decode.comm.bandwidthSec);
  });

  it('η is continuous: the bandwidth sum at η = 0.25 is that at η = 0.5 plus the all-to-all share again', () => {
    const quarter = run(input({ allToAllEfficiency: 0.25 }));
    const sum = (p: PhaseResult) => a2aResults(p).reduce((s, r) => s + r.commBandwidthSec, 0);
    expect(sum(quarter.decode)).toBeCloseTo(2 * sum(half.decode), 12);
  });
});

describe('dispatch / combine formats change the all-to-all bytes, hence its bandwidth term', () => {
  const base = run(input({}));
  const nvfp4 = run(input({ dispatchFormat: 'nvfp4' }));

  it('NVFP4 dispatch: the dispatch bytes are 0.5625/2 of bf16; combine and the rest unchanged', () => {
    const bytes = (r: SimResult) =>
      a2aResults(r.decode).map((x) => (x.op.kind === 'collective' ? x.op.messageBytes : 0));
    const b = bytes(base);
    const n = bytes(nvfp4);
    // ops alternate dispatch, combine per MoE layer (the head's sampling all-to-all is not part of this model's layer)
    b.forEach((v, i) => {
      const isDispatch = a2aResults(base.decode)[i].op.name === 'moe.dispatch';
      expect(n[i]).toBeCloseTo(isDispatch ? (v * 0.5625) / 2 : v, 9);
    });
    expect(nvfp4.decode.comm.bandwidthSec).toBeLessThan(base.decode.comm.bandwidthSec);
  });
});

describe('validation', () => {
  const errors = (p: Partial<ParallelConfig>) => validateInput(input(p));
  it.each([0, -0.5, 1.01, Number.NaN, Infinity])('allToAllEfficiency %s is an error', (eta) => {
    expect(errors({ allToAllEfficiency: eta }).join('\n')).toMatch(
      /parallel\.allToAllEfficiency must be a number in \(0, 1\]/,
    );
  });
  it.each([0.01, 0.7, 1])('allToAllEfficiency %s is valid', (eta) => {
    expect(errors({ allToAllEfficiency: eta })).toEqual([]);
  });
  it('an unknown format is an error; every known one is valid', () => {
    expect(errors({ dispatchFormat: 'fp8' as never }).join('\n')).toMatch(
      /parallel\.dispatchFormat must be one of/,
    );
    expect(errors({ combineFormat: 'int3' as never }).join('\n')).toMatch(
      /parallel\.combineFormat must be one of/,
    );
    for (const f of ['bf16', 'fp8_e4m3', 'fp8_block128', 'mxfp8', 'nvfp4', 'mxfp4'] as const) {
      expect(errors({ dispatchFormat: f, combineFormat: f })).toEqual([]);
    }
  });
});
