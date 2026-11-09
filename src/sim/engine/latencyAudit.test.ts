import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_IB } from '../__fixtures__/synthetic';
import { REF_DENSE_8B } from '../models/reference';
import { lowerModel, type LoweringContext } from '../lower/lower';
import { simulate } from './simulate';
import type { SimInput } from './types';

const ctx: LoweringContext = {
  phase: 'decode',
  seqs: 2,
  qLen: 3,
  kvLen: 1024,
  quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
  parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
};
function input(): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: ctx.quant,
    parallel: ctx.parallel,
    workload: { batch: 1, promptLen: 128, decodeContextLen: 1024 },
    hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
  };
}

describe('G2 PP sampled-token loop-back', () => {
  it('decode returns b·qLen·4 = 2·3·4 = 24 B; prefill has no return', () => {
    expect(lowerModel(REF_DENSE_8B, ctx).stages.at(-1)).toHaveProperty(
      'sendToFirst.messageBytes',
      24,
    );
    expect(lowerModel(REF_DENSE_8B, ctx).stages.at(-1)).toHaveProperty(
      'sendToFirst.dtype',
      'int32',
    );
    expect(lowerModel(REF_DENSE_8B, { ...ctx, phase: 'prefill' }).stages.at(-1)).not.toHaveProperty(
      'sendToFirst',
    );
  });
  it.each(['ideal', 'none'] as const)(
    '%s: pp2 adds 1 µs + 4/100 GB/s exactly once after the last stage',
    (overlap) => {
      const r = simulate({ ...input(), options: { overlap } });
      const stages = r.decode.stages;
      const forward = stages[0].sendToNext!;
      const path =
        stages.reduce((sum, s) => sum + s.sec, 0) + (overlap === 'ideal' ? forward.sec : 0);
      expect(r.decode.cycleSec - path).toBeCloseTo(1e-6 + 4 / 100e9, 14);
      expect(r.decode.comm.latencySec).toBeCloseTo(2e-6, 14);
      expect(r.prefill.comm.latencySec).toBeCloseTo(1e-6, 14);
    },
  );
  it('pp4 chain return 3→0 crosses 3 hops: 15 µs + 3·4/50 GB/s (existing fabric flow budget)', () => {
    const i = input();
    const r = simulate({
      ...i,
      parallel: { ...i.parallel, pp: 4 },
      hardware: {
        ...i.hardware,
        topology: {
          ...SYNTH_CLUSTER,
          devicesPerNode: 1,
          nodes: 4,
          intraNode: undefined,
          interNode: SYNTH_IB,
          interNodeFabric: { kind: 'chain' },
        },
      },
    });
    const forwardSec = r.decode.stages.reduce((sum, s) => sum + (s.sendToNext?.sec ?? 0), 0);
    const pathSec = r.decode.stages.reduce((sum, s) => sum + s.sec, 0) + forwardSec;
    expect(r.decode.cycleSec - pathSec).toBeCloseTo(15e-6 + (3 * 4) / 50e9, 14);
  });
});

describe('G7 best algorithm uses exposed stage time under ideal overlap', () => {
  function tp4(bandwidthBytesPerSec = 100e9): SimInput {
    const i = input();
    return {
      ...i,
      parallel: { ...i.parallel, tp: 4, pp: 1 },
      workload: { ...i.workload, promptLen: 4096 },
      hardware: {
        ...i.hardware,
        topology: {
          ...SYNTH_CLUSTER,
          intraNode: {
            ...SYNTH_CLUSTER.intraNode!,
            bandwidthPerDirectionBytesPerSec: bandwidthBytesPerSec,
          },
        },
      },
    };
  }
  it('hidden β: tp4 prefill tree removes 2 µs × 73 AR = 146 µs vs min-total ring', () => {
    const i = tp4();
    const best = simulate(i).prefill;
    const ring = simulate({ ...i, options: { collectiveAlgorithm: 'ring' } }).prefill;
    const tree = simulate({ ...i, options: { collectiveAlgorithm: 'tree' } }).prefill;
    expect(best.cycleSec).toBe(tree.cycleSec);
    expect(ring.cycleSec - best.cycleSec).toBeCloseTo(146e-6, 12);
  });
  it('exposed β at 1 MB/s: ring stays faster than tree, no-overlap preserves min-total', () => {
    const i = tp4(1e6);
    for (const overlap of ['ideal', 'none'] as const) {
      const run = (collectiveAlgorithm: 'best' | 'ring' | 'tree') =>
        simulate({ ...i, options: { overlap, collectiveAlgorithm } }).prefill;
      expect(run('best').cycleSec).toBe(run('ring').cycleSec);
      expect(run('best').cycleSec).toBeLessThan(run('tree').cycleSec);
    }
  });
});

describe('G8 PP host-bounce shares host bandwidth across TP×DP ranks', () => {
  it.each([
    [2, 1, 8, 24e9],
    [2, 2, 8, 12e9],
    [8, 1, 4, 12e9],
  ])(
    'tp%i dp%i dpn%i: β uses B_eff=%i B/s for forward and return',
    (tp, dp, devicesPerNode, bandwidth) => {
      const i = input();
      const link = {
        ...SYNTH_IB,
        bandwidthPerDirectionBytesPerSec: 64e9,
        p2p: 'host_bounce' as const,
        sources: [
          ...SYNTH_IB.sources,
          {
            url: 'synthetic://test',
            title: 'Test host',
            fields: ['hostBounce.hostMemBandwidthBytesPerSec', 'hostBounce.extraLatencySec'],
          },
        ],
        hostBounce: { hostMemBandwidthBytesPerSec: 96e9, extraLatencySec: 12e-6 },
      };
      const r = simulate({
        ...i,
        parallel: { tp, dp, pp: 2, ep: 1 },
        workload: { ...i.workload, batch: dp },
        hardware: {
          ...i.hardware,
          topology: {
            ...SYNTH_CLUSTER,
            devicesPerNode,
            nodes: 4,
            intraNode: link,
            interNode: link,
          },
        },
      });
      const send = r.decode.stages[0].sendToNext!;
      expect(send.commBandwidthSec).toBeCloseTo((REF_DENSE_8B.hidden * 2) / bandwidth, 15);
      // Return contributes another 4 B/bandwidth beyond stage collective β and the activation send.
      const stageBeta = r.decode.stages.reduce((s, st) => s + st.commBandwidthSec, 0);
      expect(r.decode.comm.bandwidthSec - stageBeta - send.commBandwidthSec).toBeCloseTo(
        4 / bandwidth,
        15,
      );
    },
  );
});
