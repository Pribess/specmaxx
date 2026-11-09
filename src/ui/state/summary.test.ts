import { describe, expect, it } from 'vitest';
import type { OpResult, RoutedExpertRule, SimInput } from '../../sim/engine/types';
import type { BlockDraftSpec } from '../../sim/ir/model';
import { simulate } from '../../sim/engine/simulate';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import {
  REFERENCE_MODEL,
  buildSimInput,
  loadedFromHf,
  modelKey,
  type LoadedModel,
} from './buildInput';
import { offloadPlacements, summarize, type Breakdown } from './summary';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import { SWEEP_STEPS, batchSweep } from './sweep';
import {
  formatBytes,
  formatParams,
  formatSeconds,
  formatShare,
  formatTokensPerSec,
  formatTokensPerSecCompact,
} from './format';

const QWEN_ID = 'Qwen/Qwen3.8-27B';
const qwen = (): LoadedModel => {
  const r = mapHfModel(fixtureRepoFiles(QWEN_ID));
  if (r.kind !== 'ok') throw new Error(r.kind);
  return loadedFromHf({ kind: 'hf', id: QWEN_ID }, r);
};

/** 2 × RTX 5090 with the P2P unlock, TP 2, Qwen3.8-27B: comm rows and two devices. */
function multiDeviceState(): AppState {
  return [
    { type: 'setModel', model: { kind: 'hf', id: QWEN_ID } },
    { type: 'setCounts', devicesPerNode: 2 },
    { type: 'setUnlock', axisId: 'pcie-p2p', optionId: 'direct' },
    { type: 'setParallel', parallel: { tp: 2 } },
    { type: 'setWorkload', workload: { batch: 4 } },
  ].reduce((s, a) => reduce(s, a as Parameters<typeof reduce>[1]), DEFAULT_STATE);
}

function expectConsistent(b: Breakdown): void {
  for (const rows of [b.byOp, b.byKind, b.byLayer]) {
    const shareSum = rows.reduce((s, r) => s + r.share, 0);
    expect(Math.abs(shareSum - 1)).toBeLessThan(1e-9);
    const secSum = rows.reduce((s, r) => s + r.sec, 0);
    expect(secSum / b.totalSec).toBeCloseTo(1, 12);
    for (const r of rows) {
      const bound = Object.values(r.boundSec).reduce((a, b) => a + b, 0);
      expect(Math.abs(bound - r.sec)).toBeLessThanOrEqual(1e-12 * Math.max(1, r.sec));
    }
  }
}

describe('buildSimInput', () => {
  it('builds the default state without network: docs §10 numbers', () => {
    const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const r = simulate(built.input);
    expect(r.decode.tokensPerSec).toBeCloseTo(34.3774, 3);
    expect(r.prefill.tokensPerSec).toBeCloseTo(3953.0115, 3);
    expect(r.memory.fits).toBe(false);
    expect(r.memory.maxSequencesPerReplica).toBe(0);
  });

  it('reports a model that is not loaded (or stale) as a general issue', () => {
    const s = reduce(DEFAULT_STATE, { type: 'setModel', model: { kind: 'hf', id: QWEN_ID } });
    expect(buildSimInput(s, null)).toMatchObject({
      kind: 'issues',
      issues: [{ field: 'general' }],
    });
    expect(buildSimInput(s, REFERENCE_MODEL).kind).toBe('issues');
    expect(modelKey({ kind: 'hf', id: QWEN_ID, revision: 'abc' })).toBe(`hf:${QWEN_ID}@abc`);
  });

  it('stops at field errors (TP 3) and applies quant overrides', () => {
    const s = reduce(multiDeviceState(), { type: 'setParallel', parallel: { tp: 3 } });
    const bad = buildSimInput(s, qwen());
    expect(bad.kind).toBe('issues');
    if (bad.kind === 'issues')
      expect(bad.issues.some((i) => i.field === 'tp' && i.level === 'error')).toBe(true);
    const kv = buildSimInput(
      reduce(multiDeviceState(), { type: 'setQuant', field: 'kvCache', dtype: 'fp8_e4m3' }),
      qwen(),
    );
    if (kv.kind !== 'ok') throw new Error(JSON.stringify(kv.issues));
    expect(kv.input.quant.kvCache).toEqual({ dtype: 'fp8_e4m3' });
    expect(kv.notes.some((n) => n.includes('p2p direct'))).toBe(true);
  });

  it('reports engine errors left after the field checks as general issues', () => {
    const s = reduce(DEFAULT_STATE, {
      type: 'setEdit',
      target: 'device',
      path: 'memory.bandwidthBytesPerSec',
      value: 0,
    });
    const built = buildSimInput(s, BUNDLED_DEFAULT_MODEL);
    expect(built.kind).toBe('issues');
    if (built.kind === 'issues')
      expect(
        built.issues.some(
          (i) => i.field === 'general' && /bandwidthBytesPerSec must be > 0/.test(i.message),
        ),
      ).toBe(true);
  });
});

describe('summarize', () => {
  it('breakdown shares sum to 1 and bound times to the row time, for every grouping and phase', () => {
    for (const [state, loaded] of [
      [DEFAULT_STATE, BUNDLED_DEFAULT_MODEL],
      [multiDeviceState(), qwen()],
    ] as const) {
      const built = buildSimInput(state, loaded);
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      const summary = summarize(simulate(built.input), built.input, built.notes);
      expect(summary.prefill.contextTokens).toBe(state.workload.promptLen);
      expect(summary.decode.contextTokens).toBe(state.workload.decodeContextLen);
      expectConsistent(summary.breakdown.prefill);
      expectConsistent(summary.breakdown.decode);
    }
  });

  it('summarizes the multi-device result: comm rows, memory per stage, roofline, warnings', () => {
    const built = buildSimInput(multiDeviceState(), qwen());
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const result = simulate(built.input);
    const s = summarize(result, built.input, ['loader note']);
    expect(s.decode.tokensPerSec).toBe(result.decode.tokensPerSec);
    expect(s.devicesUsed).toBe(2);
    expect(
      s.breakdown.decode.byKind.some(
        (r) => r.key === 'collective' && r.boundSec.commLatency > 0 && r.boundSec.commBandwidth > 0,
      ),
    ).toBe(true);
    expect(s.breakdown.decode.byOp[0].sec).toBeGreaterThanOrEqual(s.breakdown.decode.byOp[1].sec);
    expect(s.breakdown.decode.byLayer[0].key).toBe('embed');
    expect(s.memory.stages).toHaveLength(1);
    expect(s.memory.stages[0].capacityBytes).toBe(32e9);
    expect(s.roofline.peakLabel).toBe('bf16, fp32 accumulate, dense');
    expect(s.roofline.memBandwidthBytesPerSec).toBe(1.792e12);
    expect(s.roofline.ridgeFlopPerByte).toBeCloseTo(s.roofline.peakFlopsPerSec / 1.792e12);
    // Decode sits left of the ridge (memory-bound), prefill right of it.
    expect(s.roofline.decode.intensityFlopPerByte).toBeLessThan(s.roofline.ridgeFlopPerByte);
    expect(s.roofline.prefill.intensityFlopPerByte).toBeGreaterThan(s.roofline.ridgeFlopPerByte);
    expect(s.roofline.prefill.attainedFlopsPerSec).toBeLessThanOrEqual(s.roofline.peakFlopsPerSec);
    expect(s.warnings).toContain('loader note');
  });

  it("today's result: comm split, no transfer, no sections, no annotations", () => {
    const built = buildSimInput(multiDeviceState(), qwen());
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const result = simulate(built.input);
    const s = summarize(result, built.input);
    const collective = s.breakdown.decode.byKind.find((r) => r.key === 'collective')!;
    // Every collective op: sec = commLatencySec + commBandwidthSec, so the split sums to the row.
    const ops = result.decode.stages.flatMap((st) => st.layers.flatMap((l) => l.ops));
    const coll = ops.filter((o) => o.op.kind === 'collective');
    expect(collective.boundSec.commLatency).toBeCloseTo(
      coll.reduce((a, o) => a + o.commLatencySec, 0),
      15,
    );
    expect(collective.boundSec.commBandwidth).toBeCloseTo(
      coll.reduce((a, o) => a + o.commBandwidthSec, 0),
      15,
    );
    expect(s.breakdown.decode.byKind.every((r) => r.boundSec.transfer === 0)).toBe(true);
    expect(s.breakdown.decode.sections).toBeUndefined();
    expect(s.breakdown.decode.byLayer.every((r) => r.section === undefined)).toBe(true);
    expect(s.decode.comm).toEqual(result.decode.comm);
    expect(s.decode.transfer).toBeUndefined();
    expect(s.decode.speculative).toBeUndefined();
    expect(s.memory.tiers).toEqual([]);
    expect(s.offload).toEqual([]);
    expect(s.speculative).toBeUndefined();
  });

  it.each(['ideal', 'none'] as const)(
    'PP return appears once in decode breakdown and reconciles comm totals (%s overlap)',
    (overlap) => {
      const built = buildSimInput(multiDeviceState(), qwen());
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      const input: SimInput = {
        ...built.input,
        parallel: { ...built.input.parallel, tp: 1, pp: 2 },
        options: { overlap },
      };
      const result = simulate(input);
      const summary = summarize(result, input);
      const returned = result.decode.loopBack!;
      const decode = summary.breakdown.decode;
      expect(decode.byLayer.find((r) => r.key === 'pp.return')).toMatchObject({
        sec: returned.sec,
        count: 1,
        boundSec: {
          commLatency: returned.commLatencySec,
          commBandwidth: returned.commBandwidthSec,
        },
      });
      expect(decode.byOp.find((r) => r.key === 'pp.sample_return')).toMatchObject({
        count: 1,
        sec: returned.sec,
      });
      const comm = decode.byKind.find((r) => r.key === 'collective')!;
      expect(comm.boundSec.commLatency).toBeCloseTo(result.decode.comm.latencySec, 14);
      expect(comm.boundSec.commBandwidth).toBeCloseTo(result.decode.comm.bandwidthSec, 14);
      expect(summary.breakdown.prefill.byLayer.some((r) => r.key === 'pp.return')).toBe(false);
      expectConsistent(decode);
      expectConsistent(summary.breakdown.prefill);
    },
  );

  it('a hand-built offload result: transfer rows, tiers, chips, roofline without transfer ops', () => {
    const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const base = simulate(built.input);
    const before = summarize(base, built.input);
    // One expert fetch of 2 ms (0.5 ms latency + 1.5 ms service) in front of layer 0's ops, per decode step.
    const result = structuredClone(base);
    result.decode.stages[0].layers[1].ops.unshift(transferOp(0.5e-3, 1.5e-3));
    result.decode.transfer = { sec: 2e-3, exposedSec: 1e-3, bytes: 300e6, requests: 40 };
    result.memory.tiers = [{ tier: 'host_ram', bytes: 16.3e9, breakdown: { lookupTables: 0, routedExperts: 16.3e9, draftExperts: 0 }, capacityBytes: 96e9, fits: true }];
    const input: SimInput = {
      ...built.input,
      deployment: {
        offload: {
          routedExperts: {
            tier: 'host_ram',
            residentFraction: 0.25,
            hitRatio: { kind: 'custom', h: 0.5 },
          },
          lookupTables: { tier: 'nvme' },
        },
      },
    };
    const s = summarize(result, input);
    const transferRow = s.breakdown.decode.byKind.find((r) => r.key === 'transfer')!;
    expect(transferRow.sec).toBeCloseTo(2e-3, 15);
    expect(transferRow.boundSec.transfer).toBeCloseTo(2e-3, 15);
    expect(s.breakdown.decode.byOp.find((r) => r.key === 'moe.expert_fetch')?.count).toBe(1);
    expectConsistent(s.breakdown.decode);
    // Formula: total' = total + 2 ms
    expect(s.breakdown.decode.totalSec).toBeCloseTo(before.breakdown.decode.totalSec + 2e-3, 12);
    expect(s.decode.transfer).toEqual(result.decode.transfer);
    expect(s.memory.tiers).toEqual(result.memory.tiers);
    expect(s.offload).toEqual([
      { group: 'routed_experts', tier: 'host_ram', hitRatio: 0.5 },
      { group: 'lookup_tables', tier: 'nvme' },
    ]);
    // The transfer op has no FLOP and no HBM bytes; it only adds time: same intensity, lower attained rate.
    expect(s.roofline.decode.intensityFlopPerByte).toBe(
      before.roofline.decode.intensityFlopPerByte,
    );
    expect(s.roofline.decode.attainedFlopsPerSec).toBeCloseTo(
      before.roofline.decode.flops / (before.breakdown.decode.totalSec + 2e-3),
      0,
    );
  });

  it('offload placements: one per rule, by first layer, each with its layers and hit ratio', () => {
    const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const cache = { residentFraction: 0.25, hitRatio: { kind: 'custom', h: 0.5 } } as const;
    const input: SimInput = {
      ...built.input,
      deployment: {
        offload: {
          routedExperts: [
            { tier: 'nvme', layers: { from: 24, to: 48 }, ...cache },
            { tier: 'host_ram', layers: { from: 0, to: 24 }, ...cache },
          ],
          lookupTables: [
            { tier: 'nvme', layers: { from: 30, to: 64 } },
            { tier: 'device', layers: { from: 0, to: 10 } },
            { tier: 'host_ram', layers: { from: 10, to: 30 }, scaleLayout: 'interleaved' },
          ],
        },
      },
    };
    expect(offloadPlacements(input)).toEqual([
      { group: 'routed_experts', tier: 'host_ram', hitRatio: 0.5, layers: { from: 0, to: 24 } },
      { group: 'routed_experts', tier: 'nvme', hitRatio: 0.5, layers: { from: 24, to: 48 } },
      // Tables left on the device are not a placement.
      { group: 'lookup_tables', tier: 'host_ram', layers: { from: 10, to: 30 } },
      { group: 'lookup_tables', tier: 'nvme', layers: { from: 30, to: 64 } },
    ]);
    // The one-rule object form gives the same placement as its one-rule list.
    const one = { tier: 'nvme', ...cache } as const;
    const withRules = (routedExperts: RoutedExpertRule | readonly RoutedExpertRule[]) =>
      offloadPlacements({ ...built.input, deployment: { offload: { routedExperts } } });
    expect(withRules(one)).toEqual(withRules([one]));
    expect(withRules([])).toEqual([]);
  });

  it('a hand-built spec-dec decode: draft and verify sections, speedup, drafter badge', () => {
    const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const result = structuredClone(simulate(built.input));
    const stage = result.decode.stages[0];
    // A one-layer drafter: a copy of layer 0 named draft.layer.0, after the head.
    stage.layers.push({
      ...structuredClone(stage.layers[1]),
      name: 'draft.layer.0',
      layerIndex: null,
    });
    result.decode.speculative = {
      draftSec: stage.layers.at(-1)!.sec,
      verifySec: 10e-3,
      expectedTokensPerStep: 2.31,
      speedup: 1.67,
    };
    const input: SimInput = {
      ...built.input,
      deployment: {
        speculative: {
          method: 'dflash2',
          draftLength: 3,
          acceptance: { kind: 'iid', alpha: 0.6 },
          drafter: { kind: 'hypothetical', spec: {} as BlockDraftSpec },
        },
      },
    };
    const s = summarize(result, input);
    const b = s.breakdown.decode;
    expectConsistent(b);
    expect(b.sections?.map((r) => r.key)).toEqual(['verify', 'draft']);
    const [verify, draft] = b.sections!;
    expect(draft.sec).toBeCloseTo(stage.layers.at(-1)!.sec, 15);
    expect(verify.sec + draft.sec).toBeCloseTo(b.totalSec, 15);
    expect(b.byLayer.at(-1)).toMatchObject({ key: 'draft.layer.0', section: 'draft' });
    expect(b.byLayer[0]).toMatchObject({ key: 'embed', section: 'verify' });
    expect(s.decode.speculative).toEqual(result.decode.speculative);
    expect(s.speculative).toEqual({ method: 'dflash2', draftLength: 3, hypothetical: true });
    // Prefill has no speculative block: no sections.
    expect(s.breakdown.prefill.sections).toBeUndefined();
  });
});

/** A routed-expert fetch over the host link: `latencySec` + `bandwidthSec` seconds, no FLOP, no HBM bytes. */
function transferOp(latencySec: number, bandwidthSec: number): OpResult {
  const zero = {
    weightBytes: 0,
    activationReadBytes: 0,
    activationWriteBytes: 0,
    kvReadBytes: 0,
    kvWriteBytes: 0,
    stateReadBytes: 0,
    stateWriteBytes: 0,
    totalBytes: 0,
  };
  return {
    op: {
      kind: 'transfer',
      name: 'moe.expert_fetch',
      group: 'routed_experts',
      source: 'host_ram',
      reads: [{ count: 40, bytesEach: 7.5e6 }],
      usefulBytes: 300e6,
      prefetchedFraction: 1,
    },
    flops: 0,
    bytes: zero,
    computeSec: 0,
    memorySec: 0,
    commLatencySec: 0,
    commBandwidthSec: 0,
    transferLatencySec: latencySec,
    transferBandwidthSec: bandwidthSec,
    sec: latencySec + bandwidthSec,
    bound: 'transfer',
  };
}

describe('batchSweep', () => {
  it('sweeps batch = dp · 2^k and loses fit at large batch', () => {
    const built = buildSimInput(
      reduce(multiDeviceState(), {
        type: 'setWorkload',
        workload: { decodeContextLen: 32768, promptLen: 32768 },
      }),
      BUNDLED_DEFAULT_MODEL,
    );
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const points = batchSweep(built.input);
    expect(points.map((p) => p.batch)).toEqual(
      Array.from({ length: SWEEP_STEPS }, (_, k) => 2 ** k),
    );
    expect(points[1].decodeTokensPerSec).toBeGreaterThan(points[0].decodeTokensPerSec);
    expect(points[0].fits).toBe(true);
    expect(points[SWEEP_STEPS - 1].fits).toBe(false);
  });
});

describe('format', () => {
  it('formats rates, bytes, times, params and shares', () => {
    expect(formatTokensPerSec(25_710.4)).toBe('25,710');
    expect(formatTokensPerSec(231.44)).toBe('231.4');
    expect(formatTokensPerSec(3.271)).toBe('3.27');
    expect(formatTokensPerSec(Infinity)).toBe('—');
    expect(formatTokensPerSecCompact(25_710)).toBe('25.7k');
    expect(formatTokensPerSecCompact(231.4)).toBe('231');
    expect(formatBytes(19.1e9)).toBe('19.1GB');
    expect(formatBytes(512e6)).toBe('512MB');
    expect(formatSeconds(0.00091)).toBe('0.91ms');
    expect(formatSeconds(12.5e-6)).toBe('12.5µs');
    expect(formatParams(27.4e9)).toBe('27.4B');
    expect(formatShare(0.2104)).toBe('21%');
    expect(formatShare(0.004)).toBe('0.4%');
  });
});
