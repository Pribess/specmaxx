import { describe, expect, it } from 'vitest';
import { simulate } from '../../sim/engine/simulate';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { REFERENCE_MODEL, buildSimInput, loadedFromHf, modelKey, type LoadedModel } from './buildInput';
import { summarize, type Breakdown } from './summary';
import { SWEEP_STEPS, batchSweep } from './sweep';
import { formatBytes, formatParams, formatSeconds, formatShare, formatTokensPerSec, formatTokensPerSecCompact } from './format';

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
      const bound = r.boundSec.compute + r.boundSec.memory + r.boundSec.comm;
      expect(Math.abs(bound - r.sec)).toBeLessThanOrEqual(1e-12 * Math.max(1, r.sec));
    }
  }
}

describe('buildSimInput', () => {
  it('builds the default state without network: docs §10 numbers', () => {
    const built = buildSimInput(DEFAULT_STATE, REFERENCE_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const r = simulate(built.input);
    expect(r.decode.tokensPerSec).toBeCloseTo(117.1, 0);
    expect(r.prefill.tokensPerSec).toBeGreaterThan(12_000);
    expect(r.prefill.tokensPerSec).toBeLessThan(13_500);
  });

  it('reports a model that is not loaded (or stale) as a general issue', () => {
    const s = reduce(DEFAULT_STATE, { type: 'setModel', model: { kind: 'hf', id: QWEN_ID } });
    expect(buildSimInput(s, null)).toMatchObject({ kind: 'issues', issues: [{ field: 'general' }] });
    expect(buildSimInput(s, REFERENCE_MODEL).kind).toBe('issues');
    expect(modelKey({ kind: 'hf', id: QWEN_ID, revision: 'abc' })).toBe(`hf:${QWEN_ID}@abc`);
  });

  it('stops at field errors (TP 3) and applies quant overrides', () => {
    const s = reduce(multiDeviceState(), { type: 'setParallel', parallel: { tp: 3 } });
    const bad = buildSimInput(s, qwen());
    expect(bad.kind).toBe('issues');
    if (bad.kind === 'issues') expect(bad.issues.some((i) => i.field === 'tp' && i.level === 'error')).toBe(true);
    const kv = buildSimInput(reduce(multiDeviceState(), { type: 'setQuant', field: 'kvCache', dtype: 'fp8_e4m3' }), qwen());
    if (kv.kind !== 'ok') throw new Error(JSON.stringify(kv.issues));
    expect(kv.input.quant.kvCache).toEqual({ dtype: 'fp8_e4m3' });
    expect(kv.notes.some((n) => n.includes('p2p direct'))).toBe(true);
  });

  it('reports engine errors left after the field checks as general issues', () => {
    const s = reduce(DEFAULT_STATE, { type: 'setEdit', target: 'device', path: 'memory.bandwidthBytesPerSec', value: 0 });
    const built = buildSimInput(s, REFERENCE_MODEL);
    expect(built.kind).toBe('issues');
    if (built.kind === 'issues') expect(built.issues.some((i) => i.field === 'general' && /bandwidthBytesPerSec must be > 0/.test(i.message))).toBe(true);
  });
});

describe('summarize', () => {
  it('breakdown shares sum to 1 and bound times to the row time, for every grouping and phase', () => {
    for (const [state, loaded] of [[DEFAULT_STATE, REFERENCE_MODEL], [multiDeviceState(), qwen()]] as const) {
      const built = buildSimInput(state, loaded);
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      const summary = summarize(simulate(built.input), built.input, built.notes);
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
    expect(s.breakdown.decode.byKind.some((r) => r.key === 'collective' && r.boundSec.comm > 0)).toBe(true);
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
});

describe('batchSweep', () => {
  it('sweeps batch = dp · 2^k and loses fit at large batch', () => {
    const built = buildSimInput(reduce(DEFAULT_STATE, { type: 'setWorkload', workload: { decodeContextLen: 32768, promptLen: 32768 } }), REFERENCE_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const points = batchSweep(built.input);
    expect(points.map((p) => p.batch)).toEqual(Array.from({ length: SWEEP_STEPS }, (_, k) => 2 ** k));
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
    expect(formatBytes(19.1e9)).toBe('19.1 GB');
    expect(formatBytes(512e6)).toBe('512 MB');
    expect(formatSeconds(0.00091)).toBe('0.91 ms');
    expect(formatSeconds(12.5e-6)).toBe('12.5 µs');
    expect(formatParams(27.4e9)).toBe('27.4 B');
    expect(formatShare(0.2104)).toBe('21%');
    expect(formatShare(0.004)).toBe('0.4%');
  });
});
