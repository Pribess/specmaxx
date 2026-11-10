/**
 * Prefill/decode disaggregation in the UI state (docs/design/prefill-decode-disaggregation.md §6, §8): the pool view
 * and its reducer write-back, the URL keys (`pd`, `osl`, `pdr`, `dec.*`), the state → DisaggregatedInput step, the
 * summary, the formatters and the runner. ID-PD-3: with the knob off, every existing state, URL, input and run is
 * exactly as before. Model: GLM-5.3-Flash (288 experts, divisible by both 32 and 144) on H100 SXM nodes of 8.
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { simulate } from '../../sim/engine/simulate';
import { poolInput, simulateDisaggregated } from '../../sim/engine/disaggregation';
import {
  DEFAULT_STATE,
  poolView,
  reduce,
  reduceInPool,
  type AppAction,
  type AppState,
  type PoolAction,
} from './appState';
import {
  buildDisaggregatedInput,
  buildSimInput,
  DECODE_POOL_ISSUE_PREFIX,
  sharedInput,
  loadedFromHf,
  REFERENCE_MODEL,
  type LoadedModel,
} from './buildInput';
import { decodeState, encodeState, POOL_URL_KEYS } from './url';
import { effectivePrefillPhase, summarize, summarizeDisaggregated } from './summary';
import { batchSweep } from './sweep';
import { formatRatio, formatSystemRate, formatWorkloadQuote } from './format';
import { DEFAULT_KV_LINK_ID, KV_LINK_PRESETS, findKvLink, kvTransferConfig } from './kvLink';
import { inlineRunner } from '../sim/runner';

const ID = 'zai-org/GLM-5.3-Flash';
function loadGlm(): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(`${ID}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id: ID }, r);
}
const GLM = loadGlm();

/** The state of a query (asserts nothing was dropped). */
function decoded(query: string): AppState {
  const { state, dropped } = decodeState(query);
  expect(dropped).toEqual([]);
  return state;
}
const apply = (s: AppState, ...actions: AppAction[]) => actions.reduce(reduce, s);
const inPool = (action: PoolAction): AppAction => ({ type: 'inPool', pool: 'decode', action });

/** DeepSeek's prefill shape: DP attention + EP 32 on 4 nodes of 8 H100, batch 32. */
const PREFILL_QUERY = `m=${ID}&d=h100-sxm&n=8&nodes=4&dp=32&ep=32&lay=dep&b=32`;
const PREFILL = decoded(PREFILL_QUERY);
/** The same with disaggregation on and the decode pool at DEP144 on 18 nodes, batch 144. */
const PD = apply(
  PREFILL,
  { type: 'setDisaggregation', on: true },
  inPool({ type: 'setCounts', nodes: 18 }),
  inPool({ type: 'setParallelLayout', layout: 'dpAttentionEp' }),
  inPool({ type: 'setWorkload', workload: { batch: 144 } }),
);

/** Every literal `decodeState('…')` query of the other state tests: the existing URL fixtures. */
const FIXTURE_QUERIES: string[] = (() => {
  const files = import.meta.glob<string>('./*.test.ts', {
    query: '?raw',
    import: 'default',
    eager: true,
  });
  const queries = new Set<string>(['', PREFILL_QUERY]);
  for (const [path, text] of Object.entries(files)) {
    if (path.endsWith('disaggregation.test.ts')) continue;
    for (const m of text.matchAll(/decodeState\(\s*'([^']*)'/g)) queries.add(m[1]);
  }
  return [...queries];
})();

describe('ID-PD-3: off is identical', () => {
  it('scans enough existing URL fixtures', () => {
    expect(FIXTURE_QUERIES.length).toBeGreaterThan(40);
  });

  it('no existing fixture decodes to a state with disaggregation; its encoding has none of the new keys', () => {
    for (const q of FIXTURE_QUERIES) {
      const { state } = decodeState(q);
      expect('disaggregation' in state, q).toBe(false);
      const encoded = encodeState(state);
      expect(encoded, q).not.toMatch(/(^|&)(pd|osl|pdr|dec\.[a-z]+)=/);
      // The encoding is a fixed point (decode ∘ encode ∘ decode = decode).
      expect(decodeState(encoded).state, q).toEqual(state);
    }
  });

  it('every existing action leaves `disaggregation` absent', () => {
    const actions: AppAction[] = [
      { type: 'setDevice', deviceId: 'h100-sxm' },
      { type: 'setCounts', devicesPerNode: 8, nodes: 2 },
      { type: 'setParallel', parallel: { tp: 8, dp: 2 } },
      { type: 'setParallelLayout', layout: 'dpAttentionEp' },
      { type: 'setWorkload', workload: { batch: 16, promptLen: 4096 } },
      { type: 'setModel', model: { kind: 'hf', id: ID } },
      { type: 'setDisaggregationParams', outputLen: 2048 },
      inPool({ type: 'setCounts', nodes: 4 }),
    ];
    let s = DEFAULT_STATE;
    for (const a of actions) {
      s = reduce(s, a);
      expect('disaggregation' in s).toBe(false);
    }
    // inPool and the params are no-ops while off.
    expect(reduce(PREFILL, inPool({ type: 'setCounts', nodes: 4 }))).toBe(PREFILL);
    expect(reduce(PREFILL, { type: 'setDisaggregationParams', outputLen: 7 })).toBe(PREFILL);
  });

  it('off: buildDisaggregatedInput is undefined; buildSimInput and the runner are as before (no pd key)', async () => {
    expect(buildDisaggregatedInput(PREFILL, GLM)).toBeUndefined();
    expect(buildDisaggregatedInput(DEFAULT_STATE, REFERENCE_MODEL)).toBeUndefined();
    const built = buildSimInput(PREFILL, GLM);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const run = await inlineRunner.run(built.input, built.notes);
    expect('pd' in run).toBe(false);
    expect(run).toEqual({
      summary: summarize(simulate(built.input), built.input, built.notes),
      sweep: batchSweep(built.input),
    });
  });

  it('turning it on and off again gives back the exact state and URL', () => {
    const back = reduce(PD, { type: 'setDisaggregation', on: false });
    expect(back).toEqual(PREFILL);
    expect('disaggregation' in back).toBe(false);
    expect(encodeState(back)).toBe(encodeState(PREFILL));
  });
});

describe('reducer: the pool view', () => {
  it('turning on copies the prefill pool into the decode pool; on again keeps it', () => {
    const on = reduce(PREFILL, { type: 'setDisaggregation', on: true });
    expect(on.disaggregation).toEqual({
      decodePool: {
        hw: PREFILL.hw,
        parallel: PREFILL.parallel,
        parallelLayout: 'dpAttentionEp',
        batch: 32,
      },
    });
    expect(reduce(PD, { type: 'setDisaggregation', on: true })).toBe(PD);
    // The decode view of a fresh copy is the prefill state itself (same four fields).
    expect(poolView(on, 'decode')).toEqual(on);
    expect(poolView(on, 'prefill')).toBe(on);
  });

  it('pool actions change only the decode pool; the prefill pool keeps its four fields', () => {
    expect(PD.hw).toEqual(PREFILL.hw);
    expect(PD.parallel).toEqual(PREFILL.parallel);
    expect(PD.parallelLayout).toBe('dpAttentionEp');
    expect(PD.workload).toEqual(PREFILL.workload);
    const pool = PD.disaggregation!.decodePool;
    expect(pool.hw.nodes).toBe(18);
    expect(pool.hw.devicesPerNode).toBe(8);
    expect(pool.parallel).toEqual({ tp: 1, pp: 1, dp: 144, ep: 144 });
    expect(pool.parallelLayout).toBe('dpAttentionEp');
    expect(pool.batch).toBe(144);
    const view = poolView(PD, 'decode');
    expect(view.hw).toBe(pool.hw);
    expect(view.workload).toEqual({ ...PD.workload, batch: 144 });
  });

  it('every hardware and parallel action through the pool leaves the prefill fields untouched', () => {
    const actions: PoolAction[] = [
      { type: 'setDevice', deviceId: 'h200-sxm' },
      { type: 'setCapacity', index: 0 },
      { type: 'setEngine', engineId: undefined },
      { type: 'setLink', level: 'inter', linkId: undefined },
      { type: 'setCabling', cabling: undefined },
      { type: 'setEdit', target: 'device', path: 'memory.bandwidthBytesPerSec', value: 5e12 },
      { type: 'resetEdits' },
      { type: 'setMemoryFit', field: 'runtimeReserveBytes', value: 2e9 },
      { type: 'setHost', host: { ramCapacityBytes: 512e9 } },
      { type: 'setParallel', parallel: { tp: 8, dp: 18, ep: 144, dcp: 8 } },
      { type: 'setParallelLayout', layout: 'custom' },
      { type: 'setParallel', parallel: { epImbalance: 1.5, redundantExpertsPerRank: 2 } },
      { type: 'setWorkload', workload: { batch: 288 } },
      { type: 'setHardware', hw: PREFILL.hw },
    ];
    let s = PD;
    for (const a of actions) {
      const next = reduceInPool(s, a);
      expect(next).toEqual(reduce(s, inPool(a)));
      expect(next.hw, a.type).toEqual(PREFILL.hw);
      expect(next.parallel, a.type).toEqual(PREFILL.parallel);
      expect(next.parallelLayout, a.type).toBe('dpAttentionEp');
      expect(next.workload, a.type).toEqual(PREFILL.workload);
      // The view after the action is what the ordinary reducer makes of the view before it.
      expect({ ...poolView(next, 'decode'), disaggregation: undefined }, a.type).toEqual({
        ...reduce(poolView(s, 'decode'), a),
        disaggregation: undefined,
      });
      s = next;
    }
    expect(s.disaggregation!.decodePool.parallel).toEqual({
      tp: 8,
      pp: 1,
      dp: 18,
      ep: 144,
      dcp: 8,
      epImbalance: 1.5,
      redundantExpertsPerRank: 2,
    });
    expect(s.disaggregation!.decodePool.parallelLayout).toBeUndefined();
    expect(s.disaggregation!.decodePool.batch).toBe(288);
  });

  it('shared fields through the pool change for both pools (P, N, quant, options)', () => {
    const s = apply(
      PD,
      inPool({ type: 'setWorkload', workload: { promptLen: 4096, decodeContextLen: 4608 } }),
      inPool({ type: 'setQuant', field: 'kvCache', dtype: 'fp8_e4m3' }),
      inPool({ type: 'setOptions', options: { overlap: 'none' } }),
    );
    expect(s.workload).toEqual({ ...PREFILL.workload, promptLen: 4096, decodeContextLen: 4608 });
    expect(s.quant).toEqual({ kvCache: 'fp8_e4m3' });
    expect(s.options).toEqual({ overlap: 'none' });
    expect(s.disaggregation!.decodePool).toEqual(PD.disaggregation!.decodePool);
    expect(poolView(s, 'decode').workload).toEqual({ ...s.workload, batch: 144 });
  });

  it('settings: defaults are stored as absent', () => {
    const s = reduce(PD, {
      type: 'setDisaggregationParams',
      outputLen: 2048,
      instances: { prefill: 1, decode: 3 },
    });
    expect(s.disaggregation).toMatchObject({
      outputLen: 2048,
      instances: { prefill: 1, decode: 3 },
    });
    const back = reduce(s, {
      type: 'setDisaggregationParams',
      outputLen: 1024,
      instances: { prefill: 1, decode: 1 },
    });
    expect(back).toEqual(PD);
    // A key absent from the action is kept.
    const only = reduce(s, { type: 'setDisaggregationParams', outputLen: 1024 });
    expect(only.disaggregation).toEqual({
      ...PD.disaggregation,
      instances: { prefill: 1, decode: 3 },
    });
  });
});

describe('URL', () => {
  const FULL = reduce(PD, {
    type: 'setDisaggregationParams',
    outputLen: 2048,
    instances: { prefill: 1, decode: 3 },
  });

  it('round trip of the DeepSeek-shaped state, keys after the prefill ones', () => {
    const q = encodeState(FULL);
    expect(q).toBe(
      `${PREFILL_QUERY}&pd=1&osl=2048&pdr=1:3` +
        '&dec.d=h100-sxm&dec.n=8&dec.nodes=18&dec.dp=144&dec.ep=144&dec.lay=dep&dec.b=144',
    );
    expect(decoded(q)).toEqual(FULL);
    expect(decoded(encodeState(PD))).toEqual(PD);
    expect(encodeState(PD)).toBe(
      `${PREFILL_QUERY}&pd=1&dec.d=h100-sxm&dec.n=8&dec.nodes=18&dec.dp=144&dec.ep=144&dec.lay=dep&dec.b=144`,
    );
  });

  it('a decode pool with every pool key round-trips (DCP, EP balance, host, edits, memory fit)', () => {
    const s = apply(
      PD,
      inPool({ type: 'setParallelLayout', layout: 'custom' }),
      inPool({ type: 'setParallel', parallel: { tp: 8, dp: 18, dcp: 8 } }),
      inPool({
        type: 'setParallel',
        parallel: { epImbalance: 1.5, epImbalanceScalesAllToAll: true },
      }),
      inPool({ type: 'setParallel', parallel: { redundantExpertsPerRank: 2 } }),
      inPool({
        type: 'setParallel',
        parallel: { dispatchFormat: 'fp8_e4m3', combineFormat: 'bf16', allToAllEfficiency: 0.7 },
      }),
      inPool({
        type: 'setEdit',
        target: 'device',
        path: 'memory.bandwidthBytesPerSec',
        value: 4e12,
      }),
      inPool({ type: 'setMemoryFit', field: 'runtimeReserveBytes', value: 1.5e9 }),
      inPool({ type: 'setHost', host: { ramCapacityBytes: 512e9 } }),
    );
    const q = encodeState(s);
    expect(q).toContain(
      '&dec.tp=8&dec.dp=18&dec.ep=144&dec.dcp=8&dec.epi=1.5&dec.epa=1&dec.rex=2&dec.dsp=fp8_e4m3&dec.cmb=bf16&dec.a2e=70',
    );
    // The prefill pool keeps its own (default) all-to-all settings.
    expect(q).not.toMatch(/(^|&)(dsp|cmb|a2e)=/);
    expect(q).toContain('dec.x=memory.bandwidthBytesPerSec:4e12');
    expect(q).toContain('dec.rr=1.5');
    expect(q).toContain('dec.hrc=512');
    expect(decoded(q)).toEqual(s);
  });

  it("pool keys use the same defaults as the state keys (not the prefill pool's)", () => {
    const s = apply(DEFAULT_STATE, { type: 'setDisaggregation', on: true });
    expect(encodeState(s)).toBe('pd=1');
    expect(decoded('pd=1')).toEqual(s);
    expect(POOL_URL_KEYS).toContain('dcp');
  });

  it('pd, osl, pdr and dec.* without pd=1 are dropped', () => {
    const r = decodeState(`${PREFILL_QUERY}&osl=2048&pdr=1:3&dec.nodes=18&dec.b=144`);
    expect(r.state).toEqual(PREFILL);
    expect(r.dropped).toEqual(['osl=2048', 'pdr=1:3', 'dec.nodes=18', 'dec.b=144']);
    const bad = decodeState(`${PREFILL_QUERY}&pd=2&dec.nodes=18`);
    expect(bad.state).toEqual(PREFILL);
    expect(bad.dropped).toEqual(['pd=2', 'dec.nodes=18']);
  });

  it('invalid values are dropped with their dec. key; unknown dec. keys are unknown', () => {
    const r = decodeState(
      `${PREFILL_QUERY}&pd=1&osl=0&pdr=0:3&dec.d=nope&dec.nodes=x&dec.b=0&dec.p=4096&dec.tp=2&dec.lay=dep`,
    );
    expect(r.dropped).toEqual([
      'dec.p=4096',
      'osl=0',
      'pdr=0:3',
      'dec.d=nope',
      'dec.nodes=x',
      'dec.lay=dep',
      'dec.b=0',
    ]);
    expect(r.state.disaggregation).toEqual({
      decodePool: { hw: DEFAULT_STATE.hw, parallel: { tp: 2, pp: 1, dp: 1, ep: 1 }, batch: 1 },
    });
  });

  it("the decode pool's lay=dep is dropped unless its shape holds; defaults osl=1024 and pdr=1:1 stay absent", () => {
    const shape = `${PREFILL_QUERY}&pd=1&dec.d=h100-sxm&dec.n=8&dec.dp=8&dec.ep=4&dec.lay=dep`;
    const r = decodeState(shape);
    expect(r.dropped).toEqual(['dec.lay=dep']);
    expect(r.state.disaggregation!.decodePool.parallelLayout).toBeUndefined();
    const defaults = decodeState(`pd=1&osl=1024&pdr=1:1`);
    expect(defaults.dropped).toEqual([]);
    expect(defaults.state.disaggregation).toEqual({
      decodePool: {
        hw: DEFAULT_STATE.hw,
        parallel: DEFAULT_STATE.parallel,
        batch: DEFAULT_STATE.workload.batch,
      },
    });
    expect(encodeState(defaults.state)).toBe('pd=1');
  });
});

describe('buildDisaggregatedInput', () => {
  it('prefill = the state, decode = the pool view; shared = the prefill input without hardware and parallel', () => {
    const r = buildDisaggregatedInput(PD, GLM);
    if (r?.kind !== 'ok') throw new Error(JSON.stringify(r?.issues));
    const prefill = buildSimInput(PD, GLM);
    const decode = buildSimInput(poolView(PD, 'decode'), GLM);
    if (prefill.kind !== 'ok' || decode.kind !== 'ok') throw new Error('pool build failed');
    expect(r.prefill).toEqual(prefill);
    expect(r.decode).toEqual(decode);
    const shared = sharedInput(prefill.input);
    expect('hardware' in shared || 'parallel' in shared).toBe(false);
    expect(r.input).toEqual({
      shared,
      prefillPool: {
        hardware: prefill.input.hardware,
        parallel: prefill.input.parallel,
        batch: 32,
      },
      decodePool: { hardware: decode.input.hardware, parallel: decode.input.parallel, batch: 144 },
      config: {
        outputLen: 1024,
        prefillInstances: 1,
        decodeInstances: 1,
        // Always set while on: the default link (InfiniBand NDR400), overlapped layer by layer.
        kvTransfer: {
          link: {
            name: 'InfiniBand NDR400 (one NIC per GPU)',
            bandwidthPerDirectionBytesPerSec: 50e9,
            latencySec: 1e-6,
          },
          overlap: 'layerwise',
        },
      },
    });
    expect(r.input.decodePool.parallel).toEqual({ tp: 1, pp: 1, dp: 144, ep: 144 });
    // The decode pool's own input is exactly its single-pool input.
    expect(poolInput(r.input.shared, r.input.decodePool, 'decode')).toEqual(decode.input);
  });

  it("the decode pool's issues are prefixed; settings out of range are errors on their fields", () => {
    const bad = apply(PD, inPool({ type: 'setWorkload', workload: { batch: 100 } }), {
      type: 'setDisaggregationParams',
      outputLen: 0,
      instances: { prefill: 0, decode: 2 },
    });
    const r = buildDisaggregatedInput(bad, GLM);
    expect(r?.kind).toBe('issues');
    const errors = r!.issues.filter((i) => i.level === 'error');
    expect(errors.some((i) => i.message.startsWith(DECODE_POOL_ISSUE_PREFIX))).toBe(true);
    expect(errors.filter((i) => i.field === 'outputLen')).toHaveLength(1);
    expect(errors.filter((i) => i.field === 'instances')).toHaveLength(1);
    // The decode pool's own result keeps the unprefixed messages for its panels.
    expect(r!.decode.kind).toBe('issues');
    expect(r!.decode.issues.some((i) => i.message.startsWith(DECODE_POOL_ISSUE_PREFIX))).toBe(
      false,
    );
    // The prefill pool is fine.
    expect(r!.prefill.kind).toBe('ok');
  });
});

describe('summary and runner', () => {
  const built = buildDisaggregatedInput(
    reduce(PD, { type: 'setDisaggregationParams', instances: { prefill: 1, decode: 3 } }),
    GLM,
  );
  if (built?.kind !== 'ok') throw new Error(JSON.stringify(built?.issues));
  const pd = built.input;

  it('summarizeDisaggregated: each pool summarized with its own input, plus the plain cross-pool numbers', () => {
    const result = simulateDisaggregated(pd);
    const s = summarizeDisaggregated(result, pd, ['p note'], ['d note']);
    expect(s.prefillPool).toEqual(
      summarize(result.prefillPool, poolInput(pd.shared, pd.prefillPool, 'prefill'), ['p note']),
    );
    expect(s.decodePool).toEqual(
      summarize(result.decodePool, poolInput(pd.shared, pd.decodePool, 'decode'), ['d note']),
    );
    expect(s.system).toEqual({
      promptLen: 8192,
      outputLen: 1024,
      decodeContextLen: 8192,
      prefill: result.prefill,
      decode: result.decode,
      balance: result.balance,
      system: result.system,
      kvTransfer: result.kvTransfer,
    });
    expect(s.system.system.prefillInstances).toBe(1);
    expect(s.system.system.decodeInstances).toBe(3);
    expect(s.system.system.devices).toBe(32 + 3 * 144);
    // Plain data: survives a structured clone (the worker posts it).
    expect(structuredClone(s)).toEqual(s);
  });

  it('run with the disaggregated request: the single-pool part as before, plus pd', async () => {
    const run = await inlineRunner.run(built.prefill.input, built.prefill.notes, {
      input: pd,
      decodeNotes: built.decode.notes,
    });
    const plain = await inlineRunner.run(built.prefill.input, built.prefill.notes);
    expect(run.summary).toEqual(plain.summary);
    expect(run.sweep).toEqual(plain.sweep);
    expect(run.pd).toEqual({
      summary: summarizeDisaggregated(
        simulateDisaggregated(pd),
        pd,
        built.prefill.notes,
        built.decode.notes,
      ),
      decodeSweep: batchSweep(poolInput(pd.shared, pd.decodePool, 'decode')),
    });
    expect(run.pd!.decodeSweep[0].batch).toBe(144);
  });
});

describe('all-to-all options per pool', () => {
  it('dispatch/combine formats and η are pool-level: they reach only the pool that set them', () => {
    // Both pools in MoE mode B (DP attention + EP: EP > TP), so the engine reads the knobs in each.
    const s = reduce(
      PD,
      inPool({
        type: 'setParallel',
        parallel: { dispatchFormat: 'fp8_e4m3', allToAllEfficiency: 0.7 },
      }),
    );
    expect(s.parallel).toEqual(PREFILL.parallel);
    const r = buildDisaggregatedInput(s, GLM);
    if (r?.kind !== 'ok') throw new Error(JSON.stringify(r?.issues));
    expect(r.input.decodePool.parallel).toMatchObject({
      dispatchFormat: 'fp8_e4m3',
      allToAllEfficiency: 0.7,
    });
    expect(r.input.prefillPool.parallel.dispatchFormat).toBeUndefined();
    expect(r.input.prefillPool.parallel.allToAllEfficiency).toBeUndefined();
    expect('dispatchFormat' in r.input.shared).toBe(false);
  });
});

describe('communication overlap with two pools (shared options, per-pool effect)', () => {
  it('per-class overlap: each pool sees the classes its own layout has', () => {
    // Prefill DEP 32 has only the EP class; a decode pool at TP 8 × DP 18, EP 144 has TP and EP.
    const s = apply(
      PD,
      inPool({ type: 'setParallelLayout', layout: 'custom' }),
      inPool({ type: 'setParallel', parallel: { tp: 8, dp: 18 } }),
      { type: 'setOverlapClass', overlapClass: 'tp', fraction: 0.5 },
      { type: 'setOverlapClass', overlapClass: 'ep', fraction: 0.25 },
    );
    const r = buildDisaggregatedInput(s, GLM);
    if (r?.kind !== 'ok') throw new Error(JSON.stringify(r?.issues));
    expect(r.prefill.input.options?.overlapByClass).toEqual({ ep: 0.25 });
    expect(r.decode.input.options?.overlapByClass).toEqual({ tp: 0.5, ep: 0.25 });
    expect(r.input.shared.options?.overlapByClass).toEqual({ tp: 0.5, ep: 0.25 });
    // Each pool's run equals its own single-pool run (the engine ignores a class a run lacks).
    const result = simulateDisaggregated(r.input);
    expect(result.decodePool).toEqual(simulate(r.decode.input));
    expect(result.prefillPool.prefill).toEqual(simulate(r.prefill.input).prefill);
  });

  it('two-batch overlap: on in both DP attention + EP pools; off with a warning when one pool cannot use it', () => {
    const both = reduce(PD, { type: 'setTwoBatchOverlap', on: true });
    const r = buildDisaggregatedInput(both, GLM);
    if (r?.kind !== 'ok') throw new Error(JSON.stringify(r?.issues));
    expect(r.input.shared.deployment?.twoBatchOverlap).toBe(true);
    expect(r.issues.some((i) => i.field === 'twoBatchOverlap')).toBe(false);
    // A decode pool at TP 8, EP 8 (mode A: no all-to-all) cannot use it.
    const one = apply(
      both,
      inPool({ type: 'setParallelLayout', layout: 'custom' }),
      inPool({ type: 'setParallel', parallel: { tp: 8, dp: 18, ep: 8 } }),
    );
    const o = buildDisaggregatedInput(one, GLM);
    if (o?.kind !== 'ok') throw new Error(JSON.stringify(o?.issues));
    expect(o.input.shared.deployment?.twoBatchOverlap).toBeUndefined();
    expect(o.issues.filter((i) => i.field === 'twoBatchOverlap').map((i) => i.level)).toContain(
      'warning',
    );
    expect(() => simulateDisaggregated(o.input)).not.toThrow();
  });
});

describe('effectivePrefillPhase', () => {
  const summaryOf = (s: AppState) => {
    const b = buildDisaggregatedInput(s, GLM);
    if (b?.kind !== 'ok') throw new Error(JSON.stringify(b?.issues));
    return summarizeDisaggregated(simulateDisaggregated(b.input), b.input);
  };

  it('a slow, not overlapped KV link lowers the prefill figure by cycle_P / cycle_eff; the compute-only rate is kept', () => {
    const fast = summaryOf(PD);
    const slow = summaryOf(
      apply(
        PD,
        { type: 'setKvEdit', path: 'bandwidthPerDirectionBytesPerSec', value: 1e9 },
        { type: 'setKvOverlap', overlap: 'none' },
      ),
    );
    const f = effectivePrefillPhase(fast);
    expect(f.tokensPerSec).toBe(fast.system.prefill.tokensPerSec);
    expect(f.computeOnlyTokensPerSec).toBe(fast.prefillPool.prefill.tokensPerSec);
    const e = effectivePrefillPhase(slow);
    const kv = slow.system.kvTransfer!;
    const factor = kv.prefillCycleSec / kv.effectivePrefillCycleSec;
    expect(factor).toBeLessThan(1);
    expect(e.tokensPerSec).toBeCloseTo(slow.prefillPool.prefill.tokensPerSec * factor, 6);
    expect(e.tokensPerSec).toBeLessThan(f.tokensPerSec);
    expect(e.perSequenceTokensPerSec).toBeCloseTo(
      slow.prefillPool.prefill.perSequenceTokensPerSec * factor,
      6,
    );
    expect(e.computeOnlyTokensPerSec).toBe(slow.prefillPool.prefill.tokensPerSec);
    expect(e.contextTokens).toBe(8192);
  });
});

describe('KV transfer settings', () => {
  const XDR = 'infiniband-xdr800-one-nic-per-gpu';

  it('the link presets are the catalog RDMA NICs; NDR400 is the default', () => {
    expect(KV_LINK_PRESETS.map((l) => l.name)).toEqual(
      expect.arrayContaining([
        'InfiniBand HDR200 (one NIC per GPU)',
        'InfiniBand NDR400 (one NIC per GPU)',
        'InfiniBand XDR800 (one NIC per GPU)',
      ]),
    );
    expect(KV_LINK_PRESETS.every((l) => l.transport === 'rdma')).toBe(true);
    expect(DEFAULT_KV_LINK_ID).toBe('infiniband-ndr400-one-nic-per-gpu');
    expect(findKvLink('pcie-5-0-x16-host-bounce')).toBeUndefined();
  });

  it('reducer: defaults stored as absent; a new link clears the edits; no-ops while off', () => {
    const s = apply(
      PD,
      { type: 'setKvLink', linkId: XDR },
      { type: 'setKvEdit', path: 'bandwidthPerDirectionBytesPerSec', value: 1e9 },
      { type: 'setKvEdit', path: 'latencySec', value: 5e-6 },
      { type: 'setKvOverlap', overlap: 'none' },
    );
    expect(s.disaggregation).toMatchObject({
      kvLinkId: XDR,
      kvEdits: { bandwidthPerDirectionBytesPerSec: 1e9, latencySec: 5e-6 },
      kvOverlap: 'none',
    });
    expect(kvTransferConfig(s.disaggregation!)).toEqual({
      link: {
        name: 'InfiniBand XDR800 (one NIC per GPU)',
        bandwidthPerDirectionBytesPerSec: 1e9,
        latencySec: 5e-6,
      },
      overlap: 'none',
    });
    const back = apply(
      s,
      { type: 'setKvEdit', path: 'latencySec', value: undefined },
      { type: 'setKvLink', linkId: DEFAULT_KV_LINK_ID },
      { type: 'setKvOverlap', overlap: 'layerwise' },
    );
    expect(back).toEqual(PD);
    expect(reduce(PREFILL, { type: 'setKvOverlap', overlap: 'none' })).toBe(PREFILL);
    expect(reduce(PREFILL, { type: 'setKvLink', linkId: XDR })).toBe(PREFILL);
  });

  it('URL: kvl, kvx, kvo after pdr, round trip; invalid dropped; without pd dropped', () => {
    const s = apply(
      PD,
      { type: 'setDisaggregationParams', instances: { prefill: 1, decode: 3 } },
      { type: 'setKvLink', linkId: XDR },
      { type: 'setKvEdit', path: 'bandwidthPerDirectionBytesPerSec', value: 1e9 },
      { type: 'setKvOverlap', overlap: 'none' },
    );
    const q = encodeState(s);
    expect(q).toContain(
      `&pd=1&pdr=1:3&kvl=${XDR}&kvx=bandwidthPerDirectionBytesPerSec:1e9&kvo=none&dec.d=h100-sxm`,
    );
    expect(decoded(q)).toEqual(s);
    const bad = decodeState(
      `pd=1&kvl=pcie-5-0-x16-host-bounce&kvx=latencySec:2,bandwidthPerDirectionBytesPerSec:0,foo:1&kvo=ideal`,
    );
    expect(bad.dropped).toEqual([
      'kvl=pcie-5-0-x16-host-bounce',
      'kvx=latencySec:2,bandwidthPerDirectionBytesPerSec:0,foo:1',
      'kvo=ideal',
    ]);
    expect(bad.state.disaggregation).toEqual(decoded('pd=1').disaggregation);
    // Without pd, in the query's order.
    expect(decodeState(`kvo=none&kvl=${XDR}`).dropped).toEqual(['kvo=none', 'kvl=' + XDR]);
    expect(decoded(`pd=1&kvl=${DEFAULT_KV_LINK_ID}`)).toEqual(decoded('pd=1'));
  });

  it('build: the link reaches the input; a zero bandwidth or negative latency is an error', () => {
    const slow = apply(
      PD,
      { type: 'setKvEdit', path: 'bandwidthPerDirectionBytesPerSec', value: 1e9 },
      { type: 'setKvOverlap', overlap: 'none' },
    );
    const r = buildDisaggregatedInput(slow, GLM);
    if (r?.kind !== 'ok') throw new Error(JSON.stringify(r?.issues));
    expect(r.input.config.kvTransfer?.link.bandwidthPerDirectionBytesPerSec).toBe(1e9);
    expect(r.input.config.kvTransfer?.overlap).toBe('none');
    const zero = apply(
      PD,
      { type: 'setKvEdit', path: 'bandwidthPerDirectionBytesPerSec', value: 0 },
      { type: 'setKvEdit', path: 'latencySec', value: -1 },
    );
    const z = buildDisaggregatedInput(zero, GLM);
    expect(z?.kind).toBe('issues');
    expect(z!.issues.map((i) => i.field)).toEqual(
      expect.arrayContaining(['kvBandwidth', 'kvLatency']),
    );
  });
});

describe('format', () => {
  it('formatWorkloadQuote: P/O, then the decode context', () => {
    expect(formatWorkloadQuote(8192, 1024, 8704)).toBe('@ 8k/1k, decode @ 8,704');
    expect(formatWorkloadQuote(8192, 1024, 8192)).toBe('@ 8k/1k, decode @ 8k');
    expect(formatWorkloadQuote(1000, 300, 1150)).toBe('@ 1,000/300, decode @ 1,150');
  });

  it('formatRatio: the smaller side at 1, two decimals (design note W-PD1)', () => {
    expect(formatRatio(0.390625)).toBe('1 : 2.56');
    expect(formatRatio(1.3343442)).toBe('1.33 : 1');
    expect(formatRatio(0.5)).toBe('1 : 2.00');
    expect(formatRatio(1)).toBe('1 : 1.00');
    expect(formatRatio(0)).toBe('—');
    expect(formatRatio(Infinity)).toBe('—');
  });

  it('formatSystemRate: tok/s/GPU with the full quote', () => {
    expect(formatSystemRate(533.3333, 8192, 1024, 8704)).toBe(
      '533.3 tok/s/GPU @ 8k/1k, decode @ 8,704',
    );
  });
});
