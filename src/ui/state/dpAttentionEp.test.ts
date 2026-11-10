/**
 * The DP attention + EP layout in the UI state: the reducer, the URL keys (`lay`, `dn`), the per-field checks and
 * hints, the degree choice on hardware changes, and the state → SimInput step. GLM-5.3-Flash (288 experts, dense FFN
 * width 12288, 45 layers) on a GB200 NVL72 rack (72 GPUs in one NVLink domain).
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { buildSimInput, loadedFromHf, REFERENCE_MODEL, type LoadedModel } from './buildInput';
import { resolveHardware } from './hardware';
import {
  chooseDegrees,
  effectiveParallel,
  fieldIssues,
  shardingField,
  uiShardingMessage,
  dpAttentionEpSplits,
} from './parallelism';
import { decodeState, encodeState } from './url';

const ID = 'zai-org/GLM-5.3-Flash';

function loadGlm(): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(`${ID}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id: ID }, r);
}

const GLM = loadGlm();
const RACK = `m=${ID}&d=gb200&n=72&b=72`;
/** The rack state of a query (asserts nothing was dropped). */
function decoded(query: string): AppState {
  const { state, dropped } = decodeState(query);
  expect(dropped).toEqual([]);
  return state;
}
const RACK_STATE = decoded(RACK);
const TOPOLOGY = resolveHardware(RACK_STATE.hw).topology;
const WORKLOAD = RACK_STATE.workload;
const apply = (s: AppState, ...actions: Parameters<typeof reduce>[1][]) =>
  actions.reduce(reduce, s);

describe('reducer', () => {
  it('switching to DP attention + EP sets TP 1 and DP = EP = devices / PP', () => {
    const s = apply(RACK_STATE, { type: 'setParallelLayout', layout: 'dpAttentionEp' });
    expect(s.parallelLayout).toBe('dpAttentionEp');
    expect(s.parallel).toEqual({ tp: 1, pp: 1, dp: 72, ep: 72 });
    // PP 2 divides 72 and is kept (36 per stage); PP 5 does not and becomes 1.
    const pp2 = apply(
      RACK_STATE,
      { type: 'setParallel', parallel: { tp: 4, pp: 2, dp: 9 } },
      { type: 'setParallelLayout', layout: 'dpAttentionEp' },
    );
    expect(pp2.parallel).toEqual({ tp: 1, pp: 2, dp: 36, ep: 36 });
    const pp5 = apply(
      RACK_STATE,
      { type: 'setParallel', parallel: { pp: 5 } },
      { type: 'setParallelLayout', layout: 'dpAttentionEp' },
    );
    expect(pp5.parallel).toEqual({ tp: 1, pp: 1, dp: 72, ep: 72 });
  });

  it('dense TP is set and cleared; degrees that leave the shape leave the layout and dense TP', () => {
    const dep = apply(RACK_STATE, { type: 'setParallelLayout', layout: 'dpAttentionEp' });
    const tp = reduce(dep, { type: 'setParallel', parallel: { denseTpGroup: 'ep' } });
    expect(tp.parallel.denseTpGroup).toBe('ep');
    const cleared = reduce(tp, { type: 'setParallel', parallel: { denseTpGroup: undefined } });
    expect(cleared).toEqual(dep);
    expect('denseTpGroup' in cleared.parallel).toBe(false);
    // EP and DP together keep the shape.
    expect(reduce(tp, { type: 'setParallel', parallel: { ep: 24, dp: 24, pp: 3 } })).toMatchObject({
      parallelLayout: 'dpAttentionEp',
      parallel: { tp: 1, pp: 3, dp: 24, ep: 24, denseTpGroup: 'ep' },
    });
    // TP 2 (or DP ≠ EP) is a custom layout: no layout, no dense TP.
    const left = reduce(tp, { type: 'setParallel', parallel: { tp: 2, dp: 36 } });
    expect('parallelLayout' in left).toBe(false);
    expect(left.parallel).toEqual({ tp: 2, pp: 1, dp: 36, ep: 72 });
    // Dense TP outside the layout is never stored.
    expect(
      'denseTpGroup' in
        reduce(RACK_STATE, { type: 'setParallel', parallel: { denseTpGroup: 'ep' } }).parallel,
    ).toBe(false);
  });

  it('switching back to custom keeps the degrees and drops dense TP', () => {
    const tp = apply(
      RACK_STATE,
      { type: 'setParallelLayout', layout: 'dpAttentionEp' },
      { type: 'setParallel', parallel: { denseTpGroup: 'ep' } },
    );
    const custom = reduce(tp, { type: 'setParallelLayout', layout: 'custom' });
    expect(custom).toEqual({ ...RACK_STATE, parallel: { tp: 1, pp: 1, dp: 72, ep: 72 } });
    expect('parallelLayout' in custom).toBe(false);
  });
});

describe('URL', () => {
  it('round-trips the layout and dense TP as lay=dep and dn=tp', () => {
    const s = apply(RACK_STATE, { type: 'setParallelLayout', layout: 'dpAttentionEp' });
    expect(encodeState(s)).toBe(
      `${'m=zai-org/GLM-5.3-Flash&d=gb200&n=72'}&dp=72&ep=72&lay=dep&b=72`,
    );
    expect(decoded(encodeState(s))).toEqual(s);
    const tp = apply(s, {
      type: 'setParallel',
      parallel: { pp: 3, dp: 24, ep: 24, denseTpGroup: 'ep' },
    });
    expect(encodeState(tp)).toBe(
      'm=zai-org/GLM-5.3-Flash&d=gb200&n=72&pp=3&dp=24&ep=24&lay=dep&dn=tp&b=72',
    );
    expect(decoded(encodeState(tp))).toEqual(tp);
  });

  it('an old URL (no lay, no dn) decodes to exactly the state it did before', () => {
    const old = decoded(`${RACK}&dp=72&ep=72`);
    expect(old).toEqual({ ...RACK_STATE, parallel: { tp: 1, pp: 1, dp: 72, ep: 72 } });
    expect('parallelLayout' in old).toBe(false);
    expect('denseTpGroup' in old.parallel).toBe(false);
    expect(encodeState(old)).toBe('m=zai-org/GLM-5.3-Flash&d=gb200&n=72&dp=72&ep=72&b=72');
  });

  it('drops lay without its shape and dn without the layout or with an unknown value', () => {
    expect(decodeState(`${RACK}&tp=2&dp=36&ep=72&lay=dep`).dropped).toEqual(['lay=dep']);
    expect(decodeState(`${RACK}&dp=72&ep=36&lay=dep&dn=tp`).dropped).toEqual(['lay=dep', 'dn=tp']);
    expect(decodeState(`${RACK}&dp=72&ep=72&dn=tp`).dropped).toEqual(['dn=tp']);
    expect(decodeState(`${RACK}&dp=72&ep=72&lay=dep&dn=dp`).dropped).toEqual(['dn=dp']);
    expect(decodeState(`${RACK}&dp=72&ep=72&lay=custom`).dropped).toEqual(['lay=custom']);
  });
});

describe('effectiveParallel', () => {
  const tp = { tp: 1, pp: 1, dp: 72, ep: 72, denseTpGroup: 'ep' as const };
  it('keeps dense TP only on a MoE model in DP attention + EP', () => {
    expect(effectiveParallel(tp, GLM.model, 'dpAttentionEp')).toEqual(tp);
    expect(effectiveParallel(tp, GLM.model)).toEqual({ tp: 1, pp: 1, dp: 72, ep: 72 });
    expect(effectiveParallel(tp, REFERENCE_MODEL.model, 'dpAttentionEp')).toEqual({
      tp: 1,
      pp: 1,
      dp: 72,
      ep: 1,
    });
  });
});

describe('per-field checks and hints', () => {
  const dep = { tp: 1, pp: 1, dp: 72, ep: 72 };
  it('DP attention + EP 72 is clean; dense TP at 72 is an error on the dense-layers control', () => {
    expect(fieldIssues(GLM.model, GLM.quant.quant, dep, WORKLOAD, TOPOLOGY)).toEqual([]);
    const issues = fieldIssues(
      GLM.model,
      GLM.quant.quant,
      { ...dep, denseTpGroup: 'ep' },
      WORKLOAD,
      TOPOLOGY,
    );
    expect(issues).toEqual([
      {
        field: 'denseTp',
        level: 'error',
        message: 'Dense layers on TP need EP 72 to divide their width 12288. Use DP.',
      },
    ]);
    // At EP 24 × PP 3 the width splits (12288 / 24 = 512).
    expect(
      fieldIssues(
        GLM.model,
        GLM.quant.quant,
        { tp: 1, pp: 3, dp: 24, ep: 24, denseTpGroup: 'ep' },
        WORKLOAD,
        TOPOLOGY,
      ),
    ).toEqual([]);
  });

  it('attributes the engine dense-TP message by its wording', () => {
    const engine = 'mlp intermediate (12288) must be divisible by the dense TP group (72)';
    expect(shardingField(engine)).toBe('denseTp');
    expect(uiShardingMessage(engine)).toBe(
      'Dense layers on TP need EP 72 to divide their width 12288. Use DP.',
    );
  });

  it('splits: EP × PP = 72 with PP ≤ 45 layers, N dividing the 288 experts; with dense TP also the width 12288', () => {
    const splits = (parallel: Parameters<typeof dpAttentionEpSplits>[2], model = GLM) =>
      dpAttentionEpSplits(model.model, model.quant.quant, parallel, WORKLOAD, TOPOLOGY);
    // PP 72 (EP 1) would exceed the 45 layers; every other divisor pair of 72 works.
    expect(splits(dep)).toEqual(
      [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36].map((pp) => ({ pp, n: 72 / pp })),
    );
    // The current EP / PP do not matter: every split is checked as its own coupled state.
    expect(splits({ tp: 1, pp: 3, dp: 24, ep: 24 })).toEqual(splits(dep));
    // Dense TP: N must also divide 12288 = 2^12 · 3, so 72, 36, 18 and 9 go (and with them PP 1, 2, 4, 8).
    expect(splits({ tp: 1, pp: 3, dp: 24, ep: 24, denseTpGroup: 'ep' })).toEqual(
      [3, 6, 9, 12, 18, 24, 36].map((pp) => ({ pp, n: 72 / pp })),
    );
    // A dense model has no EP group.
    expect(splits(dep, REFERENCE_MODEL)).toEqual([]);
  });

  it('in the layout, the device-count error is worded as EP × PP', () => {
    const issues = (layout?: 'dpAttentionEp') =>
      fieldIssues(
        GLM.model,
        GLM.quant.quant,
        { tp: 1, pp: 5, dp: 24, ep: 24 },
        WORKLOAD,
        TOPOLOGY,
        layout,
      ).filter((i) => i.field === 'devices');
    expect(issues('dpAttentionEp').map((i) => i.message)).toEqual([
      'EP × PP = 120, but the rig has 72 devices. They must be equal.',
    ]);
    expect(issues().map((i) => i.message)).toEqual([
      'TP × PP × DP = 120, but the rig has 72 devices. They must be equal.',
    ]);
  });

  it('chooseDegrees keeps the layout shape: the current PP if it fits, else the smallest that does', () => {
    const choose = (current: Parameters<typeof chooseDegrees>[4]) =>
      chooseDegrees(GLM.model, GLM.quant.quant, WORKLOAD, TOPOLOGY, {
        layout: 'dpAttentionEp',
        ...current,
      });
    expect(choose({})).toEqual({ tp: 1, pp: 1, dp: 72, ep: 72 });
    expect(choose({ pp: 2 })).toEqual({ tp: 1, pp: 2, dp: 36, ep: 36 });
    // Dense TP: 12288 % 72 and % 36 ≠ 0, so PP 3 with 24 per stage.
    expect(choose({ denseTpGroup: 'ep' })).toEqual({ tp: 1, pp: 3, dp: 24, ep: 24 });
    // The batch split does not block the layout (it is shown on the batch field).
    expect(
      chooseDegrees(GLM.model, GLM.quant.quant, { ...WORKLOAD, batch: 1 }, TOPOLOGY, {
        layout: 'dpAttentionEp',
      }),
    ).toEqual({ tp: 1, pp: 1, dp: 72, ep: 72 });
    // Custom: as before (the layout fields are ignored).
    expect(
      chooseDegrees(GLM.model, GLM.quant.quant, WORKLOAD, TOPOLOGY, { dp: 72, ep: 72 }),
    ).toEqual(
      chooseDegrees(GLM.model, GLM.quant.quant, WORKLOAD, TOPOLOGY, {
        dp: 72,
        ep: 72,
        pp: 2,
        layout: undefined,
      }),
    );
  });
});

describe('buildSimInput', () => {
  it('the layout with dense DP gives exactly the input of the old mode-B URL', () => {
    const old = buildSimInput(decoded(`${RACK}&dp=72&ep=72`), GLM);
    const dep = buildSimInput(decoded(`${RACK}&dp=72&ep=72&lay=dep`), GLM);
    expect(old.kind).toBe('ok');
    expect(dep).toEqual(old);
    if (old.kind === 'ok') expect(old.input.parallel).toEqual({ tp: 1, pp: 1, dp: 72, ep: 72 });
  });

  it('dense TP reaches the engine as denseTpGroup ep; at 72 it stops on the dense-layers control', () => {
    const tp = buildSimInput(decoded(`${RACK}&pp=3&dp=24&ep=24&lay=dep&dn=tp`), GLM);
    expect(tp.kind).toBe('ok');
    if (tp.kind === 'ok')
      expect(tp.input.parallel).toEqual({ tp: 1, pp: 3, dp: 24, ep: 24, denseTpGroup: 'ep' });
    const bad = buildSimInput(decoded(`${RACK}&dp=72&ep=72&lay=dep&dn=tp`), GLM);
    expect(bad.kind).toBe('issues');
    expect(bad.issues.map((i) => i.field)).toEqual(['denseTp']);
  });

  it('a default-state build is unchanged by the layout code', () => {
    const s = { ...DEFAULT_STATE, model: { kind: 'hf' as const, id: ID } };
    const built = buildSimInput(s, GLM);
    if (built.kind === 'ok') expect(built.input.parallel).toEqual({ tp: 1, pp: 1, dp: 1, ep: 1 });
  });
});
