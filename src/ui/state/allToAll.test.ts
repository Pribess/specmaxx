/**
 * The expert all-to-all knobs in the UI state (`ParallelConfig.dispatchFormat`, `combineFormat`,
 * `allToAllEfficiency`, docs §5 "Dispatch and combine precision", §6 "All-to-all efficiency"): the reducer keeps only
 * non-default values, the URL codec round-trips them (`dsp`, `cmb`, `a2e` in percent), the checks and the SimInput see
 * them only in MoE mode B (EP > TP). GLM-5.3-Flash on a GB200 NVL72 rack.
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { DEFAULT_STATE, normalizeAllToAll, reduce } from './appState';
import { buildSimInput, loadedFromHf, REFERENCE_MODEL, type LoadedModel } from './buildInput';
import { resolveHardware } from './hardware';
import { effectiveParallel, fieldIssues } from './parallelism';
import { decodeState, encodeState } from './url';

const ID = 'zai-org/GLM-5.3-Flash';
function loadGlm(): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(`${ID}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id: ID }, r);
}
const GLM = loadGlm();
const RACK = `m=${ID}&d=gb200&n=72&b=72`;
const EP72 = { tp: 1, pp: 1, dp: 72, ep: 72 };
const KNOBS = {
  dispatchFormat: 'nvfp4',
  combineFormat: 'fp8_e4m3',
  allToAllEfficiency: 0.7,
} as const;

describe('all-to-all knobs', () => {
  it('the reducer keeps only non-default values (η 1 and absent formats are absent)', () => {
    expect(
      normalizeAllToAll({ ...EP72, allToAllEfficiency: 1, dispatchFormat: undefined }),
    ).toEqual(EP72);
    expect(normalizeAllToAll({ ...EP72, ...KNOBS })).toEqual({ ...EP72, ...KNOBS });
    expect(normalizeAllToAll({ ...EP72, allToAllEfficiency: 0 })).toEqual({
      ...EP72,
      allToAllEfficiency: 0,
    });
    const s = reduce(DEFAULT_STATE, { type: 'setParallel', parallel: { dispatchFormat: 'mxfp4' } });
    expect(s.parallel.dispatchFormat).toBe('mxfp4');
    expect(reduce(s, { type: 'setParallel', parallel: { dispatchFormat: undefined } })).toEqual(
      DEFAULT_STATE,
    );
  });

  it('URL: dsp, cmb, a2e (percent); defaults omitted; invalid values dropped', () => {
    const q = `m=${ID}&d=gb200&n=72&dp=72&ep=72&dsp=nvfp4&cmb=fp8_e4m3&a2e=70&b=72`;
    const { state, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(state.parallel).toEqual({ ...EP72, ...KNOBS });
    expect(encodeState(state)).toBe(q);
    expect(decodeState('ep=2&dp=2&a2e=100')).toEqual(decodeState('ep=2&dp=2'));
    const set = reduce(DEFAULT_STATE, { type: 'setParallel', parallel: KNOBS });
    expect(encodeState(set)).toBe('dsp=nvfp4&cmb=fp8_e4m3&a2e=70');
    expect(
      encodeState(
        reduce(set, {
          type: 'setParallel',
          parallel: { allToAllEfficiency: 1, dispatchFormat: undefined, combineFormat: undefined },
        }),
      ),
    ).toBe('');
    for (const [key, v] of [
      ['dsp', 'fp8'],
      ['cmb', 'x'],
      ['a2e', '0'],
      ['a2e', '101'],
      ['a2e', 'x'],
    ]) {
      const d = decodeState(`ep=2&dp=2&${key}=${v}`);
      expect(d.dropped).toEqual([`${key}=${v}`]);
      expect(d.state.parallel).toEqual({ tp: 1, pp: 1, dp: 2, ep: 2 });
    }
    // 33.3% survives the float round trip
    expect(encodeState(decodeState('a2e=33.3').state)).toBe('a2e=33.3');
  });

  it('effectiveParallel keeps them only on a MoE model with EP > TP (mode B)', () => {
    expect(effectiveParallel({ ...EP72, ...KNOBS }, GLM.model)).toEqual({ ...EP72, ...KNOBS });
    expect(effectiveParallel({ ...EP72, ep: 1, ...KNOBS }, GLM.model)).toEqual({ ...EP72, ep: 1 });
    // mode A: EP inside TP
    expect(effectiveParallel({ tp: 8, pp: 1, dp: 9, ep: 8, ...KNOBS }, GLM.model)).toEqual({
      tp: 8,
      pp: 1,
      dp: 9,
      ep: 8,
    });
    expect(effectiveParallel({ ...EP72, ...KNOBS }, REFERENCE_MODEL.model)).toEqual({
      ...EP72,
      ep: 1,
    });
  });

  it('an invalid η is an error on its own field', () => {
    const topology = resolveHardware(decodeState(RACK).state.hw).topology;
    const workload = decodeState(RACK).state.workload;
    for (const eta of [0, 1.5, -1]) {
      const issues = fieldIssues(
        GLM.model,
        GLM.quant.quant,
        { ...EP72, allToAllEfficiency: eta },
        workload,
        topology,
      );
      expect(issues.map((i) => [i.field, i.level])).toEqual([['allToAllEfficiency', 'error']]);
    }
    expect(
      fieldIssues(GLM.model, GLM.quant.quant, { ...EP72, ...KNOBS }, workload, topology),
    ).toEqual([]);
  });

  it('the SimInput carries them only when set, in mode B; absent knobs give the input from before', () => {
    const plain = buildSimInput(decodeState(`${RACK}&dp=72&ep=72`).state, GLM);
    const set = buildSimInput(
      decodeState(`${RACK}&dp=72&ep=72&dsp=nvfp4&cmb=fp8_e4m3&a2e=70`).state,
      GLM,
    );
    if (plain.kind !== 'ok' || set.kind !== 'ok') throw new Error('expected ok');
    expect(plain.input.parallel).toEqual(EP72);
    expect(set.input.parallel).toEqual({ ...EP72, ...KNOBS });
    const ep1 = buildSimInput(decodeState(`${RACK}&dp=72&dsp=nvfp4&a2e=70`).state, GLM);
    const ep1Plain = buildSimInput(decodeState(`${RACK}&dp=72`).state, GLM);
    if (ep1.kind !== 'ok' || ep1Plain.kind !== 'ok') throw new Error('expected ok');
    expect(ep1.input).toEqual(ep1Plain.input);
  });
});
