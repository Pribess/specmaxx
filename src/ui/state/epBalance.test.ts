/**
 * The EP balance knobs in the UI state (`ParallelConfig.epImbalance` ρ and `redundantExpertsPerRank`, docs §5
 * "EP load imbalance", "Redundant experts"): the reducer keeps only non-default values, the checks see them only on
 * a MoE model at EP > 1, and the SimInput carries them only then. GLM-5.3-Flash on a GB200 NVL72 rack.
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { DEFAULT_STATE, normalizeEpBalance, reduce } from './appState';
import { buildSimInput, loadedFromHf, REFERENCE_MODEL, type LoadedModel } from './buildInput';
import { resolveHardware } from './hardware';
import { effectiveParallel, fieldIssues, validDegrees } from './parallelism';
import { decodeState } from './url';

const ID = 'zai-org/GLM-5.3-Flash';
function loadGlm(): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(`${ID}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id: ID }, r);
}
const GLM = loadGlm();
const RACK = `m=${ID}&d=gb200&n=72&b=72`;
const EP72 = { tp: 1, pp: 1, dp: 72, ep: 72 };

describe('EP balance knobs', () => {
  it('the reducer keeps only non-default values (ρ 1.2 and 0 replicas are absent)', () => {
    expect(
      normalizeEpBalance({
        ...EP72,
        epImbalance: 1.2,
        epImbalanceScalesAllToAll: false,
        redundantExpertsPerRank: 0,
      }),
    ).toEqual(EP72);
    expect(normalizeEpBalance({ ...EP72, epImbalanceScalesAllToAll: true })).toEqual({
      ...EP72,
      epImbalanceScalesAllToAll: true,
    });
    expect(normalizeEpBalance({ ...EP72, epImbalance: 0, redundantExpertsPerRank: 1.5 })).toEqual({
      ...EP72,
      epImbalance: 0,
      redundantExpertsPerRank: 1.5,
    });
    const s = reduce(DEFAULT_STATE, { type: 'setParallel', parallel: { epImbalance: 1.5 } });
    expect(s.parallel).toEqual({ ...DEFAULT_STATE.parallel, epImbalance: 1.5 });
    expect(reduce(s, { type: 'setParallel', parallel: { tp: 2 } }).parallel.epImbalance).toBe(1.5);
  });

  it('effectiveParallel keeps them only on a MoE model at EP > 1', () => {
    const knobs = { epImbalance: 1.5, epImbalanceScalesAllToAll: true, redundantExpertsPerRank: 1 };
    expect(effectiveParallel({ ...EP72, ...knobs }, GLM.model)).toEqual({ ...EP72, ...knobs });
    expect(effectiveParallel({ ...EP72, ep: 1, ...knobs }, GLM.model)).toEqual({ ...EP72, ep: 1 });
    expect(effectiveParallel({ ...EP72, ...knobs }, REFERENCE_MODEL.model)).toEqual({
      ...EP72,
      ep: 1,
    });
  });

  it('invalid values are errors on their own fields', () => {
    const topology = resolveHardware(decodeState(RACK).state.hw).topology;
    const workload = decodeState(RACK).state.workload;
    const issues = fieldIssues(
      GLM.model,
      GLM.quant.quant,
      { ...EP72, epImbalance: 0.5, redundantExpertsPerRank: -1 },
      workload,
      topology,
    );
    expect(issues).toEqual([
      { field: 'epImbalance', level: 'error', message: 'EP imbalance must be a number ≥ 1.' },
      {
        field: 'redundantExperts',
        level: 'error',
        message: 'Redundant experts must be a whole number ≥ 0.',
      },
    ]);
    // They do not stop the other checks: the degree hints are the ones without them
    const bad = { ...EP72, epImbalance: 0.5 };
    expect(validDegrees('ep', GLM.model, GLM.quant.quant, bad, workload, topology)).toEqual(
      validDegrees('ep', GLM.model, GLM.quant.quant, EP72, workload, topology),
    );
  });

  it('the SimInput carries them only when set; absent knobs give the input from before', () => {
    const plain = buildSimInput(decodeState(`${RACK}&dp=72&ep=72`).state, GLM);
    const set = buildSimInput(decodeState(`${RACK}&dp=72&ep=72&epi=1.5&epa=1&rex=1`).state, GLM);
    if (plain.kind !== 'ok' || set.kind !== 'ok') throw new Error('expected ok');
    expect(plain.input.parallel).toEqual(EP72);
    expect(set.input.parallel).toEqual({
      ...EP72,
      epImbalance: 1.5,
      epImbalanceScalesAllToAll: true,
      redundantExpertsPerRank: 1,
    });
    // At EP 1 the stored knobs are dropped: the input equals the one without them
    const ep1 = buildSimInput(decodeState(`${RACK}&dp=72&epi=1.5&epa=1&rex=1`).state, GLM);
    const ep1Plain = buildSimInput(decodeState(`${RACK}&dp=72`).state, GLM);
    if (ep1.kind !== 'ok' || ep1Plain.kind !== 'ok') throw new Error('expected ok');
    expect(ep1.input).toEqual(ep1Plain.input);
  });
});
