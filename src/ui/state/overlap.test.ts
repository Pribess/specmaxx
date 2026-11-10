/**
 * Per-collective overlap (`SimOptions.overlapByClass`) and two-batch overlap (`DeploymentChoice.twoBatchOverlap`) in
 * the UI state (docs/simulation-model.md §7 "Stage time and overlap", "Two-batch overlap"): the reducer keeps set
 * entries only, the URL codec round-trips them (`ovt`, `ove`, `ovp` in percent, `tbo=1`) and leaves old URLs alone,
 * and the SimInput carries them only for the classes / layouts the run has. GLM-5.3-Flash on a GB200 NVL72 rack.
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { DEFAULT_STATE, normalizeOptions, reduce } from './appState';
import { buildSimInput, loadedFromHf, REFERENCE_MODEL, type LoadedModel } from './buildInput';
import { effectiveOptions, overlapClassesOf, twoBatchOverlapApplies } from './overlap';
import { decodeState, encodeState } from './url';

const ID = 'zai-org/GLM-5.3-Flash';
function loadGlm(): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(`${ID}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id: ID }, r);
}
const GLM = loadGlm();
const DENSE = REFERENCE_MODEL.model;
const RACK = `m=${ID}&d=gb200&n=72&b=72`;
const EP72 = { tp: 1, pp: 1, dp: 72, ep: 72 };

describe('per-class overlap and two-batch overlap', () => {
  it('normalizeOptions keeps set classes (even at a default value) and drops an empty map', () => {
    expect(normalizeOptions({ overlapByClass: {} })).toEqual({});
    expect(normalizeOptions({ overlapByClass: { tp: undefined } })).toEqual({});
    expect(
      normalizeOptions({ overlap: 'ideal', overlapByClass: { tp: 1, ep: undefined } }),
    ).toEqual({ overlapByClass: { tp: 1 } });
  });

  it('the reducer sets and clears one class; two-batch on clears an explicit EP choice', () => {
    let s = reduce(DEFAULT_STATE, { type: 'setOverlapClass', overlapClass: 'ep', fraction: 0.5 });
    s = reduce(s, { type: 'setOverlapClass', overlapClass: 'tp', fraction: 0 });
    expect(s.options).toEqual({ overlapByClass: { ep: 0.5, tp: 0 } });
    const tbo = reduce(s, { type: 'setTwoBatchOverlap', on: true });
    expect(tbo.options).toEqual({ overlapByClass: { tp: 0 } });
    expect(tbo.deployment).toEqual({ twoBatchOverlap: true });
    const off = reduce(tbo, { type: 'setTwoBatchOverlap', on: false });
    expect(off.deployment).toEqual({});
    expect(
      reduce(off, { type: 'setOverlapClass', overlapClass: 'tp', fraction: undefined }),
    ).toEqual(DEFAULT_STATE);
  });

  it('URL: ovt, ove, ovp in percent and tbo=1, right after ov; defaults omitted; invalid values dropped', () => {
    const q = `m=${ID}&d=gb200&n=72&dp=72&ep=72&b=72&ov=none&ovt=0&ove=37.5&ovp=100&tbo=1`;
    const { state, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(state.options).toEqual({ overlap: 'none', overlapByClass: { tp: 0, ep: 0.375, pp: 1 } });
    expect(state.deployment).toEqual({ twoBatchOverlap: true });
    expect(encodeState(state)).toBe(q);
    // Old URLs are unchanged: nothing set, nothing written.
    expect(encodeState(decodeState(`${RACK}&ov=none`).state)).toBe(`${RACK}&ov=none`);
    expect(encodeState(DEFAULT_STATE)).toBe('');
    // ≤ 4 decimals of percent survive the float round trip.
    expect(encodeState(decodeState('ove=33.3333').state)).toBe('ove=33.3333');
    for (const [key, v] of [
      ['ovt', '-1'],
      ['ove', '101'],
      ['ovp', 'x'],
      ['ovt', ''],
      ['tbo', '0'],
      ['tbo', 'yes'],
    ]) {
      const d = decodeState(`${key}=${v}`);
      expect(d.dropped).toEqual([`${key}=${v}`]);
      expect(d.state).toEqual(DEFAULT_STATE);
    }
  });

  it('overlapClassesOf: tp with TP > 1, ep in MoE mode B, pp with PP > 1', () => {
    expect(overlapClassesOf(GLM.model, EP72)).toEqual(['ep']);
    expect(overlapClassesOf(GLM.model, { tp: 8, pp: 2, dp: 1, ep: 8 })).toEqual(['tp', 'pp']);
    expect(overlapClassesOf(GLM.model, { tp: 2, pp: 2, dp: 4, ep: 8 })).toEqual(['tp', 'ep', 'pp']);
    expect(overlapClassesOf(DENSE, { tp: 1, pp: 1, dp: 8, ep: 8 })).toEqual([]);
    expect(twoBatchOverlapApplies(GLM.model, EP72, false)).toBe(true);
    expect(twoBatchOverlapApplies(GLM.model, EP72, true)).toBe(false);
    expect(twoBatchOverlapApplies(DENSE, EP72, false)).toBe(false);
    expect(
      effectiveOptions({ overlap: 'none', overlapByClass: { tp: 0.5 } }, GLM.model, EP72),
    ).toEqual({
      overlap: 'none',
    });
  });

  it('the SimInput carries them only where they apply; absent choices give the input from before', () => {
    const build = (q: string) => {
      const r = buildSimInput(decodeState(`${RACK}&${q}`).state, GLM);
      if (r.kind !== 'ok') throw new Error(JSON.stringify(r.issues));
      return r;
    };
    const plain = build('dp=72&ep=72').input;
    expect(plain.options).toBeUndefined();
    expect(plain.deployment).toBeUndefined();
    const set = build('dp=72&ep=72&ove=50&tbo=1').input;
    expect(set.options).toEqual({ overlapByClass: { ep: 0.5 } });
    expect(set.deployment).toEqual({ twoBatchOverlap: true });
    // A class the run lacks (TP 1, PP 1) and two-batch overlap on a layout without an all-to-all change nothing.
    expect(build('dp=72&ovt=50&ovp=0&tbo=1').input).toEqual(build('dp=72').input);
    // With offloaded experts two-batch overlap is not modelled: off, with a warning on its switch.
    const offload = build('dp=72&ep=72&tbo=1&ox=ram');
    expect(offload.input.deployment?.twoBatchOverlap).toBeUndefined();
    expect(offload.issues.filter((i) => i.field === 'twoBatchOverlap').map((i) => i.level)).toEqual(
      ['warning'],
    );
  });
});
