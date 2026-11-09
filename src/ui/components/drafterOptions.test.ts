import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import type { ModelSpec } from '../../sim/ir/model';
import { isValidModelId } from '../../hf/urls';
import { DEFAULT_STATE, decodeState, drafterSource, encodeState, reduce } from '../state';
import { DEFAULT_DRAFTER_OPTION, drafterChoices, drafterFromCommit } from './drafterOptions';

function load(id: string): ModelSpec {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${r.kind}`);
  return r.model;
}
const DEEPSEEK_ID = 'deepseek-ai/DeepSeek-V4.1-Flash';
const DEEPSEEK = load(DEEPSEEK_ID);
const QWEN_ID = 'Qwen/Qwen3.8-27B';
const QWEN = load(QWEN_ID);
const QWEN_DRAFTER = 'incoai/Qwen3.8-27B-DFlash2';

/** One user step on the drafter field: commit `value`, as the combobox does, and return the new URL query. */
function pick(query: string, id: string, model: ModelSpec, value: string): string {
  const s = decodeState(query).state;
  const spec = s.deployment.specDec;
  if (spec === undefined || spec.method === 'mtp') throw new Error('block method expected');
  const choices = drafterChoices(spec.method, id, model, spec.drafter);
  const next = drafterFromCommit(value, choices, isValidModelId);
  if (next === 'invalid') return query;
  const out = reduce(s, { type: 'setSpecDec', specDec: { ...spec, drafter: next } });
  return encodeState(out);
}

describe('drafterChoices', () => {
  it('lists the checkpoint’s own drafter first as the model default, then the stand-in', () => {
    expect(DEEPSEEK.draft?.kind).toBe('block_drafter');
    const c = drafterChoices('dspark', DEEPSEEK_ID, DEEPSEEK, undefined);
    expect(c.options.map((o) => o.label)).toEqual([
      'Model default (checkpoint’s DSpark drafter)',
      'Hypothetical drafter',
    ]);
    expect(c.options[0].value).toBe(DEFAULT_DRAFTER_OPTION);
    expect(c.options[0].description).toBe('3 layers');
    expect(c.selected).toBe(DEFAULT_DRAFTER_OPTION);
    expect(c.text).toBe(c.options[0].label);
  });

  it('has no checkpoint-default entry for a method the checkpoint does not ship', () => {
    const c = drafterChoices('dflash2', DEEPSEEK_ID, DEEPSEEK, undefined);
    expect(c.options.map((o) => o.label)).toEqual(['Hypothetical drafter']);
    expect(c.selected).toBe('hyp');
  });

  it('lists a released default once, as the model default, without a second entry for its repo', () => {
    const c = drafterChoices('dflash2', QWEN_ID, QWEN, undefined);
    expect(c.options.map((o) => o.label)).toEqual([
      `Model default (${QWEN_DRAFTER})`,
      'Hypothetical drafter',
    ]);
    expect(c.options.filter((o) => o.value === QWEN_DRAFTER)).toEqual([]);
    expect(c.text).toBe(`Model default (${QWEN_DRAFTER})`);
  });

  it('shows an arbitrary repo id as itself and the hypothetical choice as selected', () => {
    const repo = drafterChoices('dspark', DEEPSEEK_ID, DEEPSEEK, 'someone/my-drafter');
    expect([repo.selected, repo.text]).toEqual(['someone/my-drafter', 'someone/my-drafter']);
    const hyp = drafterChoices('dspark', DEEPSEEK_ID, DEEPSEEK, 'hyp');
    expect([hyp.selected, hyp.text]).toEqual(['hyp', 'Hypothetical drafter']);
  });

  it('treats the released default’s repo id (from the URL) as the model default', () => {
    const c = drafterChoices('dflash2', QWEN_ID, QWEN, QWEN_DRAFTER);
    expect(c.selected).toBe(DEFAULT_DRAFTER_OPTION);
    expect(c.text).toBe(`Model default (${QWEN_DRAFTER})`);
  });
});

describe('drafterFromCommit', () => {
  const c = drafterChoices('dspark', DEEPSEEK_ID, DEEPSEEK, undefined);
  it('maps the default, its label, and empty text to no override; keeps arbitrary repo ids', () => {
    expect(drafterFromCommit(DEFAULT_DRAFTER_OPTION, c, isValidModelId)).toBeUndefined();
    expect(drafterFromCommit(c.options[0].label, c, isValidModelId)).toBeUndefined();
    expect(drafterFromCommit('  ', c, isValidModelId)).toBeUndefined();
    expect(drafterFromCommit('hyp', c, isValidModelId)).toBe('hyp');
    expect(drafterFromCommit(' org/custom-drafter ', c, isValidModelId)).toBe('org/custom-drafter');
    expect(drafterFromCommit('not a repo', c, isValidModelId)).toBe('invalid');
  });
});

describe('drafter picker URL round trips', () => {
  it('DeepSeek-V4.1-Flash + DSpark: hypothetical, then back to the checkpoint’s drafter drops sdr', () => {
    const base = `m=${DEEPSEEK_ID}&sd=dspark`;
    const start = decodeState(base).state.deployment.specDec;
    expect(drafterSource(start, DEEPSEEK, DEEPSEEK_ID)?.kind).toBe('checkpoint');

    const hyp = pick(base, DEEPSEEK_ID, DEEPSEEK, 'hyp');
    expect(hyp).toContain('sdr=hyp');
    const hypSpec = decodeState(hyp).state.deployment.specDec;
    expect(drafterSource(hypSpec, DEEPSEEK, DEEPSEEK_ID)?.kind).toBe('hypothetical');

    const back = pick(hyp, DEEPSEEK_ID, DEEPSEEK, DEFAULT_DRAFTER_OPTION);
    expect(back).not.toContain('sdr');
    const backSpec = decodeState(back).state.deployment.specDec;
    expect(drafterSource(backSpec, DEEPSEEK, DEEPSEEK_ID)?.kind).toBe('checkpoint');

    const typed = pick(base, DEEPSEEK_ID, DEEPSEEK, 'someone/my-drafter');
    expect(typed).toContain('sdr=someone/my-drafter');
    expect(
      drafterSource(decodeState(typed).state.deployment.specDec, DEEPSEEK, DEEPSEEK_ID),
    ).toEqual({
      kind: 'repo',
      repoId: 'someone/my-drafter',
      known: false,
    });
  });

  it('a released default round-trips too: hypothetical, then back to the released repo, no sdr', () => {
    const base = `m=${QWEN_ID}&sd=dflash2`;
    const hyp = pick(base, QWEN_ID, QWEN, 'hyp');
    expect(hyp).toContain('sdr=hyp');
    const back = pick(hyp, QWEN_ID, QWEN, DEFAULT_DRAFTER_OPTION);
    expect(back).not.toContain('sdr');
    expect(drafterSource(decodeState(back).state.deployment.specDec, QWEN, QWEN_ID)).toEqual({
      kind: 'repo',
      repoId: QWEN_DRAFTER,
      known: true,
    });
    // Typing the default's repo id is the default as well, not a second spelling in the URL.
    expect(pick(hyp, QWEN_ID, QWEN, QWEN_DRAFTER)).not.toContain('sdr');
  });

  it('keeps the default state free of sdr', () => {
    expect(encodeState(DEFAULT_STATE)).not.toContain('sdr');
  });
});
