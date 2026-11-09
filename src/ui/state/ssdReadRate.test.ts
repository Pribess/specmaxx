import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, reduce } from './appState';
import { buildSimInput, loadedFromHf } from './buildInput';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { normalizeOffload, type OffloadChoice } from './deployment';
import { decodeState, encodeState } from './url';

/** Qwen3.8-Flash-Next (a MoE with PLE tables), mapped from the committed fixtures. */
function flashNext() {
  const id = 'Qwen/Qwen3.8-Flash-Next';
  const r = mapHfModel(fixtureRepoFiles(id, { dir: '' }));
  if (r.kind !== 'ok') throw new Error(r.kind);
  return loadedFromHf({ kind: 'hf', id }, r);
}
describe('SSD read rate', () => {
  it.each<OffloadChoice>([
    { experts: { tier: 'nvme' } },
    { experts: { tier: 'split' } },
    { tables: { tier: 'nvme' } },
  ])('round-trips sustained mode with each SSD group: %j', (placement) => {
    const state = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { ...placement, ssdReadRate: 'sustained' },
    });
    const query = encodeState(state);
    expect(query).toContain('osr=sustained');
    const decoded = decodeState(query);
    expect(decoded.dropped).toEqual([]);
    expect(decoded.state.deployment).toEqual(state.deployment);
    expect(encodeState(decoded.state)).toBe(query);
  });

  it('omits datasheet defaults and removes sustained mode without SSD placement', () => {
    const placement = { experts: { tier: 'nvme' as const } };
    expect(normalizeOffload({ ...placement, ssdReadRate: 'datasheet' })).toEqual(placement);
    expect(encodeState(decodeState('ox=ssd&osr=datasheet').state)).toBe('ox=ssd');
    expect(normalizeOffload({ ssdReadRate: 'sustained' })).toEqual({});
    expect(normalizeOffload({ experts: { tier: 'host_ram' }, ssdReadRate: 'sustained' })).toEqual({
      experts: { tier: 'host_ram' },
    });
    expect(encodeState(DEFAULT_STATE)).toBe('');
  });

  it.each(['osr=sustained', 'ox=ram&osr=sustained', 'ox=ssd&osr=invalid'])(
    'reports unsupported or orphan URL values: %s',
    (query) => {
      const decoded = decodeState(query);
      expect(decoded.dropped).toEqual([query.split('&').at(-1)]);
      expect(decoded.state.deployment.offload?.ssdReadRate).toBeUndefined();
    },
  );

  it('passes sustained mode to the engine but leaves datasheet mode absent', () => {
    const flash = flashNext();
    for (const rate of ['datasheet', 'sustained'] as const) {
      const state = reduce(
        { ...DEFAULT_STATE, model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' } },
        { type: 'setOffload', offload: { experts: { tier: 'nvme' }, ssdReadRate: rate } },
      );
      const built = buildSimInput(state, flash);
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      expect(built.input.deployment?.offload?.ssdReadRate).toBe(
        rate === 'sustained' ? rate : undefined,
      );
    }
  });
});
