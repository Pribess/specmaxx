import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, reduce } from './appState';
import { buildSimInput } from './buildInput';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import { normalizeOffload, type OffloadChoice } from './deployment';
import { decodeState, encodeState } from './url';

describe('SSD read rate', () => {
  it.each<OffloadChoice>([
    { experts: { tier: 'nvme' } },
    { dense: { tier: 'nvme', layersOnDevice: 0 } },
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
    const placement = { dense: { tier: 'nvme' as const, layersOnDevice: 0 } };
    expect(normalizeOffload({ ...placement, ssdReadRate: 'datasheet' })).toEqual(placement);
    expect(encodeState(decodeState('odl=0&odt=ssd&osr=datasheet').state)).toBe('odl=0&odt=ssd');
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
    for (const rate of ['datasheet', 'sustained'] as const) {
      const state = reduce(DEFAULT_STATE, {
        type: 'setOffload',
        offload: { dense: { tier: 'nvme', layersOnDevice: 0 }, ssdReadRate: rate },
      });
      const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      expect(built.input.deployment?.offload?.ssdReadRate).toBe(
        rate === 'sustained' ? rate : undefined,
      );
    }
  });
});
