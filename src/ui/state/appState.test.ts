import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, reduce, type AppState } from './appState';

const apply = (s: AppState, ...actions: Parameters<typeof reduce>[1][]) => actions.reduce(reduce, s);

describe('reduce', () => {
  it('changing the device resets unlocks, engine, capacity, links and edits, keeping the counts', () => {
    const s = apply(
      DEFAULT_STATE,
      { type: 'setCounts', devicesPerNode: 2 },
      { type: 'setUnlock', axisId: 'pcie-p2p', optionId: 'direct' },
      { type: 'setLink', level: 'intra', linkId: 'pcie-5-0-x16-direct-p2p' },
      { type: 'setEdit', target: 'device', path: 'memory.bandwidthBytesPerSec', value: 2e12 },
      { type: 'setEdit', target: 'intra', path: 'latencySec', value: 1e-6 },
      { type: 'setEngine', engineId: 'gpu' },
    );
    expect(s.hw.intraLinkId).toBe('pcie-5-0-x16-direct-p2p');
    const next = reduce(s, { type: 'setDevice', deviceId: 'rtx-4090' });
    expect(next.hw).toEqual({ ...DEFAULT_STATE.hw, deviceId: 'rtx-4090', devicesPerNode: 2 });
  });

  it('changing the counts re-picks the default links and clears link edits', () => {
    const s = apply(
      DEFAULT_STATE,
      { type: 'setCounts', devicesPerNode: 2, nodes: 2 },
      { type: 'setLink', level: 'inter', linkId: '10gbe-tcp' },
      { type: 'setEdit', target: 'inter', path: 'latencySec', value: 1e-6 },
      { type: 'setEdit', target: 'device', path: 'memory.bandwidthBytesPerSec', value: 2e12 },
      { type: 'setCounts', nodes: 3 },
    );
    expect(s.hw.interLinkId).toBeUndefined();
    expect(s.hw.interEdits).toEqual({});
    expect(s.hw.deviceEdits).toEqual({ 'memory.bandwidthBytesPerSec': 2e12 });
    expect(s.hw).toMatchObject({ devicesPerNode: 2, nodes: 3 });
  });

  it('setLink clears that level edits; setEdit undefined removes one edit; resetEdits clears', () => {
    let s = apply(DEFAULT_STATE, { type: 'setEdit', target: 'intra', path: 'latencySec', value: 1e-6 }, { type: 'setLink', level: 'intra', linkId: 'nvlink4-nvswitch' });
    expect(s.hw.intraEdits).toEqual({});
    s = apply(s, { type: 'setEdit', target: 'device', path: 'a', value: 1 }, { type: 'setEdit', target: 'device', path: 'b', value: 2 }, { type: 'setEdit', target: 'device', path: 'a', value: undefined });
    expect(s.hw.deviceEdits).toEqual({ b: 2 });
    expect(reduce(s, { type: 'resetEdits' }).hw.deviceEdits).toEqual({});
  });

  it('quant overrides: set, clear, reset; a new model clears them; rev main is normalized', () => {
    let s = apply(DEFAULT_STATE, { type: 'setQuant', field: 'kvCache', dtype: 'fp8_e4m3' }, { type: 'setQuant', field: 'weights', dtype: 'nvfp4' });
    expect(s.quant).toEqual({ kvCache: 'fp8_e4m3', weights: 'nvfp4' });
    expect(reduce(s, { type: 'setQuant', field: 'weights', dtype: undefined }).quant).toEqual({ kvCache: 'fp8_e4m3' });
    expect(reduce(s, { type: 'resetQuant' }).quant).toEqual({});
    s = reduce(s, { type: 'setModel', model: { kind: 'hf', id: 'Qwen/Qwen3.8-27B', revision: 'main' } });
    expect(s.model).toEqual({ kind: 'hf', id: 'Qwen/Qwen3.8-27B' });
    expect(s.quant).toEqual({});
  });

  it('options keep only non-default values; unlocks reset the capacity index', () => {
    const s = apply(DEFAULT_STATE, { type: 'setOptions', options: { overlap: 'none', collectiveAlgorithm: 'best' } });
    expect(s.options).toEqual({ overlap: 'none' });
    const c = apply(DEFAULT_STATE, { type: 'setDevice', deviceId: 'apple-m3-ultra-80c' }, { type: 'setCapacity', index: 2 });
    expect(c.hw.capacityIndex).toBe(2);
    expect(reduce(c, { type: 'setUnlock', axisId: 'x', optionId: undefined }).hw.capacityIndex).toBe(0);
  });

  it('never mutates the input state', () => {
    const frozen = JSON.parse(JSON.stringify(DEFAULT_STATE)) as AppState;
    apply(frozen, { type: 'setParallel', parallel: { tp: 2 } }, { type: 'setWorkload', workload: { batch: 4 } });
    expect(frozen).toEqual(DEFAULT_STATE);
  });
});
