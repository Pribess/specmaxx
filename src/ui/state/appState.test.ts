import { describe, expect, it } from 'vitest';
import { linkPresetId } from '../../sim/hw/presets/topologies';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { localLinkChoices, resolveHardware } from './hardware';

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

  it('changing the counts keeps a level\'s link and edits while the level exists, and drops them when it goes', () => {
    const s = apply(
      DEFAULT_STATE,
      { type: 'setCounts', devicesPerNode: 2, nodes: 2 },
      { type: 'setLink', level: 'inter', linkId: '10gbe-tcp' },
      { type: 'setEdit', target: 'inter', path: 'latencySec', value: 1e-6 },
      { type: 'setEdit', target: 'device', path: 'memory.bandwidthBytesPerSec', value: 2e12 },
      { type: 'setCounts', nodes: 4 },
    );
    expect(s.hw.interLinkId).toBe('10gbe-tcp');
    expect(s.hw.interEdits).toEqual({ latencySec: 1e-6 });
    expect(s.hw.deviceEdits).toEqual({ 'memory.bandwidthBytesPerSec': 2e12 });
    expect(s.hw).toMatchObject({ devicesPerNode: 2, nodes: 4 });
    const one = reduce(s, { type: 'setCounts', nodes: 1 });
    expect(one.hw.interLinkId).toBeUndefined();
    expect(one.hw.interEdits).toEqual({});
  });

  it('cabling: kept while the counts and link allow it, cleared otherwise; a default cabling is stored as absent', () => {
    const spark = apply(DEFAULT_STATE, { type: 'setDevice', deviceId: 'dgx-spark' }, { type: 'setCounts', nodes: 4 });
    const ring = reduce(spark, { type: 'setCabling', cabling: 'ring' });
    expect(ring.hw.cabling).toBe('ring');
    // 3 machines: a ring of three is the full mesh, so it is not offered; the default (switch) applies.
    expect(reduce(ring, { type: 'setCounts', nodes: 3 }).hw.cabling).toBeUndefined();
    expect(reduce(ring, { type: 'setCabling', cabling: 'switched' }).hw.cabling).toBeUndefined();
    expect(reduce(ring, { type: 'setCabling', cabling: undefined }).hw.cabling).toBeUndefined();
    // Another device resets the cabling.
    expect(reduce(ring, { type: 'setDevice', deviceId: 'ryzen-ai-max-plus-395' }).hw.cabling).toBeUndefined();
    // A link change keeps an allowed cabling.
    const halo = apply(DEFAULT_STATE, { type: 'setDevice', deviceId: 'ryzen-ai-max-plus-395' }, { type: 'setCounts', nodes: 4 }, { type: 'setCabling', cabling: 'chain' });
    expect(halo.hw.cabling).toBe('chain');
    expect(reduce(halo, { type: 'setLink', level: 'inter', linkId: 'strix-halo-usb4-tcp-20-gb-s-negotiated' }).hw.cabling).toBe('chain');
  });

  it('another device snaps the counts to the nearest offered ones', () => {
    const s = apply(DEFAULT_STATE, { type: 'setCounts', devicesPerNode: 4 }, { type: 'setDevice', deviceId: 'dgx-spark' });
    expect(s.hw.devicesPerNode).toBe(1);
  });

  it('the 2-card NVLink bridge stays valid: more GPUs fall back to PCIe, picking it snaps to 2', () => {
    const bridge = apply(
      DEFAULT_STATE,
      { type: 'setDevice', deviceId: 'rtx-3090' },
      { type: 'setCounts', devicesPerNode: 2 },
      { type: 'setLink', level: 'intra', linkId: 'rtx-3090-nvlink-bridge' },
      { type: 'setEdit', target: 'intra', path: 'latencySec', value: 1e-6 },
    );
    expect(bridge.hw).toMatchObject({ devicesPerNode: 2, intraLinkId: 'rtx-3090-nvlink-bridge' });
    // 4 GPUs: the bridge (and its edits) is dropped, so the default PCIe link applies.
    const four = reduce(bridge, { type: 'setCounts', devicesPerNode: 4 });
    expect(four.hw.devicesPerNode).toBe(4);
    expect(four.hw.intraLinkId).toBeUndefined();
    expect(four.hw.intraEdits).toEqual({});
    // Picking the bridge at 4 GPUs moves the count to 2.
    const back = reduce(four, { type: 'setLink', level: 'intra', linkId: 'rtx-3090-nvlink-bridge' });
    expect(back.hw).toMatchObject({ devicesPerNode: 2, intraLinkId: 'rtx-3090-nvlink-bridge' });
    // An unrestricted link keeps the count.
    expect(reduce(four, { type: 'setLink', level: 'intra', linkId: 'pcie-4-0-x16-host-bounce' }).hw.devicesPerNode).toBe(4);
  });

  it('CMP 170HX: the local-link choices follow the PCIe unlocks, and removing an unlock drops a link it no longer runs', () => {
    const ids = (s: AppState) => localLinkChoices(s.hw).map((l) => linkPresetId(l.link));
    const stock = apply(DEFAULT_STATE, { type: 'setDevice', deviceId: 'cmp-170hx' }, { type: 'setCounts', devicesPerNode: 2 });
    expect(ids(stock)).toEqual(['pcie-1-0-x4-host-bounce']);
    expect(resolveHardware(stock.hw).topology.intraNode?.name).toBe('PCIe 1.0 x4 (host bounce)');
    // A link the stock interface cannot run is refused.
    expect(reduce(stock, { type: 'setLink', level: 'intra', linkId: 'pcie-2-0-x16-host-bounce' }).hw.intraLinkId).toBeUndefined();
    const cap = reduce(stock, { type: 'setUnlock', axisId: 'pcie-capacitor-mod', optionId: 'x16' });
    expect(ids(cap)).toEqual(['pcie-1-0-x16-host-bounce', 'pcie-1-0-x4-host-bounce']);
    const drv = reduce(stock, { type: 'setUnlock', axisId: 'compute-path', optionId: 'cmpunlocker' });
    expect(ids(drv)).toEqual(['pcie-2-0-x4-host-bounce', 'pcie-1-0-x4-host-bounce']);
    const both = reduce(cap, { type: 'setUnlock', axisId: 'compute-path', optionId: 'cmpunlocker' });
    expect(ids(both)).toEqual([
      'pcie-2-0-x16-host-bounce',
      'pcie-1-0-x16-host-bounce',
      'pcie-2-0-x4-host-bounce',
      'pcie-1-0-x4-host-bounce',
    ]);
    // With both unlocks the default is the fastest link; picking Gen2 x16 explicitly keeps it with its edits.
    expect(resolveHardware(both.hw).topology.intraNode?.name).toBe('PCIe 2.0 x16 (host bounce)');
    const picked = apply(
      both,
      { type: 'setLink', level: 'intra', linkId: 'pcie-2-0-x16-host-bounce' },
      { type: 'setEdit', target: 'intra', path: 'latencySec', value: 2e-6 },
    );
    expect(picked.hw).toMatchObject({ intraLinkId: 'pcie-2-0-x16-host-bounce', intraEdits: { latencySec: 2e-6 } });
    // Removing the driver patch (Gen2) drops the stale Gen2 link and its edits; the Gen1 x16 default applies.
    const noDrv = reduce(picked, { type: 'setUnlock', axisId: 'compute-path', optionId: undefined });
    expect(noDrv.hw.intraLinkId).toBeUndefined();
    expect(noDrv.hw.intraEdits).toEqual({});
    expect(resolveHardware(noDrv.hw).topology.intraNode?.name).toBe('PCIe 1.0 x16 (host bounce)');
    // A Gen1 x16 pick survives removing the driver patch but not removing the capacitor mod.
    const gen1x16 = reduce(both, { type: 'setLink', level: 'intra', linkId: 'pcie-1-0-x16-host-bounce' });
    expect(reduce(gen1x16, { type: 'setUnlock', axisId: 'compute-path', optionId: undefined }).hw.intraLinkId).toBe('pcie-1-0-x16-host-bounce');
    expect(reduce(gen1x16, { type: 'setUnlock', axisId: 'pcie-capacitor-mod', optionId: undefined }).hw.intraLinkId).toBeUndefined();
  });

  it('CMP 170HX: a stale link in the state (set without the reducer) is never simulated', () => {
    const stale: AppState = {
      ...DEFAULT_STATE,
      hw: { ...DEFAULT_STATE.hw, deviceId: 'cmp-170hx', devicesPerNode: 2, intraLinkId: 'pcie-2-0-x16-host-bounce', intraEdits: { latencySec: 2e-6 } },
    };
    const r = resolveHardware(stale.hw);
    expect(r.topology.intraNode?.name).toBe('PCIe 1.0 x4 (host bounce)');
    expect(r.topology.intraNode?.latencySec).toBe(1e-6); // the stale link's edits are not applied either
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

  it('opens on the default model at 8k / 8k', () => {
    expect(DEFAULT_STATE.model).toEqual({ kind: 'hf', id: 'Qwen/Qwen3.8-27B' });
    expect(DEFAULT_STATE.workload).toEqual({ batch: 1, promptLen: 8192, decodeContextLen: 8192 });
  });

  it('setHardware replaces the whole hardware choice', () => {
    const hw = { ...DEFAULT_STATE.hw, deviceId: 'rtx-3090', devicesPerNode: 2, intraLinkId: 'x' };
    expect(reduce(DEFAULT_STATE, { type: 'setHardware', hw }).hw).toEqual(hw);
  });
});
