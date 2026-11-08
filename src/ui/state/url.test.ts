import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { decodeState, encodeNumber, encodeState } from './url';

/** A state built from the default by reducer actions (so it is in the reducer's canonical form). */
function state(over: Omit<Partial<AppState>, 'hw'> & { hw?: Partial<AppState['hw']> } = {}): AppState {
  return { ...DEFAULT_STATE, ...over, hw: { ...DEFAULT_STATE.hw, ...over.hw } };
}

const STATES: [string, AppState][] = [
  ['default', DEFAULT_STATE],
  ['HF model', state({ model: { kind: 'hf', id: 'Qwen/Qwen3.8-27B' } })],
  ['HF model with revision', state({ model: { kind: 'hf', id: 'nvidia/Qwen3.8-27B-NVFP4', revision: 'a1b2c3d' } })],
  ['quant overrides', state({ quant: { weights: 'nvfp4', activations: 'fp8_e4m3', kvCache: 'fp8_e4m3' } })],
  ['EXL3 weights with bpw and head bpw', state({ quant: { weights: 'exl3', bpw: 2.5, headBpw: 5 } })],
  ['EXL3 weights at the default bpw', state({ quant: { weights: 'exl3' } })],
  ['unlocks + capacity', state({ hw: { deviceId: 'rtx-4090', unlocks: { 'vram-mod': '48gb', 'pcie-p2p': 'direct' }, capacityIndex: 0 } })],
  ['engine + capacity index', state({ hw: { deviceId: 'apple-m5-max-40c', engineId: 'gpu-simd', capacityIndex: 2 } })],
  ['device edits', state({ hw: { deviceEdits: { 'memory.bandwidthBytesPerSec': 2e12, 'engines.0.matmulPeaks.0.flopsPerSec': 1.5e14 } } })],
  [
    'multi-node with links and link edits',
    state({
      hw: {
        deviceId: 'h100-sxm', devicesPerNode: 8, nodes: 2, intraLinkId: 'nvlink4-nvswitch',
        interLinkId: 'infiniband-xdr800-one-nic-per-gpu', intraEdits: { latencySec: 2e-6 },
        interEdits: { bandwidthPerDirectionBytesPerSec: 5e10 },
      },
      parallel: { tp: 8, pp: 2, dp: 1, ep: 1 },
    }),
  ],
  ['parallel + workload', state({ parallel: { tp: 2, pp: 1, dp: 2, ep: 4 }, workload: { batch: 8, promptLen: 4096, decodeContextLen: 32768 } })],
  ['every option', state({ options: { overlap: 'none', elementwiseFusion: 'ideal', collectiveAlgorithm: 'ring' } })],
];

describe('URL codec', () => {
  it('encodes the default state to the empty string', () => {
    expect(encodeState(DEFAULT_STATE)).toBe('');
    expect(decodeState('')).toEqual({ state: DEFAULT_STATE, dropped: [] });
  });

  it.each(STATES)('round-trips: %s', (_name, s) => {
    const q = encodeState(s);
    const decoded = decodeState(q);
    expect(decoded.dropped).toEqual([]);
    expect(decoded.state).toEqual(s);
    expect(decodeState(`?${q}`).state).toEqual(s);
  });

  it.each([
    'm=zai-org/GLM-5.3&kv=fp8_e4m3&d=rtx-4090&u=vram-mod.48gb&n=2&tp=2&b=4',
    'm=Qwen/Qwen3.8-27B&rev=main2&w=fp8_e4m3&a=fp8_e4m3',
    'u=pcie-p2p.direct,power-clock.400w&n=2&il=pcie-5-0-x16-host-bounce&xi=latencySec:5e-7&tp=2',
    'x=memory.bandwidthBytesPerSec:2e12&b=16&p=4096&c=4096',
    'n=4&nodes=2&xl=10gbe-tcp&xe=bandwidthPerDirectionBytesPerSec:2.5e9&tp=4&dp=2&ov=none&fu=ideal&ar=tree',
  ])('keeps a canonical query unchanged: %s', (q) => {
    const { state: s, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(encodeState(s)).toBe(q);
  });

  it('drops EXL3 bpw values that are out of range or lack w=exl3', () => {
    expect(decodeState('w=exl3&bpw=2.5&hbpw=16').state.quant).toEqual({ weights: 'exl3', bpw: 2.5, headBpw: 16 });
    const d = decodeState('w=exl3&bpw=12&hbpw=0');
    expect(d.state.quant).toEqual({ weights: 'exl3' });
    expect(d.dropped).toEqual(['bpw=12', 'hbpw=0']);
    expect(decodeState('w=fp8_e4m3&bpw=3').dropped).toEqual(['bpw=3']);
    expect(encodeState(state({ quant: { weights: 'fp8_e4m3', bpw: 3 } }))).toBe('w=fp8_e4m3');
  });

  it('keeps / : , . readable', () => {
    const q = encodeState(state({ model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' }, hw: { deviceEdits: { 'memory.bandwidthBytesPerSec': 2e12 } } }));
    expect(q).toBe('m=Qwen/Qwen3.8-Flash-Next&x=memory.bandwidthBytesPerSec:2e12');
  });

  it('drops junk values and unknown keys, never throws', () => {
    const { state: s, dropped } = decodeState(
      'm=not a model&w=fp7&d=gtx-1080&tp=0&pp=-1&dp=1.5&b=abc&ov=sometimes&zzz=1&cap=9&eng=npu&n=0',
    );
    expect(s).toEqual(DEFAULT_STATE);
    expect(dropped.sort()).toEqual(
      ['m=not a model', 'w=fp7', 'd=gtx-1080', 'tp=0', 'pp=-1', 'dp=1.5', 'b=abc', 'ov=sometimes', 'zzz=1', 'cap=9', 'eng=npu', 'n=0'].sort(),
    );
  });

  it('drops unknown unlocks, edit paths and links, and link edits for a level that does not exist', () => {
    const { state: s, dropped } = decodeState(
      'u=vram-mod.48gb,nope.x&x=memory.bandwidthBytesPerSec:1e12,memory.bogus:3,engines.0.computeUnits:-1&il=warp-drive&xi=latencySec:1e-6',
    );
    // rtx-5090 has no vram-mod axis.
    expect(s.hw.unlocks).toEqual({});
    expect(s.hw.deviceEdits).toEqual({ 'memory.bandwidthBytesPerSec': 1e12 });
    expect(s.hw.intraLinkId).toBeUndefined();
    expect(s.hw.intraEdits).toEqual({});
    expect(dropped).toEqual([
      'u=vram-mod.48gb,nope.x',
      'x=memory.bogus:3,engines.0.computeUnits:-1',
      'il=warp-drive',
      'xi=latencySec:1e-6',
    ]);
  });

  it('checks the capacity index against the unlocked device', () => {
    // The 4090's 48 GB mod replaces the capacity options with one entry: index 0 only.
    expect(decodeState('d=rtx-4090&u=vram-mod.48gb&cap=1').dropped).toEqual(['cap=1']);
    expect(decodeState('d=apple-m3-ultra-80c&cap=2').state.hw.capacityIndex).toBe(2);
  });

  it('never writes an HF token', () => {
    const q = encodeState(state({ model: { kind: 'hf', id: 'meta/gated' } }));
    expect(q).not.toMatch(/token|hf_/i);
  });

  it('normalizes rev=main and default-valued options away', () => {
    expect(encodeState(decodeState('m=a/b&rev=main&ov=ideal&ar=best').state)).toBe('m=a/b');
    const s = reduce(DEFAULT_STATE, { type: 'setOptions', options: { overlap: 'none' } });
    expect(encodeState(reduce(s, { type: 'setOptions', options: { overlap: 'ideal' } }))).toBe('');
  });

  it('writes numbers in their shortest exact form', () => {
    for (const v of [2e12, 1.792e12, 1024, 0.5, 5e-7, 123456789, 1 / 3]) {
      expect(Number(encodeNumber(v))).toBe(v);
    }
    expect(encodeNumber(2e12)).toBe('2e12');
    expect(encodeNumber(1024)).toBe('1024');
  });

  it('omits the default model and hardware; another model and old links still work', () => {
    expect(encodeState(DEFAULT_STATE)).toBe('');
    expect(encodeState(state({ model: { kind: 'hf', id: 'zai-org/GLM-5.3' } }))).toBe('m=zai-org/GLM-5.3');
    expect(decodeState('m=zai-org/GLM-5.3').state.model).toEqual({ kind: 'hf', id: 'zai-org/GLM-5.3' });
    // Old link without m: the default model; `?b=2` keeps the default hardware.
    const old = decodeState('b=2');
    expect(old.state.model).toEqual(DEFAULT_STATE.model);
    expect(old.state.hw).toEqual(DEFAULT_STATE.hw);
    expect(old.state.workload.batch).toBe(2);
  });

  it('prefill length and decode context default to 8192 and are written only when different', () => {
    expect(encodeState(state({ workload: { batch: 1, promptLen: 8192, decodeContextLen: 32768 } }))).toBe('c=32768');
  });
});
