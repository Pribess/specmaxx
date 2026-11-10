import { describe, expect, it } from 'vitest';
import { TINY_CSA, TINY_MLA_SPARSE } from '../../sim/models/reference';
import { offloadChipText } from '../components/ResultHero';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import { buildSimInput } from './buildInput';
import {
  normalizeKvCache,
  normalizeOffload,
  offloadsAnything,
  offloadsToSsd,
  type OffloadChoice,
} from './deployment';
import { kvCacheConfig } from './deploymentInput';
import { offloadPlacements } from './summary';
import { decodeState, encodeState } from './url';

const DENSE = BUNDLED_DEFAULT_MODEL.model;
const withOffload = (offload: OffloadChoice, state: AppState = DEFAULT_STATE) =>
  reduce(state, { type: 'setOffload', offload });

describe('KV-cache offload: state', () => {
  it('drops a choice that keeps everything on the device, and the defaults', () => {
    expect(normalizeKvCache({ tier: 'host_ram' })).toBeUndefined();
    expect(normalizeKvCache({ tier: 'nvme', deviceFraction: 1 })).toBeUndefined();
    expect(
      normalizeKvCache({ tier: 'nvme', deviceFraction: 1, prefixDeviceFraction: 1 }),
    ).toBeUndefined();
    expect(
      normalizeKvCache({
        tier: 'host_ram',
        deviceFraction: 0.25,
        ramFraction: 0.5,
        prefixDeviceFraction: 0.25,
      }),
    ).toEqual({ tier: 'host_ram', deviceFraction: 0.25 });
    // Prefix-only: f_dev = 1 is the default and not stored.
    expect(
      normalizeKvCache({ tier: 'nvme', deviceFraction: 1, prefixDeviceFraction: 0.5 }),
    ).toEqual({ tier: 'nvme', prefixDeviceFraction: 0.5 });
    expect(normalizeOffload({ kvCache: { tier: 'host_ram', deviceFraction: 1 } })).toEqual({});
  });

  it('counts as an offload (host controls, queue depth) and as SSD use on the SSD or a split', () => {
    const kv = { tier: 'host_ram', deviceFraction: 0.5 } as const;
    expect(offloadsAnything({ kvCache: kv })).toBe(true);
    expect(offloadsToSsd({ kvCache: kv })).toBe(false);
    expect(offloadsToSsd({ kvCache: { ...kv, tier: 'split' } })).toBe(true);
    expect(normalizeOffload({ kvCache: kv, queueDepth: 8 })).toEqual({
      kvCache: kv,
      queueDepth: 8,
    });
  });
});

describe('KV-cache offload: engine input', () => {
  it('maps host RAM, SSD and a split to hostRamFraction (absent, 0, clamped share)', () => {
    expect(kvCacheConfig({ tier: 'host_ram', deviceFraction: 0.25 }, DENSE, 0)).toEqual({
      deviceFraction: 0.25,
    });
    expect(kvCacheConfig({ tier: 'nvme', deviceFraction: 0.25 }, DENSE, 0)).toEqual({
      deviceFraction: 0.25,
      hostRamFraction: 0,
    });
    // Split default: half of the off-device share.
    expect(kvCacheConfig({ tier: 'split', deviceFraction: 0.5 }, DENSE, 0)).toEqual({
      deviceFraction: 0.5,
      hostRamFraction: 0.25,
    });
    // f_ram ≤ 1 − f_dev.
    expect(
      kvCacheConfig({ tier: 'split', deviceFraction: 0.5, ramFraction: 0.9 }, DENSE, 0),
    ).toEqual({ deviceFraction: 0.5, hostRamFraction: 0.5 });
  });

  it('sends the custom hit ratio only for top-k models and the prefix share only with prefix hits', () => {
    const kv = { tier: 'host_ram', deviceFraction: 0.5, hitRatio: 0.8 } as const;
    expect(kvCacheConfig(kv, DENSE, 0)).toEqual({ deviceFraction: 0.5 });
    expect(kvCacheConfig(kv, TINY_MLA_SPARSE, 0)).toEqual({
      deviceFraction: 0.5,
      hitRatio: { kind: 'custom', h: 0.8 },
    });
    const prefixOnly = { tier: 'nvme', prefixDeviceFraction: 0.5 } as const;
    expect(kvCacheConfig(prefixOnly, DENSE, 0)).toBeUndefined();
    expect(kvCacheConfig(prefixOnly, DENSE, 0.5)).toEqual({
      deviceFraction: 1,
      hostRamFraction: 0,
      prefixDeviceFraction: 0.5,
    });
  });

  it('buildSimInput emits kvCache and the host, and nothing with KV offload off', () => {
    const off = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
    if (off.kind !== 'ok') throw new Error('not ok');
    expect('deployment' in off.input).toBe(false);
    const on = buildSimInput(
      withOffload({ kvCache: { tier: 'host_ram', deviceFraction: 0.25 } }),
      BUNDLED_DEFAULT_MODEL,
    );
    if (on.kind !== 'ok') throw new Error('not ok');
    expect(on.input.deployment).toEqual({ offload: { kvCache: { deviceFraction: 0.25 } } });
    expect(on.input.hardware.host).toBeDefined();
  });

  it('warns and places nothing for a model without a growing KV cache', () => {
    const built = buildSimInput(
      withOffload({ kvCache: { tier: 'host_ram', deviceFraction: 0.25 } }),
      {
        ...BUNDLED_DEFAULT_MODEL,
        model: TINY_CSA,
      },
    );
    expect(
      built.issues.some(
        (i) => i.field === 'offload' && i.level === 'warning' && /KV cache/.test(i.message),
      ),
    ).toBe(true);
    expect(built.kind === 'ok' && 'deployment' in built.input).toBe(false);
  });
});

describe('KV-cache offload: URL', () => {
  it('round-trips okt/okd/okr/okh/okp in percent, after oet', () => {
    const q = 'oet=ssd&okt=split&okd=25&okr=50&okh=80&okp=10';
    const { state, dropped } = decodeState(q);
    expect(dropped).toEqual([]);
    expect(state.deployment.offload?.kvCache).toEqual({
      tier: 'split',
      deviceFraction: 0.25,
      ramFraction: 0.5,
      hitRatio: 0.8,
      prefixDeviceFraction: 0.1,
    });
    expect(encodeState(state)).toBe(q);
    expect(encodeState(decodeState('okt=ram&okd=25').state)).toBe('okt=ram&okd=25');
    // The dependent keys turn on the queue depth and the SSD read rate.
    expect(decodeState('okt=ssd&okd=0&oqd=4&osr=sustained').dropped).toEqual([]);
  });

  it('drops dependent keys without their parent, and out-of-range values', () => {
    expect(decodeState('okd=25').dropped).toEqual(['okd=25']);
    expect(decodeState('okt=ram&okd=25&okr=10').dropped).toEqual(['okr=10']);
    expect(decodeState('okt=ram&okd=125').dropped).toEqual(['okd=125']);
    expect(decodeState('okt=disk').dropped).toEqual(['okt=disk']);
    // All on the device: nothing is stored, nothing is written.
    expect(encodeState(decodeState('okt=ram&okd=100').state)).toBe('');
    // f_ram is clamped to 1 − f_dev.
    expect(decodeState('okt=split&okd=50&okr=80').state.deployment.offload?.kvCache).toEqual({
      tier: 'split',
      deviceFraction: 0.5,
      ramFraction: 0.5,
    });
  });

  it('leaves the URL of a state without KV offload unchanged', () => {
    expect(encodeState(DEFAULT_STATE)).toBe('');
    expect(encodeState(decodeState('ox=ram&oxf=0.25').state)).toBe('ox=ram&oxf=0.25');
  });
});

describe('KV-cache offload: result chip', () => {
  const chip = (state: AppState, model = BUNDLED_DEFAULT_MODEL) => {
    const built = buildSimInput(state, model);
    if (built.kind !== 'ok') throw new Error('not ok');
    return offloadPlacements(built.input).map(offloadChipText);
  };

  it('names the device and tier shares', () => {
    expect(chip(withOffload({ kvCache: { tier: 'host_ram', deviceFraction: 0.25 } }))).toEqual([
      'KV cache: 25% on device, 75% host RAM',
    ]);
    expect(
      chip(withOffload({ kvCache: { tier: 'split', deviceFraction: 0.5, ramFraction: 0.25 } })),
    ).toEqual(['KV cache: 50% on device, 25% host RAM, 25% SSD']);
    const prefix = reduce(DEFAULT_STATE, {
      type: 'setWorkload',
      workload: { prefixCacheHitRatio: 0.5 },
    });
    expect(
      chip(withOffload({ kvCache: { tier: 'nvme', prefixDeviceFraction: 0.5 } }, prefix)),
    ).toEqual(['KV cache: cached prefix 50% on device, rest SSD']);
  });
});
