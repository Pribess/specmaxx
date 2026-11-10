/**
 * In-network all-reduce (NVLS, SHARP): the worked numbers of docs/simulation-model.md §6 "In-network reduction
 * (NVLS, SHARP)". NVLink4: B 450e9 B/s, α 0.8 µs; InfiniBand NDR400: B 50e9 B/s, α 1 µs; S = 8e6 bytes.
 */

import { describe, expect, it } from 'vitest';
import { IB_NDR_400, NVLINK_4 } from '../hw/presets/links';
import type { LinkSpec } from '../hw/types';
import {
  allReduceSec,
  collectiveTime,
  effectiveLink,
  hierarchicalAllReduceSec,
  inNetworkAllReduceSec,
  nvlsAllReduceSec,
  sharpAllReduceSec,
  type CollectivePath,
  type CollectiveTime,
} from './collectives';

const S = 8e6;
const NVL = effectiveLink(NVLINK_4, 8);
const IB = effectiveLink(IB_NDR_400, 1);

function expectTime(t: CollectiveTime, latencyUs: number, bandwidthUs: number): void {
  expect(Math.abs(t.latencySec - latencyUs * 1e-6)).toBeLessThanOrEqual(1e-9 * latencyUs * 1e-6);
  expect(Math.abs(t.bandwidthSec - bandwidthUs * 1e-6)).toBeLessThanOrEqual(
    1e-9 * bandwidthUs * 1e-6,
  );
}

describe('nvlsAllReduceSec: t = 2·α + (n+1)/n · S/B', () => {
  it('n 8 on NVLink4: 1.6 µs + (9/8)·8e6/450e9 = 20 µs', () => {
    const t = nvlsAllReduceSec(S, 8, NVL);
    expect(t.algorithm).toBe('nvls');
    expectTime(t, 1.6, 20);
  });

  it('n 2: 1.6 µs + 26.67 µs', () =>
    expectTime(nvlsAllReduceSec(S, 2, NVL), 1.6, (1.5 * S) / 450e3));

  it('n 1 costs nothing', () =>
    expect(nvlsAllReduceSec(S, 1, NVL)).toEqual({
      latencySec: 0,
      bandwidthSec: 0,
      totalSec: 0,
      algorithm: 'none',
    }));
});

describe('sharpAllReduceSec: t = 2·α + S/B', () => {
  it('n 4 on NDR400: 2 µs + 160 µs', () => {
    const t = sharpAllReduceSec(S, 4, IB);
    expect(t.algorithm).toBe('sharp');
    expectTime(t, 2, 160);
  });

  it('n 1 costs nothing', () => expect(sharpAllReduceSec(S, 1, IB).algorithm).toBe('none'));

  it('inNetworkAllReduceSec dispatches by kind', () => {
    expect(inNetworkAllReduceSec('nvls', S, 8, NVL)).toEqual(nvlsAllReduceSec(S, 8, NVL));
    expect(inNetworkAllReduceSec('sharp', S, 4, IB)).toEqual(sharpAllReduceSec(S, 4, IB));
  });
});

describe("allReduceSec: in-network is a 'best' candidate only", () => {
  it('NVLS n 8 beats ring (11.2 + 31.11 µs) and tree (4.8 + 35.56 µs)', () => {
    expectTime(allReduceSec(S, 8, NVL, 'ring'), 11.2, (14 / 8) * (S / 450e3));
    expectTime(allReduceSec(S, 8, NVL, 'tree'), 4.8, (2 * S) / 450e3);
    const best = allReduceSec(S, 8, NVL, 'best', NVL, 'nvls');
    expect(best.algorithm).toBe('nvls');
    expectTime(best, 1.6, 20);
  });

  it('NVLS n 2 loses to ring (1.6 + 17.78 µs vs 1.6 + 26.67 µs)', () => {
    const best = allReduceSec(S, 2, NVL, 'best', NVL, 'nvls');
    expect(best).toEqual(allReduceSec(S, 2, NVL, 'ring'));
  });

  it('SHARP n 4 beats ring (6 + 240 µs) and tree (4 + 320 µs)', () => {
    expectTime(allReduceSec(S, 4, IB, 'ring'), 6, 240);
    expectTime(allReduceSec(S, 4, IB, 'tree'), 4, 320);
    const best = allReduceSec(S, 4, IB, 'best', IB, 'sharp');
    expect(best.algorithm).toBe('sharp');
    expectTime(best, 2, 160);
  });

  it('explicit ring / tree ignore the capability', () => {
    expect(allReduceSec(S, 8, NVL, 'ring', NVL, 'nvls')).toEqual(allReduceSec(S, 8, NVL, 'ring'));
    expect(allReduceSec(S, 8, NVL, 'tree', NVL, 'nvls')).toEqual(allReduceSec(S, 8, NVL, 'tree'));
  });

  it('a tie keeps ring: SHARP n 2 = ring n 2 = 2·α + S/B', () => {
    const ring = allReduceSec(S, 2, IB, 'ring');
    expect(sharpAllReduceSec(S, 2, IB).totalSec).toBe(ring.totalSec);
    expect(allReduceSec(S, 2, IB, 'best', IB, 'sharp')).toEqual(ring);
  });

  it('n 1 costs nothing, whatever the capability', () =>
    expect(allReduceSec(S, 1, NVL, 'best', NVL, 'nvls').algorithm).toBe('none'));
});

describe('hierarchical all-reduce 8 × 4 (NVLink4 + NDR400), shard S/n1 = 1e6', () => {
  // Without in-network: RS_intra 5.6 + 15.56, AR_inter ring 6 + 30, AG_intra 5.6 + 15.56.
  const rsUs = (7 / 8) * (S / 450e3);
  const intraNvls = { latencyUs: 1.6, bandwidthUs: 20 };
  const interRing = { latencyUs: 6, bandwidthUs: 30 };
  const interSharp = { latencyUs: 2, bandwidthUs: 20 };
  const hier = (inNetwork?: { intra?: 'nvls'; inter?: 'sharp' }) =>
    hierarchicalAllReduceSec(S, 8, 4, NVL, IB, 'best', IB, inNetwork);

  it('none: 2·RS + ring inter', () => {
    const t = hier();
    expect(t.algorithm).toBe('hierarchical');
    expectTime(t, 5.6 * 2 + interRing.latencyUs, 2 * rsUs + interRing.bandwidthUs);
    expect(hier({})).toEqual(t);
  });

  it('intra NVLS only: NVLS replaces RS + AG', () =>
    expectTime(
      hier({ intra: 'nvls' }),
      intraNvls.latencyUs + interRing.latencyUs,
      intraNvls.bandwidthUs + interRing.bandwidthUs,
    ));

  it('inter SHARP only: SHARP replaces the inter ring', () =>
    expectTime(
      hier({ inter: 'sharp' }),
      5.6 * 2 + interSharp.latencyUs,
      2 * rsUs + interSharp.bandwidthUs,
    ));

  it('both', () =>
    expectTime(
      hier({ intra: 'nvls', inter: 'sharp' }),
      intraNvls.latencyUs + interSharp.latencyUs,
      intraNvls.bandwidthUs + interSharp.bandwidthUs,
    ));

  it('explicit ring ignores both', () =>
    expect(
      hierarchicalAllReduceSec(S, 8, 4, NVL, IB, 'ring', IB, { intra: 'nvls', inter: 'sharp' }),
    ).toEqual(hierarchicalAllReduceSec(S, 8, 4, NVL, IB, 'ring', IB)));
});

describe('collectiveTime reads the path fields', () => {
  const flat = (link: LinkSpec, inNetwork?: 'nvls' | 'sharp'): CollectivePath => ({
    kind: 'flat',
    level: 'intra',
    link,
    ...(inNetwork !== undefined ? { inNetworkReduction: inNetwork } : {}),
  });

  it('a flat path with inNetworkReduction uses NVLS for all-reduce only', () => {
    expect(collectiveTime('all_reduce', S, 8, flat(NVLINK_4, 'nvls'), 'best').algorithm).toBe(
      'nvls',
    );
    expect(collectiveTime('all_gather', S, 8, flat(NVLINK_4, 'nvls'), 'best')).toEqual(
      collectiveTime('all_gather', S, 8, flat(NVLINK_4), 'best'),
    );
    // Without the field: tree (4.8 + 35.56 µs) beats ring (11.2 + 31.11 µs).
    expect(collectiveTime('all_reduce', S, 8, flat(NVLINK_4), 'best').algorithm).toBe('tree');
  });

  it('a hierarchical path with both fields matches hierarchicalAllReduceSec', () => {
    const path: CollectivePath = {
      kind: 'hierarchical',
      devicesPerNode: 8,
      nodes: 4,
      intra: NVLINK_4,
      inter: IB_NDR_400,
      intraInNetworkReduction: 'nvls',
      interInNetworkReduction: 'sharp',
    };
    expect(collectiveTime('all_reduce', S, 32, path, 'best')).toEqual(
      hierarchicalAllReduceSec(S, 8, 4, NVL, effectiveLink(IB_NDR_400, 8), 'best', undefined, {
        intra: 'nvls',
        inter: 'sharp',
      }),
    );
  });
});
