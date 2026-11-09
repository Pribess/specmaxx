import { describe, expect, it } from 'vitest';
import { SYNTH_NVLINK } from '../__fixtures__/synthetic';
import { collectiveTime } from './collectives';

describe('G3 hierarchical all-to-all retains both concurrent latency floors', () => {
  it('host-bounce intra 13 µs vs inter 1 µs: α = 13 µs (+12 µs), even with oversubscription', () => {
    const intra = {
      ...SYNTH_NVLINK,
      p2p: 'host_bounce' as const,
      hostBounce: { hostMemBandwidthBytesPerSec: 96e9, extraLatencySec: 12e-6 },
    };
    for (const interFabric of [undefined, { kind: 'switched' as const, oversubscription: 2 }]) {
      const t = collectiveTime(
        'all_to_all',
        8000,
        8,
        {
          kind: 'hierarchical',
          devicesPerNode: 4,
          nodes: 2,
          intra,
          inter: SYNTH_NVLINK,
          interFabric,
        },
        'best',
      );
      expect(t.latencySec).toBeCloseTo(13e-6, 15);
      // Intra: 3/8·8000/(96e9/(2·4)) = 250 ns; inter ≤ 80 ns.
      expect(t.bandwidthSec).toBeCloseTo(250e-9, 15);
    }
  });
});

describe('G4 flat all-to-all per-hop latency', () => {
  it.each([
    ['ring', 2],
    ['chain', 3],
  ] as const)(
    '%s4: farthest destination is %i hops at 1.5 µs, already delivered by fabric shapes',
    (kind, hops) => {
      const link = { ...SYNTH_NVLINK, latencySec: 1.5e-6 };
      const t = collectiveTime(
        'all_to_all',
        0,
        4,
        { kind: 'flat', level: 'inter', link, fabric: { kind } },
        'best',
      );
      expect(t.latencySec).toBeCloseTo(hops * 1.5e-6, 15);
    },
  );
  it('host-bounce4: 3 serialized peer rounds × 13 µs = 39 µs (+26 µs); n2 remains 13 µs', () => {
    const link = {
      ...SYNTH_NVLINK,
      p2p: 'host_bounce' as const,
      hostBounce: { hostMemBandwidthBytesPerSec: 96e9, extraLatencySec: 12e-6 },
    };
    const path = { kind: 'flat' as const, level: 'intra' as const, link };
    expect(collectiveTime('all_to_all', 0, 4, path, 'best').latencySec).toBeCloseTo(39e-6, 15);
    expect(collectiveTime('all_to_all', 0, 2, path, 'best').latencySec).toBeCloseTo(13e-6, 15);
  });
});

describe('G8 fabric dispatch preserves send/recv host sharing', () => {
  it('2 active ranks, 96 GB/s host, o2 switch: 96/(2·2)/2 = 12 GB/s (β for 12000 B = 1 µs)', () => {
    const link = {
      ...SYNTH_NVLINK,
      p2p: 'host_bounce' as const,
      hostBounce: { hostMemBandwidthBytesPerSec: 96e9, extraLatencySec: 12e-6 },
    };
    const t = collectiveTime(
      'send_recv',
      12000,
      2,
      {
        kind: 'flat',
        level: 'inter',
        link,
        fabric: { kind: 'switched', oversubscription: 2 },
        sendActiveDevices: 2,
      },
      'best',
    );
    expect(t.bandwidthSec).toBeCloseTo(1e-6, 15);
    expect(t.latencySec).toBeCloseTo(13e-6, 15);
  });
});
