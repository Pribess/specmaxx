import { describe, expect, it } from 'vitest';
import { SYNTH_PCIE_HOSTBOUNCE_LINK } from '../__fixtures__/synthetic';
import type { LinkSpec } from '../hw/types';
import {
  allReduceSec,
  allToAllSec,
  collectiveTime,
  effectiveLink,
  ringAllGatherSec,
  ringAllReduceSec,
  sendRecvSec,
  treeAllReduceSec,
} from './collectives';
import type { EffectiveLink } from './collectives';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

function link(bandwidthPerDirectionBytesPerSec: number, latencySec: number): LinkSpec {
  return {
    kind: 'other',
    name: 'test',
    bandwidthPerDirectionBytesPerSec,
    latencySec,
    p2p: 'direct',
    sources: [
      {
        url: 'synthetic://test',
        title: 't',
        fields: ['bandwidthPerDirectionBytesPerSec', 'latencySec'],
      },
    ],
    estimated: [],
  };
}

// n 8, B 1e11 B/s, α 1e-6 s
const L: EffectiveLink = { bandwidthBytesPerSec: 1e11, latencySec: 1e-6 };

describe('flat alpha-beta collectives (T7)', () => {
  it('S 8e6: ring 154e-6 beats tree 166e-6', () => {
    // ring: 2·7·1e-6 + 2·7/8·8e6/1e11 = 14e-6 + 140e-6
    expectClose(ringAllReduceSec(8e6, 8, L).totalSec, 154e-6);
    // tree: 2·3·1e-6 + 2·8e6/1e11 = 6e-6 + 160e-6
    expectClose(treeAllReduceSec(8e6, 8, L).totalSec, 166e-6);
    const best = allReduceSec(8e6, 8, L, 'best');
    expect(best.algorithm).toBe('ring');
    expectClose(best.latencySec, 14e-6);
    expectClose(best.bandwidthSec, 140e-6);
  });

  it('S 8e3: tree 6.16e-6 beats ring 14.14e-6', () => {
    expectClose(ringAllReduceSec(8e3, 8, L).totalSec, 14.14e-6); // 14e-6 + 0.14e-6
    expectClose(treeAllReduceSec(8e3, 8, L).totalSec, 6.16e-6); // 6e-6 + 0.16e-6
    expect(allReduceSec(8e3, 8, L, 'best').algorithm).toBe('tree');
    expect(allReduceSec(8e3, 8, L, 'ring').algorithm).toBe('ring');
  });

  it('all-gather 77e-6, all-to-all 71e-6, send/recv 81e-6', () => {
    expectClose(ringAllGatherSec(8e6, 8, L).totalSec, 77e-6); // 7·1e-6 + 7/8·8e6/1e11 = 7e-6 + 70e-6
    expectClose(allToAllSec(8e6, 8, L).totalSec, 71e-6); // 1e-6 + 70e-6
    expectClose(sendRecvSec(8e6, L).totalSec, 81e-6); // 1e-6 + 8e6/1e11
  });

  it('a group of one costs nothing', () => {
    expect(ringAllReduceSec(8e6, 1, L).totalSec).toBe(0);
    expect(collectiveTime('all_reduce', 8e6, 1, { kind: 'none' }, 'best').totalSec).toBe(0);
  });
});

describe('hierarchical collectives (T7)', () => {
  it('all-reduce n 16 = 2 nodes × 8: 38.111e-6 + 50e-6 + 38.111e-6 = 126.22e-6', () => {
    const path = {
      kind: 'hierarchical',
      devicesPerNode: 8,
      nodes: 2,
      intra: link(450e9, 1e-6),
      inter: link(50e9, 5e-6),
    } as const;
    const t = collectiveTime('all_reduce', 16e6, 16, path, 'best');
    // RS_intra(16e6, 8) = 7·1e-6 + 7/8·16e6/450e9 = 7e-6 + 31.111e-6 = 38.111e-6 (AG_intra the same)
    const intraPhase = 7e-6 + ((7 / 8) * 16e6) / 450e9;
    // AR_inter(2e6, 2): ring 2·1·5e-6 + 2·1/2·2e6/50e9 = 10e-6 + 40e-6 = 50e-6 (tree 10e-6 + 80e-6 = 90e-6)
    const interPhase = 50e-6;
    expectClose(t.totalSec, 2 * intraPhase + interPhase);
    expectClose(t.totalSec, 126.2222222e-6, 1e-8);
    expect(t.algorithm).toBe('hierarchical');
    expectClose(t.latencySec, 7e-6 + 10e-6 + 7e-6);
  });
});

describe('host-bounce links (T7)', () => {
  it('ring all-reduce n 2 S 1e8: B_eff = min(64e9, 80e9/4) = 20e9, α_eff 7e-6 → 14e-6 + 5e-3', () => {
    const eff = effectiveLink(SYNTH_PCIE_HOSTBOUNCE_LINK, 2);
    expect(eff.bandwidthBytesPerSec).toBe(20e9);
    expectClose(eff.latencySec, 7e-6); // 2e-6 + 5e-6
    const path = { kind: 'flat', level: 'intra', link: SYNTH_PCIE_HOSTBOUNCE_LINK } as const;
    const t = collectiveTime('all_reduce', 1e8, 2, path, 'best');
    expect(t.algorithm).toBe('ring'); // tree: 14e-6 + 2·1e8/20e9 = 10.014e-3
    expectClose(t.latencySec, 14e-6); // 2·1·7e-6
    expectClose(t.bandwidthSec, 5e-3); // 2·(1/2)·1e8/20e9
  });

  it('send/recv S 1e8: B_eff = min(64e9, 80e9/2) = 40e9 → 7e-6 + 2.5e-3', () => {
    const path = { kind: 'flat', level: 'intra', link: SYNTH_PCIE_HOSTBOUNCE_LINK } as const;
    const t = collectiveTime('send_recv', 1e8, 2, path, 'best');
    expectClose(t.latencySec, 7e-6);
    expectClose(t.bandwidthSec, 2.5e-3);
  });
});
