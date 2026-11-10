/** Multi-NIC nodes (docs/simulation-model.md §6 "Multi-NIC nodes"): B_dev = min(N_nic · B_nic / dpn, C). */

import { describe, expect, it } from 'vitest';
import { collectiveTime } from '../engine/collectives';
import { collectivePath, groupSpan } from '../engine/placement';
import {
  interNodeBandwidthPerDeviceBytesPerSec,
  interNodeLinkPerDevice,
  nicCountErrors,
  nicsPerNodeOf,
} from './nics';
import { IB_NDR_400, NVLINK_4 } from './presets/links';
import { DGX_SPARK_4, nodeTopology } from './presets/topologies';
import type { TopologySpec } from './types';

function expectClose(actual: number, expected: number, rel = 1e-12): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** 2 nodes × 8 GPUs, NVLink4 inside, NDR400 (B_nic 50e9 B/s, α 1 µs) between nodes, `nics` NICs per node. */
function hgx2(nics?: number): TopologySpec {
  return nodeTopology({
    devicesPerNode: 8,
    nodes: 2,
    intraNode: NVLINK_4,
    interNode: IB_NDR_400,
    ...(nics !== undefined ? { nicsPerNode: nics } : {}),
  });
}

describe('nicsPerNodeOf', () => {
  it('defaults to one NIC per device', () => {
    expect(nicsPerNodeOf(hgx2())).toBe(8);
    expect(nicsPerNodeOf(hgx2(4))).toBe(4);
  });
});

describe('interNodeBandwidthPerDeviceBytesPerSec and interNodeLinkPerDevice (worked numbers)', () => {
  it('4 NICs per 8 GPUs: 4 · 50e9 / 8 = 25e9 B/s per device', () => {
    const topo = hgx2(4);
    expect(interNodeBandwidthPerDeviceBytesPerSec(topo)).toBe(25e9);
    const link = interNodeLinkPerDevice(topo);
    expect(link).toEqual({ ...IB_NDR_400, bandwidthPerDirectionBytesPerSec: 25e9 });
  });

  it('16 NICs per 8 GPUs: 100e9 B/s per device', () => {
    expect(interNodeBandwidthPerDeviceBytesPerSec(hgx2(16))).toBe(100e9);
  });

  it('absent, or equal to devicesPerNode, returns the topology link itself (same object)', () => {
    expect(interNodeLinkPerDevice(hgx2())).toBe(IB_NDR_400);
    expect(interNodeLinkPerDevice({ ...hgx2(), nicsPerNode: 8 })).toBe(IB_NDR_400);
    expect(interNodeBandwidthPerDeviceBytesPerSec(hgx2())).toBe(50e9);
  });

  it('nodeTopology stores nicsPerNode = devicesPerNode as absent, keeping the id', () => {
    expect(hgx2(8)).toEqual(hgx2());
    expect(hgx2(4).id).toBe(`${hgx2().id}-nics4`);
    expect(hgx2(4).name).toContain('(4 NICs)');
    expect(hgx2(4).nicsPerNode).toBe(4);
  });

  it('no inter-node link: undefined; B_dev throws', () => {
    const single = nodeTopology({ devicesPerNode: 8, nodes: 1, intraNode: NVLINK_4 });
    expect(interNodeLinkPerDevice(single)).toBeUndefined();
    expect(() => interNodeBandwidthPerDeviceBytesPerSec(single)).toThrow(/no interNode/);
  });

  it('DGX Spark (all-ports cap 25e9) with 2 NICs: min(2 · 25e9, 25e9) = 25e9, the same link', () => {
    const topo: TopologySpec = { ...DGX_SPARK_4, nicsPerNode: 2 };
    expect(interNodeBandwidthPerDeviceBytesPerSec(topo)).toBe(25e9);
    expect(interNodeLinkPerDevice(topo)).toBe(DGX_SPARK_4.interNode);
  });
});

describe('multi-NIC hierarchical all-reduce, inter phase (2 × 8, S/n1 = 1e6 bytes, n2 = 2)', () => {
  // Inter phase: ring all-reduce of 1e6 B over 2 nodes: 2·α + 2·(1/2)·1e6/B_dev.
  const S = 8e6;
  const interPhaseSec = (topo: TopologySpec): { latencySec: number; bandwidthSec: number } => {
    const path = collectivePath(groupSpan(16, topo), topo);
    const all = collectiveTime('all_reduce', S, 16, path, 'ring');
    const intraOnly = collectiveTime('all_reduce', S, 16, { ...hgxIntraPath(path) }, 'ring');
    return {
      latencySec: all.latencySec - intraOnly.latencySec,
      bandwidthSec: all.bandwidthSec - intraOnly.bandwidthSec,
    };
  };
  /** The same hierarchical path with a free inter-node level, to isolate the inter phase. */
  const hgxIntraPath = (path: ReturnType<typeof collectivePath>) => {
    if (path.kind !== 'hierarchical') throw new Error('expected hierarchical');
    return {
      ...path,
      inter: { ...path.inter, latencySec: 0, bandwidthPerDirectionBytesPerSec: Infinity },
    };
  };

  it('4 NICs: B_dev 25e9 → 2 µs + 40 µs', () => {
    const t = interPhaseSec(hgx2(4));
    expectClose(t.latencySec, 2e-6, 1e-9);
    expectClose(t.bandwidthSec, 40e-6, 1e-9);
  });

  it('16 NICs: B_dev 100e9 → 2 µs + 10 µs', () => {
    const t = interPhaseSec(hgx2(16));
    expectClose(t.latencySec, 2e-6, 1e-9);
    expectClose(t.bandwidthSec, 10e-6, 1e-9);
  });

  it('8 NICs on 8 GPUs: identical to absent', () => {
    const a = hgx2();
    const b: TopologySpec = { ...a, nicsPerNode: 8 };
    const path = (t: TopologySpec) => collectivePath(groupSpan(16, t), t);
    expect(path(b)).toEqual(path(a));
    expect(collectiveTime('all_reduce', S, 16, path(b), 'best')).toEqual(
      collectiveTime('all_reduce', S, 16, path(a), 'best'),
    );
  });
});

describe('nicCountErrors', () => {
  it('accepts divisors and multiples of devicesPerNode', () => {
    for (const nics of [1, 2, 4, 8, 16, 32]) expect(nicCountErrors(nics, 8)).toEqual([]);
    expect(nicCountErrors(2, 1)).toEqual([]);
  });

  it('rejects non-integers, < 1 and rail-asymmetric counts', () => {
    expect(nicCountErrors(0, 8)).toHaveLength(1);
    expect(nicCountErrors(2.5, 8)).toHaveLength(1);
    expect(nicCountErrors(Number.NaN, 8)).toHaveLength(1);
    expect(nicCountErrors(3, 8)[0]).toMatch(/divide devicesPerNode \(8\) or be a multiple/);
    expect(nicCountErrors(12, 8)).toHaveLength(1);
  });
});
