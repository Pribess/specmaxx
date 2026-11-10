/**
 * Placement of the multi-NIC share and the in-network reduction fields (docs/simulation-model.md §6 "Multi-NIC
 * nodes", "In-network reduction (NVLS, SHARP)").
 */

import { describe, expect, it } from 'vitest';
import { SYNTH_IB } from '../__fixtures__/synthetic';
import { IB_NDR_400, NVLINK_3, NVLINK_4, NVLINK_5_NVL4, PCIE_5_X16 } from '../hw/presets/links';
import { HGX_H100_16_NDR, HGX_H100_8, nodeTopology, socCluster } from '../hw/presets/topologies';
import type { LinkSpec, TopologySpec } from '../hw/types';
import { collectivePath, expertTpCollectivePath, groupSpan, ppSendPath } from './placement';
import type { HostStaging } from './staging';

const SHARP_IB: LinkSpec = { ...SYNTH_IB, inNetworkReduction: 'sharp' };
const STAGING: HostStaging = {
  pcie: { bandwidthBytesPerSec: 50e9, latencySec: 1e-6 },
  socketCopy: false,
  hostMemBandwidthBytesPerSec: 200e9,
  activeDevicesPerNode: 8,
  stepBytes: 1 << 20,
} as HostStaging;

const pathOf = (size: number, topo: TopologySpec, inNetwork: 'off' | 'on', staging?: HostStaging) =>
  collectivePath(groupSpan(size, topo), topo, staging, inNetwork);

describe('in-network reduction fields', () => {
  it("HGX H100 TP8: flat intra gets 'nvls' only with the switch on", () => {
    expect(pathOf(8, HGX_H100_8, 'on')).toEqual({
      kind: 'flat',
      level: 'intra',
      link: NVLINK_4,
      inNetworkReduction: 'nvls',
    });
    expect(pathOf(8, HGX_H100_8, 'off')).toEqual({ kind: 'flat', level: 'intra', link: NVLINK_4 });
    expect(pathOf(8, HGX_H100_8, 'off')).toEqual(
      collectivePath(groupSpan(8, HGX_H100_8), HGX_H100_8),
    );
  });

  it("HGX H100 TP16 over NDR: hierarchical gets intra 'nvls' and inter 'sharp'", () => {
    expect(pathOf(16, HGX_H100_16_NDR, 'on')).toEqual({
      kind: 'hierarchical',
      devicesPerNode: 8,
      nodes: 2,
      intra: NVLINK_4,
      inter: IB_NDR_400,
      intraInNetworkReduction: 'nvls',
      interInNetworkReduction: 'sharp',
    });
    expect(pathOf(16, HGX_H100_16_NDR, 'off')).not.toHaveProperty('intraInNetworkReduction');
  });

  it('a staged inter-node level gets none; the intra level keeps NVLS', () => {
    const path = pathOf(16, HGX_H100_16_NDR, 'on', STAGING);
    expect(path).toMatchObject({ intraInNetworkReduction: 'nvls', interStaging: STAGING });
    expect(path).not.toHaveProperty('interInNetworkReduction');
    const flat = nodeTopology({ devicesPerNode: 1, nodes: 4, interNode: SHARP_IB });
    expect(pathOf(4, flat, 'on')).toMatchObject({ inNetworkReduction: 'sharp' });
    expect(pathOf(4, flat, 'on', STAGING)).not.toHaveProperty('inNetworkReduction');
  });

  it('links without a switch get none: NVL4 bridge, NVLink3, PCIe', () => {
    for (const intraNode of [NVLINK_5_NVL4, NVLINK_3, PCIE_5_X16]) {
      const topo = nodeTopology({ devicesPerNode: 4, nodes: 1, intraNode });
      expect(pathOf(4, topo, 'on')).toEqual({ kind: 'flat', level: 'intra', link: intraNode });
    }
  });

  it('switchless (mesh, ring, chain) and oversubscribed flat paths get none', () => {
    for (const fabric of [
      { kind: 'full_mesh' },
      { kind: 'ring' },
      { kind: 'chain' },
      { kind: 'switched', oversubscription: 2 },
    ] as const) {
      const topo = socCluster({ id: 'soc', name: 'SoC' }, 4, SHARP_IB, fabric);
      const path = pathOf(4, topo, 'on');
      expect(path).toHaveProperty('fabric');
      expect(path).not.toHaveProperty('inNetworkReduction');
    }
  });

  it('expert-TP paths take the switch too', () => {
    expect(expertTpCollectivePath({ tp: 8, ep: 2 }, HGX_H100_8, undefined, 'on')).toMatchObject({
      inNetworkReduction: 'nvls',
    });
    expect(expertTpCollectivePath({ tp: 8, ep: 2 }, HGX_H100_8)).not.toHaveProperty(
      'inNetworkReduction',
    );
  });
});

describe('the NIC share on every inter-node link of a path', () => {
  const topo = nodeTopology({
    devicesPerNode: 8,
    nodes: 2,
    intraNode: NVLINK_4,
    interNode: IB_NDR_400,
    nicsPerNode: 4,
  });
  const shared = { ...IB_NDR_400, bandwidthPerDirectionBytesPerSec: 25e9 };

  it('hierarchical inter', () => expect(pathOf(16, topo, 'off')).toMatchObject({ inter: shared }));

  it('flat inter (expert-TP one member per node)', () =>
    expect(expertTpCollectivePath({ tp: 16, ep: 8 }, topo)).toMatchObject({
      kind: 'flat',
      level: 'inter',
      link: shared,
    }));

  it('PP send across nodes', () =>
    expect(ppSendPath({ tp: 8, pp: 2, dp: 1, ep: 1 }, topo, 0)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: shared,
    }));

  it('absent nicsPerNode: the topology link object itself', () => {
    const path = pathOf(16, HGX_H100_16_NDR, 'off');
    if (path.kind !== 'hierarchical') throw new Error('expected hierarchical');
    expect(path.inter).toBe(HGX_H100_16_NDR.interNode);
  });
});
