import { describe, expect, it } from 'vitest';
import { effectiveLink, ringAllReduceSec } from '../../engine/collectives';
import { applyLinkModifiers } from '../modifiers';
import { validateLink, validateLinkModifiers, validateTopology } from '../validate';
import { allUnlockSelections, resolveUnlocks } from '../unlocks';
import {
  ETHERNET_5G_TCP, ETHERNET_10G_TCP, HOST_BOUNCE_PCIE_LINKS, IB_NDR_400, IB_XDR_800,
  LINK_PRESETS, NVLINK_4, NVLINK_5, PCIE_4_X8, PCIE_4_X16, PCIE_5_X8, PCIE_5_X16,
  RTX_3090_NVLINK, SPARK_CONNECTX_7, TB4_TCP, TB4_TCP_20G, TB5_RDMA, USB4_HALO_TCP,
  XGMI_MI300X_PER_PEER, XGMI_MI300X_8_GPU,
} from './links';
import { RTX_3090, RTX_3090_UNLOCKS, RTX_4090_UNLOCKS, RTX_5090_UNLOCKS } from './nvidia';
import {
  ARC_PRO_B60_PCIE_2, DGX_SPARK_3, DGX_SPARK_4, DGX_SPARK_4_RING, DGX_SPARK_PAIR, HGX_B200_16_NDR, HGX_B300_16_XDR,
  HGX_H100_16_NDR, M3_ULTRA_TB5_CLUSTERS, MAC_TB4_TCP_CLUSTERS, MI300X_XGMI_8, RTX_3090_NVLINK_2,
  STRIX_HALO_5GBE_CLUSTERS, STRIX_HALO_USB4_RDMA_CLUSTERS, STRIX_HALO_USB4_TCP_CLUSTERS, TOPOLOGY_PRESETS,
  fabricIdSuffix, fabricWord, hgxCluster, linkPresetId, nodeTopology, pcieBox, socCluster,
} from './topologies';
import { DGX_SPARK } from './nvidia-datacenter';
import { RYZEN_AI_MAX_PLUS_395 } from './amd';

describe('interconnect presets', () => {
  it.each(LINK_PRESETS)('$name is valid and all numeric fields have provenance', (link) => {
    expect(validateLink(link)).toEqual([]);
    expect(link.sources.length).toBeGreaterThan(0);
  });

  it.each(HOST_BOUNCE_PCIE_LINKS)('$name accepts a direct-P2P unlock without leaving hostBounce behind', (link) => {
    const mods = [{
      kind: 'p2p' as const, p2p: 'direct' as const, note: 'Test direct P2P on a compatible platform',
      sources: [], estimatedReason: 'Test-only compatibility assumption',
    }];
    expect(validateLinkModifiers(link, mods)).toEqual([]);
    expect(applyLinkModifiers(link, mods).hostBounce).toBeUndefined();
    expect(link.p2p).toBe('host_bounce');
    expect(link.hostBounce).toBeDefined();
  });

  it('accepts every real GeForce unlock combination on every host-bounce PCIe lane configuration', () => {
    for (const axes of [RTX_3090_UNLOCKS, RTX_4090_UNLOCKS, RTX_5090_UNLOCKS]) {
      for (const selection of allUnlockSelections(axes)) {
        const { intraNodeLinkModifiers } = resolveUnlocks(axes, selection);
        for (const link of HOST_BOUNCE_PCIE_LINKS) {
          expect(validateLinkModifiers(link, intraNodeLinkModifiers)).toEqual([]);
        }
      }
    }
  });

  it('uses the PCIe payload rate: line rate after 128b/130b encoding × 256/280 TLP payload efficiency, B/s per direction', () => {
    // Gen5 32 GT/s/lane × 16 lanes × 128/130 coding efficiency ÷ 8 bits/byte × 256/(256 + 24) payload efficiency.
    expect(PCIE_5_X16.bandwidthPerDirectionBytesPerSec).toBeCloseTo(32e9 * 16 * 128 / 130 / 8 * 256 / 280);
    expect(PCIE_5_X16.bandwidthPerDirectionBytesPerSec).toBeCloseTo(57.614e9, -6);
    expect(PCIE_4_X16.bandwidthPerDirectionBytesPerSec).toBe(PCIE_5_X16.bandwidthPerDirectionBytesPerSec / 2);
    expect(PCIE_5_X8.bandwidthPerDirectionBytesPerSec).toBe(PCIE_4_X16.bandwidthPerDirectionBytesPerSec);
    expect(PCIE_4_X8.bandwidthPerDirectionBytesPerSec).toBe(PCIE_4_X16.bandwidthPerDirectionBytesPerSec / 2);
  });

  it('converts vendor bidirectional NVLink totals to per-direction injection', () => {
    expect(RTX_3090_NVLINK.bandwidthPerDirectionBytesPerSec).toBe(112.5e9 / 2);
    expect(NVLINK_4.bandwidthPerDirectionBytesPerSec).toBe(900e9 / 2);
    expect(NVLINK_5.bandwidthPerDirectionBytesPerSec).toBe(1.8e12 / 2);
    // Verbs/remote-store proxies, not software NCCL floors.
    expect(NVLINK_5.latencySec).toBe(0.8e-6);
    expect(SPARK_CONNECTX_7.latencySec).toBe(1.5e-6);
  });

  it('converts network bit rates to bytes without summing directions', () => {
    for (const [link, gigabits] of [
      [IB_NDR_400, 400], [IB_XDR_800, 800], [SPARK_CONNECTX_7, 200], [TB5_RDMA, 80],
      [TB4_TCP, 40], [TB4_TCP_20G, 20], [USB4_HALO_TCP, 20], [ETHERNET_5G_TCP, 5], [ETHERNET_10G_TCP, 10],
    ] as const) {
      expect(link.bandwidthPerDirectionBytesPerSec).toBe(gigabits * 1e9 / 8);
    }
  });

  it('shares estimated host DRAM across active devices and removes staging costs when unlocked', () => {
    // DDR5-6000 × two channels × 8 B = 96 GB/s, divided by write+read and active devices.
    expect(PCIE_5_X16.hostBounce?.hostMemBandwidthBytesPerSec).toBe(96e9);
    expect(PCIE_5_X16.hostBounce?.extraLatencySec).toBe(12e-6);
    expect(effectiveLink(PCIE_5_X16, 2).bandwidthBytesPerSec).toBe(96e9 / (2 * 2));
    expect(effectiveLink(PCIE_5_X16, 4).bandwidthBytesPerSec).toBe(96e9 / (2 * 4));
    expect(effectiveLink(PCIE_5_X16, 2).latencySec).toBeCloseTo(13e-6);
    const unlocked = applyLinkModifiers(PCIE_5_X16, [{
      kind: 'p2p', p2p: 'direct', note: 'Test direct path', sources: [], estimatedReason: 'Test assumption',
    }]);
    expect(effectiveLink(unlocked, 4)).toEqual({
      bandwidthBytesPerSec: PCIE_5_X16.bandwidthPerDirectionBytesPerSec, latencySec: 1e-6,
    });
  });

  it('makes the full-node xGMI injection assumption explicit and also offers a per-peer link', () => {
    expect(XGMI_MI300X_PER_PEER.bandwidthPerDirectionBytesPerSec).toBe(128e9 / 2);
    expect(XGMI_MI300X_8_GPU.bandwidthPerDirectionBytesPerSec).toBe(7 * 64e9);
    // Ring AR beta = 2(n−1)/n × S/B; 8 GPUs can stripe over all seven outgoing links.
    const cost = ringAllReduceSec(1e9, 8, effectiveLink(XGMI_MI300X_8_GPU, 8));
    expect(cost.bandwidthSec).toBe(2 * 7 / 8 * 1e9 / (7 * 64e9));
  });
});

describe('topology presets and builders', () => {
  it.each(TOPOLOGY_PRESETS)('$id validates including its links', (topology) => {
    expect(validateTopology(topology)).toEqual([]);
  });

  it('uses distinct stable ids for every preset and builder lane choice', () => {
    expect(new Set(TOPOLOGY_PRESETS.map((t) => t.id)).size).toBe(TOPOLOGY_PRESETS.length);
    expect(pcieBox(RTX_3090, 2, PCIE_4_X8).id).not.toBe(pcieBox(RTX_3090, 2, PCIE_4_X16).id);
    expect(pcieBox(RTX_3090, 2, PCIE_4_X16).id).toBe('pcie-2-rtx-3090-pcie-4-0-x16-host-bounce');
    expect(pcieBox(RTX_3090, 2, PCIE_5_X16).id).not.toMatch(/63015384615|\.384615/);
  });

  it('puts SoC clusters on the inter-node path and PCIe boxes on the intra-node path', () => {
    expect(DGX_SPARK_PAIR).toMatchObject({ devicesPerNode: 1, nodes: 2, interNode: SPARK_CONNECTX_7 });
    expect(DGX_SPARK_PAIR.intraNode).toBeUndefined();
    expect(ARC_PRO_B60_PCIE_2).toMatchObject({ devicesPerNode: 2, nodes: 1 });
    expect(ARC_PRO_B60_PCIE_2.intraNode?.p2p).toBe('direct');
    expect(ARC_PRO_B60_PCIE_2.interNode).toBeUndefined();
    expect(RTX_3090_NVLINK_2.intraNode).toBe(RTX_3090_NVLINK);
  });

  it('uses eight-device scale-up nodes and one inter-node NIC budget per GPU', () => {
    expect(HGX_H100_16_NDR).toMatchObject({ devicesPerNode: 8, nodes: 2, intraNode: NVLINK_4, interNode: IB_NDR_400 });
    expect(HGX_B200_16_NDR).toMatchObject({ devicesPerNode: 8, nodes: 2, intraNode: NVLINK_5, interNode: IB_NDR_400 });
    expect(HGX_B300_16_XDR).toMatchObject({ devicesPerNode: 8, nodes: 2, intraNode: NVLINK_5, interNode: IB_XDR_800 });
    expect(MI300X_XGMI_8.intraNode).toBe(XGMI_MI300X_8_GPU);
    expect(hgxCluster('b300').interNode).toBeUndefined();
    expect(hgxCluster('b300', 2, IB_NDR_400).interNode).toBe(IB_NDR_400);
  });

  it('supports x8 boxes and configurable SoC node counts without mutating the link', () => {
    expect(validateTopology(pcieBox(RTX_3090, 4, PCIE_4_X8))).toEqual([]);
    const cluster = socCluster({ id: 'example-soc', name: 'Example SoC' }, 3, TB5_RDMA);
    expect(cluster).toMatchObject({ devicesPerNode: 1, nodes: 3, interNode: TB5_RDMA });
    expect(validateTopology(cluster)).toEqual([]);
    expect(validateTopology(socCluster(RTX_3090, 0, TB5_RDMA))).not.toEqual([]);
  });

  it('gives every link preset a unique, readable id', () => {
    const ids = LINK_PRESETS.map(linkPresetId);
    expect(new Set(ids).size).toBe(LINK_PRESETS.length);
    expect(linkPresetId(PCIE_5_X16)).toBe('pcie-5-0-x16-host-bounce');
    for (const id of ids) expect(id).toMatch(/^[a-z0-9]+(-[a-z0-9]+)*$/);
  });

  it('builds general node topologies with links only at the levels that exist', () => {
    const single = nodeTopology({ devicesPerNode: 1, nodes: 1, intraNode: PCIE_5_X16, interNode: IB_NDR_400 });
    expect(single.intraNode).toBeUndefined();
    expect(single.interNode).toBeUndefined();
    const box = nodeTopology({ devicesPerNode: 4, nodes: 1, intraNode: PCIE_4_X16, interNode: IB_NDR_400, device: RTX_3090 });
    expect(box).toMatchObject({ devicesPerNode: 4, nodes: 1, intraNode: PCIE_4_X16 });
    expect(box.interNode).toBeUndefined();
    expect(box.id).toBe('node-4x1-rtx-3090-intra-pcie-4-0-x16-host-bounce');
    expect(box.name).toBe(`4 × ${RTX_3090.name} / PCIe 4.0 x16 (host bounce)`);
    const cluster = nodeTopology({ devicesPerNode: 8, nodes: 2, intraNode: NVLINK_4, interNode: IB_NDR_400 });
    expect(cluster).toMatchObject({ intraNode: NVLINK_4, interNode: IB_NDR_400 });
    for (const t of [single, box, cluster]) expect(validateTopology(t)).toEqual([]);
    expect(new Set([single.id, box.id, cluster.id]).size).toBe(3);
  });
});

describe('fabric shapes in the builders', () => {
  it('fabricIdSuffix and fabricWord: nothing for the default, a suffix and a word for every other shape', () => {
    expect(fabricIdSuffix(undefined)).toBe('');
    expect(fabricIdSuffix({ kind: 'switched' })).toBe('');
    expect(fabricIdSuffix({ kind: 'switched', oversubscription: 1 })).toBe('');
    expect(fabricIdSuffix({ kind: 'switched', oversubscription: 2 })).toBe('-switched-o2');
    expect(fabricIdSuffix({ kind: 'full_mesh' })).toBe('-mesh');
    expect(fabricIdSuffix({ kind: 'full_mesh', cablesPerPair: 2 })).toBe('-mesh2');
    expect(fabricIdSuffix({ kind: 'ring' })).toBe('-ring');
    expect(fabricIdSuffix({ kind: 'chain' })).toBe('-chain');
    expect(fabricWord(undefined)).toBeUndefined();
    expect(fabricWord({ kind: 'switched' })).toBeUndefined();
    expect(fabricWord({ kind: 'switched', oversubscription: 2 })).toBe('switch, 2:1 oversubscribed');
    expect(fabricWord({ kind: 'full_mesh', cablesPerPair: 2 })).toBe('full mesh, 2 cables per pair');
    expect(fabricWord({ kind: 'ring' })).toBe('ring');
  });

  it('nodeTopology keeps the fabric only with an inter-node link, and omits the default (pre-fabric topologies unchanged)', () => {
    const base = { devicesPerNode: 1, nodes: 4, interNode: SPARK_CONNECTX_7, device: DGX_SPARK };
    const plain = nodeTopology(base);
    expect(nodeTopology({ ...base, fabric: { kind: 'switched' } })).toEqual(plain);
    expect(nodeTopology({ ...base, fabric: { kind: 'switched', oversubscription: 1 } })).toEqual(plain);
    expect(plain.interNodeFabric).toBeUndefined();
    const ring = nodeTopology({ ...base, fabric: { kind: 'ring' } });
    expect(ring.interNodeFabric).toEqual({ kind: 'ring' });
    expect(ring.id).toBe(`${plain.id}-ring`);
    expect(ring.name).toBe(`4 nodes × (1 × ${DGX_SPARK.name}) / ${SPARK_CONNECTX_7.name} (ring)`);
    expect(validateTopology(ring)).toEqual([]);
    // One node: no inter-node link, so no fabric either.
    expect(nodeTopology({ ...base, nodes: 1, fabric: { kind: 'ring' } }).interNodeFabric).toBeUndefined();
    // A fabric without an inter-node link is dropped too.
    expect(nodeTopology({ devicesPerNode: 1, nodes: 4, fabric: { kind: 'ring' } }).interNodeFabric).toBeUndefined();
  });

  it('socCluster: the fabric suffix, word and field, and the pre-fabric id without one', () => {
    const mesh = socCluster(DGX_SPARK, 3, SPARK_CONNECTX_7, { kind: 'full_mesh' });
    expect(mesh.id).toBe('cluster-3-dgx-spark-dgx-spark-connectx-7-200gbe-roce-mesh');
    expect(mesh.name).toBe(`3 × ${DGX_SPARK.name} / ${SPARK_CONNECTX_7.name} (full mesh)`);
    expect(mesh.interNodeFabric).toEqual({ kind: 'full_mesh' });
    expect(socCluster(DGX_SPARK, 3, SPARK_CONNECTX_7).id).toBe('cluster-3-dgx-spark-dgx-spark-connectx-7-200gbe-roce');
    expect(socCluster(DGX_SPARK, 3, SPARK_CONNECTX_7, { kind: 'switched' })).toEqual(socCluster(DGX_SPARK, 3, SPARK_CONNECTX_7));
  });

  it('DGX Spark: pair and switched four unchanged; triangle is a mesh; the switchless four is a ring, each citing the playbook', () => {
    expect(DGX_SPARK_PAIR.interNodeFabric).toBeUndefined();
    expect(DGX_SPARK_4.interNodeFabric).toBeUndefined();
    expect(DGX_SPARK_4.id).toBe('cluster-4-dgx-spark-dgx-spark-connectx-7-200gbe-roce');
    expect(DGX_SPARK_3).toMatchObject({ nodes: 3, interNode: SPARK_CONNECTX_7, interNodeFabric: { kind: 'full_mesh' } });
    expect(DGX_SPARK_4_RING).toMatchObject({ nodes: 4, interNode: SPARK_CONNECTX_7, interNodeFabric: { kind: 'ring' } });
    expect(DGX_SPARK_4_RING.id).toBe(`${DGX_SPARK_4.id}-ring`);
    for (const t of [DGX_SPARK_3, DGX_SPARK_4_RING]) {
      expect(t.sources.some((s) => s.url.includes('dgx-spark-playbooks') && s.fields.includes('interNodeFabric'))).toBe(true);
      expect(validateTopology(t)).toEqual([]);
    }
  });

  it('point-to-point clusters: Macs are meshes; Strix Halo over USB4 is a mesh up to 3 and a ring at 4; Ethernet stays switched', () => {
    for (const t of [...M3_ULTRA_TB5_CLUSTERS, ...MAC_TB4_TCP_CLUSTERS]) expect(t.interNodeFabric).toEqual({ kind: 'full_mesh' });
    for (const clusters of [STRIX_HALO_USB4_RDMA_CLUSTERS, STRIX_HALO_USB4_TCP_CLUSTERS]) {
      expect(clusters.map((t) => t.interNodeFabric)).toEqual([{ kind: 'full_mesh' }, { kind: 'full_mesh' }, { kind: 'ring' }]);
      expect(clusters[2].id).toBe(`cluster-4-${RYZEN_AI_MAX_PLUS_395.id}-${linkPresetId(clusters[2].interNode!)}-ring`);
    }
    for (const t of STRIX_HALO_5GBE_CLUSTERS) expect(t.interNodeFabric).toBeUndefined();
  });
});
