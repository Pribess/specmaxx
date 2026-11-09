import { describe, expect, it } from 'vitest';
import {
  A100_NVLINK_BRIDGE,
  IB_HDR_200,
  NVLINK_3,
  NVLINK_2_HCM_8,
  NVLINK_2_MESH_4,
  NVLINK_2_PAIR,
  PCIE_4_X16_DIRECT,
} from './links';
import { validateTopology } from '../validate';
import {
  CABLING_IDS, POPULAR_SETUPS, allowedCablings, allowedLocalLinks, cablingLabel, cablingOf, catalogDeviceIds, composeTopology,
  defaultCabling, fabricOf, linkById, networkOf, popularSetupById, resolveNetwork, setupErrors, setupLabel, setupOptionsFor,
  type CablingId, type SetupChoice,
} from './catalog';
import { DEVICE_PRESETS } from './index';
import {
  ETHERNET_10G_TCP, IB_NDR_400, IB_XDR_800, NVLINK_4, NVLINK_5, PCIE_4_X16, PCIE_5_X16, PCIE_5_X16_DIRECT,
  PCIE_4_X8, PCIE_4_X8_DIRECT, PCIE_5_X8, PCIE_5_X8_DIRECT, RTX_3090_NVLINK, SPARK_CONNECTX_7, TB4_TCP, TB4_TCP_20G, TB5_RDMA,
  TB5_TCP,
  USB4_HALO_RDMA, XGMI_MI300X_8_GPU,
} from './links';
import { linkPresetId } from './topologies';

/** A one-SoC-per-machine choice over the device's default network. */
function soc(deviceId: string, nodes: number, extra: Partial<SetupChoice> = {}): SetupChoice {
  return { deviceId, devicesPerNode: 1, nodes, ...extra };
}

describe('setup catalog: every device', () => {
  it('has an entry for every device preset, and nothing else', () => {
    expect(catalogDeviceIds().sort()).toEqual(DEVICE_PRESETS.map((d) => d.id).sort());
    expect(() => setupOptionsFor('no-such-device')).toThrow(/no setup options/);
  });

  it.each(DEVICE_PRESETS.map((d) => [d.id] as const))('%s: well-formed options', (id) => {
    const o = setupOptionsFor(id);
    expect(o.deviceId).toBe(id);
    expect(o.shortName.trim()).not.toBe('');
    expect(o.networks.length).toBeGreaterThan(0);
    expect(o.devicesPerNodeCounts[0]).toBe(1);
    expect(o.nodeCounts[0]).toBe(1);
    for (const n of o.networks) {
      expect(n.portsPerNode).toBeGreaterThanOrEqual(1);
      expect(n.maxCablesPerPair).toBeGreaterThanOrEqual(1);
      expect(n.sources.length).toBeGreaterThan(0);
      expect(n.sources.some((s) => s.fields.includes('portsPerNode')), `${id} ${n.shortName} cites its port count`).toBe(true);
      for (const s of n.sources) expect(s.url.startsWith('https://')).toBe(true);
    }
    // No two networks or local links share a link.
    expect(new Set(o.networks.map((n) => linkPresetId(n.link))).size).toBe(o.networks.length);
    expect(new Set(o.localLinks.map((l) => linkPresetId(l.link))).size).toBe(o.localLinks.length);
    // SoCs have no local link and one device per machine; cards and accelerators have both.
    if (o.localLinks.length === 0) expect(o.devicesPerNodeCounts).toEqual([1]);
    else expect(o.devicesPerNodeCounts).toContain(2);
  });

  it.each(DEVICE_PRESETS.map((d) => [d.id] as const))('%s: every offered machine count has a cabling on every network, and the composed topology validates', (id) => {
    const o = setupOptionsFor(id);
    for (const net of o.networks) {
      for (const nodes of o.nodeCounts) {
        if (nodes === 1) {
          expect(allowedCablings(net, 1, 1)).toEqual([]);
          expect(defaultCabling(net, 1, 1)).toBeUndefined();
          continue;
        }
        const allowed = allowedCablings(net, nodes, 1);
        expect(allowed.length, `${id} ×${nodes} over ${net.shortName}`).toBeGreaterThan(0);
        const d = defaultCabling(net, nodes, 1);
        expect(d).toBeDefined();
        expect(allowed).toContain(d);
        for (const cabling of allowed) {
          const choice = soc(id, nodes, { networkLinkId: linkPresetId(net.link), cabling });
          expect(setupErrors(choice), setupLabel(choice)).toEqual([]);
          expect(validateTopology(composeTopology(choice))).toEqual([]);
        }
      }
    }
    for (const devicesPerNode of o.devicesPerNodeCounts) {
      // Every offered count has at least one local link (SoCs have none and offer only 1 device per machine).
      if (devicesPerNode > 1) expect(allowedLocalLinks(o, devicesPerNode).length, `${id} ×${devicesPerNode}`).toBeGreaterThan(0);
      for (const local of allowedLocalLinks(o, devicesPerNode)) {
        const choice: SetupChoice = { deviceId: id, devicesPerNode, nodes: 1, localLinkId: linkPresetId(local.link) };
        expect(setupErrors(choice)).toEqual([]);
        expect(validateTopology(composeTopology(choice))).toEqual([]);
      }
    }
  });

  /**
   * The defaults the UI used before the catalog (`defaultLinks` in `src/ui/state/hardware.ts`, by device id), which
   * the catalog's `[0]` entries must reproduce so every old URL without link ids resolves to the same links.
   */
  const OLD_DEFAULTS: [string, string | undefined, string][] = [
    ['rtx-5090', linkPresetId(PCIE_5_X16), linkPresetId(ETHERNET_10G_TCP)],
    ['rtx-4090', linkPresetId(PCIE_4_X16), linkPresetId(ETHERNET_10G_TCP)],
    ['rtx-3090', linkPresetId(PCIE_4_X16), linkPresetId(ETHERNET_10G_TCP)],
    ['apple-m5-ultra-80c', undefined, linkPresetId(TB5_RDMA)],
    ['apple-m5-ultra-64c', undefined, linkPresetId(TB5_RDMA)],
    ['apple-m5-max-40c', undefined, linkPresetId(TB5_RDMA)],
    ['apple-m5-max-32c', undefined, linkPresetId(TB5_RDMA)],
    ['apple-m4-max-40c', undefined, linkPresetId(TB5_RDMA)],
    ['apple-m4-max-32c', undefined, linkPresetId(TB5_RDMA)],
    ['apple-m3-ultra-80c', undefined, linkPresetId(TB5_RDMA)],
    ['apple-m3-ultra-60c', undefined, linkPresetId(TB5_RDMA)],
    ['radeon-ai-pro-r9700', linkPresetId(PCIE_5_X16), linkPresetId(ETHERNET_10G_TCP)],
    ['radeon-rx-9070-xt', linkPresetId(PCIE_5_X16), linkPresetId(ETHERNET_10G_TCP)],
    ['ryzen-ai-max-plus-395', undefined, linkPresetId(USB4_HALO_RDMA)],
    ['arc-pro-b70', linkPresetId(PCIE_5_X16), linkPresetId(ETHERNET_10G_TCP)],
    ['arc-pro-b65', linkPresetId(PCIE_5_X16), linkPresetId(ETHERNET_10G_TCP)],
    ['arc-pro-b60', linkPresetId(PCIE_5_X8_DIRECT), linkPresetId(ETHERNET_10G_TCP)],
    // Deliberate change from the pre-catalog PCIe 5.0 x16: the B580 is a PCIe 4.0 x8 card (practicality review, rank 3).
    ['arc-b580', linkPresetId(PCIE_4_X8), linkPresetId(ETHERNET_10G_TCP)],
    ['dgx-spark', undefined, linkPresetId(SPARK_CONNECTX_7)],
    ['b300', linkPresetId(NVLINK_5), linkPresetId(IB_XDR_800)],
    ['b200', linkPresetId(NVLINK_5), linkPresetId(IB_NDR_400)],
    ['h200-sxm', linkPresetId(NVLINK_4), linkPresetId(IB_NDR_400)],
    ['h100-sxm', linkPresetId(NVLINK_4), linkPresetId(IB_NDR_400)],
    ['instinct-mi455x', linkPresetId(PCIE_5_X16_DIRECT), linkPresetId(IB_NDR_400)],
    ['instinct-mi355x', linkPresetId(PCIE_5_X16_DIRECT), linkPresetId(IB_NDR_400)],
    ['instinct-mi325x', linkPresetId(PCIE_5_X16_DIRECT), linkPresetId(IB_NDR_400)],
    ['instinct-mi300x', linkPresetId(XGMI_MI300X_8_GPU), linkPresetId(IB_NDR_400)],
  ];

  it.each(OLD_DEFAULTS)('%s: the default links equal the pre-catalog UI defaults', (id, intra, inter) => {
    const o = setupOptionsFor(id);
    expect(o.localLinks[0] === undefined ? undefined : linkPresetId(o.localLinks[0].link)).toBe(intra);
    expect(linkPresetId(o.networks[0].link)).toBe(inter);
  });

  it('covers every pre-catalog device in the defaults table (the M3 Max is new: Thunderbolt 4 TCP only)', () => {
    const covered = new Set(OLD_DEFAULTS.map(([id]) => id));
    const uncovered = DEVICE_PRESETS.map((d) => d.id).filter(
      (id) => !covered.has(id) && !/^(rtx-pro-|a100-|v100-)/.test(id),
    );
    expect(uncovered.sort()).toEqual(['apple-m3-max-30c', 'apple-m3-max-40c']);
    for (const id of uncovered) {
      const o = setupOptionsFor(id);
      expect(o.networks.map((n) => linkPresetId(n.link))).toEqual([linkPresetId(TB4_TCP), linkPresetId(TB4_TCP_20G)]);
      expect(o.localLinks).toEqual([]);
    }
  });

  it('RTX PRO defaults to direct PCIe P2P: 6000 WS/Max-Q support 2/4/8 cards and SFF never offers x16', () => {
    for (const deviceId of ['rtx-pro-6000-ws', 'rtx-pro-6000-max-q']) {
      const options = setupOptionsFor(deviceId);
      expect(options.devicesPerNodeCounts).toEqual([1, 2, 4, 8]);
      expect(options.localLinks[0].link).toBe(PCIE_5_X16_DIRECT);
      expect(options.localLinks.every((l) => l.link.kind === 'pcie')).toBe(true);
      for (const devicesPerNode of [2, 4, 8]) {
        const choice = { deviceId, devicesPerNode, nodes: 1 };
        expect(setupErrors(choice)).toEqual([]);
        expect(composeTopology(choice).intraNode).toBe(PCIE_5_X16_DIRECT);
      }
    }
    const sff = setupOptionsFor('rtx-pro-4000-sff');
    expect(sff.localLinks.map((l) => l.link)).toEqual([PCIE_5_X8_DIRECT, PCIE_5_X8]);
    expect(
      setupLabel({ deviceId: 'rtx-pro-5000', memCapacityBytes: 72e9, devicesPerNode: 1, nodes: 1 }),
    ).toBe('RTX PRO 5000 72GB');
  });
});

/** Generation and lanes of a PCIe link preset, from the name its factory gives it (`'PCIe 4.0 x8 (host bounce)'`). */
function pcieGenLanes(name: string): { generation: number; lanes: number } {
  const m = /^PCIe (\d)\.0 x(\d+) /.exec(name);
  if (m === null) throw new Error(`not a PCIe preset name: '${name}'`);
  return { generation: Number(m[1]), lanes: Number(m[2]) };
}

describe('setup catalog: PCIe links never exceed the card', () => {
  /**
   * Accelerators without a `hostInterface` (OAM modules, whose host attachment no datasheet states as a slot) that
   * are offered a PCIe link as the stand-in for an xGMI fabric the library lacks. Listed so a new device that offers
   * PCIe without a cited `hostInterface` fails here instead of slipping through.
   */
  const NO_HOST_INTERFACE_PCIE = ['instinct-mi325x', 'instinct-mi355x', 'instinct-mi455x'];

  it.each(DEVICE_PRESETS.map((d) => [d.id] as const))('%s: every offered PCIe link is no wider and no newer than hostInterface', (id) => {
    const device = DEVICE_PRESETS.find((d) => d.id === id)!;
    const pcie = setupOptionsFor(id).localLinks.filter((l) => l.link.kind === 'pcie');
    if (device.hostInterface === undefined) {
      expect(pcie.length === 0 || NO_HOST_INTERFACE_PCIE.includes(id), `${id} offers PCIe without a hostInterface`).toBe(true);
      return;
    }
    const { generation, lanes } = device.hostInterface;
    for (const opt of pcie) {
      const link = pcieGenLanes(opt.link.name);
      expect(link.generation, `${id}: ${opt.link.name} vs Gen${generation}`).toBeLessThanOrEqual(generation);
      expect(link.lanes, `${id}: ${opt.link.name} vs x${lanes}`).toBeLessThanOrEqual(lanes);
    }
  });

  it('Arc B580 (PCIe 4.0 x8): Gen4 x8 host bounce by default, Gen4 x8 direct as the option; Arc Pro B60 (Gen5 x8): x8 only', () => {
    expect(setupOptionsFor('arc-b580').localLinks.map((l) => l.link)).toEqual([PCIE_4_X8, PCIE_4_X8_DIRECT]);
    expect(setupOptionsFor('arc-pro-b60').localLinks.map((l) => l.link)).toEqual([PCIE_5_X8_DIRECT, PCIE_5_X8]);
  });
});

describe('cablings', () => {
  it('fabricOf and cablingOf are inverses over every cabling', () => {
    for (const c of CABLING_IDS) expect(cablingOf(fabricOf(c))).toBe(c);
    expect(fabricOf('switched')).toEqual({ kind: 'switched' });
    expect(fabricOf('mesh2')).toEqual({ kind: 'full_mesh', cablesPerPair: 2 });
    expect(cablingOf(undefined)).toBe('switched');
    expect(cablingOf({ kind: 'switched', oversubscription: 2 })).toBeUndefined();
    expect(cablingOf({ kind: 'full_mesh', cablesPerPair: 3 })).toBeUndefined();
  });

  it('labels', () => {
    expect(CABLING_IDS.map((c) => cablingLabel(c, 4))).toEqual(['switch', 'mesh', 'mesh, 2 cables per pair', 'ring', 'chain']);
    expect(cablingLabel('mesh2', 2)).toBe('mesh, 2 cables');
  });

  const tb5 = (id: string) => networkOf(setupOptionsFor(id), linkPresetId(TB5_RDMA));
  const cx7 = networkOf(setupOptionsFor('dgx-spark'), undefined);
  const usb4 = networkOf(setupOptionsFor('ryzen-ai-max-plus-395'), undefined);

  it('port counts: M3 Ultra 5 (RDMA) of 6, M4 Max 3 of 4, M5 Max 3, M5 Ultra 6, M3 Max 3 TB4, Spark 2, Halo 2', () => {
    expect(tb5('apple-m3-ultra-80c')?.portsPerNode).toBe(5);
    expect(networkOf(setupOptionsFor('apple-m3-ultra-80c'), linkPresetId(TB5_TCP))?.portsPerNode).toBe(6);
    expect(tb5('apple-m4-max-40c')?.portsPerNode).toBe(3);
    expect(tb5('apple-m5-max-40c')?.portsPerNode).toBe(3);
    expect(tb5('apple-m5-ultra-80c')?.portsPerNode).toBe(6);
    expect(tb5('apple-m3-max-40c')).toBeUndefined();
    expect(networkOf(setupOptionsFor('apple-m3-max-40c'), undefined)?.portsPerNode).toBe(3);
    expect(cx7).toMatchObject({ portsPerNode: 2, switchable: true, maxCablesPerPair: 1, maxSwitchlessNodes: 4 });
    expect(usb4).toMatchObject({ portsPerNode: 2, switchable: false, maxCablesPerPair: 1 });
    expect(tb5('apple-m3-ultra-80c')).toMatchObject({ switchable: false, maxCablesPerPair: 2 });
  });

  it('allowedCablings: Thunderbolt has no switch; mesh while the ports allow; ring from 4; chain from 3', () => {
    const m3u = tb5('apple-m3-ultra-80c');
    expect(allowedCablings(m3u, 2, 1)).toEqual(['mesh', 'mesh2']);
    expect(allowedCablings(m3u, 3, 1)).toEqual(['mesh', 'mesh2', 'chain']);
    expect(allowedCablings(m3u, 4, 1)).toEqual(['mesh', 'ring', 'chain']); // mesh2 needs 6 ports, 5 usable
    expect(allowedCablings(m3u, 6, 1)).toEqual(['mesh', 'ring', 'chain']);
    expect(allowedCablings(m3u, 7, 1)).toEqual(['ring', 'chain']);
    expect(allowedCablings(m3u, 8, 1)).toEqual(['ring', 'chain']);
    const m4m = tb5('apple-m4-max-40c');
    expect(allowedCablings(m4m, 4, 1)).toEqual(['mesh', 'ring', 'chain']);
    expect(allowedCablings(m4m, 5, 1)).toEqual(['ring', 'chain']);
    const m5u = tb5('apple-m5-ultra-80c');
    expect(allowedCablings(m5u, 4, 1)).toEqual(['mesh', 'mesh2', 'ring', 'chain']);
    expect(allowedCablings(m5u, 7, 1)).toEqual(['mesh', 'ring', 'chain']);
  });

  it('allowedCablings: Spark has a switch, no striping, direct cabling up to 4; Halo 4 is a ring; Ethernet is a switch only', () => {
    expect(allowedCablings(cx7, 2, 1)).toEqual(['switched', 'mesh']);
    expect(allowedCablings(cx7, 3, 1)).toEqual(['switched', 'mesh', 'chain']);
    expect(allowedCablings(cx7, 4, 1)).toEqual(['switched', 'ring', 'chain']);
    expect(allowedCablings(cx7, 8, 1)).toEqual(['switched']);
    expect(allowedCablings(usb4, 2, 1)).toEqual(['mesh']);
    expect(allowedCablings(usb4, 3, 1)).toEqual(['mesh', 'chain']);
    expect(allowedCablings(usb4, 4, 1)).toEqual(['ring', 'chain']);
    const eth = networkOf(setupOptionsFor('ryzen-ai-max-plus-395'), linkPresetId(ETHERNET_10G_TCP));
    expect(allowedCablings(eth, 4, 1)).toEqual(['switched']);
    // Several devices per machine: a switch only; an uncatalogued link: a switch only; one machine: nothing.
    expect(allowedCablings(cx7, 4, 2)).toEqual(['switched']);
    expect(allowedCablings(undefined, 4, 1)).toEqual(['switched']);
    expect(allowedCablings(usb4, 1, 1)).toEqual([]);
  });

  it('defaultCabling: switch, else mesh, else ring, else chain', () => {
    expect(defaultCabling(cx7, 4, 1)).toBe('switched');
    expect(defaultCabling(tb5('apple-m3-ultra-80c'), 4, 1)).toBe('mesh');
    expect(defaultCabling(tb5('apple-m3-ultra-80c'), 8, 1)).toBe('ring');
    expect(defaultCabling(usb4, 4, 1)).toBe('ring');
    expect(defaultCabling(usb4, 2, 1)).toBe('mesh');
    expect(defaultCabling(undefined, 3, 1)).toBe('switched');
    expect(defaultCabling(usb4, 1, 1)).toBeUndefined();
  });
});

describe('setupErrors', () => {
  const TB5 = linkPresetId(TB5_RDMA);

  it('rejects a mesh beyond the ports: 5 Macs on the M4 Max (3 ports), 7 on the M3 Ultra (5 ports)', () => {
    expect(setupErrors(soc('apple-m4-max-40c', 5, { networkLinkId: TB5, cabling: 'mesh' }))).toEqual([
      'a full mesh of 5 needs 4 ports per node; this machine has 3 usable',
    ]);
    expect(setupErrors(soc('apple-m3-ultra-80c', 7, { networkLinkId: TB5, cabling: 'mesh' }))).toHaveLength(1);
    expect(setupErrors(soc('apple-m3-ultra-80c', 6, { networkLinkId: TB5, cabling: 'mesh' }))).toEqual([]);
  });

  it('two cables per pair: rejected on the M3 Ultra at 4 (6 > 5 ports), accepted on the M5 Ultra', () => {
    expect(setupErrors(soc('apple-m3-ultra-80c', 4, { networkLinkId: TB5, cabling: 'mesh2' }))).toEqual([
      'a full mesh of 4 needs 6 ports per node; this machine has 5 usable',
    ]);
    expect(setupErrors(soc('apple-m5-ultra-80c', 4, { networkLinkId: TB5, cabling: 'mesh2' }))).toEqual([]);
    expect(setupErrors(soc('dgx-spark', 2, { cabling: 'mesh2' }))).toEqual(['ConnectX-7 does not stripe two cables between a pair']);
  });

  it('no switch for Thunderbolt or USB4; a ring needs 3+ machines; Spark direct cabling stops at 4', () => {
    expect(setupErrors(soc('apple-m3-ultra-80c', 4, { cabling: 'switched' }))).toEqual(['no switch exists for this link']);
    expect(setupErrors(soc('ryzen-ai-max-plus-395', 2, { cabling: 'ring' }))).toEqual(['a ring needs at least 3 nodes (got 2); two nodes are a full mesh']);
    expect(setupErrors(soc('dgx-spark', 8, { cabling: 'ring' }))).toEqual([
      'ConnectX-7: direct cabling is supported for up to 4 machines (got 8); more go through a switch',
    ]);
    expect(setupErrors(soc('dgx-spark', 8))).toEqual([]);
  });

  it('the RTX 3090 NVLink bridge joins exactly two cards; other counts fall back to PCIe', () => {
    const o = setupOptionsFor('rtx-3090');
    const bridge = linkPresetId(RTX_3090_NVLINK);
    expect(allowedLocalLinks(o, 2).map((l) => linkPresetId(l.link))).toContain(bridge);
    for (const n of [1, 4, 8]) expect(allowedLocalLinks(o, n).map((l) => linkPresetId(l.link))).not.toContain(bridge);
    expect(allowedLocalLinks(o, 4)[0].link).toBe(PCIE_4_X16);
    expect(setupErrors({ deviceId: 'rtx-3090', devicesPerNode: 2, nodes: 1, localLinkId: bridge })).toEqual([]);
    expect(setupErrors({ deviceId: 'rtx-3090', devicesPerNode: 4, nodes: 1, localLinkId: bridge })).toEqual([
      'NVLink bridge joins exactly 2 devices per machine (got 4)',
    ]);
    expect(setupErrors({ deviceId: 'rtx-3090', devicesPerNode: 8, nodes: 1, localLinkId: bridge })).toHaveLength(1);
    // The RTX 3090 exception above predates the count-restricted A100/V100 links tested separately below.
    for (const id of DEVICE_PRESETS.map((d) => d.id).filter((id) => !/^(a100-|v100-)/.test(id))) {
      const opts = setupOptionsFor(id);
      for (const n of opts.devicesPerNodeCounts.filter((x) => x > 1)) {
        expect(allowedLocalLinks(opts, n).length, `${id} ×${n}`).toBe(opts.localLinks.length - (id === 'rtx-3090' && n !== 2 ? 1 : 0));
      }
    }
  });

  it('Thunderbolt TCP: TB5 Macs get the 80 Gb/s TB5 bridge, the M3 Max the TB4 bridge; old TB4 URLs on a TB5 Mac still resolve', () => {
    const tb5Id = linkPresetId(TB5_TCP);
    const tb4Id = linkPresetId(TB4_TCP);
    for (const id of ['apple-m5-ultra-80c', 'apple-m5-ultra-64c', 'apple-m5-max-40c', 'apple-m5-max-32c', 'apple-m4-max-40c', 'apple-m4-max-32c', 'apple-m3-ultra-80c', 'apple-m3-ultra-60c']) {
      const o = setupOptionsFor(id);
      const tcp = networkOf(o, tb5Id);
      expect(tcp?.link, id).toBe(TB5_TCP);
      expect(tcp?.link.bandwidthPerDirectionBytesPerSec).toBe(10e9);
      expect(networkOf(o, tb4Id), id).toBeUndefined();
      expect(setupErrors(soc(id, 3, { networkLinkId: tb5Id, cabling: 'mesh' }))).toEqual([]);
      expect(composeTopology(soc(id, 3, { networkLinkId: tb5Id })).interNode).toBe(TB5_TCP);
      // An old URL with the TB4 bridge on a TB5 Mac: the link still resolves (uncatalogued for this device, switched only).
      const old = soc(id, 2, { networkLinkId: tb4Id });
      expect(resolveNetwork(o, tb4Id)).toEqual({ link: TB4_TCP });
      expect(composeTopology(old).interNode).toBe(TB4_TCP);
      expect(setupErrors(old)).toEqual([]);
      expect(setupLabel(old)).toContain('Thunderbolt 4 bridge (TCP)');
    }
    for (const id of ['apple-m3-max-40c', 'apple-m3-max-30c']) {
      const o = setupOptionsFor(id);
      // Default the bonded 40 Gb/s Mac ↔ Mac bridge; the 20 Gb/s negotiated variant as the option.
      expect(o.networks.map((n) => n.link)).toEqual([TB4_TCP, TB4_TCP_20G]);
      expect(TB4_TCP.bandwidthPerDirectionBytesPerSec).toBe(5e9);
      expect(TB4_TCP_20G.bandwidthPerDirectionBytesPerSec).toBe(2.5e9);
      expect(networkOf(o, tb5Id)).toBeUndefined();
      expect(composeTopology(soc(id, 2)).interNode).toBe(TB4_TCP);
      const negotiated = soc(id, 2, { networkLinkId: linkPresetId(TB4_TCP_20G) });
      expect(setupErrors(negotiated)).toEqual([]);
      expect(composeTopology(negotiated).interNode).toBe(TB4_TCP_20G);
    }
  });

  it('several GPUs per machine network through a switch only', () => {
    expect(setupErrors({ deviceId: 'rtx-5090', devicesPerNode: 2, nodes: 2, cabling: 'mesh' })).toEqual([
      'a full mesh needs one device per node (got 2); nodes with several devices network through a switch',
    ]);
    expect(setupErrors({ deviceId: 'h100-sxm', devicesPerNode: 8, nodes: 2 })).toEqual([]);
  });

  it('device, memory, counts and link ids', () => {
    expect(setupErrors(soc('no-such-device', 2))).toEqual(["unknown device 'no-such-device'"]);
    expect(setupErrors(soc('apple-m5-ultra-80c', 2, { memCapacityBytes: 128e9 }))).toEqual(['128000000000 B is not a memory option of apple-m5-ultra-80c']);
    expect(setupErrors(soc('apple-m5-ultra-80c', 2, { memCapacityBytes: 256e9 }))).toEqual([]);
    expect(setupErrors(soc('dgx-spark', 0))).toEqual(['nodes must be an integer ≥ 1 (got 0)']);
    expect(setupErrors({ deviceId: 'rtx-5090', devicesPerNode: 1.5, nodes: 1 })).toEqual(['devicesPerNode must be an integer ≥ 1 (got 1.5)']);
    expect(setupErrors(soc('dgx-spark', 2, { networkLinkId: 'no-such-link' }))).toEqual(["unknown link 'no-such-link'"]);
    // A link preset the device does not list (an old URL): allowed as a switch, nothing else.
    const foreign = soc('apple-m3-ultra-80c', 3, { networkLinkId: linkPresetId(SPARK_CONNECTX_7) });
    expect(setupErrors(foreign)).toEqual([]);
    expect(setupErrors({ ...foreign, cabling: 'mesh' })).toEqual([
      "link 'dgx-spark-connectx-7-200gbe-roce' is not catalogued for apple-m3-ultra-80c: only a switch is modeled for it",
    ]);
  });
});

describe('composeTopology', () => {
  it('builds the counts, the chosen or default links, and the fabric (omitted for a switch)', () => {
    const ring = composeTopology(soc('dgx-spark', 4, { cabling: 'ring' }));
    expect(ring).toMatchObject({ devicesPerNode: 1, nodes: 4, interNode: SPARK_CONNECTX_7, interNodeFabric: { kind: 'ring' } });
    expect(ring.intraNode).toBeUndefined();
    expect(ring.id).toBe('node-1x4-dgx-spark-inter-dgx-spark-connectx-7-200gbe-roce-ring');
    const switched = composeTopology(soc('dgx-spark', 4));
    expect(switched.interNodeFabric).toBeUndefined();
    expect(switched.id).toBe('node-1x4-dgx-spark-inter-dgx-spark-connectx-7-200gbe-roce');
    // The default cabling of a Thunderbolt pair is a mesh (no switch exists).
    const pair = composeTopology(soc('apple-m5-ultra-80c', 2));
    expect(pair).toMatchObject({ interNode: TB5_RDMA, interNodeFabric: { kind: 'full_mesh' } });
    const mesh2 = composeTopology(soc('apple-m5-ultra-80c', 4, { cabling: 'mesh2' }));
    expect(mesh2.interNodeFabric).toEqual({ kind: 'full_mesh', cablesPerPair: 2 });
    expect(mesh2.id.endsWith('-mesh2')).toBe(true);
    // Several GPUs in one machine: the local link, no network, no fabric.
    const box = composeTopology({ deviceId: 'rtx-5090', devicesPerNode: 2, nodes: 1 });
    expect(box).toMatchObject({ devicesPerNode: 2, nodes: 1, intraNode: PCIE_5_X16 });
    expect(box.interNode).toBeUndefined();
    expect(box.interNodeFabric).toBeUndefined();
    // A chosen local link; an unknown id falls back to the default.
    expect(composeTopology({ deviceId: 'rtx-5090', devicesPerNode: 2, nodes: 1, localLinkId: linkPresetId(PCIE_5_X16_DIRECT) }).intraNode).toBe(PCIE_5_X16_DIRECT);
    expect(composeTopology({ deviceId: 'rtx-5090', devicesPerNode: 2, nodes: 1, localLinkId: 'nope' }).intraNode).toBe(PCIE_5_X16);
    expect(composeTopology(soc('dgx-spark', 2, { networkLinkId: 'nope' })).interNode).toBe(SPARK_CONNECTX_7);
    // One device: no links at all.
    const one = composeTopology(soc('apple-m3-max-30c', 1));
    expect(one.interNode).toBeUndefined();
    expect(one.interNodeFabric).toBeUndefined();
    expect(() => composeTopology(soc('no-such-device', 2))).toThrow(/unknown device/);
  });
});

describe('setupLabel', () => {
  it('names the exact SKU: chip, GPU-core binning, memory, count, link and cabling', () => {
    expect(setupLabel(soc('apple-m5-ultra-80c', 2, { memCapacityBytes: 256e9, cabling: 'mesh' }))).toBe('M5 Ultra 80-core GPU, 256GB ×2 · TB5 mesh');
    expect(setupLabel(soc('apple-m5-ultra-64c', 2))).toBe('M5 Ultra 64-core GPU, 96GB ×2 · TB5 mesh');
    expect(setupLabel(soc('apple-m3-ultra-80c', 4, { memCapacityBytes: 512e9 }))).toBe('M3 Ultra 80-core GPU, 512GB ×4 · TB5 mesh');
    expect(setupLabel(soc('apple-m5-ultra-80c', 4, { memCapacityBytes: 512e9, cabling: 'mesh2' }))).toBe('M5 Ultra 80-core GPU, 512GB ×4 · TB5 mesh, 2 cables per pair');
    expect(setupLabel(soc('apple-m3-max-40c', 3))).toBe('M3 Max 40-core GPU, 48GB ×3 · TB TCP mesh');
    expect(setupLabel(soc('dgx-spark', 4, { cabling: 'ring' }))).toBe('DGX Spark 128GB ×4 · ConnectX-7 ring');
    expect(setupLabel(soc('dgx-spark', 4))).toBe('DGX Spark 128GB ×4 · ConnectX-7 switch');
    expect(setupLabel(soc('ryzen-ai-max-plus-395', 4))).toBe('Strix Halo 395 128GB ×4 · USB4 ring');
    expect(setupLabel({ deviceId: 'rtx-5090', devicesPerNode: 2, nodes: 1 })).toBe('RTX 5090 32GB ×2 · PCIe 5.0 x16');
    expect(setupLabel({ deviceId: 'rtx-5090', devicesPerNode: 1, nodes: 1 })).toBe('RTX 5090 32GB');
    expect(setupLabel({ deviceId: 'h100-sxm', devicesPerNode: 8, nodes: 2 })).toBe('H100 SXM 80GB ×8 × 2 machines · NVLink 4 / IB NDR');
    // A link the device does not list is named by its preset name.
    expect(setupLabel(soc('apple-m3-ultra-80c', 3, { networkLinkId: linkPresetId(SPARK_CONNECTX_7) }))).toBe(
      'M3 Ultra 80-core GPU, 96GB ×3 · DGX Spark ConnectX-7 200GbE RoCE switch',
    );
  });
});

describe('POPULAR_SETUPS', () => {
  it('is at most 10, grouped by vendor, with unique ids and names equal to setupLabel', () => {
    expect(POPULAR_SETUPS.length).toBeLessThanOrEqual(10);
    expect(new Set(POPULAR_SETUPS.map((p) => p.id)).size).toBe(POPULAR_SETUPS.length);
    expect(new Set(POPULAR_SETUPS.map((p) => p.name)).size).toBe(POPULAR_SETUPS.length);
    for (const p of POPULAR_SETUPS) expect(p.name).toBe(setupLabel(p.choice));
    expect(POPULAR_SETUPS.map((p) => p.id)).toEqual([
      'rtx-5090-x2-pcie5',
      'rtx-3090-x2-nvlink',
      'dgx-spark-x2',
      'dgx-spark-x3-mesh',
      'dgx-spark-x4-ring',
      'dgx-spark-x4',
      'm3-ultra-80c-512gb-x4-tb5-mesh',
      'm5-ultra-80c-256gb-x2-tb5',
      'm5-ultra-80c-512gb-x4-tb5-mesh2',
      'strix-halo-128gb-x4-usb4-ring',
    ]);
    expect(POPULAR_SETUPS.map((p) => p.name)).toEqual([
      'RTX 5090 32GB ×2 · PCIe 5.0 x16',
      'RTX 3090 24GB ×2 · NVLink bridge',
      'DGX Spark 128GB ×2 · ConnectX-7 mesh',
      'DGX Spark 128GB ×3 · ConnectX-7 mesh',
      'DGX Spark 128GB ×4 · ConnectX-7 ring',
      'DGX Spark 128GB ×4 · ConnectX-7 switch',
      'M3 Ultra 80-core GPU, 512GB ×4 · TB5 mesh',
      'M5 Ultra 80-core GPU, 256GB ×2 · TB5 mesh',
      'M5 Ultra 80-core GPU, 512GB ×4 · TB5 mesh, 2 cables per pair',
      'Strix Halo 395 128GB ×4 · USB4 ring',
    ]);
  });

  it.each(POPULAR_SETUPS.map((p) => [p.id, p] as const))('%s: valid, multi-device, cited, and its topology validates', (_id, p) => {
    expect(setupErrors(p.choice)).toEqual([]);
    expect(p.choice.devicesPerNode * p.choice.nodes).toBeGreaterThan(1);
    expect(p.sources.length).toBeGreaterThan(0);
    for (const s of p.sources) expect(s.url.startsWith('https://')).toBe(true);
    const t = composeTopology(p.choice);
    expect(validateTopology(t)).toEqual([]);
    expect(t.devicesPerNode).toBe(p.choice.devicesPerNode);
    expect(t.nodes).toBe(p.choice.nodes);
    // Every popular setup names its links and cabling explicitly (no reliance on defaults).
    if (p.choice.nodes > 1) {
      expect(p.choice.networkLinkId).toBeDefined();
      expect(p.choice.cabling).toBeDefined();
    } else expect(p.choice.localLinkId).toBeDefined();
    expect(popularSetupById(p.id)).toBe(p);
  });

  it('the Spark ring and switch differ only in cabling, and the ring topology carries the ring fabric', () => {
    const ring = popularSetupById('dgx-spark-x4-ring')!;
    const sw = popularSetupById('dgx-spark-x4')!;
    expect({ ...ring.choice, cabling: undefined }).toEqual({ ...sw.choice, cabling: undefined });
    expect(composeTopology(ring.choice).interNodeFabric).toEqual({ kind: 'ring' });
    expect(composeTopology(sw.choice).interNodeFabric).toBeUndefined();
    expect(composeTopology(popularSetupById('strix-halo-128gb-x4-usb4-ring')!.choice).interNodeFabric).toEqual({ kind: 'ring' });
    expect(composeTopology(popularSetupById('m5-ultra-80c-512gb-x4-tb5-mesh2')!.choice).interNodeFabric).toEqual({ kind: 'full_mesh', cablesPerPair: 2 });
  });
});

describe('linkById', () => {
  it('finds link presets by id and rejects unknown ids', () => {
    expect(linkById(linkPresetId(TB5_RDMA))).toBe(TB5_RDMA);
    expect(linkById('thunderbolt-5-rdma-macos-26-2')).toBe(TB5_RDMA);
    expect(linkById('nope')).toBeUndefined();
  });

  it('every cabling id is a valid fabric for some setup', () => {
    const ids: CablingId[] = ['switched', 'mesh', 'mesh2', 'ring', 'chain'];
    expect([...CABLING_IDS]).toEqual(ids);
  });
});

describe('A100 / V100 catalog', () => {
  it('restricts A100 PCIe bridges to pairs and A100 networks to HDR200', () => {
    for (const deviceId of ['a100-pcie-40gb', 'a100-pcie-80gb']) {
      expect(
        setupErrors({
          deviceId,
          devicesPerNode: 2,
          nodes: 1,
          localLinkId: linkPresetId(A100_NVLINK_BRIDGE),
        }),
      ).toEqual([]);
      expect(
        setupErrors({
          deviceId,
          devicesPerNode: 4,
          nodes: 1,
          localLinkId: linkPresetId(A100_NVLINK_BRIDGE),
        }).length,
      ).toBeGreaterThan(0);
      expect(setupOptionsFor(deviceId).localLinks[0].link).toBe(PCIE_4_X16_DIRECT);
    }
    for (const deviceId of ['a100-sxm4-40gb', 'a100-sxm4-80gb']) {
      const options = setupOptionsFor(deviceId);
      expect(options.localLinks[0].link).toBe(NVLINK_3);
      expect(options.networks.map((n) => n.link)).toEqual([IB_HDR_200]);
      expect(setupErrors({ deviceId, devicesPerNode: 8, nodes: 2 })).toEqual([]);
    }
  });
  it('offers V100 board-specific NVLink only at matching node sizes, PCIe never NVLink', () => {
    for (const [count, link] of [
      [2, NVLINK_2_PAIR],
      [4, NVLINK_2_MESH_4],
      [8, NVLINK_2_HCM_8],
    ] as const) {
      expect(
        setupErrors({
          deviceId: 'v100-sxm2',
          devicesPerNode: count,
          nodes: 1,
          localLinkId: linkPresetId(link),
        }),
      ).toEqual([]);
      expect(
        composeTopology({
          deviceId: 'v100-sxm2',
          devicesPerNode: count,
          nodes: 1,
          localLinkId: linkPresetId(link),
        }).intraNode,
      ).toBe(link);
      const wrongCount = count === 8 ? 4 : 8;
      expect(
        setupErrors({
          deviceId: 'v100-sxm2',
          devicesPerNode: wrongCount,
          nodes: 1,
          localLinkId: linkPresetId(link),
        }).length,
      ).toBeGreaterThan(0);
    }
    expect(setupOptionsFor('v100-pcie').localLinks.every((l) => l.link.kind === 'pcie')).toBe(true);
    for (const devicesPerNode of [2, 4])
      expect(setupErrors({ deviceId: 'v100-pcie', devicesPerNode, nodes: 1 })).toEqual([]);
    expect(
      setupLabel({ deviceId: 'v100-pcie', memCapacityBytes: 16e9, devicesPerNode: 1, nodes: 1 }),
    ).toBe('V100 PCIe 16GB');
  });
});
