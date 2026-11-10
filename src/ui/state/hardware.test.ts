import { describe, expect, it } from 'vitest';
import { simulate } from '../../sim/engine/simulate';
import { DEVICE_PRESETS } from '../../sim/hw/presets/index';
import { POPULAR_SETUPS, composeTopology } from '../../sim/hw/presets/catalog';
import {
  LINK_PRESETS,
  PCIE_4_X16,
  PCIE_5_X16,
  SPARK_CONNECTX_7,
  TB5_RDMA,
  USB4_HALO_RDMA,
  NVLINK_4,
  NVLINK_C2C,
  XGMI_MI300X_8_GPU,
  hostPcieLink,
} from '../../sim/hw/presets/links';
import { DEFAULT_HOST_RAM, graceHostRam } from '../../sim/hw/presets/storage';
import { validateDevice, validateLink, validateTopology } from '../../sim/hw/validate';
import { REF_DENSE_8B } from '../../sim/models/reference';
import { SINGLE_DEVICE_HW, type HardwareChoice } from './appState';
import {
  EDITED_REASON,
  UNITS,
  applyEdits,
  cablingName,
  cablingOptions,
  defaultLinks,
  editableFields,
  editableLinkFields,
  effectiveCabling,
  findDevice,
  fromDisplay,
  inNetworkReductionsOf,
  matchingPopularSetup,
  nicLabel,
  nicOptions,
  normalizeNics,
  nearestCount,
  normalizeCabling,
  popularSetupChoice,
  resolveDevice,
  resolveHardware,
  resolveHost,
  toDisplay,
} from './hardware';
import { REFERENCE_QUANT_DEFAULTS } from './quant';

const hw = (over: Partial<HardwareChoice>): HardwareChoice => ({ ...SINGLE_DEVICE_HW, ...over });

describe('resolveHost', () => {
  const plain = DEVICE_PRESETS.filter((d) => d.hostLink === undefined);
  it.each(plain.map((d) => [d.id, d] as const))(
    '%s (no hostLink): the desktop RAM default and its PCIe interface, as before',
    (_id, device) => {
      for (const devicesPerNode of [1, 8, 72]) {
        const host = resolveHost({ host: undefined, devicesPerNode }, device);
        expect(host.ram).toBe(DEFAULT_HOST_RAM);
        expect(host.ramDefault).toBe(true);
        expect(host.link).toEqual(
          device.hostInterface &&
            hostPcieLink(device.hostInterface.generation, device.hostInterface.lanes),
        );
        expect(host.linkOverridden).toBe(false);
        const overridden = resolveHost(
          { host: { link: { generation: 4, lanes: 8 } }, devicesPerNode },
          device,
        );
        expect(overridden.link).toEqual(hostPcieLink(4, 8));
        expect(overridden.linkOverridden).toBe(true);
      }
    },
  );

  it('GB200 / GB300: NVLink-C2C and the Grace pool of ceil(GPUs / 2) CPUs by default', () => {
    const gb200 = findDevice('gb200')!;
    const gb300 = findDevice('gb300')!;
    const rack = resolveHost({ host: undefined, devicesPerNode: 72 }, gb200);
    expect(rack.link).toBe(NVLINK_C2C);
    expect(rack.ram).toEqual(graceHostRam('gb200', 72));
    expect(rack.ram.capacityBytes).toBe(36 * 480e9);
    expect(rack.ram.bandwidthBytesPerSec).toBe(36 * 384e9);
    expect(resolveHost({ host: undefined, devicesPerNode: 4 }, gb200).ram.capacityBytes).toBe(
      2 * 480e9,
    );
    expect(resolveHost({ host: undefined, devicesPerNode: 1 }, gb300).ram).toEqual(
      graceHostRam('gb300', 1),
    );
  });

  it('GB200: ignores a PCIe override, keeps a picked RAM preset and an edited capacity', () => {
    const gb200 = findDevice('gb200')!;
    const pcie = resolveHost(
      { host: { link: { generation: 5, lanes: 16 } }, devicesPerNode: 72 },
      gb200,
    );
    expect(pcie.link).toBe(NVLINK_C2C);
    expect(pcie.linkOverridden).toBe(false);
    const picked = resolveHost({ host: { ramId: DEFAULT_HOST_RAM.id }, devicesPerNode: 72 }, gb200);
    expect(picked.ram).toBe(DEFAULT_HOST_RAM);
    expect(picked.ramDefault).toBe(false);
    const edited = resolveHost({ host: { ramCapacityBytes: 1e12 }, devicesPerNode: 72 }, gb200);
    expect(edited.ram.capacityBytes).toBe(1e12);
    expect(edited.ram.bandwidthBytesPerSec).toBe(36 * 384e9);
    expect(edited.ram.estimated).toContainEqual({ field: 'capacityBytes', reason: EDITED_REASON });
  });

  it('GB200 NVL72 with offloading: the engine gets the C2C link and the 36-CPU Grace pool', () => {
    const stack = resolveHardware(
      hw({ deviceId: 'gb200', devicesPerNode: 72, nodes: 1 }),
      true,
    ).stack;
    expect(stack.host?.link).toBe(NVLINK_C2C);
    expect(stack.host?.ram).toEqual(graceHostRam('gb200', 72));
  });
});

describe('defaultLinks and topologies', () => {
  it.each(DEVICE_PRESETS.map((d) => [d.id, d] as const))(
    '%s: default links and node topologies validate',
    (_id, device) => {
      const links = defaultLinks(device);
      if (links.intraNode !== undefined) expect(validateLink(links.intraNode)).toEqual([]);
      expect(validateLink(links.interNode)).toEqual([]);
      for (const [devicesPerNode, nodes] of [
        [1, 1],
        [2, 1],
        [1, 2],
        [4, 2],
      ]) {
        const r = resolveHardware(hw({ deviceId: device.id, devicesPerNode, nodes }));
        expect(validateTopology(r.topology)).toEqual([]);
        expect(r.topology.interNode === undefined).toBe(nodes === 1);
        if (devicesPerNode === 1) expect(r.topology.intraNode).toBeUndefined();
      }
    },
  );

  it('picks the expected defaults', () => {
    expect(defaultLinks(findDevice('rtx-5090')!).intraNode).toBe(PCIE_5_X16);
    expect(defaultLinks(findDevice('rtx-3090')!).intraNode).toBe(PCIE_4_X16);
    expect(defaultLinks(findDevice('h100-sxm')!).intraNode).toBe(NVLINK_4);
    expect(defaultLinks(findDevice('instinct-mi300x')!).intraNode).toBe(XGMI_MI300X_8_GPU);
    expect(defaultLinks(findDevice('apple-m3-ultra-60c')!)).toEqual({ interNode: TB5_RDMA });
    expect(defaultLinks(findDevice('ryzen-ai-max-plus-395')!).interNode).toBe(USB4_HALO_RDMA);
    expect(defaultLinks(findDevice('dgx-spark')!).interNode).toBe(SPARK_CONNECTX_7);
  });

  it('offers the cablings the ports allow and defaults to a switch, else a mesh, else a ring', () => {
    const spark = (nodes: number) => hw({ deviceId: 'dgx-spark', nodes });
    expect(cablingOptions(spark(1))).toEqual([]);
    expect(effectiveCabling(spark(1))).toBeUndefined();
    expect(effectiveCabling(spark(4))).toBe('switched');
    expect(cablingOptions(spark(4))).toContain('ring');
    expect(cablingOptions(spark(3))).not.toContain('ring');
    const halo = (nodes: number) => hw({ deviceId: 'ryzen-ai-max-plus-395', nodes });
    expect(effectiveCabling(halo(2))).toBe('mesh');
    expect(effectiveCabling(halo(3))).toBe('mesh');
    expect(effectiveCabling(halo(4))).toBe('ring');
    // Several GPUs per machine: a switch only.
    expect(cablingOptions(hw({ devicesPerNode: 2, nodes: 2 }))).toEqual(['switched']);
  });

  it('a cabling the link does not allow falls back to the default; normalizeCabling drops it and defaults', () => {
    const ring3 = hw({ deviceId: 'dgx-spark', nodes: 3, cabling: 'ring' });
    expect(effectiveCabling(ring3)).toBe('switched');
    expect(normalizeCabling(ring3).cabling).toBeUndefined();
    expect(
      normalizeCabling(hw({ deviceId: 'dgx-spark', nodes: 4, cabling: 'switched' })).cabling,
    ).toBeUndefined();
    expect(normalizeCabling(hw({ deviceId: 'dgx-spark', nodes: 4, cabling: 'ring' })).cabling).toBe(
      'ring',
    );
    // A link the device's catalog does not list (an older URL) is switched only.
    expect(
      cablingOptions(
        hw({
          deviceId: 'ryzen-ai-max-plus-395',
          nodes: 4,
          interLinkId: 'infiniband-ndr400-one-nic-per-gpu',
        }),
      ),
    ).toEqual(['switched']);
  });

  it('names cablings in plain language, two machines as cables', () => {
    expect(cablingName('mesh', 2)).toBe('One cable');
    expect(cablingName('mesh2', 2)).toBe('Two cables');
    expect(cablingName('mesh', 4)).toBe('Full mesh');
    expect(cablingName('chain', 4)).toBe('Daisy chain');
    expect(cablingName('switched', 4)).toBe('Switch');
  });

  it('snaps a count to the nearest offered one', () => {
    expect(nearestCount([1, 2, 4, 8], 3)).toBe(2);
    expect(nearestCount([1, 2, 4, 8], 6)).toBe(4);
    expect(nearestCount([1, 2, 4, 8], 16)).toBe(8);
    expect(nearestCount([1], 4)).toBe(1);
  });

  it('a ring cabling reaches the topology; a switch leaves it absent', () => {
    const ring = resolveHardware(hw({ deviceId: 'dgx-spark', nodes: 4, cabling: 'ring' }));
    expect(ring.topology.interNodeFabric).toEqual({ kind: 'ring' });
    expect(ring.cabling).toBe('ring');
    expect(validateTopology(ring.topology)).toEqual([]);
    const sw = resolveHardware(hw({ deviceId: 'dgx-spark', nodes: 4 }));
    expect(sw.topology.interNodeFabric).toBeUndefined();
    expect(sw.cabling).toBe('switched');
  });

  it('applies the P2P unlock to the host-bounce intra-node link', () => {
    const stock = resolveHardware(hw({ devicesPerNode: 2 }));
    expect(stock.topology.intraNode?.p2p).toBe('host_bounce');
    const p2p = resolveHardware(hw({ devicesPerNode: 2, unlocks: { 'pcie-p2p': 'direct' } }));
    expect(p2p.topology.intraNode?.p2p).toBe('direct');
    expect(p2p.notes.some((n) => n.startsWith('intraNode link variant: p2p direct'))).toBe(true);
    expect(validateTopology(p2p.topology)).toEqual([]);
  });
});

describe('popular setups', () => {
  it.each(POPULAR_SETUPS.map((p) => [p.id, p] as const))(
    '%s: round-trips through popularSetupChoice and resolves to its topology',
    (_id, p) => {
      const choice = popularSetupChoice(p);
      expect(matchingPopularSetup(choice)).toBe(p);
      const r = resolveHardware(choice);
      const expected = composeTopology(p.choice);
      expect(validateTopology(r.topology)).toEqual([]);
      expect(r.topology.devicesPerNode).toBe(expected.devicesPerNode);
      expect(r.topology.nodes).toBe(expected.nodes);
      expect(r.topology.intraNode).toBe(expected.intraNode);
      expect(r.topology.interNode).toBe(expected.interNode);
      expect(r.topology.interNodeFabric).toEqual(expected.interNodeFabric);
      const device = findDevice(p.choice.deviceId)!;
      expect(r.memCapacityBytes).toBe(
        p.choice.memCapacityBytes ?? device.memory.capacityOptionsBytes[0],
      );
    },
  );

  it('RTX 5090 ×2 sets the device, counts and the PCIe link', () => {
    const p = POPULAR_SETUPS.find((s) => s.id === 'rtx-5090-x2-pcie5')!;
    expect(popularSetupChoice(p)).toEqual({
      ...SINGLE_DEVICE_HW,
      deviceId: 'rtx-5090',
      devicesPerNode: 2,
      intraLinkId: 'pcie-5-0-x16-host-bounce',
    });
  });

  it('M5 Ultra 256GB picks the 256GB option', () => {
    const p = POPULAR_SETUPS.find((s) => s.id === 'm5-ultra-80c-256gb-x2-tb5')!;
    expect(popularSetupChoice(p).capacityIndex).toBe(1);
  });

  it('DGX Spark ×4 ring and switch are told apart by the cabling', () => {
    const sparks = POPULAR_SETUPS.filter(
      (s) => s.choice.deviceId === 'dgx-spark' && s.choice.nodes === 4,
    );
    expect(sparks.length).toBe(2);
    const ring = sparks.find((s) => s.choice.cabling === 'ring')!;
    const other = sparks.find((s) => s !== ring)!;
    expect(matchingPopularSetup(popularSetupChoice(ring))).toBe(ring);
    expect(matchingPopularSetup(popularSetupChoice(other))).toBe(other);
    expect(matchingPopularSetup(hw({ deviceId: 'dgx-spark', nodes: 4 }))).toBe(other);
    expect(matchingPopularSetup(hw({ deviceId: 'dgx-spark', nodes: 4, cabling: 'ring' }))).toBe(
      ring,
    );
  });

  it('a default-link choice with the same counts matches; any change matches nothing', () => {
    const p = POPULAR_SETUPS.find((s) => s.id === 'rtx-5090-x2-pcie5')!;
    expect(matchingPopularSetup(hw({ devicesPerNode: 2 }))).toBe(p);
    expect(matchingPopularSetup(SINGLE_DEVICE_HW)).toBeUndefined();
    const choice = popularSetupChoice(p);
    expect(matchingPopularSetup({ ...choice, devicesPerNode: 3 })).toBeUndefined();
    expect(
      matchingPopularSetup({ ...choice, intraLinkId: 'pcie-5-0-x8-host-bounce' }),
    ).toBeUndefined();
    expect(
      matchingPopularSetup({ ...choice, deviceEdits: { 'memory.bandwidthBytesPerSec': 2e12 } }),
    ).toBeUndefined();
    expect(matchingPopularSetup({ ...choice, unlocks: { 'pcie-p2p': 'direct' } })).toBeUndefined();
  });
});

describe('spec edits', () => {
  it('apply after the unlocks, rename the device and keep it valid', () => {
    const r = resolveDevice({
      deviceId: 'rtx-4090',
      unlocks: { 'vram-mod': '48gb' },
      deviceEdits: { 'memory.capacityOptionsBytes.0': 40e9 },
    });
    expect(r.unlocked.memory.capacityOptionsBytes).toEqual([48e9]);
    expect(r.device.memory.capacityOptionsBytes).toEqual([40e9]);
    expect(r.device.id).toBe('rtx-4090-custom');
    expect(r.device.name).toBe(`${r.base.name} (edited)`);
    expect(r.edited).toBe(true);
    expect(r.device.estimated).toContainEqual({
      field: 'memory.capacityOptionsBytes.0',
      reason: EDITED_REASON,
    });
    expect(validateDevice(r.device)).toEqual([]);
    expect(r.base.memory.capacityOptionsBytes).toEqual([24e9]); // presets are never mutated
  });

  it('ignore unknown paths and return the spec itself without edits', () => {
    const device = findDevice('rtx-5090')!;
    expect(applyEdits(device, {}).spec).toBe(device);
    expect(applyEdits(device, { 'memory.nope': 1 })).toEqual({
      spec: device,
      unknownPaths: ['memory.nope'],
    });
    expect(resolveDevice({ deviceId: 'rtx-5090', unlocks: {}, deviceEdits: {} }).edited).toBe(
      false,
    );
  });

  it('reach simulate: doubling the memory bandwidth doubles the (memory-bound) decode rate', () => {
    const run = (choice: HardwareChoice) =>
      simulate({
        model: REF_DENSE_8B,
        quant: REFERENCE_QUANT_DEFAULTS.quant,
        parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
        workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
        hardware: resolveHardware(choice).stack,
      });
    const stock = run(hw({}));
    const edited = run(hw({ deviceEdits: { 'memory.bandwidthBytesPerSec': 2 * 1.792e12 } }));
    expect(stock.decode.tokensPerSec).toBeCloseTo(117.1, 0);
    expect(edited.decode.tokensPerSec / stock.decode.tokensPerSec).toBeCloseTo(2, 2);
    expect(edited.effectiveDevice.name).toMatch(/\(edited\)$/);
  });

  it('on links: edited values reach the topology and stay valid', () => {
    const r = resolveHardware(
      hw({ devicesPerNode: 2, intraEdits: { bandwidthPerDirectionBytesPerSec: 10e9 } }),
    );
    expect(r.topology.intraNode?.bandwidthPerDirectionBytesPerSec).toBe(10e9);
    expect(r.topology.intraNode?.name).toMatch(/\(edited\)$/);
    expect(r.edited).toBe(true);
    expect(validateTopology(r.topology)).toEqual([]);
  });
});

describe('editable fields', () => {
  it.each(DEVICE_PRESETS.map((d) => [d.id, d] as const))(
    '%s: every editable field has a human label and a unit, never its raw path',
    (_id, device) => {
      for (const f of editableFields(device)) {
        expect(f.label, f.path).not.toBe(f.path);
        expect(f.unit, f.path).toBeDefined();
      }
      expect(editableFields(device).some((f) => f.path.startsWith('hostLink.'))).toBe(false);
    },
  );

  it.each(LINK_PRESETS.map((l) => [l.name, l] as const))(
    '%s: every editable link field has a human label, never its raw path',
    (_name, link) => {
      for (const f of editableLinkFields(link)) expect(f.label, f.path).not.toBe(f.path);
    },
  );
  it('covers every numeric leaf with a group, a label and a unit', () => {
    for (const device of DEVICE_PRESETS) {
      const fields = editableFields(device);
      expect(fields.length).toBeGreaterThan(5);
      for (const f of fields) expect(f.label).not.toBe('');
    }
    const fields = editableFields(findDevice('rtx-5090')!);
    const bw = fields.find((f) => f.path === 'memory.bandwidthBytesPerSec')!;
    expect(bw).toMatchObject({ group: 'memory', unit: UNITS.gbps });
    expect(toDisplay(bw.value, bw.unit)).toBeCloseTo(1792);
    const bf16 = fields.find(
      (f) => f.group === 'matmul' && f.label === 'bf16, fp32 accumulate, dense',
    );
    expect(bf16?.unit).toBe(UNITS.tflops);
    expect(fields.some((f) => f.group === 'vector')).toBe(true);
    expect(fields.some((f) => f.group === 'tiling')).toBe(true);
    expect(
      editableFields(findDevice('apple-m5-max-40c')!).some((f) => f.label.includes(': ')),
    ).toBe(true);
  });

  it('converts display units exactly for round numbers', () => {
    expect(fromDisplay(1792, UNITS.gbps)).toBe(1.792e12);
    expect(fromDisplay(0.5, UNITS.us)).toBeCloseTo(5e-7, 15);
    expect(toDisplay(209.5e12, UNITS.tflops)).toBeCloseTo(209.5);
  });

  it('labels link fields', () => {
    const fields = editableLinkFields(PCIE_5_X16);
    expect(fields.map((f) => f.path).sort()).toEqual([
      'bandwidthPerDirectionBytesPerSec',
      'hostBounce.extraLatencySec',
      'hostBounce.hostMemBandwidthBytesPerSec',
      'latencySec',
    ]);
    expect(fields.find((f) => f.path === 'latencySec')?.unit).toBe(UNITS.us);
  });

  it('labels the multi-port link fields: the cap over all ports and the relay latency', () => {
    const fields = editableLinkFields(SPARK_CONNECTX_7);
    expect(fields.find((f) => f.path === 'allPortsCapPerDirectionBytesPerSec')).toMatchObject({
      label: 'Bandwidth cap, all ports',
      unit: UNITS.gbps,
    });
    expect(fields.find((f) => f.path === 'forwardingLatencySec')).toMatchObject({
      label: 'Relay latency per hop',
      unit: UNITS.us,
    });
  });
});

describe('network: NICs per machine and in-network reduction', () => {
  const h100 = (over: Partial<HardwareChoice> = {}) =>
    hw({ deviceId: 'h100-sxm', devicesPerNode: 8, nodes: 2, ...over });

  it('offers the divisors of the GPUs per machine and 2 and 4 per GPU, only between switched multi-GPU machines', () => {
    expect(nicOptions(h100())).toEqual([1, 2, 4, 8, 16, 32]);
    expect(nicOptions(h100({ devicesPerNode: 4 }))).toEqual([1, 2, 4, 8, 16]);
    expect(nicOptions(h100({ nodes: 1 }))).toEqual([]);
    expect(nicOptions(h100({ devicesPerNode: 1 }))).toEqual([]);
    expect(nicOptions(hw({ deviceId: 'dgx-spark', nodes: 4 }))).toEqual([]);
  });

  it('labels a count with its share per GPU', () => {
    expect(nicLabel(4, 8)).toBe('4 (1 per 2 GPUs)');
    expect(nicLabel(1, 8)).toBe('1 (1 per 8 GPUs)');
    expect(nicLabel(8, 8)).toBe('8 (1 per GPU)');
    expect(nicLabel(16, 8)).toBe('16 (2 per GPU)');
  });

  it('normalizes away the default and counts no longer offered; the topology carries the rest', () => {
    expect(normalizeNics(h100({ nicsPerNode: 8 })).nicsPerNode).toBeUndefined();
    expect(normalizeNics(h100({ nicsPerNode: 3 })).nicsPerNode).toBeUndefined();
    expect(normalizeNics(h100({ nicsPerNode: 4 })).nicsPerNode).toBe(4);
    const four = resolveHardware(h100({ nicsPerNode: 4 })).topology;
    expect(four.nicsPerNode).toBe(4);
    expect(four.id).toMatch(/-nics4$/);
    expect(validateTopology(four)).toEqual([]);
    // The default is exactly the topology from before NIC counts.
    expect(resolveHardware(h100({ nicsPerNode: 8 })).topology).toEqual(
      resolveHardware(h100()).topology,
    );
    expect(resolveHardware(h100()).topology.nicsPerNode).toBeUndefined();
  });

  it('lists the in-network reductions of the links in use: NVLS on NVLink 4, SHARP on InfiniBand', () => {
    expect(inNetworkReductionsOf(resolveHardware(h100()))).toEqual(['nvls', 'sharp']);
    expect(inNetworkReductionsOf(resolveHardware(h100({ nodes: 1 })))).toEqual(['nvls']);
    expect(inNetworkReductionsOf(resolveHardware(h100({ devicesPerNode: 1 })))).toEqual(['sharp']);
    // A100's NVSwitch has none; its HDR network has SHARP.
    expect(
      inNetworkReductionsOf(
        resolveHardware(hw({ deviceId: 'a100-sxm4-80gb', devicesPerNode: 8, nodes: 1 })),
      ),
    ).toEqual([]);
    expect(
      inNetworkReductionsOf(
        resolveHardware(hw({ deviceId: 'a100-sxm4-80gb', devicesPerNode: 8, nodes: 2 })),
      ),
    ).toEqual(['sharp']);
    // One RTX 5090 and a DGX Spark cluster: none.
    expect(inNetworkReductionsOf(resolveHardware(SINGLE_DEVICE_HW))).toEqual([]);
    expect(inNetworkReductionsOf(resolveHardware(hw({ deviceId: 'dgx-spark', nodes: 4 })))).toEqual(
      [],
    );
  });
});
