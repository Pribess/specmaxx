import { describe, expect, it } from 'vitest';
import { simulate } from '../../sim/engine/simulate';
import { DEVICE_PRESETS, SYSTEM_PRESETS } from '../../sim/hw/presets/index';
import { PCIE_4_X16, PCIE_5_X16, SPARK_CONNECTX_7, TB5_RDMA, USB4_HALO_RDMA, NVLINK_4, XGMI_MI300X_8_GPU } from '../../sim/hw/presets/links';
import { validateDevice, validateLink, validateTopology } from '../../sim/hw/validate';
import { REF_DENSE_8B } from '../../sim/models/reference';
import { SINGLE_DEVICE_HW, type HardwareChoice } from './appState';
import {
  EDITED_REASON, UNITS, applyEdits, defaultLinks, editableFields, editableLinkFields, findDevice, fromDisplay,
  matchingSystem, resolveDevice, resolveHardware, systemChoice, toDisplay, topologyPresetsFor,
} from './hardware';
import { REFERENCE_QUANT_DEFAULTS } from './quant';

const hw = (over: Partial<HardwareChoice>): HardwareChoice => ({ ...SINGLE_DEVICE_HW, ...over });

describe('defaultLinks and topologies', () => {
  it.each(DEVICE_PRESETS.map((d) => [d.id, d] as const))('%s: default links and node topologies validate', (_id, device) => {
    const links = defaultLinks(device);
    if (links.intraNode !== undefined) expect(validateLink(links.intraNode)).toEqual([]);
    expect(validateLink(links.interNode)).toEqual([]);
    for (const [devicesPerNode, nodes] of [[1, 1], [2, 1], [1, 2], [4, 2]]) {
      const r = resolveHardware(hw({ deviceId: device.id, devicesPerNode, nodes }));
      expect(validateTopology(r.topology)).toEqual([]);
      expect(r.topology.interNode === undefined).toBe(nodes === 1);
      if (devicesPerNode === 1) expect(r.topology.intraNode).toBeUndefined();
    }
  });

  it('picks the expected defaults', () => {
    expect(defaultLinks(findDevice('rtx-5090')!).intraNode).toBe(PCIE_5_X16);
    expect(defaultLinks(findDevice('rtx-3090')!).intraNode).toBe(PCIE_4_X16);
    expect(defaultLinks(findDevice('h100-sxm')!).intraNode).toBe(NVLINK_4);
    expect(defaultLinks(findDevice('instinct-mi300x')!).intraNode).toBe(XGMI_MI300X_8_GPU);
    expect(defaultLinks(findDevice('apple-m3-ultra-60c')!)).toEqual({ interNode: TB5_RDMA });
    expect(defaultLinks(findDevice('ryzen-ai-max-plus-395')!).interNode).toBe(USB4_HALO_RDMA);
    expect(defaultLinks(findDevice('dgx-spark')!).interNode).toBe(SPARK_CONNECTX_7);
  });

  it('lists the named topologies of a device as quick picks', () => {
    expect(topologyPresetsFor(findDevice('rtx-5090')!).map((t) => t.devicesPerNode)).toEqual([2, 4]);
    expect(topologyPresetsFor(findDevice('h100-sxm')!).length).toBe(2);
    expect(topologyPresetsFor(findDevice('instinct-mi300x')!).map((t) => t.id)).toEqual(['mi300x-xgmi-8']);
    // The R9700 has a named 4-card workstation now, so it gets a quick pick.
    expect(topologyPresetsFor(findDevice('radeon-ai-pro-r9700')!).map((t) => t.devicesPerNode)).toEqual([4]);
    expect(topologyPresetsFor(findDevice('dgx-spark')!).map((t) => t.nodes)).toEqual([2, 4]);
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

describe('multi-device setups', () => {
  it.each(SYSTEM_PRESETS.map((p) => [p.id, p] as const))('%s: round-trips through systemChoice and resolves to its topology', (_id, p) => {
    const choice = systemChoice(p);
    expect(matchingSystem(choice)).toBe(p);
    const r = resolveHardware(choice);
    expect(validateTopology(r.topology)).toEqual([]);
    expect(r.topology.devicesPerNode).toBe(p.topology.devicesPerNode);
    expect(r.topology.nodes).toBe(p.topology.nodes);
    expect(r.topology.intraNode).toBe(p.topology.intraNode);
    expect(r.topology.interNode).toBe(p.topology.interNode);
    expect(r.memCapacityBytes).toBe(p.memCapacityBytes ?? p.device.memory.capacityOptionsBytes[0]);
  });

  it('RTX 5090 ×2 (PCIe) sets the device, counts and the PCIe link', () => {
    const p = SYSTEM_PRESETS.find((s) => s.id === 'rtx-5090-x2-pcie5')!;
    expect(systemChoice(p)).toEqual({
      ...SINGLE_DEVICE_HW,
      deviceId: 'rtx-5090',
      devicesPerNode: 2,
      intraLinkId: 'pcie-5-0-x16-host-bounce',
    });
  });

  it('M5 Ultra 256GB picks the 256GB option', () => {
    const p = SYSTEM_PRESETS.find((s) => s.id === 'm5-ultra-80c-256gb-x2-tb5')!;
    expect(systemChoice(p).capacityIndex).toBe(1);
  });

  it('a default-link choice with the same counts matches; any change matches nothing', () => {
    const p = SYSTEM_PRESETS.find((s) => s.id === 'rtx-5090-x2-pcie5')!;
    expect(matchingSystem(hw({ devicesPerNode: 2 }))).toBe(p);
    expect(matchingSystem(SINGLE_DEVICE_HW)).toBeUndefined();
    const choice = systemChoice(p);
    expect(matchingSystem({ ...choice, devicesPerNode: 3 })).toBeUndefined();
    expect(matchingSystem({ ...choice, intraLinkId: 'pcie-5-0-x8-host-bounce' })).toBeUndefined();
    expect(matchingSystem({ ...choice, deviceEdits: { 'memory.bandwidthBytesPerSec': 2e12 } })).toBeUndefined();
    expect(matchingSystem({ ...choice, unlocks: { 'pcie-p2p': 'direct' } })).toBeUndefined();
  });
});

describe('spec edits', () => {
  it('apply after the unlocks, rename the device and keep it valid', () => {
    const r = resolveDevice({ deviceId: 'rtx-4090', unlocks: { 'vram-mod': '48gb' }, deviceEdits: { 'memory.capacityOptionsBytes.0': 40e9 } });
    expect(r.unlocked.memory.capacityOptionsBytes).toEqual([48e9]);
    expect(r.device.memory.capacityOptionsBytes).toEqual([40e9]);
    expect(r.device.id).toBe('rtx-4090-custom');
    expect(r.device.name).toBe(`${r.base.name} (edited)`);
    expect(r.edited).toBe(true);
    expect(r.device.estimated).toContainEqual({ field: 'memory.capacityOptionsBytes.0', reason: EDITED_REASON });
    expect(validateDevice(r.device)).toEqual([]);
    expect(r.base.memory.capacityOptionsBytes).toEqual([24e9]); // presets are never mutated
  });

  it('ignore unknown paths and return the spec itself without edits', () => {
    const device = findDevice('rtx-5090')!;
    expect(applyEdits(device, {}).spec).toBe(device);
    expect(applyEdits(device, { 'memory.nope': 1 })).toEqual({ spec: device, unknownPaths: ['memory.nope'] });
    expect(resolveDevice({ deviceId: 'rtx-5090', unlocks: {}, deviceEdits: {} }).edited).toBe(false);
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
    const r = resolveHardware(hw({ devicesPerNode: 2, intraEdits: { bandwidthPerDirectionBytesPerSec: 10e9 } }));
    expect(r.topology.intraNode?.bandwidthPerDirectionBytesPerSec).toBe(10e9);
    expect(r.topology.intraNode?.name).toMatch(/\(edited\)$/);
    expect(r.edited).toBe(true);
    expect(validateTopology(r.topology)).toEqual([]);
  });
});

describe('editable fields', () => {
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
    const bf16 = fields.find((f) => f.group === 'matmul' && f.label === 'bf16, fp32 accumulate, dense');
    expect(bf16?.unit).toBe(UNITS.tflops);
    expect(fields.some((f) => f.group === 'vector')).toBe(true);
    expect(fields.some((f) => f.group === 'tiling')).toBe(true);
    expect(editableFields(findDevice('apple-m5-max-40c')!).some((f) => f.label.includes(': '))).toBe(true);
  });

  it('converts display units exactly for round numbers', () => {
    expect(fromDisplay(1792, UNITS.gbps)).toBe(1.792e12);
    expect(fromDisplay(0.5, UNITS.us)).toBeCloseTo(5e-7, 15);
    expect(toDisplay(209.5e12, UNITS.tflops)).toBeCloseTo(209.5);
  });

  it('labels link fields', () => {
    const fields = editableLinkFields(PCIE_5_X16);
    expect(fields.map((f) => f.path).sort()).toEqual(
      ['bandwidthPerDirectionBytesPerSec', 'hostBounce.extraLatencySec', 'hostBounce.hostMemBandwidthBytesPerSec', 'latencySec'],
    );
    expect(fields.find((f) => f.path === 'latencySec')?.unit).toBe(UNITS.us);
  });
});
