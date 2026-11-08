import { describe, expect, it } from 'vitest';
import { validateDevice, validateTopology } from '../validate';
import { DEVICE_PRESETS } from './index';
import { SPARK_CONNECTX_7, PCIE_5_X16 } from './links';
import { SYSTEM_PRESETS } from './setups';
import { DGX_SPARK_4, R9700_PCIE_4, TOPOLOGY_PRESETS } from './topologies';

describe('SYSTEM_PRESETS', () => {
  it("lists the user's setups in order, with unique ids", () => {
    expect(SYSTEM_PRESETS.map((p) => p.id)).toEqual([
      'rtx-5090-x2-pcie5',
      'rtx-5090-x4-pcie5',
      'rtx-3090-x2-nvlink',
      'dgx-spark-x2',
      'dgx-spark-x4',
      'm5-ultra-80c-256gb-x2-tb5',
      'strix-halo-128gb-x2-usb4',
      'r9700-x4-pcie5',
    ]);
    expect(new Set(SYSTEM_PRESETS.map((p) => p.name)).size).toBe(SYSTEM_PRESETS.length);
  });

  it.each(SYSTEM_PRESETS.map((p) => [p.id, p] as const))(
    '%s: stock device preset, valid memory option and topology',
    (_id, p) => {
      expect(DEVICE_PRESETS).toContain(p.device);
      expect(validateDevice(p.device)).toEqual([]);
      if (p.memCapacityBytes !== undefined)
        expect(p.device.memory.capacityOptionsBytes).toContain(p.memCapacityBytes);
      expect(validateTopology(p.topology)).toEqual([]);
      expect(TOPOLOGY_PRESETS).toContain(p.topology);
      // The topology is built for this device (its id names it).
      expect(p.topology.id).toContain(p.device.id);
      expect(p.topology.devicesPerNode * p.topology.nodes).toBeGreaterThan(1);
    },
  );

  it('the new topologies: DGX Spark ×4 over ConnectX-7, R9700 ×4 on cited PCIe 5.0 x16', () => {
    expect(DGX_SPARK_4).toMatchObject({ devicesPerNode: 1, nodes: 4, interNode: SPARK_CONNECTX_7 });
    expect(R9700_PCIE_4).toMatchObject({ devicesPerNode: 4, nodes: 1, intraNode: PCIE_5_X16 });
    expect(R9700_PCIE_4.sources[0].url.startsWith('https://www.amd.com/')).toBe(true);
  });
});
