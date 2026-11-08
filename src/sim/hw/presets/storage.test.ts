import { describe, expect, it } from 'vitest';
import type { DeviceSpec, StorageTierSpec } from '../types';
import {
  isPathCovered,
  numericLeafPaths,
  validateHost,
  validateLink,
  validateStorageTier,
} from '../validate';
import { DEVICE_PRESETS } from './index';
import {
  HOST_PCIE_LINKS,
  PCIE_4_X16,
  PCIE_4_X8,
  PCIE_5_X16,
  PCIE_5_X16_DIRECT,
  PCIE_5_X8,
  hostPcieLink,
  pcieBandwidthPerDirectionBytesPerSec,
} from './links';
import {
  APPLE_SSD_M3_M4_MAX_1TB,
  DDR5_4800_12CH,
  DDR5_5600_2CH,
  DDR5_6000_12CH,
  DDR5_6000_2CH,
  DDR5_6400_2CH,
  DDR5_6400_8CH,
  DEFAULT_HOST_RAM,
  DEFAULT_SSD,
  DGX_SPARK_SSD_4TB,
  HOST_RAM_PRESETS,
  NVME_990_PRO_2TB,
  NVME_9100_PRO_4TB,
  NVME_GEN4_X4_GENERIC,
  NVME_SN850X_2TB,
  NVME_T705_2TB,
  SSD_PRESETS,
  ddr5BandwidthBytesPerSec,
} from './storage';

const TIERS: StorageTierSpec[] = [...HOST_RAM_PRESETS, ...SSD_PRESETS];

/** Every field path a source or an estimate covers. */
function covers(t: { sources: { fields: string[] }[]; estimated: { field: string }[] }): string[] {
  return [...t.sources.flatMap((s) => s.fields), ...t.estimated.map((e) => e.field)];
}

describe('storage-tier presets', () => {
  it('lists the 6 host-RAM and 8 SSD rows of design §2.2, with unique ids', () => {
    expect(HOST_RAM_PRESETS.map((t) => t.id)).toEqual([
      'ddr5-5600-2ch',
      'ddr5-6000-2ch',
      'ddr5-6400-2ch',
      'ddr5-6400-8ch',
      'ddr5-4800-12ch',
      'ddr5-6000-12ch',
    ]);
    expect(SSD_PRESETS.map((t) => t.id)).toEqual([
      'nvme-990-pro-2tb',
      'nvme-sn850x-2tb',
      'nvme-9100-pro-4tb',
      'nvme-t705-2tb',
      'nvme-gen4-x4-generic',
      'apple-ssd-m3-m4-max-1tb',
      'apple-ssd-m5-pro-max',
      'dgx-spark-ssd-4tb',
    ]);
    expect(new Set(TIERS.map((t) => t.id)).size).toBe(TIERS.length);
  });

  it.each(TIERS.map((t) => [t.id, t] as const))('%s passes validateStorageTier', (_id, t) => {
    expect(validateStorageTier(t)).toEqual([]);
  });

  it.each(TIERS.map((t) => [t.id, t] as const))(
    '%s: every numeric field has a source or an estimate, each with a URL or reason',
    (_id, t) => {
      const c = covers(t);
      for (const leaf of numericLeafPaths(t))
        expect(isPathCovered(leaf.path, c), `${t.id}: ${leaf.path}`).toBe(true);
      for (const s of t.sources) {
        expect(s.url).toMatch(/^https:\/\//);
        expect(s.fields.length).toBeGreaterThan(0);
      }
      for (const e of t.estimated) expect(e.reason.trim()).not.toBe('');
    },
  );

  it('defaults: DDR5-6000 dual channel and the Samsung 990 PRO 2 TB', () => {
    expect(DEFAULT_HOST_RAM).toBe(DDR5_6000_2CH);
    expect(DEFAULT_SSD).toBe(NVME_990_PRO_2TB);
  });

  it('host-RAM bandwidth = MT/s × 8 B × channels, as the vendor and design figures', () => {
    expect(ddr5BandwidthBytesPerSec(6000, 2)).toBe(96e9);
    expect(DDR5_5600_2CH.bandwidthBytesPerSec).toBe(89.6e9);
    expect(DDR5_6000_2CH.bandwidthBytesPerSec).toBe(96e9);
    expect(DDR5_6400_2CH.bandwidthBytesPerSec).toBe(102.4e9);
    expect(DDR5_6400_8CH.bandwidthBytesPerSec).toBe(409.6e9);
    expect(DDR5_4800_12CH.bandwidthBytesPerSec).toBe(460.8e9); // AMD: "Per Socket Mem BW 460.8 GB/s"
    expect(DDR5_6000_12CH.bandwidthBytesPerSec).toBe(576e9);
    // The default host RAM is the same 96 GB/s the host-bounce PCIe links already assume for staging.
    expect(DDR5_6000_2CH.bandwidthBytesPerSec).toBe(
      PCIE_5_X16.hostBounce?.hostMemBandwidthBytesPerSec,
    );
    for (const t of HOST_RAM_PRESETS) {
      expect(t.readGranularityBytes).toBe(64);
      expect(t.maxOutstanding).toBe(64);
    }
  });

  it('SSD figures as their datasheets quote them', () => {
    // 990 PRO 2TB: 7,450 MB/s; QD1 22K IOPS ⇒ 45.45 µs; 1,400K IOPS at QD32 × 16.
    expect(NVME_990_PRO_2TB.seqReadBandwidthBytesPerSec).toBe(7.45e9);
    expect(NVME_990_PRO_2TB.randomReadLatencySec).toBe(1 / 22_000); // 45.45 µs
    expect(NVME_990_PRO_2TB.randomRead).toEqual({ iops: 1.4e6, queueDepth: 32, threads: 16 });
    expect(NVME_990_PRO_2TB.maxOutstanding).toBe(512);
    expect(NVME_SN850X_2TB.seqReadBandwidthBytesPerSec).toBe(7.3e9);
    expect(NVME_SN850X_2TB.randomRead?.iops).toBe(1.2e6);
    // 9100 PRO 4TB: 14,800 MB/s; 2,200K IOPS at QD256 × 32 ⇒ 8192 outstanding.
    expect(NVME_9100_PRO_4TB.seqReadBandwidthBytesPerSec).toBe(14.8e9);
    expect(NVME_9100_PRO_4TB.maxOutstanding).toBe(256 * 32);
    expect(NVME_T705_2TB.seqReadBandwidthBytesPerSec).toBe(14.5e9);
    expect(NVME_T705_2TB.randomRead?.iops).toBe(1.55e6);
    // Apple 1 TB: 4096 B / 60 MB/s = 68.3 µs.
    expect(APPLE_SSD_M3_M4_MAX_1TB.randomReadLatencySec).toBeCloseTo(68.27e-6, 8);
    // DGX Spark: 11.4 GiB/s = 12.24e9 B/s.
    expect(DGX_SPARK_SSD_4TB.seqReadBandwidthBytesPerSec).toBeCloseTo(12.2407e9, -5);
    // Generic Gen4 x4: the slot ceiling 16e9 × 4 × 128/130 / 8 = 7.877e9 B/s binds above the 7.0e9 drive rate.
    expect(NVME_GEN4_X4_GENERIC.linkBandwidthBytesPerSec).toBeCloseTo(7.8769e9, -5);
    for (const t of SSD_PRESETS) expect(t.readGranularityBytes).toBe(4096);
  });

  it('every estimate on DGX Spark random reads is marked (no published figure)', () => {
    const est = DGX_SPARK_SSD_4TB.estimated.map((e) => e.field);
    expect(est).toEqual(
      expect.arrayContaining(['randomReadLatencySec', 'randomRead', 'maxOutstanding']),
    );
  });
});

describe('hostPcieLink', () => {
  it('Gen5 x16 bandwidth equals PCIE_5_X16 exactly (63,015,384,615.38… B/s)', () => {
    const link = hostPcieLink(5, 16);
    expect(link.bandwidthPerDirectionBytesPerSec).toBe(PCIE_5_X16.bandwidthPerDirectionBytesPerSec);
    expect(link.bandwidthPerDirectionBytesPerSec).toBeCloseTo(63_015_384_615.38, 1);
    expect(hostPcieLink(4, 16).bandwidthPerDirectionBytesPerSec).toBe(
      PCIE_4_X16.bandwidthPerDirectionBytesPerSec,
    );
    expect(hostPcieLink(4, 8).bandwidthPerDirectionBytesPerSec).toBe(
      PCIE_4_X8.bandwidthPerDirectionBytesPerSec,
    );
    expect(hostPcieLink(5, 8).bandwidthPerDirectionBytesPerSec).toBe(
      PCIE_5_X8.bandwidthPerDirectionBytesPerSec,
    );
  });

  it('the existing device-to-device PCIe links keep their exact bandwidths', () => {
    // 32e9 × 16 × 128/130 / 8 and 16e9 × 16 × 128/130 / 8, in the order the old private formula computed them.
    expect(PCIE_5_X16.bandwidthPerDirectionBytesPerSec).toBe((32e9 * 16 * (128 / 130)) / 8);
    expect(PCIE_5_X16_DIRECT.bandwidthPerDirectionBytesPerSec).toBe((32e9 * 16 * (128 / 130)) / 8);
    expect(PCIE_4_X16.bandwidthPerDirectionBytesPerSec).toBe((16e9 * 16 * (128 / 130)) / 8);
  });

  it('Gen3 is half Gen4 and x4 a quarter of x16', () => {
    expect(pcieBandwidthPerDirectionBytesPerSec(3, 16)).toBe(
      pcieBandwidthPerDirectionBytesPerSec(4, 16) / 2,
    );
    expect(pcieBandwidthPerDirectionBytesPerSec(4, 4)).toBe(
      pcieBandwidthPerDirectionBytesPerSec(4, 16) / 4,
    );
  });

  it('is a direct, cited PCIe link with an estimated 1 µs α', () => {
    const link = hostPcieLink(3, 4);
    expect(link).toMatchObject({
      kind: 'pcie',
      name: 'PCIe 3.0 x4 (host link)',
      latencySec: 1e-6,
      p2p: 'direct',
    });
    expect(link.hostBounce).toBeUndefined();
    expect(link.estimated.map((e) => e.field)).toEqual(['latencySec']);
    expect(validateLink(link)).toEqual([]);
  });

  it('HOST_PCIE_LINKS lists all 9 generation × width combinations, each valid', () => {
    expect(HOST_PCIE_LINKS.map((l) => l.name)).toEqual([
      'PCIe 5.0 x16 (host link)',
      'PCIe 5.0 x8 (host link)',
      'PCIe 5.0 x4 (host link)',
      'PCIe 4.0 x16 (host link)',
      'PCIe 4.0 x8 (host link)',
      'PCIe 4.0 x4 (host link)',
      'PCIe 3.0 x16 (host link)',
      'PCIe 3.0 x8 (host link)',
      'PCIe 3.0 x4 (host link)',
    ]);
    for (const l of HOST_PCIE_LINKS) expect(validateLink(l)).toEqual([]);
  });
});

describe('DeviceSpec.hostInterface', () => {
  const byId = (id: string): DeviceSpec => DEVICE_PRESETS.find((d) => d.id === id)!;

  it('is set, cited, on the PCIe cards; absent on SXM/OAM modules and SoCs', () => {
    const expected: Record<string, [number, number]> = {
      'rtx-5090': [5, 16],
      'rtx-4090': [4, 16],
      'rtx-3090': [4, 16],
      'radeon-ai-pro-r9700': [5, 16],
      'radeon-rx-9070-xt': [5, 16],
      'arc-pro-b70': [5, 16],
      'arc-pro-b65': [5, 16],
      'arc-pro-b60': [5, 8], // x16 connector, x8 electrical
      'arc-b580': [4, 8],
    };
    for (const d of DEVICE_PRESETS) {
      const want = expected[d.id];
      if (want === undefined) {
        expect(d.hostInterface, d.id).toBeUndefined();
        continue;
      }
      expect(d.hostInterface, d.id).toEqual({ generation: want[0], lanes: want[1] });
      const c = covers(d);
      expect(isPathCovered('hostInterface.generation', c), d.id).toBe(true);
      expect(isPathCovered('hostInterface.lanes', c), d.id).toBe(true);
      // Cited by a source, not an estimate.
      const cited = d.sources.flatMap((s) => s.fields);
      expect(
        isPathCovered('hostInterface.generation', cited) &&
          isPathCovered('hostInterface.lanes', cited),
        d.id,
      ).toBe(true);
    }
    expect(byId('rtx-5090').memory.kind).toBe('dedicated');
  });

  it('the default host link of the RTX 5090 is PCIe 5.0 x16, and validateHost accepts the default host', () => {
    const { generation, lanes } = byId('rtx-5090').hostInterface!;
    const link = hostPcieLink(generation, lanes);
    expect(link.bandwidthPerDirectionBytesPerSec).toBe(PCIE_5_X16.bandwidthPerDirectionBytesPerSec);
    expect(validateHost({ link, ram: DEFAULT_HOST_RAM, ssd: DEFAULT_SSD })).toEqual([]);
  });
});
