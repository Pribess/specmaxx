import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH, NVME_990_PRO_2TB, SSD_PRESETS } from '../hw/presets/storage';
import { validateStorageTier } from '../hw/validate';
import { REF_MOE_30B_A3B } from '../models/reference';
import { transferPath } from './host';
import { simulate } from './simulate';
import { transferTime, type TransferPath } from './transfer';
import type { OffloadConfig, SimInput } from './types';
import { validateInput } from './validate';

const sustained = { fraction: 0.75, blockBytes: 1_048_576, queueDepth: 1, threads: 1 };
const ssd = { ...NVME_990_PRO_2TB, randomReadLatencySec: 45e-6, sustainedRead: sustained };
const base: TransferPath = { tier: ssd, queueDepth: 512, nStream: 1 };
const path = { ...base, nvmeSustainedRead: sustained };
const experts = [{ count: 8, bytesEach: 2_654_208 }];
const rows = [{ count: 16, bytesEach: 320 }];
const bytes = 21_233_664;

function input(ssdReadRate?: OffloadConfig['ssdReadRate']): SimInput {
  return {
    model: REF_MOE_30B_A3B,
    quant: {
      weights: { dtype: 'bf16' },
      activations: 'bf16',
      kvCache: { dtype: 'bf16' },
      modules: { routed_experts: { weights: { dtype: 'int4', extraBitsPerElem: 0.5 } } },
    },
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 64, decodeContextLen: 64 },
    hardware: { device: RTX_5090, topology: singleDevice(RTX_5090), host: { ssd } },
    deployment: {
      offload: {
        ssdReadRate,
        routedExperts: { tier: 'nvme', residentFraction: 0, hitRatio: { kind: 'uniform' } },
      },
    },
  };
}

describe('opt-in sustained large NVMe reads', () => {
  it('costs large reads with the measured fraction; applies the slot cap after the fraction', () => {
    expect(transferTime(experts, bytes, path).sec).toBeCloseTo(45e-6 + bytes / (0.75 * 7.45e9), 12);
    const capped = { ...path, tier: { ...ssd, linkBandwidthBytesPerSec: 4e9 } };
    expect(transferTime(experts, bytes, capped).tierSec).toBe(bytes / 4e9);
  });
  it('leaves tiny rows unchanged and sums mixed read-class bandwidth times, sharing each by n_stream', () => {
    expect(transferTime(rows, 5120, path)).toEqual(transferTime(rows, 5120, base));
    const mixed = transferTime([...experts, ...rows], bytes + 5120, { ...path, nStream: 2 });
    expect(mixed.tierSec).toBe(2 * (bytes / (0.75 * 7.45e9) + (16 * 4096) / 7.45e9));
    expect(transferTime(experts, bytes, { ...path, tier: DDR5_6000_2CH })).toEqual(
      transferTime(experts, bytes, { ...base, tier: DDR5_6000_2CH }),
    );
  });
  it('only enables the fraction for an NVMe path in sustained mode', () => {
    const host = { ssd, ram: DDR5_6000_2CH, warnings: [] };
    expect(transferPath('nvme', host, undefined, 1)).toEqual(
      transferPath('nvme', host, undefined, 1, 'datasheet'),
    );
    expect(transferPath('nvme', host, undefined, 1, 'sustained').nvmeSustainedRead).toEqual(
      sustained,
    );
    expect(
      transferPath('host_ram', host, undefined, 1, 'sustained').nvmeSustainedRead,
    ).toBeUndefined();
  });
  it('keeps absent/datasheet whole results byte-identical and derates SSD-bound decode only when opted in', () => {
    const omitted = simulate(input());
    expect(JSON.stringify(simulate(input('datasheet')))).toBe(JSON.stringify(omitted));
    const ratio = simulate(input('sustained')).decode.tokensPerSec / omitted.decode.tokensPerSec;
    expect(ratio).toBeGreaterThan(0.74);
    expect(ratio).toBeLessThan(0.76);
  });
  it('warns and falls back to datasheet when the SSD has no scenario data', () => {
    const i = input('sustained');
    const unmeasured = { ...ssd, sustainedRead: undefined };
    i.hardware.host = { ssd: unmeasured };
    const result = simulate(i);
    expect(result.warnings.some((w) => w.includes('no sustained-read data'))).toBe(true);
    i.deployment!.offload!.ssdReadRate = 'datasheet';
    expect(result.decode).toEqual(simulate(i).decode);
  });
  it('rejects invalid mode, fraction, block size and measurement counts', () => {
    const i = input();
    Object.assign(i.deployment!.offload!, { ssdReadRate: 'other' });
    expect(validateInput(i)).toContain(
      "deployment.offload.ssdReadRate must be 'datasheet' or 'sustained'",
    );
    for (const value of [
      { fraction: 0 },
      { fraction: 1.01 },
      { fraction: NaN },
      { blockBytes: 1024 },
      { queueDepth: 0 },
      { threads: 1.5 },
    ]) {
      expect(
        validateStorageTier({ ...ssd, sustainedRead: { ...sustained, ...value } }).length,
      ).toBeGreaterThan(0);
    }
  });
  it('pins every preset fraction and its large-block condition', () => {
    expect(SSD_PRESETS.map((s) => s.sustainedRead?.fraction)).toEqual([
      0.646, 0.747, 0.72, 0.646, 0.666, 0.51, 0.84, 0.65,
    ]);
    for (const s of SSD_PRESETS)
      expect(s.sustainedRead).toMatchObject({ blockBytes: 1_048_576, queueDepth: 1, threads: 1 });
  });
});
