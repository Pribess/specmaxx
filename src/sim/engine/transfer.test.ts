import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import { hostPcieLink } from '../hw/presets/links';
import { DDR5_6000_2CH, DGX_SPARK_SSD_4TB, NVME_990_PRO_2TB } from '../hw/presets/storage';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { NvmeSpec } from '../hw/types';
import { REF_DENSE_8B } from '../models/reference';
import { resolveHost } from './host';
import { simulate } from './simulate';
import {
  expertHitsMisses,
  hitRatio,
  nStream,
  stageNStream,
  transferTime,
  uniqueExperts,
  type TransferPath,
} from './transfer';
import { validateInput } from './validate';
import type { SimInput } from './types';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** PCIe 5.0 x16 per direction: 32e9·16·128/130/8 = 63,015,384,615.38 B/s; α 1 µs. */
const PCIE5 = hostPcieLink(5, 16);
const B_LINK = 63_015_384_615.38461;
/** The design doc's 990 PRO numbers (QD1 latency rounded to 45 µs; the preset carries 1/22,000 s). */
const SSD_45US: NvmeSpec = { ...NVME_990_PRO_2TB, randomReadLatencySec: 45e-6 };
const LINK = { latencySec: PCIE5.latencySec, bandwidthPerDirectionBytesPerSec: PCIE5.bandwidthPerDirectionBytesPerSec };

function ramPath(over: Partial<TransferPath> = {}): TransferPath {
  return { tier: DDR5_6000_2CH, link: LINK, queueDepth: 64, nStream: 1, ...over };
}
function ssdPath(over: Partial<TransferPath> = {}): TransferPath {
  return { tier: SSD_45US, link: LINK, hostStagingBandwidthBytesPerSec: 96e9, queueDepth: 512, nStream: 1, ...over };
}

describe('U-1 uniqueExperts U(T) = E·(1 − (1 − k/E)^T)', () => {
  it('U(1) = k; E 128 k 8: U(2) 15.5, U(4) 29.123046875, U(8) 51.6199…, U(16) 82.4225…, U(32) 111.771…, U(8192) 128', () => {
    expect(uniqueExperts(128, 8, 1)).toBe(8);
    expect(uniqueExperts(64, 6, 1)).toBeCloseTo(6, 12);
    expectClose(uniqueExperts(128, 8, 2), 15.5);
    expectClose(uniqueExperts(128, 8, 4), 29.123046875);
    expectClose(uniqueExperts(128, 8, 8), 51.61990735, 1e-9);
    expectClose(uniqueExperts(128, 8, 16), 82.4225113, 1e-8);
    expectClose(uniqueExperts(128, 8, 32), 111.771035, 1e-8);
    expect(uniqueExperts(128, 8, 8192)).toBe(128);
  });
});

describe('U-2 hits and misses: hits = clamp(U·h, max(0, U + C − E), min(U, C)), E 128', () => {
  it.each([
    // [U, h, f, hits, M]
    [8, 0.5, 0.5, 4, 4],
    [128, 0.87, 0.5, 64, 64],
    [128, 0.1, 0.5, 64, 64], // lower bound U + C − E = 64
    [8, 0.1, 0.5, 0.8, 7.2], // custom below f allowed
    [8, 0, 0, 0, 8], // f 0 ⇒ M = U
    [8, 0.3, 1, 8, 0], // f 1 ⇒ M = 0 whatever h
    [100, 0.2, 0.5, 36, 64], // lo = 100 + 64 − 128 = 36
  ])('U %d, h %d, f %d → hits %d, M %d', (U, h, f, hits, M) => {
    const r = expertHitsMisses(U, h, f, 128);
    expectClose(r.hits, hits);
    expect(Math.abs(r.misses - M)).toBeLessThanOrEqual(1e-12);
  });
  it('f 1 gives exactly 0 misses for a non-integer U (no rounding residue)', () => {
    expect(expertHitsMisses(uniqueExperts(128, 8, 3), 0.3, 1, 128).misses).toBe(0);
  });
});

describe('U-3 hit-ratio models and their validation', () => {
  it('at f .25: uniform .25, skewed_classic .25^.55 = .46651…, skewed_fine_grained .25^.2 = .75785…', () => {
    expect(hitRatio({ kind: 'uniform' }, 0.25)).toBe(0.25);
    expectClose(hitRatio({ kind: 'skewed_classic' }, 0.25), 0.4665164957684037);
    expectClose(hitRatio({ kind: 'skewed_fine_grained' }, 0.25), 0.757858283255199);
    expect(hitRatio({ kind: 'custom', h: 0.4 }, 0.25)).toBe(0.4);
  });
  const moeInput = (f: number, h: number): SimInput => ({
    model: { ...REF_DENSE_8B, layers: REF_DENSE_8B.layers },
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 128, decodeContextLen: 128 },
    hardware: { device: RTX_5090, topology: { ...SYNTH_CLUSTER, devicesPerNode: 1, nodes: 1 } },
    deployment: { offload: { routedExperts: { tier: 'host_ram', residentFraction: f, hitRatio: { kind: 'custom', h } } } },
  });
  it('f 0 with custom h .2 is rejected ("no experts are resident, h must be 0")', () => {
    expect(validateInput(moeInput(0, 0.2)).some((e) => e.includes('no experts are resident, h must be 0'))).toBe(true);
    expect(validateInput(moeInput(1, 0.2)).some((e) => e.includes('h must be 0'))).toBe(false);
  });
});

describe('U-4 transferTime: t = α_tier + α_link + max(Little, IOPS/n, tier/n, link, host staging)', () => {
  it('host RAM → PCIe5, reads [{8, 2,654,208}]: α 1.07 µs, link-bound service 21,233,664 / 63.015e9 = 3.36960e-4', () => {
    const t = transferTime([{ count: 8, bytesEach: 2_654_208 }], 21_233_664, ramPath());
    expectClose(t.latencySec, 1.07e-6);
    expectClose(t.linkSec, 21_233_664 / B_LINK);
    expectClose(t.tierSec, 21_233_664 / 96e9); // 2.2120e-4
    expectClose(t.littleSec, (7 * 70e-9) / 64); // 7.66e-9
    expect(t.iopsSec).toBeUndefined();
    expect(t.hostSec).toBeUndefined();
    expectClose(t.serviceSec, 3.3696e-4, 1e-5);
    expectClose(t.sec, 1.07e-6 + 21_233_664 / B_LINK);
  });
  it('NVMe → host → device, reads [{16, 320}], QD 512, n 1: bytes_tier 65,536, α 46 µs, IOPS-bound service 16/1.4e6', () => {
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, ssdPath());
    expect(t.tierBytes).toBe(65_536);
    expect(t.usefulBytes).toBe(5_120);
    expectClose(t.latencySec, 46e-6);
    expectClose(t.littleSec, (15 * 45e-6) / 512); // 1.318e-6
    expectClose(t.iopsSec ?? 0, 16 / 1.4e6); // 1.1429e-5
    expectClose(t.tierSec, 65_536 / 7.45e9); // 8.797e-6
    expectClose(t.hostSec ?? 0, 65_536 / 48e9); // 1.365e-6
    expectClose(t.linkSec, 5_120 / B_LINK); // 8.1e-8
    expectClose(t.serviceSec, 16 / 1.4e6);
    expectClose(t.sec, 46e-6 + 16 / 1.4e6); // 57.429 µs
  });
  it('the real 990 PRO preset (QD1 1/22,000 s = 45.45 µs): α = 1/22,000 + 1 µs', () => {
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, ssdPath({ tier: NVME_990_PRO_2TB }));
    expectClose(t.latencySec, 1 / 22_000 + 1e-6);
  });
  it('an NvmeSpec without randomRead (Spark-like, maxOutstanding 512, 50 µs): no IOPS term, and resolveHost warns', () => {
    const ssd: NvmeSpec = { ...DGX_SPARK_SSD_4TB, seqReadBandwidthBytesPerSec: 12.2e9 };
    delete ssd.randomRead;
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, ssdPath({ tier: ssd }));
    expect(t.iopsSec).toBeUndefined();
    expectClose(t.littleSec, (15 * 50e-6) / 512);
    expectClose(t.serviceSec, Math.max((15 * 50e-6) / 512, 65_536 / 12.2e9, 65_536 / 48e9, 5_120 / B_LINK));
    const host = resolveHost({ device: SYNTH_GPU, topology: SYNTH_CLUSTER, host: { link: PCIE5, ssd } }, SYNTH_GPU, new Set(['nvme']));
    expect(host.warnings).toContain(`offload: no random-read IOPS for ${ssd.id}; request rate from latency × queue depth`);
  });
  it('NVMe on unified memory: no link (α_link 0, no link term), no host staging', () => {
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, { tier: SSD_45US, queueDepth: 512, nStream: 1 });
    expectClose(t.latencySec, 45e-6);
    expect(t.linkSec).toBe(0);
    expect(t.hostSec).toBeUndefined();
  });
});

describe('U-11 queue depth: Qwen rows [{16, 320}] from the 990 Pro', () => {
  const rows = [{ count: 16, bytesEach: 320 }];
  it('QD 512 ⇒ 57.429 µs; QD 1 ⇒ 46 + 15·45 = 721 µs; QD 4 ⇒ 46 + max(168.75, 11.43, …) = 214.75 µs', () => {
    expectClose(transferTime(rows, 5_120, ssdPath({ queueDepth: 512 })).sec, 46e-6 + 16 / 1.4e6);
    expectClose(transferTime(rows, 5_120, ssdPath({ queueDepth: 1 })).sec, 721e-6);
    expectClose(transferTime(rows, 5_120, ssdPath({ queueDepth: 4 })).sec, 214.75e-6);
  });
});

describe('U-12 a shared SSD: four genuine consumers (tp 4, one node, every stage offloads)', () => {
  it('Qwen rows [{16, 320}]: service = max(1.318e-6, 16/(1.4e6/4), 65,536/(7.45e9/4), 65,536/(96e9/8)) ⇒ 91.71 µs', () => {
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, ssdPath({ nStream: 4 }));
    expectClose(t.iopsSec ?? 0, 16 / (1.4e6 / 4)); // 4.571e-5
    expectClose(t.tierSec, 65_536 / (7.45e9 / 4)); // 3.519e-5
    expectClose(t.hostSec ?? 0, 65_536 / (96e9 / 8)); // 5.46e-6
    expectClose(t.sec, 46e-6 + 16 / (1.4e6 / 4)); // 91.71 µs (vs 57.43 µs alone)
  });
  it('experts [{8, 2,654,208}] from the SSD: alone 46 µs + 2.8502 ms, with n 4 46 µs + 11.4006 ms', () => {
    const reads = [{ count: 8, bytesEach: 2_654_208 }];
    expectClose(transferTime(reads, 21_233_664, ssdPath()).sec, 46e-6 + 21_233_664 / 7.45e9);
    expectClose(transferTime(reads, 21_233_664, ssdPath({ nStream: 4 })).sec, 46e-6 + 21_233_664 / (7.45e9 / 4));
    expectClose(21_233_664 / 7.45e9, 2.8502e-3, 1e-4);
  });
});

describe('U-17 n_stream (one node, tp = dp = 1 unless said)', () => {
  const pp4 = { tp: 1, dp: 1, pp: 4 };
  const all = [true, true, true, true];
  it.each([
    ['(a) b 1, pp 4, none, stage 0 only', pp4, [true, false, false, false], 'none', 1, 1],
    ['(b) same with ideal', pp4, [true, false, false, false], 'ideal', 1, 1],
    ['(c) pp 4, b 1, ideal, every stage', pp4, all, 'ideal', 1, 4],
    ['(d) pp 4, b 1, none, every stage', pp4, all, 'none', 1, 1],
    ['(e) pp 4, b 2 (m 2), none, every stage', pp4, all, 'none', 2, 2],
    ['(f) pp 4, b 2, none, stages {0, 1}', pp4, [true, true, false, false], 'none', 2, 2],
    ['(g) pp 4, b 2, none, stage 0 only', pp4, [true, false, false, false], 'none', 2, 1],
    ['(h) pp 4, b 4 (m 4), none, every stage', pp4, all, 'none', 4, 4],
    ['(i) tp 4, every stage', { tp: 4, dp: 1, pp: 1 }, [true], 'ideal', 1, 4],
  ] as const)('%s ⇒ %d', (_name, parallel, streams, overlap, m, expected) => {
    expect(stageNStream(parallel, 8, streams, 0, overlap, m)).toBe(expected);
  });
  it('the optimistic schedule takes the fewest consumers, but never fewer than the stage itself', () => {
    expect(nStream([4, 1, 2, 0], 1, 2)).toBe(3); // 1 + 2
    expect(nStream([4, 1, 2, 0], 0, 2)).toBe(4); // at least its own 4
  });
  it('(e) end to end: REF_DENSE_8B pp 4, b 2, dense layers on the SSD, offloadOverlap none ⇒ tier term / 2 consumers', () => {
    const run = (offloadOverlap: 'ideal' | 'none') =>
      simulate({
        model: REF_DENSE_8B,
        quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
        parallel: { tp: 1, pp: 4, dp: 1, ep: 1 },
        workload: { batch: 2, promptLen: 128, decodeContextLen: 128 },
        hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER, host: { link: PCIE5, ssd: SSD_45US, ram: DDR5_6000_2CH } },
        options: { offloadOverlap },
        deployment: { offload: { denseLayers: { tier: 'nvme', layers: { from: 0, to: 32 } } } },
      });
    for (const [overlap, n] of [['none', 2], ['ideal', 4]] as const) {
      const fetch = run(overlap).decode.stages[1].layers[0].ops[0];
      expect(fetch.op.kind).toBe('transfer');
      if (fetch.op.kind !== 'transfer') continue;
      const tierBytes = Math.ceil(fetch.op.usefulBytes / 4096) * 4096;
      expectClose(fetch.transferBandwidthSec, tierBytes / (7.45e9 / n));
      expectClose(fetch.transferLatencySec, 46e-6);
      expect(fetch.bound).toBe('transfer');
    }
  });
});
