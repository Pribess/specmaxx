import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import { hostPcieLink } from '../hw/presets/links';
import { DDR5_6000_2CH, DGX_SPARK_SSD_4TB, NVME_990_PRO_2TB } from '../hw/presets/storage';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { NvmeSpec } from '../hw/types';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_DENSE_8B, REF_MOE_30B_A3B } from '../models/reference';
import { resolveHost } from './host';
import { simulate } from './simulate';
import {
  expertHitsMisses,
  hitRatio,
  nStream,
  prefetchesFromMultipleTiers,
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

/** PCIe 5.0 x16 per direction: 32e9·16·128/130/8 × 256/280 (TLP payload) = 57,614,065,934.07 B/s; α 1 µs. */
const PCIE5 = hostPcieLink(5, 16);
const B_LINK = 57_614_065_934.06594;
/** The design doc's 990 PRO numbers (QD1 latency rounded to 45 µs; the preset carries 1/22,000 s). */
const SSD_45US: NvmeSpec = { ...NVME_990_PRO_2TB, randomReadLatencySec: 45e-6 };
const LINK = {
  latencySec: PCIE5.latencySec,
  bandwidthPerDirectionBytesPerSec: PCIE5.bandwidthPerDirectionBytesPerSec,
};

function ramPath(over: Partial<TransferPath> = {}): TransferPath {
  return { tier: DDR5_6000_2CH, link: LINK, queueDepth: 64, nStream: 1, ...over };
}
function ssdPath(over: Partial<TransferPath> = {}): TransferPath {
  return {
    tier: SSD_45US,
    link: LINK,
    hostStagingBandwidthBytesPerSec: 96e9,
    queueDepth: 512,
    nStream: 1,
    ...over,
  };
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
    deployment: {
      offload: {
        routedExperts: { tier: 'host_ram', residentFraction: f, hitRatio: { kind: 'custom', h } },
      },
    },
  });
  it('f 0 with custom h .2 is rejected ("no experts are resident, h must be 0")', () => {
    expect(
      validateInput(moeInput(0, 0.2)).some((e) =>
        e.includes('no experts are resident, h must be 0'),
      ),
    ).toBe(true);
    expect(validateInput(moeInput(1, 0.2)).some((e) => e.includes('h must be 0'))).toBe(false);
  });
});

describe('U-4 transferTime: t = α_tier + α_link + max(Little, IOPS/n, tier/n, link, host staging)', () => {
  it('host RAM → PCIe5, reads [{8, 2,654,208}]: α 1.07 µs, link-bound service 21,233,664 / 57.614e9 = 3.68550e-4', () => {
    const t = transferTime([{ count: 8, bytesEach: 2_654_208 }], 21_233_664, ramPath());
    expectClose(t.latencySec, 1.07e-6);
    expectClose(t.linkSec, 21_233_664 / B_LINK);
    expectClose(t.tierSec, 21_233_664 / 96e9); // 2.2120e-4
    expectClose(t.littleSec, (7 * 70e-9) / 64); // 7.66e-9
    expect(t.iopsSec).toBeUndefined();
    expect(t.hostSec).toBeUndefined();
    expectClose(t.serviceSec, 3.6855e-4, 1e-5);
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
    const t = transferTime(
      [{ count: 16, bytesEach: 320 }],
      5_120,
      ssdPath({ tier: NVME_990_PRO_2TB }),
    );
    expectClose(t.latencySec, 1 / 22_000 + 1e-6);
  });
  it('an NvmeSpec without randomRead (Spark-like, maxOutstanding 512, 50 µs): no IOPS term, and resolveHost warns', () => {
    const ssd: NvmeSpec = { ...DGX_SPARK_SSD_4TB, seqReadBandwidthBytesPerSec: 12.2e9 };
    delete ssd.randomRead;
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, ssdPath({ tier: ssd }));
    expect(t.iopsSec).toBeUndefined();
    expectClose(t.littleSec, (15 * 50e-6) / 512);
    expectClose(
      t.serviceSec,
      Math.max((15 * 50e-6) / 512, 65_536 / 12.2e9, 65_536 / 48e9, 5_120 / B_LINK),
    );
    const host = resolveHost(
      { device: SYNTH_GPU, topology: SYNTH_CLUSTER, host: { link: PCIE5, ssd } },
      SYNTH_GPU,
      new Set(['nvme']),
    );
    expect(host.warnings).toContain(
      `offload: no random-read IOPS for ${ssd.id}; request rate from latency × queue depth`,
    );
  });
  it('NVMe on unified memory: no link (α_link 0, no link term), no host staging', () => {
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, {
      tier: SSD_45US,
      queueDepth: 512,
      nStream: 1,
    });
    expectClose(t.latencySec, 45e-6);
    expect(t.linkSec).toBe(0);
    expect(t.hostSec).toBeUndefined();
  });
});

describe('transferTime resource use (link bytes, host RAM bytes and time) for the overlapped stage transfer', () => {
  it('host RAM path: link = usefulBytes, host RAM = bytes_tier at its tier time', () => {
    const t = transferTime([{ count: 8, bytesEach: 2_654_208 }], 21_233_664, ramPath());
    expect(t.linkBytes).toBe(21_233_664);
    expect(t.hostRamBytes).toBe(t.tierBytes); // 64 B lines: 2,654,208 is a multiple of 64
    expect(t.hostRamBytes).toBe(21_233_664);
    expect(t.hostRamSec).toBe(t.tierSec);
    expectClose(t.hostRamSec, 21_233_664 / 96e9);
  });
  it('NVMe on a dedicated device: host RAM moves 2·bytes_tier (written, read) at the staging time', () => {
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, ssdPath());
    expect(t.linkBytes).toBe(5_120);
    expect(t.hostRamBytes).toBe(2 * 65_536);
    expect(t.hostRamSec).toBe(t.hostSec);
    expectClose(t.hostRamSec, 65_536 / 48e9);
  });
  it('NVMe on unified memory (no link, no staging): no link and no host RAM use', () => {
    const t = transferTime([{ count: 16, bytesEach: 320 }], 5_120, {
      tier: SSD_45US,
      queueDepth: 512,
      nStream: 1,
    });
    expect([t.linkBytes, t.hostRamBytes, t.hostRamSec]).toEqual([0, 0, 0]);
  });
});

describe('prefetchesFromMultipleTiers: ops with q > 0 from two or more sources', () => {
  const op = (source: 'host_ram' | 'nvme' | 'remote_pool', prefetchedFraction: number) => ({
    source,
    prefetchedFraction,
  });
  it('both tiers with q > 0 ⇒ true', () => {
    expect(prefetchesFromMultipleTiers([op('host_ram', 0.8), op('nvme', 1)])).toBe(true);
    expect(prefetchesFromMultipleTiers([op('nvme', 0), op('nvme', 0.3), op('host_ram', 0.8)])).toBe(
      true,
    );
  });
  it('one tier, any q ⇒ false', () => {
    expect(prefetchesFromMultipleTiers([op('host_ram', 0.8), op('host_ram', 1)])).toBe(false);
    expect(prefetchesFromMultipleTiers([op('nvme', 1), op('nvme', 0)])).toBe(false);
    expect(prefetchesFromMultipleTiers([])).toBe(false);
  });
  it('the remote pool counts as a source: pool + RAM or pool + SSD ⇒ true, pool alone ⇒ false', () => {
    expect(prefetchesFromMultipleTiers([op('remote_pool', 1), op('host_ram', 1)])).toBe(true);
    expect(prefetchesFromMultipleTiers([op('remote_pool', 0.5), op('nvme', 1)])).toBe(true);
    expect(prefetchesFromMultipleTiers([op('remote_pool', 1), op('remote_pool', 1)])).toBe(false);
    expect(prefetchesFromMultipleTiers([op('remote_pool', 0), op('host_ram', 1)])).toBe(false);
  });
  it('both tiers, but one (or both) only at q = 0 ⇒ false', () => {
    expect(prefetchesFromMultipleTiers([op('host_ram', 0), op('nvme', 0)])).toBe(false);
    expect(prefetchesFromMultipleTiers([op('host_ram', 0.8), op('nvme', 0)])).toBe(false);
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
    expectClose(
      transferTime(reads, 21_233_664, ssdPath({ nStream: 4 })).sec,
      46e-6 + 21_233_664 / (7.45e9 / 4),
    );
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
    // 'ideal' = prefetching on (p > 0), 'none' = off (p = 0).
    expect(stageNStream(parallel, 8, streams, 0, overlap === 'ideal', m)).toBe(expected);
  });
  it('the optimistic schedule takes the fewest consumers, but never fewer than the stage itself', () => {
    expect(nStream([4, 1, 2, 0], 1, 2)).toBe(3); // 1 + 2
    expect(nStream([4, 1, 2, 0], 0, 2)).toBe(4); // at least its own 4
  });
  it('(e) end to end: REF_MOE_30B_A3B pp 4, b 2, experts on the SSD, prefetch accuracy 0 ⇒ tier term / 2 consumers', () => {
    const run = (prefetch: 'ideal' | 'none') =>
      simulate({
        model: REF_MOE_30B_A3B,
        quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
        parallel: { tp: 1, pp: 4, dp: 1, ep: 1 },
        workload: { batch: 2, promptLen: 128, decodeContextLen: 128 },
        hardware: {
          device: SYNTH_GPU,
          topology: SYNTH_CLUSTER,
          host: { link: PCIE5, ssd: SSD_45US, ram: DDR5_6000_2CH },
        },
        deployment: {
          offload: {
            prefetchAccuracy: prefetch === 'ideal' ? 1 : 0,
            routedExperts: { tier: 'nvme', residentFraction: 0, hitRatio: { kind: 'uniform' } },
          },
        },
      });
    for (const [overlap, n] of [
      ['none', 2],
      ['ideal', 4],
    ] as const) {
      const fetch = run(overlap)
        .decode.stages[1].layers.flatMap((l) => l.ops)
        .find((o) => o.op.kind === 'transfer');
      expect(fetch?.op.kind).toBe('transfer');
      if (fetch === undefined || fetch.op.kind !== 'transfer') continue;
      // m = 2 microbatches of one sequence: M = U = k = 8 experts of 9.44 MB; the drive's term dominates
      // (staging 75.5 MB/(96e9/(2n)) and the link stay below it).
      const tierBytes = fetch.op.reads.reduce(
        (s, r) => s + r.count * Math.ceil(r.bytesEach / 4096) * 4096,
        0,
      );
      expectClose(fetch.transferBandwidthSec, tierBytes / (7.45e9 / n));
      expectClose(fetch.transferLatencySec, 46e-6);
      expect(fetch.bound).toBe('transfer');
    }
  });
});

/**
 * Mixed placement: host RAM is one resource behind host-RAM-tier reads and NVMe staging, so its sharing n_host counts
 * every stage that uses it either way (docs/simulation-model.md §12 "Sharing"):
 *   hostStreams[s] = streams_ram[s] ∨ streams_nvme[s];  n_host = stageNStream(hostStreams)
 *   RAM tier term = bytes_tier / (B_ram / n_host);  NVMe staging = bytes_tier / (B_ram / (2·n_host))
 * Qwen3.8-Flash-Next (48 MoE layers, the PLE table in layer 1) at pp 2: the table (stage 0) in DDR5-6000 ×2
 * (96e9 B/s), the experts of layers 24–47 (stage 1) on a fast SSD (200 GB/s, so the staging term, not the drive, is
 * the largest throughput term), PCIe 5.0 x16 (57.6 GB/s). Decode b 1: the table reads 16 rows of 320 B (5,120 B).
 */
describe('n_host: host RAM shared by host-RAM reads and NVMe staging (tables in RAM, experts on SSD)', () => {
  const FAST_SSD: NvmeSpec = { ...SSD_45US, seqReadBandwidthBytesPerSec: 200e9 };
  const B_RAM = 96e9;
  const qwen = (() => {
    const mapped = mapHfModel(fixtureRepoFiles('Qwen/Qwen3.8-Flash-Next', { dir: '' }));
    if (mapped.kind !== 'ok') throw new Error('fixture');
    return { model: mapped.model, quant: mapped.quant.quant };
  })();
  const run = (parallel: { tp: number; pp: number }, prefetch: number) =>
    simulate({
      ...qwen,
      parallel: { ...parallel, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 128, decodeContextLen: 128 },
      hardware: {
        device: SYNTH_GPU,
        topology: SYNTH_CLUSTER,
        host: { link: PCIE5, ssd: FAST_SSD, ram: DDR5_6000_2CH },
      },
      deployment: {
        offload: {
          prefetchAccuracy: prefetch,
          lookupTables: { tier: 'host_ram' },
          routedExperts: {
            tier: 'nvme',
            layers: { from: 24, to: 48 },
            residentFraction: 0,
            hitRatio: { kind: 'uniform' },
          },
        },
      },
    });
  /** The transfer op of `group` in decoder layer `i`, with its result. */
  const fetchOf = (
    r: ReturnType<typeof run>,
    i: number,
    group: 'lookup_table' | 'routed_experts',
  ) => {
    const layer = r.decode.stages.flatMap((s) => s.layers).find((l) => l.layerIndex === i);
    const fetch = layer?.ops.find((o) => o.op.kind === 'transfer' && o.op.group === group);
    if (fetch === undefined || fetch.op.kind !== 'transfer')
      throw new Error(`layer ${i}: no ${group} op`);
    return { fetch, op: fetch.op };
  };
  const ramBytes = (op: { reads: { count: number; bytesEach: number }[] }) =>
    op.reads.reduce((s, r) => s + r.count * Math.ceil(r.bytesEach / 64) * 64, 0);
  const ssdBytes = (op: { reads: { count: number; bytesEach: number }[] }) =>
    op.reads.reduce((s, r) => s + r.count * Math.ceil(r.bytesEach / 4096) * 4096, 0);

  it("each op reads from its rule's tier", () => {
    const r = run({ tp: 1, pp: 2 }, 1);
    expect(fetchOf(r, 1, 'lookup_table').op.source).toBe('host_ram');
    for (const i of [24, 47]) expect(fetchOf(r, i, 'routed_experts').op.source).toBe('nvme');
  });

  it('pp 2, b 1, p > 0: stage 0 (RAM table) and stage 1 (SSD experts) both use host RAM ⇒ n_host = 2', () => {
    const r = run({ tp: 1, pp: 2 }, 1);
    const table = fetchOf(r, 1, 'lookup_table');
    // RAM: max(little, tier = 5,120/(B_ram/2) = 1.07e-7 s, link = 5,120/57.6e9 = 8.9e-8 s) ⇒ tier
    // (n_host = 1 would make it link-bound)
    expectClose(table.fetch.transferBandwidthSec, ramBytes(table.op) / (B_RAM / 2));
    const experts = fetchOf(r, 24, 'routed_experts');
    // SSD: max(tier = bytes/200e9, link, staging = bytes/(B_ram/(2·2)) = bytes/24e9) ⇒ staging
    expectClose(experts.fetch.transferBandwidthSec, ssdBytes(experts.op) / (B_RAM / 4));
  });

  it('the same with p = 0, m = 1: one stage streams at a time ⇒ n_host = 1', () => {
    const r = run({ tp: 1, pp: 2 }, 0);
    const table = fetchOf(r, 1, 'lookup_table');
    // RAM: tier = 5,120/96e9 < link = 5,120/B_link ⇒ link-bound
    expectClose(table.fetch.transferBandwidthSec, table.op.usefulBytes / B_LINK);
    const experts = fetchOf(r, 24, 'routed_experts');
    // SSD: staging = bytes/(B_ram/2) = bytes/48e9 > link, tier
    expectClose(experts.fetch.transferBandwidthSec, ssdBytes(experts.op) / (B_RAM / 2));
  });

  it('tp 2, both tiers in one stage: n_host = 2 = n_tier (each device counted once)', () => {
    const r = run({ tp: 2, pp: 1 }, 1);
    const table = fetchOf(r, 1, 'lookup_table');
    expectClose(table.fetch.transferBandwidthSec, ramBytes(table.op) / (B_RAM / 2));
    const experts = fetchOf(r, 24, 'routed_experts');
    expectClose(experts.fetch.transferBandwidthSec, ssdBytes(experts.op) / (B_RAM / 4));
  });

  it('transferTime: hostStagingNStream sets the staging share; absent, it is nStream (single-tier results unchanged)', () => {
    const reads = [{ count: 1, bytesEach: 409_600 }];
    expect(transferTime(reads, 409_600, ssdPath({ nStream: 2 }))).toEqual(
      transferTime(reads, 409_600, ssdPath({ nStream: 2, hostStagingNStream: 2 })),
    );
    const t = transferTime(reads, 409_600, ssdPath({ nStream: 1, hostStagingNStream: 3 }));
    expectClose(t.hostSec ?? 0, 409_600 / (96e9 / 6));
    expectClose(t.tierSec, 409_600 / 7.45e9);
    // A host-RAM path has no staging term at all.
    expect(transferTime(reads, 409_600, ramPath()).hostSec).toBeUndefined();
  });
});
