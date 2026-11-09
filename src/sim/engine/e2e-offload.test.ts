/**
 * Worked examples of weight offloading (design doc §8.4 W1, §8.6 W3/W4), end to end through `simulate`:
 * exact (rel 1e-6) on the hand-derived transfer terms and the tier bytes, within 0.1% on the quoted tok/s.
 */
import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { NVME_990_PRO_2TB } from '../hw/presets/storage';
import type { NvmeSpec } from '../hw/types';
import type { QuantConfig } from '../ir/model';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_MOE_30B_A3B } from '../models/reference';
import { simulate } from './simulate';
import { DEFAULT_PREFETCH_ACCURACY } from './transfer';
import type { OffloadConfig, SimInput, SimResult } from './types';

/**
 * Prefetch setting of a worked example: `'ideal'` = p 1 (the former offloadOverlap 'ideal'), `'none'` = p 0 (the
 * former 'none'), or an explicit p. The rows keep their pre-prefetch-accuracy numbers, so they pin both endpoints.
 */
type Prefetch = 'ideal' | 'none' | number;

/** p of a {@link Prefetch}. Dimensionless. */
function prefetchAccuracy(prefetch: Prefetch): number {
  return prefetch === 'ideal' ? 1 : prefetch === 'none' ? 0 : prefetch;
}

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** PCIe 5.0 x16 per direction, 32e9·16·128/130/8. Unit: B/s. */
const B_LINK = (32e9 * 16 * 128) / 130 / 8;
const GPU = { device: RTX_5090, topology: singleDevice(RTX_5090) };

// ---------------------------------------------------------------------------------------------------------------
// W1: Qwen3-30B-A3B-class MoE on one RTX 5090, experts in DDR5-6000 ×2 over PCIe 5.0 x16 (design doc §8.4)

const W1_QUANT: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
  modules: { routed_experts: { weights: { dtype: 'int4', extraBitsPerElem: 0.5 } } },
};
/** One expert at 4.5 bits: 3·2048·768·4.5/8. Unit: bytes. */
const W_EXPERT = 2_654_208;

function w1(f: number | undefined, prefetch: Prefetch): SimResult {
  const input: SimInput = {
    model: REF_MOE_30B_A3B,
    quant: W1_QUANT,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: GPU,
  };
  if (f !== undefined) {
    input.deployment = {
      offload: {
        prefetchAccuracy: prefetchAccuracy(prefetch),
        routedExperts: { tier: 'host_ram', residentFraction: f, hitRatio: { kind: 'uniform' } },
      },
    };
  }
  return simulate(input);
}

/** A tier breakdown with every category empty; tests spread it and set the categories they expect. */
const NO_TIER_BYTES = { lookupTables: 0, routedExperts: 0, denseLayers: 0, draftExperts: 0 };

describe('W1 experts in DDR5 over PCIe 5.0 x16 (REF_MOE_30B_A3B, int4+0.5 routed experts, b 1, @8k)', () => {
  const baseline = w1(undefined, 'ideal');
  it('baseline (all resident): decode 2.46673 ms, prefill 377.580 ms', () => {
    expectClose(baseline.decode.cycleSec, 2.46673e-3, 1e-5);
    expectClose(baseline.prefill.cycleSec, 377.58e-3, 1e-5);
  });

  // [phase, f, M per layer, 'ideal' tok/s, 'none' tok/s]
  const rows = [
    ['decode', 0, 8, 61.63, 53.5],
    ['decode', 0.5, 4, 122.87, 94.29],
    ['prefill', 0, 128, 21_696, 12_872],
    ['prefill', 0.5, 64, 21_696, 16_157],
  ] as const;
  for (const [phase, f, M, idealRate, noneRate] of rows) {
    for (const [overlap, rate] of [['ideal', idealRate], ['none', noneRate]] as const) {
      it(`${phase} @ 8k, f = h = ${f}, ${overlap}: xfer = 48·(1.07 µs + ${M}·W_expert/B_link), ${rate} tok/s @ 8k`, () => {
        const r = w1(f, overlap)[phase];
        const tPath = 1.07e-6 + (M * W_EXPERT) / B_LINK;
        expect(r.transfer).toBeDefined();
        expectClose(r.transfer?.sec ?? 0, 48 * tPath, 1e-6);
        expectClose(r.transfer?.bytes ?? 0, 48 * M * W_EXPERT, 1e-9);
        expectClose(r.transfer?.requests ?? 0, 48 * M, 1e-9);
        expectClose(r.tokensPerSec, rate, 1e-3);
        const core = baseline[phase].cycleSec;
        const expectedCycle = overlap === 'ideal' ? Math.max(core, 48 * tPath) : core + 48 * tPath;
        expectClose(r.cycleSec, expectedCycle, 1e-6);
        // The fetch sits right in front of the MoE block (before the router), one per layer.
        const layerOps = r.stages[0].layers[1].ops;
        const at = layerOps.findIndex((o) => o.op.kind === 'transfer');
        expect(layerOps[at + 1].op.name).toBe('moe.router');
        expect(r.stages[0].layers.filter((l) => l.ops.some((o) => o.op.kind === 'transfer')).length).toBe(48);
        const fetch = layerOps[at];
        expect(fetch.op.name).toBe('moe.expert_fetch');
        expect(fetch.bound).toBe('transfer');
        expectClose(fetch.transferLatencySec, 1.07e-6);
        expectClose(fetch.transferBandwidthSec, (M * W_EXPERT) / B_LINK);
      });
    }
  }

  it('memory f 0 ideal: W_s 3,082,186,752; staging 2·128·W 679,477,248; total 5,036,830,720 ≤ 32e9; host tier 16,307,453,952', () => {
    const m = w1(0, 'ideal').memory;
    expect(m.perStage[0].weightBytes).toBe(19_389_640_704 - 16_307_453_952);
    expect(m.perStage[0].stagingBytes).toBe(2 * 128 * W_EXPERT);
    expect(m.perStage[0].sequenceStateBytes).toBe(805_404_672);
    expect(m.perStage[0].activationPeakBytes).toBe(469_762_048);
    expect(m.perStage[0].totalBytes).toBe(5_036_830_720);
    expect(m.tiers).toEqual([{ tier: 'host_ram', bytes: 16_307_453_952, breakdown: { ...NO_TIER_BYTES, routedExperts: 16_307_453_952 }, capacityBytes: 96e9, fits: true }]);
    expect(m.fits).toBe(true);
  });
  it('memory f .5 ideal: W_s 11,235,913,728; staging 2·64·W 339,738,624; none halves the staging', () => {
    const m = w1(0.5, 'ideal').memory.perStage[0];
    expect(m.weightBytes).toBe(11_235_913_728);
    expect(m.stagingBytes).toBe(339_738_624);
    expect(w1(0.5, 'none').memory.perStage[0].stagingBytes).toBe(64 * W_EXPERT);
  });

  it('default p 0.8 (decode f 0): q = max(0.8, 8/128) = 0.8; t = core + xfer − q·min(core, xfer)', () => {
    const omitted = simulate({
      model: REF_MOE_30B_A3B,
      quant: W1_QUANT,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
      hardware: GPU,
      deployment: { offload: { routedExperts: { tier: 'host_ram', residentFraction: 0, hitRatio: { kind: 'uniform' } } } },
    });
    expect(omitted).toEqual(w1(0, DEFAULT_PREFETCH_ACCURACY));
    const r = omitted.decode;
    const fetch = r.stages[0].layers[1].ops.find((o) => o.op.kind === 'transfer');
    expect(fetch?.op.kind === 'transfer' ? fetch.op.prefetchedFraction : NaN).toBe(0.8);
    const core = baseline.decode.cycleSec;
    const xfer = 48 * (1.07e-6 + (8 * W_EXPERT) / B_LINK);
    expectClose(r.cycleSec, core + xfer - 0.8 * Math.min(core, xfer), 1e-6);
    expect(r.cycleSec).toBeGreaterThan(w1(0, 'ideal').decode.cycleSec);
    expect(r.cycleSec).toBeLessThan(w1(0, 'none').decode.cycleSec);
    // Prefill needs every expert (M = E − C), so q = 1: identical to perfect prefetch.
    expect(omitted.prefill).toEqual(w1(0, 'ideal').prefill);
  });

  it('f = 1 contract: every rate, op time and device memory line equals the baseline; only tiers and warnings differ', () => {
    for (const overlap of ['ideal', 'none'] as const) {
      const base = w1(undefined, overlap);
      const r = w1(1, overlap);
      expectClose(r.decode.cycleSec, base.decode.cycleSec);
      expectClose(r.decode.tokensPerSec, base.decode.tokensPerSec);
      expectClose(r.prefill.cycleSec, base.prefill.cycleSec);
      expectClose(r.prefill.latencySec, base.prefill.latencySec);
      expectClose(r.prefill.tokensPerSec, base.prefill.tokensPerSec);
      for (const phase of ['decode', 'prefill'] as const) {
        const ops = r[phase].stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
        const baseOps = base[phase].stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
        expect(ops.some((o) => o.op.kind === 'transfer')).toBe(false);
        expect(ops.length).toBe(baseOps.length);
        ops.forEach((o, i) => expectClose(o.sec, baseOps[i].sec));
        expect(r[phase].transfer).toBeUndefined();
      }
      const [d, b] = [r.memory.perStage[0], base.memory.perStage[0]];
      expectClose(d.weightBytes, b.weightBytes);
      expectClose(d.sequenceStateBytes, b.sequenceStateBytes);
      expectClose(d.activationPeakBytes, b.activationPeakBytes);
      expect(d.stagingBytes).toBe(0);
      expect(d.fits).toBe(b.fits);
      expect(r.memory.tiers).toEqual([{ tier: 'host_ram', bytes: 16_307_453_952, breakdown: { ...NO_TIER_BYTES, routedExperts: 16_307_453_952 }, capacityBytes: 96e9, fits: true }]);
      expect(base.memory.tiers).toEqual([]);
      expect(r.warnings).not.toEqual(base.warnings);
    }
  });
});

// ---------------------------------------------------------------------------------------------------------------
// W3/W4: n-gram tables off-device on one RTX 5090, decode and prefill @ 8k, prefetch accuracy 0 (design doc §8.6)

/** The design doc's 990 PRO numbers: QD1 latency rounded to 45 µs (the preset carries 1/22,000 s = 45.45 µs). */
const SSD_45US: NvmeSpec = { ...NVME_990_PRO_2TB, randomReadLatencySec: 45e-6 };

function hf(repo: string): Pick<SimInput, 'model' | 'quant'> {
  const mapped = mapHfModel(fixtureRepoFiles(repo, { dir: '' }));
  if (mapped.kind !== 'ok') throw new Error(`${repo}: ${JSON.stringify(mapped)}`);
  return { model: mapped.model, quant: mapped.quant.quant };
}

function tables(repo: string, lookupTables: OffloadConfig['lookupTables'] | undefined, prefetch: Prefetch = 'none'): SimResult {
  const input: SimInput = {
    ...hf(repo),
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: { ...GPU, host: { ssd: SSD_45US } },
  };
  if (lookupTables !== undefined) input.deployment = { offload: { prefetchAccuracy: prefetchAccuracy(prefetch), lookupTables } };
  return simulate(input);
}

const PLE = 'Qwen/Qwen3.8-Flash-Next';
const ENGRAM = 'deepseek-ai/DeepSeek-V4.1-Flash';

describe('W3/W4 n-gram tables off-device (fixtures, repo quant, RTX 5090, b 1, @8k, prefetch accuracy 0)', () => {
  // [repo, tier, phase, transfer sec (all tables), tok/s @ 8k]
  const rows = [
    // PLE: one table, R = 16 (decode) / 131,072 (prefill) reads of 320 B.
    [PLE, 'host_ram', 'decode', 1.07e-6 + 5_120 / B_LINK, 119.651], // 1.1513 µs
    [PLE, 'host_ram', 'prefill', 1.07e-6 + 41_943_040 / B_LINK, 13_429.0], // 0.66667 ms
    [PLE, 'nvme', 'decode', 46e-6 + 16 / 1.4e6, 118.851], // 57.429 µs
    [PLE, 'nvme', 'prefill', 46e-6 + 131_072 / 1.4e6, 11_652.5], // 93.669 ms
    // Engram: two tables, 24 rows per table per decode step, 196,608 per prefill; 264 B useful per row.
    [ENGRAM, 'host_ram', 'decode', 2 * (1.07e-6 + 6_336 / B_LINK), 111.336], // 2.3411 µs
    [ENGRAM, 'host_ram', 'prefill', 2 * (1.07e-6 + 51_904_512 / B_LINK), 9_313.8], // 1.6495 ms
    [ENGRAM, 'nvme', 'decode', 2 * (46e-6 + 24 / 1.4e6), 109.82], // 126.286 µs
    [ENGRAM, 'nvme', 'prefill', 2 * (46e-6 + 196_608 / 1.4e6), 7_069.0], // 280.96 ms
  ] as const;
  for (const [repo, tier, phase, sec, rate] of rows) {
    it(`${repo} tables on ${tier}, ${phase}: transfer ${(sec * 1e6).toFixed(4)} µs, ${rate} tok/s @ 8k`, () => {
      const base = tables(repo, undefined)[phase];
      const r = tables(repo, { tier })[phase];
      expectClose(r.transfer?.sec ?? 0, sec, 1e-6);
      expectClose(r.cycleSec, base.cycleSec + sec, 1e-9);
      expectClose(r.tokensPerSec, rate, 1e-3);
    });
  }

  it('p 1 and the default p (q = 1 for table rows) leave every rate at the baseline: the gathers hide behind compute', () => {
    for (const repo of [PLE, ENGRAM]) {
      const base = tables(repo, undefined);
      const r = tables(repo, { tier: 'host_ram' }, 'ideal');
      expect(tables(repo, { tier: 'host_ram' }, DEFAULT_PREFETCH_ACCURACY)).toEqual(r);
      expect(r.decode.cycleSec).toBe(base.decode.cycleSec);
      expect(r.prefill.cycleSec).toBe(base.prefill.cycleSec);
    }
  });

  it('baselines equal G-h1 / G-h2 (decode 8.356467 / 8.979504 ms); lookupTables at device changes nothing', () => {
    expectClose(tables(PLE, undefined).decode.cycleSec, 8.356467181537718e-3, 1e-6);
    expectClose(tables(ENGRAM, undefined).decode.cycleSec, 8.979504281438205e-3, 1e-6);
    expect(tables(PLE, { tier: 'device' })).toEqual(tables(PLE, undefined));
  });

  it('PLE memory: device weights 353,887,054,848 → 251,487,054,848; host RAM 102.4e9 > 96e9 does not fit; NVMe fits', () => {
    const base = tables(PLE, undefined).memory;
    expect(base.perStage[0].weightBytes).toBe(353_887_054_848);
    expect(base.perStage[0].lookupTableBytes).toBe(102_400_000_000);
    const ram = tables(PLE, { tier: 'host_ram' }).memory;
    expect(ram.perStage[0].weightBytes).toBe(251_487_054_848);
    expect(ram.perStage[0].lookupTableBytes).toBe(0);
    expect(ram.tiers).toEqual([{ tier: 'host_ram', bytes: 102_400_000_000, breakdown: { ...NO_TIER_BYTES, lookupTables: 102_400_000_000 }, capacityBytes: 96e9, fits: false }]);
    expect(ram.fits).toBe(false);
    const ssd = tables(PLE, { tier: 'nvme' }).memory;
    expect(ssd.tiers).toEqual([{ tier: 'nvme', bytes: 102_400_000_000, breakdown: { ...NO_TIER_BYTES, lookupTables: 102_400_000_000 }, capacityBytes: 2e12, fits: true }]);
  });

  it('Engram memory: device 501,382,474,992 → 298,624,442,592; separate on RAM 202,758,032,400 B does not fit; pinned_host SSD fits', () => {
    expect(tables(ENGRAM, undefined).memory.perStage[0].weightBytes).toBe(501_382_474_992);
    const ram = tables(ENGRAM, { tier: 'host_ram' }).memory;
    expect(ram.perStage[0].weightBytes).toBe(298_624_442_592);
    expect(ram.tiers).toEqual([{ tier: 'host_ram', bytes: 202_758_032_400, breakdown: { ...NO_TIER_BYTES, lookupTables: 202_758_032_400 }, capacityBytes: 96e9, fits: false }]);
    const ssd = tables(ENGRAM, { tier: 'nvme' }).memory;
    expect(ssd.tiers).toEqual([
      { tier: 'host_ram', bytes: 6_144_182_800, breakdown: { ...NO_TIER_BYTES, lookupTables: 6_144_182_800 }, capacityBytes: 96e9, fits: true },
      { tier: 'nvme', bytes: 196_613_849_600, breakdown: { ...NO_TIER_BYTES, lookupTables: 196_613_849_600 }, capacityBytes: 2e12, fits: true },
    ]);
  });

  it('interleaved gives the same numbers as the defaults on both tiers for these two models', () => {
    for (const repo of [PLE, ENGRAM]) {
      for (const tier of ['host_ram', 'nvme'] as const) {
        const def = tables(repo, { tier });
        const inter = tables(repo, { tier, scaleLayout: 'interleaved' });
        expectClose(inter.decode.transfer?.sec ?? 0, def.decode.transfer?.sec ?? 0, 1e-12);
        expectClose(inter.prefill.transfer?.sec ?? 0, def.prefill.transfer?.sec ?? 0, 1e-12);
      }
    }
  });
});
