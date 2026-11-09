/**
 * Mixed placement by layer range (docs/simulation-model.md §12 "Placement", design doc §2.11): the routed experts
 * (or whole layers) of some layers in host RAM, others on the SSD. End to end through `simulate`, plus the
 * regression pins that the single-object form is the one-rule list, bit for bit.
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import { singleDevice } from '../hw/presets';
import { hostPcieLink } from '../hw/presets/links';
import { M5_ULTRA_80C } from '../hw/presets/apple';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_5600_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import type { QuantConfig } from '../ir/model';
import type { LoweringContext } from '../lower/lower';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_MOE_30B_A3B } from '../models/reference';
import { stageTierBreakdown } from './memory';
import { simulate } from './simulate';
import type { OffloadConfig, RoutedExpertRule, SimInput, SimResult } from './types';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

function hf(repo: string): Pick<SimInput, 'model' | 'quant'> {
  const mapped = mapHfModel(fixtureRepoFiles(repo, { dir: '' }));
  if (mapped.kind !== 'ok') throw new Error(`${repo}: ${JSON.stringify(mapped)}`);
  return { model: mapped.model, quant: mapped.quant.quant };
}

/** PCIe 5.0 x16 per direction, 32e9·16·128/130/8 × 256/280 (TLP payload efficiency, links.ts). Unit: B/s. */
const B_LINK = ((32e9 * 16 * 128) / 130 / 8) * (256 / 280);
const GPU = { device: RTX_5090, topology: singleDevice(RTX_5090) };
const UNIFORM_F0 = { residentFraction: 0, hitRatio: { kind: 'uniform' } } as const;

// ---------------------------------------------------------------------------------------------------------------
// The task's check: Qwen3.8-Flash-Next on one RTX 5090 + DDR5-5600 ×2 (96 GB) + 990 PRO, decode @ 8k, b 1

/**
 * Qwen/Qwen3.8-Flash-Next (fixture): 48 decoder layers, every one with a MoE block of 512 experts, top-10 (layer 0
 * also carries the PLE n-gram table), so the layers are identical as far as the expert fetch goes. Weights fp8_e4m3:
 * one expert is W_expert = 4,915,200 B, a layer's experts 512·W_expert = 2,516,582,400 B, all 48 layers 120.8 GB:
 * more than the 96 GB of host RAM, so all-in-RAM does not fit, while 24/24 does. The PLE table (102.4 GB) is on the
 * SSD in every configuration, f = 0 (nothing resident), default prefetch accuracy 0.8.
 *
 * Per offloaded layer and decode step at b 1: M = U = 10 misses, reads [{10, W_expert}] = 49,152,000 B.
 * - host RAM layer: t_ram = α_ram + α_link + max(little, tier = 49.152e6/89.6e9 = 0.5486 ms,
 *   link = 49.152e6/B_link = 0.8531 ms) ⇒ link-bound: 1.07 µs + 0.8531 ms = 0.8542 ms.
 * - SSD layer: t_ssd = α_ssd + α_link + max(little, IOPS = 10/1.4e6, tier = 49.152e6/7.45e9 = 6.5976 ms,
 *   link 0.8531 ms, staging = 49.152e6/(89.6e9/2) = 1.0971 ms) ⇒ drive-bound: 46.45 µs + 6.5976 ms = 6.6441 ms.
 * - the PLE table rows (layer 0, SSD): t_table = α_ssd + α_link + 16/1.4e6 (IOPS-bound), the same in each config.
 * transfer.sec = Σ ops: all-RAM 48·t_ram + t_table ≈ 41.06 ms; all-SSD 48·t_ssd + t_table ≈ 318.97 ms;
 * 24/24 = 24·t_ram + 24·t_ssd + t_table = (all-RAM + all-SSD)/2 ≈ 180.02 ms (the table term cancels in the mean).
 * One device, so n_host = n_tier = 1: the two tiers never share anything but the serial per-device stream.
 * Decode @ 8k: all-RAM 23.73 tok/s > 24/24 5.522 tok/s > all-SSD 3.124 tok/s.
 */
const QWEN = 'Qwen/Qwen3.8-Flash-Next';
const QWEN_LAYERS = 48;
const W_EXPERT_FP8 = 4_915_200;
/** One layer's full expert set, 512·W_expert. Unit: bytes. */
const LAYER_EXPERT_BYTES = 512 * W_EXPERT_FP8;
const PLE_TABLE_BYTES = 102_400_000_000;

function qwen(experts: OffloadConfig['routedExperts'], device = GPU): SimResult {
  const { model, quant } = hf(QWEN);
  const fp8: QuantConfig = { ...quant, weights: { dtype: 'fp8_e4m3' } };
  return simulate({
    model,
    quant: fp8,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: { ...device, host: { ram: DDR5_5600_2CH, ssd: NVME_990_PRO_2TB } },
    deployment: { offload: { routedExperts: experts, lookupTables: { tier: 'nvme' } } },
  });
}
const allRam: RoutedExpertRule = { tier: 'host_ram', ...UNIFORM_F0 };
const allSsd: RoutedExpertRule = { tier: 'nvme', ...UNIFORM_F0 };
const split = (ramLayers: number): RoutedExpertRule[] => [
  { tier: 'host_ram', layers: { from: 0, to: ramLayers }, ...UNIFORM_F0 },
  { tier: 'nvme', layers: { from: ramLayers, to: QWEN_LAYERS }, ...UNIFORM_F0 },
];

describe(`${QWEN} fp8 on one RTX 5090 + DDR5-5600 ×2 + 990 PRO: experts all in RAM / 24 in RAM + 24 on SSD / all on SSD, b 1, decode @ 8k`, () => {
  const ram = qwen(allRam);
  const mixed = qwen(split(24));
  const ssd = qwen(allSsd);

  it('the fixture has 48 identical MoE layers (512 experts, top-10)', () => {
    const { model } = hf(QWEN);
    expect(model.layers).toHaveLength(QWEN_LAYERS);
    for (const layer of model.layers) {
      const moe = layer.blocks.filter((b) => b.kind === 'moe');
      expect(moe).toHaveLength(1);
      expect(moe[0]).toMatchObject({ experts: 512, topK: 10 });
    }
  });

  it('decode @ 8k sits between the extremes: all-RAM 23.73 > 24/24 5.522 > all-SSD 3.124 tok/s', () => {
    expect(ram.decode.tokensPerSec).toBeGreaterThan(mixed.decode.tokensPerSec);
    expect(mixed.decode.tokensPerSec).toBeGreaterThan(ssd.decode.tokensPerSec);
    expectClose(ram.decode.tokensPerSec, 23.7309, 1e-4);
    expectClose(mixed.decode.tokensPerSec, 5.52192, 1e-4);
    expectClose(ssd.decode.tokensPerSec, 3.12448, 1e-4);
  });

  it('transfer per step: 24/24 = (all-RAM + all-SSD)/2 exactly; each extreme is 48 layers of its tier plus the table', () => {
    const t = (r: SimResult) => r.decode.transfer?.sec ?? 0;
    expectClose(t(mixed), (t(ram) + t(ssd)) / 2, 1e-12);
    const alphaSsd = 1 / 22_000 + 1e-6;
    const tTable = alphaSsd + 16 / 1.4e6;
    const tRam = 70e-9 + 1e-6 + (10 * W_EXPERT_FP8) / B_LINK;
    const tSsd = alphaSsd + (10 * W_EXPERT_FP8) / 7.45e9;
    expectClose(t(ram), QWEN_LAYERS * tRam + tTable);
    expectClose(t(ssd), QWEN_LAYERS * tSsd + tTable);
    expectClose(t(mixed), 24 * tRam + 24 * tSsd + tTable);
  });

  it("per layer, each op reads from its rule's tier", () => {
    const fetches = mixed.decode.stages[0].layers.flatMap((l) =>
      l.ops.flatMap((o) =>
        o.op.kind === 'transfer' && o.op.group === 'routed_experts'
          ? [{ i: l.layerIndex, source: o.op.source }]
          : [],
      ),
    );
    expect(fetches).toHaveLength(QWEN_LAYERS);
    for (const { i, source } of fetches) expect(source).toBe((i ?? -1) < 24 ? 'host_ram' : 'nvme');
  });

  it('tier bytes: all-RAM 120.8 GB > 96 GB does not fit; 24/24 puts 60.4 GB in RAM (fits) and 60.4 GB + the table on the SSD', () => {
    const half = 24 * LAYER_EXPERT_BYTES; // 60,397,977,600
    expect(ram.memory.tiers.map((t) => [t.tier, t.breakdown.routedExperts, t.fits])).toEqual([
      ['host_ram', 2 * half, false],
      ['nvme', 0, true],
    ]);
    expect(ram.memory.fits).toBe(false);
    expect(mixed.memory.tiers).toEqual([
      {
        tier: 'host_ram',
        bytes: half,
        breakdown: { lookupTables: 0, routedExperts: half, draftExperts: 0 },
        capacityBytes: 96e9,
        fits: true,
      },
      {
        tier: 'nvme',
        bytes: half + PLE_TABLE_BYTES,
        breakdown: {
          lookupTables: PLE_TABLE_BYTES,
          routedExperts: half,
          draftExperts: 0,
        },
        capacityBytes: 2e12,
        fits: true,
      },
    ]);
    expect(mixed.memory.fits).toBe(true);
    expect(ssd.memory.tiers.map((t) => [t.tier, t.breakdown.routedExperts, t.fits])).toEqual([
      ['nvme', 2 * half, true],
    ]);
    // The device keeps the same resident weights in all three (f = 0 everywhere).
    expect(mixed.memory.perStage[0].weightBytes).toBe(ram.memory.perStage[0].weightBytes);
    expect(mixed.memory.perStage[0].weightBytes).toBe(ssd.memory.perStage[0].weightBytes);
  });

  it('list order never changes a result; a split with one side empty equals the single-tier form', () => {
    expect(qwen([...split(24)].reverse())).toEqual(mixed);
    expect(qwen([{ ...allRam, layers: { from: 0, to: QWEN_LAYERS } }])).toEqual(ram);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Regression: the single-object form is the one-rule list, bit for bit

describe('single-tier configs are unchanged: object ≡ [object], same-tier split ≡ one rule, [] ≡ absent', () => {
  const base = (offload: OffloadConfig | undefined): SimResult =>
    simulate({
      model: REF_MOE_30B_A3B,
      quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
      parallel: { tp: 2, pp: 2, dp: 1, ep: 1 },
      workload: { batch: 2, promptLen: 2048, decodeContextLen: 2048 },
      hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER, host: { link: hostPcieLink(5, 16) } },
      ...(offload === undefined ? {} : { deployment: { offload } }),
    });
  const experts: RoutedExpertRule = {
    tier: 'nvme',
    layers: { from: 4, to: 40 },
    residentFraction: 0.25,
    hitRatio: { kind: 'skewed_fine_grained' },
  };

  it('routedExperts: object ≡ [object]; [4, 20) + [20, 40) on one tier ≡ [4, 40)', () => {
    const one = base({ routedExperts: experts });
    expect(base({ routedExperts: [experts] })).toEqual(one);
    expect(
      base({
        routedExperts: [
          { ...experts, layers: { from: 4, to: 20 } },
          { ...experts, layers: { from: 20, to: 40 } },
        ],
      }),
    ).toEqual(one);
  });

  it('[] ≡ absent, for both groups', () => {
    const none = base(undefined);
    expect(base({ routedExperts: [], lookupTables: [] })).toEqual(none);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Memory fit per tier on unified memory, and the drafter's experts

describe('unified memory: host_ram rules stay device weights, nvme rules go to the SSD tier', () => {
  const M5 = { device: M5_ULTRA_80C, topology: singleDevice(M5_ULTRA_80C) };
  const run = (experts: OffloadConfig['routedExperts']): SimResult =>
    simulate({
      model: REF_MOE_30B_A3B,
      quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
      hardware: { ...M5, host: { ssd: NVME_990_PRO_2TB } },
      ...(experts === undefined ? {} : { deployment: { offload: { routedExperts: experts } } }),
    });
  const ssdHalf: RoutedExpertRule = { tier: 'nvme', layers: { from: 24, to: 48 }, ...UNIFORM_F0 };
  const mixed = run([{ tier: 'host_ram', layers: { from: 0, to: 24 }, ...UNIFORM_F0 }, ssdHalf]);
  const ssdOnly = run([ssdHalf]);

  it('the host_ram rule is dropped with a note; the result is the nvme rule alone', () => {
    expect(mixed.warnings).toContain(
      "offload: on unified memory host RAM is the device memory; the 'host_ram' placement keeps the weights on the device",
    );
    expect(mixed.memory).toEqual(ssdOnly.memory);
    expect(mixed.decode).toEqual(ssdOnly.decode);
    expect(mixed.prefill).toEqual(ssdOnly.prefill);
  });

  it('no host_ram tier; the SSD tier holds the 24 SSD layers; the device keeps layers 0–23 in full', () => {
    const none = run(undefined);
    expect(mixed.memory.tiers.map((t) => t.tier)).toEqual(['nvme']);
    const nvme = mixed.memory.tiers[0].breakdown.routedExperts;
    expect(nvme).toBeGreaterThan(0);
    // f = 0: the SSD layers' full expert sets leave the device, nothing else moves.
    expect(mixed.memory.perStage[0].weightBytes).toBe(none.memory.perStage[0].weightBytes - nvme);
    expect(run([{ tier: 'nvme', ...UNIFORM_F0 }]).memory.tiers[0].breakdown.routedExperts).toBe(
      2 * nvme,
    );
  });
});

describe("the drafter's experts follow the rule with includeDraft (DeepSeek-V4.1-Flash DSpark)", () => {
  const r = hf('deepseek-ai/DeepSeek-V4.1-Flash');
  const draft = r.model.draft;
  if (draft === undefined) throw new Error('no draft');
  const L = r.model.layers.length;
  const ctx = (draftOn: 'host_ram' | 'nvme'): LoweringContext => ({
    phase: 'decode',
    seqs: 1,
    qLen: 6,
    kvLen: 8198,
    quant: r.quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    offload: {
      prefetchAccuracy: 0.8,
      routedExperts: [
        {
          tier: 'host_ram',
          layers: { from: 0, to: 20 },
          residentFraction: 0,
          hitRatio: 0,
          includeDraft: draftOn === 'host_ram',
        },
        {
          tier: 'nvme',
          layers: { from: 20, to: L },
          residentFraction: 0,
          hitRatio: 0,
          includeDraft: draftOn === 'nvme',
        },
      ],
    },
    speculative: { draft, draftLength: 5, verifyTokens: 6, contextRows: 1, stateSnapshots: 6 },
  });

  it('includeDraft on the RAM rule puts the draft experts in host RAM, on the SSD rule on the SSD', () => {
    const onRam = stageTierBreakdown(r.model, ctx('host_ram'), 0);
    const onSsd = stageTierBreakdown(r.model, ctx('nvme'), 0);
    expect(onRam.host_ram.draftExperts).toBeGreaterThan(0);
    expect(onRam.nvme.draftExperts).toBe(0);
    expect(onSsd.nvme.draftExperts).toBe(onRam.host_ram.draftExperts);
    expect(onSsd.host_ram.draftExperts).toBe(0);
    // The target's experts do not move with it.
    expect(onSsd.host_ram.routedExperts).toBe(onRam.host_ram.routedExperts);
    expect(onSsd.nvme.routedExperts).toBe(onRam.nvme.routedExperts);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Lookup tables split across tiers: DeepSeek-V4.1-Flash's two Engram tables (layers 1 and 14)

const ENGRAM = 'deepseek-ai/DeepSeek-V4.1-Flash';
const ENGRAM_LAYERS = hf(ENGRAM).model.layers.length;

function engram(lookupTables: OffloadConfig['lookupTables']): SimResult {
  return simulate({
    ...hf(ENGRAM),
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: { ...GPU, host: { ram: DDR5_5600_2CH, ssd: NVME_990_PRO_2TB } },
    deployment: { offload: { prefetchAccuracy: 0, lookupTables } },
  });
}

describe(`${ENGRAM} tables: both in RAM / layer 1 in RAM + layer 14 on SSD / both on SSD, b 1, decode @ 8k, p 0`, () => {
  const ram = engram({ tier: 'host_ram' });
  const mixed = engram([
    { tier: 'host_ram', layers: { from: 0, to: 2 } },
    { tier: 'nvme', layers: { from: 2, to: ENGRAM_LAYERS } },
  ]);
  const ssd = engram({ tier: 'nvme' });

  it('the fixture has n-gram tables in layers 1 and 14 only', () => {
    const layers = hf(ENGRAM).model.layers.flatMap((l, i) =>
      l.blocks.some((b) => b.kind === 'ngram_embedding') ? [i] : [],
    );
    expect(layers).toEqual([1, 14]);
  });

  it("each table reads from its rule's tier", () => {
    const sources = mixed.decode.stages[0].layers.flatMap((l) =>
      l.ops.flatMap((o) =>
        o.op.kind === 'transfer' && o.op.group === 'lookup_table'
          ? [[l.layerIndex, o.op.source]]
          : [],
      ),
    );
    expect(sources).toEqual([
      [1, 'host_ram'],
      [14, 'nvme'],
    ]);
  });

  it('decode @ 8k sits between the extremes; the split transfer is the mean (the two tables read the same rows per step)', () => {
    expect(ram.decode.tokensPerSec).toBeGreaterThan(mixed.decode.tokensPerSec);
    expect(mixed.decode.tokensPerSec).toBeGreaterThan(ssd.decode.tokensPerSec);
    const t = (r: SimResult) => r.decode.transfer?.sec ?? 0;
    expectClose(t(mixed), (t(ram) + t(ssd)) / 2, 1e-12);
  });

  it('tier bytes: each table counts on its own tier (the SSD table keeps its pinned scales in host RAM)', () => {
    const bytes = (r: SimResult, tier: string) =>
      r.memory.tiers.find((x) => x.tier === tier)?.breakdown.lookupTables ?? 0;
    const total = bytes(ram, 'host_ram');
    expect(bytes(ram, 'nvme')).toBe(0);
    expect(bytes(mixed, 'host_ram') + bytes(mixed, 'nvme')).toBe(total);
    expect(bytes(ssd, 'host_ram') + bytes(ssd, 'nvme')).toBe(total);
    // Split: table 1 whole + table 14's scales in RAM; all-SSD: both tables' scales in RAM.
    expect(bytes(mixed, 'host_ram')).toBeGreaterThan(total / 2);
    expect(bytes(mixed, 'nvme')).toBeLessThan(bytes(ssd, 'nvme'));
  });

  it('object ≡ [object]; list order never changes a result; [] ≡ absent', () => {
    expect(engram([{ tier: 'host_ram' }])).toEqual(ram);
    expect(
      engram([
        { tier: 'nvme', layers: { from: 2, to: ENGRAM_LAYERS } },
        { tier: 'host_ram', layers: { from: 0, to: 2 } },
      ]),
    ).toEqual(mixed);
    expect(engram([])).toEqual(engram(undefined));
  });
});

// ---------------------------------------------------------------------------------------------------------------
// The one intended change for old (single-object) inputs: n_host

/**
 * Host RAM serves both host-RAM-tier reads and the staging hop of NVMe reads, so a legacy config whose stages stream
 * from DIFFERENT tiers now shares B_ram among them (n_host, docs/simulation-model.md §12 "Sharing"); before, each
 * stage got the full B_ram (granted twice). Qwen3.8-Flash-Next at pp 2, b 1, default p 0.8 (every stage can stream at
 * once): the PLE table (layer 1, stage 0) in host RAM, the experts of layers 24–47 (stage 1) on a fast SSD (200 GB/s, so
 * host-RAM staging bounds the SSD reads). n_host = 2: the expert fetch's staging term doubles, bytes/(B_ram/2) →
 * bytes/(B_ram/4). Configs with one physical off-device tier, or both tiers in the same stage only, are unchanged
 * (n_host = n_tier); the identity pin, the sanity reports and every single-tier test pass without edits.
 */
describe('legacy object config with RAM and SSD on different stages: moved by n_host (intended)', () => {
  const FAST_SSD = { ...NVME_990_PRO_2TB, seqReadBandwidthBytesPerSec: 200e9 };
  const run = (): SimResult =>
    simulate({
      ...hf(QWEN),
      parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 2048, decodeContextLen: 2048 },
      hardware: {
        device: SYNTH_GPU,
        topology: SYNTH_CLUSTER,
        host: { link: hostPcieLink(5, 16), ram: DDR5_5600_2CH, ssd: FAST_SSD },
      },
      deployment: {
        offload: {
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

  it('decode @ 2k: 8.642 tok/s (15.861 tok/s before n_host, which granted B_ram to both stages); the expert fetch is staging-bound at B_ram/4', () => {
    const r = run();
    expectClose(r.decode.tokensPerSec, 8.64216, 1e-5);
    const fetch = r.decode.stages[1].layers
      .flatMap((l) => l.ops)
      .find((o) => o.op.kind === 'transfer' && o.op.group === 'routed_experts');
    if (fetch === undefined || fetch.op.kind !== 'transfer') throw new Error('no expert fetch');
    const tierBytes = fetch.op.reads.reduce(
      (s, x) => s + x.count * Math.ceil(x.bytesEach / 4096) * 4096,
      0,
    );
    // Formula: staging = bytes_tier / (B_ram / (2·n_host)), n_host = 2 (stage 0 reads RAM, stage 1 stages SSD reads)
    expectClose(fetch.transferBandwidthSec, tierBytes / (89.6e9 / 4));
  });
});
