import { describe, expect, it } from 'vitest';
import { SYNTH_UNIFIED } from '../__fixtures__/synthetic';
import { decoderLayer, decoderModel } from '../ir/builders';
import type { QuantConfig } from '../ir/model';
import { TINY_DENSE, TINY_HYBRID } from '../models/reference';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_5600_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import { embeddingResidentWeightBytes, embeddingShardBytes } from '../lower/embedding';
import { lmHeadResidentWeightBytes } from '../lower/head';
import type { LoweringContext, OffloadLowering } from '../lower/lower';
import { stageTierBreakdown, stageWeightBytes, tierMemory } from './memory';
import { simulate } from './simulate';
import type { SimInput } from './types';
import { validateInput } from './validate';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

function input(over: Partial<SimInput['hardware']> = {}): SimInput {
  return {
    model: TINY_DENSE,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
    hardware: { device: SYNTH_UNIFIED, topology: singleDevice(SYNTH_UNIFIED), ...over },
  };
}

describe('T14 unified memory + engines (SYNTH_UNIFIED)', () => {
  it('memCapacityBytes 64e9 → capacity 64e9 and the unified-memory warning', () => {
    const r = simulate(input({ memCapacityBytes: 64e9 }));
    expect(r.memory.kind).toBe('unified');
    expect(r.memory.perStage[0].capacityBytes).toBe(64e9);
    expect(r.warnings.some((w) => w.includes('unified memory — the GPU may use 100%'))).toBe(true);
  });

  it('default capacity is capacityOptionsBytes[0] = 32e9', () => {
    expect(simulate(input()).memory.perStage[0].capacityBytes).toBe(32e9);
  });

  it('memCapacityBytes 50e9 (not an option) is a validation error', () => {
    const errors = validateInput(input({ memCapacityBytes: 50e9 }));
    expect(errors.some((e) => e.includes('memCapacityBytes'))).toBe(true);
    expect(() => simulate(input({ memCapacityBytes: 50e9 }))).toThrow(/memCapacityBytes/);
  });

  it("engineId 'npu' with bf16 weights: fallback warning (no bf16 → vector fp32) and memory at the 0.25e12 cap", () => {
    const r = simulate(input({ engineId: 'npu' }));
    expect(r.warnings.some((w) => w.includes("engine 'npu'") && w.includes('vector fp32'))).toBe(true);
    const ops = r.decode.stages[0].layers.flatMap((l) => l.ops);
    const qkv = ops.find((o) => o.op.name === 'attn.qkv_proj');
    if (qkv === undefined) throw new Error('no qkv');
    // memBW = min(device 0.5e12, npu cap 0.25e12) = 0.25e12 B/s
    expectClose(qkv.memorySec, qkv.bytes.totalBytes / 0.25e12);
    expect(qkv.peakFlopsPerSec).toBe(1e12); // the npu's vector fp32 peak
    expect(qkv.usedInput).toBe('fp32');
  });
});

describe('stage weights', () => {
  it('a tied head at pp = 1 is counted once (only its final norm adds)', () => {
    const tied = decoderModel({
      name: 'tied',
      hidden: 256,
      vocab: 1000,
      tieEmbeddings: true,
      layers: [decoderLayer({ hidden: 256, mixer: { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 64 }, ffn: { kind: 'dense', intermediate: 512 } })],
    });
    const untied = { ...tied, tieEmbeddings: false };
    const ctx: LoweringContext = { phase: 'decode', seqs: 1, qLen: 1, kvLen: 2, quant: BF16, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 } };
    // The difference is the head matrix: 1000 · 256 · 2 B = 512,000 B
    expect(stageWeightBytes(untied, ctx, 0) - stageWeightBytes(tied, ctx, 0)).toBe(1000 * 256 * 2);
    expect(lmHeadResidentWeightBytes(untied, ctx)).toBe(1000 * 256 * 2 + 256 * 2); // + rmsnorm γ
    expect(embeddingResidentWeightBytes(untied, ctx)).toBe(1000 * 256 * 2);
  });

  it('a tied head stored in a different format is a second copy (review P2)', () => {
    const tied = decoderModel({
      name: 'tied',
      hidden: 256,
      vocab: 1000,
      tieEmbeddings: true,
      layers: [decoderLayer({ hidden: 256, mixer: { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 64 }, ffn: { kind: 'dense', intermediate: 512 } })],
    });
    const at = (quant: QuantConfig): LoweringContext => ({ phase: 'decode', seqs: 1, qLen: 1, kvLen: 2, quant, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 } });
    const shared = stageWeightBytes(tied, at(BF16), 0); // layers + bf16 table 512,000 + final norm 512
    // Explicit bf16 head = the default bf16 embedding format: still one shared allocation.
    expect(stageWeightBytes(tied, at({ ...BF16, lmHead: { dtype: 'bf16' } }), 0)).toBe(shared);
    // int4 head: an extra 1000 · 256 · 4 bit / 8 = 128,000 B copy next to the bf16 embedding table.
    expect(stageWeightBytes(tied, at({ ...BF16, lmHead: { dtype: 'int4' } }), 0) - shared).toBe(128_000);
    // Same dtype, different group-quant overhead (bf16 + 0.15625 bit): 1000 · 256 · 16.15625 / 8 = 517,000 B extra.
    expect(stageWeightBytes(tied, at({ ...BF16, lmHead: { dtype: 'bf16', extraBitsPerElem: 0.15625 } }), 0) - shared).toBe(517_000);
    // int4 embeddings with the default head (= the embedding format): shared again; table 1000 · 256 · 0.5 B.
    expect(shared - stageWeightBytes(tied, at({ ...BF16, embeddings: { dtype: 'int4' } }), 0)).toBe(512_000 - 128_000);
  });
});

describe('U-6 memory with routed experts offloaded (TINY_HYBRID on one RTX 5090, host RAM)', () => {
  // W_expert = 3·256·128·2 B = 196,608 B; 8 experts × 4 MoE layers = 6,291,456 B of routed experts per device.
  const W_EXPERT = 196_608;
  const run = (f: number | undefined, prefetch: 'ideal' | 'none' = 'ideal') => {
    const r: SimInput = {
      model: TINY_HYBRID,
      quant: BF16,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
      hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) },
    };
    if (f !== undefined) {
      // 'ideal' = prefetch accuracy 1 (double-buffered staging), 'none' = 0 (single buffer).
      const prefetchAccuracy = prefetch === 'ideal' ? 1 : 0;
      r.deployment = { offload: { prefetchAccuracy, routedExperts: { tier: 'host_ram', residentFraction: f, hitRatio: { kind: 'uniform' } } } };
    }
    return simulate(r);
  };

  it('f .5: device W drops by half the expert bytes; host tier = all expert bytes; staging 2·ceil(M_max)·W_expert', () => {
    const base = run(undefined).memory;
    const m = run(0.5).memory;
    expect(base.perStage[0].weightBytes - m.perStage[0].weightBytes).toBe(6_291_456 / 2);
    expect(m.tiers).toEqual([{ tier: 'host_ram', bytes: 6_291_456, breakdown: { lookupTables: 0, routedExperts: 6_291_456, draftExperts: 0, tokenEmbeddings: 0 }, capacityBytes: 96e9, fits: true }]);
    // Prefill T = 128: U ≈ 8 = E, hits = C = 4, M_max = 4 (decode: U 2, M 1).
    expect(m.perStage[0].stagingBytes).toBe(2 * 4 * W_EXPERT);
    expect(run(0.5, 'none').memory.perStage[0].stagingBytes).toBe(4 * W_EXPERT);
    expect(m.perStage[0].totalBytes).toBe(
      m.perStage[0].weightBytes + m.perStage[0].sequenceStateBytes + m.perStage[0].activationPeakBytes + 2 * 4 * W_EXPERT,
    );
    // maxSequencesPerReplica uses W_s + staging.
    const d = m.perStage[0];
    const perSeq = d.sequenceStateBytes; // B_r = 1
    expect(m.maxSequencesPerReplica).toBe(Math.floor((d.capacityBytes - d.weightBytes - d.activationPeakBytes - d.stagingBytes) / perSeq));
  });

  it('f 1: stagingBytes 0, no TransferOp, the device lines and every rate equal the all-device baseline; tiers and warnings differ', () => {
    const base = run(undefined);
    const r = run(1);
    const [d, b] = [r.memory.perStage[0], base.memory.perStage[0]];
    expect(d.stagingBytes).toBe(0);
    expect(d.weightBytes).toBe(b.weightBytes);
    expect(d.sequenceStateBytes).toBe(b.sequenceStateBytes);
    expect(d.activationPeakBytes).toBe(b.activationPeakBytes);
    expect(d.fits).toBe(b.fits);
    for (const phase of ['decode', 'prefill'] as const) {
      expect(r[phase].stages[0].layers.flatMap((l) => l.ops).some((o) => o.op.kind === 'transfer')).toBe(false);
      expect(r[phase].cycleSec).toBe(base[phase].cycleSec);
      expect(r[phase].tokensPerSec).toBe(base[phase].tokensPerSec);
    }
    expect(r.memory.tiers).toEqual([{ tier: 'host_ram', bytes: 6_291_456, breakdown: { lookupTables: 0, routedExperts: 6_291_456, draftExperts: 0, tokenEmbeddings: 0 }, capacityBytes: 96e9, fits: true }]);
    expect(r.warnings).not.toEqual(base.warnings);
  });
});

// Whole-layer streaming (`denseLayers`) was removed; its unified-memory cases are kept here with routed experts.
describe('offloaded experts on unified memory', () => {
  const experts = (tier: 'host_ram' | 'nvme') =>
    simulate({
      model: TINY_HYBRID,
      quant: BF16,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
      hardware: { device: SYNTH_UNIFIED, topology: singleDevice(SYNTH_UNIFIED) },
      deployment: { offload: { routedExperts: { tier, residentFraction: 0, hitRatio: { kind: 'uniform' } } } },
    });
  const base = simulate({
    model: TINY_HYBRID,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
    hardware: { device: SYNTH_UNIFIED, topology: singleDevice(SYNTH_UNIFIED) },
  });

  it("'host_ram' is the device pool (no transfer, no tier, a note); 'nvme' has no link term or α_link", () => {
    const ram = experts('host_ram');
    expect(ram.decode.cycleSec).toBe(base.decode.cycleSec);
    expect(ram.memory.tiers).toEqual([]);
    expect(ram.memory.perStage[0].weightBytes).toBe(base.memory.perStage[0].weightBytes);
    expect(ram.warnings.some((w) => w.includes('host RAM is the device memory'))).toBe(true);
    const ssd = experts('nvme');
    const fetch = ssd.decode.stages[0].layers.flatMap((l) => l.ops).find((o) => o.op.kind === 'transfer');
    // The default SSD of a device with no internal-drive preset is the 990 PRO: α = 1/22,000 s only (no link hop).
    expectClose(fetch?.transferLatencySec ?? 0, 1 / 22_000);
    expect(ssd.memory.tiers.map((t) => t.tier)).toEqual(['nvme']);
    expect(ssd.warnings.some((w) => w.includes('no SSD given'))).toBe(true);
    expect(ssd.warnings.some((w) => w.includes('no host link given'))).toBe(false);
  });
});

describe('token embedding off-device: stage 0 drops exactly its shard, and the tier holds it', () => {
  // 4 layers, h 256, V 1001 (not divisible by tp 2 or 3), bf16: shard = ceil(1001/tp)·256·2 B.
  const model = decoderModel({
    name: 'emb',
    hidden: 256,
    vocab: 1001,
    tieEmbeddings: false,
    layers: Array.from({ length: 4 }, () =>
      decoderLayer({ hidden: 256, mixer: { kind: 'gqa', heads: 12, kvHeads: 6, headDim: 64 }, ffn: { kind: 'dense', intermediate: 768 } }),
    ),
  });
  const at = (parallel: Partial<LoweringContext['parallel']>, offload?: OffloadLowering): LoweringContext => {
    const c: LoweringContext = {
      phase: 'decode',
      seqs: 1,
      qLen: 1,
      kvLen: 2,
      quant: BF16,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    };
    if (offload !== undefined) c.offload = offload;
    return c;
  };
  const placed = (tier: 'host_ram' | 'nvme'): OffloadLowering => ({ prefetchAccuracy: 0.8, embeddings: { tier } });

  it.each([
    ['host_ram', 1],
    ['nvme', 1],
    ['host_ram', 3],
    ['nvme', 2],
  ] as const)('%s, tp %d: ΔW_0 = tierBytes_0 = ceil(V/tp)·h·bpe', (tier, tp) => {
    const shard = Math.ceil(1001 / tp) * 256 * 2;
    expect(embeddingShardBytes(model, at({ tp }))).toBe(shard);
    expect(stageWeightBytes(model, at({ tp }), 0) - stageWeightBytes(model, at({ tp }, placed(tier)), 0)).toBe(shard);
    const tiers = stageTierBreakdown(model, at({ tp }, placed(tier)), 0);
    expect(tiers[tier]).toEqual({ lookupTables: 0, routedExperts: 0, draftExperts: 0, tokenEmbeddings: shard });
    const other = tier === 'host_ram' ? 'nvme' : 'host_ram';
    expect(tiers[other].tokenEmbeddings).toBe(0);
  });

  it('pp 2: stage 1 is unchanged and places nothing on a tier', () => {
    const base = at({ pp: 2 });
    const ram = at({ pp: 2 }, placed('host_ram'));
    expect(stageWeightBytes(model, ram, 1)).toBe(stageWeightBytes(model, base, 1));
    expect(stageTierBreakdown(model, ram, 1).host_ram.tokenEmbeddings).toBe(0);
    expect(stageWeightBytes(model, base, 0) - stageWeightBytes(model, ram, 0)).toBe(1001 * 256 * 2);
  });

  it('one node of tp 2 × dp 2: host RAM holds tp·dp shards (each stage-0 device its own)', () => {
    const host = { ram: DDR5_5600_2CH, ssd: NVME_990_PRO_2TB, warnings: [] };
    const shard = Math.ceil(1001 / 2) * 256 * 2;
    const [entry] = tierMemory(model, at({ tp: 2, dp: 2 }, placed('host_ram')), 8, host);
    expect(entry).toMatchObject({ tier: 'host_ram', bytes: 4 * shard, fits: true });
    expect(entry.breakdown.tokenEmbeddings).toBe(4 * shard);
  });

  it('a tied head with the embedding off-device keeps its own matrix (defense; validation rejects this input)', () => {
    const tied = { ...model, tieEmbeddings: true };
    // Off-device, nothing is shared: W_0 = layers + the full head, the same as the untied model's.
    expect(stageWeightBytes(tied, at({}, placed('host_ram')), 0)).toBe(stageWeightBytes(model, at({}, placed('host_ram')), 0));
    expect(lmHeadResidentWeightBytes(tied, at({}))).toBe(1001 * 256 * 2 + 256 * 2);
  });
});
