import { describe, expect, it } from 'vitest';
import { SYNTH_UNIFIED } from '../__fixtures__/synthetic';
import { decoderLayer, decoderModel } from '../ir/builders';
import type { QuantConfig } from '../ir/model';
import { TINY_DENSE, TINY_HYBRID } from '../models/reference';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { DeviceSpec } from '../hw/types';
import { embeddingResidentWeightBytes } from '../lower/embedding';
import { lmHeadResidentWeightBytes } from '../lower/head';
import { layerWeightBytesOnDevice, type LoweringContext } from '../lower/lower';
import { stageWeightBytes } from './memory';
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
    expect(r.warnings.some((w) => w.includes('shared with the CPU/OS'))).toBe(true);
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
  const run = (f: number | undefined, offloadOverlap: 'ideal' | 'none' = 'ideal') => {
    const r: SimInput = {
      model: TINY_HYBRID,
      quant: BF16,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
      hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) },
      options: { offloadOverlap },
    };
    if (f !== undefined) {
      r.deployment = { offload: { routedExperts: { tier: 'host_ram', residentFraction: f, hitRatio: { kind: 'uniform' } } } };
    }
    return simulate(r);
  };

  it('f .5: device W drops by half the expert bytes; host tier = all expert bytes; staging 2·ceil(M_max)·W_expert', () => {
    const base = run(undefined).memory;
    const m = run(0.5).memory;
    expect(base.perStage[0].weightBytes - m.perStage[0].weightBytes).toBe(6_291_456 / 2);
    expect(m.tiers).toEqual([{ tier: 'host_ram', bytes: 6_291_456, capacityBytes: 96e9, fits: true }]);
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
    expect(r.memory.tiers).toEqual([{ tier: 'host_ram', bytes: 6_291_456, capacityBytes: 96e9, fits: true }]);
    expect(r.warnings).not.toEqual(base.warnings);
  });
});

describe('dense layers offloaded and unified memory', () => {
  const dense = (device: DeviceSpec, tier: 'host_ram' | 'nvme') =>
    simulate({
      model: TINY_DENSE,
      quant: BF16,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
      hardware: { device, topology: singleDevice(device) },
      deployment: { offload: { denseLayers: { tier, layers: { from: 1, to: 2 } } } },
    });
  const ctx: LoweringContext = { phase: 'decode', seqs: 1, qLen: 1, kvLen: 2, quant: BF16, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 } };
  const W_LAYER = layerWeightBytesOnDevice(TINY_DENSE.layers[1], ctx);

  it('a dense layer on host RAM: resident 0, streamed whole every step (reads [{1, W_layer}]), host tier W_layer, staging 2·W_layer', () => {
    const r = dense(RTX_5090, 'host_ram');
    const base = simulate({ ...input(), hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) } });
    expect(base.memory.perStage[0].weightBytes - r.memory.perStage[0].weightBytes).toBe(W_LAYER);
    expect(r.memory.tiers).toEqual([{ tier: 'host_ram', bytes: W_LAYER, capacityBytes: 96e9, fits: true }]);
    expect(r.memory.perStage[0].stagingBytes).toBe(2 * W_LAYER);
    for (const phase of [r.decode, r.prefill]) {
      const fetch = phase.stages[0].layers.find((l) => l.layerIndex === 1)?.ops[0];
      expect(fetch?.op).toEqual({
        kind: 'transfer',
        name: 'layer.fetch',
        group: 'dense_layer',
        source: 'host_ram',
        reads: [{ count: 1, bytesEach: W_LAYER }],
        usefulBytes: W_LAYER,
      });
      expect(phase.stages[0].layers.find((l) => l.layerIndex === 0)?.ops.some((o) => o.op.kind === 'transfer')).toBe(false);
    }
  });

  it("unified memory: 'host_ram' is the device pool (no transfer, no tier, a note); 'nvme' has no link term or α_link", () => {
    const base = simulate(input());
    const ram = dense(SYNTH_UNIFIED, 'host_ram');
    expect(ram.decode.cycleSec).toBe(base.decode.cycleSec);
    expect(ram.memory.tiers).toEqual([]);
    expect(ram.memory.perStage[0].weightBytes).toBe(base.memory.perStage[0].weightBytes);
    expect(ram.warnings.some((w) => w.includes('host RAM is the device memory'))).toBe(true);
    const ssd = dense(SYNTH_UNIFIED, 'nvme');
    const fetch = ssd.decode.stages[0].layers.find((l) => l.layerIndex === 1)?.ops[0];
    // The default SSD of a device with no internal-drive preset is the 990 PRO: α = 1/22,000 s only (no link hop).
    expectClose(fetch?.transferLatencySec ?? 0, 1 / 22_000);
    expect(ssd.memory.tiers.map((t) => t.tier)).toEqual(['nvme']);
    expect(ssd.warnings.some((w) => w.includes('no SSD given'))).toBe(true);
    expect(ssd.warnings.some((w) => w.includes('no host link given'))).toBe(false);
  });
});
