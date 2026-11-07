import { describe, expect, it } from 'vitest';
import { SYNTH_UNIFIED } from '../__fixtures__/synthetic';
import { decoderLayer, decoderModel } from '../ir/builders';
import type { QuantConfig } from '../ir/model';
import { TINY_DENSE } from '../models/reference';
import { singleDevice } from '../hw/presets';
import { embeddingResidentWeightBytes } from '../lower/embedding';
import { lmHeadResidentWeightBytes } from '../lower/head';
import type { LoweringContext } from '../lower/lower';
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
