/**
 * The token embedding as a 1-gram lookup table off the device (docs/simulation-model.md §12 "Token embedding
 * off-device"): its row fetch, read classes, prefetch share, and resident bytes.
 */
import { describe, expect, it } from 'vitest';
import { decoderLayer, decoderModel } from '../ir/builders';
import type { QuantConfig } from '../ir/model';
import { tpAllReduce } from './common';
import {
  embeddingFetchOp,
  embeddingResidentWeightBytes,
  embeddingShardBytes,
  lowerEmbedding,
} from './embedding';
import type { LoweringContext, OffloadLowering } from './lower';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
/** A Qwen-27B-sized embedding: h 5120, V 248,320 (not divisible by 3), bf16 → 10,240 B per row. */
const MODEL = decoderModel({
  name: 'emb',
  hidden: 5120,
  vocab: 248_320,
  tieEmbeddings: false,
  layers: [
    decoderLayer({
      hidden: 5120,
      mixer: { kind: 'gqa', heads: 40, kvHeads: 8, headDim: 128 },
      ffn: { kind: 'dense', intermediate: 17_408 },
    }),
  ],
});

function ctx(
  phase: 'prefill' | 'decode',
  opts: { tp?: number; offload?: OffloadLowering; quant?: QuantConfig } = {},
): LoweringContext {
  const T = phase === 'prefill' ? 8192 : 1;
  const c: LoweringContext = {
    phase,
    seqs: 1,
    qLen: T,
    kvLen: phase === 'prefill' ? 8192 : 8193,
    quant: opts.quant ?? BF16,
    parallel: { tp: opts.tp ?? 1, pp: 1, dp: 1, ep: 1 },
  };
  if (opts.offload !== undefined) c.offload = opts.offload;
  return c;
}
const RAM: OffloadLowering = { prefetchAccuracy: 0.8, embeddings: { tier: 'host_ram' } };

describe('lowerEmbedding', () => {
  it('without an off-device embedding, the ops are the gather and the TP all-reduce, as before', () => {
    for (const tp of [1, 2]) {
      const c = ctx('prefill', { tp });
      expect(lowerEmbedding(MODEL, c)).toEqual([
        {
          kind: 'embedding',
          name: 'embed',
          tokens: 8192,
          hidden: 5120,
          vocabShard: Math.ceil(248_320 / tp),
          weight: { dtype: 'bf16' },
          outputDtype: 'bf16',
        },
        ...tpAllReduce('embed.tp_all_reduce', 8192, 5120, c),
      ]);
    }
    // Another group's placement does not place the embedding.
    const other: OffloadLowering = {
      prefetchAccuracy: 0.8,
      lookupTables: [{ tier: 'nvme', layers: { from: 0, to: 1 }, scaleLayout: 'pinned_host' }],
    };
    expect(lowerEmbedding(MODEL, ctx('prefill', { offload: other }))).toEqual(
      lowerEmbedding(MODEL, ctx('prefill')),
    );
  });

  it('host RAM, tp 1, prefill T 8192: the row fetch comes first, 8192 rows of 10,240 B = 83,886,080 B, q = 1', () => {
    const ops = lowerEmbedding(MODEL, ctx('prefill', { offload: RAM }));
    expect(ops.map((o) => o.name)).toEqual(['embed.row_fetch', 'embed']);
    expect(ops[0]).toEqual({
      kind: 'transfer',
      name: 'embed.row_fetch',
      group: 'token_embedding',
      source: 'host_ram',
      reads: [{ count: 8192, bytesEach: 10_240 }],
      usefulBytes: 83_886_080,
      prefetchedFraction: 1,
    });
    // The gather is unchanged: it reads the staged rows.
    expect(ops[1]).toEqual(lowerEmbedding(MODEL, ctx('prefill'))[0]);
  });

  it('decode: q = 0 at any p (the row is the token just sampled); p = 0: q = 0 in prefill too', () => {
    for (const p of [0.8, 1]) {
      const op = embeddingFetchOp(
        'embed',
        1,
        5120,
        ctx('decode', { offload: { ...RAM, prefetchAccuracy: p } }),
      );
      expect(op).toMatchObject({ reads: [{ count: 1, bytesEach: 10_240 }], prefetchedFraction: 0 });
    }
    const off = embeddingFetchOp(
      'embed',
      8192,
      5120,
      ctx('prefill', { offload: { ...RAM, prefetchAccuracy: 0 } }),
    );
    expect(off?.prefetchedFraction).toBe(0);
  });

  it('nvme: the source is the SSD; no rows, no op', () => {
    const nvme: OffloadLowering = { prefetchAccuracy: 0.8, embeddings: { tier: 'nvme' } };
    expect(embeddingFetchOp('embed', 1, 5120, ctx('decode', { offload: nvme }))?.source).toBe(
      'nvme',
    );
    expect(embeddingFetchOp('embed', 0, 5120, ctx('decode', { offload: nvme }))).toBeUndefined();
    expect(embeddingFetchOp('embed', 1, 5120, ctx('decode'))).toBeUndefined();
  });

  it('tp 3 (V 248,320 not divisible by 3): R = T/3 rows per device; shard = ceil(V/3)·h·2', () => {
    const op = embeddingFetchOp('embed', 8192, 5120, ctx('prefill', { tp: 3, offload: RAM }));
    expect(op?.reads).toEqual([{ count: 8192 / 3, bytesEach: 10_240 }]);
    expect(op?.usefulBytes).toBeCloseTo((8192 / 3) * 10_240, 6);
    // Formula: ceil(248,320/3) = 82,774 rows · 5120 · 2 B
    expect(embeddingShardBytes(MODEL, ctx('prefill', { tp: 3 }))).toBe(82_774 * 5120 * 2);
  });

  it('a block-scaled format (mxfp8) reads values and scales separately: [{R, h}, {R, h/32}]', () => {
    const quant: QuantConfig = { ...BF16, embeddings: { dtype: 'mxfp8' } };
    const op = embeddingFetchOp('embed', 8192, 5120, ctx('prefill', { offload: RAM, quant }));
    // valueBytes = 5120·8/8 = 5120 B; scaleBytes = 5120·(8/32)/8 = 160 B; rowBytes = 5280 B
    expect(op?.reads).toEqual([
      { count: 8192, bytesEach: 5120 },
      { count: 8192, bytesEach: 160 },
    ]);
    expect(op?.usefulBytes).toBe(8192 * 5280);
  });

  it('resident bytes: the shard on the device, 0 when placed off-device', () => {
    expect(embeddingResidentWeightBytes(MODEL, ctx('decode'))).toBe(2_542_796_800);
    expect(embeddingShardBytes(MODEL, ctx('decode', { offload: RAM }))).toBe(2_542_796_800);
    expect(embeddingResidentWeightBytes(MODEL, ctx('decode', { offload: RAM }))).toBe(0);
  });
});
