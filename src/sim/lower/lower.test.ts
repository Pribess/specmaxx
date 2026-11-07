import { describe, expect, it } from 'vitest';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  lowerModel,
  stageLayerRange,
  type LoweringContext,
} from './lower';
import type { Op } from '../ir/ops';
import type { ParallelConfig, QuantConfig } from '../ir/model';
import { decoderLayer, decoderModel } from '../ir/builders';
import { REF_DENSE_8B } from '../models/reference';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

function ctx(phase: 'prefill' | 'decode', parallel: Partial<ParallelConfig> = {}, seqs = 1): LoweringContext {
  const qLen = phase === 'prefill' ? 128 : 1;
  return {
    phase,
    seqs,
    qLen,
    kvLen: phase === 'prefill' ? 128 : 1024,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
  };
}

/** A compact label per op: gemm/attention/collective by name, elementwise by fn. */
function label(op: Op): string {
  switch (op.kind) {
    case 'elementwise':
      return op.fn;
    case 'collective':
      return op.collective;
    default:
      return op.name;
  }
}

const allOps = (m: ReturnType<typeof lowerModel>): Op[] => m.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));

describe('T4 dense structure (REF_DENSE_8B)', () => {
  it('tp 1 decode: 36 identical layers of 12 ops, plus embed and head', () => {
    const lowered = lowerModel(REF_DENSE_8B, ctx('decode'));
    expect(lowered.stages).toHaveLength(1);
    const layers = lowered.stages[0].layers;
    expect(layers.map((l) => l.name)[0]).toBe('embed');
    expect(layers.at(-1)?.name).toBe('head');
    expect(layers).toHaveLength(38); // embed + 36 + head
    for (const layer of layers.slice(1, -1)) {
      expect(layer.ops.map(label)).toEqual([
        'rmsnorm',
        'attn.qkv_proj',
        'qk_norm',
        'rope',
        'attn.core',
        'attn.o_proj',
        'residual_add',
        'rmsnorm',
        'mlp.gate_up',
        'silu_mul',
        'mlp.down',
        'residual_add',
      ]);
    }
  });

  it('tp 8: 2 all-reduces per layer + 1 for the embedding, KV heads sharded 8/8 = 1', () => {
    const ops = allOps(lowerModel(REF_DENSE_8B, ctx('decode', { tp: 8 })));
    expect(ops.filter((o) => o.kind === 'collective' && o.collective === 'all_reduce')).toHaveLength(73); // 36·2 + 1
    const qkv = ops.find((o) => o.name === 'attn.qkv_proj');
    expect(qkv?.kind === 'gemm' && qkv.n).toBe(768); // (32/8 + 2·1)·128
    const ar = ops.find((o) => o.kind === 'collective');
    expect(ar?.kind === 'collective' && ar.messageBytes).toBe(8192); // T 1 · h 4096 · 2 B
  });

  it('tp 16 replicates KV heads (KVl = 1)', () => {
    const ops = allOps(lowerModel(REF_DENSE_8B, ctx('decode', { tp: 16 })));
    const attn = ops.find((o) => o.kind === 'attention');
    expect(attn?.kind === 'attention' && [attn.heads, attn.kvHeads]).toEqual([2, 1]); // 32/16; 8 < 16 → 1
  });

  it('pp 2 splits layers 0–17 / 18–35 with a send only from stage 0', () => {
    expect(stageLayerRange(36, 2, 0)).toEqual([0, 18]);
    expect(stageLayerRange(36, 2, 1)).toEqual([18, 36]);
    const lowered = lowerModel(REF_DENSE_8B, ctx('decode', { pp: 2 }, 4));
    const [s0, s1] = lowered.stages;
    expect(s0.layers.map((l) => l.layerIndex)).toEqual([null, ...Array.from({ length: 18 }, (_, i) => i)]);
    expect(s1.layers.map((l) => l.layerIndex)).toEqual([...Array.from({ length: 18 }, (_, i) => i + 18), null]);
    expect(s0.sendToNext?.messageBytes).toBe(32_768); // T 4 · 4096 · 2 B
    expect(s1.sendToNext).toBeUndefined();
    expect(() => lowerModel(REF_DENSE_8B, ctx('decode', { pp: 37 }))).toThrow();
  });

  it('prefill computes logits for the last position only (lm_head m = seqs)', () => {
    const ops = allOps(lowerModel(REF_DENSE_8B, ctx('prefill', {}, 3)));
    const head = ops.find((o) => o.name === 'lm_head');
    expect(head?.kind === 'gemm' && [head.m, head.n, head.k]).toEqual([3, 151936, 4096]);
    const embed = ops.find((o) => o.kind === 'embedding');
    expect(embed?.kind === 'embedding' && embed.tokens).toBe(384); // 3 seqs · 128
  });

  it('heads not divisible by tp throws', () => {
    expect(() => lowerModel(REF_DENSE_8B, ctx('decode', { tp: 3 }))).toThrow();
  });

  it('sliding window → kvSelection window, KV cache capped at the window', () => {
    const model = decoderModel({
      name: 'swa',
      hidden: 256,
      vocab: 1000,
      tieEmbeddings: true,
      layers: [
        decoderLayer({
          hidden: 256,
          mixer: { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 64, slidingWindow: 512 },
          ffn: { kind: 'dense', intermediate: 512 },
        }),
      ],
    });
    const attn = allOps(lowerModel(model, ctx('decode'))).find((o) => o.kind === 'attention');
    expect(attn?.kind === 'attention' && attn.kvSelection).toEqual({ kind: 'window', windowTokens: 512 });
    const block = model.layers[0].blocks[1];
    // min(4096, 512) tokens · (2·2·64 = 256) elems · 2 B
    expect(blockSequenceStateBytes(block, ctx('decode'), 4096)).toBe(262_144);
  });

  it('sparse GQA emits the indexer (q from h) and a top-k selection', () => {
    const model = decoderModel({
      name: 'dsa-gqa',
      hidden: 256,
      vocab: 1000,
      tieEmbeddings: false,
      layers: [
        decoderLayer({
          hidden: 256,
          mixer: { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 64, sparse: { kind: 'indexer', indexer: { heads: 2, headDim: 16, topK: 32 } } },
          ffn: { kind: 'dense', intermediate: 512 },
        }),
      ],
    });
    const ops = allOps(lowerModel(model, ctx('decode')));
    const idxQ = ops.find((o) => o.name === 'idx.q');
    expect(idxQ?.kind === 'gemm' && [idxQ.n, idxQ.k]).toEqual([32, 256]); // heads_I·hd_I = 2·16; k = h
    expect(ops.some((o) => o.kind === 'indexer')).toBe(true);
    const attn = ops.find((o) => o.kind === 'attention');
    expect(attn?.kind === 'attention' && attn.kvSelection).toEqual({ kind: 'topk', topK: 32 });
    // KV 4096·2·2·64·2 = 2,097,152 + indexer keys 4096·16·2 = 131,072
    expect(blockSequenceStateBytes(model.layers[0].blocks[1], ctx('decode'), 4096)).toBe(2_228_224);
  });
});

describe('resident weights and sequence state (REF_DENSE_8B)', () => {
  const layer = REF_DENSE_8B.layers[0];
  it('one layer resident at tp 1 = params · 2 B', () => {
    const bytes = layer.blocks.reduce((s, b) => s + blockResidentWeightBytes(b, ctx('decode')), 0);
    expect(bytes).toBe(385_892_864); // 192,946,432 params · 2 B (all bf16)
  });
  it('KV per sequence per layer', () => {
    const attn = layer.blocks[1];
    expect(blockSequenceStateBytes(attn, ctx('decode'), 1024)).toBe(4_194_304); // 1024·2·8·128·2
    expect(blockSequenceStateBytes(attn, ctx('decode', { tp: 8 }), 1024)).toBe(524_288); // KVl 1: 1024·2·128·2
    expect(blockSequenceStateBytes(layer.blocks[5], ctx('decode'), 1024)).toBe(0); // mlp
  });
});
