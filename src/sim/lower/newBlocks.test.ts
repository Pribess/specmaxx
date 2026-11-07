// Lowering of the shortlist block kinds on the TINY_* fixtures (made-up dims, see models/reference.ts), with
// hand-derived FLOP and bytes, in prefill and decode, at tp 1 and 2. bf16 everywhere (2 B/element).

import { describe, expect, it } from 'vitest';
import { opBytes, opFlops } from '../ir/metrics';
import type { ParallelConfig, QuantConfig } from '../ir/model';
import type { AttentionOp, CollectiveOp, GemmOp, IndexerOp, Op, RecurrentOp } from '../ir/ops';
import { blockParams } from '../ir/params';
import { modelSpecErrors } from '../ir/validateModel';
import { TINY_CSA, TINY_HC_NGRAM, TINY_INKLING, TINY_KDA, TINY_MLA_SHARED } from '../models/reference';
import { crossStageWarnings } from '../engine/simulate';
import { blockResidentWeightBytes, blockSequenceStateBytes, lowerModel, type LoweringContext } from './lower';
import type { ModelSpec } from '../ir/model';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

/** Prefill: 2 sequences of 64 tokens (T = 128). Decode: 2 sequences at a 256-token context (T = 2). */
function ctx(phase: 'prefill' | 'decode', tp = 1, pp = 1): LoweringContext {
  const parallel: ParallelConfig = { tp, pp, dp: 1, ep: 1 };
  return { phase, seqs: 2, qLen: phase === 'prefill' ? 64 : 1, kvLen: phase === 'prefill' ? 64 : 256, quant: BF16, parallel };
}

/** The ops of layer `i` (with its stage's lowering). */
function layerOps(model: ModelSpec, i: number, c: LoweringContext): Op[] {
  const layer = lowerModel(model, c).stages.flatMap((s) => s.layers).find((l) => l.layerIndex === i);
  if (layer === undefined) throw new Error(`no layer ${i}`);
  return layer.ops;
}
const byName = (ops: Op[], name: string): Op => {
  const op = ops.find((o) => o.name === name);
  if (op === undefined) throw new Error(`no op ${name} in ${ops.map((o) => o.name).join(', ')}`);
  return op;
};
const names = (ops: Op[]): string[] => ops.map((o) => o.name);

describe('fixtures are valid and their resident weights equal params · 2 B at tp 1', () => {
  const models = [TINY_CSA, TINY_HC_NGRAM, TINY_KDA, TINY_MLA_SHARED, TINY_INKLING];
  it.each(models.map((m) => [m.name, m] as const))('%s', (_name, model) => {
    expect(modelSpecErrors(model)).toEqual([]);
    for (const layer of model.layers) {
      for (const block of layer.blocks) {
        // KDA's A_log (vH) and dt_bias (vH·dk) are not modeled as weight reads (§9).
        const kdaVectors = block.kind === 'linear_attention' && block.variant === 'kda' ? block.vHeads * (1 + block.keyDim) : 0;
        expect(blockResidentWeightBytes(block, ctx('decode'))).toBe(2 * (blockParams(block) - kdaVectors));
      }
    }
  });
});

describe('CSA (TINY_CSA: h 128, H 4, hd 32, dr 8, rq 32, G 2 × R_o 16, W 16; r = 2, indexer 2 × 16, top-8 entries)', () => {
  it('owner layer: ops in order, compressor and indexer', () => {
    expect(names(layerOps(TINY_CSA, 1, ctx('decode')))).toEqual([
      'input_norm',
      'csa.q_a',
      'csa.q_norm',
      'csa.q_b',
      'csa.q_rope',
      'csa.wkv',
      'csa.kv_norm',
      'csa.k_rope',
      'csa.wkv_c',
      'csa.wgate_c',
      'csa.kv_pool',
      'csa.compress_norm',
      'csa.compress_rope',
      'idx.q',
      'idx.k',
      'idx.k_norm',
      'idx.w',
      'idx.score',
      'attn.core',
      'csa.o_inverse_rope',
      'csa.wo_a',
      'csa.wo_b',
      'mixer_residual',
      'ffn_norm',
      'mlp.gate_up',
      'mlp.act',
      'mlp.down',
      'ffn_residual',
    ]);
    // consumer (reuse): no compressor, no indexer; Reindex: idx.q, idx.w, scoring only
    expect(names(layerOps(TINY_CSA, 2, ctx('decode'))).filter((n) => n.startsWith('idx') || n.includes('_c'))).toEqual([]);
    expect(names(layerOps(TINY_CSA, 3, ctx('decode'))).filter((n) => n.startsWith('idx'))).toEqual(['idx.q', 'idx.w', 'idx.score']);
  });

  it('prefill attention: window + compressed entries in one softmax', () => {
    const att = byName(layerOps(TINY_CSA, 1, ctx('prefill')), 'attn.core') as AttentionOp;
    // window 16, qLen 64: 16·17/2 + 48·16 = 904 pairs; entries r = 2, top-8: j0 = 16 → 136/2 + 48·8 = 452
    // F = 2·seqs 2·H 4·(904 + 452)·(32 + 32) = 1,388,544
    expect(opFlops(att)).toBe(1_388_544);
    // tp 2 halves the heads
    expect(opFlops(byName(layerOps(TINY_CSA, 1, ctx('prefill', 2)), 'attn.core'))).toBe(694_272);
    // KV written per token: window hd 32 + compressed hd/r 16 → 2·64·(32 + 16)·2 = 12,288
    expect(opBytes(att).kvWriteBytes).toBe(12_288);
  });

  it('decode attention and indexer bytes (owner, consumer, Reindex)', () => {
    const owner = layerOps(TINY_CSA, 1, ctx('decode'));
    const att = byName(owner, 'attn.core') as AttentionOp;
    // pairs 16 (window) + 8 (top-8 of 128 entries); F = 2·2·4·24·64 = 24,576
    expect(opFlops(att)).toBe(24_576);
    // read: 2 seqs·(16 window tokens + 8 entries)·32·2 = 3072; write: 2·(32 + 16)·2 = 192
    expect(opBytes(att).kvReadBytes).toBe(3072);
    expect(opBytes(att).kvWriteBytes).toBe(192);
    const idx = byName(owner, 'idx.score') as IndexerOp;
    // 256/2 = 128 keys: F = 2·2·2 heads·128·16 = 16,384; keys read 2·128·16·2 = 8192; write 2·(16/2)·2 = 32
    expect(opFlops(idx)).toBe(16_384);
    expect(opBytes(idx).kvReadBytes).toBe(8192);
    expect(opBytes(idx).kvWriteBytes).toBe(32);
    // consumer: only the window vector is written
    expect(opBytes(byName(layerOps(TINY_CSA, 2, ctx('decode')), 'attn.core')).kvWriteBytes).toBe(128);
    // Reindex: at most 32 keys scored, nothing written
    const reindex = byName(layerOps(TINY_CSA, 3, ctx('decode')), 'idx.score') as IndexerOp;
    expect(opFlops(reindex)).toBe(4096); // 2·2·2·32·16
    expect(opBytes(reindex).kvWriteBytes).toBe(0);
  });

  it('grouped output projection and TP sharding', () => {
    const woA = byName(layerOps(TINY_CSA, 0, ctx('decode', 2)), 'csa.wo_a') as GemmOp;
    // Gl = 2/2 = 1 group per rank: batch 1, n = 16, k = H·hd/G = 64
    expect([woA.batch, woA.n, woA.k]).toEqual([1, 16, 64]);
    const qB = byName(layerOps(TINY_CSA, 0, ctx('decode', 2)), 'csa.q_b') as GemmOp;
    expect(qB.n).toBe(64); // Hl·hd = 2·32
  });

  it('sequence state: window, owned compressed cache and tail, owned indexer keys', () => {
    const state = (i: number): number => {
      const block = TINY_CSA.layers[i].blocks[1];
      return blockSequenceStateBytes(block, ctx('decode'), 256);
    };
    // window min(256, 16)·32·2 = 1024
    expect(state(0)).toBe(1024);
    // + ceil(256/2)·32·2 = 8192 + fp32 tail 2·r 2·32·4 = 512 + keys ceil(256/2)·16·2 = 4096
    expect(state(1)).toBe(1024 + 8192 + 512 + 4096);
    expect(state(2)).toBe(1024);
    expect(state(3)).toBe(1024);
  });
});

describe('hyper-connections and n-gram (TINY_HC_NGRAM: h 64, N 2, fn 128 → 8, Sinkhorn 4; table 1000 × 8, 4 rows/token)', () => {
  it('layer layout: ngram, hc_pre, norm, attention, hc_post, …, final collapse in the last layer', () => {
    const ops = names(layerOps(TINY_HC_NGRAM, 1, ctx('decode')));
    expect(ops.slice(0, 7)).toEqual([
      'engram.gather',
      'engram.wkv',
      'engram.gate',
      'mixer_hc_pre.norm',
      'mixer_hc_pre.fn',
      'mixer_hc_pre.sinkhorn',
      'mixer_hc_pre.collapse',
    ]);
    expect(ops.slice(-2)).toEqual(['final_hc_pre.norm', 'final_hc_pre.collapse']);
  });

  it('prefill hc FLOP and bytes (T = 128)', () => {
    const ops = layerOps(TINY_HC_NGRAM, 0, ctx('prefill'));
    // sinkhorn: T·N² = 512 elems × 4·4 = 8192 FLOP
    expect(opFlops(byName(ops, 'mixer_hc_pre.sinkhorn'))).toBe(8192);
    // collapse: read T·N·D·2 = 32,768, write T·D·2 = 16,384, FLOP T·D·2N = 32,768; weights 11 vectors · 2 B
    const collapse = opBytes(byName(ops, 'mixer_hc_pre.collapse'));
    expect([collapse.activationReadBytes, collapse.activationWriteBytes, collapse.weightBytes]).toEqual([32_768, 16_384, 22]);
    // expand (mixing): T·N·D·(2N + 2) = 16,384·6 = 98,304 FLOP; read (T·N·D + T·D)·2 = 49,152, write 32,768
    const expand = byName(ops, 'mixer_hc_post.expand');
    expect(opFlops(expand)).toBe(98_304);
    expect(opBytes(expand).activationReadBytes).toBe(49_152);
    // stats-only norm reads the streams once and writes nothing
    expect(opBytes(byName(ops, 'mixer_hc_pre.norm')).activationWriteBytes).toBe(0);
  });

  it('n-gram gather is row-parallel over tp; the rest is replicated', () => {
    const ops = layerOps(TINY_HC_NGRAM, 1, ctx('prefill', 2));
    const gather = byName(ops, 'engram.gather');
    if (gather.kind !== 'embedding') throw new Error('expected an embedding op');
    // T·g = 512 rows of 8: 512·8·2 = 8192 B gathered; 500 rows per rank
    expect([gather.tokens, gather.vocabShard, opBytes(gather).weightBytes]).toEqual([512, 500, 8192]);
    // all-reduce of the gathered rows: T·g·d·2 = 8192 B
    expect((byName(ops, 'engram.tp_all_reduce') as CollectiveOp).messageBytes).toBe(8192);
    // resident at tp 2: table 500·8·2 = 8000 + wkv 32·192·2 = 12,288 + gate vectors 256·2 = 512
    expect(blockResidentWeightBytes(TINY_HC_NGRAM.layers[1].blocks[0], ctx('decode', 2))).toBe(20_800);
  });

  it('PP send carries all N streams', () => {
    const send = lowerModel(TINY_HC_NGRAM, ctx('prefill', 1, 2)).stages[0].sendToNext;
    // T·N·h·2 = 128·2·64·2
    expect(send?.messageBytes).toBe(32_768);
  });
});

describe('KDA and NoPE MLA with a k-pool indexer (TINY_KDA)', () => {
  it('KDA projections, gates and sharding', () => {
    const ops = layerOps(TINY_KDA, 0, ctx('decode'));
    expect(names(ops).slice(1, 9)).toEqual([
      'lin.qkv',
      'lin.f_a',
      'lin.f_b',
      'lin.g_a',
      'lin.g_b',
      'lin.conv1d',
      'lin.recurrent',
      'lin.out_norm',
    ]);
    // qkv n = 2·2·32 + 2·32 + 2 (β) = 194; tp 2: 2·1·32 + 32 + 1 = 97
    expect((byName(ops, 'lin.qkv') as GemmOp).n).toBe(194);
    expect((byName(layerOps(TINY_KDA, 0, ctx('decode', 2)), 'lin.qkv') as GemmOp).n).toBe(97);
    expect((byName(layerOps(TINY_KDA, 0, ctx('decode', 2)), 'lin.f_a') as GemmOp).n).toBe(16); // replicated rank
    const rec = byName(ops, 'lin.recurrent') as RecurrentOp;
    expect(rec.variant).toBe('kda');
    // read: q,k 2·T 2·2·32·2 = 512; v 2·2·32·2 = 256; gates T·vH·(dk + 1)·2 = 2·2·33·2 = 264
    expect(opBytes(rec).activationReadBytes).toBe(1032);
    // step c_s 7: F = 2·2 heads·7·32·32 = 28,672
    expect(opFlops(rec)).toBe(28_672);
  });

  it('NoPE MLA emits no rope; the pooled indexer scores T/4 keys', () => {
    const ops = layerOps(TINY_KDA, 1, ctx('prefill'));
    expect(names(ops).some((n) => n.includes('rope'))).toBe(false);
    expect(names(ops).filter((n) => n.startsWith('idx'))).toEqual(['idx.q', 'idx.k', 'idx.k_norm', 'idx.pool_gate', 'idx.kv_pool', 'idx.w', 'idx.score']);
    const idx = byName(ops, 'idx.score') as IndexerOp;
    // causal pairs (64·65/2)/4 = 520: F = 2·2·2·520·16 = 66,560; top-16 tokens = 4 pools → indices 2·64·4·4 = 2048
    expect(opFlops(idx)).toBe(66_560);
    expect(opBytes(idx).activationWriteBytes).toBe(2048);
    // key cache written: hd/p = 4 elements per token → 2·64·4·2 = 1024
    expect(opBytes(idx).kvWriteBytes).toBe(1024);
    // state: latent 256·32·2 = 16,384 + pooled keys ceil(256/4)·16·2 = 2048
    expect(blockSequenceStateBytes(TINY_KDA.layers[1].blocks[1], ctx('decode'), 256)).toBe(18_432);
  });
});

describe('cross-layer sharing (TINY_MLA_SHARED: H 4, rkv 32, dn 16, dr 8, dv 16; indexer 2 × 16, top-16)', () => {
  it('reuse emits no indexer ops; a latent consumer has no kv_a and writes no cache', () => {
    const reuse = layerOps(TINY_MLA_SHARED, 1, ctx('decode'));
    expect(names(reuse).some((n) => n.startsWith('idx'))).toBe(false);
    expect((byName(reuse, 'attn.core') as AttentionOp).kvSelection).toEqual({ kind: 'topk', topK: 16 });
    const consumer = layerOps(TINY_MLA_SHARED, 2, ctx('decode', 2));
    expect(names(consumer).some((n) => n.startsWith('mla.kv_a'))).toBe(false);
    // rope only on q: T·Hl·dr = 2·2·8 = 32 elements
    if (byName(consumer, 'mla.rope').kind !== 'elementwise') throw new Error('expected rope');
    expect(opFlops(byName(consumer, 'mla.rope'))).toBe(96);
    expect(opBytes(byName(consumer, 'attn.core')).kvWriteBytes).toBe(0);
  });

  it('sequence state: owner latent + keys, reuse latent only, consumer nothing', () => {
    const state = (i: number): number => blockSequenceStateBytes(TINY_MLA_SHARED.layers[i].blocks[1], ctx('decode'), 256);
    expect(state(0)).toBe(256 * 40 * 2 + 256 * 16 * 2);
    expect(state(1)).toBe(256 * 40 * 2);
    expect(state(2)).toBe(0);
  });

  it('pipeline stages split a source from its consumer: one warning per reference', () => {
    expect(crossStageWarnings(TINY_MLA_SHARED, 1)).toEqual([]);
    // pp 3: one layer per stage → layer 1 (top-k of 0), layer 2 (kv cache of 0, top-k of 0)
    expect(crossStageWarnings(TINY_MLA_SHARED, 3)).toHaveLength(3);
  });
});

describe('Inkling relative bias, K/V convs and short convs (TINY_INKLING: H 4, KV 2, hd 32, relDim 8)', () => {
  it('prefill FLOP (T = 128) at tp 1 and tp 2', () => {
    const ops = layerOps(TINY_INKLING, 1, ctx('prefill'));
    // kv_conv (+ residual) over T·2·KVl·hd = 128·128 elems × (2·4 + 1) FLOP = 147,456; state written 2·128·3·2 = 1536
    expect(opFlops(byName(ops, 'attn.kv_conv'))).toBe(147_456);
    expect(opBytes(byName(ops, 'attn.kv_conv')).stateWriteBytes).toBe(1536);
    // rel_table: m = T·H = 512, n = extent 64, k = relDim 8 → 2·512·64·8 = 524,288
    expect(opFlops(byName(ops, 'attn.rel_table'))).toBe(524_288);
    // short conv after the branch: 128·128·(2·4 + 1) = 147,456
    expect(opFlops(byName(ops, 'mixer_short_conv'))).toBe(147_456);
    expect(names(ops).some((n) => n.includes('rope'))).toBe(false);
    const tp2 = layerOps(TINY_INKLING, 1, ctx('prefill', 2));
    expect((byName(tp2, 'attn.rel_proj') as GemmOp).n).toBe(16); // Hl·relDim = 2·8
    expect(opFlops(byName(tp2, 'attn.kv_conv'))).toBe(73_728); // KVl 1
  });

  it('decode sequence state: window KV + K/V conv state; short conv state', () => {
    const blocks = TINY_INKLING.layers[0].blocks;
    // window min(256, 32)·2·2·32·2 = 8192 + K/V conv 3·2·2·32·2 = 768
    expect(blockSequenceStateBytes(blocks[1], ctx('decode'), 256)).toBe(8960);
    expect(blocks[2].kind).toBe('short_conv');
    expect(blockSequenceStateBytes(blocks[2], ctx('decode'), 256)).toBe(3 * 128 * 2);
    // decode reads the prior conv state
    expect(opBytes(byName(layerOps(TINY_INKLING, 0, ctx('decode')), 'attn.kv_conv')).stateReadBytes).toBe(2 * 3 * 128 * 2);
  });
});
