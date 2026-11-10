import { describe, expect, it } from 'vitest';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  lowerBlock,
  type LoweringContext,
} from './lower';
import type { MlaAttentionBlock } from '../ir/model';
import type { AttentionOp, GemmOp, IndexerOp, Op } from '../ir/ops';
import { blockParams } from '../ir/params';
import { TINY_MLA_SPARSE } from '../models/reference';

const block = TINY_MLA_SPARSE.layers[0].blocks[1];
if (block.kind !== 'mla_attention') throw new Error('expected an MLA block');
const MLA: MlaAttentionBlock = block;
const { sparse: _sparse, ...denseFields } = MLA;
void _sparse;
const MLA_DENSE: MlaAttentionBlock = denseFields;

function ctx(phase: 'prefill' | 'decode', seqs = 1, tp = 1): LoweringContext {
  return {
    phase,
    seqs,
    qLen: phase === 'prefill' ? 512 : 1,
    kvLen: phase === 'prefill' ? 512 : 1000,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp, pp: 1, dp: 1, ep: 1 },
  };
}

const find = <K extends Op['kind']>(ops: Op[], kind: K): Extract<Op, { kind: K }> | undefined =>
  ops.find((o): o is Extract<Op, { kind: K }> => o.kind === kind);
const gemm = (ops: Op[], name: string): GemmOp | undefined =>
  ops.find((o): o is GemmOp => o.kind === 'gemm' && o.name === name);

describe('T4c TINY_MLA_SPARSE (heads 8, rq 128, rkv 64, dn 32, dr 16, dv 32; indexer 4×32, top-64)', () => {
  it('decode uses the absorbed form on the latent', () => {
    const ops = lowerBlock(MLA, ctx('decode'));
    const attn = find(ops, 'attention') as AttentionOp;
    expect([attn.heads, attn.kvHeads, attn.qkHeadDim, attn.vHeadDim]).toEqual([8, 1, 80, 64]); // qk rkv+dr, v rkv
    expect(attn.kvRead.elemsPerToken).toBe(80); // rkv + dr
    expect(attn.kvWrite.elemsPerToken).toBe(80);
    const qAbsorb = gemm(ops, 'mla.q_absorb') as GemmOp;
    expect([qAbsorb.batch, qAbsorb.m, qAbsorb.n, qAbsorb.k]).toEqual([8, 1, 64, 32]); // Hl, T, rkv, dn
    const vAbsorb = gemm(ops, 'mla.v_absorb') as GemmOp;
    expect([vAbsorb.batch, vAbsorb.n, vAbsorb.k]).toEqual([8, 32, 64]); // Hl, dv, rkv
    const idx = find(ops, 'indexer') as IndexerOp;
    expect([idx.heads, idx.headDim, idx.topK, idx.kvLen]).toEqual([4, 32, 64, 1000]);
    expect(attn.kvSelection).toEqual({ kind: 'topk', topK: 64 });
    expect((gemm(ops, 'idx.q') as GemmOp).k).toBe(128); // the indexer query reads the q latent (rq)
  });

  it('op order', () => {
    expect(lowerBlock(MLA, ctx('decode')).map((o) => o.name)).toEqual([
      'mla.q_a',
      'mla.q_a_norm',
      'mla.q_b',
      'mla.kv_a',
      'mla.kv_a_norm',
      'mla.rope',
      'idx.q',
      'idx.k',
      'idx.k_norm',
      'idx.w',
      'idx.score',
      'mla.q_absorb',
      'attn.core',
      'mla.v_absorb',
      'mla.o_proj',
    ]);
  });

  it('prefill with sparse stays absorbed', () => {
    const ops = lowerBlock(MLA, ctx('prefill'));
    expect(gemm(ops, 'mla.q_absorb')).toBeDefined();
    expect(gemm(ops, 'mla.kv_b')).toBeUndefined();
    expect((find(ops, 'attention') as AttentionOp).kvSelection).toEqual({ kind: 'topk', topK: 64 });
  });

  it('prefill without sparse uses the MHA form', () => {
    const ops = lowerBlock(MLA_DENSE, ctx('prefill', 2));
    const kvB = gemm(ops, 'mla.kv_b') as GemmOp;
    expect([kvB.m, kvB.n, kvB.k]).toEqual([1024, 512, 64]); // seqs·kvLen = 2·512; Hl·(dn+dv) = 8·64; rkv
    const attn = find(ops, 'attention') as AttentionOp;
    expect([attn.kvHeads, attn.qkHeadDim, attn.vHeadDim]).toEqual([8, 48, 32]); // Hl, dn+dr, dv
    expect(attn.kvRead).toEqual({ format: { dtype: 'bf16' }, elemsPerToken: 528 }); // 8·64 + 16
    expect(attn.kvWrite.elemsPerToken).toBe(80); // the latent is still cached
    expect(attn.kvSelection).toEqual({ kind: 'full' });
    expect(find(ops, 'indexer')).toBeUndefined();
  });

  it('tp 2 shards heads, replicates the latent path, adds an all-reduce', () => {
    const ops = lowerBlock(MLA, ctx('decode', 1, 2));
    expect((find(ops, 'attention') as AttentionOp).heads).toBe(4); // 8/2
    expect((gemm(ops, 'mla.kv_a') as GemmOp).n).toBe(80); // replicated: rkv + dr
    expect((gemm(ops, 'mla.q_b') as GemmOp).n).toBe(192); // 4·(32 + 16)
    expect(find(ops, 'collective')?.collective).toBe('all_reduce');
  });

  it('sequence state = latent + indexer keys; resident weights = params·2 B at tp 1', () => {
    // 1000·(64 + 16)·2 = 160,000 + indexer 1000·32·2 = 64,000
    expect(blockSequenceStateBytes(MLA, ctx('decode'), 1000)).toBe(224_000);
    // absorbed GEMMs carry exactly kv_b's weights, so resident bytes = blockParams·2 (all bf16)
    expect(blockResidentWeightBytes(MLA, ctx('decode'))).toBe(blockParams(MLA) * 2); // 354,560·2
    expect(blockParams(MLA)).toBe(354_560);
  });
});
