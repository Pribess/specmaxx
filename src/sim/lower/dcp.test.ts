/**
 * Decode context parallelism (DCP) at the lowering level: per-rank KV bytes, the two extra collectives' exact
 * sizes, the sharded attention op, prefill unchanged, d = 1 unchanged, and the per-block error messages.
 * See docs/simulation-model.md §5 "Decode context parallelism (DCP)".
 */
import { describe, expect, it } from 'vitest';
import type {
  BlockSpec,
  GqaAttentionBlock,
  MlaAttentionBlock,
  ModelSpec,
  QuantConfig,
} from '../ir/model';
import type { AttentionOp, CollectiveOp, Op } from '../ir/ops';
import { attentionScores, opBytes, opFlops } from '../ir/metrics';
import { TINY_CSA, TINY_HYBRID, TINY_INKLING, TINY_MLA_SPARSE } from '../models/reference';
import { DCP_LSE_BYTES, dcpKvReplicationError, dcpKvTokens, dcpUnsupportedError } from './dcp';
import { blockSequenceStateBytes, lowerBlock, type LoweringContext } from './lower';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const FP8_ACT: QuantConfig = {
  weights: { dtype: 'fp8_e4m3' },
  activations: 'fp8_e4m3',
  kvCache: { dtype: 'fp8_e4m3' },
};

/** Kimi-K2-shaped MLA block (64 heads, rq 1536, rkv 512, dn 128, dr 64, dv 128, h 7168). */
const MLA: MlaAttentionBlock = {
  kind: 'mla_attention',
  name: 'attn',
  hidden: 7168,
  heads: 64,
  qLoraRank: 1536,
  kvLoraRank: 512,
  qkNopeDim: 128,
  qkRopeDim: 64,
  vHeadDim: 128,
};

/** A full-attention GQA block: 32 heads, `kvHeads` KV heads, hd 128, h 4096. */
function gqa(kvHeads: number, extra: Partial<GqaAttentionBlock> = {}): GqaAttentionBlock {
  return {
    kind: 'gqa_attention',
    name: 'attn',
    hidden: 4096,
    heads: 32,
    kvHeads,
    headDim: 128,
    qkvBias: false,
    outBias: false,
    qkNorm: true,
    rope: true,
    causal: true,
    outputGate: false,
    ...extra,
  };
}

interface CtxOptions {
  tp?: number;
  dcp?: number;
  seqs?: number;
  qLen?: number;
  kvLen?: number;
  quant?: QuantConfig;
}

/** A lowering context; `dcp` is left out of `parallel` when undefined (the field absent, as before DCP). */
function ctx(phase: 'prefill' | 'decode', o: CtxOptions = {}): LoweringContext {
  const qLen = o.qLen ?? (phase === 'prefill' ? 512 : 1);
  return {
    phase,
    seqs: o.seqs ?? 1,
    qLen,
    kvLen: o.kvLen ?? (phase === 'prefill' ? qLen : 1000),
    quant: o.quant ?? BF16,
    parallel: {
      tp: o.tp ?? 8,
      pp: 1,
      dp: 1,
      ep: 1,
      ...(o.dcp === undefined ? {} : { dcp: o.dcp }),
    },
  };
}

const attentionOf = (ops: Op[]): AttentionOp => {
  const a = ops.find((o): o is AttentionOp => o.kind === 'attention');
  if (a === undefined) throw new Error('no attention op');
  return a;
};
const collective = (ops: Op[], name: string): CollectiveOp => {
  const c = ops.find((o): o is CollectiveOp => o.kind === 'collective' && o.name === name);
  if (c === undefined) throw new Error(`no collective ${name}`);
  return c;
};
const blockOf = <K extends BlockSpec['kind']>(
  model: ModelSpec,
  layer: number,
  kind: K,
): Extract<BlockSpec, { kind: K }> => {
  const b = model.layers[layer].blocks.find(
    (x): x is Extract<BlockSpec, { kind: K }> => x.kind === kind,
  );
  if (b === undefined) throw new Error(`no ${kind} in layer ${layer}`);
  return b;
};

describe('dcpKvTokens: ceil(ctx / d), exactly ctx at d = 1', () => {
  it.each([
    [131072, 1, 131072],
    [131073, 1, 131073],
    [131072, 8, 16384],
    [131073, 8, 16385],
    [5, 8, 1],
    [1001, 4, 251],
  ])('ctx %i, d %i → %i', (c, d, n) => {
    expect(dcpKvTokens(c, d)).toBe(n);
  });
});

describe('per-rank KV bytes (memory, any phase)', () => {
  const ctxTokens = 131073; // not divisible by any d > 1: the ceil shows
  it.each([1, 2, 4, 8])('MLA at tp 8, d %i: ceil(ctx/d)·(rkv + dr)·bpe(kv)', (d) => {
    for (const phase of ['prefill', 'decode'] as const) {
      expect(blockSequenceStateBytes(MLA, ctx(phase, { dcp: d }), ctxTokens)).toBe(
        Math.ceil(ctxTokens / d) * 576 * 2,
      );
    }
  });
  it('MLA: dcp absent = dcp 1 = ctx·576·bpe (the latent replicated on every TP rank)', () => {
    expect(blockSequenceStateBytes(MLA, ctx('decode'), ctxTokens)).toBe(ctxTokens * 576 * 2);
  });
  it.each([1, 2, 4])(
    'GQA kvHeads 2 at tp 8 (r = 4), d %i: ceil(ctx/d)·2·KVl·hd·bpe(kv), KVl = 1',
    (d) => {
      expect(blockSequenceStateBytes(gqa(2), ctx('decode', { dcp: d }), ctxTokens)).toBe(
        Math.ceil(ctxTokens / d) * 2 * 1 * 128 * 2,
      );
    },
  );
  it('a sliding-window GQA block keeps min(ctx, W) tokens: unsharded', () => {
    const win = gqa(2, { slidingWindow: 4096 });
    expect(blockSequenceStateBytes(win, ctx('decode', { dcp: 4 }), ctxTokens)).toBe(
      4096 * 2 * 128 * 2,
    );
    expect(blockSequenceStateBytes(win, ctx('decode', { dcp: 4 }), ctxTokens)).toBe(
      blockSequenceStateBytes(win, ctx('decode'), ctxTokens),
    );
  });
  it('linear-attention state is unchanged (not token-shardable)', () => {
    const lin = blockOf(TINY_HYBRID, 0, 'linear_attention');
    expect(blockSequenceStateBytes(lin, ctx('decode', { tp: 4, dcp: 4 }), ctxTokens)).toBe(
      blockSequenceStateBytes(lin, ctx('decode', { tp: 4 }), ctxTokens),
    );
  });
});

describe('the two DCP collectives, exact bytes', () => {
  // MLA, tp 8 ⇒ Hl = 8; d = 8 ⇒ gathered heads Hl·d = 64.
  it('MLA bf16, seqs 3, qLen 1: S_q = T·64·(rkv + dr)·2, S_o = T·64·(rkv·2 + 4)', () => {
    const ops = lowerBlock(MLA, ctx('decode', { dcp: 8, seqs: 3 }));
    const T = 3;
    expect(collective(ops, 'attn.dcp_q_all_gather')).toEqual({
      kind: 'collective',
      name: 'attn.dcp_q_all_gather',
      collective: 'all_gather',
      group: 'dcp',
      groupSize: 8,
      messageBytes: T * 64 * 576 * 2, // 221,184
      dtype: 'bf16',
    });
    expect(collective(ops, 'attn.dcp_out_reduce_scatter')).toEqual({
      kind: 'collective',
      name: 'attn.dcp_out_reduce_scatter',
      collective: 'reduce_scatter',
      group: 'dcp',
      groupSize: 8,
      messageBytes: T * 64 * (512 * 2 + DCP_LSE_BYTES), // 197,376
      dtype: 'bf16',
    });
    expect(collective(ops, 'attn.dcp_q_all_gather').messageBytes).toBe(221184);
    expect(collective(ops, 'attn.dcp_out_reduce_scatter').messageBytes).toBe(197376);
  });
  it('MLA spec-dec verify (seqs 3, qLen 4 ⇒ T 12): both scale with T', () => {
    const ops = lowerBlock(MLA, ctx('decode', { dcp: 8, seqs: 3, qLen: 4, kvLen: 1004 }));
    expect(collective(ops, 'attn.dcp_q_all_gather').messageBytes).toBe(12 * 64 * 576 * 2); // 884,736
    expect(collective(ops, 'attn.dcp_out_reduce_scatter').messageBytes).toBe(12 * 64 * 1028); // 789,504
  });
  it('MLA fp8 activations: the gather moves 1 B/elem, the reduce-scatter stays 16-bit (reduceBpe 2)', () => {
    const ops = lowerBlock(MLA, ctx('decode', { dcp: 8, seqs: 3, quant: FP8_ACT }));
    expect(collective(ops, 'attn.dcp_q_all_gather')).toMatchObject({
      messageBytes: 3 * 64 * 576 * 1,
      dtype: 'fp8_e4m3',
    });
    expect(collective(ops, 'attn.dcp_out_reduce_scatter')).toMatchObject({
      messageBytes: 3 * 64 * (512 * 2 + 4),
      dtype: 'bf16',
    });
  });
  it('GQA kvHeads 4 at tp 8 (Hl 4), d 2, seqs 5: S_q = 5·8·128·2, S_o = 5·8·(128·2 + 4)', () => {
    const ops = lowerBlock(gqa(4), ctx('decode', { dcp: 2, seqs: 5 }));
    expect(collective(ops, 'attn.dcp_q_all_gather')).toMatchObject({
      groupSize: 2,
      messageBytes: 10240,
    });
    expect(collective(ops, 'attn.dcp_out_reduce_scatter')).toMatchObject({
      groupSize: 2,
      messageBytes: 10400,
    });
  });
});

describe('op order with DCP', () => {
  it('MLA: q_absorb → Q all-gather → attention → reduce-scatter → v_absorb', () => {
    expect(lowerBlock(MLA, ctx('decode', { dcp: 8 })).map((o) => o.name)).toEqual([
      'mla.q_a',
      'mla.q_a_norm',
      'mla.q_b',
      'mla.kv_a',
      'mla.kv_a_norm',
      'mla.rope',
      'mla.q_absorb',
      'attn.dcp_q_all_gather',
      'attn.core',
      'attn.dcp_out_reduce_scatter',
      'mla.v_absorb',
      'mla.o_proj',
      'attn.tp_all_reduce',
    ]);
  });
  it('gated GQA: rope → Q all-gather → attention → reduce-scatter → output gate → o_proj', () => {
    const gated = blockOf(TINY_HYBRID, 3, 'gqa_attention'); // 4 heads, 1 KV head: r = tp = 4
    expect(lowerBlock(gated, ctx('decode', { tp: 4, dcp: 4 })).map((o) => o.name)).toEqual([
      'attn.qkv_proj',
      'attn.qk_norm',
      'attn.rope',
      'attn.dcp_q_all_gather',
      'attn.core',
      'attn.dcp_out_reduce_scatter',
      'attn.output_gate',
      'attn.o_proj',
      'attn.tp_all_reduce',
    ]);
  });
});

describe('the sharded attention op', () => {
  const kvLen = 131072; // divisible by 8: kvRead exactly 1/8
  const base = attentionOf(lowerBlock(MLA, ctx('decode', { kvLen, seqs: 4 })));
  const sharded = attentionOf(lowerBlock(MLA, ctx('decode', { kvLen, seqs: 4, dcp: 8 })));
  it('heads = Hl·d, kvTokenShards = d, every other field as at d = 1', () => {
    expect(sharded).toEqual({ ...base, heads: 8 * 8, kvTokenShards: 8 });
    expect(base.heads).toBe(8);
  });
  it('FLOP and softmax scores per rank are unchanged', () => {
    expect(opFlops(sharded)).toBe(opFlops(base));
    expect(attentionScores(sharded)).toBe(attentionScores(base));
  });
  it('kvReadBytes is exactly 1/d; Q read and O write grow d× (the gathered heads)', () => {
    const b = opBytes(base);
    const s = opBytes(sharded);
    expect(s.kvReadBytes).toBe(b.kvReadBytes / 8);
    expect(s.kvReadBytes).toBe(4 * (kvLen / 8) * 576 * 2);
    expect(s.activationReadBytes).toBe(8 * b.activationReadBytes);
    expect(s.activationWriteBytes).toBe(8 * b.activationWriteBytes);
    expect(s.kvWriteBytes).toBe(b.kvWriteBytes); // ceil(1/8) = 1 new token on the busiest rank
  });
  it('qLen 4 > kvLen/d (kvLen 5, d 8): pairs /d stays positive, tokens read/written ceil to 1', () => {
    const c = { qLen: 4, kvLen: 5, seqs: 1 };
    const b = attentionOf(lowerBlock(MLA, ctx('decode', c)));
    const s = attentionOf(lowerBlock(MLA, ctx('decode', { ...c, dcp: 8 })));
    expect(opFlops(s)).toBe(opFlops(b));
    expect(opFlops(s)).toBeGreaterThan(0);
    expect(opBytes(s).kvReadBytes).toBe(1 * 576 * 2);
    expect(opBytes(s).kvWriteBytes).toBe(1 * 576 * 2);
  });
  it('GQA: kvHeads stay KVl = 1, kvRead 1/d', () => {
    const b = attentionOf(lowerBlock(gqa(4), ctx('decode', { kvLen: 4096 })));
    const s = attentionOf(lowerBlock(gqa(4), ctx('decode', { kvLen: 4096, dcp: 2 })));
    expect(s).toEqual({ ...b, heads: 8, kvTokenShards: 2 });
    expect(s.kvHeads).toBe(1);
    expect(opFlops(s)).toBe(opFlops(b));
    expect(opBytes(s).kvReadBytes).toBe(opBytes(b).kvReadBytes / 2);
  });
});

describe('identity: d = 1 and prefill', () => {
  const blocks: [string, BlockSpec, number][] = [
    ['MLA', MLA, 8],
    ['GQA kvHeads 4', gqa(4), 8],
    ['gated GQA (hybrid)', blockOf(TINY_HYBRID, 3, 'gqa_attention'), 4],
    ['linear attention', blockOf(TINY_HYBRID, 0, 'linear_attention'), 4],
  ];
  it.each(blocks)(
    '%s: dcp 1 lowers exactly like dcp absent, with no kvTokenShards and no dcp op',
    (_n, block, tp) => {
      for (const phase of ['prefill', 'decode'] as const) {
        const absent = lowerBlock(block, ctx(phase, { tp }));
        const one = lowerBlock(block, ctx(phase, { tp, dcp: 1 }));
        expect(one).toStrictEqual(absent);
        expect(one.some((o) => 'kvTokenShards' in o)).toBe(false);
        expect(one.some((o) => o.kind === 'collective' && o.group === 'dcp')).toBe(false);
      }
    },
  );
  it.each(blocks)('%s: prefill at d > 1 lowers exactly like d absent', (_n, block, tp) => {
    const d = block.kind === 'gqa_attention' && block.kvHeads === 4 ? 2 : tp;
    expect(lowerBlock(block, ctx('prefill', { tp, dcp: d }))).toStrictEqual(
      lowerBlock(block, ctx('prefill', { tp })),
    );
  });
  it('a sliding-window GQA block lowers exactly like d absent in decode too', () => {
    const win = gqa(2, { slidingWindow: 4096 });
    expect(lowerBlock(win, ctx('decode', { dcp: 4 }))).toStrictEqual(
      lowerBlock(win, ctx('decode')),
    );
  });
});

describe('per-block errors (d > 1, any phase)', () => {
  it('GQA: d must divide r = tp / kvHeads', () => {
    expect(() => lowerBlock(gqa(4), ctx('decode', { dcp: 4 }))).toThrow(
      'attn dcp (4) must divide tp / kvHeads (2): KV heads are replicated on 2 ranks and DCP shards that copy',
    );
    expect(dcpKvReplicationError('attn', 4, 2)).toBe(
      'attn dcp (4) must divide tp / kvHeads (2): KV heads are replicated on 2 ranks and DCP shards that copy',
    );
    expect(() => lowerBlock(gqa(4), ctx('decode', { dcp: 2 }))).not.toThrow();
  });
  it('GQA with kvHeads ≥ tp (r = 1): only d = 1', () => {
    expect(() => lowerBlock(gqa(8), ctx('decode', { dcp: 2 }))).toThrow(
      'attn dcp (2) must divide tp / kvHeads (1): KV heads are replicated on 1 ranks and DCP shards that copy',
    );
    expect(() => blockSequenceStateBytes(gqa(8), ctx('prefill', { dcp: 2 }), 100)).toThrow(
      'must divide tp / kvHeads (1)',
    );
  });
  it('sparse MLA', () => {
    const sparse = blockOf(TINY_MLA_SPARSE, 0, 'mla_attention');
    const msg = 'attn dcp (2) > 1 is not supported with sparse (top-k) attention';
    expect(dcpUnsupportedError('attn', 2, 'sparse')).toBe(msg);
    expect(() => lowerBlock(sparse, ctx('decode', { dcp: 2 }))).toThrow(msg);
    expect(() => lowerBlock(sparse, ctx('prefill', { dcp: 2 }))).toThrow(msg);
    expect(() => blockSequenceStateBytes(sparse, ctx('decode', { dcp: 2 }), 100)).toThrow(msg);
  });
  it('CSA', () => {
    const csa = blockOf(TINY_CSA, 0, 'csa_attention');
    const msg = 'attn dcp (2) > 1 is not supported with compressed sparse attention (CSA)';
    expect(() => lowerBlock(csa, ctx('decode', { tp: 1, dcp: 2 }))).toThrow(msg);
    expect(() => blockSequenceStateBytes(csa, ctx('decode', { tp: 1, dcp: 2 }), 100)).toThrow(msg);
  });
  it('relative-position bias / K/V convs on a full-attention block; the window block of the same model is fine', () => {
    const full = blockOf(TINY_INKLING, 1, 'gqa_attention');
    const window = blockOf(TINY_INKLING, 0, 'gqa_attention');
    expect(() => lowerBlock(full, ctx('decode', { tp: 2, dcp: 2 }))).toThrow(
      'attn dcp (2) > 1 is not supported with a relative-position bias or K/V short convs',
    );
    expect(() => lowerBlock(window, ctx('decode', { tp: 2, dcp: 2 }))).not.toThrow();
  });
  it('d = 1 never throws a DCP error', () => {
    expect(() =>
      lowerBlock(blockOf(TINY_MLA_SPARSE, 0, 'mla_attention'), ctx('decode', { dcp: 1 })),
    ).not.toThrow();
    expect(() => lowerBlock(gqa(8), ctx('decode', { dcp: 1 }))).not.toThrow();
  });
});
