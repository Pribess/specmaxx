import { describe, expect, it } from 'vitest';
import { attendedPairs, attentionPairs, attentionScores, indexerPairs, kvTokensRead, opBytes, opFlops } from './metrics';
import type { AttentionOp, ElementwiseOp, GemmOp, IndexerOp, RecurrentOp, TransferOp } from './ops';
import { lowerBlock, type LoweringContext } from '../lower/lower';
import { REF_DENSE_8B } from '../models/reference';
import type { QuantConfig } from './model';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

const gemm = (sparsity: 'dense' | '2:4'): GemmOp => ({
  kind: 'gemm',
  name: 'test',
  batch: 1,
  m: 16,
  n: 4096,
  k: 4096,
  inputDtype: 'bf16',
  weight: { dtype: 'bf16' },
  outputDtype: 'bf16',
  math: 'bf16',
  accumulate: 'fp32',
  sparsity,
});

describe('T2 GEMM b1 m16 n4096 k4096 bf16', () => {
  it('counts FLOP and bytes', () => {
    const op = gemm('dense');
    expect(opFlops(op)).toBe(536_870_912); // 2·16·4096·4096
    const b = opBytes(op);
    expect(b.weightBytes).toBe(33_554_432); // 4096·4096·2
    expect(b.activationReadBytes).toBe(131_072); // 16·4096·2
    expect(b.activationWriteBytes).toBe(131_072); // 16·4096·2
    expect(b.totalBytes).toBe(33_554_432 + 2 * 131_072);
  });
  it('2:4 stores half the values plus 2-bit indices, same FLOP', () => {
    const op = gemm('2:4');
    expect(opBytes(op).weightBytes).toBe(18_874_368); // 4096²·(16/2 + 1)/8 = 4096²·9/8
    expect(opFlops(op)).toBe(536_870_912); // dense-equivalent: unchanged
  });
  it('adds the bias', () => {
    const op: GemmOp = { ...gemm('dense'), biasDtype: 'bf16' };
    expect(opBytes(op).weightBytes).toBe(33_554_432 + 8192); // + n·2 = 4096·2
  });
});

describe('T2b attendedPairs', () => {
  const full = { kind: 'full' } as const;
  it('causal full', () => {
    expect(attendedPairs(4096, 4096, true, full)).toBe(8_390_656); // 4096·4097/2
    expect(attendedPairs(1, 1024, true, full)).toBe(1024); // decode: 1·1023 + 1
    expect(attendedPairs(4, 10, true, full)).toBe(34); // 4·6 + 4·5/2 = 24 + 10
  });
  it('sliding window', () => {
    // W 4, q8 kv8: past 0, j0 = 4 → 0 + 4·5/2 + (8 − 4)·4 = 10 + 16
    expect(attendedPairs(8, 8, true, { kind: 'window', windowTokens: 4 })).toBe(26);
    // W 4, q4 kv10: past 6, j0 = clamp(4 − 6) = 0 → 4·4
    expect(attendedPairs(4, 10, true, { kind: 'window', windowTokens: 4 })).toBe(16);
  });
  it('non-causal', () => {
    expect(attendedPairs(3, 5, false, full)).toBe(15); // 3·5
    expect(attendedPairs(3, 5, false, { kind: 'topk', topK: 2 })).toBe(6); // 3·min(2, 5)
  });
  it('top-k', () => {
    // topK 256, q = kv = 512: j0 = 256 → 256·257/2 + 256·256 = 32,896 + 65,536
    expect(attendedPairs(512, 512, true, { kind: 'topk', topK: 256 })).toBe(98_432);
    expect(attendedPairs(1, 1000, true, { kind: 'topk', topK: 256 })).toBe(256); // decode: min(256, 1000)
  });
});

describe('kvTokensRead', () => {
  it('caps the union of touched keys', () => {
    expect(kvTokensRead(8, 100, { kind: 'full' })).toBe(100);
    expect(kvTokensRead(8, 100, { kind: 'window', windowTokens: 4 })).toBe(11); // 4 + 8 − 1
    expect(kvTokensRead(1, 1000, { kind: 'topk', topK: 64 })).toBe(64); // min(1000, 64·1)
    expect(kvTokensRead(512, 512, { kind: 'topk', topK: 256 })).toBe(512); // min(512, 256·512)
  });
});

describe('T2c REF_DENSE_8B attention layer', () => {
  const attnBlock = REF_DENSE_8B.layers[0].blocks.find((b) => b.kind === 'gqa_attention');
  if (attnBlock === undefined) throw new Error('no attention block');
  const ctxOf = (phase: 'prefill' | 'decode', qLen: number, kvLen: number): LoweringContext => ({
    phase,
    seqs: 1,
    qLen,
    kvLen,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
  });
  const attentionOf = (ctx: LoweringContext): AttentionOp => {
    const op = lowerBlock(attnBlock, ctx).find((o) => o.kind === 'attention');
    if (op === undefined || op.kind !== 'attention') throw new Error('no attention op');
    return op;
  };
  it('prefill T 4096 FLOP: exact causal triangle', () => {
    // 2·heads·pairs·(qk + v) = 2·32·8,390,656·(128 + 128). (The plan's 137,438,953,472 used 4096²/2 pairs,
    // i.e. it omitted the diagonal; the triangle with the diagonal is 4096·4097/2.)
    expect(opFlops(attentionOf(ctxOf('prefill', 4096, 4096)))).toBe(137_472_507_904);
  });
  it('decode ctx 1024 reads the whole KV cache', () => {
    // 1024 tokens · (2·8 KV heads·128) elems · 2 B
    expect(opBytes(attentionOf(ctxOf('decode', 1, 1024))).kvReadBytes).toBe(4_194_304);
    expect(opBytes(attentionOf(ctxOf('decode', 1, 1024))).kvWriteBytes).toBe(4096); // 1 token · 2048 elems · 2 B
  });
});

describe('T2d indexer', () => {
  const op: IndexerOp = {
    kind: 'indexer',
    name: 'idx',
    seqs: 1,
    heads: 4,
    headDim: 32,
    qLen: 1,
    kvLen: 1000,
    causal: true,
    topK: 256,
    keyCache: { dtype: 'fp8_e4m3' },
    keyElemsPerKey: 32,
    keyPoolTokens: 1,
    keyWriteElemsPerToken: 32,
    headWeights: true,
    math: 'fp8',
    accumulate: 'fp32',
    activationDtype: 'bf16',
  };
  it('scores every visible key', () => {
    expect(opFlops(op)).toBe(256_000); // 2·4 heads·1000 pairs·32
    const b = opBytes(op);
    expect(b.kvReadBytes).toBe(32_000); // 1000 tokens·32 elems·1 B
    expect(b.kvWriteBytes).toBe(32); // 1 new token·32·1 B
    expect(b.activationWriteBytes).toBe(1024); // 256 int32 indices·4 B
    expect(b.activationReadBytes).toBe(264); // q 4·32·2 = 256 + head weights 4·2 = 8
  });
});

describe('pooled / compressed keys (r tokens per key)', () => {
  it('r = 1 reproduces the token formulas exactly', () => {
    for (const sel of [{ kind: 'full' } as const, { kind: 'topk', topK: 64 } as const, { kind: 'window', windowTokens: 32 } as const]) {
      expect(attendedPairs(100, 300, true, sel, 1)).toBe(attendedPairs(100, 300, true, sel));
      expect(kvTokensRead(100, 300, sel, 1)).toBe(kvTokensRead(100, 300, sel));
    }
  });
  it('r = 2, by hand', () => {
    // full: qLen 4, past 6: (4·6 + 4·5/2)/2 = (24 + 10)/2 = 17   (queries see 3.5, 4, 4.5, 5 entries)
    expect(attendedPairs(4, 10, true, { kind: 'full' }, 2)).toBe(17);
    // capped at 4 entries: j0 = clamp(4·2 − 6, 0, 4) = 2 → (2·6 + 3)/2 + 2·4 = 7.5 + 8 = 15.5
    expect(attendedPairs(4, 10, true, { kind: 'topk', topK: 4 }, 2)).toBe(15.5);
    // keys read: full ceil(10/2) = 5; top-k min(5, 2·4) = 5; decode at 1000: top-k 64 of 500 → 64
    expect(kvTokensRead(4, 10, { kind: 'full' }, 2)).toBe(5);
    expect(kvTokensRead(1, 1000, { kind: 'topk', topK: 64 }, 2)).toBe(64);
  });
  it('attention with a secondary (compressed) key set in the same softmax', () => {
    const op: AttentionOp = {
      kind: 'attention',
      name: 'csa',
      seqs: 1,
      heads: 2,
      kvHeads: 1,
      qkHeadDim: 64,
      vHeadDim: 64,
      qLen: 1,
      kvLen: 1000,
      causal: true,
      kvSelection: { kind: 'window', windowTokens: 128 },
      kvRead: { format: { dtype: 'bf16' }, elemsPerToken: 64 },
      kvWrite: { format: { dtype: 'bf16' }, elemsPerToken: 64 },
      secondaryKv: {
        tokensPerEntry: 4,
        kvSelection: { kind: 'topk', topK: 32 },
        kvRead: { format: { dtype: 'fp8_e4m3' }, elemsPerToken: 64 },
        kvWrite: { format: { dtype: 'fp8_e4m3' }, elemsPerToken: 16 },
      },
      math: 'bf16',
      accumulate: 'fp32',
      activationDtype: 'bf16',
    };
    expect(attentionPairs(op)).toBe(128 + 32);
    expect(opFlops(op)).toBe(2 * 2 * 160 * 128); // 2·heads·pairs·(qk + v)
    const b = opBytes(op);
    expect(b.kvReadBytes).toBe(128 * 64 * 2 + 32 * 64); // window bf16 + 32 entries fp8
    expect(b.kvWriteBytes).toBe(64 * 2 + 16); // window vector + hd/r of the compressed cache
    expect(b.activationReadBytes).toBe(2 * 64 * 2); // Q read once
  });
  it('indexer with pooled keys, a scoring cap, no head weights, foreign keys', () => {
    const op: IndexerOp = {
      kind: 'indexer',
      name: 'idx',
      seqs: 1,
      heads: 4,
      headDim: 32,
      qLen: 1,
      kvLen: 1000,
      causal: true,
      topK: 16,
      keyCache: { dtype: 'bf16' },
      keyElemsPerKey: 32,
      keyPoolTokens: 4,
      maxScoredKeys: 100,
      keyWriteElemsPerToken: 0,
      headWeights: false,
      math: 'bf16',
      accumulate: 'fp32',
      activationDtype: 'bf16',
    };
    // 1000/4 = 250 visible keys, capped at 100
    expect(indexerPairs(op)).toBe(100);
    expect(opFlops(op)).toBe(2 * 4 * 100 * 32);
    const b = opBytes(op);
    expect(b.kvReadBytes).toBe(100 * 32 * 2);
    expect(b.kvWriteBytes).toBe(0);
    expect(b.activationReadBytes).toBe(4 * 32 * 2); // q only
    expect(indexerPairs({ ...op, maxScoredKeys: undefined } as IndexerOp)).toBe(250);
  });
  it('KDA reads dk + 1 gate elements per token and head', () => {
    const op: RecurrentOp = {
      kind: 'recurrent',
      name: 'kda',
      variant: 'kda',
      seqs: 1,
      qLen: 1,
      heads: 2,
      qkHeads: 2,
      keyDim: 64,
      valueDim: 64,
      mode: 'step',
      chunkLen: 1,
      stateFlopsPerElemPerToken: 7,
      gateElemsPerTokenPerHead: 65,
      hasPriorState: true,
      stateDtype: 'fp32',
      activationDtype: 'bf16',
      math: 'bf16',
      accumulate: 'fp32',
    };
    // q,k 2·2·64·2 = 512; v 2·64·2 = 256; gates 2·65·2 = 260
    expect(opBytes(op).activationReadBytes).toBe(512 + 256 + 260);
  });
});
describe('T2e recurrent GatedDeltaNet (vH 8, qkH 4, dk = dv = 64, fp32 state)', () => {
  const base: RecurrentOp = {
    kind: 'recurrent',
    name: 'gdn',
    variant: 'gated_delta_net',
    seqs: 2,
    qLen: 1,
    heads: 8,
    qkHeads: 4,
    keyDim: 64,
    valueDim: 64,
    mode: 'step',
    chunkLen: 1,
    stateFlopsPerElemPerToken: 7,
    gateElemsPerTokenPerHead: 2,
    hasPriorState: true,
    stateDtype: 'fp32',
    activationDtype: 'bf16',
    math: 'bf16',
    accumulate: 'fp32',
  };
  it('step, seqs 2', () => {
    expect(opFlops(base)).toBe(458_752); // 2 seqs·1·8 heads·7·4096
    const b = opBytes(base);
    expect(b.stateReadBytes).toBe(262_144); // 2·8·4096·4
    expect(b.stateWriteBytes).toBe(262_144);
  });
  it('chunked, seqs 1, q128, C 64, no prior state', () => {
    const op: RecurrentOp = {
      ...base,
      seqs: 1,
      qLen: 128,
      mode: 'chunked',
      chunkLen: 64,
      stateFlopsPerElemPerToken: 4,
      hasPriorState: false,
    };
    expect(opFlops(op)).toBe(25_296_896); // 128·8·(4·4096 + 65·128)
    const b = opBytes(op);
    expect(b.stateReadBytes).toBe(0);
    expect(b.stateWriteBytes).toBe(131_072); // 1·8·4096·4
  });
});

describe('recurrent Mamba2 (Nemotron-3.5-Lightning layer, tp 1: H 64, G 8, N 128, P 64, fp32 state)', () => {
  const step: RecurrentOp = {
    kind: 'recurrent',
    name: 'lin.recurrent',
    variant: 'mamba2',
    seqs: 1,
    qLen: 1,
    heads: 64,
    qkHeads: 8,
    keyDim: 128,
    valueDim: 64,
    mode: 'step',
    chunkLen: 1,
    stateFlopsPerElemPerToken: 5,
    gateElemsPerTokenPerHead: 2,
    hasPriorState: true,
    stateDtype: 'fp32',
    activationDtype: 'bf16',
    math: 'bf16',
    accumulate: 'fp32',
  };
  it('decode step: F = T·H·5·N·P; state 64·128·64·4 B read and written', () => {
    expect(opFlops(step)).toBe(2_621_440); // 1·64·5·8192
    const b = opBytes(step);
    expect(b.stateReadBytes).toBe(2_097_152);
    expect(b.stateWriteBytes).toBe(2_097_152);
    // B, C 2·1·8·128·2 = 4096 + x 64·64·2 = 8192 + gates 64·2·2 = 256
    expect(b.activationReadBytes).toBe(12_544);
    expect(b.activationWriteBytes).toBe(8192); // y 64·64·2
  });
  it('prefill chunked SSD: C·Bᵀ once per group, F = T·(H·4·N·P + (Q+1)·(G·N + H·P))', () => {
    const op: RecurrentOp = { ...step, qLen: 4096, mode: 'chunked', chunkLen: 128, stateFlopsPerElemPerToken: 4, hasPriorState: false };
    // 4096·(64·4·8192 + 129·(8·128 + 64·64)) = 4096·(2,097,152 + 660,480) = 4096·2,757,632
    expect(opFlops(op)).toBe(11_295_260_672);
    // The per-head form of the other variants would count C·Bᵀ for all 64 heads: 4096·64·(4·8192 + 129·192)
    expect(opFlops({ ...op, variant: 'gated_delta_net' })).toBe(15_082_717_184);
    expect(opBytes(op).stateReadBytes).toBe(0);
  });
});

describe('deployment fields (design doc §7.1): defaults reproduce today, overrides scale only their term', () => {
  const attention: AttentionOp = {
    kind: 'attention',
    name: 'attn.core',
    seqs: 1,
    heads: 32,
    kvHeads: 8,
    qkHeadDim: 128,
    vHeadDim: 128,
    qLen: 13,
    kvLen: 2061,
    causal: false,
    kvSelection: { kind: 'window', windowTokens: 2048 },
    kvRead: { format: { dtype: 'bf16' }, elemsPerToken: 2048 },
    kvWrite: { format: { dtype: 'bf16' }, elemsPerToken: 2048 },
    math: 'bf16',
    accumulate: 'fp32',
    activationDtype: 'bf16',
  };
  it('AttentionOp.kvWriteTokens absent = qLen; 5 keeps only the context rows (block drafter m_ctx 5)', () => {
    expect(opBytes(attention).kvWriteBytes).toBe(53_248); // 13 tokens · 2·8·128 elems · 2 B
    expect(opBytes({ ...attention, kvWriteTokens: 13 })).toEqual(opBytes(attention));
    const kept = opBytes({ ...attention, kvWriteTokens: 5 });
    expect(kept.kvWriteBytes).toBe(20_480); // 5 · 2048 · 2
    expect(kept.kvReadBytes).toBe(opBytes(attention).kvReadBytes); // reads unchanged
    expect(opFlops({ ...attention, kvWriteTokens: 5 })).toBe(opFlops(attention));
  });
  it('AttentionOp.kvWriteTokens also applies to the secondary (compressed) cache write', () => {
    const withSecondary: AttentionOp = {
      ...attention,
      secondaryKv: {
        tokensPerEntry: 4,
        kvSelection: { kind: 'full' },
        kvRead: { format: { dtype: 'bf16' }, elemsPerToken: 512 },
        kvWrite: { format: { dtype: 'bf16' }, elemsPerToken: 128 },
      },
    };
    // primary 5·2048·2 = 20,480 + secondary 5·128·2 = 1,280
    expect(opBytes({ ...withSecondary, kvWriteTokens: 5 }).kvWriteBytes).toBe(21_760);
    expect(opBytes(withSecondary).kvWriteBytes).toBe(53_248 + 3_328); // 13·128·2 = 3,328
  });
  it('RecurrentOp.stateSnapshots multiplies the state write only (k = 3 verify: 4 snapshots)', () => {
    const op: RecurrentOp = {
      kind: 'recurrent',
      name: 'gdn',
      variant: 'gated_delta_net',
      seqs: 2,
      qLen: 4,
      heads: 8,
      qkHeads: 4,
      keyDim: 64,
      valueDim: 64,
      mode: 'step',
      chunkLen: 1,
      stateFlopsPerElemPerToken: 7,
      gateElemsPerTokenPerHead: 2,
      hasPriorState: true,
      stateDtype: 'fp32',
      activationDtype: 'bf16',
      math: 'bf16',
      accumulate: 'fp32',
    };
    expect(opBytes({ ...op, stateSnapshots: 1 })).toEqual(opBytes(op));
    const b = opBytes({ ...op, stateSnapshots: 4 });
    expect(b.stateReadBytes).toBe(262_144); // read once: 2·8·4096·4
    expect(b.stateWriteBytes).toBe(1_048_576); // 4 · 262,144
    expect(b.totalBytes - opBytes(op).totalBytes).toBe(3 * 262_144);
  });
  it('ElementwiseOp.kvWriteBytes maps to ByteBreakdown.kvWriteBytes (absent = 0)', () => {
    // The U-15 kv_cache_write shape: elems 2048·2·8·128 = 4,194,304, bf16 act read and kv write 8,388,608 B.
    const op: ElementwiseOp = {
      kind: 'elementwise',
      name: 'draft.kv_cache_write',
      fn: 'residual_add',
      elems: 4_194_304,
      flopsPerElem: 1,
      dtype: 'bf16',
      activationReadBytes: 8_388_608,
      activationWriteBytes: 0,
      weightBytes: 0,
      stateReadBytes: 0,
      stateWriteBytes: 0,
    };
    expect(opBytes(op).kvWriteBytes).toBe(0);
    expect(opBytes(op).totalBytes).toBe(8_388_608);
    const b = opBytes({ ...op, kvWriteBytes: 8_388_608 });
    expect(b.kvWriteBytes).toBe(8_388_608);
    expect(b.totalBytes).toBe(16_777_216);
    expect(opFlops(op)).toBe(4_194_304);
  });
  it('TransferOp has no FLOP and no device-memory bytes (costed by the tier/link model)', () => {
    const op: TransferOp = {
      kind: 'transfer',
      name: 'moe.expert_fetch',
      group: 'routed_experts',
      source: 'host_ram',
      reads: [{ count: 8, bytesEach: 2_654_208 }],
      usefulBytes: 21_233_664,
      prefetchedFraction: 1,
    };
    expect(opFlops(op)).toBe(0);
    expect(opBytes(op)).toEqual({
      weightBytes: 0,
      activationReadBytes: 0,
      activationWriteBytes: 0,
      kvReadBytes: 0,
      kvWriteBytes: 0,
      stateReadBytes: 0,
      stateWriteBytes: 0,
      totalBytes: 0,
    });
  });
});

describe('attentionScores: softmax scores the kernel computes (docs §7 "Attention softmax")', () => {
  const base: AttentionOp = {
    kind: 'attention',
    name: 'attn.core',
    seqs: 1,
    heads: 32,
    kvHeads: 8,
    qkHeadDim: 128,
    vHeadDim: 128,
    qLen: 4096,
    kvLen: 4096,
    causal: true,
    kvSelection: { kind: 'full' },
    kvRead: { format: { dtype: 'bf16' }, elemsPerToken: 2048 },
    kvWrite: { format: { dtype: 'bf16' }, elemsPerToken: 2048 },
    math: 'bf16',
    accumulate: 'fp32',
    activationDtype: 'bf16',
  };
  it('causal prefill: exactly heads·P(P+1)/2 (about half the square)', () => {
    expect(attentionScores(base)).toBe(32 * ((4096 * 4097) / 2));
  });
  it('non-causal: heads·qLen·kvLen', () => {
    expect(attentionScores({ ...base, causal: false })).toBe(32 * 4096 * 4096);
  });
  it('sliding window W: capped above the window, the triangle below it', () => {
    // W = 128 < P: the first 128 queries see 1..128 keys, the rest 128 each.
    const w = { ...base, kvSelection: { kind: 'window', windowTokens: 128 } } as const;
    expect(attentionScores(w)).toBe(32 * ((128 * 129) / 2 + (4096 - 128) * 128));
    // W = 8192 > P: identical to full causal.
    expect(attentionScores({ ...base, kvSelection: { kind: 'window', windowTokens: 8192 } })).toBe(attentionScores(base));
  });
  it('top-k (DSA): each query scores at most k keys', () => {
    const decode = { ...base, qLen: 1, kvLen: 100_001, kvSelection: { kind: 'topk', topK: 2048 } } as const;
    expect(attentionScores(decode)).toBe(32 * 2048);
  });
  it('CSA secondary entries share the softmax and add their pairs', () => {
    const op: AttentionOp = {
      ...base,
      qLen: 1,
      kvLen: 1000,
      kvSelection: { kind: 'window', windowTokens: 128 },
      secondaryKv: {
        tokensPerEntry: 4,
        kvSelection: { kind: 'topk', topK: 32 },
        kvRead: { format: { dtype: 'fp8_e4m3' }, elemsPerToken: 64 },
        kvWrite: { format: { dtype: 'fp8_e4m3' }, elemsPerToken: 16 },
      },
    };
    expect(attentionScores(op)).toBe(32 * (128 + 32));
  });
  it('decode: heads·kvLen per sequence, times seqs', () => {
    const decode: AttentionOp = { ...base, qLen: 1, kvLen: 8193 };
    expect(attentionScores(decode)).toBe(32 * 8193);
    expect(attentionScores({ ...decode, seqs: 64 })).toBe(64 * 32 * 8193);
  });
  it('scores · 2·(qk + v) = the attention FLOP (same pairs as the MMAs)', () => {
    expect(attentionScores(base) * 2 * (128 + 128)).toBe(opFlops(base));
  });
});
