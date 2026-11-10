import { describe, expect, it } from 'vitest';
import { blockSequenceStateBytes, lowerBlock, type LoweringContext } from './lower';
import { opBytes } from '../ir/metrics';
import type { AttentionOp } from '../ir/ops';
import type { Dtype } from '../ir/dtype';
import type { GqaAttentionBlock, MlaAttentionBlock } from '../ir/model';

/** KV bytes per token per layer = elements/token · bytesPerElement(kv dtype), block-scale bits included. */
function ctxFor(kv: Dtype, tp = 1): LoweringContext {
  return {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 1,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: kv } },
    parallel: { tp, pp: 1, dp: 1, ep: 1 },
  };
}

/** DeepSeek-V3 MLA attention: latent rkv = 512 plus a shared RoPE key dr = 64, 61 layers. */
const DSV3_MLA: MlaAttentionBlock = {
  kind: 'mla_attention',
  name: 'attn',
  hidden: 7168,
  heads: 128,
  qLoraRank: 1536,
  kvLoraRank: 512,
  qkNopeDim: 128,
  qkRopeDim: 64,
  vHeadDim: 128,
};
const DSV3_LAYERS = 61;

/** Llama-3-70B-like GQA: 8 KV heads of 128. */
const GQA: GqaAttentionBlock = {
  kind: 'gqa_attention',
  name: 'attn',
  hidden: 8192,
  heads: 64,
  kvHeads: 8,
  headDim: 128,
  qkvBias: false,
  outBias: false,
  qkNorm: false,
  rope: true,
  causal: true,
  outputGate: false,
};
const GQA_LAYERS = 80;

// bytes per element = (valueBits + blockScaleBitsPerElem) / 8
const BPE: [Dtype, number][] = [
  ['bf16', 2],
  ['fp16', 2],
  ['fp8_e4m3', 1],
  ['fp8_e5m2', 1],
  ['mxfp8', 1 + 1 / 32], // 8-bit E8M0 scale per 32 elements
  ['fp4_e2m1', 0.5],
  ['mxfp4', 0.5 + 1 / 32], // 8 bits / 32 elements = 0.25 bit/elem
  ['nvfp4', 0.5 + 1 / 16], // 8-bit E4M3 scale per 16 elements = 0.5 bit/elem
];

describe('KV bytes per token by dtype', () => {
  it.each(BPE)('MLA (DeepSeek-V3 shape), %s', (dtype, bpe) => {
    // Formula: per token = layers · (rkv + dr) · bpe
    const perToken = DSV3_LAYERS * (512 + 64) * bpe;
    const measured = DSV3_LAYERS * blockSequenceStateBytes(DSV3_MLA, ctxFor(dtype), 1);
    expect(measured).toBeCloseTo(perToken, 9);
  });

  it('DeepSeek-V3 FP8: 61 · 576 · 1 B = 35,136 B/token, half of BF16', () => {
    expect(DSV3_LAYERS * blockSequenceStateBytes(DSV3_MLA, ctxFor('fp8_e4m3'), 1)).toBe(35136);
    expect(DSV3_LAYERS * blockSequenceStateBytes(DSV3_MLA, ctxFor('bf16'), 1)).toBe(70272);
  });

  it.each(BPE)('GQA (8 KV heads × 128), %s', (dtype, bpe) => {
    // Formula: per token = layers · 2·KVh·hd · bpe
    const perToken = GQA_LAYERS * 2 * 8 * 128 * bpe;
    const measured = GQA_LAYERS * blockSequenceStateBytes(GQA, ctxFor(dtype), 1);
    expect(measured).toBeCloseTo(perToken, 9);
  });

  it('GQA KV is sharded across TP ranks; MLA latent is replicated', () => {
    expect(blockSequenceStateBytes(GQA, ctxFor('nvfp4', 4), 1)).toBeCloseTo(2 * 2 * 128 * 0.5625, 9);
    expect(blockSequenceStateBytes(DSV3_MLA, ctxFor('nvfp4', 4), 1)).toBeCloseTo(576 * 0.5625, 9);
  });

  it('scaled FP4 KV costs the scale bits: NVFP4 is 9/16 of FP8 per element, not 1/2', () => {
    const fp8 = blockSequenceStateBytes(GQA, ctxFor('fp8_e4m3'), 1000);
    const nv = blockSequenceStateBytes(GQA, ctxFor('nvfp4'), 1000);
    expect(nv / fp8).toBeCloseTo(0.5625 / 1, 9);
  });
});

describe('decode KV read bytes carry the block-scale overhead', () => {
  const SEQS = 3;
  const KV_LEN = 1000;
  const decodeCtx = (kv: Dtype): LoweringContext => ({ ...ctxFor(kv), seqs: SEQS, kvLen: KV_LEN });
  const attention = (block: GqaAttentionBlock | MlaAttentionBlock, kv: Dtype): AttentionOp =>
    lowerBlock(block, decodeCtx(kv)).find((o): o is AttentionOp => o.kind === 'attention')!;
  const cases: [Dtype, number][] = [
    ['bf16', 2],
    ['fp8_e4m3', 1],
    ['mxfp8', 1 + 1 / 32],
    ['nvfp4', 0.5625],
  ];

  it.each(cases)('GQA, %s: kvRead = seqs·kvLen·2·KVl·hd·bpe', (dtype, bpe) => {
    // Formula: kvRead = seqs · kvTokensRead · elemsPerToken · bpe(kv)
    const expected = SEQS * KV_LEN * 2 * 8 * 128 * bpe;
    expect(opBytes(attention(GQA, dtype)).kvReadBytes).toBeCloseTo(expected, 6);
  });

  it.each(cases)('MLA, %s: kvRead = seqs·kvLen·(rkv+dr)·bpe', (dtype, bpe) => {
    const expected = SEQS * KV_LEN * (512 + 64) * bpe;
    expect(opBytes(attention(DSV3_MLA, dtype)).kvReadBytes).toBeCloseTo(expected, 6);
  });
});
