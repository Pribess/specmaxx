import { describe, expect, it } from 'vitest';
import { activeBlockParams, activeParamsPerToken, blockParams, countParams } from './params';
import { decoderLayer } from './builders';
import type { MlaAttentionBlock, MoeBlock } from './model';
import { REF_DENSE_8B, TINY_DENSE, TINY_HYBRID, TINY_HYBRID_DENSE_FIRST, TINY_MLA_SPARSE } from '../models/reference';

describe('T3 countParams', () => {
  it('REF_DENSE_8B matches Qwen3-8B', () => {
    // layer = 41,943,040 attn + 256 qk-norm + 150,994,944 mlp + 8192 norms = 192,946,432; ×36 = 6,946,071,552
    // + embed + head 2·151936·4096 = 1,244,659,712 + final norm 4096
    expect(countParams(REF_DENSE_8B)).toBe(8_190_735_360);
  });

  it('MLA block with DeepSeek-V3 dims', () => {
    const block: MlaAttentionBlock = {
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
    // h·rq 11,010,048 + rq 1536 + rq·H·(dn+dr) 37,748,736 + h·(rkv+dr) 4,128,768 + rkv 512
    // + rkv·H·(dn+dv) 16,777,216 + H·dv·h 117,440,512
    expect(blockParams(block)).toBe(187_107_328);
  });

  it('TINY_DENSE', () => {
    // attn 256·(4+4)·64 + 4·64·256 = 131,072 + 65,536 = 196,608; mlp 3·256·512 = 393,216; norms 512
    // layer 590,336 ×2 = 1,180,672; + 2·1000·256 = 512,000 + final norm 256
    expect(countParams(TINY_DENSE)).toBe(1_692_928);
  });

  it('TINY_HYBRID', () => {
    // GDN: in_proj 256·(2·2·32 + 4·32·2 + 2·4) = 256·392 = 100,352; conv (128 + 128)·4 = 1024; A/dt 8;
    //      out-norm 32; out_proj 128·256 = 32,768 → 134,184
    // MoE: 8·3·256·128 = 786,432 + router 8·256 = 2048 + shared 3·256·256 = 196,608 → 985,088
    // gated GQA: 256·(4·2 + 2·1)·64 = 163,840 + o 65,536 + qk-norm 128 → 229,504
    // layers: 3·(134,184 + 985,088 + 512) + (229,504 + 985,088 + 512) = 3,359,352 + 1,215,104 = 4,574,456
    // + 512,000 + 256
    expect(countParams(TINY_HYBRID)).toBe(5_086_712);
    // dense first: layer 0 FFN 3·256·512 = 393,216 instead of 985,088 → −591,872
    expect(countParams(TINY_HYBRID_DENSE_FIRST)).toBe(4_494_840);
  });

  it('TINY_MLA_SPARSE', () => {
    // q: 512·128 + 128 + 128·8·48 = 114,816; kv: 512·80 + 64 + 64·8·64 = 73,792; o: 8·32·512 = 131,072
    // indexer: 128·4·32 + 512·32 + 512·4 + 2·32 = 34,880 → attn 354,560
    // mlp 3·512·1024 = 1,572,864; norms 1024 → layer 1,928,448 ×2 = 3,856,896; + 1,024,000 + 512
    expect(countParams(TINY_MLA_SPARSE)).toBe(4_881_408);
  });
});

describe('activeParamsPerToken', () => {
  it('MoE block counts top-k + router + shared', () => {
    const moe: MoeBlock = {
      kind: 'moe',
      name: 'moe',
      hidden: 256,
      experts: 8,
      topK: 2,
      expertIntermediate: 128,
      gated: true,
      activation: 'silu',
      sharedExperts: 1,
      sharedIntermediate: 256,
    };
    // 2·3·256·128 = 196,608 + router 2048 + shared 196,608
    expect(activeBlockParams(moe)).toBe(395_264);
    expect(blockParams(moe)).toBe(985_088);
  });
  it('dense model: active = total', () => {
    expect(activeParamsPerToken(REF_DENSE_8B)).toBe(countParams(REF_DENSE_8B));
  });
  it('TINY_HYBRID drops 6 inactive experts in each of 4 MoE layers', () => {
    // 5,086,712 − 4·6·3·256·128 = 5,086,712 − 2,359,296
    expect(activeParamsPerToken(TINY_HYBRID)).toBe(2_727_416);
  });
});

describe('builders', () => {
  it('decoderLayer orders blocks pre-norm style', () => {
    const layer = decoderLayer({
      hidden: 64,
      mixer: { kind: 'gqa', heads: 2, kvHeads: 1, headDim: 32 },
      ffn: { kind: 'dense', intermediate: 128 },
      postNorms: true,
    });
    expect(layer.blocks.map((b) => b.kind)).toEqual([
      'norm',
      'gqa_attention',
      'norm',
      'residual',
      'norm',
      'mlp',
      'norm',
      'residual',
    ]);
  });
  it('rejects slidingWindow with sparse', () => {
    expect(() =>
      decoderLayer({
        hidden: 64,
        mixer: { kind: 'gqa', heads: 2, kvHeads: 1, headDim: 32, slidingWindow: 8, sparse: { heads: 1, headDim: 8, topK: 4 } },
        ffn: { kind: 'dense', intermediate: 128 },
      }),
    ).toThrow();
  });
});
