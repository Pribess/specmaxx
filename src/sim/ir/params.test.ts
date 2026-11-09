import { describe, expect, it } from 'vitest';
import { activeBlockParams, activeParamsPerToken, blockParams, countParams } from './params';
import { decoderLayer, decoderModel, hyperCollapse, singleBlockLayer, type HcPreParams } from './builders';
import type {
  CsaAttentionBlock,
  GqaAttentionBlock,
  HcPreBlock,
  IndexerSpec,
  LinearAttentionBlock,
  MlaAttentionBlock,
  MoeBlock,
  NgramEmbeddingBlock,
} from './model';
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
  it('latent MoE: experts at l, router and shared at h, plus 2·h·l projections', () => {
    // h 64, l 32, E 8, k 2, fe 16, S 1, fs 64
    const moe: MoeBlock = {
      kind: 'moe',
      name: 'moe',
      hidden: 64,
      experts: 8,
      topK: 2,
      expertIntermediate: 16,
      gated: true,
      activation: 'silu',
      sharedExperts: 1,
      sharedIntermediate: 64,
      latentDim: 32,
    };
    // routed 8·3·32·16 = 12,288 + router 8·64 = 512 + latent 2·64·32 = 4096 + shared 3·64·64 = 12,288
    expect(blockParams(moe)).toBe(29_184);
    // active: routed 2·3·32·16 = 3072 + 512 + 4096 + 12,288
    expect(activeBlockParams(moe)).toBe(19_968);
    // non-gated: 2 matrices per expert and shared expert: 8·2·32·16 = 8192 + 512 + 4096 + 2·64·64 = 8192
    const plain: MoeBlock = { ...moe, gated: false, activation: 'gelu' };
    expect(blockParams(plain)).toBe(20_992);
    // active: 2·2·32·16 = 2048 + 512 + 4096 + 8192
    expect(activeBlockParams(plain)).toBe(14_848);
    // latentDim = h still adds the two projections (they exist in the checkpoint)
    expect(blockParams({ ...moe, latentDim: 64 })).toBe(8 * 3 * 64 * 16 + 512 + 2 * 64 * 64 + 12_288);
  });
  it('dense model: active = total', () => {
    expect(activeParamsPerToken(REF_DENSE_8B)).toBe(countParams(REF_DENSE_8B));
  });
  it('TINY_HYBRID drops 6 inactive experts in each of 4 MoE layers', () => {
    // 5,086,712 − 4·6·3·256·128 = 5,086,712 − 2,359,296
    expect(activeParamsPerToken(TINY_HYBRID)).toBe(2_727_416);
  });
});

describe('Nemotron-H blocks (Nemotron-3.5-Super-VL-120B-A12B, checked against the checkpoint headers)', () => {
  it('latent relu² MoE layer: E 512, k 22, fe 2688, l 1024, S 1, fs 5376, ungated', () => {
    const moe: MoeBlock = {
      kind: 'moe',
      name: 'moe',
      hidden: 4096,
      experts: 512,
      topK: 22,
      expertIntermediate: 2688,
      gated: false,
      activation: 'relu2',
      sharedExperts: 1,
      sharedIntermediate: 5376,
      latentDim: 1024,
    };
    // routed 512·2·1024·2688 = 2,818,572,288 + router 512·4096 = 2,097,152 + shared 2·4096·5376 = 44,040,192
    // + latent 2·4096·1024 = 8,388,608
    expect(blockParams(moe)).toBe(2_873_098_240);
    // active: routed 22·2·1024·2688 = 121,110,528 + 2,097,152 + 44,040,192 + 8,388,608
    expect(activeBlockParams(moe)).toBe(175_636_480);
  });
  it('Mamba-2 mixer: h 4096, G 8, H 128, N 128, P 64, K 4, conv bias', () => {
    const mamba: LinearAttentionBlock = {
      kind: 'linear_attention',
      name: 'mixer',
      variant: 'mamba2',
      hidden: 4096,
      qkHeads: 8,
      vHeads: 128,
      keyDim: 128,
      valueDim: 64,
      convKernel: 4,
      convBias: true,
      outputGate: true,
    };
    // in_proj 4096·(2048 + 16384 + 128) = 76,021,760; conv 10240·4 = 40,960 + bias 10,240; A_log, D, dt_bias 384;
    // norm γ 8192; out_proj 8192·4096 = 33,554,432 (headers: in_proj [18560, 4096], conv1d [10240, 1, 4], ...)
    expect(blockParams(mamba)).toBe(109_635_968);
    // without the conv bias: 10,240 fewer; a GDN block with the same dims keeps 2 gates per head, 2 vectors, γ dv
    expect(blockParams({ ...mamba, convBias: false })).toBe(109_625_728);
    expect(blockParams({ ...mamba, variant: 'gated_delta_net' })).toBe(109_635_968 + 4096 * 128 - 128 - 8192 + 64);
  });
});

describe('singleBlockLayer (one sublayer per layer, Nemotron-H)', () => {
  it('mixer layer: norm → Mamba-2 → residual, with convBias / chunkTokens passed through', () => {
    const layer = singleBlockLayer({
      hidden: 64,
      name: 'layer.0',
      mixer: { kind: 'linear', variant: 'mamba2', qkHeads: 1, vHeads: 2, keyDim: 16, valueDim: 8, outputGate: true, convBias: true, chunkTokens: 128 },
    });
    expect(layer.name).toBe('layer.0');
    expect(layer.blocks.map((b) => `${b.kind}:${b.name}`)).toEqual(['norm:input_norm', 'linear_attention:linear_attn', 'residual:mixer_residual']);
    const mixer = layer.blocks[1];
    expect(mixer.kind === 'linear_attention' && [mixer.convBias, mixer.chunkTokens, mixer.convKernel]).toEqual([true, 128, 4]);
  });
  it('ffn layer: ffn_norm → moe → residual; optional fields absent when not given', () => {
    const layer = singleBlockLayer({
      hidden: 64,
      ffn: { kind: 'moe', experts: 4, topK: 2, expertIntermediate: 32, gated: false, activation: 'relu2' },
    });
    expect(layer.blocks.map((b) => `${b.kind}:${b.name}`)).toEqual(['norm:ffn_norm', 'moe:moe', 'residual:ffn_residual']);
    const plain = singleBlockLayer({ hidden: 64, mixer: { kind: 'linear', variant: 'mamba2', qkHeads: 1, vHeads: 2, keyDim: 16, valueDim: 8 } });
    expect(Object.keys(plain.blocks[1])).not.toContain('convBias');
    expect(Object.keys(plain.blocks[1])).not.toContain('chunkTokens');
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
        mixer: { kind: 'gqa', heads: 2, kvHeads: 1, headDim: 32, slidingWindow: 8, sparse: { kind: 'indexer', indexer: { heads: 1, headDim: 8, topK: 4 } } },
        ffn: { kind: 'dense', intermediate: 128 },
      }),
    ).toThrow();
  });
});

// Hand-derived counts for the new blocks, built with the real dims of the shortlist models. Tensor shapes from the
// research notes (safetensors headers): Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash, GLM-5.3-Flash, Mistral Large 4,
// Inkling-Small.

/** A GLM-5.3-Flash KDA layer: 64 heads, dk = dv = 128, conv 4, gate rank 128, h 4096. */
const GLM_KDA: LinearAttentionBlock = {
  kind: 'linear_attention',
  name: 'linear_attn',
  variant: 'kda',
  hidden: 4096,
  qkHeads: 64,
  vHeads: 64,
  keyDim: 128,
  valueDim: 128,
  convKernel: 4,
  outputGate: true,
  gateRank: 128,
};

/** DeepSeek-V4.1-Flash CSA attention without its global branch (layers 0-1). */
const DS_CSA_SWA: CsaAttentionBlock = {
  kind: 'csa_attention',
  name: 'attn',
  hidden: 5120,
  heads: 64,
  headDim: 512,
  ropeDim: 64,
  qLoraRank: 1280,
  outGroups: 8,
  outLoraRank: 1024,
  windowTokens: 128,
};

/** DeepSeek-V4.1 layer-2 indexer: 32 heads × 128, top-512 entries, keys from the 512-wide latent pooled by r = 2. */
const DS_INDEXER: IndexerSpec = { heads: 32, headDim: 128, topK: 512, keyPoolTokens: 2, keyInputDim: 512 };

describe('new block parameter counts', () => {
  it('KDA (GLM-5.3-Flash): q,k,v,f,g,b projections 104,071,168', () => {
    // q,k,v 4096·3·8192 = 100,663,296; β 4096·64 = 262,144; forget 4096·128 + 128·8192 = 1,572,864;
    // out gate 1,572,864 → 104,071,168 (R3 §5 "104.07 M"); conv 3·8192·4 = 98,304; A_log 64; dt_bias 8192;
    // o_norm 128; o_proj 8192·4096 = 33,554,432
    expect(blockParams(GLM_KDA)).toBe(104_071_168 + 98_304 + 64 + 8192 + 128 + 33_554_432);
  });

  it('CSA (DeepSeek-V4.1-Flash): attention without compressor/indexer is 126.6 M', () => {
    // q_a 5120·1280 = 6,553,600 + q_norm 1280 + q_b 1280·64·512 = 41,943,040; wkv 5120·512 = 2,621,440 + kv_norm 512;
    // wo_a 8·1024·4096 = 33,554,432; wo_b 8192·5120 = 41,943,040
    expect(blockParams(DS_CSA_SWA)).toBe(126_617_344);
  });

  it('CSA kv-source layer (r = 2) adds the compressor and the indexer; a consumer adds nothing', () => {
    const owner: CsaAttentionBlock = {
      ...DS_CSA_SWA,
      global: { compressRatio: 2, selection: { kind: 'indexer', indexer: DS_INDEXER } },
    };
    // compressor wkv + wgate 2·5120·512 + norm 512 = 5,243,392
    // indexer wq_b 1280·32·128 = 5,242,880 + wk 512·128 = 65,536 + key norm 2·128 + weights_proj 5120·32 = 163,840
    //   = 5,472,512 (the checkpoint's RMSNorm k_norm has 128 params, not 256)
    expect(blockParams(owner)).toBe(126_617_344 + 5_243_392 + 5_472_512);
    const r1Owner: CsaAttentionBlock = { ...owner, global: { ...owner.global!, compressRatio: 1 } };
    // r = 1: wkv 5120·512 + norm 512 = 2,621,952
    expect(blockParams(r1Owner)).toBe(126_617_344 + 2_621_952 + 5_472_512);
    const consumer: CsaAttentionBlock = {
      ...DS_CSA_SWA,
      global: { compressRatio: 2, kvCacheFromLayer: 2, selection: { kind: 'reuse', topK: 512 } },
    };
    expect(blockParams(consumer)).toBe(126_617_344);
    // Reindex layer (24): own query and head weights, keys read from layer 20
    const reindex: CsaAttentionBlock = {
      ...DS_CSA_SWA,
      global: {
        compressRatio: 1,
        kvCacheFromLayer: 20,
        selection: { kind: 'indexer', indexer: { ...DS_INDEXER, keyPoolTokens: 1, ownsKeys: false, maxScoredKeys: 16384 } },
      },
    };
    expect(blockParams(reindex)).toBe(126_617_344 + 5_242_880 + 163_840);
  });

  it('indexer variants: GLM k-pool gate, Qwen QSA without head weights, reuse', () => {
    const mla = (sparse: MlaAttentionBlock['sparse']): MlaAttentionBlock => ({
      kind: 'mla_attention',
      name: 'attn',
      hidden: 4096,
      heads: 64,
      qLoraRank: 1536,
      kvLoraRank: 512,
      qkNopeDim: 256,
      qkRopeDim: 0,
      vHeadDim: 256,
      ...(sparse !== undefined ? { sparse } : {}),
    });
    const glmIndexer: IndexerSpec = { heads: 32, headDim: 128, topK: 2048, keyPoolTokens: 4, keyPoolGate: true };
    // wq_b 1536·32·128 = 6,291,456; wk 4096·128 + k_norm 256 = 524,544; gate 4096·128 + APE 4·128 = 524,800;
    // weights_proj 4096·32 = 131,072 → 7,471,872 (R3 §5 "indexer 7.4 M")
    expect(blockParams(mla({ kind: 'indexer', indexer: glmIndexer })) - blockParams(mla(undefined))).toBe(7_471_872);
    expect(blockParams(mla({ kind: 'reuse', topK: 2048 }))).toBe(blockParams(mla(undefined)));

    const qwenAttn: GqaAttentionBlock = {
      kind: 'gqa_attention',
      name: 'attn',
      hidden: 2560,
      heads: 24,
      kvHeads: 2,
      headDim: 256,
      qkvBias: false,
      outBias: false,
      qkNorm: true,
      rope: true,
      causal: true,
      outputGate: true,
    };
    const qsa: IndexerSpec = { heads: 4, headDim: 128, topK: 2048, keyPoolTokens: 4, headWeights: false };
    // index_qk_proj 640·2560 = 1,638,400 = q 2560·4·128 + k 2560·128; q/k layernorm 2·128 = 256
    const withQsa = blockParams({ ...qwenAttn, sparse: { kind: 'indexer', indexer: qsa } });
    expect(withQsa - blockParams(qwenAttn)).toBe(1_638_400 + 256);
  });

  it('GQA with relative bias and K/V convs (Inkling-Small global layer)', () => {
    const inkling: GqaAttentionBlock = {
      kind: 'gqa_attention',
      name: 'attn',
      hidden: 4096,
      heads: 32,
      kvHeads: 8,
      headDim: 128,
      qkvBias: false,
      outBias: false,
      qkNorm: true,
      rope: false,
      causal: true,
      outputGate: false,
      relativeBias: { relDim: 16, extentTokens: 1024 },
      kvConvKernel: 4,
    };
    // wq 4096·4096 = 16,777,216; wk, wv 2·1024·4096 = 8,388,608; wo 16,777,216; q/k norm 256;
    // wr 4096·32·16 = 2,097,152; rel profile 16·1024 = 16,384; k/v sconv 2·1024·4 = 8192
    expect(blockParams(inkling)).toBe(44_065_024);
  });

  it('MLA reading an earlier latent drops kv_a and its norm (Mistral Large 4 layer 39)', () => {
    const own: MlaAttentionBlock = {
      kind: 'mla_attention',
      name: 'attn',
      hidden: 6144,
      heads: 64,
      qLoraRank: 0,
      kvLoraRank: 512,
      qkNopeDim: 128,
      qkRopeDim: 0,
      vHeadDim: 128,
    };
    // h·(rkv + dr) + rkv = 6144·512 + 512
    expect(blockParams(own) - blockParams({ ...own, kvCacheFromLayer: 35 })).toBe(3_146_240);
  });

  it('hc_pre: Qwen Gated Residual, DeepSeek mHC, GLM mHC; hc_post is free', () => {
    const qwenPre: HcPreParams = {
      normGamma: true,
      projections: [
        { name: 'hc.down', inputDim: 10240, outputDim: 320 },
        { name: 'hc.up', inputDim: 320, outputDim: 10240 },
        { name: 'hc.inject', inputDim: 10240, outputDim: 4 },
      ],
      sinkhornIters: 0,
      vectorParams: 0,
    };
    const qwen: HcPreBlock = { kind: 'hc_pre', name: 'hc', streams: 4, dim: 2560, ...qwenPre };
    // hc_norm 10240 + down 3,276,800 + up 3,276,800 + inject 40,960 (R1 tensors) → ×2·48 = 634,060,800
    expect(blockParams(qwen)).toBe(6_604_800);
    // final hyper_connection_mixer: hc_norm + down + up, no inject
    const final = hyperCollapse({ streams: 4, dim: 2560, pre: { ...qwenPre, projections: qwenPre.projections.slice(0, 2) } });
    expect(blockParams(final)).toBe(6_563_840);

    const mhc = (dim: number): HcPreBlock => ({
      kind: 'hc_pre',
      name: 'hc',
      streams: 4,
      dim,
      normGamma: false,
      projections: [{ name: 'hc.fn', inputDim: 4 * dim, outputDim: 24 }],
      sinkhornIters: 20,
      vectorParams: 24 + 3,
    });
    // DeepSeek hc_attn_fn [24, 20480] + base 24 + scale 3; two per layer × 40 = 39.3 M
    expect(blockParams(mhc(5120))).toBe(24 * 20480 + 27);
    expect(40 * 2 * blockParams(mhc(5120))).toBe(39_323_760);
    // GLM hc_attn_fn [24, 16384] + 27; two per layer = 786,486 ("0.79 M")
    expect(2 * blockParams(mhc(4096))).toBe(786_486);
    expect(blockParams({ kind: 'hc_post', name: 'p', streams: 4, dim: 4096, mixesStreams: true })).toBe(0);
  });

  it('ngram: Qwen PLE table counts in total, only the gathered rows in active', () => {
    const ple: NgramEmbeddingBlock = {
      kind: 'ngram_embedding',
      name: 'ple',
      // 16 heads · 20,000,000 rounded up to 128; the checkpoint's 128 × 2,500,012 = 320,001,536 rows (rel. err 4.8e-6)
      tableRows: 320_000_000,
      rowDim: 160,
      rowsPerToken: 16,
      projections: [
        { name: 'ple.key', inputDim: 2560, outputDim: 10240 },
        { name: 'ple.value', inputDim: 2560, outputDim: 2560 },
      ],
      outputDim: 10240,
      conv: { kernel: 4, dilation: 3 },
      vectorParams: 3 * 10240,
    };
    // table 51,200,000,000; key 26,214,400 + value 6,553,600 + conv 10240·4 = 40,960 + norms 30,720 = 32,839,680
    expect(blockParams(ple)).toBe(51_200_000_000 + 32_839_680);
    expect(Math.abs(320_000_000 * 160 - 51_200_245_760) / 51_200_245_760).toBeLessThan(1e-4);
    // active: the 16 gathered rows of 160 instead of the table
    expect(activeBlockParams(ple)).toBe(32_839_680 + 16 * 160);

    const engram: NgramEmbeddingBlock = {
      kind: 'ngram_embedding',
      name: 'engram',
      tableRows: 384_006_168,
      rowDim: 256,
      rowsPerToken: 24,
      projections: [{ name: 'engram.wkv', inputDim: 6144, outputDim: 5 * 5120 }],
      outputDim: 4 * 5120,
      vectorParams: 2 * 4 * 5120,
    };
    // table 98,305,579,008 + wkv 157,286,400 + q/k weights 40,960
    expect(blockParams(engram)).toBe(98_305_579_008 + 157_286_400 + 40_960);
  });

  it('short_conv: dim·K', () => {
    expect(blockParams({ kind: 'short_conv', name: 'sc', dim: 4096, kernel: 4 })).toBe(16_384);
  });

  it("finalNorm 'none' has no final-norm params", () => {
    const layers = TINY_DENSE.layers;
    const none = decoderModel({ name: 'n', hidden: 256, vocab: 1000, tieEmbeddings: false, layers, finalNorm: 'none' });
    expect(countParams(TINY_DENSE) - countParams(none)).toBe(256);
  });
});

describe('builders: hyper-connections, n-gram, short conv, beforeHead', () => {
  const pre: HcPreParams = {
    normGamma: false,
    projections: [{ name: 'hc.fn', inputDim: 256, outputDim: 24 }],
    sinkhornIters: 20,
    vectorParams: 27,
  };
  it('hyper layout with sub-norms (DeepSeek/GLM) and without (Qwen)', () => {
    const p = {
      hidden: 64,
      mixer: { kind: 'gqa', heads: 2, kvHeads: 1, headDim: 32 },
      ffn: { kind: 'dense', intermediate: 128 },
    } as const;
    const ds = decoderLayer({ ...p, residual: { kind: 'hyper', streams: 4, pre, mixesStreams: true, subNorms: true } });
    expect(ds.blocks.map((b) => b.kind)).toEqual(['hc_pre', 'norm', 'gqa_attention', 'hc_post', 'hc_pre', 'norm', 'mlp', 'hc_post']);
    const qwen = decoderLayer({ ...p, residual: { kind: 'hyper', streams: 4, pre, mixesStreams: false, subNorms: false } });
    expect(qwen.blocks.map((b) => b.kind)).toEqual(['hc_pre', 'gqa_attention', 'hc_post', 'hc_pre', 'mlp', 'hc_post']);
  });
  it('ngram goes first; short convs follow each branch (Inkling)', () => {
    const layer = decoderLayer({
      hidden: 64,
      mixer: { kind: 'gqa', heads: 2, kvHeads: 1, headDim: 32 },
      ffn: { kind: 'dense', intermediate: 128 },
      shortConv: { kernel: 4 },
      ngram: { tableRows: 100, rowDim: 8, rowsPerToken: 2, projections: [], outputDim: 64, vectorParams: 0 },
    });
    expect(layer.blocks.map((b) => b.kind)).toEqual([
      'ngram_embedding',
      'norm',
      'gqa_attention',
      'short_conv',
      'residual',
      'norm',
      'mlp',
      'short_conv',
      'residual',
    ]);
  });
  it('beforeHead blocks are appended to the last layer', () => {
    const layer = (name: string) =>
      decoderLayer({ name, hidden: 64, mixer: { kind: 'gqa', heads: 2, kvHeads: 1, headDim: 32 }, ffn: { kind: 'dense', intermediate: 128 } });
    const model = decoderModel({
      name: 'm',
      hidden: 64,
      vocab: 10,
      tieEmbeddings: true,
      layers: [layer('layer.0'), layer('layer.1')],
      residualStreams: 4,
      beforeHead: [hyperCollapse({ streams: 4, dim: 64, pre: { ...pre, projections: [] } })],
    });
    expect(model.residualStreams).toBe(4);
    expect(model.layers[0].blocks.at(-1)?.kind).toBe('residual');
    expect(model.layers[1].blocks.at(-1)?.kind).toBe('hc_pre');
  });
});
