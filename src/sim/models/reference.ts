/**
 * Reference models with HAND-WRITTEN dims (no Hugging Face config dependency), used by tests and the demo.
 * `REF_DENSE_8B` copies Qwen3-8B's published shape so its size can be sanity-checked; the `TINY_*` fixtures have
 * made-up dims that exercise every block kind (hybrid linear attention, MoE with shared experts, dense first
 * layers, gated attention, MLA with a sparse indexer, DeepSeek-V4.1 CSA, hyper-connections and n-gram tables, KDA,
 * cross-layer KV and top-k sharing, Inkling relative bias and short convs, Kimi K3 latent MoE, gated MLA and
 * Attention Residuals).
 */

import {
  attnResCollapse,
  decoderLayer,
  decoderModel,
  hyperCollapse,
  repeatLayers,
} from '../ir/builders';
import type { FfnParams, MixerParams } from '../ir/builders';
import type { ModelSpec } from '../ir/model';

/**
 * Qwen3-style dense GQA decoder (q/k-norm, SwiGLU, RMSNorm), dims equal to Qwen3-8B: h 4096, 36 layers,
 * 32 heads, 8 KV heads, head dim 128, ffn 12288, vocab 151936, untied embeddings. 8,190,735,360 params.
 * Shape source: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json (copied by hand).
 */
export const REF_DENSE_8B: ModelSpec = decoderModel({
  name: 'REF_DENSE_8B (Qwen3-8B shape)',
  hidden: 4096,
  vocab: 151936,
  tieEmbeddings: false,
  source: 'https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json (dims copied by hand)',
  layers: repeatLayers(36, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 4096,
      mixer: { kind: 'gqa', heads: 32, kvHeads: 8, headDim: 128, qkNorm: true },
      ffn: { kind: 'dense', intermediate: 12288 },
    }),
  ),
});

/**
 * Qwen3-30B-A3B shape: h 2048, 48 layers, 32/4 GQA heads of 128 elements, 128 routed experts,
 * top-8, expert width 768, no shared experts, untied vocab 151936. 30,532,122,624 parameters.
 * Source: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json (dims copied by hand).
 */
export const REF_MOE_30B_A3B: ModelSpec = decoderModel({
  name: 'REF_MOE_30B_A3B (Qwen3-30B-A3B shape)',
  hidden: 2048,
  vocab: 151936,
  tieEmbeddings: false,
  source: 'https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json (dims copied by hand)',
  layers: repeatLayers(48, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 2048,
      mixer: { kind: 'gqa', heads: 32, kvHeads: 4, headDim: 128, qkNorm: true },
      ffn: { kind: 'moe', experts: 128, topK: 8, expertIntermediate: 768 },
    }),
  ),
});
/** Tiny dense GQA model (made-up dims): h 256, 2 layers, 4 heads, 2 KV heads, head dim 64, ffn 512, vocab 1000. */
export const TINY_DENSE: ModelSpec = decoderModel({
  name: 'TINY_DENSE',
  hidden: 256,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture',
  layers: repeatLayers(2, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 256,
      mixer: { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 64 },
      ffn: { kind: 'dense', intermediate: 512 },
    }),
  ),
});

/** GatedDeltaNet mixer of the tiny hybrid: 2 q/k heads, 4 value heads, dk = dv = 32, conv 4, z-gate. */
const TINY_GDN: MixerParams = {
  kind: 'linear',
  variant: 'gated_delta_net',
  qkHeads: 2,
  vHeads: 4,
  keyDim: 32,
  valueDim: 32,
  convKernel: 4,
  outputGate: true,
};

/** Gated full attention of the tiny hybrid: 4 heads, 1 KV head, head dim 64, q/k-norm, output gate. */
const TINY_GATED_GQA: MixerParams = {
  kind: 'gqa',
  heads: 4,
  kvHeads: 1,
  headDim: 64,
  qkNorm: true,
  outputGate: true,
};

/** MoE of the tiny hybrid: 8 experts, top-2, expert width 128, one shared expert of width 256. */
const TINY_MOE: FfnParams = {
  kind: 'moe',
  experts: 8,
  topK: 2,
  expertIntermediate: 128,
  sharedExperts: 1,
  sharedIntermediate: 256,
};

/** Builds the tiny hybrid: layers 0–2 GDN, layer 3 gated GQA, all MoE except layer 0 when `denseFirst`. */
function tinyHybrid(name: string, denseFirst: boolean): ModelSpec {
  return decoderModel({
    name,
    hidden: 256,
    vocab: 1000,
    tieEmbeddings: false,
    source: 'made-up test fixture (Qwen3-Next-style 3:1 hybrid)',
    layers: repeatLayers(4, (i) =>
      decoderLayer({
        name: `layer.${i}`,
        hidden: 256,
        mixer: i < 3 ? TINY_GDN : TINY_GATED_GQA,
        ffn: denseFirst && i === 0 ? { kind: 'dense', intermediate: 512 } : TINY_MOE,
      }),
    ),
  });
}

/**
 * Tiny Qwen3-Next-style hybrid (made-up dims): h 256, 4 layers; layers 0–2 GatedDeltaNet (qkH 2, vH 4,
 * dk = dv = 32, conv 4, gate), layer 3 gated GQA (4 heads, 1 KV, hd 64, qk-norm); every FFN a MoE
 * (E 8, top-2, fe 128, 1 shared of 256). Vocab 1000.
 */
export const TINY_HYBRID: ModelSpec = tinyHybrid('TINY_HYBRID', false);

/** {@link TINY_HYBRID} with a dense first FFN (layer 0: dense MLP of width 512). */
export const TINY_HYBRID_DENSE_FIRST: ModelSpec = tinyHybrid('TINY_HYBRID_DENSE_FIRST', true);

/**
 * Tiny DeepSeek-V3.2-style MLA model (made-up dims): h 512, 2 layers, MLA (8 heads, rq 128, rkv 64, dn 32,
 * dr 16, dv 32) with a DSA indexer (4 heads, hd 32, top-64); dense ffn 1024; vocab 1000.
 */
export const TINY_MLA_SPARSE: ModelSpec = decoderModel({
  name: 'TINY_MLA_SPARSE',
  hidden: 512,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture (DeepSeek-V3.2-style)',
  layers: repeatLayers(2, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 512,
      mixer: {
        kind: 'mla',
        heads: 8,
        qLoraRank: 128,
        kvLoraRank: 64,
        qkNopeDim: 32,
        qkRopeDim: 16,
        vHeadDim: 32,
        sparse: { kind: 'indexer', indexer: { heads: 4, headDim: 32, topK: 64 } },
      },
      ffn: { kind: 'dense', intermediate: 1024 },
    }),
  ),
});

/** CSA mixer of {@link TINY_CSA} (made-up dims): 4 heads, hd 32 (K = V), rope 8, rq 32, 2 output groups of rank 16, window 16. */
const TINY_CSA_BASE: Omit<Extract<MixerParams, { kind: 'csa' }>, 'global'> = {
  kind: 'csa',
  heads: 4,
  headDim: 32,
  ropeDim: 8,
  qLoraRank: 32,
  outGroups: 2,
  outLoraRank: 16,
  windowTokens: 16,
};

/**
 * Tiny DeepSeek-V4.1-style CSA model (made-up dims): h 128, 4 layers, dense ffn 256, vocab 1000.
 * Layer 0: window only. Layer 1: owns the r = 2 compressed cache and runs an indexer (2 heads × 16, top-8 entries,
 * keys from the 32-wide latent). Layer 2: reads layer 1's cache and reuses its top-k. Layer 3 ("Reindex"): reads
 * layer 1's cache and indexer keys, runs its own query/scoring, at most 32 keys scored.
 */
export const TINY_CSA: ModelSpec = decoderModel({
  name: 'TINY_CSA',
  hidden: 128,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture (DeepSeek-V4.1-style)',
  layers: repeatLayers(4, (i) => {
    const owner = { heads: 2, headDim: 16, topK: 8, keyPoolTokens: 2, keyInputDim: 32 };
    const mixers: MixerParams[] = [
      TINY_CSA_BASE,
      {
        ...TINY_CSA_BASE,
        global: { compressRatio: 2, selection: { kind: 'indexer', indexer: owner } },
      },
      {
        ...TINY_CSA_BASE,
        global: { compressRatio: 2, kvCacheFromLayer: 1, selection: { kind: 'reuse', topK: 8 } },
      },
      {
        ...TINY_CSA_BASE,
        global: {
          compressRatio: 2,
          kvCacheFromLayer: 1,
          selection: { kind: 'indexer', indexer: { ...owner, ownsKeys: false, maxScoredKeys: 32 } },
        },
      },
    ];
    return decoderLayer({
      name: `layer.${i}`,
      hidden: 128,
      mixer: mixers[i],
      ffn: { kind: 'dense', intermediate: 256 },
    });
  }),
});

/**
 * Tiny model with mHC hyper-connections and an Engram-style n-gram table (made-up dims): h 64, N = 2 streams,
 * 2 layers of GQA (2 heads, 1 KV, hd 32) + dense ffn 128, sub-norms kept. Each hc_pre has one `fn` projection
 * 128 → (2+N)·N = 8, Sinkhorn 4 iterations, 11 vector params; hc_post mixes the streams. Layer 1 starts with an
 * n-gram block: 1000 rows of 8, 4 rows per token, one projection 32 → 192, writing 128 (N·D). The final collapse
 * has no projection. Vocab 1000.
 */
export const TINY_HC_NGRAM: ModelSpec = (() => {
  const pre = {
    normGamma: false,
    projections: [{ name: 'fn', inputDim: 128, outputDim: 8 }],
    sinkhornIters: 4,
    vectorParams: 11,
  };
  return decoderModel({
    name: 'TINY_HC_NGRAM',
    hidden: 64,
    vocab: 1000,
    tieEmbeddings: false,
    residualStreams: 2,
    source: 'made-up test fixture (DeepSeek-V4.1 mHC + Engram style)',
    layers: repeatLayers(2, (i) =>
      decoderLayer({
        name: `layer.${i}`,
        hidden: 64,
        mixer: { kind: 'gqa', heads: 2, kvHeads: 1, headDim: 32 },
        ffn: { kind: 'dense', intermediate: 128 },
        residual: { kind: 'hyper', streams: 2, pre, mixesStreams: true, subNorms: true },
        ...(i === 1
          ? {
              ngram: {
                name: 'engram',
                tableRows: 1000,
                rowDim: 8,
                rowsPerToken: 4,
                projections: [{ name: 'wkv', inputDim: 32, outputDim: 192 }],
                outputDim: 128,
                vectorParams: 256,
              },
            }
          : {}),
      }),
    ),
    beforeHead: [
      hyperCollapse({
        streams: 2,
        dim: 64,
        pre: { ...pre, projections: [], sinkhornIters: 0, vectorParams: 0 },
      }),
    ],
  });
})();

/**
 * Tiny GLM-5.3-Flash-style hybrid (made-up dims): h 128, layer 0 KDA (2 q/k heads, 2 value heads, dk = dv = 32,
 * conv 4, gate rank 16), layer 1 NoPE MLA (4 heads, rq 32, rkv 32, dn 16, dr 0, dv 16) with a k-pool indexer
 * (2 heads × 16, top-16 tokens, 4 tokens per key, gated pooling, q from rq); dense ffn 256; vocab 1000.
 */
export const TINY_KDA: ModelSpec = decoderModel({
  name: 'TINY_KDA',
  hidden: 128,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture (GLM-5.3-Flash-style)',
  layers: [
    decoderLayer({
      name: 'layer.0',
      hidden: 128,
      mixer: {
        kind: 'linear',
        variant: 'kda',
        qkHeads: 2,
        vHeads: 2,
        keyDim: 32,
        valueDim: 32,
        convKernel: 4,
        outputGate: true,
        gateRank: 16,
      },
      ffn: { kind: 'dense', intermediate: 256 },
    }),
    decoderLayer({
      name: 'layer.1',
      hidden: 128,
      mixer: {
        kind: 'mla',
        heads: 4,
        qLoraRank: 32,
        kvLoraRank: 32,
        qkNopeDim: 16,
        qkRopeDim: 0,
        vHeadDim: 16,
        sparse: {
          kind: 'indexer',
          indexer: { heads: 2, headDim: 16, topK: 16, keyPoolTokens: 4, keyPoolGate: true },
        },
      },
      ffn: { kind: 'dense', intermediate: 256 },
    }),
  ],
});

/** MLA mixer of {@link TINY_MLA_SHARED}: 4 heads, no q compression, rkv 32, dn 16, dr 8, dv 16. */
const TINY_SHARED_MLA: Extract<MixerParams, { kind: 'mla' }> = {
  kind: 'mla',
  heads: 4,
  qLoraRank: 0,
  kvLoraRank: 32,
  qkNopeDim: 16,
  qkRopeDim: 8,
  vHeadDim: 16,
};

/**
 * Tiny Mistral-Large-4 / GLM-5.3-style MLA model with cross-layer sharing (made-up dims): h 128, 3 layers, dense
 * ffn 256, vocab 1000. Layer 0 runs a DSA indexer (2 heads × 16, top-16); layer 1 reuses its top-k; layer 2 reads
 * layer 0's latent cache and reuses the top-k.
 */
export const TINY_MLA_SHARED: ModelSpec = decoderModel({
  name: 'TINY_MLA_SHARED',
  hidden: 128,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture (Mistral Large 4 / GLM-5.3 style)',
  layers: repeatLayers(3, (i) => {
    const mixers: MixerParams[] = [
      {
        ...TINY_SHARED_MLA,
        sparse: { kind: 'indexer', indexer: { heads: 2, headDim: 16, topK: 16 } },
      },
      { ...TINY_SHARED_MLA, sparse: { kind: 'reuse', topK: 16 } },
      { ...TINY_SHARED_MLA, kvCacheFromLayer: 0, sparse: { kind: 'reuse', topK: 16 } },
    ];
    return decoderLayer({
      name: `layer.${i}`,
      hidden: 128,
      mixer: mixers[i],
      ffn: { kind: 'dense', intermediate: 256 },
    });
  }),
});

/**
 * Tiny Inkling-style model (made-up dims): h 128, 2 layers of GQA without RoPE (4 heads, 2 KV, hd 32, q/k-norm)
 * with a relative-position bias (relDim 8; layer 0 sliding window 32 with extent 32, layer 1 full with extent 64)
 * and K/V convs of 4 taps; short convs of 4 taps after the attention and the ffn (dense 256). Vocab 1000.
 */
export const TINY_INKLING: ModelSpec = decoderModel({
  name: 'TINY_INKLING',
  hidden: 128,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture (Inkling-style)',
  layers: repeatLayers(2, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 128,
      mixer: {
        kind: 'gqa',
        heads: 4,
        kvHeads: 2,
        headDim: 32,
        qkNorm: true,
        rope: false,
        relativeBias: { relDim: 8, extentTokens: i === 0 ? 32 : 64 },
        kvConvKernel: 4,
        ...(i === 0 ? { slidingWindow: 32 } : {}),
      },
      ffn: { kind: 'dense', intermediate: 256 },
      shortConv: { kernel: 4 },
    }),
  ),
});

/**
 * Tiny Kimi-K3-style hybrid (made-up dims): h 128, 2 layers with Attention Residuals of block size 1 (layer 0 pushes
 * a summary, so its ffn mixes 2 candidates; layer 1 mixes 2 before its mixer and 3 before its ffn; the final mix 3).
 * Layer 0: KDA (2 q/k heads, 2 value heads, dk = dv = 32, conv 4, forget-gate rank 16, FULL-RANK output gate) and a
 * dense `situ` ffn of 256. Layer 1: MLA (4 heads, rq 32, rkv 32, dn 16, dr 8, dv 16) with a sigmoid output gate and
 * a latent MoE (E 8, top-2, fe 64, latent width L 64 with norm, one shared expert of 128). Vocab 1000.
 */
export const TINY_KIMI: ModelSpec = decoderModel({
  name: 'TINY_KIMI',
  hidden: 128,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture (Kimi K3-style)',
  layers: [
    decoderLayer({
      name: 'layer.0',
      hidden: 128,
      mixer: {
        kind: 'linear',
        variant: 'kda',
        qkHeads: 2,
        vHeads: 2,
        keyDim: 32,
        valueDim: 32,
        convKernel: 4,
        outputGate: true,
        gateRank: 16,
        fullRankOutputGate: true,
      },
      ffn: { kind: 'dense', intermediate: 256, activation: 'situ' },
      residual: { kind: 'attn_res', blockSize: 1, layerIndex: 0 },
    }),
    decoderLayer({
      name: 'layer.1',
      hidden: 128,
      mixer: {
        kind: 'mla',
        heads: 4,
        qLoraRank: 32,
        kvLoraRank: 32,
        qkNopeDim: 16,
        qkRopeDim: 8,
        vHeadDim: 16,
        outputGate: true,
      },
      ffn: {
        kind: 'moe',
        experts: 8,
        topK: 2,
        expertIntermediate: 64,
        sharedExperts: 1,
        sharedIntermediate: 128,
        activation: 'situ',
        latentDim: 64,
        latentNorm: true,
      },
      residual: { kind: 'attn_res', blockSize: 1, layerIndex: 1 },
    }),
  ],
  beforeHead: [attnResCollapse({ blockSize: 1, layers: 2, dim: 128 })],
});
