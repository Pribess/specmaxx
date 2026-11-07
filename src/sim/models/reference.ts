/**
 * Reference models with HAND-WRITTEN dims (no Hugging Face config dependency), used by tests and the demo.
 * `REF_DENSE_8B` copies Qwen3-8B's published shape so its size can be sanity-checked; the `TINY_*` fixtures have
 * made-up dims that exercise every block kind (hybrid linear attention, MoE with shared experts, dense first
 * layers, gated attention, MLA with a sparse indexer).
 */

import { decoderLayer, decoderModel, repeatLayers } from '../ir/builders';
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
const TINY_GATED_GQA: MixerParams = { kind: 'gqa', heads: 4, kvHeads: 1, headDim: 64, qkNorm: true, outputGate: true };

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
        sparse: { heads: 4, headDim: 32, topK: 64 },
      },
      ffn: { kind: 'dense', intermediate: 1024 },
    }),
  ),
});
