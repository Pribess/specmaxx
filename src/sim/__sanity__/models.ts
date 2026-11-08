/**
 * Hand-written model specs of the sibling models the community reports use (the HF loader does not map their
 * families: `gpt_oss`, `glm4_moe`, `qwen3_moe`, `llama`, `qwen2`, `qwen3_next`). Dims are copied from each repo's
 * config.json (fetched 2026-11-07; URL in `source`). Architecture features the IR has no block for are approximated
 * by the nearest one; each spec says which. Shapes only: no weights, no tokenizer.
 */

import { decoderLayer, decoderModel, repeatLayers } from '../ir/builders';
import type { ModelSpec } from '../ir/model';
import { simpleDecoder } from './harness';

/**
 * gpt-oss decoder (both sizes): h 2880, 64 heads, 8 KV heads, head dim 64, q/k/v and o biases, layers alternate a
 * 128-token sliding window (even layers) and full attention (odd layers), MoE with top-4 of E experts of width
 * 2880 (SwiGLU), vocab 201088, untied.
 * Approximations: the per-head attention sinks (one learned logit per head; ≈ 0 FLOP and bytes) and the expert
 * biases (2 × 2880-element vectors per expert, < 0.1 % of the expert bytes) are left out; the clamped SwiGLU is a
 * plain SwiGLU.
 */
function gptOss(name: string, layers: number, experts: number, repo: string): ModelSpec {
  return simpleDecoder({
    name,
    source: `https://huggingface.co/${repo}/blob/main/config.json`,
    hidden: 2880,
    vocab: 201088,
    tieEmbeddings: false,
    layers,
    mixer: (i) => ({
      kind: 'gqa',
      heads: 64,
      kvHeads: 8,
      headDim: 64,
      qkvBias: true,
      outBias: true,
      ...(i % 2 === 0 ? { slidingWindow: 128 } : {}),
    }),
    ffn: { kind: 'moe', experts, topK: 4, expertIntermediate: 2880 },
  });
}

/** openai/gpt-oss-120b: 36 layers, 128 experts top-4 (≈ 117 B params, ≈ 5.1 B active). */
export const GPT_OSS_120B = gptOss('gpt-oss-120b', 36, 128, 'openai/gpt-oss-120b');

/** openai/gpt-oss-20b: 24 layers, 32 experts top-4 (≈ 21 B params, ≈ 3.6 B active). */
export const GPT_OSS_20B = gptOss('gpt-oss-20b', 24, 32, 'openai/gpt-oss-20b');

/**
 * Qwen3-30B-A3B (also Qwen3-Coder-30B-A3B-Instruct and the -2507 instructs: same config): h 2048, 48 layers,
 * 32 heads, 4 KV heads, head dim 128, q/k-norm, MoE 128 experts top-8 of width 768, no shared expert, vocab 151936.
 */
export const QWEN3_30B_A3B = simpleDecoder({
  name: 'Qwen3-30B-A3B',
  source: 'https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json',
  hidden: 2048,
  vocab: 151936,
  tieEmbeddings: false,
  layers: 48,
  mixer: { kind: 'gqa', heads: 32, kvHeads: 4, headDim: 128, qkNorm: true },
  ffn: { kind: 'moe', experts: 128, topK: 8, expertIntermediate: 768 },
});

/**
 * Qwen3-235B-A22B (`qwen3_moe`): h 4096, 94 layers, 64 heads, 4 KV heads, head dim 128, q/k-norm, MoE 128 experts
 * top-8 of width 1536 in every layer (`mlp_only_layers` empty), no shared expert, vocab 151936, untied.
 */
export const QWEN3_235B_A22B = simpleDecoder({
  name: 'Qwen3-235B-A22B',
  source: 'https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json',
  hidden: 4096,
  vocab: 151936,
  tieEmbeddings: false,
  layers: 94,
  mixer: { kind: 'gqa', heads: 64, kvHeads: 4, headDim: 128, qkNorm: true },
  ffn: { kind: 'moe', experts: 128, topK: 8, expertIntermediate: 1536 },
});

/**
 * GLM-4.5-Air (`glm4_moe`): h 4096, 46 layers, 96 heads, 8 KV heads, head dim 128, q/k/v biases, layer 0 a dense
 * MLP of width 10944, layers 1–45 MoE 128 experts top-8 of width 1408 plus one shared expert of width 1408, vocab
 * 151552. Approximations: partial RoPE (factor 0.5) costed as full RoPE (elementwise, negligible); the MTP layer
 * (`num_nextn_predict_layers: 1`) is left out (llama.cpp does not run it).
 */
export const GLM_4_5_AIR = simpleDecoder({
  name: 'GLM-4.5-Air',
  source: 'https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json',
  hidden: 4096,
  vocab: 151552,
  tieEmbeddings: false,
  layers: 46,
  mixer: { kind: 'gqa', heads: 96, kvHeads: 8, headDim: 128, qkvBias: true },
  ffn: (i) =>
    i === 0
      ? { kind: 'dense', intermediate: 10944 }
      : { kind: 'moe', experts: 128, topK: 8, expertIntermediate: 1408, sharedExperts: 1, sharedIntermediate: 1408 },
});

/**
 * Llama-3.x-70B shape (DeepSeek-R1-Distill-Llama-70B, Llama-3.3-70B, Shisa V2 70B): h 8192, 80 layers, 64 heads,
 * 8 KV heads, head dim 128, SwiGLU 28672, vocab 128256, untied.
 */
export const LLAMA3_70B = simpleDecoder({
  name: 'Llama-3.x-70B',
  source: 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/config.json',
  hidden: 8192,
  vocab: 128256,
  tieEmbeddings: false,
  layers: 80,
  mixer: { kind: 'gqa', heads: 64, kvHeads: 8, headDim: 128 },
  ffn: { kind: 'dense', intermediate: 28672 },
});

/**
 * QwQ-32B (`qwen2`, Qwen2.5-32B shape): h 5120, 64 layers, 40 heads, 8 KV heads, head dim 128 (5120 / 40), q/k/v
 * biases, SwiGLU 27648, vocab 152064, untied (sliding window disabled: `use_sliding_window: false`).
 */
export const QWQ_32B = simpleDecoder({
  name: 'QwQ-32B',
  source: 'https://huggingface.co/Qwen/QwQ-32B/blob/main/config.json',
  hidden: 5120,
  vocab: 152064,
  tieEmbeddings: false,
  layers: 64,
  mixer: { kind: 'gqa', heads: 40, kvHeads: 8, headDim: 128, qkvBias: true },
  ffn: { kind: 'dense', intermediate: 27648 },
});

/**
 * Qwen3-Next-80B-A3B shape (Qwen3-Coder-Next, `qwen3_next`): h 2048, 48 layers; every 4th layer (3, 7, …) gated
 * full attention (16 heads, 2 KV heads, head dim 256, q/k-norm, output gate), the others GatedDeltaNet (16 q/k heads,
 * 32 value heads, dk = dv = 128, conv 4, z-gate); every layer MoE 512 experts top-10 of width 512 plus one shared
 * expert of width 512; vocab 151936. Approximations: the shared expert's sigmoid gate (one h-vector per layer) and
 * partial RoPE (0.25) are left out; the MTP head is not run.
 */
export const QWEN3_NEXT_80B_A3B: ModelSpec = decoderModel({
  name: 'Qwen3-Next-80B-A3B (Qwen3-Coder-Next shape)',
  hidden: 2048,
  vocab: 151936,
  tieEmbeddings: false,
  source: 'https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/config.json',
  layers: repeatLayers(48, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 2048,
      mixer:
        i % 4 === 3
          ? { kind: 'gqa', heads: 16, kvHeads: 2, headDim: 256, qkNorm: true, outputGate: true }
          : {
              kind: 'linear',
              variant: 'gated_delta_net',
              qkHeads: 16,
              vHeads: 32,
              keyDim: 128,
              valueDim: 128,
              convKernel: 4,
              outputGate: true,
            },
      ffn: { kind: 'moe', experts: 512, topK: 10, expertIntermediate: 512, sharedExperts: 1, sharedIntermediate: 512 },
    }),
  ),
});
