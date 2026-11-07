/**
 * Model architecture spec: WHAT the model computes, independent of tokens, batch and parallelism.
 *
 * A model is an ordered list of heterogeneous layers, each an ordered list of blocks. That covers hybrid stacks
 * (e.g. 3 GatedDeltaNet layers per full-attention layer), dense first layers before MoE layers, and alternating
 * sliding-window layers. Lowering (`src/sim/lower/`) turns blocks into per-device ops for a given workload and
 * parallelism. All dimensions are LOGICAL (unsharded) sizes. See docs/simulation-model.md §2.
 */

import type { Dtype, TensorFormat } from './dtype';

/** A decoder-only language model. */
export interface ModelSpec {
  /** Display name, e.g. `'Qwen3-8B'`. */
  name: string;
  /** Residual-stream width d_model. Unit: elements. */
  hidden: number;
  /** Vocabulary size (embedding and LM-head rows). Count. */
  vocab: number;
  /** True if the LM head reuses the embedding matrix (stored once). */
  tieEmbeddings: boolean;
  /** Decoder layers in execution order (heterogeneous). */
  layers: LayerSpec[];
  /** Norm applied before the LM head. */
  finalNorm: 'rmsnorm' | 'layernorm';
  /** Where the dims came from (e.g. a Hugging Face model ID or paper URL). */
  source?: string;
}

/** One decoder layer: blocks in execution order (norms and residual adds are explicit blocks). */
export interface LayerSpec {
  /** Display name, e.g. `'layer.0'`. */
  name: string;
  /** Blocks in execution order. */
  blocks: BlockSpec[];
}

/** Any block of a layer. Dispatch with `switch (block.kind)`. */
export type BlockSpec =
  | NormBlock
  | ResidualBlock
  | GqaAttentionBlock
  | MlaAttentionBlock
  | LinearAttentionBlock
  | MlpBlock
  | MoeBlock;

/** A normalization over the last dimension (per token). */
export interface NormBlock {
  kind: 'norm';
  /** Display name, e.g. `'input_norm'`. */
  name: string;
  /** Norm function (rmsnorm: one gamma vector; layernorm: gamma and beta). */
  norm: 'rmsnorm' | 'layernorm';
  /** Normalized width. Unit: elements. */
  dim: number;
}

/** Residual add: `x = x + f(x)` over the residual stream. */
export interface ResidualBlock {
  kind: 'residual';
  /** Display name, e.g. `'attn_residual'`. */
  name: string;
  /** Residual width. Unit: elements. */
  dim: number;
}

/** Sparse-attention indexer parameters (DeepSeek-V3.2 DSA "lightning indexer"). */
export interface IndexerSpec {
  /** Indexer heads. Count. */
  heads: number;
  /** Indexer head dimension (also the indexer key-cache elements per token). Unit: elements. */
  headDim: number;
  /** Keys kept per query. Unit: tokens. */
  topK: number;
  /** Storage dtype of the indexer key cache; default: the KV-cache dtype. */
  keyDtype?: Dtype;
}

/**
 * Multi-head / grouped-query / multi-query attention (MHA when kvHeads = heads, MQA when kvHeads = 1), with
 * optional q/k-norm, output gate, sliding window, or top-k sparse selection.
 */
export interface GqaAttentionBlock {
  kind: 'gqa_attention';
  /** Display name, e.g. `'attn'`. */
  name: string;
  /** Input/output width (the residual stream). Unit: elements. */
  hidden: number;
  /** Query heads. Count. */
  heads: number;
  /** Key/value heads. Count. */
  kvHeads: number;
  /** Head dimension (Q, K and V). Unit: elements. */
  headDim: number;
  /** True if the q/k/v projections have biases. */
  qkvBias: boolean;
  /** True if the output projection has a bias. */
  outBias: boolean;
  /** True if q and k are RMS-normalized per head before RoPE (Qwen3, Gemma 3). */
  qkNorm: boolean;
  /** True if rotary position embeddings are applied to q and k. */
  rope: boolean;
  /** True for causal (decoder) attention. */
  causal: boolean;
  /** Gated attention output (Qwen3-Next): `sigmoid(gate) ⊙ attn_out`, the gate from a wider q projection. */
  outputGate: boolean;
  /** Sliding-window size; absent = full attention. Mutually exclusive with `sparse`. Unit: tokens. */
  slidingWindow?: number;
  /** Top-k sparse attention via an indexer; absent = dense. Mutually exclusive with `slidingWindow`. */
  sparse?: IndexerSpec;
}

/**
 * Multi-head latent attention (DeepSeek-V2/V3): K/V are compressed to a low-rank latent; the KV cache holds
 * `kvLoraRank + qkRopeDim` elements per token per layer (shared by all heads).
 */
export interface MlaAttentionBlock {
  kind: 'mla_attention';
  /** Display name, e.g. `'attn'`. */
  name: string;
  /** Input/output width (the residual stream). Unit: elements. */
  hidden: number;
  /** Attention heads. Count. */
  heads: number;
  /** Query compression rank rq; 0 = no q compression (q projected straight from hidden). Unit: elements. */
  qLoraRank: number;
  /** KV latent rank rkv (cached per token). Unit: elements. */
  kvLoraRank: number;
  /** Per-head q/k dimension without RoPE (dn). Unit: elements. */
  qkNopeDim: number;
  /** Per-head q/k RoPE dimension (dr); the k part is shared by all heads and cached. Unit: elements. */
  qkRopeDim: number;
  /** Per-head value dimension (dv). Unit: elements. */
  vHeadDim: number;
  /** DeepSeek-V3.2 DSA top-k sparse attention; absent = dense. */
  sparse?: IndexerSpec;
}

/**
 * Linear-attention / SSM token mixer (GatedDeltaNet, Mamba2, plain linear attention) with a fixed-size state.
 * State per sequence = `vHeads · keyDim · valueDim` elements, plus the conv1d rolling state.
 */
export interface LinearAttentionBlock {
  kind: 'linear_attention';
  /** Display name, e.g. `'linear_attn'`. */
  name: string;
  /** Recurrence variant. */
  variant: 'gated_delta_net' | 'mamba2' | 'linear';
  /** Input/output width (the residual stream). Unit: elements. */
  hidden: number;
  /** q/k heads (Mamba2: B/C groups). Count. */
  qkHeads: number;
  /** Value (state) heads. Count. */
  vHeads: number;
  /** State key dimension dk (Mamba2: d_state). Unit: elements. */
  keyDim: number;
  /** State value dimension dv (Mamba2: head dim). Unit: elements. */
  valueDim: number;
  /** Causal depthwise conv width over the q, k, v channels; 0 = no conv. Unit: tokens (taps). */
  convKernel: number;
  /** True if the output passes a z-gate (gated rmsnorm) before the output projection. */
  outputGate: boolean;
}

/** Dense feed-forward block: `down(act(gate(x)) ⊙ up(x))` if gated, else `down(act(up(x)))`. */
export interface MlpBlock {
  kind: 'mlp';
  /** Display name, e.g. `'mlp'`. */
  name: string;
  /** Input/output width. Unit: elements. */
  hidden: number;
  /** Intermediate width f. Unit: elements. */
  intermediate: number;
  /** True for GLU-style (gate and up projections, e.g. SwiGLU). */
  gated: boolean;
  /** Activation function. */
  activation: 'silu' | 'gelu' | 'relu';
  /** True if the projections have biases. */
  bias: boolean;
}

/** Mixture-of-experts feed-forward block with a top-k router and optional always-on shared experts. */
export interface MoeBlock {
  kind: 'moe';
  /** Display name, e.g. `'moe'`. */
  name: string;
  /** Input/output width. Unit: elements. */
  hidden: number;
  /** Routed experts E. Count. */
  experts: number;
  /** Routed experts per token k. Count. */
  topK: number;
  /** Intermediate width of one routed expert. Unit: elements. */
  expertIntermediate: number;
  /** True for GLU-style experts. */
  gated: boolean;
  /** Expert activation function. */
  activation: 'silu' | 'gelu' | 'relu';
  /** Shared experts applied to every token (0 = none). Count. */
  sharedExperts: number;
  /** Intermediate width of EACH shared expert. Unit: elements. */
  sharedIntermediate: number;
}

/** Numeric formats of a deployment (weights, activations, caches). */
export interface QuantConfig {
  /** Linear-layer weights (attention/mixer projections, MLP, experts). */
  weights: TensorFormat;
  /** GEMM inputs and intermediate activations. */
  activations: Dtype;
  /** KV, MLA latent and indexer key caches (unless `IndexerSpec.keyDtype` is set). */
  kvCache: TensorFormat;
  /** Recurrent (linear-attention/SSM) state; default `'fp32'`. */
  recurrentState?: Dtype;
  /** Embedding table; default `{ dtype: highPrecision }`. */
  embeddings?: TensorFormat;
  /** LM head; default: `embeddings`. */
  lmHead?: TensorFormat;
  /** Norm weights, router, biases, conv taps; default `'bf16'`. */
  highPrecision?: Dtype;
  /** Attention-core math dtype (Q·Kᵀ, P·V); default: `activations`. */
  attentionMath?: Dtype;
  /** Requested float matmul accumulator; default `'fp32'` (bf16/tf32/fp32 inputs always accumulate in fp32). */
  accumulate?: 'fp32' | 'fp16';
  /** Structured sparsity of linear-layer weights; default `'dense'`. */
  weightSparsity?: 'dense' | '2:4';
}

/**
 * Parallelism degrees. Each is ≥ 1. World size = tp · pp · dp devices; the expert-parallel group (ep) is laid
 * over the tp × dp ranks of a stage (docs §5, MoE modes A/B).
 */
export interface ParallelConfig {
  /** Tensor-parallel degree (devices sharing each layer's weights). Count. */
  tp: number;
  /** Pipeline-parallel degree (stages, contiguous layer ranges). Count. */
  pp: number;
  /** Data-parallel degree (independent replicas, each serving batch / dp sequences). Count. */
  dp: number;
  /** Expert-parallel degree (devices sharing the routed experts of a MoE layer). Count. */
  ep: number;
}
