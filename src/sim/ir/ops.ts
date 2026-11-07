/**
 * The op catalogue: the unit of work the engine costs.
 *
 * An op is **the work of ONE device on its pipeline stage's critical path, for one microbatch step**.
 * Sharding is assumed balanced, so all devices of a stage are symmetric and one device's ops describe the stage.
 * Every count below is therefore per device (after tensor/expert parallel sharding) unless a field says otherwise.
 *
 * Shapes are concrete numbers (elements, tokens, sequences), never symbolic. FLOP and byte counts are derived
 * from these fields by `ir/metrics.ts` (docs/simulation-model.md §3), not stored on the op.
 */

import type { AccumulateType, Dtype, MathFormat, TensorFormat } from './dtype';

/** Any op one device executes. Dispatch with `switch (op.kind)`. */
export type Op = GemmOp | AttentionOp | IndexerOp | RecurrentOp | ElementwiseOp | EmbeddingOp | CollectiveOp;

/**
 * A (possibly batched/grouped) matrix multiply against weights:
 * `out[b] (m×n) = in[b] (m×k) · W[b] (k×n)` for `b` in `0..batch-1`.
 * Batch entries have DISTINCT weights (grouped GEMM: MoE experts, MLA per-head absorb matrices).
 * FLOP = 2·batch·m·n·k (dense-equivalent; 2:4 sparsity does not halve the count). See docs §3 "GEMM".
 */
export interface GemmOp {
  kind: 'gemm';
  /** Stable dotted name for reports, e.g. `'attn.qkv_proj'`, `'moe.gate_up'`. */
  name: string;
  /** Number of independent GEMMs, each with its own weight matrix (1 for a plain linear layer). Count. */
  batch: number;
  /** Rows of the input/output per batch entry: tokens on this device. Unit: elements (= tokens). */
  m: number;
  /** Output features per batch entry (weight columns on this device, after sharding). Unit: elements. */
  n: number;
  /** Input features per batch entry (the reduction dimension). Unit: elements. */
  k: number;
  /** Storage dtype of the activation input `in`, read from memory. */
  inputDtype: Dtype;
  /** Storage format of the weights `W` (with quantization overhead), read from memory once per step. */
  weight: TensorFormat;
  /** Storage dtype of the output `out`, written to memory. */
  outputDtype: Dtype;
  /** Matmul input class the tensor cores run in (selects the peak; from `resolveGemmMath`). */
  math: MathFormat;
  /** Accumulator type (selects the peak; from `resolveAccumulate`). */
  accumulate: AccumulateType;
  /**
   * Structured sparsity of the weight operand: `'dense'`, or `'2:4'` (2 of every 4 weights kept; uses the
   * device's 2:4 sparse peak and stores half the values plus 2-bit indices per kept value).
   */
  sparsity: 'dense' | '2:4';
  /** Dtype of the bias vector; present iff the GEMM adds a bias of `n` elements per batch entry. */
  biasDtype?: Dtype;
}

/** Which keys each query attends to. */
export type KvSelection =
  /** Every causally visible key (or every key, if not causal). */
  | { kind: 'full' }
  /** Sliding window: each query sees only the last `windowTokens` visible keys. Unit: tokens. */
  | { kind: 'window'; windowTokens: number }
  /** Top-k sparse attention: each query sees `topK` keys chosen by a preceding {@link IndexerOp}. Unit: tokens. */
  | { kind: 'topk'; topK: number };

/** A stream of K/V (or latent) elements read or written per token, per device per layer. */
export interface KvStream {
  /** Storage format of the stream's elements (e.g. the KV-cache format). */
  format: TensorFormat;
  /**
   * Elements per token per device per layer, K and V together (e.g. GQA: 2·kvHeadsLocal·headDim;
   * MLA latent: kvLoraRank + qkRopeDim). 0 = no traffic. Unit: elements/token.
   */
  elemsPerToken: number;
}

/**
 * Fused attention core, `softmax(Q·Kᵀ)·V`, FlashAttention-style: scores never reach memory, so only Q, K/V and
 * the output move. Covers MHA/GQA/MQA, sliding window, top-k sparse, and MLA (absorbed: kvHeads 1, qk ≠ v dims).
 * FLOP = 2·seqs·heads·pairs·(qkHeadDim + vHeadDim), with the causal triangle counted exactly. See docs §3 "Attention".
 */
export interface AttentionOp {
  kind: 'attention';
  /** Stable dotted name, e.g. `'attn.core'`. */
  name: string;
  /** Sequences in this microbatch on this device. Count. */
  seqs: number;
  /** Query heads on this device (after TP sharding). Count. */
  heads: number;
  /** K/V heads on this device (after TP sharding or replication; 1 for absorbed MLA). Count. */
  kvHeads: number;
  /** Q/K head dimension (absorbed MLA: kvLoraRank + qkRopeDim, e.g. 576). Unit: elements. */
  qkHeadDim: number;
  /** V/output head dimension (absorbed MLA: kvLoraRank, e.g. 512). Unit: elements. */
  vHeadDim: number;
  /** Query tokens per sequence in this step (prefill: prompt chunk; decode: 1). Unit: tokens. */
  qLen: number;
  /** Keys visible to the LAST query of the step (= past tokens + qLen). Unit: tokens. */
  kvLen: number;
  /** True if query i only sees keys at positions ≤ its own (decoder self-attention). */
  causal: boolean;
  /** Which of the visible keys each query attends to (full, sliding window, or top-k). */
  kvSelection: KvSelection;
  /**
   * K/V read per visible-and-selected token: the KV cache, or decompressed K/V produced by a preceding GEMM
   * (MLA prefill, MHA form).
   */
  kvRead: KvStream;
  /** Appended to the KV cache per NEW token (qLen tokens per sequence); `elemsPerToken` 0 = no cache write. */
  kvWrite: KvStream;
  /** Matmul input class of Q·Kᵀ and P·V (selects the peak). */
  math: MathFormat;
  /** Accumulator type of Q·Kᵀ and P·V (selects the peak). */
  accumulate: AccumulateType;
  /** Storage dtype of Q (read) and the output O (written). */
  activationDtype: Dtype;
}

/**
 * Top-k key scorer for sparse attention (DeepSeek-V3.2 DSA "lightning indexer"):
 * `score(q, s) = Σ_h w_h · ReLU(q_h · k_s)`; keeps the `topK` best keys per query. It reads its own key cache
 * (one key vector per token, shared by all indexer heads), has no V, and writes int32 indices.
 * It scores EVERY visible key, so its FLOP use full causal pairs. See docs §3 "Indexer".
 */
export interface IndexerOp {
  kind: 'indexer';
  /** Stable dotted name, e.g. `'attn.indexer'`. */
  name: string;
  /** Sequences in this microbatch on this device. Count. */
  seqs: number;
  /** Indexer heads; replicated on every TP rank (not sharded). Count. */
  heads: number;
  /** Indexer head dimension. Unit: elements. */
  headDim: number;
  /** Query tokens per sequence in this step. Unit: tokens. */
  qLen: number;
  /** Keys visible to the last query (= past + qLen). Unit: tokens. */
  kvLen: number;
  /** True if query i only scores keys at positions ≤ its own. */
  causal: boolean;
  /** Keys kept per query (and int32 indices written per query). Unit: tokens. */
  topK: number;
  /** Storage format of the indexer key cache. */
  keyCache: TensorFormat;
  /** Indexer key-cache elements per token per device per layer (usually `headDim`). Unit: elements/token. */
  keyCacheElemsPerToken: number;
  /** Matmul input class of the q·k scoring (selects the peak). */
  math: MathFormat;
  /** Accumulator type of the scoring (selects the peak). */
  accumulate: AccumulateType;
  /** Storage dtype of the indexer queries and head weights (read). */
  activationDtype: Dtype;
}

/**
 * Linear-attention / SSM token mixer with a FIXED-size recurrent state per sequence (GatedDeltaNet, Mamba2 SSD,
 * plain linear attention). State per sequence per device = `heads · keyDim · valueDim` elements, independent
 * of context length. See docs §3 "Recurrent".
 */
export interface RecurrentOp {
  kind: 'recurrent';
  /** Stable dotted name, e.g. `'lin.recurrent'`. */
  name: string;
  /** Which recurrence (sets the step-mode FLOP constant, see `stateFlopsPerElemPerToken`). */
  variant: 'gated_delta_net' | 'mamba2' | 'linear';
  /** Sequences in this microbatch on this device. Count. */
  seqs: number;
  /** New tokens per sequence in this step. Unit: tokens. */
  qLen: number;
  /** State (value) heads on this device. Count. */
  heads: number;
  /** q/k heads on this device (Mamba2: B/C groups). Count. */
  qkHeads: number;
  /** State key dimension dk (Mamba2: d_state). Unit: elements. */
  keyDim: number;
  /** State value dimension dv (Mamba2: head dim). Unit: elements. */
  valueDim: number;
  /** `'chunked'`: prefill, chunk-parallel form (tensor-core friendly); `'step'`: decode, one recurrent update. */
  mode: 'chunked' | 'step';
  /** Tokens per chunk in chunked mode (default 64); 1 in step mode. Unit: tokens. */
  chunkLen: number;
  /**
   * FLOP per state element per token for the state terms (c_s in docs §3: chunked 4; step: gated_delta_net 7,
   * mamba2 5, linear 4). A documented approximation set by lowering. Unit: FLOP/(element·token).
   */
  stateFlopsPerElemPerToken: number;
  /** False on the first prefill step of a sequence (no prior state to read); true otherwise. */
  hasPriorState: boolean;
  /** Storage dtype of the recurrent state (read and written once per step). */
  stateDtype: Dtype;
  /** Storage dtype of q, k, v, gates (read) and the output (written). */
  activationDtype: Dtype;
  /** Matmul input class of the chunked form (selects the peak). */
  math: MathFormat;
  /** Accumulator type of the chunked form (selects the peak). */
  accumulate: AccumulateType;
}

/**
 * Memory-bound elementwise or small-reduction op (norms, activations, RoPE, residual adds, conv1d, routing).
 * Runs on the vector units: FLOP = elems · flopsPerElem; bytes are given explicitly by the byte fields
 * (lowering computes them, since they differ per function). See docs §3 "Elementwise".
 */
export interface ElementwiseOp {
  kind: 'elementwise';
  /** Stable dotted name, e.g. `'attn.qk_norm'`. */
  name: string;
  /**
   * Function computed; its FLOP-per-element approximation is in the docs §3 table (`ELEMENTWISE_FLOPS_PER_ELEM` in
   * lower/common.ts). `silu` and `relu_mul` cover the ungated-SiLU and gated-ReLU MLPs.
   */
  fn:
    | 'rmsnorm'
    | 'layernorm'
    | 'residual_add'
    | 'rope'
    | 'qk_norm'
    | 'silu'
    | 'silu_mul'
    | 'gelu_mul'
    | 'gelu'
    | 'relu'
    | 'relu_mul'
    | 'sigmoid_mul'
    | 'gated_rmsnorm'
    | 'causal_conv1d'
    | 'router_topk'
    | 'moe_combine';
  /** Output elements computed on this device. Unit: elements. */
  elems: number;
  /** Vector FLOP per output element (e.g. rmsnorm 4; causal_conv1d 2·kernelWidth). Unit: FLOP/element. */
  flopsPerElem: number;
  /** Storage dtype of the activations (selects the vector peak). */
  dtype: Dtype;
  /** Activation bytes read from memory on this device. Unit: bytes. */
  activationReadBytes: number;
  /** Activation bytes written to memory on this device. Unit: bytes. */
  activationWriteBytes: number;
  /** Weight bytes read on this device (norm gamma, conv taps); 0 if none. Unit: bytes. */
  weightBytes: number;
  /** Per-sequence state bytes read (causal_conv1d rolling state); 0 otherwise. Unit: bytes. */
  stateReadBytes: number;
  /** Per-sequence state bytes written (causal_conv1d rolling state); 0 otherwise. Unit: bytes. */
  stateWriteBytes: number;
}

/**
 * Token-embedding gather: one row of the embedding table per token. The table is vocab-parallel over TP
 * (each rank holds `vocabShard` rows; a TP all-reduce combines the partial results). FLOP = 0.
 */
export interface EmbeddingOp {
  kind: 'embedding';
  /** Stable dotted name, e.g. `'embed'`. */
  name: string;
  /** Tokens embedded on this device in this step. Unit: tokens. */
  tokens: number;
  /** Hidden size (row length). Unit: elements. */
  hidden: number;
  /** Vocabulary rows held by this device (`ceil(vocab / tp)`). Count. */
  vocabShard: number;
  /** Storage format of the embedding table. */
  weight: TensorFormat;
  /** Storage dtype of the embedded activations written. */
  outputDtype: Dtype;
}

/**
 * A collective among the devices of one parallel group. Its time comes from the alpha-beta model
 * (engine/collectives.ts, docs §6); it moves no HBM bytes in this model.
 *
 * `messageBytes` per device, by collective:
 * - `all_reduce`: the full buffer S (input size = output size);
 * - `all_gather`: the full OUTPUT (after gathering);
 * - `reduce_scatter`: the full INPUT (before scattering);
 * - `all_to_all`: the total this device sends, including the share it keeps for itself;
 * - `send_recv`: the bytes sent to the peer.
 */
export interface CollectiveOp {
  kind: 'collective';
  /** Stable dotted name, e.g. `'attn.tp_all_reduce'`. */
  name: string;
  /** Collective pattern. */
  collective: 'all_reduce' | 'all_gather' | 'reduce_scatter' | 'all_to_all' | 'send_recv';
  /** Parallel group it runs over: tensor (`tp`), expert (`ep`) or pipeline (`pp`, send/recv between stages). */
  group: 'tp' | 'ep' | 'pp';
  /** Devices in the group (send_recv: 2). Count. */
  groupSize: number;
  /** Per-device message size, by the convention above. Unit: bytes. */
  messageBytes: number;
  /** Element dtype on the wire (informational; the size is already in `messageBytes`). */
  dtype: Dtype;
}
