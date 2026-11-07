/**
 * Shared lowering helpers: TP sharding rules, op constructors with the quantization rules applied, and the
 * elementwise FLOP-per-element table. See docs/simulation-model.md §5 (Parallelism) and §3 (Elementwise).
 */

import {
  bytesPerElement,
  mathFormatOf,
  resolveAccumulate,
  resolveGemmMath,
  type AccumulateType,
  type Dtype,
  type MathFormat,
  type TensorFormat,
} from '../ir/dtype';
import type { CollectiveOp, ElementwiseOp, GemmOp } from '../ir/ops';
import type { LoweringContext } from './lower';

/**
 * Vector FLOP per output element of each elementwise function (documented approximations; they only feed the
 * vector-compute time). `causal_conv1d` is `2·kernelWidth` and is set by its lowering. `router_topk` is per logit.
 * Unit: FLOP/element. See docs/simulation-model.md §3 "Elementwise".
 */
export const ELEMENTWISE_FLOPS_PER_ELEM: Record<Exclude<ElementwiseOp['fn'], 'causal_conv1d'>, number> = {
  rmsnorm: 4,
  layernorm: 8,
  residual_add: 1,
  rope: 3,
  qk_norm: 4,
  silu: 4,
  silu_mul: 5,
  gelu_mul: 9,
  gelu: 8,
  relu: 1,
  relu_mul: 2,
  sigmoid_mul: 5,
  gated_rmsnorm: 9,
  router_topk: 5,
  moe_combine: 2,
};

/** Tokens processed per device step: `T = seqs·qLen`. Unit: tokens. */
export function stepTokens(ctx: LoweringContext): number {
  // Formula: T = seqs·qLen
  return ctx.seqs * ctx.qLen;
}

/** Bytes per element of the activation dtype. Unit: bytes/element. */
export function actBpe(ctx: LoweringContext): number {
  return bytesPerElement({ dtype: ctx.quant.activations });
}

/** High-precision dtype for norms, router, biases and conv taps (default 'bf16'). */
export function highPrecision(ctx: LoweringContext): Dtype {
  return ctx.quant.highPrecision ?? 'bf16';
}

/** Bytes per element of the high-precision dtype. Unit: bytes/element. */
export function hpBpe(ctx: LoweringContext): number {
  return bytesPerElement({ dtype: highPrecision(ctx) });
}

/**
 * Bytes per element carried by reductions (all-reduce, reduce-scatter, MoE combine): at least 16-bit.
 * Unit: bytes/element. See docs/simulation-model.md §5 "Reductions dtype".
 */
export function reduceBpe(ctx: LoweringContext): number {
  // Formula: reduceBpe = max(bpe(act), 2)
  return Math.max(actBpe(ctx), 2);
}

/** Wire dtype of reductions: the activation dtype if it is at least 16-bit, else bf16 (see {@link reduceBpe}). */
export function reduceDtype(ctx: LoweringContext): Dtype {
  return actBpe(ctx) >= 2 ? ctx.quant.activations : 'bf16';
}

/**
 * Query (or other evenly sharded) heads per TP rank: `heads / tp`; throws unless `tp` divides `heads`. Count.
 * See docs/simulation-model.md §5 "TP sharding".
 */
export function shardHeads(heads: number, tp: number, what: string): number {
  if (heads % tp !== 0) throw new Error(`${what} (${heads}) must be divisible by tp (${tp})`);
  // Formula: Hl = heads / tp
  return heads / tp;
}

/**
 * KV (or linear-attention q/k, v) heads per TP rank with replication: `kvHeads / tp` when kvHeads ≥ tp,
 * else 1 (each rank holds one replicated head). Requires `kvHeads % tp == 0` or `tp % kvHeads == 0`. Count.
 * See docs/simulation-model.md §5 "KV-head replication".
 */
export function shardKvHeads(kvHeads: number, tp: number, what: string): number {
  if (kvHeads % tp !== 0 && tp % kvHeads !== 0) {
    throw new Error(`${what} (${kvHeads}) and tp (${tp}) must divide one another`);
  }
  // Formula: KVl = kvHeads ≥ tp ? kvHeads / tp : 1
  return kvHeads >= tp ? kvHeads / tp : 1;
}

/** Evenly shard a width over `parts` ranks; throws unless divisible. Unit: elements. */
export function shardWidth(width: number, parts: number, what: string): number {
  if (width % parts !== 0) throw new Error(`${what} (${width}) must be divisible by ${parts}`);
  return width / parts;
}

/** Math format and accumulator of the attention-like cores (attentionMath ?? activations). */
export function attentionMath(ctx: LoweringContext): { math: MathFormat; accumulate: AccumulateType } {
  const math = mathFormatOf(ctx.quant.attentionMath ?? ctx.quant.activations);
  return { math, accumulate: resolveAccumulate(math, ctx.quant.accumulate ?? 'fp32') };
}

/** Options of {@link linearGemm}. */
export interface GemmOptions {
  /** Grouped-GEMM batch (distinct weights per entry); default 1. Count. */
  batch?: number;
  /** Weight format; default `quant.weights` (a linear layer). */
  weight?: TensorFormat;
  /** Weight sparsity; default `quant.weightSparsity ?? 'dense'` for linear layers, 'dense' if `weight` is given. */
  sparsity?: 'dense' | '2:4';
  /** Input dtype; default `quant.activations`. */
  inputDtype?: Dtype;
  /** Adds an `n`-element bias per batch entry in the high-precision dtype. */
  bias?: boolean;
}

/**
 * A GEMM `m×k · k×n` with the quantization rules applied: `math = resolveGemmMath(w, act)`,
 * `accumulate = resolveAccumulate(math, quant.accumulate ?? 'fp32')`. Shapes in elements (m = tokens).
 */
export function linearGemm(
  name: string,
  shape: { m: number; n: number; k: number },
  ctx: LoweringContext,
  opts: GemmOptions = {},
): GemmOp {
  const act = ctx.quant.activations;
  const weight = opts.weight ?? ctx.quant.weights;
  const sparsity = opts.sparsity ?? (opts.weight === undefined ? (ctx.quant.weightSparsity ?? 'dense') : 'dense');
  const math = resolveGemmMath(weight.dtype, act);
  const op: GemmOp = {
    kind: 'gemm',
    name,
    batch: opts.batch ?? 1,
    m: shape.m,
    n: shape.n,
    k: shape.k,
    inputDtype: opts.inputDtype ?? act,
    weight,
    outputDtype: act,
    math,
    accumulate: resolveAccumulate(math, ctx.quant.accumulate ?? 'fp32'),
    sparsity,
  };
  if (opts.bias === true) op.biasDtype = highPrecision(ctx);
  return op;
}

/** Byte and FLOP fields of an elementwise op (everything but kind/name/fn/dtype). */
export interface ElementwiseFields {
  /** Output elements. Unit: elements. */
  elems: number;
  /** FLOP per output element; default from {@link ELEMENTWISE_FLOPS_PER_ELEM}. Unit: FLOP/element. */
  flopsPerElem?: number;
  /** Activation bytes read. Unit: bytes. */
  activationReadBytes: number;
  /** Activation bytes written. Unit: bytes. */
  activationWriteBytes: number;
  /** Weight bytes read (gamma, taps); default 0. Unit: bytes. */
  weightBytes?: number;
  /** State bytes read; default 0. Unit: bytes. */
  stateReadBytes?: number;
  /** State bytes written; default 0. Unit: bytes. */
  stateWriteBytes?: number;
}

/** An elementwise op over activations of dtype `quant.activations`. */
export function elementwise(
  name: string,
  fn: ElementwiseOp['fn'],
  f: ElementwiseFields,
  ctx: LoweringContext,
): ElementwiseOp {
  let flopsPerElem = f.flopsPerElem;
  if (flopsPerElem === undefined) {
    if (fn === 'causal_conv1d') throw new Error('causal_conv1d needs an explicit flopsPerElem (2·kernelWidth)');
    flopsPerElem = ELEMENTWISE_FLOPS_PER_ELEM[fn];
  }
  return {
    kind: 'elementwise',
    name,
    fn,
    elems: f.elems,
    flopsPerElem,
    dtype: ctx.quant.activations,
    activationReadBytes: f.activationReadBytes,
    activationWriteBytes: f.activationWriteBytes,
    weightBytes: f.weightBytes ?? 0,
    stateReadBytes: f.stateReadBytes ?? 0,
    stateWriteBytes: f.stateWriteBytes ?? 0,
  };
}

/**
 * A per-token norm over `tokens × dim` activations: read and write `tokens·dim·bpe(act)`, gamma (and beta for
 * layernorm) `dim·bpe(hp)` each. Replicated on every TP rank (no sequence parallelism).
 */
export function normOp(
  name: string,
  norm: 'rmsnorm' | 'layernorm',
  tokens: number,
  dim: number,
  ctx: LoweringContext,
): ElementwiseOp {
  const elems = tokens * dim;
  const vectors = norm === 'layernorm' ? 2 : 1;
  // Formula: read = write = T·dim·bpe(act); weights = (layernorm ? 2 : 1)·dim·bpe(hp)
  // See docs/simulation-model.md §5 "Norms and residuals"
  return elementwise(
    name,
    norm,
    {
      elems,
      activationReadBytes: elems * actBpe(ctx),
      activationWriteBytes: elems * actBpe(ctx),
      weightBytes: vectors * dim * hpBpe(ctx),
    },
    ctx,
  );
}

/**
 * The TP all-reduce of a `tokens × hidden` partial sum, carried at `reduceBpe`; none when tp = 1.
 * messageBytes = the full buffer S per device.
 */
export function tpAllReduce(name: string, tokens: number, hidden: number, ctx: LoweringContext): CollectiveOp[] {
  const tp = ctx.parallel.tp;
  if (tp === 1) return [];
  // Formula: S = T·h·reduceBpe
  // See docs/simulation-model.md §5 "TP sharding"
  return [
    {
      kind: 'collective',
      name,
      collective: 'all_reduce',
      group: 'tp',
      groupSize: tp,
      messageBytes: tokens * hidden * reduceBpe(ctx),
      dtype: reduceDtype(ctx),
    },
  ];
}
