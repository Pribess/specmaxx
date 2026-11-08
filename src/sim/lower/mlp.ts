/**
 * Lowering of dense MLP blocks (also used for MoE shared experts): column-parallel gate/up projection,
 * activation, row-parallel down projection, TP all-reduce. See docs/simulation-model.md §5 "TP sharding".
 */

import type { ElementwiseOp, Op } from '../ir/ops';
import type { MlpBlock } from '../ir/model';
import { actBpe, elementwise, linearGemm, shardWidth, stepTokens, tpAllReduce } from './common';
import type { LoweringContext } from './lower';

/** The shape of a dense feed-forward network to lower. */
export interface DenseFfnShape {
  /** Input/output width h. Unit: elements. */
  hidden: number;
  /** Intermediate width f (logical, before sharding). Unit: elements. */
  intermediate: number;
  /** GLU-style (gate and up). */
  gated: boolean;
  /** Activation function. */
  activation: 'silu' | 'gelu' | 'relu';
  /** Projection biases. */
  bias: boolean;
}

/** The elementwise function of an activation, fused with the gate multiply when gated. */
export function activationFn(activation: 'silu' | 'gelu' | 'relu', gated: boolean): ElementwiseOp['fn'] {
  switch (activation) {
    case 'silu':
      return gated ? 'silu_mul' : 'silu';
    case 'gelu':
      return gated ? 'gelu_mul' : 'gelu';
    case 'relu':
      return gated ? 'relu_mul' : 'relu';
    default: {
      const _never: never = activation;
      throw new Error(`unknown activation ${String(_never)}`);
    }
  }
}

/**
 * The activation op over `rows × width` elements: reads `(gated ? 2 : 1)·rows·width·bpe(act)` (gate and up),
 * writes `rows·width·bpe(act)`.
 */
export function activationOp(
  name: string,
  shape: Pick<DenseFfnShape, 'gated' | 'activation'>,
  rows: number,
  width: number,
  ctx: LoweringContext,
): ElementwiseOp {
  const elems = rows * width;
  // Formula: read = (gated ? 2 : 1)·elems·bpe(act); write = elems·bpe(act)
  // See docs/simulation-model.md §3 "Elementwise"
  return elementwise(
    name,
    activationFn(shape.activation, shape.gated),
    {
      elems,
      activationReadBytes: (shape.gated ? 2 : 1) * elems * actBpe(ctx),
      activationWriteBytes: elems * actBpe(ctx),
    },
    ctx,
  );
}

/**
 * gate_up → activation → down for `tokens` rows, the intermediate width sharded over `shards` ranks; no
 * reduction (callers add it). `prefix` names the ops (`'mlp'` → `'mlp.gate_up'`, `'mlp.act'`, `'mlp.down'`);
 * `module` picks the weight format (`'dense_mlp'` or `'shared_experts'`).
 */
export function lowerDenseFfn(
  prefix: string,
  shape: DenseFfnShape,
  tokens: number,
  shards: number,
  module: 'dense_mlp' | 'shared_experts',
  ctx: LoweringContext,
): Op[] {
  const fLocal = shardWidth(shape.intermediate, shards, `${prefix} intermediate`);
  const upWidth = (shape.gated ? 2 : 1) * fLocal;
  // Formula: gate_up: m = T, n = (gated ? 2 : 1)·f/tp, k = h; down: m = T, n = h, k = f/tp
  // See docs/simulation-model.md §5 "TP sharding"
  return [
    linearGemm(`${prefix}.gate_up`, { m: tokens, n: upWidth, k: shape.hidden }, ctx, { module, bias: shape.bias, outParts: shape.gated ? [fLocal, fLocal] : [fLocal] }),
    activationOp(`${prefix}.act`, shape, tokens, fLocal, ctx),
    linearGemm(`${prefix}.down`, { m: tokens, n: shape.hidden, k: fLocal }, ctx, { module, bias: shape.bias }),
  ];
}

/** A dense MLP block: {@link lowerDenseFfn} sharded by tp, then the TP all-reduce of `T·h`. */
export function lowerMlp(block: MlpBlock, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  return [
    ...lowerDenseFfn('mlp', block, T, ctx.parallel.tp, 'dense_mlp', ctx),
    ...tpAllReduce('mlp.tp_all_reduce', T, block.hidden, ctx),
  ];
}
