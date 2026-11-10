/**
 * Lowering of dense MLP blocks (also used for MoE shared experts): column-parallel gate/up projection,
 * activation, row-parallel down projection, TP all-reduce. See docs/simulation-model.md §5 "TP sharding"; with
 * the dense FFN sharded over the EP group, §5 "DP attention + EP".
 */

import type { ElementwiseOp, Op } from '../ir/ops';
import type { FfnActivation, MlpBlock } from '../ir/model';
import {
  actBpe,
  denseOverEp,
  denseTpGroupSize,
  elementwise,
  linearGemm,
  reduceBpe,
  reduceDtype,
  shardWidth,
  stepTokens,
  tpAllReduce,
} from './common';
import type { LoweringContext } from './lower';

/** The shape of a dense feed-forward network to lower. */
export interface DenseFfnShape {
  /** Input/output width h. Unit: elements. */
  hidden: number;
  /** Intermediate width f (logical, before sharding). Unit: elements. */
  intermediate: number;
  /** GLU-style (gate and up). */
  gated: boolean;
  /** Activation function (`relu2` = `relu(x)²`, ungated only). */
  activation: FfnActivation;
  /** Projection biases. */
  bias: boolean;
}

/**
 * The elementwise function of an activation, fused with the gate multiply when gated. `relu2` exists only ungated
 * (Nemotron-H experts); a gated `relu2` throws, since no model uses it and its FLOP count is not defined.
 */
export function activationFn(activation: FfnActivation, gated: boolean): ElementwiseOp['fn'] {
  switch (activation) {
    case 'silu':
      return gated ? 'silu_mul' : 'silu';
    case 'gelu':
      return gated ? 'gelu_mul' : 'gelu';
    case 'relu':
      return gated ? 'relu_mul' : 'relu';
    case 'relu2':
      if (gated) throw new Error('activation relu2 is supported only ungated (gated: false)');
      return 'relu2';
    case 'situ':
      return gated ? 'situ_mul' : 'situ';
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
    linearGemm(`${prefix}.gate_up`, { m: tokens, n: upWidth, k: shape.hidden }, ctx, {
      module,
      bias: shape.bias,
      outParts: shape.gated ? [fLocal, fLocal] : [fLocal],
    }),
    activationOp(`${prefix}.act`, shape, tokens, fLocal, ctx),
    linearGemm(`${prefix}.down`, { m: tokens, n: shape.hidden, k: fLocal }, ctx, {
      module,
      bias: shape.bias,
    }),
  ];
}

/**
 * A dense MLP block. Default: {@link lowerDenseFfn} sharded by tp, then the TP all-reduce of `T·h`.
 * With the dense FFN sharded over the EP group ({@link denseOverEp}, tp = 1): {@link lowerDenseFfnOverEp}.
 */
export function lowerMlp(block: MlpBlock, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  if (denseOverEp(ctx)) return lowerDenseFfnOverEp(block, T, ctx);
  return [
    ...lowerDenseFfn('mlp', block, T, ctx.parallel.tp, 'dense_mlp', ctx),
    ...tpAllReduce('mlp.tp_all_reduce', T, block.hidden, ctx),
  ];
}

/**
 * The error when the intermediate width f does not split evenly over the dense TP group G. Its wording is stable:
 * the UI attributes it to the dense-layers control by matching it.
 */
export function denseGroupWidthError(intermediate: number, groupSize: number): string {
  return `mlp intermediate (${intermediate}) must be divisible by the dense TP group (${groupSize})`;
}

/**
 * A dense MLP block tensor-parallel over the whole EP group of G = ep devices (`denseTpGroup: 'ep'`, tp = 1;
 * SGLang's dense TP under `--enable-dp-attention`). Each device holds `T` own tokens and an `f/G` slice:
 *
 * ```
 * mlp.dense_all_gather      all_gather(ep, G)       S = G·T·h·bpe(act)     (full output: every device's T rows)
 * mlp.gate_up / act / down  m = G·T, n = (gated?2:1)·f/G | k = f/G   (lowerDenseFfn, shards = G)
 * mlp.dense_reduce_scatter  reduce_scatter(ep, G)   S = G·T·h·reduceBpe   (full input; each keeps its T·h)
 * ```
 *
 * This is the minimal traffic: the gather and scatter together move the bytes of one all-reduce of `G·T·h`.
 * Throws {@link denseGroupWidthError} unless G divides f. See docs/simulation-model.md §5 "DP attention + EP".
 */
export function lowerDenseFfnOverEp(block: MlpBlock, T: number, ctx: LoweringContext): Op[] {
  const G = denseTpGroupSize(ctx);
  if (block.intermediate % G !== 0) throw new Error(denseGroupWidthError(block.intermediate, G));
  // Formula: T_G = G·T   (rows after the gather: the tokens of every device in the group)
  const groupTokens = G * T;
  // Formula: S_gather = G·T·h·bpe(act);  S_scatter = G·T·h·reduceBpe
  // See docs/simulation-model.md §5 "DP attention + EP"
  const gatherBytes = groupTokens * block.hidden * actBpe(ctx);
  const scatterBytes = groupTokens * block.hidden * reduceBpe(ctx);
  return [
    {
      kind: 'collective',
      name: 'mlp.dense_all_gather',
      collective: 'all_gather',
      group: 'ep',
      groupSize: G,
      messageBytes: gatherBytes,
      dtype: ctx.quant.activations,
    },
    ...lowerDenseFfn('mlp', block, groupTokens, G, 'dense_mlp', ctx),
    {
      kind: 'collective',
      name: 'mlp.dense_reduce_scatter',
      collective: 'reduce_scatter',
      group: 'ep',
      groupSize: G,
      messageBytes: scatterBytes,
      dtype: reduceDtype(ctx),
    },
  ];
}
