/**
 * Lowering: model spec + workload step + quantization + parallelism → the ops ONE device runs per stage.
 *
 * Conventions (docs/simulation-model.md §5): `T = seqs · qLen` tokens per device step; TP shards heads and
 * intermediate widths (`Hl = heads / tp`; KV heads are replicated when `kvHeads < tp`); PP splits layers into
 * contiguous stages ({@link stageLayerRange}); DP replicas are independent and not lowered separately.
 */

import { opBytes } from '../ir/metrics';
import type { CollectiveOp, Op } from '../ir/ops';
import type { BlockSpec, ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { actBpe, stepTokens } from './common';
import { csaSequenceStateBytes, lowerCsa } from './csa';
import { lowerEmbedding } from './embedding';
import { gqaSequenceStateBytes, lowerGqa } from './gqa';
import { lowerHead } from './head';
import { lowerHcPost, lowerHcPre } from './hyperConnection';
import { linearAttentionSequenceStateBytes, lowerLinearAttention } from './linearAttention';
import { lowerMla, mlaSequenceStateBytes } from './mla';
import { lowerMlp } from './mlp';
import { lowerMoe, moeResidentWeightBytes } from './moe';
import { lowerNgram, ngramResidentWeightBytes, ngramSequenceStateBytes } from './ngram';
import { lowerNorm, lowerResidual } from './norm';
import { lowerShortConv, shortConvSequenceStateBytes } from './shortConv';

/** One phase step of one microbatch, as seen by the lowering. */
export interface LoweringContext {
  /** Which phase is lowered (prefill: the whole prompt; decode: one new token per sequence). */
  phase: 'prefill' | 'decode';
  /** Sequences in the microbatch (b_mb), per replica. Count. */
  seqs: number;
  /** New query tokens per sequence in this step (prefill: prompt length; decode: 1). Unit: tokens. */
  qLen: number;
  /** Keys visible to the last query (past + qLen). Unit: tokens. */
  kvLen: number;
  /** Numeric formats. */
  quant: QuantConfig;
  /** Parallelism degrees. */
  parallel: ParallelConfig;
}

/** The ops of one lowered layer on one device of its stage, in execution order. */
export interface LoweredLayer {
  /** `'embed'`, `'layer.<i>'` or `'head'`. */
  name: string;
  /** Index into `ModelSpec.layers`, or `null` for the embedding and the head. */
  layerIndex: number | null;
  /** Per-device ops in execution order. */
  ops: Op[];
}

/** One pipeline stage: its layers (stage 0 starts with `'embed'`, the last stage ends with `'head'`). */
export interface LoweredStage {
  /** Stage index, 0-based. */
  index: number;
  /** Lowered layers of this stage, in execution order. */
  layers: LoweredLayer[];
  /** Activation send to the next stage (absent on the last stage). */
  sendToNext?: CollectiveOp;
}

/** A model lowered for one phase step: per-stage, per-device ops. */
export interface LoweredModel {
  /** Pipeline stages, `parallel.pp` of them, in order. */
  stages: LoweredStage[];
}

/** Lower a whole model for one phase step (embedding, all layers split into PP stages, final norm + LM head). */
export function lowerModel(model: ModelSpec, ctx: LoweringContext): LoweredModel {
  const pp = ctx.parallel.pp;
  const L = model.layers.length;
  if (pp < 1 || pp > L) throw new Error(`pp (${pp}) must be between 1 and the number of layers (${L})`);
  const stages: LoweredStage[] = [];
  for (let s = 0; s < pp; s++) {
    const layers: LoweredLayer[] = [];
    if (s === 0) layers.push({ name: 'embed', layerIndex: null, ops: lowerEmbedding(model, ctx) });
    const [start, end] = stageLayerRange(L, pp, s);
    for (let i = start; i < end; i++) {
      const ops = model.layers[i].blocks.flatMap((block) => lowerBlock(block, ctx));
      layers.push({ name: `layer.${i}`, layerIndex: i, ops });
    }
    const stage: LoweredStage = { index: s, layers };
    if (s === pp - 1) {
      layers.push({ name: 'head', layerIndex: null, ops: lowerHead(model, ctx) });
    } else {
      // Formula: send_recv S = T·(N·h)·bpe(act), N = residualStreams (default 1)
      // See docs/simulation-model.md §5 "PP split"
      stage.sendToNext = {
        kind: 'collective',
        name: 'pp.send_recv',
        collective: 'send_recv',
        group: 'pp',
        groupSize: 2,
        messageBytes: stepTokens(ctx) * (model.residualStreams ?? 1) * model.hidden * actBpe(ctx),
        dtype: ctx.quant.activations,
      };
    }
    stages.push(stage);
  }
  return { stages };
}

/** Lower one block to its per-device ops, in execution order (a `switch` on `block.kind` into `lower/<kind>.ts`). */
export function lowerBlock(block: BlockSpec, ctx: LoweringContext): Op[] {
  switch (block.kind) {
    case 'norm':
      return lowerNorm(block, ctx);
    case 'residual':
      return lowerResidual(block, ctx);
    case 'gqa_attention':
      return lowerGqa(block, ctx);
    case 'mla_attention':
      return lowerMla(block, ctx);
    case 'csa_attention':
      return lowerCsa(block, ctx);
    case 'linear_attention':
      return lowerLinearAttention(block, ctx);
    case 'mlp':
      return lowerMlp(block, ctx);
    case 'moe':
      return lowerMoe(block, ctx);
    case 'hc_pre':
      return lowerHcPre(block, ctx);
    case 'hc_post':
      return lowerHcPost(block, ctx);
    case 'ngram_embedding':
      return lowerNgram(block, ctx);
    case 'short_conv':
      return lowerShortConv(block, ctx);
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Weight bytes of one block resident on ONE device (after TP/EP sharding; all local experts, not only the
 * active ones). Unit: bytes.
 *
 * Every weight of a non-MoE block is read exactly once per step whatever the token count, so its resident
 * bytes equal the weight bytes of its ops in any step: we lower a one-token decode step and sum them (this keeps
 * resident and streamed weights consistent by construction). A MoE step touches only active experts, and an
 * n-gram step only its gathered table rows, so those use their own formulas ({@link moeResidentWeightBytes},
 * {@link ngramResidentWeightBytes}).
 */
export function blockResidentWeightBytes(block: BlockSpec, ctx: LoweringContext): number {
  switch (block.kind) {
    case 'moe':
      return moeResidentWeightBytes(block, ctx);
    case 'ngram_embedding':
      return ngramResidentWeightBytes(block, ctx);
    case 'norm':
    case 'residual':
    case 'gqa_attention':
    case 'mla_attention':
    case 'linear_attention':
    case 'csa_attention':
    case 'hc_pre':
    case 'hc_post':
    case 'short_conv':
    case 'mlp': {
      const oneTokenStep: LoweringContext = { ...ctx, phase: 'decode', seqs: 1, qLen: 1, kvLen: 2 };
      // Formula: W_block = Σ_ops weightBytes(op) over a one-token step
      // See docs/simulation-model.md §8 "Memory fit"
      return lowerBlock(block, oneTokenStep).reduce((sum, op) => sum + opBytes(op).weightBytes, 0);
    }
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Per-sequence state bytes of one block on ONE device for a context of `contextTokens`: KV / MLA latent /
 * indexer-key caches × contextTokens, plus the fixed recurrent and conv1d state. Hyper-connection steps keep
 * none. Unit: bytes per sequence.
 */
export function blockSequenceStateBytes(block: BlockSpec, ctx: LoweringContext, contextTokens: number): number {
  switch (block.kind) {
    case 'gqa_attention':
      return gqaSequenceStateBytes(block, ctx, contextTokens);
    case 'mla_attention':
      return mlaSequenceStateBytes(block, ctx, contextTokens);
    case 'csa_attention':
      return csaSequenceStateBytes(block, ctx, contextTokens);
    case 'linear_attention':
      return linearAttentionSequenceStateBytes(block, ctx);
    case 'ngram_embedding':
      return ngramSequenceStateBytes(block, ctx);
    case 'short_conv':
      return shortConvSequenceStateBytes(block, ctx);
    case 'norm':
    case 'residual':
    case 'mlp':
    case 'moe':
    case 'hc_pre':
    case 'hc_post':
      return 0;
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Layer index range `[start, end)` of stage `s` when `L` layers are split into `pp` stages:
 * `[floor(s·L/pp), floor((s+1)·L/pp))`. Counts and indices (no unit).
 */
export function stageLayerRange(L: number, pp: number, s: number): [number, number] {
  // Formula: [floor(s·L/pp), floor((s+1)·L/pp))
  // See docs/simulation-model.md §5 "PP split"
  return [Math.floor((s * L) / pp), Math.floor(((s + 1) * L) / pp)];
}
