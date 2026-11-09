/**
 * Lowering: model spec + workload step + quantization + parallelism → the ops ONE device runs per stage.
 *
 * Conventions (docs/simulation-model.md §5): `T = seqs · qLen` tokens per device step; TP shards heads and
 * intermediate widths (`Hl = heads / tp`; KV heads are replicated when `kvHeads < tp`); PP splits layers into
 * contiguous stages ({@link stageLayerRange}); DP replicas are independent and not lowered separately.
 */

import { opBytes } from '../ir/metrics';
import type { CollectiveOp, Op, TransferOp } from '../ir/ops';
import type {
  BlockSpec,
  DraftSpec,
  LayerSpec,
  ModelSpec,
  ParallelConfig,
  QuantConfig,
} from '../ir/model';
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
import { lowerNorm, lowerResidual, lowerResidualMix } from './norm';
import { denseLayerPlacementAt, prefetchedFraction } from './offload';
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
  /**
   * Resolved off-device placement (deployment `offload`, design doc §2.1); absent = everything on the device, and
   * lowering emits no `TransferOp` (identity).
   */
  offload?: OffloadLowering;
  /**
   * Resolved speculative-decoding step (deployment `speculative`, design doc §3.3); absent = plain decode (`qLen 1`,
   * no `draft.*` layers, no snapshot overrides).
   */
  speculative?: SpeculativeLowering;
}

/**
 * The off-device placement lowering applies, resolved from `OffloadConfig` with every default filled in (built once
 * per run by `simulate`). Each present group adds one `TransferOp` in front of the block that consumes it, only
 * when it has bytes to move. Layer ranges are `[from, to)` over `ModelSpec.layers` (zero-based indices).
 */
export interface OffloadLowering {
  /**
   * Prefetch accuracy p (`OffloadConfig.prefetchAccuracy`, default filled in): sets each transfer op's
   * `prefetchedFraction` (docs §12 "Expert prefetch"). Dimensionless, in [0, 1].
   */
  prefetchAccuracy: number;
  /** Routed experts of the MoE layers in `layers` live on `tier`; a resident cache keeps a fraction of them. */
  routedExperts?: {
    /** Tier holding each offloaded layer's full expert set. */
    tier: 'host_ram' | 'nvme';
    /** Offloaded layer range `[from, to)` (all MoE layers when the user gave none); non-MoE layers are skipped. */
    layers: { from: number; to: number };
    /** Fraction f of each offloaded layer's experts resident on the device, in [0, 1]. Dimensionless. */
    residentFraction: number;
    /**
     * h(f): probability that a unique needed expert is resident, already evaluated by the hit-ratio model
     * (design doc §2.4), before the physical clamp of the hit count. Dimensionless, in [0, 1].
     */
    hitRatio: number;
    /** Also offload the drafter's own MoE experts (DSpark MoE drafters). */
    includeDraft: boolean;
  };
  /** Whole decoder layers `[from, to)` streamed from `tier` every step. */
  denseLayers?: { tier: 'host_ram' | 'nvme'; layers: { from: number; to: number } };
  /** n-gram lookup tables read row by row from `tier` (absent = on the device), with their scale layout (design doc §2.10). */
  lookupTables?: {
    tier: 'host_ram' | 'nvme';
    scaleLayout: 'separate' | 'pinned_host' | 'interleaved';
  };
  /**
   * Unified memory: host RAM is the device pool, so pinned-host table scales (and any host-RAM group) are device
   * bytes, never a separate tier. Absent/false = dedicated memory (host RAM is its own tier).
   */
  hostRamIsDevice?: boolean;
}

/**
 * One speculative-decoding step as lowering sees it, resolved from `SpecDecodeConfig` (design doc §3.3, §3.5).
 * The verify pass is the target lowered with `qLen = verifyTokens`; the drafter's ops are appended to the last stage
 * as `draft.*` layers.
 */
export interface SpeculativeLowering {
  /** The drafter to lower: the checkpoint's MTP module(s) or the block drafter. */
  draft: DraftSpec;
  /** Draft tokens per step k. Count. */
  draftLength: number;
  /** Target tokens verified per sequence per step, k + 1. Unit: tokens. */
  verifyTokens: number;
  /**
   * Newly committed tokens of the previous step that the drafter ingests, `m_ctx = ceil(E[τ])` (design doc §3.3).
   * Unit: tokens.
   */
  contextRows: number;
  /** Recurrent/conv state snapshots written by the verify's state ops, k + 1 (`RecurrentOp.stateSnapshots`). Count. */
  stateSnapshots: number;
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
  /** Decode-only sampled uint32 token IDs returned from the last stage to stage 0 (b_mb·qLen·4 bytes). */
  sendToFirst?: CollectiveOp;
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
  if (pp < 1 || pp > L)
    throw new Error(`pp (${pp}) must be between 1 and the number of layers (${L})`);
  const stages: LoweredStage[] = [];
  for (let s = 0; s < pp; s++) {
    const layers: LoweredLayer[] = [];
    if (s === 0) layers.push({ name: 'embed', layerIndex: null, ops: lowerEmbedding(model, ctx) });
    const [start, end] = stageLayerRange(L, pp, s);
    for (let i = start; i < end; i++) {
      const fetch = denseLayerFetchOp(model.layers[i], i, ctx);
      const blockOps = model.layers[i].blocks.flatMap((block) => lowerBlock(block, ctx, i));
      const ops = fetch === undefined ? blockOps : [fetch, ...blockOps];
      layers.push({ name: `layer.${i}`, layerIndex: i, ops });
    }
    const stage: LoweredStage = { index: s, layers };
    if (s === pp - 1) {
      layers.push({ name: 'head', layerIndex: null, ops: lowerHead(model, ctx) });
      if (pp > 1 && ctx.phase === 'decode') {
        // Formula: S_return = b_mb·(k+1)·4 B; plain decode has qLen = 1.
        // See docs/simulation-model.md §5 "PP split" (audit G2).
        stage.sendToFirst = {
          kind: 'collective',
          name: 'pp.sample_return',
          collective: 'send_recv',
          group: 'pp',
          groupSize: 2,
          messageBytes: stepTokens(ctx) * 4,
          dtype: 'int32',
        };
      }
    } else {
      // Formula: send_recv S = T·(N_live·h)·bpe(act), N_live = ppBoundaryResidualVectors(model, end)
      // See docs/simulation-model.md §5 "PP split"
      const vectors = ppBoundaryResidualVectors(model, end);
      stage.sendToNext = {
        kind: 'collective',
        name: 'pp.send_recv',
        collective: 'send_recv',
        group: 'pp',
        groupSize: 2,
        messageBytes: stepTokens(ctx) * vectors * model.hidden * actBpe(ctx),
        dtype: ctx.quant.activations,
      };
    }
    stages.push(stage);
  }
  return { stages };
}

/**
 * Residual vectors per token, each `hidden` wide, that a PP stage boundary before layer `layerIndex` must carry: the
 * activations the layers from `layerIndex` on still read. Count (multiply by `T·h·bpe(act)` for bytes).
 *
 * - Plain residual: 1 (the running sum). Hyper-connections: `residualStreams` N.
 * - Attention Residuals (Kimi K3): the running sum plus every block summary stored so far, `1 + ceil(i/B)`
 *   (`builders.ts: attnResSummariesBefore`); later layers and the final mix read all of them. This equals the
 *   `candidates` of layer i's first `residual_mix` (its pre-mixer mix, which mixes exactly those vectors), so it is
 *   read from there rather than recomputed. Kimi K3 at pp 2 (boundary before layer 46, B 12): 1 + 4 = 5.
 *
 * Formula: N_live = C(first residual_mix of layer i) if it has one, else residualStreams (default 1).
 * See docs/simulation-model.md §5 "PP split".
 */
export function ppBoundaryResidualVectors(model: ModelSpec, layerIndex: number): number {
  const mix = model.layers[layerIndex]?.blocks.find((b) => b.kind === 'residual_mix');
  if (mix !== undefined && mix.kind === 'residual_mix') return mix.candidates;
  return model.residualStreams ?? 1;
}

/**
 * Bytes of ONE decoder layer resident on one device with everything on the device: Σ of its blocks' resident weight
 * bytes (all local experts, the whole table shard). What a `denseLayers` placement streams for it every step. Unit: bytes.
 */
export function layerWeightBytesOnDevice(layer: LayerSpec, ctx: LoweringContext): number {
  const onDevice: LoweringContext = { ...ctx, offload: undefined };
  // Formula: W_layer = Σ_blocks W_block   (no offload)
  return layer.blocks.reduce((sum, block) => sum + blockResidentWeightBytes(block, onDevice), 0);
}

/**
 * The whole-layer fetch of a layer in the `denseLayers` range (design doc §2.5): `reads [{ 1, W_layer }]`,
 * `usefulBytes = W_layer`, every step (no hit ratio). Undefined when the layer is on the device or has no weights.
 */
export function denseLayerFetchOp(
  layer: LayerSpec,
  layerIndex: number,
  ctx: LoweringContext,
): TransferOp | undefined {
  const placement = denseLayerPlacementAt(ctx, layerIndex);
  if (placement === undefined) return undefined;
  const W_layer = layerWeightBytesOnDevice(layer, ctx);
  if (!(W_layer > 0)) return undefined;
  // Formula: reads = [{ 1, W_layer }];  usefulBytes = W_layer
  // See docs/simulation-model.md §12 ("Placement")
  return {
    kind: 'transfer',
    name: 'layer.fetch',
    group: 'dense_layer',
    source: placement.tier,
    reads: [{ count: 1, bytesEach: W_layer }],
    usefulBytes: W_layer,
    prefetchedFraction: prefetchedFraction(ctx.offload?.prefetchAccuracy ?? 0, {
      group: 'dense_layer',
    }),
  };
}

/**
 * Lower one block to its per-device ops, in execution order (a `switch` on `block.kind` into `lower/<kind>.ts`).
 * `layerIndex` (index into `ModelSpec.layers`) locates the block for the offload placement; without it (draft layers,
 * one-token weight probes) no transfer op is emitted.
 */
export function lowerBlock(block: BlockSpec, ctx: LoweringContext, layerIndex?: number): Op[] {
  switch (block.kind) {
    case 'norm':
      return lowerNorm(block, ctx);
    case 'residual':
      return lowerResidual(block, ctx);
    case 'residual_mix':
      return lowerResidualMix(block, ctx);
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
      return lowerMoe(block, ctx, layerIndex);
    case 'hc_pre':
      return lowerHcPre(block, ctx);
    case 'hc_post':
      return lowerHcPost(block, ctx);
    case 'ngram_embedding':
      return lowerNgram(block, ctx, layerIndex);
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
 * {@link ngramResidentWeightBytes}), which also drop the off-device part of an offloaded layer `layerIndex`. A layer
 * of a `denseLayers` range is handled by the caller (resident 0).
 */
export function blockResidentWeightBytes(
  block: BlockSpec,
  ctx: LoweringContext,
  layerIndex?: number,
): number {
  switch (block.kind) {
    case 'moe':
      return moeResidentWeightBytes(block, ctx, layerIndex);
    case 'ngram_embedding':
      return ngramResidentWeightBytes(block, ctx, layerIndex);
    case 'norm':
    case 'residual':
    case 'residual_mix':
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
export function blockSequenceStateBytes(
  block: BlockSpec,
  ctx: LoweringContext,
  contextTokens: number,
): number {
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
    case 'residual_mix':
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
