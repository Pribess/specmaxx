/**
 * Decode context parallelism (DCP): the one place with its sharding rules, collective sizes and error messages.
 * See docs/simulation-model.md §5 "Decode context parallelism (DCP)".
 *
 * DCP degree `d` (`ParallelConfig.dcp`, absent = 1) shards the decode KV cache of full-attention layers along the
 * TOKEN axis over `d` consecutive ranks of each TP group (the DCP group; no new devices). Rank i of a TP group still
 * owns query heads `[i·Hl, (i+1)·Hl)`; DCP group g is TP ranks `[g·d, (g+1)·d)`. Per decode step, per sharded layer
 * and rank (T = seqs·qLen, Hl = heads/tp):
 *
 * 1. Q all-gather over the DCP group: every rank gets the queries of all `Hl·d` heads of its group.
 * 2. Attention of those `Hl·d` heads on this rank's share of the keys (`AttentionOp.kvTokenShards = d`): attended
 *    pairs / d, KV tokens read and written ceil(·/d). FLOP per rank is unchanged (Hl·d heads × pairs/d).
 * 3. Reduce-scatter of the partial outputs with their log-sum-exp: every rank gets the merged output of its own
 *    `Hl` heads. The (o, lse) merge is the associative online-softmax combine, so one reduce-scatter suffices.
 *
 * Which blocks are sharded: MLA without `sparse`; GQA with full attention (no sliding window, no sparse, no
 * relative-position bias, no K/V convs). A sliding-window GQA block stays local (its cache is ≤ W tokens).
 * Linear-attention, SSM and short-conv state is not token-shardable and is unchanged. Sparse (top-k) attention,
 * CSA, relative-position bias and K/V convs are rejected (not modelled). Prefill is unchanged.
 *
 * With d = 1 none of this applies: no op, no field and no byte differs from before.
 */

import type {
  CsaAttentionBlock,
  GqaAttentionBlock,
  MlaAttentionBlock,
  ModelSpec,
} from '../ir/model';
import type { AttentionOp, CollectiveOp } from '../ir/ops';
import { actBpe, reduceBpe, reduceDtype } from './common';
import type { LoweringContext } from './lower';

/**
 * Bytes of one log-sum-exp value: fp32, one per (token, head) partial output, carried with the output so the
 * partial results of the d key shards can be merged. Unit: bytes.
 */
export const DCP_LSE_BYTES = 4;

/** The exact error for DCP with a block drafter (DFlash 2, DSpark): its cache is not sharded (§9). */
export const DCP_BLOCK_DRAFTER_ERROR =
  'parallel.dcp > 1 with a block drafter (dflash2, dspark) is not supported yet';

/** The exact error when `d` does not divide `tp`: the DCP group must lie inside one TP group. */
export function dcpDivideTpError(d: number, tp: number): string {
  return `parallel.dcp (${d}) must divide tp (${tp})`;
}

/** The exact error for a model without any block DCP could shard (MLA, full-attention GQA). */
export function dcpNoBlockError(d: number): string {
  return `dcp (${d}) > 1 but the model has no full-attention (MLA or GQA) block to shard`;
}

/**
 * The exact error for a GQA block whose KV-head replication factor `r = tp / kvHeads` (1 when kvHeads ≥ tp) is
 * not a multiple of `d`: the d ranks of a DCP group must hold the same KV head, so they must lie inside the r ranks
 * that replicate it. vLLM requires the same divisibility (`tp % dcp == 0` for MLA, `(tp // kv_heads) % dcp == 0` for
 * GQA; https://vllm.ai/blog/2026-08-07-decode-context-parallelism); its docs state the bound `dcp ∈ [1, tp / H]`
 * (https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/).
 */
export function dcpKvReplicationError(blockName: string, d: number, r: number): string {
  return `${blockName} dcp (${d}) must divide tp / kvHeads (${r}): KV heads are replicated on ${r} ranks and DCP shards that copy`;
}

/** What a block kind cannot be combined with DCP in this model (not a vLLM claim; a modelling limit, §9). */
export type DcpUnsupportedFeature = 'sparse' | 'csa' | 'relativeBiasOrKvConv';

/** The exact error for a block DCP > 1 does not model. */
export function dcpUnsupportedError(
  blockName: string,
  d: number,
  feature: DcpUnsupportedFeature,
): string {
  const what =
    feature === 'sparse'
      ? 'sparse (top-k) attention'
      : feature === 'csa'
        ? 'compressed sparse attention (CSA)'
        : 'a relative-position bias or K/V short convs';
  return `${blockName} dcp (${d}) > 1 is not supported with ${what}`;
}

/** DCP degree d of the lowering: `parallel.dcp ?? 1`. Count (ranks). */
export function dcpDegree(ctx: LoweringContext): number {
  return ctx.parallel.dcp ?? 1;
}

/**
 * True when this step shards the KV of full-attention layers over tokens: a decode step with d > 1. Prefill lowers
 * exactly as without DCP (memory is sharded in every phase, see {@link dcpKvTokens}).
 */
export function dcpActive(ctx: LoweringContext): boolean {
  return ctx.phase === 'decode' && dcpDegree(ctx) > 1;
}

/**
 * KV tokens per sequence one rank of a DCP group stores for a context of `contextTokens`: `ceil(ctx / d)` (tokens
 * are interleaved over the d ranks, so the busiest holds at most one more than the mean); exactly `contextTokens`
 * when d = 1. Unit: tokens.
 */
export function dcpKvTokens(contextTokens: number, d: number): number {
  if (d === 1) return contextTokens;
  // Formula: N_rank = ceil(ctx / d)
  // See docs/simulation-model.md §5 "Decode context parallelism (DCP)"
  return Math.ceil(contextTokens / d);
}

/**
 * Ranks of a TP group that hold a copy of the same KV head, `r = tp / kvHeads` when kvHeads < tp, else 1 (every
 * rank holds distinct KV heads). MLA's single latent head gives r = tp. Count.
 */
export function dcpKvReplication(kvHeads: number, tp: number): number {
  // Formula: r = kvHeads < tp ? tp / kvHeads : 1
  return kvHeads < tp ? tp / kvHeads : 1;
}

/** How DCP treats a block: its KV is token-sharded, it stays as without DCP, or the combination is rejected. */
export type DcpBlockRole = 'sharded' | 'local' | { unsupported: DcpUnsupportedFeature };

/**
 * The role of an attention block under DCP (independent of d): MLA without sparse and full-attention GQA are
 * sharded; sliding-window GQA is local; sparse MLA/GQA, CSA, and GQA with a relative-position bias or K/V convs are
 * unsupported.
 */
export function dcpBlockRole(
  block: MlaAttentionBlock | GqaAttentionBlock | CsaAttentionBlock,
): DcpBlockRole {
  switch (block.kind) {
    case 'mla_attention':
      return block.sparse === undefined ? 'sharded' : { unsupported: 'sparse' };
    case 'csa_attention':
      return { unsupported: 'csa' };
    case 'gqa_attention':
      if (block.slidingWindow !== undefined) return 'local';
      if (block.sparse !== undefined) return { unsupported: 'sparse' };
      if (block.relativeBias !== undefined || block.kvConvKernel !== undefined) {
        return { unsupported: 'relativeBiasOrKvConv' };
      }
      return 'sharded';
    default: {
      const _never: never = block;
      throw new Error(`unknown attention block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * True when the model has an attention block DCP could apply to: an MLA or CSA block, or a GQA block without a
 * sliding window (unsupported ones count, so their own error is the one reported, not {@link dcpNoBlockError}).
 */
export function dcpHasCandidateBlock(model: ModelSpec): boolean {
  return model.layers.some((l) =>
    l.blocks.some(
      (b) =>
        b.kind === 'mla_attention' ||
        b.kind === 'csa_attention' ||
        (b.kind === 'gqa_attention' && b.slidingWindow === undefined),
    ),
  );
}

/**
 * Checks an attention block against DCP when d > 1 (every phase, so memory-only callers fail early too) and returns
 * whether its KV is token-sharded. Throws {@link dcpUnsupportedError} for an unsupported block and, for a sharded
 * GQA block, {@link dcpKvReplicationError} unless d divides `r = tp / kvHeads`. The rule `d | tp` is global
 * (`engine/validate.ts: dcpErrors`), so the GQA rule is checked only when it holds. d = 1: always false, no check.
 */
export function dcpShardsBlock(
  block: MlaAttentionBlock | GqaAttentionBlock | CsaAttentionBlock,
  ctx: LoweringContext,
): boolean {
  const d = dcpDegree(ctx);
  if (d === 1) return false;
  const role = dcpBlockRole(block);
  if (role === 'local') return false;
  if (role !== 'sharded') throw new Error(dcpUnsupportedError(block.name, d, role.unsupported));
  const tp = ctx.parallel.tp;
  if (block.kind === 'gqa_attention' && tp % d === 0) {
    const r = dcpKvReplication(block.kvHeads, tp);
    if (r % d !== 0) throw new Error(dcpKvReplicationError(block.name, d, r));
  }
  return true;
}

/**
 * The Q all-gather in front of a sharded attention core: each rank contributes the queries of its `localHeads`
 * heads and receives those of all `localHeads·d` heads of its DCP group. `qkDim` is the query width per head
 * (absorbed MLA: rkv + dr; GQA: hd). messageBytes = the full OUTPUT (the all-gather convention of `CollectiveOp`).
 */
export function dcpGatherQ(
  tokens: number,
  localHeads: number,
  qkDim: number,
  ctx: LoweringContext,
): CollectiveOp {
  const d = dcpDegree(ctx);
  const gatheredHeads = localHeads * d;
  // Formula: S_q = T·(Hl·d)·qkDim·bpe(act)                  (all_gather: full output)
  // See docs/simulation-model.md §5 "Decode context parallelism (DCP)"
  const messageBytes = tokens * gatheredHeads * qkDim * actBpe(ctx);
  return {
    kind: 'collective',
    name: 'attn.dcp_q_all_gather',
    collective: 'all_gather',
    group: 'dcp',
    groupSize: d,
    messageBytes,
    dtype: ctx.quant.activations,
  };
}

/**
 * The reduce-scatter after a sharded attention core: each rank holds a partial output (over its key shard) of all
 * `localHeads·d` heads plus one fp32 log-sum-exp per (token, head), and gets back the merged output of its own
 * `localHeads` heads. `vDim` is the output width per head (absorbed MLA: rkv, still in latent space; GQA: hd).
 * messageBytes = the full INPUT (the reduce-scatter convention of `CollectiveOp`).
 */
export function dcpReduceOut(
  tokens: number,
  localHeads: number,
  vDim: number,
  ctx: LoweringContext,
): CollectiveOp {
  const d = dcpDegree(ctx);
  const gatheredHeads = localHeads * d;
  // Formula: S_o = T·(Hl·d)·(vDim·reduceBpe + 4)             (reduce_scatter: full input; 4 B = fp32 LSE)
  // See docs/simulation-model.md §5 "Decode context parallelism (DCP)"
  const messageBytes = tokens * gatheredHeads * (vDim * reduceBpe(ctx) + DCP_LSE_BYTES);
  return {
    kind: 'collective',
    name: 'attn.dcp_out_reduce_scatter',
    collective: 'reduce_scatter',
    group: 'dcp',
    groupSize: d,
    messageBytes,
    dtype: reduceDtype(ctx),
  };
}

/**
 * The attention core of one rank of a DCP group: the gathered `heads·d` query heads over this rank's 1/d share of
 * the keys (`kvTokenShards = d`); every other field as without DCP. KV heads are unchanged (MLA 1; GQA KVl = 1,
 * since d | tp / kvHeads).
 */
export function dcpShardAttention(op: AttentionOp, ctx: LoweringContext): AttentionOp {
  const d = dcpDegree(ctx);
  // Formula: heads_rank = Hl·d; kvTokenShards = d   (FLOP: Hl·d · pairs/d = Hl · pairs, unchanged)
  // See docs/simulation-model.md §5 "Decode context parallelism (DCP)"
  return { ...op, heads: op.heads * d, kvTokenShards: d };
}
