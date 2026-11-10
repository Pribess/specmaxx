/**
 * KV-cache tiering (`OffloadConfig.kvCache`): the one place with its rules. Which attention KV may live off the
 * device, how many bytes one token of it is, and the per-step fetch of the non-resident past tokens from host RAM,
 * the SSD and the remote KV pool. See docs/simulation-model.md §12 ("KV cache off-device", "Remote KV pool").
 *
 * Notation (per device, per microbatch step, per offloadable attention layer):
 * - `b` sequences of the microbatch, `qLen` new tokens each, `past = kvLen − qLen` tokens already cached;
 * - `kvB` bytes per cached token on this device (GQA `2·KVl·hd·bpe(kv)`, MLA `(rkv + dr)·bpe(kv)`);
 * - `f` the device-resident share (decode: f_dev; prefill: f_pre, the share of the prefix-cache hit on the device);
 * - `N_res = f·past` resident past tokens, `M` missed (fetched) tokens per sequence, split over the tiers as
 *   `M_ram = M·ramShare`, `M_ssd = M·nvmeShare`, `M_pool = M·remotePoolShare`.
 *
 * Offloadable: GQA K/V of full and top-k layers (no sliding window), and the MLA latent (of the owning layer; a layer
 * reading another layer's latent fetches the same misses again). Stays on the device: sliding-window caches,
 * indexer keys, K/V conv state, recurrent/short-conv state, CSA caches, and the drafter's KV.
 */

import { bytesPerElement } from '../ir/dtype';
import type { BlockSpec, GqaAttentionBlock, MlaAttentionBlock, ModelSpec } from '../ir/model';
import type { AttentionOp, TransferOp } from '../ir/ops';
import { expertHitsMisses } from '../engine/transfer';
import { shardKvHeads } from './common';
import { dcpDegree, dcpKvTokens, dcpShardsBlock } from './dcp';
import type { KvCachePlacement, LoweringContext } from './lower';
import { prefetchedFraction } from './offload';

/** An attention block whose KV can be tiered: GQA (incl. top-k) without a sliding window, or MLA. */
export type KvOffloadBlock = GqaAttentionBlock | MlaAttentionBlock;

/**
 * True if the block's KV grows with the context and may live off the device: GQA without a sliding window (full or
 * top-k), and MLA. Sliding-window GQA (≤ W tokens), CSA and every non-attention block are not.
 */
export function kvOffloadableBlock(block: BlockSpec): block is KvOffloadBlock {
  if (block.kind === 'gqa_attention') return block.slidingWindow === undefined;
  return block.kind === 'mla_attention';
}

/** True if any layer of the model has an offloadable KV cache ({@link kvOffloadableBlock}). */
export function modelHasOffloadableKv(model: ModelSpec): boolean {
  return model.layers.some((l) => l.blocks.some(kvOffloadableBlock));
}

/**
 * True if any offloadable layer of the model uses top-k (sparse) attention, so the KV hit ratio
 * (`KvCacheOffloadConfig.hitRatio`) has an effect.
 */
export function modelHasTopKKv(model: ModelSpec): boolean {
  return model.layers.some((l) =>
    l.blocks.some((b) => kvOffloadableBlock(b) && b.sparse !== undefined),
  );
}

/**
 * True if the model has a CSA (DeepSeek-V4 compressed sparse attention) block, whose caches stay on the device under
 * KV-cache offload (a validation warning).
 */
export function modelHasCsaBlock(model: ModelSpec): boolean {
  return model.layers.some((l) => l.blocks.some((b) => b.kind === 'csa_attention'));
}

/**
 * Bytes of one cached token of the block's KV on this device: GQA `2·KVl·hd·bpe(kv)` (K and V of the KVl local KV
 * heads), MLA `(rkv + dr)·bpe(kv)` (the latent plus the shared rope key, replicated on every TP rank). The same for a
 * layer that reads another layer's latent (`kvCacheFromLayer`): it reads those bytes. Unit: bytes per token.
 */
export function kvBytesPerToken(block: KvOffloadBlock, ctx: LoweringContext): number {
  const bpe = bytesPerElement(ctx.quant.kvCache);
  if (block.kind === 'mla_attention') {
    // Formula: kvB = (rkv + dr)·bpe(kv)   (MLA latent)
    return (block.kvLoraRank + block.qkRopeDim) * bpe;
  }
  const KVl = shardKvHeads(block.kvHeads, ctx.parallel.tp, `${block.name} kvHeads`);
  // Formula: kvB = 2·KVl·hd·bpe(kv)   (GQA K and V)
  return 2 * KVl * block.headDim * bpe;
}

/**
 * Per-sequence bytes of the block's offloadable KV on this device for a context of `contextTokens`: the tokens this
 * device stores (`ceil(ctx/d)` under decode context parallelism) × kvB, for a block that OWNS its cache (an MLA layer
 * reading another layer's latent stores none). 0 for blocks that are not offloadable. This is the part of
 * `blockSequenceStateBytes` that the tier fractions split; the rest of the block's state stays on the device.
 * Unit: bytes per sequence.
 */
export function kvOffloadableStateBytes(
  block: BlockSpec,
  ctx: LoweringContext,
  contextTokens: number,
): number {
  if (!kvOffloadableBlock(block)) return 0;
  if (block.kind === 'mla_attention' && block.kvCacheFromLayer !== undefined) return 0;
  const tokens = dcpShardsBlock(block, ctx)
    ? dcpKvTokens(contextTokens, dcpDegree(ctx))
    : contextTokens;
  // Formula: S_off = N_rank·kvB,  N_rank = ceil(ctx/d)   (= ctx without DCP)
  // See docs/simulation-model.md §12 ("KV cache off-device")
  return tokens * kvBytesPerToken(block, ctx);
}

/** Expected missed past tokens of one attention step (see {@link kvMisses}). Unit: tokens per sequence (real). */
export interface KvMisses {
  /** Past tokens this device caches for one sequence, `past` (`ceil(past/d)` under DCP). Unit: tokens. */
  pastTokens: number;
  /** Resident past tokens `N_res = f·past`. Unit: tokens (real). */
  residentTokens: number;
  /** Past tokens the attention reads, `U`: every past token (full, prefill) or `min(past, topK·qLen)` (top-k). */
  attendedTokens: number;
  /** Missed (fetched) past tokens `M = U − hits`. Unit: tokens (real). */
  misses: number;
  /** True for a top-k decode step (scattered rows, q by the expert rule). */
  topK: boolean;
}

/**
 * Missed past tokens of one attention op, per sequence (docs §12 "KV cache off-device"):
 * - full attention, and every prefill (the prefix-cache hit is loaded whole): `U = past`, `hits = N_res`;
 * - top-k decode: `U = min(past, topK·qLen)` selected past tokens, `hits = clamp(U·h, max(0, U + N_res − past),
 *   min(U, N_res))` (the expert hit rule of `engine/transfer.ts: expertHitsMisses`, past tokens for experts);
 * then `M = U − hits`. `past = kvLen − qLen`, this rank's `ceil(past/d)` when the op is DCP-sharded.
 * A chunk of chunked pipeline prefill (`ctx.prefixCacheTokens` set, docs §7 "Chunked pipeline prefill") splits its
 * past into the prefix-cache hit `past_pre = min(past, prefixCacheTokens)` (placed by f_pre) and the earlier chunks of
 * the same prefill `past_live = past − past_pre`, live KV placed by f_dev like any other (the memory fit counts only
 * that share on the device): `N_res = f_pre·past_pre + f_dev·past_live`,
 * `M = (1 − f_pre)·past_pre + (1 − f_dev)·past_live`.
 */
export function kvMisses(
  attention: AttentionOp,
  ctx: LoweringContext,
  placement: KvCachePlacement,
): KvMisses {
  const shards = attention.kvTokenShards ?? 1;
  // Formula: past = kvLen − qLen   (per rank ceil(past/d) under DCP)
  const pastTokens = dcpKvTokens(attention.kvLen - attention.qLen, shards);
  // Formula: f = decode ? f_dev : f_pre
  const f = ctx.phase === 'decode' ? placement.deviceFraction : placement.prefixDeviceFraction;
  const topK = ctx.phase === 'decode' && attention.kvSelection.kind === 'topk';
  if (ctx.phase === 'prefill' && ctx.prefixCacheTokens !== undefined) {
    // Formula: past_pre = min(past, cached);  past_live = past − past_pre
    //          N_res = f_pre·past_pre + f_dev·past_live;  M = (1 − f_pre)·past_pre + (1 − f_dev)·past_live
    // See docs/simulation-model.md §7 "Chunked pipeline prefill" (KV-cache offload)
    const prefixTokens = Math.min(pastTokens, dcpKvTokens(ctx.prefixCacheTokens, shards));
    const liveTokens = pastTokens - prefixTokens;
    const fLive = placement.deviceFraction;
    const prefix = expertHitsMisses(prefixTokens, f, f, prefixTokens);
    const live = expertHitsMisses(liveTokens, fLive, fLive, liveTokens);
    return {
      pastTokens,
      residentTokens: prefix.residentExperts + live.residentExperts,
      attendedTokens: pastTokens,
      misses: prefix.misses + live.misses,
      topK: false,
    };
  }
  // Formula: U = full / prefill: past;  top-k decode: min(past, topK·qLen)
  const attendedTokens =
    attention.kvSelection.kind === 'topk' && topK
      ? Math.min(pastTokens, attention.kvSelection.topK * attention.qLen)
      : pastTokens;
  // Formula: hits = full: N_res;  top-k: clamp(U·h, max(0, U + N_res − past), min(U, N_res));  M = U − hits
  const { residentExperts: residentTokens, misses } = expertHitsMisses(
    attendedTokens,
    topK ? placement.hitRatio : f,
    f,
    pastTokens,
  );
  return { pastTokens, residentTokens, attendedTokens, misses, topK };
}

/**
 * The KV fetch ops in front of one offloadable attention op (`attention`, after DCP sharding): one `TransferOp` per
 * tier with missed tokens, `attn.kv_fetch` (host RAM), `attn.kv_fetch_ssd` (SSD) and `attn.kv_fetch_pool` (remote KV
 * pool). Per tier t, with `M_t = M·share_t` and `b = attention.seqs`:
 * - reads: full attention and prefill `[{ b, M_t·kvB }]` (one contiguous span per sequence); top-k decode
 *   `[{ b·M_t, kvB }]` (one row per selected token; an SSD pays its read granularity for each, the pool none);
 * - usefulBytes `b·M_t·kvB`;
 * - q from {@link prefetchedFraction} (`kv_cache` target).
 * Empty when the context has no KV placement, or nothing is missed (f = 1, no past tokens).
 */
export function kvFetchOps(
  block: KvOffloadBlock,
  ctx: LoweringContext,
  attention: AttentionOp,
): TransferOp[] {
  const placement = ctx.offload?.kvCache;
  if (placement === undefined) return [];
  const miss = kvMisses(attention, ctx, placement);
  if (!(miss.misses > 0)) return [];
  const kvB = kvBytesPerToken(block, ctx);
  const b = attention.seqs;
  // Formula: q = p = 0: 0;  full / prefill: 1;  top-k decode: max(p, M/(past − N_res))
  const q = prefetchedFraction(ctx.offload?.prefetchAccuracy ?? 0, {
    group: 'kv_cache',
    topK: miss.topK,
    misses: miss.misses,
    nonResidentTokens: miss.pastTokens - miss.residentTokens,
  });
  const ops: TransferOp[] = [];
  const tiers = [
    { source: 'host_ram', share: placement.ramShare, name: 'attn.kv_fetch' },
    { source: 'nvme', share: placement.nvmeShare, name: 'attn.kv_fetch_ssd' },
    { source: 'remote_pool', share: placement.remotePoolShare, name: 'attn.kv_fetch_pool' },
  ] as const;
  for (const { source, share, name } of tiers) {
    // Formula: M_t = M·share_t
    const missedTokens = miss.misses * share;
    if (!(missedTokens > 0)) continue;
    // Formula: reads_t = full / prefill: [{ b, M_t·kvB }];  top-k decode: [{ b·M_t, kvB }];  useful_t = b·M_t·kvB
    // See docs/simulation-model.md §12 ("KV cache off-device", "Remote KV pool")
    const reads = miss.topK
      ? [{ count: b * missedTokens, bytesEach: kvB }]
      : [{ count: b, bytesEach: missedTokens * kvB }];
    ops.push({
      kind: 'transfer',
      name,
      group: 'kv_cache',
      source,
      reads,
      usefulBytes: b * missedTokens * kvB,
      prefetchedFraction: q,
    });
  }
  return ops;
}
