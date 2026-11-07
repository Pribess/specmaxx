/**
 * Lowering of DeepSeek-V4.1 compressed sparse attention (`csa_attention`). TP shards the query heads
 * (`Hl = heads/tp`) and the output groups (`Gl = outGroups/tp`); the single K = V vector per token, the compressor,
 * the indexer and every cache are REPLICATED on each TP rank (one KV head). See docs/simulation-model.md §5 "CSA".
 *
 * Notation: hd = headDim, dr = ropeDim, rq = qLoraRank, G = outGroups, R_o = outLoraRank, W = windowTokens,
 * r = global compress ratio, T = seqs·qLen.
 *
 * One attention op covers both key sets in one softmax: the raw window (primary keys) and the top-k compressed
 * entries (`secondaryKv`), so Q is read and O written once.
 */

import { bytesPerElement, type TensorFormat } from '../ir/dtype';
import type { CsaAttentionBlock } from '../ir/model';
import type { AttentionOp, Op } from '../ir/ops';
import { actBpe, attentionMath, elementwise, linearGemm, normOp, shardHeads, stepTokens, tpAllReduce } from './common';
import { indexerKeyStateBytes, lowerSparseSelection, sparseTopK } from './indexer';
import type { LoweringContext } from './lower';

/** Storage format of the CSA compressed (global) KV cache: `quant.compressedKvCache`, else `kvCache`. */
export function compressedKvFormat(ctx: LoweringContext): TensorFormat {
  return ctx.quant.compressedKvCache ?? ctx.quant.kvCache;
}

/** A rope op over `elems` activations (read and written once). */
function ropeOp(name: string, elems: number, ctx: LoweringContext): Op {
  const act = actBpe(ctx);
  return elementwise(name, 'rope', { elems, activationReadBytes: elems * act, activationWriteBytes: elems * act }, ctx);
}

/**
 * The compressor of a kv-source layer, run on all T new tokens. r > 1: `csa.wkv_c` and `csa.wgate_c` (n = hd, k = h)
 * → `csa.kv_pool` (softmax-weighted pooling of r tokens) → rmsnorm over T/r·hd → rope over T/r·dr; r = 1: one
 * `csa.wkv_c` GEMM → rmsnorm → rope. T/r is continuous (decode pools once every r steps; amortized).
 */
function compressorOps(block: CsaAttentionBlock, r: number, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  const h = block.hidden;
  const hd = block.headDim;
  const act = actBpe(ctx);
  const m = { module: 'attention_kv_down' } as const;
  const ops: Op[] = [linearGemm('csa.wkv_c', { m: T, n: hd, k: h }, ctx, m)];
  if (r > 1) {
    // Formula: wkv_c, wgate_c: n = hd, k = h; kv_pool over T·hd: read 2·T·hd, write (T/r)·hd
    // See docs/simulation-model.md §5 "CSA"
    ops.push(linearGemm('csa.wgate_c', { m: T, n: hd, k: h }, ctx, m));
    const elems = T * hd;
    ops.push(
      elementwise(
        'csa.kv_pool',
        'kv_pool',
        { elems, activationReadBytes: 2 * elems * act, activationWriteBytes: (elems / r) * act },
        ctx,
      ),
    );
  }
  ops.push(normOp('csa.compress_norm', 'rmsnorm', T / r, hd, ctx));
  ops.push(ropeOp('csa.compress_rope', (T / r) * block.ropeDim, ctx));
  return ops;
}

/**
 * CSA ops, in order: q_a → q_norm → q_b → rope(q) → wkv → kv_norm → rope(k) → [compressor] → [indexer ops] →
 * attention (window + compressed top-k, one softmax) → inverse rope(o) → wo_a (grouped) → wo_b → [TP all-reduce].
 */
export function lowerCsa(block: CsaAttentionBlock, ctx: LoweringContext): Op[] {
  const tp = ctx.parallel.tp;
  const T = stepTokens(ctx);
  const h = block.hidden;
  const hd = block.headDim;
  const dr = block.ropeDim;
  const rq = block.qLoraRank;
  const Hl = shardHeads(block.heads, tp, `${block.name} heads`);
  const Gl = shardHeads(block.outGroups, tp, `${block.name} outGroups`);
  const attn = { module: 'attention' } as const;
  const ops: Op[] = [];

  // Formula: q_a: n = rq, k = h (replicated) → rmsnorm(T·rq) → q_b: n = Hl·hd, k = rq → rope over T·Hl·dr
  // Formula: wkv: n = hd, k = h (replicated) → rmsnorm(T·hd) → rope over T·dr
  // See docs/simulation-model.md §5 "CSA"
  ops.push(linearGemm('csa.q_a', { m: T, n: rq, k: h }, ctx, attn));
  ops.push(normOp('csa.q_norm', 'rmsnorm', T, rq, ctx));
  ops.push(linearGemm('csa.q_b', { m: T, n: Hl * hd, k: rq }, ctx, attn));
  ops.push(ropeOp('csa.q_rope', T * Hl * dr, ctx));
  ops.push(linearGemm('csa.wkv', { m: T, n: hd, k: h }, ctx, attn));
  ops.push(normOp('csa.kv_norm', 'rmsnorm', T, hd, ctx));
  ops.push(ropeOp('csa.k_rope', T * dr, ctx));

  const g = block.global;
  const window = { format: ctx.quant.kvCache, elemsPerToken: hd };
  const attention: AttentionOp = {
    kind: 'attention',
    name: 'attn.core',
    seqs: ctx.seqs,
    heads: Hl,
    kvHeads: 1,
    qkHeadDim: hd,
    vHeadDim: hd,
    qLen: ctx.qLen,
    kvLen: ctx.kvLen,
    causal: true,
    kvSelection: { kind: 'window', windowTokens: block.windowTokens },
    // K = V: one hd-wide vector per token is read and written.
    kvRead: window,
    kvWrite: window,
    ...attentionMath(ctx),
    activationDtype: ctx.quant.activations,
  };
  if (g !== undefined) {
    const r = g.compressRatio;
    const ownsCache = g.kvCacheFromLayer === undefined;
    if (ownsCache) ops.push(...compressorOps(block, r, ctx));
    ops.push(...lowerSparseSelection(g.selection, h, rq, ctx, r));
    // Formula: secondary: r tokens/entry, topk over entries, read hd per entry, write hd/r per new token (owner only)
    // See docs/simulation-model.md §3 "Attention"
    const fmt = compressedKvFormat(ctx);
    attention.secondaryKv = {
      tokensPerEntry: r,
      kvSelection: { kind: 'topk', topK: sparseTopK(g.selection) },
      kvRead: { format: fmt, elemsPerToken: hd },
      kvWrite: { format: fmt, elemsPerToken: ownsCache ? hd / r : 0 },
    };
  }
  ops.push(attention);
  ops.push(ropeOp('csa.o_inverse_rope', T * Hl * dr, ctx));

  // Formula: wo_a: batch = Gl, m = T, n = R_o, k = H·hd/G; wo_b: m = T, n = h, k = Gl·R_o
  ops.push(linearGemm('csa.wo_a', { m: T, n: block.outLoraRank, k: (block.heads * hd) / block.outGroups }, ctx, { ...attn, batch: Gl }));
  ops.push(linearGemm('csa.wo_b', { m: T, n: h, k: Gl * block.outLoraRank }, ctx, attn));
  ops.push(...tpAllReduce('attn.tp_all_reduce', T, h, ctx));
  return ops;
}

/**
 * Per-sequence cache bytes of a CSA block on one device for `contextTokens` (all replicated per TP rank): the raw
 * window, plus for a compressed-cache owner its `ceil(ctx/r)` entries and the compressor's fp32 tail (r > 1), plus
 * the indexer keys this layer owns. Unit: bytes.
 */
export function csaSequenceStateBytes(block: CsaAttentionBlock, ctx: LoweringContext, contextTokens: number): number {
  const hd = block.headDim;
  // Formula: S = min(ctx, W)·hd·bpe(kv) + [owner] (ceil(ctx/r)·hd·bpe(ckv) + [r > 1] 2·r·hd·4) + [owns keys] ceil(ctx/p)·hd_I·bpe(key)
  // See docs/simulation-model.md §8 "Memory fit"
  const windowBytes = Math.min(contextTokens, block.windowTokens) * hd * bytesPerElement(ctx.quant.kvCache);
  const g = block.global;
  if (g === undefined) return windowBytes;
  const r = g.compressRatio;
  const fp32Bytes = 4;
  const ownedBytes =
    g.kvCacheFromLayer === undefined
      ? Math.ceil(contextTokens / r) * hd * bytesPerElement(compressedKvFormat(ctx)) + (r > 1 ? 2 * r * hd * fp32Bytes : 0)
      : 0;
  const keyBytes = g.selection.kind === 'indexer' ? indexerKeyStateBytes(g.selection.indexer, ctx, contextTokens) : 0;
  return windowBytes + ownedBytes + keyBytes;
}
