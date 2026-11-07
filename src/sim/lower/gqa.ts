/**
 * Lowering of MHA/GQA/MQA attention blocks (with optional q/k-norm, output gate, sliding window, top-k sparse,
 * Inkling's relative-position bias and K/V short convs). TP shards query heads (`Hl = heads/tp`); KV heads are
 * sharded or replicated (`KVl`, see common.ts).
 * See docs/simulation-model.md §5 "TP sharding", "Relative-position bias", "Short convs" and §3 "Attention".
 */

import { bytesPerElement } from '../ir/dtype';
import type { GqaAttentionBlock } from '../ir/model';
import type { AttentionOp, KvSelection, Op } from '../ir/ops';
import {
  actBpe,
  causalConvOp,
  convStateTokens,
  attentionMath,
  elementwise,
  highPrecision,
  hpBpe,
  linearGemm,
  shardHeads,
  shardKvHeads,
  stepTokens,
  tpAllReduce,
} from './common';
import { indexerSelection, lowerSparseSelection, sparseSelectionStateBytes } from './indexer';
import type { LoweringContext } from './lower';

/** Key selection of a GQA block: sliding window, top-k (sparse), or full. */
function gqaSelection(block: GqaAttentionBlock): KvSelection {
  if (block.slidingWindow !== undefined && block.sparse !== undefined) {
    throw new Error(`${block.name}: slidingWindow and sparse are mutually exclusive`);
  }
  if (block.slidingWindow !== undefined) return { kind: 'window', windowTokens: block.slidingWindow };
  if (block.sparse !== undefined) return indexerSelection(block.sparse);
  return { kind: 'full' };
}

/**
 * GQA attention ops, in order: qkv_proj → [kv_conv] → [rel_proj → rel_table] → [qk_norm] → [rope] →
 * [indexer ops] → attention → [sigmoid_mul gate] → o_proj → [TP all-reduce of T·h].
 */
export function lowerGqa(block: GqaAttentionBlock, ctx: LoweringContext): Op[] {
  const tp = ctx.parallel.tp;
  const T = stepTokens(ctx);
  const h = block.hidden;
  const hd = block.headDim;
  const Hl = shardHeads(block.heads, tp, `${block.name} heads`);
  const KVl = shardKvHeads(block.kvHeads, tp, `${block.name} kvHeads`);
  const act = actBpe(ctx);
  const ops: Op[] = [];

  // Formula: qkv_proj: m = T, n = (Hl·(gate ? 2 : 1) + 2·KVl)·hd, k = h
  // See docs/simulation-model.md §5 "TP sharding"
  const qkvN = (Hl * (block.outputGate ? 2 : 1) + 2 * KVl) * hd;
  ops.push(linearGemm('attn.qkv_proj', { m: T, n: qkvN, k: h }, ctx, { module: 'attention', bias: block.qkvBias }));

  if (block.kvConvKernel !== undefined) {
    // Formula: kv_conv: causal_conv1d + residual over T·2·KVl·hd (K and V; 2·K + 1 FLOP/elem), state (K−1)·2·KVl·hd
    // See docs/simulation-model.md §5 "Short convs"
    ops.push(causalConvOp('attn.kv_conv', T, 2 * KVl * hd, { kernel: block.kvConvKernel, residual: true }, ctx));
  }
  const rb = block.relativeBias;
  if (rb !== undefined) {
    // Formula: rel_proj: m = T, n = Hl·relDim, k = h; rel_table: m = T·Hl, n = extent, k = relDim (weights in hp)
    // The bias gather and add inside the attention kernel are vector work, excluded like softmax (§9).
    // See docs/simulation-model.md §5 "Relative-position bias"
    ops.push(linearGemm('attn.rel_proj', { m: T, n: Hl * rb.relDim, k: h }, ctx, { module: 'attention' }));
    ops.push(
      linearGemm('attn.rel_table', { m: T * Hl, n: rb.extentTokens, k: rb.relDim }, ctx, {
        format: { weights: { dtype: highPrecision(ctx) } },
      }),
    );
  }

  // q/k-norm and RoPE act on the q and k heads of this rank: T·(Hl + KVl)·hd elements.
  const qkElems = T * (Hl + KVl) * hd;
  if (block.qkNorm) {
    ops.push(
      elementwise(
        'attn.qk_norm',
        'qk_norm',
        {
          elems: qkElems,
          activationReadBytes: qkElems * act,
          activationWriteBytes: qkElems * act,
          weightBytes: 2 * hd * hpBpe(ctx),
        },
        ctx,
      ),
    );
  }
  if (block.rope) {
    ops.push(
      elementwise(
        'attn.rope',
        'rope',
        { elems: qkElems, activationReadBytes: qkElems * act, activationWriteBytes: qkElems * act },
        ctx,
      ),
    );
  }

  if (block.sparse !== undefined) ops.push(...lowerSparseSelection(block.sparse, h, h, ctx));

  // Formula: kvRead = kvWrite = { kvCache, 2·KVl·hd } elements per token
  // See docs/simulation-model.md §3 "Attention"
  const kvStream = { format: ctx.quant.kvCache, elemsPerToken: 2 * KVl * hd };
  const attention: AttentionOp = {
    kind: 'attention',
    name: 'attn.core',
    seqs: ctx.seqs,
    heads: Hl,
    kvHeads: KVl,
    qkHeadDim: hd,
    vHeadDim: hd,
    qLen: ctx.qLen,
    kvLen: ctx.kvLen,
    causal: block.causal,
    kvSelection: gqaSelection(block),
    kvRead: kvStream,
    kvWrite: kvStream,
    ...attentionMath(ctx),
    activationDtype: ctx.quant.activations,
  };
  ops.push(attention);

  if (block.outputGate) {
    const gateElems = T * Hl * hd;
    // Formula: sigmoid_mul over T·Hl·hd: read 2·elems (out, gate), write elems
    ops.push(
      elementwise(
        'attn.output_gate',
        'sigmoid_mul',
        { elems: gateElems, activationReadBytes: 2 * gateElems * act, activationWriteBytes: gateElems * act },
        ctx,
      ),
    );
  }

  // Formula: o_proj: m = T, n = h, k = Hl·hd
  ops.push(linearGemm('attn.o_proj', { m: T, n: h, k: Hl * hd }, ctx, { module: 'attention', bias: block.outBias }));
  ops.push(...tpAllReduce('attn.tp_all_reduce', T, h, ctx));
  return ops;
}

/**
 * Per-sequence KV-cache bytes of a GQA block on one device for a context of `contextTokens`. A sliding-window
 * layer keeps only its last `slidingWindow` tokens; a sparse block adds the owned indexer key cache, and K/V convs
 * their rolling state. Unit: bytes.
 */
export function gqaSequenceStateBytes(block: GqaAttentionBlock, ctx: LoweringContext, contextTokens: number): number {
  const KVl = shardKvHeads(block.kvHeads, ctx.parallel.tp, `${block.name} kvHeads`);
  const cachedTokens = block.slidingWindow === undefined ? contextTokens : Math.min(contextTokens, block.slidingWindow);
  // Formula: S = min(ctx, W)·2·KVl·hd·bpe(kv) [+ ceil(ctx/p)·hd_I·bpe(key) if it owns indexer keys]
  //            [+ (K−1)·2·KVl·hd·bpe(act) with kvConvKernel]
  // See docs/simulation-model.md §8 "Memory fit"
  const kvBytes = cachedTokens * 2 * KVl * block.headDim * bytesPerElement(ctx.quant.kvCache);
  const indexerBytes = block.sparse === undefined ? 0 : sparseSelectionStateBytes(block.sparse, ctx, contextTokens);
  const convBytes =
    block.kvConvKernel === undefined
      ? 0
      : convStateTokens({ kernel: block.kvConvKernel }) * 2 * KVl * block.headDim * actBpe(ctx);
  return kvBytes + indexerBytes + convBytes;
}
