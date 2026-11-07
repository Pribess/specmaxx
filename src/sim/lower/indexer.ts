/**
 * Lowering of the DSA top-k indexer (DeepSeek-V3.2 "lightning indexer"), shared by GQA and MLA blocks with
 * `sparse`. The indexer is REPLICATED on every TP rank: each rank scores all keys for its own query heads'
 * selection (the indexer heads are not sharded). See docs/simulation-model.md §5 "Sparse attention indexer".
 */

import { bytesPerElement, resolveAccumulate, resolveGemmMath, type TensorFormat } from '../ir/dtype';
import type { IndexerSpec } from '../ir/model';
import type { IndexerOp, KvSelection, Op } from '../ir/ops';
import { actBpe, elementwise, hpBpe, linearGemm, stepTokens } from './common';
import type { LoweringContext } from './lower';

/** Storage format of the indexer key cache: `keyDtype` if given, else the KV-cache format. */
export function indexerKeyFormat(spec: IndexerSpec, ctx: LoweringContext): TensorFormat {
  return spec.keyDtype !== undefined ? { dtype: spec.keyDtype } : ctx.quant.kvCache;
}

/** The main attention's selection when an indexer precedes it: the top-k keys. */
export function indexerSelection(spec: IndexerSpec): KvSelection {
  return { kind: 'topk', topK: spec.topK };
}

/**
 * Indexer ops for one sparse attention block, in order:
 * 1. `idx.q` (m = T, n = heads_I·hd_I, k = qInputDim; qInputDim = h for GQA, rq for MLA with q compression);
 * 2. `idx.k` (m = T, n = hd_I, k = h); 3. `idx.k_norm` (LayerNorm over T·hd_I);
 * 4. `idx.w` head weights (m = T, n = heads_I, k = h); 5. the `indexer` scoring op.
 * `hidden` is the block's input width.
 */
export function lowerIndexer(spec: IndexerSpec, hidden: number, qInputDim: number, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  const keyCache = indexerKeyFormat(spec, ctx);
  // The scoring math runs in the wider of the key-cache and attention-math formats (same rule as GEMMs).
  const scoreMath = resolveGemmMath(keyCache.dtype, ctx.quant.attentionMath ?? ctx.quant.activations);
  const kNormElems = T * spec.headDim;
  const indexer: IndexerOp = {
    kind: 'indexer',
    name: 'idx.score',
    seqs: ctx.seqs,
    heads: spec.heads,
    headDim: spec.headDim,
    qLen: ctx.qLen,
    kvLen: ctx.kvLen,
    causal: true,
    topK: spec.topK,
    keyCache,
    keyCacheElemsPerToken: spec.headDim,
    math: scoreMath,
    accumulate: resolveAccumulate(scoreMath, ctx.quant.accumulate ?? 'fp32'),
    activationDtype: ctx.quant.activations,
  };
  // Formula: idx.q: n = heads_I·hd_I, k = qIn; idx.k: n = hd_I, k = h; idx.w: n = heads_I, k = h
  // See docs/simulation-model.md §5 "Sparse attention indexer"
  return [
    linearGemm('idx.q', { m: T, n: spec.heads * spec.headDim, k: qInputDim }, ctx),
    linearGemm('idx.k', { m: T, n: spec.headDim, k: hidden }, ctx),
    elementwise(
      'idx.k_norm',
      'layernorm',
      {
        elems: kNormElems,
        activationReadBytes: kNormElems * actBpe(ctx),
        activationWriteBytes: kNormElems * actBpe(ctx),
        weightBytes: 2 * spec.headDim * hpBpe(ctx),
      },
      ctx,
    ),
    linearGemm('idx.w', { m: T, n: spec.heads, k: hidden }, ctx),
    indexer,
  ];
}

/** Indexer key-cache bytes per sequence on one device for `contextTokens` tokens (replicated per TP rank). Unit: bytes. */
export function indexerSequenceStateBytes(spec: IndexerSpec, ctx: LoweringContext, contextTokens: number): number {
  // Formula: S_idx = ctx·hd_I·bpe(key)
  // See docs/simulation-model.md §8 "Memory fit"
  return contextTokens * spec.headDim * bytesPerElement(indexerKeyFormat(spec, ctx));
}
