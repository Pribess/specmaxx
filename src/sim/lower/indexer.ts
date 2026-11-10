/**
 * Lowering of sparse-attention selections: a top-k indexer (DeepSeek-V3.2 DSA "lightning indexer" and its
 * relatives: GLM k-pool, Qwen QSA, MiniMax MSA, DeepSeek-V4 / V4.1 CSA), or a `'reuse'` of an earlier layer's top-k,
 * which emits no ops. The indexer is REPLICATED on every TP rank: each rank scores all keys for its own query heads'
 * selection (the indexer heads are not sharded). See docs/simulation-model.md §5 "Sparse attention indexer".
 *
 * Notation: H_I indexer heads, hd_I indexer head dim, p = keyPoolTokens, T = seqs·qLen.
 */

import {
  bytesPerElement,
  resolveAccumulate,
  resolveGemmMath,
  type TensorFormat,
} from '../ir/dtype';
import type { IndexerSpec, SparseSelection } from '../ir/model';
import type { IndexerOp, KvSelection, Op } from '../ir/ops';
import { actBpe, elementwise, hpBpe, linearGemm, stepTokens } from './common';
import type { LoweringContext } from './lower';

/** Storage format of the indexer key cache: `quant.indexerKeyCache`, else the KV-cache format. */
export function indexerKeyFormat(ctx: LoweringContext): TensorFormat {
  return ctx.quant.indexerKeyCache ?? ctx.quant.kvCache;
}

/** Keys the main attention reads per query under a sparse selection, in the attention's key units. */
export function sparseTopK(sel: SparseSelection): number {
  return sel.kind === 'indexer' ? sel.indexer.topK : sel.topK;
}

/** The main attention's selection when a sparse selection precedes it: the top-k keys. */
export function indexerSelection(sel: SparseSelection): KvSelection {
  return { kind: 'topk', topK: sparseTopK(sel) };
}

/**
 * Indexer ops for one sparse attention block (none for a `'reuse'` selection), in order:
 * 1. `idx.q` (m = T, n = H_I·hd_I, k = qInputDim; qInputDim = h for GQA, rq for MLA with q compression and CSA);
 * 2. if the layer owns its keys: `idx.k` (n = coff·hd_I, k = keyInputDim ?? h) and the key LayerNorm over T·hd_I;
 * 3. with `keyPoolGate`: `idx.pool_gate` (n = coff·hd_I, k = h) and `idx.kv_pool` (T·coff·hd_I → T·hd_I/p);
 *    coff = 2 with `keyPoolOverlap` (DeepSeek-V4 overlapping key windows), else 1;
 * 4. with `headWeights`: `idx.w` (n = H_I, k = h);
 * 5. the `indexer` scoring op over the visible keys (each pooling p tokens), capped at `maxScoredKeys`.
 * `hidden` is the block's input width; `attentionKeyTokens` is how many tokens one key of the MAIN attention
 * stands for (1 for gqa/mla, r for csa entries), used to turn the attention's `topK` into indexer keys.
 * Documented approximation: the key GEMM runs on all T tokens even when keys come from pooled latents.
 */
export function lowerSparseSelection(
  sel: SparseSelection,
  hidden: number,
  qInputDim: number,
  ctx: LoweringContext,
  attentionKeyTokens = 1,
): Op[] {
  if (sel.kind === 'reuse') return [];
  const spec = sel.indexer;
  const T = stepTokens(ctx);
  const p = spec.keyPoolTokens ?? 1;
  const hdI = spec.headDim;
  const ownsKeys = spec.ownsKeys !== false;
  // coff = 2 with overlapping key pools (each key pools 2p tokens from two hd_I-wide halves), else 1.
  const coff = spec.keyPoolOverlap === true ? 2 : 1;
  const keyProjDim = coff * hdI;
  const act = actBpe(ctx);
  const ops: Op[] = [];

  // Formula: idx.q: n = H_I·hd_I, k = qIn
  // See docs/simulation-model.md §5 "Sparse attention indexer"
  ops.push(
    linearGemm('idx.q', { m: T, n: spec.heads * hdI, k: qInputDim }, ctx, { module: 'indexer' }),
  );
  if (ownsKeys) {
    // Formula: idx.k: n = coff·hd_I, k = kIn; key LayerNorm over T·hd_I
    ops.push(
      linearGemm('idx.k', { m: T, n: keyProjDim, k: spec.keyInputDim ?? hidden }, ctx, {
        module: 'indexer_key',
      }),
    );
    const kNormElems = T * hdI;
    ops.push(
      elementwise(
        'idx.k_norm',
        'layernorm',
        {
          elems: kNormElems,
          activationReadBytes: kNormElems * act,
          activationWriteBytes: kNormElems * act,
          weightBytes: 2 * hdI * hpBpe(ctx),
        },
        ctx,
      ),
    );
    if (spec.keyPoolGate === true) {
      // Formula: idx.pool_gate: n = coff·hd_I, k = h; kv_pool over T·coff·hd_I: read 2·T·coff·hd_I (keys, gate
      //          logits), write T·hd_I/p, APE p·coff·hd_I·bpe(hp)
      ops.push(
        linearGemm('idx.pool_gate', { m: T, n: keyProjDim, k: hidden }, ctx, {
          module: 'indexer_key',
        }),
      );
      const poolElems = T * keyProjDim;
      const keyElems = T * hdI;
      ops.push(
        elementwise(
          'idx.kv_pool',
          'kv_pool',
          {
            elems: poolElems,
            activationReadBytes: 2 * poolElems * act,
            activationWriteBytes: (keyElems / p) * act,
            weightBytes: p * keyProjDim * hpBpe(ctx),
          },
          ctx,
        ),
      );
    }
  }
  if (spec.headWeights !== false) {
    // Formula: idx.w: n = H_I, k = h
    ops.push(
      linearGemm('idx.w', { m: T, n: spec.heads, k: hidden }, ctx, {
        module: 'indexer_head_weights',
      }),
    );
  }

  const keyCache = indexerKeyFormat(ctx);
  // The scoring math runs in the wider of the key-cache and attention-math formats (same rule as GEMMs).
  const scoreMath = resolveGemmMath(
    keyCache.dtype,
    ctx.quant.attentionMath ?? ctx.quant.activations,
  );
  // Formula: topK_I = ceil(topK·attentionKeyTokens/p)   (indexer keys kept per query)
  const indexer: IndexerOp = {
    kind: 'indexer',
    name: 'idx.score',
    seqs: ctx.seqs,
    heads: spec.heads,
    headDim: hdI,
    qLen: ctx.qLen,
    kvLen: ctx.kvLen,
    causal: true,
    topK: Math.ceil((spec.topK * attentionKeyTokens) / p),
    keyCache,
    keyElemsPerKey: hdI,
    keyPoolTokens: p,
    ...(spec.maxScoredKeys !== undefined ? { maxScoredKeys: spec.maxScoredKeys } : {}),
    keyWriteElemsPerToken: ownsKeys ? hdI / p : 0,
    headWeights: spec.headWeights !== false,
    math: scoreMath,
    accumulate: resolveAccumulate(scoreMath, ctx.quant.accumulate ?? 'fp32'),
    activationDtype: ctx.quant.activations,
  };
  ops.push(indexer);
  return ops;
}

/**
 * Indexer key-cache bytes per sequence on one device for `contextTokens` tokens (replicated per TP rank): the keys
 * this layer owns, `ceil(ctx/p)` of them; 0 for a `'reuse'` selection or an indexer reading another layer's keys.
 * Unit: bytes.
 */
export function sparseSelectionStateBytes(
  sel: SparseSelection,
  ctx: LoweringContext,
  contextTokens: number,
): number {
  if (sel.kind === 'reuse') return 0;
  return indexerKeyStateBytes(sel.indexer, ctx, contextTokens);
}

/** Owned indexer key-cache bytes per sequence: `ceil(ctx/p)·hd_I·bpe(key)`, 0 if `ownsKeys` is false. Unit: bytes. */
export function indexerKeyStateBytes(
  spec: IndexerSpec,
  ctx: LoweringContext,
  contextTokens: number,
): number {
  if (spec.ownsKeys === false) return 0;
  // Formula: S_idx = ceil(ctx/p)·hd_I·bpe(key)
  // See docs/simulation-model.md §8 "Memory fit"
  return (
    Math.ceil(contextTokens / (spec.keyPoolTokens ?? 1)) *
    spec.headDim *
    bytesPerElement(indexerKeyFormat(ctx))
  );
}
