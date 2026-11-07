/**
 * Lowering of multi-head latent attention (DeepSeek-V2/V3, V3.2 with DSA). TP shards the heads
 * (`Hl = heads/tp`); the down-projections (`q_a`, `kv_a`) and the latent cache are REPLICATED on every TP rank.
 * See docs/simulation-model.md §5 "MLA forms".
 *
 * Notation: rq = qLoraRank, rkv = kvLoraRank, dn = qkNopeDim, dr = qkRopeDim, dv = vHeadDim, T = seqs·qLen.
 *
 * Two mathematically equivalent forms; lowering picks one per phase (a documented rule, not a knob):
 * - **Absorbed** (decode, and prefill with `sparse`): W_UK is folded into the query and W_UV into the output, so
 *   attention runs directly on the cached latent (kvHeads 1, qk = rkv+dr, v = rkv). Per (query, key, head):
 *   2·(2·rkv + dr) FLOP, but only the latent is read.
 * - **MHA** (prefill without `sparse`): K/V are decompressed by `kv_b` for every visible token, then standard
 *   attention runs with qk = dn+dr, v = dv: 2·(dn + dr + dv) FLOP per (query, key, head), cheaper for long
 *   prefills, where compute binds.
 */

import { bytesPerElement } from '../ir/dtype';
import type { MlaAttentionBlock } from '../ir/model';
import type { AttentionOp, KvSelection, Op } from '../ir/ops';
import {
  actBpe,
  attentionMath,
  elementwise,
  linearGemm,
  normOp,
  shardHeads,
  stepTokens,
  tpAllReduce,
} from './common';
import { indexerSelection, indexerSequenceStateBytes, lowerIndexer } from './indexer';
import type { LoweringContext } from './lower';

/** Which MLA form a phase uses: absorbed for decode or sparse prefill, MHA for dense prefill. */
export function mlaForm(block: MlaAttentionBlock, ctx: LoweringContext): 'absorbed' | 'mha' {
  return ctx.phase === 'decode' || block.sparse !== undefined ? 'absorbed' : 'mha';
}

/**
 * MLA ops, in order: q path (q_a → q_a_norm → q_b, or q) → kv_a → kv_a_norm → rope → [indexer ops] →
 * absorbed (q_absorb → attention → v_absorb) or MHA (kv_b → attention) → o_proj → [TP all-reduce].
 */
export function lowerMla(block: MlaAttentionBlock, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  const h = block.hidden;
  const rq = block.qLoraRank;
  const rkv = block.kvLoraRank;
  const dn = block.qkNopeDim;
  const dr = block.qkRopeDim;
  const dv = block.vHeadDim;
  const Hl = shardHeads(block.heads, ctx.parallel.tp, `${block.name} heads`);
  const act = actBpe(ctx);
  const ops: Op[] = [];

  // Query path. Formula: rq > 0: q_a (n = rq, k = h, replicated) → rmsnorm(T·rq) → q_b (n = Hl·(dn+dr), k = rq);
  //                      else  q (n = Hl·(dn+dr), k = h)
  // See docs/simulation-model.md §5 "MLA forms"
  if (rq > 0) {
    ops.push(linearGemm('mla.q_a', { m: T, n: rq, k: h }, ctx));
    ops.push(normOp('mla.q_a_norm', 'rmsnorm', T, rq, ctx));
    ops.push(linearGemm('mla.q_b', { m: T, n: Hl * (dn + dr), k: rq }, ctx));
  } else {
    ops.push(linearGemm('mla.q', { m: T, n: Hl * (dn + dr), k: h }, ctx));
  }

  // Latent path. Formula: kv_a (n = rkv + dr, k = h, replicated) → rmsnorm(T·rkv); rope over T·(Hl·dr + dr)
  ops.push(linearGemm('mla.kv_a', { m: T, n: rkv + dr, k: h }, ctx));
  ops.push(normOp('mla.kv_a_norm', 'rmsnorm', T, rkv, ctx));
  const ropeElems = T * (Hl * dr + dr);
  ops.push(
    elementwise(
      'mla.rope',
      'rope',
      { elems: ropeElems, activationReadBytes: ropeElems * act, activationWriteBytes: ropeElems * act },
      ctx,
    ),
  );

  let selection: KvSelection = { kind: 'full' };
  if (block.sparse !== undefined) {
    ops.push(...lowerIndexer(block.sparse, h, rq > 0 ? rq : h, ctx));
    selection = indexerSelection(block.sparse);
  }

  // The latent cache: rkv + dr elements per token, shared by all heads, written for every new token.
  const latent = { format: ctx.quant.kvCache, elemsPerToken: rkv + dr };
  const common = {
    kind: 'attention' as const,
    name: 'attn.core',
    seqs: ctx.seqs,
    heads: Hl,
    qLen: ctx.qLen,
    kvLen: ctx.kvLen,
    causal: true,
    kvSelection: selection,
    kvWrite: latent,
    ...attentionMath(ctx),
    activationDtype: ctx.quant.activations,
  };

  const form = mlaForm(block, ctx);
  switch (form) {
    case 'absorbed': {
      // Formula: q_absorb: batch = Hl, m = T, n = rkv, k = dn; attention: kvHeads 1, qk = rkv + dr, v = rkv,
      //          kvRead = kvWrite = {kvCache, rkv + dr}; v_absorb: batch = Hl, m = T, n = dv, k = rkv
      // See docs/simulation-model.md §5 "MLA forms"
      ops.push(linearGemm('mla.q_absorb', { m: T, n: rkv, k: dn }, ctx, { batch: Hl }));
      const attention: AttentionOp = { ...common, kvHeads: 1, qkHeadDim: rkv + dr, vHeadDim: rkv, kvRead: latent };
      ops.push(attention);
      ops.push(linearGemm('mla.v_absorb', { m: T, n: dv, k: rkv }, ctx, { batch: Hl }));
      break;
    }
    case 'mha': {
      // Formula: kv_b: m = seqs·kvLen, n = Hl·(dn + dv), k = rkv (reads the latent: inputDtype = kvCache dtype);
      //          attention: kvHeads = Hl, qk = dn + dr, v = dv, kvRead = {act, Hl·(dn + dv) + dr}, kvWrite = latent
      // See docs/simulation-model.md §5 "MLA forms"
      ops.push(
        linearGemm('mla.kv_b', { m: ctx.seqs * ctx.kvLen, n: Hl * (dn + dv), k: rkv }, ctx, {
          inputDtype: ctx.quant.kvCache.dtype,
        }),
      );
      const decompressed = { format: { dtype: ctx.quant.activations }, elemsPerToken: Hl * (dn + dv) + dr };
      const attention: AttentionOp = { ...common, kvHeads: Hl, qkHeadDim: dn + dr, vHeadDim: dv, kvRead: decompressed };
      ops.push(attention);
      break;
    }
    default: {
      const _never: never = form;
      throw new Error(`unknown MLA form ${String(_never)}`);
    }
  }

  // Formula: o_proj: m = T, n = h, k = Hl·dv
  ops.push(linearGemm('mla.o_proj', { m: T, n: h, k: Hl * dv }, ctx));
  ops.push(...tpAllReduce('attn.tp_all_reduce', T, h, ctx));
  return ops;
}

/**
 * Per-sequence latent (+ indexer key) cache bytes of an MLA block on one device (replicated per TP rank).
 * Unit: bytes.
 */
export function mlaSequenceStateBytes(block: MlaAttentionBlock, ctx: LoweringContext, contextTokens: number): number {
  // Formula: S = ctx·(rkv + dr)·bpe(kv) [+ ctx·hd_I·bpe(key) if sparse]
  // See docs/simulation-model.md §8 "Memory fit"
  const latentBytes = contextTokens * (block.kvLoraRank + block.qkRopeDim) * bytesPerElement(ctx.quant.kvCache);
  const indexerBytes = block.sparse === undefined ? 0 : indexerSequenceStateBytes(block.sparse, ctx, contextTokens);
  return latentBytes + indexerBytes;
}
