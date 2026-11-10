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
import {
  dcpActive,
  dcpDegree,
  dcpGatherQ,
  dcpKvTokens,
  dcpReduceOut,
  dcpShardAttention,
  dcpShardsBlock,
} from './dcp';
import { indexerSelection, lowerSparseSelection, sparseSelectionStateBytes } from './indexer';
import { kvFetchOps } from './kvOffload';
import type { LoweringContext } from './lower';

/** Which MLA form a phase uses: absorbed for decode or sparse prefill, MHA for dense prefill. */
export function mlaForm(block: MlaAttentionBlock, ctx: LoweringContext): 'absorbed' | 'mha' {
  return ctx.phase === 'decode' || block.sparse !== undefined ? 'absorbed' : 'mha';
}

/**
 * MLA ops, in order: q path (q_a → q_a_norm → q_b, or q) → [g_proj] → [own latent: kv_a → kv_a_norm] → [rope] →
 * [indexer ops] → absorbed (q_absorb → [KV fetch] → [DCP Q all-gather] → attention → [DCP output reduce-scatter] →
 * v_absorb) or MHA ([KV fetch] → kv_b → attention) → [sigmoid_mul gate] → o_proj → [TP all-reduce]. The KV fetch
 * appears only with KV-cache offload when past latents are off the device (lower/kvOffload.ts). A layer with
 * `kvCacheFromLayer` reads an earlier layer's latent: no kv_a, no kv_a_norm, no k-rope, no cache write. `qkRopeDim` 0
 * (NoPE) emits no rope op.
 * With `outputGate` (Kimi K3) the gate projection `g_proj` reads the layer input and its sigmoid multiplies the
 * attention output before `o_proj`. With decode context parallelism (`parallel.dcp` d > 1, decode) the attention
 * core runs the d·Hl gathered heads over this rank's 1/d of the latent tokens (lower/dcp.ts).
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
  // Throws for a sparse MLA block with d > 1 (any phase); true when this decode step shards the latent over tokens.
  const dcpSharded = dcpShardsBlock(block, ctx) && dcpActive(ctx);

  // Query path. Formula: rq > 0: q_a (n = rq, k = h, replicated) → rmsnorm(T·rq) → q_b (n = Hl·(dn+dr), k = rq);
  //                      else  q (n = Hl·(dn+dr), k = h)
  // See docs/simulation-model.md §5 "MLA forms"
  if (rq > 0) {
    ops.push(linearGemm('mla.q_a', { m: T, n: rq, k: h }, ctx, { module: 'attention' }));
    ops.push(normOp('mla.q_a_norm', 'rmsnorm', T, rq, ctx));
    ops.push(
      linearGemm('mla.q_b', { m: T, n: Hl * (dn + dr), k: rq }, ctx, { module: 'attention' }),
    );
  } else {
    ops.push(linearGemm('mla.q', { m: T, n: Hl * (dn + dr), k: h }, ctx, { module: 'attention' }));
  }
  if (block.outputGate === true) {
    // Formula: g_proj: m = T, n = Hl·dv, k = h (the gate, sharded with the heads)
    // See docs/simulation-model.md §5 "MLA output gate"
    ops.push(linearGemm('mla.g_proj', { m: T, n: Hl * dv, k: h }, ctx, { module: 'attention' }));
  }

  // Latent path. Formula: [own latent] kv_a (n = rkv + dr, k = h, replicated) → rmsnorm(T·rkv);
  //               rope over T·(Hl·dr + [own latent] dr)
  // See docs/simulation-model.md §5 "KV sharing"
  const ownsLatent = block.kvCacheFromLayer === undefined;
  if (ownsLatent) {
    ops.push(
      linearGemm('mla.kv_a', { m: T, n: rkv + dr, k: h }, ctx, { module: 'attention_kv_down' }),
    );
    ops.push(normOp('mla.kv_a_norm', 'rmsnorm', T, rkv, ctx));
  }
  const ropeElems = T * (Hl * dr + (ownsLatent ? dr : 0));
  if (ropeElems > 0) {
    ops.push(
      elementwise(
        'mla.rope',
        'rope',
        {
          elems: ropeElems,
          activationReadBytes: ropeElems * act,
          activationWriteBytes: ropeElems * act,
        },
        ctx,
      ),
    );
  }

  let selection: KvSelection = { kind: 'full' };
  if (block.sparse !== undefined) {
    ops.push(...lowerSparseSelection(block.sparse, h, rq > 0 ? rq : h, ctx));
    selection = indexerSelection(block.sparse);
  }

  // The latent cache: rkv + dr elements per token, shared by all heads, written for every new token (by its owner).
  const latent = { format: ctx.quant.kvCache, elemsPerToken: rkv + dr };
  const latentWrite = { format: ctx.quant.kvCache, elemsPerToken: ownsLatent ? rkv + dr : 0 };
  const common = {
    kind: 'attention' as const,
    name: 'attn.core',
    seqs: ctx.seqs,
    heads: Hl,
    qLen: ctx.qLen,
    kvLen: ctx.kvLen,
    causal: true,
    kvSelection: selection,
    kvWrite: latentWrite,
    ...attentionMath(ctx),
    activationDtype: ctx.quant.activations,
  };

  const form = mlaForm(block, ctx);
  switch (form) {
    case 'absorbed': {
      // Formula: q_absorb: batch = Hl, m = T, n = rkv, k = dn; attention: kvHeads 1, qk = rkv + dr, v = rkv,
      //          kvRead = kvWrite = {kvCache, rkv + dr}; v_absorb: batch = Hl, m = T, n = dv, k = rkv
      // See docs/simulation-model.md §5 "MLA forms"
      // The stored kv_b matrix is (Hl·(dn + dv)) × rkv with Hl·(dn + dv) row and rkv column side vectors (EXL3).
      // q_absorb owns the Hl·dn rows of its k_nope part and the rkv column vector; v_absorb the Hl·dv rows of
      // its v part, so both forms charge the same stored bytes.
      ops.push(
        linearGemm('mla.q_absorb', { m: T, n: rkv, k: dn }, ctx, {
          module: 'attention_kv_up',
          batch: Hl,
          sideVectorElems: Hl * dn + rkv,
        }),
      );
      const attention: AttentionOp = {
        ...common,
        kvHeads: 1,
        qkHeadDim: rkv + dr,
        vHeadDim: rkv,
        kvRead: latent,
      };
      if (dcpSharded) {
        // Formula: S_q = T·(Hl·d)·(rkv + dr)·bpe(act); attention heads Hl·d over 1/d of the latent tokens;
        //          S_o = T·(Hl·d)·(rkv·reduceBpe + 4)   (the output is still latent; v_absorb runs after the merge)
        // See docs/simulation-model.md §5 "Decode context parallelism (DCP)"
        const sharded = dcpShardAttention(attention, ctx);
        // KV-cache offload: fetch this rank's non-resident latent tokens first (docs §12 "KV cache off-device").
        ops.push(...kvFetchOps(block, ctx, sharded));
        ops.push(dcpGatherQ(T, Hl, rkv + dr, ctx));
        ops.push(sharded);
        ops.push(dcpReduceOut(T, Hl, rkv, ctx));
      } else {
        // KV-cache offload: fetch the non-resident latent tokens first (docs §12 "KV cache off-device").
        ops.push(...kvFetchOps(block, ctx, attention));
        ops.push(attention);
      }
      ops.push(
        linearGemm('mla.v_absorb', { m: T, n: dv, k: rkv }, ctx, {
          module: 'attention_kv_up',
          batch: Hl,
          sideVectorElems: Hl * dv,
        }),
      );
      break;
    }
    case 'mha': {
      // Formula: kv_b: m = seqs·kvLen, n = Hl·(dn + dv), k = rkv (reads the latent: inputDtype = kvCache dtype);
      //          attention: kvHeads = Hl, qk = dn + dr, v = dv, kvRead = {act, Hl·(dn + dv) + dr}, kvWrite = latent
      // See docs/simulation-model.md §5 "MLA forms"
      const decompressed = {
        format: { dtype: ctx.quant.activations },
        elemsPerToken: Hl * (dn + dv) + dr,
      };
      const attention: AttentionOp = {
        ...common,
        kvHeads: Hl,
        qkHeadDim: dn + dr,
        vHeadDim: dv,
        kvRead: decompressed,
      };
      // KV-cache offload: kv_b reads every visible latent, so the non-resident ones are fetched before it
      // (docs §12 "KV cache off-device").
      ops.push(...kvFetchOps(block, ctx, attention));
      ops.push(
        linearGemm('mla.kv_b', { m: ctx.seqs * ctx.kvLen, n: Hl * (dn + dv), k: rkv }, ctx, {
          module: 'attention_kv_up',
          inputDtype: ctx.quant.kvCache.dtype,
        }),
      );
      ops.push(attention);
      break;
    }
    default: {
      const _never: never = form;
      throw new Error(`unknown MLA form ${String(_never)}`);
    }
  }

  if (block.outputGate === true) {
    const gateElems = T * Hl * dv;
    // Formula: sigmoid_mul over T·Hl·dv: read 2·elems (out, gate), write elems
    // See docs/simulation-model.md §5 "MLA output gate"
    ops.push(
      elementwise(
        'attn.output_gate',
        'sigmoid_mul',
        {
          elems: gateElems,
          activationReadBytes: 2 * gateElems * act,
          activationWriteBytes: gateElems * act,
        },
        ctx,
      ),
    );
  }

  // Formula: o_proj: m = T, n = h, k = Hl·dv
  ops.push(linearGemm('mla.o_proj', { m: T, n: h, k: Hl * dv }, ctx, { module: 'attention' }));
  ops.push(...tpAllReduce('attn.tp_all_reduce', T, h, ctx));
  return ops;
}

/**
 * Per-sequence latent (+ owned indexer key) cache bytes of an MLA block on one device (replicated per TP rank, or
 * with decode context parallelism d > 1 this rank's `ceil(ctx/d)` tokens of it). A layer reading another layer's
 * latent (`kvCacheFromLayer`) holds none of it. Unit: bytes.
 */
export function mlaSequenceStateBytes(
  block: MlaAttentionBlock,
  ctx: LoweringContext,
  contextTokens: number,
): number {
  // Formula: S = [own latent] N_rank·(rkv + dr)·bpe(kv) [+ ceil(ctx/p)·hd_I·bpe(key) if it owns indexer keys],
  //          N_rank = ceil(ctx/d)   (= ctx without DCP)
  // See docs/simulation-model.md §8 "Memory fit", §5 "Decode context parallelism (DCP)"
  const latentTokens = dcpShardsBlock(block, ctx)
    ? dcpKvTokens(contextTokens, dcpDegree(ctx))
    : contextTokens;
  const latentElems = block.kvCacheFromLayer === undefined ? block.kvLoraRank + block.qkRopeDim : 0;
  const latentBytes = latentTokens * latentElems * bytesPerElement(ctx.quant.kvCache);
  const indexerBytes =
    block.sparse === undefined ? 0 : sparseSelectionStateBytes(block.sparse, ctx, contextTokens);
  return latentBytes + indexerBytes;
}
