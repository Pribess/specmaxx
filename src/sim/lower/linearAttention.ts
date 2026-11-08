/**
 * Lowering of linear-attention / SSM mixers (GatedDeltaNet, Kimi Delta Attention, Mamba2, plain linear attention).
 * TP shards q/k heads with the KV-head replication rule (`qkL`, see common.ts) and value (state) heads evenly
 * (`vL = vHeads / tp`, which must divide): each value head's output feeds the row-parallel out_proj, so a
 * replicated value head would be summed tp/vHeads times by the TP all-reduce.
 * See docs/simulation-model.md §5 "Linear-attention sharding" and §3 "Recurrent".
 *
 * Notation: dk = keyDim, dv = valueDim, K = convKernel, C = 2·qkL·dk + vL·dv conv channels per device, r = gateRank.
 */

import { bytesPerElement } from '../ir/dtype';
import type { LinearAttentionBlock } from '../ir/model';
import type { Op, RecurrentOp } from '../ir/ops';
import {
  actBpe,
  attentionMath,
  elementwise,
  hpBpe,
  linearGemm,
  normOp,
  shardHeads,
  shardKvHeads,
  stepTokens,
  tpAllReduce,
} from './common';
import type { LoweringContext } from './lower';

/** Tokens per chunk of the chunked (prefill) form; shorter steps use one chunk of qLen. Unit: tokens. */
export const RECURRENT_CHUNK_TOKENS = 64;

/**
 * FLOP per state element per token `c_s` (documented approximations, docs §3 "Recurrent"):
 * chunked form 4 for every variant (inter-chunk output 2 + state update 2); step form: GatedDeltaNet and KDA 7
 * (Sᵀk 2 + delta outer product 2 + decay 1 + output 2; KDA's per-channel decay is still 1 FLOP per state element),
 * Mamba2 5, linear 4. Unit: FLOP/(element·token).
 */
export function stateFlopsPerElemPerToken(
  variant: LinearAttentionBlock['variant'],
  mode: RecurrentOp['mode'],
): number {
  if (mode === 'chunked') return 4;
  switch (variant) {
    case 'gated_delta_net':
    case 'kda':
      return 7;
    case 'mamba2':
      return 5;
    case 'linear':
      return 4;
    default: {
      const _never: never = variant;
      throw new Error(`unknown recurrent variant ${String(_never)}`);
    }
  }
}

/** Per-device head counts and conv channels of a linear-attention block. */
interface LinearShard {
  /** q/k heads per device. Count. */
  qkL: number;
  /** Value (state) heads per device. Count. */
  vL: number;
  /** Conv channels per device `C = 2·qkL·dk + vL·dv`. Unit: elements. */
  convChannels: number;
}

/**
 * Shards a linear-attention block over tp. q/k heads follow the KV-head replication rule (a replicated q/k head
 * is an input shared by the local value heads, like a GQA KV head). Value heads must split evenly
 * (`vHeads % tp == 0`, else this throws): replicating them would duplicate their out_proj contributions in the
 * all-reduce. See docs/simulation-model.md §5 "Linear-attention sharding".
 */
function linearShard(block: LinearAttentionBlock, tp: number): LinearShard {
  // Formula: qkL = qkHeads ≥ tp ? qkHeads/tp : 1;  vL = vHeads/tp (must divide)
  const qkL = shardKvHeads(block.qkHeads, tp, `${block.name} qkHeads`);
  const vL = shardHeads(block.vHeads, tp, `${block.name} vHeads`);
  // Formula: C = 2·qkL·dk + vL·dv
  return { qkL, vL, convChannels: 2 * qkL * block.keyDim + vL * block.valueDim };
}

/**
 * Linear-attention ops, in order: input projections → [causal_conv1d] → recurrent → gated_rmsnorm (or rmsnorm) →
 * out_proj → [TP all-reduce of T·h]. Input projections: `lin.in_proj` (one fused GEMM), or for KDA `lin.qkv`
 * (q, k, v and β) plus the low-rank gates `lin.f_a → lin.f_b` (forget) and `lin.g_a → lin.g_b` (output); the
 * rank-r down-projections are replicated on every TP rank.
 */
export function lowerLinearAttention(block: LinearAttentionBlock, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  const h = block.hidden;
  const dk = block.keyDim;
  const dv = block.valueDim;
  const K = block.convKernel;
  const { qkL, vL, convChannels: C } = linearShard(block, ctx.parallel.tp);
  const act = actBpe(ctx);
  // A prior state exists unless this step starts the sequence (first prefill chunk).
  const hasPriorState = ctx.kvLen > ctx.qLen;
  const ops: Op[] = [];

  if (block.variant === 'kda') {
    const r = block.gateRank ?? 0;
    // Formula: lin.qkv: n = 2·qkL·dk + vL·dv + vL (β), k = h; f_a: n = r, k = h; f_b: n = vL·dk, k = r;
    //          g_a: n = r, k = h; g_b: n = vL·dv, k = r
    // See docs/simulation-model.md §5 "KDA"
    const m = { module: 'linear_attention' } as const;
    ops.push(linearGemm('lin.qkv', { m: T, n: 2 * qkL * dk + vL * dv + vL, k: h }, ctx, { ...m, outParts: [qkL * dk, qkL * dk, vL * dv, vL] }));
    ops.push(linearGemm('lin.f_a', { m: T, n: r, k: h }, ctx, m));
    ops.push(linearGemm('lin.f_b', { m: T, n: vL * dk, k: r }, ctx, m));
    ops.push(linearGemm('lin.g_a', { m: T, n: r, k: h }, ctx, m));
    ops.push(linearGemm('lin.g_b', { m: T, n: vL * dv, k: r }, ctx, m));
  } else {
    // Formula: in_proj: m = T, n = 2·qkL·dk + vL·dv·(gate ? 2 : 1) + 2·vL, k = h
    // See docs/simulation-model.md §5 "Linear-attention sharding"
    const inN = 2 * qkL * dk + vL * dv * (block.outputGate ? 2 : 1) + 2 * vL;
    const inParts = [qkL * dk, qkL * dk, vL * dv, ...(block.outputGate ? [vL * dv] : []), vL, vL];
    ops.push(linearGemm('lin.in_proj', { m: T, n: inN, k: h }, ctx, { module: 'linear_attention', outParts: inParts }));
  }

  if (K > 0) {
    const elems = T * C;
    const convStateBytes = ctx.seqs * C * (K - 1) * act;
    // Formula: conv1d over T·C: flopsPerElem = 2·K; rolling state seqs·C·(K−1)·bpe(act) read (if prior) and written;
    //          taps C·K·bpe(hp)
    // See docs/simulation-model.md §5 "Linear-attention sharding"
    ops.push(
      elementwise(
        'lin.conv1d',
        'causal_conv1d',
        {
          elems,
          flopsPerElem: 2 * K,
          activationReadBytes: elems * act,
          activationWriteBytes: elems * act,
          weightBytes: C * K * hpBpe(ctx),
          stateReadBytes: hasPriorState ? convStateBytes : 0,
          stateWriteBytes: convStateBytes,
        },
        ctx,
      ),
    );
  }

  const mode: RecurrentOp['mode'] = ctx.phase === 'prefill' ? 'chunked' : 'step';
  const recurrent: RecurrentOp = {
    kind: 'recurrent',
    name: 'lin.recurrent',
    variant: block.variant,
    seqs: ctx.seqs,
    qLen: ctx.qLen,
    heads: vL,
    qkHeads: qkL,
    keyDim: dk,
    valueDim: dv,
    mode,
    chunkLen: mode === 'chunked' ? Math.min(RECURRENT_CHUNK_TOKENS, ctx.qLen) : 1,
    stateFlopsPerElemPerToken: stateFlopsPerElemPerToken(block.variant, mode),
    // Formula: gate elems per token per head: KDA dk + 1 (per-channel decay + β), else 2
    gateElemsPerTokenPerHead: block.variant === 'kda' ? dk + 1 : 2,
    hasPriorState,
    stateDtype: ctx.quant.recurrentState ?? 'fp32',
    activationDtype: ctx.quant.activations,
    ...attentionMath(ctx),
  };
  ops.push(recurrent);

  const outElems = T * vL * dv;
  if (block.outputGate) {
    // Formula: gated_rmsnorm over T·vL·dv: read 2·elems (o, z), write elems, gamma dv·bpe(hp)
    ops.push(
      elementwise(
        'lin.out_norm',
        'gated_rmsnorm',
        {
          elems: outElems,
          activationReadBytes: 2 * outElems * act,
          activationWriteBytes: outElems * act,
          weightBytes: dv * hpBpe(ctx),
        },
        ctx,
      ),
    );
  } else {
    // Per-head norm of width dv (gamma shared across heads): T·vL rows of dv.
    ops.push(normOp('lin.out_norm', 'rmsnorm', T * vL, dv, ctx));
  }

  // Formula: out_proj: m = T, n = h, k = vL·dv
  ops.push(linearGemm('lin.out_proj', { m: T, n: h, k: vL * dv }, ctx, { module: 'linear_attention' }));
  ops.push(...tpAllReduce('lin.tp_all_reduce', T, h, ctx));
  return ops;
}

/**
 * Fixed per-sequence state of a linear-attention block on one device, independent of context length:
 * the recurrent state plus the conv1d rolling state. Unit: bytes.
 */
export function linearAttentionSequenceStateBytes(block: LinearAttentionBlock, ctx: LoweringContext): number {
  const { vL, convChannels: C } = linearShard(block, ctx.parallel.tp);
  const stateBpe = bytesPerElement({ dtype: ctx.quant.recurrentState ?? 'fp32' });
  const convTaps = Math.max(block.convKernel - 1, 0);
  // Formula: S = vL·dk·dv·bpe(state) + C·(K−1)·bpe(act)
  // See docs/simulation-model.md §8 "Memory fit"
  return vL * block.keyDim * block.valueDim * stateBpe + C * convTaps * actBpe(ctx);
}
