/**
 * Lowering of the final norm and LM head (last pipeline stage). The LM head is vocab-parallel over the dense TP
 * group of G ranks (`ceil(vocab/G)` columns per rank; G = tp unless `denseTpGroup: 'ep'`, then G = ep).
 * **Prefill computes logits only for the last position of each sequence** (`Tout = seqs`); decode computes them
 * for every new token (`Tout = seqs·qLen`). Sampling exchanges one fp32 score + uint32 token ID per rank and
 * output position (not the full logits).
 * See docs/simulation-model.md §5 "Embedding and LM head" and "DP attention + EP".
 */

import { matrixStoredBytes, type TensorFormat } from '../ir/dtype';
import type { ModelSpec } from '../ir/model';
import type { Op } from '../ir/ops';
import {
  actBpe,
  denseOverEp,
  denseTpGroupSize,
  hpBpe,
  linearGemm,
  normOp,
  stepTokens,
} from './common';
import { embeddingFormat } from './embedding';
import type { LoweringContext } from './lower';

/** Storage format of the LM head: `quant.lmHead`, default the embedding format. */
export function lmHeadFormat(ctx: LoweringContext): TensorFormat {
  return ctx.quant.lmHead ?? embeddingFormat(ctx);
}

/** Tokens whose logits are computed: prefill `seqs` (last position only), decode `seqs·qLen`. Unit: tokens. */
export function headTokens(ctx: LoweringContext): number {
  // Formula: Tout = prefill ? seqs : T
  return ctx.phase === 'prefill' ? ctx.seqs : stepTokens(ctx);
}

/**
 * Sampling exchange for outputTokens logit rows (tokens, not all prompt rows): one fp32 score and uint32 ID per
 * TP rank, full all-gather output outputTokens·tp·8 bytes. No op for tp1. name is the IR display label.
 * Shared by target and draft heads; see docs/simulation-model.md §5 "Embedding and LM head" (audit G1).
 */
export function samplingExchangeOps(
  outputTokens: number,
  ctx: LoweringContext,
  name: string,
): Op[] {
  if (ctx.parallel.tp <= 1) return [];
  // Formula: S_output = outputTokens·tp·8 B (one score + token ID per shard).
  return [
    {
      kind: 'collective',
      name,
      collective: 'all_gather',
      group: 'tp',
      groupSize: ctx.parallel.tp,
      messageBytes: outputTokens * ctx.parallel.tp * 8,
      dtype: 'fp32',
    },
  ];
}
/**
 * Vocabulary columns of the LM head held by one device: `ceil(vocab/G)` over the dense TP group of G ranks
 * ({@link denseTpGroupSize}: tp, or ep with `denseTpGroup: 'ep'`). Count (vocabulary rows of the head matrix).
 * See docs/simulation-model.md §5 "Embedding and LM head", "DP attention + EP".
 */
export function lmHeadVocabShard(model: ModelSpec, ctx: LoweringContext): number {
  // Formula: V_l = ceil(vocab/G)
  return Math.ceil(model.vocab / denseTpGroupSize(ctx));
}

/**
 * Head ops: final norm over `Tout·h` (none when `finalNorm` is 'none'), then `lm_head` (m = Tout, n = ceil(vocab/G),
 * k = h; G = tp) and the tp sampling exchange. With the head sharded over the EP group ({@link denseOverEp}, tp = 1,
 * G = ep):
 *
 * ```
 * final_norm               elems = Tout·h                    (own rows only)
 * head.dense_all_gather    all_gather(ep, G)   S = G·Tout·h·bpe(act)
 * lm_head                  m = G·Tout, n = ceil(V/G), k = h
 * head.sample_all_to_all   all_to_all(ep, G)   S = G·Tout·8 B  (fp32 score + uint32 ID per gathered row, back to its owner)
 * ```
 */
export function lowerHead(model: ModelSpec, ctx: LoweringContext): Op[] {
  const Tout = headTokens(ctx);
  const overEp = denseOverEp(ctx);
  const G = denseTpGroupSize(ctx);
  // Formula: m_head = (overEp ? G : 1)·Tout   (rows after the group gather)
  const headRows = overEp ? G * Tout : Tout;
  // Formula: lm_head: m = m_head, n = ceil(vocab/G), k = h   (G = tp by default)
  // See docs/simulation-model.md §5 "Embedding and LM head", "DP attention + EP"
  const vocabShard = lmHeadVocabShard(model, ctx);
  // GEMM input quantized to `quant.lmHeadGemmInput` (default activations), e.g. NVFP4 W4A4 heads.
  const format = {
    weights: lmHeadFormat(ctx),
    gemmInput: ctx.quant.lmHeadGemmInput ?? ctx.quant.activations,
  };
  const lmHead = linearGemm('lm_head', { m: headRows, n: vocabShard, k: model.hidden }, ctx, {
    format,
  });
  const ops: Op[] =
    model.finalNorm === 'none'
      ? []
      : [normOp('final_norm', model.finalNorm, Tout, model.hidden, ctx)];
  if (!overEp) {
    ops.push(lmHead, ...samplingExchangeOps(Tout, ctx, 'head.sample_all_gather'));
    return ops;
  }
  // Formula: S_gather = G·Tout·h·bpe(act);  S_a2a = G·Tout·8 B
  ops.push(
    {
      kind: 'collective',
      name: 'head.dense_all_gather',
      collective: 'all_gather',
      group: 'ep',
      groupSize: G,
      messageBytes: headRows * model.hidden * actBpe(ctx),
      dtype: ctx.quant.activations,
    },
    lmHead,
    {
      kind: 'collective',
      name: 'head.sample_all_to_all',
      collective: 'all_to_all',
      group: 'ep',
      groupSize: G,
      messageBytes: headRows * 8,
      dtype: 'fp32',
    },
  );
  return ops;
}

/**
 * Resident LM-head bytes on one device: `ceil(vocab/G)·h·bpe(head)` ({@link lmHeadVocabShard}; G = tp by default)
 * plus the final-norm weights. A tied head shares the embedding table; the caller decides whether to count it once
 * (same device) or twice. Unit: bytes.
 */
export function lmHeadResidentWeightBytes(model: ModelSpec, ctx: LoweringContext): number {
  const normVectors = model.finalNorm === 'none' ? 0 : model.finalNorm === 'layernorm' ? 2 : 1;
  // Formula: W_head = ceil(vocab/G)·h·bpe(head) + (none ? 0 : layernorm ? 2 : 1)·h·bpe(hp)
  // See docs/simulation-model.md §8 "Memory fit"
  const head = matrixStoredBytes(lmHeadFormat(ctx), lmHeadVocabShard(model, ctx), model.hidden);
  return head + normVectors * model.hidden * hpBpe(ctx);
}
