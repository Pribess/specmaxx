/**
 * Lowering of the final norm and LM head (last pipeline stage). The LM head is vocab-parallel over TP
 * (`ceil(vocab/tp)` columns per rank). **Prefill computes logits only for the last position of each sequence**
 * (`Tout = seqs`); decode computes them for every new token (`Tout = seqs·qLen`). Sampling
 * exchanges one fp32 score + uint32 token ID per rank and output position (not the full logits).
 * See docs/simulation-model.md §5 "Embedding and LM head".
 */

import { matrixStoredBytes, type TensorFormat } from '../ir/dtype';
import type { ModelSpec } from '../ir/model';
import type { Op } from '../ir/ops';
import { hpBpe, linearGemm, normOp, stepTokens } from './common';
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
/** Head ops: final norm over `Tout·h` (none when `finalNorm` is 'none'), then `lm_head` (m = Tout, n = ceil(vocab/tp), k = h). */
export function lowerHead(model: ModelSpec, ctx: LoweringContext): Op[] {
  const Tout = headTokens(ctx);
  // Formula: lm_head: m = Tout, n = ceil(vocab/tp), k = h
  // See docs/simulation-model.md §5 "Embedding and LM head"
  const vocabShard = Math.ceil(model.vocab / ctx.parallel.tp);
  // GEMM input quantized to `quant.lmHeadGemmInput` (default activations), e.g. NVFP4 W4A4 heads.
  const format = {
    weights: lmHeadFormat(ctx),
    gemmInput: ctx.quant.lmHeadGemmInput ?? ctx.quant.activations,
  };
  const lmHead = linearGemm('lm_head', { m: Tout, n: vocabShard, k: model.hidden }, ctx, {
    format,
  });
  const ops: Op[] =
    model.finalNorm === 'none'
      ? [lmHead]
      : [normOp('final_norm', model.finalNorm, Tout, model.hidden, ctx), lmHead];
  ops.push(...samplingExchangeOps(Tout, ctx, 'head.sample_all_gather'));
  return ops;
}

/**
 * Resident LM-head bytes on one device: `ceil(vocab/tp)·h·bpe(head)` plus the final-norm weights. A tied head
 * shares the embedding table; the caller decides whether to count it once (same device) or twice. Unit: bytes.
 */
export function lmHeadResidentWeightBytes(model: ModelSpec, ctx: LoweringContext): number {
  const normVectors = model.finalNorm === 'none' ? 0 : model.finalNorm === 'layernorm' ? 2 : 1;
  // Formula: W_head = ceil(vocab/tp)·h·bpe(head) + (none ? 0 : layernorm ? 2 : 1)·h·bpe(hp)
  // See docs/simulation-model.md §8 "Memory fit"
  const head = matrixStoredBytes(
    lmHeadFormat(ctx),
    Math.ceil(model.vocab / ctx.parallel.tp),
    model.hidden,
  );
  return head + normVectors * model.hidden * hpBpe(ctx);
}
