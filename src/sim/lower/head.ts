/**
 * Lowering of the final norm and LM head (last pipeline stage). The LM head is vocab-parallel over TP
 * (`ceil(vocab/tp)` columns per rank). **Prefill computes logits only for the last position of each sequence**
 * (`Tout = seqs`); decode computes them for every new token (`Tout = seqs·qLen`). No logits gather: sampling
 * across shards exchanges O(Tout·tp) scalars, treated as free (docs §9).
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

/** Head ops: final norm over `Tout·h` (none when `finalNorm` is 'none'), then `lm_head` (m = Tout, n = ceil(vocab/tp), k = h). */
export function lowerHead(model: ModelSpec, ctx: LoweringContext): Op[] {
  const Tout = headTokens(ctx);
  // Formula: lm_head: m = Tout, n = ceil(vocab/tp), k = h
  // See docs/simulation-model.md §5 "Embedding and LM head"
  const vocabShard = Math.ceil(model.vocab / ctx.parallel.tp);
  // GEMM input quantized to `quant.lmHeadGemmInput` (default activations), e.g. NVFP4 W4A4 heads.
  const format = { weights: lmHeadFormat(ctx), gemmInput: ctx.quant.lmHeadGemmInput ?? ctx.quant.activations };
  const lmHead = linearGemm('lm_head', { m: Tout, n: vocabShard, k: model.hidden }, ctx, { format });
  if (model.finalNorm === 'none') return [lmHead];
  return [normOp('final_norm', model.finalNorm, Tout, model.hidden, ctx), lmHead];
}

/**
 * Resident LM-head bytes on one device: `ceil(vocab/tp)·h·bpe(head)` plus the final-norm weights. A tied head
 * shares the embedding table; the caller decides whether to count it once (same device) or twice. Unit: bytes.
 */
export function lmHeadResidentWeightBytes(model: ModelSpec, ctx: LoweringContext): number {
  const normVectors = model.finalNorm === 'none' ? 0 : model.finalNorm === 'layernorm' ? 2 : 1;
  // Formula: W_head = ceil(vocab/tp)·h·bpe(head) + (none ? 0 : layernorm ? 2 : 1)·h·bpe(hp)
  // See docs/simulation-model.md §8 "Memory fit"
  const head = matrixStoredBytes(lmHeadFormat(ctx), Math.ceil(model.vocab / ctx.parallel.tp), model.hidden);
  return head + normVectors * model.hidden * hpBpe(ctx);
}
