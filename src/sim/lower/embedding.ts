/**
 * Lowering of the token embedding (first pipeline stage). The table is vocab-parallel over TP: each rank holds
 * `ceil(vocab/tp)` rows, and a TP all-reduce of `T·h` combines the partial lookups.
 * See docs/simulation-model.md §5 "Embedding and LM head".
 */

import { bytesPerElement, type TensorFormat } from '../ir/dtype';
import type { ModelSpec } from '../ir/model';
import type { EmbeddingOp, Op } from '../ir/ops';
import { highPrecision, stepTokens, tpAllReduce } from './common';
import type { LoweringContext } from './lower';

/** Storage format of the embedding table: `quant.embeddings`, default `{ dtype: highPrecision }`. */
export function embeddingFormat(ctx: LoweringContext): TensorFormat {
  return ctx.quant.embeddings ?? { dtype: highPrecision(ctx) };
}

/** Embedding ops: the gather of T rows, then the TP all-reduce of `T·h·reduceBpe` when tp > 1. */
export function lowerEmbedding(model: ModelSpec, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  // Formula: vocabShard = ceil(vocab/tp)
  const embedding: EmbeddingOp = {
    kind: 'embedding',
    name: 'embed',
    tokens: T,
    hidden: model.hidden,
    vocabShard: Math.ceil(model.vocab / ctx.parallel.tp),
    weight: embeddingFormat(ctx),
    outputDtype: ctx.quant.activations,
  };
  return [embedding, ...tpAllReduce('embed.tp_all_reduce', T, model.hidden, ctx)];
}

/** Resident embedding-table bytes on one device: `ceil(vocab/tp)·h·bpe(emb)`. Unit: bytes. */
export function embeddingResidentWeightBytes(model: ModelSpec, ctx: LoweringContext): number {
  // Formula: W_emb = ceil(vocab/tp)·h·bpe(emb)
  // See docs/simulation-model.md §8 "Memory fit"
  return Math.ceil(model.vocab / ctx.parallel.tp) * model.hidden * bytesPerElement(embeddingFormat(ctx));
}
