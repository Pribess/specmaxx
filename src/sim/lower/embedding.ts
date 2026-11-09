/**
 * Lowering of the token embedding (first pipeline stage). The table is vocab-parallel over TP: each rank holds
 * `ceil(vocab/tp)` rows, and a TP all-reduce of `T·h` combines the partial lookups.
 * See docs/simulation-model.md §5 "Embedding and LM head".
 *
 * Off-device table (deployment `offload.embeddings`): the embedding is a 1-gram lookup table (one row of `h` per
 * token), so, like an off-device n-gram table, a `TransferOp` of the gathered rows goes in front of the gather, whose
 * device-memory bytes stay as they are (it reads the staged rows), and the shard leaves the device's resident weights.
 * See docs/simulation-model.md §12 ("Token embedding off-device").
 */

import { bytesPerElement, type TensorFormat } from '../ir/dtype';
import type { ModelSpec } from '../ir/model';
import type { EmbeddingOp, Op, TransferOp } from '../ir/ops';
import { highPrecision, stepTokens, tpAllReduce } from './common';
import type { LoweringContext } from './lower';
import { tableRowBytes, tableRowReads } from './ngram';
import { prefetchedFraction } from './offload';

/** Storage format of the embedding table: `quant.embeddings`, default `{ dtype: highPrecision }`. */
export function embeddingFormat(ctx: LoweringContext): TensorFormat {
  return ctx.quant.embeddings ?? { dtype: highPrecision(ctx) };
}

/**
 * The row fetch in front of a gather from the off-device token-embedding table (`ctx.offload.embeddings`), or
 * undefined when the table is on the device or the step gathers no rows. `tokens` rows of `hidden` elements are
 * gathered; the table is vocab-parallel, so a rank serves the expected 1/tp of them: `R = tokens/tp` rows per device
 * (real, an expectation), read in the `'separate'` layout ({@link tableRowReads}: `[{ R, rowBytes }]`, or value and
 * block-scale reads for a block-scaled format). `usefulBytes = R·rowBytes` with `rowBytes = h·bpe(emb)`.
 * q = 1 in prefill and 0 in decode when prefetching is on ({@link prefetchedFraction}). Used by the target's
 * `embed` and the drafter's shared-embedding gathers (`name` is the gather's name; the op is `<name>.row_fetch`).
 */
export function embeddingFetchOp(
  name: string,
  tokens: number,
  hidden: number,
  ctx: LoweringContext,
): TransferOp | undefined {
  const placement = ctx.offload?.embeddings;
  if (placement === undefined) return undefined;
  // Formula: R = T/tp   (rows per device, vocab-parallel table)
  const rows = tokens / ctx.parallel.tp;
  if (!(rows > 0)) return undefined;
  const format = embeddingFormat(ctx);
  const { rowBytes } = tableRowBytes(format, hidden);
  // Formula: reads = separate layout of R rows of h;  usefulBytes = R·rowBytes;  q = prefill ? 1 : 0   (p > 0)
  // See docs/simulation-model.md §12 ("Token embedding off-device")
  return {
    kind: 'transfer',
    name: `${name}.row_fetch`,
    group: 'token_embedding',
    source: placement.tier,
    reads: tableRowReads(rows, format, hidden, 'separate'),
    usefulBytes: rows * rowBytes,
    prefetchedFraction: prefetchedFraction(ctx.offload?.prefetchAccuracy ?? 0, {
      group: 'token_embedding',
      phase: ctx.phase,
    }),
  };
}

/**
 * Embedding ops: [row fetch, when the table is off-device] → the gather of T rows → the TP all-reduce of
 * `T·h·reduceBpe` when tp > 1.
 */
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
  const fetch = embeddingFetchOp('embed', T, model.hidden, ctx);
  return [
    ...(fetch === undefined ? [] : [fetch]),
    embedding,
    ...tpAllReduce('embed.tp_all_reduce', T, model.hidden, ctx),
  ];
}

/**
 * Bytes of the token-embedding table shard on one stage-0 device, `ceil(vocab/tp)·h·bpe(emb)`, wherever it lives
 * (device or tier). Unit: bytes.
 */
export function embeddingShardBytes(model: ModelSpec, ctx: LoweringContext): number {
  // Formula: W_emb = ceil(vocab/tp)·h·bpe(emb)
  return Math.ceil(model.vocab / ctx.parallel.tp) * model.hidden * bytesPerElement(embeddingFormat(ctx));
}

/**
 * Resident embedding-table bytes on one device: the shard {@link embeddingShardBytes} when the table is on the device,
 * 0 when it is off-device (`ctx.offload.embeddings`; it then counts on its tier). Unit: bytes.
 */
export function embeddingResidentWeightBytes(model: ModelSpec, ctx: LoweringContext): number {
  // Formula: W_emb = [on device] ceil(vocab/tp)·h·bpe(emb)
  // See docs/simulation-model.md §8 "Memory fit", §12 ("Token embedding off-device")
  return ctx.offload?.embeddings === undefined ? embeddingShardBytes(model, ctx) : 0;
}
