/**
 * Lowering of hashed n-gram embeddings (`ngram_embedding`: Qwen4Exp PLE, DeepSeek-V4.1 Engram). The lookup table
 * is row-parallel over TP like the token embedding (`ceil(rows/tp)` rows per rank, a TP all-reduce combines the
 * gathered rows); the projections, the gate and the conv are REPLICATED on every TP rank.
 * See docs/simulation-model.md §5 "n-gram embeddings".
 *
 * Notation: g = rowsPerToken, d = rowDim, O = outputDim, T = seqs·qLen.
 */

import { bytesPerElement, type TensorFormat } from '../ir/dtype';
import { opBytes } from '../ir/metrics';
import type { NgramEmbeddingBlock } from '../ir/model';
import type { EmbeddingOp, Op } from '../ir/ops';
import { actBpe, causalConvOp, convStateTokens, elementwise, hpBpe, linearGemm, stepTokens, tpAllReduce } from './common';
import { embeddingFormat } from './embedding';
import type { LoweringContext } from './lower';

/** Storage format of n-gram lookup tables: `quant.lookupTables`, else the embedding format. */
export function lookupTableFormat(ctx: LoweringContext): TensorFormat {
  return ctx.quant.lookupTables ?? embeddingFormat(ctx);
}

/**
 * n-gram ops, in order: the row gather (embedding op over T·g rows of d) → [TP all-reduce of T·g·d] → each
 * projection (m = T, module `ngram`) → `ngram_gate` over T·O (reads the projection outputs and the stream, writes
 * the stream; carries the `vectorParams` weights) → [dilated causal conv over T·O].
 */
export function lowerNgram(block: NgramEmbeddingBlock, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  const g = block.rowsPerToken;
  const d = block.rowDim;
  const O = block.outputDim;
  const act = actBpe(ctx);
  // Formula: gather: tokens = T·g, hidden = d, vocabShard = ceil(rows/tp)
  // See docs/simulation-model.md §5 "n-gram embeddings"
  const gather: EmbeddingOp = {
    kind: 'embedding',
    name: `${block.name}.gather`,
    tokens: T * g,
    hidden: d,
    vocabShard: Math.ceil(block.tableRows / ctx.parallel.tp),
    weight: lookupTableFormat(ctx),
    outputDtype: ctx.quant.activations,
  };
  const ops: Op[] = [gather, ...tpAllReduce(`${block.name}.tp_all_reduce`, T * g, d, ctx)];
  let projectedElems = 0;
  for (const p of block.projections) {
    ops.push(linearGemm(`${block.name}.${p.name}`, { m: T, n: p.outputDim, k: p.inputDim }, ctx, { module: 'ngram' }));
    projectedElems += T * p.outputDim;
  }
  // Formula: ngram_gate over T·O: read Σ T·out (projections) + T·O (stream), write T·O
  ops.push(
    elementwise(
      `${block.name}.gate`,
      'ngram_gate',
      {
        elems: T * O,
        activationReadBytes: (projectedElems + T * O) * act,
        activationWriteBytes: T * O * act,
        weightBytes: block.vectorParams * hpBpe(ctx),
      },
      ctx,
    ),
  );
  if (block.conv !== undefined) ops.push(causalConvOp(`${block.name}.conv`, T, O, block.conv, ctx));
  return ops;
}

/**
 * Resident weight bytes of an n-gram block on one device: the table shard `ceil(rows/tp)·d·bpe(lookup)` plus the
 * projections, vectors and conv taps (the weights of a one-token step without its gathered rows). A step's ops read
 * only the gathered rows, so this has its own formula, like MoE. Unit: bytes.
 */
export function ngramResidentWeightBytes(block: NgramEmbeddingBlock, ctx: LoweringContext): number {
  const oneToken: LoweringContext = { ...ctx, phase: 'decode', seqs: 1, qLen: 1, kvLen: 2 };
  const nonTable = lowerNgram(block, oneToken)
    .filter((op) => op.kind !== 'embedding')
    .reduce((sum, op) => sum + opBytes(op).weightBytes, 0);
  // Formula: W = ceil(rows/tp)·d·bpe(lookup) + Σ weights(projections, gate vectors, conv taps)
  // See docs/simulation-model.md §8 "Memory fit"
  const table = Math.ceil(block.tableRows / ctx.parallel.tp) * block.rowDim * bytesPerElement(lookupTableFormat(ctx));
  return table + nonTable;
}

/**
 * Per-sequence state of an n-gram block on one device: the conv rolling state `(K−1)·dilation·O·bpe(act)`. The
 * last n−1 token ids the hashes need are a few bytes and are ignored. Unit: bytes.
 */
export function ngramSequenceStateBytes(block: NgramEmbeddingBlock, ctx: LoweringContext): number {
  // Formula: S = [conv] (K−1)·dilation·O·bpe(act)
  // See docs/simulation-model.md §8 "Memory fit"
  return block.conv === undefined ? 0 : convStateTokens(block.conv) * block.outputDim * actBpe(ctx);
}
