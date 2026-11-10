/**
 * Lowering of hashed n-gram embeddings (`ngram_embedding`: Qwen4Exp PLE, DeepSeek-V4.1 Engram). The lookup table
 * is row-parallel over TP like the token embedding (`ceil(rows/tp)` rows per rank, a TP all-reduce combines the
 * gathered rows); the projections, the gate and the conv are REPLICATED on every TP rank.
 * See docs/simulation-model.md §5 "n-gram embeddings".
 *
 * Notation: g = rowsPerToken, d = rowDim, O = outputDim, T = seqs·qLen.
 *
 * Off-device tables (deployment `offload.lookupTables`, design doc §2.10): a `TransferOp` of the gathered rows goes in
 * front of the gather, whose device-memory bytes stay as they are.
 */

import { bytesPerElement, DTYPE_INFO, type TensorFormat } from '../ir/dtype';
import { opBytes } from '../ir/metrics';
import type { NgramEmbeddingBlock } from '../ir/model';
import type { EmbeddingOp, Op, TransferOp } from '../ir/ops';
import {
  actBpe,
  causalConvOp,
  convStateTokens,
  elementwise,
  hpBpe,
  linearGemm,
  stepTokens,
  tpAllReduce,
} from './common';
import { embeddingFormat } from './embedding';
import type { LoweringContext, TablePlacement } from './lower';
import { prefetchedFraction, tablePlacementAt } from './offload';

/** Storage format of n-gram lookup tables: `quant.lookupTables`, else the embedding format. */
export function lookupTableFormat(ctx: LoweringContext): TensorFormat {
  return ctx.quant.lookupTables ?? embeddingFormat(ctx);
}

/**
 * Bytes of one logical table row of `d` elements in `format`, split once into the value part and the block-scale part
 * (design doc §2.10): `valueBytes = d·(elementBits + extraBits)/8`, `scaleBytes = d·blockScaleBitsPerElem/8`,
 * `rowBytes = d·bpe(format) = valueBytes + scaleBytes` (Engram mxfp8 d 256: 256 + 8 = 264 B; PLE bf16 d 160: 320 + 0).
 * Unit: bytes.
 */
export function tableRowBytes(
  format: TensorFormat,
  d: number,
): { valueBytes: number; scaleBytes: number; rowBytes: number } {
  const info = DTYPE_INFO[format.dtype];
  // Formula: valueBytes = d·(valueBits + extraBits)/8;  scaleBytes = d·blockScaleBits/8;  rowBytes = d·bpe
  // See docs/simulation-model.md §12 ("n-gram tables off-device")
  const valueBytes =
    (d * ((format.valueBitsPerElem ?? info.elementBits) + (format.extraBitsPerElem ?? 0))) / 8;
  const scaleBytes = (d * info.blockScaleBitsPerElem) / 8;
  return { valueBytes, scaleBytes, rowBytes: d * bytesPerElement(format) };
}

/**
 * The physical read classes of `R = rows` gathered rows of `d` elements in `format` (design doc §2.10), per the
 * scale layout: `interleaved` `[{ R, rowBytes }]`; `separate` `[{ R, valueBytes }, { R, scaleBytes }]`;
 * `pinned_host` `[{ R, valueBytes }]` (the scales sit in host RAM; their read is not costed). A format without block
 * scales reads `[{ R, rowBytes }]` in every layout. Shared by the n-gram tables and the token embedding (1-gram,
 * always `separate`). Unit: `count` requests (expected, real), `bytesEach` bytes.
 */
export function tableRowReads(
  rows: number,
  format: TensorFormat,
  d: number,
  scaleLayout: TablePlacement['scaleLayout'],
): TransferOp['reads'] {
  const { valueBytes, scaleBytes, rowBytes } = tableRowBytes(format, d);
  if (scaleBytes === 0 || scaleLayout === 'interleaved') {
    // Formula: reads = [{ R, rowBytes }]
    return [{ count: rows, bytesEach: rowBytes }];
  }
  if (scaleLayout === 'separate') {
    // Formula: reads = [{ R, valueBytes }, { R, scaleBytes }]
    return [
      { count: rows, bytesEach: valueBytes },
      { count: rows, bytesEach: scaleBytes },
    ];
  }
  // Formula: reads = [{ R, valueBytes }]   (pinned_host: scales from host RAM, not costed)
  return [{ count: rows, bytesEach: valueBytes }];
}

/**
 * The row fetch of an off-device n-gram table (design doc §2.5, §2.10): `R = T·g/tp` rows per device (the table is
 * row-parallel, so a rank serves the expected 1/tp of the gathered rows), read per the scale layout
 * ({@link tableRowReads}). `usefulBytes = R·rowBytes` whatever the layout.
 */
export function tableFetchOp(
  block: NgramEmbeddingBlock,
  ctx: LoweringContext,
  placement: TablePlacement,
): TransferOp | undefined {
  const rows = (stepTokens(ctx) * block.rowsPerToken) / ctx.parallel.tp;
  if (!(rows > 0)) return undefined;
  const format = lookupTableFormat(ctx);
  const { rowBytes } = tableRowBytes(format, block.rowDim);
  // Formula: usefulBytes = R·rowBytes
  // See docs/simulation-model.md §12 ("n-gram tables off-device")
  return {
    kind: 'transfer',
    name: `${block.name}.row_fetch`,
    group: 'lookup_table',
    source: placement.tier,
    reads: tableRowReads(rows, format, block.rowDim, placement.scaleLayout),
    usefulBytes: rows * rowBytes,
    prefetchedFraction: prefetchedFraction(ctx.offload?.prefetchAccuracy ?? 0, {
      group: 'lookup_table',
    }),
  };
}

/**
 * n-gram ops, in order: [row fetch, when the table is off-device] → the row gather (embedding op over T·g rows of d)
 * → [TP all-reduce of T·g·d] → each projection (m = T, module `ngram`) → `ngram_gate` over T·O (reads the
 * projection outputs and the stream, writes the stream; carries the `vectorParams` weights) → [dilated causal conv
 * over T·O].
 */
export function lowerNgram(
  block: NgramEmbeddingBlock,
  ctx: LoweringContext,
  layerIndex?: number,
): Op[] {
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
  const ops: Op[] = [];
  const placement = tablePlacementAt(ctx, layerIndex);
  if (placement !== undefined) {
    const fetch = tableFetchOp(block, ctx, placement);
    if (fetch !== undefined) ops.push(fetch);
  }
  ops.push(gather, ...tpAllReduce(`${block.name}.tp_all_reduce`, T * g, d, ctx));
  let projectedElems = 0;
  for (const p of block.projections) {
    ops.push(
      linearGemm(`${block.name}.${p.name}`, { m: T, n: p.outputDim, k: p.inputDim }, ctx, {
        module: 'ngram',
      }),
    );
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
 * Bytes of an n-gram block's table shard on one device, `ceil(rows/tp)·d·bpe(lookup)`. Unit: bytes.
 */
export function ngramTableShardBytes(block: NgramEmbeddingBlock, ctx: LoweringContext): number {
  // Formula: W_table = ceil(rows/tp)·d·bpe(lookup)
  return (
    Math.ceil(block.tableRows / ctx.parallel.tp) *
    block.rowDim *
    bytesPerElement(lookupTableFormat(ctx))
  );
}

/**
 * Bytes of an n-gram block's quantization scales on one device, `ceil(rows/tp)·scaleBytes(row)` (0 for a table
 * format without scales). Unit: bytes.
 */
export function pinnedScaleBytes(block: NgramEmbeddingBlock, ctx: LoweringContext): number {
  const { scaleBytes } = tableRowBytes(lookupTableFormat(ctx), block.rowDim);
  // Formula: S_scales = ceil(rows/tp)·scaleBytes
  // See docs/simulation-model.md §12 ("Memory tiers")
  return Math.ceil(block.tableRows / ctx.parallel.tp) * scaleBytes;
}

/**
 * Resident weight bytes of an n-gram block on one device: the table shard `ceil(rows/tp)·d·bpe(lookup)` plus the
 * projections, vectors and conv taps (the weights of a one-token step without its gathered rows). A step's ops read
 * only the gathered rows, so this has its own formula, like MoE. With the table off-device ((`layerIndex` given and a
 * lookup-table placement), only the non-table part stays (design doc §2.7), plus, on unified memory
 * (`offload.hostRamIsDevice`) with `scaleLayout 'pinned_host'`, the scales: host RAM is the device pool there, so the
 * pinned scales are device bytes, not a tier. Unit: bytes.
 */
export function ngramResidentWeightBytes(
  block: NgramEmbeddingBlock,
  ctx: LoweringContext,
  layerIndex?: number,
): number {
  const oneToken: LoweringContext = { ...ctx, phase: 'decode', seqs: 1, qLen: 1, kvLen: 2 };
  const nonTable = lowerNgram(block, oneToken)
    .filter((op) => op.kind !== 'embedding')
    .reduce((sum, op) => sum + opBytes(op).weightBytes, 0);
  // Formula: W = ceil(rows/tp)·d·bpe(lookup) + Σ weights(projections, gate vectors, conv taps)
  // Formula (table off-device): W = Σ weights(projections, gate vectors, conv taps)
  // Formula (off-device, unified, pinned_host): W = Σ weights(...) + ceil(rows/tp)·scaleBytes
  // See docs/simulation-model.md §8 "Memory fit", §12 ("Memory tiers")
  const placement = tablePlacementAt(ctx, layerIndex);
  if (placement !== undefined) {
    const scalesOnDevice =
      placement.scaleLayout === 'pinned_host' && ctx.offload?.hostRamIsDevice === true;
    return scalesOnDevice ? nonTable + pinnedScaleBytes(block, ctx) : nonTable;
  }
  const table = ngramTableShardBytes(block, ctx);
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
