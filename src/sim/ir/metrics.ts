/**
 * FLOP and byte counts of ops (pure functions of the op's fields). Each formula is written in a `// Formula:`
 * comment matching docs/simulation-model.md §3 (Op catalogue).
 *
 * Notation (as in the docs): `bpe(f)` = {@link bytesPerElement}; `past = kvLen − qLen`; `cap` = the window size
 * or top-k of a {@link KvSelection}.
 */

import { bitsPerElement, bytesPerElement, type Dtype, type TensorFormat } from './dtype';
import type {
  AttentionOp,
  EmbeddingOp,
  ElementwiseOp,
  GemmOp,
  IndexerOp,
  KvSelection,
  Op,
  RecurrentOp,
} from './ops';

/**
 * Per-device memory (HBM/GDDR/unified DRAM) traffic of one op, split by what is moved.
 * Compulsory traffic only: each operand is read once and each result written once (ideal on-chip reuse).
 * All fields are per device, per microbatch step. Unit: bytes.
 */
export interface ByteBreakdown {
  /** Weight bytes read (incl. quantization scales, bias, 2:4 metadata). Unit: bytes. */
  weightBytes: number;
  /** Activation bytes read (op inputs other than weights, KV and state). Unit: bytes. */
  activationReadBytes: number;
  /** Activation bytes written (op outputs other than KV and state). Unit: bytes. */
  activationWriteBytes: number;
  /** KV-cache (or MLA latent / indexer key cache / decompressed K/V) bytes read. Unit: bytes. */
  kvReadBytes: number;
  /** KV-cache bytes appended for the new tokens. Unit: bytes. */
  kvWriteBytes: number;
  /** Fixed-size per-sequence state bytes read (recurrent state, conv1d rolling state). Unit: bytes. */
  stateReadBytes: number;
  /** Fixed-size per-sequence state bytes written. Unit: bytes. */
  stateWriteBytes: number;
  /** Sum of all the fields above. Unit: bytes. */
  totalBytes: number;
}

/** Bytes per element of a bare dtype (no group-quant overhead). Unit: bytes/element. */
function bpeOf(d: Dtype): number {
  return bytesPerElement({ dtype: d });
}

/** Builds a {@link ByteBreakdown} from its parts; `totalBytes` is their sum. */
function breakdown(parts: Omit<ByteBreakdown, 'totalBytes'>): ByteBreakdown {
  // Formula: total = weight + actRead + actWrite + kvRead + kvWrite + stateRead + stateWrite
  const totalBytes =
    parts.weightBytes +
    parts.activationReadBytes +
    parts.activationWriteBytes +
    parts.kvReadBytes +
    parts.kvWriteBytes +
    parts.stateReadBytes +
    parts.stateWriteBytes;
  return { ...parts, totalBytes };
}

/** All-zero traffic (collectives: their wire bytes are `messageBytes`, not memory traffic). */
const NO_BYTES: ByteBreakdown = breakdown({
  weightBytes: 0,
  activationReadBytes: 0,
  activationWriteBytes: 0,
  kvReadBytes: 0,
  kvWriteBytes: 0,
  stateReadBytes: 0,
  stateWriteBytes: 0,
});

/**
 * Stored bytes of `batch` weight matrices of `n × k` elements in format `weight`, dense or 2:4 sparse.
 * 2:4 keeps half the values plus a 2-bit index per kept value. Bias not included. Unit: bytes.
 * Also used by lowering for resident-weight accounting, so the two always agree.
 */
export function gemmWeightBytes(
  batch: number,
  n: number,
  k: number,
  weight: TensorFormat,
  sparsity: 'dense' | '2:4',
): number {
  const elems = batch * n * k;
  const bits = bitsPerElement(weight);
  switch (sparsity) {
    case 'dense':
      // Formula: W = b·n·k·bpe(w)
      // See docs/simulation-model.md §3 "GEMM"
      return elems * (bits / 8);
    case '2:4':
      // Formula: W = b·n·k·(bits/2 + 1)/8   (half the values + 2-bit index per kept value = 1 bit/elem)
      // See docs/simulation-model.md §3 "GEMM"
      return (elems * (bits / 2 + 1)) / 8;
    default: {
      const _never: never = sparsity;
      throw new Error(`unknown sparsity ${String(_never)}`);
    }
  }
}

/**
 * (query, key) pairs attended per sequence per head in one step, the causal triangle counted exactly.
 * `qLen` new queries, `kvLen` keys visible to the last query (tokens), `causal` masking, `sel` the key cap
 * (window / top-k), `tokensPerEntry` r when each key stands for r tokens (compressed or pooled keys; a query at
 * absolute position p sees p/r keys, continuous: overcounts by < qLen/2 pairs against floor(p/r)). r = 1 is the
 * plain token formula. Unit: pairs (count). See docs §3 "Attended pairs".
 */
export function attendedPairs(
  qLen: number,
  kvLen: number,
  causal: boolean,
  sel: KvSelection,
  tokensPerEntry = 1,
): number {
  const cap = selectionCap(sel);
  const r = tokensPerEntry;
  if (!causal) {
    // Formula: pairs = qLen·min(cap, ceil(kvLen/r))
    // See docs/simulation-model.md §3 "Attended pairs"
    return qLen * Math.min(cap, Math.ceil(kvLen / r));
  }
  const past = kvLen - qLen;
  if (cap === Infinity) {
    // Formula: pairs = (qLen·past + qLen(qLen+1)/2)/r      (query j sees (past + j)/r keys, j = 1..qLen)
    // See docs/simulation-model.md §3 "Attended pairs"
    return (qLen * past + (qLen * (qLen + 1)) / 2) / r;
  }
  // Query j (1-based) sees min(cap, (past + j)/r) keys. The first j0 queries are below the cap.
  // Formula: j0 = clamp(cap·r − past, 0, qLen); pairs = (j0·past + j0(j0+1)/2)/r + (qLen − j0)·cap
  // See docs/simulation-model.md §3 "Attended pairs"
  const j0 = Math.min(Math.max(cap * r - past, 0), qLen);
  return (j0 * past + (j0 * (j0 + 1)) / 2) / r + (qLen - j0) * cap;
}

/** Keys a query may see at most under a selection: window size, top-k, or Infinity for full. Unit: keys. */
function selectionCap(sel: KvSelection): number {
  switch (sel.kind) {
    case 'full':
      return Infinity;
    case 'window':
      return sel.windowTokens;
    case 'topk':
      return sel.topK;
    default: {
      const _never: never = sel;
      throw new Error(`unknown kv selection ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Distinct keys touched per sequence in one step (the union over its queries), with r tokens per key:
 * full `ceil(kvLen/r)`; window `min(ceil(kvLen/r), W + qLen − 1)`; top-k `min(ceil(kvLen/r), topK · qLen)`.
 * Unit: keys (tokens when r = 1). See docs §3.
 */
export function kvTokensRead(qLen: number, kvLen: number, sel: KvSelection, tokensPerEntry = 1): number {
  const visible = Math.ceil(kvLen / tokensPerEntry);
  switch (sel.kind) {
    case 'full':
      // Formula: keys = ceil(kvLen/r)
      return visible;
    case 'window':
      // Formula: keys = min(ceil(kvLen/r), W + qLen − 1)   (the windows of consecutive queries overlap)
      // See docs/simulation-model.md §3 "Attention"
      return Math.min(visible, sel.windowTokens + qLen - 1);
    case 'topk':
      // Formula: keys = min(ceil(kvLen/r), topK·qLen)       (worst case: disjoint selections)
      // See docs/simulation-model.md §3 "Attention"
      return Math.min(visible, sel.topK * qLen);
    default: {
      const _never: never = sel;
      throw new Error(`unknown kv selection ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * (query, key) pairs of an attention op per sequence per head: its primary keys plus its secondary (compressed)
 * entries, which share the softmax. Unit: pairs. See docs §3 "Attention".
 */
export function attentionPairs(op: AttentionOp): number {
  const primary = attendedPairs(op.qLen, op.kvLen, op.causal, op.kvSelection);
  const sec = op.secondaryKv;
  // Formula: pairs = pairs(primary) + pairs(secondary, r)
  const secondary = sec === undefined ? 0 : attendedPairs(op.qLen, op.kvLen, op.causal, sec.kvSelection, sec.tokensPerEntry);
  return primary + secondary;
}

/** The scoring cap of an indexer: every visible key, or at most `maxScoredKeys`. */
function indexerScoredSelection(op: IndexerOp): KvSelection {
  return op.maxScoredKeys === undefined ? { kind: 'full' } : { kind: 'topk', topK: op.maxScoredKeys };
}

/**
 * (query, key) pairs an indexer scores per sequence (all heads share them): every visible indexer key (each
 * pooling p tokens), capped at `maxScoredKeys`. Unit: pairs. See docs §3 "Indexer".
 */
export function indexerPairs(op: IndexerOp): number {
  // Formula: pairs_I = attendedPairs(qLen, kvLen, causal, cap = maxScoredKeys ?? ∞, r = p)
  return attendedPairs(op.qLen, op.kvLen, op.causal, indexerScoredSelection(op), op.keyPoolTokens);
}

/**
 * Dense-equivalent FLOP of one op on one device (2:4 sparsity does not halve it; the sparse peak is
 * quoted in dense-equivalent FLOP/s too). Collectives and embeddings: 0. Unit: FLOP.
 */
export function opFlops(op: Op): number {
  switch (op.kind) {
    case 'gemm':
      return gemmFlops(op);
    case 'attention':
      return attentionFlops(op);
    case 'indexer':
      return indexerFlops(op);
    case 'recurrent':
      return recurrentFlops(op);
    case 'elementwise':
      // Formula: F = elems·flopsPerElem
      // See docs/simulation-model.md §3 "Elementwise"
      return op.elems * op.flopsPerElem;
    case 'embedding':
      return 0;
    case 'collective':
      return 0;
    default: {
      const _never: never = op;
      throw new Error(`unknown op ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Per-device memory traffic of one op (see {@link ByteBreakdown}). Collectives return all zeros: their wire
 * bytes are `CollectiveOp.messageBytes`, costed by the alpha-beta model, not as memory traffic.
 */
export function opBytes(op: Op): ByteBreakdown {
  switch (op.kind) {
    case 'gemm':
      return gemmBytes(op);
    case 'attention':
      return attentionBytes(op);
    case 'indexer':
      return indexerBytes(op);
    case 'recurrent':
      return recurrentBytes(op);
    case 'elementwise':
      return elementwiseBytes(op);
    case 'embedding':
      return embeddingBytes(op);
    case 'collective':
      return NO_BYTES;
    default: {
      const _never: never = op;
      throw new Error(`unknown op ${JSON.stringify(_never)}`);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM

/** FLOP of a (grouped) GEMM. Unit: FLOP. */
function gemmFlops(op: GemmOp): number {
  // Formula: F = 2·b·m·n·k
  // See docs/simulation-model.md §3 "GEMM"
  return 2 * op.batch * op.m * op.n * op.k;
}

/** Memory traffic of a (grouped) GEMM: weights (+bias) once, input once, output once. Unit: bytes. */
function gemmBytes(op: GemmOp): ByteBreakdown {
  // Formula: W = b·n·k·bpe(w) [2:4: b·n·k·(bits/2+1)/8] + b·n·bpe(bias); read = b·m·k·bpe(in); write = b·m·n·bpe(out)
  // See docs/simulation-model.md §3 "GEMM"
  const matrixBytes = gemmWeightBytes(op.batch, op.n, op.k, op.weight, op.sparsity);
  const biasBytes = op.biasDtype === undefined ? 0 : op.batch * op.n * bpeOf(op.biasDtype);
  const inputBytes = op.batch * op.m * op.k * bpeOf(op.inputDtype);
  const outputBytes = op.batch * op.m * op.n * bpeOf(op.outputDtype);
  return breakdown({
    weightBytes: matrixBytes + biasBytes,
    activationReadBytes: inputBytes,
    activationWriteBytes: outputBytes,
    kvReadBytes: 0,
    kvWriteBytes: 0,
    stateReadBytes: 0,
    stateWriteBytes: 0,
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Attention

/** FLOP of the fused attention core (Q·Kᵀ and P·V; softmax excluded). Unit: FLOP. */
function attentionFlops(op: AttentionOp): number {
  const pairs = attentionPairs(op);
  // Formula: F = 2·seqs·heads·(pairs + pairs_secondary)·(qkHeadDim + vHeadDim)
  // See docs/simulation-model.md §3 "Attention"
  return 2 * op.seqs * op.heads * pairs * (op.qkHeadDim + op.vHeadDim);
}

/** Memory traffic of the fused attention core: Q in, O out, K/V read for touched tokens, K/V appended. Unit: bytes. */
function attentionBytes(op: AttentionOp): ByteBreakdown {
  const act = bpeOf(op.activationDtype);
  const tokensRead = kvTokensRead(op.qLen, op.kvLen, op.kvSelection);
  // Formula: readQ = seqs·qLen·heads·qk·bpe(act); writeO = seqs·qLen·heads·v·bpe(act)
  // Formula: kvRead = seqs·kvTokensRead·kvRead.elemsPerToken·bpe(kvRead); kvWrite = seqs·qLen·kvWrite.elemsPerToken·bpe(kvWrite)
  // See docs/simulation-model.md §3 "Attention"
  const qBytes = op.seqs * op.qLen * op.heads * op.qkHeadDim * act;
  const oBytes = op.seqs * op.qLen * op.heads * op.vHeadDim * act;
  const kvReadBytes = op.seqs * tokensRead * op.kvRead.elemsPerToken * bytesPerElement(op.kvRead.format);
  const kvWriteBytes = op.seqs * op.qLen * op.kvWrite.elemsPerToken * bytesPerElement(op.kvWrite.format);
  // Formula: secondary: kvRead += seqs·kvTokensRead(sel, r)·elemsPerEntry·bpe; kvWrite += seqs·qLen·elemsPerNewToken·bpe
  const sec = op.secondaryKv;
  const secReadBytes =
    sec === undefined
      ? 0
      : op.seqs *
        kvTokensRead(op.qLen, op.kvLen, sec.kvSelection, sec.tokensPerEntry) *
        sec.kvRead.elemsPerToken *
        bytesPerElement(sec.kvRead.format);
  const secWriteBytes =
    sec === undefined ? 0 : op.seqs * op.qLen * sec.kvWrite.elemsPerToken * bytesPerElement(sec.kvWrite.format);
  return breakdown({
    weightBytes: 0,
    activationReadBytes: qBytes,
    activationWriteBytes: oBytes,
    kvReadBytes: kvReadBytes + secReadBytes,
    kvWriteBytes: kvWriteBytes + secWriteBytes,
    stateReadBytes: 0,
    stateWriteBytes: 0,
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Indexer

/** FLOP of the top-k indexer: scores every visible (pooled) key, up to `maxScoredKeys`. Unit: FLOP. */
function indexerFlops(op: IndexerOp): number {
  // Formula: F = 2·seqs·heads·pairs_I·headDim
  // See docs/simulation-model.md §3 "Indexer"
  return 2 * op.seqs * op.heads * indexerPairs(op) * op.headDim;
}

/**
 * Memory traffic of the indexer: q (and head weights) in, the touched keys read, new keys appended, indices out.
 * Unit: bytes.
 */
function indexerBytes(op: IndexerOp): ByteBreakdown {
  const act = bpeOf(op.activationDtype);
  const key = bytesPerElement(op.keyCache);
  const int32Bytes = 4;
  const keysRead = kvTokensRead(op.qLen, op.kvLen, indexerScoredSelection(op), op.keyPoolTokens);
  // Formula: read = seqs·qLen·heads·headDim·bpe(act) + [headWeights] seqs·qLen·heads·bpe(act)
  // Formula: keyRead = seqs·min(ceil(kvLen/p), maxScored·qLen)·keyElems·bpe(key); keyWrite = seqs·qLen·keyWriteElems·bpe(key)
  // Formula: indices = seqs·qLen·topK·4
  // See docs/simulation-model.md §3 "Indexer"
  const qBytes = op.seqs * op.qLen * op.heads * op.headDim * act;
  const headWeightBytes = op.headWeights ? op.seqs * op.qLen * op.heads * act : 0;
  const keyReadBytes = op.seqs * keysRead * op.keyElemsPerKey * key;
  const keyWriteBytes = op.seqs * op.qLen * op.keyWriteElemsPerToken * key;
  const indexBytes = op.seqs * op.qLen * op.topK * int32Bytes;
  return breakdown({
    weightBytes: 0,
    activationReadBytes: qBytes + headWeightBytes,
    activationWriteBytes: indexBytes,
    kvReadBytes: keyReadBytes,
    kvWriteBytes: keyWriteBytes,
    stateReadBytes: 0,
    stateWriteBytes: 0,
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Recurrent

/** FLOP of a linear-attention / SSM mixer step (state terms + intra-chunk attention in chunked mode). Unit: FLOP. */
function recurrentFlops(op: RecurrentOp): number {
  const stateElemsPerHead = op.keyDim * op.valueDim;
  // Intra-chunk causal attention: on average (C+1)/2 keys per query, 2 FLOP per element over dk + dv.
  const intraChunkPerToken = op.mode === 'chunked' ? (op.chunkLen + 1) * (op.keyDim + op.valueDim) : 0;
  // Formula: F = seqs·qLen·heads·( c_s·dk·dv + (chunked ? (C+1)·(dk+dv) : 0) )
  // See docs/simulation-model.md §3 "Recurrent"
  return op.seqs * op.qLen * op.heads * (op.stateFlopsPerElemPerToken * stateElemsPerHead + intraChunkPerToken);
}

/** Memory traffic of a recurrent mixer step: q/k/v/gates in, o out, state read (if any) and written once. Unit: bytes. */
function recurrentBytes(op: RecurrentOp): ByteBreakdown {
  const act = bpeOf(op.activationDtype);
  const stateBytes = op.seqs * op.heads * op.keyDim * op.valueDim * bpeOf(op.stateDtype);
  const tokens = op.seqs * op.qLen;
  // Formula: read = 2·T·qkHeads·dk·bpe(act) (q,k) + T·heads·dv·bpe(act) (v) + T·heads·g·bpe(act) (gates; g = 2, KDA dk+1)
  // Formula: write = T·heads·dv·bpe(act); state read = hasPriorState ? seqs·heads·dk·dv·bpe(state) : 0; state write = seqs·heads·dk·dv·bpe(state)
  // See docs/simulation-model.md §3 "Recurrent"
  const qkBytes = 2 * tokens * op.qkHeads * op.keyDim * act;
  const vBytes = tokens * op.heads * op.valueDim * act;
  const gateBytes = tokens * op.heads * op.gateElemsPerTokenPerHead * act;
  const oBytes = tokens * op.heads * op.valueDim * act;
  return breakdown({
    weightBytes: 0,
    activationReadBytes: qkBytes + vBytes + gateBytes,
    activationWriteBytes: oBytes,
    kvReadBytes: 0,
    kvWriteBytes: 0,
    stateReadBytes: op.hasPriorState ? stateBytes : 0,
    stateWriteBytes: stateBytes,
  });
}

// ---------------------------------------------------------------------------------------------------------------
// Elementwise, embedding

/** Memory traffic of an elementwise op: the explicit byte fields lowering computed. Unit: bytes. */
function elementwiseBytes(op: ElementwiseOp): ByteBreakdown {
  // Formula: bytes = actRead + actWrite + weight + stateRead + stateWrite   (fields of the op)
  // See docs/simulation-model.md §3 "Elementwise"
  return breakdown({
    weightBytes: op.weightBytes,
    activationReadBytes: op.activationReadBytes,
    activationWriteBytes: op.activationWriteBytes,
    kvReadBytes: 0,
    kvWriteBytes: 0,
    stateReadBytes: op.stateReadBytes,
    stateWriteBytes: op.stateWriteBytes,
  });
}

/** Memory traffic of an embedding gather: one table row per token read, one activation row written. Unit: bytes. */
function embeddingBytes(op: EmbeddingOp): ByteBreakdown {
  // Formula: W = tokens·hidden·bpe(w) (gathered rows, conservative); write = tokens·hidden·bpe(out)
  // See docs/simulation-model.md §3 "Embedding"
  return breakdown({
    weightBytes: op.tokens * op.hidden * bytesPerElement(op.weight),
    activationReadBytes: 0,
    activationWriteBytes: op.tokens * op.hidden * bpeOf(op.outputDtype),
    kvReadBytes: 0,
    kvWriteBytes: 0,
    stateReadBytes: 0,
    stateWriteBytes: 0,
  });
}
