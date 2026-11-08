/**
 * Memory fit: resident weights + per-sequence state (KV, MLA latent, indexer keys, recurrent and conv state)
 * + peak transient activations on one device of each pipeline stage, against its memory capacity.
 * See docs/simulation-model.md §8 ("Memory fit").
 */

import { bytesPerElement, type TensorFormat } from '../ir/dtype';
import { opBytes } from '../ir/metrics';
import type { ModelSpec } from '../ir/model';
import { embeddingFormat, embeddingResidentWeightBytes } from '../lower/embedding';
import { lmHeadFormat, lmHeadResidentWeightBytes } from '../lower/head';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  stageLayerRange,
  type LoweredModel,
  type LoweringContext,
} from '../lower/lower';
import type { DeviceMemory, MemoryReport } from './types';

/** Inputs of {@link memoryReport}. */
export interface MemoryInput {
  /** The model. */
  model: ModelSpec;
  /** A lowering context carrying the quantization and parallelism (its phase and token fields are not used). */
  ctx: LoweringContext;
  /** Sequences per DP replica `B_r` (every one keeps its state on every stage). Count. */
  sequencesPerReplica: number;
  /**
   * Context length the state is sized for: `max(promptLen, decodeContextLen + 1)` (the cache after prefill, or after
   * the decode step appended its new token). Unit: tokens.
   */
  contextTokens: number;
  /** The lowered phases (prefill and decode), for the activation peak. */
  lowered: LoweredModel[];
  /** Memory capacity per device (the selected option of the modified device). Unit: bytes. */
  capacityBytes: number;
  /** Memory kind of the modified device. */
  kind: 'dedicated' | 'unified';
}

/**
 * True if two stored formats are the same bytes on device (dtype and group-quant overhead), so one allocation can
 * serve both the embedding and a tied LM head.
 */
export function sameStoredFormat(a: TensorFormat, b: TensorFormat): boolean {
  return a.dtype === b.dtype && (a.extraBitsPerElem ?? 0) === (b.extraBitsPerElem ?? 0);
}

/**
 * Resident weight bytes on one device of stage `s`: the blocks of its layers (all local experts), plus the
 * embedding shard on stage 0 and the LM-head shard (+ final norm) on the last stage. A tied head shares the
 * embedding table only when it is on the same device (pp = 1) AND stored in the same format (dtype and
 * overhead); a tied head quantized differently is a second copy and is counted in full. Unit: bytes.
 */
export function stageWeightBytes(model: ModelSpec, ctx: LoweringContext, s: number): number {
  const pp = ctx.parallel.pp;
  const [start, end] = stageLayerRange(model.layers.length, pp, s);
  let layerBytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks) layerBytes += blockResidentWeightBytes(block, ctx);
  }
  const embedBytes = s === 0 ? embeddingResidentWeightBytes(model, ctx) : 0;
  let headBytes = s === pp - 1 ? lmHeadResidentWeightBytes(model, ctx) : 0;
  if (model.tieEmbeddings && pp === 1 && sameStoredFormat(embeddingFormat(ctx), lmHeadFormat(ctx))) {
    // The shared matrix is already counted by the embedding; keep only the final-norm part of the head.
    const headMatrixBytes = Math.ceil(model.vocab / ctx.parallel.tp) * model.hidden * bytesPerElement(lmHeadFormat(ctx));
    headBytes -= headMatrixBytes;
  }
  // Formula: W_s = Σ_{blocks of s} W_block + [s = 0] W_emb + [s = pp−1] W_head   (tied, pp = 1, same format: matrix once)
  // See docs/simulation-model.md §8 ("Memory fit")
  return layerBytes + embedBytes + headBytes;
}

/** Per-sequence state bytes on one device of stage `s` for `contextTokens` of context. Unit: bytes per sequence. */
export function stageSequenceStateBytesPerSeq(model: ModelSpec, ctx: LoweringContext, s: number, contextTokens: number): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks) bytes += blockSequenceStateBytes(block, ctx, contextTokens);
  }
  // Formula: state_s = Σ_{blocks of s} blockSequenceStateBytes(ctxTokens)
  // See docs/simulation-model.md §8 ("Memory fit")
  return bytes;
}

/**
 * Peak transient activation bytes of stage `s`: the largest activation read + write of any single op of that
 * stage over the given lowered phases (the inputs and outputs of the op that is running). Unit: bytes.
 */
export function stageActivationPeakBytes(lowered: LoweredModel[], s: number): number {
  let peak = 0;
  for (const model of lowered) {
    for (const layer of model.stages[s].layers) {
      for (const op of layer.ops) {
        const b = opBytes(op);
        // Formula: act_s = max_ops (activationRead + activationWrite)
        // See docs/simulation-model.md §8 ("Memory fit")
        peak = Math.max(peak, b.activationReadBytes + b.activationWriteBytes);
      }
    }
  }
  return peak;
}

/**
 * The memory-fit report: per stage weights + B_r · state + activation peak vs capacity, and the most
 * sequences per replica that would fit at this context length:
 * `floor(min_s (capacity − W_s − act_s) / state_s)` (Infinity when no stage keeps per-sequence state; 0 when
 * the weights alone do not fit). The activation peak is taken at the evaluated batch, so this is an estimate.
 */
export function memoryReport(input: MemoryInput): MemoryReport {
  const { model, ctx, capacityBytes } = input;
  const perStage: DeviceMemory[] = [];
  let maxSequencesPerReplica = Infinity;
  for (let s = 0; s < ctx.parallel.pp; s++) {
    const weightBytes = stageWeightBytes(model, ctx, s);
    const statePerSeqBytes = stageSequenceStateBytesPerSeq(model, ctx, s, input.contextTokens);
    const activationPeakBytes = stageActivationPeakBytes(input.lowered, s);
    // Formula: total_s = W_s + B_r·state_s + act_s;  fits = total_s ≤ capacity
    // See docs/simulation-model.md §8 ("Memory fit")
    const sequenceStateBytes = input.sequencesPerReplica * statePerSeqBytes;
    const totalBytes = weightBytes + sequenceStateBytes + activationPeakBytes;
    perStage.push({
      stage: s,
      weightBytes,
      sequenceStateBytes,
      activationPeakBytes,
      totalBytes,
      capacityBytes,
      fits: totalBytes <= capacityBytes,
    });
    if (statePerSeqBytes > 0) {
      // Formula: maxSeq = floor(min_s (capacity − W_s − act_s) / state_s)
      const freeBytes = capacityBytes - weightBytes - activationPeakBytes;
      maxSequencesPerReplica = Math.min(maxSequencesPerReplica, Math.max(0, Math.floor(freeBytes / statePerSeqBytes)));
    } else if (weightBytes + activationPeakBytes > capacityBytes) {
      maxSequencesPerReplica = 0;
    }
  }
  return { kind: input.kind, perStage, fits: perStage.every((d) => d.fits), maxSequencesPerReplica };
}
