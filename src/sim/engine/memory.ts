/**
 * Memory fit: resident weights + per-sequence state (KV, MLA latent, indexer keys, recurrent and conv state)
 * + peak transient activations (+ offload staging) on one device of each pipeline stage, against its memory
 * capacity; and, with weights offloaded, the bytes each off-device tier of a node holds against its capacity.
 * See docs/simulation-model.md §8 ("Memory fit") and §12 ("Memory tiers").
 */

import { matrixStoredBytes, type TensorFormat } from '../ir/dtype';
import { opBytes } from '../ir/metrics';
import type { DraftSpec, LayerSpec, ModelSpec } from '../ir/model';
import { draftResidentWeightBytes, draftSequenceStateBytes } from '../lower/draft';
import { embeddingFormat, embeddingResidentWeightBytes } from '../lower/embedding';
import { lmHeadFormat, lmHeadResidentWeightBytes } from '../lower/head';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  layerWeightBytesOnDevice,
  stageLayerRange,
  type LoweredModel,
  type LoweringContext,
} from '../lower/lower';
import { moeLocalExpertBytes } from '../lower/moe';
import { lookupTableFormat, ngramTableShardBytes, tableRowBytes } from '../lower/ngram';
import { denseLayerPlacementAt, expertPlacementAt, tablePlacementAt } from '../lower/offload';
import type { ResolvedHost } from './host';
import type { DeviceMemory, MemoryReport, OffloadTier, SimOptions, TierMemory } from './types';

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
  /** Offload overlap (double-buffered staging under `'ideal'`); default `'ideal'`. */
  offloadOverlap?: SimOptions['offloadOverlap'];
  /** Devices per node (the host tiers are shared by a node's devices); default 1. Count. */
  devicesPerNode?: number;
  /** The resolved host side when something is offloaded (tier capacities); absent otherwise. */
  host?: ResolvedHost;
}

/**
 * True if two stored formats are the same bytes on device (dtype and group-quant overhead), so one allocation can
 * serve both the embedding and a tied LM head.
 */
export function sameStoredFormat(a: TensorFormat, b: TensorFormat): boolean {
  return (
    a.dtype === b.dtype &&
    (a.extraBitsPerElem ?? 0) === (b.extraBitsPerElem ?? 0) &&
    (a.valueBitsPerElem ?? 0) === (b.valueBitsPerElem ?? 0) &&
    (a.perDimBits ?? 0) === (b.perDimBits ?? 0)
  );
}

/**
 * Resident weight bytes on one device of stage `s`: the blocks of its layers (all local experts), plus the
 * embedding shard on stage 0 and the LM-head shard (+ final norm) on the last stage. A tied head shares the
 * embedding table only when it is on the same device (pp = 1) AND stored in the same format (dtype and
 * overhead); a tied head quantized differently is a second copy and is counted in full. With weights offloaded
 * (`ctx.offload`), a layer of a `denseLayers` range keeps nothing, an offloaded MoE layer keeps `C = f·E_loc`
 * experts, an off-device table keeps its non-table part (design doc §2.7). Unit: bytes.
 */
export function stageWeightBytes(model: ModelSpec, ctx: LoweringContext, s: number): number {
  const pp = ctx.parallel.pp;
  const [start, end] = stageLayerRange(model.layers.length, pp, s);
  let layerBytes = 0;
  for (let i = start; i < end; i++) {
    // Formula: W_layer = 0 for a layer streamed whole (denseLayers)
    if (denseLayerPlacementAt(ctx, i) !== undefined) continue;
    for (const block of model.layers[i].blocks) layerBytes += blockResidentWeightBytes(block, ctx, i);
  }
  const embedBytes = s === 0 ? embeddingResidentWeightBytes(model, ctx) : 0;
  let headBytes = s === pp - 1 ? lmHeadResidentWeightBytes(model, ctx) : 0;
  if (model.tieEmbeddings && pp === 1 && sameStoredFormat(embeddingFormat(ctx), lmHeadFormat(ctx))) {
    // The shared matrix is already counted by the embedding; keep only the final-norm part of the head.
    const headMatrixBytes = matrixStoredBytes(lmHeadFormat(ctx), Math.ceil(model.vocab / ctx.parallel.tp), model.hidden);
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
 * Bytes of n-gram table shards resident on one device of stage `s` (included in {@link stageWeightBytes}; 0 for
 * off-device tables). Unit: bytes.
 */
export function stageLookupTableBytes(model: ModelSpec, ctx: LoweringContext, s: number): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    if (denseLayerPlacementAt(ctx, i) !== undefined || tablePlacementAt(ctx, i) !== undefined) continue;
    for (const block of model.layers[i].blocks) {
      if (block.kind === 'ngram_embedding') bytes += ngramTableShardBytes(block, ctx);
    }
  }
  // Formula: W_tables,s = Σ_{on-device tables of s} ceil(rows/tp)·d·bpe(lookup)
  return bytes;
}

/**
 * Staging buffer of stage `s` for streamed weight groups (design doc §2.7): the largest single transfer of the stage
 * over the given phases, double-buffered under `offloadOverlap 'ideal'` (one buffer fills while the other is read).
 * A transfer's buffer is `Σ ceil(count)·bytesEach` for experts and dense layers (capacity is not an average, so
 * expected misses round up) and its `usefulBytes` for table rows (the gathered rows, not the tier's pages).
 * 0 when nothing streams. Unit: bytes.
 */
export function stageStagingBytes(lowered: LoweredModel[], s: number, offloadOverlap: SimOptions['offloadOverlap']): number {
  let largest = 0;
  for (const model of lowered) {
    for (const layer of model.stages[s].layers) {
      for (const op of layer.ops) {
        if (op.kind !== 'transfer') continue;
        // Formula: buffer = lookup table: usefulBytes; else Σ ceil(count_i)·bytes_i
        const buffer =
          op.group === 'lookup_table' ? op.usefulBytes : op.reads.reduce((sum, r) => sum + Math.ceil(r.count) * r.bytesEach, 0);
        largest = Math.max(largest, buffer);
      }
    }
  }
  // Formula: staging_s = (offloadOverlap = ideal ? 2 : 1) · max_{phase, transfer} buffer
  // See docs/simulation-model.md §12 ("Memory tiers")
  return (offloadOverlap === 'ideal' ? 2 : 1) * largest;
}

/**
 * Bytes one device of stage `s` places on each off-device tier (design doc §2.7): the FULL groups, since the device
 * cache is a copy: all local experts of each offloaded MoE layer, the whole weights of each `denseLayers` layer,
 * each off-device table shard; plus, for an NVMe table with `scaleLayout 'pinned_host'`, its scales
 * (`ceil(rows/tp)·scaleBytes`) on host RAM. Unit: bytes per device.
 */
export function stageTierBytes(model: ModelSpec, ctx: LoweringContext, s: number): Record<OffloadTier, number> {
  const bytes: Record<OffloadTier, number> = { host_ram: 0, nvme: 0 };
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  for (let i = start; i < end; i++) {
    const layer = model.layers[i];
    const dense = denseLayerPlacementAt(ctx, i);
    if (dense !== undefined) {
      // Formula: tier += W_layer
      bytes[dense.tier] += layerWeightBytesOnDevice(layer, ctx);
      continue;
    }
    const experts = expertPlacementAt(ctx, i);
    const tables = tablePlacementAt(ctx, i);
    for (const block of layer.blocks) {
      if (block.kind === 'moe' && experts !== undefined) {
        // Formula: tier += E_loc·W_expert
        bytes[experts.tier] += moeLocalExpertBytes(block, ctx);
      } else if (block.kind === 'ngram_embedding' && tables !== undefined) {
        const shard = ngramTableShardBytes(block, ctx);
        const { scaleBytes } = tableRowBytes(lookupTableFormat(ctx), block.rowDim);
        if (tables.scaleLayout === 'pinned_host' && scaleBytes > 0) {
          // Formula: nvme += shard − scales;  host_ram += ceil(rows/tp)·scaleBytes
          const scales = Math.ceil(block.tableRows / ctx.parallel.tp) * scaleBytes;
          bytes[tables.tier] += shard - scales;
          bytes.host_ram += scales;
        } else {
          // Formula: tier += ceil(rows/tp)·d·bpe(lookup)
          bytes[tables.tier] += shard;
        }
      }
    }
  }
  // Spec-dec with includeDraft: the drafter's own MoE experts (last stage) live on the experts' tier too (design §3.8).
  const draftExperts = ctx.offload?.routedExperts;
  if (s === ctx.parallel.pp - 1 && ctx.speculative !== undefined && draftExperts?.includeDraft === true) {
    for (const layer of draftLayerSpecs(ctx.speculative.draft)) {
      for (const block of layer.blocks) {
        // Formula: tier += E_loc·W_expert per drafter MoE block
        if (block.kind === 'moe') bytes[draftExperts.tier] += moeLocalExpertBytes(block, ctx);
      }
    }
  }
  return bytes;
}

/** The drafter's decoder layers given as `LayerSpec`s: MTP modules, or a DeepSeek-V4-style drafter's `moeLayers`. */
function draftLayerSpecs(draft: DraftSpec): LayerSpec[] {
  return draft.kind === 'mtp' ? draft.layers : (draft.moeLayers ?? []);
}

/**
 * Recurrent and conv state of one sequence on stage `s` that a speculative verify snapshots per position (design doc
 * §3.3): the context-independent state of the blocks whose lowering writes `stateSnapshots` copies (linear attention
 * recurrent + conv, short convs, n-gram convs, GQA K/V convs) = their `blockSequenceStateBytes` at context 0. KV,
 * MLA and CSA caches are not snapshotted (rejected rows are overwritten). Unit: bytes per sequence.
 */
export function stageSnapshotStateBytesPerSeq(model: ModelSpec, ctx: LoweringContext, s: number): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks) {
      const snapshotted =
        block.kind === 'linear_attention' ||
        block.kind === 'short_conv' ||
        block.kind === 'ngram_embedding' ||
        block.kind === 'gqa_attention';
      // Formula: snap_s = Σ_{state blocks of s} blockSequenceStateBytes(ctx = 0)
      // See docs/simulation-model.md §13 ("Draft memory")
      if (snapshotted) bytes += blockSequenceStateBytes(block, ctx, 0);
    }
  }
  return bytes;
}

/**
 * Off-device tier entries of the report (design doc §2.7): per node, `bytes = Σ over the node's devices` of their
 * {@link stageTierBytes} (a device of rank r sits on stage `floor(r/(tp·dp))` and node `floor(r/devicesPerNode)`,
 * docs §5 "Rank layout"), taking the fullest node; against the tier's capacity. One entry per tier with bytes.
 * On unified memory no `'host_ram'` group reaches here (it is device memory).
 */
export function tierMemory(
  model: ModelSpec,
  ctx: LoweringContext,
  devicesPerNode: number,
  host: ResolvedHost,
): TierMemory[] {
  const { tp, dp, pp } = ctx.parallel;
  const perStage = Array.from({ length: pp }, (_, s) => stageTierBytes(model, ctx, s));
  const ranks = tp * dp * pp;
  const nodes = Math.ceil(ranks / devicesPerNode);
  const tiers: TierMemory[] = [];
  for (const tier of ['host_ram', 'nvme'] as const) {
    let bytes = 0;
    for (let node = 0; node < nodes; node++) {
      let nodeBytes = 0;
      for (let r = node * devicesPerNode; r < Math.min(ranks, (node + 1) * devicesPerNode); r++) {
        // Formula: bytes_node = Σ_{ranks r of the node} tierBytes(stage(r)),  stage(r) = floor(r / (tp·dp))
        nodeBytes += perStage[Math.floor(r / (tp * dp))][tier];
      }
      bytes = Math.max(bytes, nodeBytes);
    }
    if (bytes === 0) continue;
    const spec = tier === 'host_ram' ? host.ram : host.ssd;
    const capacityBytes = spec?.capacityBytes ?? 0;
    tiers.push({ tier, bytes, capacityBytes, fits: bytes <= capacityBytes });
  }
  return tiers;
}

/**
 * The memory-fit report: per stage weights + B_r · state + activation peak + staging vs capacity, and the most
 * sequences per replica that would fit at this context length:
 * `floor(min_s (capacity − W_s − act_s − staging_s) / state_s)` (Infinity when no stage keeps per-sequence state; 0
 * when the weights alone do not fit). The activation peak is taken at the evaluated batch, so this is an estimate.
 * With weights offloaded, the off-device tiers are checked too and enter `fits`.
 */
export function memoryReport(input: MemoryInput): MemoryReport {
  const { model, ctx, capacityBytes } = input;
  const offloadOverlap = input.offloadOverlap ?? 'ideal';
  const perStage: DeviceMemory[] = [];
  let maxSequencesPerReplica = Infinity;
  for (let s = 0; s < ctx.parallel.pp; s++) {
    const weightBytes = stageWeightBytes(model, ctx, s);
    const lookupTableBytes = stageLookupTableBytes(model, ctx, s);
    const statePerSeqBytes = stageSequenceStateBytesPerSeq(model, ctx, s, input.contextTokens);
    const activationPeakBytes = stageActivationPeakBytes(input.lowered, s);
    // Deployment terms (design doc §2.7, §3.3): 0 with nothing offloaded / spec-dec off.
    const stagingBytes = stageStagingBytes(input.lowered, s, offloadOverlap);
    // Spec-dec (last stage): draft weights (shared embedding / head excluded) + B_r · draft KV/state at the context;
    // every stage: the verify's (k + 1) per-position state snapshots × B_r.
    const spec = ctx.speculative;
    let draftBytes = 0;
    let draftStatePerSeqBytes = 0;
    let snapshotPerSeqBytes = 0;
    if (spec !== undefined) {
      if (s === ctx.parallel.pp - 1) {
        draftStatePerSeqBytes = draftSequenceStateBytes(spec.draft, ctx, input.contextTokens);
        // Formula: draft_s = W_draft + B_r·S_draft(ctx)
        // See docs/simulation-model.md §13 ("Draft memory")
        draftBytes = draftResidentWeightBytes(model, spec.draft, ctx) + input.sequencesPerReplica * draftStatePerSeqBytes;
      }
      // Formula: specTransient_s = B_r·(k + 1)·snap_s
      snapshotPerSeqBytes = spec.stateSnapshots * stageSnapshotStateBytesPerSeq(model, ctx, s);
    }
    const speculativeTransientBytes = input.sequencesPerReplica * snapshotPerSeqBytes;
    // Formula: total_s = W_s + B_r·state_s + act_s + staging_s + draft_s + specTransient_s;  fits = total_s ≤ capacity
    // See docs/simulation-model.md §8 ("Memory fit"), §12 ("Memory tiers"), §13 ("Draft memory")
    const sequenceStateBytes = input.sequencesPerReplica * statePerSeqBytes;
    const totalBytes =
      weightBytes + sequenceStateBytes + activationPeakBytes + stagingBytes + draftBytes + speculativeTransientBytes;
    perStage.push({
      stage: s,
      weightBytes,
      lookupTableBytes,
      sequenceStateBytes,
      activationPeakBytes,
      stagingBytes,
      draftBytes,
      speculativeTransientBytes,
      totalBytes,
      capacityBytes,
      fits: totalBytes <= capacityBytes,
    });
    // Per-sequence bytes that grow with B_r, and the fixed bytes besides them (spec-dec terms 0 when off).
    const perSeqBytes =
      spec === undefined ? statePerSeqBytes : statePerSeqBytes + draftStatePerSeqBytes + snapshotPerSeqBytes;
    const fixedDraftBytes = draftBytes - input.sequencesPerReplica * draftStatePerSeqBytes;
    if (perSeqBytes > 0) {
      // Formula: maxSeq = floor(min_s (capacity − W_s − act_s − staging_s − W_draft) / (state_s + S_draft + (k+1)·snap_s))
      // See docs/simulation-model.md §8 ("Memory fit"), §13 ("Draft memory")
      const freeBytes =
        spec === undefined
          ? capacityBytes - weightBytes - activationPeakBytes - stagingBytes
          : capacityBytes - weightBytes - activationPeakBytes - stagingBytes - fixedDraftBytes;
      maxSequencesPerReplica = Math.min(maxSequencesPerReplica, Math.max(0, Math.floor(freeBytes / perSeqBytes)));
    } else if (weightBytes + activationPeakBytes + stagingBytes + fixedDraftBytes > capacityBytes) {
      maxSequencesPerReplica = 0;
    }
  }
  const tiers = input.host === undefined ? [] : tierMemory(model, ctx, input.devicesPerNode ?? 1, input.host);
  // Formula: fits = ∀ s fits_s ∧ ∀ tiers (bytes ≤ capacity)
  const fits = perStage.every((d) => d.fits) && tiers.every((t) => t.fits);
  return { kind: input.kind, perStage, fits, maxSequencesPerReplica, tiers };
}
