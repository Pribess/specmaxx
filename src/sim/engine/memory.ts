/**
 * Memory fit: resident weights + per-sequence state (KV, MLA latent, indexer keys, recurrent and conv state)
 * + peak transient activations (decode step and the last prefill chunk) + runtime reserve (+ offload staging) on one
 * device of each pipeline stage, against its GPU-usable capacity (the whole device on dedicated memory, the OS's GPU
 * limit on unified memory); and, with weights offloaded, the bytes each off-device tier of a node holds against its
 * capacity. See docs/simulation-model.md §8 ("Memory fit") and §12 ("Memory tiers").
 */

import { matrixStoredBytes, type TensorFormat } from '../ir/dtype';
import { opBytes } from '../ir/metrics';
import type { DraftSpec, LayerSpec, ModelSpec } from '../ir/model';
import type { TransferOp } from '../ir/ops';
import { draftResidentWeightBytes, draftSequenceStateBytes } from '../lower/draft';
import { embeddingFormat, embeddingResidentWeightBytes, embeddingShardBytes } from '../lower/embedding';
import { lmHeadFormat, lmHeadResidentWeightBytes, lmHeadVocabShard } from '../lower/head';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  stageLayerRange,
  type LoweredModel,
  type LoweringContext,
} from '../lower/lower';
import { moeLocalExpertBytes } from '../lower/moe';
import { lookupTableFormat, ngramTableShardBytes, tableRowBytes } from '../lower/ngram';
import { draftExpertPlacement, expertPlacementAt, tablePlacementAt } from '../lower/offload';
import { prefetchesFromBothTiers } from './transfer';
import type { ResolvedHost } from './host';
import type { DeviceMemory, MemoryReport, OffloadTier, TierBytes, TierMemory } from './types';

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
  /** The lowered phases (prefill and decode), for the offload staging buffers. */
  lowered: LoweredModel[];
  /**
   * The lowered phases whose ops set the activation peak: the decode step and the LAST prefill chunk
   * ({@link prefillChunkShape}); default `lowered`.
   */
  activationLowered?: LoweredModel[];
  /** Installed memory capacity per device (the selected option of the modified device). Unit: bytes. */
  capacityBytes: number;
  /** Memory kind of the modified device. */
  kind: 'dedicated' | 'unified';
  /**
   * GPU-usable fraction f of `capacityBytes` (unified memory: {@link resolveGpuUsableFraction}); default 1.
   * Dimensionless, (0, 1].
   */
  gpuUsableFraction?: number;
  /** Runtime reserve per device (framework, driver, CUDA context); default 0. Unit: bytes. */
  runtimeReserveBytes?: number;
  /** Devices per node (the host tiers are shared by a node's devices); default 1. Count. */
  devicesPerNode?: number;
  /** The resolved host side when something is offloaded (tier capacities); absent otherwise. */
  host?: ResolvedHost;
}

/**
 * Default prefill chunk C of the memory fit (`Workload.prefillChunkTokens`): 8192 tokens per step. Sources: SGLang's
 * `--chunked-prefill-size` default (8192; https://docs.sglang.ai/advanced_features/server_arguments.html) and vLLM's
 * `max_num_batched_tokens` default for the offline `LLM` class and for the API server on H100-class GPUs (8192;
 * https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py). Smaller budgets (vLLM 2048 on other
 * GPUs, llama.cpp `n_ubatch` 512, mlx-lm `prefill_step_size` 2048) only shrink the peak, so 8192 is the
 * conservative common choice. Unit: tokens.
 */
export const DEFAULT_PREFILL_CHUNK_TOKENS = 8192;

/** The shape of one prefill step of the memory fit (see {@link prefillChunkShape}). */
export interface PrefillChunkShape {
  /** Sequences in the step. Count. */
  seqs: number;
  /** New prompt tokens per sequence in the step. Unit: tokens. */
  qLen: number;
  /** Keys visible to the step's last query (past chunks + this one). Unit: tokens. */
  kvLen: number;
}

/**
 * The prefill step whose activations the memory fit counts, under chunked prefill with a budget of C tokens per
 * step: a microbatch of `seqs` prompts of P tokens, of which `N = newTokens` are computed (`N = P` without a prefix
 * cache; with a prefix-cache hit `N = P − floor(P·h) < P` and the cached tokens only add keys, so `kvLen` stays P).
 * - `seqs·N ≤ C`: the whole microbatch in one step (`seqs, qLen = N, kvLen = P`), as the rate is costed.
 * - `N ≤ C < seqs·N`: as many whole prompts as fit the budget (`floor(C/N)` sequences, `qLen = N`, `kvLen = P`).
 * - `C < N`: one sequence's LAST chunk (`qLen = C`, `kvLen = P`): the chunk that sees the whole prompt.
 * Without a cache (`N = P`) these are `qLen = kvLen = P` as before.
 * Activations of every op grow with the step's tokens `seqs·qLen`, so this step bounds the peak of all chunks.
 */
export function prefillChunkShape(
  promptLen: number,
  seqs: number,
  chunkTokens: number,
  newTokens: number = promptLen,
): PrefillChunkShape {
  // `newTokens` = P_new ≤ P: prompt tokens actually computed (prefix cache); the cached P − P_new tokens only add keys.
  // Formula: seqs·N ≤ C → (seqs, N, P);  N ≤ C → (floor(C/N), N, P);  else (1, C, P)   [N = P_new; P_new = P: no cache]
  // See docs/simulation-model.md §8 ("Activation peak")
  if (seqs * newTokens <= chunkTokens) return { seqs, qLen: newTokens, kvLen: promptLen };
  if (newTokens <= chunkTokens)
    return { seqs: Math.floor(chunkTokens / newTokens), qLen: newTokens, kvLen: promptLen };
  return { seqs: 1, qLen: chunkTokens, kvLen: promptLen };
}

/**
 * The GPU-usable fraction f the fit uses: on unified memory the stack's override (`HardwareStack.gpuUsableFraction`,
 * a raised OS limit), else the device's default (`MemorySpec.gpuUsableFraction`), else 1; on dedicated memory always
 * 1 (the device is the GPU's own). Dimensionless, (0, 1].
 */
export function resolveGpuUsableFraction(
  memory: { kind: 'dedicated' | 'unified'; gpuUsableFraction?: number },
  override: number | undefined,
): number {
  if (memory.kind !== 'unified') return 1;
  // Formula: f = override ?? f_device ?? 1   (unified memory)
  return override ?? memory.gpuUsableFraction ?? 1;
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
 * embedding shard on stage 0 (when the table is on the device) and the LM-head shard (+ final norm) on the last
 * stage. A tied head shares the embedding table only when it is on the same device (pp = 1, and the embedding is not
 * off-device) AND stored in the same format (dtype and overhead); a tied head quantized differently is a second copy
 * and is counted in full. (Validation rejects an off-device embedding on a tied model; the on-device condition is a
 * defense.) With weights offloaded (`ctx.offload`), an offloaded MoE layer keeps `C = f·E_loc` experts, an
 * off-device table keeps its non-table part (design doc §2.7), and an off-device token embedding leaves stage 0.
 * Unit: bytes.
 */
export function stageWeightBytes(model: ModelSpec, ctx: LoweringContext, s: number): number {
  const pp = ctx.parallel.pp;
  const [start, end] = stageLayerRange(model.layers.length, pp, s);
  let layerBytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks)
      layerBytes += blockResidentWeightBytes(block, ctx, i);
  }
  const embedBytes = s === 0 ? embeddingResidentWeightBytes(model, ctx) : 0;
  let headBytes = s === pp - 1 ? lmHeadResidentWeightBytes(model, ctx) : 0;
  if (
    model.tieEmbeddings &&
    pp === 1 &&
    ctx.offload?.embeddings === undefined &&
    sameStoredFormat(embeddingFormat(ctx), lmHeadFormat(ctx))
  ) {
    // The shared matrix is already counted by the embedding (its ceil(vocab/tp) rows hold the head's
    // ceil(vocab/G) shard: G = tp, or G = ep ≥ tp = 1 with denseTpGroup 'ep'); keep only the final-norm part.
    const headMatrixBytes = matrixStoredBytes(lmHeadFormat(ctx), lmHeadVocabShard(model, ctx), model.hidden);
    headBytes -= headMatrixBytes;
  }
  // Formula: W_s = Σ_{blocks of s} W_block + [s = 0 ∧ emb on device] W_emb + [s = pp−1] W_head
  //          (tied, pp = 1, emb on device, same format: matrix once)
  // See docs/simulation-model.md §8 ("Memory fit"), §12 ("Token embedding off-device")
  return layerBytes + embedBytes + headBytes;
}

/** Per-sequence state bytes on one device of stage `s` for `contextTokens` of context. Unit: bytes per sequence. */
export function stageSequenceStateBytesPerSeq(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
  contextTokens: number,
): number {
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  let bytes = 0;
  for (let i = start; i < end; i++) {
    for (const block of model.layers[i].blocks)
      bytes += blockSequenceStateBytes(block, ctx, contextTokens);
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
    if (tablePlacementAt(ctx, i) !== undefined) continue;
    for (const block of model.layers[i].blocks) {
      if (block.kind === 'ngram_embedding') bytes += ngramTableShardBytes(block, ctx);
    }
  }
  // Formula: W_tables,s = Σ_{on-device tables of s} ceil(rows/tp)·d·bpe(lookup)
  return bytes;
}

/** The transfer ops of stage `s` of one lowered phase, in op order. */
function stageTransferOps(model: LoweredModel, s: number): TransferOp[] {
  const transfers: TransferOp[] = [];
  for (const layer of model.stages[s].layers) {
    for (const op of layer.ops) if (op.kind === 'transfer') transfers.push(op);
  }
  return transfers;
}

/**
 * Staging buffer of stage `s` for streamed weight groups (design doc §2.7): the largest single transfer of the stage
 * over the given phases, double-buffered when it is prefetched (`prefetchedFraction > 0`: one buffer fills while the
 * other is read), single otherwise.
 * A phase whose transfers prefetch from both host RAM and the SSD ({@link prefetchesFromBothTiers}) has two streams
 * in flight (docs/simulation-model.md §12 "Overlap across tiers"): one buffer being read by the device plus one being
 * filled per source, `max_all + max_ram,q>0 + max_ssd,q>0`. With one source this is today's 2·max.
 * A transfer's buffer is `Σ ceil(count)·bytesEach` for experts (capacity is not an average, so
 * expected misses round up) and its `usefulBytes` for table and token-embedding rows (the gathered rows, not the
 * tier's pages). Token-embedding transfers are taken from `embeddingLowered` (the decode step and the LAST prefill
 * chunk, as for the activation peak: chunked prefill stages one chunk's rows, not the whole prompt's), every other
 * transfer from `lowered`; phase i of one pairs with phase i of the other (same length), and `embeddingLowered`
 * defaults to `lowered`. 0 when nothing streams. Unit: bytes.
 */
export function stageStagingBytes(
  lowered: LoweredModel[],
  s: number,
  embeddingLowered: LoweredModel[] = lowered,
): number {
  if (embeddingLowered.length !== lowered.length) {
    throw new Error('stageStagingBytes: embeddingLowered must list the same phases as lowered');
  }
  // Formula: buffer = table / embedding rows: usefulBytes; else Σ ceil(count_i)·bytes_i
  const bufferBytes = (op: TransferOp): number =>
    op.group === 'lookup_table' || op.group === 'token_embedding'
      ? op.usefulBytes
      : op.reads.reduce((sum, r) => sum + Math.ceil(r.count) * r.bytesEach, 0);
  let largest = 0;
  for (let phase = 0; phase < lowered.length; phase++) {
    // The phase's transfers: every group but the token embedding from `lowered`, the token embedding from the
    // matching phase of `embeddingLowered` (same source and q, only R per chunk), so the both-tiers test below
    // agrees with the stage time's (`pipeline.ts: stageTransfer`).
    const transfers = [
      ...stageTransferOps(lowered[phase], s).filter((op) => op.group !== 'token_embedding'),
      ...stageTransferOps(embeddingLowered[phase], s).filter((op) => op.group === 'token_embedding'),
    ];
    for (const op of transfers) {
      // Formula: slots = q > 0 ? 2 : 1   (double buffer for a prefetched transfer)
      const slots = op.prefetchedFraction > 0 ? 2 : 1;
      largest = Math.max(largest, slots * bufferBytes(op));
    }
    if (prefetchesFromBothTiers(transfers)) {
      // Formula: staging = max_all buffer + max_{ram,q>0} buffer + max_{ssd,q>0} buffer   (1 read + 1 fill per source)
      let maxAll = 0;
      let maxRamPrefetched = 0;
      let maxNvmePrefetched = 0;
      for (const op of transfers) {
        const buffer = bufferBytes(op);
        maxAll = Math.max(maxAll, buffer);
        if (op.prefetchedFraction <= 0) continue;
        if (op.source === 'host_ram') maxRamPrefetched = Math.max(maxRamPrefetched, buffer);
        else maxNvmePrefetched = Math.max(maxNvmePrefetched, buffer);
      }
      largest = Math.max(largest, maxAll + maxRamPrefetched + maxNvmePrefetched);
    }
  }
  // Formula: staging_s = max_phase ( both tiers prefetched ? max_all + max_ram + max_ssd
  //                                                       : max_transfer (q > 0 ? 2 : 1)·buffer )
  //          token-embedding buffers from the decode step and the LAST prefill chunk (R·rowBytes per chunk)
  // See docs/simulation-model.md §12 ("Memory tiers", "Overlap across tiers", "Token embedding off-device")
  return largest;
}

/** An all-zero {@link TierBytes}. */
function emptyTierBytes(): TierBytes {
  return { lookupTables: 0, routedExperts: 0, draftExperts: 0, tokenEmbeddings: 0 };
}

/** Sum of the categories of a {@link TierBytes}. Unit: bytes. */
export function tierBytesTotal(b: TierBytes): number {
  return b.lookupTables + b.routedExperts + b.draftExperts + b.tokenEmbeddings;
}

/**
 * Bytes one device of stage `s` places on each off-device tier, split by what they hold (design doc §2.7): the FULL
 * groups, since the device cache is a copy: all local experts of each offloaded MoE layer (`routedExperts`), each
 * off-device table shard (`lookupTables`); plus, for
 * an NVMe table with `scaleLayout 'pinned_host'`, its scales (`ceil(rows/tp)·scaleBytes`) on host RAM
 * (`lookupTables`) — on dedicated memory only: on unified memory (`hostRamIsDevice`) the scales are device bytes counted
 * in {@link stageWeightBytes}, and a `host_ram` group is an error (invariant). The draft's own experts are
 * `draftExperts`. Stage 0 with the token embedding off-device adds its shard `ceil(vocab/tp)·h·bpe(emb)` to its tier
 * (`tokenEmbeddings`). Unit: bytes per device.
 */
export function stageTierBreakdown(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
): Record<OffloadTier, TierBytes> {
  const bytes: Record<OffloadTier, TierBytes> = {
    host_ram: emptyTierBytes(),
    nvme: emptyTierBytes(),
  };
  const [start, end] = stageLayerRange(model.layers.length, ctx.parallel.pp, s);
  for (let i = start; i < end; i++) {
    const layer = model.layers[i];
    const experts = expertPlacementAt(ctx, i);
    const tables = tablePlacementAt(ctx, i);
    for (const block of layer.blocks) {
      if (block.kind === 'moe' && experts !== undefined) {
        // Formula: tier += E_loc·W_expert
        bytes[experts.tier].routedExperts += moeLocalExpertBytes(block, ctx);
      } else if (block.kind === 'ngram_embedding' && tables !== undefined) {
        const shard = ngramTableShardBytes(block, ctx);
        const { scaleBytes } = tableRowBytes(lookupTableFormat(ctx), block.rowDim);
        if (tables.scaleLayout === 'pinned_host' && scaleBytes > 0) {
          // Formula: nvme += shard − scales;  host_ram += ceil(rows/tp)·scaleBytes   (dedicated memory)
          // Formula: nvme += shard − scales;  the scales are device bytes             (unified: see stageWeightBytes)
          const scales = Math.ceil(block.tableRows / ctx.parallel.tp) * scaleBytes;
          bytes[tables.tier].lookupTables += shard - scales;
          if (ctx.offload?.hostRamIsDevice !== true) bytes.host_ram.lookupTables += scales;
        } else {
          // Formula: tier += ceil(rows/tp)·d·bpe(lookup)
          bytes[tables.tier].lookupTables += shard;
        }
      }
    }
  }
  const embeddings = ctx.offload?.embeddings;
  if (s === 0 && embeddings !== undefined) {
    // Formula: tier += ceil(vocab/tp)·h·bpe(emb)   (stage 0, token embedding off-device)
    // See docs/simulation-model.md §12 ("Token embedding off-device")
    bytes[embeddings.tier].tokenEmbeddings += embeddingShardBytes(model, ctx);
  }
  // Spec-dec with includeDraft: the drafter's own MoE experts (last stage) live on the tier of the one rule that sets
  // includeDraft (design §3.8).
  const draftExperts = draftExpertPlacement(ctx.offload);
  if (
    s === ctx.parallel.pp - 1 &&
    ctx.speculative !== undefined &&
    draftExperts !== undefined
  ) {
    for (const layer of draftLayerSpecs(ctx.speculative.draft)) {
      for (const block of layer.blocks) {
        // Formula: tier += E_loc·W_expert per drafter MoE block
        if (block.kind === 'moe')
          bytes[draftExperts.tier].draftExperts += moeLocalExpertBytes(block, ctx);
      }
    }
  }
  // Invariant: on unified memory host RAM is the device pool, so no group may land on a `host_ram` tier.
  if (ctx.offload?.hostRamIsDevice === true && tierBytesTotal(bytes.host_ram) > 0) {
    throw new Error('memory: invariant violated — a host_ram group on unified memory');
  }
  return bytes;
}

/**
 * Bytes one device of stage `s` places on each off-device tier: {@link stageTierBreakdown} summed over its
 * categories. Unit: bytes per device.
 */
export function stageTierBytes(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
): Record<OffloadTier, number> {
  const b = stageTierBreakdown(model, ctx, s);
  return { host_ram: tierBytesTotal(b.host_ram), nvme: tierBytesTotal(b.nvme) };
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
export function stageSnapshotStateBytesPerSeq(
  model: ModelSpec,
  ctx: LoweringContext,
  s: number,
): number {
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
 * On unified memory no `'host_ram'` group reaches here (it is device memory); {@link stageTierBreakdown} enforces it.
 */
export function tierMemory(
  model: ModelSpec,
  ctx: LoweringContext,
  devicesPerNode: number,
  host: ResolvedHost,
): TierMemory[] {
  const { tp, dp, pp } = ctx.parallel;
  const perStage = Array.from({ length: pp }, (_, s) => stageTierBreakdown(model, ctx, s));
  const ranks = tp * dp * pp;
  const nodes = Math.ceil(ranks / devicesPerNode);
  const tiers: TierMemory[] = [];
  for (const tier of ['host_ram', 'nvme'] as const) {
    let bytes = 0;
    let breakdown = emptyTierBytes();
    for (let node = 0; node < nodes; node++) {
      const nodeBreakdown = emptyTierBytes();
      for (let r = node * devicesPerNode; r < Math.min(ranks, (node + 1) * devicesPerNode); r++) {
        // Formula: bytes_node = Σ_{ranks r of the node} tierBytes(stage(r)),  stage(r) = floor(r / (tp·dp))
        const device = perStage[Math.floor(r / (tp * dp))][tier];
        for (const k of Object.keys(nodeBreakdown) as (keyof TierBytes)[])
          nodeBreakdown[k] += device[k];
      }
      const nodeBytes = tierBytesTotal(nodeBreakdown);
      // The fullest node: its breakdown is the one reported, so the parts always sum to `bytes`.
      if (nodeBytes > bytes) {
        bytes = nodeBytes;
        breakdown = nodeBreakdown;
      }
    }
    if (bytes === 0) continue;
    const spec = tier === 'host_ram' ? host.ram : host.ssd;
    const capacityBytes = spec?.capacityBytes ?? 0;
    tiers.push({ tier, bytes, breakdown, capacityBytes, fits: bytes <= capacityBytes });
  }
  return tiers;
}

/**
 * The memory-fit report: per stage weights + B_r · state + activation peak + runtime reserve + staging against the
 * GPU-usable capacity `cap = f · capacity` ({@link resolveGpuUsableFraction}), and the most sequences per replica that
 * would fit at this context length: `floor(min_s (cap − W_s − act_s − reserve − staging_s) / state_s)` (Infinity
 * when no stage keeps per-sequence state; 0 when the weights alone do not fit). The activation peak is taken at the
 * evaluated batch, so this is an estimate. With weights offloaded, the off-device tiers are checked too and enter
 * `fits`.
 */
export function memoryReport(input: MemoryInput): MemoryReport {
  const { model, ctx } = input;
  const gpuUsableFraction = input.gpuUsableFraction ?? 1;
  const physicalCapacityBytes = input.capacityBytes;
  // Formula: cap = f · capacity   (f = GPU-usable fraction; 1 on dedicated memory)
  // See docs/simulation-model.md §8 ("GPU-usable capacity")
  const capacityBytes = gpuUsableFraction * physicalCapacityBytes;
  const runtimeReserveBytes = input.runtimeReserveBytes ?? 0;
  const activationLowered = input.activationLowered ?? input.lowered;
  const perStage: DeviceMemory[] = [];
  let maxSequencesPerReplica = Infinity;
  for (let s = 0; s < ctx.parallel.pp; s++) {
    const weightBytes = stageWeightBytes(model, ctx, s);
    const lookupTableBytes = stageLookupTableBytes(model, ctx, s);
    const statePerSeqBytes = stageSequenceStateBytesPerSeq(model, ctx, s, input.contextTokens);
    const activationPeakBytes = stageActivationPeakBytes(activationLowered, s);
    // Deployment terms (design doc §2.7, §3.3): 0 with nothing offloaded / spec-dec off.
    // Token-embedding rows are staged per prefill chunk (the activation phases), other groups per costed phase.
    const stagingBytes = stageStagingBytes(input.lowered, s, activationLowered);
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
        draftBytes =
          draftResidentWeightBytes(model, spec.draft, ctx) +
          input.sequencesPerReplica * draftStatePerSeqBytes;
      }
      // Formula: specTransient_s = B_r·(k + 1)·snap_s
      snapshotPerSeqBytes = spec.stateSnapshots * stageSnapshotStateBytesPerSeq(model, ctx, s);
    }
    const speculativeTransientBytes = input.sequencesPerReplica * snapshotPerSeqBytes;
    // Formula: total_s = W_s + B_r·state_s + act_s + reserve + staging_s + draft_s + specTransient_s;  fits = total_s ≤ cap
    // See docs/simulation-model.md §8 ("Memory fit"), §12 ("Memory tiers"), §13 ("Draft memory")
    const sequenceStateBytes = input.sequencesPerReplica * statePerSeqBytes;
    const totalBytes =
      weightBytes +
      sequenceStateBytes +
      activationPeakBytes +
      runtimeReserveBytes +
      stagingBytes +
      draftBytes +
      speculativeTransientBytes;
    perStage.push({
      stage: s,
      weightBytes,
      lookupTableBytes,
      sequenceStateBytes,
      activationPeakBytes,
      stagingBytes,
      draftBytes,
      speculativeTransientBytes,
      runtimeReserveBytes,
      totalBytes,
      capacityBytes,
      physicalCapacityBytes,
      fits: totalBytes <= capacityBytes,
    });
    // Per-sequence bytes that grow with B_r, and the fixed bytes besides them (spec-dec terms 0 when off).
    const perSeqBytes =
      spec === undefined
        ? statePerSeqBytes
        : statePerSeqBytes + draftStatePerSeqBytes + snapshotPerSeqBytes;
    const fixedDraftBytes = draftBytes - input.sequencesPerReplica * draftStatePerSeqBytes;
    if (perSeqBytes > 0) {
      // Formula: maxSeq = floor(min_s (cap − W_s − act_s − reserve − staging_s − W_draft) / (state_s + S_draft + (k+1)·snap_s))
      // See docs/simulation-model.md §8 ("Memory fit"), §13 ("Draft memory")
      const fixedBytes = weightBytes + activationPeakBytes + runtimeReserveBytes + stagingBytes;
      const freeBytes =
        spec === undefined
          ? capacityBytes - fixedBytes
          : capacityBytes - fixedBytes - fixedDraftBytes;
      maxSequencesPerReplica = Math.min(
        maxSequencesPerReplica,
        Math.max(0, Math.floor(freeBytes / perSeqBytes)),
      );
    } else if (
      weightBytes + activationPeakBytes + runtimeReserveBytes + stagingBytes + fixedDraftBytes >
      capacityBytes
    ) {
      maxSequencesPerReplica = 0;
    }
  }
  const tiers =
    input.host === undefined ? [] : tierMemory(model, ctx, input.devicesPerNode ?? 1, input.host);
  // Formula: fits = ∀ s fits_s ∧ ∀ tiers (bytes ≤ capacity)
  const fits = perStage.every((d) => d.fits) && tiers.every((t) => t.fits);
  return {
    kind: input.kind,
    gpuUsableFraction,
    perStage,
    fits,
    maxSequencesPerReplica,
    tiers,
  };
}

/** Bytes as decimal GB with one decimal, for warnings (`96.0 GB`). */
function gbText(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/**
 * Warning lines of a unified-memory report: the GPU-usable share the fit used, and, for a stage that overflows that
 * share but not the installed memory, that it fits only with a raised OS limit ("fits by default" vs "fits with
 * override", docs §8 "GPU-usable capacity").
 */
export function unifiedMemoryWarnings(report: MemoryReport): string[] {
  const first = report.perStage[0];
  if (first === undefined) return [];
  const percent = Math.round(report.gpuUsableFraction * 1000) / 10;
  const lines = [
    `memory: unified memory — the GPU may use ${percent}% of it (${gbText(first.capacityBytes)} of ` +
      `${gbText(first.physicalCapacityBytes)}); the rest stays with the OS and CPU`,
  ];
  const overLimit = report.perStage.find((d) => !d.fits && d.totalBytes <= d.physicalCapacityBytes);
  if (overLimit !== undefined) {
    lines.push(
      `memory: needs ${gbText(overLimit.totalBytes)}, above the GPU limit of ${gbText(overLimit.capacityBytes)}; ` +
        'it fits only with a raised limit (macOS iogpu.wired_limit_mb, Linux ttm.pages_limit)',
    );
  }
  return lines;
}
