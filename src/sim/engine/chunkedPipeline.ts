/**
 * Chunked pipeline prefill (CPP, `ParallelConfig.chunkedPipelinePrefill`; Mooncake's "chunked pipeline parallelism",
 * arXiv 2407.00079 §5.1): with pp > 1, each microbatch's prompt is split into K chunks of C new tokens per sequence
 * that flow through the pipeline stages one after another, so stage s + 1 works on chunk j while stage s runs chunk
 * j + 1. A chunk is the prefix-cache shape of prefill (`qLen < kvLen`: the chunk's tokens over every earlier key), so
 * every block kind lowers it as it is. This module holds the chunk partition, the schedule (time to first token and
 * cycle) and the merge of the per-chunk stage results; `simulate.ts` lowers and costs the chunks.
 * See docs/simulation-model.md §7 "Chunked pipeline prefill".
 *
 * Notation: P = prompt tokens per sequence, cached = floor(P·h) of them already in the prefix cache, P_new = P − cached
 * computed, C = chunk size (`Workload.prefillChunkTokens`), K = chunks per microbatch, q_j / kv_j = new tokens / visible
 * keys of chunk j (per sequence), m = microbatches per replica, pp = stages; for one microbatch: t_{s,j} = time of stage
 * s on chunk j (`StageResult.sec`), p_{s,j} = full time of the send after stage s for chunk j (`sendToNext.sec`, α + β),
 * β_{s,j} = its bandwidth term (`sendToNext.commBandwidthSec`), f_pp = the PP overlap fraction. Times in seconds.
 */

import type { ByteBreakdown } from '../ir/metrics';
import type { ParallelConfig } from '../ir/model';
import { DEFAULT_PREFILL_CHUNK_TOKENS } from './memory';
import { layerResult, phaseComm, phaseTransfer, type PhaseShape } from './pipeline';
import {
  cachedPrefixTokens,
  type LayerResult,
  type OpResult,
  type PhaseResult,
  type StageResult,
  type TransferResourceUse,
  type Workload,
} from './types';

/** The warning of a run with chunked pipeline prefill on and one pipeline stage: nothing to pipeline, ignored. */
export const CHUNKED_PIPELINE_SINGLE_STAGE_WARNING =
  'chunked pipeline prefill: pp = 1, one pipeline stage, nothing to pipeline; ignored';

/** One chunk of a prompt under chunked pipeline prefill, per sequence. */
export interface PrefillChunk {
  /** q_j: new prompt tokens of the chunk (the chunk's queries). Unit: tokens. */
  qLen: number;
  /** kv_j: keys the chunk's last query sees: the cached prefix, every earlier chunk and this one. Unit: tokens. */
  kvLen: number;
}

/**
 * The chunks of one prompt of P = `promptLen` tokens, of which `cachedTokens` are in the prefix cache, with at most
 * C = `chunkTokens` new tokens per chunk:
 *   P_new = P − cached;  K = ceil(P_new / C)
 *   q_j = min(C, P_new − j·C);  kv_j = cached + j·C + q_j     (j = 0..K−1)
 * So Σ_j q_j = P_new and kv_{K−1} = P (the last chunk sees the whole prompt). Unit: tokens.
 */
export function prefillPipelineChunks(
  promptLen: number,
  cachedTokens: number,
  chunkTokens: number,
): PrefillChunk[] {
  // Formula: P_new = P − cached;  K = ceil(P_new / C)
  // See docs/simulation-model.md §7 "Chunked pipeline prefill"
  const newTokens = promptLen - cachedTokens;
  const chunkCount = Math.ceil(newTokens / chunkTokens);
  const chunks: PrefillChunk[] = [];
  for (let j = 0; j < chunkCount; j++) {
    // Formula: q_j = min(C, P_new − j·C);  kv_j = cached + j·C + q_j
    const qLen = Math.min(chunkTokens, newTokens - j * chunkTokens);
    chunks.push({ qLen, kvLen: cachedTokens + j * chunkTokens + qLen });
  }
  return chunks;
}

/** What chunked pipeline prefill does in one run ({@link chunkedPipelinePlan}). */
export interface ChunkedPipelinePlan {
  /** C: new tokens per sequence per chunk. Unit: tokens. */
  chunkTokens: number;
  /** The K ≥ 2 chunks of each sequence's prompt, in order. */
  chunks: PrefillChunk[];
}

/**
 * The chunk plan of a run, or undefined when prefill runs as whole-prompt microbatches (the path of before):
 * chunked pipeline prefill applies only when `parallel.chunkedPipelinePrefill === true`, pp > 1 and the prompt has
 * K > 1 chunks (P_new > C). C = `workload.prefillChunkTokens`, default {@link DEFAULT_PREFILL_CHUNK_TOKENS}.
 */
export function chunkedPipelinePlan(
  parallel: Pick<ParallelConfig, 'pp' | 'chunkedPipelinePrefill'>,
  workload: Pick<Workload, 'promptLen' | 'prefixCacheHitRatio' | 'prefillChunkTokens'>,
): ChunkedPipelinePlan | undefined {
  if (parallel.chunkedPipelinePrefill !== true || parallel.pp <= 1) return undefined;
  const chunkTokens = workload.prefillChunkTokens ?? DEFAULT_PREFILL_CHUNK_TOKENS;
  const chunks = prefillPipelineChunks(
    workload.promptLen,
    cachedPrefixTokens(workload),
    chunkTokens,
  );
  return chunks.length > 1 ? { chunkTokens, chunks } : undefined;
}

/** The schedule inputs of one chunk of one microbatch: per stage time and per boundary send terms. Seconds. */
export interface ChunkStageTimes {
  /** t_{s,j} per stage s = 0..pp−1 (`StageResult.sec`, which holds the share (1 − f_pp) of its send). Unit: seconds. */
  stageSec: number[];
  /** p_{s,j}: full time (α + β) of the send after stage s, s = 0..pp−2. Unit: seconds. */
  sendSec: number[];
  /** β_{s,j}: bandwidth term of the send after stage s (the link's occupancy), s = 0..pp−2. Unit: seconds. */
  sendBandwidthSec: number[];
}

/** {@link ChunkStageTimes} of one chunk's stage results (every stage but the last has a send). */
export function chunkStageTimes(stages: readonly StageResult[]): ChunkStageTimes {
  const sends = stages.slice(0, -1).map((st) => st.sendToNext);
  return {
    stageSec: stages.map((st) => st.sec),
    sendSec: sends.map((op) => op?.sec ?? 0),
    sendBandwidthSec: sends.map((op) => op?.commBandwidthSec ?? 0),
  };
}

/**
 * Time to first token of a replica batch under chunked pipeline prefill: when the last chunk of the last microbatch
 * leaves the last stage. Each stage and each boundary link is a machine that serves the units u = (i, j) (microbatch
 * i = 0..m−1, chunk j = 0..K−1) in order, microbatch-major, chunk-minor; `chunks[j]` gives unit (i, j)'s times, the
 * same for every microbatch. The link after stage s is busy for λ_s(u) = f_pp·β_{s,j} (the occupancy outside the
 * stage) and then delays the chunk by δ_s(u) = f_pp·(p_{s,j} − β_{s,j}) (its α terms: latency, no occupancy). The
 * share (1 − f_pp) of each send is already inside t_{s,j}. With F(u, s) = when unit u leaves stage s and G(u, s) =
 * when its send after stage s has left the link (F(−1, ·) = G(−1, ·) = 0):
 *   F(u, 0)   = F(u−1, 0) + t_0(u)
 *   G(u, s)   = max(G(u−1, s), F(u, s)) + λ_s(u)
 *   F(u, s+1) = max(F(u−1, s+1), G(u, s) + δ_s(u)) + t_{s+1}(u)
 *   TTFT      = F(last unit, pp−1)
 * With K = 1 this is the whole-prompt formula `Σ_s t_s + Σ_s f_pp·p_s + (m − 1)·b` of `pipeline.ts: phaseResult`
 * (at f_pp ∈ {0, 1}; up to rounding). With equal chunk times t on every stage and no sends it is (K + pp − 1)·t,
 * against pp·K·t unchunked. Unit: seconds. See docs/simulation-model.md §7 "Chunked pipeline prefill".
 */
export function chunkedPipelineLatencySec(
  chunks: readonly ChunkStageTimes[],
  microbatches: number,
  ppOverlap: number,
): number {
  const pp = chunks[0].stageSec.length;
  const leaveStage = new Array<number>(pp).fill(0); // F(u, s), updated in place unit by unit
  const leaveLink = new Array<number>(Math.max(pp - 1, 0)).fill(0); // G(u, s)
  for (let i = 0; i < microbatches; i++) {
    for (const unit of chunks) {
      // Formula: F(u, 0) = F(u−1, 0) + t_0(u)
      leaveStage[0] = leaveStage[0] + unit.stageSec[0];
      for (let s = 0; s < pp - 1; s++) {
        const occupancySec = ppOverlap * unit.sendBandwidthSec[s]; // λ_s(u) = f_pp·β
        const delaySec = ppOverlap * (unit.sendSec[s] - unit.sendBandwidthSec[s]); // δ_s(u) = f_pp·(p − β)
        // Formula: G(u, s) = max(G(u−1, s), F(u, s)) + λ_s(u)
        leaveLink[s] = Math.max(leaveLink[s], leaveStage[s]) + occupancySec;
        // Formula: F(u, s+1) = max(F(u−1, s+1), G(u, s) + δ_s(u)) + t_{s+1}(u)
        leaveStage[s + 1] =
          Math.max(leaveStage[s + 1], leaveLink[s] + delaySec) + unit.stageSec[s + 1];
      }
    }
  }
  return leaveStage[pp - 1];
}

/**
 * Pipeline cycle under chunked pipeline prefill: every stage runs all K chunks of each of the m microbatches, and
 * every boundary link carries all their sends, once per cycle, so the busiest resource bounds the steady stream:
 *   cycle = m · max( max_s Σ_j t_{s,j},  max_s Σ_j β_{s,j} )
 * With K = 1 this is `m·b` of `pipeline.ts: phaseResult`. Unit: seconds.
 */
export function chunkedPipelineCycleSec(
  chunks: readonly ChunkStageTimes[],
  microbatches: number,
): number {
  const pp = chunks[0].stageSec.length;
  let busiestSec = 0;
  for (let s = 0; s < pp; s++) {
    // Formula: Σ_j t_{s,j}   (stage s, one microbatch)
    const stageBusySec = chunks.reduce((sum, c) => sum + c.stageSec[s], 0);
    // Formula: Σ_j β_{s,j}   (the link after stage s, one microbatch; none after the last stage)
    const linkBusySec = s < pp - 1 ? chunks.reduce((sum, c) => sum + c.sendBandwidthSec[s], 0) : 0;
    busiestSec = Math.max(busiestSec, stageBusySec, linkBusySec);
  }
  // Formula: cycle = m · max(max_s Σ_j t_{s,j}, max_s Σ_j β_{s,j})
  // See docs/simulation-model.md §7 "Chunked pipeline prefill"
  return microbatches * busiestSec;
}

/** Field-wise sum of byte breakdowns. Unit: bytes. */
function sumBytes(parts: readonly ByteBreakdown[]): ByteBreakdown {
  const sum = (key: keyof ByteBreakdown): number => parts.reduce((s, b) => s + b[key], 0);
  return {
    weightBytes: sum('weightBytes'),
    activationReadBytes: sum('activationReadBytes'),
    activationWriteBytes: sum('activationWriteBytes'),
    kvReadBytes: sum('kvReadBytes'),
    kvWriteBytes: sum('kvWriteBytes'),
    stateReadBytes: sum('stateReadBytes'),
    stateWriteBytes: sum('stateWriteBytes'),
    totalBytes: sum('totalBytes'),
  };
}

/**
 * One op's results over the chunks as one result (they are the same op of the same layer, lowered at each chunk's
 * shape): its costs summed,
 *   flops = Σ_j flops_j;  bytes = Σ_j bytes_j;  sec = Σ_j sec_j;  computeSec, memorySec, α, β, transfer terms: Σ_j
 * and the op the last chunk's, with the sizes that add up summed: a collective's `messageBytes = Σ_j S_j` (for a PP
 * send: the whole microbatch's activations, Σ_j q_j = P_new), a transfer's `usefulBytes` and its `reads` (each chunk
 * re-reads its streamed weights). Every other field (the op's shape, `bound`, `efficiency`, the link level) is the last
 * chunk's. FLOP, bytes and seconds.
 */
export function sumChunkOps(results: readonly OpResult[]): OpResult {
  const last = results[results.length - 1];
  if (results.length === 1) return last;
  const sum = (f: (r: OpResult) => number): number => results.reduce((s, r) => s + f(r), 0);
  let op = last.op;
  if (op.kind === 'collective') {
    // Formula: S = Σ_j S_j
    op = { ...op, messageBytes: sum((r) => (r.op.kind === 'collective' ? r.op.messageBytes : 0)) };
  } else if (op.kind === 'transfer') {
    // Formula: usefulBytes = Σ_j usefulBytes_j;  reads = reads_0 ++ … ++ reads_{K−1}
    op = {
      ...op,
      usefulBytes: sum((r) => (r.op.kind === 'transfer' ? r.op.usefulBytes : 0)),
      reads: results.flatMap((r) => (r.op.kind === 'transfer' ? r.op.reads : [])),
    };
  }
  const merged: OpResult = {
    ...last,
    op,
    flops: sum((r) => r.flops),
    bytes: sumBytes(results.map((r) => r.bytes)),
    computeSec: sum((r) => r.computeSec),
    memorySec: sum((r) => r.memorySec),
    commLatencySec: sum((r) => r.commLatencySec),
    commBandwidthSec: sum((r) => r.commBandwidthSec),
    transferLatencySec: sum((r) => r.transferLatencySec),
    transferBandwidthSec: sum((r) => r.transferBandwidthSec),
    sec: sum((r) => r.sec),
  };
  if (last.softmaxSec !== undefined) merged.softmaxSec = sum((r) => r.softmaxSec ?? 0);
  if (last.transferResources !== undefined) {
    const use = results.map((r) => r.transferResources);
    const sumUse = (key: keyof TransferResourceUse): number =>
      use.reduce((s, u) => s + (u?.[key] ?? 0), 0);
    merged.transferResources = {
      linkBytes: sumUse('linkBytes'),
      linkSec: sumUse('linkSec'),
      hostRamBytes: sumUse('hostRamBytes'),
      hostRamSec: sumUse('hostRamSec'),
    };
  }
  return merged;
}

/**
 * The ops of one layer over the chunks, merged: the j-th occurrence of an op name with a given `bound` in every chunk
 * is one op ({@link sumChunkOps}), in first-seen order. Keying on the bound keeps the time of an op that is memory
 * bound in early chunks and compute bound in later ones (attention as kv_j grows) on the right side of the
 * compute/memory split. So a merged layer has about as many ops as one chunk's, not K times as many, and its Σ FLOP,
 * bytes and seconds are the chunks' totals.
 */
export function mergeChunkLayerOps(opsPerChunk: readonly (readonly OpResult[])[]): OpResult[] {
  const groups = new Map<string, OpResult[]>();
  for (const ops of opsPerChunk) {
    const seen = new Map<string, number>();
    for (const r of ops) {
      const base = `${r.op.name}|${r.bound}`;
      const occurrence = seen.get(base) ?? 0;
      seen.set(base, occurrence + 1);
      const key = `${base}#${occurrence}`;
      const group = groups.get(key);
      if (group === undefined) groups.set(key, [r]);
      else group.push(r);
    }
  }
  return [...groups.values()].map(sumChunkOps);
}

/** The time fields of a {@link StageResult} that {@link mergeChunkStages} sums over chunks. */
type StageTimeField =
  | 'computeSec'
  | 'commLatencySec'
  | 'commBandwidthSec'
  | 'exposedCommSec'
  | 'sec'
  | 'transferSec'
  | 'exposedTransferSec';

/**
 * The stage results of one microbatch from those of its K chunks (`chunkStages[j][s]`): per stage, every time field
 * summed over the chunks (`sec = Σ_j t_{s,j}`, `computeSec = Σ_j compute_{s,j}`, …), each layer's ops merged over the
 * chunks that have it ({@link mergeChunkLayerOps}; the last chunk alone has the `head` layer) and the sends summed
 * ({@link sumChunkOps}). So Σ op FLOP, bytes and times of a merged stage are the chunks' totals, and
 * `pipeline.ts: phaseComm, phaseTransfer` work on it unchanged.
 */
export function mergeChunkStages(chunkStages: readonly (readonly StageResult[])[]): StageResult[] {
  const last = chunkStages[chunkStages.length - 1];
  return last.map((lastStage, s) => {
    const perChunk = chunkStages.map((stages) => stages[s]);
    const sum = (key: StageTimeField): number => perChunk.reduce((total, st) => total + st[key], 0);
    const layersByName = perChunk.map((st) => new Map(st.layers.map((l) => [l.name, l])));
    // The last chunk has every layer (earlier chunks only lack the head), so its order is the merged order.
    const layers: LayerResult[] = lastStage.layers.map((layer) => {
      const opsPerChunk = layersByName.flatMap((byName) => {
        const l = byName.get(layer.name);
        return l === undefined ? [] : [l.ops];
      });
      return layerResult(layer.name, layer.layerIndex, mergeChunkLayerOps(opsPerChunk));
    });
    const merged: StageResult = {
      index: lastStage.index,
      layers,
      computeSec: sum('computeSec'),
      commLatencySec: sum('commLatencySec'),
      commBandwidthSec: sum('commBandwidthSec'),
      exposedCommSec: sum('exposedCommSec'),
      sec: sum('sec'),
      transferSec: sum('transferSec'),
      exposedTransferSec: sum('exposedTransferSec'),
    };
    const sends = perChunk.flatMap((st) => (st.sendToNext === undefined ? [] : [st.sendToNext]));
    if (sends.length > 0) merged.sendToNext = sumChunkOps(sends);
    return merged;
  });
}

/**
 * The prefill result of chunked pipeline prefill from the stage results of each chunk of one microbatch
 * (`chunkStages[j][s]`, chunk j costed as its own lowering with `seqs = b_mb`):
 *   latency (TTFT of a replica batch) = {@link chunkedPipelineLatencySec};  cycle = {@link chunkedPipelineCycleSec}
 *   tokens/s = dp·B_r·P / cycle;  computed tokens/s (prefix cache) = dp·B_r·P_new / cycle;  per sequence P / latency
 * `stages` are the merged per-microbatch sums ({@link mergeChunkStages}), from which `comm` and `transfer` follow as
 * for whole-prompt prefill. `chunkedPipeline` is left for the caller (it knows C and the unchunked result).
 * See docs/simulation-model.md §7 "Chunked pipeline prefill".
 */
export function chunkedPipelinePhase(
  chunkStages: readonly (readonly StageResult[])[],
  shape: PhaseShape,
): PhaseResult {
  if (shape.phase !== 'prefill') throw new Error('chunked pipeline: prefill only');
  const m = shape.microbatches;
  const times = chunkStages.map(chunkStageTimes);
  const latencySec = chunkedPipelineLatencySec(times, m, shape.overlap.pp);
  const cycleSec = chunkedPipelineCycleSec(times, m);
  const stages = mergeChunkStages(chunkStages);
  // Formula: tokens/s = dp·B_r·P/cycle;  computed = dp·B_r·P_new/cycle;  perSeq = P/latency
  // See docs/simulation-model.md §7 "Chunked pipeline prefill"
  const replicaSequences = shape.dp * shape.sequencesPerReplica;
  const hit = shape.computedTokens !== undefined && shape.computedTokens !== shape.promptLen;
  const result: PhaseResult = {
    phase: 'prefill',
    microbatches: m,
    microbatchSize: shape.microbatchSize,
    stages,
    comm: phaseComm(stages, shape.overlap),
    cycleSec,
    latencySec,
    tokensPerSec: (replicaSequences * shape.promptLen) / cycleSec,
    perSequenceTokensPerSec: shape.promptLen / latencySec,
  };
  if (hit) result.computedTokensPerSec = (replicaSequences * shape.computedTokens!) / cycleSec;
  const transfer = phaseTransfer(stages);
  if (transfer !== undefined) result.transfer = transfer;
  return result;
}
