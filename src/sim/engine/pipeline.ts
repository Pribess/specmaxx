/**
 * Stage time (with the communication-overlap assumption), pipeline schedule (PP microbatches) and data
 * parallelism → the throughput and latency of one phase.
 * See docs/simulation-model.md §7 ("Stage time and overlap", "Pipeline and DP").
 *
 * Notation: per stage s and microbatch, compute_s = Σ non-collective op times, lat_s = Σ collective α terms,
 * bw_s = Σ collective β terms, p2p_s = the activation send to stage s+1; m = microbatches in flight per replica,
 * B_r = sequences per DP replica.
 */

import type { LayerResult, OpResult, PhaseResult, SimOptions, StageResult } from './types';

/** Sums a layer's op times (no overlap): `sec = Σ op.sec`. */
export function layerResult(name: string, layerIndex: number | null, ops: OpResult[]): LayerResult {
  // Formula: t_layer = Σ_ops t_op
  return { name, layerIndex, ops, sec: ops.reduce((s, o) => s + o.sec, 0) };
}

/**
 * Time of one stage for one microbatch on one device, under the overlap assumption:
 * - `'none'`:  t_s = compute_s + lat_s + bw_s + p2p_s
 * - `'ideal'`: t_s = lat_s + max(compute_s, bw_s)   (β hides behind independent work, α is on the dependency
 *   chain; the PP send is not in t_s: it lengthens the latency path and, as a link resource, bounds the cycle;
 *   see {@link phaseResult})
 */
export function stageResult(
  index: number,
  layers: LayerResult[],
  sendToNext: OpResult | undefined,
  overlap: SimOptions['overlap'],
): StageResult {
  const ops = layers.flatMap((l) => l.ops);
  const computeSec = ops.filter((o) => o.op.kind !== 'collective').reduce((s, o) => s + o.sec, 0);
  const commLatencySec = ops.reduce((s, o) => s + o.commLatencySec, 0);
  const commBandwidthSec = ops.reduce((s, o) => s + o.commBandwidthSec, 0);
  const p2pSec = sendToNext?.sec ?? 0;
  let sec: number;
  switch (overlap) {
    case 'none':
      // Formula: t_s = compute_s + lat_s + bw_s + p2p_s
      // See docs/simulation-model.md §7 ("Stage time and overlap")
      sec = computeSec + commLatencySec + commBandwidthSec + p2pSec;
      break;
    case 'ideal':
      // Formula: t_s = lat_s + max(compute_s, bw_s)
      // See docs/simulation-model.md §7 ("Stage time and overlap")
      sec = commLatencySec + Math.max(computeSec, commBandwidthSec);
      break;
    default: {
      const _never: never = overlap;
      throw new Error(`unknown overlap ${String(_never)}`);
    }
  }
  // Formula: exposedComm_s = t_s − compute_s
  const exposedCommSec = sec - computeSec;
  const stage: StageResult = { index, layers, computeSec, commLatencySec, commBandwidthSec, exposedCommSec, sec };
  if (sendToNext !== undefined) stage.sendToNext = sendToNext;
  return stage;
}

/** Inputs of {@link phaseResult} besides the stages. */
export interface PhaseShape {
  /** Which phase. */
  phase: 'prefill' | 'decode';
  /** DP replicas. Count. */
  dp: number;
  /** Sequences per replica `B_r = batch / dp`. Count. */
  sequencesPerReplica: number;
  /** Microbatches in flight per replica `m = min(pp, B_r)`. Count. */
  microbatches: number;
  /** Sequences per microbatch `b_mb = ceil(B_r / m)`. Count. */
  microbatchSize: number;
  /** Prompt length (prefill tokens per sequence). Unit: tokens. */
  promptLen: number;
  /** Overlap assumption (decides whether the PP sends are already inside t_s). */
  overlap: SimOptions['overlap'];
}

/**
 * Microbatching of one replica: `m = min(pp, B_r)` microbatches of `b_mb = ceil(B_r / m)` sequences.
 * Counts (no unit).
 */
export function microbatching(pp: number, sequencesPerReplica: number): { microbatches: number; microbatchSize: number } {
  // Formula: m = min(pp, B_r);  b_mb = ceil(B_r / m)
  // See docs/simulation-model.md §7 ("Pipeline and DP")
  const microbatches = Math.min(pp, sequencesPerReplica);
  return { microbatches, microbatchSize: Math.ceil(sequencesPerReplica / microbatches) };
}

/**
 * Throughput and latency of one phase from its per-microbatch stage times.
 *
 * Every pipeline stage AND every PP boundary link is a resource that each of the m microbatches occupies once per
 * cycle: stage s for t_s, the link after stage s for its bandwidth (β) time `βp2p_s = bytes / B` (α is
 * latency, not occupancy). Even when the sends overlap compute (`'ideal'`), a link cannot move more bytes than
 * its bandwidth allows, so the bottleneck per microbatch is
 *   b = max(max_s t_s, max_s βp2p_s).
 * Under `'none'` each t_s already contains its send, so the β term never wins there.
 *
 * Decode (every sequence makes one token per cycle; a microbatch cannot start its next step before its
 * previous one left the last stage):
 *   cycle = max(m · b, Σ_s t_s + Σ p2p);  tokens/s = dp · B_r / cycle;  per sequence 1/cycle; latency = cycle.
 * Prefill (a steady stream of replica batches; the bottleneck resource bounds it):
 *   cycle = m · b;  tokens/s = dp · B_r · promptLen / cycle;
 *   latency (TTFT of a replica batch) = Σ t_s + Σ p2p + (m − 1) · b;  per sequence promptLen / latency.
 * Σ p2p is the PP sends' full time (α + β) when `overlap = 'ideal'` (they are not inside t_s); 0 under
 * `'none'`, where each t_s already contains its send.
 */
export function phaseResult(stages: StageResult[], shape: PhaseShape): PhaseResult {
  const m = shape.microbatches;
  const maxStageSec = Math.max(...stages.map((s) => s.sec));
  // Formula: βp2p_s = bandwidth (β) time of the send after stage s, per microbatch
  const maxLinkServiceSec = Math.max(0, ...stages.map((s) => s.sendToNext?.commBandwidthSec ?? 0));
  // Formula: b = max(max_s t_s, max_s βp2p_s)   (bottleneck resource time per microbatch)
  // See docs/simulation-model.md §7 ("Pipeline and DP")
  const bottleneckSec = Math.max(maxStageSec, maxLinkServiceSec);
  const sumStageSec = stages.reduce((s, st) => s + st.sec, 0);
  const p2pOnPathSec =
    shape.overlap === 'ideal' ? stages.reduce((s, st) => s + (st.sendToNext?.sec ?? 0), 0) : 0;
  const base = { phase: shape.phase, microbatches: m, microbatchSize: shape.microbatchSize, stages };
  switch (shape.phase) {
    case 'decode': {
      // Formula: cycle = max(m·b, Σ t_s + Σ p2p); tokens/s = dp·B_r/cycle; perSeq = 1/cycle
      // See docs/simulation-model.md §7 ("Pipeline and DP")
      const cycleSec = Math.max(m * bottleneckSec, sumStageSec + p2pOnPathSec);
      return {
        ...base,
        cycleSec,
        latencySec: cycleSec,
        tokensPerSec: (shape.dp * shape.sequencesPerReplica) / cycleSec,
        perSequenceTokensPerSec: 1 / cycleSec,
      };
    }
    case 'prefill': {
      // Formula: cycle = m·b; tokens/s = dp·B_r·promptLen/cycle
      // Formula: latency = Σ t_s + Σ p2p + (m−1)·b; perSeq = promptLen/latency
      // See docs/simulation-model.md §7 ("Pipeline and DP")
      const cycleSec = m * bottleneckSec;
      const latencySec = sumStageSec + p2pOnPathSec + (m - 1) * bottleneckSec;
      return {
        ...base,
        cycleSec,
        latencySec,
        tokensPerSec: (shape.dp * shape.sequencesPerReplica * shape.promptLen) / cycleSec,
        perSequenceTokensPerSec: shape.promptLen / latencySec,
      };
    }
    default: {
      const _never: never = shape.phase;
      throw new Error(`unknown phase ${String(_never)}`);
    }
  }
}
