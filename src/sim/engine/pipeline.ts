/**
 * Stage time (with the communication-overlap assumption), pipeline schedule (PP microbatches) and data
 * parallelism → the throughput and latency of one phase.
 * See docs/simulation-model.md §7 ("Stage time and overlap", "Pipeline and DP").
 *
 * Notation: per stage s and microbatch, compute_s = Σ compute-op times (every op but collectives and off-device
 * transfers), lat_s = Σ collective α terms, bw_s = Σ collective β terms, p2p_s = the activation send to stage s+1,
 * xfer_s = Σ transfer-op times (design doc §2.6); m = microbatches in flight per replica, B_r = sequences per DP
 * replica.
 */

import type { LayerResult, OpResult, PhaseResult, SimOptions, StageResult } from './types';

/** Sums a layer's op times (no overlap): `sec = Σ op.sec`. */
export function layerResult(name: string, layerIndex: number | null, ops: OpResult[]): LayerResult {
  // Formula: t_layer = Σ_ops t_op
  return { name, layerIndex, ops, sec: ops.reduce((s, o) => s + o.sec, 0) };
}

/**
 * Time of one stage for one microbatch on one device. First the core time under the overlap assumption:
 * - `'none'`:  t_core,s = compute_s + lat_s + bw_s + p2p_s
 * - `'ideal'`: t_core,s = lat_s + max(compute_s, bw_s)   (β hides behind independent work, α is on the dependency
 *   chain; the PP send is not in t_s: it lengthens the latency path and, as a link resource, bounds the cycle;
 *   see {@link phaseResult})
 * Then the off-device transfers under `offloadOverlap` (design doc §2.6):
 * - `'ideal'`: t_s = lat_s + max(core_s, xfer_s)   (core_s = max(compute_s, bw_s) under `overlap 'ideal'`,
 *   compute_s + bw_s + p2p_s under `'none'`; prefetch hides xfer behind the core)
 * - `'none'`:  t_s = t_core,s + xfer_s
 * With xfer_s = 0 both give t_s = t_core,s exactly.
 */
export function stageResult(
  index: number,
  layers: LayerResult[],
  sendToNext: OpResult | undefined,
  overlap: SimOptions['overlap'],
  offloadOverlap: SimOptions['offloadOverlap'] = 'ideal',
): StageResult {
  const ops = layers.flatMap((l) => l.ops);
  const computeSec = ops
    .filter((o) => o.op.kind !== 'collective' && o.op.kind !== 'transfer')
    .reduce((s, o) => s + o.sec, 0);
  // Formula: xfer_s = Σ transfer op sec   (design doc §2.6)
  const transferSec = ops.filter((o) => o.op.kind === 'transfer').reduce((s, o) => s + o.sec, 0);
  const commLatencySec = ops.reduce((s, o) => s + o.commLatencySec, 0);
  const commBandwidthSec = ops.reduce((s, o) => s + o.commBandwidthSec, 0);
  const p2pSec = sendToNext?.sec ?? 0;
  // t_core,s keeps today's expression (and its floating-point order) so that xfer_s = 0 is bit-identical.
  let coreSec: number;
  let coreTermSec: number;
  switch (overlap) {
    case 'none':
      // Formula: t_core,s = compute_s + lat_s + bw_s + p2p_s;  core_s = compute_s + bw_s + p2p_s
      // See docs/simulation-model.md §7 ("Stage time and overlap")
      coreSec = computeSec + commLatencySec + commBandwidthSec + p2pSec;
      coreTermSec = computeSec + commBandwidthSec + p2pSec;
      break;
    case 'ideal':
      // Formula: t_core,s = lat_s + max(compute_s, bw_s);  core_s = max(compute_s, bw_s)
      // See docs/simulation-model.md §7 ("Stage time and overlap")
      coreTermSec = Math.max(computeSec, commBandwidthSec);
      coreSec = commLatencySec + coreTermSec;
      break;
    default: {
      const _never: never = overlap;
      throw new Error(`unknown overlap ${String(_never)}`);
    }
  }
  let sec: number;
  switch (offloadOverlap) {
    case 'none':
      // Formula: t_s = t_core,s + xfer_s
      // See docs/simulation-model.md §7 ("Stage time and overlap"), design doc §2.6
      sec = transferSec === 0 ? coreSec : coreSec + transferSec;
      break;
    case 'ideal':
      // Formula: t_s = lat_s + max(core_s, xfer_s)
      // See docs/simulation-model.md §7 ("Stage time and overlap"), design doc §2.6
      sec = transferSec === 0 ? coreSec : commLatencySec + Math.max(coreTermSec, transferSec);
      break;
    default: {
      const _never: never = offloadOverlap;
      throw new Error(`unknown offload overlap ${String(_never)}`);
    }
  }
  // Formula: exposedComm_s = t_core,s − compute_s;  exposedTransfer_s = t_s − t_core,s
  const exposedCommSec = coreSec - computeSec;
  const exposedTransferSec = sec - coreSec;
  const stage: StageResult = {
    index,
    layers,
    computeSec,
    commLatencySec,
    commBandwidthSec,
    exposedCommSec,
    sec,
    transferSec,
    exposedTransferSec,
  };
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
  /**
   * Decode with speculative decoding: E[τ], tokens committed per sequence per step (design doc §3.5). Absent = 1
   * token per step (today's rule, bit for bit). Count (real, ≥ 1).
   */
  tokensPerStep?: number;
  /** Decode-only last-stage → stage-0 token-ID send; α + β seconds on the dependency path, never hidden. */
  loopBack?: OpResult;
}

/**
 * Microbatching of one replica: `m = min(pp, B_r)` microbatches of `b_mb = ceil(B_r / m)` sequences.
 * Counts (no unit).
 */
export function microbatching(
  pp: number,
  sequencesPerReplica: number,
): { microbatches: number; microbatchSize: number } {
  // Formula: m = min(pp, B_r);  b_mb = ceil(B_r / m)
  // See docs/simulation-model.md §7 ("Pipeline and DP")
  const microbatches = Math.min(pp, sequencesPerReplica);
  return { microbatches, microbatchSize: Math.ceil(sequencesPerReplica / microbatches) };
}

/**
 * Communication of a phase on its critical path, split into latency and bandwidth (display only; design doc §4.2).
 * Sums of the stage terms and the PP sends, so no rate depends on it:
 *   latency   = Σ_s lat_s + Σ α_p2p
 *   bandwidth = Σ_s bw_s + Σ β_p2p
 *   exposed   = Σ_s exposedComm_s + [overlap 'ideal'] Σ p2p   (under 'none' each send is already inside t_s,
 *               hence inside exposedComm_s)
 * The decode-only loopBack adds its α, β and full time to the respective sums in both overlap modes.
 * Unit: seconds.
 */
export function phaseComm(
  stages: StageResult[],
  overlap: SimOptions['overlap'],
  loopBack?: OpResult,
): PhaseResult['comm'] {
  // Formula: latency = Σ_s lat_s + Σ α_p2p;  bandwidth = Σ_s bw_s + Σ β_p2p
  // See docs/simulation-model.md §7 ("Comm latency vs bandwidth split")
  const latencySec = stages.reduce(
    (s, st) => s + st.commLatencySec + (st.sendToNext?.commLatencySec ?? 0),
    0,
  );
  const bandwidthSec = stages.reduce(
    (s, st) => s + st.commBandwidthSec + (st.sendToNext?.commBandwidthSec ?? 0),
    0,
  );
  // Formula: exposed = Σ_s exposedComm_s + [ideal] Σ p2p_s
  const sendsOnPathSec =
    overlap === 'ideal' ? stages.reduce((s, st) => s + (st.sendToNext?.sec ?? 0), 0) : 0;
  const exposedSec = stages.reduce((s, st) => s + st.exposedCommSec, 0) + sendsOnPathSec;
  return {
    latencySec: latencySec + (loopBack?.commLatencySec ?? 0),
    bandwidthSec: bandwidthSec + (loopBack?.commBandwidthSec ?? 0),
    exposedSec: exposedSec + (loopBack?.sec ?? 0),
  };
}

/**
 * Off-device transfers of a phase (design doc §2.5), or undefined when no stage has a transfer op:
 * `sec = Σ_s xfer_s`, `exposedSec = Σ_s exposedTransfer_s` (seconds), `bytes = Σ usefulBytes` (bytes over the host
 * link), `requests = Σ reads.count` (expected requests, real), for one microbatch through every stage.
 */
export function phaseTransfer(stages: StageResult[]): PhaseResult['transfer'] {
  let bytes = 0;
  let requests = 0;
  let any = false;
  for (const st of stages) {
    for (const layer of st.layers) {
      for (const r of layer.ops) {
        if (r.op.kind !== 'transfer') continue;
        any = true;
        // Formula: bytes = Σ usefulBytes;  requests = Σ_reads count
        bytes += r.op.usefulBytes;
        requests += r.op.reads.reduce((s, rd) => s + rd.count, 0);
      }
    }
  }
  if (!any) return undefined;
  const sec = stages.reduce((s, st) => s + st.transferSec, 0);
  const exposedSec = stages.reduce((s, st) => s + st.exposedTransferSec, 0);
  return { sec, exposedSec, bytes, requests };
}

/**
 * Throughput and latency of one phase from its per-microbatch stage times.
 *
 * Every pipeline stage AND every PP boundary link is a resource that each of the m microbatches occupies once per
 * cycle: stage s for t_s, the link after stage s for its bandwidth (β) time `βp2p_s = bytes / B` (α is
 * latency, not occupancy). Even when the sends overlap compute (`'ideal'`), a link cannot move more bytes than
 * its bandwidth allows, so the bottleneck per microbatch is
 *   b = max(max_s t_s, max_s βp2p_s, β_return).
 * Under 'none' each t_s contains its forward send. Decode return β is a separate link resource.
 *
 * Decode (every sequence makes one token per cycle; a microbatch cannot start its next step before its
 * sampled IDs have returned to stage 0):
 *   cycle = max(m · b, Σ_s t_s + Σ p2p + return_p2p); tokens/s = dp · B_r / cycle; per sequence 1/cycle.
 *   return_p2p = α_return + β_return is present under both overlap modes (G2); latency = cycle.
 *   With speculative decoding each step commits E[τ] tokens per sequence: tokens/s = dp · B_r · E[τ] / cycle;
 *   per sequence E[τ]/cycle; latency (time per output token) = cycle / E[τ].
 * Prefill (a steady stream of replica batches; the bottleneck resource bounds it):
 *   cycle = m · b;  tokens/s = dp · B_r · promptLen / cycle;
 *   latency (TTFT of a replica batch) = Σ t_s + Σ p2p + (m − 1) · b;  per sequence promptLen / latency.
 * Σ p2p is the PP sends' full time (α + β) when `overlap = 'ideal'` (they are not inside t_s); 0 under
 * 'none', where each t_s already contains its forward send. Return is never inside a stage.
 */
export function phaseResult(stages: StageResult[], shape: PhaseShape): PhaseResult {
  const m = shape.microbatches;
  const maxStageSec = Math.max(...stages.map((s) => s.sec));
  // Formula: βp2p_s = bandwidth (β) time of the send after stage s, per microbatch
  const loopBack = shape.phase === 'decode' ? shape.loopBack : undefined;
  const maxLinkServiceSec = Math.max(
    loopBack?.commBandwidthSec ?? 0,
    ...stages.map((s) => s.sendToNext?.commBandwidthSec ?? 0),
  );
  // Formula: b = max(max_s t_s, max_s βp2p_s)   (bottleneck resource time per microbatch)
  // See docs/simulation-model.md §7 ("Pipeline and DP")
  const bottleneckSec = Math.max(maxStageSec, maxLinkServiceSec);
  const sumStageSec = stages.reduce((s, st) => s + st.sec, 0);
  const p2pOnPathSec =
    shape.overlap === 'ideal' ? stages.reduce((s, st) => s + (st.sendToNext?.sec ?? 0), 0) : 0;
  const base: Pick<
    PhaseResult,
    'phase' | 'microbatches' | 'microbatchSize' | 'stages' | 'comm' | 'transfer' | 'loopBack'
  > = {
    phase: shape.phase,
    microbatches: m,
    microbatchSize: shape.microbatchSize,
    stages,
    comm: phaseComm(stages, shape.overlap, loopBack),
  };
  if (loopBack !== undefined) base.loopBack = loopBack;
  const transfer = phaseTransfer(stages);
  if (transfer !== undefined) base.transfer = transfer;
  switch (shape.phase) {
    case 'decode': {
      // Formula: cycle = max(m·b, Σ t_s + Σ forward_p2p + return_p2p); tokens/s = dp·B_r/cycle
      // Formula (spec-dec): tokens/s = dp·B_r·E[τ]/cycle; perSeq = E[τ]/cycle; latency = cycle/E[τ]
      // See docs/simulation-model.md §7 ("Pipeline and DP"), §13 ("Accepted tokens")
      // Formula (G2): cycle = max(m·b, Σ t_s + Σ forward_p2p + return_p2p).
      const cycleSec = Math.max(
        m * bottleneckSec,
        sumStageSec + p2pOnPathSec + (loopBack?.sec ?? 0),
      );
      const E = shape.tokensPerStep;
      return {
        ...base,
        cycleSec,
        latencySec: E === undefined ? cycleSec : cycleSec / E,
        tokensPerSec:
          E === undefined
            ? (shape.dp * shape.sequencesPerReplica) / cycleSec
            : (shape.dp * shape.sequencesPerReplica * E) / cycleSec,
        perSequenceTokensPerSec: E === undefined ? 1 / cycleSec : E / cycleSec,
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
