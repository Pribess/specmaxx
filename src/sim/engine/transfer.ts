/**
 * The off-device transfer model of weight offloading: expected unique experts and misses per MoE layer, the hit-ratio
 * models, the one read rule every streamed group goes through (`transferTime`), and how many concurrent consumers
 * share a node's tier (`nStream`). Pure math on plain specs, no lowering or pipeline logic.
 * See docs/simulation-model.md §12 ("Offloading") and design doc docs/design/offload-specdec-latency.md §2.3–2.4.
 *
 * Conventions: every quantity is an EXPECTED value per device and microbatch step (real numbers; expected misses are
 * not rounded to whole requests); times in seconds, bytes in bytes, bandwidths in B/s (per direction for links).
 */

import type { HostMemorySpec, NvmeSpec, StorageTierSpec } from '../hw/types';
import type { TransferOp } from '../ir/ops';
import type { HitRatioModel } from './types';

// ---------------------------------------------------------------------------------------------------------------
// Expert prefetch (docs/simulation-model.md §12 "Expert prefetch")

/**
 * Default prefetch accuracy p (`OffloadConfig.prefetchAccuracy`): the share of a step's missed routed experts that
 * are predicted one layer ahead and copied while the previous layer computes. Dimensionless, in [0, 1].
 *
 * Estimated: the middle of what training-free next-gate prediction (the next layer's gate applied to the current
 * hidden state, same set size k, exact outputs) achieves one layer ahead:
 * - Mixtral 8×7B (8 experts, top-2): 0.84 (DAOP, arXiv 2501.10375 Fig. 5), ≈0.9 (AdapMoE, arXiv 2408.10284 §6),
 *   ≈0.85 (Eliseev & Mazur, arXiv 2312.17238 Fig. 2);
 * - Qwen1.5-MoE (60, top-4): 0.79 (Fate, arXiv 2502.12224 §4.2);
 * - Qwen2-57B-A14B (64, top-8): 0.67 (ProMoE, arXiv 2410.22134 §4.2);
 * - Qwen3-30B-A3B (128, top-8): ≈0.9 beyond the first two layers (Madan et al., arXiv 2603.19289 §3.3).
 * Learned predictors reach 0.85–0.95, and 0.97–0.99 only by fetching extra experts.
 */
export const DEFAULT_PREFETCH_ACCURACY = 0.8;

/**
 * True iff a stage's transfer ops prefetch from BOTH off-device tiers: some op with a prefetched share q > 0 reads
 * host RAM and some op with q > 0 reads the SSD. Only then may the prefetched shares of the two streams overlap
 * within the device (docs/simulation-model.md §12 "Overlap across tiers"); every other stage keeps the serial
 * transfer sum. The one condition shared by the stage time (`pipeline.ts: stageTransfer`) and the staging buffers
 * (`memory.ts: stageStagingBytes`), so the two cannot drift apart. Keyed on `source` only, whatever the group.
 */
export function prefetchesFromBothTiers(
  ops: readonly Pick<TransferOp, 'source' | 'prefetchedFraction'>[],
): boolean {
  // Formula: overlapped_s ⇔ ∃ op: source = host_ram ∧ q > 0  and  ∃ op: source = nvme ∧ q > 0
  const ram = ops.some((o) => o.source === 'host_ram' && o.prefetchedFraction > 0);
  const ssd = ops.some((o) => o.source === 'nvme' && o.prefetchedFraction > 0);
  return ram && ssd;
}

// ---------------------------------------------------------------------------------------------------------------
// Expert hit ratio and misses (design doc §2.4)

/**
 * Expected distinct experts touched in one MoE layer by `tokens` tokens under independent uniform top-k routing
 * (MoESD, arXiv 2505.19645 Eq. 7–8): `U(T) = E·(1 − (1 − k/E)^T)`. The one expected-unique rule of the engine: the
 * MoE lowering's active experts (the expert weights the grouped GEMMs read, `lower/moe.ts: neededExperts`) and the
 * offloaded experts' needed set both use it. An EXPECTATION under iid-uniform routing, not a bound: skewed or
 * correlated real routing touches fewer, and one batch can touch more than the mean.
 * `U(1) = k` exactly, `U(T) → E` as T grows, and `U ≤ min(E, T·k)` always. Unit: experts (expected, real).
 */
export function uniqueExperts(experts: number, topK: number, tokens: number): number {
  // Formula: U(T) = E·(1 − (1 − k/E)^T)
  //          evaluated as E − (E − k)·(1 − k/E)^(T−1), the same value, so U(1) = E − (E − k) = k has no rounding
  // See docs/simulation-model.md §5 ("Expected distinct experts"), §12 ("Expert hit ratio")
  return experts - (experts - topK) * (1 - topK / experts) ** (tokens - 1);
}

/**
 * Expected distinct LOCAL experts one device touches at expert parallelism ep > 1 (design doc §2.9): its `E_loc`
 * experts receive `A` (token, expert) assignments, sampled with replacement, so `U_dev = E_loc·(1 − (1 − 1/E_loc)^A)`.
 * An approximation of the exact without-replacement form, which the engine uses at ep = 1. Like {@link uniqueExperts},
 * shared by the MoE lowering and offload. `U_dev(1) = 1` exactly. Unit: experts.
 */
export function uniqueLocalExperts(localExperts: number, assignments: number): number {
  // Formula: U_dev = E_loc·(1 − (1 − 1/E_loc)^A)
  //          evaluated as E_loc − (E_loc − 1)·(1 − 1/E_loc)^(A−1), the same value, exact at A = 1
  // See docs/simulation-model.md §5 ("Expected distinct experts"), §12 ("Expert hit ratio")
  return localExperts - (localExperts - 1) * (1 - 1 / localExperts) ** (assignments - 1);
}

/** Exponent γ of the skewed hit-ratio fit `h = f^γ` (design doc §5.2): classic MoEs (E ≤ 64) 0.55, fine-grained 0.2. */
export const SKEWED_HIT_RATIO_EXPONENT = {
  skewed_classic: 0.55,
  skewed_fine_grained: 0.2,
} as const;

/**
 * The hit ratio h(f): the probability that a unique routed expert a token needs is already in device memory, per
 * unique expert per layer per step at batch 1 (design doc §2.4), as a function of the resident fraction f:
 * `uniform` h = f (exact for a balanced router); `skewed_classic` h = f^0.55; `skewed_fine_grained` h = f^0.2;
 * `custom` the user's h. Before the physical clamp of {@link expertHitsMisses}. Dimensionless, in [0, 1].
 */
export function hitRatio(model: HitRatioModel, residentFraction: number): number {
  const f = residentFraction;
  switch (model.kind) {
    case 'uniform':
      // Formula: h = f
      return f;
    case 'skewed_classic':
    case 'skewed_fine_grained':
      // Formula: h = f^γ   (γ 0.55 / 0.2)
      // See docs/simulation-model.md §12 ("Expert hit ratio")
      return f ** SKEWED_HIT_RATIO_EXPONENT[model.kind];
    case 'custom':
      return model.h;
    default: {
      const _never: never = model;
      throw new Error(`unknown hit-ratio model ${JSON.stringify(_never)}`);
    }
  }
}

/** Expected hits and misses of one offloaded MoE layer in one step (see {@link expertHitsMisses}). Unit: experts. */
export interface ExpertHitsMisses {
  /** Resident experts `C = f·E` (pinned or cached). Unit: experts (real). */
  residentExperts: number;
  /** Lower bound of the hits, `max(0, U + C − E)` (pigeonhole). Unit: experts. */
  minHits: number;
  /** Upper bound of the hits, `min(U, C)`. Unit: experts. */
  maxHits: number;
  /** Expected hits, `clamp(U·h, minHits, maxHits)`. Unit: experts. */
  hits: number;
  /** Expected missed (streamed) experts `M = U − hits`. Unit: experts. */
  misses: number;
}

/**
 * Hits and misses of one layer's `U` needed experts with `C = f·E` of `E` resident (design doc §2.4): the hit
 * ratio h is applied inside the physical bounds, `hits = clamp(U·h, max(0, U + C − E), min(U, C))`, `M = U − hits`.
 * So f = 0 ⇒ M = U; f = 1 ⇒ M = 0 whatever h; a full sweep U = E ⇒ hits = C exactly. All in experts (real).
 */
export function expertHitsMisses(
  neededExperts: number,
  hitRatioValue: number,
  residentFraction: number,
  experts: number,
): ExpertHitsMisses {
  const U = neededExperts;
  // Formula: C = f·E;  lo = max(0, U + C − E);  hi = min(U, C);  hits = clamp(U·h, lo, hi);  M = U − hits
  // See docs/simulation-model.md §12 ("Expert hit ratio")
  const residentExperts = residentFraction * experts;
  // Every expert resident (f = 1): lo = U exactly (not U + C − E, which can round below U), so M = 0.
  const minHits = residentExperts >= experts ? U : Math.max(0, U + residentExperts - experts);
  const maxHits = Math.min(U, residentExperts);
  const hits = Math.min(maxHits, Math.max(minHits, U * hitRatioValue));
  return { residentExperts, minHits, maxHits, hits, misses: U - hits };
}

// ---------------------------------------------------------------------------------------------------------------
// The read rule (design doc §2.3)

/**
 * Everything a transfer op of one device needs to be costed: the source tier, the hops after it, the queue depth
 * and the tier's sharing. Plain data, built per stage and tier by `simulate`.
 */
export interface TransferPath {
  /** The tier the bytes are read from (host RAM or the node's SSD). */
  tier: StorageTierSpec;
  /** Opt-in NVMe read-size scenario; absent preserves the datasheet B/s ceiling. */
  nvmeSustainedRead?: {
    /** Low-QD / deep-queue rate ratio, dimensionless in (0, 1]. */
    fraction: number;
    /** Individual read size at which the scenario starts. Unit: bytes. */
    blockBytes: number;
  };
  /**
   * The device's host link (PCIe), per direction. Absent on unified memory, where the tier writes straight into the
   * device pool (α_link = 0, B_link = ∞).
   */
  link?: { latencySec: number; bandwidthPerDirectionBytesPerSec: number };
  /**
   * Host RAM bandwidth of the node (read + write combined), for the NVMe → host → device staging hop, which writes and
   * then reads every byte once. Present only for an NVMe tier on a dedicated-memory device. Unit: B/s.
   */
  hostStagingBandwidthBytesPerSec?: number;
  /**
   * Concurrent consumers of the node's host RAM bandwidth as seen by this device, for the staging term: every streaming
   * device of the node that reads host RAM, either as its tier or as the staging hop of NVMe reads, each counted once
   * (n_host, docs/simulation-model.md §12 "Sharing"). Host RAM is one resource behind both paths, so an SSD stream
   * and a host-RAM stream on other stages split it. Absent = `nStream` (one tier streams: n_host = n_tier). Count (≥ 1).
   */
  hostStagingNStream?: number;
  /** Outstanding read requests in flight, `QD = min(queueDepth ?? maxOutstanding, maxOutstanding)`. Count (≥ 1). */
  queueDepth: number;
  /**
   * Concurrent consumers of the tier on this device's node ({@link nStream}); they share its bandwidth and IOPS. For
   * the host-RAM tier this is n_host (host-RAM readers and NVMe stagers together). Count (≥ 1).
   */
  nStream: number;
}

/** First-byte latency of one read of a tier: host RAM load-to-use latency, or the SSD's QD1 random-read latency. Unit: seconds. */
export function tierLatencySec(tier: StorageTierSpec): number {
  return tier.kind === 'host_ram' ? tier.latencySec : tier.randomReadLatencySec;
}

/**
 * Bandwidth of a tier for streamed reads: host RAM's theoretical peak, or the SSD's sequential read capped by its
 * slot (`min(B_seq, linkBandwidth)`). Shared by the node's consumers. Unit: B/s.
 */
export function tierBandwidthBytesPerSec(tier: StorageTierSpec): number {
  // Formula: B_tier = host RAM: B_ram;  NVMe: min(B_seq, B_slot ?? ∞)
  return tier.kind === 'host_ram'
    ? tier.bandwidthBytesPerSec
    : Math.min(tier.seqReadBandwidthBytesPerSec, tier.linkBandwidthBytesPerSec ?? Infinity);
}

/**
 * Queue depth a deployment gets on a tier: the requested depth, capped by (and defaulting to) the tier's
 * `maxOutstanding`. `QD = 1` serializes every request behind its latency. Count.
 */
export function resolveQueueDepth(
  tier: HostMemorySpec | NvmeSpec,
  requested: number | undefined,
): number {
  // Formula: QD = min(queueDepth ?? maxOutstanding, maxOutstanding)
  return Math.min(requested ?? tier.maxOutstanding, tier.maxOutstanding);
}

/** The cost of one transfer op and its terms (design doc §2.3). Times in seconds, bytes in bytes. */
export interface TransferTime {
  /** Physical read requests `R = Σ count_i` (expected, real). Count. */
  requests: number;
  /** Bytes the tier moves, every read rounded up to its granularity: `Σ count_i·ceil(bytes_i/gran)·gran`. Unit: bytes. */
  tierBytes: number;
  /** Useful payload over the host link. Unit: bytes. */
  usefulBytes: number;
  /** `α_path = α_tier + α_link`: the first byte crosses every hop. Unit: seconds. */
  latencySec: number;
  /** Little's law: `(R − 1)·α_tier / QD` (at least 0). Unit: seconds. */
  littleSec: number;
  /** The tier's request rate shared by n_stream: `R / (IOPS / n)`; absent for host RAM and an SSD without `randomRead`. Unit: seconds. */
  iopsSec?: number;
  /** Sum of per-class tier bandwidth times, shared by n_stream: Σ bytes_i / (B_i / n). Unit: seconds. */
  tierSec: number;
  /** This device's host link: `usefulBytes / B_link` (0 on unified memory). Unit: seconds. */
  linkSec: number;
  /** NVMe → host → device staging through host RAM: `bytes_tier / (B_ram / (2·n_host))`; absent otherwise. Unit: seconds. */
  hostSec?: number;
  /** `service = max(little, iops, tier, link, host)`: hops are pipelined across requests and chunks. Unit: seconds. */
  serviceSec: number;
  /** `t_path = α_path + service`. Unit: seconds. */
  sec: number;
  /**
   * Bytes this op puts on the device's host link (PCIe), per device and step: `usefulBytes` when the path has a link,
   * 0 on unified memory. With {@link linkSec}, the op's use of the link for the overlapped stage transfer of
   * docs/simulation-model.md §12 ("Overlap across tiers"). Unit: bytes.
   */
  linkBytes: number;
  /**
   * Bytes this op moves through the node's host RAM, per device and step: the host-RAM tier's `tierBytes` (read once),
   * an NVMe read with staging `2·tierBytes` (written once, read once), 0 otherwise (unified memory, no staging).
   * Unit: bytes.
   */
  hostRamBytes: number;
  /**
   * Time this op occupies the node's host RAM bandwidth, on that resource alone, with B_ram shared by n_host: the
   * host-RAM tier's `tierSec` (`bytes_tier / (B_ram / n_host)`, its n_stream is n_host), an NVMe read with staging
   * its `hostSec` (`bytes_tier / (B_ram / (2·n_host))`), 0 otherwise. No latency term. Unit: seconds.
   */
  hostRamSec: number;
}

/**
 * Time of one off-device read on `path` (design doc §2.3), the one rule for every group:
 *   B_i        = NVMe sustained, bytes_i ≥ blockBytes: min(f_sus·B_seq, B_slot); otherwise B_tier
 *   tier       = Σ_i count_i·ceil(bytes_i/gran)·gran / (B_i/n)
 *   R          = Σ_i count_i
 *   bytes_tier = Σ_i count_i · ceil(bytes_i / gran) · gran
 *   α_path     = α_tier + α_link
 *   service    = max( (R − 1)·α_tier / QD,  R / (IOPS / n),  tier,  usefulBytes / B_link,
 *                     [NVMe on a dedicated device] bytes_tier / (B_ram / (2·n_host)) )
 *   t_path     = α_path + service
 *
 * n is the tier's n_stream (`path.nStream`); n_host the consumers of host RAM bandwidth (`path.hostStagingNStream`,
 * default n): host-RAM readers and NVMe stagers of the node share one B_ram (docs/simulation-model.md §12 "Sharing").
 * Latencies add (the first byte traverses every hop); throughput terms take the max (every hop moves chunks while
 * the next is read). The IOPS term exists only for an SSD with a rated `randomRead`.
 */
export function transferTime(
  reads: readonly { count: number; bytesEach: number }[],
  usefulBytes: number,
  path: TransferPath,
): TransferTime {
  const { tier, link, queueDepth: QD, nStream: n } = path;
  const gran = tier.readGranularityBytes;
  const alphaTier = tierLatencySec(tier);
  // Formula: R = Σ count_i;  bytes_tier = Σ count_i·ceil(bytes_i/gran)·gran
  const requests = reads.reduce((s, r) => s + r.count, 0);
  const tierBytes = reads.reduce((s, r) => s + r.count * Math.ceil(r.bytesEach / gran) * gran, 0);
  // Formula: α_path = α_tier + α_link   (α_link = 0 on unified memory)
  // See docs/simulation-model.md §12 ("Transfer path")
  const latencySec = alphaTier + (link?.latencySec ?? 0);
  // Formula: little = (R − 1)·α_tier / QD
  const littleSec = (Math.max(0, requests - 1) * alphaTier) / QD;
  // Formula: iops = R / (IOPS / n)   (rated SSDs only)
  const iopsSec =
    tier.kind === 'nvme' && tier.randomRead !== undefined
      ? requests / (tier.randomRead.iops / n)
      : undefined;
  // Formula: B_i = min(f_sus·B_seq, B_slot) for large NVMe reads; tier = Σ bytes_i / (B_i/n).
  // Keep the old reduction/division in datasheet mode for byte-identical default results.
  const sustained = tier.kind === 'nvme' ? path.nvmeSustainedRead : undefined;
  const tierSec =
    sustained === undefined
      ? tierBytes / (tierBandwidthBytesPerSec(tier) / n)
      : reads.reduce((seconds, read) => {
          const bytes = read.count * Math.ceil(read.bytesEach / gran) * gran;
          const bandwidth =
            tier.kind === 'nvme' && read.bytesEach >= sustained.blockBytes
              ? Math.min(
                  tier.seqReadBandwidthBytesPerSec * sustained.fraction,
                  tier.linkBandwidthBytesPerSec ?? Infinity,
                )
              : tierBandwidthBytesPerSec(tier);
          return seconds + bytes / (bandwidth / n);
        }, 0);
  // Formula: link = usefulBytes / B_link
  const linkSec = link === undefined ? 0 : usefulBytes / link.bandwidthPerDirectionBytesPerSec;
  // Formula: host = bytes_tier / (B_ram / (2·n_host))   (NVMe → host RAM → device: written once, read once;
  //          n_host = host-RAM readers + NVMe stagers of the node, each device once; one tier streaming: n_host = n)
  // See docs/simulation-model.md §12 ("Sharing")
  const B_ram = path.hostStagingBandwidthBytesPerSec;
  const n_host = path.hostStagingNStream ?? n;
  const hostSec = B_ram === undefined ? undefined : tierBytes / (B_ram / (2 * n_host));
  // Formula: linkBytes = link ? usefulBytes : 0
  //          hostRamBytes = host_ram: bytes_tier;  NVMe with staging: 2·bytes_tier;  else 0
  //          hostRamSec   = host_ram: tier (= bytes_tier/(B_ram/n_host));  NVMe with staging: host;  else 0
  // See docs/simulation-model.md §12 ("Overlap across tiers")
  const linkBytes = link === undefined ? 0 : usefulBytes;
  const hostRamBytes =
    tier.kind === 'host_ram' ? tierBytes : hostSec === undefined ? 0 : 2 * tierBytes;
  const hostRamSec = tier.kind === 'host_ram' ? tierSec : (hostSec ?? 0);
  const serviceSec = Math.max(littleSec, iopsSec ?? 0, tierSec, linkSec, hostSec ?? 0);
  const result: TransferTime = {
    requests,
    tierBytes,
    usefulBytes,
    latencySec,
    littleSec,
    tierSec,
    linkSec,
    serviceSec,
    // Formula: t_path = α_path + service
    sec: latencySec + serviceSec,
    linkBytes,
    hostRamBytes,
    hostRamSec,
  };
  if (iopsSec !== undefined) result.iopsSec = iopsSec;
  if (hostSec !== undefined) result.hostSec = hostSec;
  return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Sharing of a node's tier (design doc §2.3, "n_stream")

/**
 * Devices of pipeline stage `stage` that sit on node `node`, under the rank layout of docs/simulation-model.md §5
 * ("Rank layout"): `rank = tpIdx + tp·(dpIdx + dp·ppIdx)`, `node = floor(rank / devicesPerNode)`. The ranks of a
 * stage are the contiguous block `[stage·tp·dp, (stage+1)·tp·dp)`, so this is its overlap with the node's ranks.
 * Count.
 */
export function stageDevicesOnNode(
  tp: number,
  dp: number,
  devicesPerNode: number,
  stage: number,
  node: number,
): number {
  // Formula: |[s·tp·dp, (s+1)·tp·dp) ∩ [node·D, (node+1)·D)|
  const stageStart = stage * tp * dp;
  const stageEnd = stageStart + tp * dp;
  const nodeStart = node * devicesPerNode;
  const nodeEnd = nodeStart + devicesPerNode;
  return Math.max(0, Math.min(stageEnd, nodeEnd) - Math.max(stageStart, nodeStart));
}

/**
 * Stages that stream from a tier at the same time (design doc §2.3): with prefetching on (p > 0) every offloading
 * stage can have a prefetch in flight (`|S_tier|`); without it (p = 0) a stage streams only while a microbatch
 * occupies it, and `m` microbatches circulate, so at most `min(m, |S_tier|)`. Count.
 */
export function streamSlots(
  prefetching: boolean,
  microbatches: number,
  streamingStages: number,
): number {
  // Formula: slots = p > 0 ? |S_tier| : min(m, |S_tier|)
  return prefetching ? streamingStages : Math.min(microbatches, streamingStages);
}

/**
 * Concurrent consumers of one tier on one node, as seen by stage `stage` (design doc §2.3):
 * `consumers[s]` = devices of stage s on the node that stream from the tier (0 if s holds no group there),
 * `S_tier = { s : consumers[s] > 0 }`, and the optimistic schedule runs the `slots` stages with the FEWEST consumers
 * together: `n_stream = max(consumers[stage], Σ of the slots smallest consumers[s] over S_tier)` (an explicit
 * upper-bound choice). At least 1. Count.
 */
export function nStream(consumers: readonly number[], stage: number, slots: number): number {
  const active = consumers.filter((c) => c > 0).sort((a, b) => a - b);
  // Formula: n = max(consumers[stage], Σ_{slots smallest} consumers[s]), at least 1
  const scheduled = active.slice(0, slots).reduce((s, c) => s + c, 0);
  return Math.max(1, consumers[stage] ?? 0, scheduled);
}

/**
 * n_stream of a tier for the devices of stage `stage`: {@link nStream} on every node holding a device of the stage,
 * the largest of them (the stage's TP ranks finish together, so its slowest device sets its time). `streams[s]` is
 * true if stage s streams from the tier in this phase; `prefetching` is p > 0 ({@link streamSlots}). Count (≥ 1).
 */
export function stageNStream(
  parallel: { tp: number; dp: number; pp: number },
  devicesPerNode: number,
  streams: readonly boolean[],
  stage: number,
  prefetching: boolean,
  microbatches: number,
): number {
  const { tp, dp, pp } = parallel;
  const firstNode = Math.floor((stage * tp * dp) / devicesPerNode);
  const lastNode = Math.floor(((stage + 1) * tp * dp - 1) / devicesPerNode);
  let n = 1;
  for (let node = firstNode; node <= lastNode; node++) {
    const consumers = Array.from({ length: pp }, (_, s) =>
      streams[s] ? stageDevicesOnNode(tp, dp, devicesPerNode, s, node) : 0,
    );
    const streaming = consumers.filter((c) => c > 0).length;
    n = Math.max(n, nStream(consumers, stage, streamSlots(prefetching, microbatches, streaming)));
  }
  return n;
}
