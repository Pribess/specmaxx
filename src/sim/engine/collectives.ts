/**
 * Alpha-beta collective costs: flat (one link level), hierarchical (intra-node + inter-node), host-bounce
 * links (GeForce without P2P), and inter-node host staging (a dedicated-memory GPU without a GPUDirect path, e.g.
 * any GPU over TCP, and the TCP socket copy of every device over TCP; `staging.ts`). Each cost is split into latency
 * (α) terms, which stay on the dependency chain, and bandwidth (β) terms, which may overlap with compute.
 * See docs/simulation-model.md §6 ("Collectives").
 *
 * Notation: S = messageBytes per device (conventions on `CollectiveOp`: all_reduce = the full buffer,
 * all_gather = the full OUTPUT, reduce_scatter = the full INPUT, all_to_all = the total sent incl. the self
 * share, send_recv = bytes sent), n = group size, B = per-device per-direction bandwidth (B/s), α = latency (s).
 *
 * Inter-node fabric shapes other than a non-blocking switch (oversubscribed switch, full mesh, ring, chain) are
 * costed in `fabricCollectives.ts`; a path without a fabric runs the formulas of this file unchanged.
 *
 * In-network reduction (NVLS on an NVLink Switch, SHARP on an InfiniBand switch; `LinkSpec.inNetworkReduction`) is
 * one more all-reduce candidate of `'best'` when the path carries it ({@link nvlsAllReduceSec},
 * {@link sharpAllReduceSec}); a path without it runs exactly the ring / tree code of before.
 *
 * Not modeled (documented): PCIe switch / shared-uplink contention, the HBM traffic of collectives, contention
 * between communication and compute for HBM, and intra-node shapes (an intra-node link is always treated as
 * switched; e.g. the MI300X xGMI mesh is encoded as aggregate injection).
 */

import type { CollectiveOp } from '../ir/ops';
import type { FabricShape, InNetworkReduction, LinkSpec } from '../hw/types';
import { fabricCollectiveTime } from './fabricCollectives';
import type { HostStaging } from './staging';

/** The kinds of collective (same as `CollectiveOp.collective`). */
export type CollectiveKind = CollectiveOp['collective'];

/**
 * All-reduce algorithm choice: `'best'` = the fastest of ring, tree and (when the path carries one) the link's
 * in-network reduction for this message; `'ring'` and `'tree'` force that algorithm.
 */
export type AllReduceAlgorithm = 'best' | 'ring' | 'tree';

/** Cost of one collective on one device. */
export interface CollectiveTime {
  /** Σ latency (α) terms. Unit: seconds. */
  latencySec: number;
  /** Σ bandwidth (β) terms. Unit: seconds. */
  bandwidthSec: number;
  /** latencySec + bandwidthSec. Unit: seconds. */
  totalSec: number;
  /**
   * Algorithm used: `'ring'`, `'tree'`, `'nvls'` / `'sharp'` (in-network all-reduce, {@link InNetworkReduction}),
   * `'direct'` (all-to-all, send/recv), `'hierarchical'` (two link levels, whatever each phase used), or `'none'`
   * (n = 1).
   */
  algorithm: 'ring' | 'tree' | 'nvls' | 'sharp' | 'direct' | 'hierarchical' | 'none';
}

/** The α and β parameters one link level offers to a collective, after host-bounce and host-staging effects. */
export interface EffectiveLink {
  /** Effective per-device, per-direction bandwidth B_eff. Unit: B/s. */
  bandwidthBytesPerSec: number;
  /** Effective per-message latency α_eff. Unit: seconds. */
  latencySec: number;
}

/**
 * Where a collective runs:
 * - `'none'`: a group of one device (no communication);
 * - `'flat'`: every member on one link level (`'intra'` = one node, `'inter'` = one device per node);
 * - `'hierarchical'`: `devicesPerNode` (n1) members in each of `nodes` (n2) nodes, n = n1 · n2.
 *
 * The optional fabric, staging and in-network fields are set only when they differ from the default (placement
 * emits them only then), so an absent field always means the switched, non-blocking, ring / tree formulas of this
 * file. Inter-node links are already the per-device share of the node's NICs (`hw/nics.ts: interNodeLinkPerDevice`).
 */
export type CollectivePath =
  | { kind: 'none' }
  | {
      kind: 'flat';
      level: 'intra' | 'inter';
      link: LinkSpec;
      /**
       * Inter-node fabric as this group sees it (`groupFabric`: a ring segment is a chain); absent = switched,
       * non-blocking (the pre-fabric path). Only on `level: 'inter'`.
       */
      fabric?: FabricShape;
      /**
       * send_recv only: links the send crosses, ≥ 1 (absent = 1); also how many of the boundary's flows share the
       * busiest link. Count.
       */
      sendHops?: number;
      /** Concurrent send/recv ranks sharing one host: min(devicesPerNode, tp·dp). Count; absent = 1. */
      sendActiveDevices?: number;
      /**
       * Inter-node host staging of this path (`staging.ts`: PCIe staging and / or the TCP socket copy); absent = none
       * (intra-node paths, unified memory on RDMA, a GPUDirect pair on RDMA, and every path before staging was
       * modelled). Only on `level: 'inter'`.
       */
      interStaging?: HostStaging;
      /**
       * In-network all-reduce this path may use as a `'best'` candidate (`SimOptions.inNetworkReduction: 'on'` and a
       * capable link); absent = ring / tree only. Placement sets it on an intra-node path, and on an inter-node path
       * only without `fabric` and without `interStaging`.
       */
      inNetworkReduction?: InNetworkReduction;
    }
  | {
      kind: 'hierarchical';
      devicesPerNode: number;
      nodes: number;
      intra: LinkSpec;
      inter: LinkSpec;
      /**
       * Oversubscribed switch between nodes (oversubscription > 1); absent = non-blocking. Other shapes cannot be
       * hierarchical (they need one device per node).
       */
      interFabric?: { kind: 'switched'; oversubscription: number };
      /**
       * Host staging of the inter-node phase (`staging.ts`: PCIe staging and / or the TCP socket copy); absent = none.
       * The intra-node phase never stages.
       */
      interStaging?: HostStaging;
      /**
       * In-network all-reduce of the intra-node phase (NVLS: replaces the RS_intra + AG_intra pair when faster);
       * absent = ring phases only.
       */
      intraInNetworkReduction?: InNetworkReduction;
      /**
       * In-network all-reduce of the inter-node phase (SHARP: one more candidate of AR_inter); absent = ring / tree
       * only. Placement never sets it with `interStaging`.
       */
      interInNetworkReduction?: InNetworkReduction;
    };

/** A zero cost (group of one). */
const NO_TIME: CollectiveTime = { latencySec: 0, bandwidthSec: 0, totalSec: 0, algorithm: 'none' };

/** Builds a {@link CollectiveTime} from its α and β parts. */
function time(
  latencySec: number,
  bandwidthSec: number,
  algorithm: CollectiveTime['algorithm'],
): CollectiveTime {
  return { latencySec, bandwidthSec, totalSec: latencySec + bandwidthSec, algorithm };
}

/**
 * α and β of `link` when `activeDevices` devices of a node transfer at once, then host staging if `staging` is given.
 * A `'direct'` link is unchanged.
 * A `'host_bounce'` link stages every byte through host memory (written once, read once), so the host memory
 * bandwidth is shared by 2 · activeDevices streams, and each message pays the host copy latency.
 * `activeDevices` = group size for collectives; PP send/recv uses min(devicesPerNode, tp·dp).
 * `staging` (inter-node paths only) applies {@link stagedLink} with m = `stepMessageBytes`, the bytes of one
 * algorithm step's message (S/n for the collectives, S for send/recv). Absent ⇒ exactly the result without staging.
 */
export function effectiveLink(
  link: LinkSpec,
  activeDevices: number,
  staging?: HostStaging,
  stepMessageBytes = 0,
): EffectiveLink {
  const unstaged = hostBounceLink(link, activeDevices);
  return staging === undefined ? unstaged : stagedLink(unstaged, staging, stepMessageBytes);
}

/** α and β of `link` after host-bounce effects only (the part of {@link effectiveLink} before staging). */
function hostBounceLink(link: LinkSpec, activeDevices: number): EffectiveLink {
  if (link.p2p === 'direct' || link.hostBounce === undefined) {
    return {
      bandwidthBytesPerSec: link.bandwidthPerDirectionBytesPerSec,
      latencySec: link.latencySec,
    };
  }
  // Formula: B_eff = min(B, hostMemBW / (2 · nActive));  α_eff = α + extraLatency
  // See docs/simulation-model.md §6 ("Host bounce")
  const hostShareBytesPerSec = link.hostBounce.hostMemBandwidthBytesPerSec / (2 * activeDevices);
  const bandwidthBytesPerSec = Math.min(
    link.bandwidthPerDirectionBytesPerSec,
    hostShareBytesPerSec,
  );
  const latencySec = link.latencySec + link.hostBounce.extraLatencySec;
  return { bandwidthBytesPerSec, latencySec };
}

/**
 * a: host-DRAM accesses per byte of ONE end's copies on a staged path: 1 for the PCIe copy (the GPU's DMA writes or
 * reads the pinned buffer once), 2 for the TCP socket copy (a CPU memcpy reads and writes each byte), 3 for both.
 * Formula: a = (pcie ? 1 : 0) + (socketCopy ? 2 : 0). Count.
 */
export function stagingCopyAccessesPerByte(staging: HostStaging): number {
  return (staging.pcie === undefined ? 0 : 1) + (staging.socketCopy ? 2 : 0);
}

/**
 * B_hj: the host-DRAM share of one device when every stage of a staged message runs at once (the pipelined steady
 * state): each byte touches host DRAM 2·a times for the copies (a per end, both ends) plus 2 times for the NIC (read
 * at the sender, write at the receiver), and the n_stg devices of the node that stage at once share B_host.
 * PCIe staging only (a = 1): B_host/(4·n_stg); socket copy only (a = 2): B_host/(6·n_stg); both (a = 3):
 * B_host/(8·n_stg). Unit: B/s, per device, per direction.
 */
function hostJointBytesPerSec(staging: HostStaging): number {
  // Formula: B_hj = B_host / ((2·a + 2)·n_stg)
  const accesses = 2 * stagingCopyAccessesPerByte(staging) + 2;
  return staging.hostMemBandwidthBytesPerSec / (accesses * staging.activeDevicesPerNode);
}

/**
 * The bandwidth host staging allows one device, per direction, whatever the network: min(B_pcie, B_hj) (B_pcie = ∞
 * without PCIe staging), B_hj from {@link hostJointBytesPerSec}. PCIe staging only: min(B_pcie, B_host/(4·n_stg)),
 * each staged byte crossing the device's PCIe link once per end and host DRAM 4 times per device pair (sender: GPU
 * write + NIC read; receiver: NIC write + GPU read). Unit: B/s.
 */
export function stagingBandwidthCapBytesPerSec(staging: HostStaging): number {
  // Formula: B_stg = min(B_pcie, B_host / ((2·a + 2)·n_stg))   (B_pcie = ∞ without PCIe staging)
  // See docs/simulation-model.md §6 ("Inter-node host staging", "TCP socket copy")
  const hostShareBytesPerSec = hostJointBytesPerSec(staging);
  return Math.min(staging.pcie?.bandwidthBytesPerSec ?? Infinity, hostShareBytesPerSec);
}

/**
 * The per-device rates of the stages of one staged message, and the joint host-memory cap, for {@link stagedLink}.
 * A message crosses up to five stages: PCIe copy-out (GPU → pinned host buffer), socket copy-in (user buffer →
 * kernel), the wire, socket copy-out (kernel → user buffer), PCIe copy-in (host buffer → GPU). An absent stage has
 * rate ∞ (its step time s/∞ = 0). In a collective every node sends and receives at once, so each stage's host-DRAM
 * traffic is counted for both directions on each node:
 * - r_pcie: GPU ↔ pinned host buffer over PCIe: one DRAM access per byte per device, r_pcie = min(B_pcie,
 *   B_host/n_stg);
 * - r_sock: the kernel memcpy between user and socket buffers: a read plus a write, two DRAM accesses per byte per
 *   device, r_sock = B_host/(2·n_stg);
 * - r_net: the wire: NIC read at the sender + NIC write at the receiver, two DRAM accesses per byte per device,
 *   r_net = min(B, B_host/(2·n_stg));
 * - B_hostjoint: all stages running at once (steady state of the pipeline): B_host/((2·a + 2)·n_stg).
 * All in B/s, per device, per direction.
 */
export interface StagedStageRates {
  /** r_pcie: one PCIe copy stage (sender copy-out, receiver copy-in); ∞ without PCIe staging. Unit: B/s. */
  pcieBytesPerSec: number;
  /** r_sock: one socket copy stage (sender copy-in, receiver copy-out); ∞ without the socket copy. Unit: B/s. */
  sockBytesPerSec: number;
  /** r_net: the wire stage. Unit: B/s. */
  netBytesPerSec: number;
  /** B_host/((2·a + 2)·n_stg): the DRAM cap when every stage runs at once (pipelined steady state). Unit: B/s. */
  hostJointBytesPerSec: number;
}

/** {@link StagedStageRates} of a staged path whose unstaged network rate is `netBytesPerSec` (B, B/s). */
export function stagedStageRates(netBytesPerSec: number, staging: HostStaging): StagedStageRates {
  // Formula: r_pcie = min(B_pcie, B_host/n_stg)   (∞ without PCIe staging)
  //          r_sock = B_host/(2·n_stg)            (∞ without the socket copy)
  //          r_net  = min(B, B_host/(2·n_stg));  B_hj = B_host/((2·a + 2)·n_stg)
  // See docs/simulation-model.md §6 ("Inter-node host staging", "TCP socket copy")
  const n = staging.activeDevicesPerNode;
  const hostBytesPerSec = staging.hostMemBandwidthBytesPerSec;
  return {
    pcieBytesPerSec:
      staging.pcie === undefined
        ? Infinity
        : Math.min(staging.pcie.bandwidthBytesPerSec, hostBytesPerSec / n),
    sockBytesPerSec: staging.socketCopy ? hostBytesPerSec / (2 * n) : Infinity,
    netBytesPerSec: Math.min(netBytesPerSec, hostBytesPerSec / (2 * n)),
    hostJointBytesPerSec: hostJointBytesPerSec(staging),
  };
}

/**
 * An inter-node link level `unstaged` with host staging applied. Every message is copied GPU → pinned host buffer at
 * the sender (rate r_pcie, PCIe staging), copied into kernel socket buffers (r_sock, TCP), sent (r_net), copied out
 * of the socket buffers (r_sock) and host buffer → GPU at the receiver (r_pcie), in steps of s = min(m, c) bytes
 * (c = `staging.stepBytes`), the stages pipelined over the k = m/s steps ({@link stagedStageRates}; an absent stage
 * has rate ∞ and contributes nothing).
 *
 * A pipeline of stages with per-step times s/r_i, k steps, and steady-state period s/B_eff takes
 * Σ_i s/r_i + (k − 1)·s/B_eff = m/B_eff + (Σ_i s/r_i − s/B_eff):
 * - B_eff = min(r_pcie, r_sock, r_net, B_hostjoint) = min(B, B_pcie, B_hostjoint) is the steady-state rate (β);
 * - the fill/drain Σ_i s/r_i − s/B_eff is one step through every stage except the one that sets B_eff, so nothing
 *   is counted twice when PCIe, the copy or host memory rather than the NIC binds. It is ≥ 0 because
 *   Σ_i 1/r_i ≥ 1/B_eff.
 * - m ≤ c (k = 1): exactly the serial sum 2·m/r_pcie + 2·m/r_sock + m/r_net.
 * The fill is serial with the step chain, so it joins α; β stays m/B_eff, which the collective formulas apply.
 * No fixed latency is added for the socket copy: the measured TCP α (a 1-byte ping-pong) already holds the syscalls
 * and a 1-byte copy, so only the size-dependent part 2·s/r_sock joins.
 * PCIe staging only, the operations are exactly those of the formula before the socket copy (bit-identical).
 * `stepMessageBytes` = m (bytes); returns B_eff (B/s) and α_eff (seconds).
 */
export function stagedLink(
  unstaged: EffectiveLink,
  staging: HostStaging,
  stepMessageBytes: number,
): EffectiveLink {
  // Formula: B_eff = min(r_pcie, r_sock, r_net, B_host/((2·a + 2)·n_stg))
  //          s = min(m, c)
  //          α_eff = α + 2·α_pcie + (2·s/r_pcie + 2·s/r_sock + s/r_net − s/B_eff)
  // 2·α_pcie = α_stage: one PCIe write + flag at the sender, one PCIe poll + read at the receiver (0 without PCIe).
  // See docs/simulation-model.md §6 ("Inter-node host staging", "TCP socket copy")
  const rates = stagedStageRates(unstaged.bandwidthBytesPerSec, staging);
  const bandwidthBytesPerSec = Math.min(
    rates.pcieBytesPerSec,
    rates.sockBytesPerSec,
    rates.netBytesPerSec,
    rates.hostJointBytesPerSec,
  );
  const stageLatencySec = 2 * (staging.pcie?.latencySec ?? 0);
  const stepBytes = Math.min(stepMessageBytes, staging.stepBytes);
  // Each term is 0 when its stage is absent (rate ∞), so PCIe-only staging computes exactly the former sum.
  const pcieCopiesSec = (2 * stepBytes) / rates.pcieBytesPerSec;
  const socketCopiesSec = (2 * stepBytes) / rates.sockBytesPerSec;
  const oneStepThroughAllSec = pcieCopiesSec + socketCopiesSec + stepBytes / rates.netBytesPerSec;
  const fillSec = oneStepThroughAllSec - stepBytes / bandwidthBytesPerSec;
  const latencySec = unstaged.latencySec + stageLatencySec + fillSec;
  return { bandwidthBytesPerSec, latencySec };
}

/** Ring all-reduce of S bytes over n devices on one link level. */
export function ringAllReduceSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = 2(n−1)·α + 2(n−1)/n · S/B
  // See docs/simulation-model.md §6 ("Flat collectives")
  const latencySec = 2 * (n - 1) * link.latencySec;
  const bandwidthSec = ((2 * (n - 1)) / n) * (messageBytes / link.bandwidthBytesPerSec);
  return time(latencySec, bandwidthSec, 'ring');
}

/** Tree (reduce + broadcast) all-reduce of S bytes over n devices on one link level. */
export function treeAllReduceSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = 2⌈log2 n⌉·α + 2·S/B
  // See docs/simulation-model.md §6 ("Flat collectives")
  const latencySec = 2 * Math.ceil(Math.log2(n)) * link.latencySec;
  const bandwidthSec = (2 * messageBytes) / link.bandwidthBytesPerSec;
  return time(latencySec, bandwidthSec, 'tree');
}

/**
 * NVLS (NVLink SHARP) all-reduce of S bytes over n GPUs on one NVLink Switch. Two phases, pipelined over chunks:
 * 1. reduce-scatter in the switch: GPU c reads its reduced 1/n shard with a multimem load-reduce
 *    (`multimem.ld_reduce`); the switch reads that shard from all n GPUs, adds it up and returns S/n to c. Over
 *    all n shards each GPU's link sends its whole buffer once (egress S) and receives its shard (ingress S/n);
 * 2. multicast all-gather: GPU c writes its reduced shard once with a multicast store (`multimem.st`), which the
 *    switch copies to all n GPUs (egress S/n, ingress S).
 * Each direction of a GPU's link carries S + S/n = (n+1)/n · S, and both phases stream concurrently over chunks, so
 * the β term is the busier direction once; the α term is the two dependent steps (vs ring 2(n−1), tree 2⌈log2 n⌉).
 * `link`: the per-device, per-direction B and the α of the NVLink level. n = 1 costs nothing.
 */
export function nvlsAllReduceSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = 2·α + (n+1)/n · S/B
  // See docs/simulation-model.md §6 ("In-network reduction")
  const steps = 2;
  const bytesPerDirection = ((n + 1) / n) * messageBytes;
  const latencySec = steps * link.latencySec;
  const bandwidthSec = bytesPerDirection / link.bandwidthBytesPerSec;
  return time(latencySec, bandwidthSec, 'nvls');
}

/**
 * SHARP (InfiniBand in-network aggregation) all-reduce of S bytes over n NICs through an IB switch: each NIC streams
 * its S bytes up the switch's aggregation tree (egress S) and receives the reduced S back (ingress S), the two
 * directions at once (full duplex), so the β term is S/B once; the α term is the two dependent steps (up the tree,
 * back down). `link`: the per-device, per-direction B and the α of the inter-node level. n = 1 costs nothing.
 */
export function sharpAllReduceSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = 2·α + S/B
  // See docs/simulation-model.md §6 ("In-network reduction")
  const steps = 2;
  const bytesPerDirection = messageBytes;
  const latencySec = steps * link.latencySec;
  const bandwidthSec = bytesPerDirection / link.bandwidthBytesPerSec;
  return time(latencySec, bandwidthSec, 'sharp');
}

/** The in-network all-reduce of `kind` ({@link nvlsAllReduceSec} or {@link sharpAllReduceSec}) of S bytes over n. */
export function inNetworkAllReduceSec(
  kind: InNetworkReduction,
  messageBytes: number,
  n: number,
  link: EffectiveLink,
): CollectiveTime {
  switch (kind) {
    case 'nvls':
      return nvlsAllReduceSec(messageBytes, n, link);
    case 'sharp':
      return sharpAllReduceSec(messageBytes, n, link);
    default: {
      const _never: never = kind;
      throw new Error(`unknown in-network reduction ${String(_never)}`);
    }
  }
}

/**
 * Flat all-reduce with the chosen algorithm (`'best'` = the faster of ring and tree; a tie keeps ring). `treeLink`
 * (default `link`) is the link the tree runs on when it differs from the ring's: on a host-staged path the per-step
 * message differs (ring S/n, tree S), so each algorithm gets its own staged α before the comparison.
 * `inNetwork` (the path's in-network reduction, absent = none) adds {@link inNetworkAllReduceSec} on `link` as a
 * third candidate of `'best'` only (a tie keeps ring or tree); explicit `'ring'` / `'tree'` ignore it.
 */
export function allReduceSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
  algorithm: AllReduceAlgorithm,
  treeLink: EffectiveLink = link,
  inNetwork?: InNetworkReduction,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  const ring = ringAllReduceSec(messageBytes, n, link);
  const tree = treeAllReduceSec(messageBytes, n, treeLink);
  switch (algorithm) {
    case 'ring':
      return ring;
    case 'tree':
      return tree;
    case 'best': {
      // Formula: t = min(t_ring, t_tree)
      const ringOrTree = tree.totalSec < ring.totalSec ? tree : ring;
      if (inNetwork === undefined) return ringOrTree;
      // Formula: t = min(t_ring, t_tree, t_in)   (ties keep ring / tree)
      // See docs/simulation-model.md §6 ("In-network reduction")
      const inSwitch = inNetworkAllReduceSec(inNetwork, messageBytes, n, link);
      return inSwitch.totalSec < ringOrTree.totalSec ? inSwitch : ringOrTree;
    }
    default: {
      const _never: never = algorithm;
      throw new Error(`unknown all-reduce algorithm ${String(_never)}`);
    }
  }
}

/**
 * Ring all-gather (S = the full OUTPUT) or reduce-scatter (S = the full INPUT) over n devices: both move
 * (n−1)/n of S per device.
 */
export function ringAllGatherSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = (n−1)·α + (n−1)/n · S/B
  // See docs/simulation-model.md §6 ("Flat collectives")
  const latencySec = (n - 1) * link.latencySec;
  const bandwidthSec = ((n - 1) / n) * (messageBytes / link.bandwidthBytesPerSec);
  return time(latencySec, bandwidthSec, 'ring');
}

/** Ring reduce-scatter of an S-byte INPUT over n devices (the mirror of {@link ringAllGatherSec}; same cost). */
export function ringReduceScatterSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
): CollectiveTime {
  return ringAllGatherSec(messageBytes, n, link);
}

/**
 * All-to-all of S bytes per device (incl. the self share, which does not cross the link) over n devices,
 * every pair sending concurrently (direct, one α).
 */
export function allToAllSec(messageBytes: number, n: number, link: EffectiveLink): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = α + (n−1)/n · S/B
  // See docs/simulation-model.md §6 ("Flat collectives")
  const latencySec = link.latencySec;
  const bandwidthSec = ((n - 1) / n) * (messageBytes / link.bandwidthBytesPerSec);
  return time(latencySec, bandwidthSec, 'direct');
}

/** Point-to-point send of S bytes (pipeline-stage boundary). */
export function sendRecvSec(messageBytes: number, link: EffectiveLink): CollectiveTime {
  // Formula: t = α + S/B
  // See docs/simulation-model.md §6 ("Flat collectives")
  return time(link.latencySec, messageBytes / link.bandwidthBytesPerSec, 'direct');
}

/** Sum of phases of a hierarchical collective. */
function sumPhases(phases: CollectiveTime[]): CollectiveTime {
  const latencySec = phases.reduce((s, p) => s + p.latencySec, 0);
  const bandwidthSec = phases.reduce((s, p) => s + p.bandwidthSec, 0);
  return time(latencySec, bandwidthSec, 'hierarchical');
}

/**
 * Hierarchical all-reduce over n1 devices per node × n2 nodes: reduce-scatter inside each node, all-reduce of
 * the 1/n1 shard across nodes (the n1 shards in parallel, one per device), all-gather inside each node.
 * `inNetwork` (the path's in-network reductions per level, absent = none; used by `'best'` only):
 * - `intra` (NVLS): the RS_intra + AG_intra pair becomes one NVLS all-reduce of S over n1 when that is faster (its
 *   in-switch reduce-scatter and multicast all-gather bracket the inter-node step and pipeline over chunks as in
 *   the flat case, {@link nvlsAllReduceSec});
 * - `inter` (SHARP): one more candidate of AR_inter ({@link allReduceSec}).
 * Without either, exactly the ring / tree phases of before.
 */
export function hierarchicalAllReduceSec(
  messageBytes: number,
  n1: number,
  n2: number,
  intra: EffectiveLink,
  inter: EffectiveLink,
  algorithm: AllReduceAlgorithm,
  interTree: EffectiveLink = inter,
  inNetwork: { intra?: InNetworkReduction; inter?: InNetworkReduction } = {},
): CollectiveTime {
  // Formula: t = RS_intra(S, n1) + AR_inter(S/n1, n2) + AG_intra(S, n1)
  // See docs/simulation-model.md §6 ("Hierarchical collectives")
  const reduceScatterIntra = ringReduceScatterSec(messageBytes, n1, intra);
  const allReduceInter = allReduceSec(
    messageBytes / n1,
    n2,
    inter,
    algorithm,
    interTree,
    inNetwork.inter,
  );
  const allGatherIntra = ringAllGatherSec(messageBytes, n1, intra);
  const ringPhases = sumPhases([reduceScatterIntra, allReduceInter, allGatherIntra]);
  if (algorithm !== 'best' || inNetwork.intra === undefined) return ringPhases;
  // Formula: t = min(RS_intra + AG_intra, t_in(S, n1)) + AR_inter(S/n1, n2)   (a tie keeps the ring pair)
  // See docs/simulation-model.md §6 ("In-network reduction")
  const inSwitchIntra = inNetworkAllReduceSec(inNetwork.intra, messageBytes, n1, intra);
  const ringIntraSec = reduceScatterIntra.totalSec + allGatherIntra.totalSec;
  return inSwitchIntra.totalSec < ringIntraSec
    ? sumPhases([inSwitchIntra, allReduceInter])
    : ringPhases;
}

/**
 * Hierarchical all-gather of an S-byte OUTPUT over n1 × n2: gather the 1/n1 node shards across nodes, then
 * gather inside each node.
 */
export function hierarchicalAllGatherSec(
  messageBytes: number,
  n1: number,
  n2: number,
  intra: EffectiveLink,
  inter: EffectiveLink,
): CollectiveTime {
  // Formula: t = AG_inter(S/n1, n2) + AG_intra(S, n1)
  // See docs/simulation-model.md §6 ("Hierarchical collectives")
  const allGatherInter = ringAllGatherSec(messageBytes / n1, n2, inter);
  const allGatherIntra = ringAllGatherSec(messageBytes, n1, intra);
  return sumPhases([allGatherInter, allGatherIntra]);
}

/** Hierarchical reduce-scatter of an S-byte INPUT over n1 × n2: the mirror of {@link hierarchicalAllGatherSec}. */
export function hierarchicalReduceScatterSec(
  messageBytes: number,
  n1: number,
  n2: number,
  intra: EffectiveLink,
  inter: EffectiveLink,
): CollectiveTime {
  // Formula: t = RS_intra(S, n1) + RS_inter(S/n1, n2)
  // See docs/simulation-model.md §6 ("Hierarchical collectives")
  const reduceScatterIntra = ringReduceScatterSec(messageBytes, n1, intra);
  const reduceScatterInter = ringReduceScatterSec(messageBytes / n1, n2, inter);
  return sumPhases([reduceScatterIntra, reduceScatterInter]);
}

/**
 * Hierarchical all-to-all over n1 × n2 (n = n1 · n2): the intra-node and inter-node shares are sent
 * concurrently, so the slower one sets the β term; the larger effective α remains exposed.
 */
export function hierarchicalAllToAllSec(
  messageBytes: number,
  n1: number,
  n2: number,
  intra: EffectiveLink,
  inter: EffectiveLink,
): CollectiveTime {
  const n = n1 * n2;
  if (n <= 1) return NO_TIME;
  // Formula: t = max(α_intra, α_inter) + max((n1−1)/n · S/B_intra, (n−n1)/n · S/B_inter)
  // See docs/simulation-model.md §6 ("Hierarchical collectives")
  const intraSec = ((n1 - 1) / n) * (messageBytes / intra.bandwidthBytesPerSec);
  const interSec = ((n - n1) / n) * (messageBytes / inter.bandwidthBytesPerSec);
  const latencySec = Math.max(intra.latencySec, inter.latencySec);
  return time(latencySec, Math.max(intraSec, interSec), 'hierarchical');
}

/**
 * Cost of one collective of `kind` with S = `messageBytes` per device over `groupSize` devices on `path`.
 * Host-bounce links use nActive = group devices per node; send/recv uses path.sendActiveDevices (default 1).
 * `algorithm` selects the all-reduce algorithm (flat, and the inter-node phase of hierarchical). A flat path with a
 * `fabric` is costed by `fabricCollectiveTime` (docs §6 "Fabric shapes"); a hierarchical path with an
 * oversubscribed `interFabric` runs the formulas below with the inter-node bandwidth divided by the oversubscription.
 * An `interStaging` (inter-node host staging, docs §6 "Inter-node host staging") applies {@link stagedLink} to the
 * inter-node level with the message one step of each algorithm really moves, before ring and tree are compared:
 * ring all-reduce, all-gather, reduce-scatter and all-to-all S/n; tree all-reduce S; send/recv S; the hierarchical
 * inter phase S/n (ring, = (S/n1)/n2) or S/n1 (tree). Staging is not host bounce: an all-to-all keeps one α.
 * A path's in-network fields (`inNetworkReduction`, `intraInNetworkReduction`, `interInNetworkReduction`) add the
 * in-switch all-reduce as a `'best'` candidate on that level (docs §6 "In-network reduction (NVLS, SHARP)"); the
 * other collectives ignore them.
 */
export function collectiveTime(
  kind: CollectiveKind,
  messageBytes: number,
  groupSize: number,
  path: CollectivePath,
  algorithm: AllReduceAlgorithm,
): CollectiveTime {
  if (path.kind === 'none') return NO_TIME;
  // Formula: m_ring = S/n (one ring step's or one peer's share);  m_tree = S;  m_send = S
  const ringStepBytes = messageBytes / groupSize;
  if (path.kind === 'flat') {
    if (path.fabric !== undefined) {
      return fabricCollectiveTime(
        kind,
        messageBytes,
        groupSize,
        path.fabric,
        path.link,
        algorithm,
        path.sendHops,
        path.sendActiveDevices,
        path.interStaging,
      );
    }
    const activeDevices = kind === 'send_recv' ? (path.sendActiveDevices ?? 1) : groupSize;
    const linkFor = (stepMessageBytes: number): EffectiveLink =>
      effectiveLink(path.link, activeDevices, path.interStaging, stepMessageBytes);
    const link = linkFor(kind === 'send_recv' ? messageBytes : ringStepBytes);
    switch (kind) {
      case 'all_reduce':
        return allReduceSec(
          messageBytes,
          groupSize,
          link,
          algorithm,
          linkFor(messageBytes),
          path.inNetworkReduction,
        );
      case 'all_gather':
        return ringAllGatherSec(messageBytes, groupSize, link);
      case 'reduce_scatter':
        return ringReduceScatterSec(messageBytes, groupSize, link);
      case 'all_to_all': {
        // Formula (G4): host-bounce flat a2a α = (n−1)·α_eff; β retains the shared host-bandwidth cap.
        // See docs/simulation-model.md §6 ("Host bounce"); ring/chain paths already charge each hop above.
        const peerRounds = path.link.p2p === 'host_bounce' ? Math.max(1, groupSize - 1) : 1;
        return allToAllSec(messageBytes, groupSize, {
          ...link,
          latencySec: peerRounds * link.latencySec,
        });
      }
      case 'send_recv':
        return sendRecvSec(messageBytes, link);
      default: {
        const _never: never = kind;
        throw new Error(`unknown collective ${String(_never)}`);
      }
    }
  }
  const { devicesPerNode: n1, nodes: n2 } = path;
  if (n1 * n2 !== groupSize)
    throw new Error(`hierarchical path ${n1} × ${n2} does not match group size ${groupSize}`);
  const intra = effectiveLink(path.intra, n1);
  const interLink = effectiveLink(path.inter, n1);
  // Formula: B_inter = B_eff / o  (oversubscribed switch between nodes; o absent = 1 keeps the link untouched)
  const switchedInter: EffectiveLink =
    path.interFabric === undefined
      ? interLink
      : {
          ...interLink,
          bandwidthBytesPerSec: interLink.bandwidthBytesPerSec / path.interFabric.oversubscription,
        };
  // Staging applies after the switch: its PCIe and host-DRAM caps are the device's own, not shared by o.
  const interFor = (stepMessageBytes: number): EffectiveLink =>
    path.interStaging === undefined
      ? switchedInter
      : stagedLink(switchedInter, path.interStaging, stepMessageBytes);
  const inter = interFor(ringStepBytes);
  switch (kind) {
    case 'all_reduce':
      // Formula: inter-phase tree step m = S/n1 (the whole node shard)
      return hierarchicalAllReduceSec(
        messageBytes,
        n1,
        n2,
        intra,
        inter,
        algorithm,
        interFor(messageBytes / n1),
        { intra: path.intraInNetworkReduction, inter: path.interInNetworkReduction },
      );
    case 'all_gather':
      return hierarchicalAllGatherSec(messageBytes, n1, n2, intra, inter);
    case 'reduce_scatter':
      return hierarchicalReduceScatterSec(messageBytes, n1, n2, intra, inter);
    case 'all_to_all':
      return hierarchicalAllToAllSec(messageBytes, n1, n2, intra, inter);
    case 'send_recv':
      throw new Error('send_recv is point-to-point; use a flat path');
    default: {
      const _never: never = kind;
      throw new Error(`unknown collective ${String(_never)}`);
    }
  }
}
