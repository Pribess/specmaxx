/**
 * Alpha-beta collective costs: flat (one link level), hierarchical (intra-node + inter-node), and host-bounce
 * links (GeForce without P2P). Each cost is split into latency (α) terms, which stay on the dependency chain,
 * and bandwidth (β) terms, which may overlap with compute.
 * See docs/simulation-model.md §6 ("Collectives").
 *
 * Notation: S = messageBytes per device (conventions on `CollectiveOp`: all_reduce = the full buffer,
 * all_gather = the full OUTPUT, reduce_scatter = the full INPUT, all_to_all = the total sent incl. the self
 * share, send_recv = bytes sent), n = group size, B = per-device per-direction bandwidth (B/s), α = latency (s).
 *
 * Inter-node fabric shapes other than a non-blocking switch (oversubscribed switch, full mesh, ring, chain) are
 * costed in `fabricCollectives.ts`; a path without a fabric runs the formulas of this file unchanged.
 *
 * Not modeled (documented): NVLS/SHARP in-switch reduction, PCIe switch / shared-uplink contention, the HBM
 * traffic of collectives, contention between communication and compute for HBM, and intra-node shapes (an
 * intra-node link is always treated as switched; e.g. the MI300X xGMI mesh is encoded as aggregate injection).
 */

import type { CollectiveOp } from '../ir/ops';
import type { FabricShape, LinkSpec } from '../hw/types';
import { fabricCollectiveTime } from './fabricCollectives';

/** The kinds of collective (same as `CollectiveOp.collective`). */
export type CollectiveKind = CollectiveOp['collective'];

/** All-reduce algorithm choice: `'best'` = the faster of ring and tree for this message. */
export type AllReduceAlgorithm = 'best' | 'ring' | 'tree';

/** Cost of one collective on one device. */
export interface CollectiveTime {
  /** Σ latency (α) terms. Unit: seconds. */
  latencySec: number;
  /** Σ bandwidth (β) terms. Unit: seconds. */
  bandwidthSec: number;
  /** latencySec + bandwidthSec. Unit: seconds. */
  totalSec: number;
  /** Algorithm used: `'ring'`, `'tree'`, `'direct'` (all-to-all, send/recv), `'hierarchical'`, or `'none'` (n = 1). */
  algorithm: 'ring' | 'tree' | 'direct' | 'hierarchical' | 'none';
}

/** The α and β parameters one link level offers to a collective, after host-bounce effects. */
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
 * The optional fabric fields are set only when they differ from the pre-fabric default (placement emits them only
 * then), so an absent field always means the switched, non-blocking formulas of this file.
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
 * α and β of `link` when `activeDevices` devices of a node transfer at once. A `'direct'` link is unchanged.
 * A `'host_bounce'` link stages every byte through host memory (written once, read once), so the host memory
 * bandwidth is shared by 2 · activeDevices streams, and each message pays the host copy latency.
 * `activeDevices` = group size for collectives; PP send/recv uses min(devicesPerNode, tp·dp).
 */
export function effectiveLink(link: LinkSpec, activeDevices: number): EffectiveLink {
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

/** Flat all-reduce with the chosen algorithm (`'best'` = the faster of ring and tree; a tie keeps ring). */
export function allReduceSec(
  messageBytes: number,
  n: number,
  link: EffectiveLink,
  algorithm: AllReduceAlgorithm,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  const ring = ringAllReduceSec(messageBytes, n, link);
  const tree = treeAllReduceSec(messageBytes, n, link);
  switch (algorithm) {
    case 'ring':
      return ring;
    case 'tree':
      return tree;
    case 'best':
      // Formula: t = min(t_ring, t_tree)
      return tree.totalSec < ring.totalSec ? tree : ring;
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
 */
export function hierarchicalAllReduceSec(
  messageBytes: number,
  n1: number,
  n2: number,
  intra: EffectiveLink,
  inter: EffectiveLink,
  algorithm: AllReduceAlgorithm,
): CollectiveTime {
  // Formula: t = RS_intra(S, n1) + AR_inter(S/n1, n2) + AG_intra(S, n1)
  // See docs/simulation-model.md §6 ("Hierarchical collectives")
  const reduceScatterIntra = ringReduceScatterSec(messageBytes, n1, intra);
  const allReduceInter = allReduceSec(messageBytes / n1, n2, inter, algorithm);
  const allGatherIntra = ringAllGatherSec(messageBytes, n1, intra);
  return sumPhases([reduceScatterIntra, allReduceInter, allGatherIntra]);
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
 */
export function collectiveTime(
  kind: CollectiveKind,
  messageBytes: number,
  groupSize: number,
  path: CollectivePath,
  algorithm: AllReduceAlgorithm,
): CollectiveTime {
  if (path.kind === 'none') return NO_TIME;
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
      );
    }
    const link = effectiveLink(
      path.link,
      kind === 'send_recv' ? (path.sendActiveDevices ?? 1) : groupSize,
    );
    switch (kind) {
      case 'all_reduce':
        return allReduceSec(messageBytes, groupSize, link, algorithm);
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
  const inter: EffectiveLink =
    path.interFabric === undefined
      ? interLink
      : {
          ...interLink,
          bandwidthBytesPerSec: interLink.bandwidthBytesPerSec / path.interFabric.oversubscription,
        };
  switch (kind) {
    case 'all_reduce':
      return hierarchicalAllReduceSec(messageBytes, n1, n2, intra, inter, algorithm);
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
