/**
 * Alpha-beta collective costs on the inter-node fabric shapes other than a non-blocking switch: an oversubscribed
 * switch, a full mesh (optionally with several cables per pair), a ring and a daisy chain. `collectiveTime`
 * (`collectives.ts`) calls {@link fabricCollectiveTime} only when a flat inter-node path carries a `fabric`; without
 * one it runs the switched formulas unchanged, so every pre-fabric result stays bit-identical.
 * See docs/simulation-model.md §6 ("Fabric shapes").
 *
 * Notation (per-device conventions of `CollectiveOp` as in `collectives.ts`): S = messageBytes, N = nodes in the
 * group (one device per node), m = S/N, and from {@link FabricLinkParams}: B_L = one port's per-direction bandwidth,
 * C = the per-node cap over all ports (∞ when absent), α = per-hop latency, α_f = per-relay latency, k = cables per
 * mesh pair, o = switch oversubscription.
 *
 * Two rules give every formula below:
 * - Path latency of an h-hop message: lat(h) = h·α + (h−1)·α_f (exactly α for h = 1).
 * - Bandwidth term of a pattern: β = max(L_link / B_L, E_node / C), where L_link is the bytes on the busiest link
 *   direction and E_node the busiest node's egress (its own bytes plus the bytes it relays; ingress is symmetric in
 *   every pattern here). Relays go through the host (no switching inside these NICs), so they count against C.
 */

import type { FabricShape, LinkSpec } from '../hw/types';
import { cablesPerPairOf, maxHops, oversubscriptionOf, treeStepHops } from '../hw/fabric';
import {
  allReduceSec,
  allToAllSec,
  effectiveLink,
  ringAllGatherSec,
  ringAllReduceSec,
  ringReduceScatterSec,
  sendRecvSec,
  treeAllReduceSec,
} from './collectives';
import type {
  AllReduceAlgorithm,
  CollectiveKind,
  CollectiveTime,
  EffectiveLink,
} from './collectives';

/** The link and shape parameters a fabric collective uses. */
export interface FabricLinkParams {
  /** B_L: one port's (one cable's) per-direction bandwidth, after host-bounce effects (`effectiveLink`). Unit: B/s. */
  portBandwidthBytesPerSec: number;
  /** C: what one node can send (and, separately, receive) over all its ports at once; Infinity when uncapped. Unit: B/s. */
  allPortsCapBytesPerSec: number;
  /** α: latency of one hop (one message on one link), after host-bounce effects. Unit: seconds. */
  hopLatencySec: number;
  /** α_f: extra latency of each relaying node on a multi-hop path (0 when the link gives none). Unit: seconds. */
  forwardingLatencySec: number;
  /** k: cables between each pair of a full mesh, striped (1 for the other shapes). Count. */
  cablesPerPair: number;
  /** o: oversubscription of a switched fabric (1 for the other shapes). Dimensionless. */
  oversubscription: number;
}

/**
 * {@link FabricLinkParams} of `link` in `shape`, with `activeDevices` as in `effectiveLink` (group size for
 * collectives, 1 for send/recv).
 */
export function fabricLinkParams(
  link: LinkSpec,
  shape: FabricShape,
  activeDevices: number,
): FabricLinkParams {
  const eff = effectiveLink(link, activeDevices);
  return {
    portBandwidthBytesPerSec: eff.bandwidthBytesPerSec,
    allPortsCapBytesPerSec: link.allPortsCapPerDirectionBytesPerSec ?? Infinity,
    hopLatencySec: eff.latencySec,
    forwardingLatencySec: link.forwardingLatencySec ?? 0,
    cablesPerPair: cablesPerPairOf(shape),
    oversubscription: oversubscriptionOf(shape),
  };
}

/** Builds a {@link CollectiveTime} from its α and β parts. */
function time(
  latencySec: number,
  bandwidthSec: number,
  algorithm: CollectiveTime['algorithm'],
): CollectiveTime {
  return { latencySec, bandwidthSec, totalSec: latencySec + bandwidthSec, algorithm };
}

/** The zero cost of a group of one. */
const NO_TIME: CollectiveTime = { latencySec: 0, bandwidthSec: 0, totalSec: 0, algorithm: 'none' };

/** Latency of a message that crosses `hops` links: lat(h) = h·α + (h−1)·α_f (0 for h = 0). Unit: seconds. */
export function pathLatencySec(hops: number, p: FabricLinkParams): number {
  if (hops <= 0) return 0;
  // Formula: lat(h) = h·α + (h−1)·α_f
  // See docs/simulation-model.md §6 ("Fabric shapes")
  return hops * p.hopLatencySec + (hops - 1) * p.forwardingLatencySec;
}

/** B_one = min(k·B_L, C): one flow between two nodes over one (striped) pair or port. Unit: B/s. */
export function onePairBandwidthBytesPerSec(p: FabricLinkParams): number {
  // Formula: B_P = k·B_L;  B_one = min(B_P, C)
  return Math.min(p.cablesPerPair * p.portBandwidthBytesPerSec, p.allPortsCapBytesPerSec);
}

/** B_two = min(2·B_P, C): a node streaming on two ports at once (both directions of a bidirectional ring). Unit: B/s. */
function twoPortBandwidthBytesPerSec(p: FabricLinkParams): number {
  // Formula: B_two = min(2·k·B_L, C)
  return Math.min(2 * p.cablesPerPair * p.portBandwidthBytesPerSec, p.allPortsCapBytesPerSec);
}

/** B_mesh = min((N−1)·B_P, C): a full-mesh node sending to all N−1 peers at once. Unit: B/s. */
export function meshInjectionBandwidthBytesPerSec(n: number, p: FabricLinkParams): number {
  // Formula: B_mesh = min((N−1)·k·B_L, C)
  return Math.min((n - 1) * p.cablesPerPair * p.portBandwidthBytesPerSec, p.allPortsCapBytesPerSec);
}

/** An {@link EffectiveLink} with bandwidth `bandwidthBytesPerSec` and the one-hop latency α, for the switched formulas. */
function asLink(bandwidthBytesPerSec: number, p: FabricLinkParams): EffectiveLink {
  return { bandwidthBytesPerSec, latencySec: p.hopLatencySec };
}

// ---------------------------------------------------------------------------------------------------------------
// Daisy chain: linear reduce-scatter and all-gather (no relays: every transfer is between neighbours).

/**
 * Linear reduce-scatter on a chain of N ≥ 2 nodes: chunk c (S/N) is reduced at node c, partial sums flowing toward c
 * from both sides, one neighbour step at a time. Link j→j+1 carries the partials of chunks j+1 … N−1, so the busiest
 * link direction carries (N−1)·m (the end links); node j sends (N−1−j)·m right and j·m left, (N−1)·m in total.
 */
export function chainReduceScatterSec(
  messageBytes: number,
  n: number,
  p: FabricLinkParams,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = (N−1)·α + (N−1)/N · S / min(B_L, C)     (L = E = (N−1)·m)
  // See docs/simulation-model.md §6 ("Fabric shapes")
  return ringReduceScatterSec(
    messageBytes,
    n,
    asLink(Math.min(p.portBandwidthBytesPerSec, p.allPortsCapBytesPerSec), p),
  );
}

/**
 * Linear all-gather on a chain of N ≥ 2 nodes: every node's chunk (S/N) travels to both ends, one neighbour step at a
 * time. The busiest link direction carries (N−1)·m (the end links). An inner node sends its own chunk both ways and
 * forwards every other chunk once: (N+1)·m. That is forced for ANY all-gather on a chain: the N(N−1) chunk receptions
 * are all neighbour transfers, the two end nodes can only send their own chunk once each, and the N−2 inner nodes
 * share the rest, (N+1)·m each on average. With C ≥ (N+1)/(N−1)·B_L the link term dominates, the same as the ring
 * formula at B_L.
 */
export function chainAllGatherSec(
  messageBytes: number,
  n: number,
  p: FabricLinkParams,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  if (n === 2)
    return ringAllGatherSec(
      messageBytes,
      n,
      asLink(Math.min(p.portBandwidthBytesPerSec, p.allPortsCapBytesPerSec), p),
    );
  // Formula: t = (N−1)·α + max((N−1)/N · S/B_L, (N+1)/N · S/C)     (L = (N−1)·m, E = (N+1)·m)
  // See docs/simulation-model.md §6 ("Fabric shapes")
  const latencySec = (n - 1) * p.hopLatencySec;
  const linkSec = ((n - 1) / n) * (messageBytes / p.portBandwidthBytesPerSec);
  const nodeSec = ((n + 1) / n) * (messageBytes / p.allPortsCapBytesPerSec);
  return time(latencySec, Math.max(linkSec, nodeSec), 'ring');
}

/** Ring-class all-reduce on a chain: {@link chainReduceScatterSec} then {@link chainAllGatherSec}. */
export function chainAllReduceSec(
  messageBytes: number,
  n: number,
  p: FabricLinkParams,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = RS_chain(S, N) + AG_chain(S, N) = 2(N−1)·α + β_RS + β_AG
  const rs = chainReduceScatterSec(messageBytes, n, p);
  const ag = chainAllGatherSec(messageBytes, n, p);
  return time(rs.latencySec + ag.latencySec, rs.bandwidthSec + ag.bandwidthSec, 'ring');
}

// ---------------------------------------------------------------------------------------------------------------
// Per collective, per shape. `shape` is the group's view of the fabric (`groupFabric`): N = the group's nodes.

/**
 * Ring-class (reduce-scatter + all-gather) all-reduce over N nodes.
 * - switched (o ≥ 1): the switched ring at B_L/o;
 * - full_mesh: a bidirectional ring at min(2·B_P, C) for N ≥ 3 (two counter-rotating half rings use every link
 *   direction of a Hamiltonian cycle once); one striped pair at B_one for N = 2;
 * - ring: the same bidirectional ring at min(2·B_L, C);
 * - chain: {@link chainAllReduceSec}.
 */
export function fabricRingAllReduceSec(
  messageBytes: number,
  n: number,
  shape: FabricShape,
  p: FabricLinkParams,
): CollectiveTime {
  switch (shape.kind) {
    case 'switched':
      // Formula: t = 2(N−1)·α + 2(N−1)/N · S/(B_L/o)
      return ringAllReduceSec(
        messageBytes,
        n,
        asLink(p.portBandwidthBytesPerSec / p.oversubscription, p),
      );
    case 'full_mesh':
    case 'ring':
      // Formula: t = 2(N−1)·α + 2(N−1)/N · S/B_ring,  B_ring = min(2·B_P, C) (N ≥ 3), B_one (N = 2)
      return ringAllReduceSec(
        messageBytes,
        n,
        asLink(n >= 3 ? twoPortBandwidthBytesPerSec(p) : onePairBandwidthBytesPerSec(p), p),
      );
    case 'chain':
      return chainAllReduceSec(messageBytes, n, p);
  }
}

/**
 * Binomial-tree (reduce + broadcast) all-reduce over N nodes: ⌈log2 N⌉ steps each way; step k's partners sit 2^k
 * apart, h_k = `treeStepHops` links (1 on a switch or mesh). One flow per step at B_one (B_L/o on a switch).
 */
export function fabricTreeAllReduceSec(
  messageBytes: number,
  n: number,
  shape: FabricShape,
  p: FabricLinkParams,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  if (shape.kind === 'switched') {
    // Formula: t = 2⌈log2 N⌉·α + 2·S/(B_L/o)
    return treeAllReduceSec(
      messageBytes,
      n,
      asLink(p.portBandwidthBytesPerSec / p.oversubscription, p),
    );
  }
  if (shape.kind === 'full_mesh') {
    // Formula: t = 2⌈log2 N⌉·α + 2·S/B_one
    return treeAllReduceSec(messageBytes, n, asLink(onePairBandwidthBytesPerSec(p), p));
  }
  // Formula: t = 2·Σ_{k=0}^{⌈log2 N⌉−1} lat(h_k) + 2·S/B_one
  // See docs/simulation-model.md §6 ("Fabric shapes")
  const steps = Math.ceil(Math.log2(n));
  let halfLatencySec = 0;
  for (let k = 0; k < steps; k++) halfLatencySec += pathLatencySec(treeStepHops(shape, n, k), p);
  const bandwidthSec = (2 * messageBytes) / onePairBandwidthBytesPerSec(p);
  return time(2 * halfLatencySec, bandwidthSec, 'tree');
}

/**
 * Direct all-reduce on a full mesh: a one-shot all-pairs reduce-scatter, then a one-shot all-pairs all-gather. Each
 * phase sends m to each of the N−1 peers at once: m per pair, (N−1)·m per node.
 */
export function meshDirectAllReduceSec(
  messageBytes: number,
  n: number,
  p: FabricLinkParams,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  // Formula: t = 2α + 2·(N−1)/N · S/B_mesh,  B_mesh = min((N−1)·B_P, C)
  // See docs/simulation-model.md §6 ("Fabric shapes")
  const latencySec = 2 * p.hopLatencySec;
  const bandwidthSec =
    2 * (((n - 1) / n) * (messageBytes / meshInjectionBandwidthBytesPerSec(n, p)));
  return time(latencySec, bandwidthSec, 'direct');
}

/**
 * All-reduce on a fabric with the chosen algorithm. `'best'` = the fastest of ring and tree, and on a full mesh of
 * N ≥ 3 also direct (ties keep ring, then direct).
 */
export function fabricAllReduceSec(
  messageBytes: number,
  n: number,
  shape: FabricShape,
  p: FabricLinkParams,
  algorithm: AllReduceAlgorithm,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  const ring = fabricRingAllReduceSec(messageBytes, n, shape, p);
  const tree = fabricTreeAllReduceSec(messageBytes, n, shape, p);
  switch (algorithm) {
    case 'ring':
      return ring;
    case 'tree':
      return tree;
    case 'best': {
      // Formula: t = min(t_ring, t_direct (full mesh, N ≥ 3), t_tree)
      let best = ring;
      if (shape.kind === 'full_mesh' && n >= 3) {
        const direct = meshDirectAllReduceSec(messageBytes, n, p);
        if (direct.totalSec < best.totalSec) best = direct;
      }
      return tree.totalSec < best.totalSec ? tree : best;
    }
    default: {
      const _never: never = algorithm;
      throw new Error(`unknown all-reduce algorithm ${String(_never)}`);
    }
  }
}

/**
 * All-gather (S = the full OUTPUT) or reduce-scatter (S = the full INPUT) over N nodes.
 * - switched: the switched ring at B_L/o;
 * - full_mesh, N ≥ 3: direct (one-shot all-pairs), α + (N−1)/N · S/B_mesh, which beats the ring on both terms;
 *   N = 2: one striped pair at B_one;
 * - ring: the bidirectional ring at min(2·B_L, C);
 * - chain: {@link chainAllGatherSec} / {@link chainReduceScatterSec} (they differ when C binds).
 */
export function fabricAllGatherOrReduceScatterSec(
  kind: 'all_gather' | 'reduce_scatter',
  messageBytes: number,
  n: number,
  shape: FabricShape,
  p: FabricLinkParams,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  switch (shape.kind) {
    case 'switched':
      // Formula: t = (N−1)·α + (N−1)/N · S/(B_L/o)
      return ringAllGatherSec(
        messageBytes,
        n,
        asLink(p.portBandwidthBytesPerSec / p.oversubscription, p),
      );
    case 'full_mesh': {
      if (n === 2)
        return ringAllGatherSec(messageBytes, n, asLink(onePairBandwidthBytesPerSec(p), p));
      // Formula: t = α + (N−1)/N · S/B_mesh
      // See docs/simulation-model.md §6 ("Fabric shapes")
      const bandwidthSec = ((n - 1) / n) * (messageBytes / meshInjectionBandwidthBytesPerSec(n, p));
      return time(p.hopLatencySec, bandwidthSec, 'direct');
    }
    case 'ring':
      // Formula: t = (N−1)·α + (N−1)/N · S/min(2·B_L, C)
      return ringAllGatherSec(messageBytes, n, asLink(twoPortBandwidthBytesPerSec(p), p));
    case 'chain':
      return kind === 'all_gather'
        ? chainAllGatherSec(messageBytes, n, p)
        : chainReduceScatterSec(messageBytes, n, p);
  }
}

/**
 * All-to-all of S bytes per device (m = S/N to each node, the self share included in S) over N nodes, every flow on
 * its shortest path at once; α is the farthest destination's lat(maxHops).
 * - switched: α + (N−1)/N · S/(B_L/o);
 * - full_mesh: α + (N−1)/N · S/B_mesh (N = 2: B_one);
 * - ring: each link direction carries ⌊N²/4⌋/2 · m (half of the N/2-distance flows each way for even N), a node's
 *   egress is twice that over its two ports, so β = ⌊N²/4⌋·m / min(2·B_L, C);
 * - chain: link j→j+1 carries (j+1)(N−1−j)·m ≤ ⌊N²/4⌋·m; inner node j sends [N−1 + 2j(N−1−j)]·m ≤ E_N·m,
 *   E_N = (N−1) + 2⌊(N−1)²/4⌋.
 */
export function fabricAllToAllSec(
  messageBytes: number,
  n: number,
  shape: FabricShape,
  p: FabricLinkParams,
): CollectiveTime {
  if (n <= 1) return NO_TIME;
  switch (shape.kind) {
    case 'switched':
      // Formula: t = α + (N−1)/N · S/(B_L/o)
      return allToAllSec(
        messageBytes,
        n,
        asLink(p.portBandwidthBytesPerSec / p.oversubscription, p),
      );
    case 'full_mesh':
      // Formula: t = α + (N−1)/N · S/B_mesh     (B_mesh = B_one for N = 2)
      return allToAllSec(messageBytes, n, asLink(meshInjectionBandwidthBytesPerSec(n, p), p));
    case 'ring': {
      // Formula: t = lat(⌊N/2⌋) + (⌊N²/4⌋/N) · S/min(2·B_L, C)
      // See docs/simulation-model.md §6 ("Fabric shapes")
      const latencySec = pathLatencySec(maxHops(shape, n), p);
      const linkLoadPerNodeShare = Math.floor((n * n) / 4) / n;
      const bandwidthSec = linkLoadPerNodeShare * (messageBytes / twoPortBandwidthBytesPerSec(p));
      return time(latencySec, bandwidthSec, 'direct');
    }
    case 'chain': {
      // Formula: t = lat(N−1) + max(⌊N²/4⌋/N · S/B_L, E_N/N · S/C),  E_N = (N−1) + 2⌊(N−1)²/4⌋
      // See docs/simulation-model.md §6 ("Fabric shapes")
      const latencySec = pathLatencySec(maxHops(shape, n), p);
      const busiestLinkShare = Math.floor((n * n) / 4) / n;
      const busiestNodeShare = (n - 1 + 2 * Math.floor(((n - 1) * (n - 1)) / 4)) / n;
      const linkSec = busiestLinkShare * (messageBytes / p.portBandwidthBytesPerSec);
      const nodeSec = busiestNodeShare * (messageBytes / p.allPortsCapBytesPerSec);
      return time(latencySec, Math.max(linkSec, nodeSec), 'direct');
    }
  }
}

/**
 * Point-to-point send of S bytes (a pipeline-stage boundary) over `hops` links. The boundary's flows all run the
 * same way (rank r → r + tp·dp), so on a ring or chain the busiest link and the busiest relay each carry `hops` of
 * them (documented approximation: on a ring they could split over both directions; not modeled).
 * - switched: α + S/(B_L/o) (hops = 1);
 * - full_mesh: α + S/B_one (hops = 1);
 * - ring, chain: lat(h) + h·S/B_one.
 */
export function fabricSendRecvSec(
  messageBytes: number,
  hops: number,
  shape: FabricShape,
  p: FabricLinkParams,
): CollectiveTime {
  switch (shape.kind) {
    case 'switched':
      // Formula: t = α + S/(B_L/o)
      return sendRecvSec(messageBytes, asLink(p.portBandwidthBytesPerSec / p.oversubscription, p));
    case 'full_mesh':
      // Formula: t = α + S/B_one
      return sendRecvSec(messageBytes, asLink(onePairBandwidthBytesPerSec(p), p));
    case 'ring':
    case 'chain': {
      if (hops <= 1) return sendRecvSec(messageBytes, asLink(onePairBandwidthBytesPerSec(p), p));
      // Formula: t = lat(h) + h·S/B_one
      // See docs/simulation-model.md §6 ("Fabric shapes")
      const bandwidthSec = hops * (messageBytes / onePairBandwidthBytesPerSec(p));
      return time(pathLatencySec(hops, p), bandwidthSec, 'direct');
    }
  }
}

/**
 * Cost of one collective of `kind` with S = `messageBytes` per device over `groupSize` nodes (one device each) of the
 * fabric `shape` (the group's view, `groupFabric`), on `link`. `sendHops` = links a send_recv crosses (default 1).
 * `sendActiveDevices` = simultaneous send/recv ranks per host (count, default 1; PP uses min(dpn, tp·dp)).
 * Two nodes on a full mesh or a chain are one (striped) link at B_one: the switched formulas at that bandwidth, so a
 * single cable with C ≥ B_L gives exactly the switched result.
 */
export function fabricCollectiveTime(
  kind: CollectiveKind,
  messageBytes: number,
  groupSize: number,
  shape: FabricShape,
  link: LinkSpec,
  algorithm: AllReduceAlgorithm,
  sendHops = 1,
  sendActiveDevices = 1,
): CollectiveTime {
  const p = fabricLinkParams(link, shape, kind === 'send_recv' ? sendActiveDevices : groupSize);
  if (
    shape.kind !== 'switched' &&
    shape.kind !== 'ring' &&
    groupSize === 2 &&
    kind !== 'send_recv'
  ) {
    // Formula: two nodes = one link at B_one = min(k·B_L, C)
    const single = asLink(onePairBandwidthBytesPerSec(p), p);
    switch (kind) {
      case 'all_reduce':
        return allReduceSec(messageBytes, groupSize, single, algorithm);
      case 'all_gather':
        return ringAllGatherSec(messageBytes, groupSize, single);
      case 'reduce_scatter':
        return ringReduceScatterSec(messageBytes, groupSize, single);
      case 'all_to_all':
        return allToAllSec(messageBytes, groupSize, single);
    }
  }
  switch (kind) {
    case 'all_reduce':
      return fabricAllReduceSec(messageBytes, groupSize, shape, p, algorithm);
    case 'all_gather':
    case 'reduce_scatter':
      return fabricAllGatherOrReduceScatterSec(kind, messageBytes, groupSize, shape, p);
    case 'all_to_all':
      return fabricAllToAllSec(messageBytes, groupSize, shape, p);
    case 'send_recv':
      return fabricSendRecvSec(messageBytes, sendHops, shape, p);
    default: {
      const _never: never = kind;
      throw new Error(`unknown collective ${String(_never)}`);
    }
  }
}
