/**
 * Placement of parallel groups on a topology: which link level each TP / EP group and each pipeline-stage
 * boundary uses, and whether the parallelism fits the devices at all.
 * See docs/simulation-model.md §5 ("Rank layout").
 *
 * Rank layout: `rank = tpIdx + tp · (dpIdx + dp · ppIdx)`; ranks fill node 0 first (`node = floor(rank / dpn)`,
 * dpn = devicesPerNode). TP groups (and EP groups, which are `ep` consecutive ranks) are therefore contiguous.
 */

import type { ParallelConfig } from '../ir/model';
import type { TopologySpec } from '../hw/types';
import type { CollectivePath } from './collectives';

/** How a contiguous group of devices lies on the topology. */
export interface GroupSpan {
  /** Devices in the group. Count. */
  n: number;
  /** Group members in each node it touches (n1). Count. */
  devicesPerNodeInGroup: number;
  /** Nodes the group touches (n2); n = n1 · n2. Count. */
  nodesSpanned: number;
}

/**
 * Errors placing a contiguous group of `size` devices; empty = valid. A group is intra-node if
 * `size ≤ dpn` and `dpn % size == 0` (it never straddles a node boundary), hierarchical if `size % dpn == 0`
 * and `size / dpn ≤ nodes`; anything else straddles nodes unevenly and is an error.
 */
export function groupSpanErrors(size: number, topo: TopologySpec): string[] {
  const dpn = topo.devicesPerNode;
  if (!Number.isInteger(size) || size < 1) return [`group size must be an integer ≥ 1 (got ${size})`];
  if (size <= dpn && dpn % size === 0) return [];
  if (size % dpn === 0 && size / dpn <= topo.nodes) return [];
  if (size % dpn === 0) return [`a group of ${size} needs ${size / dpn} nodes of ${dpn}; the topology has ${topo.nodes}`];
  return [`a group of ${size} devices does not tile nodes of ${dpn} devices (needs size | ${dpn} or ${dpn} | size)`];
}

/** Span of a contiguous group of `size` devices (TP and EP groups). Throws if {@link groupSpanErrors} reports any. */
export function groupSpan(size: number, topo: TopologySpec): GroupSpan {
  const errors = groupSpanErrors(size, topo);
  if (errors.length > 0) throw new Error(errors.join('\n'));
  const dpn = topo.devicesPerNode;
  if (size <= dpn) return { n: size, devicesPerNodeInGroup: size, nodesSpanned: 1 };
  // Formula: n1 = dpn;  n2 = n / dpn
  return { n: size, devicesPerNodeInGroup: dpn, nodesSpanned: size / dpn };
}

/**
 * True if the boundary between pipeline stage `stage` and `stage + 1` crosses nodes: some rank r of the stage
 * sends to r + tp·dp in another node (`floor(r / dpn)` differs).
 */
export function ppBoundaryIsInterNode(par: ParallelConfig, topo: TopologySpec, stage: number): boolean {
  const dpn = topo.devicesPerNode;
  const stride = par.tp * par.dp;
  for (let i = 0; i < stride; i++) {
    // Formula: r = tpIdx + tp·(dpIdx + dp·s) = s·tp·dp + i,  partner = r + tp·dp
    const rank = stage * stride + i;
    const partner = rank + stride;
    if (Math.floor(rank / dpn) !== Math.floor(partner / dpn)) return true;
  }
  return false;
}

/**
 * The {@link CollectivePath} of a group with span `span`: `'none'` for one device, flat intra-node, flat
 * inter-node (one member per node), or hierarchical. Throws if the topology lacks the needed link.
 */
export function collectivePath(span: GroupSpan, topo: TopologySpec): CollectivePath {
  if (span.n <= 1) return { kind: 'none' };
  const needIntra = span.devicesPerNodeInGroup > 1;
  const needInter = span.nodesSpanned > 1;
  if (needIntra && topo.intraNode === undefined) throw new Error(`topology ${topo.id} has no intraNode link`);
  if (needInter && topo.interNode === undefined) throw new Error(`topology ${topo.id} has no interNode link`);
  if (needIntra && needInter && topo.intraNode !== undefined && topo.interNode !== undefined) {
    return {
      kind: 'hierarchical',
      devicesPerNode: span.devicesPerNodeInGroup,
      nodes: span.nodesSpanned,
      intra: topo.intraNode,
      inter: topo.interNode,
    };
  }
  if (needInter && topo.interNode !== undefined) return { kind: 'flat', level: 'inter', link: topo.interNode };
  if (topo.intraNode !== undefined) return { kind: 'flat', level: 'intra', link: topo.intraNode };
  throw new Error(`topology ${topo.id} has no link for a group of ${span.n}`);
}

/** The flat path of the activation send from stage `stage` to `stage + 1` (intra- or inter-node link). Throws if missing. */
export function ppSendPath(par: ParallelConfig, topo: TopologySpec, stage: number): CollectivePath {
  const inter = ppBoundaryIsInterNode(par, topo, stage);
  const link = inter ? topo.interNode : topo.intraNode;
  if (link === undefined) throw new Error(`topology ${topo.id} has no ${inter ? 'interNode' : 'intraNode'} link for the PP boundary after stage ${stage}`);
  return { kind: 'flat', level: inter ? 'inter' : 'intra', link };
}

/** Result of {@link checkPlacement}: errors make the input invalid; warnings are informational lines. */
export interface PlacementCheck {
  /** Invalid placement (too many devices, a group that straddles nodes unevenly, a missing link). */
  errors: string[];
  /** E.g. unused devices. */
  warnings: string[];
}

/**
 * Checks that `par` fits `topo`: world `tp·pp·dp ≤ dpn·nodes` (unused devices are a warning); TP and EP groups
 * place per {@link groupSpanErrors}; every link a TP/EP group or a PP boundary needs exists.
 */
export function checkPlacement(par: ParallelConfig, topo: TopologySpec): PlacementCheck {
  const errors: string[] = [];
  const warnings: string[] = [];
  const world = par.tp * par.pp * par.dp;
  const devices = topo.devicesPerNode * topo.nodes;
  if (world > devices) {
    errors.push(`tp·pp·dp = ${world} devices needed, the topology has ${devices}`);
    return { errors, warnings };
  }
  if (world < devices) warnings.push(`${devices - world} of ${devices} devices are unused (tp·pp·dp = ${world})`);
  for (const [name, size] of [['tp', par.tp], ['ep', par.ep]] as const) {
    const spanErrors = groupSpanErrors(size, topo).map((e) => `${name}: ${e}`);
    errors.push(...spanErrors);
    if (spanErrors.length === 0) {
      try {
        collectivePath(groupSpan(size, topo), topo);
      } catch (e) {
        errors.push(`${name}: ${(e as Error).message}`);
      }
    }
  }
  for (let stage = 0; stage + 1 < par.pp; stage++) {
    try {
      ppSendPath(par, topo, stage);
    } catch (e) {
      errors.push(`pp: ${(e as Error).message}`);
    }
  }
  return { errors, warnings };
}
