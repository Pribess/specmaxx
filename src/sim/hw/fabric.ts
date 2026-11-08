/**
 * Geometry of the inter-node fabric shapes ({@link FabricShape}): hop counts between nodes, ports each node needs,
 * the hops of a binomial-tree step, how a contiguous group of nodes sees the fabric, and the structural errors of a
 * shape. Pure functions over plain data, shared by placement (`engine/placement.ts`), the cost formulas
 * (`engine/fabricCollectives.ts`), the setup catalog and the UI.
 * See docs/simulation-model.md §6 ("Fabric shapes").
 *
 * Nodes are numbered 0 … n−1 in rank order (`node = floor(rank / devicesPerNode)`, docs §5 "Rank layout"); a ring
 * cables node i to i ± 1 (mod n), a chain cables node i to i + 1 only.
 */

import type { FabricShape } from './types';

/** The fabric shape kinds, in display order. */
export const FABRIC_SHAPE_KINDS = ['switched', 'full_mesh', 'ring', 'chain'] as const;

/** One of {@link FABRIC_SHAPE_KINDS}. */
export type FabricShapeKind = FabricShape['kind'];

/** The shape an absent `TopologySpec.interNodeFabric` means: one port per node into a non-blocking switch. */
export const DEFAULT_FABRIC: FabricShape = { kind: 'switched' };

/**
 * True for the pre-fabric convention: no shape given, or `'switched'` with oversubscription absent or 1. The engine
 * then runs exactly the formulas it ran before fabric shapes existed (bit-identical results).
 */
export function isNonBlockingSwitched(shape: FabricShape | undefined): boolean {
  if (shape === undefined) return true;
  return (
    shape.kind === 'switched' &&
    (shape.oversubscription === undefined || shape.oversubscription === 1)
  );
}

/** Cables between each pair of nodes, k: `cablesPerPair` of a full mesh (absent = 1), 1 for every other shape. Count. */
export function cablesPerPairOf(shape: FabricShape | undefined): number {
  return shape?.kind === 'full_mesh' ? (shape.cablesPerPair ?? 1) : 1;
}

/** Oversubscription o of a switched fabric (absent = 1, non-blocking), 1 for every other shape. Dimensionless. */
export function oversubscriptionOf(shape: FabricShape | undefined): number {
  return shape?.kind === 'switched' ? (shape.oversubscription ?? 1) : 1;
}

/**
 * Links a message from node `i` to node `j` crosses in a fabric of `n` nodes (0 when i = j). Count.
 * - switched, full_mesh: 1 (the switch traversal is inside the link's latency);
 * - ring: min(|i−j|, n−|i−j|), the shorter way round;
 * - chain: |i−j|.
 */
export function nodeHops(shape: FabricShape | undefined, i: number, j: number, n: number): number {
  if (i === j) return 0;
  const d = Math.abs(i - j);
  switch (shape?.kind ?? 'switched') {
    case 'switched':
    case 'full_mesh':
      return 1;
    case 'ring':
      // Formula: h = min(|i−j|, n − |i−j|)
      return Math.min(d, n - d);
    case 'chain':
      // Formula: h = |i−j|
      return d;
  }
}

/** Most links any message crosses in a fabric of `n` nodes: 1, 1, ⌊n/2⌋, n−1 for switched, full_mesh, ring, chain (0 when n ≤ 1). Count. */
export function maxHops(shape: FabricShape | undefined, n: number): number {
  if (n <= 1) return 0;
  switch (shape?.kind ?? 'switched') {
    case 'switched':
    case 'full_mesh':
      return 1;
    case 'ring':
      return Math.floor(n / 2);
    case 'chain':
      return n - 1;
  }
}

/**
 * Ports of the inter-node link one node needs for the shape over `n` nodes (0 when n ≤ 1). Count.
 * - switched: 1;
 * - full_mesh: (n−1)·k (k = cables per pair);
 * - ring: 2;
 * - chain: 2 for n ≥ 3 (the inner nodes), 1 for n = 2.
 */
export function portsNeeded(shape: FabricShape | undefined, n: number): number {
  if (n <= 1) return 0;
  switch (shape?.kind ?? 'switched') {
    case 'switched':
      return 1;
    case 'full_mesh':
      return (n - 1) * cablesPerPairOf(shape);
    case 'ring':
      return 2;
    case 'chain':
      return n >= 3 ? 2 : 1;
  }
}

/**
 * Links crossed by a message of step `k` of a binomial tree (reduce or broadcast) over `n` nodes, where step-k
 * partners sit 2^k apart in node order (k = 0 … ⌈log2 n⌉ − 1). Count.
 * - ring: min(2^k, n − 2^k);
 * - chain: 2^k;
 * - switched, full_mesh: 1.
 */
export function treeStepHops(shape: FabricShape | undefined, n: number, k: number): number {
  const distance = 2 ** k;
  switch (shape?.kind ?? 'switched') {
    case 'switched':
    case 'full_mesh':
      return 1;
    case 'ring':
      // Formula: h_k = min(2^k, n − 2^k)
      return Math.min(distance, n - distance);
    case 'chain':
      // Formula: h_k = 2^k
      return distance;
  }
}

/**
 * The fabric as a contiguous group of `groupNodes` of the topology's `totalNodes` nodes sees it. A ring that the group
 * does not cover entirely is a chain of `groupNodes` (a segment; the long way round is ignored, and concurrent groups
 * use disjoint segments). Every other shape, and a ring the group covers, is unchanged.
 */
export function groupFabric(
  shape: FabricShape | undefined,
  groupNodes: number,
  totalNodes: number,
): FabricShape | undefined {
  if (shape?.kind === 'ring' && groupNodes < totalNodes) return { kind: 'chain' };
  return shape;
}

/** What {@link fabricShapeErrors} checks a shape against. */
export interface FabricShapeContext {
  /** Nodes the fabric connects. Count. */
  nodes: number;
  /** Devices per node; every shape but `'switched'` needs 1. Count. */
  devicesPerNode: number;
  /** Usable ports of the link per node (optional; the setup catalog knows it, `TopologySpec` does not). Count. */
  portsPerNode?: number;
  /** Whether a switch exists for the link (optional; `false` rejects `'switched'`). */
  switchable?: boolean;
}

/** Words for a shape kind in error messages. */
const SHAPE_WORDS: Record<FabricShapeKind, string> = {
  switched: 'switched fabric',
  full_mesh: 'full mesh',
  ring: 'ring',
  chain: 'chain',
};

/**
 * Errors of a fabric shape in a context; empty = valid. Structural rules (what `validateTopology` uses): a shape other
 * than `'switched'` needs one device per node; a ring needs ≥ 3 nodes; `cablesPerPair` is an integer ≥ 1;
 * `oversubscription` is finite ≥ 1. With `portsPerNode`: the shape needs at most that many ports. With
 * `switchable: false`: no `'switched'` shape.
 */
export function fabricShapeErrors(shape: FabricShape, ctx: FabricShapeContext): string[] {
  const errors: string[] = [];
  if (shape.kind !== 'switched' && ctx.devicesPerNode !== 1) {
    errors.push(
      `a ${SHAPE_WORDS[shape.kind]} needs one device per node (got ${ctx.devicesPerNode}); nodes with several devices network through a switch`,
    );
  }
  if (shape.kind === 'ring' && ctx.nodes < 3)
    errors.push(`a ring needs at least 3 nodes (got ${ctx.nodes}); two nodes are a full mesh`);
  if (
    shape.kind === 'full_mesh' &&
    shape.cablesPerPair !== undefined &&
    !(Number.isInteger(shape.cablesPerPair) && shape.cablesPerPair >= 1)
  ) {
    errors.push(`cablesPerPair must be an integer ≥ 1 (got ${shape.cablesPerPair})`);
  }
  if (
    shape.kind === 'switched' &&
    shape.oversubscription !== undefined &&
    !(Number.isFinite(shape.oversubscription) && shape.oversubscription >= 1)
  ) {
    errors.push(`oversubscription must be finite and ≥ 1 (got ${shape.oversubscription})`);
  }
  if (ctx.portsPerNode !== undefined) {
    const needed = portsNeeded(shape, ctx.nodes);
    if (needed > ctx.portsPerNode) {
      errors.push(
        `a ${SHAPE_WORDS[shape.kind]} of ${ctx.nodes} needs ${needed} ports per node; this machine has ${ctx.portsPerNode} usable`,
      );
    }
  }
  if (ctx.switchable === false && shape.kind === 'switched')
    errors.push('no switch exists for this link');
  return errors;
}
