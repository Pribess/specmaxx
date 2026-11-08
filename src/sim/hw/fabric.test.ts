import { describe, expect, it } from 'vitest';
import {
  FABRIC_SHAPE_KINDS,
  cablesPerPairOf,
  fabricShapeErrors,
  groupFabric,
  isNonBlockingSwitched,
  maxHops,
  nodeHops,
  oversubscriptionOf,
  portsNeeded,
  treeStepHops,
} from './fabric';
import type { FabricShape } from './types';

const SWITCHED: FabricShape = { kind: 'switched' };
const MESH: FabricShape = { kind: 'full_mesh' };
const MESH2: FabricShape = { kind: 'full_mesh', cablesPerPair: 2 };
const RING: FabricShape = { kind: 'ring' };
const CHAIN: FabricShape = { kind: 'chain' };
const NS = [2, 3, 4, 5, 6, 7, 8];

/** Shortest-path hops between every pair of nodes of a graph given by its edges (breadth-first search). */
function bfsHops(n: number, edges: [number, number][]): number[][] {
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adj[a].push(b);
    adj[b].push(a);
  }
  return Array.from({ length: n }, (_, s) => {
    const dist = Array<number>(n).fill(Infinity);
    dist[s] = 0;
    const queue = [s];
    while (queue.length > 0) {
      const u = queue.shift()!;
      for (const v of adj[u]) {
        if (dist[v] !== Infinity) continue;
        dist[v] = dist[u] + 1;
        queue.push(v);
      }
    }
    return dist;
  });
}

/** Cables of a shape over n nodes (one entry per cable; a mesh with k cables per pair lists each pair once). */
function cables(shape: FabricShape, n: number): [number, number][] {
  const out: [number, number][] = [];
  if (shape.kind === 'chain' || shape.kind === 'ring')
    for (let i = 0; i + 1 < n; i++) out.push([i, i + 1]);
  if (shape.kind === 'ring') out.push([n - 1, 0]);
  if (shape.kind === 'full_mesh')
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) out.push([i, j]);
  return out;
}

describe('fabric shape geometry', () => {
  it('lists the kinds in display order', () => {
    expect(FABRIC_SHAPE_KINDS).toEqual(['switched', 'full_mesh', 'ring', 'chain']);
  });

  it('isNonBlockingSwitched: absent, switched, switched o = 1 only', () => {
    expect(isNonBlockingSwitched(undefined)).toBe(true);
    expect(isNonBlockingSwitched(SWITCHED)).toBe(true);
    expect(isNonBlockingSwitched({ kind: 'switched', oversubscription: 1 })).toBe(true);
    expect(isNonBlockingSwitched({ kind: 'switched', oversubscription: 2 })).toBe(false);
    for (const s of [MESH, MESH2, RING, CHAIN]) expect(isNonBlockingSwitched(s)).toBe(false);
  });

  it('cablesPerPairOf and oversubscriptionOf default to 1', () => {
    expect([undefined, SWITCHED, MESH, MESH2, RING, CHAIN].map(cablesPerPairOf)).toEqual([
      1, 1, 1, 2, 1, 1,
    ]);
    expect(
      [undefined, SWITCHED, { kind: 'switched', oversubscription: 3 } as FabricShape, MESH].map(
        oversubscriptionOf,
      ),
    ).toEqual([1, 1, 3, 1]);
  });

  it('maxHops for N = 2..8', () => {
    expect(NS.map((n) => maxHops(SWITCHED, n))).toEqual([1, 1, 1, 1, 1, 1, 1]);
    expect(NS.map((n) => maxHops(MESH, n))).toEqual([1, 1, 1, 1, 1, 1, 1]);
    expect(NS.map((n) => maxHops(RING, n))).toEqual([1, 1, 2, 2, 3, 3, 4]);
    expect(NS.map((n) => maxHops(CHAIN, n))).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(maxHops(RING, 1)).toBe(0);
  });

  it('portsNeeded for N = 2..8', () => {
    expect(NS.map((n) => portsNeeded(SWITCHED, n))).toEqual([1, 1, 1, 1, 1, 1, 1]);
    expect(NS.map((n) => portsNeeded(MESH, n))).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(NS.map((n) => portsNeeded(MESH2, n))).toEqual([2, 4, 6, 8, 10, 12, 14]);
    expect(NS.map((n) => portsNeeded(RING, n))).toEqual([2, 2, 2, 2, 2, 2, 2]);
    expect(NS.map((n) => portsNeeded(CHAIN, n))).toEqual([1, 2, 2, 2, 2, 2, 2]);
    expect(portsNeeded(MESH, 1)).toBe(0);
  });

  it('nodeHops: hand-checked rows', () => {
    expect([0, 1, 2, 3, 4].map((j) => nodeHops(RING, 0, j, 5))).toEqual([0, 1, 2, 2, 1]);
    expect([0, 1, 2, 3, 4, 5].map((j) => nodeHops(RING, 0, j, 6))).toEqual([0, 1, 2, 3, 2, 1]);
    expect([0, 1, 2, 3].map((j) => nodeHops(CHAIN, 1, j, 4))).toEqual([1, 0, 1, 2]);
    expect([0, 1, 2, 3].map((j) => nodeHops(MESH, 1, j, 4))).toEqual([1, 0, 1, 1]);
    expect([0, 1, 2, 3].map((j) => nodeHops(undefined, 3, j, 4))).toEqual([1, 1, 1, 0]);
  });

  it('nodeHops and maxHops equal shortest paths over the cables (BFS), ring N = 3..8, chain and mesh N = 2..8', () => {
    for (const shape of [RING, CHAIN, MESH]) {
      for (const n of NS) {
        if (shape.kind === 'ring' && n < 3) continue;
        const bfs = bfsHops(n, cables(shape, n));
        let worst = 0;
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            expect(nodeHops(shape, i, j, n), `${shape.kind} n ${n} ${i}→${j}`).toBe(bfs[i][j]);
            worst = Math.max(worst, bfs[i][j]);
          }
        }
        expect(maxHops(shape, n), `${shape.kind} n ${n}`).toBe(worst);
      }
    }
  });

  it('treeStepHops: partners 2^k apart', () => {
    const steps = (s: FabricShape, n: number) =>
      Array.from({ length: Math.ceil(Math.log2(n)) }, (_, k) => treeStepHops(s, n, k));
    expect(steps(RING, 4)).toEqual([1, 2]);
    expect(steps(RING, 5)).toEqual([1, 2, 1]);
    expect(steps(RING, 6)).toEqual([1, 2, 2]);
    expect(steps(RING, 8)).toEqual([1, 2, 4]);
    expect(steps(CHAIN, 4)).toEqual([1, 2]);
    expect(steps(CHAIN, 8)).toEqual([1, 2, 4]);
    expect(steps(MESH, 8)).toEqual([1, 1, 1]);
    expect(steps(SWITCHED, 8)).toEqual([1, 1, 1]);
    // The step-k hops are nodeHops of nodes 0 and 2^k.
    for (const n of [3, 4, 5, 6, 7, 8]) {
      for (const s of [RING, CHAIN])
        steps(s, n).forEach((h, k) => expect(h).toBe(nodeHops(s, 0, 2 ** k, n)));
    }
  });

  it('groupFabric: a ring segment is a chain; everything else is unchanged', () => {
    expect(groupFabric(RING, 2, 4)).toEqual(CHAIN);
    expect(groupFabric(RING, 3, 4)).toEqual(CHAIN);
    expect(groupFabric(RING, 4, 4)).toBe(RING);
    expect(groupFabric(CHAIN, 2, 4)).toBe(CHAIN);
    expect(groupFabric(MESH2, 2, 4)).toBe(MESH2);
    expect(groupFabric(undefined, 2, 4)).toBeUndefined();
  });
});

describe('fabricShapeErrors', () => {
  it('accepts the plain shapes', () => {
    expect(fabricShapeErrors(SWITCHED, { nodes: 4, devicesPerNode: 8 })).toEqual([]);
    expect(fabricShapeErrors(RING, { nodes: 4, devicesPerNode: 1 })).toEqual([]);
    expect(fabricShapeErrors(CHAIN, { nodes: 2, devicesPerNode: 1 })).toEqual([]);
    expect(fabricShapeErrors(MESH2, { nodes: 2, devicesPerNode: 1 })).toEqual([]);
  });

  it('a switchless shape needs one device per node', () => {
    for (const s of [MESH, RING, CHAIN])
      expect(fabricShapeErrors(s, { nodes: 4, devicesPerNode: 2 })).toHaveLength(1);
  });

  it('a ring needs at least 3 nodes', () => {
    expect(fabricShapeErrors(RING, { nodes: 2, devicesPerNode: 1 })).toEqual([
      'a ring needs at least 3 nodes (got 2); two nodes are a full mesh',
    ]);
  });

  it('cablesPerPair and oversubscription ranges', () => {
    expect(
      fabricShapeErrors({ kind: 'full_mesh', cablesPerPair: 0 }, { nodes: 2, devicesPerNode: 1 }),
    ).toHaveLength(1);
    expect(
      fabricShapeErrors({ kind: 'full_mesh', cablesPerPair: 1.5 }, { nodes: 2, devicesPerNode: 1 }),
    ).toHaveLength(1);
    expect(
      fabricShapeErrors(
        { kind: 'switched', oversubscription: 0.5 },
        { nodes: 2, devicesPerNode: 1 },
      ),
    ).toHaveLength(1);
    expect(
      fabricShapeErrors(
        { kind: 'switched', oversubscription: Infinity },
        { nodes: 2, devicesPerNode: 1 },
      ),
    ).toHaveLength(1);
    expect(
      fabricShapeErrors({ kind: 'switched', oversubscription: 4 }, { nodes: 2, devicesPerNode: 8 }),
    ).toEqual([]);
  });

  it('ports: a full mesh beyond the ports is rejected', () => {
    expect(fabricShapeErrors(MESH, { nodes: 5, devicesPerNode: 1, portsPerNode: 3 })).toEqual([
      'a full mesh of 5 needs 4 ports per node; this machine has 3 usable',
    ]);
    expect(fabricShapeErrors(MESH, { nodes: 4, devicesPerNode: 1, portsPerNode: 3 })).toEqual([]);
    expect(fabricShapeErrors(MESH2, { nodes: 4, devicesPerNode: 1, portsPerNode: 5 })).toHaveLength(
      1,
    );
    expect(fabricShapeErrors(MESH2, { nodes: 4, devicesPerNode: 1, portsPerNode: 6 })).toEqual([]);
    expect(fabricShapeErrors(RING, { nodes: 8, devicesPerNode: 1, portsPerNode: 2 })).toEqual([]);
  });

  it('switchable: false rejects a switch', () => {
    expect(fabricShapeErrors(SWITCHED, { nodes: 4, devicesPerNode: 1, switchable: false })).toEqual(
      ['no switch exists for this link'],
    );
    expect(fabricShapeErrors(RING, { nodes: 4, devicesPerNode: 1, switchable: false })).toEqual([]);
  });
});
