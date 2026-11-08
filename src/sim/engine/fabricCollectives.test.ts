import { describe, expect, it } from 'vitest';
import type { FabricShape, LinkSpec } from '../hw/types';
import { collectiveTime } from './collectives';
import type {
  AllReduceAlgorithm,
  CollectiveKind,
  CollectivePath,
  CollectiveTime,
} from './collectives';
import {
  fabricAllToAllSec,
  fabricLinkParams,
  meshDirectAllReduceSec,
  pathLatencySec,
} from './fabricCollectives';

/**
 * Worked examples of docs/simulation-model.md §6 ("Fabric shapes"). Synthetic link: B_L = 1e10 B/s, α = 2 µs,
 * α_f = 1 µs; S = 8e6 B (S/B_L = 800 µs), N = 4 unless stated. Expected values are [α, β] in µs.
 */

const US = 1e-6;

/** A link with the given per-port bandwidth, latency, relay latency and (optional) all-ports cap; every number sourced. */
function link(bandwidth: number, latency: number, forwarding?: number, cap?: number): LinkSpec {
  return {
    kind: 'other',
    name: 'test',
    bandwidthPerDirectionBytesPerSec: bandwidth,
    latencySec: latency,
    p2p: 'direct',
    ...(forwarding !== undefined ? { forwardingLatencySec: forwarding } : {}),
    ...(cap !== undefined ? { allPortsCapPerDirectionBytesPerSec: cap } : {}),
    sources: [
      {
        url: 'synthetic://test',
        title: 't',
        fields: [
          'bandwidthPerDirectionBytesPerSec',
          'latencySec',
          'forwardingLatencySec',
          'allPortsCapPerDirectionBytesPerSec',
        ],
      },
    ],
    estimated: [],
  };
}

const B = 1e10;
const FREE = link(B, 2 * US, 1 * US); // C = ∞
const CAPPED = link(B, 2 * US, 1 * US, B); // C = B_L

const SWITCHED: FabricShape = { kind: 'switched' };
const O2: FabricShape = { kind: 'switched', oversubscription: 2 };
const RING: FabricShape = { kind: 'ring' };
const CHAIN: FabricShape = { kind: 'chain' };
const MESH: FabricShape = { kind: 'full_mesh' };
const MESH2: FabricShape = { kind: 'full_mesh', cablesPerPair: 2 };

/** A flat inter-node path; the non-blocking switch is the pre-fabric path (no `fabric` field), as placement emits it. */
function path(l: LinkSpec, fabric: FabricShape, sendHops?: number): CollectivePath {
  if (fabric.kind === 'switched' && fabric.oversubscription === undefined)
    return { kind: 'flat', level: 'inter', link: l };
  return {
    kind: 'flat',
    level: 'inter',
    link: l,
    fabric,
    ...(sendHops !== undefined ? { sendHops } : {}),
  };
}

function cost(
  kind: CollectiveKind,
  s: number,
  n: number,
  p: CollectivePath,
  algorithm: AllReduceAlgorithm = 'best',
): CollectiveTime {
  return collectiveTime(kind, s, n, p, algorithm);
}

/** Asserts [α, β] in µs to a relative 1e-12. */
function expectUs(t: CollectiveTime, [latUs, bwUs]: [number, number], label: string): void {
  expect(t.latencySec / US, `${label} α`).toBeCloseTo(latUs, 9);
  expect(t.bandwidthSec / US, `${label} β`).toBeCloseTo(bwUs, 9);
  expect(Math.abs(t.latencySec / US - latUs)).toBeLessThanOrEqual(1e-12 * Math.max(1, latUs));
  expect(Math.abs(t.bandwidthSec / US - bwUs)).toBeLessThanOrEqual(1e-12 * Math.max(1, bwUs));
}

type Cell = [number, number];
/** One column of the worked-example table: a fabric on a link. */
interface Column {
  name: string;
  link: LinkSpec;
  fabric: FabricShape;
  ringAr: Cell;
  treeAr: Cell;
  ag: Cell;
  rs: Cell;
  a2a: Cell;
  send1: Cell;
  send2?: Cell;
}

const S = 8e6;
const COLUMNS: Column[] = [
  {
    name: 'switched',
    link: FREE,
    fabric: SWITCHED,
    ringAr: [12, 1200],
    treeAr: [8, 1600],
    ag: [6, 600],
    rs: [6, 600],
    a2a: [2, 600],
    send1: [2, 800],
  },
  {
    name: 'switched o=2',
    link: FREE,
    fabric: O2,
    ringAr: [12, 2400],
    treeAr: [8, 3200],
    ag: [6, 1200],
    rs: [6, 1200],
    a2a: [2, 1200],
    send1: [2, 1600],
  },
  {
    name: 'ring C=∞',
    link: FREE,
    fabric: RING,
    ringAr: [12, 600],
    treeAr: [14, 1600],
    ag: [6, 300],
    rs: [6, 300],
    a2a: [5, 400],
    send1: [2, 800],
    send2: [5, 1600],
  },
  {
    name: 'ring C=B_L',
    link: CAPPED,
    fabric: RING,
    ringAr: [12, 1200],
    treeAr: [14, 1600],
    ag: [6, 600],
    rs: [6, 600],
    a2a: [5, 800],
    send1: [2, 800],
    send2: [5, 1600],
  },
  {
    name: 'chain C=∞',
    link: FREE,
    fabric: CHAIN,
    ringAr: [12, 1200],
    treeAr: [14, 1600],
    ag: [6, 600],
    rs: [6, 600],
    a2a: [8, 800],
    send1: [2, 800],
    send2: [5, 1600],
  },
  // Chain with C = B_L: the all-gather's inner nodes send (N+1)·m = 5/4·S (own chunk both ways + forwards), so
  // AG β = max(3/4·800, 5/4·800) = 1000 and AR = RS 600 + AG 1000. RS stays (N−1)·m per node.
  {
    name: 'chain C=B_L',
    link: CAPPED,
    fabric: CHAIN,
    ringAr: [12, 1600],
    treeAr: [14, 1600],
    ag: [6, 1000],
    rs: [6, 600],
    a2a: [8, 1400],
    send1: [2, 800],
    send2: [5, 1600],
  },
  {
    name: 'mesh k=1 C=∞',
    link: FREE,
    fabric: MESH,
    ringAr: [12, 600],
    treeAr: [8, 1600],
    ag: [2, 200],
    rs: [2, 200],
    a2a: [2, 200],
    send1: [2, 800],
  },
  {
    name: 'mesh k=1 C=B_L',
    link: CAPPED,
    fabric: MESH,
    ringAr: [12, 1200],
    treeAr: [8, 1600],
    ag: [2, 600],
    rs: [2, 600],
    a2a: [2, 600],
    send1: [2, 800],
  },
  {
    name: 'mesh k=2 C=∞',
    link: FREE,
    fabric: MESH2,
    ringAr: [12, 300],
    treeAr: [8, 800],
    ag: [2, 100],
    rs: [2, 100],
    a2a: [2, 100],
    send1: [2, 400],
  },
];

describe('fabric collectives: the worked examples (N = 4, S = 8e6 B)', () => {
  for (const c of COLUMNS) {
    it(c.name, () => {
      const p = path(c.link, c.fabric);
      expectUs(cost('all_reduce', S, 4, p, 'ring'), c.ringAr, 'ring AR');
      expectUs(cost('all_reduce', S, 4, p, 'tree'), c.treeAr, 'tree AR');
      expectUs(cost('all_gather', S, 4, p), c.ag, 'AG');
      expectUs(cost('reduce_scatter', S, 4, p), c.rs, 'RS');
      expectUs(cost('all_to_all', S, 4, p), c.a2a, 'a2a');
      expectUs(cost('send_recv', S, 2, p), c.send1, 'send h=1');
      if (c.send2 !== undefined)
        expectUs(cost('send_recv', S, 2, path(c.link, c.fabric, 2)), c.send2, 'send h=2');
    });
  }

  it('direct all-reduce on a full mesh: 2α + 2(N−1)/N · S/B_mesh', () => {
    const direct = (l: LinkSpec, f: FabricShape) =>
      meshDirectAllReduceSec(S, 4, fabricLinkParams(l, f, 4));
    expectUs(direct(FREE, MESH), [4, 400], 'mesh k=1 C=∞');
    expectUs(direct(CAPPED, MESH), [4, 1200], 'mesh k=1 C=B_L');
    expectUs(direct(FREE, MESH2), [4, 200], 'mesh k=2 C=∞');
    expect(direct(FREE, MESH).algorithm).toBe('direct');
  });

  it('lat(h) = h·α + (h−1)·α_f', () => {
    const p = fabricLinkParams(FREE, RING, 4);
    expect([0, 1, 2, 3, 4].map((h) => pathLatencySec(h, p) / US)).toEqual([0, 2, 5, 8, 11]);
  });
});

describe("fabric collectives: 'best' at a decode-sized message (S = 1e4 B, N = 4)", () => {
  const best = (l: LinkSpec, f: FabricShape) => cost('all_reduce', 1e4, 4, path(l, f));
  it('switched → tree, 8 + 2 µs', () => {
    expect(best(FREE, SWITCHED).algorithm).toBe('tree');
    expectUs(best(FREE, SWITCHED), [8, 2], 'switched');
  });
  it('ring C = B_L → ring, 12 + 1.5 µs (the tree would be 14 + 2)', () => {
    expect(best(CAPPED, RING).algorithm).toBe('ring');
    expectUs(best(CAPPED, RING), [12, 1.5], 'ring');
  });
  it('ring C = ∞ → ring, 12 + 0.75 µs', () => expectUs(best(FREE, RING), [12, 0.75], 'ring'));
  it('chain → ring: 12 + 1.5 µs (C = ∞), 12 + 2 µs (C = B_L)', () => {
    expect(best(FREE, CHAIN).algorithm).toBe('ring');
    expectUs(best(FREE, CHAIN), [12, 1.5], 'chain');
    expectUs(best(CAPPED, CHAIN), [12, 2], 'chain C=B_L');
  });
  it('full mesh k = 1, C = ∞ → direct, 4 + 0.5 µs', () => {
    expect(best(FREE, MESH).algorithm).toBe('direct');
    expectUs(best(FREE, MESH), [4, 0.5], 'mesh');
  });
});

describe('fabric collectives: other N (C = ∞ unless stated)', () => {
  it('ring N = 8 all-to-all: lat(4) = 11 µs, β 800 µs (1600 with C = B_L); switched 2 + 700', () => {
    expectUs(cost('all_to_all', S, 8, path(FREE, RING)), [11, 800], 'ring');
    expectUs(cost('all_to_all', S, 8, path(CAPPED, RING)), [11, 1600], 'ring C=B_L');
    expectUs(cost('all_to_all', S, 8, path(FREE, SWITCHED)), [2, 700], 'switched');
  });
  it('tree all-reduce N = 8 on a ring and a chain: hops 1, 2, 4 → 2·(2 + 5 + 11) = 36 µs; switched 12 µs', () => {
    for (const f of [RING, CHAIN])
      expectUs(cost('all_reduce', S, 8, path(FREE, f), 'tree'), [36, 1600], f.kind);
    expectUs(cost('all_reduce', S, 8, path(FREE, SWITCHED), 'tree'), [12, 1600], 'switched');
  });
  it('chain N = 3 all-to-all: α 5, β 533.33 (1066.67 with C = B_L: E_3 = 4)', () => {
    expectUs(cost('all_to_all', S, 3, path(FREE, CHAIN)), [5, 1600 / 3], 'chain');
    expectUs(cost('all_to_all', S, 3, path(CAPPED, CHAIN)), [5, 3200 / 3], 'chain C=B_L');
  });
  it('ring N = 3 all-to-all equals the full mesh of 3 (the same cabling): α 2, β 266.67', () => {
    const ring = cost('all_to_all', S, 3, path(FREE, RING));
    expectUs(ring, [2, 800 / 3], 'ring');
    expect(ring).toEqual(cost('all_to_all', S, 3, path(FREE, MESH)));
  });
});

describe('fabric collectives: exact identities with the switched formulas', () => {
  const KINDS: [CollectiveKind, AllReduceAlgorithm][] = [
    ['all_reduce', 'best'],
    ['all_reduce', 'ring'],
    ['all_reduce', 'tree'],
    ['all_gather', 'best'],
    ['reduce_scatter', 'best'],
    ['all_to_all', 'best'],
    ['send_recv', 'best'],
  ];
  const SIZES = [1e3, 1e4, 1e6, 8e6, 1e8];

  it('two nodes on a full mesh (k = 1) or a chain = one cable = switched, for every kind (C = ∞ and C = B_L)', () => {
    for (const l of [FREE, CAPPED, link(25e9, 1.5 * US, 5 * US, 25e9)]) {
      for (const f of [MESH, CHAIN]) {
        for (const [kind, a] of KINDS) {
          for (const s of SIZES)
            expect(cost(kind, s, 2, path(l, f), a), `${f.kind} ${kind} ${a} ${s}`).toEqual(
              cost(kind, s, 2, path(l, SWITCHED), a),
            );
        }
      }
    }
  });

  it('ring with C = B_L = switched for the ring all-reduce and AG / RS (bit for bit)', () => {
    for (const n of [3, 4, 5, 8]) {
      for (const s of SIZES) {
        for (const [kind, a] of [
          ['all_reduce', 'ring'],
          ['all_gather', 'best'],
          ['reduce_scatter', 'best'],
        ] as const) {
          const t = cost(kind, s, n, path(CAPPED, RING), a);
          expect(t, `${kind} n ${n} S ${s}`).toEqual(cost(kind, s, n, path(CAPPED, SWITCHED), a));
        }
      }
    }
  });

  it('DGX Spark-like cap: two cables per pair do not beat one (B_one = min(2·25, 25) GB/s)', () => {
    const spark = link(25e9, 1.5 * US, 5 * US, 25e9);
    for (const [kind, a] of KINDS) {
      for (const s of SIZES)
        expect(cost(kind, s, 2, path(spark, MESH2), a)).toEqual(
          cost(kind, s, 2, path(spark, MESH), a),
        );
    }
  });

  it('switched with oversubscription 1 is the pre-fabric path', () => {
    const o1: CollectivePath = {
      kind: 'flat',
      level: 'inter',
      link: FREE,
      fabric: { kind: 'switched', oversubscription: 1 },
    };
    for (const [kind, a] of KINDS)
      for (const s of SIZES)
        expect(cost(kind, s, 4, o1, a)).toEqual(cost(kind, s, 4, path(FREE, SWITCHED), a));
  });

  it('hierarchical with an oversubscribed switch divides only the inter-node bandwidth', () => {
    const intra = link(1e11, 1 * US);
    const base: CollectivePath = {
      kind: 'hierarchical',
      devicesPerNode: 8,
      nodes: 2,
      intra,
      inter: FREE,
    };
    const over: CollectivePath = {
      ...base,
      interFabric: { kind: 'switched', oversubscription: 2 },
    };
    const halved: CollectivePath = { ...base, inter: link(B / 2, 2 * US, 1 * US) };
    for (const kind of ['all_reduce', 'all_gather', 'reduce_scatter', 'all_to_all'] as const) {
      expect(cost(kind, S, 16, over)).toEqual(cost(kind, S, 16, halved));
    }
  });
});

describe('DGX Spark ring (B_L = C = 25 GB/s, α 1.5 µs, α_f 5 µs estimated), N = 4', () => {
  const spark = link(25e9, 1.5 * US, 5 * US, 25e9);
  it('ring AR is bit-identical to switched: 9 + 480 µs', () => {
    const ring = cost('all_reduce', S, 4, path(spark, RING), 'ring');
    expect(ring).toEqual(cost('all_reduce', S, 4, path(spark, SWITCHED), 'ring'));
    expectUs(ring, [9, 480], 'ring');
  });
  it('tree AR 19 + 640, all-to-all 8 + 320, send h = 2: 8 + 640', () => {
    expectUs(cost('all_reduce', S, 4, path(spark, RING), 'tree'), [19, 640], 'tree');
    expectUs(cost('all_to_all', S, 4, path(spark, RING)), [8, 320], 'a2a');
    expectUs(cost('send_recv', S, 2, path(spark, RING, 2)), [8, 640], 'send');
    expectUs(cost('all_to_all', S, 4, path(spark, SWITCHED)), [1.5, 240], 'switched a2a');
  });
  it("decode (S = 1e4): switched 'best' = tree 6.8 µs; the ring fabric = ring 9.6 µs", () => {
    const sw = cost('all_reduce', 1e4, 4, path(spark, SWITCHED));
    const ring = cost('all_reduce', 1e4, 4, path(spark, RING));
    expect(sw.algorithm).toBe('tree');
    expect(sw.totalSec / US).toBeCloseTo(6.8, 9);
    expect(ring.algorithm).toBe('ring');
    expect(ring.totalSec / US).toBeCloseTo(9.6, 9);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// Brute-force routing: the closed-form link and node loads, checked against explicit flows.

/** Bytes per directed link ("u>v") and per node egress (own + relayed). */
interface Loads {
  link: Map<string, number>;
  egress: number[];
}

function emptyLoads(n: number): Loads {
  return { link: new Map(), egress: Array<number>(n).fill(0) };
}

/** Adds `bytes` along the node sequence `route` (each hop u→v loads link u>v and u's egress). */
function addRoute(loads: Loads, route: number[], bytes: number): void {
  for (let i = 0; i + 1 < route.length; i++) {
    const key = `${route[i]}>${route[i + 1]}`;
    loads.link.set(key, (loads.link.get(key) ?? 0) + bytes);
    loads.egress[route[i]] += bytes;
  }
}

/** The node sequence from `src` stepping by `dir` (±1) `hops` times, mod n. */
function walk(src: number, dir: number, hops: number, n: number): number[] {
  return Array.from({ length: hops + 1 }, (_, i) => (((src + dir * i) % n) + n) % n);
}

/** All-to-all flows of m bytes on shortest paths (ring: a tie at distance n/2 splits half each way). */
function allToAllLoads(kind: 'ring' | 'chain', n: number, m: number): Loads {
  const loads = emptyLoads(n);
  for (let src = 0; src < n; src++) {
    for (let dst = 0; dst < n; dst++) {
      if (src === dst) continue;
      if (kind === 'chain') {
        addRoute(loads, walk(src, dst > src ? 1 : -1, Math.abs(dst - src), n), m);
        continue;
      }
      const cw = (dst - src + n) % n;
      const ccw = n - cw;
      if (cw < ccw) addRoute(loads, walk(src, 1, cw, n), m);
      else if (ccw < cw) addRoute(loads, walk(src, -1, ccw, n), m);
      else {
        addRoute(loads, walk(src, 1, cw, n), m / 2);
        addRoute(loads, walk(src, -1, ccw, n), m / 2);
      }
    }
  }
  return loads;
}

const max = (xs: Iterable<number>) => Math.max(...xs);

describe('brute-force routing (N = 2..8)', () => {
  const m = 1000;
  it('ring all-to-all: busiest link ⌊N²/4⌋/2 · m, busiest egress twice that', () => {
    for (let n = 3; n <= 8; n++) {
      const loads = allToAllLoads('ring', n, m);
      const busiestLink = (Math.floor((n * n) / 4) / 2) * m;
      expect(max(loads.link.values()), `n ${n}`).toBeCloseTo(busiestLink, 9);
      expect(max(loads.egress), `n ${n}`).toBeCloseTo(2 * busiestLink, 9);
      // Every link direction is equally loaded (symmetry), so β = max(L/B_L, 2L/C) = L / min(B_L, C/2).
      expect(Math.min(...loads.link.values())).toBeCloseTo(busiestLink, 9);
    }
  });

  it('chain all-to-all: busiest link ⌊N²/4⌋ · m, busiest egress E_N · m, E_N = (N−1) + 2⌊(N−1)²/4⌋', () => {
    for (let n = 2; n <= 8; n++) {
      const loads = allToAllLoads('chain', n, m);
      expect(max(loads.link.values()), `n ${n}`).toBe(Math.floor((n * n) / 4) * m);
      expect(max(loads.egress), `n ${n}`).toBe(
        (n - 1 + 2 * Math.floor(((n - 1) * (n - 1)) / 4)) * m,
      );
    }
  });

  it('the all-to-all formulas equal max(L/B_L, E/C) of the brute-force loads (S = N·m)', () => {
    for (const [bl, cap] of [
      [1e10, Infinity],
      [1e10, 1e10],
      [1e10, 1.5e10],
      [1e10, 3e10],
    ]) {
      const l = link(bl, 2 * US, 1 * US, Number.isFinite(cap) ? cap : undefined);
      for (const kind of ['ring', 'chain'] as const) {
        for (let n = kind === 'ring' ? 3 : 2; n <= 8; n++) {
          const loads = allToAllLoads(kind, n, m);
          const expected = Math.max(max(loads.link.values()) / bl, max(loads.egress) / cap);
          const t = fabricAllToAllSec(n * m, n, { kind }, fabricLinkParams(l, { kind }, n));
          expect(t.bandwidthSec / expected, `${kind} n ${n} C ${cap}`).toBeCloseTo(1, 12);
        }
      }
    }
  });

  it('chain linear reduce-scatter: busiest link (N−1)·m, every node sends (N−1)·m', () => {
    for (let n = 2; n <= 8; n++) {
      const loads = emptyLoads(n);
      // The partial of chunk c flows toward node c from both sides, one neighbour step at a time.
      for (let c = 0; c < n; c++) {
        for (let j = 0; j < c; j++) addRoute(loads, [j, j + 1], m);
        for (let j = n - 1; j > c; j--) addRoute(loads, [j, j - 1], m);
      }
      expect(max(loads.link.values())).toBe((n - 1) * m);
      expect(loads.egress.every((e) => e === (n - 1) * m)).toBe(true);
    }
  });

  it('chain linear all-gather: busiest link (N−1)·m, inner nodes send (N+1)·m, the ends m', () => {
    for (let n = 3; n <= 8; n++) {
      const loads = emptyLoads(n);
      // Chunk c travels from node c to both ends.
      for (let c = 0; c < n; c++) {
        addRoute(loads, walk(c, 1, n - 1 - c, n), m);
        addRoute(loads, walk(c, -1, c, n), m);
      }
      expect(max(loads.link.values())).toBe((n - 1) * m);
      expect(loads.egress.slice(1, -1).every((e) => e === (n + 1) * m)).toBe(true);
      expect([loads.egress[0], loads.egress[n - 1]]).toEqual([m, m]);
      // Lower bound for ANY chain all-gather: N(N−1) receptions, the ends send m each, the inner nodes the rest.
      expect(loads.egress.reduce((a, b) => a + b, 0)).toBe(n * (n - 1) * m);
    }
  });
});
