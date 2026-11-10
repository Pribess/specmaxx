import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_IB, SYNTH_NVLINK } from '../__fixtures__/synthetic';
import {
  checkPlacement,
  collectivePath,
  lookupTableCollectivePaths,
  hasInterNodeTraffic,
  expertTpCollectivePath,
  expertTpGroupSpan,
  groupSpan,
  ppBoundaryIsInterNode,
  ppSendHops,
  ppSendPath,
} from './placement';
import type { TopologySpec } from '../hw/types';

// SYNTH_CLUSTER: 8 devices per node × 2 nodes, NVLink intra, InfiniBand inter.
describe('placement on SYNTH_CLUSTER (T8)', () => {
  it('tp 8 is intra-node', () => {
    const span = groupSpan(8, SYNTH_CLUSTER);
    expect(span).toEqual({ n: 8, devicesPerNodeInGroup: 8, nodesSpanned: 1 });
    expect(collectivePath(span, SYNTH_CLUSTER)).toEqual({
      kind: 'flat',
      level: 'intra',
      link: SYNTH_NVLINK,
    });
  });

  it('tp 16 is hierarchical: n1 8, n2 2', () => {
    const span = groupSpan(16, SYNTH_CLUSTER);
    expect(span).toEqual({ n: 16, devicesPerNodeInGroup: 8, nodesSpanned: 2 }); // 16 % 8 == 0
    expect(collectivePath(span, SYNTH_CLUSTER).kind).toBe('hierarchical');
  });

  it('DP groups are strided by TP, not contiguous groups of dp devices', () => {
    expect(lookupTableCollectivePaths({ tp: 8, dp: 2, pp: 1, ep: 1 }, SYNTH_CLUSTER).dp).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
    });
    expect(
      lookupTableCollectivePaths({ tp: 4, dp: 4, pp: 1, ep: 1 }, SYNTH_CLUSTER).dp,
    ).toMatchObject({ kind: 'hierarchical', devicesPerNode: 2, nodes: 2 });
    const par = { tp: 1, dp: 2, pp: 1, ep: 1 };
    const topo = { ...SYNTH_CLUSTER, devicesPerNode: 1 };
    expect(hasInterNodeTraffic(par, topo)).toBe(false);
    expect(hasInterNodeTraffic(par, topo, true)).toBe(true);
    expect(() =>
      lookupTableCollectivePaths(
        { tp: 16, dp: 2, pp: 1, ep: 1 },
        {
          ...SYNTH_CLUSTER,
          nodes: 4,
          interNodeFabric: { kind: 'ring' },
        },
      ),
    ).toThrow(/DP group.*2 apart/);
  });

  it('tp 3 is an error (3 ∤ 8 and 8 ∤ 3)', () => {
    expect(() => groupSpan(3, SYNTH_CLUSTER)).toThrow();
    expect(
      checkPlacement({ tp: 3, pp: 1, dp: 1, ep: 1 }, SYNTH_CLUSTER).errors.length,
    ).toBeGreaterThan(0);
  });

  it('tp 8 pp 2: the boundary is inter-node (ranks 0–7 → 8–15)', () => {
    const par = { tp: 8, pp: 2, dp: 1, ep: 1 };
    expect(ppBoundaryIsInterNode(par, SYNTH_CLUSTER, 0)).toBe(true);
    expect(ppSendPath(par, SYNTH_CLUSTER, 0)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
    });
    expect(checkPlacement(par, SYNTH_CLUSTER)).toEqual({ errors: [], warnings: [] });
  });

  it('tp 4 pp 2: the boundary is intra-node (ranks 0–3 → 4–7); 8 devices unused', () => {
    const par = { tp: 4, pp: 2, dp: 1, ep: 1 };
    expect(ppBoundaryIsInterNode(par, SYNTH_CLUSTER, 0)).toBe(false);
    const check = checkPlacement(par, SYNTH_CLUSTER);
    expect(check.errors).toEqual([]);
    expect(check.warnings).toHaveLength(1); // 16 − 8 unused
  });

  it('world > 16 is an error', () => {
    // tp·pp·dp = 8·2·2 = 32 > 8·2
    expect(
      checkPlacement({ tp: 8, pp: 2, dp: 2, ep: 1 }, SYNTH_CLUSTER).errors.length,
    ).toBeGreaterThan(0);
  });

  it('a missing required link is an error', () => {
    const noInter: TopologySpec = { ...SYNTH_CLUSTER, interNode: undefined };
    const errors = checkPlacement({ tp: 16, pp: 1, dp: 1, ep: 1 }, noInter).errors;
    expect(errors.some((e) => e.includes('no interNode link'))).toBe(true);
  });

  it('one device per node: a group across nodes is flat inter-node', () => {
    const macs: TopologySpec = {
      id: 'm',
      name: 'm',
      devicesPerNode: 1,
      nodes: 4,
      interNode: SYNTH_IB,
      sources: [],
    };
    expect(collectivePath(groupSpan(4, macs), macs)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
    });
  });
});

// Four single-device nodes cabled as a ring (node i to i ± 1), then the other shapes.
describe('placement with an inter-node fabric', () => {
  const ring4: TopologySpec = {
    id: 'r',
    name: 'r',
    devicesPerNode: 1,
    nodes: 4,
    interNode: SYNTH_IB,
    interNodeFabric: { kind: 'ring' },
    sources: [],
  };

  it('a TP group over the whole ring sees the ring', () => {
    expect(collectivePath(groupSpan(4, ring4), ring4)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
      fabric: { kind: 'ring' },
    });
  });

  it('a TP subgroup on a ring is a chain (a segment)', () => {
    expect(collectivePath(groupSpan(2, ring4), ring4)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
      fabric: { kind: 'chain' },
    });
    expect(
      collectivePath(groupSpan(3, { ...ring4, nodes: 8 }), { ...ring4, nodes: 8 }),
    ).toMatchObject({ fabric: { kind: 'chain' } });
  });

  it('pp only on a ring: stages on neighbours, no sendHops', () => {
    const par = { tp: 1, pp: 4, dp: 1, ep: 1 };
    for (const stage of [0, 1, 2]) {
      expect(ppSendPath(par, ring4, stage)).toEqual({
        kind: 'flat',
        level: 'inter',
        link: SYNTH_IB,
        fabric: { kind: 'ring' },
      });
    }
  });

  it('tp 2 × pp 2 on a ring of 4: ranks 0, 1 send to 2, 3 over 2 hops', () => {
    const par = { tp: 2, pp: 2, dp: 1, ep: 1 };
    expect(ppSendHops(par, ring4, 0)).toBe(2);
    expect(ppSendPath(par, ring4, 0)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
      fabric: { kind: 'ring' },
      sendHops: 2,
    });
    expect(checkPlacement(par, ring4)).toEqual({ errors: [], warnings: [] });
  });

  it('a full mesh is seen whole by every group, and a send is 1 hop', () => {
    const mesh: TopologySpec = {
      ...ring4,
      interNodeFabric: { kind: 'full_mesh', cablesPerPair: 2 },
    };
    expect(collectivePath(groupSpan(2, mesh), mesh)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
      fabric: { kind: 'full_mesh', cablesPerPair: 2 },
    });
    expect(ppSendPath({ tp: 2, pp: 2, dp: 1, ep: 1 }, mesh, 0)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
      fabric: { kind: 'full_mesh', cablesPerPair: 2 },
    });
  });

  it('a non-blocking switch, given explicitly, emits the pre-fabric paths', () => {
    for (const f of [{ kind: 'switched' }, { kind: 'switched', oversubscription: 1 }] as const) {
      const t: TopologySpec = { ...ring4, interNodeFabric: f };
      expect(collectivePath(groupSpan(4, t), t)).toEqual({
        kind: 'flat',
        level: 'inter',
        link: SYNTH_IB,
      });
      expect(ppSendPath({ tp: 2, pp: 2, dp: 1, ep: 1 }, t, 0)).toEqual({
        kind: 'flat',
        level: 'inter',
        link: SYNTH_IB,
      });
    }
  });

  it('an oversubscribed switch: flat paths carry it, hierarchical paths carry interFabric', () => {
    const over: TopologySpec = {
      ...SYNTH_CLUSTER,
      interNodeFabric: { kind: 'switched', oversubscription: 2 },
    };
    expect(collectivePath(groupSpan(16, over), over)).toMatchObject({
      kind: 'hierarchical',
      interFabric: { kind: 'switched', oversubscription: 2 },
    });
    expect(collectivePath(groupSpan(8, over), over)).toEqual({
      kind: 'flat',
      level: 'intra',
      link: SYNTH_NVLINK,
    });
    const send = ppSendPath({ tp: 8, pp: 2, dp: 1, ep: 1 }, over, 0);
    expect(send).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
      fabric: { kind: 'switched', oversubscription: 2 },
    });
  });
});

// The expert-TP group of mode A: the etp = tp/ep ranks j + i·ep of a TP group (EP groups are ep consecutive ranks).
describe('expert-TP group span', () => {
  it('SYNTH_CLUSTER (8 × 2): strided members per node', () => {
    expect(expertTpGroupSpan({ tp: 16, ep: 2 }, SYNTH_CLUSTER)).toEqual({
      n: 8,
      devicesPerNodeInGroup: 4,
      nodesSpanned: 2,
      nodeStride: 1,
    });
    expect(expertTpGroupSpan({ tp: 16, ep: 4 }, SYNTH_CLUSTER)).toEqual({
      n: 4,
      devicesPerNodeInGroup: 2,
      nodesSpanned: 2,
      nodeStride: 1,
    });
    expect(expertTpGroupSpan({ tp: 16, ep: 8 }, SYNTH_CLUSTER)).toEqual({
      n: 2,
      devicesPerNodeInGroup: 1,
      nodesSpanned: 2,
      nodeStride: 1,
    });
    expect(expertTpGroupSpan({ tp: 8, ep: 2 }, SYNTH_CLUSTER)).toEqual({
      n: 4,
      devicesPerNodeInGroup: 4,
      nodesSpanned: 1,
      nodeStride: 1,
    });
    // etp 1 (ep = tp, or mode B)
    expect(expertTpGroupSpan({ tp: 16, ep: 16 }, SYNTH_CLUSTER).n).toBe(1);
    expect(expertTpGroupSpan({ tp: 2, ep: 8 }, SYNTH_CLUSTER).n).toBe(1);
    const p = expertTpCollectivePath({ tp: 16, ep: 2 }, SYNTH_CLUSTER);
    expect(p).toMatchObject({ kind: 'hierarchical', devicesPerNode: 4, nodes: 2 });
    expect(expertTpCollectivePath({ tp: 16, ep: 8 }, SYNTH_CLUSTER)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
    });
    expect(expertTpCollectivePath({ tp: 8, ep: 2 }, SYNTH_CLUSTER)).toEqual({
      kind: 'flat',
      level: 'intra',
      link: SYNTH_NVLINK,
    });
  });

  it('ep > devices per node: one member per node, nodes ep/dpn apart', () => {
    const four: TopologySpec = { ...SYNTH_CLUSTER, nodes: 4 };
    // tp 32, ep 16 on 4 × 8: ranks j and j + 16 sit on nodes 0 and 2
    expect(expertTpGroupSpan({ tp: 32, ep: 16 }, four)).toEqual({
      n: 2,
      devicesPerNodeInGroup: 1,
      nodesSpanned: 2,
      nodeStride: 2,
    });
    expect(expertTpCollectivePath({ tp: 32, ep: 16 }, four)).toEqual({
      kind: 'flat',
      level: 'inter',
      link: SYNTH_IB,
    });
  });

  it('non-adjacent member nodes on a ring are rejected; adjacent ones (ep 1 = the TP group) are not', () => {
    const ring4: TopologySpec = {
      id: 'r',
      name: 'r',
      devicesPerNode: 1,
      nodes: 4,
      interNode: SYNTH_IB,
      interNodeFabric: { kind: 'ring' },
      sources: [],
    };
    expect(() => expertTpCollectivePath({ tp: 4, ep: 2 }, ring4)).toThrow(
      /2 nodes 2 apart on a ring/,
    );
    expect(expertTpCollectivePath({ tp: 4, ep: 1 }, ring4)).toEqual(
      collectivePath(groupSpan(4, ring4), ring4),
    );
  });
});
