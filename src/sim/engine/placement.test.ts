import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_IB, SYNTH_NVLINK } from '../__fixtures__/synthetic';
import { checkPlacement, collectivePath, groupSpan, ppBoundaryIsInterNode, ppSendPath } from './placement';
import type { TopologySpec } from '../hw/types';

// SYNTH_CLUSTER: 8 devices per node × 2 nodes, NVLink intra, InfiniBand inter.
describe('placement on SYNTH_CLUSTER (T8)', () => {
  it('tp 8 is intra-node', () => {
    const span = groupSpan(8, SYNTH_CLUSTER);
    expect(span).toEqual({ n: 8, devicesPerNodeInGroup: 8, nodesSpanned: 1 });
    expect(collectivePath(span, SYNTH_CLUSTER)).toEqual({ kind: 'flat', level: 'intra', link: SYNTH_NVLINK });
  });

  it('tp 16 is hierarchical: n1 8, n2 2', () => {
    const span = groupSpan(16, SYNTH_CLUSTER);
    expect(span).toEqual({ n: 16, devicesPerNodeInGroup: 8, nodesSpanned: 2 }); // 16 % 8 == 0
    expect(collectivePath(span, SYNTH_CLUSTER).kind).toBe('hierarchical');
  });

  it('tp 3 is an error (3 ∤ 8 and 8 ∤ 3)', () => {
    expect(() => groupSpan(3, SYNTH_CLUSTER)).toThrow();
    expect(checkPlacement({ tp: 3, pp: 1, dp: 1, ep: 1 }, SYNTH_CLUSTER).errors.length).toBeGreaterThan(0);
  });

  it('tp 8 pp 2: the boundary is inter-node (ranks 0–7 → 8–15)', () => {
    const par = { tp: 8, pp: 2, dp: 1, ep: 1 };
    expect(ppBoundaryIsInterNode(par, SYNTH_CLUSTER, 0)).toBe(true);
    expect(ppSendPath(par, SYNTH_CLUSTER, 0)).toEqual({ kind: 'flat', level: 'inter', link: SYNTH_IB });
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
    expect(checkPlacement({ tp: 8, pp: 2, dp: 2, ep: 1 }, SYNTH_CLUSTER).errors.length).toBeGreaterThan(0);
  });

  it('a missing required link is an error', () => {
    const noInter: TopologySpec = { ...SYNTH_CLUSTER, interNode: undefined };
    const errors = checkPlacement({ tp: 16, pp: 1, dp: 1, ep: 1 }, noInter).errors;
    expect(errors.some((e) => e.includes('no interNode link'))).toBe(true);
  });

  it('one device per node: a group across nodes is flat inter-node', () => {
    const macs: TopologySpec = { id: 'm', name: 'm', devicesPerNode: 1, nodes: 4, interNode: SYNTH_IB, sources: [] };
    expect(collectivePath(groupSpan(4, macs), macs)).toEqual({ kind: 'flat', level: 'inter', link: SYNTH_IB });
  });
});
