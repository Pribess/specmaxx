import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_IB } from '../__fixtures__/synthetic';
import type { LinkSpec, TopologySpec } from './types';
import { validateLink, validateTopology } from './validate';

/** Four single-device nodes on {@link SYNTH_IB}. */
const FOUR_NODES: TopologySpec = {
  id: 'four',
  name: 'four',
  devicesPerNode: 1,
  nodes: 4,
  interNode: SYNTH_IB,
  sources: [],
};

describe('validateTopology: interNodeFabric', () => {
  it('accepts every shape on single-device nodes, and an oversubscribed switch on multi-device nodes', () => {
    for (const f of [
      { kind: 'switched' },
      { kind: 'full_mesh', cablesPerPair: 2 },
      { kind: 'ring' },
      { kind: 'chain' },
    ] as const) {
      expect(validateTopology({ ...FOUR_NODES, interNodeFabric: f })).toEqual([]);
    }
    expect(
      validateTopology({
        ...SYNTH_CLUSTER,
        interNodeFabric: { kind: 'switched', oversubscription: 2 },
      }),
    ).toEqual([]);
  });

  it('a switchless shape with devicesPerNode > 1 is an error', () => {
    const errors = validateTopology({ ...SYNTH_CLUSTER, interNodeFabric: { kind: 'ring' } });
    expect(errors).toHaveLength(2); // one device per node; a ring of 2 nodes
    expect(errors.some((e) => e.includes('one device per node'))).toBe(true);
  });

  it('oversubscription < 1 is an error', () => {
    expect(
      validateTopology({
        ...FOUR_NODES,
        interNodeFabric: { kind: 'switched', oversubscription: 0.5 },
      }),
    ).toHaveLength(1);
  });

  it('a fabric without an interNode link, or on one node, is an error', () => {
    const { interNode: _dropped, ...noLink } = FOUR_NODES;
    void _dropped;
    expect(validateTopology({ ...noLink, interNodeFabric: { kind: 'chain' } })).toEqual([
      'topology four: interNodeFabric needs an interNode link',
    ]);
    expect(
      validateTopology({ ...FOUR_NODES, nodes: 1, interNodeFabric: { kind: 'switched' } }),
    ).toEqual(['topology four: interNodeFabric needs nodes > 1 (got 1)']);
  });

  it('a ring of 2 nodes is an error', () => {
    expect(
      validateTopology({ ...FOUR_NODES, nodes: 2, interNodeFabric: { kind: 'ring' } }),
    ).toHaveLength(1);
  });
});

describe('validateLink: the new fabric fields', () => {
  const sourced = (l: Partial<LinkSpec>): LinkSpec => ({
    ...SYNTH_IB,
    ...l,
    sources: [
      {
        ...SYNTH_IB.sources[0],
        fields: [
          'bandwidthPerDirectionBytesPerSec',
          'latencySec',
          'allPortsCapPerDirectionBytesPerSec',
          'forwardingLatencySec',
        ],
      },
    ],
  });

  it('accepts a cap ≥ the per-port bandwidth and a relay latency', () => {
    expect(
      validateLink(
        sourced({ allPortsCapPerDirectionBytesPerSec: 50e9, forwardingLatencySec: 1e-6 }),
      ),
    ).toEqual([]);
    expect(validateLink(sourced({ allPortsCapPerDirectionBytesPerSec: 100e9 }))).toEqual([]);
  });

  it('a cap below the per-port bandwidth is an error', () => {
    const errors = validateLink(sourced({ allPortsCapPerDirectionBytesPerSec: 25e9 }));
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain('allPortsCapPerDirectionBytesPerSec');
  });

  it('the new fields must be sourced or estimated, finite and ≥ 0', () => {
    const unsourced: LinkSpec = {
      ...SYNTH_IB,
      allPortsCapPerDirectionBytesPerSec: 50e9,
      forwardingLatencySec: 1e-6,
    };
    expect(validateLink(unsourced).filter((e) => e.includes('not covered'))).toHaveLength(2);
    expect(
      validateLink(sourced({ forwardingLatencySec: -1 })).some((e) =>
        e.includes('forwardingLatencySec must be finite'),
      ),
    ).toBe(true);
  });
});
