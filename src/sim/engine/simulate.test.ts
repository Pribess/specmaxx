import { describe, expect, it } from 'vitest';
import type { ParallelConfig, QuantConfig } from '../ir/model';
import type { AttentionOp } from '../ir/ops';
import { REF_DENSE_8B } from '../models/reference';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DGX_SPARK } from '../hw/presets/nvidia-datacenter';
import type { LinkSpec, TopologySpec } from '../hw/types';
import { simulate } from './simulate';
import type { PhaseResult, SimInput } from './types';

/**
 * The context semantics of `Workload` (docs §7 "Pipeline and DP"):
 * - prefill @ P: a P-token prompt from an empty KV cache → attention `qLen = kvLen = P`;
 * - decode @ N: N tokens already cached, one new token → attention `qLen = 1`, `kvLen = N + 1`.
 */

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const P1: ParallelConfig = { tp: 1, pp: 1, dp: 1, ep: 1 };

function run(promptLen: number, decodeContextLen: number): ReturnType<typeof simulate> {
  const input: SimInput = {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: P1,
    workload: { batch: 1, promptLen, decodeContextLen },
    hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) },
  };
  return simulate(input);
}

const attentionOps = (p: PhaseResult): AttentionOp[] =>
  p.stages
    .flatMap((s) => s.layers.flatMap((l) => l.ops.map((o) => o.op)))
    .filter((op): op is AttentionOp => op.kind === 'attention');

/** Σ KV-cache bytes read by one phase step, per device. Unit: bytes. */
const kvReadBytes = (p: PhaseResult): number =>
  p.stages
    .flatMap((s) => s.layers.flatMap((l) => l.ops))
    .reduce((sum, o) => sum + o.bytes.kvReadBytes, 0);

describe('workload context semantics', () => {
  it('prefill @ P attends from an empty cache: qLen = kvLen = P', () => {
    const ops = attentionOps(run(8192, 1024).prefill);
    expect(ops).toHaveLength(36);
    for (const op of ops) {
      expect(op.qLen).toBe(8192);
      expect(op.kvLen).toBe(8192);
    }
  });

  it('decode @ N has N tokens cached plus the new one: qLen = 1, kvLen = N + 1', () => {
    const ops = attentionOps(run(128, 8192).decode);
    expect(ops).toHaveLength(36);
    for (const op of ops) {
      expect(op.qLen).toBe(1);
      expect(op.kvLen).toBe(8193);
    }
  });

  it('decode KV reads grow linearly in N: 36 layers · (N + 1) tokens · 4096 B', () => {
    // REF_DENSE_8B bf16: K + V per token per layer = 2 · 8 KV heads · 128 · 2 B = 4096 B.
    const perTokenBytes = 36 * 4096;
    for (const n of [1, 1024, 8192, 32768]) {
      expect(kvReadBytes(run(128, n).decode)).toBe(perTokenBytes * (n + 1));
    }
  });

  it('decode @ 1 attends to the cached token and the new one', () => {
    expect(attentionOps(run(128, 1).decode)[0].kvLen).toBe(2);
  });

  it('memory holds max(P, N + 1) tokens of KV', () => {
    // N + 1 > P: sized for the cache after the decode step.
    expect(run(128, 8192).memory.perStage[0].sequenceStateBytes).toBe(36 * 4096 * 8193);
    // P > N + 1: sized for the prompt.
    expect(run(8192, 1024).memory.perStage[0].sequenceStateBytes).toBe(36 * 4096 * 8192);
  });
});

describe('4 × DGX Spark-like nodes: ring fabric vs switch (docs §6 "Fabric shapes")', () => {
  // A Spark-like link built here (not the preset), so the test pins the engine, not preset numbers: one port
  // 25 GB/s per direction, both ports capped at 25 GB/s together, α 1.5 µs, relay 5 µs.
  const link: LinkSpec = {
    kind: 'roce',
    name: 'Spark-like',
    bandwidthPerDirectionBytesPerSec: 25e9,
    latencySec: 1.5e-6,
    allPortsCapPerDirectionBytesPerSec: 25e9,
    forwardingLatencySec: 5e-6,
    p2p: 'direct',
    sources: [{ url: 'synthetic://test', title: 't', fields: ['bandwidthPerDirectionBytesPerSec', 'latencySec', 'allPortsCapPerDirectionBytesPerSec', 'forwardingLatencySec'] }],
    estimated: [],
  };
  const switched: TopologySpec = { id: 's', name: 's', devicesPerNode: 1, nodes: 4, interNode: link, sources: [] };
  const ring: TopologySpec = { ...switched, interNodeFabric: { kind: 'ring' } };
  const sim = (topology: TopologySpec) =>
    simulate({
      model: REF_DENSE_8B,
      quant: BF16,
      parallel: { tp: 4, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 4096, decodeContextLen: 8192 },
      hardware: { device: DGX_SPARK, topology },
    });

  it('decode is slower on the ring (latency-bound: the tree needs a 2-hop step), prefill is bit-equal (ring at the cap)', () => {
    const a = sim(switched);
    const b = sim(ring);
    expect(b.decode.tokensPerSec).toBeLessThan(a.decode.tokensPerSec);
    expect(b.prefill.tokensPerSec).toBe(a.prefill.tokensPerSec);
  });
});
