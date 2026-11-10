/**
 * End to end: in-network reduction (`SimOptions.inNetworkReduction`) and multi-NIC nodes (`TopologySpec.nicsPerNode`)
 * through `simulate` (docs/simulation-model.md §6 "In-network reduction (NVLS, SHARP)", "Multi-NIC nodes").
 */

import { describe, expect, it } from 'vitest';
import { H100_SXM } from '../hw/presets/nvidia-datacenter';
import { IB_NDR_400, NVLINK_4 } from '../hw/presets/links';
import { HGX_H100_16_NDR, HGX_H100_8, nodeTopology } from '../hw/presets/topologies';
import type { TopologySpec } from '../hw/types';
import type { QuantConfig } from '../ir/model';
import { REF_DENSE_8B } from '../models/reference';
import { simulate } from './simulate';
import type { OpResult, PhaseResult, SimInput, SimOptions } from './types';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

function input(tp: number, topology: TopologySpec, options: Partial<SimOptions> = {}): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: { tp, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 4096, decodeContextLen: 4096 },
    hardware: { device: H100_SXM, topology },
    options,
  };
}

/** The TP all-reduces of a phase. */
function tpAllReduces(phase: PhaseResult): OpResult[] {
  return phase.stages
    .flatMap((s) => s.layers)
    .flatMap((l) => l.ops)
    .filter(
      (o) => o.op.kind === 'collective' && o.op.group === 'tp' && o.op.collective === 'all_reduce',
    );
}

describe('NVLS on HGX H100 (TP8, one NVLink Switch domain)', () => {
  const off = simulate(input(8, HGX_H100_8));
  const on = simulate(input(8, HGX_H100_8, { inNetworkReduction: 'on' }));

  it('decode and prefill are faster with the switch on', () => {
    expect(on.decode.tokensPerSec).toBeGreaterThan(off.decode.tokensPerSec);
    expect(on.prefill.tokensPerSec).toBeGreaterThan(off.prefill.tokensPerSec);
  });

  it('every TP all-reduce costs the NVLS 2·α = 1.6 µs latency', () => {
    const ops = [...tpAllReduces(on.decode), ...tpAllReduces(on.prefill)];
    expect(ops.length).toBeGreaterThan(0);
    for (const o of ops) expect(o.commLatencySec).toBeCloseTo(1.6e-6, 15);
  });

  it("'off' deep-equals the run without the option; explicit 'ring' ignores 'on'", () => {
    expect(simulate(input(8, HGX_H100_8, { inNetworkReduction: 'off' }))).toEqual(off);
    expect(
      simulate(input(8, HGX_H100_8, { collectiveAlgorithm: 'ring', inNetworkReduction: 'on' })),
    ).toEqual(simulate(input(8, HGX_H100_8, { collectiveAlgorithm: 'ring' })));
  });
});

describe('NVLS + SHARP on 2 × HGX H100 over NDR400 (TP16, hierarchical)', () => {
  it('prefill and decode are faster with the switch on', () => {
    const off = simulate(input(16, HGX_H100_16_NDR));
    const on = simulate(input(16, HGX_H100_16_NDR, { inNetworkReduction: 'on' }));
    expect(on.decode.tokensPerSec).toBeGreaterThan(off.decode.tokensPerSec);
    expect(on.prefill.tokensPerSec).toBeGreaterThan(off.prefill.tokensPerSec);
  });
});

describe('NICs per node on 2 × 8 H100 over NDR400 (TP16 prefill, no overlap)', () => {
  const topo = (nicsPerNode?: number) =>
    nodeTopology({
      devicesPerNode: 8,
      nodes: 2,
      intraNode: NVLINK_4,
      interNode: IB_NDR_400,
      device: H100_SXM,
      ...(nicsPerNode !== undefined ? { nicsPerNode } : {}),
    });
  const prefill = (nicsPerNode?: number) =>
    simulate(input(16, topo(nicsPerNode), { overlap: 'none' })).prefill.tokensPerSec;

  it('4 NICs is slower and 16 NICs faster than the default 8 (one per GPU)', () => {
    expect(prefill(4)).toBeLessThan(prefill());
    expect(prefill(16)).toBeGreaterThan(prefill());
  });

  it('8 NICs deep-equals the default', () =>
    expect(simulate(input(16, { ...topo(), nicsPerNode: 8 }))).toEqual(
      simulate(input(16, topo())),
    ));
});
