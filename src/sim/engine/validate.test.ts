import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_PCIE_HOSTBOUNCE, SYNTH_PCIE_HOSTBOUNCE_LINK } from '../__fixtures__/synthetic';
import { decoderLayer, decoderModel } from '../ir/builders';
import type { ParallelConfig, QuantConfig } from '../ir/model';
import { REF_DENSE_8B, TINY_HYBRID } from '../models/reference';
import { singleDevice } from '../hw/presets';
import type { TopologySpec } from '../hw/types';
import type { SimInput } from './types';
import { validateInput } from './validate';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

function input(parallel: Partial<ParallelConfig> = {}, over: Partial<SimInput> = {}): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
    hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
    ...over,
  };
}

/** Asserts exactly that some error line matches `pattern`. */
function expectError(i: SimInput, pattern: RegExp): void {
  const errors = validateInput(i);
  expect(errors.some((e) => pattern.test(e)), `errors: ${errors.join(' | ')}`).toBe(true);
}

describe('T17 input errors', () => {
  it('a valid input has no errors', () => {
    expect(validateInput(input())).toEqual([]);
    expect(validateInput(input({ tp: 8 }))).toEqual([]);
  });

  it('heads % tp', () => {
    // 32 heads over tp 3 (placement also rejects tp 3 on nodes of 8)
    expectError(input({ tp: 3 }), /heads \(32\) must be divisible by tp \(3\)/);
  });

  it('batch % dp', () => {
    expectError(input({ dp: 2 }, { workload: { batch: 3, promptLen: 128, decodeContextLen: 256 } }), /batch \(3\) must be divisible by dp \(2\)/);
  });

  it('pp > layers', () => {
    const topo: TopologySpec = { ...SYNTH_CLUSTER, nodes: 8 }; // 64 devices, so only the layer rule fails
    expectError(input({ pp: 37 }, { hardware: { device: SYNTH_GPU, topology: topo } }), /pp \(37\) must not exceed the number of layers \(36\)/);
  });

  it('world > devices', () => {
    expectError(input({ tp: 8, dp: 4 }), /32 devices needed, the topology has 16/);
  });

  it('missing interNode link', () => {
    const noInter: TopologySpec = { ...SYNTH_CLUSTER, interNode: undefined };
    expectError(input({ tp: 16 }, { hardware: { device: SYNTH_GPU, topology: noInter } }), /no interNode link/);
  });

  it('invalid ep (MoE: 8 experts, ep 3) and ep on a dense model', () => {
    expectError(input({ ep: 3 }, { model: TINY_HYBRID }), /experts \(8\) must be divisible by ep \(3\)/);
    expectError(input({ ep: 2 }), /model has no MoE block/);
  });

  it('unknown engineId', () => {
    expectError(input({}, { hardware: { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU), engineId: 'npu' } }), /engineId 'npu'/);
  });

  it('non-integer degrees and empty workloads', () => {
    expectError(input({ tp: 0 }), /parallel\.tp must be an integer ≥ 1/);
    expectError(input({}, { workload: { batch: 1, promptLen: 0, decodeContextLen: 256 } }), /promptLen must be an integer ≥ 1/);
  });

  it('an unmodified device is validated too (review P2): zero memory bandwidth, zero peak, no engines, unsourced', () => {
    const on = (device: SimInput['hardware']['device']): SimInput => input({}, { hardware: { device, topology: singleDevice(device) } });
    expectError(on({ ...SYNTH_GPU, memory: { ...SYNTH_GPU.memory, bandwidthBytesPerSec: 0 } }), /memory\.bandwidthBytesPerSec must be > 0/);
    const zeroPeak = { ...SYNTH_GPU.engines[0], matmulPeaks: SYNTH_GPU.engines[0].matmulPeaks.map((p) => ({ ...p, flopsPerSec: 0 })) };
    expectError(on({ ...SYNTH_GPU, engines: [zeroPeak] }), /matmulPeaks\.0\.flopsPerSec must be > 0/);
    expectError(on({ ...SYNTH_GPU, engines: [] }), /no engines/);
    expectError(on({ ...SYNTH_GPU, sources: [] }), /is not covered by any sources/);
  });

  it('a host-bounce link with zero host bandwidth is rejected', () => {
    const link = { ...SYNTH_PCIE_HOSTBOUNCE_LINK, hostBounce: { hostMemBandwidthBytesPerSec: 0, extraLatencySec: 5e-6 } };
    expectError(input({ tp: 2 }, { hardware: { device: SYNTH_GPU, topology: { ...SYNTH_PCIE_HOSTBOUNCE, intraNode: link } } }), /hostMemBandwidthBytesPerSec must be > 0/);
  });

  it('value heads must divide tp (review P2): GDN vHeads 2 at tp 4', () => {
    const model = decoderModel({
      name: 'gdn',
      hidden: 256,
      vocab: 1000,
      tieEmbeddings: false,
      layers: [
        decoderLayer({
          hidden: 256,
          mixer: { kind: 'linear', variant: 'gated_delta_net', qkHeads: 1, vHeads: 2, keyDim: 64, valueDim: 64, outputGate: true },
          ffn: { kind: 'dense', intermediate: 512 },
        }),
      ],
    });
    expectError(input({ tp: 4 }, { model }), /vHeads \(2\) must be divisible by tp \(4\)/);
    expect(validateInput(input({ tp: 2 }, { model }))).toEqual([]);
  });
  it('invalid modifiers', () => {
    expectError(
      input({}, { hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER, deviceModifiers: [{ kind: 'clockScale', factor: 0, note: 'x', sources: [], estimatedReason: 'test' }] } }),
      /factor must be a finite number > 0/,
    );
    expectError(
      input({}, { hardware: { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU), linkModifiers: { intraNode: [{ kind: 'linkBandwidth', bandwidthPerDirectionBytesPerSec: 1e9, note: 'x', sources: [], estimatedReason: 'test' }] } } }),
      /has no intraNode link/,
    );
  });
});
