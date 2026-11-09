import { describe, expect, it } from 'vitest';
import { PCIE_5_X16_DIRECT } from '../../sim/hw/presets/links';
import { RTX_5090 } from '../../sim/hw/presets/nvidia';
import { nodeTopology } from '../../sim/hw/presets/topologies';
import type { QuantConfig } from '../../sim/ir/model';
import { REF_DENSE_8B } from '../../sim/models/reference';
import type { SimInput } from '../../sim/engine/types';
import { batchSweep, SWEEP_STEPS } from './sweep';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

describe('batch sweep with DP > 1', () => {
  it('DP 2: every point is a batch DP divides (2·2^k), each replica at 2^k, aggregate = 2 × one GPU', () => {
    const at = (n: number, dp: number): SimInput => ({
      model: REF_DENSE_8B,
      quant: BF16,
      parallel: { tp: 1, pp: 1, dp, ep: 1 },
      workload: { batch: dp, promptLen: 4096, decodeContextLen: 4096 },
      hardware: {
        device: RTX_5090,
        topology: nodeTopology({ devicesPerNode: n, nodes: 1, intraNode: PCIE_5_X16_DIRECT, device: RTX_5090 }),
      },
    });
    const dp = batchSweep(at(2, 2));
    const one = batchSweep(at(1, 1));
    expect(dp.map((p) => p.batch)).toEqual(Array.from({ length: SWEEP_STEPS }, (_, k) => 2 * 2 ** k));
    expect(dp).toHaveLength(one.length);
    dp.forEach((p, i) => {
      expect(p.decodeTokensPerSec).toBeCloseTo(2 * one[i].decodeTokensPerSec, 6);
      expect(p.perSeqTokensPerSec).toBeCloseTo(one[i].perSeqTokensPerSec, 9);
      expect(p.fits).toBe(one[i].fits);
    });
  });
});
