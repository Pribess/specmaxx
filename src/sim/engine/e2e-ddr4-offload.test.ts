/**
 * DDR4 vs DDR5 host RAM for expert offload, end to end through `simulate`: with the experts in host RAM and the
 * PCIe 5.0 x16 link (63 GB/s per direction) faster than DDR4-3200 × 2 (51.2 GB/s), the expert fetch is bound by the
 * host RAM and its bandwidth term is M·W / 51.2e9; on DDR5-6000 × 2 (96 GB/s) the link binds instead.
 */
import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR4_3200_2CH, DDR5_6000_2CH } from '../hw/presets/storage';
import type { HostMemorySpec } from '../hw/types';
import type { QuantConfig } from '../ir/model';
import { REF_MOE_30B_A3B } from '../models/reference';
import { simulate } from './simulate';

const QUANT: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
  modules: { routed_experts: { weights: { dtype: 'int4', extraBitsPerElem: 0.5 } } },
};
/** One expert at 4.5 bits: 3·2048·768·4.5/8. Unit: bytes. */
const W_EXPERT = 2_654_208;
/** Experts fetched per layer per decode step at f = 0, batch 1: the top-8 routed experts. Count. */
const M = 8;
/** PCIe 5.0 x16 per direction. Unit: B/s. */
const B_LINK = (32e9 * 16 * 128) / 130 / 8;

function fetchOf(ram: HostMemorySpec) {
  const r = simulate({
    model: REF_MOE_30B_A3B,
    quant: QUANT,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: { device: RTX_5090, topology: singleDevice(RTX_5090), host: { ram } },
    deployment: {
      offload: {
        prefetchAccuracy: 1,
        routedExperts: { tier: 'host_ram', residentFraction: 0, hitRatio: { kind: 'uniform' } },
      },
    },
  });
  const fetch = r.decode.stages[0].layers[1].ops.find((o) => o.op.kind === 'transfer');
  if (!fetch) throw new Error('no transfer op');
  return fetch;
}

describe('expert transfer on DDR4-3200 ×2 vs DDR5-6000 ×2', () => {
  const ddr4 = fetchOf(DDR4_3200_2CH);
  const ddr5 = fetchOf(DDR5_6000_2CH);
  it('DDR4 is host-bandwidth-bound: M·W / 51.2e9', () => {
    expect(DDR4_3200_2CH.bandwidthBytesPerSec).toBe(51.2e9);
    expect(ddr4.bound).toBe('transfer');
    expect(ddr4.transferBandwidthSec).toBeCloseTo((M * W_EXPERT) / 51.2e9, 12);
  });
  it('DDR5 is link-bound: M·W / B_link', () => {
    expect(ddr5.transferBandwidthSec).toBeCloseTo((M * W_EXPERT) / B_LINK, 12);
  });
  it('the DDR4 fetch is slower by B_link / 51.2e9', () => {
    expect(ddr4.transferBandwidthSec / ddr5.transferBandwidthSec).toBeCloseTo(B_LINK / 51.2e9, 6);
  });
});
