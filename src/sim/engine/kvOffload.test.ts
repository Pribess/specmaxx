/**
 * KV-cache tiering in the engine (docs/simulation-model.md §12 "KV cache off-device"): resolving the placement
 * (defaults, unified memory, the no-op drop), the validation messages, the tiers' own sequence bound, PD
 * disaggregation (the decode pool fits more, the KV-transfer bytes stay the full cache), and the drafter (no fetch).
 */
import { describe, expect, it } from 'vitest';
import { B200, composeTopology, hostPcieLink, singleDevice } from '../hw/presets';
import { M5_ULTRA_80C } from '../hw/presets/apple';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import type { QuantConfig } from '../ir/model';
import { REF_DENSE_8B, TINY_CSA, TINY_HC_NGRAM, TINY_MLA_SPARSE } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import { simulateDisaggregated, type DisaggregatedInput } from './disaggregation';
import { KV_CACHE_CSA_WARNING, offloadTiers, resolveKvCache, resolveOffload } from './host';
import { simulate } from './simulate';
import type { KvCacheOffloadConfig, SimInput } from './types';
import { KV_CACHE_NO_OFFLOADABLE_ERROR, kvCacheErrors, validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const FP8_W: QuantConfig = { ...BF16, weights: { dtype: 'fp8_e4m3' } };

function expectClose(actual: number, expected: number, rel = 1e-12): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

function input(kvCache: KvCacheOffloadConfig | undefined, over: Partial<SimInput> = {}): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: FP8_W,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 8, promptLen: 32768, decodeContextLen: 32768 },
    hardware: {
      device: RTX_5090,
      topology: singleDevice(RTX_5090),
      host: { ram: DDR5_6000_2CH, ssd: NVME_990_PRO_2TB },
    },
    ...(kvCache === undefined ? {} : { deployment: { offload: { kvCache } } }),
    ...over,
  };
}

describe('resolveKvCache', () => {
  it('defaults: f_ram = 1 − f_dev (all off-device KV in host RAM), f_pre = f_dev, uniform h = f_dev', () => {
    const w: string[] = [];
    expect(resolveKvCache({ deviceFraction: 0.25 }, 'dedicated', 0, w)).toEqual({
      deviceFraction: 0.25,
      ramShare: 1,
      nvmeShare: 0,
      hitRatio: 0.25,
      prefixDeviceFraction: 1,
    });
    // With a prefix hit, f_pre defaults to f_dev.
    expect(
      resolveKvCache({ deviceFraction: 0.25 }, 'dedicated', 0.5, w)?.prefixDeviceFraction,
    ).toBe(0.25);
    expect(w).toEqual([]);
  });

  it('split: ramShare = f_ram/(1 − f_dev), nvmeShare = f_ssd/(1 − f_dev); custom h kept', () => {
    const r = resolveKvCache(
      { deviceFraction: 0.5, hostRamFraction: 0.125, hitRatio: { kind: 'custom', h: 0.9 } },
      'dedicated',
      0,
      [],
    );
    expect(r?.ramShare).toBe(0.25);
    expect(r?.nvmeShare).toBe(0.75);
    expect(r?.hitRatio).toBe(0.9);
  });

  it('prefix-only (f_dev 1): hostRamFraction absent ⇒ host RAM, hostRamFraction 0 ⇒ the SSD', () => {
    const ram = resolveKvCache(
      { deviceFraction: 1, prefixDeviceFraction: 0.5 },
      'dedicated',
      0.4,
      [],
    );
    expect([ram?.ramShare, ram?.nvmeShare]).toEqual([1, 0]);
    const ssd = resolveKvCache(
      { deviceFraction: 1, hostRamFraction: 0, prefixDeviceFraction: 0.5 },
      'dedicated',
      0.4,
      [],
    );
    expect([ssd?.ramShare, ssd?.nvmeShare]).toEqual([0, 1]);
  });

  it('no-op configs resolve to nothing: f_dev 1 without a prefix hit, or with f_pre 1', () => {
    expect(resolveKvCache({ deviceFraction: 1 }, 'dedicated', 0, [])).toBeUndefined();
    expect(
      resolveKvCache({ deviceFraction: 1, prefixDeviceFraction: 0 }, 'dedicated', 0, []),
    ).toBeUndefined();
    expect(resolveKvCache({ deviceFraction: 1 }, 'dedicated', 0.5, [])).toBeUndefined();
    expect(
      resolveOffload(REF_DENSE_8B, { kvCache: { deviceFraction: 1 } }, 'dedicated', 1),
    ).toEqual({
      offload: undefined,
      warnings: [],
    });
  });

  it('unified memory: host RAM folds into the device (with a warning); only the SSD share moves', () => {
    const w: string[] = [];
    // f_dev 0.25, f_ram 0.25, f_ssd 0.5: nvmeShare 2/3 → f_dev' = 1 − 0.75·2/3 = 0.5, all of it on the SSD.
    const r = resolveKvCache({ deviceFraction: 0.25, hostRamFraction: 0.25 }, 'unified', 0, w);
    expectClose(r?.deviceFraction ?? NaN, 0.5);
    expect([r?.ramShare, r?.nvmeShare]).toEqual([0, 1]);
    expect(w).toHaveLength(1);
    expect(w[0]).toMatch(/KV cache in host RAM stays on the device/);
    // All in host RAM on unified memory: nothing moves.
    expect(resolveKvCache({ deviceFraction: 0.25 }, 'unified', 0, [])).toBeUndefined();
    const resolved = resolveOffload(
      REF_DENSE_8B,
      { kvCache: { deviceFraction: 0.25, hostRamFraction: 0 } },
      'unified',
      1,
    );
    expect(resolved.offload?.hostRamIsDevice).toBe(true);
    expect(offloadTiers(resolved.offload!)).toEqual(new Set(['nvme']));
  });

  it('a unified-memory run with the KV cache on the SSD puts no KV in host RAM', () => {
    const r = simulate(
      input(
        { deviceFraction: 0.5, hostRamFraction: 0 },
        {
          hardware: {
            device: M5_ULTRA_80C,
            topology: singleDevice(M5_ULTRA_80C),
            host: { ssd: NVME_990_PRO_2TB },
          },
        },
      ),
    );
    expect(r.memory.tiers.map((t) => t.tier)).toEqual(['nvme']);
    expect(r.decode.transfer?.bytes).toBeGreaterThan(0);
  });

  it('a model with CSA blocks warns that their caches stay on the device', () => {
    const r = resolveOffload(TINY_CSA, { kvCache: { deviceFraction: 0.5 } }, 'dedicated', 1);
    expect(r.warnings).toContain(KV_CACHE_CSA_WARNING);
  });
});

describe('validation of deployment.offload.kvCache', () => {
  it.each([
    [
      { deviceFraction: 1.5 },
      'deployment.offload.kvCache.deviceFraction: the KV cache share on the device must be in [0, 1] (got 1.5)',
    ],
    [
      { deviceFraction: 0.5, hostRamFraction: 0.75 },
      'deployment.offload.kvCache.hostRamFraction: the KV cache share in host RAM must be in [0, 1 − deviceFraction] (got 0.75)',
    ],
    [
      { deviceFraction: 0.5, prefixDeviceFraction: -0.1 },
      "deployment.offload.kvCache.prefixDeviceFraction: the cached prefix's KV cache share on the device must be in [0, 1] (got -0.1)",
    ],
    [
      { deviceFraction: 0.5, hitRatio: { kind: 'custom', h: 2 } },
      'deployment.offload.kvCache.hitRatio.h: the KV cache hit ratio must be in [0, 1] (got 2)',
    ],
    [
      { deviceFraction: 0, hitRatio: { kind: 'custom', h: 0.5 } },
      'deployment.offload.kvCache.hitRatio: no KV cache is on the device (deviceFraction 0), h must be 0',
    ],
  ] as [KvCacheOffloadConfig, string][])('%j', (kv, message) => {
    expect(kvCacheErrors(kv, TINY_MLA_SPARSE, true)).toEqual([message]);
    expect(validateInput(input(kv))).toContain(message);
  });

  it('an unknown hit model is rejected', () => {
    const kv = {
      deviceFraction: 0.5,
      hitRatio: { kind: 'skewed' },
    } as unknown as KvCacheOffloadConfig;
    expect(kvCacheErrors(kv, REF_DENSE_8B, true)).toEqual([
      "deployment.offload.kvCache.hitRatio: the KV cache hit model must be 'uniform' or 'custom'",
    ]);
  });

  it('a model without offloadable KV is rejected when the placement moves anything', () => {
    expect(kvCacheErrors({ deviceFraction: 0.5 }, TINY_CSA, true)).toEqual([
      KV_CACHE_NO_OFFLOADABLE_ERROR,
    ]);
    expect(kvCacheErrors({ deviceFraction: 1 }, TINY_CSA, false)).toEqual([]);
  });

  it('a valid placement passes; two-batch overlap with KV offload is rejected as for other offload', () => {
    expect(validateInput(input({ deviceFraction: 0.25, hostRamFraction: 0.25 }))).toEqual([]);
    const tbo = input({ deviceFraction: 0.25 });
    tbo.deployment = { ...tbo.deployment, twoBatchOverlap: true };
    expect(validateInput(tbo)).toContain(
      'deployment.twoBatchOverlap with weights offloaded is not modelled',
    );
  });
});

describe('memory: the tiers bound the batch too', () => {
  it('a host RAM too small for 8 sequences caps maxSequencesPerReplica and fails the fit', () => {
    // Off-device KV per sequence: 0.75·36·32,769·4096 B = 3.624 GB; a 16 GB host RAM holds floor(16e9/3.624e9) = 4.
    const small = { ...DDR5_6000_2CH, capacityBytes: 16e9 };
    const r = simulate(
      input(
        { deviceFraction: 0.25 },
        {
          hardware: {
            device: RTX_5090,
            topology: singleDevice(RTX_5090),
            host: { ram: small },
          },
        },
      ),
    );
    const perSeq = 0.75 * 36 * 32769 * 4096;
    expect(r.memory.maxSequencesPerReplica).toBe(Math.floor(16e9 / perSeq));
    expect(r.memory.tiers[0].fits).toBe(false);
    expect(r.memory.fits).toBe(false);
  });
});

describe('KV-cache offload leaves the other groups where they are', () => {
  it('an n-gram model with its tables (and token embedding) on the device keeps them there', () => {
    const over = {
      model: TINY_HC_NGRAM,
      quant: BF16,
      workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
    };
    const r = simulate(input({ deviceFraction: 0.5 }, over));
    const base = simulate(input(undefined, over));
    const groups = new Set(
      [...r.prefill.stages, ...r.decode.stages].flatMap((s) =>
        s.layers.flatMap((l) =>
          l.ops.flatMap((o) => (o.op.kind === 'transfer' ? [o.op.group] : [])),
        ),
      ),
    );
    expect(groups).toEqual(new Set(['kv_cache']));
    for (const t of r.memory.tiers) expect(t.breakdown.kvCache).toBe(t.bytes);
    expect(r.memory.perStage[0].weightBytes).toBe(base.memory.perStage[0].weightBytes);
    expect(r.memory.perStage[0].lookupTableBytes).toBe(base.memory.perStage[0].lookupTableBytes);
  });
});

describe('PD disaggregation with KV-cache offload in the shared deployment', () => {
  // B200 has no cited host link: give it PCIe 5.0 x16 and a 1 TB DDR5 host (both pools validate the offload; a
  // 96 GB host would cap the decode pool at floor(96e9 / 3.62e9) = 26 sequences).
  const hardware = {
    device: B200,
    topology: composeTopology({ deviceId: 'b200', devicesPerNode: 8, nodes: 1 }),
    host: { link: hostPcieLink(5, 16), ram: { ...DDR5_6000_2CH, capacityBytes: 1e12 } },
  };
  function pd(kvCache: KvCacheOffloadConfig | undefined): DisaggregatedInput {
    const pool = { hardware, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 }, batch: 32 };
    return {
      shared: {
        model: REF_DENSE_8B,
        quant: BF16,
        workload: { batch: 32, promptLen: 32768, decodeContextLen: 32768 },
        ...(kvCache === undefined ? {} : { deployment: { offload: { kvCache } } }),
      },
      prefillPool: pool,
      decodePool: pool,
      config: {
        outputLen: 1024,
        prefillInstances: 1,
        decodeInstances: 1,
        kvTransfer: {
          link: { name: 'test link', latencySec: 5e-6, bandwidthPerDirectionBytesPerSec: 50e9 },
          overlap: 'layerwise',
        },
      },
    };
  }

  it('the decode pool fits a larger batch; the KV-transfer bytes stay the full cache', () => {
    const on = simulateDisaggregated(pd(undefined));
    const off = simulateDisaggregated(pd({ deviceFraction: 0.25 }));
    expect(off.decodePool.memory.maxSequencesPerReplica).toBeGreaterThan(
      3 * on.decodePool.memory.maxSequencesPerReplica,
    );
    expect(off.kvTransfer?.requestBytes).toBe(on.kvTransfer?.requestBytes);
    expect(off.kvTransfer?.batchBytesPerDevice).toBe(on.kvTransfer?.batchBytesPerDevice);
    // 36 layers × 32768 tokens × 4096 B per request.
    expect(on.kvTransfer?.requestBytes).toBe(36 * 32768 * 4096);
    // Without a prefix hit the prefill pool has nothing to fetch.
    expect(off.prefillPool.prefill.transfer).toBeUndefined();
    expect(off.decodePool.decode.transfer?.bytes).toBeGreaterThan(0);
  });
});

describe('speculative decoding: the drafter keeps its KV on the device', () => {
  // pp 2 on two B200s of one node (spec-dec runs at pp > 1 since the PP refinement): the drafter on the last stage
  // still keeps its KV on the device, and every target layer of both stages fetches.
  const b200x2 = {
    device: B200,
    topology: composeTopology({ deviceId: 'b200', devicesPerNode: 2, nodes: 1 }),
    host: { link: hostPcieLink(5, 16), ram: { ...DDR5_6000_2CH, capacityBytes: 1e12 } },
  };
  it.each([1, 2])('pp %d: MTP drafter layers get no kv_fetch op, the target layers do', (pp) => {
    const r = simulate({
      ...input({ deviceFraction: 0.5 }, { model: REF_DENSE_8B_MTP, quant: BF16 }),
      ...(pp === 1 ? {} : { hardware: b200x2 }),
      parallel: { tp: 1, pp, dp: 1, ep: 1 },
      workload: { batch: pp, promptLen: 8192, decodeContextLen: 8192 },
      deployment: {
        offload: { kvCache: { deviceFraction: 0.5 } },
        speculative: { method: 'mtp', draftLength: 2, acceptance: { kind: 'iid', alpha: 0.7 } },
      },
    });
    const layers = r.decode.stages.flatMap((s) => s.layers);
    const fetchesIn = (pred: (name: string) => boolean): number =>
      layers
        .filter((l) => pred(l.name))
        .flatMap((l) => l.ops)
        .filter((o) => o.op.kind === 'transfer' && o.op.group === 'kv_cache').length;
    expect(fetchesIn((n) => n.startsWith('draft.'))).toBe(0);
    expect(fetchesIn((n) => n.startsWith('layer.'))).toBe(36);
    expect(r.decode.stages.every((s) => s.transferSec > 0)).toBe(true);
  });
});
