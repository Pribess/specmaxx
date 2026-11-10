/**
 * KV-cache tiering end to end through `simulate` (docs/simulation-model.md §12 "KV cache off-device"): a batch whose
 * KV cache does not fit the device fits with 75 % of it in host RAM, and decode then streams that share over PCIe
 * every step, so it is bound by the host link. Every rate below is quoted at its context.
 *
 * Setup: REF_DENSE_8B (Qwen3-8B shape: 36 layers, 8 KV heads × hd 128) with fp8 weights and a bf16 KV cache, on one
 * RTX 5090 (32 GB) with DDR5-6000 ×2 (96 GB/s, 96 GB) behind PCIe 5.0 x16, b 8, P 32768, decode @ 32k.
 * - kvB = 2·8·128·2 = 4096 B per token per layer; one sequence at 32,769 tokens holds 36·32,769·4096 = 4.832 GB.
 * - Device only: 8 sequences need 38.66 GB of KV: does not fit (at most 4 sequences fit).
 * - f_dev 0.25: 9.66 GB resident, 28.99 GB in host RAM; fits, and at most 16 sequences fit (≈ 4× = 1/f).
 * - Decode reads 8·0.75·32,768·4096 = 805,306,368 B per layer from host RAM; the link (57.6 GB/s) is slower than
 *   DDR5 (96 GB/s), so each layer's fetch takes α + 805,306,368 / B_link = 13.979 ms, and the step 36 × that =
 *   0.5032 s: 15.897 tok/s @ 32k (the prefetched fetch hides the 26.2 ms of compute), against 305.35 tok/s @ 32k
 *   on a device big enough to hold it all.
 */
import { describe, expect, it } from 'vitest';
import { GB200 } from '../hw/presets/nvidia-datacenter';
import { singleDevice } from '../hw/presets';
import { NVLINK_C2C } from '../hw/presets/links';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import type { HardwareStack } from '../hw/types';
import type { QuantConfig } from '../ir/model';
import type { TransferOp } from '../ir/ops';
import { REF_DENSE_8B } from '../models/reference';
import { simulate } from './simulate';
import type { KvCacheOffloadConfig, OpResult, SimInput, SimResult } from './types';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const FP8_W: QuantConfig = {
  weights: { dtype: 'fp8_e4m3' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
/** PCIe 5.0 x16 per direction, 32e9·16·128/130/8 × 256/280 (TLP payload efficiency, links.ts). Unit: B/s. */
const B_LINK = ((32e9 * 16 * 128) / 130 / 8) * (256 / 280);
/** α of a host-RAM read: DDR5 70 ns + PCIe 1 µs. Unit: seconds. */
const ALPHA_RAM = 70e-9 + 1e-6;
const LAYERS = 36;
const KVB = 2 * 8 * 128 * 2;
const B = 8;
const N = 32768;

const GPU: HardwareStack = {
  device: RTX_5090,
  topology: singleDevice(RTX_5090),
  host: { ram: DDR5_6000_2CH, ssd: NVME_990_PRO_2TB },
};

function run(
  kvCache: KvCacheOffloadConfig | undefined,
  extra: { prefetchAccuracy?: number; workload?: Partial<SimInput['workload']> } = {},
  hardware: HardwareStack = GPU,
): SimResult {
  return simulate({
    model: REF_DENSE_8B,
    quant: FP8_W,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: B, promptLen: N, decodeContextLen: N, ...extra.workload },
    hardware,
    ...(kvCache === undefined
      ? {}
      : {
          deployment: {
            offload: {
              kvCache,
              ...(extra.prefetchAccuracy === undefined
                ? {}
                : { prefetchAccuracy: extra.prefetchAccuracy }),
            },
          },
        }),
  });
}

/** The KV fetch op results of a phase, over every stage and layer. */
function kvFetches(r: SimResult, phase: 'prefill' | 'decode'): OpResult[] {
  return r[phase].stages.flatMap((s) =>
    s.layers.flatMap((l) =>
      l.ops.filter((o) => o.op.kind === 'transfer' && o.op.group === 'kv_cache'),
    ),
  );
}

describe('REF_DENSE_8B fp8 on one RTX 5090 + DDR5-6000 ×2 over PCIe 5.0 x16, b 8, decode @ 32k: KV cache 25 % on the GPU, 75 % in host RAM', () => {
  const base = run(undefined);
  const ram = run({ deviceFraction: 0.25 });

  it('(a) device-only does not fit; with f_dev 0.25 it fits and maxSequencesPerReplica grows ≈ 1/f (4 → 16)', () => {
    expect(base.memory.fits).toBe(false);
    expect(base.memory.maxSequencesPerReplica).toBe(4);
    // Formula: device KV = B·f·36·(N + 1)·kvB; tier KV = B·(1 − f)·36·(N + 1)·kvB
    const fullKv = B * LAYERS * (N + 1) * KVB;
    expect(base.memory.perStage[0].sequenceStateBytes).toBe(fullKv);
    expect(ram.memory.fits).toBe(true);
    expectClose(ram.memory.perStage[0].sequenceStateBytes, 0.25 * fullKv);
    const [tier] = ram.memory.tiers;
    expect(tier.tier).toBe('host_ram');
    expectClose(tier.breakdown.kvCache, 0.75 * fullKv);
    expect(tier.bytes).toBe(tier.breakdown.kvCache);
    expect(ram.memory.maxSequencesPerReplica).toBe(16);
    expect(ram.memory.maxSequencesPerReplica / base.memory.maxSequencesPerReplica).toBe(4);
    // Staging: the largest layer fetch, double-buffered (q = 1): 2·805,306,368 B.
    expect(ram.memory.perStage[0].stagingBytes).toBe(2 * B * 0.75 * N * KVB);
  });

  it('(b) decode is bound by the PCIe link: each fetch is α + useful/B_link, 15.897 tok/s @ 32k (305.35 device-only)', () => {
    const fetches = kvFetches(ram, 'decode');
    expect(fetches).toHaveLength(LAYERS);
    const useful = B * 0.75 * N * KVB; // 805,306,368 B per layer
    for (const o of fetches) {
      const op = o.op as TransferOp;
      expect(op.usefulBytes).toBe(useful);
      expect(op.prefetchedFraction).toBe(1);
      // Service = the link term, above the host-RAM term (96 GB/s) and the request terms.
      expectClose(o.transferBandwidthSec, useful / B_LINK);
      expect(o.transferResources?.hostRamSec ?? Infinity).toBeLessThan(o.transferBandwidthSec);
      expectClose(o.sec, ALPHA_RAM + useful / B_LINK);
    }
    // Formula: tok/s = B / Σ_ℓ (α + useful/B_link)   (q = 1 and transfer > compute: t_s = max(core, xfer) = xfer)
    const step = LAYERS * (ALPHA_RAM + useful / B_LINK);
    expectClose(ram.decode.tokensPerSec, B / step, 1e-9);
    expectClose(ram.decode.tokensPerSec, 15.8972, 1e-5);
    expectClose(base.decode.tokensPerSec, 305.354, 1e-5);
    expect(ram.decode.stages[0].computeSec).toBeLessThan(step);
    // Without prefetching (p = 0), the transfers add to the step: B / (core + Σ_ℓ t_ℓ), 15.11 tok/s @ 32k.
    const serial = run({ deviceFraction: 0.25 }, { prefetchAccuracy: 0 });
    expectClose(serial.decode.tokensPerSec, B / (serial.decode.stages[0].computeSec + step), 1e-9);
    expectClose(serial.decode.tokensPerSec, 15.1106, 1e-5);
  });

  it('(c) half of the off-device KV on the SSD (f_ram 0.25, f_ssd 0.5) is slower: 3.08 tok/s @ 32k', () => {
    const split = run({ deviceFraction: 0.25, hostRamFraction: 0.25 });
    const names = kvFetches(split, 'decode').map((o) => o.op.name);
    expect(names.filter((n) => n === 'attn.kv_fetch')).toHaveLength(LAYERS);
    expect(names.filter((n) => n === 'attn.kv_fetch_ssd')).toHaveLength(LAYERS);
    expect(split.decode.tokensPerSec).toBeLessThan(ram.decode.tokensPerSec);
    expectClose(split.decode.tokensPerSec, 3.0817, 1e-4);
    const byTier = Object.fromEntries(split.memory.tiers.map((t) => [t.tier, t.breakdown.kvCache]));
    const fullKv = B * LAYERS * (N + 1) * KVB;
    expectClose(byTier.host_ram, 0.25 * fullKv);
    expectClose(byTier.nvme, 0.5 * fullKv);
  });

  it("(c) staging of the split: both tiers' fetches of a layer are one buffer (prefetch on: 2·Σ, off: Σ)", () => {
    // Per layer: host RAM 8·0.25·32,768·4096 = 268,435,456 B, SSD 8·0.5·32,768·4096 = 536,870,912 B, Σ = 805,306,368 B.
    const ram = B * 0.25 * N * KVB;
    const ssd = B * 0.5 * N * KVB;
    // Prefetching (q = 1, both tiers): the layer being read (Σ) + the next layer filling from each source (ram + ssd).
    const on = run({ deviceFraction: 0.25, hostRamFraction: 0.25 });
    expect(on.memory.perStage[0].stagingBytes).toBe(2 * (ram + ssd));
    // No prefetching (q = 0): one layer's fetch, both tiers together.
    const off = run({ deviceFraction: 0.25, hostRamFraction: 0.25 }, { prefetchAccuracy: 0 });
    expect(off.memory.perStage[0].stagingBytes).toBe(ram + ssd);
    // The same total as the all-in-RAM placement's buffers: splitting the tiers does not shrink the staging.
    expect(
      run({ deviceFraction: 0.25 }, { prefetchAccuracy: 0 }).memory.perStage[0].stagingBytes,
    ).toBe(ram + ssd);
  });

  it('(d) a cached prefix half in host RAM: fetched at prefill, hidden with prefetching, exposed without', () => {
    // P 32768, prefix hit 0.5 → 16,384 cached tokens per sequence, f_pre 0 → all fetched; decode KV stays on the GPU.
    const workload = { batch: 4, promptLen: N, prefixCacheHitRatio: 0.5 };
    const kv = { deviceFraction: 1, prefixDeviceFraction: 0 };
    const off = run(undefined, { workload });
    const hidden = run(kv, { workload });
    const exposed = run(kv, { workload, prefetchAccuracy: 0 });
    const fetches = kvFetches(hidden, 'prefill');
    expect(fetches).toHaveLength(LAYERS);
    expect((fetches[0].op as TransferOp).reads).toEqual([{ count: 4, bytesEach: 16384 * KVB }]);
    expect(kvFetches(hidden, 'decode')).toEqual([]);
    expect(hidden.decode.tokensPerSec).toBe(off.decode.tokensPerSec);
    // Prefetched (q = 1) and far shorter than the compute-bound prefill: identical rate.
    expect(hidden.prefill.tokensPerSec).toBe(off.prefill.tokensPerSec);
    expect(hidden.prefill.transfer?.exposedSec).toBe(0);
    // p = 0: the cycle grows by exactly Σ_ℓ t_fetch.
    const xfer = exposed.prefill.transfer?.sec ?? 0;
    expectClose(xfer, LAYERS * (ALPHA_RAM + (4 * 16384 * KVB) / B_LINK));
    expectClose(exposed.prefill.cycleSec, off.prefill.cycleSec + xfer, 1e-12);
    // The memory fit and the tiers: decode KV is all resident (f_dev 1), so no tier bytes.
    expect(hidden.memory.tiers).toEqual([]);
  });
});

describe('GB200 (single device) reads the off-device KV over NVLink-C2C from Grace LPDDR5X, decode @ 32k', () => {
  it('the KV fetch uses the C2C link (450 GB/s) and Grace RAM, with no host given', () => {
    const r = run({ deviceFraction: 0.25 }, {}, { device: GB200, topology: singleDevice(GB200) });
    const [o] = kvFetches(r, 'decode');
    const useful = (o.op as TransferOp).usefulBytes;
    expect(NVLINK_C2C.bandwidthPerDirectionBytesPerSec).toBe(450e9);
    expectClose(o.transferResources?.linkSec ?? 0, useful / 450e9);
    const res = o.transferResources;
    expectClose(o.transferBandwidthSec, Math.max(res?.linkSec ?? 0, res?.hostRamSec ?? 0));
    expect(r.warnings.some((w) => w.includes('NVLink-C2C') || w.includes('C2C'))).toBe(true);
  });
});
