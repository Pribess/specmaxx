/**
 * The remote KV pool tier (docs/simulation-model.md §12 "Remote KV pool"): resolving its share and link, the fetch
 * cost rule, validation, the memory report's pool entry (per deployment, unlimited or capped), identity (a pool with
 * no share changes nothing), the overlap of a pool stream with a host-RAM stream, prefix-cache hits served from the
 * pool at prefill, and an end-to-end decode that is NIC-bound. Every rate below is quoted at its context.
 *
 * Setup (as `e2e-kv-offload.test.ts`): REF_DENSE_8B (36 layers, 8 KV heads × hd 128, kvB 4096 B) with fp8 weights and
 * a bf16 KV cache on one RTX 5090 + DDR5-6000 ×2 behind PCIe 5.0 x16, b 8, P = N = 32768. The stack has no inter-node
 * link, so the pool is reached over the default InfiniBand NDR400 NIC (50 GB/s per GPU, α 1 µs), slower than the GPU's
 * PCIe 5.0 x16 (57.6 GB/s): decode is bound by the NIC.
 * - Decode reads 8·0.75·32,768·4096 = 805,306,368 B per layer from the pool: α + useful/B_pool = 16.107 ms per layer,
 *   0.5799 s per step: 13.796 tok/s @ 32k (15.897 tok/s @ 32k with the same share in host RAM over PCIe).
 */
import { describe, expect, it } from 'vitest';
import { hostPcieLink, singleDevice } from '../hw/presets';
import { IB_NDR_400, NVLINK_4 } from '../hw/presets/links';
import { B200 } from '../hw/presets/nvidia-datacenter';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import { nodeTopology } from '../hw/presets/topologies';
import type { HardwareStack } from '../hw/types';
import type { QuantConfig } from '../ir/model';
import type { TransferOp } from '../ir/ops';
import { REF_DENSE_8B } from '../models/reference';
import { resolveKvCache, resolveRemotePool } from './host';
import { simulate } from './simulate';
import {
  MOONCAKE_MEASURED_POOL_BYTES_PER_SEC_PER_NODE,
  prefetchesFromMultipleTiers,
  remotePoolTransferTime,
} from './transfer';
import type { KvCacheOffloadConfig, OpResult, SimInput, SimResult } from './types';
import { kvCacheErrors, validateInput } from './validate';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const FP8_W: QuantConfig = {
  weights: { dtype: 'fp8_e4m3' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
/** PCIe 5.0 x16 per direction (links.ts: 128/130 encoding × 256/280 TLP payload). Unit: B/s. */
const B_LINK = ((32e9 * 16 * 128) / 130 / 8) * (256 / 280);
/** The default pool link: one InfiniBand NDR400 NIC per GPU. Unit: B/s, seconds. */
const B_POOL = 400e9 / 8;
const ALPHA_POOL = 1e-6;
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

function input(
  kvCache: KvCacheOffloadConfig | undefined,
  extra: { prefetchAccuracy?: number; workload?: Partial<SimInput['workload']> } = {},
  hardware: HardwareStack = GPU,
): SimInput {
  return {
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
  };
}

const run = (...args: Parameters<typeof input>): SimResult => simulate(input(...args));

/** The KV fetch op results of a phase, over every stage and layer. */
function kvFetches(r: SimResult, phase: 'prefill' | 'decode'): OpResult[] {
  return r[phase].stages.flatMap((s) =>
    s.layers.flatMap((l) =>
      l.ops.filter((o) => o.op.kind === 'transfer' && o.op.group === 'kv_cache'),
    ),
  );
}

describe('resolveKvCache with a remote pool', () => {
  it('f_pool takes its share of the off-device part; f_ram defaults to the rest', () => {
    const w: string[] = [];
    expect(
      resolveKvCache({ deviceFraction: 0.25, remotePool: { fraction: 0.75 } }, 'dedicated', 0, w),
    ).toEqual({
      deviceFraction: 0.25,
      ramShare: 0,
      nvmeShare: 0,
      remotePoolShare: 1,
      hitRatio: 0.25,
      prefixDeviceFraction: 1,
    });
    // Formula: f_ram = 1 − f_dev − f_pool = 0.25 ⇒ ramShare 1/3, remotePoolShare 2/3.
    const mixed = resolveKvCache(
      { deviceFraction: 0.25, remotePool: { fraction: 0.5 } },
      'dedicated',
      0,
      w,
    );
    expectClose(mixed?.ramShare ?? 0, 0.25 / 0.75);
    expectClose(mixed?.remotePoolShare ?? 0, 0.5 / 0.75);
    expect(mixed?.nvmeShare).toBe(0);
    // Three tiers: f_ram 0.125 given, f_pool 0.5, f_ssd = 0.125.
    const three = resolveKvCache(
      { deviceFraction: 0.25, hostRamFraction: 0.125, remotePool: { fraction: 0.5 } },
      'dedicated',
      0,
      w,
    );
    expectClose(three?.ramShare ?? 0, 0.125 / 0.75);
    expectClose(three?.nvmeShare ?? 0, 0.125 / 0.75);
    expectClose(three?.remotePoolShare ?? 0, 0.5 / 0.75);
    expect(w).toEqual([]);
  });

  it('a pool with fraction 0 is no pool: the same placement as without it', () => {
    expect(
      resolveKvCache({ deviceFraction: 0.25, remotePool: { fraction: 0 } }, 'dedicated', 0, []),
    ).toEqual(resolveKvCache({ deviceFraction: 0.25 }, 'dedicated', 0, []));
  });

  it('prefix-only (f_dev 1) with a pool: the cached prefix is read from the pool', () => {
    const r = resolveKvCache(
      { deviceFraction: 1, prefixDeviceFraction: 0.5, remotePool: { fraction: 0 } },
      'dedicated',
      0.5,
      [],
    );
    expect(r).toMatchObject({ remotePoolShare: 1, ramShare: 0, nvmeShare: 0 });
    expect(r?.prefixDeviceFraction).toBe(0.5);
  });

  it('unified memory: host RAM folds into the device, the pool share stays (renormalised with the SSD)', () => {
    const w: string[] = [];
    // f_dev 0.25, f_ram 0.25 (default), f_pool 0.5: RAM folds in ⇒ f_dev' = 1 − 0.75·(2/3) = 0.5, pool share 1.
    const r = resolveKvCache(
      { deviceFraction: 0.25, remotePool: { fraction: 0.5 } },
      'unified',
      0,
      w,
    );
    expectClose(r?.deviceFraction ?? 0, 0.5);
    expect(r?.ramShare).toBe(0);
    expect(r?.remotePoolShare).toBe(1);
    expect(w).toHaveLength(1);
    // SSD and pool together: f_ram 0 given, f_ssd 0.25, f_pool 0.5 ⇒ nothing folds, shares 1/3 : 2/3.
    const both = resolveKvCache(
      { deviceFraction: 0.25, hostRamFraction: 0, remotePool: { fraction: 0.5 } },
      'unified',
      0,
      [],
    );
    expect(both?.deviceFraction).toBe(0.25);
    expectClose(both?.nvmeShare ?? 0, 1 / 3);
    expectClose(both?.remotePoolShare ?? 0, 2 / 3);
  });
});

describe('resolveRemotePool: the link one device reaches the pool over', () => {
  it('a stack without an inter-node link: InfiniBand NDR400 per device, with a warning; no capacity ⇒ Infinity', () => {
    const w: string[] = [];
    const r = resolveRemotePool({ fraction: 0.5 }, singleDevice(RTX_5090), w);
    expect(r.link).toEqual({
      name: IB_NDR_400.name,
      bandwidthPerDirectionBytesPerSec: B_POOL,
      latencySec: ALPHA_POOL,
    });
    expect(r.capacityBytes).toBe(Infinity);
    expect(w).toEqual([
      `offload: the stack has no inter-node link for the remote KV pool; using ${IB_NDR_400.name} per device`,
      'offload: the remote KV pool has no stated capacity; it is treated as unlimited',
    ]);
  });

  it("a multi-node stack: the inter-node link per device (the node's NICs shared: 4 × 400 Gb over 8 GPUs = 25 GB/s)", () => {
    const topo = nodeTopology({
      devicesPerNode: 8,
      nodes: 2,
      intraNode: NVLINK_4,
      interNode: IB_NDR_400,
      nicsPerNode: 4,
    });
    const w: string[] = [];
    const r = resolveRemotePool({ fraction: 0.5, capacityBytes: 2e12 }, topo, w);
    // Formula: B_dev = min(N_nic·B_nic/dpn, C) = 4·50e9/8 = 25e9
    expect(r.link.bandwidthPerDirectionBytesPerSec).toBe(25e9);
    expect(r.link.latencySec).toBe(IB_NDR_400.latencySec);
    expect(r.capacityBytes).toBe(2e12);
    expect(w).toEqual([]);
  });

  it('an explicit link wins over the stack', () => {
    const link = {
      name: 'RoCE 4 × 200 Gb',
      bandwidthPerDirectionBytesPerSec: 100e9,
      latencySec: 2e-6,
    };
    const r = resolveRemotePool({ fraction: 0.5, link }, singleDevice(RTX_5090), []);
    expect(r.link).toEqual(link);
  });

  it("Mooncake's measured per-node rates sit below the line rates this bound uses", () => {
    // 4 × 200 Gb = 100 GB/s, 8 × 400 Gb = 400 GB/s per node (line rate); measured 87 and 190 GB/s.
    expect(MOONCAKE_MEASURED_POOL_BYTES_PER_SEC_PER_NODE.roce_4x200g).toBeLessThan(4 * 25e9);
    expect(MOONCAKE_MEASURED_POOL_BYTES_PER_SEC_PER_NODE.rdma_8x400g).toBeLessThan(8 * B_POOL);
  });
});

describe('remotePoolTransferTime: α_pool + max(useful/B_pool, useful/B_link)', () => {
  it('NIC-bound on a dedicated device (B_pool < B_link); one α whatever the request count', () => {
    const path = {
      bandwidthPerDirectionBytesPerSec: 50e9,
      latencySec: 1e-6,
      link: { bandwidthPerDirectionBytesPerSec: B_LINK },
    };
    const span = remotePoolTransferTime([{ count: 8, bytesEach: 1e8 }], 8e8, path);
    const rows = remotePoolTransferTime([{ count: 8e8 / 4096, bytesEach: 4096 }], 8e8, path);
    expect(span.requests).toBe(8);
    expect(span.tierBytes).toBe(8e8);
    expectClose(span.tierSec, 8e8 / 50e9);
    expectClose(span.linkSec, 8e8 / B_LINK);
    expect(span.linkBytes).toBe(8e8);
    expect(span.serviceSec).toBe(span.tierSec);
    expect(span.sec).toBe(1e-6 + span.serviceSec);
    expect([span.littleSec, span.hostRamBytes, span.hostRamSec]).toEqual([0, 0, 0]);
    expect(span.iopsSec).toBeUndefined();
    expect(span.hostSec).toBeUndefined();
    // Scattered rows cost the same: no granularity, no per-request term.
    expect(rows.sec).toBe(span.sec);
  });

  it('PCIe-bound when the pool link is faster than the host link; no link term on unified memory', () => {
    const fast = remotePoolTransferTime([{ count: 1, bytesEach: 1e9 }], 1e9, {
      bandwidthPerDirectionBytesPerSec: 100e9,
      latencySec: 1e-6,
      link: { bandwidthPerDirectionBytesPerSec: B_LINK },
    });
    expectClose(fast.serviceSec, 1e9 / B_LINK);
    const unified = remotePoolTransferTime([{ count: 1, bytesEach: 1e9 }], 1e9, {
      bandwidthPerDirectionBytesPerSec: 100e9,
      latencySec: 1e-6,
    });
    expectClose(unified.serviceSec, 1e9 / 100e9);
    expect([unified.linkSec, unified.linkBytes]).toEqual([0, 0]);
  });
});

describe('validation of the remote pool', () => {
  const errorsOf = (kv: KvCacheOffloadConfig) => kvCacheErrors(kv, REF_DENSE_8B, true);
  it('fraction in [0, 1 − f_dev]; the shares sum to at most 1', () => {
    expect(errorsOf({ deviceFraction: 0.25, remotePool: { fraction: 0.75 } })).toEqual([]);
    expect(errorsOf({ deviceFraction: 0.25, remotePool: { fraction: 0.8 } })[0]).toMatch(
      /remotePool\.fraction.*\[0, 1 − deviceFraction\]/,
    );
    expect(
      errorsOf({ deviceFraction: 0.25, hostRamFraction: 0.5, remotePool: { fraction: 0.5 } })[0],
    ).toMatch(/exceed 1/);
    expect(
      errorsOf({ deviceFraction: 0.25, hostRamFraction: 0.25, remotePool: { fraction: 0.5 } }),
    ).toEqual([]);
  });
  it('a given link needs a positive bandwidth and a latency ≥ 0; a given capacity must be positive', () => {
    const bad = errorsOf({
      deviceFraction: 0.5,
      remotePool: {
        fraction: 0.5,
        link: { name: 'x', bandwidthPerDirectionBytesPerSec: 0, latencySec: -1 },
        capacityBytes: 0,
      },
    });
    expect(bad).toHaveLength(3);
    expect(bad[0]).toMatch(/bandwidthPerDirectionBytesPerSec/);
    expect(bad[1]).toMatch(/latencySec/);
    expect(bad[2]).toMatch(/capacityBytes/);
    expect(validateInput(input({ deviceFraction: 0.25, remotePool: { fraction: 0.75 } }))).toEqual(
      [],
    );
  });
});

describe('identity: a pool with no share changes nothing', () => {
  it('remotePool.fraction 0 next to a host-RAM placement gives the same result as without it', () => {
    const plain = run({ deviceFraction: 0.25 });
    const withPool = run({ deviceFraction: 0.25, remotePool: { fraction: 0 } });
    expect(withPool).toEqual(plain);
    // Nothing resolved: no pool warning, no pool tier.
    expect(withPool.warnings.some((w) => w.includes('remote KV pool'))).toBe(false);
    expect(withPool.memory.tiers.map((t) => t.tier)).toEqual(['host_ram']);
  });
});

describe('REF_DENSE_8B fp8 on one RTX 5090, b 8, decode @ 32k: KV cache 25 % on the GPU, 75 % in a remote pool over NDR400', () => {
  const pool = run({ deviceFraction: 0.25, remotePool: { fraction: 0.75 } });
  const ram = run({ deviceFraction: 0.25 });

  it('(a) the fetch ops read the pool: one span per sequence, 805,306,368 B per layer, q = 1', () => {
    const fetches = kvFetches(pool, 'decode');
    expect(fetches).toHaveLength(LAYERS);
    const useful = B * 0.75 * N * KVB;
    for (const o of fetches) {
      const op = o.op as TransferOp;
      expect(op).toMatchObject({ name: 'attn.kv_fetch_pool', source: 'remote_pool' });
      expect(op.reads).toEqual([{ count: B, bytesEach: 0.75 * N * KVB }]);
      expect(op.usefulBytes).toBe(useful);
      expect(op.prefetchedFraction).toBe(1);
      // Formula: t = α_pool + useful/B_pool   (B_pool 50 GB/s < B_link 57.6 GB/s: the NIC binds)
      expectClose(o.transferBandwidthSec, useful / B_POOL);
      expectClose(o.sec, ALPHA_POOL + useful / B_POOL);
      expect(o.transferResources).toMatchObject({
        linkBytes: useful,
        hostRamBytes: 0,
        hostRamSec: 0,
      });
      expectClose(o.transferResources?.linkSec ?? 0, useful / B_LINK);
    }
    expect(
      kvFetches(pool, 'decode').some(
        (o) => o.op.kind === 'transfer' && o.op.source !== 'remote_pool',
      ),
    ).toBe(false);
  });

  it('(b) decode is NIC-bound: 13.796 tok/s @ 32k, below the 15.897 tok/s @ 32k of host RAM over PCIe', () => {
    const useful = B * 0.75 * N * KVB;
    // Formula: tok/s = B / Σ_ℓ (α_pool + useful/B_pool)   (q = 1, transfer > compute: t_s = max(core, xfer) = xfer)
    const step = LAYERS * (ALPHA_POOL + useful / B_POOL);
    expect(pool.decode.stages[0].computeSec).toBeLessThan(step);
    expectClose(pool.decode.tokensPerSec, B / step, 1e-9);
    expectClose(pool.decode.tokensPerSec, 13.7962, 1e-4);
    expectClose(ram.decode.tokensPerSec, 15.8972, 1e-5);
    expect(pool.decode.tokensPerSec).toBeLessThan(ram.decode.tokensPerSec);
    expect(pool.warnings).toContain(
      `offload: the stack has no inter-node link for the remote KV pool; using ${IB_NDR_400.name} per device`,
    );
  });

  it('(c) memory: the pool holds the 75 % share for the whole deployment, unlimited unless a capacity is given', () => {
    const fullKv = B * LAYERS * (N + 1) * KVB;
    expect(pool.memory.fits).toBe(true);
    expectClose(pool.memory.perStage[0].sequenceStateBytes, 0.25 * fullKv);
    const [tier] = pool.memory.tiers;
    expect(tier.tier).toBe('remote_pool');
    expectClose(tier.breakdown.kvCache, 0.75 * fullKv);
    expect(tier.capacityBytes).toBe(Infinity);
    expect(tier.fits).toBe(true);
    // The device bound alone: 16 sequences (as with host RAM); no pool bound.
    expect(pool.memory.maxSequencesPerReplica).toBe(ram.memory.maxSequencesPerReplica);
    // Staging: the largest layer fetch, double-buffered (q = 1), as for host RAM.
    expect(pool.memory.perStage[0].stagingBytes).toBe(ram.memory.perStage[0].stagingBytes);
    // A 32 GB pool holds floor(32e9 / (0.75·36·32,769·4096)) = 8 sequences of this context: the fit is tight.
    const capped = run({
      deviceFraction: 0.25,
      remotePool: { fraction: 0.75, capacityBytes: 32e9 },
    });
    const perSeq = 0.75 * LAYERS * (N + 1) * KVB;
    expect(capped.memory.maxSequencesPerReplica).toBe(Math.floor(32e9 / perSeq));
    expect(capped.memory.tiers[0]).toMatchObject({
      tier: 'remote_pool',
      capacityBytes: 32e9,
      fits: true,
    });
    const tooSmall = run({
      deviceFraction: 0.25,
      remotePool: { fraction: 0.75, capacityBytes: 20e9 },
    });
    expect(tooSmall.memory.fits).toBe(false);
    expect(tooSmall.memory.tiers[0].fits).toBe(false);
    expect(tooSmall.memory.maxSequencesPerReplica).toBe(Math.floor(20e9 / perSeq));
  });

  it('(d) the pool is one for every node: a 2-node dp 16 B200 run sums all 16 devices into its entry', () => {
    const topo = nodeTopology({
      devicesPerNode: 8,
      nodes: 2,
      device: B200,
      intraNode: NVLINK_4,
      interNode: IB_NDR_400,
    });
    const r = simulate({
      model: REF_DENSE_8B,
      quant: FP8_W,
      parallel: { tp: 1, pp: 1, dp: 16, ep: 1 },
      workload: { batch: 16 * 8, promptLen: N, decodeContextLen: N },
      hardware: {
        device: B200,
        topology: topo,
        host: { link: hostPcieLink(5, 16), ram: { ...DDR5_6000_2CH, capacityBytes: 1e12 } },
      },
      deployment: {
        offload: {
          kvCache: { deviceFraction: 0.25, hostRamFraction: 0.25, remotePool: { fraction: 0.5 } },
        },
      },
    });
    const byTier = Object.fromEntries(r.memory.tiers.map((t) => [t.tier, t.bytes]));
    const perDeviceKv = B * LAYERS * (N + 1) * KVB;
    // Host RAM: the fullest node (8 devices × 25 %); the pool: all 16 devices × 50 %.
    expectClose(byTier.host_ram, 8 * 0.25 * perDeviceKv);
    expectClose(byTier.remote_pool, 16 * 0.5 * perDeviceKv);
    // The pool link is the stack's NDR400 per device (one NIC per GPU): each layer's pool fetch at 50 GB/s.
    const poolFetch = kvFetches(r, 'decode').find(
      (o) => o.op.kind === 'transfer' && o.op.source === 'remote_pool',
    );
    const useful = B * 0.5 * N * KVB;
    expectClose(poolFetch?.transferBandwidthSec ?? 0, useful / B_POOL);
    expect(r.warnings.some((w) => w.includes('no inter-node link for the remote KV pool'))).toBe(
      false,
    );
  });
});

describe('a pool stream next to a host-RAM stream: the overlapped stage transfer', () => {
  it('with prefetching both streams run together: xfer_s = max(Σ_ram, Σ_pool, Σ link, Σ host RAM) < the serial sum', () => {
    // f_dev 0.25, f_ram 0.25, f_pool 0.5: per layer 268,435,456 B over PCIe (57.6 GB/s) and 536,870,912 B over the NIC
    // (50 GB/s); both also cross the GPU's PCIe link: the link term Σ q·linkSec = 805,306,368 / B_link binds.
    const split = run({
      deviceFraction: 0.25,
      hostRamFraction: 0.25,
      remotePool: { fraction: 0.5 },
    });
    const ops = split.decode.stages[0].layers.flatMap((l) => l.ops);
    const transfers = ops.filter(
      (o): o is OpResult & { op: TransferOp } => o.op.kind === 'transfer',
    );
    expect(prefetchesFromMultipleTiers(transfers.map((o) => o.op))).toBe(true);
    const ramSec = transfers
      .filter((o) => o.op.source === 'host_ram')
      .reduce((s, o) => s + o.sec, 0);
    const poolSec = transfers
      .filter((o) => o.op.source === 'remote_pool')
      .reduce((s, o) => s + o.sec, 0);
    const linkSec = transfers.reduce((s, o) => s + (o.transferResources?.linkSec ?? 0), 0);
    const hostSec = transfers.reduce((s, o) => s + (o.transferResources?.hostRamSec ?? 0), 0);
    const overlapped = Math.max(ramSec, poolSec, linkSec, hostSec);
    expect(overlapped).toBe(linkSec);
    expectClose(linkSec, (LAYERS * B * 0.75 * N * KVB) / B_LINK);
    // Formula: t_s = lat + max(core, xfer)  (q = 1), xfer = D + P = 0 + overlapped
    expectClose(
      split.decode.tokensPerSec,
      B / Math.max(split.decode.stages[0].computeSec, overlapped),
      1e-9,
    );
    expect(overlapped).toBeLessThan(ramSec + poolSec);
    // Staging: the layer being read (Σ both) + one fill per source = 2·Σ.
    expect(split.memory.perStage[0].stagingBytes).toBe(2 * B * 0.75 * N * KVB);
    // p = 0: serial sum, single buffer.
    const serial = run(
      { deviceFraction: 0.25, hostRamFraction: 0.25, remotePool: { fraction: 0.5 } },
      { prefetchAccuracy: 0 },
    );
    expect(serial.memory.perStage[0].stagingBytes).toBe(B * 0.75 * N * KVB);
    const serialXfer = serial.decode.stages[0].layers
      .flatMap((l) => l.ops)
      .filter((o) => o.op.kind === 'transfer')
      .reduce((s, o) => s + o.sec, 0);
    expectClose(
      serial.decode.tokensPerSec,
      B / (serial.decode.stages[0].computeSec + serialXfer),
      1e-9,
    );
  });
});

describe('prefix-cache hits served from the pool at prefill', () => {
  it('a cached prefix wholly in the pool is fetched once per layer at prefill from the pool; decode is untouched', () => {
    // P 32768, prefix hit 0.5 → 16,384 cached tokens per sequence, f_pre 0 → all fetched from the pool.
    const workload = { batch: 4, promptLen: N, prefixCacheHitRatio: 0.5 };
    const kv: KvCacheOffloadConfig = {
      deviceFraction: 1,
      prefixDeviceFraction: 0,
      remotePool: { fraction: 0 },
    };
    const off = run(undefined, { workload });
    const hidden = run(kv, { workload });
    const exposed = run(kv, { workload, prefetchAccuracy: 0 });
    const fetches = kvFetches(hidden, 'prefill');
    expect(fetches).toHaveLength(LAYERS);
    expect(fetches[0].op).toMatchObject({ source: 'remote_pool', name: 'attn.kv_fetch_pool' });
    expect((fetches[0].op as TransferOp).reads).toEqual([{ count: 4, bytesEach: 16384 * KVB }]);
    expect(kvFetches(hidden, 'decode')).toEqual([]);
    expect(hidden.decode.tokensPerSec).toBe(off.decode.tokensPerSec);
    expect(hidden.prefill.tokensPerSec).toBe(off.prefill.tokensPerSec);
    // p = 0: the cycle grows by exactly Σ_ℓ (α_pool + useful/B_pool).
    const xfer = exposed.prefill.transfer?.sec ?? 0;
    expectClose(xfer, LAYERS * (ALPHA_POOL + (4 * 16384 * KVB) / B_POOL));
    expectClose(exposed.prefill.cycleSec, off.prefill.cycleSec + xfer, 1e-12);
    // The same prefix in host RAM crosses PCIe instead: α differs, the bandwidth term too.
    const ramXfer =
      run({ deviceFraction: 1, prefixDeviceFraction: 0 }, { workload, prefetchAccuracy: 0 }).prefill
        .transfer?.sec ?? 0;
    expectClose(ramXfer, LAYERS * (ALPHA_RAM + (4 * 16384 * KVB) / B_LINK));
    expect(hidden.memory.tiers).toEqual([]);
  });
});
