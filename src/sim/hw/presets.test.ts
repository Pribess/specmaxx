import { describe, expect, it } from 'vitest';
import {
  SYNTH_CLUSTER,
  SYNTH_GPU,
  SYNTH_PCIE_HOSTBOUNCE,
  SYNTH_UNIFIED,
} from '../__fixtures__/synthetic';
import { simulate } from '../engine/simulate';
import type { SimInput } from '../engine/types';
import type { ParallelConfig, QuantConfig } from '../ir/model';
import { REF_DENSE_8B } from '../models/reference';
import {
  AMD_DEVICE_PRESETS,
  APPLE_DEVICE_PRESETS,
  DEVICE_PRESET_GROUPS,
  DEVICE_PRESETS,
  DEVICE_UNLOCKS,
  HOST_BOUNCE_PCIE_LINKS,
  INTEL_DEVICE_PRESETS,
  LEGACY_LINK_IDS,
  LINK_PRESETS,
  NVIDIA_CMP_DEVICE_PRESETS,
  NVIDIA_CONSUMER_DEVICE_PRESETS,
  NVIDIA_PRO_DEVICE_PRESETS,
  NVIDIA_DATACENTER_DEVICE_PRESETS,
  RTX_5090,
  TOPOLOGY_PRESETS,
  deviceUnlocks,
  linkPresetId,
  singleDevice,
} from './presets';
import { allUnlockSelections, resolveUnlocks, unlockAxesErrors } from './unlocks';
import {
  numericLeafPaths,
  validateDevice,
  validateDeviceModifiers,
  validateLink,
  validateLinkModifiers,
  validateTopology,
} from './validate';
import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { DeviceSpec } from './types';

function peak(
  d: DeviceSpec,
  input: MathFormat,
  acc: AccumulateType,
  sparsity: 'dense' | '2:4',
): number {
  const p = d.engines[0].matmulPeaks.find(
    (x) => x.input === input && x.accumulate === acc && x.sparsity === sparsity,
  );
  if (p === undefined) throw new Error(`no entry ${input}/${acc}/${sparsity}`);
  return p.flopsPerSec;
}

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

describe('preset validation (T16)', () => {
  it('every preset passes validateDevice', () => {
    for (const d of DEVICE_PRESETS) expect(validateDevice(d)).toEqual([]);
    expect(validateTopology(singleDevice(RTX_5090))).toEqual([]);
  });

  it('RTX_5090: every 2:4 peak is 2 × its dense peak', () => {
    for (const p of RTX_5090.engines[0].matmulPeaks.filter((x) => x.sparsity === '2:4')) {
      // rel 1e-3: Table 3 rounds to 0.1 TFLOPS, so TF32 reads 104.8 / 209.5 (exact 104.75 / 209.5).
      expectClose(p.flopsPerSec, 2 * peak(RTX_5090, p.input, p.accumulate, 'dense'), 1e-3);
    }
  });

  it('RTX_5090 GeForce rates: fp16-acc fp16 = 2 × fp32-acc fp16; fp8 fp32-acc = 2 × bf16; fp4 = 2 × fp8 fp32-acc', () => {
    expect(peak(RTX_5090, 'fp16', 'fp16', 'dense')).toBe(
      2 * peak(RTX_5090, 'fp16', 'fp32', 'dense'),
    ); // 419 = 2·209.5
    expect(peak(RTX_5090, 'fp8', 'fp32', 'dense')).toBe(
      2 * peak(RTX_5090, 'bf16', 'fp32', 'dense'),
    ); // 419 = 2·209.5
    expect(peak(RTX_5090, 'fp4', 'fp32', 'dense')).toBe(4 * peak(RTX_5090, 'fp8', 'fp32', 'dense')); // 1676 = 4·419
    expect(peak(RTX_5090, 'fp4', 'fp32', '2:4')).toBe(3352e12); // the "3352 AI TOPS" headline
  });

  it('RTX_5090 memory and L2 as quoted', () => {
    expect(RTX_5090.memory.bandwidthBytesPerSec).toBe((512 * 28e9) / 8); // 512-bit × 28 Gbps = 1792 GB/s
    expect(RTX_5090.memory.capacityOptionsBytes).toEqual([32e9]);
    expect(RTX_5090.l2Bytes).toBe(100_663_296); // 98,304 KB × 1024
    expect(RTX_5090.engines[0].computeUnits).toBe(170);
  });

  it('fixtures validate too', () => {
    expect(validateDevice(SYNTH_GPU)).toEqual([]);
    expect(validateDevice(SYNTH_UNIFIED)).toEqual([]);
    expect(validateTopology(SYNTH_CLUSTER)).toEqual([]);
    expect(validateTopology(SYNTH_PCIE_HOSTBOUNCE)).toEqual([]);
  });
});

describe('validate rules', () => {
  it('numericLeafPaths skips provenance keys', () => {
    const paths = numericLeafPaths(RTX_5090).map((l) => l.path);
    expect(paths).toContain('engines.0.matmulPeaks.0.flopsPerSec');
    expect(paths).toContain('memory.capacityOptionsBytes.0');
    expect(paths.some((p) => p.startsWith('sources') || p.startsWith('estimated'))).toBe(false);
  });

  it('an uncovered number is an error', () => {
    const d: DeviceSpec = {
      ...SYNTH_GPU,
      sources: [{ url: 'synthetic://test', title: 't', fields: ['engines'] }],
    };
    expect(
      validateDevice(d).some((e) => e.includes('memory.bandwidthBytesPerSec is not covered')),
    ).toBe(true);
  });

  it('sparse below dense, missing fp32-acc entry, misaligned tile, empty capacity, duplicate engine id', () => {
    const e = SYNTH_GPU.engines[0];
    const bad: DeviceSpec = {
      ...SYNTH_GPU,
      engines: [
        {
          ...e,
          matmulPeaks: [
            { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: 100e12 },
            { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: 50e12 },
            { input: 'fp8', accumulate: 'fp16', sparsity: 'dense', flopsPerSec: 200e12 },
          ],
          gemmTiling: { default: { ...e.gemmTiling.default, ctaTiles: [{ m: 100, n: 128 }] } },
        },
        e,
      ],
      memory: { ...SYNTH_GPU.memory, capacityOptionsBytes: [] },
    };
    const errors = validateDevice(bad);
    expect(errors.some((x) => x.includes('below its dense'))).toBe(true);
    expect(errors.some((x) => x.includes('fp8 has peaks but no fp32-accumulate'))).toBe(true);
    expect(errors.some((x) => x.includes('not a positive multiple of mmaM'))).toBe(true);
    expect(errors.some((x) => x.includes('capacityOptionsBytes is empty'))).toBe(true);
    expect(errors.some((x) => x.includes("duplicate engine id 'gpu'"))).toBe(true);
  });

  it('hostBounce iff host_bounce', () => {
    const link = SYNTH_PCIE_HOSTBOUNCE.intraNode;
    if (link === undefined) throw new Error('fixture has an intraNode link');
    expect(validateLink(link)).toEqual([]);
    expect(
      validateLink({ ...link, p2p: 'direct' }).some((x) =>
        x.includes('hostBounce must be present iff'),
      ),
    ).toBe(true);
  });
});

/** The library's device ids, for readable failure messages (`it.each` titles). */
const DEVICE_CASES = DEVICE_PRESETS.map((d) => [d.id, d] as const);

describe('preset library: every device preset', () => {
  it('DEVICE_PRESETS holds every vendor array entry exactly once, in the groups order', () => {
    const vendorArrays = [
      NVIDIA_CONSUMER_DEVICE_PRESETS,
      NVIDIA_PRO_DEVICE_PRESETS,
      NVIDIA_CMP_DEVICE_PRESETS,
      NVIDIA_DATACENTER_DEVICE_PRESETS,
      APPLE_DEVICE_PRESETS,
      AMD_DEVICE_PRESETS,
      INTEL_DEVICE_PRESETS,
    ];
    const fromVendors = vendorArrays.flat();
    expect(DEVICE_PRESETS.length).toBe(fromVendors.length);
    expect(new Set(DEVICE_PRESETS)).toEqual(new Set(fromVendors));
    expect(DEVICE_PRESETS).toEqual(DEVICE_PRESET_GROUPS.flatMap((g) => g.devices));
    expect(DEVICE_PRESET_GROUPS.every((g) => g.devices.length > 0)).toBe(true);
    // Datacenter last (consumer / prosumer first).
    expect(DEVICE_PRESET_GROUPS[DEVICE_PRESET_GROUPS.length - 1].name).toBe('Datacenter');
  });

  it('orders every group newest generation first, then by tier', () => {
    const ids = (name: string) =>
      DEVICE_PRESET_GROUPS.find((g) => g.name === name)?.devices.map((d) => d.id);
    expect(ids('NVIDIA GeForce')).toEqual(['rtx-5090', 'rtx-4090', 'rtx-3090']);
    expect(ids('Apple silicon')).toEqual([
      'apple-m5-ultra-80c',
      'apple-m5-ultra-64c',
      'apple-m5-max-40c',
      'apple-m5-max-32c',
      'apple-m4-max-40c',
      'apple-m4-max-32c',
      'apple-m3-ultra-80c',
      'apple-m3-ultra-60c',
      'apple-m3-max-40c',
      'apple-m3-max-30c',
    ]);
    expect(ids('AMD Radeon / Ryzen AI')).toEqual([
      'radeon-ai-pro-r9700',
      'radeon-rx-9070-xt',
      'ryzen-ai-max-plus-395',
    ]);
    expect(ids('Intel Arc')).toEqual(['arc-pro-b70', 'arc-pro-b65', 'arc-pro-b60', 'arc-b580']);
    expect(ids('Datacenter')).toEqual([
      'gb300',
      'b300',
      'gb200',
      'b200',
      'h200-sxm',
      'h100-sxm',
      'a100-sxm4-80gb',
      'a100-sxm4-40gb',
      'a100-pcie-80gb',
      'a100-pcie-40gb',
      'v100-sxm2',
      'v100-pcie',
      'instinct-mi455x',
      'instinct-mi355x',
      'instinct-mi325x',
      'instinct-mi300x',
    ]);
  });

  it('device ids are unique', () => {
    const ids = DEVICE_PRESETS.map((d) => d.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it.each(DEVICE_CASES)('%s passes validateDevice', (_id, d) => {
    expect(validateDevice(d)).toEqual([]);
  });

  it.each(DEVICE_CASES)('%s: sources non-empty, each with an https url and a title', (_id, d) => {
    expect(d.sources.length).toBeGreaterThan(0);
    for (const s of d.sources) {
      expect(s.url.startsWith('https://'), s.url).toBe(true);
      expect(s.title.trim()).not.toBe('');
    }
  });

  it.each(DEVICE_CASES)(
    '%s: every matmul peak > 0 and every 2:4 peak ≥ its dense peak',
    (_id, d) => {
      for (const e of d.engines) {
        expect(e.matmulPeaks.length).toBeGreaterThan(0);
        for (const p of e.matmulPeaks) {
          expect(p.flopsPerSec).toBeGreaterThan(0);
          if (p.sparsity !== '2:4') continue;
          const dense = e.matmulPeaks.find(
            (x) => x.input === p.input && x.accumulate === p.accumulate && x.sparsity === 'dense',
          );
          expect(dense, `${e.id} ${p.input}/${p.accumulate} 2:4 has a dense twin`).toBeDefined();
          expect(p.flopsPerSec).toBeGreaterThanOrEqual(dense?.flopsPerSec ?? Infinity);
        }
      }
    },
  );

  it('device names identify the exact SKU: unique, and Apple names carry the GPU-core binning', () => {
    const names = DEVICE_PRESETS.map((d) => d.name);
    expect(new Set(names).size).toBe(names.length);
    for (const d of DEVICE_PRESETS.filter((x) => x.vendor === 'Apple'))
      expect(d.name).toMatch(/\(\d+-core GPU\)$/);
  });
});

describe('preset library: links and topologies', () => {
  it('topology ids are unique', () => {
    const ids = TOPOLOGY_PRESETS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it.each(TOPOLOGY_PRESETS.map((t) => [t.id, t] as const))(
    '%s passes validateTopology',
    (_id, t) => {
      expect(validateTopology(t)).toEqual([]);
    },
  );

  it('the fabric link fields are set where researched, cited or marked estimated, and change no existing number', () => {
    const covered = (l: (typeof LINK_PRESETS)[number], field: string) =>
      l.sources.some((s) => s.fields.includes(field)) || l.estimated.some((e) => e.field === field);
    for (const l of LINK_PRESETS) {
      if (l.allPortsCapPerDirectionBytesPerSec !== undefined)
        expect(covered(l, 'allPortsCapPerDirectionBytesPerSec'), l.name).toBe(true);
      if (l.forwardingLatencySec !== undefined) {
        expect(covered(l, 'forwardingLatencySec'), l.name).toBe(true);
        // Every relay latency is an estimate (no vendor publishes one), and is RELAY HANDLING ONLY: the engine already
        // charges the per-hop α for every hop crossed, so the reason must say so.
        expect(l.estimated.find((e) => e.field === 'forwardingLatencySec')?.reason, l.name).toMatch(
          /^ESTIMATED, relay handling only/,
        );
      }
    }
    const byId = (id: string) => LINK_PRESETS.find((l) => linkPresetId(l) === id)!;
    // DGX Spark: both QSFP ports share the 25 GB/s host attachment; a host relay adds ~4.5 µs on top of each hop's α.
    const cx7 = byId('dgx-spark-connectx-7-200gbe-roce');
    expect(cx7.bandwidthPerDirectionBytesPerSec).toBe(25e9);
    expect(cx7.allPortsCapPerDirectionBytesPerSec).toBe(25e9);
    expect(cx7.forwardingLatencySec).toBe(4.5e-6);
    expect(cx7.latencySec).toBe(1.5e-6);
    // Thunderbolt 5 RDMA: ports add up (no cap), 3 µs relay handling, α 7 µs = half the measured 13.72 µs
    // ibv_uc_pingpong round trip (the one α definition; the former 50 µs was an all-reduce upper bound).
    const tb5 = byId('thunderbolt-5-rdma-macos-26-2');
    expect(tb5.allPortsCapPerDirectionBytesPerSec).toBeUndefined();
    expect(tb5.forwardingLatencySec).toBe(3e-6);
    expect(tb5.latencySec).toBe(7e-6);
    // Thunderbolt bridge over TCP: TB5 at the 80 Gb/s line rate, TB4 at 40 Gb/s, the same software latency and relay.
    const tb5Tcp = byId('thunderbolt-5-bridge-tcp');
    const tb4Tcp = byId('thunderbolt-4-bridge-tcp');
    expect(tb5Tcp.bandwidthPerDirectionBytesPerSec).toBe(10e9);
    expect(tb4Tcp.bandwidthPerDirectionBytesPerSec).toBe(5e9);
    expect(tb5Tcp.latencySec).toBe(tb4Tcp.latencySec);
    expect(tb5Tcp.forwardingLatencySec).toBe(20e-6);
    expect(tb4Tcp.forwardingLatencySec).toBe(20e-6);
    expect(byId('strix-halo-usb4-native-rdma-20-gb-s-negotiated').forwardingLatencySec).toBe(3e-6);
    expect(byId('strix-halo-usb4-tcp-20-gb-s-negotiated').forwardingLatencySec).toBe(5e-6);
    // Switched-only links carry neither field.
    for (const id of [
      'pcie-5-0-x16-host-bounce',
      'nvlink5-nvswitch',
      'infiniband-ndr400-one-nic-per-gpu',
      '10gbe-tcp',
    ]) {
      expect(byId(id).allPortsCapPerDirectionBytesPerSec, id).toBeUndefined();
      expect(byId(id).forwardingLatencySec, id).toBeUndefined();
    }
  });

  it('relay latency is additive to the per-hop α: physical 2- and 3-hop examples (lat(h) = h·α + (h−1)·α_f)', () => {
    const byId = (id: string) => LINK_PRESETS.find((l) => linkPresetId(l) === id)!;
    const lat = (l: (typeof LINK_PRESETS)[number], hops: number) =>
      hops * l.latencySec + (hops - 1) * (l.forwardingLatencySec ?? 0);
    const us = (s: number) => Math.round(s * 1e8) / 100;
    // DGX Spark ring: the opposite Spark is 2 hops away. 2 × 1.5 + 4.5 = 7.5 µs, i.e. +6 µs over the 1-hop 1.5 µs: the
    // measured 16.9 − 10.9 µs one-shot all-reduce difference (dgx-spark-networking). 3 hops (a chain of 4): 13.5 µs.
    const cx7 = byId('dgx-spark-connectx-7-200gbe-roce');
    expect(us(lat(cx7, 1))).toBe(1.5);
    expect(us(lat(cx7, 2))).toBe(7.5);
    expect(us(lat(cx7, 2) - lat(cx7, 1))).toBe(6);
    expect(us(lat(cx7, 3))).toBe(13.5);
    // Thunderbolt 5 RDMA (α 7 µs, half the 13.72 µs ibv_uc_pingpong RTT): 2 hops 17 µs, 3 hops 27 µs.
    const tb5 = byId('thunderbolt-5-rdma-macos-26-2');
    expect(us(lat(tb5, 2))).toBe(17);
    expect(us(lat(tb5, 3))).toBe(27);
    // Thunderbolt bridge (TCP, α 45 µs): 2 hops 110 µs, 3 hops 175 µs. USB4 RDMA: 19 / 30 µs. USB4 TCP (α 9 µs): 23 / 37 µs.
    expect([2, 3].map((h) => us(lat(byId('thunderbolt-5-bridge-tcp'), h)))).toEqual([110, 175]);
    expect([2, 3].map((h) => us(lat(byId('thunderbolt-4-bridge-tcp'), h)))).toEqual([110, 175]);
    expect(
      [2, 3].map((h) => us(lat(byId('strix-halo-usb4-native-rdma-20-gb-s-negotiated'), h))),
    ).toEqual([19, 30]);
    expect([2, 3].map((h) => us(lat(byId('strix-halo-usb4-tcp-20-gb-s-negotiated'), h)))).toEqual([
      23, 37,
    ]);
    // The relay handling is always smaller than the hop it sits between: it excludes the wire.
    for (const l of LINK_PRESETS)
      if (l.forwardingLatencySec !== undefined)
        expect(l.forwardingLatencySec, l.name).toBeLessThan(l.latencySec * 4);
  });

  it('no link preset has been renamed (ids are URL state): LEGACY_LINK_IDS is empty and every id is a current one', () => {
    const ids = new Set(LINK_PRESETS.map(linkPresetId));
    for (const [from, to] of Object.entries(LEGACY_LINK_IDS)) {
      expect(ids.has(from), from).toBe(false);
      expect(ids.has(to), to).toBe(true);
    }
    expect(ids).toEqual(
      new Set([
        'pcie-3-0-x16-direct-p2p',
        'pcie-3-0-x16-host-bounce',
        'nvlink3-nvswitch',
        'a100-nvlink-bridge-2-gpus',
        'nvlink2-hybrid-cube-mesh-8-gpus',
        'nvlink2-fully-connected-mesh-4-gpus',
        'nvlink2-peer-pair-2-links',
        'infiniband-hdr200-one-nic-per-gpu',
        'pcie-4-0-x16-host-bounce',
        'pcie-4-0-x8-host-bounce',
        'pcie-5-0-x16-host-bounce',
        'pcie-5-0-x8-host-bounce',
        'pcie-4-0-x16-direct-p2p',
        'pcie-4-0-x8-direct-p2p',
        'pcie-5-0-x16-direct-p2p',
        'pcie-5-0-x8-direct-p2p',
        'rtx-3090-nvlink-bridge',
        'nvlink4-nvswitch',
        'nvlink5-nvswitch',
        'nvlink5-nvl72-nvlink-switch',
        'nvlink5-nvl4-bridge-mesh-estimate',
        'nvlink-c2c-grace-cpu-gpu',
        'mi325x-xgmi-one-peer',
        'mi325x-xgmi-8-gpu-aggregate-injection',
        'mi355x-xgmi-one-peer',
        'mi355x-xgmi-8-gpu-aggregate-injection',
        'mi455x-ualoe-helios-scale-up',
        'infiniband-ndr400-one-nic-per-gpu',
        'infiniband-xdr800-one-nic-per-gpu',
        'dgx-spark-connectx-7-200gbe-roce',
        'mi300x-xgmi-one-peer',
        'mi300x-xgmi-8-gpu-aggregate-injection',
        'thunderbolt-5-rdma-macos-26-2',
        'thunderbolt-5-bridge-tcp',
        'thunderbolt-4-bridge-tcp',
        'pcie-2-0-x16-host-bounce',
        'pcie-1-0-x16-host-bounce',
        'pcie-2-0-x4-host-bounce',
        'pcie-1-0-x4-host-bounce',
        'thunderbolt-4-bridge-tcp-20-gb-s-negotiated',
        'strix-halo-usb4-native-rdma-20-gb-s-negotiated',
        'strix-halo-usb4-tcp-20-gb-s-negotiated',
        '5gbe-tcp',
        '10gbe-tcp',
      ]),
    );
  });

  it('every link preset passes validateLink, and every single-device topology validates', () => {
    for (const l of LINK_PRESETS) expect(validateLink(l)).toEqual([]);
    for (const d of DEVICE_PRESETS) expect(validateTopology(singleDevice(d))).toEqual([]);
  });
});

describe('preset library: unlock axes', () => {
  it('DEVICE_UNLOCKS keys are preset device ids; deviceUnlocks is empty for a device without unlocks', () => {
    const ids = new Set(DEVICE_PRESETS.map((d) => d.id));
    for (const id of Object.keys(DEVICE_UNLOCKS)) expect(ids.has(id), id).toBe(true);
    expect(deviceUnlocks(RTX_5090)).toBe(DEVICE_UNLOCKS[RTX_5090.id]);
    expect(deviceUnlocks({ id: 'no-such-device' })).toEqual([]);
  });

  const withUnlocks = DEVICE_PRESETS.filter((d) => deviceUnlocks(d).length > 0).map(
    (d) => [d.id, d] as const,
  );

  it('GeForce, RTX PRO 6000 and CMP 170HX cards have unlock axes', () => {
    expect(withUnlocks.map(([id]) => id).sort()).toEqual([
      'cmp-170hx',
      'rtx-3090',
      'rtx-4090',
      'rtx-5090',
      'rtx-pro-6000-max-q',
      'rtx-pro-6000-ws',
    ]);
  });

  it.each(withUnlocks)(
    '%s: axes are well-formed and EVERY selection validates (device + each host-bounce PCIe link)',
    (_id, d) => {
      const axes = deviceUnlocks(d);
      expect(unlockAxesErrors(axes)).toEqual([]);
      const selections = allUnlockSelections(axes);
      expect(selections.length).toBe(axes.reduce((n, a) => n * (a.options.length + 1), 1));
      for (const selection of selections) {
        const resolved = resolveUnlocks(axes, selection);
        const label = JSON.stringify(selection);
        expect(validateDeviceModifiers(d, resolved.deviceModifiers), label).toEqual([]);
        for (const link of HOST_BOUNCE_PCIE_LINKS) {
          expect(
            validateLinkModifiers(link, resolved.intraNodeLinkModifiers),
            `${label} on ${link.name}`,
          ).toEqual([]);
        }
      }
    },
  );
});

describe('preset library: simulate() smoke test (REF_DENSE_8B, bf16, batch 1, prompt 4096, ctx 1024)', () => {
  const BF16: QuantConfig = {
    weights: { dtype: 'bf16' },
    activations: 'bf16',
    kvCache: { dtype: 'bf16' },
  };
  const P1: ParallelConfig = { tp: 1, pp: 1, dp: 1, ep: 1 };

  function smokeInput(d: DeviceSpec, engineId: string): SimInput {
    return {
      model: REF_DENSE_8B,
      quant: BF16,
      parallel: P1,
      workload: { batch: 1, promptLen: 4096, decodeContextLen: 1024 },
      hardware: { device: d, topology: singleDevice(d), engineId },
    };
  }

  // Every engine of every device (e.g. Apple M5's Neural Accelerators and its SIMD path).
  const cases = DEVICE_PRESETS.flatMap((d) =>
    d.engines.map((e) => [`${d.id}/${e.id}`, d, e.id] as const),
  );

  it.each(cases)(
    '%s yields finite positive prefill / decode tokens/s and a memory-fit verdict',
    (_label, d, engineId) => {
      const r = simulate(smokeInput(d, engineId));
      for (const tps of [r.prefill.tokensPerSec, r.decode.tokensPerSec]) {
        expect(Number.isFinite(tps)).toBe(true);
        expect(tps).toBeGreaterThan(0);
      }
      expect(typeof r.memory.fits).toBe('boolean');
      // Decode cannot beat the memory roofline: the bf16 GEMM weights alone, 15,136,194,560 B (e2e.test.ts T9),
      // stream once per token, so tokens/s ≤ memory bandwidth / 15e9 B.
      expect(r.decode.tokensPerSec).toBeLessThanOrEqual(d.memory.bandwidthBytesPerSec / 15e9);
    },
  );

  it('the fully unlocked RTX 4090 (every axis on) simulates', () => {
    const d = DEVICE_PRESETS.find((x) => x.id === 'rtx-4090');
    if (d === undefined) throw new Error('rtx-4090 preset missing');
    const axes = deviceUnlocks(d);
    const all = Object.fromEntries(axes.map((a) => [a.id, a.options[0].id]));
    const { deviceModifiers } = resolveUnlocks(axes, all);
    const base = simulate(smokeInput(d, 'gpu'));
    const r = simulate({
      ...smokeInput(d, 'gpu'),
      hardware: { ...smokeInput(d, 'gpu').hardware, deviceModifiers },
    });
    expect(r.decode.tokensPerSec).toBeGreaterThan(base.decode.tokensPerSec); // memory overclock
    expect(r.prefill.tokensPerSec).toBeGreaterThan(base.prefill.tokensPerSec); // bf16 tensor ×2
  });
});

describe('SFU rates (sfuOpsPerSec, docs §4 "SFU rate")', () => {
  /**
   * Devices whose vector FP32 peak is issue-throttled while the SFU is not, so the lane bound is checked against the
   * unthrottled FP32 rate (vector FP32 × divider). CMP 170HX: FP32 FMA fused to 1/32, MUFU exp2 not (arabel1a timings).
   */
  const FP32_THROTTLE_DIVIDER: Record<string, number> = { 'cmp-170hx': 32 };
  for (const d of DEVICE_PRESETS) {
    it(`${d.id}: every engine sets one, covered, between 1/64 and 1/2 of the vector FP32 FLOP/s`, () => {
      d.engines.forEach((e, i) => {
        const sfu = e.sfuOpsPerSec;
        expect(sfu, e.id).toBeGreaterThan(0);
        const field = `engines.${i}.sfuOpsPerSec`;
        const covered =
          d.sources.some((s) => s.fields.includes(field)) ||
          d.estimated.some((x) => x.field === field);
        expect(covered, field).toBe(true);
        // At most one exp per FP32 lane per clock (vector FP32 counts 2 FLOP per lane per clock), at least 1/32 lane.
        const vec = e.vectorFlopsPerSec.fp32! * (FP32_THROTTLE_DIVIDER[d.id] ?? 1);
        expect(sfu!, e.id).toBeLessThanOrEqual(vec / 2);
        expect(sfu!, e.id).toBeGreaterThanOrEqual(vec / 64);
      });
    });
  }
  it('B300 = 2 × B200 per SM per clock (Blackwell Ultra doubles the SFU exp rate)', () => {
    const b200 = DEVICE_PRESETS.find((d) => d.id === 'b200')!.engines[0];
    const b300 = DEVICE_PRESETS.find((d) => d.id === 'b300')!.engines[0];
    expect(b300.sfuOpsPerSec! / b200.sfuOpsPerSec!).toBeCloseTo(2, 12);
  });
  it('H100 = 132 SMs × 16 × the 67 TFLOP/s vector clock = 67e12 / 16', () => {
    const h100 = DEVICE_PRESETS.find((d) => d.id === 'h100-sxm')!.engines[0];
    expect(h100.sfuOpsPerSec! / (67e12 / 16)).toBeCloseTo(1, 12);
  });
  it('Apple: the GEMV row limit is set on every engine (NAX falls back to the SIMD ALUs)', () => {
    for (const d of APPLE_DEVICE_PRESETS)
      for (const e of d.engines) expect(e.gemvMaxRows, `${d.id} ${e.id}`).toBe(1);
    for (const d of [
      ...NVIDIA_CONSUMER_DEVICE_PRESETS,
      ...NVIDIA_CMP_DEVICE_PRESETS,
      ...NVIDIA_DATACENTER_DEVICE_PRESETS,
      ...AMD_DEVICE_PRESETS,
      ...INTEL_DEVICE_PRESETS,
    ]) {
      for (const e of d.engines) expect(e.gemvMaxRows, d.id).toBeUndefined();
    }
  });
});
