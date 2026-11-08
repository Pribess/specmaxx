import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_PCIE_HOSTBOUNCE, SYNTH_UNIFIED } from '../__fixtures__/synthetic';
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
  LINK_PRESETS,
  NVIDIA_CONSUMER_DEVICE_PRESETS,
  NVIDIA_DATACENTER_DEVICE_PRESETS,
  RTX_5090,
  TOPOLOGY_PRESETS,
  deviceUnlocks,
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

function peak(d: DeviceSpec, input: MathFormat, acc: AccumulateType, sparsity: 'dense' | '2:4'): number {
  const p = d.engines[0].matmulPeaks.find((x) => x.input === input && x.accumulate === acc && x.sparsity === sparsity);
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
    expect(peak(RTX_5090, 'fp16', 'fp16', 'dense')).toBe(2 * peak(RTX_5090, 'fp16', 'fp32', 'dense')); // 419 = 2·209.5
    expect(peak(RTX_5090, 'fp8', 'fp32', 'dense')).toBe(2 * peak(RTX_5090, 'bf16', 'fp32', 'dense')); // 419 = 2·209.5
    expect(peak(RTX_5090, 'fp4', 'fp32', 'dense')).toBe(4 * peak(RTX_5090, 'fp8', 'fp32', 'dense')); // 1676 = 4·419
    expect(peak(RTX_5090, 'fp4', 'fp32', '2:4')).toBe(3352e12); // the "3352 AI TOPS" headline
  });

  it('RTX_5090 memory and L2 as quoted', () => {
    expect(RTX_5090.memory.bandwidthBytesPerSec).toBe(512 * 28e9 / 8); // 512-bit × 28 Gbps = 1792 GB/s
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
    const d: DeviceSpec = { ...SYNTH_GPU, sources: [{ url: 'synthetic://test', title: 't', fields: ['engines'] }] };
    expect(validateDevice(d).some((e) => e.includes('memory.bandwidthBytesPerSec is not covered'))).toBe(true);
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
    expect(validateLink({ ...link, p2p: 'direct' }).some((x) => x.includes('hostBounce must be present iff'))).toBe(true);
  });
});

/** The library's device ids, for readable failure messages (`it.each` titles). */
const DEVICE_CASES = DEVICE_PRESETS.map((d) => [d.id, d] as const);

describe('preset library: every device preset', () => {
  it('DEVICE_PRESETS holds every vendor array entry exactly once, in the groups order', () => {
    const vendorArrays = [
      NVIDIA_CONSUMER_DEVICE_PRESETS,
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
    const ids = (name: string) => DEVICE_PRESET_GROUPS.find((g) => g.name === name)?.devices.map((d) => d.id);
    expect(ids('NVIDIA GeForce')).toEqual(['rtx-5090', 'rtx-4090', 'rtx-3090']);
    expect(ids('Apple silicon')).toEqual([
      'apple-m5-ultra-80c', 'apple-m5-ultra-64c', 'apple-m5-max-40c', 'apple-m5-max-32c',
      'apple-m4-max-40c', 'apple-m4-max-32c', 'apple-m3-ultra-80c', 'apple-m3-ultra-60c',
    ]);
    expect(ids('AMD Radeon / Ryzen AI')).toEqual(['radeon-ai-pro-r9700', 'radeon-rx-9070-xt', 'ryzen-ai-max-plus-395']);
    expect(ids('Intel Arc')).toEqual(['arc-pro-b70', 'arc-pro-b65', 'arc-pro-b60', 'arc-b580']);
    expect(ids('Datacenter')).toEqual([
      'b300', 'b200', 'h200-sxm', 'h100-sxm', 'instinct-mi455x', 'instinct-mi355x', 'instinct-mi325x', 'instinct-mi300x',
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

  it.each(DEVICE_CASES)('%s: every matmul peak > 0 and every 2:4 peak ≥ its dense peak', (_id, d) => {
    for (const e of d.engines) {
      expect(e.matmulPeaks.length).toBeGreaterThan(0);
      for (const p of e.matmulPeaks) {
        expect(p.flopsPerSec).toBeGreaterThan(0);
        if (p.sparsity !== '2:4') continue;
        const dense = e.matmulPeaks.find((x) => x.input === p.input && x.accumulate === p.accumulate && x.sparsity === 'dense');
        expect(dense, `${e.id} ${p.input}/${p.accumulate} 2:4 has a dense twin`).toBeDefined();
        expect(p.flopsPerSec).toBeGreaterThanOrEqual(dense?.flopsPerSec ?? Infinity);
      }
    }
  });
});

describe('preset library: links and topologies', () => {
  it('topology ids are unique', () => {
    const ids = TOPOLOGY_PRESETS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it.each(TOPOLOGY_PRESETS.map((t) => [t.id, t] as const))('%s passes validateTopology', (_id, t) => {
    expect(validateTopology(t)).toEqual([]);
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

  const withUnlocks = DEVICE_PRESETS.filter((d) => deviceUnlocks(d).length > 0).map((d) => [d.id, d] as const);

  it('the GeForce cards have unlock axes', () => {
    expect(withUnlocks.map(([id]) => id).sort()).toEqual(['rtx-3090', 'rtx-4090', 'rtx-5090']);
  });

  it.each(withUnlocks)('%s: axes are well-formed and EVERY selection validates (device + each host-bounce PCIe link)', (_id, d) => {
    const axes = deviceUnlocks(d);
    expect(unlockAxesErrors(axes)).toEqual([]);
    const selections = allUnlockSelections(axes);
    expect(selections.length).toBe(axes.reduce((n, a) => n * (a.options.length + 1), 1));
    for (const selection of selections) {
      const resolved = resolveUnlocks(axes, selection);
      const label = JSON.stringify(selection);
      expect(validateDeviceModifiers(d, resolved.deviceModifiers), label).toEqual([]);
      for (const link of HOST_BOUNCE_PCIE_LINKS) {
        expect(validateLinkModifiers(link, resolved.intraNodeLinkModifiers), `${label} on ${link.name}`).toEqual([]);
      }
    }
  });
});

describe('preset library: simulate() smoke test (REF_DENSE_8B, bf16, batch 1, prompt 4096, ctx 1024)', () => {
  const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };
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
  const cases = DEVICE_PRESETS.flatMap((d) => d.engines.map((e) => [`${d.id}/${e.id}`, d, e.id] as const));

  it.each(cases)('%s yields finite positive prefill / decode tokens/s and a memory-fit verdict', (_label, d, engineId) => {
    const r = simulate(smokeInput(d, engineId));
    for (const tps of [r.prefill.tokensPerSec, r.decode.tokensPerSec]) {
      expect(Number.isFinite(tps)).toBe(true);
      expect(tps).toBeGreaterThan(0);
    }
    expect(typeof r.memory.fits).toBe('boolean');
    // Decode cannot beat the memory roofline: the bf16 GEMM weights alone, 15,136,194,560 B (e2e.test.ts T9),
    // stream once per token, so tokens/s ≤ memory bandwidth / 15e9 B.
    expect(r.decode.tokensPerSec).toBeLessThanOrEqual(d.memory.bandwidthBytesPerSec / 15e9);
  });

  it('the fully unlocked RTX 4090 (every axis on) simulates', () => {
    const d = DEVICE_PRESETS.find((x) => x.id === 'rtx-4090');
    if (d === undefined) throw new Error('rtx-4090 preset missing');
    const axes = deviceUnlocks(d);
    const all = Object.fromEntries(axes.map((a) => [a.id, a.options[0].id]));
    const { deviceModifiers } = resolveUnlocks(axes, all);
    const base = simulate(smokeInput(d, 'gpu'));
    const r = simulate({ ...smokeInput(d, 'gpu'), hardware: { ...smokeInput(d, 'gpu').hardware, deviceModifiers } });
    expect(r.decode.tokensPerSec).toBeGreaterThan(base.decode.tokensPerSec); // memory overclock
    expect(r.prefill.tokensPerSec).toBeGreaterThan(base.prefill.tokensPerSec); // bf16 tensor ×2
  });
});
