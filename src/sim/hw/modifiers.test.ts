import { describe, expect, it } from 'vitest';
import { SYNTH_PCIE_HOSTBOUNCE_LINK } from '../__fixtures__/synthetic';
import { effectiveLink, ringAllReduceSec } from '../engine/collectives';
import { applyDeviceModifiers, applyLinkModifiers, describeDeviceModifier, resolveDeviceVariant } from './modifiers';
import type { DeviceModifier, LinkModifier, ModifierProvenance } from './modifiers';
import { RTX_5090 } from './presets/nvidia';
import { validateDevice, validateDeviceModifiers, validateLinkModifiers } from './validate';
import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { DeviceSpec } from './types';
import { resolveHost } from '../engine/host';
import type { OffloadTier } from '../engine/types';
import { hostPcieLink } from './presets/links';
import { singleDevice } from './presets/index';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** Synthetic provenance: these tests are about mechanics, not real unlocks. */
const PROV: ModifierProvenance = {
  note: 'synthetic test modifier',
  sources: [{ url: 'synthetic://test', title: 'Synthetic', fields: ['engines.0.matmulPeaks'] }],
};

function peak(d: DeviceSpec, input: MathFormat, acc: AccumulateType, sparsity: 'dense' | '2:4'): number {
  const p = d.engines[0].matmulPeaks.find((x) => x.input === input && x.accumulate === acc && x.sparsity === sparsity);
  if (p === undefined) throw new Error(`no entry ${input}/${acc}/${sparsity}`);
  return p.flopsPerSec;
}

const BF16_FP32_X2: DeviceModifier = { kind: 'peakScale', input: 'bf16', accumulate: 'fp32', factor: 2, ...PROV };
const CLOCK_08: DeviceModifier = { kind: 'clockScale', factor: 0.8, ...PROV, note: '450 W power limit' };

describe('device modifiers (T19)', () => {
  it('peakScale bf16/fp32 ×2 scales only the matching entries', () => {
    const d = applyDeviceModifiers(RTX_5090, [BF16_FP32_X2]);
    expect(peak(d, 'bf16', 'fp32', 'dense')).toBe(419e12); // 209.5e12 · 2
    expect(peak(d, 'bf16', 'fp32', '2:4')).toBe(838e12); // 419e12 · 2
    expect(peak(d, 'fp16', 'fp32', 'dense')).toBe(peak(RTX_5090, 'fp16', 'fp32', 'dense')); // unchanged 209.5e12
    expect(peak(d, 'fp16', 'fp16', 'dense')).toBe(peak(RTX_5090, 'fp16', 'fp16', 'dense')); // unchanged 419e12
  });

  it('memoryCapacity replaces the options', () => {
    const d = applyDeviceModifiers(RTX_5090, [{ kind: 'memoryCapacity', capacityOptionsBytes: [48e9], ...PROV }]);
    expect(d.memory.capacityOptionsBytes).toEqual([48e9]);
  });

  it('memoryBandwidth replaces the bandwidth', () => {
    const d = applyDeviceModifiers(RTX_5090, [{ kind: 'memoryBandwidth', bandwidthBytesPerSec: 2.0e12, ...PROV }]);
    expect(d.memory.bandwidthBytesPerSec).toBe(2.0e12);
  });

  it('clockScale 0.8 scales matmul and vector peaks, not memory bandwidth', () => {
    const d = applyDeviceModifiers(RTX_5090, [CLOCK_08]);
    expectClose(peak(d, 'bf16', 'fp32', 'dense'), 209.5e12 * 0.8); // 167.6e12
    expectClose(d.engines[0].vectorFlopsPerSec.fp32 ?? NaN, 104.8e12 * 0.8); // 83.84e12
    expect(d.memory.bandwidthBytesPerSec).toBe(RTX_5090.memory.bandwidthBytesPerSec); // 1.792e12
  });

  it('clockScale 0.8 scales the SFU rate with the vector peaks (same clock), and its provenance covers it', () => {
    const base = RTX_5090.engines[0].sfuOpsPerSec ?? NaN;
    const d = applyDeviceModifiers(RTX_5090, [CLOCK_08]);
    expectClose(d.engines[0].sfuOpsPerSec ?? NaN, base * 0.8);
    // An estimated clock scale records an estimate on every path it changes, the SFU rate included.
    const est = applyDeviceModifiers(RTX_5090, [{ kind: 'clockScale', factor: 0.8, note: 'sustained clock', sources: [], estimatedReason: 'test' }]);
    expect(est.estimated.some((x) => x.field === 'engines.0.sfuOpsPerSec')).toBe(true);
    // An engine without the field stays without it.
    const { sfuOpsPerSec: _sfu, ...engine } = RTX_5090.engines[0];
    void _sfu;
    const bare = applyDeviceModifiers({ ...RTX_5090, engines: [engine] }, [CLOCK_08]);
    expect(bare.engines[0].sfuOpsPerSec).toBeUndefined();
  });

  it('composition: clockScale 0.8 then peakScale ×2 → 335.2e12', () => {
    const d = applyDeviceModifiers(RTX_5090, [CLOCK_08, BF16_FP32_X2]);
    expectClose(peak(d, 'bf16', 'fp32', 'dense'), 209.5e12 * 0.8 * 2); // 335.2e12
  });

  it('is pure: the base is unchanged; provenance recorded; result validates', () => {
    const before = JSON.stringify(RTX_5090);
    const mods: DeviceModifier[] = [CLOCK_08, BF16_FP32_X2, { kind: 'memoryCapacity', capacityOptionsBytes: [48e9], ...PROV }];
    const d = applyDeviceModifiers(RTX_5090, mods);
    expect(JSON.stringify(RTX_5090)).toBe(before);
    expect(RTX_5090.appliedModifiers).toBeUndefined();
    expect(d.appliedModifiers).toHaveLength(3);
    expect(d.sources.length).toBe(RTX_5090.sources.length + 3);
    expect(validateDevice(d)).toEqual([]);
    expect(validateDeviceModifiers(RTX_5090, mods)).toEqual([]);
    // Mutating the result does not reach the base.
    d.engines[0].matmulPeaks[0].flopsPerSec = 0;
    expect(JSON.stringify(RTX_5090)).toBe(before);
  });

  it('estimatedReason becomes estimated[] entries on the covered paths', () => {
    const mod: DeviceModifier = { kind: 'memoryBandwidth', bandwidthBytesPerSec: 2e12, note: 'OC', sources: [], estimatedReason: 'forum report' };
    const d = applyDeviceModifiers(RTX_5090, [mod]);
    expect(d.estimated.at(-1)?.field).toBe('memory.bandwidthBytesPerSec');
  });

  it('resolveDeviceVariant takes id/name from the variant', () => {
    const d = resolveDeviceVariant({ id: 'rtx-5090-450w', name: 'RTX 5090 @ 450 W', base: RTX_5090, modifiers: [CLOCK_08] });
    expect(d.id).toBe('rtx-5090-450w');
    expect(d.name).toBe('RTX 5090 @ 450 W');
    expect(describeDeviceModifier(CLOCK_08)).toBe('variant: clockScale all engines ×0.8 (450 W power limit)');
  });

  it('errors: factor 0, unknown engineId, a filter matching nothing, missing provenance', () => {
    const cases: [DeviceModifier, string][] = [
      [{ kind: 'clockScale', factor: 0, ...PROV }, 'factor'],
      [{ kind: 'peakScale', engineId: 'npu', factor: 2, ...PROV }, "unknown engineId 'npu'"],
      [{ kind: 'peakScale', input: 'int4', factor: 2, ...PROV }, 'matches no matmul peak'],
      [{ kind: 'clockScale', factor: 0.9, note: 'x', sources: [] }, 'needs sources or an estimatedReason'],
    ];
    for (const [mod, message] of cases) {
      const errors = validateDeviceModifiers(RTX_5090, [mod]);
      expect(errors.some((e) => e.includes(message))).toBe(true);
      expect(() => applyDeviceModifiers(RTX_5090, [mod])).toThrow(message);
    }
  });
});

describe('device modifiers: vectorScale, computeUnits, hostInterface', () => {
  it('vectorScale fp32 ×32 scales only the fp32 vector peak (not matmul, other vector dtypes or the SFU)', () => {
    const d = applyDeviceModifiers(RTX_5090, [{ kind: 'vectorScale', dtype: 'fp32', factor: 32, ...PROV }]);
    const [e, base] = [d.engines[0], RTX_5090.engines[0]];
    expect(e.vectorFlopsPerSec.fp32).toBe(104.8e12 * 32);
    expect(e.vectorFlopsPerSec.fp16).toBe(base.vectorFlopsPerSec.fp16); // 104.8e12
    expect(e.matmulPeaks).toEqual(base.matmulPeaks);
    expect(e.sfuOpsPerSec).toBe(base.sfuOpsPerSec);
    // No dtype: every listed vector peak.
    const all = applyDeviceModifiers(RTX_5090, [{ kind: 'vectorScale', factor: 2, ...PROV }]);
    for (const k of ['fp64', 'fp32', 'fp16', 'bf16'] as const) {
      expect(all.engines[0].vectorFlopsPerSec[k]).toBe(base.vectorFlopsPerSec[k]! * 2);
    }
  });

  it('vectorScale estimates cover only the scaled vector path', () => {
    const d = applyDeviceModifiers(RTX_5090, [
      { kind: 'vectorScale', dtype: 'fp64', factor: 2, note: 'fp64 unlock', sources: [], estimatedReason: 'test' },
    ]);
    expect(d.estimated.filter((x) => x.reason.startsWith('fp64 unlock')).map((x) => x.field)).toEqual([
      'engines.0.vectorFlopsPerSec.fp64',
    ]);
  });

  it('computeUnits 170 → 85 halves every peak and the SFU rate and sets the unit count', () => {
    const d = applyDeviceModifiers(RTX_5090, [{ kind: 'computeUnits', computeUnits: 85, ...PROV }]);
    const [e, base] = [d.engines[0], RTX_5090.engines[0]];
    expect(e.computeUnits).toBe(85);
    expectClose(peak(d, 'bf16', 'fp32', 'dense'), 209.5e12 / 2);
    expectClose(e.vectorFlopsPerSec.fp32 ?? NaN, 104.8e12 / 2);
    expectClose(e.sfuOpsPerSec ?? NaN, base.sfuOpsPerSec! / 2);
    expect(d.memory.bandwidthBytesPerSec).toBe(RTX_5090.memory.bandwidthBytesPerSec);
    expect(describeDeviceModifier({ kind: 'computeUnits', computeUnits: 85, ...PROV })).toBe(
      'variant: computeUnits all engines → 85 (synthetic test modifier)',
    );
    expect(validateDevice(d)).toEqual([]);
  });
  it('hostInterface: each given field replaces only itself, so two modifiers combine (Gen2, then x8 → Gen2 x8)', () => {
    const gen2: DeviceModifier = { kind: 'hostInterface', generation: 2, ...PROV };
    const x8: DeviceModifier = { kind: 'hostInterface', lanes: 8, ...PROV };
    expect(applyDeviceModifiers(RTX_5090, [gen2]).hostInterface).toEqual({ generation: 2, lanes: 16 });
    expect(applyDeviceModifiers(RTX_5090, [x8]).hostInterface).toEqual({ generation: 5, lanes: 8 });
    expect(applyDeviceModifiers(RTX_5090, [gen2, x8]).hostInterface).toEqual({ generation: 2, lanes: 8 });
    expect(applyDeviceModifiers(RTX_5090, [x8, gen2]).hostInterface).toEqual({ generation: 2, lanes: 8 });
    expect(RTX_5090.hostInterface).toEqual({ generation: 5, lanes: 16 }); // the base is not mutated
    expect(describeDeviceModifier({ ...gen2, lanes: 16, note: 'unlock' })).toBe('variant: hostInterface Gen2 x16 (unlock)');
  });

  it('hostInterface: the modified interface is the host link simulate uses (offload streams over Gen2 x16, not Gen5 x16)', () => {
    const est = { sources: [], estimatedReason: 'test' };
    const d = applyDeviceModifiers(RTX_5090, [{ kind: 'hostInterface', generation: 2, note: 'Gen2', ...est }]);
    expect(d.estimated.filter((x) => x.reason.startsWith('Gen2')).map((x) => x.field)).toEqual(['hostInterface.generation']);
    expect(validateDevice(d)).toEqual([]);
    const tiers = new Set<OffloadTier>(['host_ram']);
    expect(resolveHost({ device: d, topology: singleDevice(d) }, d, tiers).link).toEqual(hostPcieLink(2, 16));
    expect(resolveHost({ device: RTX_5090, topology: singleDevice(RTX_5090) }, RTX_5090, tiers).link).toEqual(hostPcieLink(5, 16));
  });

  it('errors: computeUnits not an integer ≥ 1, vectorScale on an unlisted dtype, hostInterface without fields or host interface', () => {
    const { hostInterface: _h, ...noInterface } = RTX_5090;
    void _h;
    const cases: [DeviceSpec, DeviceModifier, string][] = [
      [RTX_5090, { kind: 'computeUnits', computeUnits: 0, ...PROV }, 'integer ≥ 1'],
      [RTX_5090, { kind: 'computeUnits', computeUnits: 70.5, ...PROV }, 'integer ≥ 1'],
      [RTX_5090, { kind: 'computeUnits', engineId: 'npu', computeUnits: 70, ...PROV }, "unknown engineId 'npu'"],
      [{ ...RTX_5090, engines: [{ ...RTX_5090.engines[0], vectorFlopsPerSec: { fp32: 1e12 } }] }, { kind: 'vectorScale', dtype: 'fp64', factor: 2, ...PROV }, 'matches no listed vector peak'],
      [RTX_5090, { kind: 'vectorScale', factor: -1, ...PROV }, 'factor'],
      [RTX_5090, { kind: 'hostInterface', ...PROV }, 'give generation, lanes or both'],
      [RTX_5090, { kind: 'hostInterface', generation: 6 as 5, ...PROV }, 'generation must be 1, 2, 3, 4 or 5'],
      [RTX_5090, { kind: 'hostInterface', lanes: 2 as 4, ...PROV }, 'lanes must be 4, 8 or 16'],
      [noInterface, { kind: 'hostInterface', lanes: 16, ...PROV }, 'no hostInterface to change'],
    ];
    for (const [base, mod, message] of cases) {
      expect(validateDeviceModifiers(base, [mod]).some((e) => e.includes(message)), message).toBe(true);
      expect(() => applyDeviceModifiers(base, [mod])).toThrow(message);
    }
  });
});

describe('link modifiers (T19)', () => {
  const P2P_ON: LinkModifier = {
    kind: 'p2p',
    p2p: 'direct',
    note: 'P2P-enabled driver',
    sources: [{ url: 'synthetic://test', title: 'Synthetic', fields: ['p2p'] }],
  };

  it("p2p 'direct' drops hostBounce; ring AR n2 S 1e8 = 2·1·2e-6 + 1e8/64e9", () => {
    const link = applyLinkModifiers(SYNTH_PCIE_HOSTBOUNCE_LINK, [P2P_ON]);
    expect(link.p2p).toBe('direct');
    expect(link.hostBounce).toBeUndefined();
    expect(SYNTH_PCIE_HOSTBOUNCE_LINK.hostBounce).toBeDefined(); // base untouched
    expect(link.appliedModifiers).toHaveLength(1);
    expect(validateLinkModifiers(SYNTH_PCIE_HOSTBOUNCE_LINK, [P2P_ON])).toEqual([]);
    // Ring: 2(n−1)·α + 2(n−1)/n · S/B = 4e-6 + 1e8/64e9 = 4e-6 + 1.5625e-3
    const t = ringAllReduceSec(1e8, 2, effectiveLink(link, 2));
    expectClose(t.latencySec, 4e-6);
    expectClose(t.bandwidthSec, 1.5625e-3);
  });

  it("linkBandwidth replaces the per-direction bandwidth; 'host_bounce' without hostBounce is an error", () => {
    const link = applyLinkModifiers(SYNTH_PCIE_HOSTBOUNCE_LINK, [
      { kind: 'linkBandwidth', bandwidthPerDirectionBytesPerSec: 32e9, note: 'x8 slot', sources: [], estimatedReason: 'PCIe 5.0 x8 = half of x16' },
    ]);
    expect(link.bandwidthPerDirectionBytesPerSec).toBe(32e9);
    expect(link.estimated.at(-1)?.field).toBe('bandwidthPerDirectionBytesPerSec');
    const bad: LinkModifier = { kind: 'p2p', p2p: 'host_bounce', ...PROV };
    expect(validateLinkModifiers(SYNTH_PCIE_HOSTBOUNCE_LINK, [bad]).some((e) => e.includes('requires hostBounce'))).toBe(true);
  });
});
