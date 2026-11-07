import { describe, expect, it } from 'vitest';
import { SYNTH_PCIE_HOSTBOUNCE_LINK } from '../__fixtures__/synthetic';
import { effectiveLink, ringAllReduceSec } from '../engine/collectives';
import { applyDeviceModifiers, applyLinkModifiers, describeDeviceModifier, resolveDeviceVariant } from './modifiers';
import type { DeviceModifier, LinkModifier, ModifierProvenance } from './modifiers';
import { RTX_5090 } from './presets/nvidia';
import { validateDevice, validateDeviceModifiers, validateLinkModifiers } from './validate';
import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { DeviceSpec } from './types';

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
