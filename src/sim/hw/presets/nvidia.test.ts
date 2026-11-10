import { describe, expect, it } from 'vitest';
import type { AccumulateType, MathFormat } from '../../ir/dtype';
import { applyDeviceModifiers, applyLinkModifiers } from '../modifiers';
import type { DeviceSpec, LinkSpec } from '../types';
import { allUnlockSelections, resolveUnlocks, unlockAxesErrors } from '../unlocks';
import {
  validateDevice,
  validateDeviceModifiers,
  validateLink,
  validateLinkModifiers,
} from '../validate';
import {
  NVIDIA_CONSUMER_DEVICE_PRESETS,
  RTX_3090,
  RTX_3090_UNLOCKS,
  RTX_4090,
  RTX_4090_UNLOCKS,
  RTX_5090,
  RTX_5090_UNLOCKS,
} from './nvidia';

function peak(
  device: DeviceSpec,
  input: MathFormat,
  accumulate: AccumulateType,
  sparsity: 'dense' | '2:4' = 'dense',
): number {
  const value = device.engines[0].matmulPeaks.find(
    (p) => p.input === input && p.accumulate === accumulate && p.sparsity === sparsity,
  );
  if (value === undefined)
    throw new Error(`Missing ${device.id} ${input}/${accumulate}/${sparsity}`);
  return value.flopsPerSec;
}

// Synthetic local fixture: validation tests must not depend on a sibling's topology implementation.
const HOST_BOUNCE_PCIE: LinkSpec = {
  kind: 'pcie',
  name: 'Synthetic host-bounce PCIe fixture',
  bandwidthPerDirectionBytesPerSec: 32e9,
  latencySec: 2e-6,
  p2p: 'host_bounce',
  hostBounce: { hostMemBandwidthBytesPerSec: 100e9, extraLatencySec: 5e-6 },
  sources: [
    {
      url: 'synthetic://pcie-fixture',
      title: 'Synthetic test values, not product specs',
      fields: ['bandwidthPerDirectionBytesPerSec', 'latencySec', 'hostBounce'],
    },
  ],
  estimated: [],
};

const CASES = [
  { device: RTX_3090, axes: RTX_3090_UNLOCKS, selections: 4 },
  { device: RTX_4090, axes: RTX_4090_UNLOCKS, selections: 16 },
  { device: RTX_5090, axes: RTX_5090_UNLOCKS, selections: 6 },
];

describe('NVIDIA consumer vendor tables', () => {
  it('all devices validate with every numeric field covered by provenance', () => {
    expect(NVIDIA_CONSUMER_DEVICE_PRESETS).toEqual([RTX_5090, RTX_4090, RTX_3090]); // newest generation first
    for (const device of NVIDIA_CONSUMER_DEVICE_PRESETS) expect(validateDevice(device)).toEqual([]);
  });

  it('2:4 doubles dense and GeForce FP16 accumulation doubles FP32 accumulation (table rounding)', () => {
    for (const device of NVIDIA_CONSUMER_DEVICE_PRESETS) {
      for (const sparse of device.engines[0].matmulPeaks.filter((p) => p.sparsity === '2:4')) {
        const expected = 2 * peak(device, sparse.input, sparse.accumulate);
        // Ampere TF32 35.6/71 is rounded to whole TFLOPS; maximum relative discrepancy is 0.3%.
        expect(Math.abs(sparse.flopsPerSec / expected - 1)).toBeLessThan(0.003);
      }
      for (const sparsity of ['dense', '2:4'] as const) {
        const ratio =
          peak(device, 'fp16', 'fp16', sparsity) / peak(device, 'fp16', 'fp32', sparsity);
        // Ada table's 330.3 vs 2×165.2 differs by 0.1 TFLOPS, deliberately not silently corrected.
        expect(Math.abs(ratio / 2 - 1)).toBeLessThan(0.001);
      }
    }
  });

  it('3090 Table 9 spot values: 82 SM, 71 BF16, 1136 sparse INT4, 936 GB/s, 6 MiB L2', () => {
    expect(RTX_3090.engines[0].computeUnits).toBe(82);
    expect(peak(RTX_3090, 'bf16', 'fp32')).toBe(71e12);
    expect(peak(RTX_3090, 'int4', 'int32', '2:4')).toBe(1136e12);
    expect(RTX_3090.memory.capacityOptionsBytes).toEqual([24e9]);
    expect(RTX_3090.memory.bandwidthBytesPerSec).toBe(936e9);
    expect(RTX_3090.l2Bytes).toBe(6 * 1024 ** 2);
    expect(RTX_3090.engines[0].gemmTiling.int4?.mmaK).toBe(64);
    expect(RTX_3090.engines[0].matmulPeaks.some((p) => p.input === 'fp8')).toBe(false);
  });

  it('4090 corrected Table 2: 128 SM, 165.2 BF16, 330.3 FP8/FP32, 1008 GB/s, 72 MiB L2', () => {
    expect(RTX_4090.engines[0].computeUnits).toBe(128);
    expect(peak(RTX_4090, 'bf16', 'fp32')).toBe(165.2e12);
    expect(peak(RTX_4090, 'fp8', 'fp32')).toBe(330.3e12);
    expect(peak(RTX_4090, 'int4', 'int32', '2:4')).toBe(2642.4e12);
    expect(RTX_4090.memory.bandwidthBytesPerSec).toBe(1008e9);
    expect(RTX_4090.l2Bytes).toBe(72 * 1024 ** 2);
    expect(RTX_4090.engines[0].gemmTiling.fp8?.mmaK).toBe(32);
  });

  it('5090 Table 3: 209.5 BF16, sparse FP4 headline 3352, 1792 GB/s, 96 MiB L2', () => {
    expect(peak(RTX_5090, 'bf16', 'fp32')).toBe(209.5e12);
    expect(peak(RTX_5090, 'fp4', 'fp32', '2:4')).toBe(3352e12);
    expect(RTX_5090.memory.bandwidthBytesPerSec).toBe(1792e9);
    expect(RTX_5090.l2Bytes).toBe(96 * 1024 ** 2);
    expect(
      RTX_5090.engines[0].matmulPeaks.some((p) => p.input === 'int4' || p.input === 'fp6'),
    ).toBe(false);
    expect(RTX_5090.engines[0].gemmTiling.fp4?.mmaK).toBe(64);
  });
});

describe('independent NVIDIA consumer unlock axes', () => {
  it('every stock/option combination validates against base device and host-bounce PCIe', () => {
    expect(validateLink(HOST_BOUNCE_PCIE)).toEqual([]);
    for (const { device, axes, selections } of CASES) {
      expect(unlockAxesErrors(axes)).toEqual([]);
      const combinations = allUnlockSelections(axes);
      expect(combinations).toHaveLength(selections);
      for (const selection of combinations) {
        const resolved = resolveUnlocks(axes, selection);
        expect(validateDeviceModifiers(device, resolved.deviceModifiers)).toEqual([]);
        expect(validateLinkModifiers(HOST_BOUNCE_PCIE, resolved.intraNodeLinkModifiers)).toEqual(
          [],
        );
        expect(validateDevice(applyDeviceModifiers(device, resolved.deviceModifiers))).toEqual([]);
        expect(
          validateLink(applyLinkModifiers(HOST_BOUNCE_PCIE, resolved.intraNodeLinkModifiers)),
        ).toEqual([]);
      }
    }
  });

  it('tensor unlock only doubles FP16/BF16/TF32 with FP32 accumulation', () => {
    for (const { device, axes } of CASES.slice(0, 2)) {
      const mods = resolveUnlocks(axes, { 'tensor-rate': 'full-rate' }).deviceModifiers;
      const unlocked = applyDeviceModifiers(device, mods);
      for (const base of device.engines[0].matmulPeaks) {
        const scale =
          base.accumulate === 'fp32' && ['fp16', 'bf16', 'tf32'].includes(base.input) ? 2 : 1;
        expect(peak(unlocked, base.input, base.accumulate, base.sparsity)).toBe(
          scale * base.flopsPerSec,
        );
      }
      expect(unlocked.memory).toEqual(device.memory);
      expect(unlocked.engines[0].vectorFlopsPerSec).toEqual(device.engines[0].vectorFlopsPerSec);
    }
    expect(RTX_5090_UNLOCKS.some((a) => a.kind === 'tensorRate')).toBe(false);
  });

  it('capacity and sample memory overclock are independent; P2P drops host staging without altering link peak', () => {
    const vram = applyDeviceModifiers(
      RTX_4090,
      resolveUnlocks(RTX_4090_UNLOCKS, { 'vram-mod': '48gb' }).deviceModifiers,
    );
    expect(vram.memory.capacityOptionsBytes).toEqual([48e9]);
    expect(vram.memory.bandwidthBytesPerSec).toBe(1008e9);
    const combined = resolveUnlocks(RTX_4090_UNLOCKS, {
      'vram-mod': '48gb',
      'memory-overclock': '1545mhz',
      'pcie-p2p': 'direct',
    });
    const modified = applyDeviceModifiers(RTX_4090, combined.deviceModifiers);
    expect(modified.memory.capacityOptionsBytes).toEqual([48e9]);
    expect(modified.memory.bandwidthBytesPerSec).toBeCloseTo(1186.56e9, -2);
    const direct = applyLinkModifiers(HOST_BOUNCE_PCIE, combined.intraNodeLinkModifiers);
    expect(direct.p2p).toBe('direct');
    expect(direct.hostBounce).toBeUndefined();
    expect(direct.bandwidthPerDirectionBytesPerSec).toBe(
      HOST_BOUNCE_PCIE.bandwidthPerDirectionBytesPerSec,
    );
    expect(direct.latencySec).toBe(HOST_BOUNCE_PCIE.latencySec);
  });

  it('measured sustained clocks scale vendor peaks from their 2407 MHz boost baseline', () => {
    for (const [option, measuredClockMhz] of [
      ['575w', 2768],
      ['400w', 2465],
    ] as const) {
      const mods = resolveUnlocks(RTX_5090_UNLOCKS, { 'power-clock': option }).deviceModifiers;
      const measured = applyDeviceModifiers(RTX_5090, mods);
      const factor = measuredClockMhz / 2407;
      expect(peak(measured, 'bf16', 'fp32')).toBe(209.5e12 * factor);
      expect(measured.memory).toEqual(RTX_5090.memory);
      expect(measured.engines[0].vectorFlopsPerSec.fp32).toBe(104.8e12 * factor);
      expect(factor).toBeGreaterThan(1);
    }
  });

  it('all options retain UI notes and non-vendor evidence classes', () => {
    for (const { axes } of CASES)
      for (const axis of axes) {
        expect(axis.description.length).toBeGreaterThan(50);
        for (const option of axis.options)
          for (const modifier of [
            ...option.deviceModifiers,
            ...(option.intraNodeLinkModifiers ?? []),
          ]) {
            expect(modifier.note.length).toBeGreaterThan(20);
            expect(modifier.sources.length).toBeGreaterThan(0);
            for (const source of modifier.sources) {
              expect(['community', 'review']).toContain(source.kind);
              expect(source.note?.length).toBeGreaterThan(20);
            }
          }
      }
  });
});
