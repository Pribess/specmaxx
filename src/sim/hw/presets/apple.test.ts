import { describe, expect, it } from 'vitest';
import type { DeviceSpec } from '../types';
import { gb, gbPerSec, tbPerSec } from '../units';
import { validateDevice } from '../validate';
import {
  APPLE_DEVICE_PRESETS, M3_ULTRA_60C, M3_ULTRA_80C, M4_MAX_32C, M4_MAX_40C,
  M5_MAX_32C, M5_MAX_40C, M5_ULTRA_64C, M5_ULTRA_80C,
} from './apple';

// Capacities and bandwidth as quoted in Apple Mac Studio tech specs (122211 / 128107).
const skuFacts: [DeviceSpec, number, number[], number][] = [
  [M4_MAX_32C, 32, [36], gbPerSec(410)],
  [M4_MAX_40C, 40, [48, 64, 128], gbPerSec(546)],
  [M3_ULTRA_60C, 60, [96, 256], gbPerSec(819)],
  [M3_ULTRA_80C, 80, [96, 256, 512], gbPerSec(819)],
  [M5_MAX_32C, 32, [36], gbPerSec(460)],
  [M5_MAX_40C, 40, [48, 64, 128], gbPerSec(614)],
  [M5_ULTRA_64C, 64, [96, 256], tbPerSec(1.2)],
  [M5_ULTRA_80C, 80, [96, 256, 512], tbPerSec(1.2)],
];

describe('Apple per-SKU presets', () => {
  it('has eight distinct SKUs, and no nonexistent M4 Ultra', () => {
    expect(APPLE_DEVICE_PRESETS).toHaveLength(8);
    expect(new Set(APPLE_DEVICE_PRESETS.map((d) => d.id)).size).toBe(8);
    // Newest generation first, then Ultra before Max, then more GPU cores first.
    expect(APPLE_DEVICE_PRESETS).toEqual([
      M5_ULTRA_80C, M5_ULTRA_64C, M5_MAX_40C, M5_MAX_32C, M4_MAX_40C, M4_MAX_32C, M3_ULTRA_80C, M3_ULTRA_60C,
    ]);
    expect(APPLE_DEVICE_PRESETS.some((d) => d.name.includes('M4 Ultra'))).toBe(false);
  });

  it.each(skuFacts)('validates and preserves Apple memory facts for $id', (device, cores, capacities, bandwidth) => {
    expect(validateDevice(device)).toEqual([]);
    expect(device.memory.kind).toBe('unified');
    expect(device.memory.capacityOptionsBytes).toEqual(capacities.map(gb));
    expect(device.memory.bandwidthBytesPerSec).toBe(bandwidth);
    expect(device.engines.every((e) => e.computeUnits === cores)).toBe(true);
    expect(device.engines.some((e) => e.kind === 'npu')).toBe(false);
    expect(device.sources.some((s) => s.kind === 'community')).toBe(true);
    device.engines.forEach((engine, i) => {
      expect(device.estimated.some((e) => e.field === `engines.${i}.matmulPeaks`)).toBe(true);
      expect(device.estimated.some((e) => e.field === `engines.${i}.vectorFlopsPerSec`)).toBe(true);
      expect(engine.matmulPeaks.every((p) => p.sparsity === 'dense')).toBe(true);
      expect(engine.matmulPeaks.some((p) => ['fp4', 'fp8', 'int4'].includes(p.input))).toBe(false);
    });
  });

  it.each([
    [M4_MAX_32C, 32, 1.578e9], [M4_MAX_40C, 40, 1.578e9],
    [M3_ULTRA_60C, 60, 1.398e9], [M3_ULTRA_80C, 80, 1.398e9],
  ] as const)('uses SIMD matmul at vector rate for $id', (device, cores, clockHz) => {
    const engine = device.engines[0];
    expect(device.engines).toHaveLength(1);
    expect(engine.id).toBe('gpu');
    // Four 32-lane partitions × two FLOP per FMA, at the explicitly estimated clock.
    const expectedFlopsPerSec = cores * 4 * 32 * 2 * clockHz;
    expect(engine.vectorFlopsPerSec.fp32).toBe(expectedFlopsPerSec);
    expect(engine.vectorFlopsPerSec.fp16).toBe(expectedFlopsPerSec);
    expect(engine.vectorFlopsPerSec.bf16).toBe(expectedFlopsPerSec);
    expect(engine.matmulPeaks.map((p) => [p.input, p.accumulate, p.flopsPerSec])).toEqual([
      ['fp16', 'fp32', expectedFlopsPerSec],
      ['bf16', 'fp32', expectedFlopsPerSec],
      ['fp32', 'fp32', expectedFlopsPerSec],
    ]);
    expect(engine.gemmTiling.default).toMatchObject({ mmaM: 8, mmaN: 8, mmaK: 8 });
  });

  it.each([M5_MAX_32C, M5_MAX_40C, M5_ULTRA_64C, M5_ULTRA_80C])('offers separate NAX and SIMD paths for $id', (device) => {
    const [nax, simd] = device.engines;
    expect(device.engines.map((e) => e.id)).toEqual(['gpu-na', 'gpu-simd']);
    // 512 FMA/core/clock × two FLOP/FMA; 1.62 GHz measured on Max, assumed on Ultra.
    const halfFlopsPerSec = nax.computeUnits * 512 * 2 * 1.62e9;
    expect(nax.matmulPeaks.map((p) => [p.input, p.accumulate, p.flopsPerSec])).toEqual([
      ['fp16', 'fp32', halfFlopsPerSec], ['bf16', 'fp32', halfFlopsPerSec],
      // INT8 is an explicit 2× extrapolation, not an Apple TOPS claim.
      ['int8', 'int32', 2 * halfFlopsPerSec],
    ]);
    expect(nax.matmulPeaks.some((p) => p.input === 'fp32')).toBe(false);
    expect(nax.vectorFlopsPerSec).toEqual(simd.vectorFlopsPerSec);
    expect(simd.vectorFlopsPerSec.fp32).toBe(nax.computeUnits * 256 * 1.62e9);
    expect(simd.vectorFlopsPerSec.fp16).toBe(2 * simd.vectorFlopsPerSec.fp32!);
    expect(simd.vectorFlopsPerSec.bf16).toBe(simd.vectorFlopsPerSec.fp16);
    expect(nax.gemmTiling.default.ctaTiles).toEqual([{ m: 128, n: 64 }]);
    expect(nax.gemmTiling.fp32).toEqual(simd.gemmTiling.default);
    expect(device.estimated.some((e) => e.reason.includes('INT8 extrapolated'))).toBe(true);
  });
});
