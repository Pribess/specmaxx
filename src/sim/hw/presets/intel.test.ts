import { describe, expect, it } from 'vitest';
import type { AccumulateType, MathFormat } from '../../ir/dtype';
import type { DeviceSpec } from '../types';
import { gb, gbPerSec, tflops } from '../units';
import { validateDevice } from '../validate';
import {
  ARC_B580,
  ARC_PRO_B60,
  ARC_PRO_B65,
  ARC_PRO_B70,
  INTEL_DEVICE_PRESETS,
  XE2_XMX_FP16_FLOP_PER_CLOCK_PER_CORE,
  XE2_XMX_INT4_OP_PER_CLOCK_PER_CORE,
  XE2_XMX_INT8_OP_PER_CLOCK_PER_CORE,
} from './intel';

function peak(device: DeviceSpec, input: MathFormat, accumulate: AccumulateType): number {
  const p = device.engines[0].matmulPeaks.find(
    (x) => x.input === input && x.accumulate === accumulate && x.sparsity === 'dense',
  );
  if (p === undefined) throw new Error(`Missing ${device.id} ${input}/${accumulate}`);
  return p.flopsPerSec;
}

/** [device, Xe cores, max clock (Hz), vendor INT8 TOPS, memory GB, memory GB/s]. */
const CARDS: [DeviceSpec, number, number, number, number, number][] = [
  [ARC_PRO_B70, 32, 2.8e9, 367, 32, 608],
  [ARC_PRO_B65, 20, 2.4e9, 197, 32, 608],
  [ARC_PRO_B60, 20, 2.4e9, 197, 24, 456],
  [ARC_B580, 20, 2.85e9, 233, 12, 456],
];

describe('Intel Battlemage presets', () => {
  it('lists four distinct devices that all validate', () => {
    expect(INTEL_DEVICE_PRESETS).toEqual([ARC_PRO_B70, ARC_PRO_B65, ARC_PRO_B60, ARC_B580]);
    expect(new Set(INTEL_DEVICE_PRESETS.map((d) => d.id)).size).toBe(4);
    for (const d of INTEL_DEVICE_PRESETS) expect(validateDevice(d)).toEqual([]);
  });

  it.each(CARDS)(
    '%#: Xe cores × 4096 × max clock reproduces the vendor INT8 TOPS',
    (d, cores, clockHz, int8Tops) => {
      expect(d.engines[0].computeUnits).toBe(cores);
      // Vendor INT8 is stored as quoted; the formula must match it to 3 significant digits (< 0.5 %).
      expect(peak(d, 'int8', 'int32')).toBe(tflops(int8Tops));
      const derived = cores * XE2_XMX_INT8_OP_PER_CLOCK_PER_CORE * clockHz;
      expect(Math.abs(derived / tflops(int8Tops) - 1)).toBeLessThan(0.005);
    },
  );

  it.each(CARDS)(
    '%#: FP16/BF16 = INT8 / 2 and INT4 = INT8 × 2 (Xe2 per-core rates 2048 / 4096 / 8192)',
    (d, cores, clockHz) => {
      const fp16 = cores * XE2_XMX_FP16_FLOP_PER_CLOCK_PER_CORE * clockHz;
      expect(peak(d, 'fp16', 'fp32')).toBe(fp16);
      expect(peak(d, 'bf16', 'fp32')).toBe(fp16);
      expect(peak(d, 'int4', 'int32')).toBe(cores * XE2_XMX_INT4_OP_PER_CLOCK_PER_CORE * clockHz);
      expect(peak(d, 'int4', 'int32') / peak(d, 'fp16', 'fp32')).toBe(4);
      // Vector FP16 = 2 × FP32 (deck p. 9: 512 vs 256 FLOP/clk/core). B60/B65 FP32 is Intel's rounded 12.28.
      const v = d.engines[0].vectorFlopsPerSec;
      expect(v.fp16).toBe(cores * 512 * clockHz);
      expect(Math.abs((v.fp16 ?? 0) / (2 * (v.fp32 ?? 0)) - 1)).toBeLessThan(0.005);
    },
  );

  it.each(CARDS)('%#: memory as Intel quotes it', (d, _c, _h, _t, capacityGb, bandwidthGbps) => {
    expect(d.memory.kind).toBe('dedicated');
    expect(d.memory.technology).toBe('GDDR6');
    expect(d.memory.capacityOptionsBytes).toEqual([gb(capacityGb)]);
    expect(d.memory.bandwidthBytesPerSec).toBe(gbPerSec(bandwidthGbps));
  });

  it('spot values against the datasheets', () => {
    // B70: 32 × 2048 × 2.8 GHz = 183.5 TFLOP/s FP16; 32 × 8192 × 2.8 GHz = 734 TOP/s INT4; FP32 22.94 (ark).
    expect(peak(ARC_PRO_B70, 'fp16', 'fp32') / 1e12).toBeCloseTo(183.5, 1);
    expect(peak(ARC_PRO_B70, 'int4', 'int32') / 1e12).toBeCloseTo(734.0, 0);
    expect(ARC_PRO_B70.engines[0].vectorFlopsPerSec.fp32).toBe(tflops(22.94));
    expect(ARC_PRO_B70.l2Bytes).toBe(24 * 1024 * 1024);
    // B60: 20 × 2048 × 2.4 GHz = 98.3 TFLOP/s FP16; FP32 12.28 (datasheet).
    expect(peak(ARC_PRO_B60, 'bf16', 'fp32') / 1e12).toBeCloseTo(98.3, 1);
    expect(ARC_PRO_B60.engines[0].vectorFlopsPerSec.fp32).toBe(tflops(12.28));
    expect(ARC_PRO_B60.l2Bytes).toBe(18 * 1024 * 1024);
    // B580: 20 × 256 × 2.85 GHz = 14.59 TFLOP/s FP32 (derived; PCGH 14.6).
    expect((ARC_B580.engines[0].vectorFlopsPerSec.fp32 ?? 0) / 1e12).toBeCloseTo(14.59, 2);
  });

  it('lists only formats Xe2 XMX has: dense, fp32/int32 accumulate, no fp8/fp4/tf32/fp64', () => {
    for (const d of INTEL_DEVICE_PRESETS) {
      const peaks = d.engines[0].matmulPeaks;
      expect(peaks.map((p) => p.input)).toEqual(['fp16', 'bf16', 'int8', 'int4']);
      expect(peaks.every((p) => p.sparsity === 'dense')).toBe(true);
      expect(peaks.every((p) => p.accumulate !== 'fp16')).toBe(true);
      expect(d.engines[0].vectorFlopsPerSec.fp64).toBeUndefined();
    }
  });

  it('marks the derived peaks as estimated and the vendor INT8 as sourced', () => {
    for (const d of INTEL_DEVICE_PRESETS) {
      const estimatedFields = d.estimated.map((e) => e.field);
      for (const i of [0, 1, 3]) expect(estimatedFields).toContain(`engines.0.matmulPeaks.${i}`);
      expect(estimatedFields).toContain('engines.0.vectorFlopsPerSec.fp16');
      expect(estimatedFields).not.toContain('engines.0.matmulPeaks.2');
      expect(d.sources.some((s) => s.fields.includes('engines.0.matmulPeaks.2'))).toBe(true);
    }
  });

  it('uses DPAS M8 × N16 tiles with K 16 / 32 / 64 for fp16 / int8 / int4', () => {
    const t = ARC_PRO_B70.engines[0].gemmTiling;
    expect([t.fp16?.mmaM, t.fp16?.mmaN, t.fp16?.mmaK]).toEqual([8, 16, 16]);
    expect(t.int8?.mmaK).toBe(32);
    expect(t.int4?.mmaK).toBe(64);
    expect(t.default.mmaK).toBe(16);
  });
});
