import { describe, expect, it } from 'vitest';
import type { AccumulateType, MathFormat } from '../../ir/dtype';
import type { DeviceSpec } from '../types';
import { validateDevice } from '../validate';
import { B200, B300, DGX_SPARK, H100_SXM, H200_SXM, NVIDIA_DATACENTER_DEVICE_PRESETS } from './nvidia-datacenter';

/** The engines[0] peak for (input, accumulate, sparsity); throws if the preset lists none. Unit: FLOP/s (OP/s). */
function peak(d: DeviceSpec, input: MathFormat, acc: AccumulateType, sparsity: 'dense' | '2:4'): number {
  const p = d.engines[0].matmulPeaks.find((x) => x.input === input && x.accumulate === acc && x.sparsity === sparsity);
  if (p === undefined) throw new Error(`${d.id}: no entry ${input}/${acc}/${sparsity}`);
  return p.flopsPerSec;
}

/** |actual − expected| ≤ rel · |expected|. */
function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

describe('NVIDIA datacenter presets', () => {
  it('every preset passes validateDevice and has sourced URLs', () => {
    expect(NVIDIA_DATACENTER_DEVICE_PRESETS.map((d) => d.id)).toEqual(['dgx-spark', 'b300', 'b200', 'h200-sxm', 'h100-sxm']);
    for (const d of NVIDIA_DATACENTER_DEVICE_PRESETS) {
      expect(validateDevice(d)).toEqual([]);
      expect(d.sources.length).toBeGreaterThan(0);
      for (const s of d.sources) expect(s.url).toMatch(/^https:\/\//);
    }
  });

  it('datacenter parts list fp32-accumulate entries only for float inputs (single rate)', () => {
    for (const d of NVIDIA_DATACENTER_DEVICE_PRESETS) {
      expect(d.engines[0].matmulPeaks.some((p) => p.accumulate === 'fp16')).toBe(false);
    }
  });

  it('2:4 = 2 × dense everywhere NVIDIA states it (all but B300 FP4)', () => {
    for (const d of NVIDIA_DATACENTER_DEVICE_PRESETS) {
      for (const p of d.engines[0].matmulPeaks.filter((x) => x.sparsity === '2:4')) {
        if (d.id === 'b300' && p.input === 'fp4') continue; // 18 sparse vs 14 dense (datasheet)
        expectClose(p.flopsPerSec, 2 * peak(d, p.input, p.accumulate, 'dense'));
      }
    }
  });

  it('DGX Spark: 1 PFLOP FP4 headline is sparse; dense rates halve per step; unified 128/64 GB at 273 GB/s', () => {
    expect(peak(DGX_SPARK, 'fp4', 'fp32', '2:4')).toBe(1000e12); // "Up to 1 PFLOP FP4" (with sparsity)
    expect(peak(DGX_SPARK, 'fp4', 'fp32', 'dense')).toBe(500e12); // 1000 / 2
    expect(peak(DGX_SPARK, 'fp8', 'fp32', 'dense')).toBe(250e12); // 500 / 2
    expect(peak(DGX_SPARK, 'bf16', 'fp32', 'dense')).toBe(125e12); // 250 / 2 (Carmack: 125 TF dense BF16)
    expect(peak(DGX_SPARK, 'fp16', 'fp32', 'dense')).toBe(125e12);
    expect(peak(DGX_SPARK, 'tf32', 'fp32', 'dense')).toBe(62.5e12); // 125 / 2
    expect(DGX_SPARK.engines[0].computeUnits).toBe(48);
    expect(DGX_SPARK.memory.kind).toBe('unified');
    expect(DGX_SPARK.memory.capacityOptionsBytes).toEqual([128e9, 64e9]);
    expect(DGX_SPARK.memory.bandwidthBytesPerSec).toBe(273e9); // 256 bit × 8533 MT/s / 8 = 273.06, quoted 273
    expectClose(DGX_SPARK.engines[0].vectorFlopsPerSec.fp32 ?? 0, 29.4912e12); // 6144 × 2 × 2.4 GHz
    expect(DGX_SPARK.l2Bytes).toBe(25_165_824); // 24 MiB
    expect(DGX_SPARK.engines[0].gemmTiling.default.mmaM).toBe(16); // mma.sync m16n8k16
    expect(DGX_SPARK.engines[0].gemmTiling.fp4?.mmaK).toBe(64); // kind::mxf4nvf4 m16n8k64
  });

  it('DGX Spark: estimated dense peaks and the community source are labeled', () => {
    const estimatedFields = DGX_SPARK.estimated.map((e) => e.field);
    const peaks = DGX_SPARK.engines[0].matmulPeaks;
    const fp4SparseIdx = peaks.findIndex((p) => p.input === 'fp4' && p.sparsity === '2:4');
    peaks.forEach((_, i) => {
      expect(estimatedFields.includes(`engines.0.matmulPeaks.${i}`)).toBe(i !== fp4SparseIdx);
    });
    expect(DGX_SPARK.sources.some((s) => s.kind === 'community')).toBe(true);
  });

  it('H100 SXM: dense = ½ of the sparse product-page figures', () => {
    expect(peak(H100_SXM, 'bf16', 'fp32', 'dense')).toBe(989.5e12); // 1,979 sparse / 2
    expect(peak(H100_SXM, 'fp16', 'fp32', '2:4')).toBe(1979e12);
    expect(peak(H100_SXM, 'fp8', 'fp32', 'dense')).toBe(1979e12); // 3,958 sparse / 2
    expect(peak(H100_SXM, 'tf32', 'fp32', 'dense')).toBe(494.5e12); // 989 sparse / 2
    expect(peak(H100_SXM, 'int8', 'int32', 'dense')).toBe(1979e12); // 3,958 TOPS sparse / 2
    expect(H100_SXM.engines[0].vectorFlopsPerSec).toEqual({ fp32: 67e12, fp64: 34e12 });
    expect(H100_SXM.memory.capacityOptionsBytes).toEqual([80e9]);
    expect(H100_SXM.memory.bandwidthBytesPerSec).toBe(3.35e12);
    expect(H100_SXM.engines[0].computeUnits).toBe(132);
    expect(H100_SXM.engines[0].gemmTiling.default).toMatchObject({ mmaM: 64, mmaN: 8, mmaK: 16 }); // wgmma m64nNk16
  });

  it('H200 SXM: H100 compute, 141 GB HBM3e at 4.8 TB/s', () => {
    expect(H200_SXM.engines[0].matmulPeaks).toEqual(H100_SXM.engines[0].matmulPeaks);
    expect(H200_SXM.memory.capacityOptionsBytes).toEqual([141e9]);
    expect(H200_SXM.memory.bandwidthBytesPerSec).toBe(4.8e12);
    expect(H200_SXM.estimated.map((e) => e.field)).toContain('engines.0.computeUnits');
  });

  it('B200: tech-brief per-GPU rates; 180 GB as shipped, 7.7 TB/s', () => {
    expect(peak(B200, 'fp4', 'fp32', 'dense')).toBe(9000e12); // "9/18 petaFLOPS"
    expect(peak(B200, 'fp8', 'fp32', 'dense')).toBe(4500e12); // "4.5/9"
    expect(peak(B200, 'fp6', 'fp32', 'dense')).toBe(peak(B200, 'fp8', 'fp32', 'dense')); // "FP8/FP6" one rate
    expect(peak(B200, 'int8', 'int32', 'dense')).toBe(4500e12); // "4.5/9 petaOPS"
    expect(peak(B200, 'bf16', 'fp32', 'dense')).toBe(2250e12); // 4.5 sparse / 2 (brief rounds to 2.2)
    expect(peak(B200, 'tf32', 'fp32', 'dense')).toBe(1100e12); // "1.1/2.2"
    expect(B200.memory.capacityOptionsBytes).toEqual([180e9]); // 1,440 GB / 8
    expect(B200.memory.bandwidthBytesPerSec).toBe(7.7e12);
    expect(B200.engines[0].computeUnits).toBe(148);
    expect(B200.engines[0].gemmTiling.default).toMatchObject({ mmaM: 128, mmaN: 8, mmaK: 16 }); // tcgen05 kind::f16
  });

  it('B300: FP4 14 dense / 18 sparse; INT8 cut to 153.5 TOPS dense; FP64 1.2 TF; 270 GB', () => {
    expect(peak(B300, 'fp4', 'fp32', 'dense')).toBe(14000e12);
    expect(peak(B300, 'fp4', 'fp32', '2:4')).toBe(18000e12);
    expect(peak(B300, 'fp4', 'fp32', '2:4')).toBe(peak(B200, 'fp4', 'fp32', '2:4')); // sparse FP4 unchanged vs B200
    expect(peak(B300, 'fp8', 'fp32', 'dense')).toBe(4500e12); // 9 sparse / 2
    expect(peak(B300, 'fp6', 'fp32', 'dense')).toBe(4500e12);
    expect(peak(B300, 'bf16', 'fp32', 'dense')).toBe(2250e12); // 4.5 sparse / 2
    expect(peak(B300, 'tf32', 'fp32', 'dense')).toBe(1100e12); // 2.2 sparse / 2
    expect(peak(B300, 'int8', 'int32', 'dense')).toBe(153.5e12); // 307 TOPS sparse / 2 (brief: 0.15 POPS)
    expect(B300.engines[0].vectorFlopsPerSec).toEqual({ fp32: 75e12, fp64: 1.2e12 });
    expect(B300.memory.capacityOptionsBytes).toEqual([270e9]);
    expect(B300.memory.bandwidthBytesPerSec).toBe(7.7e12);
    expect(B300.l2Bytes).toBeUndefined(); // not published
  });
});
