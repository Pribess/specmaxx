import { describe, expect, it } from 'vitest';
import type { AccumulateType, MathFormat } from '../../ir/dtype';
import { matmulPeakFlopsPerSec } from '../peaks';
import type { DeviceSpec } from '../types';
import { validateDevice } from '../validate';
import {
  AMD_DEVICE_PRESETS,
  INSTINCT_MI300X,
  INSTINCT_MI325X,
  INSTINCT_MI355X,
  INSTINCT_MI455X,
  RADEON_AI_PRO_R9700,
  RADEON_RX_9070_XT,
  RYZEN_AI_MAX_PLUS_395,
} from './amd';

/** Engine-0 peak for an exact key, in FLOP/s; throws if the preset does not list it. */
function peak(
  device: DeviceSpec,
  input: MathFormat,
  accumulate: AccumulateType,
  sparsity: 'dense' | '2:4' = 'dense',
): number {
  const p = device.engines[0].matmulPeaks.find(
    (x) => x.input === input && x.accumulate === accumulate && x.sparsity === sparsity,
  );
  if (p === undefined) throw new Error(`missing ${device.id} ${input}/${accumulate}/${sparsity}`);
  return p.flopsPerSec;
}

/** True if engine 0 lists any peak for `input`. */
function lists(device: DeviceSpec, input: MathFormat): boolean {
  return device.engines[0].matmulPeaks.some((p) => p.input === input);
}

/** Expects `actual` (FLOP/s) to round to the vendor's quoted `quotedTflops` within half its last digit. */
function expectQuoted(actual: number, quotedTflops: number, lastDigitTflops: number): void {
  expect(Math.abs(actual / 1e12 - quotedTflops)).toBeLessThanOrEqual(lastDigitTflops / 2 + 1e-9);
}

describe('AMD presets', () => {
  it('every preset validates (all numbers sourced or estimated)', () => {
    expect(AMD_DEVICE_PRESETS.map((d) => d.id)).toEqual([
      'radeon-ai-pro-r9700',
      'radeon-rx-9070-xt',
      'ryzen-ai-max-plus-395',
      'instinct-mi455x',
      'instinct-mi355x',
      'instinct-mi325x',
      'instinct-mi300x',
    ]);
    for (const d of AMD_DEVICE_PRESETS) {
      expect(validateDevice(d)).toEqual([]);
      expect(d.sources.length).toBeGreaterThan(0);
      for (const s of d.sources) expect(s.url).toMatch(/^https:\/\//);
    }
  });

  it('4:2 sparse (recorded as 2:4) is exactly 2× dense wherever listed, and only fp32/int32 accumulate is listed', () => {
    for (const d of AMD_DEVICE_PRESETS) {
      for (const p of d.engines[0].matmulPeaks) {
        expect(p.accumulate).not.toBe('fp16');
        if (p.sparsity === '2:4') expect(p.flopsPerSec).toBe(2 * peak(d, p.input, p.accumulate));
      }
    }
  });

  it('fp16-accumulate requests fall back to the fp32-accumulate rate (AMD: same rate)', () => {
    const warnings: string[] = [];
    const hit = matmulPeakFlopsPerSec(
      RADEON_AI_PRO_R9700.engines[0],
      'fp16',
      'fp16',
      'dense',
      warnings,
    );
    expect(hit.flopsPerSec).toBe(peak(RADEON_AI_PRO_R9700, 'fp16', 'fp32'));
    expect(warnings).toHaveLength(1);
  });

  it('R9700 spot values vs the product page (64 CU × per-CU rate × 2.92 GHz)', () => {
    const d = RADEON_AI_PRO_R9700;
    expect(d.engines[0].computeUnits).toBe(64);
    // FP16 64 × 1024 × 2.92e9 = 191.4 → page "191"; sparse 382.7 → "383".
    expectQuoted(peak(d, 'fp16', 'fp32'), 191, 1);
    expectQuoted(peak(d, 'fp16', 'fp32', '2:4'), 383, 1);
    // FP8 64 × 2048 × 2.92e9 = 382.7 → "383"; INT4 sparse 64 × 8192 × 2.92e9 = 1530.9 → "1531".
    expectQuoted(peak(d, 'fp8', 'fp32'), 383, 1);
    expectQuoted(peak(d, 'int4', 'int32', '2:4'), 1531, 1);
    // BF16 = FP16 rate (GPUOpen per-CU table).
    expect(peak(d, 'bf16', 'fp32')).toBe(peak(d, 'fp16', 'fp32'));
    // Vector: single-issue 4096 × 2 × 2.92e9 = 23.92 TF = half AMD's dual-issue 47.8; packed fp16 = "47.8".
    expectQuoted(2 * (d.engines[0].vectorFlopsPerSec.fp32 ?? 0), 47.8, 0.1);
    expectQuoted(d.engines[0].vectorFlopsPerSec.fp16 ?? 0, 47.8, 0.1);
    expect(d.memory.capacityOptionsBytes).toEqual([32e9]);
    expect(d.memory.bandwidthBytesPerSec).toBe(640e9);
  });

  it('RX 9070 XT spot values vs the product page (64 CU × per-CU rate × 2.97 GHz)', () => {
    const d = RADEON_RX_9070_XT;
    // FP16 64 × 1024 × 2.97e9 = 194.6 → "195"; FP8 sparse 64 × 4096 × 2.97e9 = 778.6 → "779".
    expectQuoted(peak(d, 'fp16', 'fp32'), 195, 1);
    expectQuoted(peak(d, 'fp8', 'fp32', '2:4'), 779, 1);
    // INT8 64 × 2048 × 2.97e9 = 389.3 → "389"; INT4 sparse 64 × 8192 × 2.97e9 = 1557.1 → "1557".
    expectQuoted(peak(d, 'int8', 'int32'), 389, 1);
    expectQuoted(peak(d, 'int4', 'int32', '2:4'), 1557, 1);
    expectQuoted(d.engines[0].vectorFlopsPerSec.fp16 ?? 0, 48.7, 0.1);
    expect(d.memory.capacityOptionsBytes).toEqual([16e9]);
    expect(d.memory.bandwidthBytesPerSec).toBe(640e9);
    // RDNA4 int4 WMMA reaches its full rate as 16×16×32.
    expect(d.engines[0].gemmTiling.int4?.mmaK).toBe(32);
  });

  it('Ryzen AI Max+ 395: 40 CU × 512 × 2.9 GHz = 59.4 ("up to 60 TFLOPS"), unified 256 GB/s, no NPU engine', () => {
    const d = RYZEN_AI_MAX_PLUS_395;
    expect(d.engines.map((e) => e.id)).toEqual(['gpu']);
    expect(d.engines[0].computeUnits).toBe(40);
    // ROCm blog: "59.4 on the 8060S"; AMD page rounds up to 60.
    expectQuoted(peak(d, 'fp16', 'fp32'), 59.4, 0.1);
    expect(peak(d, 'bf16', 'fp32')).toBe(peak(d, 'fp16', 'fp32'));
    // INT4 = 2 × INT8 (1024 vs 512 per CU).
    expect(peak(d, 'int4', 'int32')).toBe(2 * peak(d, 'int8', 'int32'));
    // RDNA 3.5: no sparse WMMA, no FP8.
    expect(d.engines[0].matmulPeaks.every((p) => p.sparsity === 'dense')).toBe(true);
    expect(lists(d, 'fp8')).toBe(false);
    // Vector single-issue 40 × 64 × 2 × 2.9e9 = 14.85 TF; WMMA fp16 = 2× packed fp16 vector.
    expectQuoted(d.engines[0].vectorFlopsPerSec.fp32 ?? 0, 14.85, 0.01);
    expect(peak(d, 'fp16', 'fp32')).toBe(2 * (d.engines[0].vectorFlopsPerSec.fp16 ?? 0));
    expect(d.memory.kind).toBe('unified');
    expect(d.memory.capacityOptionsBytes).toEqual([128e9, 64e9]);
    // 256-bit × 8000 MT/s / 8.
    expect(d.memory.bandwidthBytesPerSec).toBe((256 * 8000e6) / 8);
  });

  it('MI300X spot values vs the datasheet (304 CU × per-CU rate × 2.1 GHz)', () => {
    const d = INSTINCT_MI300X;
    expect(d.engines[0].computeUnits).toBe(304);
    // Datasheet: TF32 653.7, FP16/BF16 1307.4, FP8 2614.9, INT8 2614.9; sparse FP8 5229.8 (0.1 TF digits).
    expectQuoted(peak(d, 'tf32', 'fp32'), 653.7, 0.1);
    expectQuoted(peak(d, 'fp16', 'fp32'), 1307.4, 0.1);
    expectQuoted(peak(d, 'bf16', 'fp32', '2:4'), 2614.9, 0.1);
    expectQuoted(peak(d, 'fp8', 'fp32'), 2614.9, 0.1);
    expectQuoted(peak(d, 'fp8', 'fp32', '2:4'), 5229.8, 0.1);
    expectQuoted(peak(d, 'int8', 'int32'), 2614.9, 0.1);
    expectQuoted(peak(d, 'fp32', 'fp32'), 163.4, 0.1);
    // Vector FP64 81.7, FP32 163.4, FP16 163.4.
    expectQuoted(d.engines[0].vectorFlopsPerSec.fp64 ?? 0, 81.7, 0.1);
    expectQuoted(d.engines[0].vectorFlopsPerSec.fp32 ?? 0, 163.4, 0.1);
    expectQuoted(d.engines[0].vectorFlopsPerSec.fp16 ?? 0, 163.4, 0.1);
    // CDNA3 vendor ratios: FP16 = 2 × TF32, FP8 = 2 × FP16.
    expect(peak(d, 'fp16', 'fp32')).toBe(2 * peak(d, 'tf32', 'fp32'));
    expect(peak(d, 'fp8', 'fp32')).toBe(2 * peak(d, 'fp16', 'fp32'));
    expect(d.memory.capacityOptionsBytes).toEqual([192e9]);
    expect(d.memory.bandwidthBytesPerSec).toBe(5.3e12);
  });

  it('MI325X: MI300X compute with 256 GB HBM3E at 6 TB/s', () => {
    expect(INSTINCT_MI325X.engines).toEqual(INSTINCT_MI300X.engines);
    expect(INSTINCT_MI325X.memory.capacityOptionsBytes).toEqual([256e9]);
    expect(INSTINCT_MI325X.memory.bandwidthBytesPerSec).toBe(6e12);
  });

  it('MI355X spot values vs the product page (256 CU × per-CU rate × 2.4 GHz)', () => {
    const d = INSTINCT_MI355X;
    expect(d.engines[0].computeUnits).toBe(256);
    // FP16 256 × 4096 × 2.4e9 = 2516.6 → "2.5 PF"; sparse 5033.2 → "5 PF".
    expectQuoted(peak(d, 'fp16', 'fp32'), 2500, 100);
    expectQuoted(peak(d, 'fp16', 'fp32', '2:4'), 5000, 100);
    // FP8 dense 5033.2 → "5 PF"; sparse 10066.3 → "10.1 PF".
    expectQuoted(peak(d, 'fp8', 'fp32'), 5000, 100);
    expectQuoted(peak(d, 'fp8', 'fp32', '2:4'), 10100, 100);
    // MXFP4 / MXFP6 dense 256 × 16384 × 2.4e9 = 10066.3 → "10.1 PF" (whitepaper typo "16834" would give 10340).
    expectQuoted(peak(d, 'fp4', 'fp32'), 10100, 100);
    expect(peak(d, 'fp6', 'fp32')).toBe(peak(d, 'fp4', 'fp32'));
    expect(
      d.engines[0].matmulPeaks.some(
        (p) => (p.input === 'fp4' || p.input === 'fp6') && p.sparsity === '2:4',
      ),
    ).toBe(false);
    // Vendor ratios: FP8 = 2 × FP16, FP4 = 4 × FP16; 2× the MI300X's per-CU FP16 rate.
    expect(peak(d, 'fp8', 'fp32')).toBe(2 * peak(d, 'fp16', 'fp32'));
    expect(peak(d, 'fp4', 'fp32')).toBe(4 * peak(d, 'fp16', 'fp32'));
    expectQuoted(peak(d, 'fp32', 'fp32'), 157.3, 0.1);
    expectQuoted(d.engines[0].vectorFlopsPerSec.fp64 ?? 0, 78.6, 0.1);
    expect(d.memory.capacityOptionsBytes).toEqual([288e9]);
    expect(d.memory.bandwidthBytesPerSec).toBe(8e12);
  });

  it('MI355X has no TF32 hardware: a tf32 request falls back to the fp32 matrix rate', () => {
    expect(lists(INSTINCT_MI355X, 'tf32')).toBe(false);
    const warnings: string[] = [];
    const hit = matmulPeakFlopsPerSec(
      INSTINCT_MI355X.engines[0],
      'tf32',
      'fp32',
      'dense',
      warnings,
    );
    expect(hit.usedInput).toBe('fp32');
    expect(hit.flopsPerSec).toBe(peak(INSTINCT_MI355X, 'fp32', 'fp32'));
  });

  it('MI455X: published per-device datasheet figures, sparse only where published', () => {
    const d = INSTINCT_MI455X;
    // Datasheet TFLOPS table: FP16 / BF16 / INT8 5033 dense, 10066 sparse.
    expect(peak(d, 'fp16', 'fp32')).toBe(5033e12);
    expect(peak(d, 'bf16', 'fp32', '2:4')).toBe(10066e12);
    expect(peak(d, 'int8', 'int32', '2:4')).toBe(10066e12);
    // OCP FP8 / MXFP6 20133, MXFP4 40265; no sparse figure for them.
    expect(peak(d, 'fp8', 'fp32')).toBe(20133e12);
    expect(peak(d, 'fp6', 'fp32')).toBe(20133e12);
    expect(peak(d, 'fp4', 'fp32')).toBe(40265e12);
    for (const input of ['fp8', 'fp6', 'fp4'] as const) {
      expect(d.engines[0].matmulPeaks.some((p) => p.input === input && p.sparsity === '2:4')).toBe(
        false,
      );
    }
    expect(peak(d, 'fp32', 'fp32')).toBe(315e12);
    expect(d.engines[0].vectorFlopsPerSec).toEqual({ fp64: 5e12, fp32: 315e12, fp16: 315e12 });
    expect(d.memory.capacityOptionsBytes).toEqual([432e9]);
    expect(d.memory.bandwidthBytesPerSec).toBe(23.3e12);
    // computeUnits is flagged estimated (CU vs WGP labelling conflict); l2Bytes omitted.
    expect(d.estimated.some((e) => e.field === 'engines.0.computeUnits')).toBe(true);
    expect(d.l2Bytes).toBeUndefined();
  });
});
