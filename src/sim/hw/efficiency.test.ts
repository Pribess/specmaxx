import { describe, expect, it } from 'vitest';
import { SYNTH_GPU, SYNTH_TILING } from '../__fixtures__/synthetic';
import { gemmComputeSec } from './efficiency';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

// SYNTH_GPU: 100 units × 1 CTA = 100 slots; tiling mma m64 n8 k16, one CTA tile 128×128.
const PEAK = 100e12;
const UNITS = SYNTH_GPU.engines[0].computeUnits;

describe('gemmComputeSec (T5, SYNTH_GPU)', () => {
  it('1280×1280×1024: 10·10 = 100 tiles, 1 wave, efficiency 1', () => {
    const r = gemmComputeSec({ batch: 1, m: 1280, n: 1280, k: 1024 }, PEAK, UNITS, SYNTH_TILING);
    expect(r.tiles).toBe(100); // ceil(1280/128)·ceil(1280/128)
    expect(r.waves).toBe(1); // ceil(100/100)
    expectClose(r.efficiency, 1);
    // ideal = 2·1280·1280·1024 / 100e12 = 3.3554432e-5 s
    expectClose(r.computeSec, (2 * 1280 * 1280 * 1024) / PEAK);
  });

  it('128×12928×1024: 101 tiles → 2 waves, efficiency 0.505', () => {
    const r = gemmComputeSec({ batch: 1, m: 128, n: 12928, k: 1024 }, PEAK, UNITS, SYNTH_TILING);
    expect(r.tiles).toBe(101); // 1 · ceil(12928/128) = 101
    expect(r.waves).toBe(2); // ceil(101/100)
    // efficiency = useful / (2 waves · 100 slots · 128·128 per tile) = 128·12928 / (2·100·128·128) = 0.505
    expectClose(r.efficiency, 0.505);
  });

  it('M 1, N 12800, K 1024: swap-AB wins with tile 128×8, efficiency 0.125 (unswapped 1/64)', () => {
    const r = gemmComputeSec({ batch: 1, m: 1, n: 12800, k: 1024 }, PEAK, UNITS, SYNTH_TILING);
    expect(r.swapped).toBe(true);
    expect(r.tile).toEqual({ m: 128, n: 8 }); // A' = 12800 → tm 128; B' = roundUp(1, 8) = 8 → tn 8
    expect(r.tiles).toBe(100); // 12800/128 · 1
    // efficiency = 1·12800 / (100 tiles · 128·8) = 0.125
    expectClose(r.efficiency, 0.125);

    // Unswapped layout, by hand: m pads to 64 (tm 64), tn 128 → 100 tiles, 1 wave;
    // efficiency = 1·12800 / (100 · 64·128) = 1/64, so it is 8× slower than the swapped one.
    const unswappedSec = (1 * 2 * 64 * 128 * 1024) / (PEAK / 100); // 1 wave of 64×128 tiles, K' 1024
    const idealSec = (2 * 1 * 12800 * 1024) / PEAK;
    expectClose(idealSec / unswappedSec, 1 / 64);
    expect(r.computeSec).toBeLessThan(unswappedSec);
  });

  it('K 1000 pads to K\' 1008 (mmaK 16): efficiency × 1000/1008', () => {
    const r = gemmComputeSec({ batch: 1, m: 1280, n: 1280, k: 1000 }, PEAK, UNITS, SYNTH_TILING);
    expectClose(r.efficiency, 1000 / 1008); // roundUp(1000, 16) = 1008
  });

  it('batch multiplies tiles; computeSec = F / (peak · efficiency)', () => {
    const shape = { batch: 3, m: 256, n: 256, k: 512 };
    const r = gemmComputeSec(shape, PEAK, UNITS, SYNTH_TILING);
    expect(r.tiles).toBe(12); // 3 · 2 · 2
    expect(r.waves).toBe(1);
    const flops = 2 * 3 * 256 * 256 * 512;
    expectClose(r.computeSec, flops / (PEAK * r.efficiency));
    // 12 tiles of 100 slots in one wave: efficiency = 12/100
    expectClose(r.efficiency, 0.12);
  });

  it('a GEMM with no work costs 0 s', () => {
    const r = gemmComputeSec({ batch: 1, m: 0, n: 4096, k: 4096 }, PEAK, UNITS, SYNTH_TILING);
    expect(r.computeSec).toBe(0);
    expect(r.efficiency).toBe(1);
  });
});
