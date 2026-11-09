import { describe, expect, it } from 'vitest';
import type { AccumulateType, MathFormat } from '../../ir/dtype';
import { applyDeviceModifiers } from '../modifiers';
import { matmulPeakFlopsPerSec } from '../peaks';
import type { DeviceSpec } from '../types';
import { allUnlockSelections, resolveUnlocks, type UnlockSelection } from '../unlocks';
import { validateDevice, validateDeviceModifiers } from '../validate';
import { hostPcieLink } from './links';
import { CMP_170HX, CMP_170HX_UNLOCKS } from './nvidia-cmp';
import { A100_SXM4_80GB } from './nvidia-datacenter';

function expectClose(actual: number, expected: number, rel = 1e-12): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** The CMP 170HX under `selection`. */
function unlocked(selection: UnlockSelection): DeviceSpec {
  return applyDeviceModifiers(
    CMP_170HX,
    resolveUnlocks(CMP_170HX_UNLOCKS, selection).deviceModifiers,
  );
}

/** The peak the engine uses for (input, accumulate, sparsity) and the key it came from, after the documented fallback. */
function lookup(
  d: DeviceSpec,
  input: MathFormat,
  acc: AccumulateType,
  sparsity: 'dense' | '2:4' = 'dense',
): { rate: number; key: string } {
  const warnings: string[] = [];
  const p = matmulPeakFlopsPerSec(d.engines[0], input, acc, sparsity, warnings);
  return { rate: p.flopsPerSec, key: `${p.usedInput}/${p.usedAccumulate}/${p.usedSparsity}` };
}

/** The peak rate the engine uses for (input, accumulate, dense). Unit: FLOP/s (OP/s for integers). */
const used = (d: DeviceSpec, input: MathFormat, acc: AccumulateType): number =>
  lookup(d, input, acc).rate;

/** 70 SMs × r FLOP/clk/SM × 1.41 GHz. */
const at70 = (r: number): number => 70 * r * 1.41e9;

/** Theoretical HBM bandwidth at the stock vBIOS 67 clock (1458 MHz) and the vBIOS 6D clock (1728 MHz). Unit: B/s. */
const STOCK_BW = (1458e6 * 2 * 4096) / 8;
const VBIOS_6D_BW = (1728e6 * 2 * 4096) / 8;
const HBM_1890_BW = (1890e6 * 2 * 4096) / 8;

describe('CMP 170HX stock', () => {
  it('validates, every numeric field covered', () => {
    expect(validateDevice(CMP_170HX)).toEqual([]);
  });

  it('peaks: tensor ÷32 of GA100 per SM × 70 × 1.41 GHz; FP32 FMA ÷32; HFMA2 not throttled', () => {
    const e = CMP_170HX.engines[0];
    expect(e.computeUnits).toBe(70);
    expectClose(used(CMP_170HX, 'fp16', 'fp32'), at70(2048) / 32); // 6.317e12
    expectClose(used(CMP_170HX, 'bf16', 'fp32'), at70(2048) / 32);
    expectClose(used(CMP_170HX, 'tf32', 'fp32'), at70(1024) / 32); // 3.158e12
    expectClose(used(CMP_170HX, 'int8', 'int32'), at70(4096) / 32); // 12.63e12
    expectClose(used(CMP_170HX, 'int4', 'int32'), at70(8192) / 32); // 25.27e12
    expectClose(e.vectorFlopsPerSec.fp32!, at70(128) / 32); // 0.3948e12 (measured 0.39)
    expectClose(e.vectorFlopsPerSec.fp64!, at70(64) / 32);
    expectClose(e.vectorFlopsPerSec.fp16!, at70(512)); // 50.53e12
    expectClose(e.sfuOpsPerSec!, 70 * 16 * 1.41e9);
  });

  it('FP16 accumulators have their own entries (both fused, so equal at stock), each request resolving to its own', () => {
    expect(lookup(CMP_170HX, 'fp16', 'fp16')).toEqual({
      rate: at70(2048) / 32,
      key: 'fp16/fp16/dense',
    });
    expect(lookup(CMP_170HX, 'fp16', 'fp32')).toEqual({
      rate: at70(2048) / 32,
      key: 'fp16/fp32/dense',
    });
  });

  it('memory 8 GB at 1458 MHz × 2 × 4096 bit / 8 = 1492.99 GB/s; host interface Gen1 x4 (0.914 GB/s payload)', () => {
    expect(CMP_170HX.memory.capacityOptionsBytes).toEqual([8e9]);
    expectClose(CMP_170HX.memory.bandwidthBytesPerSec, STOCK_BW);
    expect(CMP_170HX.l2Bytes).toBe(32 * 1024 * 1024);
    expect(CMP_170HX.hostInterface).toEqual({ generation: 1, lanes: 4 });
    expectClose(
      hostPcieLink(1, 4).bandwidthPerDirectionBytesPerSec,
      ((2.5e9 * 4 * 0.8) / 8) * (256 / 280),
    );
  });
});

describe('CMP 170HX vs the A100 preset (same GA100 SM)', () => {
  it("fully unlocked, every matmul and vector peak is the A100 preset's × 70/108 SMs (within the A100 table rounding)", () => {
    const d = unlocked({ 'compute-path': 'cmpunlocker' });
    const a100 = A100_SXM4_80GB.engines[0];
    const ratio = 70 / a100.computeUnits;
    for (const p of a100.matmulPeaks) {
      // The A100 lists one fp16 accumulator; the CMP lists both at the same rate.
      const rate = lookup(d, p.input, p.accumulate, p.sparsity).rate;
      expect(
        Math.abs(rate / (p.flopsPerSec * ratio) - 1),
        `${p.input}/${p.accumulate}/${p.sparsity}`,
      ).toBeLessThan(0.005);
    }
    for (const k of ['fp32', 'fp64', 'fp16', 'bf16'] as const) {
      const cmp = d.engines[0].vectorFlopsPerSec[k]!;
      expect(Math.abs(cmp / (a100.vectorFlopsPerSec[k]! * ratio) - 1), k).toBeLessThan(0.01);
    }
  });
});
describe('CMP 170HX unlocks', () => {
  it('three axes in stage order (firmware, driver, board); every selection (2 × 6 × 2 = 24) validates', () => {
    expect(CMP_170HX_UNLOCKS.map((a) => a.id)).toEqual([
      'vbios-6d',
      'compute-path',
      'pcie-capacitor-mod',
    ]);
    expect(CMP_170HX_UNLOCKS[1].options.map((o) => o.id)).toEqual([
      'locked-workarounds',
      'cmpunlocker',
      'cmpunlocker-hbm-1890',
      'cmpunlocker-74sm',
      'cmpunlocker-74sm-hbm-1890',
    ]);
    const selections = allUnlockSelections(CMP_170HX_UNLOCKS);
    expect(selections).toHaveLength(24);
    for (const s of selections) {
      expect(
        validateDeviceModifiers(CMP_170HX, resolveUnlocks(CMP_170HX_UNLOCKS, s).deviceModifiers),
        JSON.stringify(s),
      ).toEqual([]);
    }
  });

  it('every offered combination is physically possible: what needs cmpunlocker only comes with it, once', () => {
    for (const s of allUnlockSelections(CMP_170HX_UNLOCKS)) {
      const label = JSON.stringify(s);
      const mods = resolveUnlocks(CMP_170HX_UNLOCKS, s).deviceModifiers;
      const d = applyDeviceModifiers(CMP_170HX, mods);
      const option = s['compute-path'];
      const driver = option !== undefined && option.startsWith('cmpunlocker');
      const sm74 = option === 'cmpunlocker-74sm' || option === 'cmpunlocker-74sm-hbm-1890';
      const smFactor = sm74 ? 74 / 70 : 1;
      // The +4 SM patch appears exactly once in its options, never elsewhere, and only with cmpunlocker.
      expect(
        mods.filter((m) => m.kind === 'computeUnits'),
        label,
      ).toHaveLength(sm74 ? 1 : 0);
      expect(d.engines[0].computeUnits, label).toBe(sm74 ? 74 : 70);
      if (d.engines[0].computeUnits === 74) expect(driver, label).toBe(true);
      // cmpunlocker's modifiers appear exactly once (or not at all): 64 GB geometry, Gen2, ×32 compute.
      expect(
        mods.filter((m) => m.kind === 'memoryCapacity'),
        label,
      ).toHaveLength(driver ? 1 : 0);
      expect(
        mods.filter((m) => m.kind === 'hostInterface' && m.generation !== undefined),
        label,
      ).toHaveLength(driver ? 1 : 0);
      expect(
        mods.filter((m) => m.kind === 'peakScale' && m.factor === 32),
        label,
      ).toHaveLength(driver ? 1 : 0);
      // Bandwidth above the 6D ROM's 1769 GB/s needs the HBM overclock, which needs cmpunlocker (its PLM patch).
      if (d.memory.bandwidthBytesPerSec > VBIOS_6D_BW) {
        expect(driver, label).toBe(true);
        expect(d.memory.capacityOptionsBytes, label).toEqual([64e9, 32e9]);
        expectClose(
          used(d, 'bf16', 'fp32'),
          at70(2048) * smFactor * (s['vbios-6d'] ? 1470 / 1410 : 1),
        );
      }
      // Gen2 only with the driver patch; x16 only with the capacitor mod.
      expect(d.hostInterface, label).toEqual({
        generation: driver ? 2 : 1,
        lanes: s['pcie-capacitor-mod'] ? 16 : 4,
      });
      // Locked-card kernels never come with full-rate tensors.
      if (!driver) expect(used(d, 'bf16', 'fp32'), label).toBeLessThan(at70(2048) / 16);
    }
  });

  it('HBM bandwidth in every vBIOS × overclock combination: stock 1493, 6D 1769, overclock 1935 with or without 6D', () => {
    expectClose(unlocked({}).memory.bandwidthBytesPerSec, STOCK_BW);
    expectClose(unlocked({ 'vbios-6d': '6d' }).memory.bandwidthBytesPerSec, VBIOS_6D_BW);
    expectClose(
      unlocked({ 'vbios-6d': '6d', 'compute-path': 'cmpunlocker' }).memory.bandwidthBytesPerSec,
      VBIOS_6D_BW,
    );
    expectClose(
      unlocked({ 'compute-path': 'cmpunlocker-hbm-1890' }).memory.bandwidthBytesPerSec,
      HBM_1890_BW,
    );
    expectClose(
      unlocked({ 'vbios-6d': '6d', 'compute-path': 'cmpunlocker-hbm-1890' }).memory
        .bandwidthBytesPerSec,
      HBM_1890_BW,
    );
    // 6D also lifts the sustained SM clock 1410 → 1470 MHz.
    expectClose(
      used(unlocked({ 'vbios-6d': '6d' }), 'fp16', 'fp32'),
      (at70(2048) / 32) * (1470 / 1410),
    );
  });

  it('locked workarounds: FP32 FADD/FMUL 6.32; half2 raises only the FP16-accumulate entry; INT8 via DP2A 25.3', () => {
    const d = unlocked({ 'compute-path': 'locked-workarounds' });
    expectClose(d.engines[0].vectorFlopsPerSec.fp32!, at70(64)); // 6.317e12
    // An fp16-accumulate GEMM runs at the HFMA2 rate on its own entry...
    expect(lookup(d, 'fp16', 'fp16').key).toBe('fp16/fp16/dense');
    expectClose(lookup(d, 'fp16', 'fp16').rate, at70(512)); // 50.53e12
    expectClose(lookup(d, 'fp16', 'fp16', '2:4').rate, at70(512)); // runs dense
    // ...an fp32-accumulate one keeps the throttled tensor rate (= FMUL + FADD on the CUDA cores, 70 × 64 × 1.41e9).
    expect(lookup(d, 'fp16', 'fp32')).toEqual({ rate: at70(2048) / 32, key: 'fp16/fp32/dense' });
    expectClose(at70(2048) / 32, at70(64));
    expectClose(used(d, 'int8', 'int32'), at70(256)); // 25.27e12
    expectClose(used(d, 'bf16', 'fp32'), at70(2048) / 32); // unchanged (no measured bf16x2 path)
    expect(d.hostInterface).toEqual({ generation: 1, lanes: 4 });
    expect(d.memory.capacityOptionsBytes).toEqual([8e9]);
  });

  it('cmpunlocker: full GA100 rates on both FP16 accumulators, 64 / 32 GB, PCIe Gen2, bandwidth unchanged', () => {
    const d = unlocked({ 'compute-path': 'cmpunlocker' });
    expectClose(used(d, 'bf16', 'fp32'), at70(2048)); // 202.1e12 (measured 164–193)
    expectClose(used(d, 'fp16', 'fp16'), at70(2048));
    expectClose(used(d, 'fp16', 'fp32'), at70(2048));
    expectClose(used(d, 'int8', 'int32'), at70(4096)); // 404.3e12 (raw MMA measured 335)
    expectClose(d.engines[0].vectorFlopsPerSec.fp32!, at70(128)); // 12.63e12
    expectClose(d.engines[0].vectorFlopsPerSec.fp64!, at70(64)); // 6.317e12
    expectClose(d.engines[0].vectorFlopsPerSec.fp16!, at70(512)); // unchanged
    expect(d.memory.capacityOptionsBytes).toEqual([64e9, 32e9]);
    expect(d.memory.bandwidthBytesPerSec).toBe(CMP_170HX.memory.bandwidthBytesPerSec);
    expect(d.hostInterface).toEqual({ generation: 2, lanes: 4 });
  });

  it('PCIe: the Gen2 patch and the capacitor mod combine to Gen2 x16 (7.31 GB/s payload)', () => {
    expect(unlocked({ 'pcie-capacitor-mod': 'x16' }).hostInterface).toEqual({
      generation: 1,
      lanes: 16,
    });
    const both = unlocked({ 'compute-path': 'cmpunlocker-hbm-1890', 'pcie-capacitor-mod': 'x16' });
    expect(both.hostInterface).toEqual({ generation: 2, lanes: 16 });
    expectClose(
      hostPcieLink(2, 16).bandwidthPerDirectionBytesPerSec,
      ((5e9 * 16 * 0.8) / 8) * (256 / 280),
    );
  });

  it('+4 SMs: 74 SMs, every compute rate × 74/70, with or without the HBM overclock; memory unchanged', () => {
    const d74 = unlocked({ 'compute-path': 'cmpunlocker-74sm' });
    expect(d74.engines[0].computeUnits).toBe(74);
    expectClose(used(d74, 'bf16', 'fp32'), 74 * 2048 * 1.41e9);
    expectClose(d74.engines[0].vectorFlopsPerSec.fp32!, 74 * 128 * 1.41e9);
    expectClose(d74.engines[0].sfuOpsPerSec!, 74 * 16 * 1.41e9);
    expectClose(d74.memory.bandwidthBytesPerSec, STOCK_BW);
    expect(d74.memory.capacityOptionsBytes).toEqual([64e9, 32e9]);
    const oc = unlocked({ 'compute-path': 'cmpunlocker-74sm-hbm-1890', 'vbios-6d': '6d' });
    expect(oc.engines[0].computeUnits).toBe(74);
    expectClose(oc.memory.bandwidthBytesPerSec, HBM_1890_BW);
    expectClose(used(oc, 'bf16', 'fp32'), 74 * 2048 * 1.41e9 * (1470 / 1410));
  });
  it('every option cites community or review sources, and every axis has a description', () => {
    for (const axis of CMP_170HX_UNLOCKS) {
      expect(axis.description.trim()).not.toBe('');
      for (const o of axis.options) {
        for (const m of o.deviceModifiers) {
          expect(
            m.sources.every((s) => s.kind === 'community' || s.kind === 'review'),
            `${axis.id}/${o.id}`,
          ).toBe(true);
        }
      }
    }
  });
});
