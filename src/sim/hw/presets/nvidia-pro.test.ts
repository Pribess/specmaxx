import { describe, expect, it } from 'vitest';
import type { MathFormat } from '../../ir/dtype';
import { applyDeviceModifiers } from '../modifiers';
import type { DeviceSpec } from '../types';
import { validateDevice, validateDeviceModifiers } from '../validate';
import { DEVICE_PRESET_GROUPS, deviceUnlocks } from './index';
import {
  NVIDIA_PRO_DEVICE_PRESETS,
  RTX_PRO_4000_SFF,
  RTX_PRO_5000,
  RTX_PRO_6000_MAX_Q,
  RTX_PRO_6000_WS,
} from './nvidia-pro';

function peak(device: DeviceSpec, input: MathFormat, sparsity: 'dense' | '2:4' = 'dense'): number {
  const p = device.engines[0].matmulPeaks.find((p) => p.input === input && p.sparsity === sparsity);
  if (!p) throw new Error(`Missing ${device.id} ${input}/${sparsity}`);
  return p.flopsPerSec;
}

describe('RTX PRO Blackwell workstation presets', () => {
  it('registers all four SKUs in tier order in a dedicated picker group', () => {
    expect(NVIDIA_PRO_DEVICE_PRESETS).toEqual([
      RTX_PRO_6000_WS,
      RTX_PRO_6000_MAX_Q,
      RTX_PRO_5000,
      RTX_PRO_4000_SFF,
    ]);
    expect(DEVICE_PRESET_GROUPS.find((g) => g.name === 'NVIDIA RTX PRO')?.devices).toEqual(
      NVIDIA_PRO_DEVICE_PRESETS,
    );
  });

  it.each(NVIDIA_PRO_DEVICE_PRESETS)(
    '$id: validates, with full-rate FP32-accumulate entries and SM120 instruction shapes',
    (device) => {
      expect(validateDevice(device)).toEqual([]);
      const e = device.engines[0];
      expect(e.matmulPeaks).toHaveLength(12);
      expect(
        e.matmulPeaks.every((p) => p.accumulate === (p.input === 'int8' ? 'int32' : 'fp32')),
      ).toBe(true);
      expect(e.matmulPeaks.some((p) => p.input === 'fp6' || p.input === 'int4')).toBe(false);
      for (const p of e.matmulPeaks.filter((p) => p.sparsity === '2:4')) {
        expect(Math.abs(p.flopsPerSec / (2 * peak(device, p.input)) - 1)).toBeLessThan(0.001);
      }
      expect(e.gemmTiling.fp4).toMatchObject({ mmaM: 16, mmaN: 8, mmaK: 64, ctasPerUnit: 1 });
      expect(e.gemmTiling.tf32?.mmaK).toBe(8);
      expect(e.gemmTiling.fp8?.mmaK).toBe(32);
      expect(e.sfuOpsPerSec).toBe(e.vectorFlopsPerSec.fp32! / 16); // 16 SFU results vs 256 FP32 FLOP/clock/SM
      expect(device.gpuDirectRdma).toBe(true);
      expect(device.memory.gpuUsableFraction).toBeUndefined(); // dedicated VRAM, not unified memory
    },
  );

  it('6000 Workstation whitepaper spot checks: 188 SM, 503.8 BF16, 4030.4 sparse FP4, 1792 GB/s', () => {
    expect(RTX_PRO_6000_WS.engines[0].computeUnits).toBe(188);
    expect(peak(RTX_PRO_6000_WS, 'bf16')).toBe(503.8e12);
    expect(peak(RTX_PRO_6000_WS, 'fp4', '2:4')).toBe(4030.4e12);
    expect(RTX_PRO_6000_WS.memory.bandwidthBytesPerSec).toBe(1792e9);
    expect(RTX_PRO_6000_WS.l2Bytes).toBe(128 * 1024 ** 2);
  });

  it('Max-Q retains its own rounded table, not a power-ratio scaling of Workstation', () => {
    expect(peak(RTX_PRO_6000_MAX_Q, 'bf16')).toBe(438.9e12);
    expect(peak(RTX_PRO_6000_MAX_Q, 'tf32')).toBe(219.5e12);
    expect(peak(RTX_PRO_6000_MAX_Q, 'fp4', '2:4')).toBe(3511.4e12);
    expect(RTX_PRO_6000_MAX_Q.engines[0].vectorFlopsPerSec.fp32).toBe(109.7e12);
    expect(RTX_PRO_6000_MAX_Q.memory.capacityOptionsBytes).toEqual([96e9]);
  });

  it('5000 both capacities share 1344 GB/s and 2064 sparse FP4 TOPS; inferred rates are explicit', () => {
    expect(RTX_PRO_5000.memory.capacityOptionsBytes).toEqual([48e9, 72e9]);
    expect(RTX_PRO_5000.memory.bandwidthBytesPerSec).toBe(1344e9);
    expect(peak(RTX_PRO_5000, 'fp4', '2:4')).toBe(2064e12);
    expect(RTX_PRO_5000.engines[0].computeUnits).toBe(110);
    expect(peak(RTX_PRO_5000, 'bf16')).toBe(258e12);
    expect(RTX_PRO_5000.estimated?.some((e) => e.field === 'engines.0.matmulPeaks')).toBe(true);
  });

  it('4000 SFF is electrically x8: 24 GB, 432 GB/s, 770 sparse FP4 TOPS', () => {
    expect(RTX_PRO_4000_SFF.hostInterface).toEqual({ generation: 5, lanes: 8 });
    expect(RTX_PRO_4000_SFF.memory.capacityOptionsBytes).toEqual([24e9]);
    expect(RTX_PRO_4000_SFF.memory.bandwidthBytesPerSec).toBe(432e9);
    expect(peak(RTX_PRO_4000_SFF, 'fp4', '2:4')).toBe(770e12);
    expect(peak(RTX_PRO_4000_SFF, 'bf16')).toBe(96.25e12);
  });

  it.each([
    [RTX_PRO_6000_WS, 1872 / 2617],
    [RTX_PRO_6000_MAX_Q, 1417 / 2280],
  ] as const)(
    '%s optional sustained-clock axis scales compute and SFU, not memory',
    (device, factor) => {
      const axes = deviceUnlocks(device);
      expect(axes).toHaveLength(1);
      const modifiers = axes[0].options[0].deviceModifiers;
      for (const m of modifiers) expect(validateDeviceModifiers(device, [m])).toEqual([]);
      const changed = applyDeviceModifiers(device, modifiers);
      expect(validateDevice(changed)).toEqual([]);
      expect(changed.engines[0].sfuOpsPerSec).toBeCloseTo(
        device.engines[0].sfuOpsPerSec! * factor,
        -1,
      );
      expect(peak(changed, 'bf16')).toBeCloseTo(peak(device, 'bf16') * factor, -1);
      expect(changed.memory.bandwidthBytesPerSec).toBe(device.memory.bandwidthBytesPerSec);
    },
  );
});
