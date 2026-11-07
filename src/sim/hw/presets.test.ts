import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_PCIE_HOSTBOUNCE, SYNTH_UNIFIED } from '../__fixtures__/synthetic';
import { DEVICE_PRESETS, RTX_5090, singleDevice } from './presets';
import { numericLeafPaths, validateDevice, validateLink, validateTopology } from './validate';
import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { DeviceSpec } from './types';

function peak(d: DeviceSpec, input: MathFormat, acc: AccumulateType, sparsity: 'dense' | '2:4'): number {
  const p = d.engines[0].matmulPeaks.find((x) => x.input === input && x.accumulate === acc && x.sparsity === sparsity);
  if (p === undefined) throw new Error(`no entry ${input}/${acc}/${sparsity}`);
  return p.flopsPerSec;
}

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

describe('preset validation (T16)', () => {
  it('every preset passes validateDevice', () => {
    for (const d of DEVICE_PRESETS) expect(validateDevice(d)).toEqual([]);
    expect(validateTopology(singleDevice(RTX_5090))).toEqual([]);
  });

  it('RTX_5090: every 2:4 peak is 2 × its dense peak', () => {
    for (const p of RTX_5090.engines[0].matmulPeaks.filter((x) => x.sparsity === '2:4')) {
      // rel 1e-3: Table 3 rounds to 0.1 TFLOPS, so TF32 reads 104.8 / 209.5 (exact 104.75 / 209.5).
      expectClose(p.flopsPerSec, 2 * peak(RTX_5090, p.input, p.accumulate, 'dense'), 1e-3);
    }
  });

  it('RTX_5090 GeForce rates: fp16-acc fp16 = 2 × fp32-acc fp16; fp8 fp32-acc = 2 × bf16; fp4 = 2 × fp8 fp32-acc', () => {
    expect(peak(RTX_5090, 'fp16', 'fp16', 'dense')).toBe(2 * peak(RTX_5090, 'fp16', 'fp32', 'dense')); // 419 = 2·209.5
    expect(peak(RTX_5090, 'fp8', 'fp32', 'dense')).toBe(2 * peak(RTX_5090, 'bf16', 'fp32', 'dense')); // 419 = 2·209.5
    expect(peak(RTX_5090, 'fp4', 'fp32', 'dense')).toBe(4 * peak(RTX_5090, 'fp8', 'fp32', 'dense')); // 1676 = 4·419
    expect(peak(RTX_5090, 'fp4', 'fp32', '2:4')).toBe(3352e12); // the "3352 AI TOPS" headline
  });

  it('RTX_5090 memory and L2 as quoted', () => {
    expect(RTX_5090.memory.bandwidthBytesPerSec).toBe(512 * 28e9 / 8); // 512-bit × 28 Gbps = 1792 GB/s
    expect(RTX_5090.memory.capacityOptionsBytes).toEqual([32e9]);
    expect(RTX_5090.l2Bytes).toBe(100_663_296); // 98,304 KB × 1024
    expect(RTX_5090.engines[0].computeUnits).toBe(170);
  });

  it('fixtures validate too', () => {
    expect(validateDevice(SYNTH_GPU)).toEqual([]);
    expect(validateDevice(SYNTH_UNIFIED)).toEqual([]);
    expect(validateTopology(SYNTH_CLUSTER)).toEqual([]);
    expect(validateTopology(SYNTH_PCIE_HOSTBOUNCE)).toEqual([]);
  });
});

describe('validate rules', () => {
  it('numericLeafPaths skips provenance keys', () => {
    const paths = numericLeafPaths(RTX_5090).map((l) => l.path);
    expect(paths).toContain('engines.0.matmulPeaks.0.flopsPerSec');
    expect(paths).toContain('memory.capacityOptionsBytes.0');
    expect(paths.some((p) => p.startsWith('sources') || p.startsWith('estimated'))).toBe(false);
  });

  it('an uncovered number is an error', () => {
    const d: DeviceSpec = { ...SYNTH_GPU, sources: [{ url: 'synthetic://test', title: 't', fields: ['engines'] }] };
    expect(validateDevice(d).some((e) => e.includes('memory.bandwidthBytesPerSec is not covered'))).toBe(true);
  });

  it('sparse below dense, missing fp32-acc entry, misaligned tile, empty capacity, duplicate engine id', () => {
    const e = SYNTH_GPU.engines[0];
    const bad: DeviceSpec = {
      ...SYNTH_GPU,
      engines: [
        {
          ...e,
          matmulPeaks: [
            { input: 'bf16', accumulate: 'fp32', sparsity: 'dense', flopsPerSec: 100e12 },
            { input: 'bf16', accumulate: 'fp32', sparsity: '2:4', flopsPerSec: 50e12 },
            { input: 'fp8', accumulate: 'fp16', sparsity: 'dense', flopsPerSec: 200e12 },
          ],
          gemmTiling: { default: { ...e.gemmTiling.default, ctaTiles: [{ m: 100, n: 128 }] } },
        },
        e,
      ],
      memory: { ...SYNTH_GPU.memory, capacityOptionsBytes: [] },
    };
    const errors = validateDevice(bad);
    expect(errors.some((x) => x.includes('below its dense'))).toBe(true);
    expect(errors.some((x) => x.includes('fp8 has peaks but no fp32-accumulate'))).toBe(true);
    expect(errors.some((x) => x.includes('not a positive multiple of mmaM'))).toBe(true);
    expect(errors.some((x) => x.includes('capacityOptionsBytes is empty'))).toBe(true);
    expect(errors.some((x) => x.includes("duplicate engine id 'gpu'"))).toBe(true);
  });

  it('hostBounce iff host_bounce', () => {
    const link = SYNTH_PCIE_HOSTBOUNCE.intraNode;
    if (link === undefined) throw new Error('fixture has an intraNode link');
    expect(validateLink(link)).toEqual([]);
    expect(validateLink({ ...link, p2p: 'direct' }).some((x) => x.includes('hostBounce must be present iff'))).toBe(true);
  });
});
