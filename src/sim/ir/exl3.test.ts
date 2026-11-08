import { describe, expect, it } from 'vitest';
import { RTX_5090 } from '../hw/presets/nvidia';
import { singleDevice } from '../hw/presets';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { simulate } from '../engine/simulate';
import { bitsPerElement, matrixStoredBytes, resolveGemmMath } from './dtype';
import { EXL3_COMPUTE_DTYPE, exl3Format } from './exl3';
import { gemmWeightBytes } from './metrics';
import type { QuantConfig } from './model';

describe('exl3Format', () => {
  it('stores bpw value bits per element plus fp16 side vectors, computes as fp16', () => {
    const f = exl3Format(2.5);
    expect(f).toEqual({ dtype: 'fp16', valueBitsPerElem: 2.5, perDimBits: 16 });
    expect(bitsPerElement(f)).toBe(2.5);
    expect(EXL3_COMPUTE_DTYPE).toBe('fp16');
  });

  it('clamps to the legal 1–8 range and treats 16 as unquantized fp16', () => {
    expect(exl3Format(0.2).valueBitsPerElem).toBe(1);
    expect(exl3Format(9).valueBitsPerElem).toBe(8);
    expect(exl3Format(16)).toEqual({ dtype: 'fp16' });
  });

  it('matrix bytes = n·k·bpw/8 + 16·(n + k)/8', () => {
    const n = 4096;
    const k = 11008;
    for (const bpw of [2.5, 4]) {
      const want = (n * k * bpw) / 8 + (16 * (n + k)) / 8;
      expect(matrixStoredBytes(exl3Format(bpw), n, k)).toBeCloseTo(want, 6);
      expect(gemmWeightBytes(3, n, k, exl3Format(bpw), 'dense')).toBeCloseTo(3 * want, 6);
    }
  });

  it('runs at the activation dtype’s peak for 16-bit activations, fp16 for fp8 activations', () => {
    expect(resolveGemmMath('fp16', 'bf16')).toBe('bf16');
    expect(resolveGemmMath('fp16', 'fp16')).toBe('fp16');
    expect(resolveGemmMath('fp16', 'fp8_e4m3')).toBe('fp16');
  });
});

describe('EXL3 on Qwen3.8-27B (dense): weight bytes and decode rate', () => {
  const r = mapHfModel(fixtureRepoFiles('Qwen/Qwen3.8-27B'));
  if (r.kind !== 'ok') throw new Error('fixture');
  const hardware = { device: RTX_5090, topology: singleDevice(RTX_5090) };
  const workload = { batch: 1, promptLen: 4096, decodeContextLen: 4096 };
  const run = (weights: QuantConfig['weights']) => {
    const quant: QuantConfig = { weights, activations: 'fp16', kvCache: { dtype: 'fp16' }, highPrecision: 'fp16', lmHead: weights };
    return simulate({ model: r.model, quant, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 }, workload, hardware });
  };

  it('resident weight bytes are linear in bpw: Δ = linearElems·Δbpw/8, plus the small side-vector overhead', () => {
    const bytes = (f: QuantConfig['weights']) => run(f).memory.perStage[0].weightBytes;
    // Linear elements (all matrices incl. the head, embeddings stay fp16 so they cancel): from the 16-bit and 0-bit extremes.
    const elems = ((bytes({ dtype: 'fp16' }) - bytes({ dtype: 'fp16', valueBitsPerElem: 0 })) * 8) / 16;
    expect(elems).toBeGreaterThan(20e9);
    const plain = (bpw: number) => bytes({ dtype: 'fp16', valueBitsPerElem: bpw });
    expect(plain(4) - plain(2.5)).toBeCloseTo((elems * 1.5) / 8, -3);
    expect(plain(2.5)).toBeCloseTo(plain(0) + (elems * 2.5) / 8, -3);
    const withVectors = bytes(exl3Format(2.5)) - plain(2.5);
    expect(withVectors).toBeGreaterThan(0);
    expect(withVectors / plain(2.5)).toBeLessThan(0.01); // side vectors < 1 % of the weights
  });

  it('decode tok/s ∝ 1/bpw in the memory-bound regime (batch 1)', () => {
    const lo = run(exl3Format(2.5));
    const hi = run(exl3Format(4));
    const ratio = lo.decode.tokensPerSec / hi.decode.tokensPerSec;
    // Embedding lookups, KV and activations do not scale with bpw, so a hair under 4/2.5 = 1.6.
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(1.6);
    expect(run(exl3Format(8)).decode.tokensPerSec).toBeLessThan(hi.decode.tokensPerSec);
  });
});
