import { describe, expect, it } from 'vitest';
import { SYNTH_GPU } from '../__fixtures__/synthetic';
import type { ElementwiseOp, GemmOp } from '../ir/ops';
import { elementwiseBytesUnderFusion, engineMemBandwidthBytesPerSec, opTime, roofline, type OpTimeContext } from './opTime';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const SYNTH_GPU_ENGINE = SYNTH_GPU.engines[0];
/** SYNTH_GPU: bf16/fp32 dense 100e12 FLOP/s, vector bf16 20e12, 100 units, tile 128×128 (mma 64×8×16), 1e12 B/s. */
const CTX: OpTimeContext = {
  engine: SYNTH_GPU_ENGINE,
  memBandwidthBytesPerSec: engineMemBandwidthBytesPerSec(SYNTH_GPU_ENGINE, SYNTH_GPU.memory.bandwidthBytesPerSec),
  collectivePaths: { tp: { kind: 'none' }, ep: { kind: 'none' }, pp: { kind: 'none' } },
  collectiveAlgorithm: 'best',
  elementwiseFusion: 'none',
  warnings: [],
};

function gemm(m: number, n: number, k: number): GemmOp {
  return {
    kind: 'gemm',
    name: 'test.gemm',
    batch: 1,
    m,
    n,
    k,
    inputDtype: 'bf16',
    weight: { dtype: 'bf16' },
    outputDtype: 'bf16',
    math: 'bf16',
    accumulate: 'fp32',
    sparsity: 'dense',
  };
}

describe('T6 roofline (SYNTH_GPU)', () => {
  it('roofline picks the larger term; a tie is compute-bound', () => {
    expect(roofline(1e-3, 2e-3)).toEqual({ sec: 2e-3, bound: 'memory' });
    expect(roofline(1e-3, 1e-3)).toEqual({ sec: 1e-3, bound: 'compute' });
  });

  it('balanced elementwise op: 20e9 FLOP at 20e12 and 1e9 B at 1e12 → both terms 1e-3 s', () => {
    // The plan's "100e9 FLOP / 1e9 B" case is shape-independent only for a vector op, so it is expressed here at
    // the bf16 vector peak (20e12): 1e9 elems · 20 FLOP = 20e9 FLOP → 1e-3 s; 1e9 B / 1e12 B/s → 1e-3 s.
    const op: ElementwiseOp = {
      kind: 'elementwise',
      name: 'test.ew',
      fn: 'residual_add',
      elems: 1e9,
      flopsPerElem: 20,
      dtype: 'bf16',
      activationReadBytes: 0.5e9,
      activationWriteBytes: 0.5e9,
      weightBytes: 0,
      stateReadBytes: 0,
      stateWriteBytes: 0,
    };
    const r = opTime(op, CTX);
    expectClose(r.computeSec, 1e-3);
    expectClose(r.memorySec, 1e-3);
    expectClose(r.sec, 1e-3);
    // Ideal fusion keeps only weight/state bytes: memory term 0, compute-bound.
    const fused = opTime(op, { ...CTX, elementwiseFusion: 'ideal' });
    expect(fused.memorySec).toBe(0);
    expect(fused.bound).toBe('compute');
    expect(elementwiseBytesUnderFusion(op, 'ideal').totalBytes).toBe(0);
  });

  it('GEMM n = k = 4096, m = 1 is memory-bound', () => {
    const r = opTime(gemm(1, 4096, 4096), CTX);
    // memory: 4096·4096·2 B weights + 2·4096·2 B in/out = 33,570,816 B / 1e12 = 33.57 µs
    expectClose(r.memorySec, (4096 * 4096 * 2 + 2 * 4096 * 2) / 1e12);
    // compute: swap-AB, 32 tiles of 128×8 in one wave: 2·128·8·4096 / (100e12/100) = 8.39 µs
    expectClose(r.computeSec, (2 * 128 * 8 * 4096) / 1e12);
    expect(r.bound).toBe('memory');
  });

  it('GEMM m = n = k = 4096 is compute-bound with efficiency 1024/1100', () => {
    const r = opTime(gemm(4096, 4096, 4096), CTX);
    // 32·32 = 1024 tiles of 128×128 on 100 slots → 11 waves = 1100 tile slots → efficiency 1024/1100
    expectClose(r.efficiency ?? 0, 1024 / 1100);
    // compute = 2·4096³ / (100e12 · 1024/1100) = 1.4763 ms; memory = 3·4096²·2 B / 1e12 = 0.1007 ms
    expectClose(r.computeSec, (2 * 4096 ** 3) / (100e12 * (1024 / 1100)));
    expect(r.bound).toBe('compute');
    expect(r.peakFlopsPerSec).toBe(100e12);
    expect(r.usedInput).toBe('bf16');
  });

  it('an engine memory cap lowers the bandwidth: min(device, cap)', () => {
    expect(engineMemBandwidthBytesPerSec({ ...SYNTH_GPU_ENGINE, memBandwidthCapBytesPerSec: 0.25e12 }, 1e12)).toBe(0.25e12);
    expect(engineMemBandwidthBytesPerSec(SYNTH_GPU_ENGINE, 1e12)).toBe(1e12);
  });
});
