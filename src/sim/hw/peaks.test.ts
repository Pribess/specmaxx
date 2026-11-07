import { describe, expect, it } from 'vitest';
import { SYNTH_UNIFIED } from '../__fixtures__/synthetic';
import { matmulPeakFlopsPerSec, vectorPeakFlopsPerSec } from './peaks';
import { RTX_5090 } from './presets/nvidia';
import type { AccumulateType, MathFormat } from '../ir/dtype';

const GPU = RTX_5090.engines[0];

/** The preset's own table entry (tests derive from the preset, not from re-typed numbers). */
function entry(input: MathFormat, accumulate: AccumulateType, sparsity: 'dense' | '2:4'): number {
  const p = GPU.matmulPeaks.find((x) => x.input === input && x.accumulate === accumulate && x.sparsity === sparsity);
  if (p === undefined) throw new Error(`no entry ${input}/${accumulate}/${sparsity}`);
  return p.flopsPerSec;
}

describe('matmulPeakFlopsPerSec on RTX_5090 (T5b)', () => {
  it('exact hits push no warning', () => {
    const cases: [MathFormat, AccumulateType, 'dense' | '2:4', number][] = [
      ['bf16', 'fp32', 'dense', 209.5e12], // Table 3: BF16 Tensor with FP32 accumulate, dense
      ['fp16', 'fp16', 'dense', 419e12], // FP16 with FP16 accumulate
      ['fp16', 'fp32', 'dense', 209.5e12], // FP16 with FP32 accumulate (GeForce half rate)
      ['fp8', 'fp32', 'dense', 419e12], // FP8 with FP32 accumulate
      ['fp4', 'fp32', '2:4', 3352e12], // FP4 sparse = the "3352 AI TOPS" headline
    ];
    for (const [input, acc, sparsity, expected] of cases) {
      const warnings: string[] = [];
      const r = matmulPeakFlopsPerSec(GPU, input, acc, sparsity, warnings);
      expect(r.flopsPerSec).toBe(expected);
      expect(r.flopsPerSec).toBe(entry(input, acc, sparsity));
      expect(warnings).toEqual([]);
    }
  });

  it('int4 falls back to int8 (838e12) with a warning', () => {
    const warnings: string[] = [];
    const r = matmulPeakFlopsPerSec(GPU, 'int4', 'int32', 'dense', warnings);
    expect(r.flopsPerSec).toBe(838e12); // Table 3: INT8 Tensor TOPS dense
    expect(r.usedInput).toBe('int8');
    expect(warnings).toHaveLength(1);
  });

  it('fp16-accumulate request for bf16 uses the fp32 entry', () => {
    const warnings: string[] = [];
    const r = matmulPeakFlopsPerSec(GPU, 'bf16', 'fp16', 'dense', warnings);
    expect(r.usedAccumulate).toBe('fp32');
    expect(r.flopsPerSec).toBe(entry('bf16', 'fp32', 'dense'));
    expect(warnings).toHaveLength(1);
  });

  it('fp4 with fp16 accumulate falls back to fp4/fp32; fp6 (no rate published) falls back to fp8', () => {
    const w1: string[] = [];
    expect(matmulPeakFlopsPerSec(GPU, 'fp4', 'fp16', 'dense', w1).flopsPerSec).toBe(entry('fp4', 'fp32', 'dense'));
    const w2: string[] = [];
    const r = matmulPeakFlopsPerSec(GPU, 'fp6', 'fp32', 'dense', w2);
    expect(r.usedInput).toBe('fp8');
    expect(r.flopsPerSec).toBe(entry('fp8', 'fp32', 'dense'));
    expect(w2).toHaveLength(1);
  });

  it('a missing 2:4 entry falls back to dense', () => {
    const warnings: string[] = [];
    // SYNTH_UNIFIED gpu lists bf16 dense only (50e12).
    const r = matmulPeakFlopsPerSec(SYNTH_UNIFIED.engines[0], 'bf16', 'fp32', '2:4', warnings);
    expect(r).toEqual({ flopsPerSec: 50e12, usedInput: 'bf16', usedAccumulate: 'fp32', usedSparsity: 'dense' });
    expect(warnings).toHaveLength(1);
  });

  it('end of chain: an NPU with no bf16/fp16 peak uses its vector fp32 peak', () => {
    const warnings: string[] = [];
    const npu = SYNTH_UNIFIED.engines[1];
    const r = matmulPeakFlopsPerSec(npu, 'bf16', 'fp32', 'dense', warnings);
    expect(r.flopsPerSec).toBe(1e12); // npu vector fp32
    expect(r.usedInput).toBe('fp32');
    expect(warnings).toHaveLength(1);
  });
});

describe('vectorPeakFlopsPerSec', () => {
  it('bf16/fp16 entries when listed, else fp32', () => {
    expect(vectorPeakFlopsPerSec(GPU, 'bf16')).toBe(104.8e12); // Table 3: Peak BF16 TFLOPS (non-Tensor)
    expect(vectorPeakFlopsPerSec(GPU, 'fp32')).toBe(104.8e12);
    expect(vectorPeakFlopsPerSec(GPU, 'fp8_e4m3')).toBe(104.8e12); // computed in fp32
    expect(vectorPeakFlopsPerSec(SYNTH_UNIFIED.engines[0], 'bf16')).toBe(5e12); // no bf16 entry → fp32
  });
});
