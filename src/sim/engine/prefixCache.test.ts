import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { ParallelConfig, QuantConfig } from '../ir/model';
import { REF_DENSE_8B } from '../models/reference';
import { prefillChunkShape } from './memory';
import { simulate } from './simulate';
import { cachedPrefixTokens, computedPromptTokens, type PhaseResult, type SimInput, type Workload } from './types';
import { validateInput } from './validate';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };
const P1: ParallelConfig = { tp: 1, pp: 1, dp: 1, ep: 1 };
const hardware = { device: RTX_5090, topology: singleDevice(RTX_5090) };
const W: Workload = { batch: 1, promptLen: 4096, decodeContextLen: 1024 };
const run = (workload: Workload) => simulate({ model: REF_DENSE_8B, quant: BF16, parallel: P1, workload, hardware });
const ops = (p: PhaseResult) => p.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));

describe('prefix-cache hit ratio (docs §7 "Prefix cache")', () => {
  it('token counts: cached = floor(P·h), computed = P − cached', () => {
    expect(cachedPrefixTokens({ promptLen: 1000, prefixCacheHitRatio: 0.5 })).toBe(500);
    expect(computedPromptTokens({ promptLen: 1001, prefixCacheHitRatio: 0.5 })).toBe(501);
    expect(computedPromptTokens({ promptLen: 1000 })).toBe(1000);
    expect(computedPromptTokens({ promptLen: 3, prefixCacheHitRatio: 0.9999 })).toBe(1);
  });

  it('identity at h = 0 (absent and explicit): bit-identical result, no computed rate', () => {
    const base = run(W);
    expect(run({ ...W, prefixCacheHitRatio: 0 })).toEqual(base);
    expect(base.prefill.computedTokensPerSec).toBeUndefined();
  });

  it('prefill time scales with new tokens while attention still reads the cached KV', () => {
    const full = run(W);
    const hit = run({ ...W, prefixCacheHitRatio: 0.5 });
    // GEMM FLOPs/time shrink by ~half; the weights are read once either way.
    const gemmSec = (p: PhaseResult) => ops(p).filter((o) => o.op.kind === 'gemm').reduce((s, o) => s + o.sec, 0);
    expect(gemmSec(hit.prefill)).toBeLessThan(gemmSec(full.prefill) * 0.6);
    expect(hit.prefill.latencySec).toBeLessThan(full.prefill.latencySec * 0.6);
    // Attention: FLOPs follow qLen·kvLen (new × all), so more than the 1/2·1/2 of a cache-less half prompt;
    // KV read bytes include the cached KV (kvLen = P), so they do not shrink with the new tokens.
    const attn = (p: PhaseResult) => ops(p).filter((o) => o.op.kind === 'attention');
    const sum = (p: PhaseResult, f: 'flops' | 'kvRead') =>
      attn(p).reduce((s, o) => s + (f === 'flops' ? o.flops : o.bytes.kvReadBytes), 0);
    const half = run({ ...W, promptLen: 2048 });
    expect(sum(hit.prefill, 'flops')).toBeGreaterThan(sum(half.prefill, 'flops') * 1.5);
    expect(sum(hit.prefill, 'flops')).toBeLessThan(sum(full.prefill, 'flops') * 0.8);
    expect(sum(hit.prefill, 'kvRead')).toBeGreaterThan(sum(half.prefill, 'kvRead'));
  });

  it('reports effective tok/s per prompt token and computed tok/s', () => {
    const hit = run({ ...W, prefixCacheHitRatio: 0.5 });
    expect(hit.prefill.computedTokensPerSec).toBeCloseTo(hit.prefill.tokensPerSec / 2, 6);
    expect(hit.prefill.tokensPerSec).toBeGreaterThan(run(W).prefill.tokensPerSec);
  });

  it('decode is unaffected', () => {
    expect(run({ ...W, prefixCacheHitRatio: 0.5 }).decode).toEqual(run(W).decode);
  });

  it('memory-fit chunk shape keys on the new tokens, with kvLen = P', () => {
    expect(prefillChunkShape(8192, 4, 4096, 2048)).toEqual({ seqs: 2, qLen: 2048, kvLen: 8192 });
    expect(prefillChunkShape(1000, 8, 8192)).toEqual({ seqs: 8, qLen: 1000, kvLen: 1000 });
  });

  it('validation rejects h outside [0, 1)', () => {
    const bad = (h: number): SimInput => ({
      model: REF_DENSE_8B, quant: BF16, parallel: P1, workload: { ...W, prefixCacheHitRatio: h }, hardware,
    });
    expect(JSON.stringify(validateInput(bad(1)))).toContain('prefixCacheHitRatio');
    expect(JSON.stringify(validateInput(bad(-0.1)))).toContain('prefixCacheHitRatio');
    expect(JSON.stringify(validateInput(bad(0.5)))).not.toContain('prefixCacheHitRatio');
  });
});
