import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import type { QuantDefaults } from '../../sim/models/hf/types';
import { QUANT_DTYPE_CHOICES, REFERENCE_QUANT_DEFAULTS, applyQuantOverride, hasQuantOverride, quantSummary } from './quant';

function defaults(repo: string, dir?: string): QuantDefaults {
  const r = mapHfModel(fixtureRepoFiles(repo, dir ? { dir } : {}));
  if (r.kind !== 'ok') throw new Error(`${repo}: ${r.kind}`);
  return r.quant;
}

describe('applyQuantOverride', () => {
  const nvfp4 = defaults('nvidia/Qwen3.8-27B-NVFP4', 'quant');
  const deepseek = defaults('deepseek-ai/DeepSeek-V4.1-Flash');

  it('returns the defaults unchanged without overrides', () => {
    expect(applyQuantOverride(nvfp4.quant, {})).toEqual(nvfp4.quant);
  });

  it('weights: one dtype for every linear module (modules dropped), LM head and embeddings kept', () => {
    expect(nvfp4.quant.modules?.attention?.weights.dtype).toBe('fp8_e4m3');
    const q = applyQuantOverride(nvfp4.quant, { weights: 'bf16' });
    expect(q.weights).toEqual({ dtype: 'bf16' });
    expect(q.modules).toBeUndefined();
    expect(q.lmHead).toEqual(nvfp4.quant.lmHead);
    expect(q.lmHeadGemmInput).toBe(nvfp4.quant.lmHeadGemmInput);
    expect(q.activations).toBe(nvfp4.quant.activations);
    expect(nvfp4.quant.modules).toBeDefined(); // not mutated
  });

  it('weights drop group-quant overhead too', () => {
    const q = applyQuantOverride(deepseek.quant, { weights: 'fp8_e4m3' });
    expect(q.weights).toEqual({ dtype: 'fp8_e4m3' });
    expect(q.modules).toBeUndefined();
    expect(q.lookupTables).toEqual(deepseek.quant.lookupTables);
  });

  it('activations: set, per-module gemmInput kept unless the weights are overridden too', () => {
    const a = applyQuantOverride(nvfp4.quant, { activations: 'fp8_e4m3' });
    expect(a.activations).toBe('fp8_e4m3');
    expect(a.modules?.dense_mlp?.gemmInput).toBe('nvfp4');
    const wa = applyQuantOverride(nvfp4.quant, { weights: 'fp8_e4m3', activations: 'fp8_e4m3' });
    expect(wa.modules).toBeUndefined();
  });

  it('KV: one dtype for the KV, compressed-KV and indexer-key caches', () => {
    expect(deepseek.quant.compressedKvCache).toEqual({ dtype: 'nvfp4' });
    expect(deepseek.quant.indexerKeyCache).toEqual({ dtype: 'mxfp4' });
    const q = applyQuantOverride(deepseek.quant, { kvCache: 'fp8_e4m3' });
    expect(q.kvCache).toEqual({ dtype: 'fp8_e4m3' });
    expect(q.compressedKvCache).toBeUndefined();
    expect(q.indexerKeyCache).toBeUndefined();
    expect(q.modules).toEqual(deepseek.quant.modules);
  });
});

describe('quant labels and choices', () => {
  it('summarizes the scheme or the overrides over it', () => {
    expect(quantSummary(REFERENCE_QUANT_DEFAULTS, {})).toBe('bf16 (unquantized)');
    expect(quantSummary(REFERENCE_QUANT_DEFAULTS, { weights: 'fp8_e4m3', kvCache: 'fp8_e4m3' })).toBe(
      'weights fp8_e4m3, KV fp8_e4m3 (over bf16 (unquantized))',
    );
    expect(hasQuantOverride({})).toBe(false);
    expect(hasQuantOverride({ activations: 'bf16' })).toBe(true);
  });

  it('offers every storage dtype but int32 and tf32', () => {
    expect(QUANT_DTYPE_CHOICES).not.toContain('int32');
    expect(QUANT_DTYPE_CHOICES).not.toContain('tf32');
    expect(QUANT_DTYPE_CHOICES).toContain('nvfp4');
    expect(QUANT_DTYPE_CHOICES).toContain('bf16');
  });
});
