import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import type { QuantDefaults } from '../../sim/models/hf/types';
import {
  EXL3_CUSTOM_KEY,
  EXL3_MODEL_KEY,
  QUANT_DTYPE_CHOICES,
  REFERENCE_QUANT_DEFAULTS,
  applyQuantOverride,
  exl3PickerKey,
  hasQuantOverride,
  quantSummary,
  resolvedDefault,
} from './quant';

function defaults(repo: string, dir?: string): QuantDefaults {
  const r = mapHfModel(fixtureRepoFiles(repo, dir ? { dir } : {}));
  if (r.kind !== 'ok') throw new Error(`${repo}: ${r.kind}`);
  return r.quant;
}

describe('applyQuantOverride', () => {
  const nvfp4 = defaults('nvidia/Qwen3.8-27B-NVFP4', 'quant');
  const deepseek = defaults('deepseek-ai/DeepSeek-V4.1-Flash');

  it('lookupTables: sets the table format only; the model default is the embeddings format (bf16)', () => {
    const q = applyQuantOverride(deepseek.quant, { lookupTables: 'nvfp4' });
    expect(q.lookupTables).toEqual({ dtype: 'nvfp4' });
    expect(q.weights).toEqual(deepseek.quant.weights);
    expect(resolvedDefault(nvfp4.quant, 'lookupTables').label).toBe('bf16');
    expect(hasQuantOverride({ lookupTables: 'nvfp4' })).toBe(true);
  });

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

describe('applyQuantOverride with EXL3 weights', () => {
  const qwen = defaults('Qwen/Qwen3.8-27B');
  const exl3Repo = defaults('turboderp/Qwen3.8-27B-exl3', 'quant');

  it('uses the override, else the checkpoint’s bits / head_bits, else 4 / 6', () => {
    const own = applyQuantOverride(qwen.quant, { weights: 'exl3', bpw: 2.5 }, qwen.exl3);
    expect(own.weights).toEqual({ dtype: 'fp16', valueBitsPerElem: 2.5, perDimBits: 16 });
    expect(own.lmHead).toEqual({ dtype: 'fp16', valueBitsPerElem: 6, perDimBits: 16 });
    expect(own.modules).toBeUndefined();
    expect(own.activations).toBe('fp16');
    const fromRepo = applyQuantOverride(exl3Repo.quant, { weights: 'exl3' }, exl3Repo.exl3);
    expect(fromRepo.weights.valueBitsPerElem).toBe(4);
    expect(fromRepo.lmHead?.valueBitsPerElem).toBe(5);
    expect(applyQuantOverride(qwen.quant, { weights: 'exl3', bpw: 3, headBpw: 16 }).lmHead).toEqual(
      { dtype: 'fp16' },
    );
  });

  it('labels the override with its bpw', () => {
    expect(quantSummary(qwen, { weights: 'exl3', bpw: 2.5 })).toBe(
      'weights exl3 2.5 bpw (head 6) (over bf16 (unquantized))',
    );
  });
});

describe('quant labels and choices', () => {
  it('summarizes the scheme or the overrides over it', () => {
    expect(quantSummary(REFERENCE_QUANT_DEFAULTS, {})).toBe('bf16 (unquantized)');
    expect(
      quantSummary(REFERENCE_QUANT_DEFAULTS, { weights: 'fp8_e4m3', kvCache: 'fp8_e4m3' }),
    ).toBe('weights fp8_e4m3, KV fp8_e4m3 (over bf16 (unquantized))');
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

describe('resolvedDefault', () => {
  it('names a single format for a bf16 model', () => {
    for (const f of ['weights', 'activations', 'kvCache'] as const) {
      expect(resolvedDefault(REFERENCE_QUANT_DEFAULTS.quant, f)).toEqual({ label: 'bf16' });
    }
  });
  it('summarises a mixed fp8/nvfp4 scheme and keeps the split in the detail', () => {
    const d = defaults('nvidia/Qwen3.8-27B-NVFP4', 'quant');
    expect(resolvedDefault(d.quant, 'weights')).toEqual({
      label: 'nvfp4 + fp8_e4m3',
      detail: 'nvfp4: other linear layers, dense_mlp; fp8_e4m3: attention, linear_attention',
    });
  });
  it('abbreviates three or more formats', () => {
    const d = defaults('deepseek-ai/DeepSeek-V4.1-Flash');
    const kv = resolvedDefault(d.quant, 'kvCache');
    expect(kv.label).toMatch(/^\S+ \+ \S+ \+ …$/);
    expect(kv.detail).toBeDefined();
  });
  it('names the EXL3 bits per weight', () => {
    const d = defaults('bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-2.5bpw', 'quant');
    expect(resolvedDefault(d.quant, 'weights').label).toBe('exl3 2.52bpw');
  });
});

describe('exl3PickerKey', () => {
  it('selects the preset matching the bpw, the checkpoint entry for its own non-preset bpw, else custom', () => {
    expect(exl3PickerKey(2.5, undefined, false)).toBe('exl3:2.5');
    expect(exl3PickerKey(4, 2.9, false)).toBe('exl3:4');
    expect(exl3PickerKey(2.9, 2.9, false)).toBe(EXL3_MODEL_KEY);
    expect(exl3PickerKey(3.7, 2.9, false)).toBe(EXL3_CUSTOM_KEY);
    expect(exl3PickerKey(2.5, undefined, true)).toBe(EXL3_CUSTOM_KEY);
  });
});

describe('KV cache format lists', () => {
  it('splits the panel formats from the advanced ones with no overlap, all valid dtypes', async () => {
    const {
      KV_MAIN_DTYPES,
      KV_ADVANCED_DTYPES,
      QUANT_DTYPE_CHOICES,
      isKvAdvancedDtype,
      isQuantDtype,
    } = await import('./quant');
    expect(KV_MAIN_DTYPES).toEqual(['bf16', 'fp8_e4m3']);
    for (const d of KV_ADVANCED_DTYPES) {
      expect(KV_MAIN_DTYPES).not.toContain(d);
      expect(isQuantDtype(d)).toBe(true);
    }
    expect([...KV_MAIN_DTYPES, ...KV_ADVANCED_DTYPES].sort()).toEqual(
      [...QUANT_DTYPE_CHOICES].sort(),
    );
    for (const d of ['nvfp4', 'mxfp4', 'fp4_e2m1']) expect(isKvAdvancedDtype(d)).toBe(true);
    expect(isKvAdvancedDtype('bf16')).toBe(false);
    expect(isKvAdvancedDtype(undefined)).toBe(false);
  });
});
