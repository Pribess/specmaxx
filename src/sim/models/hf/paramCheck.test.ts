import { describe, expect, it } from 'vitest';
import { countParams } from '../../ir/params';
import { REF_DENSE_8B } from '../reference';
import { fixtureRepoFiles } from '../__fixtures__/fixtures.testutil';
import { checkParams, estimateVisionParams } from './paramCheck';
import type { HfApiModelInfo, JsonObject } from './types';

const P = countParams(REF_DENSE_8B);
const EXCL = { mtpParamsEstimate: 1_000_000, visionParamsEstimate: 2_000_000 };
const ACTUAL = P + 3_000_000;

function api(parameters: Record<string, number>, total: number, base?: string): HfApiModelInfo {
  return { id: 'org/repo', safetensors: { parameters, total }, ...(base ? { cardData: { base_model: base } } : {}) };
}

describe('checkParams', () => {
  it('ok within ±2%', () => {
    const r = checkParams(REF_DENSE_8B, EXCL, api({ BF16: ACTUAL }, ACTUAL * 1.019));
    expect(r.status).toBe('ok');
    expect(r.actual).toBe(ACTUAL);
    expect(r.relError).toBeCloseTo(1 / 1.019 - 1, 10);
  });

  it('mismatch outside ±2%', () => {
    const r = checkParams(REF_DENSE_8B, EXCL, api({ BF16: 1 }, ACTUAL * 1.05));
    expect(r.status).toBe('mismatch');
    expect(r.note).toMatch(/-4\.76%/);
  });

  it('packed for U8/U32/I32 keys, pointing at the base model', () => {
    for (const key of ['U8', 'U32', 'I32']) {
      const r = checkParams(REF_DENSE_8B, EXCL, api({ BF16: 1, [key]: 2 }, ACTUAL / 3, 'org/base'), 'gptq');
      expect(r.status, key).toBe('packed');
      expect(r.note).toContain('org/base');
    }
  });

  it('packed for I8 with a packing quant method, but not for DeepSeek fp8 (I8 = one fp4 weight each)', () => {
    expect(checkParams(REF_DENSE_8B, EXCL, api({ I8: 1 }, ACTUAL / 2), 'compressed-tensors').status).toBe('packed');
    expect(checkParams(REF_DENSE_8B, EXCL, api({ I8: 1 }, ACTUAL), 'fp8').status).toBe('ok');
  });

  it('checks against the per-dtype sum when it exceeds safetensors.total (total without MTP)', () => {
    // Sum = ACTUAL, total 4% lower: against the total this would be a +4.2% mismatch.
    const r = checkParams(REF_DENSE_8B, EXCL, api({ BF16: ACTUAL - 10, F32: 10 }, ACTUAL / 1.042));
    expect(r.status).toBe('ok');
    expect(r.expected).toBe(ACTUAL);
    expect(r.relError).toBe(0);
    expect(r.note).toMatch(/differs from the per-dtype sum .*checked against the sum/);
    // Within 1e-4, or a sum below the total: the total is used as before.
    const close = checkParams(REF_DENSE_8B, EXCL, api({ BF16: ACTUAL * (1 + 5e-5) }, ACTUAL));
    expect([close.expected, close.relError]).toEqual([ACTUAL, 0]);
    expect(checkParams(REF_DENSE_8B, EXCL, api({ BF16: 1 }, ACTUAL)).expected).toBe(ACTUAL);
    // The packed rule runs before the sum rule (a U8 sum above the total counts packed words, not params)...
    expect(checkParams(REF_DENSE_8B, EXCL, api({ U8: ACTUAL * 2 }, ACTUAL / 2)).status).toBe('packed');
    // ...but after the tolerance check: a total that matches the IR is a parameter count whatever its keys.
    expect(checkParams(REF_DENSE_8B, EXCL, api({ U8: ACTUAL * 2 }, ACTUAL)).status).toBe('ok');
  });

  it('Nemotron-3.5 fixtures: Lightning uses the sum (MTP left out of total), Super-VL is packed', () => {
    const lightning = fixtureRepoFiles('nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16').api;
    const r = checkParams(REF_DENSE_8B, EXCL, lightning);
    expect(r.expected).toBe(32_913_266_240);
    expect(r.note).toContain('safetensors.total (31577937344)');
    const zuper = fixtureRepoFiles('nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4').api;
    expect(checkParams(REF_DENSE_8B, EXCL, zuper, 'modelopt_mixed').status).toBe('packed');
  });

  it('unavailable without safetensors', () => {
    expect(checkParams(REF_DENSE_8B, EXCL, { id: 'org/repo' }).status).toBe('unavailable');
    expect(checkParams(REF_DENSE_8B, EXCL, undefined).status).toBe('unavailable');
    const mistral = fixtureRepoFiles('mistralai/Mistral-Large-4-1T-A52B');
    expect(checkParams(REF_DENSE_8B, EXCL, mistral.api).status).toBe('unavailable');
  });
});

describe('estimateVisionParams', () => {
  const vision = (repo: string) => {
    const cfg = fixtureRepoFiles(repo).config as JsonObject;
    return cfg.vision_config as JsonObject;
  };

  it('matches the hand formula on a synthetic config', () => {
    // 2·(4·10² + 2·10·40) + (10·2²)² + 40·8 = 2·1200 + 1600 + 320
    expect(estimateVisionParams({ depth: 2, hidden_size: 10, intermediate_size: 40, spatial_merge_size: 2, out_hidden_size: 8 })).toBe(4320);
    // no merge size: 3·(4·10² + 3·10·40) + 10·16 (gated silu, projection to textHidden)
    expect(estimateVisionParams({ num_hidden_layers: 3, hidden_size: 10, intermediate_size: 40, hidden_act: 'silu' }, 16)).toBe(4960);
    expect(estimateVisionParams({ n_layers: 4 })).toBe(0);
    expect(estimateVisionParams(undefined)).toBe(0);
  });

  it('lands within ±20% of the measured towers where the generic shape fits', () => {
    // Measured tower sizes from the research findings (safetensors headers).
    const cases: [string, number][] = [
      ['Qwen/Qwen3.8-27B', 0.461e9],
      ['Qwen/Qwen3.8-Flash-Next', 0.449e9],
      ['zai-org/GLM-5.3-Flash', 0.564e9],
      ['mistralai/Mistral-Large-4-1T-A52B', 1.7e9],
    ];
    for (const [repo, measured] of cases) {
      const est = estimateVisionParams(vision(repo));
      expect(Math.abs(est / measured - 1), repo).toBeLessThan(0.2);
    }
  });
});
