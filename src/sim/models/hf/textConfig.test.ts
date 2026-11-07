import { describe, expect, it } from 'vitest';
import { fixtureRepoFiles } from '../__fixtures__/fixtures.testutil';
import { unwrapTextConfig } from './textConfig';
import type { HfConfigView } from './types';

function view(config: unknown): HfConfigView {
  const r = unwrapTextConfig(config);
  if (r.kind !== 'ok') throw new Error(r.errors.join('; '));
  return r.view;
}

describe('unwrapTextConfig', () => {
  it('unwraps a nested text_config and records the wrapper', () => {
    const v = view(fixtureRepoFiles('Qwen/Qwen3.8-Flash-Next').config);
    expect(v.nested).toBe(true);
    expect(v.text.hidden_size).toBeTypeOf('number');
    expect(v.topModelType).toBe('qwen4_exp');
    expect(v.topArchitectures).toEqual(['Qwen4ExpForConditionalGeneration']);
    expect(v.textModelType).toBe('qwen4_exp_text');
    // Qwen4Exp's vision type equals the top-level type; it is kept apart, never used as the text type.
    expect(v.visionConfig?.model_type).toBe('qwen4_exp');
    expect(v.notes).toContain('language model only; vision/audio towers excluded');
  });

  it('keeps a flat config as both top and text', () => {
    const v = view(fixtureRepoFiles('zai-org/GLM-5.3').config);
    expect(v.nested).toBe(false);
    expect(v.text).toBe(v.top);
    expect(v.topModelType).toBe('glm_moe_dsa');
    expect(v.textModelType).toBeNull();
    expect(v.textArchitectures).toEqual([]);
    expect((v.quantizationConfig as { quant_method?: string }).quant_method).toBe('fp8');
    expect(v.notes).toEqual([]);
  });

  it('handles text configs without model_type (MiniMax, Inkling)', () => {
    const mm = view(fixtureRepoFiles('MiniMaxAI/MiniMax-M3').config);
    expect(mm.nested).toBe(true);
    expect(mm.topModelType).toBe('minimax_m3_vl');
    expect(mm.textModelType).toBeNull();
    expect(mm.textArchitectures).toEqual(['MiniMaxM3SparseForCausalLM']);

    const ink = view(fixtureRepoFiles('thinkingmachines/Inkling-Small').config);
    expect(ink.topModelType).toBe('inkling_mm_model');
    expect(ink.textModelType).toBeNull();
    expect(ink.notes).toContain('language model only; vision/audio towers excluded');
  });

  it('reads quantization_config from text_config when the top level has none', () => {
    const v = view({ model_type: 'm', text_config: { model_type: 't', quantization_config: { quant_method: 'fp8' } } });
    expect(v.quantizationConfig).toEqual({ quant_method: 'fp8' });
  });

  it('rejects non-object configs and text_configs', () => {
    expect(unwrapTextConfig(null)).toEqual({ kind: 'invalid_config', errors: ['config.json: expected a JSON object'] });
    expect(unwrapTextConfig({ text_config: 3 })).toEqual({ kind: 'invalid_config', errors: ['text_config: expected object'] });
  });
});
