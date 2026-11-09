import { describe, expect, it } from 'vitest';
import { countParams } from '../../sim/ir/params';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import { modelKey } from './buildInput';
import { DEFAULT_MODEL_ID, MODEL_SHORTLIST } from './modelShortlist';

describe('bundled default model', () => {
  it('maps the committed Qwen3.8-27B fixture under the default choice key', () => {
    expect(BUNDLED_DEFAULT_MODEL.key).toBe(modelKey({ kind: 'hf', id: DEFAULT_MODEL_ID }));
    expect(BUNDLED_DEFAULT_MODEL.label).toBe('Qwen/Qwen3.8-27B');
    expect(BUNDLED_DEFAULT_MODEL.model.layers).toHaveLength(64);
    expect(BUNDLED_DEFAULT_MODEL.quant.quant.weights.dtype).toBe('bf16');
  });

  it('has about 27B parameters', () => {
    const params = countParams(BUNDLED_DEFAULT_MODEL.model);
    expect(params).toBeGreaterThan(26e9);
    expect(params).toBeLessThan(29e9);
  });
});

describe('model shortlist', () => {
  it("holds the user's models (Inkling 2 Small left out until its repo is on HF), the default among them, unique ids", () => {
    expect(MODEL_SHORTLIST.map((m) => m.name)).toEqual([
      'Qwen3.8-Flash-Next',
      'Qwen3.8-27B',
      'DeepSeek-V4.1-Flash',
      'DeepSeek-V4-Flash-0731',
      'GLM-5.3 Flash',
      'GLM-5.3',
      'Mistral Large 4',
      'Minimax M3',
    ]);
    expect(new Set(MODEL_SHORTLIST.map((m) => m.id)).size).toBe(8);
    expect(MODEL_SHORTLIST.some((m) => m.id.startsWith('thinkingmachines/'))).toBe(false);
    expect(MODEL_SHORTLIST.some((m) => m.id === DEFAULT_MODEL_ID)).toBe(true);
  });
});
