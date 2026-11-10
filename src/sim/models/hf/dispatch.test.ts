import { describe, expect, it } from 'vitest';
import { fixtureRepoFiles, listFixtureRepos } from '../__fixtures__/fixtures.testutil';
import { dispatchFamily, supportedModelTypes } from './dispatch';
import { FAMILIES } from './families';
import { unwrapTextConfig } from './textConfig';
import type { FamilyEntry, HfConfigView } from './types';

function view(config: unknown): HfConfigView {
  const r = unwrapTextConfig(config);
  if (r.kind !== 'ok') throw new Error(r.errors.join('; '));
  return r.view;
}

/** Expected family of each base fixture. */
const EXPECTED: Record<string, string> = {
  'Qwen/Qwen3.8-27B': 'qwen',
  'Qwen/Qwen3.8-Flash-Next': 'qwen',
  'deepseek-ai/DeepSeek-V4.1-Flash': 'deepseek_v41',
  'deepseek-ai/DeepSeek-V4.1-Flash-1017': 'deepseek_v41',
  'deepseek-ai/DeepSeek-V4-Flash-0731': 'deepseek_v4',
  'zai-org/GLM-5.3-Flash': 'glm5_next',
  'zai-org/GLM-5.3-Flash-BF16': 'glm5_next',
  'zai-org/GLM-5.3': 'glm_moe_dsa',
  'zai-org/GLM-5.3-BF16': 'glm_moe_dsa',
  'mistralai/Mistral-Large-4-1T-A52B': 'mistral_large4',
  'thinkingmachines/Inkling-Small': 'inkling',
  'MiniMaxAI/MiniMax-M3': 'minimax',
  'MiniMaxAI/MiniMax-M3.1-Flash': 'minimax',
  'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16': 'nemotron_h',
  'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4': 'nemotron_h',
  'unsloth/Muse-Spark-Open-816B-A42B': 'muse_spark',
  'moonshotai/Kimi-K3': 'kimi_linear',
  'moonshotai/Kimi-K3.1-Pro': 'kimi_linear',
};

describe('dispatchFamily', () => {
  it('routes every base fixture by its top-level architecture', () => {
    expect(listFixtureRepos().sort()).toEqual(Object.keys(EXPECTED).sort());
    for (const [repo, family] of Object.entries(EXPECTED)) {
      const hit = dispatchFamily(view(fixtureRepoFiles(repo).config));
      expect(hit?.entry.family, repo).toBe(family);
      expect(hit?.key, repo).toBe('architecture');
    }
  });

  it('routes every claimed model type, top-level and inside text_config', () => {
    for (const f of FAMILIES) {
      for (const t of f.modelTypes) {
        expect(dispatchFamily(view({ model_type: t }))?.entry.family).toBe(f.family);
        const nested = dispatchFamily(
          view({ model_type: 'wrapper_x', text_config: { model_type: t } }),
        );
        expect(nested?.entry.family).toBe(f.family);
        expect(nested?.key).toBe('text_config.model_type');
      }
    }
  });

  it('falls back from an unknown architecture to the model type, then to text_config', () => {
    const byType = dispatchFamily(
      view({ architectures: ['UnknownForCausalLM'], model_type: 'glm_moe_dsa' }),
    );
    expect(byType).toMatchObject({ key: 'model_type', value: 'glm_moe_dsa' });
    const byTextArch = dispatchFamily(
      view({
        model_type: 'wrapper_x',
        text_config: { architectures: ['MiniMaxM3SparseForCausalLM'] },
      }),
    );
    expect(byTextArch).toMatchObject({
      key: 'text_config.architecture',
      entry: { family: 'minimax' },
    });
  });

  it('returns null for unknown types (e.g. llama_9, a DSpark draft on qwen3)', () => {
    expect(dispatchFamily(view({ model_type: 'llama_9' }))).toBeNull();
    expect(
      dispatchFamily(view({ architectures: ['DSparkDraftModel'], model_type: 'qwen3' })),
    ).toBeNull();
    expect(dispatchFamily(view({}))).toBeNull();
  });

  it('never uses vision_config.model_type', () => {
    expect(
      dispatchFamily(view({ model_type: 'llama_9', vision_config: { model_type: 'qwen4_exp' } })),
    ).toBeNull();
  });

  it('claims each key in at most one family', () => {
    const seen = new Map<string, string>();
    for (const f of FAMILIES) {
      for (const k of [...f.modelTypes, ...f.architectures]) {
        expect(seen.get(k), `${k} claimed twice`).toBeUndefined();
        seen.set(k, f.family);
      }
    }
  });

  it('uses the given table', () => {
    const only: FamilyEntry[] = [
      {
        family: 'only',
        modelTypes: ['llama_9'],
        architectures: [],
        build: () => ({ kind: 'unsupported', reason: '' }),
      },
    ];
    expect(dispatchFamily(view({ model_type: 'llama_9' }), only)?.entry.family).toBe('only');
    expect(supportedModelTypes(only)).toEqual(['llama_9']);
    expect(supportedModelTypes()).toContain('qwen3_5');
  });
});
