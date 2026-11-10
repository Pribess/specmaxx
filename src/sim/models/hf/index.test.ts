import { describe, expect, it } from 'vitest';
import { REF_DENSE_8B } from '../reference';
import { countParams } from '../../ir/params';
import { fixtureRepoFiles, listFixtureRepos } from '../__fixtures__/fixtures.testutil';
import { mapHfModel, mergeQuantHints } from './index';
import type { FamilyEntry, HfRepoFiles, QuantDefaults } from './types';

const files = (config: unknown, api?: HfRepoFiles['api']): HfRepoFiles => ({
  id: 'org/repo',
  revision: 'main',
  config,
  ...(api ? { api } : {}),
});

/** A test family that returns REF_DENSE_8B and records what it was given. */
function fakeFamily(seen: { text?: unknown; top?: unknown }): FamilyEntry {
  return {
    family: 'fake',
    modelTypes: ['fake_text'],
    architectures: ['FakeForConditionalGeneration'],
    build: (text, top) => {
      seen.text = text;
      seen.top = top;
      return {
        kind: 'ok',
        model: REF_DENSE_8B,
        notes: ['family note'],
        mtpParamsEstimate: 10,
        visionParamsEstimate: 20,
        excludedParts: ['MTP layer', 'vision tower'],
        quantHints: { recurrentState: 'bf16' },
      };
    },
  };
}

describe('mapHfModel', () => {
  it('runs unwrap → dispatch → build → quant → param check', () => {
    const seen: { text?: unknown; top?: unknown } = {};
    const text = { model_type: 'fake_text', hidden_size: 4096 };
    const config = {
      architectures: ['FakeForConditionalGeneration'],
      model_type: 'fake',
      text_config: text,
      vision_config: { depth: 1 },
    };
    const total = countParams(REF_DENSE_8B) + 30;
    const r = mapHfModel(
      files(config, { id: 'org/repo', safetensors: { parameters: { BF16: total }, total } }),
      [fakeFamily(seen)],
    );
    expect(seen.text).toBe(text);
    expect(seen.top).toBe(config);
    if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
    expect(r.family).toBe('fake');
    expect(r.modelType).toBe('fake');
    expect(r.architecture).toBe('FakeForConditionalGeneration');
    expect(r.model).toBe(REF_DENSE_8B);
    expect(r.excluded).toEqual({
      mtpParamsEstimate: 10,
      visionParamsEstimate: 20,
      parts: ['MTP layer', 'vision tower'],
    });
    expect(r.paramCheck).toMatchObject({
      status: 'ok',
      expected: total,
      actual: total,
      relError: 0,
    });
    expect(r.quant.source).toBe('none');
    expect(r.quant.quant.weights).toEqual({ dtype: 'bf16' });
    expect(r.quant.quant.recurrentState).toBe('bf16');
    expect(r.notes).toEqual(['language model only; vision/audio towers excluded', 'family note']);
  });

  it('keeps a config / metadata read from another repo as data, not a note, and notes a non-ok param check', () => {
    const r = mapHfModel(
      { ...files({ model_type: 'fake_text' }), configFrom: 'org/base', metadataFrom: 'org/mirror' },
      [fakeFamily({})],
    );
    if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
    expect(r.paramCheck.status).toBe('unavailable');
    expect(r.configFrom).toBe('org/base');
    expect(r.metadataFrom).toBe('org/mirror');
    expect(r.notes.some((n) => /read from|org\/base|org\/mirror/.test(n))).toBe(false);
    expect(r.notes.some((n) => n.startsWith('param check (unavailable)'))).toBe(true);
  });

  it('passes family build errors through', () => {
    const bad: FamilyEntry = {
      ...fakeFamily({}),
      build: () => ({
        kind: 'invalid_config',
        errors: ['text_config.hidden_size: missing number'],
      }),
    };
    expect(mapHfModel(files({ model_type: 'fake_text' }), [bad])).toEqual({
      kind: 'invalid_config',
      errors: ['text_config.hidden_size: missing number'],
    });
  });

  it('reports unknown model types as unsupported with the supported list', () => {
    const r = mapHfModel(files({ model_type: 'llama_9', architectures: ['Llama9ForCausalLM'] }));
    expect(r).toMatchObject({
      kind: 'unsupported',
      modelType: 'llama_9',
      architecture: 'Llama9ForCausalLM',
    });
    if (r.kind !== 'unsupported') throw new Error('expected unsupported');
    expect(r.supported).toContain('glm_moe_dsa');
    expect(r.reason).toBeUndefined();
  });

  it('merges family quantHints except over fields the repo sets explicitly', () => {
    const derived: QuantDefaults = {
      quant: { weights: { dtype: 'nvfp4' }, activations: 'bf16', kvCache: { dtype: 'fp8_e4m3' } },
      scheme: 's',
      source: 'quantization_config',
      notes: [],
      explicit: ['kvCache'],
    };
    const r = mergeQuantHints(derived, { kvCache: { dtype: 'mxfp8' }, recurrentState: 'bf16' });
    expect(r.quant.quant.kvCache).toEqual({ dtype: 'fp8_e4m3' });
    expect(r.quant.quant.recurrentState).toBe('bf16');
    expect(r.quant.quant.weights).toEqual({ dtype: 'nvfp4' });
    expect(r.notes).toEqual(["family hint for kvCache overridden by the repo's quant metadata"]);
    expect(mergeQuantHints(derived, undefined)).toEqual({ quant: derived, notes: [] });
  });
  it('rejects a non-object config', () => {
    expect(mapHfModel(files('not json')).kind).toBe('invalid_config');
  });

  it('maps every base fixture to an ok result', () => {
    const repos = listFixtureRepos();
    expect(repos.length).toBeGreaterThanOrEqual(8);
    for (const repo of repos) {
      const r = mapHfModel(fixtureRepoFiles(repo));
      expect(r.kind, `${repo}: ${JSON.stringify(r).slice(0, 300)}`).toBe('ok');
    }
  });
});
