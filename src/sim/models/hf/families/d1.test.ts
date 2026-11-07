import { describe, expect, it } from 'vitest';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { modelSpecErrors } from '../../../ir/validateModel';
import { fixtureRepoFiles, listFixtureRepos } from '../../__fixtures__/fixtures.testutil';
import { mapHfModel } from '../index';
import type { JsonObject } from '../types';

const targets = [
  // Header-derived text-only counts; published sizes are 27B, 125B+51B, 428B and 276B respectively.
  { id: 'Qwen/Qwen3.8-27B', total: 26_895_998_464, active: 26_895_998_464, excludeEmbeddings: 0, linear: 48, sparse: 0, moe: 0, swa: 0 },
  { id: 'Qwen/Qwen3.8-Flash-Next', total: 176_943_899_520, active: 6.64e9, excludeEmbeddings: 1, linear: 36, sparse: 12, moe: 48, swa: 0 },
  { id: 'MiniMaxAI/MiniMax-M3', total: 426.174e9, active: 23.5e9, excludeEmbeddings: 2, linear: 0, sparse: 57, moe: 57, swa: 0 },
  { id: 'thinkingmachines/Inkling-Small', total: 263.66e9, active: 12e9, excludeEmbeddings: 0, linear: 0, sparse: 0, moe: 40, swa: 35 },
];

describe('Qwen, MiniMax and Inkling fixture mappings', () => {
  for (const target of targets) it(`${target.id}: counts, layer mix and checkpoint comparison`, () => {
    const result = mapHfModel(fixtureRepoFiles(target.id));
    expect(result.kind).toBe('ok');
    if (result.kind !== 'ok') throw new Error(JSON.stringify(result));
    const { model } = result;
    const total = countParams(model);
    const active = activeParamsPerToken(model) - target.excludeEmbeddings * model.hidden * model.vocab;
    expect(Math.abs(total / target.total - 1), `total ${total} versus ${target.total}`).toBeLessThan(0.005);
    expect(Math.abs(active / target.active - 1), `active ${active} versus ${target.active}`).toBeLessThan(0.015);
    expect(modelSpecErrors(model)).toEqual([]);
    expect(result.paramCheck.status).toBe('ok');
    const blocks = model.layers.flatMap(l => l.blocks);
    expect(blocks.filter(b => b.kind === 'linear_attention')).toHaveLength(target.linear);
    expect(blocks.filter(b => b.kind === 'gqa_attention' && b.sparse)).toHaveLength(target.sparse);
    expect(blocks.filter(b => b.kind === 'moe')).toHaveLength(target.moe);
    expect(blocks.filter(b => b.kind === 'gqa_attention' && b.slidingWindow)).toHaveLength(target.swa);
    expect(result.notes.join(' ')).toMatch(/MTP/);
    expect(result.notes.join(' ')).toMatch(/vision/);
  });

  it('Qwen PLE, hyper mixers, pooled QSA and recurrent-state dtype are explicit', () => {
    const result = mapHfModel(fixtureRepoFiles('Qwen/Qwen3.8-Flash-Next'));
    if (result.kind !== 'ok') throw new Error(JSON.stringify(result));
    expect(result.model.residualStreams).toBe(4);
    expect(result.model.finalNorm).toBe('none');
    expect(result.quant.quant.recurrentState).toBe('fp32');
    const blocks = result.model.layers.flatMap(l => l.blocks);
    expect(blocks.filter(b => b.kind === 'hc_pre')).toHaveLength(97);
    expect(result.model.layers[1].blocks[0]).toMatchObject({ kind: 'ngram_embedding', rowsPerToken: 16, rowDim: 160, outputDim: 10240 });
    expect(blocks.filter(b => b.kind === 'ngram_embedding')).toHaveLength(1);
    expect(blocks.find(b => b.kind === 'gqa_attention')).toMatchObject({ outputGate: true, sparse: { indexer: { keyPoolTokens: 4, topK: 2048, headWeights: false } } });
  });

  it('MiniMax M3.1 sibling has sparse attention in all layers and no MTP estimate', () => {
    const result = mapHfModel(fixtureRepoFiles('MiniMaxAI/MiniMax-M3.1-Flash'));
    if (result.kind !== 'ok') throw new Error(JSON.stringify(result));
    expect(modelSpecErrors(result.model)).toEqual([]);
    const blocks = result.model.layers.flatMap(l => l.blocks);
    expect(blocks.filter(b => b.kind === 'gqa_attention' && b.sparse)).toHaveLength(60);
    expect(blocks.find(b => b.kind === 'gqa_attention')).toMatchObject({ sparse: { indexer: { topK: 16 * 128 } } });
    expect(result.excluded.mtpParamsEstimate).toBe(0);
    expect(blocks.find(b => b.kind === 'mlp')).toMatchObject({ intermediate: 12288 });
  });

  it('Inkling has relative bias, untied embedding and all four short convs per layer', () => {
    const result = mapHfModel(fixtureRepoFiles('thinkingmachines/Inkling-Small'));
    if (result.kind !== 'ok') throw new Error(JSON.stringify(result));
    expect(result.model.tieEmbeddings).toBe(false);
    const blocks = result.model.layers.flatMap(l => l.blocks);
    expect(blocks.filter(b => b.kind === 'short_conv')).toHaveLength(84);
    expect(blocks.filter(b => b.kind === 'gqa_attention').every(b => b.rope === false && b.kvConvKernel === 4)).toBe(true);
    expect(result.model.layers[0].blocks.find(b => b.kind === 'gqa_attention')).toMatchObject({ relativeBias: { relDim: 16, extentTokens: 512 } });
    expect(result.model.layers[5].blocks.find(b => b.kind === 'gqa_attention')).toMatchObject({ relativeBias: { relDim: 16, extentTokens: 1024 } });
    expect(result.excluded.mtpParamsEstimate).toBeGreaterThan(2e9);
    expect(result.excluded.mtpParamsEstimate).toBeLessThan(2.4e9);
  });

  it('Qwen schedule fallback and sibling dimensions come from config, never the repo name', () => {
    const files = fixtureRepoFiles('Qwen/Qwen3.8-27B');
    const config = structuredClone(files.config) as JsonObject;
    const text = config.text_config as JsonObject;
    delete text.layer_types;
    text.num_hidden_layers = 8;
    text.full_attention_interval = 2;
    text.intermediate_size = 4096;
    const result = mapHfModel({ ...files, id: 'other/entirely-unrelated-name', config });
    if (result.kind !== 'ok') throw new Error(JSON.stringify(result));
    expect(result.model.layers).toHaveLength(8);
    expect(result.model.layers.flatMap(l => l.blocks).filter(b => b.kind === 'linear_attention')).toHaveLength(4);
    expect(result.model.layers[0].blocks.find(b => b.kind === 'mlp')).toMatchObject({ intermediate: 4096 });
    expect(result.paramCheck.status).toBe('mismatch');
  });

  it.each(targets.map(t => t.id))('%s rejects malformed layer schedules without throwing', id => {
    const files = fixtureRepoFiles(id);
    const config = structuredClone(files.config) as JsonObject;
    const text = config.text_config as JsonObject;
    if (id.startsWith('Qwen')) text.layer_types = ['unknown_attention'];
    else if (id.startsWith('MiniMax')) text.moe_layer_freq = [1];
    else text.local_layer_ids = [999];
    expect(mapHfModel({ ...files, config }).kind).toBe('invalid_config');
  });
  it.each(targets.map(t => t.id))('%s rejects missing widths and an empty decoder', id => {
    const files = fixtureRepoFiles(id);
    for (const value of [undefined, 'not-a-number', 0]) {
      const config = structuredClone(files.config) as JsonObject;
      const text = config.text_config as JsonObject;
      if (value === undefined) delete text.hidden_size;
      else text.hidden_size = value;
      expect(mapHfModel({ ...files, config }).kind).toBe('invalid_config');
    }
    const config = structuredClone(files.config) as JsonObject;
    (config.text_config as JsonObject).num_hidden_layers = 0;
    expect(mapHfModel({ ...files, config }).kind).toBe('invalid_config');
  });

  it.each(targets.map(t => t.id))('%s reports unavailable and packed checkpoint totals', id => {
    const files = fixtureRepoFiles(id);
    const missingApi = mapHfModel({ ...files, api: undefined });
    if (missingApi.kind !== 'ok') throw new Error(JSON.stringify(missingApi));
    expect(missingApi.paramCheck.status).toBe('unavailable');
    const config = structuredClone(files.config) as JsonObject;
    config.quantization_config = { quant_method: 'gptq', bits: 4, group_size: 128 };
    const packed = mapHfModel({ ...files, config, api: { id, safetensors: { total: 1e9, parameters: { I32: 1e9 } } } });
    if (packed.kind !== 'ok') throw new Error(JSON.stringify(packed));
    expect(packed.paramCheck.status).toBe('packed');
  });

  it('transformers-native MiniMax schedules match the legacy schema exactly', () => {
    const base = mapHfModel(fixtureRepoFiles('MiniMaxAI/MiniMax-M3'));
    const native = mapHfModel(fixtureRepoFiles('cyankiwi/MiniMax-M3-AWQ-INT4', { dir: 'quant' }));
    expect(native.kind).toBe('ok');
    if (base.kind !== 'ok' || native.kind !== 'ok') throw new Error(JSON.stringify({ base, native }));
    expect(native.model.layers).toEqual(base.model.layers);
    expect(countParams(native.model)).toBe(countParams(base.model));
    expect(activeParamsPerToken(native.model)).toBe(activeParamsPerToken(base.model));
  });

  it.each(['mlp_layer_types', 'layer_types'])('native MiniMax rejects invalid %s schedules', key => {
    const files = fixtureRepoFiles('cyankiwi/MiniMax-M3-AWQ-INT4', { dir: 'quant' });
    for (const schedule of [['unknown'], []]) {
      const config = structuredClone(files.config) as JsonObject;
      (config.text_config as JsonObject)[key] = schedule;
      expect(mapHfModel({ ...files, config }).kind).toBe('invalid_config');
    }
  });
  it.each([
    ...listFixtureRepos('quant').map(id => ({ id, configFrom: undefined as string | undefined })),
    // This repo has only params.json; the loader takes architecture from its base repo.
    { id: 'mistralai/Mistral-Large-4-1T-A52B-NVFP4', configFrom: 'mistralai/Mistral-Large-4-1T-A52B' },
  ])('quant fixture $id maps successfully', ({ id, configFrom }) => {
    const result = mapHfModel(fixtureRepoFiles(id, { dir: 'quant', configFrom }));
    expect(result.kind, JSON.stringify(result.kind === 'ok' ? undefined : result)).toBe('ok');
  });
});
