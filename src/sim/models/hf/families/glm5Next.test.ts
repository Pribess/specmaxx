import { describe, expect, it } from 'vitest';
import type { LayerSpec } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { checkParams } from '../paramCheck';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild, JsonObject } from '../types';
import { glm5NextFamily } from './glm5Next';
import { layerParams } from './mlaCommon';

const ID = 'zai-org/GLM-5.3-Flash';

/** Build `id`'s fixture with the GLM-5.3-Flash family, optionally with text_config overrides. */
function build(id: string, override: JsonObject = {}): FamilyBuild {
  const files = fixtureRepoFiles(id);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const text = { ...u.view.text, ...override };
  const r = glm5NextFamily.build(text, { ...u.view.top, text_config: text }, files);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

const kinds = (layer: LayerSpec) => layer.blocks.map((b) => b.kind);

describe('glm5_next: GLM-5.3-Flash', () => {
  const b = build(ID);
  const m = b.model;

  it('builds 45 layers with 4 mHC streams', () => {
    expect(m.layers).toHaveLength(45);
    expect(m.hidden).toBe(4096);
    expect(m.residualStreams).toBe(4);
  });

  it('maps layer_types to 34 KDA and 11 NoPE MLA + k-pool DSA layers', () => {
    const mixers = m.layers.map((l) => l.blocks.find((x) => x.kind === 'linear_attention' || x.kind === 'mla_attention'));
    expect(mixers.filter((x) => x?.kind === 'linear_attention')).toHaveLength(34);
    expect(mixers.filter((x) => x?.kind === 'mla_attention')).toHaveLength(11);
    expect(mixers[0]).toMatchObject({ variant: 'kda', qkHeads: 64, vHeads: 64, keyDim: 128, valueDim: 128, convKernel: 4, outputGate: true, gateRank: 128 });
    expect(mixers[3]).toMatchObject({
      kind: 'mla_attention',
      heads: 64,
      qLoraRank: 1536,
      kvLoraRank: 512,
      qkNopeDim: 256,
      qkRopeDim: 0,
      vHeadDim: 256,
      sparse: { kind: 'indexer', indexer: { heads: 32, headDim: 128, topK: 2048, keyPoolTokens: 4, keyPoolGate: true } },
    });
  });

  it('wraps each sublayer in hc_pre → norm → … → hc_post and collapses the streams before the head', () => {
    expect(kinds(m.layers[0])).toEqual(['hc_pre', 'norm', 'linear_attention', 'hc_post', 'hc_pre', 'norm', 'mlp', 'hc_post']);
    expect(kinds(m.layers[44]).slice(-1)).toEqual(['hc_pre']);
    expect(m.layers[44].blocks.at(-1)).toMatchObject({ projections: [], vectorParams: 0 });
  });

  it('falls back to the transformers defaults when layer_types / mlp_layer_types are absent', () => {
    expect(build(ID, { layer_types: null, mlp_layer_types: null }).model.layers).toEqual(m.layers);
  });

  // Per-layer params from the safetensors headers (findings R3, zai-org__GLM-5.3-Flash.text.shapes.json); the IR
  // leaves out the router bias e_score_correction_bias (288 per MoE layer).
  it.each([
    [0, 289_521_910, 0], // KDA, dense FFN
    [3, 7_399_812_182, 288], // MLA + DSA, MoE
  ])('layer %i matches the checkpoint headers', (i, header, routerBias) => {
    expect(layerParams(m.layers[i])).toBe(header - routerBias);
  });

  it('total and active params match the header-derived targets', () => {
    // Text (45 layers) 313.327B = safetensors.total 321,323,031,390 − MTP 7.433B − vision 0.564B.
    expect(Math.abs(countParams(m) / 313.327e9 - 1)).toBeLessThan(0.005);
    // R3 §5: 16.74B active with lm_head, without the embedding (card: "18B active", see R3 §5).
    const active = activeParamsPerToken(m) - m.vocab * m.hidden;
    expect(Math.abs(active / 16.74e9 - 1)).toBeLessThan(0.015);
  });

  it('estimates the MTP layer and the vision tower; the param check passes', () => {
    // layers.45: MLA + indexer + MoE without hc_* tensors, + eh_proj + 3 norms (7,432,592,416 − router bias 288).
    expect(b.mtpParamsEstimate).toBe(7_432_592_416 - 288);
    expect(Math.abs(b.visionParamsEstimate / 0.564e9 - 1)).toBeLessThan(0.02);
    expect(b.excludedParts).toEqual(['MTP layer', 'vision tower']);
    const check = checkParams(m, b, fixtureRepoFiles(ID).api, 'fp8');
    expect(check.status).toBe('ok');
  });

  it('builds the -BF16 checkpoint identically', () => {
    expect(build('zai-org/GLM-5.3-Flash-BF16').model.layers).toEqual(m.layers);
  });
});
