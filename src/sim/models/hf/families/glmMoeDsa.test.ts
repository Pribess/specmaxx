import { describe, expect, it } from 'vitest';
import type { LayerSpec, MlaAttentionBlock } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { checkParams } from '../paramCheck';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild, FamilyBuildResult, JsonObject } from '../types';
import { glmMoeDsaFamily } from './glmMoeDsa';
import { layerParams } from './mlaCommon';

const ID = 'zai-org/GLM-5.3';

/** Build `id`'s fixture with the GLM-5.3 family, optionally with config overrides. */
function buildResult(id: string, override: JsonObject = {}): FamilyBuildResult {
  const files = fixtureRepoFiles(id);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const config = { ...u.view.top, ...override };
  return glmMoeDsaFamily.build(config, config, files);
}

function build(id: string, override: JsonObject = {}): FamilyBuild {
  const r = buildResult(id, override);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

const mlaOf = (layer: LayerSpec): MlaAttentionBlock => {
  const b = layer.blocks.find((x) => x.kind === 'mla_attention');
  if (b?.kind !== 'mla_attention') throw new Error(`${layer.name}: no mla block`);
  return b;
};

describe('glm_moe_dsa: GLM-5.3', () => {
  const b = build(ID);
  const m = b.model;

  it('reads MLA from qk_nope/qk_rope/v_head_dim (not head_dim 192) and 78 layers', () => {
    expect(m.layers).toHaveLength(78);
    expect(mlaOf(m.layers[0])).toMatchObject({ heads: 64, qLoraRank: 2048, kvLoraRank: 512, qkNopeDim: 192, qkRopeDim: 64, vHeadDim: 256 });
  });

  it('runs 21 indexers and reuses the top-k in 57 layers; 3 dense + 75 MoE layers', () => {
    const sel = m.layers.map((l) => mlaOf(l).sparse?.kind);
    expect(sel.filter((k) => k === 'indexer')).toHaveLength(21);
    expect(sel.filter((k) => k === 'reuse')).toHaveLength(57);
    expect(sel.slice(0, 7)).toEqual(['indexer', 'indexer', 'indexer', 'reuse', 'reuse', 'reuse', 'indexer']);
    expect(mlaOf(m.layers[0]).sparse).toEqual({ kind: 'indexer', indexer: { heads: 32, headDim: 128, topK: 2048 } });
    const ffn = m.layers.map((l) => l.blocks.find((x) => x.kind === 'mlp' || x.kind === 'moe')?.kind);
    expect(ffn.filter((k) => k === 'mlp')).toHaveLength(3);
    expect(ffn.filter((k) => k === 'moe')).toHaveLength(75);
  });

  it('derives the same indexer schedule from index_topk_freq / index_skip_topk_offset', () => {
    const derived = build(ID, { indexer_types: null });
    expect(derived.model.layers).toEqual(m.layers);
  });

  // Per-layer params from the safetensors headers (findings R3, zai-org__GLM-5.3.text.shapes.json); the IR leaves
  // out the router bias e_score_correction_bias (256 per MoE layer).
  it.each([
    [0, 400_898_816, 0], // full indexer, dense FFN
    [3, 9_868_032_768, 256], // shared indexer, MoE
    [4, 9_868_032_768, 256],
  ])('layer %i matches the checkpoint headers', (i, header, routerBias) => {
    expect(layerParams(m.layers[i])).toBe(header - routerBias);
  });

  it('total and active params match the header-derived targets', () => {
    // Text (78 layers) 743.377B = safetensors.total 753,329,940,480 − MTP layer 78 (9,952,920,576).
    expect(Math.abs(countParams(m) / 743_377e6 - 1)).toBeLessThan(0.005);
    // R3 §5: 40.30B active with lm_head, without the embedding (published: "744B parameters (40B active)").
    const active = activeParamsPerToken(m) - m.vocab * m.hidden;
    expect(Math.abs(active / 40.3e9 - 1)).toBeLessThan(0.015);
  });

  it('estimates the MTP layer exactly up to the router bias and passes the param check', () => {
    expect(b.mtpParamsEstimate).toBe(9_952_920_576 - 256);
    expect(b.visionParamsEstimate).toBe(0);
    expect(b.excludedParts).toEqual(['MTP layer']);
    const check = checkParams(m, b, fixtureRepoFiles(ID).api, 'fp8');
    expect(check.status).toBe('ok');
    expect(Math.abs(check.relError ?? 1)).toBeLessThan(1e-6);
  });

  it('builds the -BF16 checkpoint identically', () => {
    expect(build('zai-org/GLM-5.3-BF16').model.layers).toEqual(m.layers);
  });

  it('rejects a shared indexer with no earlier full layer', () => {
    const r = buildResult(ID, { indexer_types: ['shared', ...Array(77).fill('full')] });
    expect(r.kind).toBe('invalid_config');
  });

  it('reports missing fields path-qualified', () => {
    const r = buildResult(ID, { kv_lora_rank: null });
    expect(r).toEqual({ kind: 'invalid_config', errors: ['kv_lora_rank: missing non-negative integer'] });
  });
});
