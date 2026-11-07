import { describe, expect, it } from 'vitest';
import type { LayerSpec, MlaAttentionBlock } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { checkParams } from '../paramCheck';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild, FamilyBuildResult, JsonObject } from '../types';
import { layerParams } from './mlaCommon';
import { mistralLarge4Family } from './mistralLarge4';

const ID = 'mistralai/Mistral-Large-4-1T-A52B';

/** Build the fixture with the Mistral family, optionally with text_config overrides. */
function buildResult(override: JsonObject = {}): FamilyBuildResult {
  const files = fixtureRepoFiles(ID);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const text = { ...u.view.text, ...override };
  return mistralLarge4Family.build(text, { ...u.view.top, text_config: text }, files);
}

function build(override: JsonObject = {}): FamilyBuild {
  const r = buildResult(override);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

const mlaOf = (layer: LayerSpec): MlaAttentionBlock => {
  const b = layer.blocks.find((x) => x.kind === 'mla_attention');
  if (b?.kind !== 'mla_attention') throw new Error(`${layer.name}: no mla block`);
  return b;
};

describe('mistral_large4: Mistral Large 4', () => {
  const b = build();
  const m = b.model;

  it('reads MLA without q-LoRA, ignoring head_dim 64 and num_key_value_heads', () => {
    expect(m.layers).toHaveLength(72);
    expect(mlaOf(m.layers[0])).toMatchObject({ heads: 64, qLoraRank: 0, kvLoraRank: 512, qkNopeDim: 64, qkRopeDim: 64, vHeadDim: 128 });
  });

  it('makes every no_rope_layers[i] == 0 layer NoPE with the RoPE slice folded into the nope dims', () => {
    const nope = m.layers.flatMap((l, i) => (mlaOf(l).qkRopeDim === 0 ? [i] : []));
    expect(nope).toHaveLength(18);
    expect(nope.slice(0, 3)).toEqual([3, 7, 11]);
    expect(mlaOf(m.layers[3])).toMatchObject({ qkNopeDim: 128, qkRopeDim: 0 });
  });

  it('reuses the top-k in the 28 shared-indexer layers and the latent cache of layer 35 in 9 layers', () => {
    const reuse = m.layers.filter((l) => mlaOf(l).sparse?.kind === 'reuse');
    expect(reuse).toHaveLength(28);
    expect(mlaOf(m.layers[17]).sparse).toEqual({ kind: 'reuse', topK: 2048 });
    const consumers = m.layers.flatMap((l, i) => (mlaOf(l).kvCacheFromLayer !== undefined ? [i] : []));
    expect(consumers).toEqual([39, 43, 47, 51, 55, 59, 63, 67, 71]);
    expect(consumers.every((i) => mlaOf(m.layers[i]).kvCacheFromLayer === 35)).toBe(true);
    // A KV consumer keeps its own indexer (indexer_types[39] is 'full').
    expect(mlaOf(m.layers[39]).sparse?.kind).toBe('indexer');
  });

  it('has 1 dense FFN layer and 71 MoE layers with a shared expert of moe_intermediate_size', () => {
    expect(m.layers[0].blocks.find((x) => x.kind === 'mlp')).toMatchObject({ intermediate: 24576 });
    const moe = m.layers.map((l) => l.blocks.find((x) => x.kind === 'moe')).filter((x) => x !== undefined);
    expect(moe).toHaveLength(71);
    expect(moe[0]).toMatchObject({ experts: 128, topK: 4, expertIntermediate: 6144, sharedExperts: 1, sharedIntermediate: 6144 });
  });

  it('falls back to no_rope_layer_interval and an identity kv_source_layers', () => {
    const derived = build({ no_rope_layers: null });
    expect(derived.model.layers).toEqual(m.layers);
    const unshared = build({ kv_source_layers: null }).model;
    expect(unshared.layers.every((l) => mlaOf(l).kvCacheFromLayer === undefined)).toBe(true);
  });

  // Per-layer params from the consolidated safetensors headers (findings R4, hf/…tensors.json). The IR leaves
  // out the indexer's q_input_norm LayerNorm (2·6144) and the ssmax scalars (2·64 attention + 2·32 indexer).
  it.each([
    [0, 589_652_928, 12_480], // dense FFN
    [1, 14_746_215_360, 12_480], // RoPE
    [3, 14_747_919_296, 12_480], // NoPE
    [17, 14_720_053_888, 128], // shared indexer
    [39, 14_744_773_056, 12_480], // NoPE, reads layer 35's latent
  ])('layer %i matches the checkpoint headers', (i, header, extras) => {
    expect(layerParams(m.layers[i])).toBe(header - extras);
  });

  it('total and active params match the header-derived targets', () => {
    // Text-only header sum 1,049.36B (1,053.16B − vision 1.70B − MTP 2.10B).
    expect(Math.abs(countParams(m) / 1049.36e9 - 1)).toBeLessThan(0.005);
    // 52.3B active including embedding and head (card: "49 billion … 52 billion including embeddings").
    expect(Math.abs(activeParamsPerToken(m) / 52.3e9 - 1)).toBeLessThan(0.015);
  });

  it('estimates the DSpark draft from params.json; the param check is unavailable', () => {
    expect(Math.abs(b.mtpParamsEstimate / 2_099_969_024 - 1)).toBeLessThan(0.01);
    expect(Math.abs(b.visionParamsEstimate / 1.70e9 - 1)).toBeLessThan(0.05);
    expect(checkParams(m, b, fixtureRepoFiles(ID).api).status).toBe('unavailable');
  });

  it('rejects a kv_source_layers entry that points forward', () => {
    const sources = Array.from({ length: 72 }, (_, i) => (i === 10 ? 11 : i));
    expect(buildResult({ kv_source_layers: sources }).kind).toBe('invalid_config');
  });
});
