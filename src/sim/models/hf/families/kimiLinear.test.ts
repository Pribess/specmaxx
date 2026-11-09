import { describe, expect, it } from 'vitest';
import type { LayerSpec } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { mapHfModel } from '../index';
import { checkParams } from '../paramCheck';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild, JsonObject } from '../types';
import { estimateKimiVisionParams, kimiLinearFamily } from './kimiLinear';
import { layerParams } from './mlaCommon';

const K3 = 'moonshotai/Kimi-K3';
const K31 = 'moonshotai/Kimi-K3.1-Pro';

/** Build `id`'s fixture with the Kimi family, optionally with text_config overrides. */
function build(id: string, override: JsonObject = {}): FamilyBuild {
  const files = fixtureRepoFiles(id);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const text = { ...u.view.text, ...override };
  const r = kimiLinearFamily.build(text, { ...u.view.top, text_config: text }, files);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

const kinds = (layer: LayerSpec) => layer.blocks.map((b) => b.kind);
const mixerOf = (layer: LayerSpec) =>
  layer.blocks.find((b) => b.kind === 'linear_attention' || b.kind === 'mla_attention');

// Published / header-derived numbers (research: safetensors headers, params.py reproduces safetensors.total exactly).
const K3_TOTAL = 2_779_931_837_184;
const K31_TOTAL = 2_780_110_363_392;
const VISION = 447_358_976; // MoonViT-V2 tower 401,214,464 + mm_projector 46,144,512
// The IR leaves out, per model: the router bias (92 · 896), the A_log padding (69 · 32: stored [128] for 96 heads)
// and layer 0's unused mixer-side attention-residual pair (2 · 7168).
const NOT_COUNTED = 92 * 896 + 69 * 32 + 2 * 7168;

describe('kimi_linear: Kimi K3', () => {
  const b = build(K3);
  const m = b.model;

  it('dispatches both wrappers and the text model through mapHfModel', () => {
    for (const id of [K3, K31]) {
      const r = mapHfModel(fixtureRepoFiles(id));
      expect(r.kind).toBe('ok');
      if (r.kind === 'ok') expect(r.family).toBe('kimi_linear');
    }
  });

  it('builds 93 layers: 69 KDA and 24 MLA at the 1-based full_attn_layers minus one', () => {
    expect(m.layers).toHaveLength(93);
    expect(m.hidden).toBe(7168);
    expect(m.vocab).toBe(163840);
    expect(m.tieEmbeddings).toBe(false);
    const mlaAt = m.layers.flatMap((l, i) => (mixerOf(l)?.kind === 'mla_attention' ? [i] : []));
    expect(mlaAt).toEqual([...Array.from({ length: 23 }, (_, j) => 4 * j + 3), 92]);
    expect(m.layers.filter((l) => mixerOf(l)?.kind === 'linear_attention')).toHaveLength(69);
  });

  it('KDA: 96 × 128, conv 4, rank-128 forget gate, FULL-RANK output gate', () => {
    expect(mixerOf(m.layers[0])).toMatchObject({
      variant: 'kda',
      qkHeads: 96,
      vHeads: 96,
      keyDim: 128,
      valueDim: 128,
      convKernel: 4,
      outputGate: true,
      gateRank: 128,
      fullRankOutputGate: true,
    });
  });

  it('MLA: DeepSeek shapes with the RoPE slice kept (cached, mla_use_nope), a sigmoid output gate, no indexer', () => {
    expect(mixerOf(m.layers[3])).toEqual({
      kind: 'mla_attention',
      name: 'attn',
      hidden: 7168,
      heads: 96,
      qLoraRank: 1536,
      kvLoraRank: 512,
      qkNopeDim: 128,
      qkRopeDim: 64,
      vHeadDim: 128,
      outputGate: true,
    });
  });

  it('FFN: layer 0 dense situ 33792; layers 1–92 latent MoE (896 × 3072, top-16, L 3584 with norm, 2 shared fused to 6144)', () => {
    expect(m.layers[0].blocks.find((x) => x.kind === 'mlp')).toMatchObject({
      intermediate: 33792,
      activation: 'situ',
      gated: true,
    });
    const moe = m.layers[1].blocks.find((x) => x.kind === 'moe');
    expect(moe).toMatchObject({
      experts: 896,
      topK: 16,
      expertIntermediate: 3072,
      sharedExperts: 1,
      sharedIntermediate: 6144,
      activation: 'situ',
      latentDim: 3584,
      latentNorm: true,
    });
    expect(m.layers.filter((l) => l.blocks.some((x) => x.kind === 'moe'))).toHaveLength(92);
  });

  it('Attention Residuals of block 12: mixes before each sublayer (none before layer 0 mixer), 9 candidates at the end', () => {
    expect(kinds(m.layers[0])).toEqual([
      'norm',
      'linear_attention',
      'residual',
      'residual_mix',
      'norm',
      'mlp',
      'residual',
    ]);
    expect(kinds(m.layers[1])).toEqual([
      'residual_mix',
      'norm',
      'linear_attention',
      'residual',
      'residual_mix',
      'norm',
      'moe',
      'residual',
    ]);
    // Layer 24 (third block start): the mixer sees 2 summaries + the sum; the ffn 3 summaries + the sum.
    const mixes24 = m.layers[24].blocks.filter((x) => x.kind === 'residual_mix');
    expect(mixes24.map((x) => x.candidates)).toEqual([3, 4]);
    expect(m.layers[92].blocks.at(-1)).toEqual({
      kind: 'residual_mix',
      name: 'final_residual_mix',
      dim: 7168,
      candidates: 9,
    });
  });

  // Per-layer params from the safetensors headers (research params.py; the router bias and A_log padding excluded).
  it.each([
    [0, 443_740_384 + 3 * 7168 * 33792 + 4 * 7168, 0], // KDA (full-rank gate) + dense FFN + 2 norms + ffn mix (2h)
    [1, 443_740_384 + 29_784_935_936 + 6 * 7168, 0], // KDA + latent MoE + 2 norms + 2 mixes
    [3, 144_115_712 + 88_080_384 + 29_784_935_936 + 6 * 7168, 0], // MLA + gate + latent MoE
  ])('layer %i matches the header-derived count', (i, expected) => {
    expect(layerParams(m.layers[i])).toBe(expected);
  });

  it('total params match safetensors.total minus the vision tower to within the three not-counted vectors', () => {
    const lm = K3_TOTAL - VISION;
    expect(countParams(m)).toBe(lm - NOT_COUNTED);
    // Relative gap 3.6e-8; the published "2.78T total" and "~104B active" (excluding the input embedding).
    expect(Math.abs(countParams(m) / lm - 1)).toBeLessThan(1e-7);
    const active = activeParamsPerToken(m) - m.vocab * m.hidden;
    expect(Math.abs(active / 104.19e9 - 1)).toBeLessThan(0.001);
  });

  it('estimates the MoonViT-V2 tower + projector exactly and the param check passes (U8 count is logical)', () => {
    expect(b.visionParamsEstimate).toBe(VISION);
    expect(b.excludedParts).toEqual(['vision tower']);
    expect(b.mtpParamsEstimate).toBe(0);
    const check = checkParams(m, b, fixtureRepoFiles(K3).api, 'compressed-tensors');
    expect(check.status).toBe('ok');
    expect(Math.abs(check.relError ?? 1)).toBeLessThan(1e-6);
  });

  it('quant defaults: MXFP4 routed experts only; attention, KDA, shared and latent projections BF16', () => {
    const r = mapHfModel(fixtureRepoFiles(K3));
    if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
    const q = r.quant.quant;
    expect(q.weights).toEqual({ dtype: 'mxfp4' });
    expect(q.activations).toBe('bf16');
    expect(q.kvCache).toEqual({ dtype: 'bf16' });
    expect(q.modules).toEqual({
      attention: { weights: { dtype: 'bf16' } },
      linear_attention: { weights: { dtype: 'bf16' } },
      indexer: { weights: { dtype: 'bf16' } },
      dense_mlp: { weights: { dtype: 'bf16' } },
      shared_experts: { weights: { dtype: 'bf16' } },
      // moe_latent is absent: it equals its fallback (shared_experts, BF16), as the checkpoint stores it.
    });
    expect(q.modules?.moe_latent).toBeUndefined();
    expect(q.modules?.routed_experts).toBeUndefined();
    expect(r.paramCheck.status).toBe('ok');
    expect(r.notes.some((n) => n.includes('ignore:'))).toBe(false);
  });

  it('falls back to the plain layout for a sibling without attn_res_block_size or routed_expert_hidden_size', () => {
    const plain = build(K3, {
      attn_res_block_size: null,
      routed_expert_hidden_size: null,
      mla_use_output_gate: false,
    });
    expect(kinds(plain.model.layers[1])).toEqual([
      'norm',
      'linear_attention',
      'residual',
      'norm',
      'moe',
      'residual',
    ]);
    expect(plain.model.layers[1].blocks.find((x) => x.kind === 'moe')).not.toHaveProperty('latent');
    expect(mixerOf(plain.model.layers[3])).not.toHaveProperty('outputGate');
    expect(kinds(plain.model.layers[92]).at(-1)).toBe('residual');
  });

  it('rejects layer lists that do not cover every layer once', () => {
    const files = fixtureRepoFiles(K3);
    const u = unwrapTextConfig(files.config);
    if (u.kind !== 'ok') throw new Error('unwrap failed');
    const lin = u.view.text.linear_attn_config as JsonObject;
    const text = { ...u.view.text, linear_attn_config: { ...lin, full_attn_layers: [4, 8] } };
    const r = kimiLinearFamily.build(text, { ...u.view.top, text_config: text }, files);
    expect(r.kind).toBe('invalid_config');
    if (r.kind === 'invalid_config')
      expect(r.errors[0]).toMatch(/kda_layers \+ full_attn_layers must list every layer/);
  });
});

describe('kimi_linear: Kimi K3.1-Pro', () => {
  const b = build(K31);
  const m = b.model;

  it('is K3 plus a DSA indexer (32 × 128, top-2048) on all 24 MLA layers', () => {
    const k3 = build(K3).model;
    expect(m.layers.length).toBe(93);
    for (let i = 0; i < 93; i++) {
      const mixer = mixerOf(m.layers[i]);
      const base = mixerOf(k3.layers[i]);
      if (mixer?.kind === 'mla_attention') {
        expect(mixer).toEqual({
          ...base,
          sparse: { kind: 'indexer', indexer: { heads: 32, headDim: 128, topK: 2048 } },
        });
      } else {
        expect(mixer).toEqual(base);
      }
    }
  });

  it('each indexer adds 7,438,592 params (rq·H_I·hd_I + h·hd_I + 2·hd_I + h·H_I), and the total matches', () => {
    expect(layerParams(m.layers[3]) - layerParams(build(K3).model.layers[3])).toBe(
      1536 * 32 * 128 + 7168 * 128 + 2 * 128 + 7168 * 32,
    );
    expect(countParams(m)).toBe(K31_TOTAL - VISION - NOT_COUNTED);
    const active = activeParamsPerToken(m) - m.vocab * m.hidden;
    expect(Math.abs(active / 104.37e9 - 1)).toBeLessThan(0.001);
    expect(checkParams(m, b, fixtureRepoFiles(K31).api, 'compressed-tensors').status).toBe('ok');
  });

  it('quant defaults equal K3 (the ignore list names the latent projections explicitly here)', () => {
    const a = mapHfModel(fixtureRepoFiles(K3));
    const c = mapHfModel(fixtureRepoFiles(K31));
    if (a.kind !== 'ok' || c.kind !== 'ok') throw new Error('map failed');
    expect(c.quant.quant).toEqual(a.quant.quant);
    expect(c.notes.some((n) => n.includes('ignore:'))).toBe(false);
  });
});

describe('estimateKimiVisionParams', () => {
  it('matches the K3 headers and falls back to the generic estimate without vt_* keys', () => {
    const cfg = fixtureRepoFiles(K3).config as JsonObject;
    expect(estimateKimiVisionParams(cfg.vision_config as JsonObject, 7168)).toBe(VISION);
    expect(estimateKimiVisionParams(undefined, 7168)).toBe(0);
    // Generic shape: 2·(4·10² + 2·10·40) + 10·16 (projection to the text width).
    expect(estimateKimiVisionParams({ depth: 2, hidden_size: 10, intermediate_size: 40 }, 16)).toBe(
      2 * 1200 + 160,
    );
  });
});
