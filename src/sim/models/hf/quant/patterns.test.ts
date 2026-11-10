import { describe, expect, it } from 'vitest';
import type { Probe } from './common';
import {
  blockScaleBits,
  dtypeNameBits,
  formatRanges,
  groupScaleBits,
  modelDtypeOf,
} from './common';
import { buildProbes, layerLayout } from './moduleClass';
import { compilePatterns, dottedContains, firstMatch } from './patterns';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { unwrapTextConfig } from '../textConfig';

const probe = (name: string, extra: Partial<Probe> = {}): Probe => ({
  name,
  cls: 'attention',
  layer: 0,
  kind: 'linear',
  weight: 1,
  leaf: name,
  ...extra,
});

describe('patterns', () => {
  const p = probe('model.language_model.layers.3.self_attn.q_proj');

  it('plain names match at dotted boundaries (prefix, suffix, infix)', () => {
    expect(dottedContains(p.name, 'q_proj')).toBe(true);
    expect(dottedContains(p.name, 'model.language_model.layers.3')).toBe(true);
    expect(dottedContains(p.name, 'layers.3.self_attn')).toBe(true);
    expect(dottedContains(p.name, 'layers.33')).toBe(false);
    expect(dottedContains(p.name, '_proj')).toBe(false);
    expect(dottedContains('model.layers.1.mlp.gate_proj', 'model.layers.1.mlp.gate')).toBe(false);
  });

  it('globs are fnmatch over the whole name; re: is anchored at the start', () => {
    const list = compilePatterns(['*.self_attn.*', 're:.*q_proj$', 're:q_proj']);
    expect(firstMatch(list, p)).toBe(0);
    expect(firstMatch(compilePatterns(['re:.*q_proj$']), p)).toBe(0);
    expect(firstMatch(compilePatterns(['re:q_proj']), p)).toBe(-1);
    expect(firstMatch(compilePatterns(['model.language_model.layers.3*']), p)).toBe(0);
    expect(
      firstMatch(
        compilePatterns(['model.language_model.layers.3']),
        probe('model.language_model.layers.30.x'),
      ),
    ).toBe(-1);
  });

  it('class names select by probe kind', () => {
    const list = compilePatterns(['Linear']);
    expect(firstMatch(list, p)).toBe(0);
    expect(firstMatch(list, probe('embed', { kind: 'embedding' }))).toBe(-1);
    expect(
      firstMatch(
        compilePatterns(['RoutedExperts']),
        probe('layers.3.ffn.experts.0.w1', { cls: 'routed_experts' }),
      ),
    ).toBe(0);
  });

  it('exact lists (compressed-tensors) do not let a parent name cover its children', () => {
    const exact = compilePatterns(['model.language_model.layers.3.self_attn'], { exact: true });
    expect(firstMatch(exact, p)).toBe(-1);
    expect(firstMatch(compilePatterns(['model.language_model.layers.3.self_attn']), p)).toBe(0);
  });

  it('another stack prefix matches from layers.<n> on, never for MTP stacks', () => {
    const q = probe('layers.14.attn.compressor.wkv');
    expect(
      firstMatch(
        compilePatterns(['language_model.layers.14.attn.compressor.wkv'], { exact: true }),
        q,
      ),
    ).toBe(0);
    expect(
      firstMatch(compilePatterns(['mtp_dspark.backbone.layers.14.attn.compressor.wkv']), q),
    ).toBe(-1);
  });

  it('the earliest entry wins; invalid regexes are collected, not thrown', () => {
    const list = compilePatterns(['re:([', 're:.*self_attn.*', 'q_proj']);
    expect(list.invalid).toEqual(['re:([']);
    expect(firstMatch(list, p)).toBe(0);
    expect(list.entries[0].raw).toBe('re:.*self_attn.*');
  });

  it('AutoRound regex keys are searched anywhere', () => {
    const list = compilePatterns(['.*mlp\\.gate.*'], { regexKeys: true });
    expect(firstMatch(list, probe('model.layers.0.mlp.gate_proj'))).toBe(0);
    expect(firstMatch(list, probe('model.layers.0.mlp.experts.gate_up_proj'))).toBe(-1);
  });
});

describe('scale arithmetic', () => {
  it('block and group overheads', () => {
    expect(blockScaleBits(32, 128, 128)).toBeCloseTo(0.001953125, 9);
    expect(blockScaleBits(8, 32, 32)).toBeCloseTo(0.0078125, 9);
    expect(groupScaleBits(16, 0, 128)).toBeCloseTo(0.125, 9);
    expect(groupScaleBits(16, 8, 32)).toBeCloseTo(0.75, 9);
    expect(groupScaleBits(16, 16, 64)).toBeCloseTo(0.5, 9);
  });

  it('scale dtype names', () => {
    expect(dtypeNameBits('torch.float16', 0)).toBe(16);
    expect(dtypeNameBits('float32', 0)).toBe(32);
    expect(dtypeNameBits('float', 0)).toBe(32);
    expect(dtypeNameBits('float8_e8m0fnu', 0)).toBe(8);
    expect(dtypeNameBits('torch.uint8', 0)).toBe(8);
    expect(dtypeNameBits(null, 16)).toBe(16);
    expect(modelDtypeOf(undefined, 'float16')).toBe('fp16');
    expect(modelDtypeOf()).toBe('bf16');
  });

  it('formatRanges', () => {
    expect(formatRanges([5, 0, 1, 2, 7, 8])).toBe('0–2, 5, 7–8');
    expect(formatRanges([3])).toBe('3');
  });
});

describe('moduleClass', () => {
  it('layer layout from layer_types, mlp_layer_types, first_k_dense_replace, moe_layer_freq', () => {
    const l = layerLayout({
      num_hidden_layers: 4,
      layer_types: ['linear_attention', 'full_attention', 'linear_attention', 'full_attention'],
      n_routed_experts: 8,
      first_k_dense_replace: 1,
    });
    expect(l.attention).toEqual(['linear', 'full', 'linear', 'full']);
    expect(l.mlp).toEqual(['dense', 'moe', 'moe', 'moe']);
    expect(
      layerLayout({ num_hidden_layers: 3, moe_layer_freq: [0, 1, 1], num_local_experts: 4 }).mlp,
    ).toEqual(['dense', 'moe', 'moe']);
    expect(layerLayout({ num_hidden_layers: 2 }).mlp).toEqual(['dense', 'dense']);
    expect(layerLayout({ num_hidden_layers: 3, kv_source_layers: [0, 0, 2] }).kvLayers).toEqual([
      0, 2,
    ]);
    expect(layerLayout({ num_hidden_layers: 40, kv_source_layer_ids: [2, 8] }).kvLayers).toEqual([
      2, 8,
    ]);
  });

  it('probes exist only where the module exists; MTP layers are never probed', () => {
    const top = { model_type: 'qwen3_5' };
    const text = { num_hidden_layers: 2, layer_types: ['linear_attention', 'full_attention'] };
    const { probes, family } = buildProbes(top, text, '');
    expect(family).toBe('qwen3_5');
    const l0 = probes.filter((x) => x.layer === 0).map((x) => x.cls);
    const l1 = probes.filter((x) => x.layer === 1).map((x) => x.cls);
    expect(l0).toContain('linear_attention');
    expect(l0).not.toContain('attention');
    expect(l1).toContain('attention');
    expect(l1).not.toContain('linear_attention');
    expect(probes.some((x) => x.layer !== null && x.layer >= 2)).toBe(false);
    expect(probes.find((x) => x.cls === 'lm_head')?.name).toBe('lm_head');
  });

  it('the naming variant is picked by evidence in the metadata', () => {
    const top = { model_type: 'mistral_large4' };
    const text = { num_hidden_layers: 1 };
    const native = buildProbes(top, text, '["layers.0.attention.wkv_b","output"]');
    expect(
      native.probes.some(
        (x) => x.name === 'layers.0.attention.wkv_b' && x.cls === 'attention_kv_up',
      ),
    ).toBe(true);
    const hf = buildProbes(top, text, '["model.layers.0.self_attn.kv_b_proj","lm_head"]');
    expect(hf.probes.some((x) => x.name === 'model.layers.0.self_attn.kv_b_proj')).toBe(true);
    const minimax = buildProbes(
      { model_type: 'minimax_m3_vl' },
      { num_hidden_layers: 1, num_local_experts: 2 },
      '"language_model.model.layers.0.block_sparse_moe.experts.3.gate_proj"',
    );
    expect(minimax.probes.some((x) => x.name.endsWith('experts.0.gate_proj'))).toBe(true);
  });

  it('Nemotron-H: one sublayer per layer from layers_block_type, latent probes only with moe_latent_size', () => {
    const text = {
      layers_block_type: ['mamba', 'moe', 'attention', 'mlp'],
      n_routed_experts: 8,
      moe_latent_size: 1024,
    };
    const l = layerLayout(text, undefined, 'nemotron_h');
    expect(l.numLayers).toBe(4);
    expect(l.attention).toEqual(['linear', 'none', 'full', 'none']);
    expect(l.mlp).toEqual(['none', 'moe', 'none', 'dense']);
    expect(l.moeLatent).toBe(true);
    const byLayer = (set: ReturnType<typeof buildProbes>, i: number) =>
      set.probes.filter((x) => x.layer === i).map((x) => `${x.cls}:${x.leaf}`);

    // Super-VL naming (`language_model.backbone.`), picked by evidence.
    const zuper = buildProbes(
      { model_type: 'nemotron_h_omni' },
      text,
      '"language_model.backbone.layers.0.mixer.in_proj"',
    );
    expect(zuper.family).toBe('nemotron_h');
    expect(byLayer(zuper, 0)).toEqual([
      'linear_attention:mixer.in_proj',
      'linear_attention:mixer.out_proj',
    ]);
    expect(byLayer(zuper, 1)).toEqual([
      'routed_experts:mixer.experts.0.up_proj',
      'routed_experts:mixer.experts.0.down_proj',
      'shared_experts:mixer.shared_experts.up_proj',
      'shared_experts:mixer.shared_experts.down_proj',
      'moe_latent:mixer.fc1_latent_proj',
      'moe_latent:mixer.fc2_latent_proj',
    ]);
    expect(byLayer(zuper, 2)).toHaveLength(4);
    expect(byLayer(zuper, 3)).toEqual(['dense_mlp:mixer.up_proj', 'dense_mlp:mixer.down_proj']);
    expect(zuper.probes.find((x) => x.cls === 'lm_head')?.name).toBe('language_model.lm_head');
    expect(zuper.probes.find((x) => x.cls === 'embeddings')?.name).toBe(
      'language_model.backbone.embeddings',
    );

    // Lightning naming (`backbone.`), no latent: no moe_latent probes.
    const flat = { ...text, model_type: 'nemotron_h', moe_latent_size: null };
    const lightning = buildProbes(flat, flat, '');
    expect(lightning.probes.find((x) => x.cls === 'lm_head')?.name).toBe('lm_head');
    expect(
      lightning.probes.some((x) => x.name === 'backbone.layers.1.mixer.experts.0.up_proj'),
    ).toBe(true);
    expect(lightning.probes.some((x) => x.cls === 'moe_latent')).toBe(false);
  });

  it('Nemotron-H shortlist fixtures: 88 Super-VL layers with latent probes, none for Lightning', () => {
    const config = (id: string) => {
      const u = unwrapTextConfig(fixtureRepoFiles(id).config);
      if (u.kind !== 'ok') throw new Error(u.errors.join('; '));
      return u.view;
    };
    const s = config('nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4');
    const zuper = buildProbes(s.top, s.text, JSON.stringify(s.top.quantization_config));
    const count = (set: ReturnType<typeof buildProbes>, cls: string) =>
      set.probes.filter((x) => x.cls === cls).length;
    expect(zuper.layout.numLayers).toBe(88);
    // 40 Mamba, 40 MoE, 8 attention layers.
    expect([
      count(zuper, 'linear_attention'),
      count(zuper, 'routed_experts'),
      count(zuper, 'moe_latent'),
    ]).toEqual([80, 80, 80]);
    expect(count(zuper, 'attention')).toBe(32);
    const l = config('nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16');
    const lightning = buildProbes(l.top, l.text, '');
    expect(lightning.layout.numLayers).toBe(52);
    expect(count(lightning, 'moe_latent')).toBe(0);
    expect(count(lightning, 'routed_experts')).toBe(46);
  });
});
