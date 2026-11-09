import { describe, expect, it } from 'vitest';
import type { BlockSpec, LayerSpec, ModelSpec, QuantConfig } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { blockSequenceStateBytes } from '../../../lower/lower';
import type { LoweringContext } from '../../../lower/lower';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { dispatchFamily } from '../dispatch';
import { mapHfModel } from '../index';
import { checkParams } from '../paramCheck';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild, HfMapResult, JsonObject } from '../types';
import { layerParams } from './mlaCommon';
import { nemotronHFamily } from './nemotronH';

const LIGHTNING = 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16';
const SUPER = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4';

/** Build `id`'s fixture with the family directly, with `override` merged into the language-model config. */
function build(id: string, override: JsonObject = {}): FamilyBuild {
  const files = fixtureRepoFiles(id);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const text = { ...u.view.text, ...override };
  const top = u.view.nested ? { ...u.view.top, llm_config: text } : text;
  const r = nemotronHFamily.build(text, top, files);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

function mapOk(id: string): Extract<HfMapResult, { kind: 'ok' }> {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

/** Label of a layer's single non-norm, non-residual block. */
function role(layer: LayerSpec): string {
  const b = layer.blocks.find((x) => x.kind !== 'norm' && x.kind !== 'residual') as BlockSpec;
  return b.kind === 'linear_attention'
    ? 'mamba'
    : b.kind === 'gqa_attention'
      ? 'attention'
      : b.kind === 'moe'
        ? 'moe'
        : 'mlp';
}
const indicesOf = (m: ModelSpec, r: string) =>
  m.layers.flatMap((l, i) => (role(l) === r ? [i] : []));

/** Σ over every block of the per-sequence state at `tokens` of context (tp = 1). Unit: bytes. */
function stateBytes(model: ModelSpec, quant: QuantConfig, tokens: number): number {
  const ctx: LoweringContext = {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: tokens,
    quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
  };
  return model.layers.reduce(
    (s, l) => s + l.blocks.reduce((t, b) => t + blockSequenceStateBytes(b, ctx, tokens), 0),
    0,
  );
}

describe('nemotron_h dispatch', () => {
  it('claims the flat config by architecture and the omni wrapper by its own architecture', () => {
    for (const id of [LIGHTNING, SUPER]) {
      const u = unwrapTextConfig(fixtureRepoFiles(id).config);
      if (u.kind !== 'ok') throw new Error('unwrap failed');
      const hit = dispatchFamily(u.view);
      expect(hit?.entry.family, id).toBe('nemotron_h');
      expect(hit?.key, id).toBe('architecture');
    }
  });
});

describe('nemotron_h: Lightning 30B-A3B (flat config)', () => {
  const b = build(LIGHTNING);
  const m = b.model;

  it('maps layers_block_type to 23 Mamba-2, 6 attention (no RoPE), 23 MoE layers, one block each', () => {
    expect(m.layers).toHaveLength(52);
    expect(m.hidden).toBe(2688);
    expect(indicesOf(m, 'mamba')).toHaveLength(23);
    expect(indicesOf(m, 'moe')).toHaveLength(23);
    expect(indicesOf(m, 'attention')).toEqual([5, 12, 19, 26, 33, 42]);
    expect(m.layers[0].blocks.map((x) => x.kind)).toEqual(['norm', 'linear_attention', 'residual']);
    expect(m.layers[0].blocks[1]).toMatchObject({
      variant: 'mamba2',
      qkHeads: 8,
      vHeads: 64,
      keyDim: 128,
      valueDim: 64,
      convKernel: 4,
      outputGate: true,
      convBias: true,
      chunkTokens: 128,
    });
    expect(m.layers[5].blocks[1]).toMatchObject({
      kind: 'gqa_attention',
      heads: 32,
      kvHeads: 2,
      headDim: 128,
      rope: false,
      qkvBias: false,
    });
    const moe = m.layers[1].blocks[1];
    expect(moe).toMatchObject({
      kind: 'moe',
      experts: 128,
      topK: 6,
      expertIntermediate: 1856,
      sharedExperts: 1,
      sharedIntermediate: 3712,
      gated: false,
      activation: 'relu2',
    });
    expect(moe).not.toHaveProperty('latentDim'); // moe_latent_size is null
    expect(m.finalNorm).toBe('rmsnorm');
    expect(m.tieEmbeddings).toBe(false);
  });

  it('matches the checkpoint headers: per-layer, total and active parameters', () => {
    expect(layerParams(m.layers[0])).toBe(38_744_896); // Mamba-2 incl. pre-norm
    expect(layerParams(m.layers[5])).toBe(23_399_040); // attention incl. pre-norm
    expect(layerParams(m.layers[1])).toBe(1_297_468_032); // MoE incl. pre-norm
    // = api safetensors.parameters BF16 sum without the MTP module.
    expect(countParams(m)).toBe(31_577_937_344);
    // Without the embedding table (a row lookup): the model card's "3B active" (3.58B with embedding and head).
    expect(activeParamsPerToken(m) - m.vocab * m.hidden).toBe(3_227_751_872);
  });

  it('keeps MTP out of the main pass: a one-module drafter (attention + MoE) and its parameter estimate', () => {
    expect(b.mtpParamsEstimate).toBe(1_335_325_824);
    expect(b.excludedParts).toEqual(['MTP']);
    expect(m.draft).toMatchObject({ kind: 'mtp', hiddenProj: true, maxReliableDepth: 1 });
    if (m.draft?.kind !== 'mtp') throw new Error('no mtp draft');
    expect(m.draft.layers).toHaveLength(1);
    expect(m.draft.layers[0].name).toBe('mtp.0');
    expect(m.draft.layers[0].blocks.map((x) => x.kind)).toEqual([
      'norm',
      'gqa_attention',
      'residual',
      'norm',
      'moe',
      'residual',
    ]);
  });

  it("param check: ok against the per-dtype sum (MTP included); the gap is the 24 routers' e_score_correction_bias", () => {
    const r = mapOk(LIGHTNING);
    expect(r.family).toBe('nemotron_h');
    expect(r.paramCheck.status).toBe('ok');
    expect(r.paramCheck.expected).toBe(32_913_266_240);
    // actual = 31,577,937,344 + 1,335,325,824; 3072 = 24 MoE layers (23 + the MTP one) × 128 bias elements.
    expect(r.paramCheck.actual).toBe(32_913_263_168);
    expect(32_913_266_240 - r.paramCheck.actual).toBe(24 * 128);
    expect(Math.abs(r.paramCheck.relError ?? 1)).toBeLessThan(1e-7);
    expect(checkParams(m, b, fixtureRepoFiles(LIGHTNING).api, 'bf16').status).toBe('ok');
  });

  it('quant defaults: unquantized bf16, fp32 SSM state (mamba_ssm_cache_dtype)', () => {
    const q = mapOk(LIGHTNING).quant.quant;
    expect(q).toEqual({
      weights: { dtype: 'bf16' },
      activations: 'bf16',
      kvCache: { dtype: 'bf16' },
      recurrentState: 'fp32',
    });
  });

  it('sequence state: 23 × 2,134,016 B of SSM + conv state, and 6 × 2·2·128 bf16 KV entries = 6,144 B/token', () => {
    const q = mapOk(LIGHTNING).quant.quant;
    // S_layer = H·N·P·4 B (fp32) + C·(K−1)·2 B = 64·128·64·4 + 10,240... C = 4096 + 2·8·128 = 6144 → 6144·3·2.
    expect(23 * (64 * 128 * 64 * 4 + 6144 * 3 * 2)).toBe(49_082_368);
    const at = (n: number) => stateBytes(m, q, n);
    expect(at(1 << 16) - 6 * 2 * 2 * 128 * 2 * (1 << 16)).toBe(49_082_368);
    expect((at((1 << 16) + 4096) - at(1 << 16)) / 4096).toBe(6144);
  });

  it('reads the legacy hybrid_override_pattern string and the transformers layer-type names identically', () => {
    const pattern = 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME';
    expect(
      build(LIGHTNING, { layers_block_type: null, hybrid_override_pattern: pattern }).model.layers,
    ).toEqual(m.layers);
    const renamed = (m.layers.map((l) => role(l)) as string[]).map(
      (r) =>
        ({ mamba: 'linear_attention', attention: 'full_attention', moe: 'moe', mlp: 'mlp' })[r],
    );
    expect(build(LIGHTNING, { layers_block_type: renamed }).model.layers).toEqual(m.layers);
  });

  it("a '-' layer is a dense non-gated relu² MLP; a disagreeing num_hidden_layers is noted", () => {
    const r = build(LIGHTNING, { layers_block_type: null, hybrid_override_pattern: 'M-' });
    expect(r.model.layers).toHaveLength(2);
    expect(r.model.layers[1].blocks[1]).toMatchObject({
      kind: 'mlp',
      intermediate: 1856,
      gated: false,
      activation: 'relu2',
      bias: false,
    });
    expect(r.notes.some((n) => n.includes('num_hidden_layers (52) differs'))).toBe(true);
  });

  it('reports a missing or invalid layer list as invalid_config', () => {
    const files = fixtureRepoFiles(LIGHTNING);
    const u = unwrapTextConfig(files.config);
    if (u.kind !== 'ok') throw new Error('unwrap failed');
    const none = nemotronHFamily.build(
      { ...u.view.text, layers_block_type: null },
      u.view.top,
      files,
    );
    expect(none).toMatchObject({ kind: 'invalid_config' });
    const bad = nemotronHFamily.build(
      { ...u.view.text, layers_block_type: ['mamba', 'conv'] },
      u.view.top,
      files,
    );
    expect(bad).toMatchObject({
      kind: 'invalid_config',
      errors: [expect.stringContaining('layers_block_type[1]')],
    });
  });

  it('reports an inherited object key as a layer type as invalid_config, not a throw', () => {
    const files = fixtureRepoFiles(LIGHTNING);
    const config = { ...(files.config as JsonObject), layers_block_type: ['constructor'] };
    const r = mapHfModel({ ...files, config });
    expect(r).toMatchObject({
      kind: 'invalid_config',
      errors: [expect.stringContaining('layers_block_type[0]: unknown layer type "constructor"')],
    });
  });
});

describe('nemotron_h: Super-VL 120B-A12B NVFP4 (omni wrapper, llm_config)', () => {
  const b = build(SUPER);
  const m = b.model;

  it('maps 40 Mamba-2, 8 attention and 40 latent-MoE layers (88 layers)', () => {
    expect(m.layers).toHaveLength(88);
    expect(m.hidden).toBe(4096);
    expect(indicesOf(m, 'mamba')).toHaveLength(40);
    expect(indicesOf(m, 'moe')).toHaveLength(40);
    expect(indicesOf(m, 'attention')).toEqual([7, 16, 25, 36, 47, 58, 69, 78]);
    expect(m.layers[0].blocks[1]).toMatchObject({
      variant: 'mamba2',
      qkHeads: 8,
      vHeads: 128,
      keyDim: 128,
      valueDim: 64,
      chunkTokens: 128,
    });
    expect(m.layers[1].blocks[1]).toMatchObject({
      kind: 'moe',
      experts: 512,
      topK: 22,
      expertIntermediate: 2688,
      sharedIntermediate: 5376,
      latentDim: 1024,
      gated: false,
      activation: 'relu2',
    });
  });

  it('matches the checkpoint headers: per-layer, total and active parameters', () => {
    expect(layerParams(m.layers[0])).toBe(109_640_064);
    expect(layerParams(m.layers[7])).toBe(35_655_680);
    expect(layerParams(m.layers[1])).toBe(2_873_102_336);
    // Header main-model sum 120,668,707,840 minus 40 routers' e_score_correction_bias (512 each).
    expect(countParams(m)).toBe(120_668_687_360);
    expect(activeParamsPerToken(m) - m.vocab * m.hidden).toBe(12_233_346_048); // card: "12B active"
  });

  it('simulates the language model only; MTP is the drafter and the vision tower is estimated', () => {
    expect(b.mtpParamsEstimate).toBe(2_942_324_736);
    // ViT 32 × (4·1280² + 2·1280·4·1280) = 629,145,600; mlp1: 5120 + 5120·20480 + 20480·4096 = 188,748,800.
    expect(b.visionParamsEstimate).toBe(629_145_600 + 188_748_800);
    expect(b.excludedParts).toEqual(['MTP', 'vision tower', 'mlp1 connector']);
    expect(b.notes.some((n) => n.startsWith('Language model only'))).toBe(true);
    expect(m.draft).toMatchObject({ kind: 'mtp', hiddenProj: true, maxReliableDepth: 1 });
  });

  it('param check: packed (U8 words), with the NVFP4 mixed-precision defaults', () => {
    const r = mapOk(SUPER);
    expect(r.family).toBe('nemotron_h');
    expect(r.paramCheck.status).toBe('packed');
    expect(r.quant.scheme).toContain('MIXED_PRECISION');
    const q = r.quant.quant;
    expect(q.weights).toEqual({ dtype: 'nvfp4' });
    expect(q.kvCache).toEqual({ dtype: 'fp8_e4m3' });
    expect(q.recurrentState).toBe('fp32');
    expect(q.modules?.linear_attention?.weights).toEqual({ dtype: 'fp8_e4m3' });
    expect(q.modules?.attention?.weights).toEqual({ dtype: 'fp8_e4m3' });
    expect(q.modules?.routed_experts?.weights).toEqual({ dtype: 'nvfp4' });
    expect(q.modules?.shared_experts?.weights).toEqual({ dtype: 'nvfp4' });
    expect(q.modules?.moe_latent?.weights).toEqual({ dtype: 'bf16' });
    expect(q.lmHead).toEqual({ dtype: 'fp8_e4m3' });
  });

  it('sequence state: 40 × 4,255,744 B of SSM + conv state, and 8 × 2·2·128 fp8 KV entries = 4,096 B/token', () => {
    const q = mapOk(SUPER).quant.quant;
    // S_layer = 128·128·64·4 B (fp32) + C·(K−1)·2 B with C = 8192 + 2·8·128 = 10,240.
    expect(40 * (128 * 128 * 64 * 4 + 10_240 * 3 * 2)).toBe(170_229_760);
    const at = (n: number) => stateBytes(m, q, n);
    expect(at(1 << 16) - 8 * 2 * 2 * 128 * (1 << 16)).toBe(170_229_760);
    expect((at((1 << 16) + 4096) - at(1 << 16)) / 4096).toBe(4096);
  });
});
