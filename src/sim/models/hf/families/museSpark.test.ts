import { describe, expect, it } from 'vitest';
import { simulate } from '../../../engine/simulate';
import type { GqaAttentionBlock, LayerSpec, MoeBlock } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { M3_ULTRA_80C } from '../../../hw/presets/apple';
import { composeTopology, popularSetupById } from '../../../hw/presets/catalog';
import type { HardwareStack } from '../../../hw/types';
import { H100_SXM } from '../../../hw/presets/nvidia-datacenter';
import { hgxCluster } from '../../../hw/presets/topologies';
import type { OpResult, PhaseResult } from '../../../engine/types';
import { validateInput } from '../../../engine/validate';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { dispatchFamily } from '../dispatch';
import { mapHfModel } from '../index';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild, HfMapResult, JsonObject } from '../types';
import { museSparkFamily } from './museSpark';

const ID = 'unsloth/Muse-Spark-Open-816B-A42B';

/** Build the fixture with the Muse Spark family, optionally with text_config overrides. */
function build(override: JsonObject = {}): FamilyBuild {
  const files = fixtureRepoFiles(ID);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const text = { ...u.view.text, ...override };
  const r = museSparkFamily.build(text, { ...u.view.top, text_config: text }, files);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

function mapOk(): Extract<HfMapResult, { kind: 'ok' }> {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

const attn = (l: LayerSpec) =>
  l.blocks.find((b): b is GqaAttentionBlock => b.kind === 'gqa_attention');
const ffn = (l: LayerSpec) => l.blocks.find((b) => b.kind === 'moe' || b.kind === 'mlp');
const relErr = (actual: number, expected: number) => Math.abs(actual / expected - 1);

/** The 16 full-attention (NoPE) layers: every 4th counted back from the last, i = 1, 5, …, 61. */
const FULL = Array.from({ length: 16 }, (_, j) => 1 + 4 * j);

describe('muse_spark: Muse-Spark-Open-816B-A42B (unsloth mirror fixture)', () => {
  const b = build();
  const m = b.model;

  it('dispatches by architecture to muse_spark', () => {
    const u = unwrapTextConfig(fixtureRepoFiles(ID).config);
    if (u.kind !== 'ok') throw new Error('unwrap failed');
    expect(dispatchFamily(u.view)?.entry.family).toBe('muse_spark');
    const r = mapOk();
    expect(r.family).toBe('muse_spark');
    expect(r.modelType).toBe('muse_spark');
  });

  it('builds 62 layers: 46 sliding (window 2048, RoPE) + 16 full NoPE GQA layers, every layer a latent MoE', () => {
    expect(m.layers).toHaveLength(62);
    expect(m.hidden).toBe(8192);
    expect(m.vocab).toBe(202048);
    expect(m.tieEmbeddings).toBe(false);
    const mix: Record<string, number> = {};
    for (const l of m.layers) {
      const a = attn(l);
      const key = a?.slidingWindow !== undefined ? 'gqa:swa' : 'gqa:full';
      mix[key] = (mix[key] ?? 0) + 1;
      const f = ffn(l);
      if (f) mix[f.kind] = (mix[f.kind] ?? 0) + 1;
    }
    expect(mix).toEqual({ 'gqa:swa': 46, 'gqa:full': 16, moe: 62 });
    m.layers.forEach((l, i) => {
      const a = attn(l);
      const full = FULL.includes(i);
      expect(a?.slidingWindow === undefined, `layer ${i}`).toBe(full);
      // NoPE exactly on the full layers (layer_rope_theta[i] == 0).
      expect(a?.rope, `layer ${i}`).toBe(!full);
    });
    expect(attn(m.layers[0])).toMatchObject({
      heads: 128,
      kvHeads: 16,
      headDim: 64,
      qkNorm: true,
      outputGate: true,
      slidingWindow: 2048,
    });
    const moe = ffn(m.layers[0]) as MoeBlock;
    expect(moe).toMatchObject({
      experts: 256,
      topK: 8,
      expertIntermediate: 4096,
      latentDim: 4096,
      expertOutputNorm: true,
      sharedExperts: 0,
      gated: true,
    });
    // Gemma-style sandwich: pre- and post-norm around both sublayers.
    expect(m.layers[0].blocks.filter((x) => x.kind === 'norm')).toHaveLength(4);
  });

  it('total and active params match the module definitions', () => {
    // Exact text model from the code: 819,990,552,064 (= safetensors.total − vision − audio). The IR leaves out the
    // router bias, residual gates and latent norms and counts one extra norm vector + q/k-norm: −16,512 per layer.
    const total = countParams(m);
    expect(total).toBe(819_990_552_064 - 62 * 16_512);
    expect(relErr(total, 819_990_552_064)).toBeLessThan(1e-4);
    // The name's "816B" are the decoder layers (total without embedding, LM head and final norm): 816.68B.
    const decoder = total - 2 * m.vocab * m.hidden - m.hidden;
    expect(relErr(decoder, 816e9)).toBeLessThan(0.02);
    // Active: decoder layers 62 · 690,012,416 = 42.78B ("A42B"); with embedding + LM head 46.09B.
    const active = activeParamsPerToken(m);
    expect(relErr(active, 46.09e9)).toBeLessThan(1e-3);
    const activeDecoder = active - 2 * m.vocab * m.hidden - m.hidden;
    expect(relErr(activeDecoder, 62 * 690_012_416)).toBeLessThan(1e-4);
    expect(relErr(activeDecoder, 42e9)).toBeLessThan(0.02);
  });

  it('counts the vision and audio towers exactly; the param check is ok', () => {
    // Vision tower 1,852,639,744 + adapter/projection 75,497,472 + audio tower 117,840,384.
    expect(b.visionParamsEstimate).toBe(2_045_977_600);
    expect(b.mtpParamsEstimate).toBe(0);
    expect(b.excludedParts).toEqual(['vision tower', 'audio tower']);
    const r = mapOk();
    expect(r.paramCheck.status).toBe('ok');
    expect(r.paramCheck.expected).toBe(822_036_529_664);
    // The gap is the ≈1.0 M vectors the IR leaves out (router bias, residual gates, latent norms; see above):
    // −1.2e-6 of the checkpoint.
    expect(Math.abs(r.paramCheck.relError ?? 1)).toBeLessThan(1e-4);
  });

  it('defaults to bf16 weights and KV cache with no quant warnings', () => {
    const r = mapOk();
    expect(r.quant.source).toBe('none');
    expect(r.quant.quant.weights.dtype).toBe('bf16');
    expect(r.quant.quant.kvCache.dtype).toBe('bf16');
    expect(r.quant.notes).toEqual([]);
    expect(r.notes.some((n) => n.startsWith('Language model only'))).toBe(true);
  });

  it('builds a sibling from config fields: truncated lists, a dense MLP layer, full-width latent projections', () => {
    const s = build({
      num_hidden_layers: 8,
      layer_types: Array.from({ length: 8 }, (_, i) =>
        i % 2 ? 'full_attention' : 'sliding_attention',
      ),
      layer_rope_theta: Array.from({ length: 8 }, (_, i) => (i % 2 ? 0 : 1e6)),
      mlp_layer_types: ['dense', ...Array<string>(7).fill('sparse')],
      intermediate_size: 12288,
      moe_hidden_size: 8192,
      sliding_window: 1024,
    }).model;
    expect(s.layers).toHaveLength(8);
    expect(ffn(s.layers[0])).toMatchObject({ kind: 'mlp', intermediate: 12288 });
    const moe = ffn(s.layers[1]) as MoeBlock;
    expect(moe.kind).toBe('moe');
    // pre/post_expert_proj exist at any width: moe_hidden_size == hidden still gives two 8192 × 8192 projections.
    expect(moe.latentDim).toBe(8192);
    expect(moe.expertOutputNorm).toBe(true);
    expect(attn(s.layers[0])).toMatchObject({ slidingWindow: 1024, rope: true });
    expect(attn(s.layers[1])).toMatchObject({ rope: false });
    expect(attn(s.layers[1])?.slidingWindow).toBeUndefined();
  });

  it('falls back to the transformers defaults when layer_types / layer_rope_theta / mlp_layer_types are absent', () => {
    expect(
      build({ layer_types: null, layer_rope_theta: null, mlp_layer_types: null }).model.layers,
    ).toEqual(m.layers);
  });

  it('rejects unknown layer types and short lists', () => {
    const files = fixtureRepoFiles(ID);
    const u = unwrapTextConfig(files.config);
    if (u.kind !== 'ok') throw new Error('unwrap failed');
    const text = { ...u.view.text, layer_types: ['chunked_attention', 'full_attention'] };
    const r = museSparkFamily.build(text, { ...u.view.top, text_config: text }, files);
    expect(r.kind).toBe('invalid_config');
    if (r.kind !== 'invalid_config') return;
    expect(r.errors).toContain('text_config.layer_types: 2 entries for 62 layers');
    expect(r.errors).toContain("text_config.layer_types[0]: unknown type 'chunked_attention'");
  });

  it("probes Muse's checkpoint names: an fp8 export that keeps the latent projections in bf16", () => {
    // Synthetic FineGrainedFP8 config (no such repo is known): the module names are the transformers ones.
    const files = fixtureRepoFiles(ID);
    const config = {
      ...(files.config as JsonObject),
      quantization_config: {
        quant_method: 'fp8',
        weight_block_size: [128, 128],
        modules_to_not_convert: [
          'lm_head',
          'model.language_model.layers.*.mlp.pre_expert_proj',
          'model.language_model.layers.*.mlp.post_expert_proj',
        ],
      },
    };
    const r = mapHfModel({ ...files, config });
    if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
    const modules = r.quant.quant.modules;
    expect(modules?.moe_latent?.weights.dtype).toBe('bf16');
    expect(modules?.routed_experts?.weights.dtype).toBe('fp8_e4m3');
    expect(modules?.attention?.weights.dtype).toBe('fp8_e4m3');
  });
});

describe('muse_spark end to end: decode @ 8k on 4 × M3 Ultra 512 GB (TB5 RDMA mesh)', () => {
  const setup = popularSetupById('m3-ultra-80c-512gb-x4-tb5-mesh');
  if (setup === undefined) throw new Error('popular setup missing');
  const hardware: HardwareStack = {
    device: M3_ULTRA_80C,
    topology: composeTopology(setup.choice),
    memCapacityBytes: 512e9,
  };
  const r = mapOk();
  const workload = { batch: 1, promptLen: 8192, decodeContextLen: 8192 };

  it('pp = 4 fits and decodes at ≥ 0.8× of the memory-bandwidth bound', () => {
    const res = simulate({
      model: r.model,
      quant: r.quant.quant,
      parallel: { tp: 1, pp: 4, dp: 1, ep: 1 },
      workload,
      hardware,
    });
    // ≈ 410 GB of bf16 weights per stage on 512 GB machines.
    expect(res.memory.fits).toBe(true);
    for (const p of [res.prefill, res.decode])
      expect(Number.isFinite(p.tokensPerSec) && p.tokensPerSec > 0).toBe(true);
    // Bandwidth bound: at batch 1 the 4 stages run one after the other, so each token reads every active byte once
    // from one machine's memory: bf16 active weights 2·(42.78B decoder + 1.655B LM head) ≈ 88.9 GB (the embedding is
    // a row lookup), plus the KV cache @ 8k, 16 full layers · 8192 · 4096 B + 46 sliding layers · 2048 · 4096 B
    // ≈ 0.92 GB: ≈ 89.8 GB. t ≥ 89.8 GB / 819 GB/s = 0.110 s ⇒ ≤ 9.1 tok/s @ 8k.
    const activeWeights = activeParamsPerToken(r.model) - r.model.vocab * r.model.hidden;
    const kvBytes = (16 * 8192 + 46 * 2048) * 2 * 16 * 64 * 2;
    const bytesPerToken = 2 * activeWeights + kvBytes;
    const bound = M3_ULTRA_80C.memory.bandwidthBytesPerSec / bytesPerToken;
    expect(bound).toBeGreaterThan(9);
    expect(bound).toBeLessThan(9.2);
    expect(res.decode.tokensPerSec).toBeLessThanOrEqual(bound);
    expect(res.decode.tokensPerSec).toBeGreaterThanOrEqual(0.8 * bound);
    // Unified memory is not reduced by an OS share; that is the only warning allowed.
    expect(res.warnings.filter((w) => !/unified/i.test(w))).toEqual([]);
  });
});

// Regression: the expert-TP collectives of post_expert_norm used the whole TP group's path (hierarchical 8 × 2 for
// a group of 8 at TP16/EP2) and threw. The group is the ranks j + i·ep of a TP group (EP groups are ep consecutive
// ranks), so its links depend on ep (placement.ts: expertTpGroupSpan).
describe('muse_spark end to end: expert-TP collectives on 2 × 8 H100 (NVLink 4 / IB NDR)', () => {
  const hardware: HardwareStack = { device: H100_SXM, topology: hgxCluster('h100', 2) };
  const r = mapOk();
  const workload = { batch: 2, promptLen: 1024, decodeContextLen: 1024 };
  const run = (tp: number, ep: number, dp = 1) =>
    simulate({
      model: r.model,
      quant: r.quant.quant,
      parallel: { tp, pp: 1, dp, ep },
      workload,
      hardware,
    });
  const etpOps = (p: PhaseResult): OpResult[] =>
    p.stages
      .flatMap((s) => s.layers.flatMap((l) => l.ops))
      .filter((o) => o.op.kind === 'collective' && o.op.group === 'etp');

  it.each([
    // [tp, ep, dp, group size, link level]
    [16, 2, 1, 8, 'hierarchical'], // ranks 0, 2, …, 14: 4 per node × 2 nodes
    [16, 4, 1, 4, 'hierarchical'], // ranks 0, 4, 8, 12: 2 per node × 2 nodes
    [16, 8, 1, 2, 'inter'], // ranks 0, 8: one per node
    [8, 2, 2, 4, 'intra'], // a TP group inside one node
  ] as const)(
    'tp %i ep %i dp %i: expert-TP group of %i on the %s path',
    (tp, ep, dp, size, level) => {
      const res = run(tp, ep, dp);
      for (const p of [res.prefill, res.decode]) {
        expect(Number.isFinite(p.tokensPerSec) && p.tokensPerSec > 0).toBe(true);
        const ops = etpOps(p);
        // 62 layers × (reduce-scatter + sum-of-squares all-reduce)
        expect(ops).toHaveLength(2 * 62);
        for (const o of ops) {
          expect(o.op.kind === 'collective' && o.op.groupSize).toBe(size);
          expect(o.linkLevel).toBe(level);
          expect(o.sec).toBeGreaterThan(0);
        }
      }
    },
  );

  it('tp 16 ep 16 (etp 1): no expert-TP collectives', () => {
    const res = run(16, 16);
    expect(etpOps(res.decode)).toEqual([]);
    expect(res.decode.tokensPerSec).toBeGreaterThan(0);
  });

  it('a ring of single-device nodes with non-adjacent expert-TP members is a validation error, not a crash', () => {
    const setup = popularSetupById('m3-ultra-80c-512gb-x4-tb5-mesh');
    if (setup === undefined) throw new Error('popular setup missing');
    const ring = { ...composeTopology(setup.choice), interNodeFabric: { kind: 'ring' as const } };
    // tp 4, ep 2: ranks 0 and 2 hold the same experts, on nodes 0 and 2
    const input = {
      model: r.model,
      quant: r.quant.quant,
      parallel: { tp: 4, pp: 1, dp: 1, ep: 2 },
      workload,
      hardware: { device: M3_ULTRA_80C, topology: ring, memCapacityBytes: 512e9 },
    };
    expect(validateInput(input).filter((e) => e.startsWith('etp:'))).toEqual([
      'etp: expert-TP group of 2 nodes 2 apart on a ring fabric is not modeled (needs ep ≤ devices per node, or a switched or full-mesh fabric)',
    ]);
  });
});
