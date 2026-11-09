/**
 * Family `muse_spark`: Meta Muse Spark (`MuseSparkForConditionalGeneration`; language model under `text_config`,
 * `model_type` 'muse_spark_text', vision and audio towers beside it). Read from transformers 5.22.1
 * `models/muse_spark/{configuration,modeling}_muse_spark.py`.
 *
 * Every layer, from config lists:
 * - Attention: GQA (`num_attention_heads`, `num_key_value_heads`, `head_dim`) with a weightless per-head q/k RMSNorm
 *   and a sigmoid output gate (`gate_proj: h → heads·head_dim`, the IR `outputGate`).
 *   `layer_types[i]` 'sliding_attention' → a window of `sliding_window` tokens, 'full_attention' → full attention.
 *   `layer_rope_theta[i] == 0` → NoPE (no rotary on that layer).
 *   Absent lists take the transformers defaults: full and NoPE where `(n − 1 − i) % 4 == 0`, else sliding with RoPE.
 * - FFN: `mlp_layer_types[i]` 'sparse' → latent MoE: the router reads the hidden h, `pre_expert_proj` h → l
 *   (`moe_hidden_size`), `num_local_experts` SwiGLU experts of width `moe_intermediate_size` at l, each expert's
 *   output through `post_expert_norm` (IR `expertOutputNorm`), `post_expert_proj` l → h. The projections always
 *   exist, also when `moe_hidden_size == hidden_size`. 'dense' → a SwiGLU MLP of `intermediate_size`. Absent
 *   list: every layer sparse.
 * - Norms: Gemma-style sandwich (pre- and post-norm around both sublayers; IR `postNorms`).
 *
 * Approximations (all negligible next to the GEMMs): the learned residual gates are costed as plain residual adds
 * (same bytes); the weightless q/k, o and embedding norms, the latent `pre_expert_norm`, the fp32 router and the
 * logit softcap are not modeled as separate ops. Their vectors (≈ 1.0 M params, with the `post_expert_norm` γ) are
 * left out of the count.
 */

import type { FfnParams, MixerParams } from '../../../ir/builders';
import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import { modelSpecErrors } from '../../../ir/validateModel';
import type { ConfigReader } from '../reader';
import { createReader } from '../reader';
import type { FamilyBuildResult, FamilyEntry, JsonObject } from '../types';

/** `layer_types` values of the text config. */
const LAYER_TYPES = ['sliding_attention', 'full_attention'];
/** `mlp_layer_types` values of the text config. */
const MLP_TYPES = ['sparse', 'dense'];

/**
 * Params of the vision tower, its adapter and projection (`MuseSparkVisionModel`, `MuseSparkVisionAdapter`,
 * `vision_projection`), exact from the module definitions. With d = `hidden_size`, f = `intermediate_size`,
 * L = `num_hidden_layers`, p = `patch_size`, t = `patch_temporal`, o = top `out_hidden_size`,
 * a = top `projector_hidden_size`, h = text hidden:
 *
 *   P_layer   = 4·d² + 4·d (q, k, v, proj with biases) + 2·d·f + f + d (fc1, fc2 with biases) + 4·d (2 LayerNorms)
 *   P_vision  = L·P_layer + 3·t·p²·d (patch embedding, no bias) + pos_h·pos_w·d (position table) + 4·d (ln_pre, ln_post)
 *   P_adapter = o·a + a·a + a·h
 *
 * Unit: params. 0 without a `vision_config`.
 */
function visionParams(top: ConfigReader, hidden: number): number {
  const v = top.optObj('vision_config');
  if (v === undefined) return 0;
  const d = v.int('hidden_size');
  const f = v.int('intermediate_size');
  const layers = v.int('num_hidden_layers');
  const patch = v.optInt('patch_size') ?? 14;
  const temporal = v.optInt('patch_temporal') ?? 2;
  const posTable = (v.optInt('pos_emb_height') ?? 32) * (v.optInt('pos_emb_width') ?? 32);
  const out = top.optInt('out_hidden_size') ?? 6144;
  const adapter = top.optInt('projector_hidden_size') ?? 4096;
  const perLayer = 4 * d * d + 2 * d * f + 9 * d + f;
  const tower = layers * perLayer + 3 * temporal * patch * patch * d + posTable * d + 4 * d;
  return tower + out * adapter + adapter * adapter + adapter * hidden;
}

/**
 * Params of the audio tower (`MuseSparkAudioModel`), exact from the module definitions. Stage i has width
 * hᵢ = `hidden_sizes[i]`, input c = hᵢ₋₁ (1 for the first), patch pᵢ, `depths[i]` conv blocks, k = `conv_kernel_size`,
 * r = `mlp_ratio`, o = `out_hidden_size`:
 *
 *   P_stage = c·pᵢ·hᵢ + 2·hᵢ (patch proj with bias, RMSNorm)
 *           + depthᵢ·(k·hᵢ + hᵢ (depthwise conv with bias) + hᵢ (RMSNorm) + 2·r·hᵢ² + r·hᵢ + hᵢ (fc1, fc2 with biases))
 *   P_audio = Σ P_stage + h_last (norm) + h_last·o + o (proj with bias) + o (post-proj norm)
 *
 * Unit: params. 0 without an `audio_config`.
 */
function audioParams(top: ConfigReader): number {
  const a = top.optObj('audio_config');
  if (a === undefined) return 0;
  const widths = a.numArray('hidden_sizes');
  const depths = a.numArray('depths');
  const patches = a.numArray('patch_sizes');
  const kernel = a.optInt('conv_kernel_size') ?? 7;
  const ratio = a.optNum('mlp_ratio') ?? 4;
  const out = a.int('out_hidden_size');
  let total = 0;
  widths.forEach((h, i) => {
    const c = i === 0 ? 1 : widths[i - 1];
    const ffn = Math.trunc(h * ratio);
    const block = kernel * h + h + h + 2 * h * ffn + ffn + h;
    total += c * (patches[i] ?? 0) * h + 2 * h + (depths[i] ?? 0) * block;
  });
  const last = widths.at(-1) ?? 0;
  return total + last + last * out + out + out;
}

/** Build the Muse Spark language-model IR from its text config. */
export function buildMuseSpark(
  text: JsonObject,
  top: JsonObject,
  repoId: string,
): FamilyBuildResult {
  const r = createReader(text, text === top ? '' : 'text_config');
  const t = createReader(top, '', r.errors);
  const at = (key: string) => (r.path ? `${r.path}.${key}` : key);
  const hidden = r.int('hidden_size');
  const n = r.int('num_hidden_layers');
  const vocab = r.int('vocab_size');
  const tie = r.bool('tie_word_embeddings', false);
  const heads = r.int('num_attention_heads');
  const kvHeads = r.int('num_key_value_heads');
  const headDim = r.optInt('head_dim') ?? (heads > 0 ? hidden / heads : 0);
  const window = r.int('sliding_window');
  const act = r.optStr('hidden_act') ?? 'silu';
  if (act !== 'silu' && act !== 'gelu' && act !== 'relu')
    r.errors.push(`${at('hidden_act')}: unsupported activation '${act}'`);
  const activation = act === 'gelu' || act === 'relu' ? act : 'silu';
  const ropeParams = r.optObj('rope_parameters');
  const ropeTheta = ropeParams?.optNum('rope_theta') ?? r.optNum('rope_theta') ?? 500_000;

  // transformers defaults (`MuseSparkTextConfig.__post_init__`): every 4th layer counted back from the last is
  // full attention without RoPE; every MLP is sparse.
  const isDefaultFull = (i: number) => (n - 1 - i) % 4 === 0;
  const layerTypes =
    r.optStrArray('layer_types') ??
    Array.from({ length: n }, (_, i) =>
      isDefaultFull(i) ? 'full_attention' : 'sliding_attention',
    );
  const ropeThetas =
    r.optNumArray('layer_rope_theta') ??
    Array.from({ length: n }, (_, i) => (isDefaultFull(i) ? 0 : ropeTheta));
  const mlpTypes = r.optStrArray('mlp_layer_types') ?? Array<string>(n).fill('sparse');
  for (const [key, list, known] of [
    ['layer_types', layerTypes, LAYER_TYPES],
    ['layer_rope_theta', ropeThetas, null],
    ['mlp_layer_types', mlpTypes, MLP_TYPES],
  ] as const) {
    if (list.length < n) r.errors.push(`${at(key)}: ${list.length} entries for ${n} layers`);
    if (known !== null) {
      list.slice(0, n).forEach((v, i) => {
        if (!known.includes(String(v)))
          r.errors.push(`${at(key)}[${i}]: unknown type '${String(v)}'`);
      });
    }
  }

  const sparse = mlpTypes.slice(0, n).includes('sparse');
  const dense = mlpTypes.slice(0, n).includes('dense');
  const moeHidden = sparse ? r.int('moe_hidden_size') : 0;
  const moe: FfnParams | undefined = sparse
    ? {
        kind: 'moe',
        experts: r.int('num_local_experts'),
        topK: r.int('num_experts_per_tok'),
        expertIntermediate: r.int('moe_intermediate_size'),
        // The experts run at moe_hidden_size between pre/post_expert_proj, which exist at any width; each expert's
        // output passes post_expert_norm before the weighted sum (modeling_muse_spark.py MuseSparkExperts).
        latentDim: moeHidden,
        expertOutputNorm: true,
        sharedExperts: r.optInt('num_shared_experts') ?? 0,
        activation,
      }
    : undefined;
  const mlp: FfnParams | undefined = dense
    ? { kind: 'dense', intermediate: r.int('intermediate_size'), activation }
    : undefined;

  const vision = visionParams(t, hidden);
  const audio = audioParams(t);
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

  const mixer = (i: number): MixerParams => ({
    kind: 'gqa',
    heads,
    kvHeads,
    headDim,
    qkNorm: true,
    outputGate: true,
    rope: ropeThetas[i] !== 0,
    ...(layerTypes[i] === 'sliding_attention' ? { slidingWindow: window } : {}),
  });
  const layers = repeatLayers(n, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden,
      mixer: mixer(i),
      // Both are defined whenever a layer uses them (checked above).
      ffn: (mlpTypes[i] === 'dense' ? mlp : moe) as FfnParams,
      postNorms: true,
    }),
  );
  const model = decoderModel({
    name: repoId,
    hidden,
    vocab,
    tieEmbeddings: tie,
    layers,
    source: repoId,
  });
  const errors = modelSpecErrors(model);
  if (errors.length > 0) return { kind: 'invalid_config', errors };

  const full = layerTypes.slice(0, n).filter((x) => x === 'full_attention').length;
  const nope = ropeThetas.slice(0, n).filter((x) => x === 0).length;
  const excludedParts: string[] = [];
  if (vision > 0) excludedParts.push('vision tower');
  if (audio > 0) excludedParts.push('audio tower');
  return {
    kind: 'ok',
    model,
    notes: [
      'Language model only: the vision tower (with its adapter) and the audio tower are excluded; their parameters are counted from config for the param check.',
      `${n - full} sliding-window layers (window ${window}) and ${full} full-attention layers; ${nope} layers without RoPE.`,
      ...(sparse
        ? [
            `Latent MoE: routed experts at width ${moeHidden} between ${hidden} → ${moeHidden} → ${hidden} projections.`,
          ]
        : []),
      'Residual gates are costed as residual adds. Weightless q/k, o and embedding norms, the latent pre-expert norm, the fp32 router and the logit softcap are negligible approximations.',
    ],
    mtpParamsEstimate: 0,
    visionParamsEstimate: vision + audio,
    excludedParts,
  };
}

/** Family entry for Meta Muse Spark (sliding/full GQA with output gate + latent MoE). */
export const museSparkFamily: FamilyEntry = {
  family: 'muse_spark',
  modelTypes: ['muse_spark', 'muse_spark_text'],
  architectures: ['MuseSparkForConditionalGeneration'],
  build: (text, top, files) => buildMuseSpark(text, top, files.id),
};
