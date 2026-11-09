/**
 * Family `nemotron_h`: NVIDIA Nemotron-H hybrids (transformers `modeling_nemotron_h.py`): Nemotron-3.5 Lightning
 * (`NemotronHForCausalLM`, flat config) and Super-VL (`NemotronH_Omni_Reasoning_V3`, language model under
 * `llm_config`).
 *
 * Every decoder layer holds exactly ONE sublayer, `norm → mixer → residual`, chosen by the layer-role list
 * (`layers_block_type`, or the legacy `hybrid_override_pattern`; see `nemotronHPattern.ts`):
 * - `mamba` → Mamba-2 (SSD) mixer: `linear_attention` variant `'mamba2'` with `n_groups` B/C groups,
 *   `mamba_num_heads` heads of `mamba_head_dim`, state size `ssm_state_size`, `conv_kernel` taps (+ bias when
 *   `use_conv_bias`), z-gated grouped RMSNorm, and `chunk_size` for the chunked prefill form;
 * - `attention` → GQA (`num_attention_heads`, `num_key_value_heads`, `head_dim`) WITHOUT RoPE: the forward applies
 *   no rotary embedding (`rope_theta` is ignored);
 * - `moe` → routed experts + shared expert with a NON-gated activation (`mlp_hidden_act`, relu²); when
 *   `moe_latent_size` is set the routed experts run at that latent width (down/up projections around them);
 * - `mlp` → dense non-gated MLP of `intermediate_size`.
 *
 * MTP (`num_nextn_predict_layers` modules, each `mtp_layers_block_type` = ["attention", "moe"]) is kept as
 * `model.draft` (single-step MTP, costed only with speculative decoding on) and estimated for the param check.
 * The omni wrapper is simulated as its language model only; its vision tower and `mlp1` connector are estimated.
 */

import type { Dtype } from '../../../ir/dtype';
import type { FfnParams, MixerParams } from '../../../ir/builders';
import { decoderModel, repeatLayers, singleBlockLayer } from '../../../ir/builders';
import type { LayerSpec } from '../../../ir/model';
import { blockParams } from '../../../ir/params';
import { modelSpecErrors } from '../../../ir/validateModel';
import { mtpDraft, withDraft } from '../drafter';
import { nemotronBlockTypes } from '../nemotronHPattern';
import type { HybridBlockType } from '../nemotronHPattern';
import { createReader } from '../reader';
import type { ConfigReader } from '../reader';
import type { FamilyEntry, JsonObject } from '../types';

/** MLP / expert activation of a `mlp_hidden_act` string (default relu², the Nemotron-H value). */
function readActivation(r: ConfigReader): 'silu' | 'gelu' | 'relu' | 'relu2' {
  const act = r.optStr('mlp_hidden_act') ?? 'relu2';
  if (act === 'relu2' || act === 'squared_relu') return 'relu2';
  if (act === 'silu' || act === 'swish') return 'silu';
  if (act.startsWith('gelu')) return 'gelu';
  if (act === 'relu') return 'relu';
  r.errors.push(`${r.path ? `${r.path}.` : ''}mlp_hidden_act: unsupported activation '${act}'`);
  return 'relu2';
}

/**
 * Generic ViT parameter estimate of the omni wrapper, for the param check only. Unit: params.
 *
 * Formula: `P_vit = depth·(4·d² + 2·d·(mlp_ratio·d))` (q, k, v, o + 2-layer MLP; biases, norms, patch and position
 * embeddings ignored) and the `mlp1` connector `P_mlp1 = c + c·p + p·h` with `c = vit_hidden_size /
 * downsample_ratio²` (pixel-shuffled features), `p = projector_hidden_size`, `h` = LM hidden size (LayerNorm γ,
 * `c → p`, `p → h`). 0 when the config has no `vision_config` (Lightning).
 */
function visionParams(top: JsonObject, hidden: number): number {
  const v = createReader(top.vision_config, 'vision_config');
  if (top.vision_config === undefined || top.vision_config === null || v.errors.length > 0)
    return 0;
  const d = v.optInt('hidden_size') ?? 0;
  const depth = v.optInt('num_hidden_layers') ?? 0;
  const ratio = v.optNum('mlp_ratio') ?? 4;
  const tower = depth * (4 * d * d + 2 * d * ratio * d);
  const t = createReader(top, '');
  const vit = t.optInt('vit_hidden_size');
  const proj = t.optInt('projector_hidden_size');
  const down = t.optNum('downsample_ratio') ?? 0.5;
  if (vit === undefined || proj === undefined || down <= 0) return tower;
  const c = Math.round(vit / (down * down));
  return tower + c + c * proj + proj * hidden;
}

/** Family entry for Nemotron-H (`nemotron_h`, `nemotron_h_omni`). */
export const nemotronHFamily: FamilyEntry = {
  family: 'nemotron_h',
  modelTypes: ['nemotron_h', 'nemotron_h_omni'],
  architectures: ['NemotronHForCausalLM', 'NemotronH_Omni_Reasoning_V3'],
  build(text, top, files) {
    const wrapped = text !== top;
    const r = createReader(text, wrapped ? 'llm_config' : '');
    const hidden = r.int('hidden_size');
    const vocab = r.int('vocab_size');
    const tie = r.bool('tie_word_embeddings', false);
    const act = readActivation(r);
    const notes: string[] = [];

    const roles = nemotronBlockTypes(text);
    const mtpRoles = nemotronBlockTypes(text, 'mtp');
    const at = r.path ? `${r.path}.` : '';
    if (roles === undefined)
      r.errors.push(`${at}layers_block_type: missing (or hybrid_override_pattern)`);
    else if ('error' in roles) r.errors.push(`${at}${roles.error}`);
    if (mtpRoles !== undefined && 'error' in mtpRoles) r.errors.push(`${at}${mtpRoles.error}`);
    const layerRoles: HybridBlockType[] = Array.isArray(roles) ? roles : [];
    const declared = r.optInt('num_hidden_layers');
    if (declared !== undefined && declared !== layerRoles.length && Array.isArray(roles)) {
      notes.push(
        `num_hidden_layers (${declared}) differs from the layer-type list (${layerRoles.length}); using the list`,
      );
    }

    // Mamba-2 mixer. d_inner = H·P (not expand·hidden), so expand is unused.
    const once = <T>(make: () => T): (() => T) => {
      let value: T | undefined;
      return () => (value ??= make());
    };
    const mamba = once(
      (): MixerParams => ({
        kind: 'linear',
        variant: 'mamba2',
        qkHeads: r.int('n_groups'),
        vHeads: r.int('mamba_num_heads'),
        keyDim: r.int('ssm_state_size'),
        valueDim: r.int('mamba_head_dim'),
        convKernel: r.int('conv_kernel'),
        outputGate: true,
        convBias: r.bool('use_conv_bias', true),
        ...(r.optInt('chunk_size') !== undefined ? { chunkTokens: r.optInt('chunk_size') } : {}),
      }),
    );
    const attention = once((): MixerParams => {
      const heads = r.int('num_attention_heads');
      return {
        kind: 'gqa',
        heads,
        kvHeads: r.int('num_key_value_heads'),
        headDim: r.optInt('head_dim') ?? Math.floor(hidden / Math.max(heads, 1)),
        rope: false,
        qkvBias: r.bool('attention_bias', false),
        outBias: r.bool('attention_bias', false),
      };
    });
    const latent = r.optInt('moe_latent_size');
    const moe = once(
      (): FfnParams => ({
        kind: 'moe',
        experts: r.int('n_routed_experts'),
        topK: r.int('num_experts_per_tok'),
        expertIntermediate: r.int('moe_intermediate_size'),
        sharedExperts: r.optInt('n_shared_experts') ?? 0,
        sharedIntermediate: r.optInt('moe_shared_expert_intermediate_size') ?? 0,
        gated: false,
        activation: act,
        ...(latent !== undefined ? { latentDim: latent } : {}),
      }),
    );
    const mlp = once(
      (): FfnParams => ({
        kind: 'dense',
        intermediate: r.int('intermediate_size'),
        gated: false,
        activation: act,
        bias: r.bool('mlp_bias', false),
      }),
    );
    const layerOf = (role: HybridBlockType, name: string): LayerSpec => {
      const base = { hidden, name };
      switch (role) {
        case 'mamba':
          return singleBlockLayer({ ...base, mixer: mamba() });
        case 'attention':
          return singleBlockLayer({ ...base, mixer: attention() });
        case 'moe':
          return singleBlockLayer({ ...base, ffn: moe() });
        case 'mlp':
          return singleBlockLayer({ ...base, ffn: mlp() });
      }
    };

    if (r.bool('use_bias', false) || r.bool('mamba_proj_bias', false)) {
      notes.push(
        'use_bias / mamba_proj_bias is true: Mamba in/out projection biases are not modelled (< 0.01 % of parameters)',
      );
    }
    const mtpCount = r.optInt('num_nextn_predict_layers') ?? 0;
    if (mtpCount > 0 && mtpRoles === undefined) {
      r.errors.push(
        `${at}mtp_layers_block_type: missing (num_nextn_predict_layers is ${mtpCount})`,
      );
    }
    if (layerRoles.length === 0 && r.errors.length === 0)
      r.errors.push(`${at}layers_block_type: no layers`);

    // Built before the error gate: the lazy mixer/ffn readers record missing fields in `r.errors`.
    const layers = repeatLayers(layerRoles.length, (i) => layerOf(layerRoles[i], `layer.${i}`));
    // One MTP module = the blocks of its role list in order (attention then MoE), each with its own pre-norm, plus
    // eh_proj (2h → h), enorm, hnorm and the final_layernorm. No second embedding or LM head.
    const mtpModuleRoles = mtpRoles !== undefined && !('error' in mtpRoles) ? mtpRoles : [];
    const mtpLayer = (): LayerSpec => ({
      name: 'mtp',
      blocks: mtpModuleRoles.flatMap((role, k) => layerOf(role, `mtp.block.${k}`).blocks),
    });
    // Formula: P_mtp = count · (Σ P_block + 2·h² (eh_proj) + 2·h (enorm, hnorm) + h (final_layernorm))
    const moduleParams =
      mtpModuleRoles.length > 0
        ? mtpLayer().blocks.reduce((s, b) => s + blockParams(b), 0) +
          2 * hidden * hidden +
          3 * hidden
        : 0;
    const mtpParamsEstimate = mtpCount * moduleParams;
    const draft = mtpModuleRoles.length > 0 ? mtpDraft(mtpLayer, mtpCount, true) : undefined;
    if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

    const model = decoderModel({
      name: files.id,
      hidden,
      vocab,
      tieEmbeddings: tie,
      layers,
      finalNorm: 'rmsnorm',
      source: files.id,
    });
    const errors = modelSpecErrors(model);
    if (errors.length > 0) return { kind: 'invalid_config', errors };

    const vision = wrapped ? visionParams(top, hidden) : 0;
    const counts = (role: HybridBlockType) => layerRoles.filter((x) => x === role).length;
    notes.push(
      `hybrid: ${counts('mamba')} Mamba-2, ${counts('attention')} attention (no RoPE), ${counts('moe')} MoE${counts('mlp') > 0 ? `, ${counts('mlp')} dense MLP` : ''} layers (one block per layer)`,
      'Attention has no RoPE (the forward applies none); rope_theta is ignored',
      'Mamba-2 SSD cost model: omitted terms listed in the design doc; e_score_correction_bias and fp32 router logits are not modelled',
    );
    if (wrapped)
      notes.push(
        'Language model only: vision tower, mlp1 connector and sound tower are excluded and estimated from config',
      );
    if (mtpCount > 0) {
      notes.push(
        'MTP kept as the drafter (single-step); its attention is costed with the target attention format (BF16 in the NVFP4 checkpoint)',
      );
    }
    notes.push(
      'The model card serves with --mamba-ssm-cache-dtype float16 (halves the SSM state); override the recurrent-state format to compare',
    );

    const state = r.optStr('mamba_ssm_cache_dtype');
    const recurrentState: Dtype | undefined =
      state === 'float32'
        ? 'fp32'
        : state === 'float16'
          ? 'fp16'
          : state === 'bfloat16'
            ? 'bf16'
            : undefined;
    return {
      kind: 'ok',
      model: withDraft(model, draft),
      notes,
      mtpParamsEstimate,
      visionParamsEstimate: vision,
      excludedParts: [
        ...(mtpCount > 0 ? ['MTP'] : []),
        ...(vision > 0 ? ['vision tower', 'mlp1 connector'] : []),
      ],
      ...(recurrentState ? { quantHints: { recurrentState } } : {}),
    };
  },
};
