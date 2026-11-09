/**
 * Module naming per family, and the probes built from it.
 *
 * A probe is one synthetic module name per (projection, layer) in the checkpoint's naming, tagged with its
 * {@link ModuleClass}. The readers answer per probe; `assemble.ts` takes the majority per class. Probes exist
 * only where the module exists: attention projections in full-attention layers, linear-attention ones in
 * linear layers, dense-MLP ones in dense layers, experts in MoE layers, n-gram ones in n-gram layers (from
 * `layer_types`, `mlp_layer_types`, `moe_layer_freq`, `first_k_dense_replace`, Nemotron-H `layers_block_type`,
 * …). MTP layers (index ≥ `num_hidden_layers`, or names under `mtp`) and vision / audio towers are never probed:
 * they are not in the IR.
 *
 * Several naming variants exist per family (HF `self_attn.q_proj` vs Mistral-native `attention.wq`, MiniMax
 * experts `w1` vs `gate_proj`, `model.` vs `model.language_model.` prefixes). The variant whose names occur
 * most often in the quant metadata wins; ties go to the first.
 */

import type { Dtype } from '../../../ir/dtype';
import type { ModuleClass, Probe, ProbeKind } from './common';
import { numAt, numListAt, objAt, strListAt } from './common';
import { nemotronBlockTypes } from '../nemotronHPattern';
import type { JsonObject } from '../types';
import { layerListOffset } from '../layerLists';

/**
 * Where in the layer stack a leaf exists: every layer, full- / linear-attention layers, dense / MoE layers,
 * n-gram layers, KV-owning full-attention layers (`kv`: layers that compute their own KV latent, the rest
 * share one), indexer-owning full-attention layers (`index`) and MoE layers of a latent MoE (`moe_latent`).
 */
type LeafWhere = 'all' | 'full' | 'linear' | 'dense' | 'moe' | 'ngram' | 'kv' | 'index' | 'moe_latent';

/** One projection of a decoder layer (or a top-level module) in one naming. */
interface LeafSpec {
  /** Name relative to `<prefix>layers.<i>.` (or the absolute name for top-level leaves). */
  leaf: string;
  /** Class of the module. */
  cls: ModuleClass;
  /** Layers where it exists; default `'all'`. */
  where?: LeafWhere;
  /** Torch module type; default `'linear'`. */
  kind?: ProbeKind;
  /** Vote weight inside a layer; default 1. */
  weight?: number;
  /** Kept in high precision by the native checkpoint (see {@link Probe.nativeHighPrecision}). */
  nativeHighPrecision?: boolean;
  /** Dtype in the unconverted checkpoint when not the model dtype (see {@link Probe.nativeDtype}). */
  nativeDtype?: Dtype;
}

/** A group of leaves, or alternatives of which the best-evidenced one is used. */
type LeafGroup = { leaves: LeafSpec[] } | { alternatives: LeafSpec[][] };

/** One naming variant of a family. */
interface NamingVariant {
  /** Prefix of the layer stack, e.g. `'model.language_model.'` (layers are `<prefix>layers.<i>.`). */
  prefix: string;
  /** Top-level modules with absolute names (LM head, embeddings, global residual mixers). */
  top: LeafSpec[];
  /** Per-layer leaves. */
  groups: LeafGroup[];
}

/** Which layers have which attention / MLP kind, as far as the config says. */
export interface LayerLayout {
  /**
   * Decoder layers in the IR (`num_hidden_layers`, or the length of Nemotron-H's `layers_block_type`; MTP layers
   * excluded). Count.
   */
  numLayers: number;
  /**
   * Attention kind per layer: `full` (softmax attention, incl. sparse/MLA), `linear` (GDN, KDA, Mamba-2) or
   * `none` (a Nemotron-H MoE / MLP layer, which has no mixer).
   */
  attention: ('full' | 'linear' | 'none')[];
  /** MLP kind per layer; `none` for a Nemotron-H Mamba / attention layer, which has no MLP. */
  mlp: ('dense' | 'moe' | 'none')[];
  /** True when the MoE runs its routed experts in a latent width (`moe_latent_size` set). */
  moeLatent: boolean;
  /** Layers with n-gram / Engram modules (0-based), empty when none. */
  ngramLayers: number[];
  /** Layers that own their KV projections (`kv_source_layer_ids`, `kv_source_layers`), or null for all. */
  kvLayers: number[] | null;
  /** Layers that own indexer projections (`index_source_layer_ids`, `indexer_types` = full), or null for all. */
  indexLayers: number[] | null;
}

/** The probes of one config, with the family naming that produced them. */
export interface ProbeSet {
  /** Family naming used, e.g. `'qwen4_exp'` or `'generic'`. */
  family: string;
  /** Layer layout read from the config. */
  layout: LayerLayout;
  /** Every probe. */
  probes: Probe[];
}

// ---------------------------------------------------------------------------------------------------------
// Leaf tables
// ---------------------------------------------------------------------------------------------------------

/** `<scope>.<names…>` leaves of one class. */
function leaves(scope: string, names: string[], cls: ModuleClass, extra: Omit<LeafSpec, 'leaf' | 'cls'> = {}): LeafSpec[] {
  return names.map((n) => ({ leaf: scope ? `${scope}.${n}` : n, cls, ...extra }));
}

const GQA = leaves('self_attn', ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'attention', { where: 'full' });
const HF_DENSE = leaves('mlp', ['gate_proj', 'up_proj', 'down_proj'], 'dense_mlp', { where: 'dense' });
const HF_EXPERTS = leaves('mlp.experts.0', ['gate_proj', 'up_proj', 'down_proj'], 'routed_experts', { where: 'moe' });
const HF_SHARED = leaves('mlp.shared_experts', ['gate_proj', 'up_proj', 'down_proj'], 'shared_experts', {
  where: 'moe',
});

/** MLA projections in HF naming (q via LoRA). */
const HF_MLA: LeafSpec[] = [
  ...leaves('self_attn', ['q_a_proj', 'q_b_proj', 'o_proj'], 'attention', { where: 'full' }),
  { leaf: 'self_attn.kv_a_proj_with_mqa', cls: 'attention_kv_down', where: 'kv' },
  // kv_b exists in every full layer, KV-sharing ones included (absorbed into q / o).
  { leaf: 'self_attn.kv_b_proj', cls: 'attention_kv_up', where: 'full' },
];

/** DeepSeek-style indexer in HF naming (`self_attn.indexer.*`). */
const HF_INDEXER: LeafSpec[] = [
  { leaf: 'self_attn.indexer.wq_b', cls: 'indexer', where: 'index' },
  { leaf: 'self_attn.indexer.wk', cls: 'indexer_key', where: 'index' },
  { leaf: 'self_attn.indexer.weights_proj', cls: 'indexer_head_weights', where: 'index', weight: 0.1 },
];

/** Top-level modules of the usual HF layout under `prefix`. */
function hfTop(prefix: string, lmHead = 'lm_head'): LeafSpec[] {
  return [
    { leaf: lmHead, cls: 'lm_head' },
    { leaf: `${prefix}embed_tokens`, cls: 'embeddings', kind: 'embedding' },
  ];
}

const QWEN_LINEAR_ATTN: LeafSpec[] = [
  ...leaves('linear_attn', ['in_proj_qkv', 'in_proj_z', 'out_proj'], 'linear_attention', { where: 'linear' }),
  ...leaves('linear_attn', ['in_proj_a', 'in_proj_b'], 'linear_attention', { where: 'linear', weight: 0.1 }),
];

/** Kimi K3 (`kimi_linear`) KDA projections: q/k/v/o and the full-rank output gate `g_proj`, plus the small ones. */
const KIMI_KDA: LeafSpec[] = [
  ...leaves('self_attn', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'g_proj'], 'linear_attention', {
    where: 'linear',
  }),
  ...leaves(
    'self_attn',
    ['b_proj', 'f_a_proj', 'f_b_proj', 'g_a_proj', 'g_b_proj'],
    'linear_attention',
    {
      where: 'linear',
      weight: 0.1,
    },
  ),
];

/**
 * Kimi K3 MoE in its `block_sparse_moe` naming: `w1/w2/w3` experts, HF-named shared experts, and the latent-MoE
 * `routed_expert_{down,up}_proj`, which the checkpoint stores BF16 (safetensors headers) although K3's
 * compressed-tensors `ignore` list does not name them (K3.1-Pro's does): `nativeHighPrecision`.
 */
const KIMI_MOE: LeafSpec[] = [
  ...leaves('block_sparse_moe.experts.0', ['w1', 'w2', 'w3'], 'routed_experts', { where: 'moe' }),
  ...leaves(
    'block_sparse_moe.shared_experts',
    ['gate_proj', 'up_proj', 'down_proj'],
    'shared_experts',
    { where: 'moe' },
  ),
  ...leaves(
    'block_sparse_moe',
    ['routed_expert_down_proj', 'routed_expert_up_proj'],
    'moe_latent',
    {
      where: 'moe',
      nativeHighPrecision: true,
    },
  ),
];
/** Naming variants per family key (see {@link familyKey}). */
const FAMILY_NAMING: Record<string, NamingVariant[]> = {
  qwen3_5: ['model.language_model.', 'model.'].map((prefix) => ({
    prefix,
    top: hfTop(prefix),
    groups: [{ leaves: [...GQA, ...QWEN_LINEAR_ATTN, ...HF_DENSE] }],
  })),
  qwen4_exp: ['model.language_model.', 'model.'].map((prefix) => ({
    prefix,
    top: [
      ...hfTop(prefix),
      ...leaves(`${prefix}hyper_connection_mixer`, ['input_mix_weight_down', 'input_mix_weight_up'], 'residual_mixer', {
        weight: 0.1,
      }),
    ],
    groups: [
      {
        leaves: [
          ...GQA,
          { leaf: 'self_attn.indexer.index_qk_proj', cls: 'indexer', where: 'full' },
          ...QWEN_LINEAR_ATTN,
          ...leaves('mlp.experts', ['gate_up_proj', 'down_proj'], 'routed_experts', { where: 'moe' }),
          ...leaves('mlp.shared_expert', ['gate_proj', 'up_proj', 'down_proj'], 'shared_experts', { where: 'moe' }),
          ...['attn_hyper_connection', 'mlp_hyper_connection'].flatMap((hc) =>
            leaves(hc, ['input_mix_weight_down', 'input_mix_weight_up', 'block_inject_weight'], 'residual_mixer'),
          ),
          ...leaves('ple', ['key_proj', 'value_proj'], 'ngram', { where: 'ngram' }),
          { leaf: 'ple.ple_embedding.ngram_embedding', cls: 'lookup_tables', where: 'ngram', kind: 'embedding' },
        ],
      },
    ],
  })),
  deepseek_v41: [
    {
      prefix: '',
      top: [
        { leaf: 'head', cls: 'lm_head' },
        { leaf: 'embed', cls: 'embeddings', kind: 'embedding' },
      ],
      groups: [
        {
          leaves: [
            ...leaves('attn', ['wq_a', 'wq_b', 'wkv', 'wo_a', 'wo_b'], 'attention'),
            // Compressor and indexer wk only in kv_source_layer_ids; wq_b / weights_proj in index_source_layer_ids.
            ...leaves('attn.compressor', ['wkv', 'wgate'], 'attention_kv_down', { where: 'kv', nativeHighPrecision: true }),
            { leaf: 'attn.indexer.wq_b', cls: 'indexer', where: 'index' },
            { leaf: 'attn.indexer.wk', cls: 'indexer_key', where: 'kv', nativeHighPrecision: true },
            {
              leaf: 'attn.indexer.weights_proj',
              cls: 'indexer_head_weights',
              where: 'index',
              weight: 0.1,
              nativeHighPrecision: true,
            },
            ...leaves('ffn.experts.0', ['w1', 'w2', 'w3'], 'routed_experts', { where: 'moe' }),
            ...leaves('ffn.shared_experts', ['w1', 'w2', 'w3'], 'shared_experts', { where: 'moe' }),
            // Stored F32 (safetensors headers; model.py builds them under set_dtype(torch.float32)).
            ...leaves('', ['hc_attn_fn', 'hc_ffn_fn'], 'residual_mixer', { kind: 'other', nativeDtype: 'fp32' }),
            { leaf: 'engram.wkv', cls: 'ngram', where: 'ngram' },
            { leaf: 'engram.embed', cls: 'lookup_tables', where: 'ngram', kind: 'embedding' },
          ],
        },
      ],
    },
  ],
  minimax: ['language_model.model.', 'model.'].map((prefix) => ({
    prefix,
    top: hfTop(prefix, prefix === 'model.' ? 'lm_head' : 'language_model.lm_head'),
    groups: [
      {
        leaves: [
          ...GQA,
          { leaf: 'self_attn.index_q_proj', cls: 'indexer', where: 'full' },
          { leaf: 'self_attn.index_k_proj', cls: 'indexer_key', where: 'full' },
          ...leaves('block_sparse_moe.shared_experts', ['gate_proj', 'up_proj', 'down_proj'], 'shared_experts', {
            where: 'moe',
          }),
          ...HF_DENSE,
        ],
      },
      {
        alternatives: [
          leaves('block_sparse_moe.experts.0', ['w1', 'w2', 'w3'], 'routed_experts', { where: 'moe' }),
          leaves('block_sparse_moe.experts.0', ['gate_proj', 'up_proj', 'down_proj'], 'routed_experts', {
            where: 'moe',
          }),
        ],
      },
    ],
  })),
  glm5_next: ['model.language_model.', 'model.'].map((prefix) => ({
    prefix,
    top: hfTop(prefix),
    groups: [
      {
        leaves: [
          ...HF_MLA,
          ...HF_INDEXER,
          { leaf: 'self_attn.indexer.index_kpool_compress_gate', cls: 'indexer_key', where: 'full', weight: 0.1 },
          ...leaves('self_attn', ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'linear_attention', { where: 'linear' }),
          ...leaves('self_attn', ['b_proj', 'f_a_proj', 'f_b_proj', 'g_a_proj', 'g_b_proj'], 'linear_attention', {
            where: 'linear',
            weight: 0.1,
          }),
          ...HF_DENSE,
          ...HF_EXPERTS,
          ...HF_SHARED,
          ...leaves('', ['hc_attn_fn', 'hc_ffn_fn'], 'residual_mixer', { kind: 'other', weight: 0.1 }),
        ],
      },
    ],
  })),
  glm_moe_dsa: [
    {
      prefix: 'model.',
      top: hfTop('model.'),
      groups: [{ leaves: [...HF_MLA, ...HF_INDEXER, ...HF_DENSE, ...HF_EXPERTS, ...HF_SHARED] }],
    },
  ],
  // Kimi K3 / K3.1-Pro: `language_model.model.layers.<i>.*` with `language_model.lm_head` (safetensors headers).
  kimi_linear: ['language_model.model.', 'model.'].map((prefix) => ({
    prefix,
    top: hfTop(prefix, prefix === 'model.' ? 'lm_head' : 'language_model.lm_head'),
    groups: [
      {
        leaves: [
          ...HF_MLA,
          // The MLA sigmoid output gate (`mla_use_output_gate`) shares its name with the KDA gate; split by layer kind.
          { leaf: 'self_attn.g_proj', cls: 'attention', where: 'full' },
          ...HF_INDEXER,
          ...KIMI_KDA,
          ...HF_DENSE,
          ...KIMI_MOE,
        ],
      },
    ],
  })),
  mistral_large4: [
    {
      // Consolidated (params.json) naming, used by the safetensors and by most quant exports.
      prefix: '',
      top: [
        { leaf: 'output', cls: 'lm_head' },
        { leaf: 'tok_embeddings', cls: 'embeddings', kind: 'embedding' },
      ],
      groups: [
        {
          leaves: [
            ...leaves('attention', ['wq', 'wo'], 'attention'),
            { leaf: 'attention.wkv_a_with_mqa', cls: 'attention_kv_down', where: 'kv' },
            { leaf: 'attention.wkv_b', cls: 'attention_kv_up', where: 'full' },
            { leaf: 'attention.indexer.wq_b', cls: 'indexer', where: 'index' },
            { leaf: 'attention.indexer.wk', cls: 'indexer_key', where: 'index' },
            { leaf: 'attention.indexer.weights_proj', cls: 'indexer_head_weights', where: 'index', weight: 0.1 },
            ...leaves('feed_forward', ['w1', 'w2', 'w3'], 'dense_mlp', { where: 'dense' }),
            ...leaves('experts.0', ['w1', 'w2', 'w3'], 'routed_experts', { where: 'moe' }),
            ...leaves('shared_experts', ['w1', 'w2', 'w3'], 'shared_experts', { where: 'moe' }),
          ],
        },
      ],
    },
    ...['model.language_model.', 'model.'].map((prefix) => ({
      // HF transformers naming (config.json `modules_to_not_convert`).
      prefix,
      top: hfTop(prefix),
      groups: [
        {
          leaves: [
            ...leaves('self_attn', ['q_proj', 'o_proj'], 'attention'),
            { leaf: 'self_attn.kv_a_proj_with_mqa', cls: 'attention_kv_down' as const, where: 'kv' as const },
            { leaf: 'self_attn.kv_b_proj', cls: 'attention_kv_up' as const, where: 'full' as const },
            ...HF_INDEXER,
            ...HF_DENSE,
            ...HF_EXPERTS,
            ...HF_SHARED,
          ],
        },
      ],
    })),
  ],
  inkling: [
    {
      prefix: 'model.llm.',
      top: [
        { leaf: 'model.llm.unembed', cls: 'lm_head' },
        { leaf: 'model.llm.embed', cls: 'embeddings', kind: 'embedding' },
      ],
      groups: [
        {
          leaves: [
            ...leaves('attn', ['wq_du', 'wk_dv', 'wv_dv', 'wo_ud', 'wr_du'], 'attention'),
            { leaf: 'attn.rel_logits_proj.proj', cls: 'attention', weight: 0.1 },
            ...leaves('mlp.experts', ['w13_weight', 'w2_weight'], 'routed_experts', { where: 'moe' }),
            ...leaves('mlp.shared_experts', ['shared_w13_weight', 'shared_w2_weight'], 'shared_experts', {
              where: 'moe',
            }),
            ...leaves('mlp', ['w13_dn', 'w2_md'], 'dense_mlp', { where: 'dense' }),
          ],
        },
      ],
    },
  ],
  nemotron_h: [
    { prefix: 'backbone.', lm: 'lm_head' },
    { prefix: 'language_model.backbone.', lm: 'language_model.lm_head' },
  ].map(({ prefix, lm }) => ({
    // Every layer holds one sublayer under `mixer` (transformers / vLLM `NemotronHBlock`); `mixer.gate` is the
    // router, not a module.
    prefix,
    top: [
      { leaf: lm, cls: 'lm_head' as const },
      { leaf: `${prefix}embeddings`, cls: 'embeddings' as const, kind: 'embedding' as const },
    ],
    groups: [
      {
        leaves: [
          ...leaves('mixer', ['in_proj', 'out_proj'], 'linear_attention', { where: 'linear' }),
          ...leaves('mixer', ['q_proj', 'k_proj', 'v_proj', 'o_proj'], 'attention', { where: 'full' }),
          ...leaves('mixer.experts.0', ['up_proj', 'down_proj'], 'routed_experts', { where: 'moe' }),
          ...leaves('mixer.shared_experts', ['up_proj', 'down_proj'], 'shared_experts', { where: 'moe' }),
          ...leaves('mixer', ['fc1_latent_proj', 'fc2_latent_proj'], 'moe_latent', { where: 'moe_latent' }),
          ...leaves('mixer', ['up_proj', 'down_proj'], 'dense_mlp', { where: 'dense' }),
        ],
      },
    ],
  })),
  // transformers `MuseSparkForConditionalGeneration`: fused experts, latent projections around them, and an
  // attention output gate (`self_attn.gate_proj`); every attention layer is softmax ('full' in the layout).
  muse_spark: ['model.language_model.', 'model.'].map((prefix) => ({
    prefix,
    top: hfTop(prefix),
    groups: [
      {
        leaves: [
          ...GQA,
          { leaf: 'self_attn.gate_proj', cls: 'attention', where: 'full' },
          ...leaves('mlp.experts', ['gate_up_proj', 'down_proj'], 'routed_experts', { where: 'moe' }),
          ...leaves('mlp', ['pre_expert_proj', 'post_expert_proj'], 'moe_latent', { where: 'moe' }),
          ...HF_DENSE,
        ],
      },
    ],
  })),
  generic: ['model.', 'model.language_model.', 'language_model.model.', ''].map((prefix) => ({
    prefix,
    top: hfTop(prefix),
    groups: [
      {
        leaves: [...GQA, ...HF_DENSE, ...HF_EXPERTS, ...HF_SHARED],
      },
    ],
  })),
};

/** Naming family of a config, from `model_type` (top level, then `text_config`). */
export function familyKey(top: JsonObject, text: JsonObject): string {
  const types = [top.model_type, text.model_type].filter((t): t is string => typeof t === 'string');
  for (const t of types) {
    if (t.startsWith('qwen3_5')) return 'qwen3_5';
    if (t.startsWith('qwen4_exp')) return 'qwen4_exp';
    if (t.startsWith('deepseek_v4')) return 'deepseek_v41';
    if (t.startsWith('minimax')) return 'minimax';
    if (t.startsWith('glm5_next')) return 'glm5_next';
    if (t.startsWith('glm_moe_dsa')) return 'glm_moe_dsa';
    if (t.startsWith('mistral_large4')) return 'mistral_large4';
    if (t.startsWith('inkling')) return 'inkling';
    if (t.startsWith('nemotron_h')) return 'nemotron_h';
    if (t.startsWith('muse_spark')) return 'muse_spark';
    if (t === 'kimi_linear' || t.startsWith('kimi_k3')) return 'kimi_linear';
  }
  return 'generic';
}

// ---------------------------------------------------------------------------------------------------------
// Layer layout
// ---------------------------------------------------------------------------------------------------------

/** First number found under any of `keys` of `o`. */
function firstNum(o: JsonObject | undefined, keys: string[]): number | undefined {
  for (const k of keys) {
    const v = numAt(o, k);
    if (v !== undefined) return v;
  }
  return undefined;
}

/**
 * The attention and MLP kind of every decoder layer, from the usual config fields. Approximate by design: the
 * families (`families/*.ts`) do the authoritative parsing; this only decides which probes exist.
 *
 * `family` is the naming family ({@link familyKey}). For `'nemotron_h'` the layer roles come from
 * `layers_block_type` / `hybrid_override_pattern` ({@link nemotronBlockTypes}): one sublayer per layer, so a
 * Mamba layer is (`linear`, `none`), an attention layer (`full`, `none`), a MoE layer (`none`, `moe`) and an MLP
 * layer (`none`, `dense`). The list's length is the layer count (Super-VL's `llm_config` has no
 * `num_hidden_layers`; transformers also derives it from the list).
 */
export function layerLayout(text: JsonObject, paramsJson?: unknown, family?: string): LayerLayout {
  const params = typeof paramsJson === 'object' && paramsJson !== null ? (paramsJson as JsonObject) : undefined;
  const moeLatent = (numAt(text, 'moe_latent_size') ?? 0) > 0;
  const blocks = family === 'nemotron_h' ? nemotronBlockTypes(text) : undefined;
  if (Array.isArray(blocks)) {
    return {
      numLayers: blocks.length,
      attention: blocks.map((b) => (b === 'mamba' ? 'linear' : b === 'attention' ? 'full' : 'none')),
      mlp: blocks.map((b) => (b === 'moe' ? 'moe' : b === 'mlp' ? 'dense' : 'none')),
      moeLatent,
      ngramLayers: [],
      kvLayers: null,
      indexLayers: null,
    };
  }
  const numLayers = firstNum(text, ['num_hidden_layers', 'n_layers']) ?? numAt(params, 'n_layers') ?? 0;
  const layerTypes = strListAt(text, 'layer_types');
  const linCfg = objAt(text, 'linear_attn_config');
  const kdaLayers = numListAt(linCfg, 'kda_layers') ?? [];
  // GLM-5.3-Flash lists `kda_layers` 0-based; Kimi K3 1-based (`configuration_kimi_k3.py: is_kda_layer` tests
  // `layer_idx + 1`). Decided from BOTH lists, by the rule the Kimi family uses (the last layer may be MLA).
  const kdaOffset = layerListOffset([kdaLayers, numListAt(linCfg, 'full_attn_layers')], numLayers);
  const attention = Array.from({ length: numLayers }, (_, i) => {
    const t = layerTypes[i];
    if (t !== undefined)
      return /linear|kda|mamba|delta|ssm/i.test(t) ? ('linear' as const) : ('full' as const);
    return kdaLayers.includes(i + kdaOffset) ? ('linear' as const) : ('full' as const);
  });

  const mlpTypes = strListAt(text, 'mlp_layer_types');
  const moeFreq = numListAt(text, 'moe_layer_freq');
  const numExperts =
    firstNum(text, ['num_experts', 'n_routed_experts', 'num_local_experts', 'moe_num_experts']) ??
    numAt(objAt(params, 'moe'), 'num_experts');
  const firstDense = firstNum(text, ['first_k_dense_replace', 'dense_mlp_idx']) ?? 0;
  const mlpOnly = numListAt(text, 'mlp_only_layers') ?? [];
  const mlp = Array.from({ length: numLayers }, (_, i) => {
    const t = mlpTypes[i];
    if (t !== undefined) return t === 'dense' ? ('dense' as const) : ('moe' as const);
    const f = moeFreq?.[i];
    if (f !== undefined) return f === 0 ? ('dense' as const) : ('moe' as const);
    if (!numExperts) return 'dense' as const;
    return i < firstDense || mlpOnly.includes(i) ? ('dense' as const) : ('moe' as const);
  });

  // Qwen `ple_layer_ids` are 1-based (layer 2 → `layers.1.ple`, per the NVFP4 export); DeepSeek
  // `engram_layer_ids` are 0-based (`layers.1.engram`).
  const ple = (numListAt(text, 'ple_layer_ids') ?? []).map((id) => id - 1);
  const engram = numListAt(text, 'engram_layer_ids') ?? [];
  const ngramLayers = [...ple, ...engram].filter((i) => i >= 0 && i < numLayers);

  // KV owners: a per-layer source list (Mistral `kv_source_layers[i]` = the layer whose KV layer i reads)
  // or a set of ids (DeepSeek `kv_source_layer_ids`).
  const kvList = numListAt(text, 'kv_source_layers') ?? numListAt(text, 'kv_source_layer_ids');
  const kvLayers = kvList ? (kvList.length === numLayers ? kvList.filter((src, i) => src === i) : kvList) : null;
  const indexTypes = strListAt(text, 'indexer_types');
  const indexIds = numListAt(text, 'index_source_layer_ids');
  const indexLayers = indexIds
    ? indexIds
    : indexTypes.length > 0
      ? indexTypes.flatMap((t, i) => (t === 'full' ? [i] : []))
      : null;
  return { numLayers, attention, mlp, moeLatent, ngramLayers, kvLayers, indexLayers };
}

// ---------------------------------------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------------------------------------

/** Regex source of a leaf with expert indices generalized (`experts.0.` → `experts.<n>.`). */
function leafEvidence(leaf: string): RegExp {
  const src = leaf.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/experts\\\.0\\\./g, 'experts\\.\\d+\\.');
  return new RegExp(`(^|[^A-Za-z0-9_])${src}([^A-Za-z0-9_]|$)`);
}

/**
 * Evidence for a naming in the metadata text: 1000 × the number of `<prefix>layers.<n>` names (the prefix not
 * preceded by another name part, so `model.` does not count `language_model.layers.`), plus one per leaf of
 * `specs` that occurs.
 */
function evidence(meta: string, prefix: string | null, specs: LeafSpec[]): number {
  let score = 0;
  if (prefix !== null) {
    const esc = prefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    score += 1000 * (meta.match(new RegExp(`(^|[^A-Za-z0-9_.])${esc}layers\\.\\d`, 'g'))?.length ?? 0);
  }
  for (const s of specs) if (leafEvidence(s.leaf).test(meta)) score++;
  return score;
}

/** True when a leaf exists in layer `i`. */
function existsIn(where: LeafWhere, i: number, layout: LayerLayout): boolean {
  switch (where) {
    case 'all':
      return true;
    case 'full':
      return layout.attention[i] === 'full';
    case 'linear':
      return layout.attention[i] === 'linear';
    case 'dense':
      return layout.mlp[i] === 'dense';
    case 'moe':
      return layout.mlp[i] === 'moe';
    case 'ngram':
      return layout.ngramLayers.includes(i);
    case 'kv':
      return layout.attention[i] === 'full' && (layout.kvLayers === null || layout.kvLayers.includes(i));
    case 'index':
      return layout.attention[i] === 'full' && (layout.indexLayers === null || layout.indexLayers.includes(i));
    case 'moe_latent':
      return layout.mlp[i] === 'moe' && layout.moeLatent;
  }
}

/** A probe from a leaf spec. */
function probeOf(name: string, spec: LeafSpec, layer: number | null): Probe {
  const p: Probe = {
    name,
    cls: spec.cls,
    layer,
    kind: spec.kind ?? 'linear',
    weight: spec.weight ?? 1,
    leaf: spec.leaf,
  };
  if (spec.nativeHighPrecision) p.nativeHighPrecision = true;
  if (spec.nativeDtype) p.nativeDtype = spec.nativeDtype;
  return p;
}

/**
 * Build the probes of a config. `meta` is the quant metadata as text (JSON), used only to pick the naming
 * variant by evidence.
 */
export function buildProbes(top: JsonObject, text: JsonObject, meta: string, paramsJson?: unknown): ProbeSet {
  const family = familyKey(top, text);
  const layout = layerLayout(text, paramsJson, family);
  const variants = FAMILY_NAMING[family] ?? FAMILY_NAMING.generic;

  // Pick the variant: score its prefix, top names and fixed leaves.
  let best = variants[0];
  let bestScore = -1;
  for (const v of variants) {
    const fixed = v.groups.flatMap((g) => ('leaves' in g ? g.leaves : []));
    const score = evidence(meta, v.prefix, [...v.top, ...fixed]);
    if (score > bestScore) {
      best = v;
      bestScore = score;
    }
  }

  const specs: LeafSpec[] = [];
  for (const g of best.groups) {
    if ('leaves' in g) {
      specs.push(...g.leaves);
      continue;
    }
    let pick = g.alternatives[0];
    let pickScore = -1;
    for (const alt of g.alternatives) {
      const s = evidence(meta, null, alt);
      if (s > pickScore) {
        pick = alt;
        pickScore = s;
      }
    }
    specs.push(...pick);
  }

  const probes: Probe[] = best.top.map((t) => probeOf(t.leaf, t, null));
  for (let i = 0; i < layout.numLayers; i++) {
    for (const s of specs) {
      if (existsIn(s.where ?? 'all', i, layout)) probes.push(probeOf(`${best.prefix}layers.${i}.${s.leaf}`, s, i));
    }
  }
  return { family, layout, probes };
}
