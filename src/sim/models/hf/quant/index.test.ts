import { describe, expect, it } from 'vitest';
import type { Dtype } from '../../../ir/dtype';
import type { QuantConfig, WeightModule } from '../../../ir/model';
import { WEIGHT_MODULE_FALLBACK } from '../../../ir/model';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { isJsonObject } from '../reader';
import type { HfRepoFiles, QuantDefaults } from '../types';
import { deriveQuantDefaults, type QuantInputs } from './index';

/** Expected format: [dtype, extra bits/elem (default 0), GEMM-input dtype (default: activations)]. */
type F = [Dtype, number?, Dtype?];

/** Scale overheads from the plan table (bits/element). */
const FP8_128_FP32 = 32 / (128 * 128); // 0.001953
const FP8_128_UE8M0 = 8 / (128 * 128);
const FP8_32_UE8M0 = 8 / (32 * 32); // 0.0078125
const INT4_SYM_G128 = 16 / 128; // 0.125
const INT4_ASYM_G32 = (16 + 8) / 32; // 0.75
const GPTQ_G128 = (16 + 4) / 128;
const GPTQ_G32 = (16 + 4) / 32;
const BNB_NF4_DQ = 8 / 64 + 32 / (64 * 256); // ≈0.127
const MLX_G64 = 32 / 64; // 0.5

/** Inputs for a fixture repo, as mapHfModel builds them. */
function inputsOf(files: HfRepoFiles): QuantInputs {
  const top = files.config as Record<string, unknown>;
  const text = isJsonObject(top.text_config) ? top.text_config : top;
  return {
    top,
    text,
    hfQuantConfig: files.hfQuantConfig,
    paramsJson: files.paramsJson,
    quantizationConfigJson: files.quantizationConfigJson,
    mlx: top.quantization,
  };
}

/** Effective format of a module class, resolved as the lowering does (`lower/common.ts` moduleQuant). */
function effective(q: QuantConfig, m: WeightModule): { dtype: Dtype; extra: number; input: Dtype } {
  const parent = WEIGHT_MODULE_FALLBACK[m];
  const o = q.modules?.[m] ?? (parent ? q.modules?.[parent] : undefined);
  const w = o?.weights ?? (m === 'residual_mixer' ? { dtype: q.highPrecision ?? 'bf16' } : q.weights);
  return { dtype: w.dtype, extra: w.extraBitsPerElem ?? 0, input: o?.gemmInput ?? q.activations };
}

function expectFormat(label: string, got: { dtype: Dtype; extra: number; input: Dtype }, want: F, act: Dtype) {
  expect(got.dtype, `${label} dtype`).toBe(want[0]);
  expect(got.extra, `${label} extraBitsPerElem`).toBeCloseTo(want[1] ?? 0, 6);
  expect(got.input, `${label} gemmInput`).toBe(want[2] ?? act);
}

interface Case {
  repo: string;
  /** Fixture dir: `quant` (default) or the base fixtures. */
  dir?: '' | 'quant';
  /** Repo whose config.json to use (Mistral NVFP4 has none). */
  configFrom?: string;
  scheme: RegExp;
  source: QuantDefaults['source'];
  /** Activations dtype (default bf16). */
  activations?: Dtype;
  /** Effective per-module formats. */
  modules: Partial<Record<WeightModule, F>>;
  /** LM head [dtype, extra, gemmInput]; default: embeddings (bf16). */
  lmHead?: F;
  embeddings?: Dtype;
  lookupTables?: F;
  kvCache?: Dtype;
  indexerKeyCache?: Dtype;
  explicit?: (keyof QuantConfig)[];
  notes?: RegExp[];
  /** Notes that must not appear. */
  noNotes?: RegExp[];
}

const BF16_ALL: Partial<Record<WeightModule, F>> = { attention: ['bf16'], dense_mlp: ['bf16'] };

const CASES: Case[] = [
  // --- no quantization ---------------------------------------------------------------------------------------
  { repo: 'Qwen/Qwen3.8-27B', dir: '', scheme: /^bf16 \(unquantized\)$/, source: 'none', modules: { ...BF16_ALL, linear_attention: ['bf16'] }, explicit: [] },
  { repo: 'zai-org/GLM-5.3-Flash-BF16', dir: '', scheme: /unquantized/, source: 'none', modules: { routed_experts: ['bf16'], attention_kv_up: ['bf16'] } },

  // --- native fp8 -----------------------------------------------------------------------------------------------
  {
    repo: 'Qwen/Qwen3.8-27B-FP8',
    scheme: /^fp8: e4m3 fp8, block 128×128 fp32 scales, dynamic fp8 activations$/,
    source: 'quantization_config',
    modules: {
      attention: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      linear_attention: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      dense_mlp: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
    },
    notes: [/linear_attention: linear_attn\.in_proj_a, linear_attn\.in_proj_b bf16/, /match no known module .*in_proj_ba/],
  },
  {
    repo: 'Qwen/Qwen3.8-Flash-Next-FP8',
    scheme: /^fp8: e4m3 fp8, block 128×128/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      attention: ['bf16'],
      linear_attention: ['bf16'],
      indexer: ['bf16'],
      shared_experts: ['bf16'],
      residual_mixer: ['bf16'],
      ngram: ['bf16'],
    },
    lookupTables: ['fp8_e4m3', FP8_128_FP32],
    notes: [/modules_to_convert/],
  },
  {
    repo: 'deepseek-ai/DeepSeek-V4.1-Flash',
    dir: '',
    scheme: /^fp8: e4m3 fp8, block 32×32 ue8m0 scales, dynamic fp8 activations, routed experts fp4/,
    source: 'quantization_config',
    modules: {
      attention: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      shared_experts: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      indexer: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      ngram: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      routed_experts: ['mxfp4', 0, 'fp8_e4m3'],
      attention_kv_down: ['bf16'],
      indexer_key: ['bf16'],
      indexer_head_weights: ['bf16'],
      residual_mixer: ['fp32'],
    },
    lookupTables: ['mxfp8'],
    notes: [/mxfp4, fp8 GEMM input/, /Engram table/],
  },
  {
    // expert_dtype sits at the top level of config.json, next to quantization_config (V4.1 puts it inside).
    repo: 'deepseek-ai/DeepSeek-V4-Flash-0731',
    dir: '',
    scheme: /^fp8: e4m3 fp8, block 128×128 ue8m0 scales, dynamic fp8 activations, routed experts fp4/,
    source: 'quantization_config',
    modules: {
      attention: ['fp8_e4m3', FP8_128_UE8M0, 'fp8_e4m3'],
      shared_experts: ['fp8_e4m3', FP8_128_UE8M0, 'fp8_e4m3'],
      indexer: ['fp8_e4m3', FP8_128_UE8M0, 'fp8_e4m3'],
      routed_experts: ['mxfp4', 0, 'fp8_e4m3'],
      attention_kv_down: ['bf16'],
      indexer_key: ['bf16'],
      indexer_head_weights: ['bf16'],
      residual_mixer: ['fp32'],
    },
    notes: [/mxfp4, fp8 GEMM input/],
    noNotes: [/Engram table/],
  },
  {
    repo: 'zai-org/GLM-5.3-Flash',
    dir: '',
    scheme: /^fp8: e4m3 fp8, block 128×128 fp32/,
    source: 'quantization_config',
    modules: {
      attention: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      attention_kv_down: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      attention_kv_up: ['bf16'],
      linear_attention: ['bf16'],
      indexer: ['bf16'],
      dense_mlp: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      shared_experts: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      routed_experts: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
    },
  },
  {
    repo: 'mistralai/Mistral-Large-4-1T-A52B',
    dir: '',
    scheme: /block 128×128 ue8m0 scales/,
    source: 'quantization_config',
    modules: {
      attention: ['fp8_e4m3', FP8_128_UE8M0, 'fp8_e4m3'],
      attention_kv_down: ['bf16'],
      attention_kv_up: ['bf16'],
      indexer: ['fp8_e4m3', FP8_128_UE8M0, 'fp8_e4m3'],
      indexer_key: ['bf16'],
      indexer_head_weights: ['bf16'],
      routed_experts: ['fp8_e4m3', FP8_128_UE8M0, 'fp8_e4m3'],
    },
    kvCache: 'fp8_e4m3',
    indexerKeyCache: 'fp8_e4m3',
    explicit: ['kvCache', 'indexerKeyCache', 'modules'],
    notes: [/ue8m0 \(from params\.json\)/, /dsa\.fp8_kv_cache/],
    // The synthetic quantize_kv_a entries are not reported as unknown ignore entries.
    noNotes: [/no known module/],
  },
  {
    repo: 'MiniMaxAI/MiniMax-M3-MXFP8',
    scheme: /^mxfp8: mxfp8 \(block 1×32, e8m0 scales\)/,
    source: 'quantization_config',
    modules: {
      attention: ['mxfp8', 0, 'mxfp8'],
      dense_mlp: ['mxfp8', 0, 'mxfp8'],
      shared_experts: ['mxfp8', 0, 'mxfp8'],
      routed_experts: ['mxfp8', 0, 'mxfp8'],
    },
  },
  {
    repo: 'MiniMaxAI/MiniMax-M3.1-Flash',
    dir: '',
    scheme: /routed experts NVFP4 W4A4/,
    source: 'quantization_config',
    modules: { attention: ['mxfp8', 0, 'mxfp8'], routed_experts: ['nvfp4', 0, 'nvfp4'] },
  },

  // --- ModelOpt -------------------------------------------------------------------------------------------------
  {
    repo: 'nvidia/Qwen3.8-27B-NVFP4',
    scheme: /^modelopt MIXED_PRECISION \(FP8, NVFP4\)$/,
    source: 'quantization_config',
    modules: {
      dense_mlp: ['nvfp4', 0, 'nvfp4'],
      attention: ['fp8_e4m3', 0, 'fp8_e4m3'],
      linear_attention: ['fp8_e4m3', 0, 'fp8_e4m3'],
    },
    lmHead: ['nvfp4', 0, 'nvfp4'],
  },
  {
    repo: 'nvidia/Qwen3.8-Flash-Next-NVFP4',
    scheme: /^modelopt MIXED_PRECISION/,
    source: 'quantization_config',
    modules: { routed_experts: ['nvfp4', 0, 'nvfp4'], attention: ['bf16'], shared_experts: ['bf16'], linear_attention: ['bf16'] },
    lookupTables: ['fp8_e4m3'],
  },
  {
    repo: 'nvidia/DeepSeek-V4.1-Flash-NVFP4',
    scheme: /^modelopt MIXED_PRECISION \(NVFP4\)$/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['nvfp4', 0, 'nvfp4'],
      // Ignored by the export: stays native fp8 32×32 ue8m0, not bf16.
      attention: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      shared_experts: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      attention_kv_down: ['bf16'],
      residual_mixer: ['fp32'],
    },
    lookupTables: ['mxfp4'],
    notes: [/base format of the modules the ModelOpt export leaves alone/],
  },
  {
    repo: 'nvidia/GLM-5.3-Flash-FP4-Local',
    scheme: /^modelopt MIXED_PRECISION \(MXFP8, NVFP4, FP8_PB_WO, W4A16_NVFP4\)$/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['nvfp4', 0, 'nvfp4'],
      linear_attention: ['mxfp8', 0, 'mxfp8'],
      attention: ['fp8_e4m3', FP8_128_FP32],
      shared_experts: ['fp8_e4m3', FP8_128_FP32],
      dense_mlp: ['mxfp8', 0, 'mxfp8'],
    },
    lmHead: ['mxfp8', 0, 'mxfp8'],
    notes: [/linear_attention: .*self_attn\.b_proj.* bf16/],
  },
  {
    repo: 'nvidia/GLM-5.3-Flash-NVFP4',
    scheme: /^modelopt NVFP4, fp8_e4m3 KV cache$/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['nvfp4', 0, 'nvfp4'],
      dense_mlp: ['nvfp4', 0, 'nvfp4'],
      attention: ['bf16'],
      linear_attention: ['bf16'],
      shared_experts: ['bf16'],
    },
    kvCache: 'fp8_e4m3',
    explicit: ['kvCache', 'modules'],
    // `model.language_model.layers.45*` / `model.layers.45*` name the MTP layer (index = num_hidden_layers).
    noNotes: [/no known module/],
  },
  {
    repo: 'thinkingmachines/Inkling-Small-NVFP4',
    scheme: /^modelopt NVFP4$/,
    source: 'hf_quant_config.json',
    modules: { routed_experts: ['nvfp4', 0, 'nvfp4'], attention: ['bf16'], shared_experts: ['bf16'], dense_mlp: ['bf16'] },
    kvCache: 'bf16',
    notes: [/routed_experts: nvfp4, nvfp4 GEMM input in 39 of 40 layers; layer 2 is bf16/],
  },
  {
    repo: 'mistralai/Mistral-Large-4-1T-A52B-NVFP4',
    configFrom: 'mistralai/Mistral-Large-4-1T-A52B',
    scheme: /^modelopt MIXED_PRECISION \(FP8_PB_WO, NVFP4\)$/,
    source: 'params.json',
    modules: {
      routed_experts: ['nvfp4', 0, 'nvfp4'],
      attention: ['fp8_e4m3', FP8_128_FP32],
      dense_mlp: ['fp8_e4m3', FP8_128_FP32],
      shared_experts: ['fp8_e4m3', FP8_128_FP32],
      indexer: ['fp8_e4m3', FP8_128_FP32],
      attention_kv_down: ['bf16'],
      attention_kv_up: ['bf16'],
      indexer_key: ['bf16'],
      indexer_head_weights: ['bf16'],
    },
    kvCache: 'fp8_e4m3',
    indexerKeyCache: 'fp8_e4m3',
  },

  // --- compressed-tensors ---------------------------------------------------------------------------------------
  {
    repo: 'RedHatAI/Qwen3.8-27B-INT4',
    scheme: /^compressed-tensors pack-quantized \(group_0: int4\), fp8_e4m3 KV cache$/,
    source: 'quantization_config',
    modules: { dense_mlp: ['int4', INT4_SYM_G128], attention: ['int4', INT4_SYM_G128], linear_attention: ['int4', INT4_SYM_G128] },
    kvCache: 'fp8_e4m3',
    explicit: ['kvCache'],
  },
  {
    repo: 'cyankiwi/Qwen3.8-27B-AWQ-INT4',
    scheme: /pack-quantized/,
    source: 'quantization_config',
    modules: { dense_mlp: ['int4', INT4_ASYM_G32], attention: ['int4', INT4_ASYM_G32] },
  },
  {
    repo: 'cyankiwi/MiniMax-M3-AWQ-INT4',
    scheme: /pack-quantized/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['int4', INT4_ASYM_G32],
      attention: ['int4', INT4_ASYM_G32],
      shared_experts: ['bf16'],
      indexer: ['bf16'],
      dense_mlp: ['bf16'],
    },
    notes: [/attention: .* in 57 of 60 layers; layers 0–2 are bf16/, /1 entry matches no known module/],
  },
  {
    repo: 'RedHatAI/Qwen3.8-27B-MXFP4',
    scheme: /mxfp4-pack-quantized/,
    source: 'quantization_config',
    modules: { dense_mlp: ['mxfp4', 0, 'mxfp4'], attention: ['mxfp4', 0, 'mxfp4'] },
    kvCache: 'fp8_e4m3',
  },
  {
    repo: 'unsloth/Qwen3.8-27B-NVFP4',
    scheme: /^compressed-tensors mixed-precision/,
    source: 'quantization_config',
    modules: {
      dense_mlp: ['nvfp4', 0, 'nvfp4'],
      attention: ['fp8_e4m3', 0, 'fp8_e4m3'],
      linear_attention: ['fp8_e4m3', 0, 'fp8_e4m3'],
    },
    lmHead: ['fp8_e4m3', 0, 'fp8_e4m3'],
    kvCache: 'fp8_e4m3',
    notes: [/dense_mlp: nvfp4, nvfp4 GEMM input in 56 of 64 layers; layers 56–63 are fp8_e4m3/, /per-channel/],
  },
  {
    repo: 'EmbeddedLLM/MiniMax-M3-FP8-dynamic',
    scheme: /float-quantized/,
    source: 'quantization_config',
    modules: { attention: ['fp8_e4m3', 0, 'fp8_e4m3'], routed_experts: ['fp8_e4m3', 0, 'fp8_e4m3'] },
    notes: [/per-channel weight scales: overhead ≈0/],
  },
  {
    repo: 'intel-ai/DeepSeek-V4.1-Flash-MXFP4-Mixed-CT-AutoRound',
    scheme: /^compressed-tensors mixed-precision .*by auto-round$/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['mxfp4', 0, 'mxfp4'],
      attention: ['mxfp8', 0, 'mxfp8'],
      shared_experts: ['mxfp8', 0, 'mxfp8'],
      attention_kv_down: ['bf16'],
      indexer_key: ['bf16'],
    },
  },

  // --- Quark ----------------------------------------------------------------------------------------------------
  {
    repo: 'amd/Qwen3.8-27B-Quark-AWQ-MXFP4',
    scheme: /^quark: mxfp4, mxfp4 GEMM input$/,
    source: 'quantization_config',
    modules: { dense_mlp: ['mxfp4', 0, 'mxfp4'], attention: ['mxfp4', 0, 'mxfp4'] },
  },
  {
    repo: 'amd/DeepSeek-V4.1-Flash-Quark-MXFP4',
    scheme: /^quark: mxfp4.*per-layer fp8_e4m3/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['mxfp4', 0, 'mxfp4'],
      shared_experts: ['mxfp4', 0, 'mxfp4'],
      attention: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      indexer: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      ngram: ['fp8_e4m3', FP8_32_UE8M0, 'fp8_e4m3'],
      attention_kv_down: ['bf16'],
      indexer_key: ['bf16'],
      indexer_head_weights: ['bf16'],
    },
  },
  {
    repo: 'amd/Mistral-Large-4-1T-A52B-Quark-MXFP4',
    scheme: /^quark: mxfp4/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['mxfp4', 0, 'mxfp4'],
      attention: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      shared_experts: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      dense_mlp: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      indexer: ['fp8_e4m3', FP8_128_FP32, 'fp8_e4m3'],
      attention_kv_down: ['bf16'],
      attention_kv_up: ['bf16'],
      indexer_key: ['bf16'],
      indexer_head_weights: ['bf16'],
    },
  },

  // --- GPTQ / AutoRound -----------------------------------------------------------------------------------------
  {
    repo: 'sokada4/Qwen3.8-27B-GPTQ-Int4',
    scheme: /^gptq int4 g128 sym$/,
    source: 'quantization_config',
    modules: { dense_mlp: ['int4', GPTQ_G128], attention: ['int4', GPTQ_G128], linear_attention: ['int4', GPTQ_G128] },
    notes: [/packed zero point per group of 128/],
  },
  {
    repo: 'btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit',
    scheme: /^gptq int4 g32 sym$/,
    source: 'quantization_config',
    modules: { routed_experts: ['int4', GPTQ_G32] },
  },
  {
    repo: 'Intel/Qwen3.8-Flash-Next-W4A16-AutoRound',
    scheme: /^auto-round int4/,
    source: 'quantization_config',
    modules: {
      routed_experts: ['int4', GPTQ_G128],
      attention: ['bf16'],
      linear_attention: ['bf16'],
      indexer: ['bf16'],
      shared_experts: ['bf16'],
      residual_mixer: ['bf16'],
      ngram: ['bf16'],
    },
  },
  {
    repo: 'INCModel3/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound',
    scheme: /^auto-round int4 .*g32/,
    source: 'quantization_config',
    modules: { routed_experts: ['int4', GPTQ_G32] },
  },

  // --- bitsandbytes / MLX ---------------------------------------------------------------------------------------
  {
    repo: 'unsloth/Qwen3.8-27B-unsloth-bnb-4bit',
    scheme: /^bitsandbytes nf4 \+ double quant \(blocksize 64\)$/,
    source: 'quantization_config',
    activations: 'fp16',
    modules: { dense_mlp: ['int4', BNB_NF4_DQ], attention: ['int4', BNB_NF4_DQ] },
    kvCache: 'fp16',
    notes: [/nf4: 4-bit codes stored as int4/],
  },
  {
    repo: 'mlx-community/Inkling-Small-mlx-4bit',
    scheme: /^mlx int4 .*experts_only/,
    source: 'quantization (mlx)',
    modules: { routed_experts: ['int4', MLX_G64], attention: ['bf16'], shared_experts: ['bf16'], dense_mlp: ['bf16'] },
  },
  {
    repo: 'mlx-community/Mistral-Large-4-1T-A52B-Text-4bit',
    scheme: /^mlx int4 \+0\.5000 bit\/elem \(affine 4-bit g64\)$/,
    source: 'quantization (mlx)',
    modules: { routed_experts: ['int4', MLX_G64], attention: ['int4', MLX_G64] },
    embeddings: 'int4',
    lmHead: ['int4', MLX_G64],
  },
  {
    repo: 'mlx-community/Inkling-Small-mxfp4',
    scheme: /^mlx mxfp4/,
    source: 'quantization (mlx)',
    modules: { routed_experts: ['mxfp4'], attention: ['mxfp4'] },
    embeddings: 'mxfp4',
  },
];

describe('deriveQuantDefaults: per-repo defaults from the fixtures', () => {
  for (const c of CASES) {
    it(c.repo, () => {
      const files = fixtureRepoFiles(c.repo, { dir: c.dir ?? 'quant', ...(c.configFrom ? { configFrom: c.configFrom } : {}) });
      const d = deriveQuantDefaults(inputsOf(files));
      const act = c.activations ?? 'bf16';
      expect(d.scheme).toMatch(c.scheme);
      expect(d.source).toBe(c.source);
      expect(d.quant.activations).toBe(act);
      for (const [m, want] of Object.entries(c.modules) as [WeightModule, F][]) {
        expectFormat(m, effective(d.quant, m), want, act);
      }
      const embeddings = d.quant.embeddings ?? { dtype: d.quant.highPrecision ?? 'bf16' };
      expect(embeddings.dtype).toBe(c.embeddings ?? act);
      const head = d.quant.lmHead ?? embeddings;
      expectFormat(
        'lmHead',
        { dtype: head.dtype, extra: head.extraBitsPerElem ?? 0, input: d.quant.lmHeadGemmInput ?? act },
        c.lmHead ?? [embeddings.dtype, embeddings.extraBitsPerElem ?? 0],
        act,
      );
      if (c.lookupTables) {
        const t = d.quant.lookupTables;
        expect(t?.dtype).toBe(c.lookupTables[0]);
        expect(t?.extraBitsPerElem ?? 0).toBeCloseTo(c.lookupTables[1] ?? 0, 6);
      } else {
        expect(d.quant.lookupTables).toBeUndefined();
      }
      expect(d.quant.kvCache.dtype).toBe(c.kvCache ?? act);
      expect(d.quant.indexerKeyCache?.dtype).toBe(c.indexerKeyCache);
      if (c.explicit) expect(d.explicit).toEqual(c.explicit);
      for (const n of c.notes ?? []) expect(d.notes.some((x) => n.test(x)), `note ${n} in ${JSON.stringify(d.notes)}`).toBe(true);
      for (const n of c.noNotes ?? []) expect(d.notes.filter((x) => n.test(x)), `no note ${n}`).toEqual([]);
    });
  }
});

describe('deriveQuantDefaults: edge cases', () => {
  const qwenText = { model_type: 'qwen3_5_text', num_hidden_layers: 2, layer_types: ['linear_attention', 'full_attention'], dtype: 'bfloat16' };
  const derive = (qc: unknown, extra: Partial<QuantInputs> = {}) =>
    deriveQuantDefaults({ top: { model_type: 'qwen3_5', quantization_config: qc }, text: qwenText, ...extra });

  it('unknown quant_method: torch_dtype defaults plus a note, never throws', () => {
    const d = derive({ quant_method: 'exl9', bits: 3 });
    expect(d.quant.weights).toEqual({ dtype: 'bf16' });
    expect(d.quant.modules).toBeUndefined();
    expect(d.source).toBe('none');
    expect(d.notes).toContain("unrecognised quantization 'exl9' (quantization_config); using torch_dtype");
  });

  it('empty quantization_config is ignored with a note', () => {
    const d = derive({});
    expect(d.scheme).toBe('bf16 (unquantized)');
    expect(d.notes).toContain('quantization_config is empty; ignored');
  });

  it('malformed fields and unknown ignore patterns become notes', () => {
    const d = derive({
      quant_method: 'compressed-tensors',
      config_groups: { g: { targets: ['Linear', 're:(['], weights: { num_bits: 4, type: 'int', strategy: 'group', group_size: 64, symmetric: true } } },
      ignore: ['model.layers.0.made_up_proj', 're:[unclosed'],
    });
    expect(effective(d.quant, 'dense_mlp')).toEqual({ dtype: 'int4', extra: 16 / 64, input: 'bf16' });
    expect(d.notes.some((n) => /ignore: 2 entries match no known module/.test(n))).toBe(true);
  });

  it('awq: fp16 scale + packed zero point per group, decoder layers only', () => {
    const d = derive({ quant_method: 'awq', bits: 4, group_size: 128, zero_point: true, modules_to_not_convert: ['linear_attn'] });
    expect(effective(d.quant, 'dense_mlp').extra).toBeCloseTo(20 / 128, 6);
    expect(effective(d.quant, 'linear_attention').dtype).toBe('bf16');
    expect(d.quant.lmHead).toBeUndefined();
  });

  it('bitsandbytes 8-bit: int8 weight-only', () => {
    const d = derive({ quant_method: 'bitsandbytes', load_in_8bit: true });
    expect(effective(d.quant, 'attention')).toEqual({ dtype: 'int8', extra: 0, input: 'bf16' });
  });

  it('nf4 without double quant: one fp32 absmax per 64', () => {
    const d = derive({ quant_method: 'bitsandbytes', load_in_4bit: true, bnb_4bit_quant_type: 'nf4' });
    expect(effective(d.quant, 'dense_mlp').extra).toBeCloseTo(32 / 64, 6);
  });

  it('quark int4 asym per_group: scale + 4-bit zero point; fp8 per_channel ≈0 with a note', () => {
    const d = derive({
      quant_method: 'quark',
      global_quant_config: { weight: { dtype: 'uint4', qscheme: 'per_group', group_size: 128, symmetric: false } },
      layer_quant_config: { '*self_attn*': { weight: { dtype: 'fp8_e4m3', qscheme: 'per_channel' }, input_tensors: { dtype: 'fp8_e4m3' } } },
    });
    expect(effective(d.quant, 'dense_mlp').extra).toBeCloseTo(20 / 128, 6);
    expect(effective(d.quant, 'attention')).toEqual({ dtype: 'fp8_e4m3', extra: 0, input: 'fp8_e4m3' });
    expect(d.notes.some((n) => /per-channel/.test(n))).toBe(true);
  });

  it('explicit KV scheme is reported so family hints cannot override it', () => {
    const d = derive({ quant_method: 'modelopt', quant_algo: 'FP8', kv_cache_quant_algo: 'FP8' });
    expect(d.quant.kvCache).toEqual({ dtype: 'fp8_e4m3' });
    expect(d.explicit).toContain('kvCache');
    expect(derive({ quant_method: 'modelopt', quant_algo: 'FP8' }).explicit).not.toContain('kvCache');
  });

  it('params.json quantization alone (no quantization_config): native Mistral fp8', () => {
    const d = deriveQuantDefaults({
      top: { model_type: 'mistral_large4' },
      text: { model_type: 'mistral_large4_text', num_hidden_layers: 2 },
      paramsJson: {
        quantization: { qformat_weight: 'fp8_e4m3', block_size_out: 128, block_size_in: 128, use_ue8m0_for_qscales: true, qdynamic_act: true, quantize_kv_a: false },
      },
    });
    expect(d.source).toBe('params.json');
    expect(effective(d.quant, 'attention')).toEqual({ dtype: 'fp8_e4m3', extra: FP8_128_UE8M0, input: 'fp8_e4m3' });
    expect(effective(d.quant, 'attention_kv_down').dtype).toBe('bf16');
  });
});
