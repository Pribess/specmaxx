/**
 * End-to-end check of the Hugging Face loader on the user's model shortlist: fixture files → `mapHfModel` →
 * `simulate` (prefill and decode) on the RTX 5090 preset, alone (tp = 1) and as 8 devices in one node behind the
 * synthetic 100 GB/s intra-node link of the test fixtures (tp = 8, ep = 8 for MoE models; not a preset).
 *
 * The rates are not compared with vendor numbers: most of these models do not fit 32 GB devices (the memory
 * report says so) and the simulator gives theoretical maxima. What is asserted: the whole pipeline runs without
 * throwing, every rate is finite and positive, each model gets its family, param-check outcome, derived quant
 * defaults and layer mix, and the per-token growth of the sequence state matches hand derivations.
 */

import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER } from '../../__fixtures__/synthetic';
import { bytesPerElement } from '../../ir/dtype';
import type { BlockSpec, ModelSpec, ParallelConfig, QuantConfig } from '../../ir/model';
import { blockSequenceStateBytes, type LoweringContext } from '../../lower/lower';
import { simulate } from '../../engine/simulate';
import type { SimResult, Workload } from '../../engine/types';
import { singleDevice } from '../../hw/presets';
import { RTX_5090 } from '../../hw/presets/nvidia';
import type { HardwareStack, TopologySpec } from '../../hw/types';
import { fixtureRepoFiles } from '../__fixtures__/fixtures.testutil';
import { mapHfModel } from './index';
import { isJsonObject } from './reader';
import type { HfMapResult } from './types';

/** One shortlist model and what the loader must make of it. */
interface ShortlistCase {
  /** HF repo ID (fixture). */
  id: string;
  /** Family that builds it. */
  family: string;
  /** Param-check status against `safetensors.total`. */
  paramCheck: 'ok' | 'packed' | 'unavailable';
  /** Decoder layers. */
  layers: number;
  /** Count of each mixer / FFN / extra block kind (see {@link mixLabel}). */
  mix: Record<string, number>;
  /** Derived default weight format (the routed experts' when they differ) and KV-cache dtype. */
  weights: string;
  kvCache: string;
}

const SHORTLIST: ShortlistCase[] = [
  {
    id: 'Qwen/Qwen3.8-27B',
    family: 'qwen',
    paramCheck: 'ok',
    layers: 64,
    mix: { 'linear:gated_delta_net': 48, 'gqa:full': 16, mlp: 64 },
    weights: 'bf16',
    kvCache: 'bf16',
  },
  {
    id: 'Qwen/Qwen3.8-Flash-Next',
    family: 'qwen',
    paramCheck: 'ok',
    layers: 48,
    mix: { 'linear:gated_delta_net': 36, 'gqa:indexer': 12, moe: 48, ngram_embedding: 1 },
    weights: 'bf16',
    kvCache: 'bf16',
  },
  {
    id: 'deepseek-ai/DeepSeek-V4.1-Flash',
    family: 'deepseek_v41',
    paramCheck: 'ok',
    layers: 40,
    mix: { 'csa:swa': 2, 'csa:indexer': 8, 'csa:reuse': 30, moe: 40, ngram_embedding: 2 },
    weights: 'mxfp4',
    kvCache: 'mxfp8',
  },
  {
    id: 'deepseek-ai/DeepSeek-V4-Flash-0731',
    family: 'deepseek_v4',
    paramCheck: 'ok',
    layers: 43,
    mix: { 'csa:swa': 2, 'csa:indexer': 21, 'csa:all': 20, moe: 43 },
    weights: 'mxfp4',
    kvCache: 'fp8_e4m3',
  },
  {
    id: 'moonshotai/Kimi-K3',
    family: 'kimi_linear',
    paramCheck: 'ok',
    layers: 93,
    mix: { 'linear:kda': 69, 'mla:dense': 24, mlp: 1, moe: 92 },
    weights: 'mxfp4',
    kvCache: 'bf16',
  },
  {
    id: 'moonshotai/Kimi-K3.1-Pro',
    family: 'kimi_linear',
    paramCheck: 'ok',
    layers: 93,
    mix: { 'linear:kda': 69, 'mla:indexer': 24, mlp: 1, moe: 92 },
    weights: 'mxfp4',
    kvCache: 'bf16',
  },
  {
    id: 'zai-org/GLM-5.3-Flash',
    family: 'glm5_next',
    paramCheck: 'ok',
    layers: 45,
    mix: { 'linear:kda': 34, 'mla:indexer': 11, mlp: 3, moe: 42 },
    weights: 'fp8_e4m3',
    kvCache: 'bf16',
  },
  {
    id: 'zai-org/GLM-5.3',
    family: 'glm_moe_dsa',
    paramCheck: 'ok',
    layers: 78,
    mix: { 'mla:indexer': 21, 'mla:reuse': 57, mlp: 3, moe: 75 },
    weights: 'fp8_e4m3',
    kvCache: 'bf16',
  },
  {
    id: 'mistralai/Mistral-Large-4-1T-A52B',
    family: 'mistral_large4',
    paramCheck: 'unavailable',
    layers: 72,
    mix: { 'mla:indexer': 35, 'mla:indexer:kv-shared': 9, 'mla:reuse': 28, mlp: 1, moe: 71 },
    weights: 'fp8_e4m3',
    kvCache: 'fp8_e4m3',
  },
  {
    id: 'thinkingmachines/Inkling-Small',
    family: 'inkling',
    paramCheck: 'ok',
    layers: 42,
    mix: { 'gqa:swa': 35, 'gqa:full': 7, short_conv: 84, mlp: 2, moe: 40 },
    weights: 'bf16',
    kvCache: 'bf16',
  },
  {
    id: 'MiniMaxAI/MiniMax-M3',
    family: 'minimax',
    paramCheck: 'ok',
    layers: 60,
    mix: { 'gqa:full': 3, 'gqa:indexer': 57, mlp: 3, moe: 57 },
    weights: 'bf16',
    kvCache: 'bf16',
  },
  {
    id: 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
    family: 'nemotron_h',
    paramCheck: 'ok',
    layers: 52,
    mix: { 'linear:mamba2': 23, 'gqa:full': 6, moe: 23 },
    weights: 'bf16',
    kvCache: 'bf16',
  },
  {
    id: 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4',
    family: 'nemotron_h',
    paramCheck: 'packed',
    layers: 88,
    mix: { 'linear:mamba2': 40, 'gqa:full': 8, moe: 40 },
    weights: 'nvfp4',
    kvCache: 'fp8_e4m3',
  },
  {
    // The gated meta-models/Muse-Spark-Open-816B-A42B, read through its ungated mirror (fixture ID).
    id: 'unsloth/Muse-Spark-Open-816B-A42B',
    family: 'muse_spark',
    paramCheck: 'ok',
    layers: 62,
    mix: { 'gqa:swa': 46, 'gqa:full': 16, moe: 62 },
    weights: 'bf16',
    kvCache: 'bf16',
  },
];

/** Batch 1, a 4096-token prompt, decode at a 4096-token context. */
const WORKLOAD: Workload = { batch: 1, promptLen: 4096, decodeContextLen: 4096 };

/** One RTX 5090 (the only device preset). */
const ONE_5090: HardwareStack = { device: RTX_5090, topology: singleDevice(RTX_5090) };

/** 8 × RTX 5090 in one node behind the synthetic 100 GB/s intra-node link of the test fixtures. */
const EIGHT_5090_TOPOLOGY: TopologySpec = {
  ...SYNTH_CLUSTER,
  id: 'synth-8x5090',
  name: '8 × RTX 5090 (synthetic link)',
  nodes: 1,
  interNode: undefined,
};
const EIGHT_5090: HardwareStack = { device: RTX_5090, topology: EIGHT_5090_TOPOLOGY };

/** A short label for the blocks the mix counts (norms, residuals, residual mixes and hyper-connections are skipped). */
function mixLabel(b: BlockSpec): string | undefined {
  switch (b.kind) {
    case 'gqa_attention':
      return `gqa:${b.slidingWindow !== undefined ? 'swa' : (b.sparse?.kind ?? 'full')}`;
    case 'mla_attention':
      return `mla:${b.sparse?.kind ?? 'dense'}${b.kvCacheFromLayer !== undefined ? ':kv-shared' : ''}`;
    case 'csa_attention':
      return `csa:${b.global?.selection.kind ?? 'swa'}`;
    case 'linear_attention':
      return `linear:${b.variant}`;
    case 'norm':
    case 'residual':
    case 'residual_mix':
    case 'hc_pre':
    case 'hc_post':
      return undefined;
    default:
      return b.kind;
  }
}

function layerMix(m: ModelSpec): Record<string, number> {
  const mix: Record<string, number> = {};
  for (const layer of m.layers) {
    for (const b of layer.blocks) {
      const label = mixLabel(b);
      if (label !== undefined) mix[label] = (mix[label] ?? 0) + 1;
    }
  }
  return mix;
}

function mapOk(id: string): Extract<HfMapResult, { kind: 'ok' }> {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${JSON.stringify(r)}`);
  return r;
}

const hasMoe = (m: ModelSpec) => m.layers.some((l) => l.blocks.some((b) => b.kind === 'moe'));

function run(model: ModelSpec, quant: QuantConfig, parallel: ParallelConfig, hardware: HardwareStack): SimResult {
  return simulate({ model, quant, parallel, workload: WORKLOAD, hardware });
}

function expectSane(r: SimResult): void {
  for (const p of [r.prefill, r.decode]) {
    expect(Number.isFinite(p.tokensPerSec) && p.tokensPerSec > 0).toBe(true);
    expect(Number.isFinite(p.latencySec) && p.latencySec > 0).toBe(true);
  }
  for (const s of r.memory.perStage) expect(Number.isFinite(s.totalBytes) && s.totalBytes > 0).toBe(true);
}

/** Σ over every block of the per-sequence state at `tokens` of context. Unit: bytes. */
function sequenceStateBytes(model: ModelSpec, quant: QuantConfig, tokens: number): number {
  const ctx: LoweringContext = { phase: 'decode', seqs: 1, qLen: 1, kvLen: tokens, quant, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 } };
  let total = 0;
  for (const layer of model.layers) for (const b of layer.blocks) total += blockSequenceStateBytes(b, ctx, tokens);
  return total;
}

/**
 * Per-token growth of the sequence state between two long contexts: sliding windows and recurrent states are
 * constant there and cancel. Both contexts are multiples of every compress ratio. Unit: bytes/token.
 */
function stateGrowthPerToken(model: ModelSpec, quant: QuantConfig): number {
  const a = 1 << 16;
  const b = a + (1 << 12);
  return (sequenceStateBytes(model, quant, b) - sequenceStateBytes(model, quant, a)) / (b - a);
}

describe('HF shortlist end to end: mapHfModel → simulate', () => {
  it.each(SHORTLIST)('$id', (c) => {
    const r = mapOk(c.id);
    expect(r.family).toBe(c.family);
    expect(r.paramCheck.status).toBe(c.paramCheck);
    if (c.paramCheck === 'ok') expect(Math.abs(r.paramCheck.relError ?? 1)).toBeLessThan(0.005);

    const m = r.model;
    expect(m.layers).toHaveLength(c.layers);
    expect(layerMix(m)).toEqual(c.mix);
    const q = r.quant.quant;
    expect(q.modules?.routed_experts?.weights.dtype ?? q.weights.dtype).toBe(c.weights);
    expect(q.kvCache.dtype).toBe(c.kvCache);

    const one = run(m, q, { tp: 1, pp: 1, dp: 1, ep: 1 }, ONE_5090);
    expectSane(one);
    // No validation warnings at pp = 1, except DeepSeek's fp32 hyper-connection GEMMs, which the RTX 5090 runs
    // on its vector fp32 peak (it has no fp32 tensor peak).
    expect(one.warnings.filter((w) => !w.includes('using the vector fp32 peak'))).toEqual([]);
    const eight = run(m, q, { tp: 8, pp: 1, dp: 1, ep: hasMoe(m) ? 8 : 1 }, EIGHT_5090);
    expectSane(eight);
    // More devices never make a phase slower here (tensor parallelism splits the weights and the compute).
    expect(eight.decode.tokensPerSec).toBeGreaterThan(one.decode.tokensPerSec);
    expect(eight.prefill.tokensPerSec).toBeGreaterThan(one.prefill.tokensPerSec);
  });
});

describe('sequence-state growth per token, by hand', () => {
  it('Qwen3.8-27B: 16 full-attention layers × 2·4·256 elements × 2 B = 65,536 B/token (GatedDeltaNet state is constant)', () => {
    const r = mapOk('Qwen/Qwen3.8-27B');
    expect(stateGrowthPerToken(r.model, r.quant.quant)).toBe(16 * 2 * 4 * 256 * 2);
  });

  it('DeepSeek-V4.1-Flash: (3·½ + 1)·(288 + 68) = 890 B/token (README)', () => {
    // 3 sources with r = 2 and 1 with r = 1 → 2.5 compressed entries per token; each entry is 512 nvfp4
    // elements (288 B) plus a 128-element mxfp4 indexer key (68 B). The 128-token windows are constant.
    const r = mapOk('deepseek-ai/DeepSeek-V4.1-Flash');
    expect(stateGrowthPerToken(r.model, r.quant.quant)).toBe(890);
  });

  it('DeepSeek-V4-Flash-0731: (21/4 + 20/128)·583 + (21/4)·68 = 3,508.84375 B/token', () => {
    // 21 r = 4 and 20 r = 128 compressors → 21/4 + 20/128 compressed entries per token, each 512 elements of fp8
    // with bf16 RoPE dims and a ue8m0 scale per 64 (448 + 128 + 7 = 583 B); the 21 r = 4 indexers add a 128-element
    // mxfp4 key (68 B) per 4 tokens. The 128-token windows and the compressor tails are constant.
    const r = mapOk('deepseek-ai/DeepSeek-V4-Flash-0731');
    expect(stateGrowthPerToken(r.model, r.quant.quant)).toBeCloseTo((21 / 4 + 20 / 128) * 583 + (21 / 4) * 68, 6);
  });

  it('GLM-5.3: 78 latents of 576 elements + 21 indexer keys of 128 elements per token', () => {
    const r = mapOk('zai-org/GLM-5.3');
    const q = r.quant.quant;
    const expected = 78 * 576 * bytesPerElement(q.kvCache) + 21 * 128 * bytesPerElement(q.indexerKeyCache ?? q.kvCache);
    expect(stateGrowthPerToken(r.model, q)).toBe(expected);
  });

  it('Kimi K3: 24 MLA latents of 576 elements per token (the KDA state is constant)', () => {
    const r = mapOk('moonshotai/Kimi-K3');
    expect(stateGrowthPerToken(r.model, r.quant.quant)).toBe(24 * 576 * 2);
  });

  it('Kimi K3.1-Pro: the same plus 24 indexer keys of 128 elements', () => {
    const r = mapOk('moonshotai/Kimi-K3.1-Pro');
    expect(stateGrowthPerToken(r.model, r.quant.quant)).toBe(24 * 576 * 2 + 24 * 128 * 2);
  });
});

describe('quant hints vs explicit repo metadata', () => {
  it("an explicit kv_cache_scheme beats DeepSeek's native kvCache hint; the other cache hints stay", () => {
    const files = fixtureRepoFiles('intel-ai/DeepSeek-V4.1-Flash-MXFP4-Mixed-CT-AutoRound', { dir: 'quant' });
    if (!isJsonObject(files.config) || !isJsonObject(files.config.quantization_config)) throw new Error('fixture shape');
    const config = {
      ...files.config,
      quantization_config: { ...files.config.quantization_config, kv_cache_scheme: { num_bits: 8, type: 'float' } },
    };
    const r = mapHfModel({ ...files, config });
    if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
    expect(r.quant.quant.kvCache).toEqual({ dtype: 'fp8_e4m3' });
    expect(r.quant.quant.compressedKvCache).toEqual({ dtype: 'nvfp4' });
    expect(r.quant.quant.indexerKeyCache).toEqual({ dtype: 'mxfp4' });
    expect(r.notes).toContain("family hint for kvCache overridden by the repo's quant metadata");
    // Without a kv_cache_scheme the native formats stay.
    expect(mapOk('deepseek-ai/DeepSeek-V4.1-Flash').quant.quant).toMatchObject({
      kvCache: { dtype: 'mxfp8' },
      compressedKvCache: { dtype: 'nvfp4' },
      indexerKeyCache: { dtype: 'mxfp4' },
    });
  });
});
