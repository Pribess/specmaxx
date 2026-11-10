import { describe, expect, it } from 'vitest';
import { blockParams } from '../../../ir/params';
import { createReader } from '../reader';
import {
  dsaSelection,
  mhcCollapse,
  mhcResidual,
  readActivation,
  readDsaIndexer,
  readIndexerSchedule,
  readMlaMixer,
  readMlpSchedule,
  readMoeFfn,
} from './mlaCommon';

const MLA = {
  num_attention_heads: 64,
  kv_lora_rank: 512,
  qk_nope_head_dim: 64,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
};

describe('mlaCommon', () => {
  it('reads MLA dims from qk_nope/qk_rope/v_head_dim and ignores head_dim / num_key_value_heads', () => {
    const r = createReader({ ...MLA, head_dim: 999, num_key_value_heads: 1, q_lora_rank: null });
    expect(readMlaMixer(r)).toEqual({
      kind: 'mla',
      heads: 64,
      qLoraRank: 0,
      kvLoraRank: 512,
      qkNopeDim: 64,
      qkRopeDim: 64,
      vHeadDim: 128,
    });
    expect(readMlaMixer(r, false)).toMatchObject({ qkNopeDim: 128, qkRopeDim: 0 });
    expect(r.errors).toEqual([]);
  });

  it('reads the DSA indexer with an optional gated k-pool', () => {
    const base = { index_n_heads: 32, index_head_dim: 128, index_topk: 2048 };
    expect(readDsaIndexer(createReader(base))).toEqual({ heads: 32, headDim: 128, topK: 2048 });
    expect(readDsaIndexer(createReader({ ...base, index_kpool: 1 }))).toEqual({
      heads: 32,
      headDim: 128,
      topK: 2048,
    });
    expect(
      readDsaIndexer(createReader({ ...base, index_kpool: 4, index_kpool_compress: true })),
    ).toEqual({
      heads: 32,
      headDim: 128,
      topK: 2048,
      keyPoolTokens: 4,
      keyPoolGate: true,
    });
  });

  it('derives the indexer schedule like transformers', () => {
    expect(readIndexerSchedule(createReader({ indexer_types: ['full', 'shared'] }), 2)).toEqual([
      'full',
      'shared',
    ]);
    expect(readIndexerSchedule(createReader({ index_topk_pattern: 'FSSF' }), 4)).toEqual([
      'full',
      'shared',
      'shared',
      'full',
    ]);
    // GLM-5.3: freq 4, offset 3 → full at 0, 1, 2, 6, 10, …
    const glm = readIndexerSchedule(
      createReader({ index_topk_freq: 4, index_skip_topk_offset: 3 }),
      11,
    );
    expect(glm.flatMap((m, i) => (m === 'full' ? [i] : []))).toEqual([0, 1, 2, 6, 10]);
    expect(readIndexerSchedule(createReader({}), 3)).toEqual(['full', 'full', 'full']);
    const bad = createReader({ indexer_types: ['full', 'weird'] });
    readIndexerSchedule(bad, 3);
    expect(bad.errors).toEqual([
      'indexer_types: 2 entries for 3 layers',
      "indexer_types[1]: unknown mode 'weird'",
    ]);
  });

  it('derives the FFN schedule from mlp_layer_types or first_k_dense_replace', () => {
    expect(readMlpSchedule(createReader({ mlp_layer_types: ['dense', 'sparse'] }), 2, 3)).toEqual([
      'dense',
      'sparse',
    ]);
    expect(readMlpSchedule(createReader({ first_k_dense_replace: 1 }), 3, 3)).toEqual([
      'dense',
      'sparse',
      'sparse',
    ]);
    expect(readMlpSchedule(createReader({}), 4, 3)).toEqual(['dense', 'dense', 'dense', 'sparse']);
  });

  it('fuses the shared experts into one MLP of width moe_intermediate_size · n_shared_experts', () => {
    const r = createReader({
      n_routed_experts: 8,
      num_experts_per_tok: 2,
      moe_intermediate_size: 100,
      n_shared_experts: 2,
    });
    expect(readMoeFfn(r, 'silu')).toEqual({
      kind: 'moe',
      experts: 8,
      topK: 2,
      expertIntermediate: 100,
      sharedExperts: 1,
      sharedIntermediate: 200,
      activation: 'silu',
      gated: true,
    });
  });

  it('maps activations and rejects unknown ones', () => {
    expect(readActivation(createReader({}))).toBe('silu');
    expect(readActivation(createReader({ hidden_act: 'gelu_pytorch_tanh' }))).toBe('gelu');
    const bad = createReader({ hidden_act: 'tanh' });
    readActivation(bad);
    expect(bad.errors).toEqual(["hidden_act: unsupported activation 'tanh'"]);
  });

  it('builds the mHC residual: one N·D → (2+N)·N projection, (2+N)·N + 3 vectors, no γ', () => {
    const res = mhcResidual(4, 5120, 20);
    expect(res).toEqual({
      kind: 'hyper',
      streams: 4,
      pre: {
        normGamma: false,
        projections: [{ name: 'hc.fn', inputDim: 20480, outputDim: 24 }],
        sinkhornIters: 20,
        vectorParams: 27,
      },
      mixesStreams: true,
      subNorms: true,
    });
    // DeepSeek-V4.1 headers: hc_attn_fn [24, 20480] + hc_attn_base [24] + hc_attn_scale [3].
    if (res.kind === 'hyper')
      expect(blockParams({ kind: 'hc_pre', name: 'x', streams: 4, dim: 5120, ...res.pre })).toBe(
        491_547,
      );
    expect(blockParams(mhcCollapse(4, 5120))).toBe(0);
  });

  it('turns an indexer mode into a selection', () => {
    const ix = { heads: 1, headDim: 2, topK: 3 };
    expect(dsaSelection('full', ix)).toEqual({ kind: 'indexer', indexer: ix });
    expect(dsaSelection('shared', ix)).toEqual({ kind: 'reuse', topK: 3 });
  });
});
