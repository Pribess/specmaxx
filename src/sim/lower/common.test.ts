import { describe, expect, it } from 'vitest';
import type { BlockSpec, QuantConfig } from '../ir/model';
import type { GemmOp, Op } from '../ir/ops';
import { linearGemm, moduleQuant } from './common';
import { blockParams } from '../ir/params';
import { blockResidentWeightBytes, blockSequenceStateBytes, lowerBlock, lowerModel, type LoweringContext } from './lower';
import { TINY_HYBRID, TINY_MLA_SPARSE } from '../models/reference';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };

function ctx(quant: QuantConfig): LoweringContext {
  return { phase: 'decode', seqs: 4, qLen: 1, kvLen: 128, quant, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 } };
}

const gemms = (ops: Op[]): GemmOp[] => ops.filter((o): o is GemmOp => o.kind === 'gemm');
const allOps = (quant: QuantConfig): Op[] =>
  lowerModel(TINY_HYBRID, ctx(quant)).stages.flatMap((s) => s.layers.flatMap((l) => l.ops));

describe('module formats', () => {
  it('a QuantConfig with only `weights` lowers exactly like modules that repeat it', () => {
    const repeated: QuantConfig = {
      ...BF16,
      modules: {
        attention: { weights: { dtype: 'bf16' } },
        linear_attention: { weights: { dtype: 'bf16' } },
        dense_mlp: { weights: { dtype: 'bf16' } },
        shared_experts: { weights: { dtype: 'bf16' } },
        routed_experts: { weights: { dtype: 'bf16' } },
      },
    };
    expect(allOps(repeated)).toEqual(allOps(BF16));
  });

  it('routed experts in NVFP4 W4A4: only the expert GEMMs change (format, input, math)', () => {
    const quant: QuantConfig = { ...BF16, modules: { routed_experts: { weights: { dtype: 'nvfp4' }, gemmInput: 'nvfp4' } } };
    const changed = gemms(allOps(quant));
    const base = gemms(allOps(BF16));
    changed.forEach((g, i) => {
      if (g.name === 'moe.gate_up' || g.name === 'moe.down') {
        expect(g.weight.dtype).toBe('nvfp4');
        expect(g.inputDtype).toBe('nvfp4');
        expect(g.math).toBe('fp4');
        expect(g.outputDtype).toBe('bf16');
      } else {
        expect(g).toEqual(base[i]);
      }
    });
    // resident MoE bytes: the routed experts shrink, shared experts and router stay bf16
    const moe = TINY_HYBRID.layers[1].blocks.find((b) => b.kind === 'moe');
    if (moe === undefined) throw new Error('expected a MoE block');
    // experts 8·3·256·128 = 786,432 params: bf16 2 B → nvfp4 0.5625 B (4 bit + 8-bit scale per 16)
    expect(blockResidentWeightBytes(moe, ctx(BF16)) - blockResidentWeightBytes(moe, ctx(quant))).toBe(
      786_432 * (2 - 0.5625),
    );
  });

  it('residual_mixer defaults to highPrecision and is never 2:4 sparse; other modules take weightSparsity', () => {
    const quant: QuantConfig = { ...BF16, weights: { dtype: 'fp8_e4m3' }, weightSparsity: '2:4', highPrecision: 'fp32' };
    expect(moduleQuant(ctx(quant), 'residual_mixer')).toEqual({
      weights: { dtype: 'fp32' },
      gemmInput: 'bf16',
      sparsity: 'dense',
    });
    expect(moduleQuant(ctx(quant), 'ngram')).toEqual({ weights: { dtype: 'fp8_e4m3' }, gemmInput: 'bf16', sparsity: '2:4' });
    // an explicit non-module weight (router, LM head) stays dense
    expect(linearGemm('router', { m: 1, n: 8, k: 8 }, ctx(quant), { format: { weights: { dtype: 'bf16' } } }).sparsity).toBe('dense');
    expect(() => linearGemm('x', { m: 1, n: 1, k: 1 }, ctx(quant), { module: 'attention', format: { weights: { dtype: 'bf16' } } })).toThrow();
  });

  it('MLA latent projections and indexer keys/weights get their own formats, falling back to the parent', () => {
    const fp8 = { weights: { dtype: 'fp8_e4m3' } } as const;
    const byName = (quant: QuantConfig, phase: 'prefill' | 'decode'): Record<string, string> => {
      const c = { ...ctx(quant), phase, qLen: phase === 'prefill' ? 64 : 1 };
      const ops = lowerModel(TINY_MLA_SPARSE, c).stages[0].layers[1].ops;
      return Object.fromEntries(gemms(ops).map((g) => [g.name, g.weight.dtype]));
    };
    // GLM-5.3-Flash FP8: q_a/q_b/kv_a/o FP8, kv_b (and its absorbed slices) BF16
    const glm: QuantConfig = { ...BF16, modules: { attention: fp8, attention_kv_up: { weights: { dtype: 'bf16' } } } };
    expect(byName(glm, 'decode')).toMatchObject({
      'mla.q_a': 'fp8_e4m3',
      'mla.q_b': 'fp8_e4m3',
      'mla.kv_a': 'fp8_e4m3',
      'mla.q_absorb': 'bf16',
      'mla.v_absorb': 'bf16',
      'mla.o_proj': 'fp8_e4m3',
    });
    // resident MLA bytes: fp8 q_a 512·128 + q_b 128·8·48 + kv_a 512·80 + o 8·32·512 = 286,720 (1 B);
    // bf16 absorb slices of kv_b 2·8·64·32 · 2 B = 65,536; norms (128 + 64)·2 = 384; indexer 34,880·2 = 69,760
    const mla = TINY_MLA_SPARSE.layers[0].blocks[1];
    expect(blockResidentWeightBytes(mla, ctx(glm))).toBe(286_720 + 65_536 + 384 + 69_760);
    // a sub-class override without gemmInput stays weight-only under a W8A8 parent
    const w8a8: QuantConfig = {
      ...BF16,
      modules: { attention: { weights: { dtype: 'fp8_e4m3' }, gemmInput: 'fp8_e4m3' }, attention_kv_up: { weights: { dtype: 'bf16' } } },
    };
    const w8a8Ops = gemms(lowerModel(TINY_MLA_SPARSE, ctx(w8a8)).stages[0].layers[1].ops);
    expect(w8a8Ops.find((g) => g.name === 'mla.q_a')?.inputDtype).toBe('fp8_e4m3');
    expect(w8a8Ops.find((g) => g.name === 'mla.q_absorb')).toMatchObject({ inputDtype: 'bf16', math: 'bf16' });
    // Mistral Large 4 FP8: kv_a and kv_b BF16; indexer query FP8, wk and weights_proj BF16
    const mistral: QuantConfig = {
      ...BF16,
      weights: { dtype: 'fp8_e4m3' },
      modules: {
        attention_kv_down: { weights: { dtype: 'bf16' } },
        attention_kv_up: { weights: { dtype: 'bf16' } },
        indexer_key: { weights: { dtype: 'bf16' } },
        indexer_head_weights: { weights: { dtype: 'bf16' } },
      },
    };
    expect(byName(mistral, 'decode')).toMatchObject({
      'mla.q_b': 'fp8_e4m3',
      'mla.kv_a': 'bf16',
      'mla.q_absorb': 'bf16',
      'idx.q': 'fp8_e4m3',
      'idx.k': 'bf16',
      'idx.w': 'bf16',
    });
    // unset sub-classes take the parent's entry
    const parentsOnly: QuantConfig = { ...BF16, modules: { attention: fp8, indexer: fp8 } };
    expect(byName(parentsOnly, 'decode')).toMatchObject({ 'mla.kv_a': 'fp8_e4m3', 'mla.v_absorb': 'fp8_e4m3', 'idx.k': 'fp8_e4m3', 'idx.w': 'fp8_e4m3' });
    expect(moduleQuant(ctx(parentsOnly), 'attention_kv_up').weights).toEqual(fp8.weights);
  });

  it('LM head W4A4: lmHeadGemmInput sets the input dtype and the math', () => {
    const quant: QuantConfig = { ...BF16, lmHead: { dtype: 'nvfp4' }, lmHeadGemmInput: 'nvfp4' };
    const head = gemms(allOps(quant)).find((g) => g.name === 'lm_head');
    expect(head).toMatchObject({ weight: { dtype: 'nvfp4' }, inputDtype: 'nvfp4', math: 'fp4', sparsity: 'dense' });
    // default: weight-only (bf16 input, bf16 math)
    const weightOnly = gemms(allOps({ ...BF16, lmHead: { dtype: 'nvfp4' } })).find((g) => g.name === 'lm_head');
    expect(weightOnly).toMatchObject({ inputDtype: 'bf16', math: 'bf16' });
  });
});

describe('resident weights of every new block kind equal its params · 2 B (bf16, tp 1)', () => {
  const blocks: BlockSpec[] = [
    {
      kind: 'csa_attention',
      name: 'csa',
      hidden: 64,
      heads: 2,
      headDim: 32,
      ropeDim: 8,
      qLoraRank: 16,
      outGroups: 2,
      outLoraRank: 8,
      windowTokens: 8,
      global: { compressRatio: 2, selection: { kind: 'indexer', indexer: { heads: 2, headDim: 8, topK: 4, keyPoolTokens: 2, keyInputDim: 32 } } },
    },
    { kind: 'hc_pre', name: 'hc_pre', streams: 2, dim: 64, normGamma: true, projections: [{ name: 'p', inputDim: 128, outputDim: 6 }], sinkhornIters: 3, vectorParams: 9 },
    { kind: 'hc_post', name: 'hc_post', streams: 2, dim: 64, mixesStreams: true },
    {
      kind: 'ngram_embedding',
      name: 'ngram',
      tableRows: 10,
      rowDim: 4,
      rowsPerToken: 2,
      projections: [{ name: 'v', inputDim: 8, outputDim: 64 }],
      outputDim: 64,
      conv: { kernel: 4, dilation: 2 },
      vectorParams: 64,
    },
    { kind: 'short_conv', name: 'sc', dim: 64, kernel: 4 },
    {
      kind: 'mla_attention',
      name: 'mla',
      hidden: 64,
      heads: 2,
      qLoraRank: 0,
      kvLoraRank: 16,
      qkNopeDim: 8,
      qkRopeDim: 0,
      vHeadDim: 8,
      kvCacheFromLayer: 0,
      sparse: { kind: 'indexer', indexer: { heads: 2, headDim: 8, topK: 8, keyPoolTokens: 4, keyPoolGate: true } },
    },
  ];
  it.each(blocks.map((b) => [b.name, b] as const))('%s', (_name, block) => {
    expect(blockResidentWeightBytes(block, ctx(BF16))).toBe(2 * blockParams(block));
  });
  it('KDA: all but A_log and dt_bias (vH + vH·dk vectors, not modeled as weight reads)', () => {
    const kda: BlockSpec = {
      kind: 'linear_attention',
      name: 'kda',
      variant: 'kda',
      hidden: 64,
      qkHeads: 2,
      vHeads: 2,
      keyDim: 16,
      valueDim: 16,
      convKernel: 4,
      outputGate: true,
      gateRank: 8,
    };
    expect(blockResidentWeightBytes(kda, ctx(BF16))).toBe(2 * (blockParams(kda) - 2 - 2 * 16));
  });
});

describe('top-k reuse', () => {
  it("a 'reuse' selection lowers as top-k attention without indexer ops or keys", () => {
    const block = TINY_MLA_SPARSE.layers[0].blocks[1];
    if (block.kind !== 'mla_attention') throw new Error('expected an MLA block');
    const reuse = { ...block, sparse: { kind: 'reuse', topK: 64 } } as const;
    const ops = lowerBlock(reuse, ctx(BF16));
    expect(ops.some((o) => o.kind === 'indexer' || o.name.startsWith('idx.'))).toBe(false);
    expect(ops.find((o) => o.kind === 'attention')).toMatchObject({ kvSelection: { kind: 'topk', topK: 64 } });
    // latent only: 100·(64 + 16)·2 B
    expect(blockSequenceStateBytes(reuse, ctx(BF16), 100)).toBe(16_000);
  });
});
