import { describe, expect, it } from 'vitest';
import { exl3Format } from '../ir/exl3';
import { opBytes } from '../ir/metrics';
import type { GqaAttentionBlock, MlaAttentionBlock, MoeBlock, QuantConfig } from '../ir/model';
import type { GemmOp, Op } from '../ir/ops';
import { lowerGqa } from './gqa';
import type { LoweringContext } from './lower';
import { lowerDenseFfn } from './mlp';
import { lowerMla } from './mla';
import { lowerMoe, moeResidentWeightBytes } from './moe';

const BPW = 4;
const QUANT: QuantConfig = {
  weights: exl3Format(BPW),
  activations: 'fp16',
  kvCache: { dtype: 'fp16' },
  highPrecision: 'fp16',
};
const CTX: LoweringContext = {
  phase: 'decode',
  seqs: 1,
  qLen: 1,
  kvLen: 64,
  quant: QUANT,
  parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
};

/** Exact stored bytes of one physical EXL3 matrix `out × in`: trellis bits plus fp16 suh (in) and svh (out). */
const matrix = (out: number, inn: number) => (out * inn * BPW) / 8 + (16 * (out + inn)) / 8;
const weightBytes = (ops: Op[], name: string) => {
  const op = ops.find((o): o is GemmOp => o.kind === 'gemm' && o.name === name);
  if (!op) throw new Error(`no gemm ${name}`);
  return opBytes(op).weightBytes;
};

describe('EXL3 side vectors follow the physical matrices, not the fused GEMM shape', () => {
  it('GQA: qkv_proj = q + k + v matrices, o_proj = one matrix', () => {
    const h = 64;
    const block: GqaAttentionBlock = {
      kind: 'gqa_attention',
      name: 'attn',
      hidden: h,
      heads: 8,
      kvHeads: 2,
      headDim: 8,
      qkvBias: false,
      outBias: false,
      qkNorm: false,
      rope: true,
      causal: true,
      outputGate: false,
    };
    const ops = lowerGqa(block, CTX);
    expect(weightBytes(ops, 'attn.qkv_proj')).toBeCloseTo(
      matrix(64, h) + matrix(16, h) + matrix(16, h),
      9,
    );
    expect(weightBytes(ops, 'attn.o_proj')).toBeCloseTo(matrix(h, 64), 9);
    // Fused-shape accounting would have charged one 96-row matrix: strictly less.
    expect(weightBytes(ops, 'attn.qkv_proj')).toBeGreaterThan(matrix(96, h));
    // An output gate doubles q only.
    const gated = lowerGqa({ ...block, outputGate: true }, CTX);
    expect(weightBytes(gated, 'attn.qkv_proj')).toBeCloseTo(matrix(128, h) + 2 * matrix(16, h), 9);
  });

  it('dense FFN: gate_up = gate + up, down = one matrix; ungated has no second matrix', () => {
    const shape = {
      hidden: 64,
      intermediate: 96,
      gated: true,
      activation: 'silu',
      bias: false,
    } as const;
    const ops = lowerDenseFfn('mlp', shape, 1, 1, 'dense_mlp', CTX);
    expect(weightBytes(ops, 'mlp.gate_up')).toBeCloseTo(2 * matrix(96, 64), 9);
    expect(weightBytes(ops, 'mlp.down')).toBeCloseTo(matrix(64, 96), 9);
    const plain = lowerDenseFfn('mlp', { ...shape, gated: false }, 1, 1, 'dense_mlp', CTX);
    expect(weightBytes(plain, 'mlp.gate_up')).toBeCloseTo(matrix(96, 64), 9);
  });

  it('MoE: expert and shared-expert GEMMs and the resident bytes count every physical matrix once', () => {
    const moe: MoeBlock = {
      kind: 'moe',
      name: 'moe',
      hidden: 64,
      experts: 8,
      topK: 2,
      expertIntermediate: 32,
      gated: true,
      activation: 'silu',
      sharedExperts: 1,
      sharedIntermediate: 48,
    };
    const router = 8 * 64 * 2; // fp16 router
    const resident = moeResidentWeightBytes(moe, CTX);
    expect(resident).toBeCloseTo(8 * 3 * matrix(32, 64) + router + 3 * matrix(48, 64), 6);
    const ops = lowerMoe(moe, CTX);
    // a = active experts per step = min(E, T·k) = 2.
    expect(weightBytes(ops, 'moe.gate_up')).toBeCloseTo(2 * 2 * matrix(32, 64), 9);
    expect(weightBytes(ops, 'moe.down')).toBeCloseTo(2 * matrix(64, 32), 9);
  });

  it('MoE without shared experts adds no side-vector bytes for the absent matrices', () => {
    const moe: MoeBlock = {
      kind: 'moe',
      name: 'moe',
      hidden: 64,
      experts: 8,
      topK: 2,
      expertIntermediate: 32,
      gated: true,
      activation: 'silu',
      sharedExperts: 0,
      sharedIntermediate: 0,
    };
    const router = 8 * 64 * 2;
    expect(moeResidentWeightBytes(moe, CTX)).toBeCloseTo(8 * 3 * matrix(32, 64) + router, 6);
    expect(lowerMoe(moe, CTX).some((o) => o.name.includes('shared'))).toBe(false);
  });

  it('MLA: the absorbed (decode) and MHA (prefill) forms store the same kv_b bytes', () => {
    const block: MlaAttentionBlock = {
      kind: 'mla_attention',
      name: 'mla',
      hidden: 64,
      heads: 4,
      qLoraRank: 16,
      kvLoraRank: 24,
      qkNopeDim: 8,
      qkRopeDim: 4,
      vHeadDim: 8,
    };
    const Hl = 4;
    const dn = 8;
    const dv = 8;
    const rkv = 24;
    const kvB = matrix(Hl * (dn + dv), rkv); // one stored matrix: rows Hl·(dn + dv), columns rkv
    const prefill = lowerMla(block, { ...CTX, phase: 'prefill', qLen: 16, kvLen: 16 });
    const decode = lowerMla(block, CTX);
    expect(weightBytes(prefill, 'mla.kv_b')).toBeCloseTo(kvB, 9);
    expect(weightBytes(decode, 'mla.q_absorb') + weightBytes(decode, 'mla.v_absorb')).toBeCloseTo(
      kvB,
      9,
    );
  });
});
