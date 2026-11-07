import { describe, expect, it } from 'vitest';
import { blockResidentWeightBytes, blockSequenceStateBytes, lowerBlock, lowerModel, type LoweringContext } from './lower';
import type { LinearAttentionBlock } from '../ir/model';
import type { ElementwiseOp, GemmOp, Op, RecurrentOp } from '../ir/ops';
import { blockParams } from '../ir/params';
import { TINY_HYBRID, TINY_HYBRID_DENSE_FIRST } from '../models/reference';

function ctx(phase: 'prefill' | 'decode', tp = 1): LoweringContext {
  return {
    phase,
    seqs: 1,
    qLen: phase === 'prefill' ? 256 : 1,
    kvLen: phase === 'prefill' ? 256 : 300,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp, pp: 1, dp: 1, ep: 1 },
  };
}

const layerOps = (model: typeof TINY_HYBRID, c: LoweringContext, i: number): Op[] => {
  const layer = lowerModel(model, c).stages[0].layers.find((l) => l.layerIndex === i);
  if (layer === undefined) throw new Error(`no layer ${i}`);
  return layer.ops;
};
const recurrentOf = (ops: Op[]): RecurrentOp | undefined => ops.find((o): o is RecurrentOp => o.kind === 'recurrent');
const fnOf = (ops: Op[], fn: ElementwiseOp['fn']): ElementwiseOp | undefined =>
  ops.find((o): o is ElementwiseOp => o.kind === 'elementwise' && o.fn === fn);

describe('T4d TINY_HYBRID (GDN qkH 2, vH 4, dk = dv = 32, conv 4, gate; layer 3 gated GQA)', () => {
  it('layers 0–2: recurrent step (c_s 7) + causal_conv1d in decode', () => {
    for (const i of [0, 1, 2]) {
      const ops = layerOps(TINY_HYBRID, ctx('decode'), i);
      const rec = recurrentOf(ops);
      expect(rec?.mode).toBe('step');
      expect(rec?.stateFlopsPerElemPerToken).toBe(7);
      expect(rec?.chunkLen).toBe(1);
      expect(rec?.hasPriorState).toBe(true);
      expect([rec?.heads, rec?.qkHeads, rec?.keyDim, rec?.valueDim]).toEqual([4, 2, 32, 32]);
      const conv = fnOf(ops, 'causal_conv1d');
      expect(conv?.flopsPerElem).toBe(8); // 2·K
      expect(conv?.elems).toBe(256); // T 1 · C 256
      expect(conv?.stateReadBytes).toBe(1536); // 1 seq · 256 · 3 · 2 B
      expect(fnOf(ops, 'gated_rmsnorm')).toBeDefined();
      const inProj = ops.find((o): o is GemmOp => o.kind === 'gemm' && o.name === 'lin.in_proj');
      expect(inProj?.n).toBe(392); // 2·2·32 + 4·32·2 + 2·4
    }
  });

  it('layer 3: gated GQA, no recurrent', () => {
    const ops = layerOps(TINY_HYBRID, ctx('decode'), 3);
    expect(recurrentOf(ops)).toBeUndefined();
    expect(fnOf(ops, 'sigmoid_mul')?.elems).toBe(256); // T 1 · 4 heads · 64
    const qkv = ops.find((o): o is GemmOp => o.kind === 'gemm' && o.name === 'attn.qkv_proj');
    expect(qkv?.n).toBe(640); // (4·2 + 2·1)·64
  });

  it('prefill uses the chunked form; the first step has no prior state', () => {
    const ops = layerOps(TINY_HYBRID, ctx('prefill'), 0);
    const rec = recurrentOf(ops);
    expect([rec?.mode, rec?.chunkLen, rec?.stateFlopsPerElemPerToken, rec?.hasPriorState]).toEqual([
      'chunked',
      64,
      4,
      false,
    ]);
    expect(fnOf(ops, 'causal_conv1d')?.stateReadBytes).toBe(0);
  });

  it('GDN sequence state is fixed: 4·32·32·4 + 256·3·2 = 17,920 B', () => {
    const gdn = TINY_HYBRID.layers[0].blocks[1];
    // C = 2·2·32 + 4·32 = 256 conv channels; state fp32, conv state bf16
    expect(blockSequenceStateBytes(gdn, ctx('decode'), 1000)).toBe(17_920);
    expect(blockSequenceStateBytes(gdn, ctx('decode'), 100_000)).toBe(17_920); // context-independent
  });

  it('GDN resident weights = params·2 B at tp 1, minus the per-head A_log/dt_bias scalars', () => {
    const gdn = TINY_HYBRID.layers[0].blocks[1];
    // 134,184·2 − 2·vH·2 B: A_log and dt_bias (2·4 scalars) are not streamed by any op (negligible)
    expect(blockResidentWeightBytes(gdn, ctx('decode'))).toBe(blockParams(gdn) * 2 - 16);
  });

  it('tp 2 shards GDN heads (qkL 1, vL 2) and all-reduces', () => {
    const ops = layerOps(TINY_HYBRID, ctx('decode', 2), 0);
    const rec = recurrentOf(ops);
    expect([rec?.qkHeads, rec?.heads]).toEqual([1, 2]);
    expect(ops.filter((o) => o.kind === 'collective')).toHaveLength(2); // mixer + MoE all-reduce
  });

  it('dense-first variant has an mlp in layer 0 and MoE after', () => {
    const names0 = layerOps(TINY_HYBRID_DENSE_FIRST, ctx('decode'), 0).map((o) => o.name);
    expect(names0).toContain('mlp.gate_up');
    expect(names0).not.toContain('moe.router');
    expect(layerOps(TINY_HYBRID_DENSE_FIRST, ctx('decode'), 1).map((o) => o.name)).toContain('moe.router');
  });
});

describe('value-head sharding (review P2: no replicated value heads)', () => {
  /** GDN with qkHeads 1, vHeads 2, dk = dv = 64, conv 4, z-gate, h 256. */
  const gdn: LinearAttentionBlock = {
    kind: 'linear_attention',
    name: 'mixer',
    variant: 'gated_delta_net',
    hidden: 256,
    qkHeads: 1,
    vHeads: 2,
    keyDim: 64,
    valueDim: 64,
    convKernel: 4,
    outputGate: true,
  };

  it('tp 4 > vHeads 2 is rejected (lowering, resident weights, state)', () => {
    // Replicating the 2 value heads over 4 ranks would make 4 ranks each project 64 features into the
    // row-parallel out_proj: 256 summed features for 128 logical ones.
    expect(() => lowerBlock(gdn, ctx('decode', 4))).toThrow(/vHeads \(2\) must be divisible by tp \(4\)/);
    expect(() => blockResidentWeightBytes(gdn, ctx('decode', 4))).toThrow(/vHeads \(2\) must be divisible by tp \(4\)/);
    expect(() => blockSequenceStateBytes(gdn, ctx('decode', 4), 1000)).toThrow(/vHeads/);
  });

  it('tp 2: vL 1, the q/k head replicated; resident weights hand-derived', () => {
    const ops = lowerBlock(gdn, ctx('decode', 2));
    const rec = recurrentOf(ops);
    expect([rec?.qkHeads, rec?.heads]).toEqual([1, 1]);
    // in_proj: n = 2·qkL·dk + vL·dv·2 (v, z) + 2·vL (gates) = 128 + 128 + 2 = 258; 258·256·2 B = 132,096
    // conv taps: C·K·2 B with C = 2·1·64 + 1·64 = 192 → 192·4·2 = 1,536
    // gated_rmsnorm gamma: dv·2 B = 128
    // out_proj: n = 256, k = vL·dv = 64 → 256·64·2 B = 32,768
    // total 132,096 + 1,536 + 128 + 32,768 = 166,528 B
    expect(blockResidentWeightBytes(gdn, ctx('decode', 2))).toBe(166_528);
    // The out_proj rows over both ranks add up to the logical vH·dv = 128 exactly once.
    const out = ops.find((o): o is GemmOp => o.kind === 'gemm' && o.name === 'lin.out_proj');
    expect((out?.k ?? 0) * 2).toBe(2 * 64);
  });
});
