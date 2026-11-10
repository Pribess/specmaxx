import { describe, expect, it } from 'vitest';
import { opFlops } from '../ir/metrics';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  lowerBlock,
  lowerModel,
  type LoweringContext,
} from './lower';
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
const recurrentOf = (ops: Op[]): RecurrentOp | undefined =>
  ops.find((o): o is RecurrentOp => o.kind === 'recurrent');
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
    expect(layerOps(TINY_HYBRID_DENSE_FIRST, ctx('decode'), 1).map((o) => o.name)).toContain(
      'moe.router',
    );
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
    expect(() => lowerBlock(gdn, ctx('decode', 4))).toThrow(
      /vHeads \(2\) must be divisible by tp \(4\)/,
    );
    expect(() => blockResidentWeightBytes(gdn, ctx('decode', 4))).toThrow(
      /vHeads \(2\) must be divisible by tp \(4\)/,
    );
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

describe('Mamba-2 (Nemotron-3.5-Lightning mixer: h 2688, G 8, H 64, N 128, P 64, K 4, conv bias, Q 128)', () => {
  const mamba: LinearAttentionBlock = {
    kind: 'linear_attention',
    name: 'mixer',
    variant: 'mamba2',
    hidden: 2688,
    qkHeads: 8,
    vHeads: 64,
    keyDim: 128,
    valueDim: 64,
    convKernel: 4,
    convBias: true,
    chunkTokens: 128,
    outputGate: true,
  };
  const prefill = (qLen: number, tp = 1): LoweringContext => ({
    ...ctx('prefill', tp),
    qLen,
    kvLen: qLen,
  });
  const gemm = (ops: Op[], name: string): GemmOp | undefined =>
    ops.find((o): o is GemmOp => o.kind === 'gemm' && o.name === name);

  it('decode, tp 1: in_proj has ONE dt per head, conv taps + bias, out-norm γ over H·P', () => {
    const ops = lowerBlock(mamba, ctx('decode'));
    const inProj = gemm(ops, 'lin.in_proj');
    // n = 2·8·128 (B, C) + 64·64·2 (x, z) + 64 (dt) = 10,304
    expect(inProj?.n).toBe(10_304);
    expect(inProj?.outParts).toEqual([1024, 1024, 4096, 4096, 64]);
    const conv = fnOf(ops, 'causal_conv1d');
    // C = 2·8·128 + 64·64 = 6144; taps (6144·4 + 6144 bias)·2 B = 61,440
    expect(conv?.elems).toBe(6144);
    expect(conv?.weightBytes).toBe(61_440);
    // grouped RMSNorm γ over H·P = 4096 channels · 2 B
    expect(fnOf(ops, 'gated_rmsnorm')?.weightBytes).toBe(8192);
    const rec = recurrentOf(ops);
    expect([rec?.variant, rec?.mode, rec?.chunkLen, rec?.stateFlopsPerElemPerToken]).toEqual([
      'mamba2',
      'step',
      1,
      5,
    ]);
    expect([rec?.heads, rec?.qkHeads, rec?.keyDim, rec?.valueDim]).toEqual([64, 8, 128, 64]);
    expect(rec === undefined ? 0 : opFlops(rec)).toBe(2_621_440); // 64·5·8192
    expect(gemm(ops, 'lin.out_proj')?.k).toBe(4096);
  });

  it('prefill uses chunkTokens: Q = min(128, qLen); without it the default 64', () => {
    const rec = recurrentOf(lowerBlock(mamba, prefill(4096)));
    expect([rec?.mode, rec?.chunkLen, rec?.stateFlopsPerElemPerToken]).toEqual(['chunked', 128, 4]);
    // 4096·(64·4·8192 + 129·(8·128 + 64·64))
    expect(rec === undefined ? 0 : opFlops(rec)).toBe(11_295_260_672);
    expect(recurrentOf(lowerBlock(mamba, prefill(100)))?.chunkLen).toBe(100);
    const { chunkTokens: _q, ...noChunk } = mamba;
    void _q;
    expect(recurrentOf(lowerBlock(noChunk, prefill(4096)))?.chunkLen).toBe(64);
  });

  it('sequence state: 64·128·64·4 + 6144·3·2 = 2,134,016 B per layer, context-independent', () => {
    expect(blockSequenceStateBytes(mamba, ctx('decode'), 1000)).toBe(2_134_016);
    expect(blockSequenceStateBytes(mamba, ctx('decode'), 1_000_000)).toBe(2_134_016);
  });

  it('tp 8: vL 8, qkL 1, C 768', () => {
    const ops = lowerBlock(mamba, ctx('decode', 8));
    const rec = recurrentOf(ops);
    expect([rec?.heads, rec?.qkHeads]).toEqual([8, 1]);
    expect(fnOf(ops, 'causal_conv1d')?.elems).toBe(768); // 2·1·128 + 8·64
    expect(gemm(ops, 'lin.in_proj')?.n).toBe(1288); // 256 + 8·64·2 + 8
    expect(fnOf(ops, 'gated_rmsnorm')?.weightBytes).toBe(1024); // 8·64·2 B
    // 8·8192·4 + 768·3·2
    expect(blockSequenceStateBytes(mamba, ctx('decode', 8), 1000)).toBe(266_752);
  });

  it('params 38,742,208 (= headers); resident weights = params·2 B minus the 3·H unstreamed scalars', () => {
    // in_proj 2688·10304 = 27,697,152 + conv 24,576 + bias 6,144 + A_log/D/dt_bias 192 + γ 4096 + out 11,010,048
    expect(blockParams(mamba)).toBe(38_742_208);
    expect(blockResidentWeightBytes(mamba, ctx('decode'))).toBe(38_742_208 * 2 - 3 * 64 * 2);
  });

  it('chunkTokens does not change a GDN block that leaves it unset', () => {
    const gdn = TINY_HYBRID.layers[0].blocks[1];
    expect(recurrentOf(lowerBlock(gdn, ctx('prefill')))?.chunkLen).toBe(64);
  });
});
