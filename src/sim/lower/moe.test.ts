import { describe, expect, it } from 'vitest';
import { lowerMoe, moeParallelErrors, moeResidentWeightBytes } from './moe';
import type { LoweringContext } from './lower';
import type { MoeBlock, ParallelConfig } from '../ir/model';
import type { CollectiveOp, GemmOp, Op } from '../ir/ops';

const H = 64;
const MOE: MoeBlock = {
  kind: 'moe',
  name: 'moe',
  hidden: H,
  experts: 8,
  topK: 2,
  expertIntermediate: 32,
  gated: true,
  activation: 'silu',
  sharedExperts: 1,
  sharedIntermediate: 64,
};

function ctx(seqs: number, parallel: Partial<ParallelConfig>): LoweringContext {
  return {
    phase: 'decode',
    seqs,
    qLen: 1,
    kvLen: 100,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
  };
}

const gemmNamed = (ops: Op[], name: string): GemmOp => {
  const op = ops.find((o) => o.name === name);
  if (op === undefined || op.kind !== 'gemm') throw new Error(`no gemm ${name}`);
  return op;
};
const collectives = (ops: Op[]): CollectiveOp[] => ops.filter((o): o is CollectiveOp => o.kind === 'collective');

describe('T4b MoE (E 8, k 2)', () => {
  it('mode A ep 1 tp 1, T 1 → a 2, mE 1', () => {
    const up = gemmNamed(lowerMoe(MOE, ctx(1, {})), 'moe.gate_up');
    expect([up.batch, up.m, up.n, up.k]).toEqual([2, 1, 64, 64]); // A = 1·2; n = 2·32
  });

  it('mode A ep 1 tp 1, T 64 → a 8, mE 16', () => {
    const up = gemmNamed(lowerMoe(MOE, ctx(64, {})), 'moe.gate_up');
    expect([up.batch, up.m]).toEqual([8, 16]); // A = 128, a = min(8, 128), mE = 128/8
  });

  it('mode A ep = tp = 8, T 1 → a 1, mE 1, no all-to-all, one all-reduce', () => {
    const ops = lowerMoe(MOE, ctx(1, { tp: 8, ep: 8 }));
    const up = gemmNamed(ops, 'moe.gate_up');
    expect([up.batch, up.m, up.n]).toEqual([1, 1, 64]); // A = ceil(2/8) = 1; Eloc 1; etp 1
    const coll = collectives(ops);
    expect(coll.map((c) => c.collective)).toEqual(['all_reduce']);
    expect(coll[0].messageBytes).toBe(128); // T 1 · 64 · 2 B
    // Shared expert sharded by tp: n = 2·64/8
    expect(gemmNamed(ops, 'moe.shared.gate_up').n).toBe(16);
  });

  it('mode A ep 1 tp 2 shards each expert 2 ways (etp 2)', () => {
    const up = gemmNamed(lowerMoe(MOE, ctx(1, { tp: 2 })), 'moe.gate_up');
    expect([up.batch, up.n]).toEqual([2, 32]); // all 8 experts local; n = 2·32/2
  });

  it('mode B tp 1 dp 8 ep 8, T 1 → a 1, mE 2, two all-to-alls of 1·2·h·2 B', () => {
    const ops = lowerMoe(MOE, ctx(1, { dp: 8, ep: 8 }));
    const up = gemmNamed(ops, 'moe.gate_up');
    expect([up.batch, up.m]).toEqual([1, 2]); // A = Ts·k = 2, Eloc 1
    const coll = collectives(ops);
    expect(coll.map((c) => [c.collective, c.group, c.groupSize, c.messageBytes])).toEqual([
      ['all_to_all', 'ep', 8, 256], // dispatch 1·2·64·2
      ['all_to_all', 'ep', 8, 256], // combine 1·2·64·max(2, 2)
    ]);
    // Shared expert unsharded on the Ts slice
    expect(gemmNamed(ops, 'moe.shared.gate_up').n).toBe(128);
  });

  it('mode B tp 2 dp 4 ep 4 ends with a TP all-gather of T·h', () => {
    const coll = collectives(lowerMoe(MOE, ctx(4, { tp: 2, dp: 4, ep: 4 })));
    expect(coll.map((c) => c.collective)).toEqual(['all_to_all', 'all_to_all', 'all_gather']);
    expect(coll[0].messageBytes).toBe(512); // Ts = 4/2 = 2; 2·2·64·2
    expect(coll[2].messageBytes).toBe(512); // T 4 · 64 · 2
  });

  it('ep 3 is invalid', () => {
    expect(moeParallelErrors(MOE, { tp: 1, pp: 1, dp: 3, ep: 3 }).length).toBeGreaterThan(0); // 8 % 3
    expect(() => lowerMoe(MOE, ctx(1, { dp: 3, ep: 3 }))).toThrow();
    // ep 4 with tp 1 dp 2: (1·2) % 4 ≠ 0
    expect(moeParallelErrors(MOE, { tp: 1, pp: 1, dp: 2, ep: 4 }).length).toBeGreaterThan(0);
  });

  it('includes the shared expert and the router', () => {
    const names = lowerMoe(MOE, ctx(1, {})).map((o) => o.name);
    expect(names).toEqual([
      'moe.router',
      'moe.router_topk',
      'moe.gate_up',
      'moe.act',
      'moe.down',
      'moe.combine',
      'moe.shared.gate_up',
      'moe.shared.act',
      'moe.shared.down',
    ]);
  });

  it('resident weights hold all local experts', () => {
    // 8·3·64·32·2 = 98,304 + router 8·64·2 = 1024 + shared 3·64·64·2 = 24,576
    expect(moeResidentWeightBytes(MOE, ctx(1, {}))).toBe(123_904);
    // ep 8 (mode B): 1 expert 3·64·32·2 = 12,288 + router 1024 + shared (unsharded) 24,576
    expect(moeResidentWeightBytes(MOE, ctx(1, { dp: 8, ep: 8 }))).toBe(37_888);
  });
});
