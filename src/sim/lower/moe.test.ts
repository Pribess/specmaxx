import { describe, expect, it } from 'vitest';
import { opFlops } from '../ir/metrics';
import {
  expertWeightBytes,
  lowerMoe,
  moeLatentWeightBytes,
  moeLocalExpertBytes,
  moeParallelErrors,
  moeResidentWeightBytes,
} from './moe';
import type { LoweringContext, OffloadLowering } from './lower';
import type { MoeBlock, ParallelConfig } from '../ir/model';
import type { CollectiveOp, ElementwiseOp, GemmOp, Op, TransferOp } from '../ir/ops';

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

// Latent MoE (Muse Spark, Nemotron): h 64, experts at l 32 (fe 32), router and shared expert at h.
const LAT: MoeBlock = { ...MOE, latentDim: 32 };

/** Parallel configs covering mode A (ep 1, ep = tp, ep < tp) and mode B (tp 1, tp > 1). */
const PARALLELS: Partial<ParallelConfig>[] = [{}, { tp: 2 }, { tp: 4 }, { tp: 8, ep: 8 }, { tp: 4, ep: 2 }, { dp: 8, ep: 8 }, { tp: 2, dp: 4, ep: 4 }];

const elementwiseNamed = (ops: Op[], name: string): ElementwiseOp => {
  const op = ops.find((o) => o.name === name);
  if (op === undefined || op.kind !== 'elementwise') throw new Error(`no elementwise ${name}`);
  return op;
};
const shape = (g: GemmOp) => [g.batch, g.m, g.n, g.k];

describe('latent MoE (latentDim)', () => {
  it('without latentDim the ops and bytes are unchanged; with l = h only the two latent GEMMs are added', () => {
    const same: MoeBlock = { ...MOE, latentDim: MOE.hidden };
    for (const p of PARALLELS) {
      for (const seqs of [1, 64]) {
        const c = ctx(seqs, p);
        expect(lowerMoe({ ...MOE, latentDim: undefined }, c)).toEqual(lowerMoe(MOE, c));
        // Every other op already uses l: with l = h they are the plain block's ops exactly.
        const ops = lowerMoe(same, c);
        expect(ops.filter((o) => !o.name.startsWith('moe.latent_'))).toEqual(lowerMoe(MOE, c));
        expect(ops.filter((o) => o.name.startsWith('moe.latent_')).map((o) => o.name)).toEqual(['moe.latent_in', 'moe.latent_out']);
        expect(moeResidentWeightBytes(same, c) - moeResidentWeightBytes(MOE, c)).toBe(moeLatentWeightBytes(same, c));
        expect(moeLocalExpertBytes(same, c)).toBe(moeLocalExpertBytes(MOE, c));
      }
    }
    expect(moeLatentWeightBytes(MOE, ctx(1, {}))).toBe(0);
  });

  it('mode A tp 4: latent GEMMs replicated (unsharded), experts at l with fe/4, one all-reduce of T·h', () => {
    const ops = lowerMoe(LAT, ctx(1, { tp: 4 }));
    expect(ops.map((o) => o.name)).toEqual([
      'moe.router',
      'moe.router_topk',
      'moe.latent_in',
      'moe.gate_up',
      'moe.act',
      'moe.down',
      'moe.combine',
      'moe.latent_out',
      'moe.shared.gate_up',
      'moe.shared.act',
      'moe.shared.down',
      'moe.tp_all_reduce',
    ]);
    expect(shape(gemmNamed(ops, 'moe.router'))).toEqual([1, 1, 8, 64]); // router reads h
    expect(shape(gemmNamed(ops, 'moe.latent_in'))).toEqual([1, 1, 32, 64]); // T × h → l, not /tp
    expect(shape(gemmNamed(ops, 'moe.gate_up'))).toEqual([2, 1, 16, 32]); // a 2; n = 2·32/4, k = l
    expect(shape(gemmNamed(ops, 'moe.down'))).toEqual([2, 1, 32, 8]); // n = l, k = 32/4
    expect(shape(gemmNamed(ops, 'moe.latent_out'))).toEqual([1, 1, 64, 32]); // on the partial sums, not /tp
    expect(gemmNamed(ops, 'moe.shared.gate_up').n).toBe(32); // shared at h, sharded: 2·64/4
    expect(gemmNamed(ops, 'moe.shared.gate_up').k).toBe(64);
    const combine = elementwiseNamed(ops, 'moe.combine');
    // elems a·mE·l = 2·1·32; reads 64·2 B; writes T·l·reduceBpe = 1·32·2 B
    expect([combine.elems, combine.activationReadBytes, combine.activationWriteBytes]).toEqual([64, 128, 64]);
    const coll = collectives(ops);
    expect(coll.map((c) => [c.collective, c.messageBytes])).toEqual([['all_reduce', 128]]); // T 1 · h 64 · 2 B
  });

  it('mode B: dispatch and combine all-to-alls carry l, latent GEMMs on the Ts slice', () => {
    const b1 = lowerMoe(LAT, ctx(1, { dp: 8, ep: 8 }));
    expect(b1.map((o) => o.name).slice(0, 9)).toEqual([
      'moe.router',
      'moe.router_topk',
      'moe.latent_in',
      'moe.dispatch',
      'moe.gate_up',
      'moe.act',
      'moe.down',
      'moe.combine_a2a',
      'moe.combine',
    ]);
    expect(b1[9].name).toBe('moe.latent_out');
    expect(collectives(b1).map((c) => c.messageBytes)).toEqual([128, 128]); // Ts 1 · k 2 · l 32 · 2 B (h: 256)
    expect(shape(gemmNamed(b1, 'moe.gate_up'))).toEqual([1, 2, 64, 32]); // a 1, mE 2, n 2·32, k l
    const combine = elementwiseNamed(b1, 'moe.combine');
    expect([combine.elems, combine.activationWriteBytes]).toEqual([64, 64]); // Ts·k·l; Ts·l·2 B

    // tp 2 dp 4 ep 4, T 4: Ts = 2; dispatch 2·2·32·2 = 256; the closing all-gather stays at h: 4·64·2 = 512
    const b2 = lowerMoe(LAT, ctx(4, { tp: 2, dp: 4, ep: 4 }));
    expect(collectives(b2).map((c) => [c.collective, c.messageBytes])).toEqual([
      ['all_to_all', 256],
      ['all_to_all', 256],
      ['all_gather', 512],
    ]);
    expect(shape(gemmNamed(b2, 'moe.latent_in'))).toEqual([1, 2, 32, 64]);
    expect(shape(gemmNamed(b2, 'moe.latent_out'))).toEqual([1, 2, 64, 32]);
  });

  it("latent projections use the 'moe_latent' format, experts keep theirs", () => {
    const c = ctx(1, {});
    c.quant = { ...c.quant, weights: { dtype: 'nvfp4' }, modules: { moe_latent: { weights: { dtype: 'bf16' } } } };
    const ops = lowerMoe(LAT, c);
    expect(gemmNamed(ops, 'moe.latent_in').weight.dtype).toBe('bf16');
    expect(gemmNamed(ops, 'moe.latent_out').weight.dtype).toBe('bf16');
    expect(gemmNamed(ops, 'moe.gate_up').weight.dtype).toBe('nvfp4');
  });

  it('resident weights: experts at l, latent projections replicated', () => {
    // experts 8·3·32·32·2 = 49,152 + router 8·64·2 = 1024 + latent 2·64·32·2 = 8192 + shared 3·64·64·2 = 24,576
    expect(moeResidentWeightBytes(LAT, ctx(1, {}))).toBe(82_944);
    // tp 4 (mode A, etp 4): experts 8·3·32·8·2 = 12,288 + 1024 + latent 8192 (not /4) + shared 3·64·16·2 = 6144
    expect(moeResidentWeightBytes(LAT, ctx(1, { tp: 4 }))).toBe(27_648);
    // mode B ep 8: 1 expert 3·32·32·2 = 6144 + 1024 + 8192 + shared (unsharded) 24,576
    expect(moeResidentWeightBytes(LAT, ctx(1, { dp: 8, ep: 8 }))).toBe(39_936);
    expect(moeLatentWeightBytes(LAT, ctx(1, { tp: 4 }))).toBe(8192);
    expect(moeLocalExpertBytes(LAT, ctx(1, {}))).toBe(49_152);
  });

  it('offloaded experts: the fetch moves experts at l; the latent projections stay resident', () => {
    const offload: OffloadLowering = {
      prefetchAccuracy: 0,
      routedExperts: { tier: 'host_ram', layers: { from: 0, to: 4 }, residentFraction: 0.5, hitRatio: 0.5, includeDraft: false },
    };
    const c: LoweringContext = { ...ctx(1, {}), offload };
    // W_expert = 3·32·32·2 = 6144 (at h it would be 12,288)
    expect(expertWeightBytes(LAT, c)).toBe(6144);
    const fetch = lowerMoe(LAT, c, 0)[0] as TransferOp;
    expect(fetch.name).toBe('moe.expert_fetch');
    // U = 8·(1 − (1 − 2/8)^1) = 2; C = 4; hits = clamp(2·0.5, 0, 2) = 1; M = 1
    expect(fetch.reads).toEqual([{ count: 1, bytesEach: 6144 }]);
    expect(fetch.usefulBytes).toBe(6144);
    // resident: C·W_expert = 4·6144 = 24,576 + router 1024 + latent 8192 + shared 24,576
    expect(moeResidentWeightBytes(LAT, c, 0)).toBe(58_368);
    // the tier holds the full local set at l
    expect(moeLocalExpertBytes(LAT, c)).toBe(49_152);
  });
});

describe('Nemotron-H MoE (Super: h 4096, E 512, k 22, fe 2688, l 1024, S 1, fs 5376, ungated relu²)', () => {
  const SUPER: MoeBlock = {
    kind: 'moe',
    name: 'moe',
    hidden: 4096,
    experts: 512,
    topK: 22,
    expertIntermediate: 2688,
    gated: false,
    activation: 'relu2',
    sharedExperts: 1,
    sharedIntermediate: 5376,
    latentDim: 1024,
  };
  const act = (ops: Op[], name: string): ElementwiseOp => {
    const op = ops.find((o) => o.name === name);
    if (op === undefined || op.kind !== 'elementwise') throw new Error(`no elementwise ${name}`);
    return op;
  };

  it('decode T 1, tp 1: op order and shapes; relu2 act is 2 FLOP/elem and reads one input (ungated)', () => {
    const ops = lowerMoe(SUPER, ctx(1, {}));
    expect(ops.filter((o) => o.kind === 'gemm').map((o) => o.name)).toEqual([
      'moe.router',
      'moe.latent_in',
      'moe.gate_up',
      'moe.down',
      'moe.latent_out',
      'moe.shared.gate_up',
      'moe.shared.down',
    ]);
    const shape = (name: string): number[] => {
      const g = gemmNamed(ops, name);
      return [g.batch ?? 1, g.m, g.n, g.k];
    };
    expect(shape('moe.latent_in')).toEqual([1, 1, 1024, 4096]);
    expect(shape('moe.gate_up')).toEqual([22, 1, 2688, 1024]); // ungated: n = fe
    expect(shape('moe.down')).toEqual([22, 1, 1024, 2688]);
    expect(shape('moe.latent_out')).toEqual([1, 1, 4096, 1024]);
    expect(shape('moe.shared.gate_up')).toEqual([1, 1, 5376, 4096]);
    const a = act(ops, 'moe.act');
    expect(a.fn).toBe('relu2');
    expect(a.elems).toBe(59_136); // 22·2688
    expect(opFlops(a)).toBe(118_272); // 2 FLOP/elem
    expect(a.activationReadBytes).toBe(118_272); // 1·59,136·2 B
    expect(act(ops, 'moe.shared.act').fn).toBe('relu2');
  });

  it('resident bytes at tp 1, bf16 = params·2 B (latent projections included)', () => {
    // 5,637,144,576 experts + 4,194,304 router + 16,777,216 latent + 88,080,384 shared = 2·2,873,098,240
    expect(moeResidentWeightBytes(SUPER, ctx(1, {}))).toBe(5_746_196_480);
  });

  it('a gated relu2 has no defined fused op and throws', () => {
    expect(() => lowerMoe({ ...SUPER, gated: true }, ctx(1, {}))).toThrow(/relu2/);
  });
});
