/** Dense MLP lowering: default TP sharding and the dense FFN over the EP group (docs §5 "DP attention + EP"). */
import { describe, expect, it } from 'vitest';
import type { MlpBlock } from '../ir/model';
import { denseOverEp, denseTpGroupSize } from './common';
import type { LoweringContext } from './lower';
import { denseGroupWidthError, lowerMlp } from './mlp';

const block: MlpBlock = {
  kind: 'mlp',
  name: 'mlp',
  hidden: 1024,
  intermediate: 3072,
  gated: true,
  activation: 'silu',
  bias: false,
};

const ctx: LoweringContext = {
  phase: 'decode',
  seqs: 4,
  qLen: 1,
  kvLen: 100,
  quant: { weights: { dtype: 'bf16' }, activations: 'fp8_e4m3', kvCache: { dtype: 'bf16' } },
  parallel: { tp: 1, pp: 1, dp: 8, ep: 8, denseTpGroup: 'ep' },
};

describe('denseTpGroupSize / denseOverEp', () => {
  it("G = ep with 'ep' and ep > 1, else tp", () => {
    expect(denseTpGroupSize(ctx)).toBe(8);
    expect(denseOverEp(ctx)).toBe(true);
    const absent: LoweringContext = { ...ctx, parallel: { tp: 2, pp: 1, dp: 4, ep: 8 } };
    expect(denseTpGroupSize(absent)).toBe(2);
    expect(denseOverEp(absent)).toBe(false);
    const ep1: LoweringContext = {
      ...ctx,
      parallel: { tp: 1, pp: 1, dp: 8, ep: 1, denseTpGroup: 'ep' },
    };
    expect(denseTpGroupSize(ep1)).toBe(1);
    expect(denseOverEp(ep1)).toBe(false);
  });
});

describe("lowerMlp over the EP group (denseTpGroup 'ep', tp 1)", () => {
  const G = 8;
  const T = 4; // seqs·qLen
  const h = 1024;
  const f = 3072;
  const ops = lowerMlp(block, ctx);

  it('op sequence: gather → gate_up → act → down → reduce-scatter (no TP all-reduce)', () => {
    expect(ops.map((o) => o.name)).toEqual([
      'mlp.dense_all_gather',
      'mlp.gate_up',
      'mlp.act',
      'mlp.down',
      'mlp.dense_reduce_scatter',
    ]);
  });
  it('all_gather(ep): S = G·T·h·bpe(act) (full output; fp8 = 1 B)', () => {
    expect(ops[0]).toMatchObject({
      kind: 'collective',
      collective: 'all_gather',
      group: 'ep',
      groupSize: G,
      messageBytes: G * T * h * 1,
      dtype: 'fp8_e4m3',
    });
  });
  it('gate_up m = G·T, n = 2·f/G, k = h; act over G·T × f/G; down m = G·T, n = h, k = f/G', () => {
    expect(ops[1]).toMatchObject({ m: G * T, n: (2 * f) / G, k: h, outParts: [f / G, f / G] });
    expect(ops[2]).toMatchObject({ elems: G * T * (f / G) });
    expect(ops[3]).toMatchObject({ m: G * T, n: h, k: f / G });
  });
  it('reduce_scatter(ep): S = G·T·h·reduceBpe (full input; fp8 reduces in bf16 = 2 B)', () => {
    expect(ops[4]).toMatchObject({
      collective: 'reduce_scatter',
      group: 'ep',
      groupSize: G,
      messageBytes: G * T * h * 2,
      dtype: 'bf16',
    });
  });
  it('the gather and scatter together carry the bytes of one all-reduce of G·T·h (bf16 activations)', () => {
    const bf: LoweringContext = { ...ctx, quant: { ...ctx.quant, activations: 'bf16' } };
    const [gather, , , , scatter] = lowerMlp(block, bf);
    if (gather.kind !== 'collective' || scatter.kind !== 'collective')
      throw new Error('collectives expected');
    expect(gather.messageBytes).toBe(scatter.messageBytes);
    expect(gather.messageBytes).toBe(G * T * h * 2);
  });
  it('throws the stable width error when G does not divide f', () => {
    const odd: LoweringContext = { ...ctx, parallel: { ...ctx.parallel, dp: 9, ep: 9 } };
    expect(() => lowerMlp({ ...block, intermediate: 3072 }, odd)).toThrow(
      denseGroupWidthError(3072, 9),
    );
    expect(denseGroupWidthError(12288, 72)).toBe(
      'mlp intermediate (12288) must be divisible by the dense TP group (72)',
    );
  });
  it("'ep' at ep 1 equals the default lowering", () => {
    const ep1: LoweringContext = {
      ...ctx,
      parallel: { tp: 1, pp: 1, dp: 8, ep: 1, denseTpGroup: 'ep' },
    };
    const plain: LoweringContext = { ...ctx, parallel: { tp: 1, pp: 1, dp: 8, ep: 1 } };
    expect(lowerMlp(block, ep1)).toEqual(lowerMlp(block, plain));
  });
});
