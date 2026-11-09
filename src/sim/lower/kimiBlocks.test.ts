// Lowering of the Kimi K3 additions on TINY_KIMI (made-up dims, see models/reference.ts): the full-rank KDA output
// gate, the MLA sigmoid output gate, the latent MoE (modes A and B) and the Attention-Residuals mixes, with
// hand-derived params, shapes and bytes. bf16 everywhere (2 B/element).

import { describe, expect, it } from 'vitest';
import { opBytes, opFlops } from '../ir/metrics';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import type { CollectiveOp, ElementwiseOp, GemmOp, Op } from '../ir/ops';
import { activeBlockParams, blockParams } from '../ir/params';
import { modelSpecErrors } from '../ir/validateModel';
import { TINY_HC_NGRAM, TINY_KDA, TINY_KIMI } from '../models/reference';
import {
  blockResidentWeightBytes,
  lowerModel,
  ppBoundaryResidualVectors,
  type LoweringContext,
} from './lower';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

/** Decode: 2 sequences at a 256-token context (T = 2). */
function ctx(parallel: Partial<ParallelConfig> = {}): LoweringContext {
  return {
    phase: 'decode',
    seqs: 2,
    qLen: 1,
    kvLen: 256,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
  };
}

function layerOps(model: ModelSpec, i: number, c: LoweringContext): Op[] {
  const layer = lowerModel(model, c)
    .stages.flatMap((s) => s.layers)
    .find((l) => l.layerIndex === i);
  if (layer === undefined) throw new Error(`no layer ${i}`);
  return layer.ops;
}
const byName = <T extends Op>(ops: Op[], name: string): T => {
  const op = ops.find((o) => o.name === name);
  if (op === undefined) throw new Error(`no op ${name} in ${ops.map((o) => o.name).join(', ')}`);
  return op as T;
};
const names = (ops: Op[]): string[] => ops.map((o) => o.name);
const block = (i: number, kind: string) => {
  const b = TINY_KIMI.layers[i].blocks.find((x) => x.kind === kind);
  if (b === undefined) throw new Error(`no ${kind} in layer ${i}`);
  return b;
};

describe('TINY_KIMI is valid and lays out the Attention-Residuals mixes', () => {
  it('has no structural errors', () => expect(modelSpecErrors(TINY_KIMI)).toEqual([]));

  it('layer 0 mixes only before its ffn (2 candidates); layer 1 before both (2, then 3); the final mix has 3', () => {
    expect(TINY_KIMI.layers[0].blocks.map((b) => b.kind)).toEqual([
      'norm',
      'linear_attention',
      'residual',
      'residual_mix',
      'norm',
      'mlp',
      'residual',
    ]);
    expect(TINY_KIMI.layers[1].blocks.map((b) => b.kind)).toEqual([
      'residual_mix',
      'norm',
      'mla_attention',
      'residual',
      'residual_mix',
      'norm',
      'moe',
      'residual',
      'residual_mix',
    ]);
    const mixes = TINY_KIMI.layers.flatMap((l) =>
      l.blocks.filter((b) => b.kind === 'residual_mix'),
    );
    expect(mixes.map((m) => [m.name, m.candidates])).toEqual([
      ['ffn_residual_mix', 2],
      ['mixer_residual_mix', 2],
      ['ffn_residual_mix', 3],
      ['final_residual_mix', 3],
    ]);
  });

  it('rejects fullRankOutputGate on a non-KDA variant', () => {
    const bad = structuredClone(TINY_KIMI);
    const lin = bad.layers[0].blocks[1];
    if (lin.kind !== 'linear_attention') throw new Error('shape');
    lin.variant = 'gated_delta_net';
    delete lin.gateRank;
    expect(modelSpecErrors(bad)).toEqual([
      "layer 0 linear_attn: fullRankOutputGate is only for variant 'kda'",
    ]);
  });
});

describe('params by hand (h 128)', () => {
  it('KDA with a full-rank output gate: 45,154 (the low-rank TINY_KDA gate would be 3,072 instead of 8,192)', () => {
    // qkv 128·192 + β 128·2 + forget 128·16 + 16·64 + gate 128·64 + conv 192·4 + A_log 2 + dt_bias 64 + o_norm 32 + o 64·128
    expect(blockParams(block(0, 'linear_attention'))).toBe(
      24_576 + 256 + 3_072 + 8_192 + 768 + 2 + 64 + 32 + 8_192,
    );
    expect(blockParams(TINY_KDA.layers[0].blocks[1])).toBe(
      24_576 + 256 + 3_072 + 3_072 + 768 + 2 + 64 + 32 + 8_192,
    );
  });

  it('MLA with a sigmoid output gate adds h·H·dv = 8,192', () => {
    // q 128·32 + 32 + 32·4·24 = 7,200; kv_a 128·40 + 32 = 5,152; kv_b 32·4·32 = 4,096; o 64·128 = 8,192; gate 8,192
    expect(blockParams(block(1, 'mla_attention'))).toBe(7_200 + 5_152 + 4_096 + 8_192 + 8_192);
  });

  it('latent MoE: experts at L = 64, plus 2·h·L + L for the projections and norm', () => {
    // routed 8·3·64·64 + router 8·128 + shared 3·128·128 + latent 2·128·64 + 64
    expect(blockParams(block(1, 'moe'))).toBe(98_304 + 1_024 + 49_152 + 16_448);
    expect(activeBlockParams(block(1, 'moe'))).toBe(2 * 3 * 64 * 64 + 1_024 + 49_152 + 16_448);
  });

  it('a residual mix has 2·dim params', () =>
    expect(blockParams(block(0, 'residual_mix'))).toBe(256));

  it('resident weight bytes equal params · 2 B at tp 1 (KDA A_log and dt_bias excepted, §9)', () => {
    for (const layer of TINY_KIMI.layers) {
      for (const b of layer.blocks) {
        const kdaVectors =
          b.kind === 'linear_attention' && b.variant === 'kda' ? b.vHeads * (1 + b.keyDim) : 0;
        expect(blockResidentWeightBytes(b, ctx()), b.name).toBe(2 * (blockParams(b) - kdaVectors));
      }
    }
  });
});

describe('layer 0: KDA full-rank gate, situ ffn, ffn residual mix (decode, T = 2)', () => {
  const ops = layerOps(TINY_KIMI, 0, ctx());

  it('emits one lin.g GEMM instead of lin.g_a → lin.g_b', () => {
    expect(names(ops)).toEqual([
      'input_norm',
      'lin.qkv',
      'lin.f_a',
      'lin.f_b',
      'lin.g',
      'lin.conv1d',
      'lin.recurrent',
      'lin.out_norm',
      'lin.out_proj',
      'mixer_residual',
      'ffn_residual_mix',
      'ffn_norm',
      'mlp.gate_up',
      'mlp.act',
      'mlp.down',
      'ffn_residual',
    ]);
    const g = byName<GemmOp>(ops, 'lin.g');
    expect([g.m, g.n, g.k]).toEqual([2, 64, 128]);
    expect(opBytes(g).weightBytes).toBe(64 * 128 * 2);
    // tp 2: the gate is sharded with the value heads (vL = 1 → n = 32), still read from the full h.
    const g2 = byName<GemmOp>(layerOps(TINY_KIMI, 0, ctx({ tp: 2 })), 'lin.g');
    expect([g2.n, g2.k]).toEqual([32, 128]);
  });

  it('the situ activation is a situ_mul of 15 FLOP per element', () => {
    const act = byName<ElementwiseOp>(ops, 'mlp.act');
    expect(act.fn).toBe('situ_mul');
    expect(opFlops(act)).toBe(2 * 256 * 15);
  });

  it('residual_mix over T·h with C = 2: reads 2 candidates, 12 FLOP per element, 2·h weights', () => {
    const mix = byName<ElementwiseOp>(ops, 'ffn_residual_mix');
    expect(mix.fn).toBe('residual_mix');
    expect(mix.elems).toBe(2 * 128);
    expect(opFlops(mix)).toBe(2 * 128 * 6 * 2);
    expect(opBytes(mix)).toMatchObject({
      activationReadBytes: 2 * 256 * 2,
      activationWriteBytes: 256 * 2,
      weightBytes: 2 * 128 * 2,
    });
  });
});

describe('layer 1: gated MLA and latent MoE (decode, T = 2)', () => {
  it('mode A, tp 1: ops in order, with g_proj, output_gate, latent_in/norm/out and no latent all-reduce', () => {
    const ops = layerOps(TINY_KIMI, 1, ctx());
    expect(names(ops)).toEqual([
      'mixer_residual_mix',
      'input_norm',
      'mla.q_a',
      'mla.q_a_norm',
      'mla.q_b',
      'mla.g_proj',
      'mla.kv_a',
      'mla.kv_a_norm',
      'mla.rope',
      'mla.q_absorb',
      'attn.core',
      'mla.v_absorb',
      'attn.output_gate',
      'mla.o_proj',
      'mixer_residual',
      'ffn_residual_mix',
      'ffn_norm',
      'moe.router',
      'moe.router_topk',
      'moe.latent_in',
      'moe.gate_up',
      'moe.act',
      'moe.down',
      'moe.combine',
      'moe.latent_norm',
      'moe.latent_out',
      'moe.shared.gate_up',
      'moe.shared.act',
      'moe.shared.down',
      'ffn_residual',
      'final_residual_mix',
    ]);
    const g = byName<GemmOp>(ops, 'mla.g_proj');
    expect([g.m, g.n, g.k]).toEqual([2, 4 * 16, 128]);
    const gate = byName<ElementwiseOp>(ops, 'attn.output_gate');
    expect(gate.fn).toBe('sigmoid_mul');
    expect(gate.elems).toBe(2 * 4 * 16);
    expect(opBytes(gate)).toMatchObject({
      activationReadBytes: 2 * 128 * 2,
      activationWriteBytes: 128 * 2,
    });

    // Experts read and write the latent width: gate_up k = L, down n = L; grid a = min(8, T·k = 4) experts of 1 token.
    // Weights are read for U(2) = 8·(1 − 0.75²) = 3.5 expected distinct experts (E 8, k 2, T 2), not 4.
    const down = byName<GemmOp>(ops, 'moe.latent_in');
    expect([down.m, down.n, down.k]).toEqual([2, 64, 128]);
    expect(byName<GemmOp>(ops, 'moe.gate_up')).toMatchObject({
      batch: 4,
      distinctWeightMatrices: 3.5,
      m: 1,
      n: 128,
      k: 64,
    });
    expect(byName<GemmOp>(ops, 'moe.down')).toMatchObject({
      batch: 4,
      distinctWeightMatrices: 3.5,
      m: 1,
      n: 64,
      k: 64,
    });
    expect(byName<ElementwiseOp>(ops, 'moe.combine').elems).toBe(4 * 1 * 64);
    expect(byName<ElementwiseOp>(ops, 'moe.latent_norm')).toMatchObject({
      fn: 'rmsnorm',
      elems: 2 * 64,
      weightBytes: 64 * 2,
    });
    const up = byName<GemmOp>(ops, 'moe.latent_out');
    expect([up.m, up.n, up.k]).toEqual([2, 128, 64]);
    // The shared expert and the router still read h.
    expect(byName<GemmOp>(ops, 'moe.shared.gate_up').k).toBe(128);
    expect(byName<GemmOp>(ops, 'moe.router').k).toBe(128);
  });

  it('mode A, tp 2: latent_in replicated, one all-reduce of T·L before the norm, latent_out row-parallel', () => {
    const ops = layerOps(TINY_KIMI, 1, ctx({ tp: 2 }));
    expect(byName<GemmOp>(ops, 'moe.latent_in')).toMatchObject({ m: 2, n: 64, k: 128 });
    const ar = byName<CollectiveOp>(ops, 'moe.latent_all_reduce');
    expect(ar).toMatchObject({
      collective: 'all_reduce',
      group: 'tp',
      groupSize: 2,
      messageBytes: 2 * 64 * 2,
    });
    expect(names(ops).indexOf('moe.latent_all_reduce')).toBe(
      names(ops).indexOf('moe.latent_norm') - 1,
    );
    expect(byName<GemmOp>(ops, 'moe.latent_out')).toMatchObject({ m: 2, n: 128, k: 32 });
    expect(names(ops).at(-2)).toBe('ffn_residual');
    expect(names(ops).filter((n) => n === 'moe.tp_all_reduce')).toHaveLength(1);
  });

  it('mode B (ep 2 over dp 2): the a2a messages are L wide; no latent all-reduce', () => {
    const ops = layerOps(TINY_KIMI, 1, ctx({ ep: 2, dp: 2 }));
    // Formula: dispatch S = Ts·k·L·bpe(act) = 2·2·64·2 (it would be 2·2·128·2 on h)
    expect(byName<CollectiveOp>(ops, 'moe.dispatch').messageBytes).toBe(2 * 2 * 64 * 2);
    expect(byName<CollectiveOp>(ops, 'moe.combine_a2a').messageBytes).toBe(2 * 2 * 64 * 2);
    expect(names(ops)).not.toContain('moe.latent_all_reduce');
    expect(byName<GemmOp>(ops, 'moe.latent_out')).toMatchObject({ m: 2, n: 128, k: 64 });
  });
});

describe('defaults off: a model without the new fields lowers as before', () => {
  it('TINY_KDA has no lin.g, no g_proj, no output gate, no latent ops and no residual mixes', () => {
    const all = [0, 1].flatMap((i) => names(layerOps(TINY_KDA, i, ctx())));
    for (const n of [
      'lin.g',
      'mla.g_proj',
      'attn.output_gate',
      'moe.latent_in',
      'moe.latent_out',
      'moe.latent_norm',
    ]) {
      expect(all).not.toContain(n);
    }
    expect(all.filter((n) => n.endsWith('residual_mix'))).toEqual([]);
    expect(all).toContain('lin.g_a');
  });
});

describe('PP send at pp 2 carries the Attention-Residuals summaries (h 128, bf16)', () => {
  const prefill = (): LoweringContext => ({
    ...ctx({ pp: 2 }),
    phase: 'prefill',
    seqs: 1,
    qLen: 16,
    kvLen: 16,
  });
  const sendBytes = (model: ModelSpec, c: LoweringContext) =>
    lowerModel(model, c).stages[0].sendToNext?.messageBytes;

  // TINY_KIMI (B 1): the boundary sits before layer 1, which stores ceil(1/1) = 1 summary, so N_live = 1 + 1 = 2.
  // Formula: S = T·N_live·h·bpe = T·2·128·2 B
  it("TINY_KIMI: N_live 2, the candidates of layer 1's pre-mixer mix", () => {
    expect(ppBoundaryResidualVectors(TINY_KIMI, 1)).toBe(2);
    expect(block(1, 'residual_mix')).toMatchObject({ name: 'mixer_residual_mix', candidates: 2 });
  });
  it('TINY_KIMI decode (T 2): 2·2·128·2 = 1,024 B', () =>
    expect(sendBytes(TINY_KIMI, ctx({ pp: 2 }))).toBe(1024));
  it('TINY_KIMI prefill (T 16): 16·2·128·2 = 8,192 B', () =>
    expect(sendBytes(TINY_KIMI, prefill())).toBe(8192));

  it('unchanged without Attention Residuals: TINY_KDA sends T·h, TINY_HC_NGRAM T·N·h (N 2)', () => {
    expect(ppBoundaryResidualVectors(TINY_KDA, 1)).toBe(1);
    expect(sendBytes(TINY_KDA, ctx({ pp: 2 }))).toBe(2 * 128 * 2);
    expect(sendBytes(TINY_KDA, prefill())).toBe(16 * 128 * 2);
    expect(ppBoundaryResidualVectors(TINY_HC_NGRAM, 1)).toBe(2);
    expect(sendBytes(TINY_HC_NGRAM, ctx({ pp: 2 }))).toBe(2 * 2 * 64 * 2);
    expect(sendBytes(TINY_HC_NGRAM, prefill())).toBe(16 * 2 * 64 * 2);
  });
});
