/**
 * Two-batch overlap (docs/simulation-model.md §7 "Two-batch overlap"): the halves, the merge rule, the costs that
 * fall out of the per-op model (weights read twice, α twice, the activation peak halved), and the small- vs
 * large-batch crossover against the same run without it.
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_IB } from '../__fixtures__/synthetic';
import type { TopologySpec } from '../hw/types';
import type { QuantConfig } from '../ir/model';
import { lowerModel, type LoweringContext } from '../lower/lower';
import { REF_MOE_30B_A3B, TINY_DENSE } from '../models/reference';
import { stageActivationPeakBytes } from './memory';
import { simulate } from './simulate';
import { mergeTwoBatch, twoBatchHalves, TWO_BATCH_NOTHING_TO_SPLIT_WARNING } from './twoBatch';
import type { SimInput } from './types';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

function ctx(over: Partial<LoweringContext> = {}): LoweringContext {
  return {
    phase: 'prefill',
    seqs: 4,
    qLen: 256,
    kvLen: 256,
    quant: BF16,
    parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
    ...over,
  };
}

describe('twoBatchHalves', () => {
  it('even b: two halves of b/2 sequences, same qLen and kvLen', () => {
    const s = twoBatchHalves(ctx({ seqs: 4 }));
    expect(s?.tokenSplit).toBe(false);
    expect([s?.a.seqs, s?.a.qLen, s?.a.kvLen]).toEqual([2, 256, 256]);
    expect([s?.b.seqs, s?.b.qLen, s?.b.kvLen]).toEqual([2, 256, 256]);
  });
  it('odd b: ceil(b/2) and floor(b/2)', () => {
    const s = twoBatchHalves(ctx({ seqs: 5, phase: 'decode', qLen: 1, kvLen: 1025 }));
    expect([s?.a.seqs, s?.b.seqs]).toEqual([3, 2]);
    expect([s?.a.qLen, s?.a.kvLen, s?.b.qLen, s?.b.kvLen]).toEqual([1, 1025, 1, 1025]);
  });
  it('one-sequence prefill: causal token split, A = {ceil(q/2), kvLen − floor(q/2)}, B = {floor(q/2), kvLen}', () => {
    // q = 7 new tokens over kvLen = 10 (3 cached): A = 4 tokens over 7 keys, B = 3 tokens over all 10.
    const s = twoBatchHalves(ctx({ seqs: 1, qLen: 7, kvLen: 10 }));
    expect(s?.tokenSplit).toBe(true);
    expect([s?.a.seqs, s?.a.qLen, s?.a.kvLen]).toEqual([1, 4, 7]);
    expect([s?.b.seqs, s?.b.qLen, s?.b.kvLen]).toEqual([1, 3, 10]);
  });
  it('one-sequence decode, or a one-token prefill: nothing to split', () => {
    expect(twoBatchHalves(ctx({ seqs: 1, phase: 'decode', qLen: 1, kvLen: 1025 }))).toBeUndefined();
    expect(twoBatchHalves(ctx({ seqs: 1, phase: 'decode', qLen: 3, kvLen: 1027 }))).toBeUndefined();
    expect(twoBatchHalves(ctx({ seqs: 1, qLen: 1, kvLen: 1 }))).toBeUndefined();
  });
});

describe('mergeTwoBatch (TINY_DENSE, pp 2)', () => {
  it('sequence split: per layer ops_A ++ ops_B, the full layer list, the full batch sends', () => {
    const c = ctx({ seqs: 4 });
    const s = twoBatchHalves(c)!;
    const full = lowerModel(TINY_DENSE, c);
    const a = lowerModel(TINY_DENSE, s.a);
    const b = lowerModel(TINY_DENSE, s.b);
    const merged = mergeTwoBatch(full, a, b, false);
    for (const [i, stage] of merged.stages.entries()) {
      expect(stage.layers.map((l) => l.name)).toEqual(full.stages[i].layers.map((l) => l.name));
      for (const [j, layer] of stage.layers.entries())
        expect(layer.ops).toEqual([...a.stages[i].layers[j].ops, ...b.stages[i].layers[j].ops]);
      expect(stage.sendToNext).toBe(full.stages[i].sendToNext);
    }
    expect(merged.stages[0].sendToNext).toBeDefined();
  });
  it('token split: the head comes from half B alone (the last position), the other layers from both', () => {
    const c = ctx({ seqs: 1, qLen: 128, kvLen: 128 });
    const s = twoBatchHalves(c)!;
    const full = lowerModel(TINY_DENSE, c);
    const a = lowerModel(TINY_DENSE, s.a);
    const b = lowerModel(TINY_DENSE, s.b);
    const merged = mergeTwoBatch(full, a, b, true);
    const last = merged.stages.length - 1;
    const head = merged.stages[last].layers.find((l) => l.name === 'head');
    const headB = b.stages[last].layers.find((l) => l.name === 'head');
    expect(head?.ops).toEqual(headB?.ops);
    const fullHead = full.stages[last].layers.find((l) => l.name === 'head');
    // One sequence: the head of a half is the full batch's head (logits of one position).
    expect(head?.ops).toEqual(fullHead?.ops);
    const embed = merged.stages[0].layers[0];
    expect(embed.ops.length).toBe(2 * full.stages[0].layers[0].ops.length);
  });
  it('dense model, even b: the activation peak (max single op) of the merged step is exactly half', () => {
    const c = ctx({ seqs: 4 });
    const s = twoBatchHalves(c)!;
    const full = lowerModel(TINY_DENSE, c);
    const merged = mergeTwoBatch(
      full,
      lowerModel(TINY_DENSE, s.a),
      lowerModel(TINY_DENSE, s.b),
      false,
    );
    for (const st of [0, 1])
      expect(stageActivationPeakBytes([merged], st)).toBe(stageActivationPeakBytes([full], st) / 2);
  });
});

/** REF_MOE_30B_A3B, mode B (tp 1, ep 2, dp 2) on two devices of one SYNTH_CLUSTER node. */
function moeInput(over: Partial<SimInput> = {}): SimInput {
  return {
    model: REF_MOE_30B_A3B,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 2, ep: 2 },
    workload: { batch: 4, promptLen: 256, decodeContextLen: 1024 },
    hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
    options: { overlap: 'none' },
    ...over,
  };
}

describe('costs that fall out of the halves (REF_MOE_30B_A3B ep 2 dp 2, b_mb = 2, overlap none)', () => {
  const off = simulate(moeInput());
  const on = simulate(moeInput({ deployment: { twoBatchOverlap: true } }));

  it('decode at small batch: each half reads the attention weights again (weight bytes ×2)', () => {
    const weightBytes = (r: typeof off, name: string) =>
      r.decode.stages[0].layers[1].ops
        .filter((o) => o.op.name === name)
        .reduce((s, o) => s + o.bytes.weightBytes, 0);
    for (const name of ['attn.qkv_proj', 'attn.o_proj']) {
      expect(weightBytes(off, name)).toBeGreaterThan(0);
      expect(weightBytes(on, name)).toBe(2 * weightBytes(off, name));
    }
    // Memory-bound at 2 sequences: the GEMM time doubles too.
    const sec = (r: typeof off) =>
      r.decode.stages[0].layers[1].ops
        .filter((o) => o.op.name === 'attn.qkv_proj')
        .reduce((s, o) => s + o.sec, 0);
    // (just under 2×: the halves' activation bytes still add up to the full batch's)
    expect(sec(on) / sec(off)).toBeGreaterThan(1.99);
    expect(sec(on) / sec(off)).toBeLessThan(2);
  });

  it('α is paid once per half: the stage latency sum doubles', () => {
    for (const phase of ['prefill', 'decode'] as const)
      expect(on[phase].stages[0].commLatencySec).toBeCloseTo(
        2 * off[phase].stages[0].commLatencySec,
        15,
      );
  });

  it('the activation peak is the larger half’s: about half at even b_mb (MoE rows round up per half)', () => {
    const ratio =
      on.memory.perStage[0].activationPeakBytes / off.memory.perStage[0].activationPeakBytes;
    expect(ratio).toBeGreaterThanOrEqual(0.5);
    expect(ratio).toBeLessThan(0.52);
    expect(on.memory.perStage[0].weightBytes).toBe(off.memory.perStage[0].weightBytes);
  });

  it('the layer list is the full batch’s; no warning with b_mb ≥ 2', () => {
    expect(on.decode.stages[0].layers.map((l) => l.name)).toEqual(
      off.decode.stages[0].layers.map((l) => l.name),
    );
    expect(on.warnings).not.toContain(TWO_BATCH_NOTHING_TO_SPLIT_WARNING);
  });

  it('one sequence per microbatch: prefill splits its prompt, decode warns and runs unsplit', () => {
    const one = moeInput({ workload: { batch: 2, promptLen: 256, decodeContextLen: 1024 } });
    const r = simulate({ ...one, deployment: { twoBatchOverlap: true } });
    const base = simulate(one);
    expect(r.warnings).toContain(TWO_BATCH_NOTHING_TO_SPLIT_WARNING);
    // Decode: not split, and the EP default stays the global one (no other half to hide behind): exactly the run
    // without TBO.
    expect(r.decode).toEqual(base.decode);
    // An explicit EP fraction still wins in that unsplit decode.
    const explicit = { ...one, options: { overlap: 'none' as const, overlapByClass: { ep: 1 } } };
    expect(simulate({ ...explicit, deployment: { twoBatchOverlap: true } }).decode).toEqual(
      simulate(explicit).decode,
    );
    expect(simulate(explicit).decode.tokensPerSec).toBeGreaterThan(base.decode.tokensPerSec);
    // Prefill: two halves of 128 tokens; the head is lowered once.
    const head = (x: typeof r) => x.prefill.stages[0].layers.find((l) => l.name === 'head');
    expect(head(r)?.ops.length).toBe(head(base)?.ops.length);
    expect(r.prefill.stages[0].layers[1].ops.length).toBe(
      2 * base.prefill.stages[0].layers[1].ops.length,
    );
  });
});

/**
 * Physics check: REF_MOE_30B_A3B, mode B with ep 16 = dp 16 over two nodes (the EP all-to-alls cross the inter-node
 * link), on a SYNTH_CLUSTER whose inter-node link is a quarter of SYNTH_IB (12.5 GB/s per direction), global
 * `overlap 'none'` (the engine's real starting point), decode @ 1024. Tokens per device = B_r (tp 1).
 * Decode tok/s, TBO / no TBO (computed on this engine):
 *   B_r = 4:   0.585   (halves of 2 sequences: weights read twice, nothing to gain from hiding a small all-to-all)
 *   B_r = 256: 1.173   (the dispatch/combine β hides behind the other half's compute)
 */
describe('physics: TBO loses at small batch and wins at large batch (EP over a slow inter-node link)', () => {
  const topology: TopologySpec = {
    ...SYNTH_CLUSTER,
    interNode: { ...SYNTH_IB, bandwidthPerDirectionBytesPerSec: 12.5e9 },
  };
  const ratio = (perReplica: number) => {
    const base: SimInput = {
      model: REF_MOE_30B_A3B,
      quant: BF16,
      parallel: { tp: 1, pp: 1, dp: 16, ep: 16 },
      workload: { batch: 16 * perReplica, promptLen: 1024, decodeContextLen: 1024 },
      hardware: { device: SYNTH_GPU, topology },
      options: { overlap: 'none' },
    };
    const off = simulate(base);
    const on = simulate({ ...base, deployment: { twoBatchOverlap: true } });
    return {
      decode: on.decode.tokensPerSec / off.decode.tokensPerSec,
      prefill: on.prefill.tokensPerSec / off.prefill.tokensPerSec,
    };
  };

  it('B_r = 4: decode tok/s falls (≈ 0.585×)', () => {
    const r = ratio(4);
    expect(r.decode).toBeLessThan(1);
    expect(r.decode).toBeCloseTo(0.585, 2);
  });

  it('B_r = 256: decode tok/s rises (≈ 1.173×); prefill rises too', () => {
    const r = ratio(256);
    expect(r.decode).toBeGreaterThan(1);
    expect(r.decode).toBeCloseTo(1.173, 2);
    expect(r.prefill).toBeGreaterThan(1);
  });

  it('under the default overlap ideal there is nothing left to hide: TBO never helps decode', () => {
    for (const perReplica of [4, 256]) {
      const base: SimInput = {
        model: REF_MOE_30B_A3B,
        quant: BF16,
        parallel: { tp: 1, pp: 1, dp: 16, ep: 16 },
        workload: { batch: 16 * perReplica, promptLen: 1024, decodeContextLen: 1024 },
        hardware: { device: SYNTH_GPU, topology },
      };
      const off = simulate(base);
      const on = simulate({ ...base, deployment: { twoBatchOverlap: true } });
      expect(on.decode.tokensPerSec).toBeLessThanOrEqual(off.decode.tokensPerSec);
    }
  });
});
