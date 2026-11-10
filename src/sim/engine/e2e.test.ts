import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import { opBytes, opFlops } from '../ir/metrics';
import type { ParallelConfig, QuantConfig } from '../ir/model';
import type { Op } from '../ir/ops';
import { countParams } from '../ir/params';
import { lowerModel } from '../lower/lower';
import {
  REF_DENSE_8B,
  TINY_CSA,
  TINY_HC_NGRAM,
  TINY_HYBRID,
  TINY_INKLING,
  TINY_KDA,
  TINY_KIMI,
  TINY_MLA_SHARED,
  TINY_MLA_SPARSE,
} from '../models/reference';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { DeviceModifier } from '../hw/modifiers';
import type { HardwareStack } from '../hw/types';
import { DEFAULT_SIM_OPTIONS, simulate } from './simulate';
import type { OpResult, PhaseResult, SimInput, SimOptions, Workload } from './types';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** Asserts lo·ref ≤ actual ≤ hi·ref. */
function expectWithin(actual: number, ref: number, lo: number, hi: number): void {
  expect(actual).toBeGreaterThanOrEqual(lo * ref);
  expect(actual).toBeLessThanOrEqual(hi * ref);
}

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const P1: ParallelConfig = { tp: 1, pp: 1, dp: 1, ep: 1 };
const ON_5090: HardwareStack = { device: RTX_5090, topology: singleDevice(RTX_5090) };
const ON_SYNTH_1: HardwareStack = { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU) };
const ON_SYNTH_CLUSTER: HardwareStack = { device: SYNTH_GPU, topology: SYNTH_CLUSTER };

/** The T9 workload: batch 1, prompt 4096, decode at a 1024-token context. */
const T9_WORKLOAD: Workload = { batch: 1, promptLen: 4096, decodeContextLen: 1024 };

function input(over: Partial<SimInput> = {}): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: P1,
    workload: T9_WORKLOAD,
    hardware: ON_5090,
    ...over,
  };
}

const phaseOps = (p: PhaseResult): OpResult[] =>
  p.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));

/** The bf16 / fp32-accumulate dense peak of the RTX 5090 preset (whitepaper Table 3: 209.5 TFLOP/s). */
function rtx5090Bf16Peak(): number {
  const p = RTX_5090.engines[0].matmulPeaks.find(
    (x) => x.input === 'bf16' && x.accumulate === 'fp32' && x.sparsity === 'dense',
  );
  if (p === undefined) throw new Error('preset lacks bf16/fp32/dense');
  return p.flopsPerSec;
}

const isTensor = (op: Op): boolean => op.kind === 'gemm' || op.kind === 'attention';

describe('T9 REF_DENSE_8B on 1×RTX 5090, bf16 (fp32 acc), batch 1', () => {
  const memBW = RTX_5090.memory.bandwidthBytesPerSec; // 1.792e12 B/s (whitepaper: 512-bit × 28 Gbps)

  it('decode @ctx 1024 is within 3% below the weights + KV bandwidth bound (117.2 tok/s)', () => {
    const r = simulate(input());
    // Decode @ 1024: 1024 tokens already cached + the new token, so attention reads kvLen = 1025 keys.
    // Bytes that must stream per token: GEMM weights (36 layers × 192,937,984 linear params + LM head
    // 622,329,856, × 2 B) + KV cache 36 layers × 1025 tokens × (2·8·128 elems × 2 B = 4096 B):
    // 2·(36·192,937,984 + 622,329,856) + 36·1025·4096 = 15,136,194,560 + 151,142,400 = 15,287,336,960 B.
    const streamedBytes = 15_287_336_960;
    const decodeOps = lowerModel(REF_DENSE_8B, {
      phase: 'decode',
      seqs: 1,
      qLen: 1,
      kvLen: 1025,
      quant: BF16,
      parallel: P1,
    }).stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
    const gemmWeightAndKv = decodeOps.reduce(
      (s, op) => s + (op.kind === 'gemm' ? opBytes(op).weightBytes : 0) + opBytes(op).kvReadBytes,
      0,
    );
    expect(gemmWeightAndKv).toBe(streamedBytes); // the hand derivation matches the IR
    const boundTokensPerSec = memBW / streamedBytes; // 1.792e12 / 15.287e9 = 117.22 tok/s
    expectWithin(r.decode.tokensPerSec, boundTokensPerSec, 0.97, 1.0);
    // Naive "all weights once per token": 1.792e12 / (2·8,190,735,360 = 16.38e9 B) = 109.4 tok/s.
    const naive = memBW / (2 * countParams(REF_DENSE_8B));
    expectWithin(r.decode.tokensPerSec, naive, 0.9, 1.15);
    expect(r.decode.perSequenceTokensPerSec).toBe(r.decode.tokensPerSec); // batch 1
    // Every decode op of this config is memory-bound (M = 1 GEMMs, tiny attention).
    expect(
      phaseOps(r.decode)
        .filter((o) => o.op.kind !== 'collective')
        .every((o) => o.bound === 'memory'),
    ).toBe(true);
  });

  it('prefill @4096 is within the FLOP bound (13,874 tok/s); fusion none ≥ 0.83×, ideal ≥ 0.90×', () => {
    // FLOP of the tensor ops (corrected from the plan, which dropped the causal diagonal):
    //   linear GEMMs 2·(36·192,937,984 = 6,945,767,424)·4096 = 56,899,726,737,408
    // + attention 36 · 2·32·8,390,656·256 = 36 · 137,472,507,904 = 4,949,010,284,544
    //   (pairs = 4096·4097/2 = 8,390,656; qk + v = 128 + 128 = 256)
    // + LM head for the last position only 2·622,329,856 = 1,244,659,712
    // = 61,849,981,681,664 FLOP.
    const tensorFlops = 61_849_981_681_664;
    const prefillOps = lowerModel(REF_DENSE_8B, {
      phase: 'prefill',
      seqs: 1,
      qLen: 4096,
      kvLen: 4096,
      quant: BF16,
      parallel: P1,
    }).stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
    expect(prefillOps.filter(isTensor).reduce((s, op) => s + opFlops(op), 0)).toBe(tensorFlops);
    // Bound: 61.85e12 FLOP / 209.5e12 FLOP/s = 0.29523 s → 4096 / 0.29523 = 13,874 tok/s.
    const boundTokensPerSec = 4096 / (tensorFlops / rtx5090Bf16Peak());
    expectClose(boundTokensPerSec, 13_874, 1e-4);
    const none = simulate(input());
    // fusion 'none' adds ≈ 29 GB of elementwise traffic ≈ 16 ms on top of 295 ms of GEMM/attention.
    expectWithin(none.prefill.tokensPerSec, boundTokensPerSec, 0.83, 1.0);
    const ideal = simulate(input({ options: { elementwiseFusion: 'ideal' } }));
    expectWithin(ideal.prefill.tokensPerSec, boundTokensPerSec, 0.9, 1.0);
    expect(ideal.prefill.tokensPerSec).toBeGreaterThan(none.prefill.tokensPerSec);
    // Single device, batch 1: TTFT = cycle, per-sequence rate = aggregate rate.
    expectClose(none.prefill.latencySec, none.prefill.cycleSec);
    expectClose(none.prefill.perSequenceTokensPerSec, none.prefill.tokensPerSec);
  });

  it('memory fits: weights 16.38e9 + KV 0.60e9 < 32e9', () => {
    const r = simulate(input());
    const d = r.memory.perStage[0];
    expect(d.weightBytes).toBe(2 * countParams(REF_DENSE_8B)); // 16,381,470,720 B: every param in bf16, untied
    // KV at max(prompt 4096, ctx 1024) = 4096 tokens: 36 layers · 4096 tokens · 4096 B = 603,979,776 B
    expect(d.sequenceStateBytes).toBe(603_979_776);
    expect(d.capacityBytes).toBe(32e9);
    expect(r.memory.fits).toBe(true);
    expect(r.devicesUsed).toBe(1);
    expect(r.effectiveDevice).toBe(RTX_5090); // no modifiers: the base spec itself
  });
});

describe('T10 batch sweep on RTX 5090', () => {
  it('decode tok/s non-decreasing over batch [1, 8, 64]; per-sequence non-increasing', () => {
    const rs = [1, 8, 64].map((batch) =>
      simulate(input({ workload: { batch, promptLen: 128, decodeContextLen: 1024 } })),
    );
    for (let i = 1; i < rs.length; i++) {
      expect(rs[i].decode.tokensPerSec).toBeGreaterThanOrEqual(rs[i - 1].decode.tokensPerSec);
      expect(rs[i].decode.perSequenceTokensPerSec).toBeLessThanOrEqual(
        rs[i - 1].decode.perSequenceTokensPerSec,
      );
    }
  });

  it('batch 64 at ctx 8192 does not fit; maxSequencesPerReplica = 12', () => {
    // Prompt 128 keeps the prefill activation peak small (the plan's estimate ignores it).
    const r = simulate(input({ workload: { batch: 64, promptLen: 128, decodeContextLen: 8192 } }));
    const d = r.memory.perStage[0];
    // KV after the decode step at ctx 8192 holds 8193 tokens: 64 seqs · 36 layers · 8193 tokens · 4096 B
    // = 77,318,848,512 B ≈ 77.3e9
    expect(d.sequenceStateBytes).toBe(77_318_848_512);
    expect(r.memory.fits).toBe(false);
    // floor((32e9 − 16.38e9 − act) / 1.208e9): per-sequence state 36·8193·4096 = 1,208,107,008 B; the
    // activation peak (prefill silu_mul of 64·128 tokens: 3·8192·12288·2 B = 0.60e9) leaves 15.0e9 → 12.
    const statePerSeq = 1_208_107_008;
    const expected = Math.floor((32e9 - d.weightBytes - d.activationPeakBytes) / statePerSeq);
    expect(d.activationPeakBytes).toBe(3 * 64 * 128 * 12288 * 2);
    expect(expected).toBe(12);
    expect(r.memory.maxSequencesPerReplica).toBe(expected);
  });
});

describe('T11 TP 8 on SYNTH_GPU × SYNTH_CLUSTER', () => {
  const W: Workload = { batch: 1, promptLen: 1024, decodeContextLen: 1024 };
  const single = simulate(input({ workload: W, hardware: ON_SYNTH_1 }));
  const tp8 = simulate(
    input({ workload: W, hardware: ON_SYNTH_CLUSTER, parallel: { ...P1, tp: 8 } }),
  );

  it('single < decode < 8 × single; prefill ≤ 8 × single', () => {
    expect(tp8.decode.tokensPerSec).toBeGreaterThan(single.decode.tokensPerSec);
    expect(tp8.decode.tokensPerSec).toBeLessThan(8 * single.decode.tokensPerSec);
    expect(tp8.prefill.tokensPerSec).toBeLessThanOrEqual(8 * single.prefill.tokensPerSec);
    expect(tp8.warnings.some((w) => w.includes('8 of 16 devices are unused'))).toBe(true);
  });

  it('73 all-reduces (2 per layer + embed), all intra-node', () => {
    const ars = phaseOps(tp8.decode).filter(
      (o) => o.op.kind === 'collective' && o.op.collective === 'all_reduce',
    );
    expect(ars).toHaveLength(73); // 36 · 2 + 1
    expect(ars.every((o) => o.linkLevel === 'intra')).toBe(true);
  });

  it("'ideal' stage time = lat + max(compute, bw) ≤ 'none'", () => {
    const none = simulate(
      input({
        workload: W,
        hardware: ON_SYNTH_CLUSTER,
        parallel: { ...P1, tp: 8 },
        options: { overlap: 'none' },
      }),
    );
    for (const phase of ['prefill', 'decode'] as const) {
      const s = tp8[phase].stages[0];
      expectClose(s.sec, s.commLatencySec + Math.max(s.computeSec, s.commBandwidthSec));
      const n = none[phase].stages[0];
      expectClose(n.sec, n.computeSec + n.commLatencySec + n.commBandwidthSec);
      expect(s.sec).toBeLessThanOrEqual(n.sec);
    }
  });
});

describe('T15 sparsity and format fallback on RTX 5090', () => {
  const gateUp = (r: ReturnType<typeof simulate>): OpResult => {
    const o = phaseOps(r.prefill).find((x) => x.op.name === 'mlp.gate_up');
    if (o === undefined) throw new Error('no mlp.gate_up');
    return o;
  };

  it('2:4 weights: linear GEMM compute halves (419e12 peak), weight bytes × 9/16', () => {
    const dense = gateUp(simulate(input()));
    const sparse = gateUp(simulate(input({ quant: { ...BF16, weightSparsity: '2:4' } })));
    expect(sparse.peakFlopsPerSec).toBe(2 * rtx5090Bf16Peak()); // Table 3: BF16 2:4 = 419 TFLOP/s
    expectClose(sparse.computeSec, dense.computeSec / 2); // same shape, same efficiency
    // bf16 2:4: (16/2 + 1)/8 = 9/8 B per logical element vs 2 B dense → 9/16
    expectClose(sparse.bytes.weightBytes, (dense.bytes.weightBytes * 9) / 16);
  });

  it('nvfp4 weights + nvfp4 activations run at the fp4 dense peak (1676e12)', () => {
    const r = simulate(
      input({
        quant: { weights: { dtype: 'nvfp4' }, activations: 'nvfp4', kvCache: { dtype: 'bf16' } },
      }),
    );
    const o = gateUp(r);
    expect(o.usedInput).toBe('fp4');
    expect(o.peakFlopsPerSec).toBe(1676e12); // Table 3: FP4 Tensor, FP32 accumulate, dense
  });
});

describe('T18 hybrid + MLA smoke (synthetic)', () => {
  const W: Workload = { batch: 2, promptLen: 256, decodeContextLen: 1000 };
  for (const tp of [1, 2]) {
    it(`TINY_HYBRID tp ${tp}: decode reads and writes the recurrent state`, () => {
      const r = simulate(
        input({
          model: TINY_HYBRID,
          workload: W,
          hardware: ON_SYNTH_CLUSTER,
          parallel: { ...P1, tp },
        }),
      );
      expect(r.decode.tokensPerSec).toBeGreaterThan(0);
      expect(r.prefill.tokensPerSec).toBeGreaterThan(0);
      const rec = phaseOps(r.decode).filter((o) => o.op.kind === 'recurrent');
      expect(rec).toHaveLength(3); // layers 0–2 are GatedDeltaNet
      for (const o of rec) {
        // State per device: 2 seqs · (vH 4 / tp) heads · 32 · 32 · 4 B (fp32)
        const stateBytes = 2 * (4 / tp) * 32 * 32 * 4;
        expect(o.bytes.stateReadBytes).toBe(stateBytes);
        expect(o.bytes.stateWriteBytes).toBe(stateBytes);
      }
    });

    it(`TINY_MLA_SPARSE tp ${tp}: decode attention reads 64 selected latent tokens`, () => {
      const r = simulate(
        input({
          model: TINY_MLA_SPARSE,
          workload: W,
          hardware: ON_SYNTH_CLUSTER,
          parallel: { ...P1, tp },
        }),
      );
      const attn = phaseOps(r.decode).filter((o) => o.op.kind === 'attention');
      expect(attn).toHaveLength(2);
      for (const o of attn) {
        // 2 seqs · topK 64 tokens · (rkv 64 + dr 16 = 80) elems · 2 B (bf16 latent)
        expect(o.bytes.kvReadBytes).toBe(2 * 64 * 80 * 2);
      }
      expect(phaseOps(r.decode).some((o) => o.op.kind === 'indexer')).toBe(true);
    });
  }
});

describe('T19 e2e: memoryBandwidth modifier', () => {
  it('decode with memoryBandwidth 2.0e12 scales the T9 decode rate by 2.0/1.792', () => {
    const mod: DeviceModifier = {
      kind: 'memoryBandwidth',
      bandwidthBytesPerSec: 2.0e12,
      note: 'synthetic memory overclock',
      sources: [
        { url: 'synthetic://test', title: 'Synthetic', fields: ['memory.bandwidthBytesPerSec'] },
      ],
    };
    const base = simulate(input());
    const fast = simulate(input({ hardware: { ...ON_5090, deviceModifiers: [mod] } }));
    // Decode is memory-bound in every op, so its time scales with 1/bandwidth.
    expectClose(
      fast.decode.tokensPerSec,
      base.decode.tokensPerSec * (2.0e12 / RTX_5090.memory.bandwidthBytesPerSec),
      1e-6,
    );
    expect(fast.effectiveDevice.memory.bandwidthBytesPerSec).toBe(2.0e12);
    expect(RTX_5090.memory.bandwidthBytesPerSec).toBe(1.792e12); // base untouched
    expect(fast.warnings.some((w) => w.startsWith('variant: memoryBandwidth'))).toBe(true);
  });
});

/** Not a plan test: documents the default options so a change to them is a visible diff. */
describe('defaults', () => {
  it("overlap 'ideal', fusion 'none', all-reduce 'best', no per-op latency floor", () => {
    const expected: SimOptions = {
      overlap: 'ideal',
      elementwiseFusion: 'none',
      collectiveAlgorithm: 'best',
      perOpLatencyFloorSec: 0,
    };
    expect(DEFAULT_SIM_OPTIONS).toEqual(expected);
  });
});

describe('shortlist block kinds end to end (TINY fixtures on the synthetic cluster, tp 2, pp 2)', () => {
  const models = [TINY_CSA, TINY_HC_NGRAM, TINY_KDA, TINY_MLA_SHARED, TINY_INKLING, TINY_KIMI];
  it.each(models.map((m) => [m.name, m] as const))(
    '%s simulates with positive rates and memory',
    (_name, model) => {
      const r = simulate(
        input({
          model,
          parallel: { tp: 2, pp: 2, dp: 1, ep: 1 },
          workload: { batch: 2, promptLen: 256, decodeContextLen: 512 },
          hardware: ON_SYNTH_CLUSTER,
        }),
      );
      expect(r.prefill.tokensPerSec).toBeGreaterThan(0);
      expect(r.decode.tokensPerSec).toBeGreaterThan(0);
      expect(r.memory.perStage.every((s) => s.sequenceStateBytes > 0)).toBe(true);
    },
  );
  it('a cross-stage source warns; an invalid cross-layer reference is rejected', () => {
    const r = simulate(
      input({
        model: TINY_MLA_SHARED,
        parallel: { tp: 1, pp: 3, dp: 1, ep: 1 },
        hardware: ON_SYNTH_CLUSTER,
      }),
    );
    expect(r.warnings.filter((w) => w.includes('on another pipeline stage'))).toHaveLength(3);
    const broken = {
      ...TINY_MLA_SHARED,
      layers: [TINY_MLA_SHARED.layers[2], ...TINY_MLA_SHARED.layers.slice(1)],
    };
    expect(() => simulate(input({ model: broken }))).toThrow(
      /kvCacheFromLayer 0 must be an earlier layer index/,
    );
  });
});
