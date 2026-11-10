/**
 * Labelled what-if figures next to the unchanged headline (docs/review/practicality-review.md F7, F1a, F1b;
 * docs/simulation-model.md §7 "Prefill without overlap", "Per-op latency floor", "Engine dispatch overlay"):
 * - with everything off the result is bit-identical (the floor 0, no overlay);
 * - each figure and setting follows its formula;
 * - the review's probe numbers are reproduced (within a few points: the probes ran before the PCIe payload
 *   efficiency and SFU changes, and their exact workloads are not recorded).
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import { simpleDecoder, simulateReport, type SanityInput } from '../__sanity__/harness';
import { GPT_OSS_120B, LLAMA3_70B, QWEN3_30B_A3B } from '../__sanity__/models';
import { LLAMACPP_MMQ } from '../__sanity__/reports';
import { singleDevice } from '../hw/presets';
import { PCIE_4_X16 } from '../hw/presets/links';
import { RTX_3090, RTX_4090, RTX_5090 } from '../hw/presets/nvidia';
import { DGX_SPARK_PAIR, pcieBox } from '../hw/presets/topologies';
import type { QuantConfig } from '../ir/model';
import { REF_DENSE_8B, REF_MOE_30B_A3B } from '../models/reference';
import { DEFAULT_DISPATCH_SEC_PER_KERNEL, stageKernelCount } from './dispatch';
import { withLatencyFloor } from './opTime';
import { simulate } from './simulate';
import type { OpResult, SimInput, SimResult } from './types';
import { validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const P1 = { tp: 1, pp: 1, dp: 1, ep: 1 };

function input(over: Partial<SimInput> = {}): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: P1,
    workload: { batch: 1, promptLen: 4096, decodeContextLen: 1024 },
    hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) },
    ...over,
  };
}

const CASES: Record<string, SimInput> = {
  'one RTX 5090': input(),
  'tp 4 on the synthetic cluster': input({
    hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
    parallel: { ...P1, tp: 4 },
  }),
  'tp 2 pp 2, batch 4, MoE': input({
    model: REF_MOE_30B_A3B,
    hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
    parallel: { ...P1, tp: 2, pp: 2 },
    workload: { batch: 4, promptLen: 1024, decodeContextLen: 2048 },
  }),
};

/** The headline part of a result: everything but the what-if figures. */
function headline(r: SimResult): Omit<SimResult, 'prefillWithoutOverlap' | 'decodeWithDispatch'> {
  const { prefillWithoutOverlap: _p, decodeWithDispatch: _d, ...rest } = r;
  void _p;
  void _d;
  return rest;
}

/** Every op of a phase (all stages). */
function phaseOps(r: SimResult['decode']): OpResult[] {
  return r.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
}

describe('everything off: bit-identical', () => {
  it.each(Object.entries(CASES))(
    '%s: floor 0 and an empty overlay change nothing',
    (_name, base) => {
      const plain = simulate(base);
      const explicit = simulate({ ...base, options: { perOpLatencyFloorSec: 0 }, overlays: {} });
      expect(explicit).toEqual(plain);
      expect(Object.is(explicit.prefill.tokensPerSec, plain.prefill.tokensPerSec)).toBe(true);
      expect(Object.is(explicit.decode.tokensPerSec, plain.decode.tokensPerSec)).toBe(true);
      expect(plain.decodeWithDispatch).toBeUndefined();
      expect(phaseOps(plain.decode).some((o) => o.latencyFloorSec !== undefined)).toBe(false);
    },
  );
  it.each(Object.entries(CASES))(
    '%s: the dispatch overlay never touches the headline',
    (_name, base) => {
      const plain = simulate(base);
      const overlay = simulate({ ...base, overlays: { engineDispatch: {} } });
      expect(headline(overlay)).toEqual(headline(plain));
      expect(Object.is(overlay.decode.tokensPerSec, plain.decode.tokensPerSec)).toBe(true);
      expect(overlay.prefillWithoutOverlap).toEqual(plain.prefillWithoutOverlap);
    },
  );
});

describe('prefill without overlap (F7)', () => {
  it('is absent without inter-device communication (one device)', () => {
    expect(simulate(CASES['one RTX 5090']).prefillWithoutOverlap).toBeUndefined();
  });
  it("is absent when the run already uses overlap 'none' (the headline is that figure)", () => {
    const r = simulate({ ...CASES['tp 4 on the synthetic cluster'], options: { overlap: 'none' } });
    expect(r.prefillWithoutOverlap).toBeUndefined();
  });
  it.each(['tp 4 on the synthetic cluster', 'tp 2 pp 2, batch 4, MoE'])(
    "%s: equals the same input's prefill under overlap 'none', bit for bit, and never exceeds the headline",
    (name) => {
      const base = CASES[name];
      const r = simulate(base);
      const none = simulate({ ...base, options: { overlap: 'none' } }).prefill;
      expect(r.prefillWithoutOverlap).toEqual({
        tokensPerSec: none.tokensPerSec,
        perSequenceTokensPerSec: none.perSequenceTokensPerSec,
        latencySec: none.latencySec,
        cycleSec: none.cycleSec,
      });
      expect(r.prefillWithoutOverlap!.tokensPerSec).toBeLessThanOrEqual(r.prefill.tokensPerSec);
    },
  );
});

describe('prefill without overlap with per-collective overlap and two-batch overlap (docs §7)', () => {
  const rates = (p: SimResult['prefill']) => ({
    tokensPerSec: p.tokensPerSec,
    perSequenceTokensPerSec: p.perSequenceTokensPerSec,
    latencySec: p.latencySec,
    cycleSec: p.cycleSec,
  });
  const tp4 = CASES['tp 4 on the synthetic cluster'];
  it("is present with a partial fraction under global 'none', and equals the run with every fraction 0", () => {
    const r = simulate({ ...tp4, options: { overlap: 'none', overlapByClass: { tp: 0.5 } } });
    const none = simulate({ ...tp4, options: { overlap: 'none' } });
    expect(r.prefillWithoutOverlap).toEqual(rates(none.prefill));
    expect(r.prefill.tokensPerSec).toBeGreaterThan(none.prefill.tokensPerSec);
  });
  it('is absent when every fraction is 0, even under global ideal', () => {
    const r = simulate({
      ...tp4,
      options: { overlap: 'ideal', overlapByClass: { tp: 0, ep: 0, pp: 0 } },
    });
    expect(r.prefillWithoutOverlap).toBeUndefined();
  });
  it('a two-batch-overlap run: equals a separate run with every fraction 0 and two-batch overlap off', () => {
    const moe = input({
      model: REF_MOE_30B_A3B,
      hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
      parallel: { ...P1, ep: 2, dp: 2 },
      workload: { batch: 4, promptLen: 1024, decodeContextLen: 2048 },
      options: { overlap: 'none' },
    });
    const r = simulate({ ...moe, deployment: { twoBatchOverlap: true } });
    expect(r.prefillWithoutOverlap).toEqual(rates(simulate(moe).prefill));
    expect(simulate(moe).prefillWithoutOverlap).toBeUndefined();
  });
});

describe('per-op latency floor (F1a)', () => {
  /** A bare op result: `sec` split into latency `lat` and the rest (bandwidth/service) for collectives and transfers. */
  const op = (kind: 'gemm' | 'collective' | 'transfer', sec: number, lat = 0) =>
    ({
      op: { kind, name: 'x' },
      sec,
      bound: kind === 'gemm' ? 'memory' : kind === 'collective' ? 'comm' : 'transfer',
      commLatencySec: kind === 'collective' ? lat : 0,
      commBandwidthSec: kind === 'collective' ? sec - lat : 0,
      transferLatencySec: kind === 'transfer' ? lat : 0,
      transferBandwidthSec: kind === 'transfer' ? sec - lat : 0,
    }) as unknown as OpResult;
  it('raises an on-device op below the floor to it, keeps its bound, and records the floor', () => {
    const r = withLatencyFloor(op('gemm', 0.2e-6), 1e-6);
    expect(r.sec).toBe(1e-6);
    expect(r.latencyFloorSec).toBe(1e-6);
    expect(r.bound).toBe('memory');
  });
  it('leaves an op at or above the floor unchanged (same object)', () => {
    const slow = op('gemm', 3e-6);
    expect(withLatencyFloor(slow, 1e-6)).toBe(slow);
    const coll = op('collective', 3e-6, 1e-6);
    expect(withLatencyFloor(coll, 1e-6)).toBe(coll);
  });
  it('raises a collective too: the lift goes to its latency term (α′ = α + L − t), β unchanged', () => {
    const r = withLatencyFloor(op('collective', 4e-6, 1e-6), 100e-6);
    expect(r.sec).toBe(100e-6);
    expect(r.latencyFloorSec).toBe(100e-6);
    expect(r.commLatencySec).toBeCloseTo(97e-6, 18);
    expect(r.commBandwidthSec).toBe(3e-6);
    expect(r.bound).toBe('comm');
  });
  it('raises an off-device transfer too: the lift goes to its latency term, service unchanged', () => {
    const r = withLatencyFloor(op('transfer', 4e-6, 1e-6), 100e-6);
    expect(r.sec).toBe(100e-6);
    expect(r.transferLatencySec).toBeCloseTo(97e-6, 18);
    expect(r.transferBandwidthSec).toBe(3e-6);
  });
  it('tp 4: a 100 µs floor lifts every TP collective on the critical path (regression: collectives were skipped)', () => {
    const base: SimInput = {
      ...CASES['tp 4 on the synthetic cluster'],
      options: { collectiveAlgorithm: 'ring' },
    };
    const L = 100e-6;
    const plain = simulate(base);
    const floored = simulate({ ...base, options: { ...base.options, perOpLatencyFloorSec: L } });
    const collectives = phaseOps(floored.decode).filter((o) => o.op.kind === 'collective');
    expect(collectives.length).toBeGreaterThan(0);
    for (const o of collectives) expect(o.sec).toBeGreaterThanOrEqual(L);
    // Formula (overlap 'ideal', pp 1): t = lat' + max(compute', bw), with
    //   compute' = Σ_on-device max(t, L), lat' = Σ_coll α + max(0, L − t), bw = Σ_coll β
    const before = phaseOps(plain.decode);
    const coll = before.filter((o) => o.op.kind === 'collective');
    const compute = before
      .filter((o) => o.op.kind !== 'collective')
      .reduce((s, o) => s + Math.max(o.sec, L), 0);
    const lat = coll.reduce((s, o) => s + o.commLatencySec + Math.max(0, L - o.sec), 0);
    const bw = coll.reduce((s, o) => s + o.commBandwidthSec, 0);
    expect(floored.decode.cycleSec).toBeCloseTo(lat + Math.max(compute, bw), 12);
    // At least L per collective: α′ + β summed over the stage's collectives.
    const stage = floored.decode.stages[0];
    expect(stage.commLatencySec + stage.commBandwidthSec).toBeGreaterThanOrEqual(
      collectives.length * L - 1e-12,
    );
    expect(floored.decode.comm.latencySec).toBeGreaterThan(plain.decode.comm.latencySec);
  });
  it('one device: the decode step is Σ max(t_op, L) over the ops (max, not additive)', () => {
    const base = CASES['one RTX 5090'];
    const L = 1e-6;
    const plain = simulate(base);
    const floored = simulate({ ...base, options: { perOpLatencyFloorSec: L } });
    // Formula: t_step = Σ_ops max(t_op, L)
    const expected = phaseOps(plain.decode).reduce((s, o) => s + Math.max(o.sec, L), 0);
    expect(floored.decode.cycleSec).toBeCloseTo(expected, 15);
    expect(floored.decode.tokensPerSec).toBeLessThan(plain.decode.tokensPerSec);
    // Large prefill ops are far above 1 µs: only the tiny ones move.
    expect(floored.prefill.tokensPerSec / plain.prefill.tokensPerSec).toBeGreaterThan(0.999);
  });
  it('a negative or non-finite floor is a validation error', () => {
    expect(validateInput({ ...input(), options: { perOpLatencyFloorSec: -1e-6 } })).toEqual([
      'options.perOpLatencyFloorSec (-0.000001) must be a finite number ≥ 0',
    ]);
    expect(
      validateInput({ ...input(), options: { perOpLatencyFloorSec: Number.NaN } }),
    ).toHaveLength(1);
  });
});

describe('engine dispatch overlay (F1b)', () => {
  it('default τ is 2 µs (inside the review’s measured 1.3–2.9 µs)', () => {
    expect(DEFAULT_DISPATCH_SEC_PER_KERNEL).toBe(2e-6);
  });
  it('one device: cycle + (number of IR ops)·τ, kernels = the IR op count (no fusion)', () => {
    const base = CASES['one RTX 5090'];
    const r = simulate({ ...base, overlays: { engineDispatch: {} } });
    const d = r.decodeWithDispatch!;
    const n = phaseOps(r.decode).length;
    expect(d.kernelsPerStep).toBe(n);
    expect(d.perKernelSec).toBe(DEFAULT_DISPATCH_SEC_PER_KERNEL);
    // Formula: t_step' = t_step + n·τ
    expect(d.cycleSec).toBeCloseTo(r.decode.cycleSec + n * 2e-6, 15);
    expect(d.tokensPerSec).toBeCloseTo(1 / d.cycleSec, 9);
  });
  it('pp 2: each stage gets its own kernels’ dispatch, then the pipeline rule', () => {
    const base = CASES['tp 2 pp 2, batch 4, MoE'];
    const tau = 3e-6;
    const r = simulate({ ...base, overlays: { engineDispatch: { perKernelSec: tau } } });
    const d = r.decodeWithDispatch!;
    expect(d.kernelsPerStep).toBe(r.decode.stages.reduce((s, st) => s + stageKernelCount(st), 0));
    expect(d.cycleSec).toBeGreaterThan(r.decode.cycleSec);
    expect(d.tokensPerSec).toBeLessThan(r.decode.tokensPerSec);
  });
  it('τ = 0 reproduces the headline decode rate', () => {
    const base = CASES['tp 4 on the synthetic cluster'];
    const r = simulate({ ...base, overlays: { engineDispatch: { perKernelSec: 0 } } });
    expect(r.decodeWithDispatch!.tokensPerSec).toBe(r.decode.tokensPerSec);
  });
  it('a negative τ is a validation error', () => {
    expect(
      validateInput({ ...input(), overlays: { engineDispatch: { perKernelSec: -1 } } }),
    ).toEqual(['overlays.engineDispatch.perKernelSec (-1) must be a finite number ≥ 0']);
  });
});

// ---------------------------------------------------------------------------------------------------------------
// The review's probes (docs/review/practicality-review.md F1, F7).

/** Meta-Llama-3-8B (meta-llama/Meta-Llama-3-8B config.json): h 4096, 32 layers, GQA 32/8, hd 128, ffn 14336. */
const LLAMA3_8B = simpleDecoder({
  name: 'Llama-3-8B',
  source: 'https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json',
  hidden: 4096,
  vocab: 128256,
  tieEmbeddings: false,
  layers: 32,
  mixer: { kind: 'gqa', heads: 32, kvHeads: 8, headDim: 128 },
  ffn: { kind: 'dense', intermediate: 14336 },
});

/** Relative change b/a − 1 (dimensionless). */
const change = (a: number, b: number) => b / a - 1;

describe('review probe F7: prefill under overlap none', () => {
  // review: −40% (4×4090 TP4, Llama-3-70B), −31.5% (Spark pair), −24% (2×3090 TP2, Qwen3-30B-A3B)
  const probes: [string, SanityInput, number][] = [
    [
      '4×4090 TP4 (direct P2P), Llama-3-70B Q4_K_M, prefill @ 8k: −38.3% (review −40%)',
      {
        model: { kind: 'spec', spec: LLAMA3_70B },
        quant: 'q4_k_m',
        numerics: LLAMACPP_MMQ,
        hardware: {
          deviceId: 'rtx-4090',
          topology: pcieBox(RTX_4090, 4, PCIE_4_X16),
          unlocks: { 'pcie-p2p': 'direct' },
        },
        workload: { promptLen: 8192, genLen: 128 },
      },
      -0.4,
    ],
    [
      '2 × DGX Spark TP2, Llama-3-70B NVFP4, prefill @ 2k: −31.5% (review −31.5%)',
      {
        model: { kind: 'spec', spec: LLAMA3_70B },
        quant: 'nvfp4',
        hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_PAIR.id },
        workload: { promptLen: 2048, genLen: 128 },
      },
      -0.315,
    ],
    [
      '2×3090 TP2 (direct P2P), Qwen3-30B-A3B Q4_K_M fp16 GEMMs, prefill @ 512: −21.5% (review −24%)',
      {
        model: { kind: 'spec', spec: QWEN3_30B_A3B },
        quant: 'q4_k_m',
        numerics: { activations: 'fp16' },
        hardware: {
          deviceId: 'rtx-3090',
          topology: pcieBox(RTX_3090, 2, PCIE_4_X16),
          unlocks: { 'pcie-p2p': 'direct' },
        },
        workload: { promptLen: 512, genLen: 128 },
      },
      -0.24,
    ],
  ];
  it.each(probes)('%s', (_name, probe, review) => {
    const r = simulateReport(probe);
    const without = r.sim.prefillWithoutOverlap;
    expect(without).toBeDefined();
    expect(Math.abs(change(r.prefillTokensPerSec, without!.tokensPerSec) - review)).toBeLessThan(
      0.03,
    );
  });
});

describe('review probe F1: per-op floor and dispatch, decode @ 64 at batch 1', () => {
  const probe = (
    spec: SanityInput['model'],
    quant: SanityInput['quant'],
    deviceId: string,
    promptLen: number,
  ): SanityInput => ({
    model: spec,
    quant,
    ...(quant === 'q4_k_m' ? { numerics: LLAMACPP_MMQ } : {}),
    hardware: { deviceId },
    workload: { promptLen, genLen: 128 },
  });
  // review table: hardware floor 0.6 µs / 1 µs, dispatch floor 1 µs / 2 µs / 3 µs (blank cells not probed)
  const rows: [string, SanityInput, (number | undefined)[]][] = [
    [
      'Llama-3-8B, 4090',
      probe({ kind: 'spec', spec: LLAMA3_8B }, 'q4_k_m', 'rtx-4090', 512),
      [-0.025, -0.043, -0.07, -0.13, -0.19],
    ],
    [
      'Qwen3-30B-A3B, 4090',
      probe({ kind: 'spec', spec: QWEN3_30B_A3B }, 'q4_k_m', 'rtx-4090', 512),
      [-0.12, -0.19, -0.27, -0.42, -0.52],
    ],
    [
      'gpt-oss-120b, Spark',
      probe({ kind: 'spec', spec: GPT_OSS_120B }, 'mxfp4-wo', 'dgx-spark', 2048),
      [-0.011, -0.021, -0.04, undefined, -0.1],
    ],
  ];
  it.each(rows)('%s', (_name, base, review) => {
    const plain = simulateReport(base).decodeTokensPerSec;
    const floors = [0.6e-6, 1e-6].map((L) =>
      change(
        plain,
        simulateReport({ ...base, options: { perOpLatencyFloorSec: L } }).decodeTokensPerSec,
      ),
    );
    const dispatch = [1e-6, 2e-6, 3e-6].map((tau) =>
      change(
        plain,
        simulateReport({ ...base, overlays: { engineDispatch: { perKernelSec: tau } } }).sim
          .decodeWithDispatch!.tokensPerSec,
      ),
    );
    [...floors, ...dispatch].forEach((got, i) => {
      if (review[i] !== undefined) expect(Math.abs(got - review[i]!)).toBeLessThan(0.02);
    });
  });
});
