import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU, SYNTH_NVLINK } from '../__fixtures__/synthetic';
import type { QuantConfig } from '../ir/model';
import type { CollectiveOp, ElementwiseOp, TransferOp } from '../ir/ops';
import { REF_DENSE_8B, TINY_DENSE, TINY_HYBRID } from '../models/reference';
import type { TopologySpec } from '../hw/types';
import { singleDevice } from '../hw/presets';
import {
  layerResult,
  microbatching,
  phaseComm,
  phaseResult,
  phaseTransfer,
  stageResult,
  stageTransfer,
} from './pipeline';
import { simulate } from './simulate';
import type { LayerResult, OpResult, SimInput, StageResult } from './types';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

function input(over: Partial<SimInput>): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 512, decodeContextLen: 1024 },
    hardware: { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU) },
    ...over,
  };
}

describe('microbatching', () => {
  it('m = min(pp, B_r), b_mb = ceil(B_r / m)', () => {
    expect(microbatching(2, 2)).toEqual({ microbatches: 2, microbatchSize: 1 });
    expect(microbatching(4, 2)).toEqual({ microbatches: 2, microbatchSize: 1 });
    expect(microbatching(2, 5)).toEqual({ microbatches: 2, microbatchSize: 3 });
    expect(microbatching(1, 8)).toEqual({ microbatches: 1, microbatchSize: 8 });
  });
});

describe('T12 PP 2, batch 2 (REF_DENSE_8B on SYNTH_GPU × SYNTH_CLUSTER)', () => {
  const single = simulate(input({}));
  const pp2 = simulate(
    input({
      parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
      workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 },
      hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
    }),
  );

  it('decode cycle = max(2·max t_s, Σ t_s + p2p), from the result’s own stages', () => {
    const [s0, s1] = pp2.decode.stages;
    expect(pp2.decode.microbatches).toBe(2);
    expect(pp2.decode.microbatchSize).toBe(1);
    expect(s0.sendToNext?.linkLevel).toBe('intra'); // ranks 0 → 1 share node 0
    expect(s1.sendToNext).toBeUndefined();
    const p2p = s0.sendToNext?.sec ?? NaN;
    expectClose(pp2.decode.cycleSec, Math.max(2 * Math.max(s0.sec, s1.sec), s0.sec + s1.sec + p2p));
    expectClose(pp2.decode.tokensPerSec, 2 / pp2.decode.cycleSec);
    expectClose(pp2.decode.perSequenceTokensPerSec, 1 / pp2.decode.cycleSec);
  });

  it('decode throughput vs single (batch 1) ∈ [1.75, 2.0]', () => {
    // Stage 1 carries the LM head (1.24 GB of 15.1 GB streamed), so the slower stage is ≈ 0.54 of the single
    // device: 2 / (2 · 0.54) ≈ 1.85.
    const ratio = pp2.decode.tokensPerSec / single.decode.tokensPerSec;
    expect(ratio).toBeGreaterThanOrEqual(1.75);
    expect(ratio).toBeLessThanOrEqual(2.0);
  });

  it('prefill: cycle = m·max t_s; TTFT = Σ t_s + p2p + (m−1)·max t_s', () => {
    const [s0, s1] = pp2.prefill.stages;
    const maxT = Math.max(s0.sec, s1.sec);
    expectClose(pp2.prefill.cycleSec, 2 * maxT);
    expectClose(pp2.prefill.latencySec, s0.sec + s1.sec + (s0.sendToNext?.sec ?? NaN) + maxT);
    expectClose(pp2.prefill.tokensPerSec, (2 * 512) / pp2.prefill.cycleSec);
    expectClose(pp2.prefill.perSequenceTokensPerSec, 512 / pp2.prefill.latencySec);
  });
});

describe('T13 DP 2, batch 2', () => {
  it('aggregate = 2 × single exactly; per-sequence rates equal', () => {
    const single = simulate(input({}));
    const dp2 = simulate(
      input({
        parallel: { tp: 1, pp: 1, dp: 2, ep: 1 },
        workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 },
        hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
      }),
    );
    for (const phase of ['prefill', 'decode'] as const) {
      expectClose(dp2[phase].tokensPerSec, 2 * single[phase].tokensPerSec, 1e-12);
      expectClose(dp2[phase].perSequenceTokensPerSec, single[phase].perSequenceTokensPerSec, 1e-12);
    }
    expect(dp2.devicesUsed).toBe(2);
  });
});

describe('stageResult overlap rules', () => {
  it("'none' adds the PP send into t_s; 'ideal' leaves it to the latency path", () => {
    const r = simulate(
      input({
        parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
        workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 },
        hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
        options: { overlap: 'none' },
      }),
    );
    const s0 = r.decode.stages[0];
    const send = s0.sendToNext?.sec ?? NaN;
    expectClose(s0.sec, s0.computeSec + s0.commLatencySec + s0.commBandwidthSec + send);
    // Rebuilding with 'ideal' from the same layers drops the send from t_s.
    const ideal = stageResult(s0.index, s0.layers, s0.sendToNext, 'ideal');
    expectClose(ideal.sec, s0.computeSec); // tp = 1: no collectives inside the stage
    // Under 'none' the decode cycle does not add the sends a second time.
    const phase = phaseResult(r.decode.stages, {
      phase: 'decode',
      dp: 1,
      sequencesPerReplica: 2,
      microbatches: 2,
      microbatchSize: 1,
      promptLen: 512,
      overlap: 'none',
    });
    const [a, b] = r.decode.stages;
    expectClose(phase.cycleSec, Math.max(2 * Math.max(a.sec, b.sec), a.sec + b.sec));
  });
});

describe('PP link bandwidth bounds the cycle (review P1): TINY_DENSE pp 2, batch 2, prompt 128, link 1 B/s', () => {
  // Every per-stage time is microseconds; the 1 B/s link dominates by ~11 orders of magnitude, so the
  // expected values below are the link terms alone (rel 1e-6 tolerance covers the stage times and α = 1 µs).
  const slow: TopologySpec = {
    ...SYNTH_CLUSTER,
    intraNode: { ...SYNTH_NVLINK, bandwidthPerDirectionBytesPerSec: 1 },
  };
  const run = (overlap: 'ideal' | 'none') =>
    simulate(
      input({
        model: TINY_DENSE,
        parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
        workload: { batch: 2, promptLen: 128, decodeContextLen: 256 },
        hardware: { device: SYNTH_GPU, topology: slow },
        options: { overlap },
      }),
    );

  for (const overlap of ['ideal', 'none'] as const) {
    it(`${overlap}: prefill cycle ≥ 2 microbatches · 65,536 B / 1 B/s = 131,072 s; TTFT ≈ 131,072 s`, () => {
      const r = run(overlap);
      expect(r.prefill.microbatches).toBe(2); // m = min(pp 2, B_r 2), b_mb = 1
      // Send per microbatch: T·h·bpe = (1 seq · 128 tokens) · 256 · 2 B = 65,536 B → 65,536 s at 1 B/s.
      expect(
        r.prefill.stages[0].sendToNext?.op.kind === 'collective' &&
          r.prefill.stages[0].sendToNext.op.messageBytes,
      ).toBe(65_536);
      // The link serves both microbatches each cycle: cycle ≥ 2 · 65,536 = 131,072 s.
      expect(r.prefill.cycleSec).toBeGreaterThanOrEqual(131_072);
      expectClose(r.prefill.cycleSec, 131_072, 1e-6);
      // tokens/s = B_r · promptLen / cycle = 2 · 128 / 131,072 = 1/512
      expectClose(r.prefill.tokensPerSec, 1 / 512, 1e-6);
      // TTFT: the first microbatch's send (65,536 s) + the second waits one link slot (65,536 s) = 131,072 s.
      expect(r.prefill.latencySec).toBeGreaterThanOrEqual(131_072);
      expectClose(r.prefill.latencySec, 131_072, 1e-6);
    });

    it(`${overlap}: decode cycle = 2 · 512 B / 1 B/s = 1,024 s`, () => {
      const r = run(overlap);
      // Send per microbatch: 1 seq · 1 token · 256 · 2 B = 512 B → 512 s; two microbatches per cycle.
      expect(r.decode.cycleSec).toBeGreaterThanOrEqual(1_024);
      expectClose(r.decode.cycleSec, 1_024, 1e-6);
      expectClose(r.decode.tokensPerSec, 2 / 1_024, 1e-6);
    });
  }
});

// ---------------------------------------------------------------------------------------------------------------
// Off-device transfers in the stage formula (design doc §2.6, U-5) and the comm split (§4.2).

/**
 * A bare OpResult with the given times (other fields zero); `sec` follows the op kind's rule. Transfer ops get
 * `transferResources` with the given link and host-RAM times (default 0).
 */
function result(
  op: ElementwiseOp | CollectiveOp | TransferOp,
  t: {
    sec: number;
    lat?: number;
    bw?: number;
    xferLat?: number;
    xferBw?: number;
    link?: number;
    host?: number;
  },
): OpResult {
  const bytes = {
    weightBytes: 0,
    activationReadBytes: 0,
    activationWriteBytes: 0,
    kvReadBytes: 0,
    kvWriteBytes: 0,
    stateReadBytes: 0,
    stateWriteBytes: 0,
    totalBytes: 0,
  };
  const bound = op.kind === 'collective' ? 'comm' : op.kind === 'transfer' ? 'transfer' : 'compute';
  const r: OpResult = {
    op,
    flops: 0,
    bytes,
    computeSec: op.kind === 'elementwise' ? t.sec : 0,
    memorySec: 0,
    commLatencySec: t.lat ?? 0,
    commBandwidthSec: t.bw ?? 0,
    transferLatencySec: t.xferLat ?? 0,
    transferBandwidthSec: t.xferBw ?? 0,
    sec: t.sec,
    bound,
  };
  if (op.kind === 'transfer') {
    r.transferResources = {
      linkBytes: 0,
      linkSec: t.link ?? 0,
      hostRamBytes: 0,
      hostRamSec: t.host ?? 0,
    };
  }
  return r;
}
const ELEMENTWISE: ElementwiseOp = {
  kind: 'elementwise',
  name: 'x',
  fn: 'residual_add',
  elems: 1,
  flopsPerElem: 1,
  dtype: 'bf16',
  activationReadBytes: 0,
  activationWriteBytes: 0,
  weightBytes: 0,
  stateReadBytes: 0,
  stateWriteBytes: 0,
};
const ALL_REDUCE: CollectiveOp = {
  kind: 'collective',
  name: 'tp_all_reduce',
  collective: 'all_reduce',
  group: 'tp',
  groupSize: 2,
  messageBytes: 1,
  dtype: 'bf16',
};
const SEND: CollectiveOp = {
  ...ALL_REDUCE,
  name: 'pp.send_recv',
  collective: 'send_recv',
  group: 'pp',
};
const FETCH: TransferOp = {
  kind: 'transfer',
  name: 'moe.expert_fetch',
  group: 'routed_experts',
  source: 'host_ram',
  reads: [
    { count: 8, bytesEach: 2_654_208 },
    { count: 0.5, bytesEach: 64 },
  ],
  usefulBytes: 21_233_664,
  prefetchedFraction: 1,
};

describe('U-5 stage formula with transfers: compute 2, bw 1, lat 0.5, xfer 3, p2p 0.25 (seconds)', () => {
  // core_s = max(compute, bw) under overlap ideal, compute + bw + p2p under none; exposed_s = lat_s.
  /** The stage's layer with a transfer of `xfer` seconds whose prefetched share is q. */
  const layers = (xfer: number, q = 1) => [
    layerResult('layer.0', 0, [
      result(ELEMENTWISE, { sec: 2 }),
      result(ALL_REDUCE, { sec: 1.5, lat: 0.5, bw: 1 }),
      ...(xfer > 0
        ? [
            result(
              { ...FETCH, prefetchedFraction: q },
              { sec: xfer, xferLat: 1e-6, xferBw: xfer - 1e-6 },
            ),
          ]
        : []),
    ]),
  ];
  const send = result(SEND, { sec: 0.25, lat: 0.2, bw: 0.05 });
  /** Local copies of the former offloadOverlap formulas: 'ideal' t = lat + max(core, xfer), 'none' t = t_core + xfer. */
  const formerIdeal = (lat: number, core: number, xfer: number) => lat + Math.max(core, xfer);
  const formerNone = (tCore: number, xfer: number) => tCore + xfer;
  const cases = [
    // [overlap, q, t_s, exposedTransfer_s]
    ['ideal', 1, formerIdeal(0.5, 2, 3), 1], // 0.5 + max(max(2, 1), 3) = 3.5
    ['ideal', 0, formerNone(2.5, 3), 3], // 0.5 + max(2, 1) + 3 = 5.5
    ['none', 1, formerIdeal(0.5, 3.25, 3), 0], // 0.5 + max(2 + 1 + 0.25, 3) = 3.75: core 3.25 hides xfer 3
    ['none', 0, formerNone(3.75, 3), 3], // 2 + 0.5 + 1 + 0.25 + 3 = 6.75
    ['ideal', 0.5, 4.5, 2], // 0.5 + 2 + 3 − 0.5·min(2, 3)
    ['none', 0.5, 5.25, 1.5], // 0.5 + 3.25 + 3 − 0.5·min(3.25, 3)
  ] as const;
  it.each(cases)('overlap %s, q %s: t_s = %s, exposedTransfer = %s', (overlap, q, t, exposed) => {
    const s = stageResult(0, layers(3, q), send, overlap);
    expect(s.sec).toBe(t);
    expect(s.computeSec).toBe(2); // the transfer op is not compute
    expect(s.transferSec).toBe(3);
    expect(s.exposedTransferSec).toBe(exposed);
    // exposed comm is the transfer-free core's: ideal 0.5 + 2 − 2; none 0.5 + 1 + 0.25
    expect(s.exposedCommSec).toBe(overlap === 'ideal' ? 0.5 : 1.75);
  });
  it('q 0.8 (the default p at batch 1): t_s = 0.5 + 2 + 3 − 0.8·2 = 3.9', () => {
    expect(stageResult(0, layers(3, 0.8), send, 'ideal').sec).toBeCloseTo(3.9, 12);
  });
  it('mixed q: a table-row fetch (q 1, 1 s) and an expert fetch (q 0.5, 2 s) give pbar = 2/3', () => {
    const mixed = [
      layerResult('layer.0', 0, [
        result(ELEMENTWISE, { sec: 2 }),
        result(ALL_REDUCE, { sec: 1.5, lat: 0.5, bw: 1 }),
        result(
          { ...FETCH, group: 'lookup_table', prefetchedFraction: 1 },
          { sec: 1, xferLat: 0, xferBw: 1 },
        ),
        result({ ...FETCH, prefetchedFraction: 0.5 }, { sec: 2, xferLat: 0, xferBw: 2 }),
      ]),
    ];
    // Formula: t = 0.5 + 2 + 3 − (2/3)·min(2, 3)
    expect(stageResult(0, mixed, send, 'ideal').sec).toBeCloseTo(5.5 - 4 / 3, 12);
  });
  it('lat + max(core, xfer) ≤ t_s ≤ t_core + xfer for every q and both overlap modes', () => {
    for (const overlap of ['ideal', 'none'] as const) {
      const core = overlap === 'ideal' ? 2 : 3.25;
      const tCore = 0.5 + core;
      for (const xfer of [0.5, 2, 3, 10]) {
        for (const q of [0, 0.1, 0.5, 0.8, 0.95, 1]) {
          const t = stageResult(0, layers(xfer, q), send, overlap).sec;
          expect(t).toBeGreaterThanOrEqual(0.5 + Math.max(core, xfer) - 1e-12);
          expect(t).toBeLessThanOrEqual(tCore + xfer + 1e-12);
        }
      }
    }
  });
  it('xfer 0 reproduces the transfer-free stage bit for bit', () => {
    for (const overlap of ['ideal', 'none'] as const) {
      const s = stageResult(0, layers(0), send, overlap);
      expect(s.sec).toBe(overlap === 'ideal' ? 2.5 : 3.75);
      expect(s.transferSec).toBe(0);
      expect(s.exposedTransferSec).toBe(0);
    }
  });
  it('phaseTransfer sums useful bytes and expected requests; absent without transfer ops', () => {
    const stages = [
      stageResult(0, layers(3), send, 'ideal'),
      stageResult(1, layers(3), undefined, 'ideal'),
    ];
    expect(phaseTransfer(stages)).toEqual({
      sec: 6,
      exposedSec: 2,
      bytes: 2 * 21_233_664,
      requests: 2 * 8.5,
    });
    expect(phaseTransfer([stageResult(0, layers(0), send, 'ideal')])).toBeUndefined();
    const phase = phaseResult(stages, {
      phase: 'decode',
      dp: 1,
      sequencesPerReplica: 1,
      microbatches: 1,
      microbatchSize: 1,
      promptLen: 1,
      overlap: 'ideal',
    });
    expect(phase.transfer?.bytes).toBe(42_467_328);
  });
});

/**
 * Local copy of the stage time before the overlap across tiers (xfer_s = Σ op sec, P_s = Σ q·t_op), for the
 * bit-identity and t ≤ t_serial checks below. Same expressions and float order as the former `stageResult`.
 */
function serialStageSec(layers: LayerResult[], overlap: 'ideal' | 'none', sendSec = 0): number {
  const ops = layers.flatMap((l) => l.ops);
  const computeSec = ops
    .filter((o) => o.op.kind !== 'collective' && o.op.kind !== 'transfer')
    .reduce((s, o) => s + o.sec, 0);
  const transferSec = ops.filter((o) => o.op.kind === 'transfer').reduce((s, o) => s + o.sec, 0);
  let prefetchedSec = 0;
  let allPrefetched = true;
  let nonePrefetched = true;
  for (const o of ops) {
    if (o.op.kind !== 'transfer') continue;
    const q = o.op.prefetchedFraction;
    prefetchedSec += q * o.sec;
    if (q !== 1) allPrefetched = false;
    if (q !== 0) nonePrefetched = false;
  }
  const lat = ops.reduce((s, o) => s + o.commLatencySec, 0);
  const bw = ops.reduce((s, o) => s + o.commBandwidthSec, 0);
  const coreSec =
    overlap === 'none' ? computeSec + lat + bw + sendSec : lat + Math.max(computeSec, bw);
  const coreTermSec = overlap === 'none' ? computeSec + bw + sendSec : Math.max(computeSec, bw);
  if (transferSec === 0) return coreSec;
  if (allPrefetched) return lat + Math.max(coreTermSec, transferSec);
  if (nonePrefetched) return coreSec + transferSec;
  return (
    lat +
    coreTermSec +
    transferSec -
    (prefetchedSec / transferSec) * Math.min(coreTermSec, transferSec)
  );
}

describe('overlap across tiers (docs §12): a stage prefetching from host RAM and the SSD; core 2, lat 0.5 (seconds)', () => {
  const SSD_FETCH: TransferOp = { ...FETCH, source: 'nvme' };
  /** compute `core`, a collective (lat 0.5, bw 1) and the given transfer results. */
  const stage = (transfers: OpResult[], core = 2) => [
    layerResult('layer.0', 0, [
      result(ELEMENTWISE, { sec: core }),
      result(ALL_REDUCE, { sec: 1.5, lat: 0.5, bw: 1 }),
      ...transfers,
    ]),
  ];
  const ram = (q: number, sec = 1, link = 0.9, host = 0.5) =>
    result({ ...FETCH, prefetchedFraction: q }, { sec, link, host });
  const ssd = (q: number, sec = 3, link = 0.9, host = 1.2) =>
    result({ ...SSD_FETCH, prefetchedFraction: q }, { sec, link, host });

  it('p = 1, SSD-stream bound: RAM {1 s, link 0.9, host 0.5} + SSD {3 s, link 0.9, host 1.2} ⇒ P = 3, t = 3.5 (serial 4.5)', () => {
    const layers = stage([ram(1), ssd(1)]);
    const st = stageTransfer(layers[0].ops);
    expect(st).toEqual({
      transferSec: 3,
      prefetchedSec: 3,
      allPrefetched: true,
      nonePrefetched: false,
      overlapped: true,
    });
    const s = stageResult(0, layers, undefined, 'ideal');
    expect(s.transferSec).toBe(3);
    expect(s.sec).toBe(3.5); // lat + max(core, P)
    expect(serialStageSec(layers, 'ideal')).toBe(4.5);
  });
  it('p = 1, link bound: SSD 1.2 s ⇒ P = Σ link = 1.8', () => {
    const layers = stage([ram(1), ssd(1, 1.2)]);
    expect(stageTransfer(layers[0].ops).transferSec).toBe(1.8);
    expect(stageResult(0, layers, undefined, 'ideal').sec).toBe(2.5); // 0.5 + max(2, 1.8)
  });
  it('p = 1, host RAM bound: host 0.8 + 1.1, link 0.1 each ⇒ P = 1.9', () => {
    const layers = stage([ram(1, 1, 0.1, 0.8), ssd(1, 1.2, 0.1, 1.1)]);
    expect(stageTransfer(layers[0].ops).transferSec).toBeCloseTo(1.9, 12);
  });
  it('q 0.8 both: D = 0.8, P = 2.4, xfer 3.2, pbar 0.75 ⇒ t = 0.5 + 2 + 3.2 − 0.75·2 = 4.2 (serial 4.9)', () => {
    const layers = stage([ram(0.8), ssd(0.8)]);
    const st = stageTransfer(layers[0].ops);
    expect(st.overlapped).toBe(true);
    expect(st.allPrefetched || st.nonePrefetched).toBe(false);
    expect(st.transferSec).toBeCloseTo(3.2, 12);
    expect(st.prefetchedSec).toBeCloseTo(2.4, 12);
    expect(stageResult(0, layers, undefined, 'ideal').sec).toBeCloseTo(4.2, 12);
    expect(serialStageSec(layers, 'ideal')).toBeCloseTo(4.9, 12);
  });
  it('p = 0 with both tiers ≡ the serial formula, bit for bit', () => {
    for (const overlap of ['ideal', 'none'] as const) {
      const layers = stage([ram(0), ssd(0), ram(0, 0.37)]);
      const s = stageResult(0, layers, undefined, overlap);
      expect(stageTransfer(layers[0].ops).overlapped).toBe(false);
      expect(s.sec).toBe(serialStageSec(layers, overlap));
      expect(s.transferSec).toBe(1 + 3 + 0.37);
    }
  });
  it('one source (3 RAM ops at mixed q) ≡ the serial formula, bit for bit; SSD q > 0 with RAM q = 0 too', () => {
    for (const overlap of ['ideal', 'none'] as const) {
      for (const layers of [
        stage([ram(0.8, 0.3), ram(1, 0.7), ram(0.25, 1.9)]),
        stage([ram(0), ssd(0.8)]),
      ]) {
        expect(stageTransfer(layers[0].ops).overlapped).toBe(false);
        expect(stageResult(0, layers, undefined, overlap).sec).toBe(
          serialStageSec(layers, overlap),
        );
      }
    }
  });
  it('every core and q: lat + max(core, xfer) ≤ t ≤ t_core + xfer, and t ≤ t_serial', () => {
    for (const overlap of ['ideal', 'none'] as const) {
      for (const compute of [0.5, 2, 10]) {
        // ideal: core = max(compute, bw 1); none: compute + bw (no send)
        const core = overlap === 'ideal' ? Math.max(compute, 1) : compute + 1;
        const tCore = 0.5 + core;
        for (const q of [0, 0.3, 0.8, 1]) {
          for (const qSsd of [q, 1]) {
            const layers = stage([ram(q), ssd(qSsd), ram(q, 0.4, 0.3, 0.2)], compute);
            const s = stageResult(0, layers, undefined, overlap);
            expect(s.sec).toBeGreaterThanOrEqual(0.5 + Math.max(core, s.transferSec) - 1e-12);
            expect(s.sec).toBeLessThanOrEqual(tCore + s.transferSec + 1e-12);
            expect(s.sec).toBeLessThanOrEqual(serialStageSec(layers, overlap) + 1e-12);
          }
        }
      }
    }
  });
  it('a transfer result without transferResources is an error in an overlapped stage', () => {
    const bare = ssd(0.8);
    delete bare.transferResources;
    expect(() => stageTransfer(stage([ram(0.8), bare])[0].ops)).toThrow(/no transferResources/);
  });
});

describe('PhaseResult.comm: the latency vs bandwidth split equals the hand sums of design §8.3 (display only)', () => {
  const run = (over: Partial<SimInput>) =>
    simulate(
      input({
        hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
        workload: { batch: 1, promptLen: 4096, decodeContextLen: 1024 },
        ...over,
      }),
    );
  /** The comm split rebuilt from the stage fields and the sends. */
  const handSums = (stages: StageResult[], overlap: 'ideal' | 'none') => {
    let latencySec = 0;
    let bandwidthSec = 0;
    let exposedSec = 0;
    for (const s of stages) {
      latencySec += s.commLatencySec + (s.sendToNext?.commLatencySec ?? 0);
      bandwidthSec += s.commBandwidthSec + (s.sendToNext?.commBandwidthSec ?? 0);
      exposedSec += s.exposedCommSec + (overlap === 'ideal' ? (s.sendToNext?.sec ?? 0) : 0);
    }
    return { latencySec, bandwidthSec, exposedSec };
  };

  it('G-b decode (tp 4): 292 µs AR + 3 µs sampling AG, β = 11.96032 + 0.00024 µs', () => {
    const r = run({ parallel: { tp: 4, pp: 1, dp: 1, ep: 1 } });
    expectClose(r.decode.comm.latencySec, 295e-6);
    expectClose(r.decode.comm.bandwidthSec, (11.96032 + 0.00024) * 1e-6);
    // pp 1, ideal: exposed = lat + max(compute, bw) − compute = lat (bw hides)
    expectClose(r.decode.comm.exposedSec, 295e-6);
  });
  it('G-c decode: stage α 147 µs (146 AR + 1 AG), forward 1 µs, return 1 µs', () => {
    const r = run({
      parallel: { tp: 2, pp: 2, dp: 1, ep: 1 },
      workload: { batch: 2, promptLen: 4096, decodeContextLen: 1024 },
    });
    const [s0, s1] = r.decode.stages;
    expectClose(s0.commLatencySec + s1.commLatencySec, 147e-6);
    expectClose(s0.sendToNext?.commLatencySec ?? NaN, 1e-6);
    expectClose(s0.sendToNext?.commBandwidthSec ?? NaN, 0.08192e-6);
    expectClose(r.decode.comm.latencySec, 149e-6);
    expectClose(
      r.decode.comm.bandwidthSec,
      s0.commBandwidthSec + s1.commBandwidthSec + 0.08192e-6 + 4 / 100e9,
    );
    // the send is on the latency path under ideal: exposed = Σ lat_s + send α + β
    expectClose(r.decode.comm.exposedSec, 147e-6 + 2e-6 + 0.08192e-6 + 4 / 100e9);
  });
  it('G-g decode (tp8 pp2): stage α 438 + 7 AG µs, forward 5 µs, return 5 µs = 455 µs', () => {
    const r = run({ parallel: { tp: 8, pp: 2, dp: 1, ep: 1 } });
    expectClose(r.decode.comm.latencySec, 455e-6);
  });
  it('G-e (ep 2 dp 2, mode B all-to-alls) and G-f (tp 2 pp 2): match the hand sums, both overlap modes', () => {
    const cases: Partial<SimInput>[] = [
      {
        model: TINY_HYBRID,
        parallel: { tp: 1, pp: 1, dp: 2, ep: 2 },
        workload: { batch: 4, promptLen: 512, decodeContextLen: 1024 },
      },
      {
        model: TINY_HYBRID,
        parallel: { tp: 2, pp: 2, dp: 1, ep: 1 },
        workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 },
      },
    ];
    for (const c of cases)
      for (const overlap of ['ideal', 'none'] as const) {
        const r = run({ ...c, options: { overlap } });
        for (const phase of [r.prefill, r.decode]) {
          const hand = handSums(phase.stages, overlap);
          hand.latencySec += phase.loopBack?.commLatencySec ?? 0;
          hand.bandwidthSec += phase.loopBack?.commBandwidthSec ?? 0;
          hand.exposedSec += phase.loopBack?.sec ?? 0;
          for (const key of ['latencySec', 'bandwidthSec', 'exposedSec'] as const)
            expectClose(phase.comm[key], hand[key]);
          expect(phaseComm(phase.stages, overlap, phase.loopBack)).toEqual(phase.comm);
          expect(phase.transfer).toBeUndefined();
          expect(phase.speculative).toBeUndefined();
        }
      }
    expectClose(run(cases[0]).decode.comm.latencySec, 8e-6); // 4 dispatch + 4 combine a2a, α 1 µs
    expectClose(run(cases[1]).decode.comm.latencySec, 18e-6 + 1e-6 + 2e-6); // 9 AR + sampling AG + forward and return
  });
});
