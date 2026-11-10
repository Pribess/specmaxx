/**
 * Chunked pipeline prefill (docs/simulation-model.md §7 "Chunked pipeline prefill"): the chunk partition, the
 * schedule recursion against hand-computed values and the whole-prompt formula, the conservation of attention pairs,
 * GEMM FLOP and PP send bytes over the chunks, and the knob on MLA, hybrid, two-batch-overlap, spec-dec, prefix-cache
 * and 1M-token runs.
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import { B300, composeTopology } from '../hw/presets';
import { hostPcieLink } from '../hw/presets/links';
import { DDR4_3200_2CH } from '../hw/presets/storage';
import { attentionPairs } from '../ir/metrics';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_DENSE_8B, REF_MOE_30B_A3B, TINY_HYBRID, TINY_KIMI } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import {
  CHUNKED_PIPELINE_SINGLE_STAGE_WARNING,
  chunkedPipelineCycleSec,
  chunkedPipelineLatencySec,
  chunkedPipelinePlan,
  chunkStageTimes,
  prefillPipelineChunks,
  type ChunkStageTimes,
} from './chunkedPipeline';
import { simulate } from './simulate';
import type {
  DeploymentConfig,
  KvCacheOffloadConfig,
  OpResult,
  PhaseResult,
  SimInput,
  SimResult,
} from './types';
import { validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const CLUSTER = { device: SYNTH_GPU, topology: SYNTH_CLUSTER };

interface Case {
  model?: ModelSpec;
  parallel?: Partial<ParallelConfig>;
  batch?: number;
  promptLen?: number;
  chunkTokens?: number;
  prefixCacheHitRatio?: number;
  deployment?: DeploymentConfig;
}

function input(c: Case, chunked: boolean): SimInput {
  return {
    model: c.model ?? REF_DENSE_8B,
    quant: BF16,
    parallel: {
      tp: 1,
      pp: 4,
      dp: 1,
      ep: 1,
      ...c.parallel,
      ...(chunked ? { chunkedPipelinePrefill: true } : {}),
    },
    workload: {
      batch: c.batch ?? 1,
      promptLen: c.promptLen ?? 65_536,
      decodeContextLen: 1024,
      ...(c.chunkTokens === undefined ? {} : { prefillChunkTokens: c.chunkTokens }),
      ...(c.prefixCacheHitRatio === undefined
        ? {}
        : { prefixCacheHitRatio: c.prefixCacheHitRatio }),
    },
    hardware: CLUSTER,
    ...(c.deployment === undefined ? {} : { deployment: c.deployment }),
  };
}

/** The same case with chunked pipeline prefill off and on. */
function both(c: Case): { off: SimResult; on: SimResult } {
  return { off: simulate(input(c, false)), on: simulate(input(c, true)) };
}

const ops = (p: PhaseResult): OpResult[] => p.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
const sumOf = (rs: OpResult[], f: (r: OpResult) => number): number =>
  rs.reduce((s, r) => s + f(r), 0);
/** Relative difference |a − b| / |b|. */
const rel = (a: number, b: number): number => Math.abs(a - b) / Math.abs(b);

describe('chunk partition', () => {
  it('P = 20,000, C = 8192: chunks 8192 | 8192 | 3616 with kv 8192 | 16,384 | 20,000', () => {
    expect(prefillPipelineChunks(20_000, 0, 8192)).toEqual([
      { qLen: 8192, kvLen: 8192 },
      { qLen: 8192, kvLen: 16_384 },
      { qLen: 3616, kvLen: 20_000 },
    ]);
  });

  it('with 5000 cached tokens: P_new = 15,000 → 8192 | 6808 new tokens over kv 13,192 | 20,000', () => {
    expect(prefillPipelineChunks(20_000, 5000, 8192)).toEqual([
      { qLen: 8192, kvLen: 13_192 },
      { qLen: 6808, kvLen: 20_000 },
    ]);
  });

  it('plan: only with the flag, pp > 1 and K > 1; C defaults to 8192', () => {
    const w = { promptLen: 20_000 };
    expect(chunkedPipelinePlan({ pp: 2 }, w)).toBeUndefined();
    expect(chunkedPipelinePlan({ pp: 2, chunkedPipelinePrefill: false }, w)).toBeUndefined();
    expect(chunkedPipelinePlan({ pp: 1, chunkedPipelinePrefill: true }, w)).toBeUndefined();
    expect(
      chunkedPipelinePlan({ pp: 2, chunkedPipelinePrefill: true }, { promptLen: 8192 }),
    ).toBeUndefined();
    const plan = chunkedPipelinePlan({ pp: 2, chunkedPipelinePrefill: true }, w);
    expect(plan?.chunkTokens).toBe(8192);
    expect(plan?.chunks.length).toBe(3);
    // A prefix-cache hit counts only the new tokens: P_new = 20,000 − 12,000 = 8000 ≤ C, one chunk.
    expect(
      chunkedPipelinePlan(
        { pp: 2, chunkedPipelinePrefill: true },
        { promptLen: 20_000, prefixCacheHitRatio: 0.6 },
      ),
    ).toBeUndefined();
  });
});

describe('schedule (hand-computed)', () => {
  // 2 stages × 3 chunks: t_0 = 1, 2, 3; t_1 = 2, 1, 1 s; each send p = 0.5 s of which β = 0.3 s (α = 0.2 s).
  const units: ChunkStageTimes[] = [
    { stageSec: [1, 2], sendSec: [0.5], sendBandwidthSec: [0.3] },
    { stageSec: [2, 1], sendSec: [0.5], sendBandwidthSec: [0.3] },
    { stageSec: [3, 1], sendSec: [0.5], sendBandwidthSec: [0.3] },
  ];

  it('f_pp = 0: F_1 = 3, 4, 7 → TTFT 7 s (sends inside the stage times)', () => {
    expect(chunkedPipelineLatencySec(units, 1, 0)).toBeCloseTo(7, 12);
  });

  it('f_pp = 1: λ = 0.3, δ = 0.2 → F_1 = 3.5, 4.5, 7.5 → TTFT 7.5 s', () => {
    expect(chunkedPipelineLatencySec(units, 1, 1)).toBeCloseTo(7.5, 12);
  });

  it('f_pp = 0.5: λ = 0.15, δ = 0.1 → F_1 = 3.25, 4.25, 7.25 → TTFT 7.25 s', () => {
    expect(chunkedPipelineLatencySec(units, 1, 0.5)).toBeCloseTo(7.25, 12);
  });

  it('m = 2 at f_pp = 1: the second microbatch ends at 13.5 s; cycle = m·max(Σt_0 6, Σt_1 4, Σβ 0.9)', () => {
    expect(chunkedPipelineLatencySec(units, 2, 1)).toBeCloseTo(13.5, 12);
    expect(chunkedPipelineCycleSec(units, 1)).toBe(6);
    expect(chunkedPipelineCycleSec(units, 2)).toBe(12);
  });

  it('a link busier than every stage sets the cycle: m·Σ_j β', () => {
    const slowLink = units.map((u) => ({ ...u, sendSec: [5], sendBandwidthSec: [4.9] }));
    expect(chunkedPipelineCycleSec(slowLink, 2)).toBeCloseTo(2 * 3 * 4.9, 12);
  });

  it('equal chunks t on every stage, no sends: TTFT = (K + pp − 1)·t vs pp·K·t whole-prompt', () => {
    const K = 5;
    const pp = 3;
    const t = 2;
    const chunk: ChunkStageTimes = {
      stageSec: Array<number>(pp).fill(t),
      sendSec: Array<number>(pp - 1).fill(0),
      sendBandwidthSec: Array<number>(pp - 1).fill(0),
    };
    expect(chunkedPipelineLatencySec(Array(K).fill(chunk), 1, 1)).toBe((K + pp - 1) * t); // 14 s
    const whole = { ...chunk, stageSec: Array<number>(pp).fill(K * t) };
    expect(chunkedPipelineLatencySec([whole], 1, 1)).toBe(pp * K * t); // 30 s
  });

  it.each([
    ['ideal', 1],
    ['none', 0],
  ] as const)(
    'K = 1 equals the whole-prompt formula of phaseResult (pp 2, m 2, overlap %s)',
    (overlap, f) => {
      const run = simulate({
        ...input({ parallel: { pp: 2, tp: 2 }, batch: 4, promptLen: 4096 }, false),
        options: { overlap },
      });
      const times = [chunkStageTimes(run.prefill.stages)];
      expect(run.prefill.microbatches).toBe(2);
      expect(rel(chunkedPipelineLatencySec(times, 2, f), run.prefill.latencySec)).toBeLessThan(
        1e-12,
      );
      expect(rel(chunkedPipelineCycleSec(times, 2), run.prefill.cycleSec)).toBeLessThan(1e-12);
    },
  );
});

describe('REF_DENSE_8B, pp 4, P = 64k, C = 8k', () => {
  const { off, on } = both({});
  const cpp = on.prefill.chunkedPipeline;

  it('reports C, K = 8 and the whole-prompt rates', () => {
    expect(cpp).toEqual({
      chunkTokens: 8192,
      chunks: 8,
      unchunked: {
        tokensPerSec: off.prefill.tokensPerSec,
        perSequenceTokensPerSec: off.prefill.perSequenceTokensPerSec,
        latencySec: off.prefill.latencySec,
        cycleSec: off.prefill.cycleSec,
      },
    });
    expect(off.prefill.chunkedPipeline).toBeUndefined();
  });

  it('Σ attention pairs and FLOP over the chunks equal the whole prompt', () => {
    const attention = (p: PhaseResult) => ops(p).filter((r) => r.op.kind === 'attention');
    // Merged ops keep the last chunk's shape, so pairs are summed per chunk from the chunk partition.
    const P = 65_536;
    const pairsPerChunk = prefillPipelineChunks(P, 0, 8192).map(
      (c) => c.qLen * (c.kvLen - c.qLen) + (c.qLen * (c.qLen + 1)) / 2,
    );
    expect(pairsPerChunk.reduce((s, x) => s + x, 0)).toBe((P * (P + 1)) / 2);
    const wholeOp = attention(off.prefill)[0].op;
    if (wholeOp.kind !== 'attention') throw new Error('attention');
    expect(attentionPairs(wholeOp)).toBe((P * (P + 1)) / 2);
    const flops = (p: PhaseResult) => sumOf(attention(p), (r) => r.flops);
    expect(rel(flops(on.prefill), flops(off.prefill))).toBeLessThan(1e-12);
  });

  it('Σ GEMM FLOP of the non-head layers equal the whole prompt (GEMM rows m = Σ_j q_j = P)', () => {
    const gemmFlops = (p: PhaseResult) =>
      sumOf(
        p.stages.flatMap((s) => s.layers.filter((l) => l.name !== 'head').flatMap((l) => l.ops)),
        (r) => (r.op.kind === 'gemm' ? r.flops : 0),
      );
    expect(rel(gemmFlops(on.prefill), gemmFlops(off.prefill))).toBeLessThan(1e-12);
    // The head runs once, on the last chunk: the same last-position logits as the whole prompt.
    const head = (r: SimResult) => r.prefill.stages[3].layers.find((l) => l.name === 'head');
    expect(head(on)?.ops.map((o) => o.flops)).toEqual(head(off)?.ops.map((o) => o.flops));
  });

  it('Σ PP send bytes and β over the chunks equal the whole-prompt sends', () => {
    for (let s = 0; s < 3; s++) {
      const a = on.prefill.stages[s].sendToNext;
      const b = off.prefill.stages[s].sendToNext;
      if (a?.op.kind !== 'collective' || b?.op.kind !== 'collective') throw new Error('send');
      expect(a.op.messageBytes).toBe(b.op.messageBytes);
      expect(rel(a.commBandwidthSec, b.commBandwidthSec)).toBeLessThan(1e-12);
    }
  });

  it('TTFT 22.31 s → 8.74 s; cycle 5.574 s → 5.596 s (docs worked example), within 5%', () => {
    // Each chunk reads every weight once more, re-reads every earlier chunk's KV and runs GEMMs of C rows, so the
    // cycle (the steady-stream throughput) moves a little; the TTFT gains up to pp× (ideal 11/32 here).
    expect(off.prefill.latencySec).toBeCloseTo(22.31, 2);
    expect(on.prefill.latencySec).toBeCloseTo(8.74, 2);
    expect(off.prefill.cycleSec).toBeCloseTo(5.574, 3);
    expect(on.prefill.cycleSec).toBeCloseTo(5.596, 3);
    expect(rel(on.prefill.cycleSec, off.prefill.cycleSec)).toBeLessThan(0.05);
    expect(on.prefill.perSequenceTokensPerSec).toBeCloseTo(65_536 / on.prefill.latencySec, 6);
  });

  it('decode and the warnings are unchanged; the merged stages have about one chunk of ops', () => {
    expect(on.decode).toEqual(off.decode);
    expect(on.warnings).toEqual(off.warnings);
    expect(ops(on.prefill).length).toBeLessThan(2 * ops(off.prefill).length);
  });

  it('prefill without overlap is the chunked schedule under every fraction 0', () => {
    expect(on.prefillWithoutOverlap).toBeDefined();
    expect(on.prefillWithoutOverlap!.latencySec).toBeGreaterThan(on.prefill.latencySec);
    expect(on.prefillWithoutOverlap!.latencySec).toBeLessThan(
      off.prefillWithoutOverlap!.latencySec,
    );
  });
});

describe('model and deployment variants', () => {
  // The tiny models put almost all prefill time in one stage's attention, and their GEMMs at a few thousand rows lose
  // shape efficiency, so they gain little; with C = 1024 they even lose (each chunk's GEMMs run at m = 1024 and
  // re-read the weights). C = 4096 over a 64k prompt keeps the gain.
  it('MLA (dense prefill, TINY_KIMI with KDA): kv_b decompresses every visible key per chunk, Σ_j kv_j rows', () => {
    const P = 65_536;
    const C = 4096;
    const { off, on } = both({
      model: TINY_KIMI,
      parallel: { pp: 2 },
      promptLen: P,
      chunkTokens: C,
    });
    const kvb = (p: PhaseResult) => sumOf(ops(p), (r) => (r.op.name === 'mla.kv_b' ? r.flops : 0));
    const sumKv = prefillPipelineChunks(P, 0, C).reduce((s, c) => s + c.kvLen, 0);
    expect(sumKv / P).toBe(8.5); // (K + 1)/2 with K = 16
    expect(rel(kvb(on.prefill) / kvb(off.prefill), sumKv / P)).toBeLessThan(1e-12);
    expect(on.prefill.latencySec).toBeLessThan(off.prefill.latencySec);
  });

  it('hybrid linear attention (TINY_HYBRID, pp 2): later chunks start from the prior state; TTFT drops', () => {
    const { off, on } = both({
      model: TINY_HYBRID,
      parallel: { pp: 2 },
      promptLen: 65_536,
      chunkTokens: 4096,
    });
    expect(on.prefill.chunkedPipeline?.chunks).toBe(16);
    expect(on.prefill.latencySec).toBeLessThan(off.prefill.latencySec);
  });

  it('two-batch overlap (REF_MOE_30B_A3B, pp 2, ep 2 over dp 2): each chunk split in halves; TTFT drops', () => {
    const { off, on } = both({
      model: REF_MOE_30B_A3B,
      parallel: { pp: 2, dp: 2, ep: 2 },
      batch: 2,
      promptLen: 32_768,
      deployment: { twoBatchOverlap: true },
    });
    expect(on.prefill.chunkedPipeline?.chunks).toBe(4);
    expect(on.prefill.latencySec).toBeLessThan(off.prefill.latencySec);
    expect(on.prefillWithoutOverlap).toBeDefined();
    expect(on.prefillWithoutOverlap!.latencySec).toBeGreaterThan(on.prefill.latencySec);
  });

  it('spec-dec MTP with pp 2: the drafter runs on every chunk, after the head on the last', () => {
    const { off, on } = both({
      model: REF_DENSE_8B_MTP,
      parallel: { pp: 2 },
      promptLen: 32_768,
      deployment: {
        speculative: { method: 'mtp', draftLength: 2, acceptance: { kind: 'iid', alpha: 0.8 } },
      },
    });
    const names = on.prefill.stages[1].layers.map((l) => l.name);
    expect(names.indexOf('head')).toBeLessThan(names.findIndex((n) => n.startsWith('draft.')));
    expect(on.prefill.latencySec).toBeLessThan(off.prefill.latencySec);
    expect(on.decode).toEqual(off.decode);
  });

  it('prefix cache h = 0.5: chunks over P_new, Σ attention FLOP equal, computed tok/s kept', () => {
    const { off, on } = both({ promptLen: 65_536, prefixCacheHitRatio: 0.5 });
    expect(on.prefill.chunkedPipeline?.chunks).toBe(4);
    const attnFlops = (p: PhaseResult) =>
      sumOf(ops(p), (r) => (r.op.kind === 'attention' ? r.flops : 0));
    expect(rel(attnFlops(on.prefill), attnFlops(off.prefill))).toBeLessThan(1e-12);
    expect(on.prefill.computedTokensPerSec).toBeCloseTo(on.prefill.tokensPerSec / 2, 6);
    expect(on.prefill.latencySec).toBeLessThan(off.prefill.latencySec);
  });

  it('b_mb > 1: the memory fit counts a pipeline chunk of b_mb sequences × C tokens', () => {
    const { off, on } = both({ parallel: { pp: 2 }, batch: 8, promptLen: 32_768 });
    expect(on.prefill.microbatchSize).toBe(4);
    for (let s = 0; s < 2; s++)
      expect(on.memory.perStage[s].activationPeakBytes).toBeGreaterThan(
        off.memory.perStage[s].activationPeakBytes,
      );
  });

  it('pp = 1: ignored with a warning, the rest unchanged', () => {
    const { off, on } = both({ parallel: { pp: 1 } });
    expect(on.warnings).toEqual([...off.warnings, CHUNKED_PIPELINE_SINGLE_STAGE_WARNING]);
    expect({ ...on, warnings: off.warnings }).toEqual(off);
  });

  it('validation: a non-boolean flag is an error', () => {
    const bad = input({}, false);
    (bad.parallel as { chunkedPipelinePrefill?: unknown }).chunkedPipelinePrefill = 1;
    expect(validateInput(bad)).toContain(
      'parallel.chunkedPipelinePrefill must be true or false (got 1)',
    );
  });
});

describe('offload: the chunks of a microbatch stream from the node host RAM at the same time', () => {
  // Review P1 repro: REF_MOE_30B_A3B with every routed expert in host RAM (DDR4-3200 × 2, 51.2 GB/s per node, behind
  // PCIe 5.0 x16), 4 PP stages on one 8-GPU node, P = 32,768, C = 1024 (K = 32). Before the fix each chunk was costed
  // as if its stage streamed alone (m = 1), so TTFT (12.04 s) was below the node's RAM time for the bytes (36.24 s).
  const B_RAM = DDR4_3200_2CH.bandwidthBytesPerSec;
  const run = (chunked: boolean, prefetchAccuracy: number): SimResult =>
    simulate({
      ...input(
        { model: REF_MOE_30B_A3B, parallel: { pp: 4 }, promptLen: 32_768, chunkTokens: 1024 },
        chunked,
      ),
      hardware: { ...CLUSTER, host: { ram: DDR4_3200_2CH, link: hostPcieLink(5, 16) } },
      deployment: {
        offload: {
          prefetchAccuracy,
          routedExperts: { tier: 'host_ram', residentFraction: 0, hitRatio: { kind: 'uniform' } },
        },
      },
    });
  /** Bytes the node's host RAM serves for one replica batch's prefill (every stage is on the node). Unit: bytes. */
  const hostRamBytes = (p: PhaseResult): number =>
    sumOf(ops(p), (r) => r.transferResources?.hostRamBytes ?? 0);

  it.each([0, 0.5, 1])(
    'p = %s: TTFT ≥ node host-RAM bytes / B_ram, chunked and whole-prompt',
    (prefetchAccuracy) => {
      const on = run(true, prefetchAccuracy);
      const off = run(false, prefetchAccuracy);
      expect(on.prefill.chunkedPipeline?.chunks).toBe(32);
      const boundSec = hostRamBytes(on.prefill) / B_RAM;
      expect(boundSec).toBeGreaterThan(30); // 1.855e12 B / 51.2e9 B/s = 36.24 s
      expect(on.prefill.latencySec).toBeGreaterThanOrEqual(boundSec);
      expect(off.prefill.latencySec).toBeGreaterThanOrEqual(hostRamBytes(off.prefill) / B_RAM);
      // The steady stream too: one cycle serves the m = 1 microbatch's bytes.
      expect(on.prefill.cycleSec).toBeGreaterThanOrEqual(boundSec);
    },
  );
});

describe('KV-cache offload: each chunk reads the off-device prefix and earlier-chunk KV', () => {
  // REF_DENSE_8B bf16 (36 layers, kvB = 2·8·128·2 = 4096 B per token), pp 4 on one node, DDR4-3200 × 2 host RAM.
  const KVB = 4096;
  const run = (
    chunked: boolean,
    kvCache: KvCacheOffloadConfig,
    prefixCacheHitRatio?: number,
    memCapacityBytes?: number,
  ): SimResult =>
    simulate({
      ...input(
        {
          promptLen: 65_536,
          ...(prefixCacheHitRatio === undefined ? {} : { prefixCacheHitRatio }),
        },
        chunked,
      ),
      hardware: {
        ...CLUSTER,
        host: { ram: DDR4_3200_2CH, link: hostPcieLink(5, 16) },
        ...(memCapacityBytes === undefined
          ? {}
          : {
              device: {
                ...SYNTH_GPU,
                memory: { ...SYNTH_GPU.memory, capacityOptionsBytes: [memCapacityBytes] },
              },
            }),
      },
      deployment: { offload: { kvCache } },
    });
  const kvFetchBytes = (p: PhaseResult): number =>
    sumOf(ops(p), (r) =>
      r.op.kind === 'transfer' && r.op.group === 'kv_cache' ? r.op.usefulBytes : 0,
    );

  it('live KV 25% on the device (review repro, 6 GB devices): later chunks fetch 75% of the earlier chunks', () => {
    // The fit keeps f_dev = 0.25 of each sequence's KV on the device, so the earlier chunks' KV is placed the same way:
    // chunk j fetches (1 − f_dev)·j·C tokens per layer. Σ_j over K = 8: 0.75·(0 + 1 + … + 7)·8192 = 0.75·28·8192.
    const kvCache = { deviceFraction: 0.25 };
    const on = run(true, kvCache, undefined, 6e9);
    const off = run(false, kvCache, undefined, 6e9);
    expect(on.prefill.chunkedPipeline?.chunks).toBe(8);
    expect(kvFetchBytes(on.prefill)).toBeCloseTo(0.75 * 28 * 8192 * 36 * KVB, 0);
    // The whole prompt has no past at prefill: nothing to fetch.
    expect(kvFetchBytes(off.prefill)).toBe(0);
    // No false fit: the device holds f_dev of the KV, chunked or not (stage 0: 9 layers × 0.25·65,536 tokens), and the
    // most a chunk keeps resident from earlier chunks, f_dev·7·C tokens, is within it.
    const residentKv = 0.25 * 65_536 * KVB * 9;
    expect(on.memory.perStage[0].sequenceStateBytes).toBeCloseTo(residentKv, 0);
    expect(0.25 * 7 * 8192 * KVB * 9).toBeLessThan(residentKv);
    expect(on.memory.perStage[0].sequenceStateBytes).toBe(
      off.memory.perStage[0].sequenceStateBytes,
    );
    // The node's host RAM serves the fetches: TTFT ≥ bytes / B_ram.
    expect(on.prefill.latencySec).toBeGreaterThanOrEqual(
      kvFetchBytes(on.prefill) / DDR4_3200_2CH.bandwidthBytesPerSec,
    );
    expect(on.decode).toEqual(off.decode);
    // Staging: the last chunk fetches the most, 0.75·57,344·4096 B = 176,160,768 B per layer, double-buffered at the
    // default prefetch accuracy (q > 0): 352,321,536 B on every stage. Weights + resident KV + that do not fit 6 GB.
    for (const st of on.memory.perStage) expect(st.stagingBytes).toBe(2 * 0.75 * 57_344 * KVB);
    expect(on.memory.perStage[0].totalBytes).toBeGreaterThan(6e9);
    expect(on.memory.fits).toBe(false);
    expect(off.memory.fits).toBe(true); // whole-prompt prefill fetches nothing at prefill
    // The sweep's path (no chunk schedule) gives the same fit.
    const sweep = simulate(
      {
        ...input({ promptLen: 65_536 }, true),
        hardware: {
          ...CLUSTER,
          host: { ram: DDR4_3200_2CH, link: hostPcieLink(5, 16) },
          device: { ...SYNTH_GPU, memory: { ...SYNTH_GPU.memory, capacityOptionsBytes: [6e9] } },
        },
        deployment: { offload: { kvCache } },
      },
      { skipChunkedPrefillSchedule: true },
    );
    expect(sweep.memory).toEqual(on.memory);
  });

  it('prefix in host RAM (h 0.5, f_pre 0): every one of the K = 4 chunks re-reads the 32k-token prefix', () => {
    const kvCache = { deviceFraction: 1, prefixDeviceFraction: 0 };
    const on = run(true, kvCache, 0.5);
    const off = run(false, kvCache, 0.5);
    expect(on.prefill.chunkedPipeline?.chunks).toBe(4);
    // Whole prompt: 36 layers × 32,768 prefix tokens × 4096 B, once.
    expect(kvFetchBytes(off.prefill)).toBe(36 * 32_768 * KVB);
    expect(kvFetchBytes(on.prefill)).toBe(4 * 36 * 32_768 * KVB);
    // The node's host RAM serves all of it: TTFT ≥ bytes / B_ram.
    const boundSec = kvFetchBytes(on.prefill) / DDR4_3200_2CH.bandwidthBytesPerSec;
    expect(on.prefill.latencySec).toBeGreaterThanOrEqual(boundSec);
    expect(on.decode).toEqual(off.decode);
  });
});

describe('skipChunkedPrefillSchedule (the UI batch sweep)', () => {
  it('pp 4, batch 16 (b_mb 4), P = 64k: memory and decode bit-identical to the full run, prefill whole-prompt', () => {
    const c: Case = { batch: 16, promptLen: 65_536 };
    const full = simulate(input(c, true));
    const skipped = simulate(input(c, true), { skipChunkedPrefillSchedule: true });
    expect(full.prefill.microbatchSize).toBe(4);
    expect(skipped.memory).toEqual(full.memory);
    expect(skipped.decode).toEqual(full.decode);
    // The chunked fit differs from the flag-off fit here (the pipeline chunk holds b_mb sequences).
    const off = simulate(input(c, false));
    expect(skipped.memory).not.toEqual(off.memory);
    expect(skipped.prefill).toEqual(off.prefill);
    expect(skipped.prefill.chunkedPipeline).toBeUndefined();
    expect(skipped.prefillWithoutOverlap).toEqual(off.prefillWithoutOverlap);
  });

  it('no effect without chunked pipeline prefill', () => {
    const c: Case = { batch: 16, promptLen: 65_536 };
    expect(simulate(input(c, false), { skipChunkedPrefillSchedule: true })).toEqual(
      simulate(input(c, false)),
    );
  });
});

describe('1M context: Kimi-K3 on 4 × 8 B300, tp 8, pp 4, ep 8, batch 1', () => {
  it('K = 128 chunks; fits; TTFT below the whole-prompt TTFT; runs in seconds', () => {
    const mapped = mapHfModel(fixtureRepoFiles('moonshotai/Kimi-K3'));
    if (mapped.kind !== 'ok') throw new Error(mapped.kind);
    const run = (chunked: boolean) =>
      simulate({
        model: mapped.model,
        quant: mapped.quant.quant,
        parallel: {
          tp: 8,
          pp: 4,
          dp: 1,
          ep: 8,
          ...(chunked ? { chunkedPipelinePrefill: true } : {}),
        },
        workload: { batch: 1, promptLen: 1_048_576, decodeContextLen: 8192 },
        hardware: {
          device: B300,
          topology: composeTopology({ deviceId: 'b300', devicesPerNode: 8, nodes: 4 }),
        },
      });
    const started = performance.now();
    const on = run(true);
    const wallSec = (performance.now() - started) / 1000;
    const off = run(false);
    expect(on.prefill.chunkedPipeline?.chunks).toBe(128);
    expect(on.memory.fits).toBe(true);
    expect(Number.isFinite(on.prefill.tokensPerSec)).toBe(true);
    expect(on.prefill.latencySec).toBeLessThan(off.prefill.latencySec);
    expect(on.prefill.chunkedPipeline?.unchunked.latencySec).toBe(off.prefill.latencySec);
    // Wall-time guard (about 0.6–0.8 s on an M-class laptop; generous for CI).
    expect(wallSec).toBeLessThan(10);
  }, 30_000);
});
