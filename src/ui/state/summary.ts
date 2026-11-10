/**
 * `summarize`: a `SimResult` reduced to what the UI shows (headline rates, memory per stage, time breakdowns,
 * roofline points, warnings). Small and plain data, so the worker can post it. Pure; no React, no DOM.
 *
 * Conventions: times in seconds, rates in tokens/s, FLOP in FLOP, bytes in bytes, all per device (the engine's
 * convention: one device of each pipeline stage, one microbatch step). Breakdown times are Σ op `sec` before
 * comm/compute overlap, so they can exceed the stage time.
 */

import type {
  LayerRange,
  KvTier,
  OpResult,
  PhaseResult,
  SimInput,
  SimResult,
  TierMemory,
} from '../../sim/engine/types';
import { engineMemBandwidthBytesPerSec } from '../../sim/engine/opTime';
import { hitRatio } from '../../sim/engine/transfer';
import { expertRules, lookupTableRules } from '../../sim/engine/offloadRules';
import { memoryFits } from './memoryVerdict';
import type { Op } from '../../sim/ir/ops';
import {
  poolInput,
  type BalanceResult,
  type DisaggregatedInput,
  type DisaggregatedResult,
  type PoolRates,
  type SystemResult,
} from '../../sim/engine/disaggregation';
import type { KvTransferResult } from '../../sim/engine/kvTransfer';

/** Headline numbers of one phase. */
export interface PhaseSummary {
  /** Prefill prompt length or tokens already cached for decode, per sequence. Unit: tokens. */
  contextTokens: number;
  /** Aggregate rate over all DP replicas. Unit: tokens/s. */
  tokensPerSec: number;
  /** Prefill with a prefix-cache hit: computed (not cached) tokens per second; absent otherwise. Unit: tokens/s. */
  computedTokensPerSec?: number;
  /** Rate seen by one sequence. Unit: tokens/s. */
  perSequenceTokensPerSec: number;
  /** Prefill: time to first token of a replica batch; decode: time per output token. Unit: seconds. */
  latencySec: number;
  /** Pipeline cycle. Unit: seconds. */
  cycleSec: number;
  /** Microbatches in flight per replica. Count. */
  microbatches: number;
  /** Sequences per microbatch. Count. */
  microbatchSize: number;
  /** Stage time per microbatch, per stage. Unit: seconds. */
  stageSec: number[];
  /**
   * Communication on the critical path (`PhaseResult.comm`): Σ latency (α) terms, Σ bandwidth (β) terms, and the
   * part not hidden behind compute. Display only. Unit: seconds.
   */
  comm: { latencySec: number; bandwidthSec: number; exposedSec: number };
  /**
   * Off-device transfers (`PhaseResult.transfer`); absent when nothing is offloaded. `sec`, `exposedSec`: seconds;
   * `bytes`: useful bytes over the host link (bytes); `requests`: expected read requests (count, real).
   */
  transfer?: { sec: number; exposedSec: number; bytes: number; requests: number };
  /**
   * Decode with speculative decoding (`PhaseResult.speculative`); absent otherwise. `draftSec` / `verifySec`: shares
   * of one step (seconds); `expectedTokensPerStep`: E[τ] (tokens per sequence per step, incl. the bonus token);
   * `offTokensPerSec`: the rate with spec-dec off (tokens/s); `offCycleSec`: its cycle (seconds); `stepCostRatio`:
   * cycle over the off cycle (dimensionless); `speedup`: tokens/s over tokens/s with spec-dec off
   * (= E[τ] / stepCostRatio, dimensionless).
   */
  speculative?: {
    draftSec: number;
    verifySec: number;
    expectedTokensPerStep: number;
    offTokensPerSec: number;
    offCycleSec: number;
    stepCostRatio: number;
    speedup: number;
  };
  /**
   * Prefill only: the same prefill without communication/compute overlap (`SimResult.prefillWithoutOverlap`), a
   * labelled what-if; absent without inter-device communication. Rates in tokens/s.
   */
  withoutOverlap?: { tokensPerSec: number; perSequenceTokensPerSec: number };
  /**
   * Prefill only, with chunked pipeline prefill on (`PhaseResult.chunkedPipeline`); absent otherwise. `chunkTokens`:
   * chunk size C (tokens); `chunks`: chunks per microbatch K (count); `unchunkedLatencySec`: the time to first token
   * with whole-prompt microbatches (seconds), next to this phase's `latencySec` (the chunked one).
   */
  chunkedPipeline?: { chunkTokens: number; chunks: number; unchunkedLatencySec: number };
  /**
   * Decode only: the rates with the engine dispatch overlay on (`SimResult.decodeWithDispatch`), engine overhead
   * outside the theoretical maximum; absent when the overlay is off. Rates in tokens/s; `kernelsPerStep` count;
   * `perKernelSec` seconds.
   */
  withDispatch?: {
    tokensPerSec: number;
    perSequenceTokensPerSec: number;
    kernelsPerStep: number;
    perKernelSec: number;
  };
}

/** Memory of one device of one stage. Unit: bytes. */
export interface StageMemory {
  /** Stage index. */
  stage: number;
  /** Resident weights. Unit: bytes. */
  weightBytes: number;
  /** KV cache and other per-sequence state. Unit: bytes. */
  sequenceStateBytes: number;
  /** Peak transient activations. Unit: bytes. */
  activationPeakBytes: number;
  /** Weights + state + activations + runtime reserve (+ staging, draft and spec-dec transients). Unit: bytes. */
  totalBytes: number;
  /** Capacity the check uses: the GPU-usable share of the device (all of it on dedicated memory). Unit: bytes. */
  capacityBytes: number;
  /** Installed memory of the device; absent = `capacityBytes`. Unit: bytes. */
  physicalCapacityBytes?: number;
  /** Runtime reserve (framework, driver); absent = 0. Unit: bytes. */
  runtimeReserveBytes?: number;
  /** totalBytes ≤ capacityBytes. */
  fits: boolean;
  /** n-gram lookup-table bytes resident on the device, part of `weightBytes`; absent = 0. Unit: bytes. */
  lookupTableBytes?: number;
  /** Staging buffers for streamed weight groups; absent = 0. Unit: bytes. */
  stagingBytes?: number;
  /** Draft weights and draft KV/state (spec-dec); absent = 0. Unit: bytes. */
  draftBytes?: number;
  /** Per-position state snapshots of a speculative verify; absent = 0. Unit: bytes. */
  speculativeTransientBytes?: number;
}

/** The memory-fit report. */
export interface MemorySummary {
  /** Dedicated or unified memory (unified: the GPU may use only part of it, `capacityBytes` of each stage). */
  kind: 'dedicated' | 'unified';
  /** True when every stage and every off-device tier fits ({@link memoryFits}). */
  fits: boolean;
  /** Most sequences one DP replica could hold at this context (Infinity: no per-sequence state). Count. */
  maxSequencesPerReplica: number;
  /** One entry per pipeline stage. */
  stages: StageMemory[];
  /** Off-device tiers holding offloaded groups, per node (empty when nothing is offloaded); absent = none. */
  tiers?: TierMemory[];
}

/**
 * What a breakdown row's time is split by: the op bound, with collectives split into their latency (α) and
 * bandwidth (β) terms, and off-device transfers (`bound: 'transfer'`) on their own.
 */
export type BreakdownBound = 'compute' | 'memory' | 'commLatency' | 'commBandwidth' | 'transfer';

/** Display labels of the breakdown bounds, in legend order. */
export const BREAKDOWN_BOUND_LABELS: Record<BreakdownBound, string> = {
  compute: 'compute',
  memory: 'memory',
  commLatency: 'comm latency',
  commBandwidth: 'comm bandwidth',
  transfer: 'offload transfer',
};

/** One row of a time breakdown. */
export interface BreakdownRow {
  /** Group key: the op name, the op kind, or the layer name. */
  key: string;
  /** Σ op time of the group. Unit: seconds. */
  sec: number;
  /** sec / Σ op sec of the phase (all stages, before overlap). Dimensionless, in [0, 1]. */
  share: number;
  /**
   * The group's time split by what bounds each op; sums to `sec`. A collective adds its `commLatencySec` and
   * `commBandwidthSec` (which sum to its `sec`); a transfer op adds its `sec` to `transfer`. Unit: seconds.
   */
  boundSec: Record<BreakdownBound, number>;
  /** Layer rows of a speculative decode: `'draft'` for `draft.*` layers, else `'verify'`; absent otherwise. */
  section?: 'verify' | 'draft';
  /** Σ FLOP. Unit: FLOP. */
  flops: number;
  /** Σ memory traffic. Unit: bytes. */
  bytes: number;
  /** Ops in the group. Count. */
  count: number;
}

/** A phase's time breakdown grouped three ways, rows sorted by time (largest first). */
export interface Breakdown {
  /** By op name (e.g. `attn.qkv_proj`), summed over layers and stages. */
  byOp: BreakdownRow[];
  /** By op kind (`gemm`, `attention`, `collective`, ...). */
  byKind: BreakdownRow[];
  /**
   * By layer (`embed`, `layer.<i>`, `head`, `pp.send` for forward sends, `pp.return` for sampled-token return,
   * `draft.*` for drafter layers), in model order; speculative decode lists verify rows before draft rows.
   */
  byLayer: BreakdownRow[];
  /** A speculative decode only: the `byLayer` rows summed per section, keys `'verify'` and `'draft'`. */
  sections?: BreakdownRow[];
  /** Σ op sec over all stages (the share denominator). Unit: seconds. */
  totalSec: number;
}

/** A weight group or the KV cache the deployment keeps (partly) off-device, for the result chips. */
export interface OffloadPlacement {
  /** Which group: routed (MoE) experts, the n-gram lookup tables, the token-embedding table, or the KV cache. */
  group: 'routed_experts' | 'lookup_tables' | 'token_embeddings' | 'kv_cache';
  /** Tier holding the group (the KV cache: the remote pool when its share is the pool alone, else host RAM when any of it is there, else the SSD). */
  tier: KvTier;
  /**
   * KV cache only: shares of each offloadable layer's per-sequence KV on the device (f_dev), in host RAM (f_ram), on
   * the SSD (f_ssd) and in the remote pool (f_pool, absent = 0), summing to 1, and the share of a cached prefix on the
   * device (f_pre) when it differs from f_dev. Dimensionless, in [0, 1].
   */
  kvShares?: { device: number; hostRam: number; ssd: number; pool?: number; prefixDevice?: number };
  /**
   * Routed experts only: the configured hit ratio h(f) of the resident cache (the hit-ratio model at the resident
   * fraction, before the engine's physical clamp). Dimensionless, in [0, 1].
   */
  hitRatio?: number;
  /** An explicit layer range `[from, to)` of the rule (a split, or experts from a later layer), zero-based. */
  layers?: { from: number; to: number };
}

/** Speculative-decoding settings of the run, for the result annotations. */
export interface SpeculativeSummary {
  /** Draft method. */
  method: 'mtp' | 'dflash2' | 'dspark';
  /** Draft tokens per step k. Count. */
  draftLength: number;
  /** True when the drafter is the labelled hypothetical stand-in, not a released model. */
  hypothetical: boolean;
}

/** One phase's point on the roofline. */
export interface RooflinePoint {
  /** Σ FLOP of the phase's compute ops (neither collectives nor off-device transfers). Unit: FLOP. */
  flops: number;
  /** Σ memory bytes of those ops. Unit: bytes. */
  bytes: number;
  /** Arithmetic intensity = flops / bytes. Unit: FLOP/byte. */
  intensityFlopPerByte: number;
  /** Attained rate = Σ FLOP / Σ op sec (all ops, communication and transfers included). Unit: FLOP/s. */
  attainedFlopsPerSec: number;
}

/** The roofline of the rig: the compute-bound roof, the bandwidth-bound slope, and the two phase points. */
export interface RooflineSummary {
  /** Peak of the dominant GEMM format (most prefill FLOP), per device. Unit: FLOP/s. */
  peakFlopsPerSec: number;
  /** Label of that peak, e.g. `'bf16, fp32 accumulate, dense'`. */
  peakLabel: string;
  /** Memory bandwidth the engine reaches (`engineMemBandwidthBytesPerSec`), per device. Unit: B/s. */
  memBandwidthBytesPerSec: number;
  /** Ridge point = peak / bandwidth. Unit: FLOP/byte. */
  ridgeFlopPerByte: number;
  /** Prefill point. */
  prefill: RooflinePoint;
  /** Decode point. */
  decode: RooflinePoint;
}

/** What the result panel shows. */
export interface ResultSummary {
  /** Prefill headline. */
  prefill: PhaseSummary;
  /** Decode headline. */
  decode: PhaseSummary;
  /** Memory fit. */
  memory: MemorySummary;
  /** Time breakdowns per phase. */
  breakdown: { prefill: Breakdown; decode: Breakdown };
  /** Roofline. */
  roofline: RooflineSummary;
  /** Devices used (tp·pp·dp). Count. */
  devicesUsed: number;
  /** Name of the simulated device. */
  deviceName: string;
  /** Engine warnings, then the caller's notes, deduplicated. */
  warnings: string[];
  /** Weight groups placed off-device (empty when nothing is offloaded); absent = none. */
  offload?: OffloadPlacement[];
  /** Speculative decoding settings; absent when off. */
  speculative?: SpeculativeSummary;
}

function phaseSummary(p: PhaseResult, contextTokens: number): PhaseSummary {
  return {
    contextTokens,
    tokensPerSec: p.tokensPerSec,
    ...(p.computedTokensPerSec !== undefined && { computedTokensPerSec: p.computedTokensPerSec }),
    perSequenceTokensPerSec: p.perSequenceTokensPerSec,
    latencySec: p.latencySec,
    cycleSec: p.cycleSec,
    microbatches: p.microbatches,
    microbatchSize: p.microbatchSize,
    stageSec: p.stages.map((s) => s.sec),
    comm: { ...p.comm },
    ...(p.transfer && { transfer: { ...p.transfer } }),
    ...(p.speculative && { speculative: { ...p.speculative } }),
    ...(p.chunkedPipeline && {
      chunkedPipeline: {
        chunkTokens: p.chunkedPipeline.chunkTokens,
        chunks: p.chunkedPipeline.chunks,
        unchunkedLatencySec: p.chunkedPipeline.unchunked.latencySec,
      },
    }),
  };
}

/** Every phase op with its layer; forward sends use `pp.send`, the decode token return uses `pp.return`. */
function phaseOps(p: PhaseResult): { layer: string; op: OpResult }[] {
  const out: { layer: string; op: OpResult }[] = [];
  for (const stage of p.stages) {
    for (const layer of stage.layers)
      for (const op of layer.ops) out.push({ layer: layer.name, op });
    if (stage.sendToNext !== undefined) out.push({ layer: 'pp.send', op: stage.sendToNext });
  }
  if (p.loopBack !== undefined) out.push({ layer: 'pp.return', op: p.loopBack });
  return out;
}

/** Groups ops by `keyOf`, keeping first-seen order; shares use `totalSec`. */
function group(
  ops: { layer: string; op: OpResult }[],
  keyOf: (e: { layer: string; op: OpResult }) => string,
  totalSec: number,
): BreakdownRow[] {
  const rows = new Map<string, BreakdownRow>();
  for (const e of ops) {
    const key = keyOf(e);
    let row = rows.get(key);
    if (row === undefined) {
      row = {
        key,
        sec: 0,
        share: 0,
        boundSec: { compute: 0, memory: 0, commLatency: 0, commBandwidth: 0, transfer: 0 },
        flops: 0,
        bytes: 0,
        count: 0,
      };
      rows.set(key, row);
    }
    row.sec += e.op.sec;
    addBoundSec(row.boundSec, e.op);
    row.flops += e.op.flops;
    row.bytes += e.op.bytes.totalBytes;
    row.count += 1;
  }
  return [...rows.values()].map((r) => ({ ...r, share: totalSec > 0 ? r.sec / totalSec : 0 }));
}

/**
 * Adds an op's time to its bound: compute / memory as the op's `bound`; a collective split into its α and β terms
 * (`sec = commLatencySec + commBandwidthSec`); a transfer op whole into `transfer`. Unit: seconds.
 */
function addBoundSec(boundSec: Record<BreakdownBound, number>, op: OpResult): void {
  switch (op.bound) {
    case 'compute':
    case 'memory':
      boundSec[op.bound] += op.sec;
      return;
    case 'comm':
      boundSec.commLatency += op.commLatencySec;
      boundSec.commBandwidth += op.commBandwidthSec;
      return;
    case 'transfer':
      boundSec.transfer += op.sec;
      return;
  }
}

/** The section of a layer in a speculative decode step: drafter layers (`draft`, `draft.*`) or the verify pass. */
function sectionOf(layer: string): 'verify' | 'draft' {
  return /^draft(\.|$)/.test(layer) ? 'draft' : 'verify';
}

/** Op name with any `layer.<i>.` prefix removed (op names are per-layer templates). */
function opName(op: Op): string {
  return op.name.replace(/^layer\.\d+\./, '');
}

/**
 * The time breakdown of one phase, grouped by op name, op kind and layer. A speculative decode
 * (`p.speculative` set) also tags its layer rows `'verify'` / `'draft'` and sums them in `sections`.
 */
export function breakdown(p: PhaseResult): Breakdown {
  const ops = phaseOps(p);
  // Formula: share_g = Σ_{op ∈ g} sec_op / Σ_all sec_op
  const totalSec = ops.reduce((s, e) => s + e.op.sec, 0);
  const bySec = (a: BreakdownRow, b: BreakdownRow) => b.sec - a.sec;
  const result: Breakdown = {
    byOp: group(ops, (e) => opName(e.op.op), totalSec).sort(bySec),
    byKind: group(ops, (e) => e.op.op.kind, totalSec).sort(bySec),
    byLayer: group(ops, (e) => e.layer, totalSec),
    totalSec,
  };
  if (p.speculative !== undefined) {
    const order = { verify: 0, draft: 1 };
    result.byLayer = result.byLayer
      .map((r) => ({ ...r, section: sectionOf(r.key) }))
      .sort((a, b) => order[a.section] - order[b.section]);
    result.sections = group(ops, (e) => sectionOf(e.layer), totalSec).sort(
      (a, b) => order[a.key as 'verify' | 'draft'] - order[b.key as 'verify' | 'draft'],
    );
  }
  return result;
}

/** The roofline point of one phase (see {@link RooflinePoint}). */
function rooflinePoint(p: PhaseResult): RooflinePoint {
  const ops = phaseOps(p).map((e) => e.op);
  const compute = ops.filter((o) => o.op.kind !== 'collective' && o.op.kind !== 'transfer');
  const flops = compute.reduce((s, o) => s + o.flops, 0);
  const bytes = compute.reduce((s, o) => s + o.bytes.totalBytes, 0);
  const sec = ops.reduce((s, o) => s + o.sec, 0);
  // Formula: I = ΣFLOP / Σbytes;  attained = ΣFLOP / Σsec
  return {
    flops,
    bytes,
    intensityFlopPerByte: bytes > 0 ? flops / bytes : 0,
    attainedFlopsPerSec: sec > 0 ? flops / sec : 0,
  };
}

/** Roofline of a result: the peak of the GEMM format with the most prefill FLOP, the engine's memory bandwidth. */
function roofline(result: SimResult, input: SimInput): RooflineSummary {
  const device = result.effectiveDevice;
  const engine =
    device.engines.find((e) => e.id === (input.hardware.engineId ?? device.engines[0].id)) ??
    device.engines[0];
  const memBandwidthBytesPerSec = engineMemBandwidthBytesPerSec(
    engine,
    device.memory.bandwidthBytesPerSec,
  );
  const flopsByPeak = new Map<string, { flops: number; peak: number }>();
  for (const { op } of phaseOps(result.prefill)) {
    // GEMV-path GEMMs (computePath 'gemv') ran at the vector peak, not a matmul peak: not a roofline candidate.
    if (op.op.kind !== 'gemm' || op.peakFlopsPerSec === undefined || op.computePath === 'gemv')
      continue;
    const label = `${op.usedInput ?? op.op.math}, ${op.usedAccumulate ?? op.op.accumulate} accumulate, ${op.op.sparsity === 'dense' ? 'dense' : '2:4 sparse'}`;
    const entry = flopsByPeak.get(label) ?? { flops: 0, peak: op.peakFlopsPerSec };
    entry.flops += op.flops;
    flopsByPeak.set(label, entry);
  }
  let peakLabel = '';
  let peakFlopsPerSec = 0;
  let most = -1;
  for (const [label, e] of flopsByPeak) {
    if (e.flops > most) [most, peakLabel, peakFlopsPerSec] = [e.flops, label, e.peak];
  }
  return {
    peakFlopsPerSec,
    peakLabel,
    memBandwidthBytesPerSec,
    // Formula: ridge = peak / BW
    ridgeFlopPerByte: memBandwidthBytesPerSec > 0 ? peakFlopsPerSec / memBandwidthBytesPerSec : 0,
    prefill: rooflinePoint(result.prefill),
    decode: rooflinePoint(result.decode),
  };
}

/** Reduces a result for the UI. `notes` (loader notes, unlock descriptions) are appended to the warnings. */
export function summarize(result: SimResult, input: SimInput, notes: string[] = []): ResultSummary {
  return {
    prefill: {
      ...phaseSummary(result.prefill, input.workload.promptLen),
      ...(result.prefillWithoutOverlap && {
        withoutOverlap: {
          tokensPerSec: result.prefillWithoutOverlap.tokensPerSec,
          perSequenceTokensPerSec: result.prefillWithoutOverlap.perSequenceTokensPerSec,
        },
      }),
    },
    decode: {
      ...phaseSummary(result.decode, input.workload.decodeContextLen),
      ...(result.decodeWithDispatch && {
        withDispatch: {
          tokensPerSec: result.decodeWithDispatch.tokensPerSec,
          perSequenceTokensPerSec: result.decodeWithDispatch.perSequenceTokensPerSec,
          kernelsPerStep: result.decodeWithDispatch.kernelsPerStep,
          perKernelSec: result.decodeWithDispatch.perKernelSec,
        },
      }),
    },
    memory: {
      kind: result.memory.kind,
      fits: memoryFits({ stages: result.memory.perStage, tiers: result.memory.tiers }),
      maxSequencesPerReplica: result.memory.maxSequencesPerReplica,
      stages: result.memory.perStage.map((s) => ({ ...s })),
      tiers: result.memory.tiers.map((t) => ({ ...t })),
    },
    breakdown: { prefill: breakdown(result.prefill), decode: breakdown(result.decode) },
    roofline: roofline(result, input),
    devicesUsed: result.devicesUsed,
    deviceName: result.effectiveDevice.name,
    warnings: [...new Set([...result.warnings, ...notes])],
    offload: offloadPlacements(input),
    ...(input.deployment?.speculative && {
      speculative: {
        method: input.deployment.speculative.method,
        draftLength: input.deployment.speculative.draftLength,
        hypothetical: input.deployment.speculative.drafter?.kind === 'hypothetical',
      },
    }),
  };
}

/**
 * The cross-pool figures of a disaggregated result, as plain numbers (see `DisaggregatedResult` and
 * docs/design/prefill-decode-disaggregation.md §3). Every figure here is quoted with all three contexts, P / O and
 * the decode context N (`formatWorkloadQuote`).
 */
export interface DisaggregationSummary {
  /** Prompt length P of every request. Unit: tokens. */
  promptLen: number;
  /** Output tokens O per request. Unit: tokens. */
  outputLen: number;
  /** Decode context N at which the decode pool's rate is taken (for all O tokens). Unit: tokens. */
  decodeContextLen: number;
  /** One prefill instance: devices D_P, R_P (tok/s @ P), λ_P (requests/s). */
  prefill: PoolRates;
  /** One decode instance: devices D_D, R_D (tok/s @ N), λ_D (requests/s). */
  decode: PoolRates;
  /** The balanced ratio r* and the figures at balance. */
  balance: BalanceResult;
  /** The system at the user's instance ratio. */
  system: SystemResult;
  /**
   * The P→D KV transfer (bytes per request @ P, per-request and exposed seconds, effective prefill cycle, decode
   * ingress); absent when the input had none.
   */
  kvTransfer?: KvTransferResult;
}

/** {@link summarizeDisaggregated}'s result: each pool's ordinary summary and the cross-pool figures. */
export interface DisaggregatedSummary {
  /**
   * The prefill pool's run (`DisaggregatedResult.prefillPool`): its prefill phase is the headline Prefill; its memory
   * fit is at P; its decode phase (taken at max(1, P − 1)) is not meant to be shown.
   */
  prefillPool: ResultSummary;
  /** The decode pool's run: its decode phase is the headline Decode @ N. */
  decodePool: ResultSummary;
  /** Cross-pool figures. */
  system: DisaggregationSummary;
}

/**
 * The prefill pool's headline prefill phase with the KV transfer applied: R_P,eff = R_P · cycle_P / cycle_eff
 * (`DisaggregationSummary.prefill.tokensPerSec`), with the per-sequence and computed-token rates scaled by the same
 * factor cycle_P / cycle_eff. `computeOnlyTokensPerSec` is R_P, the pool's rate without the transfer (tokens/s @ P).
 * Without a KV transfer the factor is 1 and the phase equals the pool's own.
 */
export function effectivePrefillPhase(
  s: DisaggregatedSummary,
): PhaseSummary & { computeOnlyTokensPerSec: number } {
  const raw = s.prefillPool.prefill;
  // Formula: f = R_P,eff / R_P = cycle_P / cycle_eff  (1 without a transfer)
  const factor = raw.tokensPerSec > 0 ? s.system.prefill.tokensPerSec / raw.tokensPerSec : 1;
  return {
    ...raw,
    tokensPerSec: s.system.prefill.tokensPerSec,
    perSequenceTokensPerSec: raw.perSequenceTokensPerSec * factor,
    ...(raw.computedTokensPerSec !== undefined && {
      computedTokensPerSec: raw.computedTokensPerSec * factor,
    }),
    computeOnlyTokensPerSec: raw.tokensPerSec,
  };
}

/**
 * Reduces a disaggregated result for the UI: {@link summarize} on each pool's run (with that pool's own input,
 * `poolInput`), plus the cross-pool numbers. `prefillNotes` / `decodeNotes` go to each pool's warnings.
 */
export function summarizeDisaggregated(
  result: DisaggregatedResult,
  input: DisaggregatedInput,
  prefillNotes: string[] = [],
  decodeNotes: string[] = prefillNotes,
): DisaggregatedSummary {
  const w = input.shared.workload;
  return {
    prefillPool: summarize(
      result.prefillPool,
      poolInput(input.shared, input.prefillPool, 'prefill'),
      prefillNotes,
    ),
    decodePool: summarize(
      result.decodePool,
      poolInput(input.shared, input.decodePool, 'decode'),
      decodeNotes,
    ),
    system: {
      promptLen: w.promptLen,
      outputLen: input.config.outputLen,
      decodeContextLen: w.decodeContextLen,
      prefill: { ...result.prefill },
      decode: { ...result.decode },
      balance: { ...result.balance },
      system: { ...result.system },
      ...(result.kvTransfer !== undefined && {
        kvTransfer: {
          ...result.kvTransfer,
          link: { ...result.kvTransfer.link },
          stageRequestBytes: [...result.kvTransfer.stageRequestBytes],
        },
      }),
    },
  };
}

/**
 * The off-device placements of `input.deployment.offload`, in the order experts, lookup tables, token embeddings, KV
 * cache; one per rule (a host-RAM + SSD split gives two), each group's rules by first layer; tables kept on the device
 * are left out. The KV cache is one placement with its tier shares.
 */
export function offloadPlacements(input: SimInput): OffloadPlacement[] {
  const offload = input.deployment?.offload;
  if (offload === undefined) return [];
  const out: OffloadPlacement[] = [];
  const byFirstLayer = (a: { layers?: LayerRange }, b: { layers?: LayerRange }) =>
    (a.layers?.from ?? 0) - (b.layers?.from ?? 0);
  for (const experts of [...expertRules(offload)].sort(byFirstLayer)) {
    out.push({
      group: 'routed_experts',
      tier: experts.tier,
      hitRatio: hitRatio(experts.hitRatio, experts.residentFraction),
      ...(experts.layers && { layers: { ...experts.layers } }),
    });
  }
  for (const tables of [...lookupTableRules(offload)].sort(byFirstLayer)) {
    if (tables.tier === 'device') continue;
    out.push({
      group: 'lookup_tables',
      tier: tables.tier,
      ...(tables.layers && { layers: { ...tables.layers } }),
    });
  }
  const embeddings = offload.embeddings;
  if (embeddings !== undefined && embeddings.tier !== 'device')
    out.push({ group: 'token_embeddings', tier: embeddings.tier });
  const kv = offload.kvCache;
  if (kv !== undefined) {
    // Formula: f_pool = remotePool.fraction ?? 0 (0 at f_dev = 1);  f_ram = hostRamFraction ?? 1 − f_dev − f_pool;
    //          f_ssd = 1 − f_dev − f_ram − f_pool   (engine KvCacheOffloadConfig, host.ts: resolveKvCache)
    const device = kv.deviceFraction;
    const pool = device >= 1 ? 0 : Math.max(0, Math.min(1 - device, kv.remotePool?.fraction ?? 0));
    const hostRam = kv.hostRamFraction ?? 1 - device - pool;
    const ssd = Math.max(0, 1 - device - hostRam - pool);
    const prefixDevice = kv.prefixDeviceFraction;
    // The tier named: the pool when it holds all of the off-device share (also a prefix-only placement with a pool);
    // else host RAM unless a given hostRamFraction of 0 means the SSD (also for a prefix-only placement, f_dev = 1).
    const poolOnly = kv.remotePool !== undefined && hostRam <= 0 && ssd <= 0;
    out.push({
      group: 'kv_cache',
      tier: poolOnly
        ? 'remote_pool'
        : kv.hostRamFraction === undefined || hostRam > 0
          ? 'host_ram'
          : 'nvme',
      kvShares: {
        device,
        hostRam,
        ssd,
        ...(pool > 0 ? { pool } : {}),
        ...(prefixDevice !== undefined && prefixDevice !== device ? { prefixDevice } : {}),
      },
    });
  }
  return out;
}
