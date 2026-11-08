/**
 * Engine input and result types. A simulation takes a model, its numeric formats, its parallelism, a workload
 * and a hardware stack, and returns the theoretical-maximum prefill and decode rates plus a memory-fit report.
 * See docs/simulation-model.md §7 (stage, overlap, pipeline) and §8 (memory fit).
 *
 * Conventions: times are seconds; rates are tokens/s; bytes are per device unless a field says otherwise.
 */

import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { ByteBreakdown } from '../ir/metrics';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import type { Op } from '../ir/ops';
import type { DeviceSpec, HardwareStack } from '../hw/types';

/** The request load being evaluated. */
export interface Workload {
  /** GLOBAL concurrent sequences; each DP replica serves `B_r = batch / dp` (must divide). Count. */
  batch: number;
  /**
   * Prefill length P: prompt tokens per sequence that prefill processes from an EMPTY KV cache. Prefill tok/s @ P =
   * P ÷ the time for that whole prompt (so attention sees `qLen = kvLen = P`). Unit: tokens.
   */
  promptLen: number;
  /**
   * Decode context N: tokens ALREADY in the KV cache at the evaluated decode step (not counting the token being
   * generated). Decode tok/s @ N = the rate of generating one new token per sequence at that point, so attention
   * sees `qLen = 1, kvLen = N + 1` (the new token attends to itself too). Unit: tokens.
   */
  decodeContextLen: number;
}

/**
 * Documented ideal-vs-none physical assumptions (not inference-engine knobs).
 */
export interface SimOptions {
  /**
   * Communication/compute overlap. `'ideal'` (default): collective bandwidth time hides behind compute, latency
   * (α) does not: `t = lat + max(compute, bw)`. `'none'`: `t = compute + lat + bw`.
   */
  overlap: 'ideal' | 'none';
  /**
   * Elementwise fusion. `'none'` (default): every elementwise op reads and writes its activations from memory.
   * `'ideal'`: elementwise activation traffic is 0 (fused into neighbours); weight and state bytes stay.
   */
  elementwiseFusion: 'none' | 'ideal';
  /** All-reduce algorithm: `'best'` (default; min of ring and tree), `'ring'` or `'tree'`. */
  collectiveAlgorithm: 'best' | 'ring' | 'tree';
}

/** Everything one simulation needs. */
export interface SimInput {
  /** Model architecture. */
  model: ModelSpec;
  /** Numeric formats. */
  quant: QuantConfig;
  /** Parallelism degrees. */
  parallel: ParallelConfig;
  /** Request load. */
  workload: Workload;
  /** Hardware (device, topology, variant modifiers). */
  hardware: HardwareStack;
  /** Assumption switches; missing fields take their defaults. */
  options?: Partial<SimOptions>;
}

/** What limits an op: tensor/vector compute, memory bandwidth, or communication. */
export type Bound = 'compute' | 'memory' | 'comm';

/** The cost of one op on one device. */
export interface OpResult {
  /** The op costed. */
  op: Op;
  /** Dense-equivalent FLOP of the op. Unit: FLOP. */
  flops: number;
  /** Memory traffic of the op, per device. Unit: bytes. */
  bytes: ByteBreakdown;
  /** Compute time (shape efficiency included). Unit: seconds. */
  computeSec: number;
  /** Memory time = totalBytes / memory bandwidth. Unit: seconds. */
  memorySec: number;
  /** Collective latency (α) terms; 0 for non-collectives. Unit: seconds. */
  commLatencySec: number;
  /** Collective bandwidth (β) terms; 0 for non-collectives. Unit: seconds. */
  commBandwidthSec: number;
  /** Op time: the roofline max(computeSec, memorySec), or the collective time. Unit: seconds. */
  sec: number;
  /** Which term set `sec`. */
  bound: Bound;
  /** Tensor shape efficiency in (0, 1] (tensor ops only; dimensionless). */
  efficiency?: number;
  /** Peak used for the compute term, per device. Unit: FLOP/s. */
  peakFlopsPerSec?: number;
  /** Input class of the peak actually used (after fallback). */
  usedInput?: MathFormat;
  /** Accumulator of the peak actually used (after fallback). */
  usedAccumulate?: AccumulateType;
  /** Collectives only: which link level carried it. */
  linkLevel?: 'intra' | 'inter' | 'hierarchical';
}

/** The cost of one lowered layer on one device. */
export interface LayerResult {
  /** `'embed'`, `'layer.<i>'` or `'head'`. */
  name: string;
  /** Index into `ModelSpec.layers`, or `null` for the embedding and the head. */
  layerIndex: number | null;
  /** Per-op costs, in execution order. */
  ops: OpResult[];
  /** Σ op sec, without overlap. Unit: seconds. */
  sec: number;
}

/** The cost of one pipeline stage for one microbatch, on one device. */
export interface StageResult {
  /** Stage index, 0-based. */
  index: number;
  /** Per-layer costs. */
  layers: LayerResult[];
  /** Σ of non-collective op times. Unit: seconds. */
  computeSec: number;
  /** Σ collective latency (α) terms. Unit: seconds. */
  commLatencySec: number;
  /** Σ collective bandwidth (β) terms. Unit: seconds. */
  commBandwidthSec: number;
  /** Communication time not hidden behind compute under the overlap option. Unit: seconds. */
  exposedCommSec: number;
  /** Stage time per microbatch, docs §7 (`ideal`: lat + max(compute, bw); `none`: compute + lat + bw). Unit: seconds. */
  sec: number;
  /** Cost of the activation send to the next stage (absent on the last stage). */
  sendToNext?: OpResult;
}

/** The result of one phase (prefill or decode). */
export interface PhaseResult {
  /** Which phase. */
  phase: 'prefill' | 'decode';
  /** Microbatches in flight per replica, `m = min(pp, B_r)`. Count. */
  microbatches: number;
  /** Sequences per microbatch, `b_mb = ceil(B_r / m)`. Count. */
  microbatchSize: number;
  /** Per-stage costs. */
  stages: StageResult[];
  /** Pipeline cycle: time for every microbatch of a replica to pass one step (docs §7). Unit: seconds. */
  cycleSec: number;
  /** Prefill: time to first token of a replica batch; decode: time per output token. Unit: seconds. */
  latencySec: number;
  /** Aggregate rate over all DP replicas (prefill: prompt tokens; decode: generated tokens). Unit: tokens/s. */
  tokensPerSec: number;
  /** Rate seen by one sequence. Unit: tokens/s. */
  perSequenceTokensPerSec: number;
}

/** Memory use of one device of one pipeline stage. */
export interface DeviceMemory {
  /** Stage index. */
  stage: number;
  /** Resident weights on this device (incl. embedding / LM-head shards on the first / last stage). Unit: bytes. */
  weightBytes: number;
  /** Per-replica sequence state on this device: KV + MLA latent + indexer keys + recurrent/conv state, × B_r. Unit: bytes. */
  sequenceStateBytes: number;
  /** Peak transient activations (max over both phases' ops of activation read + write). Unit: bytes. */
  activationPeakBytes: number;
  /** weightBytes + sequenceStateBytes + activationPeakBytes. Unit: bytes. */
  totalBytes: number;
  /** Device memory capacity used for the check. Unit: bytes. */
  capacityBytes: number;
  /** True if totalBytes ≤ capacityBytes. */
  fits: boolean;
}

/** The memory-fit report. */
export interface MemoryReport {
  /** Memory kind of the (modified) device; `'unified'` adds a warning that the OS share is not modeled. */
  kind: 'dedicated' | 'unified';
  /** One entry per pipeline stage (all devices of a stage are symmetric). */
  perStage: DeviceMemory[];
  /** True if every stage fits. */
  fits: boolean;
  /**
   * Most sequences one DP replica could hold at this context length: `floor(min_s (capacity − W_s − act_s) / state_s)`
   * (docs §8). Infinity when no stage keeps per-sequence state; the activation peak is the evaluated batch's. Count.
   */
  maxSequencesPerReplica: number;
}

/** The full simulation result. */
export interface SimResult {
  /** Prefill phase result. */
  prefill: PhaseResult;
  /** Decode phase result (at `decodeContextLen`). */
  decode: PhaseResult;
  /** Memory fit. */
  memory: MemoryReport;
  /** Devices used, tp · pp · dp. Count. */
  devicesUsed: number;
  /** The device spec actually simulated (after modifiers), for the UI to display. */
  effectiveDevice: DeviceSpec;
  /** Peak fallbacks, unused devices, applied modifiers, unified-memory notes (human-readable lines). */
  warnings: string[];
}
