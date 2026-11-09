/**
 * Engine input and result types. A simulation takes a model, its numeric formats, its parallelism, a workload
 * and a hardware stack, and returns the theoretical-maximum prefill and decode rates plus a memory-fit report.
 * See docs/simulation-model.md §7 (stage, overlap, pipeline) and §8 (memory fit).
 *
 * Conventions: times are seconds; rates are tokens/s; bytes are per device unless a field says otherwise.
 */

import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { ByteBreakdown } from '../ir/metrics';
import type { BlockDraftSpec, ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
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
  /**
   * Overlap of off-device transfers (deployment `offload`) with compute. `'ideal'` (default): the next layer's
   * transfers are issued while this layer computes (prefetch), so a stage pays the larger of its core time and its
   * total transfer time, `t_s = exposed_s + max(core_s, xfer_s)`; `'none'`: demand fetch, serialized,
   * `t_s = exposed_s + core_s + xfer_s`. Prediction accuracy is not modelled (the bound assumes 100%). No effect
   * when nothing is offloaded (`xfer_s = 0`). See docs/simulation-model.md §7 and design doc §2.6.
   */
  offloadOverlap: 'ideal' | 'none';
}

/**
 * Deployment-scenario knobs (design doc §0): explicit opt-ins beyond the hardware/math bound of the bare model.
 * Every field is optional and off by default: `deployment` absent or `{}` gives today's results, bit for bit. With
 * a knob on, the result is still the hardware/math upper bound given the knob's stated parameters.
 */
export interface DeploymentConfig {
  /** Weight groups placed off-device (host RAM or NVMe) and streamed per step. Absent = everything on the device. */
  offload?: OffloadConfig;
  /** Speculative decoding (draft + verify). Absent = off. */
  speculative?: SpecDecodeConfig;
}

/** Off-device storage tier a weight group can be placed on (`hardware.host.ram` / `hardware.host.ssd`). */
export type OffloadTier = 'host_ram' | 'nvme';

/** A half-open range of decoder-layer indices `[from, to)` over `ModelSpec.layers`. Zero-based indices. */
export interface LayerRange {
  /** First layer index in the range (inclusive). */
  from: number;
  /** One past the last layer index (exclusive); `to > from`. */
  to: number;
}

/**
 * Expected-hit model h(f) of the device-resident expert cache (design doc §2.4): the probability that a unique
 * routed expert needed this step is already resident, as a function of the resident fraction f. `'uniform'`
 * (default): `h = f` (balanced router); `'skewed_classic'` / `'skewed_fine_grained'`: the skewed-routing fits of
 * design doc §5.2; `'custom'`: the user's `h` per unique expert per layer per step at batch 1 (dimensionless, in
 * [0, 1]), clamped to the physical bounds by the engine.
 */
export type HitRatioModel =
  | { kind: 'uniform' }
  | { kind: 'skewed_classic' }
  | { kind: 'skewed_fine_grained' }
  | { kind: 'custom'; h: number };

/** Weight offloading (design doc §2): which groups live off-device, where, and how their misses are read. */
export interface OffloadConfig {
  /**
   * `'transfer'` (default): missed weights stream to the device, which computes. `'cpu_compute'`: the host computes
   * the missed experts (design doc §2.8); a designed hook, rejected by validation for now.
   */
  mode?: 'transfer' | 'cpu_compute';
  /**
   * Outstanding read requests kept in flight per device (Little's law); default and cap: the tier's
   * `maxOutstanding`. 1 serializes the reads. Integer ≥ 1. Count.
   */
  queueDepth?: number;
  /** Routed experts of MoE layers kept partly off-device, with a resident expert cache. */
  routedExperts?: {
    /** Tier holding the full expert set of each offloaded layer. */
    tier: OffloadTier;
    /** Layers whose experts are offloaded; absent = every MoE layer. Layers of the range without a MoE block are skipped. */
    layers?: LayerRange;
    /** Fraction f of each offloaded layer's experts kept resident in device memory, in [0, 1]. Dimensionless. */
    residentFraction: number;
    /** Hit-ratio model h(f) of the resident cache. */
    hitRatio: HitRatioModel;
    /** Also offload the drafter's own MoE experts (DeepSeek-V4 DSpark); default false. */
    includeDraft?: boolean;
  };
  /** Whole decoder layers streamed from a tier every step (llama.cpp `-ngl N`); no hit ratio. */
  denseLayers?: { tier: OffloadTier; layers: LayerRange };
  /**
   * n-gram lookup-table placement (Qwen PLE, DeepSeek Engram; design doc §2.10); absent = `'device'` (identity).
   * `scaleLayout` (block-scaled tables only): default `'separate'` on host RAM, `'pinned_host'` on NVMe (the scales
   * pinned in host RAM and counted in its capacity), `'interleaved'` for a repacked table (one read per row).
   */
  lookupTables?: {
    tier: 'device' | OffloadTier;
    scaleLayout?: 'separate' | 'pinned_host' | 'interleaved';
  };
}

/**
 * Per-token draft acceptance (design doc §3.5). `'iid'`: one probability α in [0, 1) for every draft position
 * (E[τ] = (1 − α^(k+1)) / (1 − α)); `'per_position'`: α_j in [0, 1], the probability that draft j is accepted given
 * drafts 1..j−1 were (a short vector is extended with its last value; α_j = 1 is legal). Dimensionless.
 */
export type AcceptanceModel =
  | { kind: 'iid'; alpha: number }
  | { kind: 'per_position'; alphas: number[] };

/** Speculative decoding (design doc §3): a drafter proposes k tokens per step, the target verifies k + 1. */
export interface SpecDecodeConfig {
  /** `'mtp'` needs `model.draft.kind === 'mtp'`; the block methods need `drafter`. Rejected when pp > 1 (§3.6). */
  method: 'mtp' | 'dflash2' | 'dspark';
  /**
   * The block drafter: a Hugging Face repo whose config the loader read, or the labelled hypothetical stand-in.
   * Ignored for MTP. The engine reads `spec` only; `repoId` is provenance for the UI.
   */
  drafter?:
    | { kind: 'repo'; repoId: string; spec: BlockDraftSpec }
    | { kind: 'hypothetical'; spec: BlockDraftSpec };
  /** Draft tokens per step k (verify width k + 1; at most `blockSize − 1` for block drafters). Integer in 1..16. Count. */
  draftLength: number;
  /** Per-token acceptance of the drafts. */
  acceptance: AcceptanceModel;
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
  /** Deployment-scenario knobs (offload, speculative decoding); absent or `{}` = off, identical results. */
  deployment?: DeploymentConfig;
}

/**
 * What limits an op: tensor/vector compute, memory bandwidth, communication (collectives), or an off-device
 * transfer (host link / storage tier).
 */
export type Bound = 'compute' | 'memory' | 'comm' | 'transfer';

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
  /** Off-device read latency terms (α_tier + α_link) of a transfer op; 0 for other ops. Unit: seconds. */
  transferLatencySec: number;
  /**
   * Off-device service term of a transfer op (the max of its Little's-law, IOPS, tier-, host- and link-bandwidth
   * terms, design doc §2.3); 0 for other ops. Unit: seconds.
   */
  transferBandwidthSec: number;
  /** Op time: the roofline max(computeSec, memorySec), the collective time, or the transfer time. Unit: seconds. */
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
  /** Σ of compute-op times: every op that is neither a collective nor an off-device transfer. Unit: seconds. */
  computeSec: number;
  /** Σ collective latency (α) terms. Unit: seconds. */
  commLatencySec: number;
  /** Σ collective bandwidth (β) terms. Unit: seconds. */
  commBandwidthSec: number;
  /**
   * Communication time not hidden behind compute under the overlap option: `t_core,s − compute_s`, where `t_core,s`
   * is the stage time without transfers (docs §7). Unit: seconds.
   */
  exposedCommSec: number;
  /**
   * Stage time per microbatch, docs §7: `t_core,s` (`ideal`: lat + max(compute, bw); `none`: compute + lat + bw
   * + p2p) extended by the transfers under `offloadOverlap`. Unit: seconds.
   */
  sec: number;
  /** Σ transfer op times (latency + service) of the stage, `xfer_s`. Unit: seconds. */
  transferSec: number;
  /** Transfer time not hidden under `offloadOverlap`: `t_s − t_core,s` (0 when nothing is offloaded). Unit: seconds. */
  exposedTransferSec: number;
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
  /** Decode-only last-stage → stage-0 uint32 sampled-token send (α and β seconds); absent for pp1 or prefill. */
  loopBack?: OpResult;
  /** Pipeline cycle: time for every microbatch of a replica to pass one step (docs §7). Unit: seconds. */
  cycleSec: number;
  /** Prefill: time to first token of a replica batch; decode: time per output token. Unit: seconds. */
  latencySec: number;
  /** Aggregate rate over all DP replicas (prefill: prompt tokens; decode: generated tokens). Unit: tokens/s. */
  tokensPerSec: number;
  /** Rate seen by one sequence. Unit: tokens/s. */
  perSequenceTokensPerSec: number;
  /**
   * Communication of the phase on its critical path, split into latency and bandwidth (design doc §4.2). Sums of
   * numbers the engine already computes, display only (`cycleSec` does not depend on them):
   * `latencySec = Σ_s lat_s + Σ α_p2p`, `bandwidthSec = Σ_s bw_s + Σ β_p2p`,
   * `exposedSec = Σ_s exposedComm_s + the PP sends on the latency path (overlap 'ideal'; under 'none' they are
   * already inside exposedComm_s) + loopBack.sec`. The α/β sums also include loopBack, once, in both modes.
   * Unit: seconds.
   */
  comm: { latencySec: number; bandwidthSec: number; exposedSec: number };
  /**
   * Off-device transfers of the phase (design doc §2.5); absent when nothing is offloaded. `sec` = Σ_s transferSec,
   * `exposedSec` = Σ_s exposedTransferSec (seconds); `bytes` = Σ useful bytes over the host link (bytes);
   * `requests` = Σ expected read requests (count, real) of one microbatch through every stage.
   */
  transfer?: { sec: number; exposedSec: number; bytes: number; requests: number };
  /**
   * Decode with speculative decoding on (design doc §3.5); absent otherwise. `draftSec` / `verifySec`: the draft
   * (`draft.*` layers) and verify shares of one step (seconds); `expectedTokensPerStep`: E[τ], tokens committed per
   * sequence per step incl. the bonus token (count, real, in [1, k + 1]); `speedup`: tokens/s over tokens/s with
   * spec-dec off (dimensionless).
   */
  speculative?: {
    draftSec: number;
    verifySec: number;
    expectedTokensPerStep: number;
    speedup: number;
  };
}

/** Memory use of one device of one pipeline stage. */
export interface DeviceMemory {
  /** Stage index. */
  stage: number;
  /** Resident weights on this device (incl. embedding / LM-head shards on the first / last stage). Unit: bytes. */
  weightBytes: number;
  /**
   * Bytes of n-gram lookup-table shards resident on this device (`ceil(rows/tp)·d·bpe(lookup)` per table of the
   * stage), ALREADY INCLUDED in `weightBytes`; 0 when the tables are off-device or the model has none. Display only
   * (the memory card's table segment). Unit: bytes.
   */
  lookupTableBytes: number;
  /** Per-replica sequence state on this device: KV + MLA latent + indexer keys + recurrent/conv state, × B_r. Unit: bytes. */
  sequenceStateBytes: number;
  /**
   * Peak transient activations (max over both phases' ops of activation read + write). Unit: bytes.
   */
  activationPeakBytes: number;
  /**
   * Staging buffer(s) for streamed weight groups (double-buffered under `offloadOverlap: 'ideal'`; design doc §2.7);
   * 0 when nothing streams. Unit: bytes.
   */
  stagingBytes: number;
  /** Draft weights and draft KV/state (speculative decoding on; last stage); 0 otherwise. Unit: bytes. */
  draftBytes: number;
  /**
   * Per-position recurrent/conv state snapshots of a k+1-token verify, × B_r (spec-dec on, hybrid models; design doc
   * §3.3); 0 otherwise. Unit: bytes.
   */
  speculativeTransientBytes: number;
  /**
   * weightBytes + sequenceStateBytes + activationPeakBytes (+ stagingBytes + draftBytes + speculativeTransientBytes,
   * all 0 with the deployment knobs off). Unit: bytes.
   */
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
  /**
   * Off-device storage tiers holding offloaded weight groups (one entry per tier used; empty when nothing is
   * offloaded). Their `fits` enters `fits` above.
   */
  tiers: TierMemory[];
}

/** Bytes placed on one off-device storage tier of a node, against its capacity (design doc §2.7). */
export interface TierMemory {
  /** Which tier. */
  tier: OffloadTier;
  /**
   * Bytes the node's devices place on the tier: the FULL offloaded groups (all experts of the offloaded layers,
   * whole dense layers, whole table shards), since the device cache is a copy; plus pinned table scales on host
   * RAM. Unit: bytes (per node, summed over its devices).
   */
  bytes: number;
  /** Installed capacity of the tier (`HostMemorySpec.capacityBytes` / `NvmeSpec.capacityBytes`). Unit: bytes. */
  capacityBytes: number;
  /** True if `bytes ≤ capacityBytes`. */
  fits: boolean;
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
