/**
 * `summarize`: a `SimResult` reduced to what the UI shows (headline rates, memory per stage, time breakdowns,
 * roofline points, warnings). Small and plain data, so the worker can post it. Pure; no React, no DOM.
 *
 * Conventions: times in seconds, rates in tokens/s, FLOP in FLOP, bytes in bytes, all per device (the engine's
 * convention: one device of each pipeline stage, one microbatch step). Breakdown times are Σ op `sec` before
 * comm/compute overlap, so they can exceed the stage time.
 */

import type { Bound, OpResult, PhaseResult, SimInput, SimResult } from '../../sim/engine/types';
import { engineMemBandwidthBytesPerSec } from '../../sim/engine/opTime';
import type { Op } from '../../sim/ir/ops';

/** Headline numbers of one phase. */
export interface PhaseSummary {
  /** Aggregate rate over all DP replicas. Unit: tokens/s. */
  tokensPerSec: number;
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
  /** Sum of the three above. Unit: bytes. */
  totalBytes: number;
  /** Device capacity. Unit: bytes. */
  capacityBytes: number;
  /** totalBytes ≤ capacityBytes. */
  fits: boolean;
}

/** The memory-fit report. */
export interface MemorySummary {
  /** Dedicated or unified memory (unified: the OS share is not modeled). */
  kind: 'dedicated' | 'unified';
  /** True when every stage fits. */
  fits: boolean;
  /** Most sequences one DP replica could hold at this context (Infinity: no per-sequence state). Count. */
  maxSequencesPerReplica: number;
  /** One entry per pipeline stage. */
  stages: StageMemory[];
}

/** One row of a time breakdown. */
export interface BreakdownRow {
  /** Group key: the op name, the op kind, or the layer name. */
  key: string;
  /** Σ op time of the group. Unit: seconds. */
  sec: number;
  /** sec / Σ op sec of the phase (all stages, before overlap). Dimensionless, in [0, 1]. */
  share: number;
  /** The group's time split by what bounds each op; sums to `sec`. Unit: seconds. */
  boundSec: Record<Bound, number>;
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
  /** By layer (`embed`, `layer.<i>`, `head`, and `pp.send` for stage-to-stage sends), in model order. */
  byLayer: BreakdownRow[];
  /** Σ op sec over all stages (the share denominator). Unit: seconds. */
  totalSec: number;
}

/** One phase's point on the roofline. */
export interface RooflinePoint {
  /** Σ FLOP of the phase's non-collective ops. Unit: FLOP. */
  flops: number;
  /** Σ memory bytes of those ops. Unit: bytes. */
  bytes: number;
  /** Arithmetic intensity = flops / bytes. Unit: FLOP/byte. */
  intensityFlopPerByte: number;
  /** Attained rate = Σ FLOP / Σ op sec (all ops, communication included). Unit: FLOP/s. */
  attainedFlopsPerSec: number;
}

/** The roofline of the rig: one tensor ceiling, one bandwidth slope, and the two phase points. */
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
}

function phaseSummary(p: PhaseResult): PhaseSummary {
  return {
    tokensPerSec: p.tokensPerSec,
    perSequenceTokensPerSec: p.perSequenceTokensPerSec,
    latencySec: p.latencySec,
    cycleSec: p.cycleSec,
    microbatches: p.microbatches,
    microbatchSize: p.microbatchSize,
    stageSec: p.stages.map((s) => s.sec),
  };
}

/** Every op of a phase with the layer it belongs to (stage sends as layer `pp.send`). */
function phaseOps(p: PhaseResult): { layer: string; op: OpResult }[] {
  const out: { layer: string; op: OpResult }[] = [];
  for (const stage of p.stages) {
    for (const layer of stage.layers) for (const op of layer.ops) out.push({ layer: layer.name, op });
    if (stage.sendToNext !== undefined) out.push({ layer: 'pp.send', op: stage.sendToNext });
  }
  return out;
}

/** Groups ops by `keyOf`, keeping first-seen order; shares use `totalSec`. */
function group(ops: { layer: string; op: OpResult }[], keyOf: (e: { layer: string; op: OpResult }) => string, totalSec: number): BreakdownRow[] {
  const rows = new Map<string, BreakdownRow>();
  for (const e of ops) {
    const key = keyOf(e);
    let row = rows.get(key);
    if (row === undefined) {
      row = { key, sec: 0, share: 0, boundSec: { compute: 0, memory: 0, comm: 0 }, flops: 0, bytes: 0, count: 0 };
      rows.set(key, row);
    }
    row.sec += e.op.sec;
    row.boundSec[e.op.bound] += e.op.sec;
    row.flops += e.op.flops;
    row.bytes += e.op.bytes.totalBytes;
    row.count += 1;
  }
  return [...rows.values()].map((r) => ({ ...r, share: totalSec > 0 ? r.sec / totalSec : 0 }));
}

/** Op name with any `layer.<i>.` prefix removed (op names are per-layer templates). */
function opName(op: Op): string {
  return op.name.replace(/^layer\.\d+\./, '');
}

/** The time breakdown of one phase, grouped by op name, op kind and layer. */
export function breakdown(p: PhaseResult): Breakdown {
  const ops = phaseOps(p);
  // Formula: share_g = Σ_{op ∈ g} sec_op / Σ_all sec_op
  const totalSec = ops.reduce((s, e) => s + e.op.sec, 0);
  const bySec = (a: BreakdownRow, b: BreakdownRow) => b.sec - a.sec;
  return {
    byOp: group(ops, (e) => opName(e.op.op), totalSec).sort(bySec),
    byKind: group(ops, (e) => e.op.op.kind, totalSec).sort(bySec),
    byLayer: group(ops, (e) => e.layer, totalSec),
    totalSec,
  };
}

/** The roofline point of one phase (see {@link RooflinePoint}). */
function rooflinePoint(p: PhaseResult): RooflinePoint {
  const ops = phaseOps(p).map((e) => e.op);
  const compute = ops.filter((o) => o.op.kind !== 'collective');
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
  const engine = device.engines.find((e) => e.id === (input.hardware.engineId ?? device.engines[0].id)) ?? device.engines[0];
  const memBandwidthBytesPerSec = engineMemBandwidthBytesPerSec(engine, device.memory.bandwidthBytesPerSec);
  const flopsByPeak = new Map<string, { flops: number; peak: number }>();
  for (const { op } of phaseOps(result.prefill)) {
    if (op.op.kind !== 'gemm' || op.peakFlopsPerSec === undefined) continue;
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
    prefill: phaseSummary(result.prefill),
    decode: phaseSummary(result.decode),
    memory: {
      kind: result.memory.kind,
      fits: result.memory.fits,
      maxSequencesPerReplica: result.memory.maxSequencesPerReplica,
      stages: result.memory.perStage.map((s) => ({ ...s })),
    },
    breakdown: { prefill: breakdown(result.prefill), decode: breakdown(result.decode) },
    roofline: roofline(result, input),
    devicesUsed: result.devicesUsed,
    deviceName: result.effectiveDevice.name,
    warnings: [...new Set([...result.warnings, ...notes])],
  };
}
