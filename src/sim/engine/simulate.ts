/**
 * `simulate`: the theoretical-maximum prefill and decode rates of a model on a hardware stack.
 * See docs/simulation-model.md §7 and §8.
 *
 * Steps:
 * 1. {@link validateInput} (throws on any error).
 * 2. Apply the device and link modifiers once (docs §4 "Device variants and modifiers"); everything after sees
 *    plain specs.
 * 3. For each phase: lower the model (`seqs = b_mb`) → per-op times → per-stage times → pipeline + DP.
 * 4. Memory fit.
 */

import { applyDeviceModifiers, applyLinkModifiers, describeDeviceModifier, describeLinkModifier } from '../hw/modifiers';
import type { DeviceSpec, HardwareStack, TopologySpec } from '../hw/types';
import { crossLayerReferences } from '../ir/validateModel';
import type { ModelSpec } from '../ir/model';
import { lowerModel, stageLayerRange, type LoweredModel, type LoweringContext } from '../lower/lower';
import type { AllReduceAlgorithm, CollectivePath } from './collectives';
import { memoryReport } from './memory';
import { engineMemBandwidthBytesPerSec, opTime, type OpTimeContext } from './opTime';
import { layerResult, microbatching, phaseResult, stageResult } from './pipeline';
import { checkPlacement, collectivePath, groupSpan, ppSendPath } from './placement';
import type { PhaseResult, SimInput, SimOptions, SimResult } from './types';
import { validateInput } from './validate';

/** The defaults of {@link SimOptions}: no elementwise fusion, ideal comm/compute overlap, best all-reduce. */
export const DEFAULT_SIM_OPTIONS: SimOptions = { overlap: 'ideal', elementwiseFusion: 'none', collectiveAlgorithm: 'best' };

/** The hardware after modifiers: the device and topology the engine actually simulates, plus info lines. */
export interface EffectiveHardware {
  /** Device after `deviceModifiers` (the base object itself when there are none). */
  device: DeviceSpec;
  /** Topology with `linkModifiers` applied to its links. */
  topology: TopologySpec;
  /** One "variant: …" line per applied modifier. */
  notes: string[];
}

/** Applies a stack's device and link modifiers (in order), without mutating the stack. */
export function effectiveHardware(hardware: HardwareStack): EffectiveHardware {
  const deviceMods = hardware.deviceModifiers ?? [];
  const intraMods = hardware.linkModifiers?.intraNode ?? [];
  const interMods = hardware.linkModifiers?.interNode ?? [];
  const device = deviceMods.length > 0 ? applyDeviceModifiers(hardware.device, deviceMods) : hardware.device;
  const topology: TopologySpec = { ...hardware.topology };
  if (topology.intraNode !== undefined && intraMods.length > 0) topology.intraNode = applyLinkModifiers(topology.intraNode, intraMods);
  if (topology.interNode !== undefined && interMods.length > 0) topology.interNode = applyLinkModifiers(topology.interNode, interMods);
  const notes = [
    ...deviceMods.map(describeDeviceModifier),
    ...intraMods.map((m) => `intraNode ${describeLinkModifier(m)}`),
    ...interMods.map((m) => `interNode ${describeLinkModifier(m)}`),
  ];
  return { device, topology, notes };
}

/**
 * One warning per cross-layer reference (shared KV cache, reused top-k, shared indexer keys) whose source layer
 * sits on an earlier pipeline stage than its consumer. Documented approximation: the consumer's stage is assumed to
 * hold the data (no transfer is costed, and the data is not counted twice in memory). See docs §9.
 */
export function crossStageWarnings(model: ModelSpec, pp: number): string[] {
  const stageOf = (layer: number): number => {
    for (let s = 0; s < pp; s++) {
      const [start, end] = stageLayerRange(model.layers.length, pp, s);
      if (layer >= start && layer < end) return s;
    }
    return -1;
  };
  return crossLayerReferences(model)
    .filter((ref) => stageOf(ref.source) !== stageOf(ref.consumer))
    .map(
      (ref) =>
        `layer ${ref.consumer} reads the ${ref.what.replaceAll('_', ' ')} of layer ${ref.source} on another pipeline stage; ` +
        'its transfer is not modeled',
    );
}

/** Lowers and costs one phase (prefill or decode) for one microbatch, then applies the pipeline and DP rules. */
function simulatePhase(
  lowered: LoweredModel,
  phase: 'prefill' | 'decode',
  base: Omit<OpTimeContext, 'collectivePaths'> & { tpPath: CollectivePath; epPath: CollectivePath },
  input: SimInput,
  topology: TopologySpec,
  options: SimOptions,
  shape: { sequencesPerReplica: number; microbatches: number; microbatchSize: number },
): PhaseResult {
  const stages = lowered.stages.map((stage) => {
    const ppPath: CollectivePath =
      stage.sendToNext === undefined ? { kind: 'none' } : ppSendPath(input.parallel, topology, stage.index);
    const ctx: OpTimeContext = {
      engine: base.engine,
      memBandwidthBytesPerSec: base.memBandwidthBytesPerSec,
      collectivePaths: { tp: base.tpPath, ep: base.epPath, pp: ppPath },
      collectiveAlgorithm: base.collectiveAlgorithm,
      elementwiseFusion: base.elementwiseFusion,
      warnings: base.warnings,
    };
    const layers = stage.layers.map((l) => layerResult(l.name, l.layerIndex, l.ops.map((op) => opTime(op, ctx))));
    const send = stage.sendToNext === undefined ? undefined : opTime(stage.sendToNext, ctx);
    return stageResult(stage.index, layers, send, options.overlap);
  });
  return phaseResult(stages, {
    phase,
    dp: input.parallel.dp,
    sequencesPerReplica: shape.sequencesPerReplica,
    microbatches: shape.microbatches,
    microbatchSize: shape.microbatchSize,
    promptLen: input.workload.promptLen,
    overlap: options.overlap,
  });
}

/**
 * Simulate one input: validate, apply modifiers, cost prefill (the whole prompt) and decode (one token at
 * `decodeContextLen`), and check the memory fit. Throws `Error(errors.join('\n'))` on an invalid input.
 */
export function simulate(input: SimInput): SimResult {
  const errors = validateInput(input);
  if (errors.length > 0) throw new Error(errors.join('\n'));
  const options: SimOptions = { ...DEFAULT_SIM_OPTIONS, ...input.options };
  const { parallel, workload, model, quant } = input;

  // 2. Modifiers, engine, memory bandwidth, collective paths.
  const hw = effectiveHardware(input.hardware);
  const device = hw.device;
  const engine = device.engines.find((e) => e.id === (input.hardware.engineId ?? device.engines[0].id));
  if (engine === undefined) throw new Error(`engine '${input.hardware.engineId}' not found`);
  const warnings: string[] = [
    ...hw.notes,
    ...checkPlacement(parallel, hw.topology).warnings,
    ...crossStageWarnings(model, parallel.pp),
  ];
  const algorithm: AllReduceAlgorithm = options.collectiveAlgorithm;
  const base = {
    engine,
    memBandwidthBytesPerSec: engineMemBandwidthBytesPerSec(engine, device.memory.bandwidthBytesPerSec),
    collectiveAlgorithm: algorithm,
    elementwiseFusion: options.elementwiseFusion,
    warnings,
    tpPath: collectivePath(groupSpan(parallel.tp, hw.topology), hw.topology),
    epPath: collectivePath(groupSpan(parallel.ep, hw.topology), hw.topology),
  };

  // 3. Phases. Formula: B_r = batch/dp; m = min(pp, B_r); b_mb = ceil(B_r/m)  (docs §7 "Pipeline and DP")
  const sequencesPerReplica = workload.batch / parallel.dp;
  const mb = microbatching(parallel.pp, sequencesPerReplica);
  const shape = { sequencesPerReplica, ...mb };
  const prefillCtx: LoweringContext = {
    phase: 'prefill',
    seqs: mb.microbatchSize,
    qLen: workload.promptLen,
    kvLen: workload.promptLen,
    quant,
    parallel,
  };
  const decodeCtx: LoweringContext = { ...prefillCtx, phase: 'decode', qLen: 1, kvLen: workload.decodeContextLen };
  const prefillLowered = lowerModel(model, prefillCtx);
  const decodeLowered = lowerModel(model, decodeCtx);
  const prefill = simulatePhase(prefillLowered, 'prefill', base, input, hw.topology, options, shape);
  const decode = simulatePhase(decodeLowered, 'decode', base, input, hw.topology, options, shape);

  // 4. Memory fit.
  const capacityBytes = input.hardware.memCapacityBytes ?? device.memory.capacityOptionsBytes[0];
  const memory = memoryReport({
    model,
    ctx: decodeCtx,
    sequencesPerReplica,
    contextTokens: Math.max(workload.promptLen, workload.decodeContextLen),
    lowered: [prefillLowered, decodeLowered],
    capacityBytes,
    kind: device.memory.kind,
  });
  if (device.memory.kind === 'unified') {
    warnings.push('memory: unified memory — capacity is shared with the CPU/OS; the usable fraction is OS-dependent and not modeled');
  }

  return {
    prefill,
    decode,
    memory,
    devicesUsed: parallel.tp * parallel.pp * parallel.dp,
    effectiveDevice: device,
    warnings: [...new Set(warnings)],
  };
}
