/**
 * Per-field validation of the parallelism and workload inputs, and the valid degrees for hints such as
 * "Valid TP: 1, 2, 4, 8". The engine's `validateInput` stays the authority (buildInput.ts runs it once these
 * are clean); this module only attributes problems to the field that causes them.
 *
 * The UI offers TP, PP, DP and (MoE models only) EP. The device count is TP × PP × DP (`placement.ts` rank layout:
 * DP replicas inside each pipeline stage). EP adds no devices: in the engine's mode A (`lower/moe.ts`, `tp % ep == 0`)
 * the EP group lies inside the TP group; in mode B (`ep % tp == 0` and `(tp·dp) % ep == 0`, so only with DP > 1) it
 * spans `ep/tp` replicas. A valid EP also divides the expert count. See {@link effectiveParallel}.
 *
 * Attribution of engine messages relies on their wording (pinned by parallelism.test.ts):
 * - `blockShardingErrors` lines are `'<layer>: <message>'`; a message naming `ep (` goes to EP, else TP. The two
 *   MoE EP rules are reworded in UI words ({@link uiShardingMessage}).
 * - `checkPlacement` errors are prefixed `tp:`, `ep:` or `pp:`; the world-size error starts `tp·pp·dp =`.
 */

import type { Workload } from '../../sim/engine/types';
import { checkPlacement, expertTpCollectivePath } from '../../sim/engine/placement';
import { blockShardingErrors } from '../../sim/engine/validate';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../../sim/ir/model';
import type { TopologySpec } from '../../sim/hw/types';
import { hasMoe } from './deploymentInput';

/**
 * A field the UI can mark. `devices` is the device/node count; `offload` and `specDec` are the deployment sections
 * (offloading, speculative decoding); `general` is anything else.
 */
export type IssueField =
  | 'tp' | 'pp' | 'dp' | 'ep' | 'batch' | 'promptLen' | 'decodeContextLen' | 'devices' | 'bpw' | 'headBpw'
  | 'offload' | 'specDec' | 'general';

/** One problem with the inputs. Errors stop the simulation; warnings do not. */
export interface FieldIssue {
  /** The field to mark. */
  field: IssueField;
  /** Severity. */
  level: 'error' | 'warning';
  /** Human-readable message (what is wrong). */
  message: string;
}

/** Label of a parallel degree in messages. */
const DEGREE_LABEL: Record<'tp' | 'pp' | 'dp' | 'ep', string> = { tp: 'TP', pp: 'PP', dp: 'DP', ep: 'EP' };

/** Label of a workload field in messages. */
const WORKLOAD_LABEL: Record<'batch' | 'promptLen' | 'decodeContextLen', string> = {
  batch: 'Batch',
  promptLen: 'Prefill length',
  decodeContextLen: 'Decode context',
};

/** True for an integer ≥ 1. */
function isPositiveInteger(v: number): boolean {
  return Number.isInteger(v) && v >= 1;
}

/** Strips the `'<layer>: '` prefix of a `blockShardingErrors` line. */
function stripLayer(message: string): string {
  const i = message.indexOf(': ');
  return i < 0 ? message : message.slice(i + 2);
}

/** Field of a sharding message: EP when it names the expert-parallel degree, else TP. */
export function shardingField(message: string): 'tp' | 'ep' {
  return /\bep \(/.test(message) ? 'ep' : 'tp';
}

/**
 * A sharding message in UI words. The engine's two MoE EP rules (`moeParallelErrors`) become
 * "EP 3 must divide the 512 experts." and "EP 3 must divide TP 2."; with DP > 1 the second names the engine's mode-B
 * alternative too: "EP 3 must divide TP 2, or be a multiple of TP 2 that divides TP × DP = 4.". Any other message is
 * kept as is.
 */
export function uiShardingMessage(message: string): string {
  const experts = message.match(/experts \((\d+)\) must be divisible by ep \((\d+)\)/);
  if (experts) return `EP ${experts[2]} must divide the ${experts[1]} experts.`;
  const divide = message.match(/ep \((\d+)\) must divide tp \((\d+)\)(?:, or be a multiple of tp that divides tp·dp \((\d+)\))?/);
  if (divide) {
    const [, ep, tp, tpDp] = divide;
    // tp·dp = tp means DP 1: the multiple-of-TP alternative cannot hold, so it is not offered.
    if (tpDp === undefined || tpDp === tp) return `EP ${ep} must divide TP ${tp}.`;
    return `EP ${ep} must divide TP ${tp}, or be a multiple of TP ${tp} that divides TP × DP = ${tpDp}.`;
  }
  return message;
}

/** Field of a `checkPlacement` error by its prefix (`tp:`, `ep:`, `pp:`); anything else is about the devices. */
export function placementField(message: string): IssueField {
  const prefix = message.match(/^(tp|ep|pp):/)?.[1];
  return prefix === 'tp' || prefix === 'ep' || prefix === 'pp' ? prefix : 'devices';
}

/**
 * The parallel config the UI simulates from the user's degrees: EP is the user's only on a MoE model (a dense model
 * runs with EP 1, whatever the state or URL holds). Every other field, DP included, is kept.
 */
export function effectiveParallel(parallel: ParallelConfig, model: ModelSpec): ParallelConfig {
  return { ...parallel, ep: hasMoe(model) ? parallel.ep : 1 };
}

/**
 * The problems of a parallelism + workload on a model and topology, by field. `parallel` is the user's; it is
 * checked as {@link effectiveParallel} makes it (EP 1 on dense models):
 * - every degree (EP only on MoE models) and workload value is an integer ≥ 1 (if not, nothing else is checked);
 * - `tp·pp·dp` ≠ devices is an error on `devices` (the device count must equal TP × PP × DP: EP uses the TP and DP
 *   devices);
 * - `batch % dp == 0` (on `batch`: each replica serves `batch / dp` sequences), `pp ≤ layers` (on `pp`);
 * - the engine's sharding rules (`blockShardingErrors`; one message per distinct rule, layer prefix dropped;
 *   EP rules such as "EP must divide TP" on `ep`);
 * - the engine's placement rules (`checkPlacement`) by their prefix;
 * - the expert-TP group path (MoE with an expert-output norm and `tp/ep > 1`), on `ep`.
 */
export function fieldIssues(
  model: ModelSpec,
  quant: QuantConfig,
  parallel: ParallelConfig,
  workload: Workload,
  topology: TopologySpec,
): FieldIssue[] {
  const issues: FieldIssue[] = [];
  const error = (field: IssueField, message: string) => issues.push({ field, level: 'error', message });
  const effective = effectiveParallel(parallel, model);
  const degrees = hasMoe(model) ? (['tp', 'pp', 'dp', 'ep'] as const) : (['tp', 'pp', 'dp'] as const);
  for (const key of degrees) {
    if (!isPositiveInteger(effective[key])) error(key, `${DEGREE_LABEL[key]} must be a whole number ≥ 1.`);
  }
  for (const key of ['batch', 'promptLen', 'decodeContextLen'] as const) {
    if (!isPositiveInteger(workload[key])) error(key, `${WORKLOAD_LABEL[key]} must be a whole number ≥ 1.`);
  }
  if (issues.length > 0) return issues;

  // EP lies inside the TP group (mode A) or spans DP replicas (mode B): the device count must equal TP × PP × DP.
  const { tp, pp, dp, ep } = effective;
  const world = tp * pp * dp;
  const devices = topology.devicesPerNode * topology.nodes;
  if (world !== devices) {
    error('devices', `TP × PP × DP = ${world}, but the rig has ${devices} ${devices === 1 ? 'device' : 'devices'}. They must be equal.`);
  }
  // Each DP replica serves B_r = batch / dp sequences, so the batch must split evenly. The batch is what to change.
  if (workload.batch % dp !== 0) {
    error('batch', `Batch ${workload.batch} does not split evenly over DP ${dp}. Use a multiple of ${dp}.`);
  }
  const layers = model.layers.length;
  if (pp > layers) error('pp', `PP ${pp} exceeds the ${layers} layers of the model.`);

  const seen = new Set<string>();
  for (const line of blockShardingErrors({ model, quant, parallel: effective })) {
    // One engine error may hold several rules, one per line (e.g. a MoE block's two EP rules).
    for (const rule of stripLayer(line).split('\n')) {
      const message = uiShardingMessage(rule);
      if (seen.has(message)) continue;
      seen.add(message);
      error(shardingField(rule), message);
    }
  }
  if (world <= devices) {
    const placementErrors = checkPlacement(effective, topology).errors;
    for (const message of placementErrors) error(placementField(message), message);
    // As validateInput: the expert-TP group (etp = tp/ep > 1 ranks sharing experts) of a MoE block with an
    // expert-output norm needs a modeled path. Changing EP is what fixes it, so it goes on EP.
    const usesExpertTp =
      tp % ep === 0 &&
      tp / ep > 1 &&
      model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe' && b.expertOutputNorm === true));
    if (usesExpertTp && placementErrors.length === 0) {
      try {
        expertTpCollectivePath(effective, topology);
      } catch (e) {
        error('ep', `etp: ${(e as Error).message}`);
      }
    }
  }
  return issues;
}

/** True when `issues` holds an error. */
export function hasErrors(issues: FieldIssue[]): boolean {
  return issues.some((i) => i.level === 'error');
}

/** Most candidate degrees {@link validDegrees} tries (1..this). Count. */
export const MAX_DEGREE_CANDIDATES = 256;

/**
 * The degrees in 1..devices that give no error for `field`, with the other degrees as in `parallel`, except that
 * TP is checked with EP = 1 (EP is chosen after TP: a TP that the current EP does not divide shows the error on EP).
 * The device-count equality (TP × PP × DP = devices) is not part of this check: it is a property of the degrees
 * together, shown on the device count field, so a hint lists every degree that is individually valid. Neither is the
 * batch split over DP, which is shown on the batch field. EP is checked at the current TP and DP, so with DP > 1 the
 * hint includes the mode-B values (multiples of TP that divide TP × DP). EP on a dense model is `[1]`.
 * Used for hints like "Valid TP: 1, 2, 4, 8".
 */
export function validDegrees(
  field: 'tp' | 'pp' | 'dp' | 'ep',
  model: ModelSpec,
  quant: QuantConfig,
  parallel: ParallelConfig,
  workload: Workload,
  topology: TopologySpec,
): number[] {
  if (field === 'ep' && !hasMoe(model)) return [1];
  const devices = topology.devicesPerNode * topology.nodes;
  const valid: number[] = [];
  // Bounded so a huge rig cannot stall the UI.
  for (let k = 1; k <= Math.min(devices, MAX_DEGREE_CANDIDATES); k++) {
    const candidate: ParallelConfig = { ...parallel, [field]: k, ...(field === 'tp' ? { ep: 1 } : {}) };
    const blocking = fieldIssues(model, quant, candidate, workload, topology).filter(
      (i) => i.level === 'error' && i.field === field,
    );
    if (blocking.length === 0) valid.push(k);
  }
  return valid;
}

/**
 * TP, PP, DP and EP for this model and topology, used when a hardware choice changes the device count:
 * - DP is never raised on its own: the user's `current.dp` is kept when it is > 1, divides the device count, and a
 *   TP × PP split of the remaining `devices / dp` passes {@link fieldIssues} (batch split included); otherwise DP is 1.
 * - TP and PP use every remaining device (TP × PP × DP = devices) and pass {@link fieldIssues} with EP = 1: the
 *   largest valid TP (at most {@link MAX_DEGREE_CANDIDATES}), with PP = devices / (TP · DP). Falls back to TP = 1,
 *   PP = 1, DP = 1 when no split is valid (the device-count error is then shown).
 * - EP is never raised on its own either: the user's `current.ep` is kept when it is > 1, the model is MoE, and the
 *   chosen TP, PP and DP give no error with it; otherwise EP is 1.
 */
export function chooseDegrees(
  model: ModelSpec,
  quant: QuantConfig,
  workload: Workload,
  topology: TopologySpec,
  current: { dp?: number; ep?: number } = {},
): { tp: number; pp: number; dp: number; ep: number } {
  const currentDp = current.dp ?? 1;
  const currentEp = current.ep ?? 1;
  const keptDp = currentDp > 1 ? chooseTpPp(model, quant, workload, topology, currentDp) : undefined;
  const { tp, pp, dp } = keptDp
    ? { ...keptDp, dp: currentDp }
    : { ...(chooseTpPp(model, quant, workload, topology, 1) ?? { tp: 1, pp: 1 }), dp: 1 };
  if (currentEp > 1 && hasMoe(model)) {
    const kept: ParallelConfig = { tp, pp, dp, ep: currentEp };
    if (!hasErrors(fieldIssues(model, quant, kept, workload, topology))) return { tp, pp, dp, ep: currentEp };
  }
  return { tp, pp, dp, ep: 1 };
}

/**
 * The TP × PP split of {@link chooseDegrees} at a given DP, searched with EP = 1: TP × PP = devices / dp, the
 * largest TP that gives no error. Undefined when dp does not divide the devices or no split is valid.
 */
function chooseTpPp(
  model: ModelSpec,
  quant: QuantConfig,
  workload: Workload,
  topology: TopologySpec,
  dp: number,
): { tp: number; pp: number } | undefined {
  const devices = topology.devicesPerNode * topology.nodes;
  // Bounded like validDegrees, whatever the requested device count (a huge or unsafe count cannot stall the UI):
  // TP never exceeds MAX_DEGREE_CANDIDATES, and counts that are not safe positive integers get no search.
  if (!Number.isSafeInteger(devices) || devices < 1 || !Number.isSafeInteger(dp) || dp < 1 || devices % dp !== 0) {
    return undefined;
  }
  // Formula: tp · pp = devices / dp
  const perReplica = devices / dp;
  for (let tp = Math.min(perReplica, MAX_DEGREE_CANDIDATES); tp >= 1; tp--) {
    if (perReplica % tp !== 0) continue;
    const pp = perReplica / tp;
    // Pipeline stages cannot outnumber layers: no smaller TP can fit either.
    if (pp > model.layers.length) break;
    const parallel: ParallelConfig = { tp, pp, dp, ep: 1 };
    if (!hasErrors(fieldIssues(model, quant, parallel, workload, topology))) return { tp, pp };
  }
  // Not pp = devices: an absurd pipeline depth would only make the checks slow. The device-count error is shown.
  return undefined;
}
