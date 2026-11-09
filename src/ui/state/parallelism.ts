/**
 * Per-field validation of the parallelism and workload inputs, and the valid degrees for hints such as
 * "Valid TP: 1, 2, 4, 8". The engine's `validateInput` stays the authority (buildInput.ts runs it once these
 * are clean); this module only attributes problems to the field that causes them.
 *
 * The UI offers TP, PP and (MoE models only) EP; DP is fixed at 1. With DP = 1 the engine's EP is its mode A
 * (`lower/moe.ts`): the EP group lies inside the TP group, so EP adds no devices (devices = TP × PP) and a valid
 * EP divides both TP and the expert count. See {@link effectiveParallel}.
 *
 * Attribution of engine messages relies on their wording (pinned by parallelism.test.ts):
 * - `blockShardingErrors` lines are `'<layer>: <message>'`; a message naming `ep (` goes to EP, else TP. The two
 *   MoE EP rules are reworded without DP ({@link uiShardingMessage}).
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
 * "EP 3 must divide the 512 experts." and "EP 3 must divide TP 2." (the engine's "or a multiple of tp that divides
 * tp·dp" alternative needs DP > 1, which the UI does not offer); any other message is kept as is.
 */
export function uiShardingMessage(message: string): string {
  const experts = message.match(/experts \((\d+)\) must be divisible by ep \((\d+)\)/);
  if (experts) return `EP ${experts[2]} must divide the ${experts[1]} experts.`;
  const divide = message.match(/ep \((\d+)\) must divide tp \((\d+)\)/);
  if (divide) return `EP ${divide[1]} must divide TP ${divide[2]}.`;
  return message;
}

/** Field of a `checkPlacement` error by its prefix (`tp:`, `ep:`, `pp:`); anything else is about the devices. */
export function placementField(message: string): IssueField {
  const prefix = message.match(/^(tp|ep|pp):/)?.[1];
  return prefix === 'tp' || prefix === 'ep' || prefix === 'pp' ? prefix : 'devices';
}

/**
 * The parallel config the UI simulates from the user's degrees: DP is always 1, and EP is the user's only on a
 * MoE model (a dense model runs with EP 1, whatever the state or URL holds). Every other field is kept.
 */
export function effectiveParallel(parallel: ParallelConfig, model: ModelSpec): ParallelConfig {
  return { ...parallel, dp: 1, ep: hasMoe(model) ? parallel.ep : 1 };
}

/**
 * The problems of a parallelism + workload on a model and topology, by field. `parallel` is the user's; it is
 * checked as {@link effectiveParallel} makes it (DP 1, EP 1 on dense models):
 * - every degree (EP only on MoE models) and workload value is an integer ≥ 1 (if not, nothing else is checked);
 * - `tp·pp` ≠ devices is an error on `devices` (the device count must equal TP × PP: EP uses the TP devices);
 * - `pp ≤ layers` (on `pp`);
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
  const degrees = hasMoe(model) ? (['tp', 'pp', 'ep'] as const) : (['tp', 'pp'] as const);
  for (const key of degrees) {
    if (!isPositiveInteger(effective[key])) error(key, `${DEGREE_LABEL[key]} must be a whole number ≥ 1.`);
  }
  for (const key of ['batch', 'promptLen', 'decodeContextLen'] as const) {
    if (!isPositiveInteger(workload[key])) error(key, `${WORKLOAD_LABEL[key]} must be a whole number ≥ 1.`);
  }
  if (issues.length > 0) return issues;

  // DP is fixed at 1 and EP lies inside the TP group: the device count must equal TP × PP.
  const { tp, pp, ep } = effective;
  const world = tp * pp;
  const devices = topology.devicesPerNode * topology.nodes;
  if (world !== devices) {
    error('devices', `TP × PP = ${world}, but the rig has ${devices} ${devices === 1 ? 'device' : 'devices'}. They must be equal.`);
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
 * The device-count equality (TP × PP = devices) is not part of this check: it is a property of the pair, shown on
 * the device count field, so a hint lists every degree that is individually valid. EP on a dense model is `[1]`.
 * Used for hints like "Valid TP: 1, 2, 4, 8".
 */
export function validDegrees(
  field: 'tp' | 'pp' | 'ep',
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
 * TP, PP and EP for this model and topology, used when a hardware choice changes the device count:
 * - TP and PP use every device (TP × PP = devices) and pass {@link fieldIssues} with EP = 1: the largest valid TP
 *   (at most {@link MAX_DEGREE_CANDIDATES}), with PP = devices / TP. Falls back to TP = 1, PP = 1 when no split is
 *   valid (the device-count error is then shown).
 * - EP is never raised on its own: the user's `currentEp` is kept when it is > 1, the model is MoE, and the chosen
 *   TP and PP give no error with it; otherwise EP is 1.
 */
export function chooseDegrees(
  model: ModelSpec,
  quant: QuantConfig,
  workload: Workload,
  topology: TopologySpec,
  currentEp = 1,
): { tp: number; pp: number; ep: number } {
  const { tp, pp } = chooseTpPp(model, quant, workload, topology);
  if (currentEp > 1 && hasMoe(model)) {
    const kept: ParallelConfig = { tp, pp, dp: 1, ep: currentEp };
    if (!hasErrors(fieldIssues(model, quant, kept, workload, topology))) return { tp, pp, ep: currentEp };
  }
  return { tp, pp, ep: 1 };
}

/** The TP × PP split of {@link chooseDegrees}, searched with EP = 1. */
function chooseTpPp(
  model: ModelSpec,
  quant: QuantConfig,
  workload: Workload,
  topology: TopologySpec,
): { tp: number; pp: number } {
  const devices = topology.devicesPerNode * topology.nodes;
  // Bounded like validDegrees, whatever the requested device count (a huge or unsafe count cannot stall the UI):
  // TP never exceeds MAX_DEGREE_CANDIDATES, and counts that are not safe positive integers get no search.
  if (Number.isSafeInteger(devices) && devices >= 1) {
    for (let tp = Math.min(devices, MAX_DEGREE_CANDIDATES); tp >= 1; tp--) {
      if (devices % tp !== 0) continue;
      const pp = devices / tp;
      // Pipeline stages cannot outnumber layers: no smaller TP can fit either.
      if (pp > model.layers.length) break;
      const parallel: ParallelConfig = { tp, pp, dp: 1, ep: 1 };
      if (!hasErrors(fieldIssues(model, quant, parallel, workload, topology))) return { tp, pp };
    }
  }
  // Not pp = devices: an absurd pipeline depth would only make the checks slow. The device-count error is shown.
  return { tp: 1, pp: 1 };
}
