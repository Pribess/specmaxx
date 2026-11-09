/**
 * Per-field validation of the parallelism and workload inputs, and the valid degrees for hints such as
 * "Valid TP: 1, 2, 4, 8". The engine's `validateInput` stays the authority (buildInput.ts runs it once these
 * are clean); this module only attributes problems to the field that causes them.
 *
 * Attribution of engine messages relies on their wording (pinned by parallelism.test.ts):
 * - `blockShardingErrors` lines are `'<layer>: <message>'`; a message naming `ep (` goes to EP, else TP.
 * - `checkPlacement` errors are prefixed `tp:`, `ep:` or `pp:`; the world-size error starts `tp·pp·dp =`.
 */

import type { Workload } from '../../sim/engine/types';
import { checkPlacement } from '../../sim/engine/placement';
import { blockShardingErrors } from '../../sim/engine/validate';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../../sim/ir/model';
import type { TopologySpec } from '../../sim/hw/types';

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

/** Field of a `checkPlacement` error by its prefix (`tp:`, `ep:`, `pp:`); anything else is about the devices. */
export function placementField(message: string): IssueField {
  const prefix = message.match(/^(tp|ep|pp):/)?.[1];
  return prefix === 'tp' || prefix === 'ep' || prefix === 'pp' ? prefix : 'devices';
}

/**
 * The problems of a parallelism + workload on a model and topology, by field:
 * - every degree and workload value is an integer ≥ 1 (if not, nothing else is checked);
 * - DP and EP are not user-facing: they are treated as 1 whatever `parallel` says;
 * - `tp·pp` ≠ devices is an error on `devices` (the device count must equal TP × PP);
 * - `pp ≤ layers` (on `pp`);
 * - the engine's sharding rules (`blockShardingErrors`; one message per distinct rule, layer prefix dropped);
 * - the engine's placement rules (`checkPlacement`) by their prefix.
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
  for (const key of ['tp', 'pp'] as const) {
    if (!isPositiveInteger(parallel[key])) error(key, `${DEGREE_LABEL[key]} must be a whole number ≥ 1.`);
  }
  for (const key of ['batch', 'promptLen', 'decodeContextLen'] as const) {
    if (!isPositiveInteger(workload[key])) error(key, `${WORKLOAD_LABEL[key]} must be a whole number ≥ 1.`);
  }
  if (issues.length > 0) return issues;

  // DP and EP are fixed at 1 in the UI: the device count must equal TP × PP.
  const { tp, pp } = parallel;
  const world = tp * pp;
  const devices = topology.devicesPerNode * topology.nodes;
  if (world !== devices) {
    error('devices', `TP × PP = ${world}, but the rig has ${devices} ${devices === 1 ? 'device' : 'devices'}. They must be equal.`);
  }
  const layers = model.layers.length;
  if (pp > layers) error('pp', `PP ${pp} exceeds the ${layers} layers of the model.`);

  const seen = new Set<string>();
  for (const line of blockShardingErrors({ model, quant, parallel: { ...parallel, dp: 1, ep: 1 } })) {
    // One engine error may hold several rules, one per line (e.g. a MoE block's two EP rules).
    for (const message of stripLayer(line).split('\n')) {
      if (seen.has(message)) continue;
      seen.add(message);
      error(shardingField(message), message);
    }
  }
  if (world <= devices) {
    for (const message of checkPlacement({ ...parallel, dp: 1, ep: 1 }, topology).errors) error(placementField(message), message);
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
 * The degrees in 1..devices that give no error for `field`, with the other degrees as in `parallel`, except:
 * The device-count equality (TP × PP = devices) is not part of this check: it is a property of the pair, shown on
 * the device count field, so a hint lists every degree that is individually valid. Used for hints like "Valid TP: 1, 2, 4, 8".
 */
export function validDegrees(
  field: 'tp' | 'pp',
  model: ModelSpec,
  quant: QuantConfig,
  parallel: ParallelConfig,
  workload: Workload,
  topology: TopologySpec,
): number[] {
  const devices = topology.devicesPerNode * topology.nodes;
  const valid: number[] = [];
  // Bounded so a huge rig cannot stall the UI.
  for (let k = 1; k <= Math.min(devices, MAX_DEGREE_CANDIDATES); k++) {
    const candidate: ParallelConfig = { ...parallel, [field]: k };
    const blocking = fieldIssues(model, quant, candidate, workload, topology).filter(
      (i) => i.level === 'error' && i.field === field,
    );
    if (blocking.length === 0) valid.push(k);
  }
  return valid;
}

/**
 * TP and PP that use every device (TP × PP = devices) and pass {@link fieldIssues} for this model and topology:
 * the largest valid TP (at most {@link MAX_DEGREE_CANDIDATES}), with PP = devices / TP. Falls back to TP = 1, PP = 1 when no split is valid (the
 * device-count error is then shown). Used when a hardware choice changes the device count.
 */
export function chooseDegrees(
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
