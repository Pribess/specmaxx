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

/** A field the UI can mark. `devices` is the device/node count; `general` is anything else. */
export type IssueField = 'tp' | 'pp' | 'dp' | 'ep' | 'batch' | 'promptLen' | 'decodeContextLen' | 'devices' | 'general';

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
const DEGREE_LABEL: Record<keyof ParallelConfig, string> = { tp: 'TP', pp: 'PP', dp: 'DP', ep: 'EP' };

/** Label of a workload field in messages. */
const WORKLOAD_LABEL: Record<keyof Workload, string> = {
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
 * - `tp·pp·dp` > devices is an error on `devices`, < devices a warning ("2 of 4 devices are unused");
 * - `batch % dp == 0` (on `batch`), `pp ≤ layers` (on `pp`), `ep > 1` needs a MoE block (on `ep`);
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
  for (const key of ['tp', 'pp', 'dp', 'ep'] as const) {
    if (!isPositiveInteger(parallel[key])) error(key, `${DEGREE_LABEL[key]} must be a whole number ≥ 1.`);
  }
  for (const key of ['batch', 'promptLen', 'decodeContextLen'] as const) {
    if (!isPositiveInteger(workload[key])) error(key, `${WORKLOAD_LABEL[key]} must be a whole number ≥ 1.`);
  }
  if (issues.length > 0) return issues;

  const { tp, pp, dp, ep } = parallel;
  const world = tp * pp * dp;
  const devices = topology.devicesPerNode * topology.nodes;
  if (world > devices) {
    error('devices', `TP × PP × DP = ${world} needs ${world} devices; the rig has ${devices}.`);
  } else if (world < devices) {
    issues.push({ field: 'devices', level: 'warning', message: `${devices - world} of ${devices} devices are unused (TP × PP × DP = ${world}).` });
  }
  if (workload.batch % dp !== 0) error('batch', `Batch ${workload.batch} does not split evenly over DP ${dp}.`);
  const layers = model.layers.length;
  if (pp > layers) error('pp', `PP ${pp} exceeds the ${layers} layers of the model.`);
  const hasMoe = model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe'));
  if (ep > 1 && !hasMoe) error('ep', `EP ${ep} needs a mixture-of-experts model; this one is dense.`);

  const seen = new Set<string>();
  for (const line of blockShardingErrors({ model, quant, parallel })) {
    // One engine error may hold several rules, one per line (e.g. a MoE block's two EP rules).
    for (const message of stripLayer(line).split('\n')) {
      if (seen.has(message)) continue;
      seen.add(message);
      error(shardingField(message), message);
    }
  }
  if (world <= devices) {
    for (const message of checkPlacement(parallel, topology).errors) error(placementField(message), message);
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
 * TP is checked with `ep = 1` (EP is chosen after TP), EP at the current TP and DP. A degree is also rejected
 * when it makes the world exceed the devices (TP, PP, DP). Used for hints like "Valid TP: 1, 2, 4, 8".
 */
export function validDegrees(
  field: keyof ParallelConfig,
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
    const candidate: ParallelConfig = { ...parallel, [field]: k, ...(field === 'tp' ? { ep: 1 } : {}) };
    // Batch divisibility is the batch field's problem, not DP's: check DP with a batch it divides.
    const load = field === 'dp' ? { ...workload, batch: k } : workload;
    const blocking = fieldIssues(model, quant, candidate, load, topology).filter(
      (i) => i.level === 'error' && (i.field === field || (field !== 'ep' && i.field === 'devices')),
    );
    if (blocking.length === 0) valid.push(k);
  }
  return valid;
}
