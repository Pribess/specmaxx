/**
 * Input validation for `simulate`: parallel degrees and workload are positive integers, the batch divides
 * into DP replicas, PP fits the layers, every block shards evenly over TP (heads, widths) and EP (MoE modes),
 * the parallelism fits the topology and its links, the effective device (after modifiers, or the base) and the
 * topology pass hardware validation (incl. positive rates), the modifiers are valid, and the engine and memory
 * capacity selections exist, and the model spec is structurally valid (`modelSpecErrors`: cross-layer references,
 * KDA fields, residual-stream widths).
 * See docs/simulation-model.md §5 (Parallelism) and §4 ("Device variants and modifiers").
 */

import { validateDevice, validateDeviceModifiers, validateLinkModifiers, validateTopology } from '../hw/validate';
import { applyDeviceModifiers } from '../hw/modifiers';
import { modelSpecErrors } from '../ir/validateModel';
import { lowerBlock, type LoweringContext } from '../lower/lower';
import { checkPlacement } from './placement';
import type { SimInput } from './types';

/** Errors for a field that must be an integer ≥ 1. */
function positiveIntegerErrors(value: number, label: string): string[] {
  return Number.isInteger(value) && value >= 1 ? [] : [`${label} must be an integer ≥ 1 (got ${value})`];
}

/**
 * Sharding errors of every block (heads, KV heads, widths divisible by tp; MoE expert parallelism), found by
 * lowering each block once for a one-token step: the lowering rules throw on an uneven split. Each message is
 * prefixed with the layer name (`'<layer>: <rule message>'`); identical messages are reported once.
 *
 * Exported so the UI can attribute TP / EP errors to their fields (and list the valid degrees) without a full
 * {@link validateInput}, which calls it too. Only `model`, `quant` and `parallel` are read; the workload and
 * hardware do not affect sharding. Rules: docs/simulation-model.md §5 ("TP sharding", "KV-head replication",
 * "MoE modes A/B").
 */
export function blockShardingErrors(input: Pick<SimInput, 'model' | 'quant' | 'parallel'>): string[] {
  const ctx: LoweringContext = {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 2,
    quant: input.quant,
    parallel: input.parallel,
  };
  const errors = new Set<string>();
  for (const layer of input.model.layers) {
    for (const block of layer.blocks) {
      try {
        lowerBlock(block, ctx);
      } catch (e) {
        errors.add(`${layer.name}: ${(e as Error).message}`);
      }
    }
  }
  return [...errors];
}

/**
 * Check a simulation input. Returns human-readable error lines; empty = valid.
 * `simulate()` throws `new Error(errors.join('\n'))` if any.
 */
export function validateInput(input: SimInput): string[] {
  const { model, parallel, workload, hardware } = input;
  const errors: string[] = [];

  // Parallel degrees and workload.
  for (const key of ['tp', 'pp', 'dp', 'ep'] as const) errors.push(...positiveIntegerErrors(parallel[key], `parallel.${key}`));
  errors.push(...positiveIntegerErrors(workload.batch, 'workload.batch'));
  errors.push(...positiveIntegerErrors(workload.promptLen, 'workload.promptLen'));
  errors.push(...positiveIntegerErrors(workload.decodeContextLen, 'workload.decodeContextLen'));
  if (errors.length > 0) return errors;
  if (workload.batch % parallel.dp !== 0) {
    errors.push(`workload.batch (${workload.batch}) must be divisible by dp (${parallel.dp})`);
  }
  if (parallel.pp > model.layers.length) {
    errors.push(`pp (${parallel.pp}) must not exceed the number of layers (${model.layers.length})`);
  }
  const hasMoe = model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe'));
  if (parallel.ep > 1 && !hasMoe) errors.push(`ep (${parallel.ep}) > 1 but the model has no MoE block`);
  errors.push(...modelSpecErrors(model));
  errors.push(...blockShardingErrors(input));

  // Hardware: the EFFECTIVE device (after modifiers, or the base itself) is always validated: physical sanity,
  // rates used as divisors > 0, every number sourced. Then the engine and capacity selection.
  const deviceModifiers = hardware.deviceModifiers ?? [];
  const deviceErrors =
    deviceModifiers.length > 0
      ? validateDeviceModifiers(hardware.device, deviceModifiers)
      : validateDevice(hardware.device);
  errors.push(...deviceErrors);
  if (deviceErrors.length === 0) {
    const device = deviceModifiers.length > 0 ? applyDeviceModifiers(hardware.device, deviceModifiers) : hardware.device;
    if (hardware.engineId !== undefined && !device.engines.some((e) => e.id === hardware.engineId)) {
      errors.push(`hardware.engineId '${hardware.engineId}' is not an engine of ${device.id} (${device.engines.map((e) => e.id).join(', ')})`);
    }
    if (hardware.memCapacityBytes !== undefined && !device.memory.capacityOptionsBytes.includes(hardware.memCapacityBytes)) {
      errors.push(
        `hardware.memCapacityBytes ${hardware.memCapacityBytes} is not one of ${device.id}'s capacity options ` +
          `(${device.memory.capacityOptionsBytes.join(', ')})`,
      );
    }
  }
  // The topology and its base links are always validated; modified links again after their modifiers.
  const topology = hardware.topology;
  errors.push(...validateTopology(topology));
  for (const level of ['intraNode', 'interNode'] as const) {
    const mods = hardware.linkModifiers?.[level] ?? [];
    if (mods.length === 0) continue;
    const link = topology[level];
    if (link === undefined) errors.push(`hardware.linkModifiers.${level}: topology ${topology.id} has no ${level} link`);
    else errors.push(...validateLinkModifiers(link, mods).map((e) => `${level}: ${e}`));
  }

  // Placement on the topology (world size, group spans, required links).
  errors.push(...checkPlacement(parallel, topology).errors);
  return errors;
}
