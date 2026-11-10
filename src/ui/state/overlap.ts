/**
 * Per-collective overlap and two-batch overlap in the UI (docs/simulation-model.md §7 "Stage time and overlap",
 * "Two-batch overlap"): which overlap classes a run has, and the options / deployment flag as the engine sees them.
 *
 * The state keeps what the user chose (`SimOptions.overlapByClass`, `DeploymentChoice.twoBatchOverlap`) even while a
 * class or the two-batch switch does not apply (fewer devices, a dense model); the SimInput carries them only where
 * they apply, so a stale choice changes nothing (as the all-to-all knobs, `parallelism.ts: onModel`). Pure TypeScript.
 */

import type { OverlapClass, SimOptions } from '../../sim/engine/types';
import type { ModelSpec, ParallelConfig } from '../../sim/ir/model';
import { hasMoe } from './deploymentInput';

/** Every overlap class, in display and URL order: TP all-reduce, EP all-to-all, PP send. */
export const OVERLAP_CLASSES: readonly OverlapClass[] = ['tp', 'ep', 'pp'];

/**
 * The overlap classes that have communication in this run: `tp` with TP > 1 (TP, expert-TP and DCP collectives),
 * `ep` on a MoE model with EP > TP (mode B: the all-to-alls and the DP attention + EP collectives), `pp` with PP > 1
 * (the send to the next stage). In {@link OVERLAP_CLASSES} order.
 */
export function overlapClassesOf(model: ModelSpec, parallel: ParallelConfig): OverlapClass[] {
  const present: Record<OverlapClass, boolean> = {
    tp: parallel.tp > 1,
    ep: hasMoe(model) && parallel.ep > parallel.tp,
    pp: parallel.pp > 1,
  };
  return OVERLAP_CLASSES.filter((c) => present[c]);
}

/**
 * True when two-batch overlap applies: a MoE model in mode B (EP > TP, the layout with the expert all-to-all it
 * hides) with nothing offloaded (the engine does not model it with offload).
 */
export function twoBatchOverlapApplies(
  model: ModelSpec,
  parallel: ParallelConfig,
  offloads: boolean,
): boolean {
  return hasMoe(model) && parallel.ep > parallel.tp && !offloads;
}

/**
 * `options` as the engine sees them: `overlapByClass` keeps only the classes the run has
 * ({@link overlapClassesOf}) and is removed when none is left, so absent overrides give the input from before.
 */
export function effectiveOptions(
  options: Partial<SimOptions>,
  model: ModelSpec,
  parallel: ParallelConfig,
): Partial<SimOptions> {
  const { overlapByClass, ...rest } = options;
  if (overlapByClass === undefined) return options;
  const kept: Partial<Record<OverlapClass, number>> = {};
  for (const c of overlapClassesOf(model, parallel))
    if (overlapByClass[c] !== undefined) kept[c] = overlapByClass[c];
  return Object.keys(kept).length > 0 ? { ...rest, overlapByClass: kept } : rest;
}
