/**
 * The one memory-fit verdict of the UI (pure, no React): the hero, the mini result, the memory section and the batch
 * sweep all derive from it, so they cannot disagree. A configuration fits when every pipeline stage fits its device
 * AND every off-device tier (host RAM, SSD) fits its capacity.
 */

import type { TierMemory } from '../../sim/engine/types';
import type { MemorySummary, StageMemory } from './summary';

/** Fraction of per-device capacity above which a fitting stage has little headroom (dimensionless). */
export const HIGH_MEMORY_UTILISATION = 0.9;

/** Memory severity, ordered conceptually from comfortable fit to capacity overflow. */
export type MemoryState = 'fits' | 'warning' | 'danger';

/**
 * THE definition of "fits": every stage fits its device and every off-device tier fits its capacity. Takes only the
 * verdict flags so the engine's report and the UI summary can both use it.
 */
export function memoryFits(parts: {
  stages: { fits: boolean }[];
  tiers?: { fits: boolean }[];
}): boolean {
  return parts.stages.every((s) => s.fits) && (parts.tiers ?? []).every((t) => t.fits);
}

/** Classifies one stage using its physical fit verdict and total/capacity byte ratio. */
export function memoryState(stage: StageMemory): MemoryState {
  if (!stage.fits) return 'danger';
  return stage.totalBytes / stage.capacityBytes > HIGH_MEMORY_UTILISATION ? 'warning' : 'fits';
}

/** The overall verdict of a memory report. */
export interface MemoryVerdict {
  /** Every stage and every off-device tier fits ({@link memoryFits}). */
  fits: boolean;
  /** `danger` when anything does not fit, else `warning` when a stage is above the headroom threshold, else `fits`. */
  state: MemoryState;
  /** The stage to show: the first that does not fit, else the fullest one above the threshold; absent when none. */
  worstStage?: StageMemory;
  /** The off-device tiers that do not fit their capacity (empty when all fit). */
  failingTiers: TierMemory[];
}

/** Derives the {@link MemoryVerdict} of a memory report. */
export function memoryVerdict(memory: Pick<MemorySummary, 'stages' | 'tiers'>): MemoryVerdict {
  const failingTiers = (memory.tiers ?? []).filter((t) => !t.fits);
  const worstStage =
    memory.stages.find((s) => memoryState(s) === 'danger') ??
    memory.stages
      .filter((s) => memoryState(s) === 'warning')
      .sort((a, b) => b.totalBytes / b.capacityBytes - a.totalBytes / a.capacityBytes)[0];
  const fits = memoryFits({ stages: memory.stages, tiers: memory.tiers });
  const state: MemoryState = !fits ? 'danger' : worstStage !== undefined ? 'warning' : 'fits';
  return { fits, state, worstStage, failingTiers };
}
