/**
 * Which weight groups of a decoder layer are off-device (design doc §2.1), from the resolved placement of the
 * lowering context. Small pure lookups shared by `lowerModel`, `lowerMoe`, `lowerNgram` and the memory fit.
 * See docs/simulation-model.md §12 ("Placement").
 *
 * Precedence: a layer of the `denseLayers` range is streamed whole (all its weights, experts and tables included),
 * so no expert or table transfer is emitted for it on top.
 */

import type { LoweringContext, OffloadLowering } from './lower';

/** True if zero-based layer index `i` lies in the half-open range `[from, to)`. */
export function inLayerRange(range: { from: number; to: number }, i: number): boolean {
  return i >= range.from && i < range.to;
}

/** The dense-layer placement streaming layer `i` whole every step, or undefined. */
export function denseLayerPlacementAt(ctx: LoweringContext, i: number | undefined): OffloadLowering['denseLayers'] {
  const dense = ctx.offload?.denseLayers;
  return i !== undefined && dense !== undefined && inLayerRange(dense.layers, i) ? dense : undefined;
}

/**
 * The routed-expert placement of layer `i`, or undefined when its experts are on the device (no offload, layer outside
 * the range, the layer is streamed whole, or `i` is unknown: a draft layer).
 */
export function expertPlacementAt(ctx: LoweringContext, i: number | undefined): OffloadLowering['routedExperts'] {
  const experts = ctx.offload?.routedExperts;
  if (i === undefined || experts === undefined || !inLayerRange(experts.layers, i)) return undefined;
  return denseLayerPlacementAt(ctx, i) === undefined ? experts : undefined;
}

/** The lookup-table placement of the n-gram blocks of layer `i`, or undefined when the tables are on the device. */
export function tablePlacementAt(ctx: LoweringContext, i: number | undefined): OffloadLowering['lookupTables'] {
  const tables = ctx.offload?.lookupTables;
  if (i === undefined || tables === undefined) return undefined;
  return denseLayerPlacementAt(ctx, i) === undefined ? tables : undefined;
}

/**
 * What {@link prefetchedFraction} needs to know about one transfer op: its group and, for routed experts, the
 * expected misses `M` and the non-resident experts `E_loc − C` of the layer (both in experts, real).
 */
export type PrefetchTarget =
  | { group: 'routed_experts'; misses: number; nonResidentExperts: number }
  | { group: 'dense_layer' | 'lookup_table' };

/**
 * q: the share of one transfer op's time that is prefetched, i.e. may hide behind the stage's compute
 * (docs/simulation-model.md §12 "Expert prefetch"), from the prefetch accuracy p (the predictor's measured recall:
 * the share of the needed experts it picks one layer ahead):
 * - p = 0: q = 0 for every group (no prefetching at all: the former `offloadOverlap 'none'`);
 * - routed experts, p > 0: q = max(p, M/(E_loc − C)). The engine takes the better of two strategies: prefetch its
 *   predictions, of which a share p are needed misses; or, when nearly every non-resident expert is needed (prefill,
 *   large batches), copy the non-resident experts unconditionally, of which a share M/(E_loc − C) are needed. So
 *   q = p at batch 1 (M ≪ E_loc − C) and q = 1 in prefill (M = E_loc − C);
 * - dense layers and table rows, p > 0: q = 1. Their bytes are known before the step (every layer streams; rows
 *   come from token ids), and an engine that prefetches at all prefetches these deterministic transfers.
 * So p = 1 gives q = 1 everywhere (the former `'ideal'`). Dimensionless, in [0, 1].
 */
export function prefetchedFraction(p: number, target: PrefetchTarget): number {
  // Formula: q = 0   (p = 0, every group)
  if (!(p > 0)) return 0;
  if (target.group !== 'routed_experts') {
    // Formula: q = 1   (dense layers, table rows; p > 0)
    return 1;
  }
  const { misses: M, nonResidentExperts } = target;
  if (!(nonResidentExperts > 0)) return 1;
  // Formula: q = max(p, M/(E_loc − C))   (routed experts; p > 0: predicted, or unconditional copy)
  // See docs/simulation-model.md §12 ("Expert prefetch")
  const neededShare = M / nonResidentExperts;
  return Math.min(1, Math.max(p, neededShare));
}
