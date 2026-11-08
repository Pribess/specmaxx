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
