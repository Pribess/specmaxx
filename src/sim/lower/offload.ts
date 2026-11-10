/**
 * Which weight groups of a decoder layer are off-device (design doc §2.1), from the resolved placement of the
 * lowering context. Small pure lookups shared by `lowerModel`, `lowerMoe`, `lowerNgram` and the memory fit.
 * See docs/simulation-model.md §12 ("Placement").
 *
 * Each group may hold several placement rules over disjoint layer ranges (e.g. experts of layers 0–23 in host RAM,
 * 24–47 on the SSD); a layer takes the one rule whose range contains it. (Whole-layer streaming, `denseLayers`, was
 * removed: docs/simulation-model.md §12 "Placement".)
 */

import type { ExpertPlacement, LoweringContext, OffloadLowering, TablePlacement } from './lower';

/** True if zero-based layer index `i` lies in the half-open range `[from, to)`. */
export function inLayerRange(range: { from: number; to: number }, i: number): boolean {
  return i >= range.from && i < range.to;
}

/** The placement among `rules` whose layer range contains layer `i` (ranges are disjoint, so at most one), or undefined. */
function ruleAt<R extends { layers: { from: number; to: number } }>(
  rules: readonly R[] | undefined,
  i: number | undefined,
): R | undefined {
  if (i === undefined || rules === undefined) return undefined;
  return rules.find((r) => inLayerRange(r.layers, i));
}

/**
 * The routed-expert placement of layer `i`, or undefined when its experts are on the device (no offload, layer outside
 * every rule's range, or `i` is unknown: a draft layer).
 */
export function expertPlacementAt(
  ctx: LoweringContext,
  i: number | undefined,
): ExpertPlacement | undefined {
  return ruleAt(ctx.offload?.routedExperts, i);
}

/**
 * The routed-expert placement that also offloads the speculative drafter's own MoE experts (`includeDraft`), or
 * undefined. Validation allows `includeDraft` on at most one rule, so the first match is the only one.
 */
export function draftExpertPlacement(
  offload: OffloadLowering | undefined,
): ExpertPlacement | undefined {
  return offload?.routedExperts?.find((r) => r.includeDraft);
}

/**
 * The lookup-table placement of the n-gram blocks of layer `i` (the rule whose range contains it), or undefined when
 * its tables are on the device (no rule covers it, or `i` is unknown).
 */
export function tablePlacementAt(
  ctx: LoweringContext,
  i: number | undefined,
): TablePlacement | undefined {
  return ruleAt(ctx.offload?.lookupTables, i);
}

/**
 * What {@link prefetchedFraction} needs to know about one transfer op: its group and, for routed experts, the
 * expected misses `M` and the non-resident experts `E_loc − C` of the layer (both in experts, real); for token
 * embedding rows, the phase of the step; for the KV cache, whether the reads follow a top-k selection made this step
 * (`topK`, decode of a sparse layer), and then the missed tokens `M` and the non-resident past tokens `N_past − N_res`
 * per sequence (tokens, real).
 */
export type PrefetchTarget =
  | { group: 'routed_experts'; misses: number; nonResidentExperts: number }
  | { group: 'lookup_table' }
  | { group: 'token_embedding'; phase: 'prefill' | 'decode' }
  | { group: 'kv_cache'; topK: boolean; misses: number; nonResidentTokens: number };

/**
 * q: the share of one transfer op's time that is prefetched, i.e. may hide behind the stage's compute
 * (docs/simulation-model.md §12 "Expert prefetch"), from the prefetch accuracy p (the predictor's measured recall:
 * the share of the needed experts it picks one layer ahead):
 * - p = 0: q = 0 for every group (no prefetching at all: the former `offloadOverlap 'none'`);
 * - routed experts, p > 0: q = max(p, M/(E_loc − C)). The engine takes the better of two strategies: prefetch its
 *   predictions, of which a share p are needed misses; or, when nearly every non-resident expert is needed (prefill,
 *   large batches), copy the non-resident experts unconditionally, of which a share M/(E_loc − C) are needed. So
 *   q = p at batch 1 (M ≪ E_loc − C) and q = 1 in prefill (M = E_loc − C);
 * - n-gram table rows, p > 0: q = 1. Their bytes are known before the step (rows come from token ids), and an engine
 *   that prefetches at all prefetches these deterministic transfers;
 * - token-embedding rows, p > 0: q = 1 in prefill (every prompt id is known before the step); q = 0 in decode (and
 *   in the drafter's decode passes): the row's id is the token the previous step just sampled, and the gather is
 *   the step's first op, so no compute is left to hide it behind.
 * - KV cache, p > 0: q = 1 for full attention and for prefill (every non-resident past token is needed, and which
 *   ones is known before the step); top-k decode: q = max(p, M/(N_past − N_res)), the expert rule with past tokens
 *   for experts: predict the selection one layer ahead (InfiniGen, ShadowKV), or copy every non-resident token;
 * So p = 1 gives q = 1 everywhere except decode token-embedding rows (the former `'ideal'`). Dimensionless, in [0, 1].
 */
export function prefetchedFraction(p: number, target: PrefetchTarget): number {
  // Formula: q = 0   (p = 0, every group)
  if (!(p > 0)) return 0;
  if (target.group === 'token_embedding') {
    // Formula: q = prefill ? 1 : 0   (token embedding, p > 0)
    // See docs/simulation-model.md §12 ("Token embedding off-device")
    return target.phase === 'prefill' ? 1 : 0;
  }
  if (target.group === 'lookup_table') {
    // Formula: q = 1   (n-gram table rows; p > 0)
    return 1;
  }
  if (target.group === 'kv_cache') {
    // Formula: q = 1   (KV cache, full attention or prefill; p > 0)
    // Formula: q = max(p, M/(N_past − N_res))   (KV cache, top-k decode; p > 0)
    // See docs/simulation-model.md §12 ("KV cache off-device")
    if (!target.topK || !(target.nonResidentTokens > 0)) return 1;
    return Math.min(1, Math.max(p, target.misses / target.nonResidentTokens));
  }
  const { misses: M, nonResidentExperts } = target;
  if (!(nonResidentExperts > 0)) return 1;
  // Formula: q = max(p, M/(E_loc − C))   (routed experts; p > 0: predicted, or unconditional copy)
  // See docs/simulation-model.md §12 ("Expert prefetch")
  const neededShare = M / nonResidentExperts;
  return Math.min(1, Math.max(p, neededShare));
}
