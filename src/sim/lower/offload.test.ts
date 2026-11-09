import { describe, expect, it } from 'vitest';
import { DEFAULT_PREFETCH_ACCURACY, uniqueExperts } from '../engine/transfer';
import type { ExpertPlacement, LoweringContext, OffloadLowering, TablePlacement } from './lower';
import { draftExpertPlacement, expertPlacementAt, prefetchedFraction, tablePlacementAt } from './offload';

/** A decode lowering context with the given resolved placement. */
function ctxWith(offload: OffloadLowering): LoweringContext {
  return {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 8193,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    offload,
  };
}

/** Mixed placement over 8 layers: experts of 0–3 in host RAM, 4–5 on the SSD, 6–7 on the device; tables of 1 in RAM, 5 on the SSD. */
describe('per-layer rule lookup (mixed placement by layer range)', () => {
  const ram: ExpertPlacement = { tier: 'host_ram', layers: { from: 0, to: 4 }, residentFraction: 0.25, hitRatio: 0.25, includeDraft: false };
  const ssd: ExpertPlacement = { tier: 'nvme', layers: { from: 4, to: 6 }, residentFraction: 0.5, hitRatio: 0.5, includeDraft: true };
  const tableRam: TablePlacement = { tier: 'host_ram', layers: { from: 1, to: 2 }, scaleLayout: 'separate' };
  const tableSsd: TablePlacement = { tier: 'nvme', layers: { from: 5, to: 6 }, scaleLayout: 'pinned_host' };
  const ctx = ctxWith({
    prefetchAccuracy: 0.8,
    routedExperts: [ram, ssd],
    lookupTables: [tableRam, tableSsd],
  });

  it('each layer takes the rule whose range contains it, with that rule\'s f and h', () => {
    for (const i of [0, 1, 2, 3]) expect(expertPlacementAt(ctx, i)).toBe(ram);
    for (const i of [4, 5]) expect(expertPlacementAt(ctx, i)).toBe(ssd);
  });

  it('a layer outside every rule, or an unknown (draft) layer, keeps its experts on the device', () => {
    expect(expertPlacementAt(ctx, 6)).toBeUndefined();
    expect(expertPlacementAt(ctx, 7)).toBeUndefined();
    expect(expertPlacementAt(ctx, undefined)).toBeUndefined();
  });

  it('lookup tables: each layer takes its own table rule; uncovered layers keep their tables on the device', () => {
    expect(tablePlacementAt(ctx, 1)).toBe(tableRam);
    expect(tablePlacementAt(ctx, 5)).toBe(tableSsd);
    for (const i of [0, 2, 4, 6, 7]) expect(tablePlacementAt(ctx, i)).toBeUndefined();
    expect(tablePlacementAt(ctx, undefined)).toBeUndefined();
  });

  it('the drafter\'s experts follow the one rule with includeDraft', () => {
    expect(draftExpertPlacement(ctx.offload)).toBe(ssd);
    expect(draftExpertPlacement({ prefetchAccuracy: 0, routedExperts: [ram] })).toBeUndefined();
    expect(draftExpertPlacement(undefined)).toBeUndefined();
  });
});

/**
 * q = prefetchedFraction(p, target) (docs/simulation-model.md §12 "Expert prefetch"): 0 at p = 0 for every group;
 * experts q = max(p, M/(E_loc − C)) (predicted copies, or an unconditional copy when nearly every non-resident
 * expert is needed); table rows 1 for any p > 0.
 */
describe('prefetchedFraction', () => {
  // Batch-1 decode of a 128-expert, top-8 layer with nothing resident: M = 8, E_loc − C = 128.
  const decode = { group: 'routed_experts', misses: 8, nonResidentExperts: 128 } as const;
  // Prefill: every non-resident expert is needed, M = E_loc − C.
  const prefill = { group: 'routed_experts', misses: 89.6, nonResidentExperts: 89.6 } as const;

  it('p = 0: no prefetching at all, q = 0 for every group', () => {
    expect(prefetchedFraction(0, decode)).toBe(0);
    expect(prefetchedFraction(0, prefill)).toBe(0);
    expect(prefetchedFraction(0, { group: 'lookup_table' })).toBe(0);
  });

  it('p = 1: perfect prediction, q = 1 for every group (exactly)', () => {
    expect(prefetchedFraction(1, decode)).toBe(1);
    expect(prefetchedFraction(1, prefill)).toBe(1);
    expect(prefetchedFraction(1, { group: 'lookup_table' })).toBe(1);
  });

  it('batch 1 keeps the measured recall: q = p exactly (no chance hits added)', () => {
    expect(DEFAULT_PREFETCH_ACCURACY).toBe(0.8);
    expect(prefetchedFraction(0.8, decode)).toBe(0.8);
    // Mixtral (8 experts, top-2, nothing resident): M/(E − C) = 0.25 < p = 0.84 ⇒ q = 0.84.
    expect(
      prefetchedFraction(0.84, { group: 'routed_experts', misses: 2, nonResidentExperts: 8 }),
    ).toBe(0.84);
    expect(prefetchedFraction(0.8, { group: 'lookup_table' })).toBe(1);
  });

  it('prefill (M = E_loc − C) gives q = 1', () => {
    expect(prefetchedFraction(0.8, prefill)).toBe(1);
    expect(prefetchedFraction(0.3, prefill)).toBe(1);
  });

  it('intermediate batch: 32 tokens on 128 experts, top-8 ⇒ U ≈ 111.8, q = U/128 ≈ 0.873 > p', () => {
    const U = uniqueExperts(128, 8, 32);
    expect(U / 128).toBeCloseTo(0.873, 3);
    const q = prefetchedFraction(0.8, {
      group: 'routed_experts',
      misses: U,
      nonResidentExperts: 128,
    });
    expect(q).toBe(U / 128);
    // A smaller batch (4 tokens: U ≈ 29.0, 0.227) stays at the recall.
    const small = uniqueExperts(128, 8, 4);
    expect(
      prefetchedFraction(0.8, { group: 'routed_experts', misses: small, nonResidentExperts: 128 }),
    ).toBe(0.8);
  });
});
