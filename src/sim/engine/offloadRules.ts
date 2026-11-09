/**
 * Normalisation of the offload placement rules (docs/simulation-model.md §12 "Placement").
 *
 * `OffloadConfig.routedExperts` and `.lookupTables` take either one rule (the original single-tier form) or a list of
 * rules over disjoint layer ranges. These helpers give every reader the list form, so a single object and a one-rule
 * list are the same input by construction. Pure; no validation (see `validate.ts`).
 */
import type { LookupTableRule, OffloadConfig, RoutedExpertRule } from './types';

/**
 * The config's routed-expert rules as a list: absent → `[]`, one object → `[it]`, a list → itself, in the order given
 * (not sorted; the lowering sorts by `layers.from`).
 */
export function expertRules(config: OffloadConfig | undefined): readonly RoutedExpertRule[] {
  return asRuleList(config?.routedExperts);
}

/**
 * The config's lookup-table rules as a list: absent → `[]`, one object → `[it]`, a list → itself, in the order given
 * (not sorted). Rules with tier `'device'` are kept (they place nothing; the resolver skips them).
 */
export function lookupTableRules(config: OffloadConfig | undefined): readonly LookupTableRule[] {
  return asRuleList(config?.lookupTables);
}

/** `undefined` → `[]`, a list → itself, one rule → a one-rule list. */
function asRuleList<R extends object>(value: R | readonly R[] | undefined): readonly R[] {
  if (value === undefined) return [];
  // Array.isArray does not narrow readonly arrays; the cast restores the element type.
  return Array.isArray(value) ? (value as readonly R[]) : [value as R];
}
