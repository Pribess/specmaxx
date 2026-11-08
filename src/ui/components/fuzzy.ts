/** One choice in a {@link Picker} or {@link Combobox}. */
export interface PickerOption {
  /** The stored value (e.g. an HF model ID or `device:<id>`). Unique within one list. */
  value: string;
  /** The text shown in the list and matched by the search box. */
  label: string;
  /** Listbox group header. Options with the same group render together, groups in first-appearance order. */
  group?: string;
  /** Optional secondary text shown in the list (not matched by search). */
  description?: string;
  /** Extra search terms (e.g. aliases) matched in addition to label and value. */
  keywords?: string[];
}

/** Penalty added to a scattered (non-contiguous) subsequence match, so contiguous matches rank first. */
const SCATTER_BASE = 100;
/** Extra cost for a match on the value rather than the label, so label matches win ties. */
const VALUE_PENALTY = 0.5;
/** Extra cost for a match on a keyword, so label and value matches win ties. */
const KEYWORD_PENALTY = 1;

/**
 * Cost of matching one query token against one text, lower is better, or null when it does not match.
 * - A contiguous match costs its start index (0 = prefix match, so "5090" ranks "RTX 5090" high).
 * - With `allowScatter`, a scattered subsequence ("r5" in "RTX 5090") costs {@link SCATTER_BASE} plus the gap
 *   between its first and last characters, so it ranks below every contiguous match. Only labels allow it:
 *   a value such as `system:...` would otherwise match almost any token through its prefix letters.
 */
function tokenCost(text: string, token: string, allowScatter: boolean): number | null {
  const t = text.toLowerCase();
  const at = t.indexOf(token);
  if (at >= 0) return at;
  if (!allowScatter) return null;
  let pos = -1;
  let first = -1;
  for (const ch of token) {
    pos = t.indexOf(ch, pos + 1);
    if (pos < 0) return null;
    if (first < 0) first = pos;
  }
  return SCATTER_BASE + (pos - first);
}

/** Cost of one token against every searchable field of an option (the best field wins). */
function optionTokenCost(o: PickerOption, token: string): number | null {
  const costs: (number | null)[] = [tokenCost(o.label, token, true)];
  const valueCost = tokenCost(o.value, token, false);
  costs.push(valueCost === null ? null : valueCost + VALUE_PENALTY);
  for (const k of o.keywords ?? []) {
    const c = tokenCost(k, token, false);
    costs.push(c === null ? null : c + KEYWORD_PENALTY);
  }
  const found = costs.filter((c): c is number => c !== null);
  return found.length === 0 ? null : Math.min(...found);
}

/**
 * Options matching `query` (case-insensitive subsequence over label, value and keywords), best first;
 * an empty query returns every option in its original order.
 *
 * The query is split on whitespace and every token must match somewhere in an option. An option's cost is
 * the sum of its token costs (see {@link tokenCost}); the sort is stable, so ties keep the list order.
 */
export function fuzzyFilter(options: PickerOption[], query: string): PickerOption[] {
  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 0);
  if (tokens.length === 0) return options;
  const scored: { option: PickerOption; cost: number }[] = [];
  for (const option of options) {
    let cost = 0;
    for (const token of tokens) {
      const c = optionTokenCost(option, token);
      if (c === null) {
        cost = -1;
        break;
      }
      cost += c;
    }
    if (cost >= 0) scored.push({ option, cost });
  }
  scored.sort((a, b) => a.cost - b.cost);
  return scored.map((s) => s.option);
}

/** Options that share a listbox group header. `group` is undefined for options without one. */
export interface OptionGroup {
  group?: string;
  options: PickerOption[];
}

/**
 * Buckets options by `group`. Groups come in the order of their first option, and options keep their order
 * within a group, so a ranked list stays ranked and each group sits where its best match is.
 */
export function groupOptions(options: PickerOption[]): OptionGroup[] {
  const byGroup = new Map<string | undefined, OptionGroup>();
  for (const option of options) {
    let bucket = byGroup.get(option.group);
    if (!bucket) {
      bucket = { group: option.group, options: [] };
      byGroup.set(option.group, bucket);
    }
    bucket.options.push(option);
  }
  return [...byGroup.values()];
}

/** A keyboard movement inside a listbox. */
export type NavStep = 'next' | 'prev' | 'first' | 'last' | 'pageDown' | 'pageUp';

/** Rows moved by PageUp / PageDown. */
const PAGE_ROWS = 10;

/**
 * The index a keyboard step lands on, clamped to the list (no wrap-around). `current` is -1 when no option
 * is active; from there `next` and `pageDown` start at the first option and `prev` and `pageUp` at the last.
 * Returns -1 for an empty list.
 */
export function stepIndex(current: number, count: number, step: NavStep): number {
  if (count <= 0) return -1;
  const last = count - 1;
  const clamp = (i: number) => Math.min(Math.max(i, 0), last);
  switch (step) {
    case 'first':
      return 0;
    case 'last':
      return last;
    case 'next':
      return current < 0 ? 0 : clamp(current + 1);
    case 'prev':
      return current < 0 ? last : clamp(current - 1);
    case 'pageDown':
      return clamp((current < 0 ? 0 : current) + PAGE_ROWS);
    case 'pageUp':
      return clamp((current < 0 ? last : current) - PAGE_ROWS);
  }
}
