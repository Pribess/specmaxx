/**
 * Build-only provenance marker. Source presets never set this; vite/stripProvenance.ts does.
 * See docs/deploy.md ("Preset provenance sidecar"). No physical field or unit is changed.
 */
export interface StrippedProvenance {
  /** True only for bundled presets whose citations were moved to the on-demand sidecar. */
  provenanceStripped?: true;
}

/** Whether citation checks must wait for the full preset from the sidecar; physical checks still run. */
export function hasStrippedProvenance(value: unknown): boolean {
  return (
    typeof value === 'object' &&
    value !== null &&
    'provenanceStripped' in value &&
    value.provenanceStripped === true
  );
}

/**
 * Pure deep copy with citation payloads emptied. Every non-citation value, including every physical
 * number, stays identical. Empty arrays retain the spec schema; the marker prevents mistaking them for
 * unsourced custom specs. `note` stays: modifier descriptions display it, not just citations.
 * The build transform implements the same operation on preset expressions, including factory results.
 */
export function strip<T>(full: T): T {
  if (Array.isArray(full)) return full.map((value) => strip(value)) as T;
  if (full === null || typeof full !== 'object') return full;
  // Standalone helper citations (e.g. an SFU rate's source) carry a URL and title, no physical data.
  if ('url' in full && 'title' in full) return {} as T;
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(full)) {
    if (key === 'sources' || key === 'estimated') {
      result[key] = [];
      result.provenanceStripped = true;
    } else if (key === 'estimatedReason') {
      result[key] = undefined;
      result.provenanceStripped = true;
    } else if (key === 'reason') result[key] = '';
    else result[key] = strip(value);
  }
  return result as T;
}
