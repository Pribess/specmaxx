import { useSyncExternalStore } from 'react';

/**
 * Below this width the workspace stacks (phone layout): one column, document scroll, no splitter. Kept in step
 * with the `max-width: 899px` media queries in tokens.css and app.css.
 */
export const STACKED_QUERY = '(max-width: 899px)';

function matches(query: string): boolean {
  return typeof matchMedia === 'function' && matchMedia(query).matches;
}

/** Whether a media query matches now, re-rendering when it changes. False where matchMedia is missing. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      if (typeof matchMedia !== 'function') return () => {};
      const list = matchMedia(query);
      list.addEventListener?.('change', onChange);
      return () => list.removeEventListener?.('change', onChange);
    },
    () => matches(query),
    () => false,
  );
}

/** Whether the stacked (phone) layout applies. */
export function useStacked(): boolean {
  return useMediaQuery(STACKED_QUERY);
}

/** One-off check of the stacked layout, for initial state (e.g. which panels start collapsed). */
export function isStacked(): boolean {
  return matches(STACKED_QUERY);
}
