import { describe, expect, it } from 'vitest';

/**
 * Guard for the scrollbar rule: no element inside the page scrolls with a native OS scrollbar. Every scroll
 * container is a ScrollArea (src/ui/components/ScrollArea.tsx), whose Radix viewport sets its own overflow
 * inline and hides the native bar. So no stylesheet or inline style in src may declare `overflow: auto` or
 * `overflow: scroll` (or the -x / -y forms).
 *
 * The one documented exception is the page-level (document) scrollbar: rules whose selector is only
 * `html` or `body` may set overflow; that scrollbar is themed in src/styles/tokens.css with
 * scrollbar-color / scrollbar-width and ::-webkit-scrollbar.
 */
const styles = import.meta.glob('/src/**/*.css', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;
const sources = import.meta.glob(['/src/**/*.tsx', '/src/**/*.ts', '!/src/**/*.test.*'], {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

/** A declaration that gives an element its own (native) scrollbar. */
const NATIVE_SCROLL = /\boverflow(?:-x|-y|-block|-inline)?\s*:\s*(?:auto|scroll)\b/;
/** The same in a React inline style object (`overflowY: 'auto'`). */
const NATIVE_SCROLL_JS = /\boverflow(?:X|Y|Block|Inline)?\s*:\s*['"](?:auto|scroll)['"]/;
/** Selectors allowed to scroll natively: the document only. */
const PAGE_LEVEL = /^(?:html|body|:root)$/;

/** Innermost `selector { declarations }` blocks of a stylesheet, comments removed (media queries unwrap). */
function cssRules(css: string): { selector: string; body: string }[] {
  const text = css.replace(/\/\*[\s\S]*?\*\//g, '');
  return [...text.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({
    selector: m[1].trim(),
    body: m[2],
  }));
}

describe('no native scrollbars inside the page', () => {
  it('finds the stylesheets and sources it guards', () => {
    expect(Object.keys(styles)).toContain('/src/ui/app.css');
    expect(Object.keys(styles)).toContain('/src/styles/tokens.css');
    expect(Object.keys(sources).length).toBeGreaterThan(10);
  });

  it('declares overflow auto/scroll in no stylesheet rule except the page-level one', () => {
    const offenders: string[] = [];
    for (const [path, css] of Object.entries(styles)) {
      for (const { selector, body } of cssRules(css)) {
        if (!NATIVE_SCROLL.test(body)) continue;
        const selectors = selector.split(',').map((s) => s.trim());
        if (selectors.every((s) => PAGE_LEVEL.test(s))) continue;
        offenders.push(`${path}: ${selector}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('sets no overflow auto/scroll in inline styles or generated HTML', () => {
    for (const [path, raw] of Object.entries(sources)) {
      const source = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
      expect(source, path).not.toMatch(NATIVE_SCROLL_JS);
      expect(source, path).not.toMatch(NATIVE_SCROLL);
    }
  });

  it('catches a native scroller and allows the page-level one (self-check)', () => {
    const rules = cssRules('@media (x) { .list { overflow-y: auto } } html { overflow-y: scroll }');
    expect(rules.map((r) => [r.selector, NATIVE_SCROLL.test(r.body)])).toEqual([
      ['.list', true],
      ['html', true],
    ]);
    expect(PAGE_LEVEL.test('html')).toBe(true);
    expect(NATIVE_SCROLL_JS.test("style={{ overflowY: 'scroll' }}")).toBe(true);
  });
});
