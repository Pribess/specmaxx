// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import modelingMethod from '../../docs/modeling-method.md?raw';
import simulationModel from '../../docs/simulation-model.md?raw';
import aboutHtml from '../../about.html?raw';
import {
  hashPreloadScript,
  injectPreloadScript,
  preloadTargets,
  prerenderDocs,
} from '../../vite/prerenderDocs';

const sanitySource = import.meta.glob<string>('../../docs/sanity-check.md', {
  query: '?raw',
  import: 'default',
  eager: true,
})['../../docs/sanity-check.md'];
const docs = prerenderDocs({
  'docs/modeling-method.md': modelingMethod,
  'docs/simulation-model.md': simulationModel,
  'docs/sanity-check.md': sanitySource,
});
const targets = preloadTargets(docs, (slug) => `./assets/${slug}-AbCd1234.htm`);
const script = hashPreloadScript(targets);

/** Runs the head script with `hash` as the URL fragment; returns the preload links it added. */
function runWithHash(hash: string) {
  window.history.replaceState(null, '', `/about.html${hash}`);
  new Function(script)();
  return Array.from(document.head.querySelectorAll<HTMLLinkElement>('link[rel="preload"]'));
}

afterEach(() => {
  document.head.innerHTML = '';
  window.history.replaceState(null, '', '/');
});

describe('about.html preload script', () => {
  it('preloads the document a hash points into, as the runtime fetch requests it', () => {
    const [link, ...rest] = runWithHash('#4-hardware');
    expect(rest).toEqual([]);
    expect(link.getAttribute('href')).toBe('./assets/simulation-model-AbCd1234.htm');
    expect(link.getAttribute('as')).toBe('fetch');
    // fetch(url) defaults: mode cors, credentials same-origin. Only crossorigin="anonymous" matches both.
    expect(link.getAttribute('crossorigin')).toBe('anonymous');
  });

  it('maps every lazy id and slug to its own document', () => {
    for (const doc of docs.lazyDocs)
      for (const id of [doc.slug, doc.ids[0], doc.ids.at(-1)!]) {
        document.head.innerHTML = '';
        const links = runWithHash(`#${encodeURIComponent(id)}`);
        expect(
          links.map((l) => l.getAttribute('href')),
          id,
        ).toEqual([`./assets/${doc.slug}-AbCd1234.htm`]);
      }
    if (sanitySource) expect(runWithHash('#flagged-cases')[0].href).toContain('sanity-check-');
  });

  it('preloads nothing for method ids, page ids, unknown or malformed hashes, or no hash', () => {
    for (const hash of [
      '',
      '#',
      '#what-the-numbers-mean',
      '#full-reference',
      '#no-such-section',
      '#%E0%A4%A',
      '#memory fit',
      '#4-hard',
    ])
      expect(runWithHash(hash), hash).toEqual([]);
  });

  it('leaves out ids the eager page also has, and checks ids for whitespace', () => {
    const shared = prerenderDocs({
      'docs/modeling-method.md': '## Same\n## Only method',
      'docs/simulation-model.md': '# Model\n## Same',
    });
    expect(preloadTargets(shared, (slug) => slug).map((t) => t.ids)).toEqual([
      ['simulation-model', 'model'],
    ]);
    expect(() => hashPreloadScript([{ url: 'u', ids: ['a b'] }])).toThrow('Whitespace');
    expect(hashPreloadScript([{ url: '</script>', ids: ['x'] }])).not.toContain('</script>');
  });

  it('goes right after <meta charset>, before the stylesheets and module scripts', () => {
    const html = injectPreloadScript(aboutHtml, 'PRELOAD()');
    const at = html.indexOf('<script>PRELOAD()</script>');
    expect(at).toBeGreaterThan(html.indexOf('<meta charset="UTF-8" />'));
    expect(at).toBeLessThan(html.indexOf('<script type="module"'));
    expect(at).toBeLessThan(1024);
    expect(() => injectPreloadScript('<head></head>', 'x')).toThrow('<meta charset>');
  });
});
