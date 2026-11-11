// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import modelingMethod from '../../docs/modeling-method.md?raw';
import simulationModel from '../../docs/simulation-model.md?raw';
import { referenceDocs } from 'virtual:about-docs';
import { prerenderDocs } from '../../vite/prerenderDocs';
import { AboutPage, estimateHref } from './AboutPage';
import { githubSlug, renderMarkdown } from './markdown';
import { splitTables } from './segments';

const sanitySource = import.meta.glob<string>('../../docs/sanity-check.md', {
  query: '?raw',
  import: 'default',
  eager: true,
})['../../docs/sanity-check.md'];
/** What the build emits for each reference document, served by the mocked fetch below. */
const prerendered = prerenderDocs({
  'docs/modeling-method.md': modelingMethod,
  'docs/simulation-model.md': simulationModel,
  'docs/sanity-check.md': sanitySource,
});

/** Serves the prerendered reference documents at their virtual-module URLs. */
function mockDocFetch(fail = false) {
  const fetchMock = vi.fn(async (url: string | URL) => {
    const doc = referenceDocs.find((d) => d.url === String(url));
    const html = prerendered.lazyDocs.find((d) => d.slug === doc?.slug)?.html;
    if (fail || html === undefined) return { ok: false, status: 404, text: async () => '' };
    return { ok: true, status: 200, text: async () => html };
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** The scroll calls of `scrollIntoView`, recorded with the element each was called on. */
function mockScroll() {
  const scroll = vi.fn();
  Element.prototype.scrollIntoView = scroll;
  return scroll;
}

/** One held document request: its URL, and `release` to answer it with the prerendered document. */
interface HeldRequest {
  url: string;
  release: () => void;
}

/** Holds every document fetch until the test releases it; the returned list fills as requests arrive. */
function holdDocFetch(): HeldRequest[] {
  const served = mockDocFetch();
  const pending: HeldRequest[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(
      (url: string | URL) =>
        new Promise((resolve) => {
          pending.push({ url: String(url), release: () => resolve(served(url)) });
        }),
    ),
  );
  return pending;
}

/** Follows an in-page link: sets the hash and fires `hashchange`, as the browser does. */
function navigate(hash: string) {
  window.history.replaceState(null, '', `/about.html${hash}`);
  window.dispatchEvent(new HashChangeEvent('hashchange'));
}

/** Waits past the next few animation frames (and the microtasks before them). */
function nextFrames() {
  return new Promise((resolve) => setTimeout(resolve, 100));
}

/** Opens the reference `<details>` as a click on its summary would. */
function openReference(container: HTMLElement) {
  const details = container.querySelector('details#full-reference') as HTMLDetailsElement;
  details.open = true;
  fireEvent(details, new Event('toggle'));
  return details;
}

beforeEach(() => {
  mockDocFetch();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  sessionStorage.clear();
  window.history.replaceState(null, '', '/');
});

describe('repository Markdown', () => {
  it('resolves every simulation-model fragment to an element id', () => {
    const container = document.createElement('div');
    container.innerHTML = renderMarkdown(simulationModel);
    const ids = new Set(Array.from(container.querySelectorAll('[id]'), (node) => node.id));
    const links = container.querySelectorAll<HTMLAnchorElement>('a[href^="#"]');
    expect(links.length).toBeGreaterThan(10);
    for (const link of links) {
      expect(ids.has(decodeURIComponent(link.hash.slice(1))), link.hash).toBe(true);
    }
  });

  it('keeps underscores and deduplicates heading ids, including suffix collisions', () => {
    expect(githubSlug('10. Worked example: REF_DENSE_8B on one RTX 5090')).toBe(
      '10-worked-example-ref_dense_8b-on-one-rtx-5090',
    );
    const html = renderMarkdown('# Same\n# Same\n# Same-1\n# Same');
    const container = document.createElement('div');
    container.innerHTML = html;
    expect(Array.from(container.querySelectorAll('h1'), (node) => node.id)).toEqual([
      'same',
      'same-1',
      'same-1-1',
      'same-2',
    ]);
    expect(renderMarkdown('# Same')).toContain('id="same"');
  });
  it('rewrites rendered document links and leaves unavailable repository references as text', () => {
    const container = document.createElement('div');
    container.innerHTML = renderMarkdown(
      '[Model](simulation-model.md) [Checks](sanity-check.md) ' +
        '[Section](simulation-model.md#per-op-time) [Source](../src/sim/) ' +
        '[Other](assets/example.json) [Web](https://example.com)',
      ['simulation-model.md', 'sanity-check.md'],
    );
    expect(
      Array.from(container.querySelectorAll('a'), (node) => node.getAttribute('href')),
    ).toEqual(['#simulation-model', '#sanity-check', '#per-op-time', 'https://example.com']);
    expect(container.textContent).toContain('Source Other');
    expect(renderMarkdown('[Checks](sanity-check.md)')).toBe('<p>Checks</p>\n');
  });

  it('splits rendered GFM tables out of the prose', () => {
    const html = renderMarkdown(
      'Intro\n\n| Limit | Value |\n| --- | --- |\n| Memory | 32 GB |\n\nOutro',
    );
    const segments = splitTables(html);
    expect(segments.map((s) => s.kind)).toEqual(['html', 'table', 'html']);
    const container = document.createElement('div');
    container.innerHTML = segments[1].html;
    expect(container.querySelector('table tbody td')?.textContent).toBe('Memory');
    expect(segments.map((s) => s.html).join('')).toBe(html);
  });
});

describe('About page', () => {
  it('renders all documents without repository paths and resolves all fragment links', async () => {
    const { container } = render(<AboutPage />);
    openReference(container);
    const sanity = await screen.findByRole('article', { name: 'Sanity checks' });
    await waitFor(() => expect(sanity.querySelector('h1')).toBeTruthy());
    await waitFor(() =>
      expect(
        container.querySelector('#simulation-model h1#specmaxx-simulation-model'),
      ).toBeTruthy(),
    );
    const ids = new Set(Array.from(container.querySelectorAll('[id]'), (node) => node.id));
    for (const link of container.querySelectorAll<HTMLAnchorElement>('a[href]')) {
      const href = link.getAttribute('href')!;
      // External benchmark sources can legitimately end in .md (e.g. GitHub files).
      if (!/^https?:\/\//i.test(href)) expect(href, href).not.toMatch(/\.md(?:$|#)/);
      expect(href.startsWith('../'), href).toBe(false);
      if (href.startsWith('#')) {
        expect(ids.has(decodeURIComponent(href.slice(1))), href).toBe(true);
      }
    }
  });

  it('renders the method sections with a table of contents, and the reference collapsed and unfetched', () => {
    const fetchMock = mockDocFetch();
    const { container } = render(<AboutPage />);
    const method = screen.getByRole('article', { name: 'Modeling method' });
    expect(method.querySelectorAll('h2').length).toBeGreaterThanOrEqual(8);
    const toc = screen.getByRole('navigation', { name: 'Modeling method contents' });
    expect(Array.from(toc.querySelectorAll('a'), (a) => a.getAttribute('href'))).toEqual(
      Array.from(method.querySelectorAll('h2'), (h) => `#${h.id}`),
    );
    expect(Array.from(toc.querySelectorAll('a'), (a) => a.textContent)).toEqual(
      Array.from(method.querySelectorAll('h2'), (h) => h.textContent),
    );
    const details = container.querySelector('details#full-reference') as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(details.contains(method)).toBe(false);
    expect(details.querySelector('article#simulation-model')).toBeTruthy();
    expect(details.querySelector('article#simulation-model h1')).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('fetches the reference documents when the reference opens, once each', async () => {
    const fetchMock = mockDocFetch();
    const { container } = render(<AboutPage />);
    const details = openReference(container);
    expect(await screen.findAllByRole('status')).not.toHaveLength(0);
    await waitFor(() =>
      expect(details.querySelectorAll('article#simulation-model table').length).toBeGreaterThan(10),
    );
    expect(fetchMock.mock.calls.map(([url]) => String(url)).sort()).toEqual(
      referenceDocs.map((doc) => doc.url).sort(),
    );
    expect(screen.queryByRole('status')).toBeNull();
    details.open = false;
    fireEvent(details, new Event('toggle'));
    openReference(container);
    expect(fetchMock).toHaveBeenCalledTimes(referenceDocs.length);
  });

  it('opens the reference, loads it and scrolls to a heading named by the hash', async () => {
    const scroll = mockScroll();
    const fetchMock = mockDocFetch();
    window.history.replaceState(null, '', '/about.html#memory-fit');
    const { container } = render(<AboutPage />);
    const details = container.querySelector('details#full-reference') as HTMLDetailsElement;
    await waitFor(() => expect(details.open).toBe(true));
    expect(fetchMock).toHaveBeenCalled();
    await waitFor(() => expect(scroll).toHaveBeenCalled());
    expect(scroll.mock.contexts[0]).toBe(document.getElementById('memory-fit'));
    expect(details.contains(document.getElementById('memory-fit'))).toBe(true);
  });

  it('follows a hash change into an unloaded reference document', async () => {
    const scroll = mockScroll();
    render(<AboutPage />);
    window.history.replaceState(null, '', '/about.html#4-hardware');
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    await waitFor(() => expect(scroll).toHaveBeenCalled());
    expect(scroll.mock.contexts.at(-1)).toBe(document.getElementById('4-hardware'));
  });

  it('fetches the document a hash points into first, and the others once it has rendered', async () => {
    const pending = holdDocFetch();
    const fetchMock = vi.mocked(fetch);
    const scroll = mockScroll();
    window.history.replaceState(null, '', '/about.html#memory-fit');
    render(<AboutPage />);
    const simulation = referenceDocs.find((doc) => doc.slug === 'simulation-model')!;
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(pending.map((p) => p.url)).toEqual([simulation.url]);
    pending[0].release();
    await waitFor(() => expect(document.getElementById('memory-fit')).toBeTruthy());
    await waitFor(() => expect(scroll).toHaveBeenCalled());
    await waitFor(() => expect(pending).toHaveLength(referenceDocs.length));
    expect(pending.slice(1).map((p) => p.url)).toEqual(
      referenceDocs.filter((doc) => doc !== simulation).map((doc) => doc.url),
    );
    for (const p of pending.slice(1)) p.release();
    await waitFor(() => expect(screen.queryByRole('status')).toBeNull());
  });

  it('drops a pending hash target when the reader follows another hash before the document arrives', async () => {
    const pending = holdDocFetch();
    const scroll = mockScroll();
    window.history.replaceState(null, '', '/about.html#memory-fit');
    render(<AboutPage />);
    await waitFor(() => expect(pending).toHaveLength(1));
    navigate('#what-the-numbers-mean');
    await waitFor(() => expect(scroll).toHaveBeenCalledTimes(1));
    const method = document.getElementById('what-the-numbers-mean');
    expect(scroll.mock.contexts[0]).toBe(method);
    pending[0].release();
    await waitFor(() => expect(document.getElementById('memory-fit')).toBeTruthy());
    await waitFor(() => expect(pending).toHaveLength(referenceDocs.length));
    await nextFrames();
    expect(scroll.mock.contexts).toEqual([method]);
  });

  it('skips the after-fonts re-scroll once the reader is on another hash', async () => {
    let fontsReady = () => {};
    const ready = new Promise<void>((resolve) => (fontsReady = resolve));
    Object.defineProperty(document, 'fonts', {
      configurable: true,
      value: { status: 'loading', ready },
    });
    try {
      const scroll = mockScroll();
      window.history.replaceState(null, '', '/about.html#memory-fit');
      render(<AboutPage />);
      await waitFor(() => expect(scroll).toHaveBeenCalledTimes(1));
      const reference = document.getElementById('memory-fit');
      expect(scroll.mock.contexts[0]).toBe(reference);
      navigate('#what-the-numbers-mean');
      await waitFor(() => expect(scroll).toHaveBeenCalledTimes(2));
      fontsReady();
      await nextFrames();
      const method = document.getElementById('what-the-numbers-mean');
      // The method heading's own re-scroll runs; the reference heading's does not.
      expect(scroll.mock.contexts).toEqual([reference, method, method]);
    } finally {
      delete (document as { fonts?: unknown }).fonts;
    }
  });

  it('ignores a hash that names no element and no reference id', () => {
    const fetchMock = mockDocFetch();
    window.history.replaceState(null, '', '/about.html#no-such-section');
    const { container } = render(<AboutPage />);
    expect((container.querySelector('details#full-reference') as HTMLDetailsElement).open).toBe(
      false,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('shows an error with a retry when a document fails to load', async () => {
    mockDocFetch(true);
    const { container } = render(<AboutPage />);
    openReference(container);
    const alerts = await screen.findAllByRole('alert');
    expect(alerts).toHaveLength(referenceDocs.length);
    const fetchMock = mockDocFetch();
    fireEvent.click(screen.getAllByRole('button', { name: 'Retry' })[0]);
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
    expect(fetchMock).toHaveBeenCalledTimes(referenceDocs.length);
    await waitFor(() => expect(container.querySelector('[id="4-hardware"]')).toBeTruthy());
  });

  it('keeps method headings distinct from reference ids and method links resolvable', () => {
    const container = document.createElement('div');
    container.innerHTML = renderMarkdown(modelingMethod);
    const reference = document.createElement('div');
    reference.innerHTML = renderMarkdown(simulationModel);
    const referenceIds = new Set(Array.from(reference.querySelectorAll('[id]'), (n) => n.id));
    for (const heading of container.querySelectorAll('h2'))
      expect(referenceIds.has(heading.id)).toBe(false);
    const links = container.querySelectorAll<HTMLAnchorElement>('a[href^="#"]');
    expect(links.length).toBeGreaterThan(10);
    for (const link of links) expect(referenceIds.has(link.hash.slice(1)), link.hash).toBe(true);
  });

  it('renders the Memory fit heading and document navigation', async () => {
    const { container } = render(<AboutPage />);
    expect(screen.getByRole('navigation', { name: 'Documents' })).toBeTruthy();
    expect(screen.getAllByRole('link', { name: 'Back to the estimate' })).toHaveLength(2);
    openReference(container);
    expect(await screen.findByRole('heading', { name: 'Memory fit' })).toBeTruthy();
  });

  it('puts every table, eager and loaded, in a keyboard-focusable themed scroll region', async () => {
    const { container } = render(<AboutPage />);
    openReference(container);
    await waitFor(() =>
      expect(container.querySelectorAll('article#sanity-check table').length).toBeGreaterThan(0),
    );
    await waitFor(() =>
      expect(container.querySelectorAll('article#simulation-model table').length).toBeGreaterThan(
        10,
      ),
    );
    const tables = container.querySelectorAll('table');
    for (const table of tables) {
      const region = table.closest('[role="region"]');
      expect(region?.getAttribute('tabindex')).toBe('0');
      expect(region?.hasAttribute('data-radix-scroll-area-viewport')).toBe(true);
    }
  });

  it('keeps the estimate query on return, with a storage fallback', () => {
    window.history.replaceState(null, '', '/about.html?batch=8');
    sessionStorage.setItem('specmaxx:last-query', '?batch=4');
    expect(estimateHref()).toBe('./?batch=8');
    window.history.replaceState(null, '', '/about.html');
    expect(estimateHref()).toBe('./?batch=4');
  });

  it('does not use stored navigation targets as a URL', () => {
    sessionStorage.setItem('specmaxx:last-query', '//untrusted.example');
    expect(estimateHref()).toBe('./');
  });
});
