// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import modelingMethod from '../../docs/modeling-method.md?raw';
import simulationModel from '../../docs/simulation-model.md?raw';
import { AboutPage, estimateHref } from './AboutPage';
import { githubSlug, renderMarkdown, splitTables } from './markdown';

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  window.history.replaceState(null, '', '/');
});

describe('repository Markdown', () => {
  it('resolves every simulation-model fragment to an element id', () => {
    const container = document.createElement('div');
    container.innerHTML = renderMarkdown(simulationModel);
    const ids = new Set(Array.from(container.querySelectorAll('[id]'), node => node.id));
    const links = container.querySelectorAll<HTMLAnchorElement>('a[href^="#"]');
    expect(links.length).toBeGreaterThan(10);
    for (const link of links) {
      expect(ids.has(decodeURIComponent(link.hash.slice(1))), link.hash).toBe(true);
    }
  });

  it('keeps underscores and deduplicates heading ids, including suffix collisions', () => {
    expect(githubSlug('10. Worked example: REF_DENSE_8B on one RTX 5090'))
      .toBe('10-worked-example-ref_dense_8b-on-one-rtx-5090');
    const html = renderMarkdown('# Same\n# Same\n# Same-1\n# Same');
    const container = document.createElement('div');
    container.innerHTML = html;
    expect(Array.from(container.querySelectorAll('h1'), node => node.id))
      .toEqual(['same', 'same-1', 'same-1-1', 'same-2']);
    expect(renderMarkdown('# Same')).toContain('id="same"');
  });
  it('rewrites rendered document links and leaves unavailable repository references as text', () => {
    const container = document.createElement('div');
    container.innerHTML = renderMarkdown(
      '[Model](simulation-model.md) [Checks](sanity-check.md) '
      + '[Section](simulation-model.md#per-op-time) [Source](../src/sim/) '
      + '[Other](assets/example.json) [Web](https://example.com)',
      ['simulation-model.md', 'sanity-check.md'],
    );
    expect(Array.from(container.querySelectorAll('a'), node => node.getAttribute('href')))
      .toEqual(['#simulation-model', '#sanity-check', '#per-op-time', 'https://example.com']);
    expect(container.textContent).toContain('Source Other');
    expect(renderMarkdown('[Checks](sanity-check.md)')).toBe('<p>Checks</p>\n');
  });

  it('splits rendered GFM tables out of the prose', () => {
    const html = renderMarkdown('Intro\n\n| Limit | Value |\n| --- | --- |\n| Memory | 32 GB |\n\nOutro');
    const segments = splitTables(html);
    expect(segments.map((s) => s.kind)).toEqual(['html', 'table', 'html']);
    const container = document.createElement('div');
    container.innerHTML = segments[1].html;
    expect(container.querySelector('table tbody td')?.textContent).toBe('Memory');
    expect(segments.map((s) => s.html).join('')).toBe(html);
  });
});

describe('About page', () => {
  it('renders both documents without repository paths and resolves all fragment links', () => {
    const { container } = render(<AboutPage />);
    expect(screen.getByRole('article', { name: 'Sanity checks' })).toBeTruthy();
    const ids = new Set(Array.from(container.querySelectorAll('[id]'), node => node.id));
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
  it('renders the method sections with a table of contents, and the reference collapsed', () => {
    const { container } = render(<AboutPage />);
    const method = screen.getByRole('article', { name: 'Modeling method' });
    expect(method.querySelectorAll('h2').length).toBeGreaterThanOrEqual(8);
    const toc = screen.getByRole('navigation', { name: 'Modeling method contents' });
    expect(toc.querySelectorAll('a').length).toBe(method.querySelectorAll('h2').length);
    const details = container.querySelector('details#full-reference') as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(details.contains(method)).toBe(false);
    expect(details.querySelector('article#simulation-model')).toBeTruthy();
  });

  it('opens the reference and scrolls to a heading named by the hash', async () => {
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    window.history.replaceState(null, '', '/about.html#memory-fit');
    const { container } = render(<AboutPage />);
    const details = container.querySelector('details#full-reference') as HTMLDetailsElement;
    await waitFor(() => expect(details.open).toBe(true));
    await waitFor(() => expect(scroll).toHaveBeenCalled());
    expect(scroll.mock.contexts[0]).toBe(document.getElementById('memory-fit'));
  });

  it('keeps method headings distinct from reference ids and method links resolvable', () => {
    const container = document.createElement('div');
    container.innerHTML = renderMarkdown(modelingMethod);
    const reference = document.createElement('div');
    reference.innerHTML = renderMarkdown(simulationModel);
    const referenceIds = new Set(Array.from(reference.querySelectorAll('[id]'), n => n.id));
    for (const heading of container.querySelectorAll('h2')) expect(referenceIds.has(heading.id)).toBe(false);
    const links = container.querySelectorAll<HTMLAnchorElement>('a[href^="#"]');
    expect(links.length).toBeGreaterThan(10);
    for (const link of links) expect(referenceIds.has(link.hash.slice(1)), link.hash).toBe(true);
  });

  it('renders the Memory fit heading and document navigation', () => {
    render(<AboutPage />);
    expect(screen.getByRole('heading', { name: 'Memory fit' })).toBeTruthy();
    expect(screen.getByRole('navigation', { name: 'Documents' })).toBeTruthy();
    expect(screen.getAllByRole('link', { name: 'Back to the estimate' })).toHaveLength(2);
  });

  it('puts every table in a keyboard-focusable themed scroll region', () => {
    const { container } = render(<AboutPage />);
    const tables = container.querySelectorAll('table');
    expect(tables.length).toBeGreaterThan(0);
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
