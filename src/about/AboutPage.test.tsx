// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import simulationModel from '../../docs/simulation-model.md?raw';
import { AboutPage, estimateHref } from './AboutPage';
import { githubSlug, renderMarkdown } from './markdown';

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

  it('renders GFM tables in a keyboard-focusable scroll region', () => {
    const container = document.createElement('div');
    container.innerHTML = renderMarkdown('| Limit | Value |\n| --- | --- |\n| Memory | 32 GB |');
    expect(container.querySelector('table tbody td')?.textContent).toBe('Memory');
    expect(container.querySelector('[role="region"]')?.getAttribute('tabindex')).toBe('0');
  });
});

describe('About page', () => {
  it('renders the Memory fit heading and document navigation', () => {
    render(<AboutPage />);
    expect(screen.getByRole('heading', { name: 'Memory fit' })).toBeTruthy();
    expect(screen.getByRole('navigation', { name: 'Documents' })).toBeTruthy();
    expect(screen.getAllByRole('link', { name: 'Back to the estimate' })).toHaveLength(2);
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
