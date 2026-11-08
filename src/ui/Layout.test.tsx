// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { App } from './App';
import { inlineRunner } from './sim/runner';
import { STACKED_QUERY } from './layout';

/** matchMedia answering `stacked` for the phone query (jsdom has no matchMedia of its own). */
function mockViewport(stacked: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: query === STACKED_QUERY ? stacked : false,
    media: query,
    addEventListener() {},
    removeEventListener() {},
  }));
}
/** Whether `a` comes before `b` in document order. */
const before = (a: Element, b: Element) =>
  !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);

beforeEach(() => {
  history.replaceState(null, '', '/');
  localStorage.clear();
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('workspace layout', () => {
  it('splits output (left) | splitter | inputs (right) on wide screens', async () => {
    mockViewport(false);
    const { container } = render(<App runner={inlineRunner} />);
    await screen.findByTestId('decode-tps');
    const workspace = container.querySelector('main.workspace')!;
    expect(workspace.getAttribute('data-layout')).toBe('split');
    expect([...workspace.children].map((c) => c.className.split(' ')[0])).toEqual([
      'output',
      'splitter',
      'input-band',
    ]);
    expect(screen.getByRole('separator', { name: 'Resize inputs' })).toBeTruthy();
    // The roofline sits in the hero, beside the figures; every input section starts open.
    const hero = container.querySelector('.hero')!;
    expect(hero.querySelector('[data-chart="roofline"]')).not.toBeNull();
    expect(
      [...container.querySelectorAll<HTMLDetailsElement>('#inputs-pane details.panel')]
        .slice(0, 3)
        .every((d) => d.open),
    ).toBe(true);
  });

  it('stacks on phones: headline, figures, memory, then inputs, then the analysis', async () => {
    mockViewport(true);
    const { container } = render(<App runner={inlineRunner} />);
    await screen.findByTestId('decode-tps');
    const workspace = container.querySelector('main.workspace')!;
    expect(workspace.getAttribute('data-layout')).toBe('stacked');
    expect(screen.queryByRole('separator')).toBeNull();
    const order = [
      container.querySelector('.rig')!,
      screen.getByTestId('prefill-tps'),
      screen.getByTestId('decode-tps'),
      screen.getByTestId('memory-verdict'),
      screen.getByTestId('mini-result'),
      container.querySelector('#inputs-pane')!,
      container.querySelector('[data-chart="roofline"]')!,
      screen.getByRole('heading', { name: /Bottleneck estimation/ }),
      container.querySelector('[data-chart="sweep"]')!,
    ];
    for (let i = 1; i < order.length; i++) expect(before(order[i - 1], order[i])).toBe(true);
    // Input sections start collapsed, each summarising its value.
    const panels = [
      ...container.querySelectorAll<HTMLDetailsElement>('#inputs-pane details.panel'),
    ];
    expect(panels.length).toBeGreaterThan(4);
    expect(panels.every((d) => !d.open)).toBe(true);
  });
});
