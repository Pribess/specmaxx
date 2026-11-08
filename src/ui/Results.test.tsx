// @vitest-environment jsdom
import { afterEach, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { ResultHero, BreakdownTable, MemoryFit, BatchSweepChart, RooflineChart } from './Results';
import { DEFAULT_STATE, buildSimInput, formatTokensPerSec, reduce } from './state';
import { BUNDLED_DEFAULT_MODEL } from './state/bundledModel';
import { inlineRunner } from './sim/runner';

afterEach(cleanup);
async function defaultResult() {
  const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  const { summary } = await inlineRunner.run(built.input, built.notes);
  return { built, summary };
}
it('describes the effective P2P transport rather than the base host-bounce preset', async () => {
  let state = reduce(DEFAULT_STATE, { type: 'setCounts', devicesPerNode: 2 });
  state = reduce(state, { type: 'setParallel', parallel: { tp: 2 } });
  state = reduce(state, { type: 'setUnlock', axisId: 'pcie-p2p', optionId: 'direct' });
  const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error('invalid test state');
  const { summary } = await inlineRunner.run(built.input, built.notes);
  render(
    <ResultHero
      state={state}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={summary}
      pending={false}
    />,
  );
  expect(screen.getByText(/with direct P2P/)).toBeTruthy();
  expect(screen.queryByText(/host bounce/)).toBeNull();
});
it('quotes hero rates at 8k and omits redundant per-sequence rates at batch 1', async () => {
  const { built, summary } = await defaultResult();
  const { container } = render(
    <ResultHero
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={summary}
      pending={false}
    />,
  );
  for (const phase of ['prefill', 'decode']) {
    expect(screen.getByTestId(`${phase}-tps`).parentElement?.textContent).toMatch(/tok\/s @ 8k/);
  }
  expect(container.querySelectorAll('.answer small[aria-hidden="true"]')).toHaveLength(2);
  expect(container.querySelectorAll('.answer small')).toHaveLength(2);
  expect(
    [...container.querySelectorAll<HTMLElement>('.answer small')].every(
      (line) => line.style.visibility === 'hidden',
    ),
  ).toBe(true);
  const note = container.querySelector('.theory');
  expect(note?.textContent).toContain('This is a theoretical maximum.');
  expect(
    within(note as HTMLElement)
      .getByRole('link', { name: 'Read how the estimate works' })
      .getAttribute('href'),
  ).toBe('./about.html');
  expect(container.querySelector('figcaption')).toBeNull();
  expect(screen.queryByText(/memory bandwidth per device/)).toBeNull();
});
it('shows terse per-sequence rates at batch 8', async () => {
  const state = reduce(DEFAULT_STATE, { type: 'setWorkload', workload: { batch: 8 } });
  const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  const { summary } = await inlineRunner.run(built.input, built.notes);
  render(
    <ResultHero
      state={state}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={summary}
      pending={false}
    />,
  );
  for (const phase of ['prefill', 'decode'] as const) {
    expect(
      screen.getByText(
        `${formatTokensPerSec(summary[phase].perSequenceTokensPerSec)} per sequence`,
      ),
    ).toBeTruthy();
  }
});
it('scales bound bars by relative row share and uses a labels-only resource legend', async () => {
  const { summary } = await defaultResult();
  const { container } = render(<BreakdownTable result={summary} />);
  expect(screen.getByRole('heading', { name: /Bottleneck estimation/ })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Decode @ 8k' })).toBeTruthy();
  const legend = screen.getByRole('list', { name: 'Limiting resource' });
  expect([...legend.querySelectorAll('li')].map((li) => li.textContent)).toEqual([
    'compute',
    'memory',
    'comm',
  ]);
  expect(container.querySelector('p.hint')).toBeNull();
  const bars = [...container.querySelectorAll<HTMLElement>('.bound-bar')];
  const maxShare = Math.max(...summary.breakdown.decode.byOp.map((row) => row.share));
  bars.forEach((bar, index) =>
    expect(parseFloat(bar.style.width)).toBeCloseTo(
      (summary.breakdown.decode.byOp[index].share / maxShare) * 100,
    ),
  );
});
it('labels memory allocations with no-space units and supports segment focus, arrows and tap', async () => {
  const { summary } = await defaultResult();
  const { container } = render(<MemoryFit result={summary} />);
  const legend = screen.getByTestId('memory-legend');
  expect(legend.textContent).toMatch(/Weights.*GB/);
  expect(legend.textContent).not.toMatch(/\d GB/);
  expect(legend.querySelectorAll('.memory-swatch')).toHaveLength(3);
  const segments = container.querySelectorAll<HTMLElement>('[data-chart-point]');
  fireEvent.focus(segments[0]);
  expect(screen.getByRole('tooltip').textContent).toMatch(/Weights.*GB @ 8k/);
  fireEvent.keyDown(segments[0], { key: 'ArrowRight' });
  expect(document.activeElement).toBe(segments[1]);
  expect(screen.getByRole('tooltip').textContent).toContain('KV + state');
  fireEvent.pointerDown(segments[2]);
  expect(screen.getByRole('tooltip').textContent).toContain('Activations');
  expect(screen.queryByText(/Throughput is a mathematical ceiling only;/)).toBeNull();
  expect(screen.getByText(/max .* sequences per replica @ 8k/)).toBeTruthy();
});
it('shows sweep x/y/context and memory-fit status on hover, keyboard navigation and tap', () => {
  const { container } = render(
    <BatchSweepChart
      points={[
        {
          batch: 1,
          contextTokens: 8192,
          decodeTokensPerSec: 33.91,
          perSeqTokensPerSec: 33.91,
          fits: true,
        },
        {
          batch: 8,
          contextTokens: 8192,
          decodeTokensPerSec: 412.3,
          perSeqTokensPerSec: 51.54,
          fits: false,
        },
      ]}
    />,
  );
  const points = container.querySelectorAll<SVGElement>('[data-chart-point]');
  expect(points[0].getAttribute('tabindex')).toBe('0');
  expect(points[1].getAttribute('tabindex')).toBe('-1');
  const svg = container.querySelector('svg')!;
  Object.defineProperty(svg, 'getBoundingClientRect', { value: () => ({ left: 0, width: 700 }) });
  fireEvent(svg, Object.assign(new Event('pointermove', { bubbles: true }), { clientX: 650 }));
  expect(screen.getByRole('tooltip').textContent).toContain('batch 8');
  fireEvent.pointerLeave(svg);
  expect(screen.queryByRole('tooltip')).toBeNull();
  fireEvent.pointerEnter(points[0]);
  expect(screen.getByRole('tooltip').textContent).toBe('batch 1 · 33.9 tok/s @ 8k');
  fireEvent.focus(points[0]);
  fireEvent.keyDown(points[0], { key: 'ArrowRight' });
  expect(document.activeElement).toBe(points[1]);
  fireEvent.pointerLeave(svg);
  expect(screen.getByRole('tooltip').textContent).toContain('batch 8');
  expect(screen.getByRole('tooltip').textContent).toBe(
    'batch 8 · 412.3 tok/s @ 8k · does not fit in memory',
  );
  expect(container.querySelector('.chart-crosshair')).toBeTruthy();
  fireEvent.keyDown(points[1], { key: 'Home' });
  expect(document.activeElement).toBe(points[0]);
  fireEvent.keyDown(points[0], { key: 'End' });
  expect(document.activeElement).toBe(points[1]);
  fireEvent.keyDown(points[1], { key: 'Escape' });
  expect(screen.queryByRole('tooltip')).toBeNull();
  fireEvent.pointerDown(points[0]);
  expect(screen.getByRole('tooltip').textContent).toContain('batch 1');
  expect(screen.getByText('aggregate tok/s @ 8k')).toBeTruthy();
  expect(screen.queryByText(/Aggregate tok\/s\. Hollow/)).toBeNull();
});
it('shows roofline x/y units and per-phase contexts on hover, focus, arrows and tap', async () => {
  const { summary } = await defaultResult();
  const { container } = render(
    <RooflineChart roof={summary.roofline} contexts={{ prefill: 8192, decode: 32768 }} />,
  );
  const points = container.querySelectorAll<SVGElement>('[data-chart-point]');
  fireEvent.pointerEnter(points[0]);
  expect(screen.getByRole('tooltip').textContent).toMatch(/Prefill · .* FLOP\/B · .*FLOP\/s @ 8k/);
  fireEvent.focus(points[0]);
  fireEvent.keyDown(points[0], { key: 'ArrowRight' });
  expect(document.activeElement).toBe(points[1]);
  expect(screen.getByRole('tooltip').textContent).toMatch(/Decode · .* FLOP\/B · .*FLOP\/s @ 32k/);
  fireEvent.pointerDown(points[0]);
  expect(screen.getByRole('tooltip').textContent).toContain('Prefill');
  expect(screen.getByText('TFLOP/s @ 8k prefill / 32k decode')).toBeTruthy();
  expect(screen.getByText('Compute bound')).toBeTruthy();
  expect(screen.getByText('Bandwidth bound')).toBeTruthy();
  expect(container.textContent).not.toMatch(/Tensor ceiling|Bandwidth slope/);
});

it('warns about an oversized workload while still showing theoretical rates', async () => {
  const { built, summary } = await defaultResult();
  const oversized = {
    ...summary,
    memory: {
      ...summary.memory,
      fits: false,
      stages: summary.memory.stages.map((stage) => ({ ...stage, fits: false })),
    },
  };
  const { container } = render(
    <ResultHero
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={oversized}
      pending={false}
    />,
  );
  expect(screen.getByTestId('memory-verdict').textContent).toContain('Does not fit');
  expect(screen.getByTestId('memory-verdict').classList.contains('memory-danger')).toBe(true);
  expect(screen.getAllByRole('link', { name: 'Warning: does not fit' })).toHaveLength(2);
  expect(screen.getByTestId('decode-tps').textContent).toMatch(/\d/);
  expect(container.querySelector('[role="alert"]')).toBeNull();
});
