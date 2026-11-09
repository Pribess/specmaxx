// @vitest-environment jsdom
import { afterEach, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { DEFAULT_STATE, buildSimInput } from '../state';
import { BUNDLED_DEFAULT_MODEL } from '../state/bundledModel';
import { inlineRunner } from '../sim/runner';
import { ResultHero } from './ResultHero';

afterEach(cleanup);

async function showStages(utilisations: number[]) {
  const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error('invalid fixture');
  const { summary } = await inlineRunner.run(built.input, built.notes);
  summary.memory.stages = utilisations.map((used, stage) => ({
    stage,
    weightBytes: used * 30e9,
    sequenceStateBytes: used * 1e9,
    activationPeakBytes: used * 1e9,
    totalBytes: used * 32e9,
    capacityBytes: 32e9,
    fits: used <= 1,
  }));
  return render(
    <ResultHero
      result={summary}
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      pending={false}
    />,
  );
}

it.each([0.5, 0.9])('keeps a comfortable fit neutral at utilisation %s', async (used) => {
  const { container } = await showStages([used]);
  expect(screen.getByTestId('memory-verdict').className).toBe('memory-fits');
  expect(screen.getByTestId('memory-verdict').textContent).toMatch(/^Fits:/);
  expect(screen.queryByRole('img', { name: /Warning:/ })).toBeNull();
  expect(container.querySelector('.hero-memory-warning')).toBeNull();
  expect(screen.queryByRole('button', { name: 'About High memory utilisation' })).toBeNull();
});

it.each([
  [0.901, '90.1%'],
  [0.90001, '90.1%'],
  [0.9999, '99.9%'],
  [0.92, '92%'],
  [1, '100%'],
] as const)('warns in orange above 90%% while still fitting (%s)', async (used, percent) => {
  const { container } = await showStages([used]);
  const verdict = screen.getByTestId('memory-verdict');
  expect(verdict.className).toBe('memory-warning');
  expect(verdict.textContent).toContain(`(${percent})`);
  expect(screen.getAllByRole('img', { name: `Warning: ${percent} of memory used` })).toHaveLength(
    3,
  );
  const links = screen.getAllByRole('link', { name: `Warning: ${percent} of memory used` });
  expect(links).toHaveLength(2);
  expect(links[0].getAttribute('href')).toBe('#memory-fit');
  expect(container.querySelector('#memory-fit')).toBeTruthy();
  expect(container.querySelector('.capacity-bar.memory-warning')).toBeTruthy();
  fireEvent.focus(screen.getByRole('button', { name: 'About High memory utilisation' }));
  expect(screen.getByRole('tooltip').textContent).toContain(
    'Little memory is left for the OS and the runtime',
  );
  expect(screen.getByRole('tooltip').textContent).toContain('Unified memory is shared with the OS');
});

it('warns in red about overflow and keeps headline rates', async () => {
  const { container } = await showStages([1.2]);
  expect(screen.getByTestId('memory-verdict').className).toBe('memory-danger');
  expect(screen.getByTestId('memory-verdict').textContent).toContain('Does not fit');
  expect(screen.getAllByRole('img', { name: 'Warning: does not fit' })).toHaveLength(3);
  expect(container.querySelector('.capacity-bar.overflow.memory-danger')).toBeTruthy();
  expect(screen.getByTestId('decode-tps').textContent).toMatch(/\d/);
});

it.each([
  {
    used: [0.5, 0.92],
    classes: ['memory-fits', 'memory-warning'],
    label: 'Warning: 92% of memory used',
  },
  { used: [0.5, 1.2], classes: ['memory-fits', 'memory-danger'], label: 'Warning: does not fit' },
  {
    used: [0.92, 1.2],
    classes: ['memory-warning', 'memory-danger'],
    label: 'Warning: does not fit',
  },
  {
    used: [0.92, 0.98],
    classes: ['memory-warning', 'memory-warning'],
    label: 'Warning: 98% of memory used',
  },
])(
  'uses per-stage verdicts and the worst hero warning: $used',
  async ({ used, classes, label }) => {
    await showStages(used);
    expect(screen.getAllByTestId('memory-verdict').map((p) => p.className)).toEqual(classes);
    expect(screen.getAllByRole('link', { name: label })).toHaveLength(2);
  },
);

it('shows the red hero warning naming a failing tier when every stage fits', async () => {
  const { container } = await showStages([0.5]);
  cleanup();
  const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error('invalid fixture');
  const { summary } = await inlineRunner.run(built.input, built.notes);
  summary.memory.tiers = [
    {
      tier: 'nvme',
      bytes: 2.2e12,
      breakdown: { lookupTables: 2.2e12, routedExperts: 0, denseLayers: 0, draftExperts: 0 },
      capacityBytes: 2e12,
      fits: false,
    },
  ];
  const view = render(
    <ResultHero
      result={summary}
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      pending={false}
    />,
  );
  void container;
  const links = screen.getAllByRole('link', { name: /Does not fit: SSD 2\.2TB of 2\.0TB/ });
  expect(links).toHaveLength(2);
  expect(view.container.querySelectorAll('.hero-memory-warning')).toHaveLength(2);
  expect(links[0].getAttribute('href')).toBe('#memory-fit');
});

it('is red, not amber, when a stage is above 90% but a tier does not fit', async () => {
  const built = buildSimInput(DEFAULT_STATE, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error('invalid fixture');
  const { summary } = await inlineRunner.run(built.input, built.notes);
  summary.memory.stages = [
    {
      stage: 0,
      weightBytes: 29e9,
      sequenceStateBytes: 1e9,
      activationPeakBytes: 1e9,
      totalBytes: 31e9,
      capacityBytes: 32e9,
      fits: true,
    },
  ];
  summary.memory.tiers = [
    {
      tier: 'nvme',
      bytes: 2.2e12,
      breakdown: { lookupTables: 2.2e12, routedExperts: 0, denseLayers: 0, draftExperts: 0 },
      capacityBytes: 2e12,
      fits: false,
    },
  ];
  const view = render(
    <ResultHero
      result={summary}
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      pending={false}
    />,
  );
  const icons = view.container.querySelectorAll('.hero-memory-warning .memory-warning-icon');
  expect(icons).toHaveLength(2);
  for (const icon of icons) {
    expect(icon.classList.contains('memory-danger')).toBe(true);
    expect(icon.classList.contains('memory-warning')).toBe(false);
  }
});
