// @vitest-environment jsdom
import { afterEach, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { ResultHero, BreakdownTable, MemoryFit } from './Results';
import { DEFAULT_STATE, REFERENCE_MODEL, buildSimInput, reduce } from './state';
import { inlineRunner } from './sim/runner';

afterEach(cleanup);
it('describes the effective P2P transport rather than the base host-bounce preset', async () => {
  let state = reduce(DEFAULT_STATE, { type: 'setCounts', devicesPerNode: 2 });
  state = reduce(state, { type: 'setParallel', parallel: { tp: 2 } });
  state = reduce(state, { type: 'setUnlock', axisId: 'pcie-p2p', optionId: 'direct' });
  const built = buildSimInput(state, REFERENCE_MODEL);
  if (built.kind !== 'ok') throw new Error('invalid test state');
  const { summary } = await inlineRunner.run(built.input, built.notes);
  render(
    <ResultHero
      state={state}
      loaded={REFERENCE_MODEL}
      built={built}
      result={summary}
      pending={false}
    />,
  );
  expect(screen.getByText(/with direct P2P/)).toBeTruthy();
  expect(screen.queryByText(/host bounce/)).toBeNull();
});

async function referenceSummary() {
  const built = buildSimInput(DEFAULT_STATE, REFERENCE_MODEL);
  if (built.kind !== 'ok') throw new Error('invalid test state');
  return (await inlineRunner.run(built.input, built.notes)).summary;
}
it('scales bound bars by relative row share', async () => {
  const summary = await referenceSummary();
  const { container } = render(<BreakdownTable result={summary} />);
  const bars = [...container.querySelectorAll<HTMLElement>('.bound-bar')];
  const maxShare = Math.max(...summary.breakdown.decode.byOp.map((row) => row.share));
  bars.forEach((bar, index) => {
    expect(parseFloat(bar.style.width)).toBeCloseTo(
      (summary.breakdown.decode.byOp[index].share / maxShare) * 100,
    );
  });
});
it('labels each memory segment with a swatch and its GB value', async () => {
  const summary = await referenceSummary();
  render(<MemoryFit result={summary} />);
  const legend = screen.getByTestId('memory-legend');
  expect(legend.textContent).toMatch(/Weights.*GB/);
  expect(legend.textContent).toMatch(/KV \+ state.*GB/);
  expect(legend.textContent).toMatch(/Activations.*GB/);
  expect(legend.querySelectorAll('.memory-swatch')).toHaveLength(3);
});
