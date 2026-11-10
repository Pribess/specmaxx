// @vitest-environment jsdom
import { afterEach, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { ResultHero, BreakdownTable, MemoryFit, BatchSweepChart, RooflineChart } from './Results';
import { specDecChipText } from './components/ResultHero';
import { HeroMemoryWarning } from './components/MemoryWarning';
import { DEFAULT_STATE, buildSimInput, formatSeconds, formatTokensPerSec, reduce } from './state';
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
    'comm latency',
    'comm bandwidth',
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
  expect(screen.getByRole('tooltip').textContent).toBe(
    'batch 1 · 33.9 tok/s per stream · 33.9 aggregate @ 8k',
  );
  fireEvent.focus(points[0]);
  fireEvent.keyDown(points[0], { key: 'ArrowRight' });
  expect(document.activeElement).toBe(points[1]);
  fireEvent.pointerLeave(svg);
  expect(screen.getByRole('tooltip').textContent).toContain('batch 8');
  expect(screen.getByRole('tooltip').textContent).toBe(
    'batch 8 · 51.5 tok/s per stream · 412.3 aggregate @ 8k · does not fit in memory',
  );
  expect(container.querySelector('.chart-crosshair')).toBeTruthy();
  expect(container.querySelector('.sweep-line')).toBeTruthy();
  expect(container.querySelector('.sweep-line-stream')).toBeTruthy();
  expect(
    container.querySelectorAll('svg > g:not(.sweep-title-sample) > .sweep-point-stream').length,
  ).toBe(2);
  // Left axis: per stream (its ticks top out at the per-stream max); right axis: aggregate.
  const leftTitle = container.querySelector('.sweep-stream-title')!;
  const rightTitle = container.querySelector('.sweep-agg-title')!;
  expect(leftTitle.textContent).toBe('per stream, tok/s @ 8k');
  expect(leftTitle.getAttribute('text-anchor')).toBe('start');
  expect(Number(leftTitle.getAttribute('x'))).toBeLessThan(60);
  expect(rightTitle.textContent).toBe('aggregate, tok/s @ 8k');
  expect(rightTitle.getAttribute('text-anchor')).toBe('end');
  expect(Number(rightTitle.getAttribute('x'))).toBeGreaterThan(640);
  const tickTexts = (sel: string) => [...container.querySelectorAll(sel)].map((t) => t.textContent);
  const tickXs = (sel: string) =>
    [...container.querySelectorAll(sel)].map((t) => Number(t.getAttribute('x')));
  expect(tickTexts('.sweep-stream-tick')).toEqual(['0.0', '25.8', '51.5']);
  expect(tickXs('.sweep-stream-tick').every((v) => v < 60)).toBe(true);
  expect(tickTexts('.sweep-agg-tick')).toEqual(['0.0', '206', '412']);
  expect(tickXs('.sweep-agg-tick').every((v) => v > 640)).toBe(true);
  // Each series is drawn against its own axis: both series' maxima reach the plot top (y = 40).
  const streamTop = Math.min(
    ...[
      ...container.querySelectorAll('svg > g:not(.sweep-title-sample) > .sweep-point-stream'),
    ].map((r) => Number(r.getAttribute('y')) + 5),
  );
  const aggTop = Math.min(
    ...[...container.querySelectorAll('svg > g:not(.sweep-title-sample) > .sweep-point')].map((c) =>
      Number(c.getAttribute('cy')),
    ),
  );
  expect(streamTop).toBe(40);
  expect(aggTop).toBe(40);
  expect(container.querySelector('.sweep-legend')?.textContent).toBe(
    'Per stream (left axis)Aggregate (right axis)',
  );
  fireEvent.keyDown(points[1], { key: 'Home' });
  expect(document.activeElement).toBe(points[0]);
  fireEvent.keyDown(points[0], { key: 'End' });
  expect(document.activeElement).toBe(points[1]);
  fireEvent.keyDown(points[1], { key: 'Escape' });
  expect(screen.queryByRole('tooltip')).toBeNull();
  fireEvent.pointerDown(points[0]);
  expect(screen.getByRole('tooltip').textContent).toContain('batch 1');
  expect(screen.getByText('aggregate, tok/s @ 8k')).toBeTruthy();
  expect(screen.queryByText(/Aggregate tok\/s\. Hollow/)).toBeNull();
  // Spec-dec off: no off line, no legend item, no break-even marker.
  expect(container.querySelector('.sweep-line-off')).toBeNull();
  expect(screen.queryByTestId('sweep-legend-off')).toBeNull();
  expect(screen.queryByTestId('specdec-break-even')).toBeNull();
});
it('with spec-dec: draws the spec-dec-off line, its legend item, the break-even marker and the tooltip rates', () => {
  const point = (batch: number, agg: number, off: number) => ({
    batch,
    contextTokens: 8192,
    decodeTokensPerSec: agg,
    perSeqTokensPerSec: agg / batch,
    fits: true,
    offDecodeTokensPerSec: off,
    speedup: agg / off,
  });
  const { container } = render(
    <BatchSweepChart
      points={[point(1, 80, 33), point(32, 1500, 1000), point(64, 1900, 2000)]}
      breakEvenBatch={48}
    />,
  );
  const offLine = container.querySelector('polyline.sweep-line-off');
  expect(offLine).toBeTruthy();
  expect(offLine?.getAttribute('points')?.split(' ')).toHaveLength(3);
  // The off rate shares the aggregate (right) axis: its 2,000 tok/s sets the top tick.
  expect([...container.querySelectorAll('.sweep-agg-tick')].map((t) => t.textContent)).toEqual([
    '0.0',
    '1.0k',
    '2.0k',
  ]);
  expect(screen.getByTestId('sweep-legend-off').textContent).toBe(
    'Aggregate, spec-dec off (right axis)',
  );
  const marker = screen.getByTestId('specdec-break-even');
  expect(marker.querySelector('title')?.textContent).toBe('break-even batch 48');
  expect(marker.querySelector('text')?.textContent).toBe('break-even batch 48');
  const points = container.querySelectorAll<SVGElement>('[data-chart-point]');
  fireEvent.pointerEnter(points[2]);
  expect(screen.getByRole('tooltip').textContent).toBe(
    'batch 64 · 29.7 tok/s per stream · 1,900 aggregate · 2,000 without spec-dec · ×0.95 @ 8k',
  );
  fireEvent.pointerEnter(points[0]);
  expect(screen.getByRole('tooltip').textContent).toBe(
    'batch 1 · 80.0 tok/s per stream · 80.0 aggregate · 33.0 without spec-dec · ×2.42 @ 8k',
  );
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

it("keeps today's result unannotated, with the comm split in the bound legend", async () => {
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
  expect(screen.queryByTestId('spec-dec')).toBeNull();
  expect([...container.querySelectorAll('.result-chip')].map((c) => c.textContent)).toEqual([
    'spec-dec: off',
  ]);
  expect(screen.queryAllByTestId('memory-tier')).toHaveLength(0);
  expect(container.querySelector('.allocation-table')).toBeNull();
});

it('annotates a spec-dec decode: speedup with E[τ] help, verify and draft layer groups', async () => {
  const { built, summary } = await defaultResult();
  const draftRow = { ...summary.breakdown.decode.byLayer[1], key: 'draft.layer.0' };
  const decode = summary.breakdown.decode;
  const verifyRows = decode.byLayer.map((r) => ({ ...r, section: 'verify' as const }));
  const sum = (rows: typeof decode.byLayer, key: string) => ({
    ...rows[0],
    key,
    sec: rows.reduce((s, r) => s + r.sec, 0),
    share: rows.reduce((s, r) => s + r.share, 0),
  });
  const spec = {
    ...summary,
    decode: {
      ...summary.decode,
      speculative: {
        draftSec: 1.2e-3,
        verifySec: 9.8e-3,
        expectedTokensPerStep: 2.31,
        offTokensPerSec: 100,
        offCycleSec: 1e-2,
        stepCostRatio: 2.31 / 1.67,
        speedup: 1.67,
      },
    },
    speculative: { method: 'dflash2' as const, draftLength: 3, hypothetical: true },
    breakdown: {
      ...summary.breakdown,
      decode: {
        ...decode,
        byLayer: [...verifyRows, { ...draftRow, section: 'draft' as const }],
        sections: [sum(verifyRows, 'verify'), sum([draftRow], 'draft')],
      },
    },
  };
  render(
    <ResultHero
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={spec}
      pending={false}
    />,
  );
  const line = screen.getByTestId('decode-tps').parentElement!;
  expect(line.textContent).toMatch(/tok\/s @ 8k\(spec-dec ×1\.67\)/);
  const help = screen.getByRole('button', { name: 'About Speculative decoding speedup' });
  fireEvent.focus(help);
  const tipLines = [...screen.getByRole('tooltip').querySelectorAll('.help-line')].map(
    (l) => l.textContent,
  );
  expect(tipLines).toEqual([
    'DFlash 2, k=3.',
    '2.31 tokens accepted per step on average (incl. the bonus token).',
    'One step: draft 1.20ms + verify 9.80ms.',
  ]);
  const chipTexts = [...document.querySelectorAll('.result-chip')].map((c) => c.textContent);
  expect(chipTexts).toEqual(['spec-dec: DFlash 2, k=3 · hypothetical drafter']);
  cleanup();
  render(<BreakdownTable result={spec} />);
  fireEvent.click(screen.getByRole('button', { name: 'Layer' }));
  // Each section is cut on its own: the draft row shows under the long verify section without "Show all".
  expect(screen.getAllByRole('row').length).toBe(1 + 2 + 12 + 1);
  const sections = screen.getAllByTestId('breakdown-section');
  expect(sections.map((s) => s.querySelector('th')?.textContent)).toEqual(['verify', 'draft']);
  const draftBody = sections[1].parentElement!;
  expect(within(draftBody).getByText('draft.layer.0')).toBeTruthy();
});

it('annotates an offload result: chips, offload transfer bound, one memory bar per tier', async () => {
  const { built, summary } = await defaultResult();
  const transferRow = {
    key: 'moe.expert_fetch',
    sec: 2e-3,
    share: 0.5,
    boundSec: { compute: 0, memory: 0, commLatency: 0, commBandwidth: 0, transfer: 2e-3 },
    flops: 0,
    bytes: 0,
    count: 1,
  };
  const offload = {
    ...summary,
    offload: [
      { group: 'routed_experts' as const, tier: 'host_ram' as const, hitRatio: 0.5 },
      { group: 'lookup_tables' as const, tier: 'nvme' as const, layers: { from: 24, to: 36 } },
    ],
    breakdown: {
      ...summary.breakdown,
      decode: {
        ...summary.breakdown.decode,
        byOp: [transferRow, ...summary.breakdown.decode.byOp],
        byKind: [{ ...transferRow, key: 'transfer' }, ...summary.breakdown.decode.byKind],
      },
    },
    memory: {
      ...summary.memory,
      tiers: [
        {
          tier: 'host_ram' as const,
          bytes: 16.3e9,
          breakdown: {
            lookupTables: 0,
            routedExperts: 16.3e9,
            draftExperts: 0,
            tokenEmbeddings: 0,
            kvCache: 0,
          },
          capacityBytes: 96e9,
          fits: true,
        },
        {
          tier: 'nvme' as const,
          bytes: 2.2e12,
          breakdown: {
            lookupTables: 0,
            routedExperts: 2.2e12,
            draftExperts: 0,
            tokenEmbeddings: 0,
            kvCache: 0,
          },
          capacityBytes: 2e12,
          fits: false,
        },
      ],
    },
  };
  render(
    <ResultHero
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={offload}
      pending={false}
    />,
  );
  const chips = screen.getByRole('list', { name: 'Result settings' });
  expect([...chips.querySelectorAll('.result-chip')].map((c) => c.textContent)).toEqual([
    'spec-dec: off',
    'MoE experts: host RAM, h 0.50',
    'n-gram tables 24–35: SSD',
  ]);
  const tiers = screen.getAllByTestId('memory-tier');
  expect(tiers.map((t) => t.firstElementChild?.textContent)).toEqual([
    'Host RAM 16.3GB / 96.0GB',
    'SSD 2.2TB / 2.0TB: does not fit',
  ]);
  expect(tiers[1].firstElementChild?.className).toBe('memory-danger');
  fireEvent.focus(tiers[0].querySelector('[data-chart-point]')!);
  expect(screen.getByRole('tooltip').textContent).toBe('Host RAM · MoE experts · 16.3GB @ 8k');
  cleanup();
  const { container } = render(<BreakdownTable result={offload} />);
  const legend = screen.getByRole('list', { name: 'Limiting resource' });
  expect([...legend.querySelectorAll('li')].map((li) => li.textContent)).toEqual([
    'compute',
    'memory',
    'comm latency',
    'comm bandwidth',
    'offload transfer',
  ]);
  const first = container.querySelector('tbody tr')!;
  expect(first.querySelector('th')?.textContent).toBe('moe.expert_fetch');
  expect(first.querySelector('.transfer-text')?.textContent).toBe('offload transfer');
  expect((first.querySelector('.bound-bar .transfer') as HTMLElement).style.width).toBe('100%');
});

it('labels a host-RAM + SSD split in the tier bars with each tier’s layer span', async () => {
  const { summary } = await defaultResult();
  const tier = (t: 'host_ram' | 'nvme', experts: number, tables: number) => ({
    tier: t,
    bytes: experts + tables,
    breakdown: {
      lookupTables: tables,
      routedExperts: experts,
      draftExperts: 0,
      tokenEmbeddings: 0,
      kvCache: 0,
    },
    capacityBytes: 2e12,
    fits: true,
  });
  const split = {
    ...summary,
    offload: [
      {
        group: 'routed_experts' as const,
        tier: 'host_ram' as const,
        hitRatio: 0,
        layers: { from: 0, to: 24 },
      },
      {
        group: 'routed_experts' as const,
        tier: 'nvme' as const,
        hitRatio: 0,
        layers: { from: 24, to: 48 },
      },
      { group: 'lookup_tables' as const, tier: 'host_ram' as const, layers: { from: 0, to: 47 } },
      { group: 'lookup_tables' as const, tier: 'nvme' as const, layers: { from: 47, to: 48 } },
    ],
    memory: { ...summary.memory, tiers: [tier('host_ram', 40e9, 2e9), tier('nvme', 40e9, 1e8)] },
  };
  render(<MemoryFit result={split} />);
  const legends = screen.getAllByTestId('memory-tier-legend');
  const labels = legends.map((l) =>
    [...l.querySelectorAll('li:has(.memory-swatch)')].map(
      (li) => li.firstChild?.nextSibling?.textContent,
    ),
  );
  expect(labels).toEqual([
    ['Lookup tables, layers 0–46', 'MoE experts, layers 0–23'],
    ['Lookup tables, layer 47', 'MoE experts, layers 24–47'],
  ]);
  fireEvent.focus(
    screen.getAllByTestId('memory-tier')[1].querySelectorAll('[data-chart-point]')[1],
  );
  expect(screen.getByRole('tooltip').textContent).toBe(
    'SSD · MoE experts, layers 24–47 · 40.0GB @ 8k',
  );
});

it('shows a Token embeddings segment in the host-RAM tier bar', async () => {
  const { summary } = await defaultResult();
  const embedded = {
    ...summary,
    offload: [{ group: 'token_embeddings' as const, tier: 'host_ram' as const }],
    memory: {
      ...summary.memory,
      tiers: [
        {
          tier: 'host_ram' as const,
          bytes: 1.6e9,
          breakdown: {
            lookupTables: 0,
            routedExperts: 0,
            draftExperts: 0,
            tokenEmbeddings: 1.6e9,
            kvCache: 0,
          },
          capacityBytes: 96e9,
          fits: true,
        },
      ],
    },
  };
  const { container } = render(<MemoryFit result={embedded} />);
  const legend = screen.getByTestId('memory-tier-legend');
  expect(
    [...legend.querySelectorAll('li:has(.memory-swatch)')].map(
      (li) => li.firstChild?.nextSibling?.textContent,
    ),
  ).toEqual(['Token embeddings']);
  expect(container.querySelector('.allocation-tier-embed')).toBeTruthy();
});

it('hatches an on-device n-gram table and offers one-click fixes (to the SSD on unified memory)', async () => {
  const { summary } = await defaultResult();
  const stage = summary.memory.stages[0];
  const placed: string[] = [];
  const table = {
    ...summary,
    memory: {
      ...summary.memory,
      kind: 'unified' as const,
      stages: [{ ...stage, lookupTableBytes: 0.4 * stage.weightBytes }],
      tiers: [],
    },
  };
  const suggestions = {
    prefill: {
      suggestions: [
        {
          id: 'tables-nvme',
          label: 'n-gram tables → SSD',
          help: ['Moves the n-gram tables off the device to the SSD.'],
          actions: [],
          fits: true,
        },
      ],
    },
  };
  const { container } = render(
    <MemoryFit
      result={table}
      suggestions={suggestions}
      onSuggestion={(s, pool) => placed.push(`${pool}:${s.id}`)}
    />,
  );
  const legend = screen.getByTestId('memory-legend');
  expect(
    [...legend.querySelectorAll('li')].map((li) => li.firstChild?.nextSibling?.textContent),
  ).toEqual(['Weights', 'n-gram table', 'KV + state', 'Activations']);
  expect(container.querySelector('.capacity-bar .allocation-table')).toBeTruthy();
  // On unified memory host RAM is the device pool, so the table moves to the SSD.
  fireEvent.click(screen.getByRole('button', { name: 'n-gram tables → SSD' }));
  expect(placed).toEqual(['prefill:tables-nvme']);
  // Unified memory: host RAM is the device pool, so no host-RAM tier or line ever shows.
  expect(screen.queryAllByTestId('memory-tier')).toHaveLength(0);
  expect(screen.queryByTestId('memory-unified-host')).toBeNull();
});

it('under disaggregation, shows both pools with their own bars and verdicts, and fixes apply to their pool', async () => {
  const { summary } = await defaultResult();
  const stage = summary.memory.stages[0];
  const withStage = (totalBytes: number, fits: boolean) => ({
    ...summary,
    memory: { ...summary.memory, fits, tiers: [], stages: [{ ...stage, totalBytes, fits }] },
  });
  const fitting = withStage(0.5 * stage.capacityBytes, true);
  const over = withStage(1.5 * stage.capacityBytes, false);
  const ran: string[] = [];
  render(
    <MemoryFit
      result={fitting}
      pools={{ prefill: fitting, decode: over }}
      suggestions={{
        decode: {
          suggestions: [
            { id: 'kv', label: 'KV cache → host RAM', help: ['x'], actions: [], fits: true },
          ],
        },
      }}
      onSuggestion={(s, pool) => ran.push(`${pool}:${s.id}`)}
    />,
  );
  const groups = screen.getAllByTestId('memory-pool');
  expect(groups.map((g) => g.getAttribute('aria-label'))).toEqual(['Prefill pool', 'Decode pool']);
  for (const g of groups) {
    expect(g.querySelectorAll('.capacity-bar')).toHaveLength(1);
    expect(within(g).getAllByTestId('memory-verdict')).toHaveLength(1);
  }
  expect(within(groups[0]).getByTestId('memory-verdict').textContent).toMatch(/^Fits/);
  expect(within(groups[1]).getByTestId('memory-verdict').textContent).toMatch(/^Does not fit/);
  // Only the failing pool has fixes; the click names its pool.
  expect(within(groups[0]).queryByTestId('memory-fixes')).toBeNull();
  fireEvent.click(within(groups[1]).getByRole('button', { name: 'KV cache → host RAM' }));
  expect(ran).toEqual(['decode:kv']);
  // Chart tooltips name the pool.
  fireEvent.focus(groups[1].querySelector('[data-chart-point]')!);
  expect(screen.getByRole('tooltip').textContent).toMatch(/^Decode pool · Stage 1 · /);
});

it('the hero memory warning names the pool that does not fit', async () => {
  const { summary } = await defaultResult();
  const stage = summary.memory.stages[0];
  const memory = {
    ...summary.memory,
    stages: [{ ...stage, totalBytes: 2 * stage.capacityBytes, fits: false }],
  };
  render(<HeroMemoryWarning memory={memory} pool="Decode pool" />);
  expect(screen.getByRole('link').getAttribute('title')).toBe('Decode pool: does not fit');
  expect(screen.getByRole('img').getAttribute('aria-label')).toBe('Decode pool: does not fit');
});

function rigText(state: ReturnType<typeof reduce>): string {
  const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  const { container } = render(
    <ResultHero state={state} loaded={BUNDLED_DEFAULT_MODEL} built={built} pending={false} />,
  );
  return container.querySelector('.rig')?.textContent ?? '';
}
it('names the chosen memory size in the rig line for Apple and multi-capacity devices only', () => {
  const apple = reduce(
    reduce(DEFAULT_STATE, { type: 'setDevice', deviceId: 'apple-m5-ultra-64c' }),
    { type: 'setCapacity', index: 1 },
  );
  expect(rigText(apple)).toContain('Apple M5 Ultra (64-core GPU, 256GB memory)');
  cleanup();
  const halo = reduce(DEFAULT_STATE, { type: 'setDevice', deviceId: 'ryzen-ai-max-plus-395' });
  expect(rigText(halo)).toMatch(/\(.*128GB memory\)/);
  cleanup();
  expect(rigText(reduce(halo, { type: 'setCapacity', index: 1 }))).toMatch(/\(.*, 64GB memory\)/);
  cleanup();
  // A single-capacity Apple device still names its memory.
  expect(
    rigText(reduce(DEFAULT_STATE, { type: 'setDevice', deviceId: 'apple-m5-max-32c' })),
  ).toContain('(32-core GPU, 36GB memory)');
  cleanup();
  expect(rigText(reduce(DEFAULT_STATE, { type: 'setDevice', deviceId: 'rtx-5090' }))).not.toContain(
    'memory)',
  );
  cleanup();
  const edited = reduce(apple, {
    type: 'setEdit',
    target: 'device',
    path: 'memory.capacityOptionsBytes.1',
    value: 300e9,
  });
  expect(rigText(edited)).toContain('(64-core GPU, 300GB memory) (edited)');
});
it('shows no what-if figures on one device with the overlay off', async () => {
  const { built, summary } = await defaultResult();
  render(
    <ResultHero
      state={DEFAULT_STATE}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={summary}
      pending={false}
    />,
  );
  expect(summary.prefill.withoutOverlap).toBeUndefined();
  expect(screen.queryByTestId('prefill-no-overlap')).toBeNull();
  expect(screen.queryByTestId('decode-dispatch')).toBeNull();
});
it('shows prefill without overlap and decode with engine dispatch as labelled second figures, headline unchanged', async () => {
  let state = reduce(DEFAULT_STATE, { type: 'setCounts', devicesPerNode: 2 });
  state = reduce(state, { type: 'setParallel', parallel: { tp: 2 } });
  const plain = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  state = reduce(state, { type: 'setEngineDispatchOverlay', on: true });
  const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok' || plain.kind !== 'ok') throw new Error('invalid test state');
  const { summary } = await inlineRunner.run(built.input, built.notes);
  const { summary: off } = await inlineRunner.run(plain.input, plain.notes);
  expect(summary.prefill.tokensPerSec).toBe(off.prefill.tokensPerSec);
  expect(summary.decode.tokensPerSec).toBe(off.decode.tokensPerSec);
  render(
    <ResultHero
      state={state}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={summary}
      pending={false}
    />,
  );
  const prefill = screen.getByTestId('prefill-no-overlap');
  expect(prefill.textContent).toContain(
    `Without overlap: ${formatTokensPerSec(summary.prefill.withoutOverlap!.tokensPerSec)}`,
  );
  expect(summary.prefill.withoutOverlap!.tokensPerSec).toBeLessThan(summary.prefill.tokensPerSec);
  expect(within(prefill).getByRole('button')).toBeTruthy();
  const decode = screen.getByTestId('decode-dispatch');
  expect(decode.textContent).toContain(
    `With engine dispatch: ${formatTokensPerSec(summary.decode.withDispatch!.tokensPerSec)}`,
  );
  expect(summary.decode.withDispatch!.tokensPerSec).toBeLessThan(summary.decode.tokensPerSec);
  expect(within(decode).getByRole('button')).toBeTruthy();
  // The headline figures are the theoretical ones.
  expect(screen.getByTestId('decode-tps').textContent).toBe(
    formatTokensPerSec(off.decode.tokensPerSec),
  );
});

it('writes the spec-dec chip: off, MTP, DSpark with a use case, hypothetical drafter', () => {
  expect(specDecChipText(undefined)).toBe('spec-dec: off');
  expect(specDecChipText({ method: 'mtp', draftLength: 3, hypothetical: false })).toBe(
    'spec-dec: MTP, k=3',
  );
  expect(
    specDecChipText({ method: 'dspark', draftLength: 7, hypothetical: false }, { useCase: 'code' }),
  ).toBe('spec-dec: DSpark, k=7 · code');
  expect(
    specDecChipText(
      { method: 'dspark', draftLength: 7, hypothetical: false },
      { useCase: 'prose' },
    ),
  ).toBe('spec-dec: DSpark, k=7 · prose');
  expect(specDecChipText({ method: 'dflash2', draftLength: 3, hypothetical: false }, {})).toBe(
    'spec-dec: DFlash 2, k=3 · prose',
  );
  expect(specDecChipText({ method: 'mtp', draftLength: 3, hypothetical: false }, {})).toBe(
    'spec-dec: MTP, k=3',
  );
  expect(
    specDecChipText(
      { method: 'dspark', draftLength: 7, hypothetical: false },
      { useCase: 'code', alpha: 0.8 },
    ),
  ).toBe('spec-dec: DSpark, k=7');
  expect(
    specDecChipText({ method: 'dspark', draftLength: 7, hypothetical: true }, { useCase: 'code' }),
  ).toBe('spec-dec: DSpark, k=7 · code · hypothetical drafter');
});
it('chunked PP prefill shows the time to first token with and without chunking; off shows neither', async () => {
  let state = reduce(DEFAULT_STATE, { type: 'setCounts', devicesPerNode: 4 });
  state = reduce(state, { type: 'setParallel', parallel: { pp: 4 } });
  state = reduce(state, { type: 'setWorkload', workload: { promptLen: 32768 } });
  const plain = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (plain.kind !== 'ok') throw new Error(JSON.stringify(plain.issues));
  const { summary: off } = await inlineRunner.run(plain.input, plain.notes);
  expect(off.prefill.chunkedPipeline).toBeUndefined();
  render(
    <ResultHero
      state={state}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={plain}
      result={off}
      pending={false}
    />,
  );
  expect(screen.queryByTestId('prefill-ttft')).toBeNull();
  expect(screen.queryByTestId('prefill-ttft-unchunked')).toBeNull();
  expect(screen.getByRole('heading', { level: 1 }).textContent).not.toContain('chunked PP');
  cleanup();

  state = reduce(state, { type: 'setParallel', parallel: { chunkedPipelinePrefill: true } });
  const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  const { summary } = await inlineRunner.run(built.input, built.notes);
  const cpp = summary.prefill.chunkedPipeline!;
  expect(cpp.chunks).toBe(4);
  // The unchunked what-if is exactly the run with the switch off.
  expect(cpp.unchunkedLatencySec).toBeCloseTo(off.prefill.latencySec, 12);
  expect(summary.prefill.latencySec).toBeLessThan(cpp.unchunkedLatencySec);
  render(
    <ResultHero
      state={state}
      loaded={BUNDLED_DEFAULT_MODEL}
      built={built}
      result={summary}
      pending={false}
    />,
  );
  const ttft = screen.getByTestId('prefill-ttft');
  expect(ttft.textContent).toContain(
    `Time to first token: ${formatSeconds(summary.prefill.latencySec)} @ 32k`,
  );
  expect(within(ttft).getByRole('button')).toBeTruthy();
  const unchunked = screen.getByTestId('prefill-ttft-unchunked');
  expect(unchunked.textContent).toContain(
    `Without chunking: ${formatSeconds(cpp.unchunkedLatencySec)} @ 32k`,
  );
  expect(within(unchunked).getByRole('button')).toBeTruthy();
  expect(screen.getByRole('heading', { level: 1 }).textContent).toContain(', chunked PP');
});
