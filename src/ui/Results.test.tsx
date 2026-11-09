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
  expect(container.querySelector('.result-chips')).toBeNull();
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
    'DFlash 2, draft length k = 3.',
    '2.31 tokens accepted per step on average (incl. the bonus token).',
    'One step: draft 1.20ms + verify 9.80ms.',
  ]);
  expect(screen.getByText('hypothetical drafter')).toBeTruthy();
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
      { group: 'dense_layers' as const, tier: 'nvme' as const, layers: { from: 24, to: 36 } },
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
          breakdown: { lookupTables: 0, routedExperts: 16.3e9, denseLayers: 0, draftExperts: 0 },
          capacityBytes: 96e9,
          fits: true,
        },
        {
          tier: 'nvme' as const,
          bytes: 2.2e12,
          breakdown: { lookupTables: 0, routedExperts: 2.2e12, denseLayers: 0, draftExperts: 0 },
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
  const chips = screen.getByRole('list', { name: 'Offloaded weights' });
  expect([...chips.querySelectorAll('.result-chip')].map((c) => c.textContent)).toEqual([
    'experts: host RAM, h 0.50',
    'layers 24–35: SSD',
  ]);
  const tiers = screen.getAllByTestId('memory-tier');
  expect(tiers.map((t) => t.firstElementChild?.textContent)).toEqual([
    'Host RAM 16.3GB / 96.0GB',
    'SSD 2.2TB / 2.0TB: does not fit',
  ]);
  expect(tiers[1].firstElementChild?.className).toBe('memory-danger');
  fireEvent.focus(tiers[0].querySelector('[data-chart-point]')!);
  expect(screen.getByRole('tooltip').textContent).toBe('Host RAM · Routed experts · 16.3GB @ 8k');
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

it('hatches an on-device n-gram table and offers to move it (to the SSD on unified memory)', async () => {
  const { summary } = await defaultResult();
  const stage = summary.memory.stages[0];
  let placed = 0;
  const table = {
    ...summary,
    memory: {
      ...summary.memory,
      kind: 'unified' as const,
      stages: [{ ...stage, lookupTableBytes: 0.4 * stage.weightBytes }],
      tiers: [],
    },
  };
  const { container } = render(
    <MemoryFit result={table} onPlaceTablesOnHost={() => (placed += 1)} />,
  );
  const legend = screen.getByTestId('memory-legend');
  expect(
    [...legend.querySelectorAll('li')].map((li) => li.firstChild?.nextSibling?.textContent),
  ).toEqual(['Weights', 'n-gram table', 'KV + state', 'Activations']);
  expect(container.querySelector('.capacity-bar .allocation-table')).toBeTruthy();
  // On unified memory host RAM is the device pool, so the table moves to the SSD.
  fireEvent.click(screen.getByRole('button', { name: 'Place on SSD' }));
  expect(placed).toBe(1);
  // Unified memory: host RAM is the device pool, so no host-RAM tier or line ever shows.
  expect(screen.queryAllByTestId('memory-tier')).toHaveLength(0);
  expect(screen.queryByTestId('memory-unified-host')).toBeNull();
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
