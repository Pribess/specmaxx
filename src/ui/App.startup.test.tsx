// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import { App, type ModelLoader } from './App';
import { inlineRunner } from './sim/runner';
import { mapHfModel } from '../sim/models/hf';
import { fixtureRepoFiles } from '../sim/models/__fixtures__/fixtures.testutil';
import { stateMemorySuggestions } from './state/memoryFixes';

vi.mock('./state/memoryFixes', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./state/memoryFixes')>();
  return { ...actual, stateMemorySuggestions: vi.fn(actual.stateMemorySuggestions) };
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

for (const changed of [false, true]) {
  it(
    changed
      ? 'recalculates a changed HF refresh'
      : 'skips both simulation and memory probing for an identical HF refresh',
    async () => {
      history.replaceState(null, '', '/');
      let finish!: (result: Awaited<ReturnType<ModelLoader>>) => void;
      const loader: ModelLoader = () =>
        new Promise((resolve) => {
          finish = resolve;
        });
      const run = vi.fn(inlineRunner.run);
      render(<App loader={loader} runner={{ run }} />);
      expect(screen.getByTestId('result-skeleton').textContent).toBe('');
      await screen.findByTestId('decode-tps');
      expect(screen.queryByTestId('result-skeleton')).toBeNull();
      expect(run).toHaveBeenCalledTimes(1);
      const probes = vi.mocked(stateMemorySuggestions).mock.calls.length;
      expect(probes).toBe(1);
      const result = mapHfModel(fixtureRepoFiles('Qwen/Qwen3.8-27B'));
      if (result.kind !== 'ok') throw new Error('fixture failed');
      if (changed) result.model = { ...result.model, name: 'Updated model' };
      await act(async () => {
        finish(result);
      });
      await waitFor(() => expect(run).toHaveBeenCalledTimes(changed ? 2 : 1));
      if (!changed) expect(stateMemorySuggestions).toHaveBeenCalledTimes(probes);
    },
  );
}
