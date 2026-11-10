import { describe, expect, it } from 'vitest';
import { simulateDisaggregated } from '../../sim/engine/disaggregation';
import { simulate } from '../../sim/engine/simulate';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../../sim/models/hf/index';
import { DEFAULT_STATE, reduce, type AppAction, type AppState, type PoolId } from './appState';
import { buildDisaggregatedInput, buildSimInput, loadedFromHf } from './buildInput';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import { stateMemorySuggestions } from './memoryFixes';
import { summarize, summarizeDisaggregated, type ResultSummary } from './summary';

const apply = (s: AppState, ...actions: AppAction[]) => actions.reduce(reduce, s);

/** Bundled Qwen3.8-27B (bf16, ≈ 54 GB) on one RTX PRO 6000 (96 GB), batch 8, 8k prompt, 128k decode context. */
const LONG_DECODE = apply(
  DEFAULT_STATE,
  { type: 'setDevice', deviceId: 'rtx-pro-6000-ws' },
  { type: 'setWorkload', workload: { batch: 8, promptLen: 8192, decodeContextLen: 131072 } },
);

/** The reports on screen for `state` (both pools with disaggregation on), as the app gets them from the runner. */
function reports(state: AppState): Partial<Record<PoolId, ResultSummary>> {
  const pd = buildDisaggregatedInput(state, BUNDLED_DEFAULT_MODEL);
  if (pd !== undefined) {
    if (pd.kind !== 'ok') throw new Error(JSON.stringify(pd.issues));
    const s = summarizeDisaggregated(simulateDisaggregated(pd.input), pd.input);
    return { prefill: s.prefillPool, decode: s.decodePool };
  }
  const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  return { prefill: summarize(simulate(built.input), built.input, built.notes) };
}

describe('stateMemorySuggestions', () => {
  it('a long decode context that does not fit: tiering the KV cache to host RAM makes it fit', () => {
    const before = reports(LONG_DECODE);
    expect(before.prefill?.memory.fits).toBe(false);
    const fixes = stateMemorySuggestions(LONG_DECODE, BUNDLED_DEFAULT_MODEL, undefined, before);
    const kv = fixes.prefill?.suggestions.find((s) => s.label === 'KV cache → host RAM');
    expect(kv?.fits).toBe(true);
    // The same actions the manual KV controls dispatch: the result fits and the host RAM bar shows the KV cache.
    const after = reports(apply(LONG_DECODE, ...kv!.actions));
    expect(after.prefill?.memory.fits).toBe(true);
    expect(after.prefill?.memory.tiers?.[0].tier).toBe('host_ram');
    expect(after.prefill?.memory.tiers?.[0].breakdown.kvCache).toBeGreaterThan(0);
    // A fix that fits hides the ones that only shrink the overflow (token embeddings alone do not fit).
    expect(fixes.prefill!.suggestions.map((s) => s.label)).toEqual(['KV cache → host RAM']);
  });

  it('a configuration that fits has no suggestions', () => {
    const state = apply(LONG_DECODE, {
      type: 'setWorkload',
      workload: { decodeContextLen: 8192 },
    });
    const r = reports(state);
    expect(r.prefill?.memory.fits).toBe(true);
    expect(stateMemorySuggestions(state, BUNDLED_DEFAULT_MODEL, undefined, r)).toEqual({});
  });

  it('with disaggregation on, only the pool that does not fit gets suggestions, and they apply to that pool', () => {
    const pd = apply(LONG_DECODE, { type: 'setDisaggregation', on: true });
    const r = reports(pd);
    // The prefill pool holds only the 8k prompt: it fits; the decode pool at 128k does not.
    expect(r.prefill?.memory.fits).toBe(true);
    expect(r.decode?.memory.fits).toBe(false);
    const fixes = stateMemorySuggestions(pd, BUNDLED_DEFAULT_MODEL, undefined, r);
    expect(fixes.prefill).toBeUndefined();
    const kv = fixes.decode?.suggestions.find((s) => s.label === 'KV cache → host RAM');
    expect(kv?.fits).toBe(true);
    const after = reports(
      apply(
        pd,
        ...kv!.actions.map((action): AppAction => ({ type: 'inPool', pool: 'decode', action })),
      ),
    );
    expect(after.decode?.memory.fits).toBe(true);
  });
});

describe('stateMemorySuggestions with host RAM that is too small', () => {
  // Qwen3.8-Flash-Next (MoE + n-gram tables) with its experts in host RAM on 96 GB of DDR5.
  const id = 'Qwen/Qwen3.8-Flash-Next';
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${r.kind}`);
  const loaded = loadedFromHf({ kind: 'hf', id }, r);
  const state = apply(
    DEFAULT_STATE,
    { type: 'setModel', model: { kind: 'hf', id } },
    { type: 'setDevice', deviceId: 'rtx-pro-6000-ws' },
    { type: 'setQuant', field: 'weights', dtype: 'fp8_e4m3' },
    { type: 'setHost', host: { ramId: 'ddr5-5600-2ch' } },
    {
      type: 'setOffload',
      offload: { experts: { tier: 'host_ram' }, tables: { tier: 'host_ram' } },
    },
  );

  it('offers a bigger Host RAM preset and the SSD; the preset fits when applied', () => {
    const built = buildSimInput(state, loaded);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const summary = summarize(simulate(built.input), built.input, built.notes);
    expect(summary.memory.tiers?.find((t) => t.tier === 'host_ram')?.fits).toBe(false);
    const fixes = stateMemorySuggestions(state, loaded, undefined, { prefill: summary });
    const labels = fixes.prefill?.suggestions.map((s) => s.label) ?? [];
    expect(labels[0]).toMatch(/^Host RAM → \d+ GB$/);
    expect(labels.some((l) => l.endsWith('→ SSD'))).toBe(true);
    const size = fixes.prefill!.suggestions[0];
    const next = apply(state, ...size.actions);
    const nextBuilt = buildSimInput(next, loaded);
    if (nextBuilt.kind !== 'ok') throw new Error(JSON.stringify(nextBuilt.issues));
    const after = summarize(simulate(nextBuilt.input), nextBuilt.input, nextBuilt.notes);
    expect(after.memory.tiers?.find((t) => t.tier === 'host_ram')?.fits).toBe(true);
  });
});
