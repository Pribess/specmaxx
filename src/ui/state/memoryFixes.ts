/**
 * The memory card's one-click fixes for an app state (pure TypeScript, no React): {@link memorySuggestions} fed with
 * the sim as its probe. A candidate change is applied to the state with the same reducer the manual controls use (for
 * the decode pool through `inPool`), built and simulated, and its memory report decides whether it helps. Runs only for
 * a pool that does not fit, so the extra simulations cost nothing while the configuration fits.
 */

import { poolInput } from '../../sim/engine/disaggregation';
import { simulate } from '../../sim/engine/simulate';
import { reduce, poolView, type AppState, type PoolAction, type PoolId } from './appState';
import { buildDisaggregatedInput, buildSimInput, type LoadedModel } from './buildInput';
import type { DrafterStatus } from './deploymentInput';
import { memorySuggestions, type MemorySuggestions } from './memorySuggestions';
import { summarize, type MemorySummary, type ResultSummary } from './summary';

/**
 * The memory report `pool` of `state` would have after `actions` (run in that pool's view), or undefined when the
 * result is not a valid input. With disaggregation off, `pool` is `prefill` and the report is the single run's.
 * `actionPool` is the view the actions run in (default `pool`); the other pool's report after an action run in this
 * pool's view shows what a shared choice does to it.
 */
function probePool(
  state: AppState,
  loaded: LoadedModel,
  drafter: DrafterStatus | undefined,
  pool: PoolId,
  actions: PoolAction[],
  actionPool: PoolId = pool,
): MemorySummary | undefined {
  const next = actions.reduce<AppState>(
    (s, action) =>
      reduce(s, actionPool === 'decode' ? { type: 'inPool', pool: 'decode', action } : action),
    state,
  );
  try {
    if (next.disaggregation !== undefined) {
      const built = buildDisaggregatedInput(next, loaded, drafter);
      if (built?.kind !== 'ok') return undefined;
      const { shared } = built.input;
      const spec = pool === 'prefill' ? built.input.prefillPool : built.input.decodePool;
      const input = poolInput(shared, spec, pool);
      return summarize(simulate(input), input).memory;
    }
    const built = buildSimInput(next, loaded, drafter);
    if (built.kind !== 'ok') return undefined;
    return summarize(simulate(built.input), built.input, built.notes).memory;
  } catch {
    return undefined;
  }
}

/**
 * The one-click fixes of each pool whose memory does not fit. `results` are the reports on screen: the single run's
 * under `prefill`, or both pools' (disaggregation on). A pool that fits has no entry.
 */
export function stateMemorySuggestions(
  state: AppState,
  loaded: LoadedModel,
  drafter: DrafterStatus | undefined,
  results: Partial<Record<PoolId, ResultSummary>>,
): Partial<Record<PoolId, MemorySuggestions>> {
  const out: Partial<Record<PoolId, MemorySuggestions>> = {};
  for (const pool of ['prefill', 'decode'] as const) {
    const result = results[pool];
    if (result === undefined || result.memory.fits) continue;
    const other: PoolId = pool === 'prefill' ? 'decode' : 'prefill';
    const suggestions = memorySuggestions({
      model: loaded.model,
      memory: result.memory,
      offload: state.deployment.offload ?? {},
      host: poolView(state, pool).hw.host ?? {},
      sharedOffload: state.disaggregation !== undefined,
      // The offload is shared: a fix must not push the other pool over its memory.
      ...(state.disaggregation !== undefined && results[other] !== undefined
        ? {
            otherMemory: results[other].memory,
            probeOther: (actions: PoolAction[]) =>
              probePool(state, loaded, drafter, other, actions, pool),
          }
        : {}),
      probe: (actions) => probePool(state, loaded, drafter, pool, actions),
    });
    if (suggestions.suggestions.length > 0 || suggestions.notice !== undefined)
      out[pool] = suggestions;
  }
  return out;
}
