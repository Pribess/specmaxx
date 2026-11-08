import { useEffect, useMemo, useReducer, useState } from 'react';
import { loadHfModel } from '../hf/load';
import { getHfToken } from '../hf/token';
import { drafterLoadMessage, loadDrafter } from '../hf/loadDrafter';
import { ModelPanel } from './components/ModelPanel';
import { ScrollArea } from './components/ScrollArea';
import {
  decodeState,
  encodeState,
  reduce,
  modelKey,
  BUNDLED_DEFAULT_MODEL,
  loadedFromHf,
  buildSimInput,
  formatRateAtContext,
  drafterSource,
  type DrafterStatus,
  type LoadedModel,
} from './state';
import { workerRunner, type SimRunner, type RunResult } from './sim/runner';
import {
  QuantPanel,
  HardwarePanel,
  ParallelPanel,
  WorkloadPanel,
  AssumptionsPanel,
} from './Panels';
import { ResultHero, BreakdownTable, BatchSweepChart } from './Results';
import '../styles/tokens.css';
import './panels.css';
import './app.css';

export type ModelLoader = typeof loadHfModel;
export type DrafterLoader = typeof loadDrafter;
export function App({
  loader = loadHfModel,
  drafterLoader = loadDrafter,
  runner: suppliedRunner,
}: {
  loader?: ModelLoader;
  /** Loads a block drafter's config.json (speculative decoding); injectable for tests. */
  drafterLoader?: DrafterLoader;
  runner?: SimRunner;
}) {
  const initial = useMemo(() => decodeState(location.search), []);
  const [state, dispatch] = useReducer(reduce, initial.state);
  const [dropped, setDropped] = useState(initial.dropped);
  const [id, setId] = useState(initial.state.model.id);
  const [attempt, setAttempt] = useState(0);
  const [modelState, setModelState] = useState<{
    key: string;
    loaded?: LoadedModel;
    error?: string;
    gated?: boolean;
  }>({ key: BUNDLED_DEFAULT_MODEL.key, loaded: BUNDLED_DEFAULT_MODEL });
  const [simulation, setSimulation] = useState<{ key: string; result?: RunResult; error?: string }>(
    { key: '' },
  );
  const [dim, setDim] = useState(false);
  const runner = useMemo(() => suppliedRunner ?? workerRunner(), [suppliedRunner]);
  const choiceKey = modelKey(state.model);
  const isBundled = choiceKey === BUNDLED_DEFAULT_MODEL.key;
  // The bundled default model is used until (and unless) a fresher copy loads; its refresh never errors.
  const loaded =
    modelState.key === choiceKey
      ? (modelState.loaded ?? (isBundled ? BUNDLED_DEFAULT_MODEL : null))
      : isBundled
        ? BUNDLED_DEFAULT_MODEL
        : null;
  const loading =
    !isBundled && (modelState.key !== choiceKey || (!modelState.loaded && !modelState.error));
  // The drafter repo a block spec-dec method needs (none for MTP, the checkpoint's own or the hypothetical drafter).
  const source = loaded ? drafterSource(state.deployment.specDec, loaded.model, state.model.id) : undefined;
  const drafterRepo = source?.kind === 'repo' ? source.repoId : undefined;
  const [drafterState, setDrafterState] = useState<DrafterStatus | undefined>(undefined);
  const drafter = useMemo<DrafterStatus | undefined>(
    () =>
      drafterRepo === undefined
        ? undefined
        : drafterState?.repoId === drafterRepo
          ? drafterState
          : { kind: 'loading', repoId: drafterRepo },
    [drafterRepo, drafterState],
  );
  useEffect(() => {
    if (drafterRepo === undefined) return;
    let current = true;
    const fail = (message: string) => {
      if (current) setDrafterState({ kind: 'error', repoId: drafterRepo, message });
    };
    drafterLoader(drafterRepo, { token: getHfToken() })
      .then((result) => {
        if (!current) return;
        if (result.kind === 'ok')
          setDrafterState({ kind: 'ok', repoId: drafterRepo, spec: result.spec, notes: result.notes });
        else fail(drafterLoadMessage(result));
      })
      .catch(() => fail('Drafter load failed. Check your connection.'));
    return () => {
      current = false;
    };
  }, [drafterRepo, drafterLoader]);
  const built = useMemo(() => buildSimInput(state, loaded, drafter), [state, loaded, drafter]);
  const query = encodeState(state);
  useEffect(() => {
    const timer = setTimeout(() => {
      const search = query ? `?${query}` : '';
      history.replaceState(null, '', `${location.pathname}${search}${location.hash}`);
      try {
        sessionStorage.setItem('specmaxx:last-query', search);
      } catch {
        /* storage may be disabled */
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [query]);
  useEffect(() => {
    const restore = () => {
      const parsed = decodeState(location.search);
      dispatch({ type: 'replace', state: parsed.state });
      setDropped(parsed.dropped);
      setId(parsed.state.model.id);
    };
    addEventListener('popstate', restore);
    return () => removeEventListener('popstate', restore);
  }, []);
  useEffect(() => {
    const choice = state.model;
    let current = true;
    loader(choice.id, { revision: choice.revision, token: getHfToken() })
      .then((result) => {
        if (!current) return;
        if (result.kind === 'ok')
          setModelState({ key: modelKey(choice), loaded: loadedFromHf(choice, result) });
        else {
          const error =
            result.kind === 'unsupported'
              ? `Unsupported architecture: ${result.modelType ?? result.architecture ?? 'unknown'}. Supported: ${result.supported.join(', ')}. ${result.reason ?? ''}`
              : result.kind === 'invalid_config'
                ? result.errors.join('; ')
                : result.message;
          if (modelKey(choice) === BUNDLED_DEFAULT_MODEL.key)
            setModelState({ key: modelKey(choice), loaded: BUNDLED_DEFAULT_MODEL });
          else
            setModelState({
              key: modelKey(choice),
              error,
              gated: result.kind === 'unauthorized' || result.kind === 'forbidden',
            });
        }
      })
      .catch(() => {
        if (!current) return;
        if (modelKey(choice) === BUNDLED_DEFAULT_MODEL.key)
          setModelState({ key: modelKey(choice), loaded: BUNDLED_DEFAULT_MODEL });
        else
          setModelState({
            key: modelKey(choice),
            error: 'Model load failed. Check your connection and try Load again.',
          });
      });
    return () => {
      current = false;
    };
  }, [state.model, loader, attempt]);
  useEffect(() => {
    if (built.kind !== 'ok') return;
    let current = true;
    const key = query;
    const timer = setTimeout(() => {
      if (current) setDim(true);
    }, 150);
    runner
      .run(built.input, built.notes)
      .then((result) => {
        if (current) {
          setSimulation({ key, result });
          setDim(false);
        }
      })
      .catch((error: unknown) => {
        if (current) {
          setSimulation({ key, error: error instanceof Error ? error.message : String(error) });
          setDim(false);
        }
      });
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [built, query, runner]);
  const titleModel = loaded?.label ?? state.model.id;
  const titleResult = built.kind === 'ok' ? simulation.result?.summary : undefined;
  useEffect(() => {
    document.title = titleResult
      ? `${titleModel} · ${formatRateAtContext(titleResult.prefill.tokensPerSec, state.workload.promptLen)} prefill · ${formatRateAtContext(titleResult.decode.tokensPerSec, state.workload.decodeContextLen)} decode`
      : `${titleModel} · specmaxx`;
  }, [titleModel, titleResult, state.workload.promptLen, state.workload.decodeContextLen]);
  useEffect(() => () => runner.dispose?.(), [runner]);
  const props = { state, dispatch, loaded, built };
  // The memory card's n-gram table segment: one click places the tables on host RAM (the SSD on unified memory).
  const placeTablesOnHost = () => {
    const unified = built.hardware?.device.device.memory.kind === 'unified';
    dispatch({
      type: 'setOffload',
      offload: { ...state.deployment.offload, tables: { tier: unified ? 'nvme' : 'host_ram' } },
    });
  };
  const result = built.kind === 'ok' ? simulation.result?.summary : undefined;
  const error = modelState.key === choiceKey ? modelState.error : undefined;
  function load(typed?: string) {
    const nextId = (typed ?? id).trim();
    if (!nextId) return;
    const unchanged = state.model.id === nextId;
    const choice = unchanged ? state.model : { kind: 'hf' as const, id: nextId };
    if (!unchanged) dispatch({ type: 'setModel', model: choice });
    setModelState({ key: modelKey(choice) });
    setAttempt((n) => n + 1);
  }
  return (
    <>
      <header className="app-header">
        <a className="wordmark" href="./">
          specmaxx
          <span className="roof-mark" aria-hidden="true" />
        </a>
        <span className="tagline">Theoretical ceilings for your rig</span>
        <a href="./about.html">How it works</a>
      </header>
      <main className="workspace">
        <div className="input-band">
          <ScrollArea className="input-band-scroll">
            <ModelPanel
              id={id}
              onIdChange={setId}
              onLoad={load}
              onRetry={() => {
                setModelState({ key: choiceKey });
                setAttempt((n) => n + 1);
              }}
              loaded={loaded}
              loading={loading}
              error={error}
              gated={modelState.key === choiceKey && !!modelState.gated}
            />
            <QuantPanel {...props} />
            <HardwarePanel {...props} />
            <ParallelPanel {...props} />
            <WorkloadPanel {...props} drafter={drafter} />
            <AssumptionsPanel {...props} />
          </ScrollArea>
        </div>
        <div className="output">
          {dropped.length > 0 && (
            <aside className="notice">
              Ignored invalid URL values: {dropped.join(', ')}.{' '}
              <button onClick={() => setDropped([])}>Dismiss</button>
            </aside>
          )}
          {built.issues.length > 0 && (
            <aside className="notice" role="alert">
              <p>
                {built.kind === 'issues'
                  ? 'No estimate is computed until the input errors are resolved.'
                  : 'Input notes'}
              </p>
              <ul>
                {built.issues.map((issue, index) => (
                  <li className={issue.level === 'error' ? 'error' : ''} key={index}>
                    {issue.message}
                  </li>
                ))}
              </ul>
            </aside>
          )}
          {simulation.error && simulation.key === query && (
            <p role="alert" className="error">
              {simulation.error}
            </p>
          )}
          <ResultHero
            state={state}
            loaded={loaded}
            built={built}
            result={result}
            pending={dim && simulation.key !== query}
            onPlaceTablesOnHost={placeTablesOnHost}
          />
          {result && (
            <>
              <div className="compact-results">
                Prefill {formatRateAtContext(result.prefill.tokensPerSec, state.workload.promptLen)}
                {' · '}
                <span>
                  Decode {formatRateAtContext(result.decode.tokensPerSec, state.workload.decodeContextLen)}
                  {result.decode.speculative && ` (spec-dec ×${result.decode.speculative.speedup.toFixed(2)})`}
                </span>
              </div>
              <div className="detail-results">
                <BreakdownTable result={result} />
                <BatchSweepChart points={simulation.result!.sweep} />
                {result.warnings.length > 0 && (
                  <details className="notes">
                    <summary>Notes and approximations ({result.warnings.length})</summary>
                    <ul>
                      {result.warnings.map((w, n) => (
                        <li key={n}>{w}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </>
  );
}
