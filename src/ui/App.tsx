import { useEffect, useMemo, useReducer, useState } from 'react';
import { loadHfModel } from '../hf/load';
import { getHfToken } from '../hf/token';
import { ModelPanel } from './components/ModelPanel';
import {
  decodeState,
  encodeState,
  reduce,
  modelKey,
  BUNDLED_DEFAULT_MODEL,
  loadedFromHf,
  buildSimInput,
  formatRateAtContext,
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
export function App({
  loader = loadHfModel,
  runner: suppliedRunner,
}: {
  loader?: ModelLoader;
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
  const built = useMemo(() => buildSimInput(state, loaded), [state, loaded]);
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
          <WorkloadPanel {...props} />
          <AssumptionsPanel {...props} />
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
          />
          {result && (
            <>
              <div className="compact-results">
                Prefill {formatRateAtContext(result.prefill.tokensPerSec, state.workload.promptLen)}
                {' · '}
                <span>
                  Decode {formatRateAtContext(result.decode.tokensPerSec, state.workload.decodeContextLen)}
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
