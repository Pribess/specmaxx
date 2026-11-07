import { useEffect, useMemo, useReducer, useState } from 'react';
import { loadHfModel } from '../hf/load';
import { getHfToken } from '../hf/token';
import { ModelPanel } from './components/ModelPanel';
import {
  decodeState,
  encodeState,
  reduce,
  modelKey,
  REFERENCE_MODEL,
  loadedFromHf,
  buildSimInput,
  formatTokensPerSecCompact,
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
  const [id, setId] = useState(initial.state.model.kind === 'hf' ? initial.state.model.id : '');
  const [attempt, setAttempt] = useState(0);
  const [modelState, setModelState] = useState<{
    key: string;
    loaded?: LoadedModel;
    error?: string;
    gated?: boolean;
  }>({ key: 'reference', loaded: REFERENCE_MODEL });
  const [simulation, setSimulation] = useState<{ key: string; result?: RunResult; error?: string }>(
    { key: '' },
  );
  const [dim, setDim] = useState(false);
  const runner = useMemo(() => suppliedRunner ?? workerRunner(), [suppliedRunner]);
  const choiceKey = modelKey(state.model);
  const loaded =
    choiceKey === 'reference'
      ? REFERENCE_MODEL
      : modelState.key === choiceKey
        ? (modelState.loaded ?? null)
        : null;
  const loading =
    choiceKey !== 'reference' &&
    (modelState.key !== choiceKey || (!modelState.loaded && !modelState.error));
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
      setId(parsed.state.model.kind === 'hf' ? parsed.state.model.id : '');
    };
    addEventListener('popstate', restore);
    return () => removeEventListener('popstate', restore);
  }, []);
  useEffect(() => {
    if (state.model.kind !== 'hf') return;
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
          setModelState({
            key: modelKey(choice),
            error,
            gated: result.kind === 'unauthorized' || result.kind === 'forbidden',
          });
        }
      })
      .catch(() => {
        if (current)
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
  useEffect(() => () => runner.dispose?.(), [runner]);
  const props = { state, dispatch, loaded, built };
  const result = built.kind === 'ok' ? simulation.result?.summary : undefined;
  const error = modelState.key === choiceKey ? modelState.error : undefined;
  function load() {
    const nextId = id.trim();
    if (!nextId) return;
    const unchanged = state.model.kind === 'hf' && state.model.id === nextId;
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
            onReference={() => {
              dispatch({ type: 'setModel', model: { kind: 'reference' } });
              setId('');
            }}
            loaded={loaded}
            loading={loading}
            error={error}
            gated={modelState.key === choiceKey && !!modelState.gated}
            isHf={state.model.kind === 'hf'}
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
                Prefill {formatTokensPerSecCompact(result.prefill.tokensPerSec)}{' '}
                <span>Decode {formatTokensPerSecCompact(result.decode.tokensPerSec)} tok/s</span>
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
