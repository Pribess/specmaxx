import { useState } from 'react';
import { getHfToken, setHfToken, clearHfToken } from '../../hf/token';
import { countParams, activeParamsPerToken } from '../../sim/ir/params';
import { formatParams, type LoadedModel } from '../state';
import { Panel } from './Panel';

interface ModelPanelProps {
  id: string;
  onIdChange(id: string): void;
  onLoad(): void;
  onRetry(): void;
  onReference(): void;
  loaded: LoadedModel | null;
  loading: boolean;
  error?: string;
  gated: boolean;
  isHf: boolean;
}

export function ModelPanel({
  id,
  onIdChange,
  onLoad,
  onRetry,
  onReference,
  loaded,
  loading,
  error,
  gated,
  isHf,
}: ModelPanelProps) {
  const [token, setToken] = useState('');
  const [hasToken, setHasToken] = useState(() => !!getHfToken());
  return (
    <Panel title="Model" value={loaded?.label ?? id} open>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          onLoad();
        }}
      >
        <label className="field">
          <span>Hugging Face model ID</span>
          <div className="model-entry">
            <input
              value={id}
              placeholder="Qwen/Qwen3.8-27B"
              onChange={(event) => onIdChange(event.target.value)}
            />
            <button type="submit">Load</button>
          </div>
        </label>
      </form>
      <p className="hint">Paste a model ID, or keep the built-in 8B reference.</p>
      {loading && <p role="status">Loading model config…</p>}
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {gated && (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            setHfToken(token);
            setHasToken(!!token.trim());
            setToken('');
            onRetry();
          }}
        >
          <label className="field">
            <span>Hugging Face token</span>
            <input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(event) => setToken(event.target.value)}
            />
          </label>
          <p className="hint">
            Stored in this browser only. Sent only to Hugging Face; never included in a shared URL.
          </p>
          <button>Save token and retry</button>
        </form>
      )}
      {hasToken && (
        <button
          className="quiet"
          onClick={() => {
            clearHfToken();
            setHasToken(false);
          }}
        >
          Forget token
        </button>
      )}
      {loaded && (
        <div className="model-facts" data-testid="model-facts">
          <p>
            {loaded.hf
              ? `${loaded.hf.family}: ${loaded.hf.modelType ?? loaded.hf.architecture}`
              : 'Built-in dense reference'}
          </p>
          <p>
            {formatParams(countParams(loaded.model))} total,{' '}
            {formatParams(activeParamsPerToken(loaded.model))} active
            <br />
            {loaded.model.layers.length} layers
          </p>
          {loaded.hf?.excluded.parts.length ? (
            <p className="hint">Excluded: {loaded.hf.excluded.parts.join(', ')}</p>
          ) : null}
          {loaded.hf?.paramCheck.status === 'mismatch' && (
            <p className="error">{loaded.hf.paramCheck.note}</p>
          )}
        </div>
      )}
      {isHf && (
        <button className="quiet" onClick={onReference}>
          Use built-in reference
        </button>
      )}
    </Panel>
  );
}
