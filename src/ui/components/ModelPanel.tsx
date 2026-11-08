import { useState } from 'react';
import { getHfToken, setHfToken, clearHfToken } from '../../hf/token';
import { countParams, activeParamsPerToken } from '../../sim/ir/params';
import { formatParams, MODEL_SHORTLIST, type LoadedModel } from '../state';
import { Panel } from './Panel';
import { Combobox } from './Combobox';
import { HelpIcon } from './HelpIcon';

interface ModelPanelProps {
  id: string;
  onIdChange(id: string): void;
  /** Load `typedId` (a committed pick or Enter) or, when absent, the text in the field. */
  onLoad(typedId?: string): void;
  onRetry(): void;
  loaded: LoadedModel | null;
  loading: boolean;
  error?: string;
  gated: boolean;
}

export function ModelPanel({
  id,
  onIdChange,
  onLoad,
  onRetry,
  loaded,
  loading,
  error,
  gated,
}: ModelPanelProps) {
  const [token, setToken] = useState('');
  const [hasToken, setHasToken] = useState(() => !!getHfToken());
  return (
    <Panel title="Model" value={loaded?.label ?? id} open>
      <div className="model-entry">
        <Combobox
          label="Hugging Face model ID"
          value={id}
          onValueChange={onIdChange}
          onCommit={(value) => {
            onIdChange(value);
            onLoad(value);
          }}
          options={MODEL_SHORTLIST.map((m) => ({ value: m.id, label: m.name, description: m.id }))}
          placeholder="Qwen/Qwen3.8-27B"
          help="Pick a popular model or paste any Hugging Face repo ID (owner/name), then press Enter or Load. Gated models need a token."
        />
        <button type="button" onClick={() => onLoad()}>
          Load
        </button>
      </div>
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
          <HelpIcon term="Hugging Face token">
            Stored in this browser only. Sent only to Hugging Face; never included in a shared URL.
          </HelpIcon>
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
              : loaded.label}
          </p>
          <p>
            {formatParams(countParams(loaded.model))} total,{' '}
            {formatParams(activeParamsPerToken(loaded.model))} active, {loaded.model.layers.length}{' '}
            layers
          </p>
          {loaded.hf?.excluded.parts.length ? (
            <p className="hint">Excluded: {loaded.hf.excluded.parts.join(', ')}</p>
          ) : null}
          {loaded.hf?.paramCheck.status === 'mismatch' && (
            <p className="error">{loaded.hf.paramCheck.note}</p>
          )}
        </div>
      )}
    </Panel>
  );
}
