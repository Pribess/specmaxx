import { QUANT_DTYPE_CHOICES, isQuantDtype, hasQuantOverride } from '../state';
import { Panel } from './Panel';
import type { PanelProps as Props } from './types';

export function QuantPanel({ state, dispatch, loaded }: Props) {
  return (
    <Panel
      title="Quantization"
      value={state.quant.weights ?? loaded?.quant.scheme ?? 'Model default'}
      open
    >
      <div className="fields three">
        {(['weights', 'activations', 'kvCache'] as const).map((field, i) => (
          <label className="field" key={field}>
            <span>{['Weights', 'Activations', 'KV cache'][i]}</span>
            <select
              value={state.quant[field] ?? ''}
              onChange={(e) =>
                dispatch({
                  type: 'setQuant',
                  field,
                  dtype: isQuantDtype(e.target.value) ? e.target.value : undefined,
                })
              }
            >
              <option value="">Model default</option>
              {QUANT_DTYPE_CHOICES.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <p className="hint">
        Model default: {loaded?.quant.scheme ?? 'loading'}. Weight overrides replace all
        linear-module formats; embeddings and the output head keep their defaults.
      </p>
      {hasQuantOverride(state.quant) && (
        <button className="quiet" onClick={() => dispatch({ type: 'resetQuant' })}>
          Reset to model default
        </button>
      )}
    </Panel>
  );
}
