import { QUANT_DTYPE_CHOICES, isQuantDtype, hasQuantOverride } from '../state';
import { Panel } from './Panel';
import { Picker } from './Picker';
import type { PickerOption } from './fuzzy';
import type { PanelProps as Props } from './types';

/** The three quant fields, in display order, with their labels and definitions. */
const FIELDS = [
  {
    key: 'weights',
    label: 'Weights',
    help: 'Storage format of every linear-layer weight. Embeddings and the output head keep their own defaults.',
  },
  {
    key: 'activations',
    label: 'Activations',
    help: 'Format of the matrix-multiply inputs. A checkpoint’s own activation format stays unless Weights is overridden too.',
  },
  {
    key: 'kvCache',
    label: 'KV cache',
    help: 'Storage format of the attention key and value cache. Compressed KV caches follow it.',
  },
] as const;

/** The "Model default" option (value ''), then every storage dtype. */
const DEFAULT_OPTION: PickerOption = { value: '', label: 'Model default' };
const DTYPE_OPTIONS: PickerOption[] = [
  DEFAULT_OPTION,
  ...QUANT_DTYPE_CHOICES.map((d) => ({ value: d, label: d })),
];

export function QuantPanel({ state, dispatch, loaded }: Props) {
  const defaultScheme = loaded?.quant.scheme ?? 'loading';
  return (
    <Panel
      title="Quantization"
      value={state.quant.weights ?? loaded?.quant.scheme ?? 'Model default'}
      open
    >
      <div className="fields">
        {FIELDS.map((f) => (
          <Picker
            key={f.key}
            label={f.label}
            value={state.quant[f.key] ?? ''}
            options={DTYPE_OPTIONS}
            help={
              f.key === 'weights' ? (
                <>
                  {f.help} Model default: {defaultScheme}.
                </>
              ) : (
                f.help
              )
            }
            onChange={(v) =>
              dispatch({
                type: 'setQuant',
                field: f.key,
                dtype: isQuantDtype(v) ? v : undefined,
              })
            }
          />
        ))}
      </div>
      {hasQuantOverride(state.quant) && (
        <button className="quiet" onClick={() => dispatch({ type: 'resetQuant' })}>
          Reset to model default
        </button>
      )}
    </Panel>
  );
}
