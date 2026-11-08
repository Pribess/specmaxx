import { Panel } from './Panel';
import { NumberField } from './NumberField';
import {
  QUANT_DTYPE_CHOICES,
  effectiveBpw,
  hasQuantOverride,
  isValidBpw,
  isValidHeadBpw,
  isWeightFormat,
  type FieldIssue,
} from '../state';
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
/** The Weights picker also offers EXL3, which has its own bits-per-weight inputs. */
const WEIGHT_OPTIONS: PickerOption[] = [...DTYPE_OPTIONS, { value: 'exl3', label: 'exl3' }];

const BPW_HELP =
  'Bits per weight (bpw): the average number of bits stored per weight, including fractional values such as 2.5. ' +
  'EXL3 (ExLlamaV3) stores each 16×16 weight tile as a trellis code of 16·K words, with K from 1 to 8 per tensor, ' +
  'so a checkpoint’s bpw is the average over its tensors. Weight bytes, and so memory-bound decode speed, scale with it. ' +
  'Two fp16 vectors per matrix add a small overhead. The matmul runs in fp16.';
const HEAD_BPW_HELP =
  'Bits per weight of the LM head (output projection), an integer 1–8 in EXL3; 16 keeps it unquantized fp16. ' +
  'The head is quantized separately from the layers (exllamav3 default 6).';

/** Warning for a bpw outside what EXL3 can store (the simulation clamps it). */
function rangeIssue(field: 'bpw' | 'headBpw', valid: boolean): FieldIssue[] {
  return valid ? [] : [{ field, level: 'warning', message: `Outside ${field === 'bpw' ? '1–8' : '1–8 or 16'}; simulated at the nearest limit.` }];
}

export function QuantPanel({ state, dispatch, loaded }: Props) {
  const defaultScheme = loaded?.quant.scheme ?? 'loading';
  const { bpw, headBpw } = effectiveBpw(state.quant, loaded?.quant.exl3);
  return (
    <Panel
      title="Quantization"
      value={state.quant.weights === 'exl3' ? `exl3 ${bpw} bpw` : (state.quant.weights ?? loaded?.quant.scheme ?? 'Model default')}
      open
    >
      <div className="fields">
        {FIELDS.map((f) => (
          <Picker
            key={f.key}
            label={f.label}
            value={state.quant[f.key] ?? ''}
            options={f.key === 'weights' ? WEIGHT_OPTIONS : DTYPE_OPTIONS}
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
                dtype: isWeightFormat(v) && (f.key === 'weights' || v !== 'exl3') ? v : undefined,
              })
            }
          />
        ))}
        {state.quant.weights === 'exl3' && (
          <>
            <NumberField
              label="Bits per weight"
              value={bpw}
              step="0.05"
              min={1}
              max={8}
              onChange={(v) => dispatch({ type: 'setBpw', field: 'bpw', value: v })}
              issues={rangeIssue('bpw', isValidBpw(bpw))}
              help={BPW_HELP}
            />
            <NumberField
              label="Head bits per weight"
              value={headBpw}
              step="1"
              min={1}
              max={16}
              onChange={(v) => dispatch({ type: 'setBpw', field: 'headBpw', value: v })}
              issues={rangeIssue('headBpw', isValidHeadBpw(headBpw))}
              help={HEAD_BPW_HELP}
            />
          </>
        )}
      </div>
      {hasQuantOverride(state.quant) && (
        <button className="quiet" onClick={() => dispatch({ type: 'resetQuant' })}>
          Reset to model default
        </button>
      )}
    </Panel>
  );
}
