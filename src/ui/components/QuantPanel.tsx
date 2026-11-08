import { useState } from 'react';
import { Panel } from './Panel';
import { NumberField } from './NumberField';
import {
  EXL3_BPW_PRESETS,
  EXL3_CUSTOM_KEY,
  EXL3_MODEL_KEY,
  QUANT_DTYPE_CHOICES,
  effectiveBpw,
  exl3PickerKey,
  exl3PresetKey,
  hasQuantOverride,
  isValidBpw,
  isValidHeadBpw,
  isWeightFormat,
  resolvedDefault,
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

/** Listbox group heading of the EXL3 entries in the Weights picker. */
const EXL3_GROUP = 'EXL3';

/**
 * The picker options for one field: the default option (value '') labelled with what it resolves to, then every
 * dtype. For Weights, also the EXL3 group: one entry per bpw preset, the checkpoint's bpw when it is not a preset
 * (`model default`), and `EXL3 custom…`.
 */
function optionsFor(defaultLabel: string, exl3?: { checkpointBpw: number | undefined }): PickerOption[] {
  const opts: PickerOption[] = [
    { value: '', label: defaultLabel },
    ...QUANT_DTYPE_CHOICES.map((d) => ({ value: d, label: d })),
  ];
  if (!exl3) return opts;
  const exl3Bpw = exl3.checkpointBpw;
  const entries: PickerOption[] = EXL3_BPW_PRESETS.map((b) => ({
    value: exl3PresetKey(b),
    label: `EXL3 ${b.toFixed(1)}bpw`,
    group: EXL3_GROUP,
  }));
  if (exl3Bpw !== undefined && !EXL3_BPW_PRESETS.includes(exl3Bpw)) {
    entries.push({ value: EXL3_MODEL_KEY, label: `EXL3 ${exl3Bpw}bpw (model default)`, group: EXL3_GROUP });
  }
  entries.push({ value: EXL3_CUSTOM_KEY, label: 'EXL3 custom…', group: EXL3_GROUP });
  return [...opts, ...entries];
}

const BPW_HELP = (
  <>
    Bits per weight (bpw): the bits stored per weight, fractional values allowed.
    <br />
    EXL3 picks bpw per tensor, so a checkpoint’s bpw is an average over its tensors.
    <br />
    Weight bytes, and so memory-bound decode speed, scale with it.
  </>
);
const HEAD_BPW_HELP =
  'Bits per weight of the LM head (output projection), an integer 1–8 in EXL3; 16 keeps it unquantized fp16. ' +
  'The head is quantized separately from the layers (exllamav3 default 6).';

/** Warning for a bpw outside what EXL3 can store (the simulation clamps it). */
function rangeIssue(field: 'bpw' | 'headBpw', valid: boolean): FieldIssue[] {
  return valid ? [] : [{ field, level: 'warning', message: `Outside ${field === 'bpw' ? '1–8' : '1–8 or 16'}; simulated at the nearest limit.` }];
}

export function QuantPanel({ state, dispatch, loaded }: Props) {
  const { bpw, headBpw } = effectiveBpw(state.quant, loaded?.quant.exl3);
  const checkpointBpw = loaded?.quant.exl3?.bpw;
  const exl3 = state.quant.weights === 'exl3';
  // `EXL3 custom…` is a UI choice: the bpw field shows, whatever the bpw is (see exl3PickerKey).
  const [custom, setCustom] = useState(false);
  const exl3Key = exl3PickerKey(bpw, checkpointBpw, custom);

  const pickWeights = (v: string) => {
    if (v === EXL3_CUSTOM_KEY) {
      setCustom(true);
      dispatch({ type: 'setQuant', field: 'weights', dtype: 'exl3' });
      return;
    }
    if (v.startsWith('exl3:')) {
      setCustom(false);
      const value = v === EXL3_MODEL_KEY ? checkpointBpw : Number(v.slice('exl3:'.length));
      if (value !== undefined) dispatch({ type: 'setBpw', field: 'bpw', value });
      return;
    }
    setCustom(false);
    dispatch({ type: 'setQuant', field: 'weights', dtype: isWeightFormat(v) ? v : undefined });
  };

  return (
    <Panel
      title="Quantization"
      value={exl3 ? `exl3 ${bpw} bpw` : (state.quant.weights ?? loaded?.quant.scheme ?? 'Model default')}
      open
    >
      <div className="fields">
        {FIELDS.map((f) => {
          const resolved = loaded ? resolvedDefault(loaded.quant.quant, f.key) : undefined;
          const defaultLabel = resolved ? `Model default (${resolved.label})` : 'Model default';
          const help = (
            <>
              {f.help}
              {loaded && f.key === 'weights' && <> Model default: {loaded.quant.scheme}.</>}
              {resolved?.detail && <> Model default split: {resolved.detail}.</>}
            </>
          );
          if (f.key === 'weights') {
            return (
              <Picker
                key={f.key}
                label={f.label}
                value={exl3 ? exl3Key : (state.quant.weights ?? '')}
                options={optionsFor(defaultLabel, { checkpointBpw })}
                help={help}
                onChange={pickWeights}
              />
            );
          }
          return (
            <Picker
              key={f.key}
              label={f.label}
              value={state.quant[f.key] ?? ''}
              options={optionsFor(defaultLabel)}
              help={help}
              onChange={(v) =>
                dispatch({
                  type: 'setQuant',
                  field: f.key,
                  dtype: isWeightFormat(v) && v !== 'exl3' ? v : undefined,
                })
              }
            />
          );
        })}
        {exl3 && exl3Key === EXL3_CUSTOM_KEY && (
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
        )}
        {exl3 && (
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
