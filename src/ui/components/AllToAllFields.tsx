import type { A2aFormat } from '../../sim/ir/a2aFormat';
import { A2A_FORMATS, isA2aFormat } from '../../sim/ir/a2aFormat';
import { hasMoe } from '../state';
import { NumberField } from './NumberField';
import { Picker } from './Picker';
import { helpLines } from './helpLines';
import type { PanelProps } from './types';

/** Picker label of each wire format. */
const FORMAT_LABEL: Record<A2aFormat, string> = {
  bf16: 'BF16',
  fp8_e4m3: 'FP8',
  fp8_block128: 'FP8 + 128-value scales',
  mxfp8: 'MXFP8',
  nvfp4: 'NVFP4',
  mxfp4: 'MXFP4',
};

const DISPATCH_HELP = helpLines([
  'Number format of the tokens sent to the GPUs that hold their experts.',
  'Smaller formats send fewer bytes; the block scales they carry are counted.',
  'Default follows the activation format. DeepSeek uses FP8, GB200 stacks use NVFP4.',
]);

const COMBINE_HELP = helpLines([
  'Number format of the expert outputs sent back.',
  'Default follows the activation format, at least 16-bit. DeepSeek uses BF16.',
  'Smaller formats send fewer bytes; the block scales they carry are counted.',
]);

const EFFICIENCY_HELP = helpLines([
  'Share of the link speed an all-to-all actually reaches.',
  'Stands for the communication backend (DeepEP and similar), which rarely hits peak.',
  'Raises the transfer time only; 100% is the link limit.',
]);

/**
 * Hardware → Advanced → "Expert all-to-all": the wire format of the MoE dispatch and combine all-to-alls
 * (`ParallelConfig.dispatchFormat` / `combineFormat`; default follows the activation dtype) and the achieved fraction
 * of link bandwidth every all-to-all reaches (`allToAllEfficiency`; default 100%, no backend choice). Shown only for a
 * MoE model whose EP spans DP replicas (EP > TP), the one layout with an all-to-all.
 */
export function AllToAllFields({ state, dispatch, loaded, built }: PanelProps) {
  const { tp, ep } = state.parallel;
  if (loaded === null || !hasMoe(loaded.model) || !(ep > tp)) return null;
  const eta = state.parallel.allToAllEfficiency ?? 1;
  const options = (defaultLabel: string) => [
    { value: '', label: defaultLabel },
    ...A2A_FORMATS.map((f) => ({ value: f, label: FORMAT_LABEL[f] })),
  ];
  return (
    <fieldset className="unlock advanced-fields" data-testid="all-to-all-advanced">
      <legend>Expert all-to-all</legend>
      <Picker
        label="Dispatch format"
        value={state.parallel.dispatchFormat ?? ''}
        options={options('Same as activations')}
        help={DISPATCH_HELP}
        onChange={(v) =>
          dispatch({
            type: 'setParallel',
            parallel: { dispatchFormat: isA2aFormat(v) ? v : undefined },
          })
        }
      />
      <Picker
        label="Combine format"
        value={state.parallel.combineFormat ?? ''}
        options={options('Activations, 16-bit or wider')}
        help={COMBINE_HELP}
        onChange={(v) =>
          dispatch({
            type: 'setParallel',
            parallel: { combineFormat: isA2aFormat(v) ? v : undefined },
          })
        }
      />
      <NumberField
        label="All-to-all efficiency %"
        value={Math.round(eta * 1e6) / 1e4}
        min={0}
        max={100}
        step="any"
        onChange={(v) =>
          dispatch({ type: 'setParallel', parallel: { allToAllEfficiency: v / 100 } })
        }
        help={EFFICIENCY_HELP}
        issues={built.issues.filter((i) => i.field === 'allToAllEfficiency')}
      />
    </fieldset>
  );
}
