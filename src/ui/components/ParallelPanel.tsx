import { validDegrees, applyQuantOverride } from '../state';
import { Panel } from './Panel';
import { NumberField } from './NumberField';
import type { PanelProps as Props } from './types';

/** Help text per field: short sentences, each shown on its own line in the tooltip. */
const HELP = {
  tp: [
    'Splits every layer across GPUs.',
    'Each GPU holds a slice of the weights.',
    'Needs a fast link: GPUs sync after every layer.',
  ],
  pp: [
    'Splits the layers into stages, one per GPU.',
    'Only small activations cross between stages, so slow links are fine.',
    'A single sequence does not get faster.',
  ],
} as const;

function helpLines(lines: readonly string[]) {
  return lines.map((line) => (
    <span key={line} className="help-line">
      {line}
    </span>
  ));
}

export function ParallelPanel({ state, dispatch, loaded, built }: Props) {
  return (
    <Panel
      title="Parallelism"
      value={`TP ${state.parallel.tp}, PP ${state.parallel.pp}`}
      open
    >
      <div className="fields two">
        {(['tp', 'pp'] as const).map((field) => (
          <NumberField
            key={field}
            label={field.toUpperCase()}
            help={helpLines(HELP[field])}
            value={state.parallel[field]}
            onChange={(v) => dispatch({ type: 'setParallel', parallel: { [field]: v } })}
            issues={built.issues.filter((i) => i.field === field)}
            hint={
              loaded && built.hardware
                ? `Valid ${field.toUpperCase()}: ${validDegrees(field, loaded.model, applyQuantOverride(loaded.quant.quant, state.quant, loaded.quant.exl3), state.parallel, state.workload, built.hardware.topology).join(', ')}`
                : undefined
            }
          />
        ))}
      </div>
    </Panel>
  );
}
