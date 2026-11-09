import { validDegrees, applyQuantOverride, effectiveParallel, hasMoe } from '../state';
import { Panel } from './Panel';
import { NumberField } from './NumberField';
import { helpLines } from './helpLines';
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
  ep: [
    'Spreads the MoE experts across GPUs.',
    'Each GPU holds a share of the experts, not a slice of each.',
    'Uses the TP GPUs, so EP must divide TP.',
  ],
} as const;

export function ParallelPanel({ state, dispatch, loaded, built }: Props) {
  // EP is offered only for a loaded MoE model; a dense model always runs with EP 1.
  const moe = loaded !== null && hasMoe(loaded.model);
  const fields = moe ? (['tp', 'pp', 'ep'] as const) : (['tp', 'pp'] as const);
  const value = `TP ${state.parallel.tp}, PP ${state.parallel.pp}${moe ? `, EP ${state.parallel.ep}` : ''}`;
  return (
    <Panel title="Parallelism" value={value} open>
      <div className={`fields ${moe ? 'three' : 'two'}`}>
        {fields.map((field) => (
          <NumberField
            key={field}
            label={field.toUpperCase()}
            help={helpLines(HELP[field])}
            value={state.parallel[field]}
            onChange={(v) => dispatch({ type: 'setParallel', parallel: { [field]: v } })}
            issues={built.issues.filter((i) => i.field === field)}
            hint={
              loaded && built.hardware
                ? `Valid ${field.toUpperCase()}: ${validDegrees(field, loaded.model, applyQuantOverride(loaded.quant.quant, state.quant, loaded.quant.exl3), effectiveParallel(state.parallel, loaded.model), state.workload, built.hardware.topology).join(', ')}`
                : undefined
            }
          />
        ))}
      </div>
    </Panel>
  );
}
