import { validDegrees, applyQuantOverride } from '../state';
import { Panel } from './Panel';
import { NumberField } from './NumberField';
import type { PanelProps as Props } from './types';

export function ParallelPanel({ state, dispatch, loaded, built }: Props) {
  return (
    <Panel
      title="Parallelism"
      value={`TP ${state.parallel.tp}, PP ${state.parallel.pp}, DP ${state.parallel.dp}, EP ${state.parallel.ep}`}
      open
    >
      <div className="fields two">
        {(['tp', 'pp', 'dp', 'ep'] as const).map((field) => (
          <NumberField
            key={field}
            label={field.toUpperCase()}
            value={state.parallel[field]}
            onChange={(v) => dispatch({ type: 'setParallel', parallel: { [field]: v } })}
            issues={built.issues.filter((i) => i.field === field)}
            hint={
              loaded && built.hardware
                ? `Valid ${field.toUpperCase()}: ${validDegrees(field, loaded.model, applyQuantOverride(loaded.quant.quant, state.quant), state.parallel, state.workload, built.hardware.topology).join(', ')}`
                : undefined
            }
          />
        ))}
      </div>
    </Panel>
  );
}
