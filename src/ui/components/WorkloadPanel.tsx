import { Panel } from './Panel';
import { NumberField } from './NumberField';
import type { PanelProps as Props } from './types';

export function WorkloadPanel({ state, dispatch, built }: Props) {
  return (
    <Panel
      title="Workload"
      value={`Batch ${state.workload.batch}, prompt ${state.workload.promptLen}`}
      open
    >
      <div className="fields">
        <NumberField
          label="Batch size"
          value={state.workload.batch}
          onChange={(batch) => dispatch({ type: 'setWorkload', workload: { batch } })}
          issues={built.issues.filter((i) => i.field === 'batch')}
        />
        <NumberField
          label="Prompt length"
          value={state.workload.promptLen}
          onChange={(promptLen) => dispatch({ type: 'setWorkload', workload: { promptLen } })}
          issues={built.issues.filter((i) => i.field === 'promptLen')}
        />
        <NumberField
          label="Decode context length"
          value={state.workload.decodeContextLen}
          onChange={(decodeContextLen) =>
            dispatch({ type: 'setWorkload', workload: { decodeContextLen } })
          }
          issues={built.issues.filter((i) => i.field === 'decodeContextLen')}
        />
      </div>
    </Panel>
  );
}
