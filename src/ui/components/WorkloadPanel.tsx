import { Panel } from './Panel';
import { NumberField } from './NumberField';
import { formatContextTokens } from '../state';
import type { PanelProps as Props } from './types';

const K_NOTE = '1k = 1024 tokens, so 8k = 8192.';

export function WorkloadPanel({ state, dispatch, built }: Props) {
  const { batch, promptLen, decodeContextLen } = state.workload;
  return (
    <Panel
      title="Workload"
      value={`Batch ${batch}, prefill ${formatContextTokens(promptLen)}, decode @ ${formatContextTokens(decodeContextLen)}`}
      open
    >
      <div className="fields">
        <NumberField
          label="Batch size"
          value={batch}
          onChange={(v) => dispatch({ type: 'setWorkload', workload: { batch: v } })}
          issues={built.issues.filter((i) => i.field === 'batch')}
          help="The number of independent sequences processed together."
        />
        <NumberField
          label="Prefill length (tokens)"
          value={promptLen}
          onChange={(v) => dispatch({ type: 'setWorkload', workload: { promptLen: v } })}
          issues={built.issues.filter((i) => i.field === 'promptLen')}
          help={`Prefill tok/s @ N = N ÷ the time to process an N-token prompt from an empty KV cache. ${K_NOTE}`}
        />
        <NumberField
          label="Decode context (tokens)"
          value={decodeContextLen}
          onChange={(v) => dispatch({ type: 'setWorkload', workload: { decodeContextLen: v } })}
          issues={built.issues.filter((i) => i.field === 'decodeContextLen')}
          help={`Decode tok/s @ N = the rate of generating new tokens when N tokens are already in the KV cache, per sequence and aggregated over the batch. ${K_NOTE}`}
        />
      </div>
    </Panel>
  );
}
