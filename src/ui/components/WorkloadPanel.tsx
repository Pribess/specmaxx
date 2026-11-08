import { Panel } from './Panel';
import { NumberField } from './NumberField';
import { SpecDecodeSection } from './SpecDecodeSection';
import { formatContextTokens, type DrafterStatus } from '../state';
import type { PanelProps } from './types';

const K_NOTE = '1k = 1024 tokens, so 8k = 8192.';

/** Method names in the panel summary. */
const METHOD_NAMES = { mtp: 'MTP', dflash2: 'DFlash 2', dspark: 'DSpark' } as const;

export function WorkloadPanel(props: PanelProps & { drafter?: DrafterStatus }) {
  const { state, dispatch, built } = props;
  const { batch, promptLen, decodeContextLen } = state.workload;
  const spec = state.deployment.specDec;
  return (
    <Panel
      title="Workload"
      value={`Batch ${batch}, prefill ${formatContextTokens(promptLen)}, decode @ ${formatContextTokens(decodeContextLen)}${spec ? `, ${METHOD_NAMES[spec.method]}` : ''}`}
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
      <SpecDecodeSection {...props} />
    </Panel>
  );
}
