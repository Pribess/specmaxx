import { Panel } from './Panel';
import { NumberField } from './NumberField';
import { helpLines } from './helpLines';
import { SpecDecodeSection } from './SpecDecodeSection';
import { formatContextTokens, type DrafterStatus } from '../state';
import type { PanelProps } from './types';

const K_NOTE = '1k = 1024 tokens, so 8k = 8192.';

/** Method names in the panel summary. */
const METHOD_NAMES = { mtp: 'MTP', dflash2: 'DFlash 2', dspark: 'DSpark' } as const;

export function WorkloadPanel(props: PanelProps & { drafter?: DrafterStatus }) {
  const { state, dispatch, built } = props;
  const { batch, promptLen, decodeContextLen, prefixCacheHitRatio } = state.workload;
  const spec = state.deployment.specDec;
  return (
    <Panel
      title="Workload"
      value={`Batch ${batch}${props.pool ? ` (${props.pool})` : ''}, prefill ${formatContextTokens(promptLen)}, decode @ ${formatContextTokens(decodeContextLen)}${spec ? `, ${METHOD_NAMES[spec.method]}` : ''}`}
      open
    >
      <div className="fields">
        <NumberField
          label={props.pool ? `Batch size (${props.pool} pool)` : 'Batch size'}
          value={batch}
          onChange={(v) => dispatch({ type: 'setWorkload', workload: { batch: v } })}
          issues={built.issues.filter((i) => i.field === 'batch')}
          help={helpLines([
            'How many sequences run at the same time.',
            'A bigger batch raises total tokens per second and uses more memory.',
          ])}
        />
        <NumberField
          label="Prefill length (tokens)"
          value={promptLen}
          onChange={(v) => dispatch({ type: 'setWorkload', workload: { promptLen: v } })}
          issues={built.issues.filter((i) => i.field === 'promptLen')}
          help={helpLines([
            'Length of the prompt, in tokens.',
            'Prefill speed is measured while reading a prompt this long.',
            K_NOTE,
          ])}
        />
        <NumberField
          label="Prefix cache hit %"
          value={Math.round((prefixCacheHitRatio ?? 0) * 1e6) / 1e4}
          min={0}
          max={99.9999}
          step="any"
          onChange={(v) =>
            dispatch({
              type: 'setWorkload',
              workload: { prefixCacheHitRatio: v > 0 && v < 100 ? v / 100 : undefined },
            })
          }
          help={helpLines([
            'Share of each prompt already cached from earlier requests.',
            'Prefill computes only the rest, but still reads the cached part.',
            'Production caches hit 35–55% (DeepSeek, Mooncake). 0 means no cache.',
          ])}
        />
        <NumberField
          label="Decode context (tokens)"
          value={decodeContextLen}
          onChange={(v) => dispatch({ type: 'setWorkload', workload: { decodeContextLen: v } })}
          issues={built.issues.filter((i) => i.field === 'decodeContextLen')}
          help={helpLines([
            'Tokens already in the KV cache (the stored attention state) while decoding.',
            'Longer contexts read more memory per token, so decode is slower.',
            K_NOTE,
          ])}
        />
      </div>
      <SpecDecodeSection {...props} />
    </Panel>
  );
}
