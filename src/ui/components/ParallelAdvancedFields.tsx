import { NumberField } from './NumberField';
import { Segmented } from './Segmented';
import { helpLines } from './helpLines';
import type { PanelProps } from './types';

/** Help text of the DCP field: short sentences, each shown on its own line in the tooltip. */
const DCP_HELP = [
  'Splits the KV cache by token across DCP of the TP GPUs.',
  'Each GPU keeps and reads 1/DCP of the cache; a small exchange follows each attention layer.',
  'Helps long context on MLA models, where TP otherwise copies the cache on every GPU.',
];

/** Help text of the chunked PP prefill switch: short sentences, each shown on its own line in the tooltip. */
const CHUNKED_PP_HELP = [
  'Splits each prompt into chunks of the prefill chunk size that pass through the pipeline stages one after another.',
  'Later stages start before the whole prompt is done, so the first token comes sooner.',
  'Throughput stays about the same.',
];

/** The chunked PP prefill switch's segments. */
const ON_OFF = [
  { value: 'off', label: 'Off' },
  { value: 'on', label: 'On' },
];

/**
 * Parallelism → Advanced: "Decode context parallelism", the DCP degree (`ParallelConfig.dcp`; 1 = off, stored as
 * absent), shown in the custom layout only (`showDcp`: DP attention + EP has TP 1, so there is nothing to split
 * over; `hint` lists the valid values, "Valid DCP: 1, 2, 4, 8"; errors come from the `dcp` field issues). And
 * "Chunked PP prefill" (`ParallelConfig.chunkedPipelinePrefill`; off = absent), shown only with PP > 1.
 */
export function ParallelAdvancedFields({
  state,
  dispatch,
  built,
  hint,
  showDcp = true,
}: PanelProps & { hint?: string; showDcp?: boolean }) {
  return (
    <>
      {showDcp && (
        <fieldset className="unlock advanced-fields" data-testid="parallel-advanced">
          <legend>Decode context parallelism</legend>
          <NumberField
            label="DCP"
            value={state.parallel.dcp ?? 1}
            onChange={(v) =>
              dispatch({ type: 'setParallel', parallel: { dcp: v === 1 ? undefined : v } })
            }
            issues={built.issues.filter((i) => i.field === 'dcp')}
            hint={hint}
            help={helpLines(DCP_HELP)}
          />
        </fieldset>
      )}
      {state.parallel.pp > 1 && (
        <fieldset className="unlock advanced-fields" data-testid="parallel-chunked-pp">
          <legend>Pipeline prefill</legend>
          <Segmented
            label="Chunked PP prefill"
            value={state.parallel.chunkedPipelinePrefill === true ? 'on' : 'off'}
            segments={ON_OFF}
            onChange={(v) =>
              dispatch({
                type: 'setParallel',
                parallel: { chunkedPipelinePrefill: v === 'on' ? true : undefined },
              })
            }
            help={helpLines(CHUNKED_PP_HELP)}
          />
        </fieldset>
      )}
    </>
  );
}
