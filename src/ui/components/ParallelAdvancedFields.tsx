import { NumberField } from './NumberField';
import { helpLines } from './helpLines';
import type { PanelProps } from './types';

/** Help text of the DCP field: short sentences, each shown on its own line in the tooltip. */
const DCP_HELP = [
  'Splits the KV cache by token across DCP of the TP GPUs.',
  'Each GPU keeps and reads 1/DCP of the cache; a small exchange follows each attention layer.',
  'Helps long context on MLA models, where TP otherwise copies the cache on every GPU.',
];

/**
 * Parallelism → Advanced → "Decode context parallelism": the DCP degree (`ParallelConfig.dcp`; 1 = off, stored as
 * absent). Shown in the custom layout only (DP attention + EP has TP 1, so there is nothing to split over). `hint`
 * lists the valid values ("Valid DCP: 1, 2, 4, 8"); errors come from the `dcp` field issues.
 */
export function ParallelAdvancedFields({
  state,
  dispatch,
  built,
  hint,
}: PanelProps & { hint?: string }) {
  return (
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
  );
}
