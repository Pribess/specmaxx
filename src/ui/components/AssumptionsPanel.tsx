import { DEFAULT_SIM_OPTIONS } from '../../sim/engine/simulate';
import { HelpIcon } from './HelpIcon';
import { Panel } from './Panel';
import { Picker } from './Picker';
import type { PickerOption } from './fuzzy';
import type { PanelProps as Props } from './types';

/** The engine switches shown here, with their labels, option values and definitions (docs/simulation-model.md §7). */
const SWITCHES = [
  {
    key: 'overlap',
    label: 'Communication overlap',
    help: 'Ideal: collective bandwidth time hides behind compute. None: communication adds to compute time. Latency is never hidden.',
    options: [
      { value: 'ideal', label: 'ideal' },
      { value: 'none', label: 'none' },
    ],
  },
  {
    key: 'elementwiseFusion',
    label: 'Elementwise fusion',
    help: 'None: every elementwise op reads and writes its activations from memory. Ideal: that activation traffic is fused away; weights and state still move.',
    options: [
      { value: 'none', label: 'none' },
      { value: 'ideal', label: 'ideal' },
    ],
  },
  {
    key: 'collectiveAlgorithm',
    label: 'Collective algorithm',
    help: 'All-reduce algorithm. Best takes the faster of ring and tree for each size.',
    options: [
      { value: 'best', label: 'best' },
      { value: 'ring', label: 'ring' },
      { value: 'tree', label: 'tree' },
    ],
  },
  {
    key: 'offloadOverlap',
    label: 'Offload overlap',
    help: "Ideal: the next layer's transfers overlap this layer's compute (prefetch, perfect prediction); in prefill this can hide expert transfers entirely. None: fetched on demand, serialized. Only matters when weights are offloaded.",
    options: [
      { value: 'ideal', label: 'ideal' },
      { value: 'none', label: 'none' },
    ],
  },
] as const satisfies readonly {
  key: keyof typeof DEFAULT_SIM_OPTIONS;
  label: string;
  help: string;
  options: readonly PickerOption[];
}[];

export function AssumptionsPanel({ state, dispatch }: Props) {
  return (
    <Panel title="Assumptions">
      <div className="field-head panel-help">
        <HelpIcon term="Assumptions">
          Hardware and math limits only. No scheduler, paging or engine-specific overhead.
          Speculative decoding and offloading are off unless set in Workload and Hardware.
        </HelpIcon>
      </div>
      {SWITCHES.map((s) => (
        <Picker
          key={s.key}
          label={s.label}
          value={state.options[s.key] ?? DEFAULT_SIM_OPTIONS[s.key]}
          options={[...s.options]}
          help={s.help}
          onChange={(v) => dispatch({ type: 'setOptions', options: { [s.key]: v } })}
        />
      ))}
    </Panel>
  );
}
