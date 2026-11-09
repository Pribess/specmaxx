import { DEFAULT_SIM_OPTIONS } from '../../sim/engine/simulate';
import { HelpIcon } from './HelpIcon';
import { helpLines } from './helpLines';
import { Panel } from './Panel';
import { Picker } from './Picker';
import type { PickerOption } from './fuzzy';
import type { PanelProps as Props } from './types';

/** The engine switches shown here, with their labels, option values and definitions (docs/simulation-model.md §7). */
const SWITCHES = [
  {
    key: 'overlap',
    label: 'Communication overlap',
    help: [
      'Ideal: time spent sending data hides behind compute.',
      'None: sending time adds to compute time.',
      'The fixed delay of each message is never hidden.',
    ],
    options: [
      { value: 'ideal', label: 'ideal' },
      { value: 'none', label: 'none' },
    ],
  },
  {
    key: 'elementwiseFusion',
    label: 'Elementwise fusion',
    help: [
      'None: each simple op reads and writes its data from memory.',
      'Ideal: that extra memory traffic disappears.',
      'Weights and the KV cache still have to be read.',
    ],
    options: [
      { value: 'none', label: 'none' },
      { value: 'ideal', label: 'ideal' },
    ],
  },
  {
    key: 'collectiveAlgorithm',
    label: 'Collective algorithm',
    help: [
      'How GPUs combine their partial results (all-reduce).',
      'Best uses the faster of ring and tree for each message size.',
    ],
    options: [
      { value: 'best', label: 'best' },
      { value: 'ring', label: 'ring' },
      { value: 'tree', label: 'tree' },
    ],
  },
] as const satisfies readonly {
  key: keyof typeof DEFAULT_SIM_OPTIONS;
  label: string;
  help: readonly string[];
  options: readonly PickerOption[];
}[];

export function AssumptionsPanel({ state, dispatch }: Props) {
  return (
    <Panel title="Assumptions">
      <div className="field-head panel-help">
        <HelpIcon term="Assumptions">
          {helpLines([
            'Hardware and math limits only.',
            'No scheduler or engine overhead.',
            'Speculative decoding and offloading stay off until you turn them on.',
          ])}
        </HelpIcon>
      </div>
      {SWITCHES.map((s) => (
        <Picker
          key={s.key}
          label={s.label}
          value={state.options[s.key] ?? DEFAULT_SIM_OPTIONS[s.key]}
          options={[...s.options]}
          help={helpLines(s.help)}
          onChange={(v) => dispatch({ type: 'setOptions', options: { [s.key]: v } })}
        />
      ))}
    </Panel>
  );
}
