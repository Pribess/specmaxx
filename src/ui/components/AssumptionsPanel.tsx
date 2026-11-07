import { DEFAULT_SIM_OPTIONS } from '../../sim/engine/simulate';
import { Panel } from './Panel';
import type { PanelProps as Props } from './types';

export function AssumptionsPanel({ state, dispatch }: Props) {
  const choices = {
    overlap: ['ideal', 'none'],
    elementwiseFusion: ['none', 'ideal'],
    collectiveAlgorithm: ['best', 'ring', 'tree'],
  } as const;
  return (
    <Panel title="Assumptions">
      <p className="hint">
        Hardware and math limits only. No scheduler, paging, speculative decoding or engine-specific
        overhead.
      </p>
      {(Object.keys(choices) as (keyof typeof choices)[]).map((k) => (
        <label className="field" key={k}>
          <span>
            {
              {
                overlap: 'Communication overlap',
                elementwiseFusion: 'Elementwise fusion',
                collectiveAlgorithm: 'Collective algorithm',
              }[k]
            }
          </span>
          <select
            value={state.options[k] ?? DEFAULT_SIM_OPTIONS[k]}
            onChange={(e) => dispatch({ type: 'setOptions', options: { [k]: e.target.value } })}
          >
            {choices[k].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
      ))}
    </Panel>
  );
}
