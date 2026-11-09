import { MAX_LATENCY_FLOOR_US, latencyFloorSecFromMicros } from '../state';
import { NumberField } from './NumberField';
import { Segmented } from './Segmented';
import { helpLines } from './helpLines';
import type { PanelProps } from './types';

/**
 * Hardware → Advanced → "Per-op costs": the per-op latency floor (`SimOptions.perOpLatencyFloorSec`, a what-if
 * that changes the rates; 0 = off) and the engine dispatch overlay (an extra labelled decode figure; the headline is
 * unchanged). Both are off by default.
 */
export function PerOpCostFields({ state, dispatch }: PanelProps) {
  const floorUs = (state.options.perOpLatencyFloorSec ?? 0) * 1e6;
  return (
    <fieldset className="unlock advanced-fields" data-testid="per-op-costs-advanced">
      <legend>Per-op costs</legend>
      <NumberField
        label="Per-op latency floor µs"
        value={Math.round(floorUs * 1e3) / 1e3}
        min={0}
        max={MAX_LATENCY_FLOOR_US}
        step="any"
        onChange={(v) =>
          dispatch({
            type: 'setOptions',
            options: {
              perOpLatencyFloorSec:
                v > 0 && v <= MAX_LATENCY_FLOOR_US ? latencyFloorSecFromMicros(v) : undefined,
            },
          })
        }
        help={helpLines([
          'Shortest time any single operation can take, data exchanges between devices included.',
          'Tiny operations are raised to it, which slows decode most. Default 0: off.',
        ])}
      />
      <Segmented
        label="Engine dispatch overlay"
        value={state.engineDispatchOverlay ? 'on' : 'off'}
        segments={[
          { value: 'off', label: 'off' },
          { value: 'on', label: 'on' },
        ]}
        onChange={(v) => dispatch({ type: 'setEngineDispatchOverlay', on: v === 'on' })}
        help={helpLines([
          'Also shows decode with the time an engine spends launching each GPU kernel.',
          'The headline stays the hardware limit.',
        ])}
      />
    </fieldset>
  );
}
