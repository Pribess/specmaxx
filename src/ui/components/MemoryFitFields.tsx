import { DEFAULT_PREFILL_CHUNK_TOKENS } from '../../sim/engine/memory';
import { NumberField } from './NumberField';
import { helpLines } from './helpLines';
import type { PanelProps } from './types';

/**
 * Hardware → Advanced → "Memory fit": the GPU memory limit (unified memory only), the runtime reserve per device and
 * the prefill chunk of the fit. Each one changes only the fit verdict, never a rate.
 */
export function MemoryFitFields({ state, dispatch, built }: PanelProps) {
  const memory = built.hardware?.device.device.memory;
  if (memory === undefined) return null;
  const unified = memory.kind === 'unified';
  const limitPercent = (state.hw.gpuUsableFraction ?? memory.gpuUsableFraction ?? 1) * 100;
  return (
    <fieldset className="unlock advanced-fields" data-testid="memory-fit-advanced">
      <legend>Memory fit</legend>
      {unified && (
        <NumberField
          label="GPU memory limit %"
          value={Math.round(limitPercent * 10) / 10}
          min={1}
          max={100}
          step="any"
          onChange={(v) =>
            dispatch({
              type: 'setMemoryFit',
              field: 'gpuUsableFraction',
              value: v > 0 && v <= 100 ? v / 100 : undefined,
            })
          }
          help={helpLines([
            'Share of memory the OS lets the GPU use.',
            'macOS allows about 75% until you raise iogpu.wired_limit_mb; Linux on Strix Halo about 50%.',
            'Set it to your raised limit.',
          ])}
        />
      )}
      <NumberField
        label="Runtime reserve GB"
        value={(state.hw.runtimeReserveBytes ?? 0) / 1e9}
        min={0}
        step="any"
        onChange={(v) =>
          dispatch({
            type: 'setMemoryFit',
            field: 'runtimeReserveBytes',
            value: v > 0 ? Math.round(v * 1e9) : undefined,
          })
        }
        help={helpLines([
          'Memory the framework and driver keep on each device, such as the CUDA context.',
          'Counted in the memory fit. Default 0.',
        ])}
      />
      <NumberField
        label="Prefill chunk"
        value={state.workload.prefillChunkTokens ?? DEFAULT_PREFILL_CHUNK_TOKENS}
        min={1}
        onChange={(v) =>
          dispatch({
            type: 'setWorkload',
            workload: {
              prefillChunkTokens:
                Number.isInteger(v) && v >= 1 && v !== DEFAULT_PREFILL_CHUNK_TOKENS ? v : undefined,
            },
          })
        }
        help={helpLines([
          'Prompt tokens the engine processes per prefill step.',
          'Sets the activation memory of long prompts; speed is not affected.',
        ])}
      />
    </fieldset>
  );
}
