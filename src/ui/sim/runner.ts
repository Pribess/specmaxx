import type { SimInput } from '../../sim/engine/types';
import { simulate } from '../../sim/engine/simulate';
import {
  poolInput,
  simulateDisaggregated,
  type DisaggregatedInput,
} from '../../sim/engine/disaggregation';
import {
  batchSweep,
  summarize,
  summarizeDisaggregated,
  type DisaggregatedSummary,
  type ResultSummary,
  type SweepPoint,
} from '../state';

/** The disaggregated part of a run request (only while disaggregation is on). */
export interface DisaggregatedRun {
  /** The two pools and the settings (`buildDisaggregatedInput`). */
  input: DisaggregatedInput;
  /** The decode pool's notes (its unlock descriptions, loader notes), for its summary's warnings. */
  decodeNotes: string[];
}

export interface RunResult {
  summary: ResultSummary;
  sweep: SweepPoint[];
  /**
   * Disaggregated results; present only when the run had a {@link DisaggregatedRun} (absent key otherwise, so a
   * run without it is exactly the one from before disaggregation).
   */
  pd?: {
    /** Each pool's summary and the cross-pool figures. */
    summary: DisaggregatedSummary;
    /** Batch sweep of the decode pool's own input. */
    decodeSweep: SweepPoint[];
  };
}
export interface SimRunner {
  /**
   * Simulates `input` (the single-pool input: with disaggregation on, the prefill pool's state) and, with
   * `disaggregated`, the two pools too.
   */
  run(input: SimInput, notes: string[], disaggregated?: DisaggregatedRun): Promise<RunResult>;
  dispose?(): void;
}
export const inlineRunner: SimRunner = {
  async run(input, notes, disaggregated) {
    const result: RunResult = {
      summary: summarize(simulate(input), input, notes),
      sweep: batchSweep(input),
    };
    if (disaggregated === undefined) return result;
    const pd = disaggregated.input;
    return {
      ...result,
      pd: {
        summary: summarizeDisaggregated(
          simulateDisaggregated(pd),
          pd,
          notes,
          disaggregated.decodeNotes,
        ),
        decodeSweep: batchSweep(poolInput(pd.shared, pd.decodePool, 'decode')),
      },
    };
  },
};

/** Lazy worker creation also lets component tests inject the inline runner. */
export function workerRunner(): SimRunner {
  let worker: Worker | undefined;
  let nextId = 0;
  const pending = new Map<
    number,
    { resolve: (r: RunResult) => void; reject: (e: Error) => void }
  >();
  return {
    run(input, notes, disaggregated) {
      if (!worker) {
        worker = new Worker(new URL('./sim.worker.ts', import.meta.url), { type: 'module' });
        worker.onmessage = (
          event: MessageEvent<{ id: number; result?: RunResult; error?: string }>,
        ) => {
          const p = pending.get(event.data.id);
          pending.delete(event.data.id);
          if (event.data.result) p?.resolve(event.data.result);
          else p?.reject(new Error(event.data.error ?? 'Simulation failed.'));
        };
        worker.onerror = () => {
          for (const p of pending.values())
            p.reject(new Error('Simulation worker failed. Reload to retry.'));
          pending.clear();
        };
      }
      const id = ++nextId;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker!.postMessage(
          disaggregated === undefined ? { id, input, notes } : { id, input, notes, disaggregated },
        );
      });
    },
    dispose() {
      worker?.terminate();
      worker = undefined;
      for (const p of pending.values()) p.reject(new Error('Cancelled'));
      pending.clear();
    },
  };
}
