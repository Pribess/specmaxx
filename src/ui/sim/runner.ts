import type { SimInput } from '../../sim/engine/types';
import { simulate } from '../../sim/engine/simulate';
import {
  poolInput,
  simulateDisaggregated,
  type DisaggregatedInput,
} from '../../sim/engine/disaggregation';
import {
  batchSweep,
  speculativeBreakEvenBatch,
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
  /** Speculative decoding on: the smallest batch where it stops paying (sequences); absent when none was found. */
  breakEvenBatch?: number;
  /**
   * Disaggregated results; present only when the run had a {@link DisaggregatedRun} (absent key otherwise, so a
   * run without it is exactly the one from before disaggregation).
   */
  pd?: {
    /** Each pool's summary and the cross-pool figures. */
    summary: DisaggregatedSummary;
    /** Batch sweep of the decode pool's own input. */
    decodeSweep: SweepPoint[];
    /**
     * Speculative decoding on: the decode pool's break-even batch over its own sweep (its own parallel layout, PP
     * included); absent when none was found. Unit: sequences.
     */
    decodeBreakEvenBatch?: number;
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
    const sweep = batchSweep(input);
    const summary = summarize(simulate(input), input, notes);
    if (disaggregated === undefined) {
      const breakEvenBatch = speculativeBreakEvenBatch(input, sweep);
      return { summary, sweep, ...(breakEvenBatch === undefined ? {} : { breakEvenBatch }) };
    }
    // Disaggregated: the chart shows the decode pool's sweep, so the break-even batch is searched on the decode
    // pool's input (its own hardware, parallel layout and batch); the single-pool input is the prefill pool's.
    const pd = disaggregated.input;
    const decodeInput = poolInput(pd.shared, pd.decodePool, 'decode');
    const decodeSweep = batchSweep(decodeInput);
    const decodeBreakEvenBatch = speculativeBreakEvenBatch(decodeInput, decodeSweep);
    return {
      summary,
      sweep,
      pd: {
        summary: summarizeDisaggregated(
          simulateDisaggregated(pd),
          pd,
          notes,
          disaggregated.decodeNotes,
        ),
        decodeSweep,
        ...(decodeBreakEvenBatch === undefined ? {} : { decodeBreakEvenBatch }),
      },
    };
  },
};

let earlyWorker: Worker | undefined;

/** Start downloading the module worker before React renders; the first runner takes ownership. */
export function startSimWorker(): void {
  earlyWorker ??= new Worker(new URL('./sim.worker.ts', import.meta.url), { type: 'module' });
}
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
        startSimWorker();
        worker = earlyWorker!;
        earlyWorker = undefined;
        worker.onmessage = (
          event: MessageEvent<{
            id: number;
            result?: RunResult;
            error?: string;
            superseded?: boolean;
          }>,
        ) => {
          const p = pending.get(event.data.id);
          pending.delete(event.data.id);
          if (event.data.result) p?.resolve(event.data.result);
          else
            p?.reject(
              new Error(
                event.data.superseded ? 'Superseded' : (event.data.error ?? 'Simulation failed.'),
              ),
            );
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
