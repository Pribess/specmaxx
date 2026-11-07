import type { SimInput } from '../../sim/engine/types';
import { simulate } from '../../sim/engine/simulate';
import { batchSweep, summarize, type ResultSummary, type SweepPoint } from '../state';

export interface RunResult {
  summary: ResultSummary;
  sweep: SweepPoint[];
}
export interface SimRunner {
  run(input: SimInput, notes: string[]): Promise<RunResult>;
  dispose?(): void;
}
export const inlineRunner: SimRunner = {
  async run(input, notes) {
    return { summary: summarize(simulate(input), input, notes), sweep: batchSweep(input) };
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
    run(input, notes) {
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
        worker!.postMessage({ id, input, notes });
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
