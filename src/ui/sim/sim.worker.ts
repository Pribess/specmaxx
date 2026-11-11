import { latestQueue } from './latestQueue';
import { inlineRunner, type DisaggregatedRun, type RunResult } from './runner';
import type { SimInput } from '../../sim/engine/types';

const cache = new Map<string, RunResult>();
const scope = self as unknown as {
  onmessage:
    | ((
        event: MessageEvent<{
          id: number;
          input: SimInput;
          notes: string[];
          disaggregated?: DisaggregatedRun;
        }>,
      ) => void)
    | null;
  postMessage(data: unknown): void;
};
const enqueue = latestQueue<Parameters<NonNullable<typeof scope.onmessage>>[0]['data']>(
  async (data) => {
    try {
      // The disaggregated request is part of the key only when present (so the single-pool key is as before).
      const key = JSON.stringify(
        data.disaggregated === undefined
          ? [data.input, data.notes]
          : [data.input, data.notes, data.disaggregated],
      );
      let result = cache.get(key);
      if (!result) result = await inlineRunner.run(data.input, data.notes, data.disaggregated);
      cache.delete(key);
      cache.set(key, result);
      if (cache.size > 32) cache.delete(cache.keys().next().value!);
      scope.postMessage({ id: data.id, result });
    } catch (error) {
      scope.postMessage({
        id: data.id,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  },
  (data) => scope.postMessage({ id: data.id, superseded: true }),
);
scope.onmessage = ({ data }) => enqueue(data);
