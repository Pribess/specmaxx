import { inlineRunner, type RunResult } from './runner';
import type { SimInput } from '../../sim/engine/types';

const cache = new Map<string, RunResult>();
const scope = self as unknown as {
  onmessage:
    | ((event: MessageEvent<{ id: number; input: SimInput; notes: string[] }>) => void)
    | null;
  postMessage(data: unknown): void;
};
scope.onmessage = async ({ data }) => {
  try {
    const key = JSON.stringify([data.input, data.notes]);
    let result = cache.get(key);
    if (!result) result = await inlineRunner.run(data.input, data.notes);
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
};
