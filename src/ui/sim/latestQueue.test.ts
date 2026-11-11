import { afterEach, expect, it, vi } from 'vitest';
import { latestQueue } from './latestQueue';

afterEach(() => vi.useRealTimers());

it('drops queued jobs, yields between runs, and preserves the latest input', async () => {
  vi.useFakeTimers();
  let finish!: () => void;
  const ran: number[] = [];
  const dropped: number[] = [];
  const queue = latestQueue<number>(
    async (input) => {
      ran.push(input);
      if (input === 2)
        await new Promise<void>((resolve) => {
          finish = resolve;
        });
    },
    (input) => dropped.push(input),
  );
  queue(1);
  queue(2);
  await vi.advanceTimersByTimeAsync(0);
  expect(ran).toEqual([2]);
  expect(dropped).toEqual([1]);
  queue(3);
  queue(4);
  queue(8192);
  finish();
  await vi.runAllTimersAsync();
  expect(ran).toEqual([2, 8192]);
  expect(dropped).toEqual([1, 3, 4]);
  queue(9);
  await vi.runAllTimersAsync();
  expect(ran).toEqual([2, 8192, 9]);
});
