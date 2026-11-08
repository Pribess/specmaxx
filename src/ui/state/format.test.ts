import { expect, it } from 'vitest';
import {
  formatBytes,
  formatBytesPerSec,
  formatContextTokens,
  formatFlopsPerSec,
  formatParams,
  formatRateAtContext,
  formatSeconds,
} from './format';

it.each([
  [512, '512'],
  [8192, '8k'],
  [32768, '32k'],
  [131072, '128k'],
  [1048576, '1M'],
  [10000, '10,000'],
] as const)('quotes %i context tokens as %s', (tokens, expected) => {
  expect(formatContextTokens(tokens)).toBe(expected);
});
it('carries the exact context with each rate and uses no-space SI units', () => {
  expect(formatRateAtContext(33.91, 8192)).toBe('33.9 tok/s @ 8k');
  expect(formatRateAtContext(33.91, 1048576)).toBe('33.9 tok/s @ 1M');
  expect(formatParams(27e9)).toBe('27B');
  expect(formatBytes(3.2e9)).toBe('3.2GB');
  expect(formatBytesPerSec(1.792e12)).toBe('1.79TB/s');
  expect(formatFlopsPerSec(209.5e12)).toBe('209.5TFLOP/s');
  expect(formatSeconds(0.00123)).toBe('1.23ms');
});
