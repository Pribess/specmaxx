import { expect, test } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * One-click memory fixes and the per-pool memory card, on the bundled default model (Qwen/Qwen3.8-27B, bf16, ≈ 54 GB)
 * on one RTX PRO 6000 (96 GB), batch 8. At a 128k context the KV + state alone is ≈ 70 GB, so it does not fit.
 */

test('a memory fix button makes the overflowing bar fit, and the panels show the new choice', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?d=rtx-pro-6000-ws&b=8&p=131072&c=131072');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByTestId('memory-verdict')).toContainText('Does not fit');
  const fixes = page.getByTestId('memory-fixes');
  await fixes.getByRole('button', { name: 'KV cache → host RAM', exact: true }).click();
  await expect(page).toHaveURL(/okt=ram/);
  await expect(page.getByTestId('memory-verdict')).toContainText('Fits');
  await expect(page.getByTestId('memory-fixes')).toHaveCount(0);
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toContainText(
    'KV cache',
  );
  expect(errors).toEqual([]);
});

test('with disaggregation on, the card shows both pools; a fix applies to the pool that does not fit', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  // An 8k prompt fits the prefill pool; the decode pool at 128k does not.
  await page.goto('/?d=rtx-pro-6000-ws&b=8&p=8192&c=131072&pd=1&dec.d=rtx-pro-6000-ws&dec.b=8');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const pools = page.getByTestId('memory-pool');
  await expect(pools).toHaveCount(2);
  const prefill = pools.nth(0);
  const decode = pools.nth(1);
  await expect(prefill.getByRole('heading', { name: 'Prefill pool' })).toBeVisible();
  await expect(decode.getByRole('heading', { name: 'Decode pool' })).toBeVisible();
  await expect(prefill.getByTestId('memory-verdict')).toContainText('Fits');
  await expect(decode.getByTestId('memory-verdict')).toContainText('Does not fit');
  // The decode headline carries the warning, naming the pool.
  await expect(page.locator('.hero-memory-warning')).toHaveAttribute(
    'title',
    'Decode pool: does not fit',
  );
  await expect(prefill.getByTestId('memory-fixes')).toHaveCount(0);
  await decode.getByRole('button', { name: 'KV cache → host RAM', exact: true }).click();
  await expect(decode.getByTestId('memory-verdict')).toContainText('Fits');
  await expect(pools.nth(0).getByTestId('memory-verdict')).toContainText('Fits');
  await expect(page.locator('.hero-memory-warning[title*="does not fit"]')).toHaveCount(0);
  expect(errors).toEqual([]);
});
