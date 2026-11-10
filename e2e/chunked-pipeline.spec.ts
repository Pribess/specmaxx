import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * Chunked pipeline prefill in the Parallelism panel's Advanced, on mocked Hub fixtures: Kimi-K3 on 4 × 8 B300 (TP 8,
 * PP 4) with one 1M-token prompt (batch 1). The switch is absent at PP 1. Turned on, the prefill headline still
 * quotes its context (`tok/s @ 1M`), the time to first token is below the one without chunking, the URL keeps it
 * (`cpp=1`) across a reload, and the panel value and the rig line name it. Turned off, the two time-to-first-token
 * rows disappear and the URL loses `cpp`.
 */

const MODEL = 'm=moonshotai/Kimi-K3';
const RIG = `${MODEL}&d=b300&n=8&nodes=4&tp=8&pp=4&ep=8&b=1&p=1048576`;

/** The Parallelism panel. */
const parallelism = (page: Page) =>
  page.locator('details.panel', { has: page.locator('> summary', { hasText: 'Parallelism' }) });

/** The Parallelism panel's Advanced block, opened. */
async function openAdvanced(page: Page) {
  const advanced = parallelism(page).locator('details.advanced');
  if ((await advanced.getAttribute('open')) === null) await advanced.locator('> summary').click();
  return advanced;
}

/** Waits for a prefill rate and asserts that no validation issue is shown. */
async function computed(page: Page) {
  await expect(page.getByTestId('prefill-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('aside.notice[role="alert"]')).toHaveCount(0);
}

/** A time-to-first-token row's value in seconds (its `data-seconds`). */
async function seconds(page: Page, testId: string): Promise<number> {
  const value = Number(await page.getByTestId(testId).getAttribute('data-seconds'));
  expect(Number.isFinite(value) && value > 0).toBe(true);
  return value;
}

test('no chunked PP prefill switch at PP 1', async ({ page }) => {
  await mockHf(page);
  await page.goto(`/?${MODEL}&d=b300&n=8&tp=8`);
  await computed(page);
  const advanced = await openAdvanced(page);
  await expect(advanced.getByTestId('parallel-advanced')).toBeVisible();
  await expect(advanced.getByRole('radiogroup', { name: 'Chunked PP prefill' })).toHaveCount(0);
});

test('Kimi-K3 at 1M on TP 8 × PP 4: chunked PP prefill cuts the time to first token and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto(`/?${RIG}`);
  await expect(page.getByTestId('model-facts')).toContainText(/kimi/i);
  await computed(page);
  // Off by default: no time-to-first-token rows, nothing named.
  await expect(page.getByTestId('prefill-ttft')).toHaveCount(0);
  await expect(page.locator('h1.rig')).not.toContainText('chunked PP');

  const advanced = await openAdvanced(page);
  const toggle = advanced.getByRole('radiogroup', { name: 'Chunked PP prefill' });
  await toggle.getByRole('radio', { name: 'On' }).check();
  await computed(page);
  await expect(page.locator('.answer').filter({ hasText: 'Prefill' })).toContainText('tok/s @ 1M');
  await expect(page.getByTestId('prefill-ttft')).toContainText('@ 1M');
  const chunked = await seconds(page, 'prefill-ttft');
  const unchunked = await seconds(page, 'prefill-ttft-unchunked');
  expect(chunked).toBeLessThan(unchunked);
  await expect(parallelism(page).locator('> summary .panel-value')).toHaveText(
    'TP 8, PP 4, DP 1, EP 8, chunked PP',
  );
  await expect(page.locator('h1.rig')).toContainText('TP 8, chunked PP');
  await expect(page).toHaveURL(/[?&]pp=4&ep=8&cpp=1(&|$)/);

  await page.reload();
  await computed(page);
  await expect(page.getByTestId('prefill-ttft')).toBeVisible();
  const again = await openAdvanced(page);
  const onAgain = again.getByRole('radiogroup', { name: 'Chunked PP prefill' });
  await expect(onAgain.getByRole('radio', { name: 'On' })).toBeChecked();

  // Off: the rows disappear and the URL is the one from before.
  await onAgain.getByRole('radio', { name: 'Off' }).check();
  await computed(page);
  await expect(page.getByTestId('prefill-ttft')).toHaveCount(0);
  await expect(page.getByTestId('prefill-ttft-unchunked')).toHaveCount(0);
  await expect(page).not.toHaveURL(/cpp=/);
  expect(errors).toEqual([]);
});
