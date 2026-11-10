import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * Decode context parallelism (DCP) in the Parallelism panel's Advanced, on mocked Hub fixtures:
 * - Kimi-K3 (MLA: one latent KV head, so TP 8 copies the cache on every GPU) on one 8 × B300 node at TP 8 and a
 *   128k decode context: DCP 8 runs, the panel value and the rig line name it, the URL keeps it (`dcp=8`) across a
 *   reload, and more sequences fit than with DCP 1; DCP 3 shows its error on the field and stops the result.
 * - Qwen3.8-27B (GQA, 4 KV heads) at TP 8: only DCP 1 and 2 are valid; DCP 4 is an error on the field.
 */

const KIMI = 'm=moonshotai/Kimi-K3&d=b300&n=8&tp=8&c=131072';

/** The Parallelism panel. */
const parallelism = (page: Page) =>
  page.locator('details.panel', { has: page.locator('> summary', { hasText: 'Parallelism' }) });

/** The Parallelism panel's Advanced block, opened. */
async function openAdvanced(page: Page) {
  const advanced = parallelism(page).locator('details.advanced');
  if ((await advanced.getAttribute('open')) === null) await advanced.locator('> summary').click();
  return advanced;
}

/** The memory card's "max N sequences per replica" count. */
async function maxSequences(page: Page): Promise<number> {
  const text = (await page.locator('.memory-limit').first().textContent()) ?? '';
  const match = text.match(/max\s+([\d,]+)\s+sequences/);
  if (!match) throw new Error(`no sequence count in "${text}"`);
  return Number(match[1].replace(/,/g, ''));
}

/** Waits for a decode rate and asserts that no validation issue is shown. */
async function computed(page: Page) {
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('aside.notice[role="alert"]')).toHaveCount(0);
}

test('Kimi-K3 on 8 × B300: DCP 8 fits more sequences at 128k, names itself and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto(`/?${KIMI}`);
  await expect(page.getByTestId('model-facts')).toContainText(/kimi/i);
  await computed(page);
  const before = await maxSequences(page);
  expect(before).toBeGreaterThanOrEqual(1);
  await expect(page.locator('h1.rig')).not.toContainText('DCP');

  const advanced = await openAdvanced(page);
  await expect(advanced.getByText('Valid DCP: 1, 2, 4, 8')).toBeVisible();
  await advanced.getByRole('spinbutton', { name: 'DCP' }).fill('8');
  await computed(page);
  await expect(page.locator('.answer').filter({ hasText: 'Decode' })).toContainText('tok/s @ 128k');
  await expect(parallelism(page).locator('> summary .panel-value')).toHaveText(
    'TP 8, PP 1, DP 1, EP 1, DCP 8',
  );
  await expect(page.locator('h1.rig')).toContainText('TP 8, DCP 8');
  await expect(page).toHaveURL(/[?&]tp=8&dcp=8(&|$)/);
  await expect.poll(() => maxSequences(page)).toBeGreaterThan(before);

  await page.reload();
  await computed(page);
  const again = await openAdvanced(page);
  await expect(again.getByRole('spinbutton', { name: 'DCP' })).toHaveValue('8');

  // DCP 3 does not divide TP 8: the error is on the field and there is no result.
  await again.getByRole('spinbutton', { name: 'DCP' }).fill('3');
  await expect(again.getByText('DCP 3 must divide TP 8.')).toBeVisible();
  await expect(again.getByRole('spinbutton', { name: 'DCP' })).toHaveAttribute(
    'aria-invalid',
    'true',
  );
  await expect(page.getByTestId('decode-tps')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('Qwen3.8-27B (4 KV heads) at TP 8: only DCP 1 and 2 are valid', async ({ page }) => {
  await mockHf(page);
  await page.goto('/?d=b200&n=8&tp=8');
  await computed(page);
  const advanced = await openAdvanced(page);
  await expect(advanced.getByText('Valid DCP: 1, 2')).toBeVisible();
  await advanced.getByRole('spinbutton', { name: 'DCP' }).fill('4');
  await expect(
    advanced.getByText('DCP 4 must divide TP ÷ KV heads = 2', { exact: false }),
  ).toBeVisible();
  await advanced.getByRole('spinbutton', { name: 'DCP' }).fill('2');
  await computed(page);
  await expect(page).toHaveURL(/[?&]dcp=2(&|$)/);
});
