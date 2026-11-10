import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * The DP attention + EP layout on one GB200 NVL72 rack (72 GPUs in one NVLink domain), with GLM-5.3-Flash (288
 * routed experts, 3 dense layers of FFN width 12288, 45 layers), batch 72:
 * - pick the layout in the Parallelism panel: EP 72 runs, the panel and the rig line name the layout, and a reload
 *   keeps it (`lay=dep`);
 * - dense layers on TP at EP 72 is an error (12288 / 72 is not whole); at EP 24 × PP 3 (12288 / 24 = 512) it runs,
 *   and the last stage (which holds the LM head, split 24 ways on TP) needs less memory than with DP.
 */

const QUERY = 'm=zai-org/GLM-5.3-Flash&d=gb200&n=72&b=72';

/** The Parallelism panel. */
const parallelism = (page: Page) =>
  page.locator('details.panel', { has: page.locator('> summary', { hasText: 'Parallelism' }) });

/** Bytes of the last stage's memory verdict ("Stage 3. Fits: 12.3GB of 186GB" → 1.23e10). */
async function lastStageBytes(page: Page): Promise<number> {
  const text = (await page.getByTestId('memory-verdict').last().textContent()) ?? '';
  const match = text.match(/([\d.,]+)\s?(KB|MB|GB|TB)/);
  if (!match) throw new Error(`no size in "${text}"`);
  const unit = { KB: 1e3, MB: 1e6, GB: 1e9, TB: 1e12 }[match[2] as 'KB' | 'MB' | 'GB' | 'TB'];
  return Number(match[1].replace(/,/g, '')) * unit;
}

/** Waits for a decode rate and asserts that no validation issue is shown. */
async function computed(page: Page) {
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('aside.notice[role="alert"]')).toHaveCount(0);
}

test('GB200 NVL72: GLM-5.3-Flash as DP attention + EP 72, kept on reload', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto(`/?${QUERY}`);
  await expect(page.getByTestId('model-facts')).toContainText(/glm/i);
  const panel = parallelism(page);
  // The segment's native radio covers its label (Segmented.tsx), so it is checked directly.
  await panel.getByRole('radio', { name: 'DP attention + EP' }).check();
  await panel.getByRole('spinbutton', { name: 'EP' }).fill('72');
  await computed(page);
  await expect(panel.locator('> summary .panel-value')).toHaveText('DP attention + EP 72');
  await expect(page.locator('h1.rig')).toContainText('DP attention + EP 72');
  await expect(panel.getByRole('spinbutton', { name: 'TP' })).toHaveCount(0);
  await expect(page).toHaveURL(/[?&]dp=72&ep=72&lay=dep/);

  await page.reload();
  await computed(page);
  await expect(parallelism(page).locator('> summary .panel-value')).toHaveText(
    'DP attention + EP 72',
  );
  await expect(parallelism(page).getByRole('radio', { name: 'DP attention + EP' })).toBeChecked();
  expect(errors).toEqual([]);
});

test('GB200 NVL72: dense layers on TP need EP to divide 12288; at EP 24 × PP 3 they run with less memory', async ({
  page,
}) => {
  await mockHf(page);
  await page.goto(`/?${QUERY}&dp=72&ep=72&lay=dep`);
  await computed(page);
  const panel = parallelism(page);
  const dense = panel.getByRole('radiogroup', { name: 'Dense layers' });
  await dense.getByRole('radio', { name: 'TP' }).check();
  await expect(
    panel.getByText('Dense layers on TP need EP 72 to divide their width 12288. Use DP.'),
  ).toBeVisible();

  // PP 3 refills each stage: EP = DP = 72 / 3 = 24.
  await panel.getByRole('spinbutton', { name: 'PP' }).fill('3');
  await expect(panel.getByRole('spinbutton', { name: 'EP' })).toHaveValue('24');
  await computed(page);
  await expect(panel.locator('> summary .panel-value')).toHaveText(
    'DP attention + EP 24, PP 3, dense TP',
  );
  await expect(page).toHaveURL(/[?&]pp=3&dp=24&ep=24&lay=dep&dn=tp/);
  await expect(page.getByTestId('memory-verdict')).toHaveCount(3);
  const tpBytes = await lastStageBytes(page);

  await dense.getByRole('radio', { name: 'DP' }).check();
  await computed(page);
  await expect(panel.locator('> summary .panel-value')).toHaveText('DP attention + EP 24, PP 3');
  // The LM head (and the dense FFN weights of stage 1) are whole on every GPU instead of split 24 ways.
  await expect.poll(() => lastStageBytes(page)).toBeGreaterThan(tpBytes);
});
