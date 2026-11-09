import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * Deployment knobs on mocked Hub fixtures: speculative decoding on the bundled default model (Qwen/Qwen3.8-27B,
 * which ships an MTP head) and routed-expert offload on a MoE (Qwen/Qwen3.8-Flash-Next, 512 experts). Each knob shows
 * its result annotation (the spec-dec speedup on the decode rate, the Host RAM tier bar) and survives a reload.
 */

/** Picks `option` in the custom picker named `label`. */
async function pick(page: Page, label: string, option: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click();
  await page.getByRole('option', { name: option, exact: true }).click();
}

test('MTP speculative decoding shows the speedup on the decode rate and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const before = await page.getByTestId('decode-tps').textContent();
  await pick(page, 'Speculative decoding', 'MTP (model head)');
  await expect(page).toHaveURL(/sd=mtp/);
  await expect(page.getByTestId('spec-dec')).toContainText(/spec-dec ×\d+\.\d\d/);
  await expect(page.getByTestId('decode-tps')).not.toHaveText(before ?? '');
  const after = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(
    page.getByRole('combobox', { name: 'Speculative decoding', exact: true }),
  ).toHaveText(/MTP/);
  await expect(page.getByTestId('decode-tps')).toHaveText(after ?? '');
  // Off again: the URL and the rate are back to the defaults.
  await pick(page, 'Speculative decoding', 'Off');
  await expect(page).not.toHaveURL(/sd=/);
  await expect(page.getByTestId('decode-tps')).toHaveText(before ?? '');
  await expect(page.getByTestId('spec-dec')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('DSpark without a released drafter uses the hypothetical one, badged', async ({ page }) => {
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await pick(page, 'Speculative decoding', 'DSpark');
  await expect(page.getByTestId('hypothetical-badge')).toBeVisible();
  await expect(page.getByTestId('spec-dec')).toBeVisible();
});

test('DFlash 2 loads its released drafter config from the Hub', async ({ page }) => {
  await mockHf(page);
  await page.goto('/?sd=dflash2');
  await expect(page.getByRole('combobox', { name: 'Drafter', exact: true })).toHaveValue(
    'Model default (incoai/Qwen3.8-27B-DFlash2)',
  );
  await expect(page.getByTestId('spec-dec')).toBeVisible();
  await expect(page.getByTestId('hypothetical-badge')).toHaveCount(0);
});

test('expert offload to host RAM shows the tier bar and survives a reload', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next');
  await expect(page.getByTestId('model-facts')).toContainText('qwen');
  await expect(page.getByTestId('ngram-params')).toContainText('n-gram tables');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByTestId('memory-tier')).toHaveCount(0);
  await pick(page, 'Routed experts', 'Host RAM');
  await expect(page).toHaveURL(/ox=ram/);
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toBeVisible();
  const decode = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Routed experts', exact: true })).toHaveText(
    'Host RAM',
  );
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toBeVisible();
  await expect(page.getByTestId('decode-tps')).toHaveText(decode ?? '');
  expect(errors).toEqual([]);
});

test('every deployment and host URL key loads, stays canonical and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  // Every new key with a non-default value, in the codec's canonical order (url.ts).
  const query =
    'm=Qwen/Qwen3.8-Flash-Next&hr=ddr5-6400-8ch&hrc=512&ssd=nvme-t705-2tb&hl=4x8' +
    '&ox=ssd&oxl=3&oxf=0.25&oxh=fine&odl=40&odt=ssd&olt=ram&ols=interleaved&oqd=32&opf=0' +
    '&sd=dflash2&sdr=hyp&sdk=5&sdu=code';
  await page.goto(`/?${query}`);
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByText('Ignored invalid URL values')).toHaveCount(0);
  await expect(page.getByTestId('spec-dec')).toBeVisible();
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'SSD' })).toBeVisible();
  // The app rewrites the URL from its state (after a short debounce): it must come back unchanged.
  await page.waitForTimeout(400);
  expect(new URL(page.url()).search).toBe(`?${query}`);
  const decode = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(page.getByTestId('decode-tps')).toHaveText(decode ?? '');
  await page.waitForTimeout(400);
  expect(new URL(page.url()).search).toBe(`?${query}`);
  // A custom acceptance replaces the use case in the URL.
  await page.goto(`/?${query.replace('sdu=code', 'sda=0.8')}`);
  await expect(page.getByRole('combobox', { name: 'Use case', exact: true })).toHaveText('Custom');
  await expect(page.getByLabel('Acceptance α', { exact: true })).toHaveValue('0.8');
  expect(errors).toEqual([]);
});

test('a legacy offload-overlap URL is rewritten to the equivalent prefetch accuracy', async ({
  page,
}) => {
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&oo=none&ox=ram');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByText('Ignored invalid URL values')).toHaveCount(0);
  await page.waitForTimeout(400);
  expect(new URL(page.url()).search).toBe('?m=Qwen/Qwen3.8-Flash-Next&ox=ram&opf=0');
  await page.locator('details.advanced > summary').click();
  await expect(page.getByTestId('offload-advanced')).toBeVisible();
  await expect(page.getByLabel('Prefetch accuracy %', { exact: true })).toHaveValue('0');
});

test('Host RAM picker groups DDR5 before DDR4, searches, and a DDR4 id round-trips in the URL', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&ox=ram');
  await page.getByRole('combobox', { name: 'Host RAM', exact: true }).click();
  const groups = page.getByRole('listbox').getByRole('group');
  await expect(groups.nth(0)).toContainText('DDR5');
  await expect(groups.nth(1)).toContainText('DDR4');
  await page.screenshot({ path: 'test-results/host-ram-picker.png' });
  await page.getByRole('combobox', { name: 'Host RAM', exact: true }).press('Escape');
  await page.getByRole('combobox', { name: 'Host RAM', exact: true }).click();
  await page.keyboard.type('ddr4 3200 8');
  await page.getByRole('option', { name: /DDR4-3200, 8 channels/ }).click();
  await expect(page).toHaveURL(/hr=ddr4-3200-8ch/);
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Host RAM', exact: true })).toHaveText(
    /DDR4-3200, 8 channels/,
  );
  expect(errors).toEqual([]);
});

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test(`SSD read rate picker persists and only appears for SSD offload (${viewport.width}px)`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    await mockHf(page);
    await page.goto('/?odl=0&odt=ssd');
    if (viewport.width < 600)
      await page.locator('.panel > summary').filter({ hasText: 'Hardware' }).click();
    await page.locator('details.advanced > summary').click();
    const rate = page.getByRole('combobox', { name: 'SSD read rate', exact: true });
    await expect(rate).toHaveText('Datasheet');
    await expect(page).not.toHaveURL(/osr=/);
    await pick(page, 'SSD read rate', 'Sustained');
    await expect(page).toHaveURL(/osr=sustained/);
    await page.reload();
    if (viewport.width < 600)
      await page.locator('.panel > summary').filter({ hasText: 'Hardware' }).click();
    await page.locator('details.advanced > summary').click();
    await expect(rate).toHaveText('Sustained');
    await rate.scrollIntoViewIfNeeded();
    if (process.env.SSD_SCREENSHOTS_DIR) {
      await page.screenshot({
        path: `${process.env.SSD_SCREENSHOTS_DIR}/ssd-rate-${viewport.width}.png`,
      });
      await page.emulateMedia({ colorScheme: 'dark' });
      await page.screenshot({
        path: `${process.env.SSD_SCREENSHOTS_DIR}/ssd-rate-${viewport.width}-dark.png`,
      });
    }
    await pick(page, 'SSD read rate', 'Datasheet');
    await expect(page).not.toHaveURL(/osr=/);
    await page.goto('/?odl=0');
    if (viewport.width < 600)
      await page.locator('.panel > summary').filter({ hasText: 'Hardware' }).click();
    await page.locator('details.advanced > summary').click();
    await expect(rate).toHaveCount(0);
  });
}
