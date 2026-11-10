import { expect, test } from '@playwright/test';
import { mockHf } from './hf-mock';

test('HF fixture loads and computes real worker rates', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/');
  // The bundled default model gives numbers before any network call.
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await page
    .getByRole('combobox', { name: 'Hugging Face model ID' })
    .fill('Qwen/Qwen3.8-Flash-Next');
  await page.getByRole('combobox', { name: 'Hugging Face model ID' }).press('Enter');
  await expect(page.getByTestId('model-facts')).toContainText('qwen');
  for (const phase of ['prefill', 'decode']) {
    await expect(page.getByTestId(`${phase}-tps`)).toHaveText(/[1-9]/);
  }
  await expect(page.getByTestId('memory-verdict')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Bottleneck estimation' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('EXL3 weights at 2.5 bpw change the rate and survive a reload', async ({ page }) => {
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const before = await page.getByTestId('decode-tps').textContent();
  await page.getByRole('combobox', { name: 'Weights' }).click();
  await page.getByRole('option', { name: 'EXL3 2.5bpw', exact: true }).click();
  await expect(page).toHaveURL(/w=exl3/);
  await expect(page).toHaveURL(/bpw=2\.5/);
  await expect(page.getByLabel('Bits per weight', { exact: true })).toHaveCount(0);
  await expect(page.getByTestId('decode-tps')).not.toHaveText(before ?? '');
  const after = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Weights' })).toHaveText(/EXL3 2\.5bpw/);
  await expect(page.getByTestId('decode-tps')).toHaveText(after ?? '');
});

test('prefix cache hit 50% shows a computed prefill rate and survives a reload', async ({
  page,
}) => {
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('prefill-tps')).toHaveText(/[1-9]/);
  await expect(page.getByTestId('prefill-computed')).toHaveCount(0);
  const num = async (id: string) =>
    Number((await page.getByTestId(id).textContent())!.replace(/[^\d.]/g, ''));
  const before = await num('prefill-tps');
  await page.getByRole('spinbutton', { name: 'Prefix cache hit %' }).fill('50');
  await expect(page).toHaveURL(/ph=50/);
  await expect(page.getByTestId('prefill-computed')).toBeVisible();
  // Effective rate rises (~2× less compute, attention still reads the cache); computed ≈ half of effective.
  await expect.poll(() => num('prefill-tps')).toBeGreaterThan(before * 1.3);
  const effective = await num('prefill-tps');
  const computed = await num('prefill-computed-tps');
  expect(computed / effective).toBeGreaterThan(0.45);
  expect(computed / effective).toBeLessThan(0.55);
  await page.reload();
  await expect(page.getByRole('spinbutton', { name: 'Prefix cache hit %' })).toHaveValue('50');
  await expect(page.getByTestId('prefill-computed')).toBeVisible();
});
