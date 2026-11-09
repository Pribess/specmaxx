import { expect, test } from '@playwright/test';
import { mockHf } from './hf-mock';

test('a gated shortlist model loads from its mirror without a token or any mention of the mirror', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await mockHf(page); // answers 401 for the official Muse Spark repo: no token is set
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await page.getByRole('combobox', { name: 'Hugging Face model ID' }).fill('Muse');
  await page.getByRole('option', { name: /Muse Spark/ }).click();
  await expect(page.getByTestId('model-facts')).toContainText('muse_spark');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByLabel('Hugging Face token')).toHaveCount(0);
  const text = (await page.locator('body').textContent()) ?? '';
  // `unsloth/` as a repo path: a hardware note names "Unsloth" the trainer, which is unrelated.
  expect(text).not.toMatch(/read from|unsloth\//i);
  expect(errors).toEqual([]);
});
