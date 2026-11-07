import { expect, test } from '@playwright/test';
import { mockHf } from './hf-mock';

test('HF fixture loads and computes real worker rates', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await mockHf(page);
  await page.goto('/');
  await page.getByLabel('Hugging Face model ID').fill('Qwen/Qwen3.8-27B');
  await page.getByLabel('Hugging Face model ID').press('Enter');
  await expect(page.getByTestId('model-facts')).toContainText('qwen');
  for (const phase of ['prefill', 'decode']) {
    await expect(page.getByTestId(`${phase}-tps`)).toHaveText(/[1-9]/);
  }
  await expect(page.getByTestId('memory-verdict')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Where the time goes' })).toBeVisible();
  expect(errors).toEqual([]);
});
