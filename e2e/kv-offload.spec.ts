import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/** The Hardware panel's Advanced block, which holds the Offload subsection with the KV-cache fields. */
function hardwareAdvanced(page: Page) {
  return page.locator('details.advanced', { has: page.getByTestId('offload-advanced') });
}

/** The hero's decode rate as a number (tok/s), after checking it is quoted at the 128k decode context. */
async function decodeAt128k(page: Page): Promise<number> {
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('.answer').filter({ hasText: 'Decode' })).toContainText('tok/s @ 128k');
  return Number((await page.getByTestId('decode-tps').textContent())?.replace(/,/g, ''));
}

/**
 * KV-cache tiering on the bundled default model (Qwen/Qwen3.8-27B, bf16, ≈ 53.8 GB of weights) on one RTX PRO 6000
 * (96 GB), batch 8, prompt and decode context 128k. All on the GPU, the KV + state is ≈ 70 GB, so the device needs
 * ≈ 125 GB: it does not fit. With 25% of each sequence's KV on the GPU (the rest in host RAM) the resident state is
 * ≈ 18 GB, plus ≈ 6.4 GB of staging for one layer's fetched KV: ≈ 79 GB, which fits. Decode then streams the 75%
 * off-GPU KV over PCIe every step, so it is much slower than the (non-fitting) all-on-GPU ceiling
 * (≈ 117 → ≈ 9 tok/s @ 128k when this was written).
 */
test('KV cache 25% on GPU in host RAM makes a long context fit, decodes slower, and survives a reload', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?d=rtx-pro-6000-ws&b=8&p=131072&c=131072');
  const onGpu = await decodeAt128k(page);
  await expect(page.getByTestId('memory-verdict')).toContainText('Does not fit');
  await expect(page.getByTestId('memory-tier')).toHaveCount(0);

  await hardwareAdvanced(page).locator('> summary').click();
  const share = page.getByLabel('KV cache on GPU %', { exact: true });
  await expect(share).toHaveValue('100');
  await share.fill('25');
  await expect(page).toHaveURL(/okt=ram&okd=25/);
  await expect(page.getByRole('combobox', { name: 'KV cache tier', exact: true })).toHaveText(
    'Host RAM',
  );
  await expect(page.getByTestId('memory-verdict')).toContainText('Fits');
  await expect(page.getByTestId('memory-tier').filter({ hasText: 'Host RAM' })).toContainText(
    'KV cache',
  );
  await expect(page.getByText('KV cache: 25% on device, 75% host RAM')).toBeVisible();
  const offloaded = await decodeAt128k(page);
  expect(offloaded).toBeLessThan(onGpu);

  await page.reload();
  expect(await decodeAt128k(page)).toBe(offloaded);
  await hardwareAdvanced(page).locator('> summary').click();
  await expect(page.getByLabel('KV cache on GPU %', { exact: true })).toHaveValue('25');

  // Back to 100%: the KV keys leave the URL and the original result returns.
  await page.getByLabel('KV cache on GPU %', { exact: true }).fill('100');
  await expect(page).not.toHaveURL(/ok[tdrhp]=/);
  expect(await decodeAt128k(page)).toBe(onGpu);
  expect(errors).toEqual([]);
});
