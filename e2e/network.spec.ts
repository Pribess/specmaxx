import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * Hardware → Advanced → Network on 2 × 8 H100 (NVLink 4 + InfiniBand NDR), GLM-5.3-Flash at TP 16 with overlap off,
 * so the inter-machine all-reduces of prefill are on the critical path (docs/simulation-model.md §6 "Multi-NIC nodes",
 * "In-network reduction (NVLS, SHARP)"):
 * - 4 NICs per machine halve each GPU's network share (one NIC per 2 GPUs), so prefill gets slower; `nic=4`.
 * - In-network reduction on (NVLS inside each machine, SHARP between them) gives the `best` all-reduce a faster
 *   candidate, so prefill gets faster again; `inr=on`.
 * Both survive a reload. Screenshots at desktop and phone width go to test-results/ (or `$NETWORK_SCREENSHOTS_DIR`).
 */

/** The rate a hero number shows, as a number (`1,234.5` → 1234.5). */
async function rate(page: Page, phase: 'prefill' | 'decode'): Promise<number> {
  const text = (await page.getByTestId(`${phase}-tps`).textContent()) ?? '';
  return Number(text.replace(/[^0-9.]/g, ''));
}

/** Picks `option` in the custom picker named `label`. */
async function pick(page: Page, label: string, option: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click();
  await page.getByRole('option', { name: option, exact: true }).click();
}

/** Opens the Hardware panel (folded on a phone) and its Advanced block. */
async function openNetwork(page: Page, phone: boolean) {
  if (phone) await page.locator('.panel > summary').filter({ hasText: 'Hardware' }).click();
  await page
    .locator('details.advanced', { has: page.getByTestId('network-advanced') })
    .locator('> summary')
    .click();
}

const URL = '/?m=zai-org/GLM-5.3-Flash&d=h100-sxm&n=8&nodes=2&tp=16&ov=none';

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test(`network: NICs per machine and in-network reduction move prefill and persist (${viewport.width}px)`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    const phone = viewport.width < 600;
    await page.setViewportSize(viewport);
    await mockHf(page);
    await page.goto(URL);
    await expect(page.getByTestId('prefill-tps')).toHaveText(/[1-9]/);
    const base = await rate(page, 'prefill');
    const baseText = await page.getByTestId('prefill-tps').textContent();
    await openNetwork(page, phone);
    const nics = page.getByRole('combobox', { name: 'NICs per machine', exact: true });
    await expect(nics).toHaveText('8 (1 per GPU)');
    await pick(page, 'NICs per machine', '4 (1 per 2 GPUs)');
    await expect(page).toHaveURL(/nic=4/);
    await expect(page.locator('.rig')).toContainText('(4 NICs per machine)');
    await expect(page.getByTestId('prefill-tps')).not.toHaveText(baseText ?? '');
    const fewer = await rate(page, 'prefill');
    expect(fewer).toBeLessThan(base);
    const fewerText = await page.getByTestId('prefill-tps').textContent();

    await expect(page.getByRole('combobox', { name: 'In-network reduction' })).toHaveText('Off');
    await pick(page, 'In-network reduction', 'On (NVLS, SHARP)');
    await expect(page).toHaveURL(/inr=on/);
    await expect(page.getByTestId('prefill-tps')).not.toHaveText(fewerText ?? '');
    const reduced = await rate(page, 'prefill');
    expect(reduced).toBeGreaterThan(fewer);
    const reducedText = await page.getByTestId('prefill-tps').textContent();

    const dir = process.env.NETWORK_SCREENSHOTS_DIR ?? 'test-results';
    await page.getByTestId('network-advanced').scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${dir}/network-${viewport.width}.png` });
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.screenshot({ path: `${dir}/network-${viewport.width}-dark.png` });
    await page.emulateMedia({ colorScheme: 'light' });

    // Both survive a reload (they are in the URL).
    await page.reload();
    await expect(page.getByTestId('prefill-tps')).toHaveText(reducedText ?? '');
    await openNetwork(page, phone);
    await expect(nics).toHaveText('4 (1 per 2 GPUs)');
    await expect(page.getByRole('combobox', { name: 'In-network reduction' })).toHaveText(
      'On (NVLS, SHARP)',
    );
    expect(errors).toEqual([]);
  });
}

test('network: hidden on one RTX 5090', async ({ page }) => {
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByTestId('network-advanced')).toHaveCount(0);
});
