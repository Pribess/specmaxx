import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * Rack scale on GB200: a DeepSeek-V3-class MoE (deepseek-ai/DeepSeek-V4-Flash-0731, the checked-in fixture) as
 * DP 64 × EP 64 with batch 64.
 * - 64 GPUs in one machine: one NVL72 NVLink Switch domain, so the expert all-to-all stays on NVLink and the rig line
 *   names no network link.
 * - 16 machines × 4 GPUs on the NVL4 bridge: the same 64 GPUs as NVL4 trays over InfiniBand, so most all-to-all
 *   traffic crosses the much slower network and decode is slower.
 * - With experts in host RAM, the Offload section shows the Grace pool and the fixed NVLink-C2C host link.
 */

const MODEL = 'deepseek-ai/DeepSeek-V4-Flash-0731';
const DP_EP = `m=${MODEL}&d=gb200&tp=1&dp=64&ep=64&b=64`;
/** URL id of the GB200 NVL4 bridge (`linkPresetId(NVLINK_5_NVL4)`). */
const NVL4 = 'nvlink5-nvl4-bridge-mesh-estimate';

/** The rate a hero number shows, as a number (`1,234.5` → 1234.5). */
async function rate(page: Page, phase: 'prefill' | 'decode'): Promise<number> {
  const text = (await page.getByTestId(`${phase}-tps`).textContent()) ?? '';
  return Number(text.replace(/[^0-9.]/g, ''));
}

/** Loads a query and waits for a decode rate; asserts no validation issue is shown. */
async function load(page: Page, query: string) {
  await page.goto(`/?${query}`);
  await expect(page.getByTestId('model-facts')).toContainText('deepseek');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('aside.notice[role="alert"]')).toHaveCount(0);
}

test('GB200: DP 64 × EP 64 in one NVL72 domain, then as 16 NVL4 trays over InfiniBand (slower decode)', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);

  await load(page, `${DP_EP}&n=64`);
  const rig = page.locator('h1.rig');
  await expect(rig).toContainText('64 × NVIDIA GB200');
  await expect(rig).toContainText('NVL72 NVLink Switch');
  await expect(rig).not.toContainText('InfiniBand');
  await expect(rig).not.toContainText(' IB ');
  const nvl72 = await rate(page, 'decode');
  expect(nvl72).toBeGreaterThan(0);

  await load(page, `${DP_EP}&n=4&nodes=16&il=${NVL4}`);
  await expect(rig).toContainText('NVL4');
  await expect(rig).toContainText(/InfiniBand|IB/);
  const trays = await rate(page, 'decode');
  expect(trays).toBeGreaterThan(0);
  expect(trays).toBeLessThan(nvl72);
  expect(errors).toEqual([]);
});

test('GB200 with experts in host RAM: the Grace pool of 32 CPUs and the NVLink-C2C host link', async ({
  page,
}) => {
  await mockHf(page);
  await load(page, `${DP_EP}&n=64&ox=ram&oxf=0.5`);
  const hardware = page.locator('details.panel', {
    has: page.locator('> summary', { hasText: 'Hardware' }),
  });
  if ((await hardware.getAttribute('open')) === null) await hardware.locator('> summary').click();
  const section = page.getByTestId('offload-section');
  await expect(section.getByRole('combobox', { name: 'Host RAM', exact: true })).toHaveText(
    /Grace LPDDR5X \(32 CPUs, 15360 GB\)/,
  );
  await expect(section.getByRole('spinbutton', { name: 'Host RAM capacity (GB)' })).toHaveValue(
    '15360',
  );
  await expect(section.getByTestId('host-link-static')).toHaveText(/^NVLink-C2C, 450\.0\s?GB\/s$/);
  await expect(section.getByRole('combobox', { name: 'Host link' })).toHaveCount(0);
});
