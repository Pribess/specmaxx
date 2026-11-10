import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * Prefill/decode disaggregation in DeepSeek's 32 : 144 shape (docs/design/prefill-decode-disaggregation.md): GLM-5.3-
 * Flash (288 experts, divisible by 32 and 144) on H100 SXM nodes of 8. The prefill pool is DP attention + EP 32 on 4
 * machines, batch 32; the decode pool, set through the UI, DP attention + EP 144 on 18 machines, batch 144. The
 * test checks the rig line, the rates with their contexts, the System answer, the URL, a reload, the KV link
 * (a slow link not overlapped lowers the system rate), and that turning it off restores exactly the earlier page.
 */

const QUERY = 'm=zai-org/GLM-5.3-Flash&d=h100-sxm&n=8&nodes=4&dp=32&ep=32&lay=dep&b=32';

/** An input panel by its title (a prefix, e.g. "Hardware" also matches "Hardware (decode pool)"). */
const panel = (page: Page, title: string) =>
  page.locator('details.panel', { has: page.locator('> summary', { hasText: title }) });

/** Waits for the hero to settle with a decode rate and no validation issue. */
async function computed(page: Page) {
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('section.hero')).not.toHaveClass(/pending/);
  await expect(page.locator('aside.notice[role="alert"]')).toHaveCount(0);
}

/** The number at the start of the System answer ("533.3 tok/s/GPU @ …" → 533.3). */
async function systemRate(page: Page): Promise<number> {
  const text = (await page.getByTestId('pd-system-tps').textContent()) ?? '';
  return Number(text.replace(/,/g, '').match(/[\d.]+/)?.[0]);
}

/** The number of a rate figure by test id ("387,477" → 387477). */
async function rate(page: Page, testId: string): Promise<number> {
  const text = (await page.getByTestId(testId).textContent()) ?? '';
  return Number(text.replace(/,/g, '').match(/[\d.]+/)?.[0]);
}

/** Opens a panel when it is collapsed (phones start collapsed). */
async function open(page: Page, title: string) {
  const p = panel(page, title);
  if ((await p.getAttribute('open')) === null) await p.locator('> summary').click();
  return p;
}

test('DeepSeek 32 : 144 shape: two pools, system figures, URL, reload, KV link, and off restores the page', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto(`/?${QUERY}`);
  await expect(page.getByTestId('model-facts')).toContainText(/glm/i);
  await computed(page);
  await expect(page).toHaveURL(/[?&]lay=dep/);
  const before = {
    hero: await page.locator('section.hero').innerText(),
    title: await page.title(),
    url: page.url(),
  };

  // On, then the decode pool: 18 machines (DP attention + EP 144 follows the device count), batch 144.
  const pd = await open(page, 'Disaggregation');
  await pd
    .getByRole('radiogroup', { name: 'Prefill/decode disaggregation' })
    .getByRole('radio', { name: 'On' })
    .check();
  await pd
    .getByRole('radiogroup', { name: 'Edit pool' })
    .getByRole('radio', { name: 'Decode' })
    .check();
  const hw = await open(page, 'Hardware (decode pool)');
  await hw.getByRole('combobox', { name: 'Machines' }).click();
  await page.getByRole('option', { name: '18', exact: true }).click();
  const parallel = await open(page, 'Parallelism (decode pool)');
  await expect(parallel.getByRole('radio', { name: 'DP attention + EP' })).toBeChecked();
  await expect(parallel.getByRole('spinbutton', { name: 'EP', exact: true })).toHaveValue('144');
  const workload = await open(page, 'Workload');
  await workload.getByRole('spinbutton', { name: 'Batch size (decode pool)' }).fill('144');
  await computed(page);

  const rig = page.getByTestId('pd-rig');
  await expect(rig).toContainText('32 × NVIDIA H100 SXM 80GB prefill');
  await expect(rig).toContainText('144 × NVIDIA H100 SXM 80GB decode');
  await expect(rig).toContainText('DP attention + EP 32 / 144');
  await expect(page.locator('.answer').filter({ hasText: 'Prefill' }).first()).toContainText(
    /tok\/s @ 8k/,
  );
  await expect(page.getByTestId('prefill-tps')).toHaveText(/[1-9]/);
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByTestId('pd-system-tps')).toHaveText(
    /^[\d,.]+\s*tok\/s\/GPU @ 8k\/1k, decode @ 8k$/,
  );
  await expect(page.getByTestId('pd-balanced')).toHaveText(/^(1 : [\d.]+|[\d.]+ : 1)$/);
  await expect(page.getByTestId('pd-bottleneck')).toHaveText(/^(Bottleneck: |Balanced)/);
  await expect(page.getByTestId('pd-kv')).toContainText(/KV .+\/token, .+ per request @ 8k/);
  await expect(page).toHaveURL(/[?&]pd=1(&|$)/);
  await expect(page).toHaveURL(/dec\.nodes=18&dec\.dp=144&dec\.ep=144&dec\.lay=dep/);
  await expect(page).toHaveURL(/dec\.b=144/);

  // A reload keeps it all (the edited pool is not in the URL: it is the prefill pool again).
  const on = {
    system: (await page.getByTestId('pd-system-tps').textContent()) ?? '',
    balanced: (await page.getByTestId('pd-balanced').textContent()) ?? '',
    rig: (await rig.textContent()) ?? '',
  };
  await page.reload();
  await computed(page);
  await expect(page.getByTestId('pd-system-tps')).toHaveText(on.system);
  await expect(page.getByTestId('pd-balanced')).toHaveText(on.balanced);
  await expect(page.getByTestId('pd-rig')).toHaveText(on.rig);

  // KV link: with enough decode instances that prefill limits the system, 1 GB/s and then not overlapped lower
  // the system rate; kvo=none in the URL.
  const kv = await open(page, 'Disaggregation');
  await kv.getByRole('spinbutton', { name: 'Decode instances' }).fill('16');
  await computed(page);
  await expect(page.getByTestId('pd-bottleneck')).toHaveText(/^Bottleneck: prefill pool/);
  await expect(page).toHaveURL(/[?&]pdr=1:16/);
  // At the default 50 GB/s overlapped link the transfer hides behind prefill: no "Without KV transfer" line.
  await expect(page.getByTestId('prefill-no-kv')).toHaveCount(0);
  const fastPrefill = (await page.getByTestId('prefill-tps').textContent()) ?? '';
  await kv.getByRole('spinbutton', { name: 'KV bandwidth (GB/s)' }).fill('1');
  await computed(page);
  await expect(page).toHaveURL(/kvx=bandwidthPerDirectionBytesPerSec:1e9/);
  const overlapped = await systemRate(page);
  const overlappedPrefill = await rate(page, 'prefill-tps');
  await kv
    .getByRole('radiogroup', { name: 'KV transfer' })
    .getByRole('radio', { name: 'Not overlapped' })
    .check();
  await computed(page);
  await expect(page).toHaveURL(/[?&]kvo=none/);
  await expect.poll(() => systemRate(page)).toBeLessThan(overlapped);
  // The Prefill figure is the prefill pool's rate with the transfer: lower too, and the rate without the transfer
  // (the one shown at the default fast link) moves to its own line.
  await expect.poll(() => rate(page, 'prefill-tps')).toBeLessThan(overlappedPrefill);
  await expect(page.getByTestId('prefill-no-kv')).toContainText(fastPrefill);

  // Off: exactly the page and URL from before.
  await kv
    .getByRole('radiogroup', { name: 'Prefill/decode disaggregation' })
    .getByRole('radio', { name: 'Off' })
    .check();
  await computed(page);
  await expect(page.getByTestId('pd-system-tps')).toHaveCount(0);
  await expect.poll(() => page.locator('section.hero').innerText()).toBe(before.hero);
  await expect.poll(() => page.title()).toBe(before.title);
  await expect.poll(() => page.url()).toBe(before.url);
  expect(errors).toEqual([]);
});
