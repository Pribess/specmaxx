import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * 4 × DGX Spark (GB10) over ConnectX-7, switched vs ring, on a shortlist MoE (Qwen/Qwen3.8-Flash-Next: 512 experts,
 * 48 layers), TP 4. Why the numbers differ (docs/simulation-model.md §6 "Fabric shapes"):
 * - The UI fixes EP at 1, so the MoE runs with its experts tensor-parallel and the only inter-node traffic is the
 *   TP all-reduce of every layer; no all-to-all.
 * - Decode is latency bound (one token: small messages). On the switch the 'best' all-reduce is the binomial tree,
 *   2⌈log2 4⌉ = 4 steps of α (1.5 µs each, 6 µs). On the ring the tree's second step crosses 2 links and pays a relay
 *   (forwarding latency) on top, so the ring algorithm's 2(N − 1) = 6 steps of α (9 µs) is the best there: every
 *   all-reduce costs more, so ring decode is SLOWER than switched decode.
 * - Prefill is bandwidth bound and uses the ring algorithm on both fabrics. Each node can push only ~25 GB/s over
 *   both ConnectX-7 ports together (their shared PCIe attachment), which is exactly the one port a switch gives
 *   it, so the ring's bandwidth equals the switch's and prefill is IDENTICAL.
 */

/** The rate a hero number shows, as a number (`1,234.5` → 1234.5). */
async function rate(page: Page, phase: 'prefill' | 'decode'): Promise<number> {
  const text = (await page.getByTestId(`${phase}-tps`).textContent()) ?? '';
  return Number(text.replace(/[^0-9.]/g, ''));
}

test('4 × DGX Spark: ring decode is slower than switched, prefill is the same', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await mockHf(page);
  await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&d=dgx-spark&nodes=4&tp=4');
  await expect(page.getByTestId('model-facts')).toContainText('qwen');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.locator('.rig')).toContainText(
    '4 × NVIDIA DGX Spark (GB10) over ConnectX-7, with',
  );
  const switched = { prefill: await rate(page, 'prefill'), decode: await rate(page, 'decode') };
  const switchedPrefillText = await page.getByTestId('prefill-tps').textContent();
  const switchedDecodeText = await page.getByTestId('decode-tps').textContent();

  await page.getByRole('combobox', { name: 'Cabling' }).click();
  await page.getByRole('option', { name: 'Ring', exact: true }).click();
  await expect(page).toHaveURL(/fab=ring/);
  await expect(page.locator('.rig')).toContainText('over ConnectX-7 (ring)');
  await expect(page.getByTestId('decode-tps')).not.toHaveText(switchedDecodeText ?? '');
  const ring = { prefill: await rate(page, 'prefill'), decode: await rate(page, 'decode') };
  expect(switched.decode).toBeGreaterThan(ring.decode);
  expect(ring.prefill).toBe(switched.prefill);
  await expect(page.getByTestId('prefill-tps')).toHaveText(switchedPrefillText ?? '');

  // The cabling survives a reload (it is in the URL).
  const ringDecodeText = await page.getByTestId('decode-tps').textContent();
  await page.reload();
  await expect(page.getByTestId('decode-tps')).toHaveText(ringDecodeText ?? '');
  await expect(page.getByTestId('prefill-tps')).toHaveText(switchedPrefillText ?? '');
  await expect(page.locator('.rig')).toContainText('(ring)');
  await expect(page.getByRole('combobox', { name: 'Cabling' })).toHaveText('Ring');
  expect(errors).toEqual([]);
});
