import { expect, test } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { mockHf } from './hf-mock';

for (const size of ['desktop', 'phone'] as const) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`${size} ${colorScheme}`, async ({ page }) => {
      await page.setViewportSize(
        size === 'desktop' ? { width: 1440, height: 900 } : { width: 390, height: 844 },
      );
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      await mockHf(page);
      await page.goto('/?m=Qwen/Qwen3.8-27B&n=2&tp=2&u=pcie-p2p.direct&b=2');
      await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
      await page.evaluate(() => document.fonts.ready);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
        true,
      );
      await mkdir('docs/screenshots', { recursive: true });
      await page.screenshot({
        path: `docs/screenshots/${size}-${colorScheme}.png`,
        fullPage: true,
      });
      if (size === 'desktop') {
        const height = await page
          .locator('.input-band')
          .evaluate((element) => element.scrollHeight + 100);
        await page.setViewportSize({ width: 1440, height });
        await page.screenshot({
          path: `docs/screenshots/desktop-full-${colorScheme}.png`,
          fullPage: true,
        });
        await page.setViewportSize({ width: 1440, height: 900 });
      }
      if (size === 'desktop' && colorScheme === 'light') {
        await page.getByRole('link', { name: 'How it works', exact: true }).click();
        await expect(page.getByRole('heading', { name: 'Memory fit', exact: true })).toBeVisible();
        await page.evaluate(() => document.fonts.ready);
        await page.screenshot({ path: 'docs/screenshots/about.png', fullPage: false });
      }
    });
  }
}
