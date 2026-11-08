import { expect, test, type Page } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { mockHf } from './hf-mock';

/** Output directory of every screenshot (committed). */
const DIR = 'docs/screenshots';

async function open(page: Page, size: 'desktop' | 'phone', colorScheme: 'light' | 'dark') {
  await page.setViewportSize(size === 'desktop' ? { width: 1440, height: 900 } : { width: 390, height: 844 });
  await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await mkdir(DIR, { recursive: true });
}

for (const size of ['desktop', 'phone'] as const) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const name = `${DIR}/${size}-${colorScheme}`;
      await page.screenshot({ path: `${name}.png`, fullPage: true });

      // Element crops: headline numbers, bottleneck section, charts. The sticky compact-results bar would
      // overlay the crops, so it is hidden while they are taken and restored afterwards.
      const setBar = (hidden: boolean) =>
        page.addStyleTag({ content: `.compact-results{visibility:${hidden ? 'hidden' : 'visible'} !important}` });
      await setBar(true);
      await page.locator('section.hero').screenshot({ path: `${name}-hero.png` });
      await page.locator('section.breakdown').screenshot({ path: `${name}-breakdown.png` });
      await page.locator('section.sweep').screenshot({ path: `${name}-sweep.png` });

      // Chart tooltips in the hovered state.
      const sweepPoint = page.locator('[data-chart="sweep"] [data-chart-point]').nth(3);
      await sweepPoint.scrollIntoViewIfNeeded();
      await sweepPoint.hover({ force: true });
      await sweepPoint.focus();
      await expect(page.locator('[data-chart="sweep"] [role="tooltip"]')).toBeVisible();
      await page.locator('section.sweep').screenshot({ path: `${name}-sweep-tooltip.png` });
      const roofPoint = page.locator('[data-chart="roofline"] [data-chart-point]').first();
      await roofPoint.scrollIntoViewIfNeeded();
      await roofPoint.hover({ force: true });
      await roofPoint.focus();
      await expect(page.locator('[data-chart="roofline"] [role="tooltip"]')).toBeVisible();
      await page.locator('section.hero').screenshot({ path: `${name}-roofline-tooltip.png` });

      await setBar(false);

      // The opened device picker with a search typed.
      await page.evaluate(() => scrollTo(0, 0));
      if (size === 'phone') {
        // Panels start collapsed on phones.
        await page.locator('details.panel > summary', { hasText: 'Hardware' }).click();
      }
      await page.getByRole('combobox', { name: 'Device' }).click();
      await page.keyboard.type('m5');
      await expect(page.getByRole('option').first()).toBeVisible();
      await page.screenshot({ path: `${name}-device-picker.png` });
      await page.keyboard.press('Escape');

      if (size === 'desktop') {
        // Scrolled to the bottom: the sidebar must still span the viewport.
        await page.evaluate(() => scrollTo(0, document.documentElement.scrollHeight));
        await page.screenshot({ path: `${name}-bottom.png` });
        const sidebar = await page.locator('.input-band').evaluate((el) => el.getBoundingClientRect());
        expect(sidebar.bottom).toBeGreaterThanOrEqual(899);
        if (colorScheme === 'light') {
          await page.getByRole('link', { name: 'How it works', exact: true }).click();
          await expect(page.getByRole('heading', { name: 'Memory fit', exact: true })).toBeVisible();
          await page.evaluate(() => document.fonts.ready);
          await page.screenshot({ path: `${DIR}/about.png`, fullPage: false });
        }
      }
    });
  }
}
