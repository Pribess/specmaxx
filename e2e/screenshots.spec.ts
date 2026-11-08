import { expect, test, type Page } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { mockHf } from './hf-mock';

/** Output directory of every screenshot (committed); `SCREENSHOTS_DIR` writes them elsewhere (e.g. a scratch dir). */
const DIR = process.env.SCREENSHOTS_DIR ?? 'docs/screenshots';

async function open(page: Page, size: 'desktop' | 'phone', colorScheme: 'light' | 'dark') {
  await page.setViewportSize(
    size === 'desktop' ? { width: 1440, height: 900 } : { width: 390, height: 844 },
  );
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
    test(`hardware setups ${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const name = (s: string) => `${DIR}/hardware-${s}-${size}-${colorScheme}.png`;
      const panel = page.locator('details.panel', {
        has: page.locator('summary', { hasText: 'Hardware' }),
      });
      /** Loads a query, expands the Hardware panel on phones, and returns once the setup row is shown. */
      const load = async (query: string) => {
        await page.goto(`/${query}`);
        await expect(page.locator('.rig')).toBeVisible();
        if (size === 'phone') await panel.locator('> summary').click();
        await expect(page.getByTestId('setup-row')).toBeVisible();
        await page.evaluate(() => document.fonts.ready);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
          true,
        );
      };
      // The setup row (machine count, link, cabling) for the switchless cases and a GeForce pair.
      const cases: [string, string][] = [
        ['strix-halo-x4-usb4-ring', '?d=ryzen-ai-max-plus-395&nodes=4&pp=4'],
        ['m3-ultra-x4-tb5-mesh', '?d=apple-m3-ultra-80c&cap=2&nodes=4&pp=4'],
        ['m5-ultra-x4-tb5-mesh2', '?d=apple-m5-ultra-80c&cap=2&nodes=4&fab=mesh2&pp=4'],
        ['dgx-spark-x4-ring', '?d=dgx-spark&nodes=4&fab=ring&tp=4'],
        ['rtx-5090-x2', '?n=2&tp=2'],
      ];
      for (const [id, query] of cases) {
        await load(query);
        await panel.screenshot({ path: name(id) });
      }
      // The cabling picker open, and its "?" help (switch / full mesh / ring / daisy chain).
      await load('?d=ryzen-ai-max-plus-395&nodes=4&pp=4');
      await page.getByRole('combobox', { name: 'Cabling' }).click();
      await expect(page.getByRole('option').first()).toBeVisible();
      await page.screenshot({ path: name('cabling-picker') });
      await page.keyboard.press('Escape');
      await page.getByRole('button', { name: 'About Cabling' }).focus();
      await expect(page.getByRole('tooltip').filter({ hasText: 'Daisy chain' })).toBeVisible();
      await page.screenshot({ path: name('cabling-help') });
      // The device picker open on its "Popular setups" group.
      await page.getByRole('combobox', { name: 'Device', exact: true }).click();
      await expect(page.getByRole('group', { name: 'Popular setups' })).toBeVisible();
      await page.screenshot({ path: name('device-picker-popular') });
      await page.keyboard.press('Escape');
    });
    test(`memory states ${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const bar = page.locator('.capacity-bar');
      await expect(page.getByTestId('memory-verdict')).toHaveClass('memory-danger');
      // Derive the exact default allocation from its capacity marker (default: 32 GB RTX 5090).
      const capacityPercent = await bar.evaluate((el) =>
        parseFloat((el as HTMLElement).style.getPropertyValue('--capacity')),
      );
      const totalBytes = (32e9 * 100) / capacityPercent;
      for (const state of ['danger', 'warning', 'fits'] as const) {
        if (state !== 'danger') {
          const capacityBytes = Math.ceil(totalBytes / (state === 'warning' ? 0.92 : 0.65));
          await page.goto(`/?x=memory.capacityOptionsBytes.0:${capacityBytes}`);
        }
        await expect(page.getByTestId('memory-verdict')).toHaveClass(`memory-${state}`);
        await page.evaluate(() => document.fonts.ready);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
          true,
        );
        const warnings = page.locator('.hero-memory-warning');
        await expect(warnings).toHaveCount(state === 'fits' ? 0 : 2);
        if (state !== 'fits') {
          await warnings.first().click();
          await expect(page.locator('#memory-fit')).toBeFocused();
        }
        // Hide the sticky duplicate rates while capturing the complete hero and memory report.
        await page.addStyleTag({ content: '.compact-results { visibility: hidden !important }' });
        await page.evaluate(() => scrollTo(0, 0));
        await page
          .locator('section.hero')
          .screenshot({ path: `${DIR}/memory-${state}-${size}-${colorScheme}.png` });
      }
    });

    test(`${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const name = `${DIR}/${size}-${colorScheme}`;
      await page.screenshot({ path: `${name}.png`, fullPage: true });

      // Element crops: headline numbers, bottleneck section, charts. The sticky compact-results bar would
      // overlay the crops, so it is hidden while they are taken and restored afterwards.
      const setBar = (hidden: boolean) =>
        page.addStyleTag({
          content: `.compact-results{visibility:${hidden ? 'hidden' : 'visible'} !important}`,
        });
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

      // The opened Weights picker with the EXL3 group.
      if (size === 'phone') await page.locator('details.panel > summary', { hasText: 'Quantization' }).click();
      await page.getByRole('combobox', { name: 'Weights' }).click();
      await page.getByRole('option', { name: 'EXL3 custom…' }).scrollIntoViewIfNeeded();
      await expect(page.getByRole('option', { name: 'EXL3 custom…' })).toBeVisible();
      await page.screenshot({ path: `${name}-weights-picker.png` });
      await page.keyboard.press('Escape');

      // The opened device picker scrolled to the middle of its list: rows scroll beneath the search header,
      // the popup is as wide as its trigger, and its themed scrollbar sits inside the rounded border.
      const deviceTrigger = page.getByRole('combobox', { name: 'Device', exact: true });
      await deviceTrigger.click();
      const viewport = page.locator('.picker-popup [data-radix-scroll-area-viewport]');
      await viewport.evaluate((el) => {
        el.scrollTop = (el.scrollHeight - el.clientHeight) / 2;
      });
      await viewport.hover();
      await expect(page.locator('.picker-popup .scroll-area-thumb')).toBeVisible();
      const popup = await page.locator('.picker-popup').boundingBox();
      const anchor = await deviceTrigger.boundingBox();
      expect(Math.abs(popup!.width - anchor!.width)).toBeLessThan(1);
      expect(Math.abs(popup!.x - anchor!.x)).toBeLessThan(1);
      expect(popup!.y + popup!.height).toBeLessThanOrEqual((page.viewportSize()?.height ?? 0) + 1);
      await page.screenshot({ path: `${name}-device-picker-scrolled.png` });
      await page.keyboard.press('Escape');

      if (size === 'desktop') {
        // Scrolled to the bottom: the sidebar must still span the viewport.
        await page.evaluate(() => scrollTo(0, document.documentElement.scrollHeight));
        await page.screenshot({ path: `${name}-bottom.png` });
        const sidebar = await page
          .locator('.input-band')
          .evaluate((el) => el.getBoundingClientRect());
        expect(sidebar.bottom).toBeGreaterThanOrEqual(899);
        if (colorScheme === 'light') {
          await page.getByRole('link', { name: 'How it works', exact: true }).click();
          await expect(
            page.getByRole('heading', { name: 'Memory fit', exact: true }),
          ).toBeVisible();
          await page.evaluate(() => document.fonts.ready);
          await page.screenshot({ path: `${DIR}/about.png`, fullPage: false });
        }
      }
    });
  }
}
