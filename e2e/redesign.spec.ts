import { expect, test, type Locator, type Page } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { mockHf } from './hf-mock';

/**
 * Before/after capture contract. Every state uses the SAME viewport-only screenshot (fullPage: false),
 * at 1440×900 and 390×844, in both schemes. Interaction states scroll their subject into view; they
 * intentionally do not resize the viewport, hide UI, or depend on sidebar/panel styling.
 * Separate spec-dec/offload and overflow/high-utilisation frames keep each subject legible on phones.
 */
const STATES = [
  // Unmodified default: Qwen3.8-27B, one RTX 5090, overflowing device memory.
  { name: '01-default', query: '', action: 'default' },
  // FP8 weights on a tensor-parallel pair: assert memory-fits, not just successful simulation.
  { name: '02-fitting', query: '?w=fp8_e4m3&n=2&tp=2', action: 'fits' },
  // Four DGX Spark nodes connected as a ring; reveal Hardware to show the setup/cabling.
  { name: '03-multi-node', query: '?d=dgx-spark&nodes=4&fab=ring&tp=4', action: 'hardware' },
  // Reveal Hardware, open Device and type m5 without choosing an option.
  { name: '04-device-picker', query: '', action: 'picker' },
  // Reveal Hardware and keyboard-focus the input's About Device help button.
  { name: '05-input-help', query: '', action: 'help' },
  // MTP enabled with k=3; reveal Workload and scroll the speculative controls into view.
  { name: '06-spec-dec', query: '?w=fp8_e4m3&n=2&tp=2&sd=mtp&sdk=3', action: 'specdec' },
  // Routed-expert cache (5% resident), host RAM experts + lookup tables; reveal Hardware Advanced (its Offload part).
  {
    name: '07-offload',
    query:
      '?m=Qwen/Qwen3.8-Flash-Next&w=fp8_e4m3&n=2&hr=ddr5-6000-12ch&tp=2&ox=ram&oxf=0.05&oxh=fine&olt=ram',
    action: 'offload',
  },
  // Hover AND focus a roofline point, keeping its unit-bearing tooltip visible.
  { name: '08-chart-tooltip', query: '?w=fp8_e4m3&n=2&tp=2', action: 'chart' },
  // Scroll the default red overflow verdict into view (distinct from the default landing viewport).
  { name: '09-memory-overflow', query: '', action: 'danger' },
  // Default allocation is 55.3 GB: a 60 GB capacity override produces ~92% utilisation.
  {
    name: '10-memory-high',
    query: '?x=memory.capacityOptionsBytes.0:60000000000',
    action: 'warning',
  },
  // Standalone documentation landing viewport.
  { name: '11-about', query: '', action: 'about' },
] as const;

const DIR = process.env.REDESIGN_DIR ?? 'docs/design/redesign/before';

/** Native details has inconsistent button-role exposure: try accessible buttons, then semantic summary. */
async function reveal(page: Page, title: string, subject: Locator) {
  if (await subject.isVisible()) return;
  const name = new RegExp(`^${title}`);
  const button = page.getByRole('button', { name }).first();
  if (await button.count()) {
    if ((await button.getAttribute('aria-expanded')) !== 'true') await button.click();
  } else {
    const summary = page.locator('summary').filter({ hasText: name }).first();
    await expect(summary).toBeVisible();
    await summary.click();
  }
  await expect(subject).toBeVisible();
}

async function ready(page: Page) {
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await page.evaluate(() => document.fonts.ready);
}

/** Center the subject in whichever pane/document owns its scroll, including the future split layout. */
async function center(subject: Locator) {
  await subject.evaluate((element) =>
    element.scrollIntoView({ block: 'center', inline: 'nearest' }),
  );
}

for (const size of ['desktop', 'phone'] as const) {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`redesign ${size} ${colorScheme}`, async ({ page }) => {
      await page.setViewportSize(
        size === 'desktop' ? { width: 1440, height: 900 } : { width: 390, height: 844 },
      );
      await page.emulateMedia({ colorScheme, reducedMotion: 'reduce' });
      await mockHf(page);
      await mkdir(DIR, { recursive: true });

      for (const state of STATES) {
        await test.step(state.name, async () => {
          await page.goto(state.action === 'about' ? '/about.html' : `/${state.query}`);
          if (state.action === 'about') {
            await expect(
              page.getByRole('heading', { name: 'Memory fit', exact: true }),
            ).toBeVisible();
            await page.evaluate(() => document.fonts.ready);
          } else {
            await ready(page);
            const verdict = page.getByTestId('memory-verdict').first();
            const device = page.getByRole('combobox', { name: 'Device', exact: true });
            switch (state.action) {
              case 'default':
                await expect(verdict).toHaveClass(/memory-danger/);
                break;
              case 'fits':
                await expect(verdict).toHaveClass(/memory-fits/);
                break;
              case 'hardware':
              case 'picker':
              case 'help': {
                await reveal(page, 'Hardware', device);
                await center(device);
                if (state.action === 'hardware') {
                  await expect(page.getByRole('combobox', { name: 'Cabling' })).toHaveText('Ring');
                  await expect(page.getByTestId('setup-row')).toBeVisible();
                } else if (state.action === 'picker') {
                  await device.click();
                  await page.keyboard.type('m5');
                  await expect(page.getByRole('option').first()).toBeVisible();
                  await expect(page.getByRole('option').first()).toContainText(/M5/i);
                } else {
                  const help = page.getByRole('button', { name: 'About Device', exact: true });
                  await help.focus();
                  await expect(page.getByRole('tooltip')).toBeVisible();
                }
                break;
              }
              case 'specdec': {
                const controls = page.getByTestId('specdec-section');
                await reveal(page, 'Workload', controls);
                await center(controls);
                await expect(
                  page.getByRole('combobox', { name: 'Speculative decoding', exact: true }),
                ).toHaveText('MTP (model head)');
                await expect(page.getByTestId('spec-dec')).toBeAttached();
                break;
              }
              case 'offload': {
                const controls = page.getByTestId('offload-section');
                await reveal(page, 'Hardware', controls);
                await reveal(page, 'Advanced', page.getByLabel('Queue depth', { exact: true }));
                // This fieldset is taller than a phone viewport: keep its title and enabled expert
                // controls in frame rather than centering its middle and clipping their labels.
                await controls.evaluate((element) => {
                  element.scrollIntoView({ block: 'start', inline: 'nearest' });
                  for (let parent = element.parentElement; parent; parent = parent.parentElement) {
                    if (
                      parent.scrollHeight > parent.clientHeight &&
                      /auto|scroll/.test(getComputedStyle(parent).overflowY)
                    ) {
                      parent.scrollTop -= 96;
                      return;
                    }
                  }
                  window.scrollBy(0, -96);
                });
                await expect(
                  page.getByRole('combobox', { name: 'Routed experts', exact: true }),
                ).toHaveText('Host RAM');
                break;
              }
              case 'chart': {
                // data-chart is a chart identity hook, not a presentation class.
                const chart = page.locator('[data-chart="roofline"]');
                await center(chart);
                const point = chart.getByRole('button').first();
                await point.hover({ force: true });
                await point.focus();
                await expect(chart.getByRole('tooltip')).toBeVisible();
                await expect(chart.getByRole('tooltip')).toContainText('FLOP/B');
                break;
              }
              case 'warning': {
                await expect(verdict).toHaveClass(/memory-warning/);
                await center(verdict);
                await expect(
                  page.getByRole('button', { name: 'About High memory utilisation' }),
                ).toBeVisible();
                break;
              }
              case 'danger':
                await expect(verdict).toHaveClass(/memory-danger/);
                await center(verdict);
                break;
            }
          }
          expect(
            await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
          ).toBe(true);
          await page.screenshot({
            path: `${DIR}/${state.name}-${size}-${colorScheme}.png`,
            fullPage: false,
          });
        });
      }
    });
  }
}
