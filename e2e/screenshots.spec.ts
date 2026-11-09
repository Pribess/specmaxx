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
    test(`deployment knobs ${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const name = (s: string) => `${DIR}/deployment-${s}-${size}-${colorScheme}.png`;
      const panel = (title: string) =>
        page.locator('details.panel', { has: page.locator('> summary', { hasText: title }) });
      /** Loads a query, opens the given panels (collapsed on phones), and waits for numbers. */
      const load = async (query: string, panels: string[] = []) => {
        await page.goto(`/${query}`);
        await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
        for (const title of panels) {
          const p = panel(title);
          if ((await p.getAttribute('open')) === null) await p.locator('> summary').click();
        }
        // The sticky duplicate rates would overlay the element crops.
        await page.addStyleTag({ content: '.compact-results { visibility: hidden !important }' });
        await page.evaluate(() => document.fonts.ready);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
          true,
        );
      };
      /**
       * Screenshots an element of the input band, which scrolls on its own on desktop: the viewport grows to fit the
       * element, the element is scrolled into view, and the page is clipped to its box.
       */
      const shoot = async (locator: ReturnType<typeof page.locator>, path: string) => {
        const before = page.viewportSize()!;
        const height = (await locator.boundingBox())!.height;
        await page.setViewportSize({
          width: before.width,
          height: Math.max(before.height, Math.ceil(height) + 120),
        });
        await locator.evaluate((el) => el.scrollIntoView({ block: 'center' }));
        await page.waitForTimeout(50);
        const box = (await locator.boundingBox())!;
        await page.screenshot({
          path,
          clip: { x: box.x - 4, y: box.y - 4, width: box.width + 8, height: box.height + 8 },
        });
        await page.setViewportSize(before);
      };

      // Speculative decoding: MTP k=3 on the default model (an MTP head with any depth), FP8 weights on
      // 2 × RTX 5090 with TP 2, so the model fits.
      await load('?w=fp8_e4m3&n=2&tp=2&sd=mtp&sdk=3', ['Workload']);
      await expect(page.getByTestId('memory-verdict').first()).toHaveClass(/memory-fits/);
      await expect(page.getByTestId('spec-dec')).toBeVisible();
      await shoot(panel('Workload'), name('specdec-panel'));
      await page.evaluate(() => scrollTo(0, 0));
      await page.getByRole('button', { name: 'About Speculative decoding speedup' }).focus();
      await expect(
        page.getByRole('tooltip').filter({ hasText: 'tokens accepted per step' }),
      ).toBeVisible();
      await page.screenshot({ path: name('specdec-rate-help') });
      await page.keyboard.press('Escape');
      const breakdown = page.locator('section.breakdown');
      await breakdown.getByRole('button', { name: 'Layer', exact: true }).click();
      await expect(breakdown.getByTestId('breakdown-section').first()).toBeVisible();
      await breakdown.screenshot({ path: name('breakdown-draft-verify') });

      // The hypothetical drafter (no released DSpark drafter for this model).
      await load('?w=fp8_e4m3&n=2&tp=2&sd=dspark', ['Workload']);
      await expect(page.getByTestId('hypothetical-badge')).toBeVisible();
      await shoot(page.getByTestId('specdec-section'), name('specdec-hypothetical'));

      // Offload on dedicated memory: 2 × RTX 5090 (TP 2, so collectives exist), FP8 weights, 5% of the experts
      // resident and the rest in 12-channel DDR5 host RAM, the PLE table in host RAM.
      const offload =
        '?m=Qwen/Qwen3.8-Flash-Next&w=fp8_e4m3&n=2&hr=ddr5-6000-12ch&tp=2&ox=ram&oxf=0.05&oxh=fine&olt=ram';
      await load(offload, ['Hardware']);
      const section = page.getByTestId('offload-section');
      await shoot(section, name('offload-panel'));
      // The less common offload knobs are the Offload subsection of the Hardware panel's Advanced.
      await page.locator('details.advanced > summary').click();
      await shoot(page.getByTestId('offload-advanced'), name('offload-advanced'));
      await page.evaluate(() => scrollTo(0, 0));
      await expect(page.getByTestId('memory-tier').first()).toBeVisible();
      await page.locator('section.hero').screenshot({ path: name('offload-hero') });
      await breakdown.getByRole('button', { name: 'Kind', exact: true }).click();
      await breakdown.screenshot({ path: name('breakdown-comm-transfer') });

      // Host RAM + SSD split: DeepSeek-V4.1-Flash on one RTX 5090 with 256 GB of DDR5-5600 × 2: half of the expert
      // layers in host RAM and half on the SSD, and its two Engram table layers split one each: the split fields (main
      // section and Advanced) and both tier bars with their layer spans.
      await load('?m=deepseek-ai/DeepSeek-V4.1-Flash&hr=ddr5-5600-2ch&hrc=256&ox=split&olt=split', [
        'Hardware',
      ]);
      await expect(page.getByTestId('split-fields').first()).toBeVisible();
      await shoot(page.getByTestId('offload-section'), name('offload-split-panel'));
      await page.locator('details.advanced > summary').click();
      await shoot(page.getByTestId('offload-advanced'), name('offload-split-advanced'));
      await page.evaluate(() => scrollTo(0, 0));
      const memory = page.locator('section.memory');
      await expect(memory.getByTestId('memory-tier')).toHaveCount(2);
      await memory.scrollIntoViewIfNeeded();
      // No help popup left open from an earlier hover or focus.
      await page.mouse.move(0, 0);
      await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
      await expect(page.getByRole('tooltip')).toHaveCount(0);
      await memory.screenshot({ path: name('offload-split-memory') });

      // Unified memory: DGX Spark, experts on its SSD ("Storage").
      await load('?d=dgx-spark&m=Qwen/Qwen3.8-Flash-Next&ox=ssd', ['Hardware']);
      const storage = page.getByTestId('offload-section');
      await expect(storage.locator('legend')).toHaveText('Storage');
      await shoot(storage, name('storage-panel'));
      await page.locator('details.advanced > summary').click();
      await shoot(page.getByTestId('offload-advanced'), name('storage-advanced'));
      await page.evaluate(() => scrollTo(0, 0));
      await page.locator('section.hero').screenshot({ path: name('storage-hero') });
    });
    test(`token embeddings ${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const name = (s: string) => `${DIR}/embeddings-${s}-${size}-${colorScheme}.png`;
      const openAdvanced = async (query: string) => {
        await page.goto(`/${query}`);
        await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
        // Phones collapse the Hardware panel; the Advanced block is closed until opened.
        const hardware = page.locator('details.panel', {
          has: page.locator('> summary', { hasText: 'Hardware' }),
        });
        if ((await hardware.getAttribute('open')) === null)
          await hardware.locator('> summary').click();
        const advanced = page.locator('details.advanced');
        if ((await advanced.getAttribute('open')) === null)
          await advanced.locator('> summary').click();
        await page.addStyleTag({ content: '.compact-results { visibility: hidden !important }' });
        await page.evaluate(() => document.fonts.ready);
        const section = page.getByTestId('offload-advanced');
        await section.evaluate((el) => el.scrollIntoView({ block: 'center' }));
        return section;
      };
      // Enabled: the picker on its default, then on Host RAM (the host pickers appear) with its help open.
      let section = await openAdvanced('?m=Qwen/Qwen3.8-27B&oet=ram');
      await page.getByRole('button', { name: 'About Token embeddings' }).focus();
      await expect(
        page.getByRole('tooltip').filter({ hasText: 'turns tokens into vectors' }),
      ).toBeVisible();
      await page.screenshot({ path: name('advanced-enabled') });
      await page.keyboard.press('Escape');
      await section.screenshot({ path: name('advanced-enabled-section') });
      // Disabled: a tied model (the config served with tie_word_embeddings true); even an oet in the URL keeps it on the GPU.
      await page.route(
        'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/**/config.json',
        async (route) => {
          const response = await route.fetch();
          const body = (await response.text()).replaceAll(
            '"tie_word_embeddings": false',
            '"tie_word_embeddings": true',
          );
          await route.fulfill({ response, body });
        },
      );
      // The Hub cache (sessionStorage) would serve the untied config again.
      await page.evaluate(() => sessionStorage.clear());
      section = await openAdvanced('?m=Qwen/Qwen3.8-27B&oet=ssd');
      await expect(section.getByRole('combobox', { name: 'Token embeddings' })).toBeDisabled();
      await page.getByRole('button', { name: 'About Token embeddings' }).focus();
      await expect(
        page.getByRole('tooltip').filter({ hasText: 'so it stays on the GPU' }),
      ).toBeVisible();
      await page.screenshot({ path: name('advanced-disabled-tied') });
      await page.keyboard.press('Escape');
      await section.screenshot({ path: name('advanced-disabled-tied-section') });
    });
    test(`parallelism moe ${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const name = (s: string) => `${DIR}/parallelism-${s}-${size}-${colorScheme}.png`;
      const panel = page.locator('details.panel', {
        has: page.locator('> summary', { hasText: 'Parallelism' }),
      });
      // A MoE model on 4 GPUs with TP 4 and EP 2: the EP field shows next to TP and PP.
      await page.goto('/?m=Qwen/Qwen3.8-Flash-Next&w=fp8_e4m3&n=4&tp=4&ep=2');
      await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
      if ((await panel.getAttribute('open')) === null) await panel.locator('> summary').click();
      await expect(panel.getByText('Valid EP: 1, 2, 4')).toBeVisible();
      // The sticky duplicate rates would overlay the element crop.
      await page.addStyleTag({ content: '.compact-results { visibility: hidden !important }' });
      await page.evaluate(() => document.fonts.ready);
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
        true,
      );
      await panel.evaluate((el) => el.scrollIntoView({ block: 'center' }));
      await page.waitForTimeout(50);
      await panel.screenshot({ path: name('panel') });
      // The EP help, on keyboard focus.
      await panel.getByRole('button', { name: 'About EP' }).focus();
      await expect(page.getByRole('tooltip').filter({ hasText: 'must divide TP' })).toBeVisible();
      await page.screenshot({ path: name('ep-help') });
    });
    test(`parallelism dp ${size} ${colorScheme}`, async ({ page }) => {
      await open(page, size, colorScheme);
      const name = (s: string) => `${DIR}/parallelism-dp-${s}-${size}-${colorScheme}.png`;
      const panel = page.locator('details.panel', {
        has: page.locator('> summary', { hasText: 'Parallelism' }),
      });
      const shoot = async (query: string, valid: string, file: string) => {
        await page.goto(query);
        await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
        if ((await panel.getAttribute('open')) === null) await panel.locator('> summary').click();
        await expect(panel.getByText(valid)).toBeVisible();
        await page.addStyleTag({ content: '.compact-results { visibility: hidden !important }' });
        await page.evaluate(() => document.fonts.ready);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
          true,
        );
        await panel.evaluate((el) => el.scrollIntoView({ block: 'center' }));
        await page.waitForTimeout(50);
        await panel.screenshot({ path: name(file) });
      };
      // A dense model on 4 GPUs as TP 2 × DP 2: TP, PP and DP in one row; the rig line names DP.
      await shoot('/?m=Qwen/Qwen3.8-27B&n=4&tp=2&dp=2&b=2', 'Valid DP: 1, 2, 3, 4', 'dense-panel');
      await expect(page.locator('h1.rig')).toContainText('with TP 2, DP 2');
      // A MoE model as TP 2 × DP 2 with EP 4 spanning both copies (mode B).
      await shoot(
        '/?m=Qwen/Qwen3.8-Flash-Next&w=fp8_e4m3&n=4&tp=2&dp=2&ep=4&b=2',
        'Valid EP: 1, 2, 4',
        'moe-panel',
      );
      // The DP help, on keyboard focus.
      await panel.getByRole('button', { name: 'About DP' }).focus();
      await expect(page.getByRole('tooltip').filter({ hasText: 'full copies' })).toBeVisible();
      await page.screenshot({ path: name('help') });
    });
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
      if (size === 'phone')
        await page.locator('details.panel > summary', { hasText: 'Quantization' }).click();
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
