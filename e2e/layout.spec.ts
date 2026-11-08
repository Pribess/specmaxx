import { expect, test, type Locator } from '@playwright/test';
import { mockHf } from './hf-mock';

/** Top edge of an element in page coordinates (document scroll included). */
async function top(locator: Locator) {
  return locator.evaluate((el) => el.getBoundingClientRect().top + window.scrollY);
}

test('desktop: output left 2/3, inputs right 1/3, draggable splitter that persists and resets', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const output = page.locator('.workspace > .output');
  const inputs = page.locator('#inputs-pane');
  const splitter = page.getByRole('separator', { name: 'Resize inputs' });
  const outputBox = (await output.boundingBox())!;
  const inputsBox = (await inputs.boundingBox())!;
  expect(outputBox.x).toBeLessThan(inputsBox.x);
  expect(inputsBox.width / 1440).toBeCloseTo(1 / 3, 1);
  // The hero figures and the roofline are above the fold.
  for (const subject of [
    page.getByTestId('prefill-tps'),
    page.locator('[data-chart="roofline"] svg'),
  ]) {
    const box = (await subject.boundingBox())!;
    expect(box.y + box.height).toBeLessThan(900);
  }
  // The panes scroll on their own: the document does not.
  expect(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight)).toBe(
    true,
  );

  // Drag the splitter 240px to the left: the inputs pane widens, and the width survives a reload.
  const handle = (await splitter.boundingBox())!;
  await page.mouse.move(handle.x + handle.width / 2, 450);
  await page.mouse.down();
  await page.mouse.move(handle.x - 240, 450, { steps: 6 });
  await page.mouse.up();
  const widened = (await inputs.boundingBox())!.width;
  expect(widened).toBeGreaterThan(inputsBox.width + 200);
  await page.reload();
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  expect(Math.abs((await inputs.boundingBox())!.width - widened)).toBeLessThan(2);

  // Keyboard: ArrowRight narrows it; End clamps at half the width; double-click resets to 1/3.
  await splitter.focus();
  await page.keyboard.press('End');
  await expect(splitter).toHaveAttribute('aria-valuenow', '50');
  await splitter.dblclick();
  await expect(splitter).toHaveAttribute('aria-valuenow', '33');
  expect(await page.evaluate(() => localStorage.getItem('specmaxx:split'))).toBeNull();
});

test('desktop: the mini result bar appears once the hero scrolls out of the output pane', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const mini = page.getByTestId('mini-result');
  await expect(mini).toBeHidden();
  await page.locator('[data-chart="sweep"]').scrollIntoViewIfNeeded();
  await expect(mini).toBeVisible();
  await expect(mini).toContainText(/Prefill.*tok\/s @ 8k.*Decode.*tok\/s @ 8k/);
});

test('phone: headline, figures and memory first, then inputs, then the analysis', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await mockHf(page);
  await page.goto('/');
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  await expect(page.getByRole('separator')).toHaveCount(0);
  const sequence = [
    page.locator('.rig'),
    page.getByTestId('decode-tps'),
    page.getByTestId('memory-verdict'),
    page.locator('#inputs-pane'),
    page.locator('[data-chart="roofline"]'),
    page.getByRole('heading', { name: /Bottleneck estimation/ }),
    page.locator('[data-chart="sweep"]'),
  ];
  const tops = [];
  for (const subject of sequence) tops.push(await top(subject));
  expect([...tops].sort((a, b) => a - b)).toEqual(tops);
  // The figures are on the first screen; the mini bar takes over once they scroll away.
  expect(tops[1]).toBeLessThan(844);
  const mini = page.getByTestId('mini-result');
  await expect(mini).toBeHidden();
  await page.locator('#inputs-pane').scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, 300));
  await expect(mini).toBeVisible();
  // It slides in from 6px above (no motion when reduced motion is asked for); it settles at the window's top.
  await expect.poll(async () => (await mini.boundingBox())!.y).toBe(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
