import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * The "?" help popups stay open while the pointer or focus is inside them, so their source links can be followed by
 * mouse and by keyboard (the Network link help of a TCP link has them). External pages are stubbed.
 */
const URL = '/?m=Qwen/Qwen3.8-Flash-Next&nodes=2&tp=2';
const SOURCE = /forum\.level1techs\.com/;

async function openRig(page: Page) {
  await mockHf(page);
  await page
    .context()
    .route(/^https:\/\/(forum\.level1techs\.com|github\.com)\//, (route) =>
      route.fulfill({ status: 200, contentType: 'text/html', body: '<title>source</title>' }),
    );
  await page.goto(URL);
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const button = page.getByRole('button', { name: 'About Network link' });
  await expect(button).toBeVisible();
  return button;
}

test('a help source link opens by mouse', async ({ page }) => {
  const button = await openRig(page);
  await button.click();
  const tip = page.getByRole('tooltip').filter({ hasText: 'one-way TCP floor' });
  await expect(tip).toBeVisible();
  const link = tip.getByRole('link', { name: 'source', exact: true }).first();
  const opened = page.context().waitForEvent('page');
  await link.click();
  const source = await opened;
  await expect.poll(() => source.url()).toMatch(SOURCE);
  await source.close();
  // A click outside closes the pinned popup.
  await page.mouse.click(5, 300);
  await expect(tip).toBeHidden();
});

test('a help source link is reached by Tab and opens with Enter; Escape closes and returns focus', async ({
  page,
}) => {
  const button = await openRig(page);
  await button.focus();
  const tip = page.getByRole('tooltip').filter({ hasText: 'one-way TCP floor' });
  await expect(tip).toBeVisible();
  await page.keyboard.press('Tab');
  const link = tip.getByRole('link', { name: 'source', exact: true }).first();
  await expect(link).toBeFocused();
  await expect(tip).toBeVisible();
  const opened = page.context().waitForEvent('page');
  await page.keyboard.press('Enter');
  const source = await opened;
  await expect.poll(() => source.url()).toMatch(SOURCE);
  await source.close();
  await page.bringToFront();
  await link.focus();
  await page.keyboard.press('Escape');
  await expect(tip).toBeHidden();
  await expect(button).toBeFocused();
});
