import { expect, test, type Page } from '@playwright/test';
import { mockHf } from './hf-mock';

/**
 * The "?" help popups open on hover and focus, stay open when pinned by a click, and close on a click outside or
 * on Escape (the Network link help of a TCP link is the rig here).
 */
const URL = '/?m=Qwen/Qwen3.8-Flash-Next&nodes=2&tp=2';

async function openRig(page: Page) {
  await mockHf(page);
  await page.goto(URL);
  await expect(page.getByTestId('decode-tps')).toHaveText(/[1-9]/);
  const button = page.getByRole('button', { name: 'About Network link' });
  await expect(button).toBeVisible();
  return button;
}

test('a help popup opens on hover and closes after the pointer leaves', async ({ page }) => {
  const button = await openRig(page);
  const tip = page.getByRole('tooltip').filter({ hasText: 'one way' });
  await button.hover();
  await expect(tip).toBeVisible();
  await page.mouse.move(5, 300);
  await expect(tip).toBeHidden();
});

test('a click pins a help popup; a click outside closes it', async ({ page }) => {
  const button = await openRig(page);
  const tip = page.getByRole('tooltip').filter({ hasText: 'one way' });
  await button.click();
  await expect(tip).toBeVisible();
  await page.mouse.move(5, 300);
  await expect(tip).toBeVisible();
  await page.mouse.click(5, 300);
  await expect(tip).toBeHidden();
});

test('a help popup opens on keyboard focus; Escape closes it and keeps focus on the icon', async ({
  page,
}) => {
  const button = await openRig(page);
  const tip = page.getByRole('tooltip').filter({ hasText: 'one way' });
  await button.focus();
  await expect(tip).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(tip).toBeHidden();
  await expect(button).toBeFocused();
});
