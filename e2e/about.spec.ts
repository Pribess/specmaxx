import { expect, test, type Page } from '@playwright/test';

/**
 * The About page ships the method prerendered and fetches the reference documents (hashed `.htm` assets) on demand:
 * when the reference opens, or when the URL hash points into one of them.
 */

/** Records the URLs of the lazily loaded reference documents the page requests. */
function docRequests(page: Page): string[] {
  const urls: string[] = [];
  page.on('request', (request) => {
    if (/\/assets\/(simulation-model|sanity-check)-[\w-]{8}\.htm$/.test(request.url()))
      urls.push(request.url());
  });
  return urls;
}

/** The table scroll regions inside one article: themed Radix viewports, keyboard focusable. */
function tableRegions(page: Page, article: string) {
  return page.locator(
    `article#${article} [data-radix-scroll-area-viewport][role="region"][tabindex="0"]:has(table)`,
  );
}

test('About loads the reference on opening it, with every table in a scroll area', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const requests = docRequests(page);
  await page.goto('/about.html');
  await expect(page.getByRole('heading', { name: 'What the numbers mean' })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Modeling method contents' })).toBeVisible();
  const details = page.locator('details#full-reference');
  await expect(details).not.toHaveAttribute('open');
  expect(requests).toEqual([]);

  await page.locator('details#full-reference > summary').click();
  await expect(page.getByRole('heading', { name: '4. Hardware', exact: true })).toBeAttached();
  expect(await tableRegions(page, 'simulation-model').count()).toBeGreaterThan(10);
  expect(await page.locator('article#simulation-model table').count()).toBe(
    await tableRegions(page, 'simulation-model').count(),
  );
  expect(requests.some((url) => url.includes('/simulation-model-'))).toBe(true);
  expect(errors).toEqual([]);
});

test('a deep link into the reference loads it, opens it and scrolls to the heading', async ({
  page,
}) => {
  const requests = docRequests(page);
  const warnings: string[] = [];
  page.on('console', (message) => {
    if (/preload/i.test(message.text())) warnings.push(message.text());
  });
  await page.goto('/about.html#4-hardware');
  // The head script preloads the document before the page's JS runs, with the fetch's mode and credentials.
  await expect(page.locator('head link[rel="preload"][as="fetch"]')).toHaveAttribute(
    'href',
    /^\.\/assets\/simulation-model-[\w-]{8}\.htm$/,
  );
  await expect(page.locator('head link[rel="preload"][as="fetch"]')).toHaveAttribute(
    'crossorigin',
    'anonymous',
  );
  await expect(page.locator('details#full-reference')).toHaveAttribute('open');
  const heading = page.locator('[id="4-hardware"]');
  await expect(heading).toBeInViewport();
  await expect(tableRegions(page, 'simulation-model').first()).toBeAttached();
  expect(await tableRegions(page, 'simulation-model').count()).toBeGreaterThan(10);
  // The scroll areas render their tables (the viewport has the table's height, not zero).
  const box = await tableRegions(page, 'simulation-model').first().boundingBox();
  expect(box?.height ?? 0).toBeGreaterThan(40);
  // The fetch reuses the preloaded response, and the other document follows once: one request each.
  await expect(page.locator('article#sanity-check h1')).toBeAttached();
  expect(requests.filter((url) => url.includes('/simulation-model-'))).toHaveLength(1);
  expect(requests.filter((url) => url.includes('/sanity-check-'))).toHaveLength(1);
  expect(requests[0]).toContain('/simulation-model-');
  expect(warnings).toEqual([]);
});

test('a link from the method into the reference follows the hash change', async ({ page }) => {
  await page.goto('/about.html');
  await expect(page.locator('details#full-reference')).not.toHaveAttribute('open');
  await page.locator('article#modeling-method a[href="#memory-fit"]').first().click();
  await expect(page).toHaveURL(/#memory-fit$/);
  await expect(page.locator('details#full-reference')).toHaveAttribute('open');
  await expect(page.locator('[id="memory-fit"]')).toBeInViewport();
});
