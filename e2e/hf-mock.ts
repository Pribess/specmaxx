import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import type { Page } from '@playwright/test';
import { GATED_MIRRORS } from '../src/hf/mirrors';

/**
 * Fixture directories, searched in order: target models, then drafter configs (kept apart because the identity pin
 * maps every target fixture as a model).
 */
const FIXTURE_DIRS = ['src/sim/models/__fixtures__', 'src/sim/models/hf/__fixtures__'];

/**
 * Serve checked-in fixtures only; every other HF request is a deterministic 404. Repos with a known ungated mirror
 * (`GATED_MIRRORS`) are gated like on the Hub: 401 without an `Authorization` header.
 */
export async function mockHf(page: Page) {
  await page.route('https://huggingface.co/**', async (route) => {
    const path = decodeURIComponent(new URL(route.request().url()).pathname);
    const api = path.match(/^\/api\/models\/([^/]+\/[^/]+)(?:\/revision\/[^/]+)?$/);
    const config = path.match(/^\/([^/]+\/[^/]+)\/resolve\/[^/]+\/config.json$/);
    const id = api?.[1] ?? config?.[1];
    if (
      id &&
      Object.hasOwn(GATED_MIRRORS, id) &&
      !(await route.request().headerValue('authorization'))
    ) {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: '{"error":"gated"}',
      });
      return;
    }
    if (id && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(id)) {
      for (const dir of FIXTURE_DIRS) {
        try {
          const body = await readFile(
            resolve(dir, `${id.replace('/', '__')}.${api ? 'api' : 'config'}.json`),
            'utf8',
          );
          await route.fulfill({ status: 200, contentType: 'application/json', body });
          return;
        } catch {
          /* not in this directory: try the next, never the real Hub */
        }
      }
    }
    await route.fulfill({ status: 404, contentType: 'application/json', body: '{}' });
  });
}
