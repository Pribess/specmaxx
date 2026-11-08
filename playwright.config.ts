import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright runs against the production build (`vite preview`), Chromium only. Two projects:
 * - `smoke` (`bun run e2e`, part of `bun run check`): the app loads and computes in the real worker.
 * - `screenshots` (`bun run screenshots`): writes docs/screenshots/*.png.
 * The HF Hub is mocked per test (no real network). `@playwright/test` is pinned to 1.64.1, whose
 * chromium-headless-shell revision (1248) is the one installed locally.
 */
export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: 'list',
  use: { baseURL: 'http://localhost:4173', ...devices['Desktop Chrome'] },
  webServer: {
    command: 'bunx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: 'smoke', testMatch: /smoke\.spec\.ts/ },
    { name: 'screenshots', testMatch: /screenshots\.spec\.ts/ },
  ],
});
