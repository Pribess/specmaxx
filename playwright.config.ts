import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright runs against the production build (`vite preview`), Chromium only. Three projects:
 * - `smoke` (`bun run e2e`, part of `bun run check`): the app loads and computes in the real worker; the fabric
 *   test compares a 4 × DGX Spark ring with the switch; the rack-scale test runs DP 64 × EP 64 on one GB200 NVL72
 *   domain vs 16 NVL4 trays over InfiniBand; the deployment test turns on speculative decoding and
 *   expert offload; the layout test checks the desktop split (splitter drag, keyboard, reset, persistence) and the
 *   phone stacking order; the gated-mirror test loads a gated shortlist model from its mirror without a token.
 * - `screenshots` (`bun run screenshots`): writes docs/screenshots/*.png (or to `$SCREENSHOTS_DIR`).
 * - `redesign` (`bun run screenshots:redesign`): viewport-only before/after states (`$REDESIGN_DIR`).
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
    {
      name: 'smoke',
      testMatch: /(smoke|fabric|rack-scale|deployment|layout|help|gated-mirror)\.spec\.ts/,
    },
    { name: 'screenshots', testMatch: /screenshots\.spec\.ts/ },
    { name: 'redesign', testMatch: /redesign\.spec\.ts/ },
  ],
});
