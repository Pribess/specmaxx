import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative asset URLs, so the same build works at a site root (specmaxx.pages.dev/) and under a path prefix
  // (cro.sh/specmaxx/); see docs/deploy.md.
  base: './',
  plugins: [react()],
  // Module workers (`new Worker(new URL(...), { type: 'module' })`) are bundled as ES modules.
  worker: { format: 'es' },
  build: {
    // Two HTML entries: the estimator and the About page (renders docs/simulation-model.md).
    rollupOptions: { input: { main: 'index.html', about: 'about.html' } },
  },
  test: {
    // Unit and component tests only; Playwright specs live in e2e/*.spec.ts and never match.
    // The environment is node; component tests opt in with a `// @vitest-environment jsdom` docblock.
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'node',
    // Polyfills jsdom lacks for Radix and cmdk (no-op in the node environment).
    setupFiles: ['src/test/setupDom.ts'],
  },
});
