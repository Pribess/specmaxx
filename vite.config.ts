import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

/**
 * The simulation core and the UI state core are pure modules: no top-level effects, only declarations. Telling the
 * bundler so lets it drop what a chunk does not use out of barrel files (`export *`) instead of keeping every module
 * a barrel mentions: the worker no longer carries the hardware presets it never reads. Components and CSS stay as
 * the bundler's default (side effects assumed).
 */
const TREESHAKE = {
  moduleSideEffects: [{ test: /\/src\/(sim|ui\/state)\//, sideEffects: false }],
};

export default defineConfig({
  // Relative asset URLs, so the same build works at a site root (localhost:5173/) and under a path prefix
  // (cro.sh/specmaxx/); see docs/deploy.md.
  base: './',
  plugins: [react()],
  // Module workers (`new Worker(new URL(...), { type: 'module' })`) are bundled as ES modules.
  worker: { format: 'es', rollupOptions: { treeshake: TREESHAKE } },
  build: {
    // Two HTML entries: the estimator and the About page (renders docs/simulation-model.md).
    rollupOptions: { input: { main: 'index.html', about: 'about.html' }, treeshake: TREESHAKE },
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
