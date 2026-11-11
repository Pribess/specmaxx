import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import { prerenderAboutDocs } from './vite/prerenderDocs.ts';
import { stripProvenance } from './vite/stripProvenance.ts';

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
  plugins: [
    react(),
    stripProvenance(),
    prerenderAboutDocs(),
    {
      name: 'preload-sim-worker',
      transformIndexHtml: {
        order: 'post',
        handler(_html, context) {
          if (context.path !== '/index.html' || !context.bundle) return [];
          const worker = Object.values(context.bundle).find((file) =>
            /^assets\/sim\.worker-.*\.js$/.test(file.fileName),
          );
          if (!worker) throw new Error('Missing simulation worker asset');
          // Cold Slow 4G: ~550ms earlier results for ~300ms FCP. Low/high measured alike;
          // prefer low so the browser may favor render-critical resources on other links.
          return [
            {
              tag: 'link',
              attrs: { rel: 'modulepreload', href: `./${worker.fileName}`, fetchpriority: 'low' },
              injectTo: 'head',
            },
          ];
        },
      },
    },
  ],
  // Module workers (`new Worker(new URL(...), { type: 'module' })`) are bundled as ES modules.
  worker: {
    format: 'es',
    plugins: () => [stripProvenance()],
    rollupOptions: { treeshake: TREESHAKE },
  },
  build: {
    // Two HTML entries: the estimator and the About page (prerendered docs; see vite/prerenderDocs.ts).
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
