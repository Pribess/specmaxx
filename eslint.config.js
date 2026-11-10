import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

/** React and DOM storage/network globals banned where code must stay pure (src/sim, src/ui/state). */
const NO_REACT = ['error', { patterns: ['react', 'react-dom', 'react/*', 'react-dom/*'] }];
const NO_DOM_GLOBALS = [
  'error',
  'window',
  'document',
  'localStorage',
  'sessionStorage',
  'fetch',
  'navigator',
];

export default tseslint.config(
  { ignores: ['dist', 'deploy', 'coverage', 'test-results', 'playwright-report', 'blob-report'] },
  {
    files: ['**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { ecmaVersion: 2022, globals: globals.browser },
  },
  {
    files: ['src/**/*.tsx'],
    extends: [reactHooks.configs.flat.recommended],
  },
  {
    // Playwright specs and config files run in Node.
    files: ['e2e/**/*.ts', 'playwright.config.ts', 'vite.config.ts'],
    languageOptions: { globals: globals.node },
  },
  {
    // The simulation core is pure TypeScript: no React, no DOM, plain data + functions only.
    files: ['src/sim/**/*.ts'],
    rules: {
      'no-restricted-imports': NO_REACT,
      'no-restricted-globals': NO_DOM_GLOBALS,
      'no-restricted-syntax': [
        'error',
        { selector: 'ClassDeclaration', message: 'Plain data + functions only in src/sim.' },
      ],
    },
  },
  {
    // The UI state core is pure TypeScript too (it also runs in the worker); URL / URLSearchParams are allowed.
    files: ['src/ui/state/**/*.ts'],
    rules: {
      'no-restricted-imports': NO_REACT,
      'no-restricted-globals': NO_DOM_GLOBALS,
    },
  },
);
