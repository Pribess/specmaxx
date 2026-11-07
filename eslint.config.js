import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'coverage'] },
  {
    files: ['**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { ecmaVersion: 2022, globals: globals.browser },
  },
  {
    // The simulation core is pure TypeScript: no React, no DOM, plain data + functions only.
    files: ['src/sim/**/*.ts'],
    rules: {
      'no-restricted-imports': ['error', { patterns: ['react', 'react-dom', 'react/*', 'react-dom/*'] }],
      'no-restricted-globals': ['error', 'window', 'document', 'localStorage', 'sessionStorage', 'fetch', 'navigator'],
      'no-restricted-syntax': [
        'error',
        { selector: 'ClassDeclaration', message: 'Plain data + functions only in src/sim.' },
      ],
    },
  },
);
