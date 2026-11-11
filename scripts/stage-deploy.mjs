// Stages the production build for Workers Static Assets: copies dist/ to deploy/specmaxx/. The Worker serves
// requests for cro.sh/specmaxx* and matches asset paths against the full request path, so the files must sit under
// the same `/specmaxx/` prefix (assets.directory in wrangler.jsonc is ./deploy). Run `bun run build` first.
//
// It also writes deploy/_headers (the file must sit at the root of the assets directory, not under specmaxx/):
// content-hashed files in assets/ are immutable for a year, HTML is revalidated on every visit.
import { cpSync, existsSync, readdirSync, rmSync, writeFileSync } from 'node:fs';

/** Vite names emitted files `name-<8 chars of base64url hash>.ext`. */
const HASHED = /-[A-Za-z0-9_-]{8}\.[A-Za-z0-9]+$/;
const IMMUTABLE = 'Cache-Control: public, max-age=31536000, immutable';
const REVALIDATE = 'Cache-Control: public, max-age=0, must-revalidate';

if (!existsSync('dist/index.html')) throw new Error('dist/ is missing: run `bun run build` first');
rmSync('deploy', { recursive: true, force: true });
cpSync('dist', 'deploy/specmaxx', { recursive: true });

const assets = existsSync('dist/assets') ? readdirSync('dist/assets') : [];
const unhashed = assets.filter((f) => !HASHED.test(f));
// All hashed: one splat rule. Otherwise list the hashed files one by one so unhashed ones keep revalidating.
const immutablePaths =
  unhashed.length === 0
    ? ['/specmaxx/assets/*']
    : assets.filter((f) => HASHED.test(f)).map((f) => `/specmaxx/assets/${f}`);
if (unhashed.length)
  console.warn(`not immutable (no content hash in name): ${unhashed.join(', ')}`);

const rules = [
  ...immutablePaths.map((p) => `${p}\n  ${IMMUTABLE}`),
  ...['/specmaxx/', '/specmaxx/index.html', '/specmaxx/about', '/specmaxx/about.html'].map(
    (p) => `${p}\n  ${REVALIDATE}`,
  ),
];
writeFileSync('deploy/_headers', rules.join('\n\n') + '\n');
console.log('staged dist/ -> deploy/specmaxx/ (+ deploy/_headers)');
