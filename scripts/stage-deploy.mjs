// Stages the production build for Workers Static Assets: copies dist/ to deploy/specmaxx/. The Worker serves
// requests for cro.sh/specmaxx* and matches asset paths against the full request path, so the files must sit under
// the same `/specmaxx/` prefix (assets.directory in wrangler.jsonc is ./deploy). Run `bun run build` first.
import { cpSync, existsSync, rmSync } from 'node:fs';

if (!existsSync('dist/index.html')) throw new Error('dist/ is missing: run `bun run build` first');
rmSync('deploy', { recursive: true, force: true });
cpSync('dist', 'deploy/specmaxx', { recursive: true });
console.log('staged dist/ -> deploy/specmaxx/');
