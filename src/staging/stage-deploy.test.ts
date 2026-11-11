// @vitest-environment node
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const script = resolve(__dirname, '../../scripts/stage-deploy.mjs');

function stage(assetNames: string[]): string {
  const dir = mkdtempSync(join(tmpdir(), 'stage-'));
  mkdirSync(join(dir, 'dist/assets'), { recursive: true });
  writeFileSync(join(dir, 'dist/index.html'), '');
  for (const f of assetNames) writeFileSync(join(dir, 'dist/assets', f), '');
  execFileSync('node', [script], { cwd: dir, stdio: 'pipe' });
  return readFileSync(join(dir, 'deploy/_headers'), 'utf8');
}

describe('stage-deploy _headers', () => {
  it('makes hashed assets immutable and keeps HTML revalidated', () => {
    const h = stage(['index-Ab3_dE-9.js', 'font-12345678.woff2']);
    expect(h).toContain('/specmaxx/assets/*\n  Cache-Control: public, max-age=31536000, immutable');
    expect(h).toContain('/specmaxx/\n  Cache-Control: public, max-age=0, must-revalidate');
    expect(h).toContain(
      '/specmaxx/about.html\n  Cache-Control: public, max-age=0, must-revalidate',
    );
  });

  it('excludes an unhashed asset from the immutable rule', () => {
    const h = stage(['index-Ab3_dE-9.js', 'plain.js']);
    expect(h).not.toContain('/specmaxx/assets/*');
    expect(h).toContain(
      '/specmaxx/assets/index-Ab3_dE-9.js\n  Cache-Control: public, max-age=31536000, immutable',
    );
    expect(h).not.toContain('plain.js');
  });
});
