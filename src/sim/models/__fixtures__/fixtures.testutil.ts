/**
 * Test-only loader for the Hugging Face fixtures in this directory (see README.md).
 *
 * Files are imported as raw text through Vite's `import.meta.glob` and parsed with `JSON.parse`, so
 * TypeScript never infers literal types for MB-sized JSON. Needs vite types, so tsconfig.sim.json excludes
 * `*.testutil.ts`; only tests import it.
 *
 * Naming: `<org>__<repo>.<file>.json`, where `<file>` is `config`, `api` (trimmed `/api/models/<id>`),
 * `params`, `hf_quant_config` or `quantization_config`. Sub-directories (e.g. `quant/`) use the same scheme.
 */

import type { HfApiModelInfo, HfRepoFiles } from '../hf/types';

/** Raw fixture text by path relative to this directory, e.g. `'./Qwen__Qwen3.8-27B.config.json'`. */
const RAW: Record<string, string> = import.meta.glob<string>('./**/*.json', {
  query: '?raw',
  import: 'default',
  eager: true,
});

/** File stem of a repo ID: `'Qwen/Qwen3.8-27B'` → `'Qwen__Qwen3.8-27B'`. */
export function fixtureStem(repoId: string): string {
  return repoId.replace('/', '__');
}

/** Parsed fixture at `relPath` (relative to this directory, with or without `./`); undefined if absent. */
export function readFixtureJson(relPath: string): unknown {
  const key = relPath.startsWith('./') ? relPath : `./${relPath}`;
  const raw = RAW[key];
  return raw === undefined ? undefined : (JSON.parse(raw) as unknown);
}

/** Every fixture path, relative to this directory (with `./`), sorted. */
export function listFixturePaths(): string[] {
  return Object.keys(RAW).sort();
}

/** Repo IDs that have a `.config.json` fixture in `dir` (`''` = this directory, `'quant'` = `quant/`). */
export function listFixtureRepos(dir = ''): string[] {
  const prefix = dir ? `./${dir}/` : './';
  return listFixturePaths()
    .filter((p) => p.startsWith(prefix) && p.endsWith('.config.json') && !p.slice(prefix.length).includes('/'))
    .map((p) => p.slice(prefix.length, -'.config.json'.length).replace('__', '/'));
}

/**
 * Assemble {@link HfRepoFiles} for `repoId` from the fixtures in `dir`.
 * With `configFrom`, config.json is read from that repo's fixture (repos without one, e.g. Mistral NVFP4).
 * Throws when the config fixture is missing (a test bug, not an input case).
 */
export function fixtureRepoFiles(repoId: string, opts: { dir?: string; configFrom?: string } = {}): HfRepoFiles {
  const base = opts.dir ? `${opts.dir}/` : '';
  const file = (id: string, kind: string) => readFixtureJson(`${base}${fixtureStem(id)}.${kind}.json`);
  const configRepo = opts.configFrom ?? repoId;
  const config = file(configRepo, 'config') ?? readFixtureJson(`${fixtureStem(configRepo)}.config.json`);
  if (config === undefined) throw new Error(`fixture missing: ${base}${fixtureStem(configRepo)}.config.json`);
  const api = file(repoId, 'api') as HfApiModelInfo | undefined;
  return {
    id: repoId,
    revision: api?.sha ?? 'main',
    config,
    ...(opts.configFrom ? { configFrom: opts.configFrom } : {}),
    ...(api ? { api } : {}),
    ...optional('hfQuantConfig', file(repoId, 'hf_quant_config')),
    ...optional('paramsJson', file(repoId, 'params')),
    ...optional('quantizationConfigJson', file(repoId, 'quantization_config')),
  };
}

/** `{ [key]: value }` when value is defined, else `{}` (keeps absent files absent). */
function optional<K extends string>(key: K, value: unknown): Partial<Record<K, unknown>> {
  return value === undefined ? {} : ({ [key]: value } as Record<K, unknown>);
}
