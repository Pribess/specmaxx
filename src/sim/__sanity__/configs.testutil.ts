/**
 * Test-only loader for the sanity-check model configs: `__sanity__/configs/<org>__<repo>.<file>.json` (same naming
 * and trimming rules as `models/__fixtures__/README.md`), falling back to the shared fixtures in
 * `models/__fixtures__/`. Needs vite types (`import.meta.glob`), hence `.testutil.ts` (excluded from
 * tsconfig.sim.json). Only models of a family the HF loader maps can be loaded this way; see harness.ts. Block drafters
 * (speculative decoding) come from the drafter fixtures in `models/hf/__fixtures__/`.
 */

import type { BlockDraftSpec } from '../ir/model';
import { fixtureRepoFiles, fixtureStem } from '../models/__fixtures__/fixtures.testutil';
import { parseDrafterConfig } from '../models/hf/drafter';
import type { HfApiModelInfo, HfRepoFiles, JsonObject } from '../models/hf/types';

/** Raw config text by path relative to this directory, e.g. `'./configs/Qwen__Qwen3.8-27B.config.json'`. */
const RAW: Record<string, string> = import.meta.glob<string>('./configs/*.json', {
  query: '?raw',
  import: 'default',
  eager: true,
});

/** Parsed `configs/<stem>.<kind>.json`, or undefined if absent. */
function readConfig(stem: string, kind: string): unknown {
  const raw = RAW[`./configs/${stem}.${kind}.json`];
  return raw === undefined ? undefined : (JSON.parse(raw) as unknown);
}

/**
 * {@link HfRepoFiles} of `repoId`: from `configs/` when it has `<stem>.config.json`, else from the shared fixtures
 * (`fixtureRepoFiles`, in `models/__fixtures__/<dir>` when `dir` is given, e.g. `'quant'` for quantized repos; it
 * throws when that is missing too).
 */
export function sanityRepoFiles(repoId: string, opts: { dir?: string } = {}): HfRepoFiles {
  const stem = fixtureStem(repoId);
  const config = readConfig(stem, 'config');
  if (config === undefined) return fixtureRepoFiles(repoId, opts);
  const api = readConfig(stem, 'api') as HfApiModelInfo | undefined;
  return { id: repoId, revision: api?.sha ?? 'main', config, ...(api ? { api } : {}) };
}

/** Raw drafter configs by path, e.g. `'../models/hf/__fixtures__/incoai__Qwen3.8-27B-DFlash2.config.json'`. */
const DRAFTER_RAW: Record<string, string> = import.meta.glob<string>(
  '../models/hf/__fixtures__/*.config.json',
  {
    query: '?raw',
    import: 'default',
    eager: true,
  },
);

/**
 * The block drafter of `repoId` (a separate drafter repo, e.g. `'incoai/Qwen3.8-27B-DFlash2'`), parsed by
 * `parseDrafterConfig` from its saved `config.json` in `models/hf/__fixtures__/`. Throws when the fixture is missing
 * or does not parse.
 */
export function sanityDrafterSpec(repoId: string): BlockDraftSpec {
  const raw = DRAFTER_RAW[`../models/hf/__fixtures__/${fixtureStem(repoId)}.config.json`];
  if (raw === undefined) throw new Error(`no drafter fixture for ${repoId}`);
  const r = parseDrafterConfig(JSON.parse(raw) as JsonObject, repoId);
  if (r.kind !== 'ok') throw new Error(`${repoId}: ${r.kind}`);
  return r.spec;
}
