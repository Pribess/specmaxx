import { describe, expect, it } from 'vitest';
import { mapHfModel, type FamilyEntry } from '../sim/models/hf';
import { REF_DENSE_8B } from '../sim/models/reference';
import { fixtureStem, readFixtureJson } from '../sim/models/__fixtures__/fixtures.testutil';
import { createCache } from './cache';
import { isHfError } from './errors';
import { fetchHfRepoFiles } from './fetch';
import { loadHfModel } from './load';
import { memoryStorage } from './storage';
import { apiUrl, resolveUrl } from './urls';

describe('loadHfModel', () => {
  const make = (config: unknown) =>
    (async (url: string) => {
      if (url === apiUrl('a/b')) return new Response('{"siblings":[]}');
      if (url === resolveUrl('a/b', 'config.json')) return new Response(JSON.stringify(config));
      return new Response('', { status: 404 });
    }) as unknown as typeof fetch;
  const opts = (f: typeof fetch) => ({
    fetchImpl: f,
    cache: createCache(memoryStorage(), new Map()),
    tokenStorage: memoryStorage(),
  });

  it('returns an unsupported result for an unknown model_type', async () => {
    const r = await loadHfModel('a/b', opts(make({ model_type: 'nope_xyz' })));
    expect(r.kind).toBe('unsupported');
  });

  it('passes fetch errors through', async () => {
    const f = (async () => new Response('', { status: 401 })) as unknown as typeof fetch;
    const r = await loadHfModel('a/b', opts(f));
    expect(r.kind).toBe('unauthorized');
  });
});

describe('loadHfModel through a gated mirror', () => {
  const official = 'meta-models/Muse-Spark-Open-816B-A42B';
  const mirror = 'unsloth/Muse-Spark-Open-816B-A42B';

  /** The Hub as seen without a token: the official repo answers 401, the mirror serves its checked-in fixtures. */
  const gatedHub = (async (url: string, init?: RequestInit) => {
    if ((init?.headers as Record<string, string> | undefined)?.Authorization) return new Response('', { status: 500 });
    if (url.includes(official)) return new Response('', { status: 401 });
    const kind = url === apiUrl(mirror) ? 'api' : url === resolveUrl(mirror, 'config.json') ? 'config' : null;
    const body = kind && readFixtureJson(`${fixtureStem(mirror)}.${kind}.json`);
    return body ? new Response(JSON.stringify(body)) : new Response('', { status: 404 });
  }) as unknown as typeof fetch;
  const opts = () => ({ fetchImpl: gatedHub, cache: createCache(memoryStorage(), new Map()), tokenStorage: memoryStorage() });

  /** Mentions of the mirror a user could see; none may appear (the user's instruction). */
  const MIRROR_TEXT = /read from|unsloth\//i;

  it('maps the mirror config under the official id, with the mirror kept as data only', async () => {
    // A stand-in muse_spark family until the real one exists, so this pins the loader path on its own.
    const stub: FamilyEntry = {
      family: 'muse_spark',
      modelTypes: ['muse_spark', 'muse_spark_text'],
      architectures: ['MuseSparkForConditionalGeneration'],
      build: (_text, _top, files) => ({
        kind: 'ok',
        model: { ...REF_DENSE_8B, name: files.id, source: files.id },
        notes: [],
        mtpParamsEstimate: 0,
        visionParamsEstimate: 0,
        excludedParts: [],
      }),
    };
    const files = await fetchHfRepoFiles(official, opts());
    if (isHfError(files)) throw new Error(files.message);
    const r = mapHfModel(files, [stub]);
    if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
    expect(r.model.name).toBe(official);
    expect(r.model.source).toBe(official);
    expect(r.configFrom).toBe(mirror);
    expect(r.metadataFrom).toBe(mirror);
    expect(r.notes.filter((n) => MIRROR_TEXT.test(n))).toEqual([]);
  });

  it('loads Muse Spark from the mirror with the real family, with no mirror note', async () => {
    const r = await loadHfModel(official, opts());
    if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
    expect(r.family).toBe('muse_spark');
    expect(r.model.name).toBe(official);
    expect(r.model.source).toBe(official);
    expect(r.notes.filter((n) => MIRROR_TEXT.test(n))).toEqual([]);
  });
});
