import { describe, expect, it } from 'vitest';
import { createCache } from './cache';
import { isHfError } from './errors';
import { fetchHfRepoFiles } from './fetch';
import { GATED_MIRRORS, gatedMirrorOf } from './mirrors';
import { memoryStorage } from './storage';
import { getHfToken, HF_TOKEN_KEY, setHfToken } from './token';
import { apiUrl, isValidModelId, resolveUrl } from './urls';

type Seen = { url: string; auth?: string };

/** Fake fetch serving `routes` (url → body object | status number). */
function fakeFetch(routes: Record<string, unknown>) {
  const seen: Seen[] = [];
  const impl = (async (url: string, init?: RequestInit) => {
    const headers = init?.headers as Record<string, string> | undefined;
    seen.push({ url, auth: headers?.Authorization });
    const r = routes[url];
    if (r === undefined) return new Response('nf', { status: 404 });
    if (typeof r === 'number') return new Response('x', { status: r });
    return new Response(JSON.stringify(r), { status: 200 });
  }) as unknown as typeof fetch;
  return { impl, seen };
}

const opts = (impl: typeof fetch) => ({
  fetchImpl: impl,
  cache: createCache(memoryStorage(), new Map()),
  tokenStorage: memoryStorage(),
});

describe('urls', () => {
  it('builds urls and validates ids', () => {
    expect(resolveUrl('a/b', 'config.json')).toBe(
      'https://huggingface.co/a/b/resolve/main/config.json',
    );
    expect(resolveUrl('a/b', 'config.json', 'refs/pr/1')).toContain('/resolve/refs%2Fpr%2F1/');
    expect(apiUrl('a/b')).toBe('https://huggingface.co/api/models/a/b');
    expect(apiUrl('a/b', 'v1')).toBe('https://huggingface.co/api/models/a/b/revision/v1');
    expect(isValidModelId('Qwen/Qwen3.8-27B')).toBe(true);
    expect(isValidModelId('nope')).toBe(false);
    expect(isValidModelId('a/b/c')).toBe(false);
  });
});

describe('fetchHfRepoFiles', () => {
  const api = { siblings: [{ rfilename: 'config.json' }], safetensors: { total: 5 } };
  const routes = {
    [apiUrl('a/b')]: api,
    [resolveUrl('a/b', 'config.json')]: { model_type: 'x' },
  };

  it('rejects invalid ids without fetching', async () => {
    const { impl, seen } = fakeFetch(routes);
    const r = await fetchHfRepoFiles('bad', opts(impl));
    expect(isHfError(r) && r.kind).toBe('invalid_id');
    expect(seen).toHaveLength(0);
  });

  it('fetches api + config, no sidecar probing', async () => {
    const { impl, seen } = fakeFetch(routes);
    const r = await fetchHfRepoFiles('a/b', opts(impl));
    expect(isHfError(r)).toBe(false);
    if (isHfError(r)) return;
    expect(r.config).toEqual({ model_type: 'x' });
    expect(r.hfQuantConfig).toBeUndefined();
    expect(seen.map((s) => s.url)).toEqual([apiUrl('a/b'), resolveUrl('a/b', 'config.json')]);
  });

  it('maps 401, 403, 404, 429 and network errors', async () => {
    for (const [status, kind] of [
      [401, 'unauthorized'],
      [403, 'forbidden'],
      [404, 'not_found'],
      [429, 'rate_limited'],
    ] as const) {
      const { impl } = fakeFetch({ [apiUrl('a/b')]: status });
      const r = await fetchHfRepoFiles('a/b', opts(impl));
      expect(isHfError(r) && r.kind).toBe(kind);
      if (isHfError(r) && kind === 'unauthorized') expect(r.message).toMatch(/token/i);
    }
    const boom = (() =>
      Promise.reject(new TypeError('Failed to fetch'))) as unknown as typeof fetch;
    const r = await fetchHfRepoFiles('a/b', opts(boom));
    expect(isHfError(r) && r.kind).toBe('network');
  });

  it('errors clearly when the repo has no config.json', async () => {
    const { impl } = fakeFetch({ [apiUrl('a/b')]: { siblings: [] } });
    const r = await fetchHfRepoFiles('a/b', opts(impl));
    expect(isHfError(r) && r.kind).toBe('not_found');
    expect(isHfError(r) && r.message).toMatch(/no config\.json/);
  });

  it('never mistakes a JSON body with kind/message for an error', async () => {
    const { impl } = fakeFetch({
      [apiUrl('a/b')]: { siblings: [] },
      [resolveUrl('a/b', 'config.json')]: { kind: 'x', message: 'y', model_type: 'z' },
    });
    const r = await fetchHfRepoFiles('a/b', opts(impl));
    expect(isHfError(r)).toBe(false);
  });

  it('reports bad json', async () => {
    const impl = (async () => new Response('<html>', { status: 200 })) as unknown as typeof fetch;
    const r = await fetchHfRepoFiles('a/b', opts(impl));
    expect(isHfError(r) && r.kind).toBe('bad_json');
  });

  it('sends the explicit and the stored token as Bearer', async () => {
    const { impl, seen } = fakeFetch(routes);
    await fetchHfRepoFiles('a/b', { ...opts(impl), token: 'hf_abc' });
    expect(seen.every((s) => s.auth === 'Bearer hf_abc')).toBe(true);

    const o = opts(impl);
    setHfToken('hf_stored', o.tokenStorage);
    expect(o.tokenStorage.getItem(HF_TOKEN_KEY)).toBe('hf_stored');
    seen.length = 0;
    await fetchHfRepoFiles('a/b', o);
    expect(seen[0]?.auth).toBe('Bearer hf_stored');
    expect(getHfToken(o.tokenStorage)).toBe('hf_stored');
  });

  it('uses the revision in urls', async () => {
    const { impl, seen } = fakeFetch({
      [apiUrl('a/b', 'v2')]: api,
      [resolveUrl('a/b', 'config.json', 'v2')]: {},
    });
    const r = await fetchHfRepoFiles('a/b', { ...opts(impl), revision: 'v2' });
    expect(isHfError(r)).toBe(false);
    expect(seen).toHaveLength(2);
  });

  it('caches in memory/sessionStorage and does not cache errors', async () => {
    const { impl, seen } = fakeFetch(routes);
    const storage = memoryStorage();
    const o = {
      fetchImpl: impl,
      cache: createCache(storage, new Map()),
      tokenStorage: memoryStorage(),
    };
    await fetchHfRepoFiles('a/b', o);
    await fetchHfRepoFiles('a/b', o);
    expect(seen).toHaveLength(2);
    // fresh memory level, same storage: round trip through sessionStorage
    const cold = createCache(storage, new Map());
    expect(cold.get('specmaxx:hf:a/b@main:config.json')).toEqual({ text: '{"model_type":"x"}' });

    // 401 then success after token: errors not cached
    const calls: string[] = [];
    const flaky = (async (url: string, init?: RequestInit) => {
      calls.push(url);
      const auth = (init?.headers as Record<string, string> | undefined)?.Authorization;
      if (!auth) return new Response('', { status: 401 });
      return new Response(JSON.stringify(routes[url] ?? {}), { status: 200 });
    }) as unknown as typeof fetch;
    const s2 = memoryStorage();
    const m2 = {
      fetchImpl: flaky,
      cache: createCache(s2, new Map()),
      tokenStorage: memoryStorage(),
    };
    const first = await fetchHfRepoFiles('c/d', m2);
    expect(isHfError(first) && first.kind).toBe('unauthorized');
    const second = await fetchHfRepoFiles('c/d', { ...m2, token: 'hf_x' });
    expect(isHfError(second)).toBe(false);
  });

  it('tolerates a throwing storage', async () => {
    const { impl } = fakeFetch(routes);
    const bad = {
      getItem: () => {
        throw new Error('x');
      },
      setItem: () => {
        throw new Error('full');
      },
      removeItem: () => {},
    };
    const r = await fetchHfRepoFiles('a/b', {
      fetchImpl: impl,
      cache: createCache(bad, new Map()),
      tokenStorage: bad,
    });
    expect(isHfError(r)).toBe(false);
  });

  it('fetches listed sidecars only', async () => {
    const { impl, seen } = fakeFetch({
      [apiUrl('a/b')]: {
        siblings: [{ rfilename: 'config.json' }, { rfilename: 'hf_quant_config.json' }],
      },
      [resolveUrl('a/b', 'config.json')]: {},
      [resolveUrl('a/b', 'hf_quant_config.json')]: { quantization: { quant_algo: 'NVFP4' } },
    });
    const r = await fetchHfRepoFiles('a/b', opts(impl));
    if (isHfError(r)) throw new Error(r.message);
    expect(r.hfQuantConfig).toEqual({ quantization: { quant_algo: 'NVFP4' } });
    expect(seen.some((s) => s.url.endsWith('params.json'))).toBe(false);
  });

  it('falls back to base_model config for Mistral params.json repos', async () => {
    const { impl } = fakeFetch({
      [apiUrl('m/nvfp4')]: {
        siblings: [{ rfilename: 'params.json' }],
        cardData: { base_model: ['m/base'] },
      },
      [resolveUrl('m/nvfp4', 'params.json')]: { dim: 1 },
      [resolveUrl('m/base', 'config.json')]: { model_type: 'mistral_large4' },
    });
    const r = await fetchHfRepoFiles('m/nvfp4', opts(impl));
    if (isHfError(r)) throw new Error(r.message);
    expect(r.paramsJson).toEqual({ dim: 1 });
    expect(r.config).toEqual({ model_type: 'mistral_large4' });
    expect(r.configFrom).toBe('m/base');
  });
});

describe('fetchHfRepoFiles gated-mirror fallback', () => {
  const official = 'meta-models/Muse-Spark-Open-816B-A42B';
  const mirror = 'unsloth/Muse-Spark-Open-816B-A42B';
  const mirrorApi = {
    id: mirror,
    siblings: [{ rfilename: 'config.json' }],
    safetensors: { total: 7 },
  };
  const mirrorConfig = { model_type: 'muse_spark' };
  /** The official repo is gated (401 without a token), the mirror is public. */
  const gatedRoutes = {
    [apiUrl(official)]: 401,
    [resolveUrl(official, 'config.json')]: 401,
    [apiUrl(mirror)]: mirrorApi,
    [resolveUrl(mirror, 'config.json')]: mirrorConfig,
  };

  it('maps the mirror to its official repo', () => {
    expect(gatedMirrorOf(official)).toBe(mirror);
    expect(GATED_MIRRORS[official]).toBe(mirror);
    expect(gatedMirrorOf('a/b')).toBeUndefined();
    expect(gatedMirrorOf('toString')).toBeUndefined();
  });

  it('(a) without a token, reads config and metadata from the mirror under the official id', async () => {
    const { impl, seen } = fakeFetch(gatedRoutes);
    const r = await fetchHfRepoFiles(official, opts(impl));
    if (isHfError(r)) throw new Error(r.message);
    expect(seen.map((s) => s.url)).toEqual([
      apiUrl(official),
      apiUrl(mirror),
      resolveUrl(mirror, 'config.json'),
    ]);
    expect(seen.every((s) => s.auth === undefined)).toBe(true);
    expect(r.id).toBe(official);
    expect(r.revision).toBe('main');
    expect(r.config).toEqual(mirrorConfig);
    expect(r.api).toEqual(mirrorApi);
    expect(r.configFrom).toBe(mirror);
    expect(r.metadataFrom).toBe(mirror);
  });

  it('also falls back when only the official config.json is gated', async () => {
    const { impl, seen } = fakeFetch({
      ...gatedRoutes,
      [apiUrl(official)]: { siblings: [{ rfilename: 'config.json' }] },
    });
    const r = await fetchHfRepoFiles(official, opts(impl));
    if (isHfError(r)) throw new Error(r.message);
    expect(seen.map((s) => s.url)).toEqual([
      apiUrl(official),
      resolveUrl(official, 'config.json'),
      apiUrl(mirror),
      resolveUrl(mirror, 'config.json'),
    ]);
    expect(r.configFrom).toBe(mirror);
  });

  it('(b) with a token, reads the official repo only and sends the token', async () => {
    const { impl, seen } = fakeFetch({
      ...gatedRoutes,
      [apiUrl(official)]: { siblings: [{ rfilename: 'config.json' }] },
      [resolveUrl(official, 'config.json')]: { model_type: 'official' },
    });
    const r = await fetchHfRepoFiles(official, { ...opts(impl), token: 'hf_abc' });
    if (isHfError(r)) throw new Error(r.message);
    expect(seen.map((s) => s.url)).toEqual([apiUrl(official), resolveUrl(official, 'config.json')]);
    expect(seen.every((s) => s.auth === 'Bearer hf_abc')).toBe(true);
    expect(r.config).toEqual({ model_type: 'official' });
    expect(r.configFrom).toBeUndefined();
    expect(r.metadataFrom).toBeUndefined();
  });

  it('(c) with a rejected token, returns 401 without asking the mirror', async () => {
    const { impl, seen } = fakeFetch(gatedRoutes);
    const r = await fetchHfRepoFiles(official, { ...opts(impl), token: 'hf_bad' });
    expect(isHfError(r) && r.kind).toBe('unauthorized');
    expect(isHfError(r) && r.message).toMatch(/rejected the token/);
    expect(seen.some((s) => s.url.includes(mirror))).toBe(false);
  });

  it('(d) a gated repo without a known mirror stays a 401 with the token prompt', async () => {
    const { impl, seen } = fakeFetch({ [apiUrl('other/gated')]: 401 });
    const r = await fetchHfRepoFiles('other/gated', opts(impl));
    expect(isHfError(r) && r.kind).toBe('unauthorized');
    expect(isHfError(r) && r.message).toMatch(/Add a Hugging Face access token/);
    expect(seen).toHaveLength(1);
  });

  it('(e) a non-default revision stays a 401, no mirror', async () => {
    const { impl, seen } = fakeFetch({ ...gatedRoutes, [apiUrl(official, 'v2')]: 401 });
    const r = await fetchHfRepoFiles(official, { ...opts(impl), revision: 'v2' });
    expect(isHfError(r) && r.kind).toBe('unauthorized');
    expect(seen.map((s) => s.url)).toEqual([apiUrl(official, 'v2')]);
  });

  it('(f) a failing mirror returns the original 401', async () => {
    for (const broken of [
      { [apiUrl(mirror)]: 404 },
      { [resolveUrl(mirror, 'config.json')]: 500 },
    ]) {
      const { impl } = fakeFetch({ ...gatedRoutes, ...broken });
      const r = await fetchHfRepoFiles(official, opts(impl));
      expect(isHfError(r) && r.kind).toBe('unauthorized');
      expect(isHfError(r) && r.message).toMatch(new RegExp(`${official} \\(api\\) is gated`));
    }
  });
});
