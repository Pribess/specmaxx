import { describe, expect, it } from 'vitest';
import { createCache } from './cache';
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
