import { describe, expect, it } from 'vitest';
import { createCache } from './cache';
import { loadDrafter } from './loadDrafter';

/** A fetch that answers every URL from `routes` (404 otherwise) and records each request. */
function fakeFetch(routes: Record<string, { status: number; body: string }>) {
  const calls: { url: string; auth?: string }[] = [];
  const impl = (async (url: string, init?: RequestInit) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ url, auth: headers.Authorization });
    const r = routes[url] ?? { status: 404, body: '{}' };
    return new Response(r.body, { status: r.status });
  }) as unknown as typeof fetch;
  return { impl, calls };
}

const URL = 'https://huggingface.co/a/drafter/resolve/main/config.json';

describe('loadDrafter', () => {
  it('fetches <repo>/resolve/main/config.json with the token, and caches it', async () => {
    const { impl, calls } = fakeFetch({
      [URL]: { status: 200, body: '{"architectures":["LlamaForCausalLM"]}' },
    });
    const cache = createCache(null, new Map());
    const first = await loadDrafter('a/drafter', { fetchImpl: impl, token: 'hf_x', cache });
    expect(first.kind).toBe('invalid_config');
    await loadDrafter('a/drafter', { fetchImpl: impl, token: 'hf_x', cache });
    expect(calls).toEqual([{ url: URL, auth: 'Bearer hf_x' }]);
  });

  it('maps a 404 and a 401 to Hub errors, and a non-object body to invalid_config', async () => {
    const cache = () => createCache(null, new Map());
    expect(
      (
        await loadDrafter('a/drafter', {
          fetchImpl: fakeFetch({}).impl,
          token: null,
          cache: cache(),
        })
      ).kind,
    ).toBe('not_found');
    const gated = fakeFetch({ [URL]: { status: 401, body: '' } });
    expect(
      (await loadDrafter('a/drafter', { fetchImpl: gated.impl, token: null, cache: cache() })).kind,
    ).toBe('unauthorized');
    const list = fakeFetch({ [URL]: { status: 200, body: '[1]' } });
    expect(
      (await loadDrafter('a/drafter', { fetchImpl: list.impl, token: null, cache: cache() })).kind,
    ).toBe('invalid_config');
    expect((await loadDrafter('not an id', { token: null })).kind).toBe('invalid_id');
  });
});
