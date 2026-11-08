import { createCache, cacheKey, type HfCache } from './cache';
import { errorFromStatus, type HfError } from './errors';
import { getHfToken } from './token';
import type { HfApiModelInfo, HfFetchOptions, HfRepoFiles } from './types';
import { apiUrl, DEFAULT_REVISION, isValidModelId, resolveUrl } from './urls';

/** Sidecar files fetched only when the repo lists them (no 404 probing). */
const SIDECARS = [
  ['hf_quant_config.json', 'hfQuantConfig'],
  ['quantization_config.json', 'quantizationConfigJson'],
  ['params.json', 'paramsJson'],
] as const;

interface Ctx {
  fetchImpl: typeof fetch;
  token: string | null;
  cache: HfCache;
  revision: string;
}

/** Outcome of one request; tagged so a JSON body can never be mistaken for an error. */
type Fetched<T> = { tag: 'ok'; value: T } | { tag: 'absent' } | { tag: 'error'; error: HfError };
const fail = (error: HfError): { tag: 'error'; error: HfError } => ({ tag: 'error', error });

/** GET `url` with cache; 404 → `absent` when `absentOk`, errors are never cached. */
async function getText(
  ctx: Ctx,
  id: string,
  file: string,
  url: string,
  absentOk: boolean,
): Promise<Fetched<string>> {
  const key = cacheKey(id, ctx.revision, file);
  const hit = ctx.cache.get(key);
  if (hit && 'text' in hit) return { tag: 'ok', value: hit.text };
  if (hit && absentOk) return { tag: 'absent' };
  let res: Response;
  try {
    res = await ctx.fetchImpl(url, {
      headers: ctx.token ? { Authorization: `Bearer ${ctx.token}` } : undefined,
    });
  } catch (e) {
    return fail({
      kind: 'network',
      message: `Could not reach Hugging Face (${e instanceof Error ? e.message : 'network error'}). You may be offline, or the request was blocked by CORS.`,
    });
  }
  if (res.status === 404 && absentOk) {
    ctx.cache.set(key, { absent: true });
    return { tag: 'absent' };
  }
  if (!res.ok) return fail(errorFromStatus(res.status, `${id} (${file})`, ctx.token !== null));
  const text = await res.text();
  ctx.cache.set(key, { text });
  return { tag: 'ok', value: text };
}

function parseJson(text: string, what: string): Fetched<unknown> {
  try {
    return { tag: 'ok', value: JSON.parse(text) as unknown };
  } catch {
    return fail({ kind: 'bad_json', message: `${what} is not valid JSON.` });
  }
}

/** Fetch + parse one JSON file; `absent` on a 404 when `absentOk`. */
async function getJson(
  ctx: Ctx,
  id: string,
  file: string,
  url: string,
  absentOk: boolean,
): Promise<Fetched<unknown>> {
  const r = await getText(ctx, id, file, url, absentOk);
  if (r.tag !== 'ok') return r;
  return parseJson(r.value, `${id}/${file}`);
}

/**
 * Fetch and parse one JSON file of a Hub repo (e.g. a drafter's `config.json`), with the same cache, token and
 * error mapping as {@link fetchHfRepoFiles}. A 404 is a `not_found` error.
 * @param id repo id `owner/name`.
 * @param file path inside the repo.
 * @param options revision, token, injectable fetch/storage (see {@link HfFetchOptions}).
 * @returns the parsed JSON value, or an {@link HfError} (use `isHfError`).
 */
export async function fetchHfJsonFile(id: string, file: string, options: HfFetchOptions = {}): Promise<{ value: unknown } | HfError> {
  if (!isValidModelId(id)) {
    return { kind: 'invalid_id', message: `"${id}" is not a Hugging Face model id. Use the form owner/name.` };
  }
  const revision = options.revision?.trim() || DEFAULT_REVISION;
  const f = options.fetchImpl ?? globalThis.fetch?.bind(globalThis);
  if (!f) return { kind: 'network', message: 'fetch is not available in this environment.' };
  const ctx: Ctx = {
    fetchImpl: f,
    token: options.token === undefined ? getHfToken(options.tokenStorage) : options.token || null,
    cache: options.cache ?? (options.storage === undefined ? createCache() : createCache(options.storage)),
    revision,
  };
  const r = await getJson(ctx, id, file, resolveUrl(id, file, revision), false);
  if (r.tag === 'ok') return { value: r.value };
  return r.tag === 'error' ? r.error : { kind: 'not_found', status: 404, message: `${id} has no ${file}.` };
}

function baseModelOf(api: HfApiModelInfo): string | undefined {
  const b = api.cardData?.base_model;
  const first = Array.isArray(b) ? b[0] : b;
  return first && isValidModelId(first) ? first : undefined;
}

/**
 * Fetch the metadata and JSON files of a Hub repo needed to build a model.
 *
 * 1. `GET /api/models/<id>`; 401/403/404 become errors.
 * 2. `config.json`; on 404 the repo is Mistral-format: use `params.json`, and if
 *    `cardData.base_model` is set, take that repo's `config.json` (`configFrom`).
 * 3. Sidecars (`hf_quant_config.json`, `quantization_config.json`, `params.json`)
 *    only when `api.siblings` lists them.
 *
 * @param id repo id `owner/name`.
 * @param options revision, token, injectable fetch/storage (see {@link HfFetchOptions}).
 * @returns the parsed files, or an {@link HfError} (use `isHfError`).
 */
export async function fetchHfRepoFiles(
  id: string,
  options: HfFetchOptions = {},
): Promise<HfRepoFiles | HfError> {
  if (!isValidModelId(id)) {
    return {
      kind: 'invalid_id',
      message: `"${id}" is not a Hugging Face model id. Use the form owner/name, e.g. Qwen/Qwen3-8B.`,
    };
  }
  const revision = options.revision?.trim() || DEFAULT_REVISION;
  const f = options.fetchImpl ?? globalThis.fetch?.bind(globalThis);
  if (!f) return { kind: 'network', message: 'fetch is not available in this environment.' };
  const ctx: Ctx = {
    fetchImpl: f,
    token: options.token === undefined ? getHfToken(options.tokenStorage) : options.token || null,
    cache:
      options.cache ?? (options.storage === undefined ? createCache() : createCache(options.storage)),
    revision,
  };

  const apiRes = await getJson(ctx, id, 'api', apiUrl(id, options.revision), false);
  if (apiRes.tag === 'error') return apiRes.error;
  const apiJson = apiRes.tag === 'ok' ? apiRes.value : null;
  if (typeof apiJson !== 'object' || apiJson === null) {
    return { kind: 'bad_json', message: `Metadata of ${id} has an unexpected shape.` };
  }
  const api = { ...(apiJson as HfApiModelInfo), id: (apiJson as Partial<HfApiModelInfo>).id ?? id };
  const listed = new Set((api.siblings ?? []).map((s) => s.rfilename));

  const file = (name: string, repo = id) =>
    getJson(ctx, repo, name, resolveUrl(repo, name, repo === id ? revision : DEFAULT_REVISION), true);

  const out: HfRepoFiles = { id, revision, api, config: null };
  const config = await file('config.json');
  if (config.tag === 'error') return config.error;
  if (config.tag === 'ok') out.config = config.value;

  for (const [name, field] of SIDECARS) {
    if (!listed.has(name)) continue;
    const v = await file(name);
    if (v.tag === 'error') return v.error;
    if (v.tag === 'ok') out[field] = v.value;
  }

  if (out.config === null) {
    const base = baseModelOf(api);
    if (base && out.paramsJson !== undefined) {
      const baseConfig = await file('config.json', base);
      if (baseConfig.tag === 'error') return baseConfig.error;
      if (baseConfig.tag === 'ok') {
        out.config = baseConfig.value;
        out.configFrom = base;
      }
    }
  }
  if (out.config === null) {
    return {
      kind: 'not_found',
      status: 404,
      message:
        out.paramsJson !== undefined
          ? `${id} has no config.json (Mistral-format params.json only) and names no base model to read one from.`
          : `${id} has no config.json, so its architecture cannot be read. It may not be a transformers model.`,
    };
  }
  return out;
}
