import type { HfApiModelInfo, HfRepoFiles } from '../sim/models/hf/types';

/** The fetched repo files and Hub metadata: one definition, owned by the mapping core. */
export type { HfApiModelInfo, HfRepoFiles };

/** Options shared by the fetch functions; all optional, all injectable for tests. */
export interface HfFetchOptions {
  /** Git revision (branch/tag/sha). Default `main`. */
  revision?: string;
  /** Access token; sent as `Authorization: Bearer`. Default: the one in localStorage. */
  token?: string | null;
  /** Fetch implementation. Default `globalThis.fetch`. */
  fetchImpl?: typeof fetch;
  /** Persistent cache level. Default sessionStorage (null disables it). */
  storage?: import('./storage').KeyValueStorage | null;
  /** Ready-made cache (overrides `storage`); tests use it to isolate the memory level. */
  cache?: import('./cache').HfCache;
  /** Token store, default localStorage. Only read when `token` is undefined. */
  tokenStorage?: import('./storage').KeyValueStorage | null;
}
