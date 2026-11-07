import { defaultSessionStorage, type KeyValueStorage } from './storage';

/** A cached response: body text, or `null` for "file absent" (404 on an optional file). */
export type CacheEntry = { text: string } | { absent: true };

/** Two-level (memory, then sessionStorage) cache of Hub responses. */
export interface HfCache {
  get(key: string): CacheEntry | undefined;
  set(key: string, entry: CacheEntry): void;
}

/** Process-wide memory level, shared by caches that do not bring their own. */
const sharedMemory = new Map<string, CacheEntry>();

/**
 * Cache key for one file of one repo revision. Never contains the token.
 * @param id repo id.
 * @param revision git revision.
 * @param file `config.json`, or `api` for the metadata JSON.
 */
export function cacheKey(id: string, revision: string, file: string): string {
  return `specmaxx:hf:${id}@${revision}:${file}`;
}

/**
 * Build a cache. Errors are never stored by callers, only successes and absences.
 * @param storage persistent level (default sessionStorage); a full or missing one is tolerated.
 * @param memory memory level (default: shared process-wide map).
 */
export function createCache(
  storage: KeyValueStorage | null = defaultSessionStorage(),
  memory: Map<string, CacheEntry> = sharedMemory,
): HfCache {
  return {
    get(key) {
      const hit = memory.get(key);
      if (hit) return hit;
      try {
        const raw = storage?.getItem(key);
        if (raw == null) return undefined;
        const parsed = JSON.parse(raw) as CacheEntry;
        memory.set(key, parsed);
        return parsed;
      } catch {
        return undefined;
      }
    },
    set(key, entry) {
      memory.set(key, entry);
      try {
        storage?.setItem(key, JSON.stringify(entry));
      } catch {
        /* quota exceeded or unavailable: memory level still works */
      }
    },
  };
}
