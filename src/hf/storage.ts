/** The subset of the Web Storage API that this module uses. */
export interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** In-memory {@link KeyValueStorage}; handy for tests and as a fallback. */
export function memoryStorage(): KeyValueStorage {
  const m = new Map<string, string>();
  return {
    getItem: (k) => m.get(k) ?? null,
    setItem: (k, v) => void m.set(k, v),
    removeItem: (k) => void m.delete(k),
  };
}

/** `window.sessionStorage` if usable, else null (SSR, tests, privacy modes). */
export function defaultSessionStorage(): KeyValueStorage | null {
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

/** `window.localStorage` if usable, else null. */
export function defaultLocalStorage(): KeyValueStorage | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}
