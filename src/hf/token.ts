import { defaultLocalStorage, type KeyValueStorage } from './storage';

/** localStorage key of the optional user access token. */
export const HF_TOKEN_KEY = 'specmaxx:hfToken';

/**
 * Read the stored HF token.
 * @param storage defaults to localStorage.
 * @returns the token, or null if none is stored.
 */
export function getHfToken(storage: KeyValueStorage | null = defaultLocalStorage()): string | null {
  try {
    const t = storage?.getItem(HF_TOKEN_KEY)?.trim();
    return t ? t : null;
  } catch {
    return null;
  }
}

/**
 * Store the HF token (an empty string clears it). Never logged.
 * @param token the access token (`hf_...`).
 * @param storage defaults to localStorage.
 */
export function setHfToken(
  token: string,
  storage: KeyValueStorage | null = defaultLocalStorage(),
): void {
  const t = token.trim();
  try {
    if (t) storage?.setItem(HF_TOKEN_KEY, t);
    else storage?.removeItem(HF_TOKEN_KEY);
  } catch {
    /* storage unavailable: token just isn't persisted */
  }
}

/**
 * Remove the stored token.
 * @param storage defaults to localStorage.
 */
export function clearHfToken(storage: KeyValueStorage | null = defaultLocalStorage()): void {
  setHfToken('', storage);
}
