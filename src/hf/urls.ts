/** Hub origin used for all requests. */
export const HF_ORIGIN = 'https://huggingface.co';

/** Default git revision (branch) when the caller gives none. */
export const DEFAULT_REVISION = 'main';

const ID_PATTERN = /^[\w.-]+\/[\w.-]+$/;

/**
 * True when `id` looks like a Hub repo id `owner/name` (letters, digits, `_`, `.`, `-`).
 * @param id candidate model id.
 */
export function isValidModelId(id: string): boolean {
  return ID_PATTERN.test(id);
}

/**
 * URL of one raw file of a repo. The Hub answers with a 307 to
 * `/api/resolve-cache/...`, which `fetch` follows.
 * @param id repo id, `owner/name` (must already be validated).
 * @param file path inside the repo, e.g. `config.json`.
 * @param revision branch, tag or commit sha.
 */
export function resolveUrl(id: string, file: string, revision: string = DEFAULT_REVISION): string {
  return `${HF_ORIGIN}/${id}/resolve/${encodeURIComponent(revision)}/${file}`;
}

/**
 * URL of the model metadata endpoint (siblings, safetensors counts, tags, cardData).
 * @param id repo id, `owner/name` (must already be validated).
 * @param revision when given, metadata of that revision instead of the default branch.
 */
export function apiUrl(id: string, revision?: string): string {
  const base = `${HF_ORIGIN}/api/models/${id}`;
  return revision && revision !== DEFAULT_REVISION
    ? `${base}/revision/${encodeURIComponent(revision)}`
    : base;
}
