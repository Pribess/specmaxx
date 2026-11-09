/**
 * Ungated public mirrors of gated Hub repos.
 *
 * {@link fetchHfRepoFiles} reads `config.json` and the `/api/models` metadata from the mirror only when the
 * official repo answers 401, no token is set and the revision is the default branch; with a token the
 * official repo is used as before. The UI never shows that a mirror was used (the user's instruction);
 * `HfRepoFiles.configFrom` / `metadataFrom` keep it as data.
 *
 * Official repo ID `owner/name` → mirror repo ID `owner/name`. Add a mirror only when its config is the
 * official one (e.g. a re-upload by a known org that names the official repo as its `base_model`).
 */
export const GATED_MIRRORS: Readonly<Record<string, string>> = {
  // Meta's gated release; unsloth re-uploads it ungated (cardData.base_model names the official repo).
  'meta-models/Muse-Spark-Open-816B-A42B': 'unsloth/Muse-Spark-Open-816B-A42B',
};

/**
 * The ungated mirror of a gated repo, if one is known.
 * @param id official repo ID `owner/name` (case-sensitive, as the Hub serves it).
 * @returns the mirror's repo ID, or undefined.
 */
export function gatedMirrorOf(id: string): string | undefined {
  return Object.hasOwn(GATED_MIRRORS, id) ? GATED_MIRRORS[id] : undefined;
}
