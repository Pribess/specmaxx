import type { BlockDraftSpec } from '../sim/ir/model';
import { parseDrafterConfig } from '../sim/models/hf/drafter';
import { isHfError, type HfError } from './errors';
import { fetchHfJsonFile } from './fetch';
import type { HfFetchOptions } from './types';

/** Result of {@link loadDrafter}. */
export type DrafterLoadResult =
  /** The drafter's architecture, read from its `config.json`, and the parser's notes. */
  | { kind: 'ok'; spec: BlockDraftSpec; notes: string[] }
  /** The config was fetched but is not a drafter the simulator knows (DFlash 2, DSpark). */
  | { kind: 'invalid_config'; errors: string[] }
  /** The fetch failed. */
  | HfError;

/**
 * Load a block drafter (DFlash 2, DSpark) from the Hub: `<repo>/resolve/<revision>/config.json` through
 * {@link fetchHfJsonFile} (cached, with the user's token), mapped by `parseDrafterConfig` (design doc §3.2).
 * @param repoId drafter repo id `owner/name`.
 * @param options revision, token and injectable fetch/storage.
 */
export async function loadDrafter(
  repoId: string,
  options: HfFetchOptions = {},
): Promise<DrafterLoadResult> {
  const fetched = await fetchHfJsonFile(repoId, 'config.json', options);
  if (isHfError(fetched)) return fetched;
  const config = fetched.value;
  if (typeof config !== 'object' || config === null || Array.isArray(config)) {
    return { kind: 'invalid_config', errors: [`${repoId}/config.json is not a JSON object.`] };
  }
  return parseDrafterConfig(config as Parameters<typeof parseDrafterConfig>[0], repoId);
}

/** A user-facing message for a failed {@link loadDrafter}. */
export function drafterLoadMessage(result: Exclude<DrafterLoadResult, { kind: 'ok' }>): string {
  return result.kind === 'invalid_config' ? result.errors.join('; ') : result.message;
}
