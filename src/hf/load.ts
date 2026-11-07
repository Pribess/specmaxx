import { mapHfModel } from '../sim/models/hf';
import type { HfMapResult } from '../sim/models/hf';
import { isHfError, type HfError } from './errors';
import { fetchHfRepoFiles } from './fetch';
import type { HfFetchOptions } from './types';

/**
 * Load a model from the Hub and lower its config to the model IR:
 * {@link fetchHfRepoFiles} then `mapHfModel`.
 * @param id repo id `owner/name`.
 * @param options revision, token and injectable fetch/storage.
 * @returns the mapping result (which itself may be `unsupported` / `invalid_config`), or a fetch {@link HfError}.
 */
export async function loadHfModel(id: string, options: HfFetchOptions = {}): Promise<HfMapResult | HfError> {
  const files = await fetchHfRepoFiles(id, options);
  if (isHfError(files)) return files;
  return mapHfModel(files);
}
