/**
 * The default model, bundled: the committed Hub fixtures of `DEFAULT_MODEL_ID` (config.json and the
 * /api/models response) mapped with `mapHfModel` at module load, so the first paint has numbers without any
 * network. The app still refreshes the model from the Hub in the background.
 */

import { mapHfModel } from '../../sim/models/hf';
import type { HfApiModelInfo } from '../../sim/models/hf/types';
import apiJson from '../../sim/models/__fixtures__/Qwen__Qwen3.8-27B.api.json?raw';
import configJson from '../../sim/models/__fixtures__/Qwen__Qwen3.8-27B.config.json?raw';
import { loadedFromHf, type LoadedModel } from './buildInput';
import { DEFAULT_MODEL_ID } from './modelShortlist';

/**
 * Maps the bundled fixture. Throws when it does not map: the fixture is committed, so a failure is a build
 * bug, not an input case.
 */
function bundledDefaultModel(): LoadedModel {
  const result = mapHfModel({
    id: DEFAULT_MODEL_ID,
    revision: 'main',
    config: JSON.parse(configJson) as unknown,
    api: JSON.parse(apiJson) as HfApiModelInfo,
  });
  if (result.kind !== 'ok')
    throw new Error(`bundled ${DEFAULT_MODEL_ID} does not map: ${result.kind}`);
  return loadedFromHf({ kind: 'hf', id: DEFAULT_MODEL_ID }, result);
}

/** `DEFAULT_MODEL_ID` loaded from the bundled fixtures; `key` = `modelKey({ kind: 'hf', id: DEFAULT_MODEL_ID })`. */
export const BUNDLED_DEFAULT_MODEL: LoadedModel = bundledDefaultModel();
