/**
 * The table of supported model families. Each family lives in `families/<name>.ts` and declares the
 * `model_type` and `architectures` values it handles; dispatch (`dispatch.ts`) picks the entry.
 *
 * To add a family: write `families/<name>.ts` exporting a {@link FamilyEntry}, then list it here.
 * A model type or architecture must appear in at most one entry (a test checks this).
 */

import { deepseekV41Family, deepseekV4Family } from './families/deepseekV41';
import { glm5NextFamily } from './families/glm5Next';
import { glmMoeDsaFamily } from './families/glmMoeDsa';
import { inklingFamily } from './families/inkling';
import { minimaxFamily } from './families/minimax';
import { mistralLarge4Family } from './families/mistralLarge4';
import { qwenFamily } from './families/qwen';
import type { FamilyEntry } from './types';

/** Every supported family, in no particular order (keys are unique across entries). */
export const FAMILIES: readonly FamilyEntry[] = [
  qwenFamily,
  minimaxFamily,
  inklingFamily,
  deepseekV41Family,
  deepseekV4Family,
  glm5NextFamily,
  glmMoeDsaFamily,
  mistralLarge4Family,
];
