/**
 * Pick the family that builds a config.
 *
 * Keys are tried in this order; the first one some family claims wins:
 * 1. top-level `architectures[0]`
 * 2. top-level `model_type`
 * 3. `text_config.architectures[0]`
 * 4. `text_config.model_type`
 *
 * Architectures come first because they name the model class (a repo may reuse another family's
 * `model_type`), model types second because text-only extractions often drop `architectures`.
 * `vision_config.model_type` is never used: Qwen4Exp's equals the top-level type.
 */

import { FAMILIES } from './families';
import type { FamilyEntry, HfConfigView } from './types';

/** Which key matched, for notes and tests. */
export type DispatchKey =
  | 'architecture'
  | 'model_type'
  | 'text_config.architecture'
  | 'text_config.model_type';

/** A dispatch hit: the family entry and the key that selected it. */
export interface DispatchHit {
  /** The family that claims the config. */
  entry: FamilyEntry;
  /** Which config key matched. */
  key: DispatchKey;
  /** The matched value, e.g. `'Qwen3_5ForConditionalGeneration'`. */
  value: string;
}

/** The family claiming `view`, or null when none does. `families` defaults to {@link FAMILIES}. */
export function dispatchFamily(
  view: HfConfigView,
  families: readonly FamilyEntry[] = FAMILIES,
): DispatchHit | null {
  const candidates: {
    key: DispatchKey;
    value: string | null | undefined;
    field: 'architectures' | 'modelTypes';
  }[] = [
    { key: 'architecture', value: view.topArchitectures[0], field: 'architectures' },
    { key: 'model_type', value: view.topModelType, field: 'modelTypes' },
    { key: 'text_config.architecture', value: view.textArchitectures[0], field: 'architectures' },
    { key: 'text_config.model_type', value: view.textModelType, field: 'modelTypes' },
  ];
  for (const c of candidates) {
    if (!c.value) continue;
    const entry = families.find((f) => f[c.field].includes(c.value as string));
    if (entry) return { entry, key: c.key, value: c.value };
  }
  return null;
}

/** All model types the families claim, sorted, for the "unsupported" message. */
export function supportedModelTypes(families: readonly FamilyEntry[] = FAMILIES): string[] {
  return families.flatMap((f) => f.modelTypes).sort();
}
