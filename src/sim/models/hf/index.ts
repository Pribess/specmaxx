/**
 * Map a Hugging Face repo (already fetched) to the model IR.
 *
 * Pipeline: unwrap `text_config` (`textConfig.ts`) → pick the family (`dispatch.ts`) → family build
 * (`families/*.ts`) → quant defaults (`quant/index.ts`) with the family's `quantHints` merged over them →
 * param check against `safetensors.total` (`paramCheck.ts`). Pure: no fetch, never throws for bad input.
 */

import type { QuantConfig } from '../../ir/model';
import { dispatchFamily, supportedModelTypes } from './dispatch';
import { checkParams } from './paramCheck';
import { deriveQuantDefaults } from './quant/index';
import { isJsonObject } from './reader';
import { unwrapTextConfig } from './textConfig';
import { FAMILIES } from './families';
import type { FamilyEntry, HfMapResult, HfRepoFiles, QuantDefaults } from './types';

export * from './types';
export { createReader, isJsonObject } from './reader';
export type { ConfigReader } from './reader';
export { unwrapTextConfig } from './textConfig';
export type { UnwrapResult } from './textConfig';
export { dispatchFamily, supportedModelTypes } from './dispatch';
export type { DispatchHit, DispatchKey } from './dispatch';
export { FAMILIES } from './families';
export { checkParams, estimateVisionParams } from './paramCheck';
export { deriveQuantDefaults } from './quant/index';
export type { QuantInputs } from './quant/index';

/**
 * Map one repo's files to the IR, or say why it cannot be mapped.
 * `families` defaults to the built-in table ({@link FAMILIES}); tests pass their own.
 */
export function mapHfModel(files: HfRepoFiles, families: readonly FamilyEntry[] = FAMILIES): HfMapResult {
  const unwrapped = unwrapTextConfig(files.config);
  if (unwrapped.kind === 'invalid_config') return unwrapped;
  const view = unwrapped.view;

  const modelType = view.topModelType ?? view.textModelType;
  const architecture = view.topArchitectures[0] ?? view.textArchitectures[0] ?? null;
  const hit = dispatchFamily(view, families);
  const supported = supportedModelTypes(families);
  if (!hit) return { kind: 'unsupported', modelType, architecture, supported };

  const built = hit.entry.build(view.text, view.top, files);
  if (built.kind === 'invalid_config') return built;
  if (built.kind === 'unsupported') {
    return { kind: 'unsupported', modelType, architecture, supported, reason: built.reason };
  }

  const derived = deriveQuantDefaults({
    top: view.top,
    text: view.text,
    hfQuantConfig: files.hfQuantConfig,
    paramsJson: files.paramsJson,
    quantizationConfigJson: files.quantizationConfigJson,
    mlx: view.top.quantization,
  });
  const merged = mergeQuantHints(derived, built.quantHints);
  const quant = merged.quant;

  const excluded = {
    mtpParamsEstimate: built.mtpParamsEstimate,
    visionParamsEstimate: built.visionParamsEstimate,
    parts: built.excludedParts,
  };
  const qc = view.quantizationConfig;
  const quantMethod = isJsonObject(qc) && typeof qc.quant_method === 'string' ? qc.quant_method : undefined;
  const paramCheck = checkParams(built.model, excluded, files.api, quantMethod);

  const notes = [...view.notes, ...built.notes, ...quant.notes, ...merged.notes];
  if (files.configFrom && files.configFrom !== files.id) notes.push(`config.json read from ${files.configFrom}`);
  if (paramCheck.status !== 'ok') notes.push(`param check (${paramCheck.status}): ${paramCheck.note}`);

  return {
    kind: 'ok',
    model: built.model,
    family: hit.entry.family,
    modelType,
    architecture,
    notes,
    excluded,
    quant,
    paramCheck,
  };
}

/**
 * Merge a family's architecture-native `quantHints` over the derived defaults. Explicit repo settings win:
 * a hint for a field in `derived.explicit` is dropped, with a note.
 */
export function mergeQuantHints(
  derived: QuantDefaults,
  hints: Partial<QuantConfig> | undefined,
): { quant: QuantDefaults; notes: string[] } {
  if (!hints) return { quant: derived, notes: [] };
  const kept: Partial<QuantConfig> = {};
  const notes: string[] = [];
  for (const key of Object.keys(hints) as (keyof QuantConfig)[]) {
    if (hints[key] === undefined) continue;
    if (derived.explicit.includes(key)) notes.push(`family hint for ${key} overridden by the repo's quant metadata`);
    else Object.assign(kept, { [key]: hints[key] });
  }
  return { quant: { ...derived, quant: { ...derived.quant, ...kept } }, notes };
}
