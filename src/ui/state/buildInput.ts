/**
 * The loaded model and the state → `SimInput` step. The simulation runs only when this returns an input;
 * otherwise the UI shows the issues by field.
 */

import type { SimInput } from '../../sim/engine/types';
import { validateInput } from '../../sim/engine/validate';
import type { ModelSpec } from '../../sim/ir/model';
import { REF_DENSE_8B } from '../../sim/models/reference';
import type { HfMapResult, QuantDefaults } from '../../sim/models/hf/types';
import type { AppState, ModelChoice } from './appState';
import { resolveHardware, type ResolvedHardware } from './hardware';
import { fieldIssues, hasErrors, type FieldIssue } from './parallelism';
import { REFERENCE_QUANT_DEFAULTS, applyQuantOverride } from './quant';

/** A successful `mapHfModel` result. */
export type HfMapOk = Extract<HfMapResult, { kind: 'ok' }>;

/** A model ready to simulate: the built-in reference or a mapped Hugging Face repo. */
export interface LoadedModel {
  /** Identity of the choice it was loaded for ({@link modelKey}); a result for another key is stale. */
  key: string;
  /** Display name: the repo id, or the reference model's name. */
  label: string;
  /** The model IR. */
  model: ModelSpec;
  /** Default formats (before the user's overrides) and the scheme label. */
  quant: QuantDefaults;
  /** Loader notes (unwrap, family, quant, param check); empty for the reference. */
  notes: string[];
  /** The full mapping result for an HF model (family, modelType, architecture, excluded parts, param check). */
  hf?: HfMapOk;
}

/** Identity of a model choice: `'hf:<id>@<revision>'` (revision default `main`). */
export function modelKey(choice: ModelChoice): string {
  return `hf:${choice.id}@${choice.revision ?? 'main'}`;
}

/** The built-in reference model (REF_DENSE_8B, bf16), available without network. */
export const REFERENCE_MODEL: LoadedModel = {
  key: 'reference',
  label: REF_DENSE_8B.name,
  model: REF_DENSE_8B,
  quant: REFERENCE_QUANT_DEFAULTS,
  notes: [],
};

/** Wraps a successful mapping as a {@link LoadedModel} for `choice`. */
export function loadedFromHf(choice: Extract<ModelChoice, { kind: 'hf' }>, result: HfMapOk): LoadedModel {
  return { key: modelKey(choice), label: choice.id, model: result.model, quant: result.quant, notes: result.notes, hf: result };
}

/** Result of {@link buildSimInput}. */
export type BuildResult =
  | {
      kind: 'ok';
      /** The engine input. */
      input: SimInput;
      /** The resolved hardware (for display: rig sentence, spec editor, "edited" marks). */
      hardware: ResolvedHardware;
      /** Warnings by field (e.g. unused devices); never errors. */
      issues: FieldIssue[];
      /** Unlock modifier descriptions and loader notes, for the Notes panel. */
      notes: string[];
    }
  | {
      kind: 'issues';
      /** At least one error; the simulation does not run. */
      issues: FieldIssue[];
      /** The resolved hardware, when it resolved (always, for states the decoder or reducer made). */
      hardware?: ResolvedHardware;
    };

/**
 * State + loaded model → engine input, or the issues that stop it. `loaded` must be the model of
 * `state.model` (same {@link modelKey}); otherwise (not loaded yet, or stale) the result is a general issue.
 * Steps: resolve the hardware; apply the quant overrides; `fieldIssues` (errors stop here); then the engine's
 * `validateInput`, whose remaining errors are general issues.
 */
export function buildSimInput(state: AppState, loaded: LoadedModel | null): BuildResult {
  if (loaded === null || loaded.key !== modelKey(state.model)) {
    return { kind: 'issues', issues: [{ field: 'general', level: 'error', message: 'The model is not loaded yet.' }] };
  }
  let hardware: ResolvedHardware;
  try {
    hardware = resolveHardware(state.hw);
  } catch (e) {
    return { kind: 'issues', issues: [{ field: 'general', level: 'error', message: (e as Error).message }] };
  }
  const quant = applyQuantOverride(loaded.quant.quant, state.quant, loaded.quant.exl3);
  const issues = fieldIssues(loaded.model, quant, state.parallel, state.workload, hardware.topology);
  if (hasErrors(issues)) return { kind: 'issues', issues, hardware };
  const input: SimInput = {
    model: loaded.model,
    quant,
    parallel: state.parallel,
    workload: state.workload,
    hardware: hardware.stack,
    ...(Object.keys(state.options).length > 0 ? { options: state.options } : {}),
  };
  const engineErrors = validateInput(input);
  if (engineErrors.length > 0) {
    const general = engineErrors.map((message): FieldIssue => ({ field: 'general', level: 'error', message }));
    return { kind: 'issues', issues: [...issues, ...general], hardware };
  }
  return { kind: 'ok', input, hardware, issues, notes: [...hardware.notes, ...loaded.notes] };
}
