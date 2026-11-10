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
import { effectiveParallel, fieldIssues, hasErrors, type FieldIssue } from './parallelism';
import { REFERENCE_QUANT_DEFAULTS, applyQuantOverride } from './quant';
import { deploymentConfig, type DrafterStatus } from './deploymentInput';

/** The panel an engine validation message belongs to, by its wording (speculative decoding, offloading, else general). */
function engineIssueField(message: string): FieldIssue['field'] {
  if (/speculative|draft/i.test(message)) return 'specDec';
  if (/offload|host link|host_ram|nvme|queue depth|resident|hit ratio|lookup|cpu_compute/i.test(message)) return 'offload';
  return 'general';
}

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
 * State + loaded model (+ the fetch state of a block drafter's config, when spec-dec needs one) → engine input, or
 * the issues that stop it. `loaded` must be the model of `state.model` (same {@link modelKey}); otherwise (not loaded
 * yet, or stale) the result is a general issue. Steps: the deployment config (`deploymentConfig`); resolve the
 * hardware (with `host` only while something is offloaded); apply the quant overrides; `fieldIssues` and the
 * deployment issues (errors stop here); then the engine's `validateInput`, whose remaining errors go to the
 * spec-dec / offload field by their wording, else general. `deployment` is in the input only when a knob is on.
 */
export function buildSimInput(state: AppState, loaded: LoadedModel | null, drafter?: DrafterStatus): BuildResult {
  if (loaded === null || loaded.key !== modelKey(state.model)) {
    return { kind: 'issues', issues: [{ field: 'general', level: 'error', message: 'The model is not loaded yet.' }] };
  }
  const deployment = deploymentConfig(state, loaded.model, drafter);
  let hardware: ResolvedHardware;
  try {
    hardware = resolveHardware(state.hw, deployment.config?.offload !== undefined);
  } catch (e) {
    return { kind: 'issues', issues: [{ field: 'general', level: 'error', message: (e as Error).message }] };
  }
  const quant = applyQuantOverride(loaded.quant.quant, state.quant, loaded.quant.exl3);
  const parallel = effectiveParallel(state.parallel, loaded.model, state.parallelLayout);
  const issues = [
    ...fieldIssues(loaded.model, quant, parallel, state.workload, hardware.topology, state.parallelLayout),
    ...deployment.issues,
  ];
  if (hasErrors(issues)) return { kind: 'issues', issues, hardware };
  const input: SimInput = {
    model: loaded.model,
    quant,
    // The user's degrees; EP is the user's on a MoE model, else 1. `denseTpGroup` only when dense TP over the EP
    // group is on (MoE, DP attention + EP), so without it the input is exactly the one from before the layout. The
    // EP balance knobs (ρ, EPLB replicas) only when the user set a non-default value on a MoE model. `dcp` likewise
    // only when DCP is on (> 1): with DCP off the input is exactly the one from before DCP.
    parallel: {
      tp: parallel.tp,
      pp: parallel.pp,
      dp: parallel.dp,
      ep: parallel.ep,
      ...(parallel.denseTpGroup !== undefined ? { denseTpGroup: parallel.denseTpGroup } : {}),
      ...(parallel.epImbalance !== undefined ? { epImbalance: parallel.epImbalance } : {}),
      ...(parallel.epImbalanceScalesAllToAll === true ? { epImbalanceScalesAllToAll: true } : {}),
      ...(parallel.redundantExpertsPerRank !== undefined ? { redundantExpertsPerRank: parallel.redundantExpertsPerRank } : {}),
      ...(parallel.dcp !== undefined && parallel.dcp > 1 ? { dcp: parallel.dcp } : {}),
    },
    workload: state.workload,
    hardware: hardware.stack,
    ...(Object.keys(state.options).length > 0 ? { options: state.options } : {}),
    // Only when a knob is on: with every knob off the input is exactly the one from before deployment knobs.
    ...(deployment.config !== undefined ? { deployment: deployment.config } : {}),
    // The dispatch overlay adds a labelled figure only; off, the input has no `overlays` at all.
    ...(state.engineDispatchOverlay ? { overlays: { engineDispatch: {} } } : {}),
  };
  const engineErrors = validateInput(input);
  if (engineErrors.length > 0) {
    const general = engineErrors.map((message): FieldIssue => ({ field: engineIssueField(message), level: 'error', message }));
    return { kind: 'issues', issues: [...issues, ...general], hardware };
  }
  const drafterNotes = input.deployment?.speculative !== undefined && drafter?.kind === 'ok' ? drafter.notes : [];
  return { kind: 'ok', input, hardware, issues, notes: [...hardware.notes, ...loaded.notes, ...drafterNotes] };
}
