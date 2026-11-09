/**
 * The UI's deployment choices (`DeploymentChoice`, `deployment.ts`) + the loaded model → the engine's
 * `DeploymentConfig`, with the problems by field. Also which drafter a block method uses (`drafterSource`), so the
 * app knows which drafter `config.json` to fetch, and the effective spec-dec defaults the panels show.
 *
 * `deploymentConfig` returns `config: undefined` when every knob is off (or each one that is on does nothing for the
 * model), so `buildSimInput` then emits no `deployment` and the results are identical to before.
 */

import type {
  AcceptanceModel,
  LookupTableRule,
  DeploymentConfig,
  OffloadConfig,
  RoutedExpertRule,
  SpecDecodeConfig,
} from '../../sim/engine/types';
import { defaultDraftLength, presetAcceptance, type SpecUseCase } from '../../sim/engine/specdec';
import type { BlockDraftSpec, ModelSpec } from '../../sim/ir/model';
import { hypotheticalDrafter, knownDrafters } from '../../sim/models/hf/drafter';
import type { AppState } from './appState';
import {
  HYPOTHETICAL_DRAFTER,
  splitTierRanges,
  type HitRatioChoice,
  type PlacementChoice,
  type SpecDecChoice,
  type SpecDecMethod,
  type TierRange,
  type TierSplitChoice,
} from './deployment';
import type { FieldIssue } from './parallelism';

/** Where a block method's drafter comes from. */
export type DrafterSource =
  /** A drafter repo on the Hub, whose `config.json` must be fetched (`known`: the default for this target). */
  | { kind: 'repo'; repoId: string; known: boolean }
  /** The drafter the checkpoint itself ships (DeepSeek-V4.1 DSpark under `mtp.*`): `model.draft`. */
  | { kind: 'checkpoint'; spec: BlockDraftSpec }
  /** The labelled hypothetical stand-in (design doc §3.2 defaults). */
  | { kind: 'hypothetical' };

/** The fetch state of a drafter repo's config, as the app holds it. */
export type DrafterStatus =
  | { kind: 'loading'; repoId: string }
  | { kind: 'ok'; repoId: string; spec: BlockDraftSpec; notes: string[] }
  | { kind: 'error'; repoId: string; message: string };

/**
 * The drafter a block method uses for `model` (repo id `targetId`): the user's choice, else the checkpoint's own
 * drafter of that method, else the known released drafter for the target, else the hypothetical one. Undefined for
 * MTP and when spec-dec is off.
 */
export function drafterSource(
  specDec: SpecDecChoice | undefined,
  model: ModelSpec,
  targetId: string,
): DrafterSource | undefined {
  if (specDec === undefined || specDec.method === 'mtp') return undefined;
  const method = specDec.method;
  if (specDec.drafter === HYPOTHETICAL_DRAFTER) return { kind: 'hypothetical' };
  if (specDec.drafter !== undefined) {
    const known = knownDrafters(targetId).some((d) => d.repoId === specDec.drafter);
    return { kind: 'repo', repoId: specDec.drafter, known };
  }
  if (model.draft?.kind === 'block_drafter' && model.draft.method === method)
    return { kind: 'checkpoint', spec: model.draft };
  const known = knownDrafters(targetId).find((d) => d.method === method);
  return known !== undefined
    ? { kind: 'repo', repoId: known.repoId, known: true }
    : { kind: 'hypothetical' };
}

/**
 * The draft spec the engine costs for a method (MTP: the checkpoint's head), or undefined when it is not available
 * (no MTP head, a drafter repo not loaded yet or failed).
 */
export function draftSpecOf(
  method: SpecDecMethod,
  source: DrafterSource | undefined,
  model: ModelSpec,
  status: DrafterStatus | undefined,
): ModelSpec['draft'] {
  if (method === 'mtp') return model.draft?.kind === 'mtp' ? model.draft : undefined;
  if (source === undefined) return undefined;
  switch (source.kind) {
    case 'checkpoint':
      return source.spec;
    case 'hypothetical':
      return hypotheticalDrafter(model, method);
    case 'repo':
      return status?.kind === 'ok' && status.repoId === source.repoId ? status.spec : undefined;
  }
}

/** The draft length k in effect: the user's, else the method's default for this draft (1 for single-step MTP heads). Count. */
export function effectiveDraftLength(specDec: SpecDecChoice, draft: ModelSpec['draft']): number {
  return specDec.k ?? defaultDraftLength(specDec.method, draft);
}

/** The acceptance in effect: the user's α (i.i.d.), else the method × use-case preset (with the MTP tail rule). */
export function effectiveAcceptance(
  specDec: SpecDecChoice,
  k: number,
  draft: ModelSpec['draft'],
): AcceptanceModel {
  if (specDec.alpha !== undefined) return { kind: 'iid', alpha: specDec.alpha };
  const useCase: SpecUseCase = specDec.useCase ?? 'prose';
  return presetAcceptance(specDec.method, useCase, k, draft);
}

/** True if any layer of the model has a routed-expert (MoE) block. */
export function hasMoe(model: ModelSpec): boolean {
  return model.layers.some((l) => l.blocks.some((b) => b.kind === 'moe'));
}

/** True if any layer of the model has an n-gram lookup table (Qwen PLE, DeepSeek Engram). */
export function hasLookupTables(model: ModelSpec): boolean {
  return model.layers.some((l) => l.blocks.some((b) => b.kind === 'ngram_embedding'));
}

/** True if the model ships an n-gram table meant to live off-device (`offDeviceByDesign`). */
export function tablesOffDeviceByDesign(model: ModelSpec): boolean {
  return model.layers.some((l) =>
    l.blocks.some((b) => b.kind === 'ngram_embedding' && b.offDeviceByDesign === true),
  );
}

/** Indices of the decoder layers that have an n-gram lookup table, ascending. Zero-based. */
export function tableLayerIndices(model: ModelSpec): number[] {
  return model.layers.flatMap((l, i) =>
    l.blocks.some((b) => b.kind === 'ngram_embedding') ? [i] : [],
  );
}

/**
 * The layer ranges of a lookup-table placement: `splitTierRanges` over the K table layers `t_0 < … < t_{K−1}` (so a
 * split counts table layers), each table-index range `[i, j)` widened to the decoder layers
 * `[i = 0 ? 0 : t_i, j = K ? L : t_j)`, so the ranges cover `[0, L)` without gaps. One range for a single tier.
 */
function tableRanges(
  tier: PlacementChoice,
  split: TierSplitChoice | undefined,
  tableLayers: readonly number[],
  layerCount: number,
): TierRange[] {
  const k = tableLayers.length;
  const byTableIndex = placementRanges(tier, split, 0, k);
  return byTableIndex.map(({ tier: t, layers: { from, to } }) => ({
    tier: t,
    layers: {
      from: from === 0 ? 0 : tableLayers[from],
      to: to === k ? layerCount : tableLayers[to],
    },
  }));
}

/** Parameters of the model's n-gram tables (Σ tableRows · rowDim over every table). Unit: params. */
export function lookupTableParams(model: ModelSpec): number {
  let params = 0;
  for (const layer of model.layers)
    for (const b of layer.blocks)
      if (b.kind === 'ngram_embedding') params += b.tableRows * b.rowDim;
  return params;
}

/** The engine's hit-ratio model of a UI choice. */
function hitRatioModel(h: HitRatioChoice | undefined): RoutedExpertRule['hitRatio'] {
  if (typeof h === 'number') return { kind: 'custom', h };
  return { kind: h ?? 'uniform' };
}

/**
 * The tier ranges of a group's offloaded layers `[from, L)`: the one tier over the whole range, or the host-RAM / SSD
 * ranges of a split (`splitTierRanges`; one range when a side is empty).
 */
function placementRanges(
  tier: PlacementChoice,
  split: TierSplitChoice | undefined,
  from: number,
  layerCount: number,
): TierRange[] {
  return tier === 'split'
    ? splitTierRanges(from, layerCount, split)
    : [{ tier, layers: { from, to: layerCount } }];
}

/** Result of {@link deploymentConfig}. */
export interface DeploymentResult {
  /** The engine config; undefined when nothing applies (then `SimInput.deployment` is omitted). */
  config?: DeploymentConfig;
  /** Problems by field (`offload`, `specDec`); errors stop the simulation. */
  issues: FieldIssue[];
}

/**
 * The engine's `DeploymentConfig` of the state's deployment choices for `model`, with the problems:
 * - offload: experts only on MoE models, tables only on models with a table (else a warning and nothing is placed);
 *   a first offloaded layer at or past the last layer places nothing (a warning);
 * - a host-RAM + SSD split (`tier: 'split'`) becomes two rules over disjoint layer ranges (experts: both with the same
 *   f and h; tables: split by table layers, with the same scale layout); a split with an empty side, like a single
 *   tier, becomes the one-rule object form;
 * - spec-dec: MTP needs the checkpoint's head; a block method needs its drafter loaded (`drafter`).
 */
export function deploymentConfig(
  state: AppState,
  model: ModelSpec,
  drafter: DrafterStatus | undefined,
): DeploymentResult {
  const issues: FieldIssue[] = [];
  const warn = (field: 'offload' | 'specDec', message: string) =>
    issues.push({ field, level: 'warning', message });
  const error = (field: 'offload' | 'specDec', message: string) =>
    issues.push({ field, level: 'error', message });
  const layerCount = model.layers.length;
  const choice = state.deployment;

  let offload: OffloadConfig | undefined;
  const o = choice.offload;
  if (o !== undefined) {
    const out: OffloadConfig = {};
    if (o.experts !== undefined) {
      const from = o.experts.fromLayer ?? 0;
      if (!hasMoe(model))
        warn('offload', 'The model has no routed experts; expert offload has no effect.');
      else if (from >= layerCount)
        warn(
          'offload',
          `Offloaded layers start at ${from}, past the last layer (${layerCount - 1}); nothing is offloaded.`,
        );
      else {
        const residentFraction = o.experts.residentFraction ?? 0;
        const hitRatio = hitRatioModel(o.experts.hitRatio);
        const rule = (range: TierRange, whole: boolean): RoutedExpertRule => ({
          tier: range.tier,
          // A rule over every layer keeps the original form (no `layers`).
          ...(whole ? {} : { layers: range.layers }),
          residentFraction,
          hitRatio,
        });
        const ranges = placementRanges(o.experts.tier, o.experts.split, from, layerCount);
        // One tier (or a split with an empty side): the single-rule object form, as before splits existed.
        out.routedExperts =
          ranges.length === 1 ? rule(ranges[0], from === 0) : ranges.map((r) => rule(r, false));
      }
    }
    if (o.tables !== undefined) {
      if (!hasLookupTables(model))
        warn('offload', 'The model has no n-gram lookup tables; their placement has no effect.');
      else {
        const split = o.tables.tier === 'split';
        const chosenLayout = o.tables.scaleLayout;
        /**
         * The rule's scale layout. 'pinned_host' (scales pinned in host RAM, rows read from the SSD) only means
         * something for an SSD rule: on a split the host-RAM side keeps its tier default (its scales are already in RAM).
         */
        const layoutFor = (tier: TierRange['tier']) =>
          split && tier === 'host_ram' && chosenLayout === 'pinned_host' ? undefined : chosenLayout;
        const rule = (range: TierRange, whole: boolean): LookupTableRule => {
          const scaleLayout = layoutFor(range.tier);
          return {
            tier: range.tier,
            // A rule over every layer keeps the original form (no `layers`).
            ...(whole ? {} : { layers: range.layers }),
            ...(scaleLayout !== undefined ? { scaleLayout } : {}),
          };
        };
        const ranges = tableRanges(
          o.tables.tier,
          o.tables.split,
          tableLayerIndices(model),
          layerCount,
        );
        out.lookupTables =
          ranges.length === 1 ? rule(ranges[0], true) : ranges.map((r) => rule(r, false));
      }
    }
    const placed = out.routedExperts !== undefined || out.lookupTables !== undefined;
    if (placed)
      offload = {
        ...out,
        ...(o.queueDepth !== undefined ? { queueDepth: o.queueDepth } : {}),
        ...(o.ssdReadRate === 'sustained' ? { ssdReadRate: o.ssdReadRate } : {}),
        ...(o.prefetchAccuracy !== undefined ? { prefetchAccuracy: o.prefetchAccuracy } : {}),
      };
  }

  let speculative: SpecDecodeConfig | undefined;
  const s = choice.specDec;
  if (s !== undefined) {
    const source = drafterSource(s, model, state.model.id);
    const draft = draftSpecOf(s.method, source, model, drafter);
    if (s.method === 'mtp' && draft === undefined)
      error('specDec', 'This model has no MTP head; pick another method.');
    else if (source?.kind === 'repo' && draft === undefined) {
      if (drafter?.kind === 'error' && drafter.repoId === source.repoId)
        error('specDec', `Drafter ${source.repoId}: ${drafter.message}`);
      else error('specDec', `Loading drafter ${source.repoId}…`);
    } else if (draft !== undefined) {
      const k = effectiveDraftLength(s, draft);
      const acceptance = effectiveAcceptance(s, k, draft);
      const drafterConfig: SpecDecodeConfig['drafter'] =
        draft.kind === 'block_drafter'
          ? source?.kind === 'hypothetical'
            ? { kind: 'hypothetical', spec: draft }
            : {
                kind: 'repo',
                repoId: source?.kind === 'repo' ? source.repoId : state.model.id,
                spec: draft,
              }
          : undefined;
      speculative = {
        method: s.method,
        ...(drafterConfig !== undefined ? { drafter: drafterConfig } : {}),
        draftLength: k,
        acceptance,
      };
    }
  }

  const config: DeploymentConfig = {
    ...(offload !== undefined ? { offload } : {}),
    ...(speculative !== undefined ? { speculative } : {}),
  };
  return { ...(Object.keys(config).length > 0 ? { config } : {}), issues };
}
