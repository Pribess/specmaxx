/**
 * Device variants as a base spec plus an ordered list of typed partial overrides (VRAM mods, tensor-rate
 * unlocks, power/clock limits, P2P over PCIe), instead of copied specs. Device-level and link-level modifiers
 * are separate unions because P2P is a property of the link (`LinkSpec.p2p`), not of the chip.
 * The engine applies them once at the start of `simulate`, so cost code only sees plain specs.
 * See docs/simulation-model.md §4 ("Device variants and modifiers").
 */

import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { ComputeEngine, DeviceSpec, EstimatedField, GemmTiling, LinkSpec, MatmulPeak, SourceRef } from './types';

/** Provenance every modifier must carry (project rule: every number is sourced). */
export interface ModifierProvenance {
  /** Human description shown in the UI, e.g. "48 GB VRAM mod (2 GB → 4 GB GDDR modules)". */
  note: string;
  /** Forum, measurement or vendor URLs; `fields` are paths in the MODIFIED spec. Non-empty unless `estimatedReason` is set. */
  sources: SourceRef[];
  /** Why the numbers are believed when no source exists; required when `sources` is empty. */
  estimatedReason?: string;
}

/** One typed partial override of a {@link DeviceSpec}. Applied in list order by {@link applyDeviceModifiers}. */
export type DeviceModifier =
  /** Replace the SKU capacity options (VRAM mod), per device. Unit: bytes. */
  | ({ kind: 'memoryCapacity'; capacityOptionsBytes: number[] } & ModifierProvenance)
  /** Replace the peak memory bandwidth (memory mod/overclock, or a lower-clocked SKU), per device. Unit: B/s. */
  | ({ kind: 'memoryBandwidth'; bandwidthBytesPerSec: number } & ModifierProvenance)
  /**
   * Multiply every MatmulPeak that matches ALL given filters (an omitted filter matches any value) by the
   * dimensionless `factor` (> 0). Must match ≥ 1 peak. Example: full-rate fp32 accumulate on GeForce =
   * `{ input: 'bf16', accumulate: 'fp32', factor: 2 }` (and one for fp16).
   */
  | ({
      kind: 'peakScale';
      engineId?: string;
      input?: MathFormat;
      accumulate?: AccumulateType;
      sparsity?: 'dense' | '2:4';
      factor: number;
    } & ModifierProvenance)
  /**
   * Power/clock limit: multiply ALL matmul and vector peaks of one engine (default: every engine) by the
   * dimensionless `factor` (> 0). Memory bandwidth is unaffected (use `memoryBandwidth` for that).
   */
  | ({ kind: 'clockScale'; engineId?: string; factor: number } & ModifierProvenance);

/** One typed partial override of a {@link LinkSpec}. Applied in list order by {@link applyLinkModifiers}. */
export type LinkModifier =
  /**
   * Enable or disable direct P2P (e.g. a driver that enables GeForce P2P over PCIe). Setting `'direct'` drops
   * `hostBounce`; setting `'host_bounce'` requires `hostBounce` (host memory B/s, extra latency in seconds).
   */
  | ({
      kind: 'p2p';
      p2p: 'direct' | 'host_bounce';
      hostBounce?: { hostMemBandwidthBytesPerSec: number; extraLatencySec: number };
    } & ModifierProvenance)
  /** Replace the per-device, per-direction bandwidth (e.g. a GPU running at PCIe x8). Unit: B/s. */
  | ({ kind: 'linkBandwidth'; bandwidthPerDirectionBytesPerSec: number } & ModifierProvenance);

/** A named device variant: the form presets use (the preset-library task fills these in). */
export interface DeviceVariant {
  /** Stable id of the variant, e.g. `'rtx-4090-48gb'`. */
  id: string;
  /** Display name of the variant. */
  name: string;
  /** The unmodified base device. */
  base: DeviceSpec;
  /** Modifiers applied to `base`, in order. */
  modifiers: DeviceModifier[];
}

// ---------------------------------------------------------------------------------------------------------
// Copies (explicit spreads: structuredClone is not in the ES2022 lib). The base is never mutated.

function copyTiling(t: GemmTiling): GemmTiling {
  return { ...t, ctaTiles: t.ctaTiles.map((c) => ({ ...c })) };
}

function copyEngine(e: ComputeEngine): ComputeEngine {
  const tiling: ComputeEngine['gemmTiling'] = { default: copyTiling(e.gemmTiling.default) };
  for (const key of Object.keys(e.gemmTiling) as (keyof ComputeEngine['gemmTiling'])[]) {
    const t = e.gemmTiling[key];
    if (key !== 'default' && t !== undefined) tiling[key] = copyTiling(t);
  }
  return {
    ...e,
    matmulPeaks: e.matmulPeaks.map((p) => ({ ...p })),
    vectorFlopsPerSec: { ...e.vectorFlopsPerSec },
    gemmTiling: tiling,
  };
}

function copySources(sources: SourceRef[]): SourceRef[] {
  return sources.map((s) => ({ ...s, fields: [...s.fields] }));
}

function copyDevice(d: DeviceSpec): DeviceSpec {
  return {
    ...d,
    engines: d.engines.map(copyEngine),
    memory: { ...d.memory, capacityOptionsBytes: [...d.memory.capacityOptionsBytes] },
    sources: copySources(d.sources),
    estimated: d.estimated.map((e) => ({ ...e })),
    ...(d.appliedModifiers !== undefined ? { appliedModifiers: [...d.appliedModifiers] } : {}),
  };
}

function copyLink(l: LinkSpec): LinkSpec {
  return {
    ...l,
    ...(l.hostBounce !== undefined ? { hostBounce: { ...l.hostBounce } } : {}),
    sources: copySources(l.sources),
    estimated: l.estimated.map((e) => ({ ...e })),
    ...(l.appliedModifiers !== undefined ? { appliedModifiers: [...l.appliedModifiers] } : {}),
  };
}

// ---------------------------------------------------------------------------------------------------------
// Matching and checks

/** True if `engine` is selected by a modifier's optional `engineId` (omitted = every engine). */
function engineSelected(engineId: string | undefined, engine: ComputeEngine): boolean {
  return engineId === undefined || engine.id === engineId;
}

/** True if `peak` matches ALL given filters of a `peakScale` modifier (an omitted filter matches any value). */
function peakMatches(mod: Extract<DeviceModifier, { kind: 'peakScale' }>, peak: MatmulPeak): boolean {
  return (
    (mod.input === undefined || peak.input === mod.input) &&
    (mod.accumulate === undefined || peak.accumulate === mod.accumulate) &&
    (mod.sparsity === undefined || peak.sparsity === mod.sparsity)
  );
}

/** Errors in the provenance of a modifier (sources non-empty, or an estimatedReason). */
function provenanceErrors(mod: ModifierProvenance, label: string): string[] {
  const errors: string[] = [];
  if (mod.note.trim() === '') errors.push(`${label}: note is empty`);
  if (mod.sources.length === 0 && (mod.estimatedReason === undefined || mod.estimatedReason.trim() === '')) {
    errors.push(`${label}: needs sources or an estimatedReason (every number must be sourced)`);
  }
  return errors;
}

/** Errors for a positive finite number field. */
function positiveErrors(value: number, label: string): string[] {
  return Number.isFinite(value) && value > 0 ? [] : [`${label} must be a finite number > 0 (got ${value})`];
}

/**
 * Errors of ONE device modifier against the spec it is applied to (the base after the earlier modifiers):
 * provenance present, numbers finite and > 0, `engineId` (if given) exists, a `peakScale` filter matches
 * ≥ 1 peak. Empty = valid. Used by {@link applyDeviceModifiers} (throws) and by `validateDeviceModifiers`.
 */
export function deviceModifierErrors(spec: DeviceSpec, mod: DeviceModifier): string[] {
  const label = `modifier ${mod.kind}`;
  const errors = provenanceErrors(mod, label);
  switch (mod.kind) {
    case 'memoryCapacity':
      if (mod.capacityOptionsBytes.length === 0) errors.push(`${label}: capacityOptionsBytes is empty`);
      mod.capacityOptionsBytes.forEach((c, i) => errors.push(...positiveErrors(c, `${label}: capacityOptionsBytes[${i}]`)));
      break;
    case 'memoryBandwidth':
      errors.push(...positiveErrors(mod.bandwidthBytesPerSec, `${label}: bandwidthBytesPerSec`));
      break;
    case 'peakScale':
    case 'clockScale': {
      errors.push(...positiveErrors(mod.factor, `${label}: factor`));
      if (mod.engineId !== undefined && !spec.engines.some((e) => e.id === mod.engineId)) {
        errors.push(`${label}: unknown engineId '${mod.engineId}'`);
      }
      if (mod.kind === 'peakScale') {
        const matched = spec.engines
          .filter((e) => engineSelected(mod.engineId, e))
          .reduce((count, e) => count + e.matmulPeaks.filter((p) => peakMatches(mod, p)).length, 0);
        if (matched === 0) errors.push(`${label}: filter matches no matmul peak (typo?)`);
      }
      break;
    }
    default: {
      const _never: never = mod;
      throw new Error(`unknown device modifier ${String(_never)}`);
    }
  }
  return errors;
}

/**
 * Errors of ONE link modifier against the link it is applied to: provenance present, numbers finite and > 0,
 * `hostBounce` given iff `p2p === 'host_bounce'`. Empty = valid.
 */
export function linkModifierErrors(link: LinkSpec, mod: LinkModifier): string[] {
  void link;
  const label = `link modifier ${mod.kind}`;
  const errors = provenanceErrors(mod, label);
  switch (mod.kind) {
    case 'p2p':
      if (mod.p2p === 'host_bounce') {
        if (mod.hostBounce === undefined) {
          errors.push(`${label}: p2p 'host_bounce' requires hostBounce`);
        } else {
          errors.push(...positiveErrors(mod.hostBounce.hostMemBandwidthBytesPerSec, `${label}: hostBounce.hostMemBandwidthBytesPerSec`));
          if (!(Number.isFinite(mod.hostBounce.extraLatencySec) && mod.hostBounce.extraLatencySec >= 0)) {
            errors.push(`${label}: hostBounce.extraLatencySec must be finite and ≥ 0`);
          }
        }
      } else if (mod.hostBounce !== undefined) {
        errors.push(`${label}: p2p 'direct' must not carry hostBounce`);
      }
      break;
    case 'linkBandwidth':
      errors.push(...positiveErrors(mod.bandwidthPerDirectionBytesPerSec, `${label}: bandwidthPerDirectionBytesPerSec`));
      break;
    default: {
      const _never: never = mod;
      throw new Error(`unknown link modifier ${String(_never)}`);
    }
  }
  return errors;
}

/** One-line human description of a device modifier, e.g. "variant: clockScale ×0.8 (450 W power limit)". */
export function describeDeviceModifier(mod: DeviceModifier): string {
  switch (mod.kind) {
    case 'memoryCapacity':
      return `variant: memoryCapacity [${mod.capacityOptionsBytes.map((c) => `${c / 1e9} GB`).join(', ')}] (${mod.note})`;
    case 'memoryBandwidth':
      return `variant: memoryBandwidth ${mod.bandwidthBytesPerSec / 1e9} GB/s (${mod.note})`;
    case 'peakScale': {
      const filter = [mod.engineId, mod.input, mod.accumulate, mod.sparsity].filter((x) => x !== undefined).join('/');
      return `variant: peakScale ${filter === '' ? 'all' : filter} ×${mod.factor} (${mod.note})`;
    }
    case 'clockScale':
      return `variant: clockScale ${mod.engineId ?? 'all engines'} ×${mod.factor} (${mod.note})`;
    default: {
      const _never: never = mod;
      throw new Error(`unknown device modifier ${String(_never)}`);
    }
  }
}

/** One-line human description of a link modifier, e.g. "link variant: p2p direct (P2P driver)". */
export function describeLinkModifier(mod: LinkModifier): string {
  switch (mod.kind) {
    case 'p2p':
      return `link variant: p2p ${mod.p2p} (${mod.note})`;
    case 'linkBandwidth':
      return `link variant: linkBandwidth ${mod.bandwidthPerDirectionBytesPerSec / 1e9} GB/s per direction (${mod.note})`;
    default: {
      const _never: never = mod;
      throw new Error(`unknown link modifier ${String(_never)}`);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// Apply

/** Dotted spec paths a device modifier changes (its provenance covers these). */
function deviceModifierPaths(spec: DeviceSpec, mod: DeviceModifier): string[] {
  switch (mod.kind) {
    case 'memoryCapacity':
      return ['memory.capacityOptionsBytes'];
    case 'memoryBandwidth':
      return ['memory.bandwidthBytesPerSec'];
    case 'peakScale':
      return spec.engines.flatMap((e, i) => (engineSelected(mod.engineId, e) ? [`engines.${i}.matmulPeaks`] : []));
    case 'clockScale':
      return spec.engines.flatMap((e, i) =>
        engineSelected(mod.engineId, e) ? [`engines.${i}.matmulPeaks`, `engines.${i}.vectorFlopsPerSec`] : [],
      );
    default: {
      const _never: never = mod;
      throw new Error(`unknown device modifier ${String(_never)}`);
    }
  }
}

/** Applies ONE (already checked) device modifier to a spec we own (a copy), returning the new spec. */
function applyOneDeviceModifier(spec: DeviceSpec, mod: DeviceModifier): DeviceSpec {
  switch (mod.kind) {
    case 'memoryCapacity':
      return { ...spec, memory: { ...spec.memory, capacityOptionsBytes: [...mod.capacityOptionsBytes] } };
    case 'memoryBandwidth':
      return { ...spec, memory: { ...spec.memory, bandwidthBytesPerSec: mod.bandwidthBytesPerSec } };
    case 'peakScale':
      // Formula: peak' = peak · factor   for every peak matching all filters
      // See docs/simulation-model.md §4 ("Device variants and modifiers")
      return {
        ...spec,
        engines: spec.engines.map((e) =>
          engineSelected(mod.engineId, e)
            ? {
                ...e,
                matmulPeaks: e.matmulPeaks.map((p) =>
                  peakMatches(mod, p) ? { ...p, flopsPerSec: p.flopsPerSec * mod.factor } : p,
                ),
              }
            : e,
        ),
      };
    case 'clockScale': {
      // Formula: peak' = peak · factor   for every matmul AND vector peak of the selected engines
      // See docs/simulation-model.md §4 ("Device variants and modifiers")
      const scaleVector = (v: ComputeEngine['vectorFlopsPerSec']): ComputeEngine['vectorFlopsPerSec'] => {
        const out: ComputeEngine['vectorFlopsPerSec'] = {};
        for (const key of Object.keys(v) as (keyof ComputeEngine['vectorFlopsPerSec'])[]) {
          const value = v[key];
          if (value !== undefined) out[key] = value * mod.factor;
        }
        return out;
      };
      return {
        ...spec,
        engines: spec.engines.map((e) =>
          engineSelected(mod.engineId, e)
            ? {
                ...e,
                matmulPeaks: e.matmulPeaks.map((p) => ({ ...p, flopsPerSec: p.flopsPerSec * mod.factor })),
                vectorFlopsPerSec: scaleVector(e.vectorFlopsPerSec),
              }
            : e,
        ),
      };
    }
    default: {
      const _never: never = mod;
      throw new Error(`unknown device modifier ${String(_never)}`);
    }
  }
}

/** Provenance entries of a modifier: its sources as given, and its estimatedReason on each covered path. */
function provenanceOf(mod: ModifierProvenance, paths: string[]): { sources: SourceRef[]; estimated: EstimatedField[] } {
  return {
    sources: copySources(mod.sources),
    estimated:
      mod.estimatedReason !== undefined ? paths.map((field) => ({ field, reason: `${mod.note}: ${mod.estimatedReason}` })) : [],
  };
}

/**
 * Pure: returns a NEW DeviceSpec (the base is never mutated). Applies `mods` in order (scales compose
 * multiplicatively), appends each modifier's sources to `sources` and its `estimatedReason` to `estimated`
 * (field = the path the modifier kind covers), and records the list in `appliedModifiers`.
 * Throws on an invalid modifier (call `validateDeviceModifiers` first).
 */
export function applyDeviceModifiers(base: DeviceSpec, mods: DeviceModifier[]): DeviceSpec {
  let spec = copyDevice(base);
  for (const mod of mods) {
    const errors = deviceModifierErrors(spec, mod);
    if (errors.length > 0) throw new Error(errors.join('\n'));
    const provenance = provenanceOf(mod, deviceModifierPaths(spec, mod));
    spec = applyOneDeviceModifier(spec, mod);
    spec = {
      ...spec,
      sources: [...spec.sources, ...provenance.sources],
      estimated: [...spec.estimated, ...provenance.estimated],
    };
  }
  return { ...spec, appliedModifiers: [...(base.appliedModifiers ?? []), ...mods] };
}

/** Dotted link paths a link modifier changes (its provenance covers these). */
function linkModifierPaths(mod: LinkModifier): string[] {
  switch (mod.kind) {
    case 'p2p':
      return ['p2p', 'hostBounce'];
    case 'linkBandwidth':
      return ['bandwidthPerDirectionBytesPerSec'];
    default: {
      const _never: never = mod;
      throw new Error(`unknown link modifier ${String(_never)}`);
    }
  }
}

/** Applies ONE (already checked) link modifier, returning the new link. */
function applyOneLinkModifier(link: LinkSpec, mod: LinkModifier): LinkSpec {
  switch (mod.kind) {
    case 'p2p': {
      // 'direct' drops hostBounce; 'host_bounce' takes the modifier's staging parameters.
      const { hostBounce: _dropped, ...rest } = link;
      void _dropped;
      return mod.p2p === 'direct' || mod.hostBounce === undefined
        ? { ...rest, p2p: mod.p2p }
        : { ...rest, p2p: mod.p2p, hostBounce: { ...mod.hostBounce } };
    }
    case 'linkBandwidth':
      return { ...link, bandwidthPerDirectionBytesPerSec: mod.bandwidthPerDirectionBytesPerSec };
    default: {
      const _never: never = mod;
      throw new Error(`unknown link modifier ${String(_never)}`);
    }
  }
}

/** Pure: returns a NEW LinkSpec with `mods` applied in order and recorded in `appliedModifiers` (as {@link applyDeviceModifiers}). */
export function applyLinkModifiers(base: LinkSpec, mods: LinkModifier[]): LinkSpec {
  let link = copyLink(base);
  for (const mod of mods) {
    const errors = linkModifierErrors(link, mod);
    if (errors.length > 0) throw new Error(errors.join('\n'));
    const provenance = provenanceOf(mod, linkModifierPaths(mod));
    link = applyOneLinkModifier(link, mod);
    link = {
      ...link,
      sources: [...link.sources, ...provenance.sources],
      estimated: [...link.estimated, ...provenance.estimated],
    };
  }
  return { ...link, appliedModifiers: [...(base.appliedModifiers ?? []), ...mods] };
}

/** The spec of a variant: `id`/`name` from the variant, everything else from `applyDeviceModifiers(base, modifiers)`. */
export function resolveDeviceVariant(v: DeviceVariant): DeviceSpec {
  return { ...applyDeviceModifiers(v.base, v.modifiers), id: v.id, name: v.name };
}
