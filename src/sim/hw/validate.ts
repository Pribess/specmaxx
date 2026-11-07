/**
 * Validation of hardware specs and modifiers: physical sanity (finite, non-negative numbers; sparse ≥ dense;
 * tile shapes aligned to the MMA shape) and the project rule that EVERY number is sourced (covered by a
 * `sources[].fields` path or an `estimated[].field`). The preset-library task reuses these.
 * See docs/simulation-model.md §4 ("Estimated fields", "Device variants and modifiers").
 */

import type { MathFormat } from '../ir/dtype';
import { applyDeviceModifiers, applyLinkModifiers, deviceModifierErrors, linkModifierErrors } from './modifiers';
import type { DeviceModifier, LinkModifier } from './modifiers';
import type { ComputeEngine, DeviceSpec, EstimatedField, GemmTiling, LinkSpec, SourceRef, TopologySpec } from './types';

/** One numeric leaf of a spec object. */
export interface NumericLeaf {
  /** Dotted path from the spec root, array indices included (e.g. `'engines.0.matmulPeaks.3.flopsPerSec'`). */
  path: string;
  /** The number found there (unit as the field defines it). */
  value: number;
}

/** Keys that hold provenance, not physical numbers; {@link numericLeafPaths} skips them. */
const PROVENANCE_KEYS = new Set(['sources', 'estimated', 'appliedModifiers']);

/**
 * Every numeric leaf of `obj` (recursing through objects and arrays), with its dotted path, skipping the
 * provenance keys `sources`, `estimated` and `appliedModifiers`. Strings and booleans are not leaves.
 */
export function numericLeafPaths(obj: unknown, prefix = ''): NumericLeaf[] {
  if (typeof obj === 'number') return [{ path: prefix, value: obj }];
  if (obj === null || typeof obj !== 'object') return [];
  const leaves: NumericLeaf[] = [];
  const entries: [string, unknown][] = Array.isArray(obj)
    ? obj.map((v, i) => [String(i), v])
    : Object.entries(obj as Record<string, unknown>);
  for (const [key, value] of entries) {
    if (!Array.isArray(obj) && PROVENANCE_KEYS.has(key)) continue;
    leaves.push(...numericLeafPaths(value, prefix === '' ? key : `${prefix}.${key}`));
  }
  return leaves;
}

/** True if `path` equals one of `covers` or lies in the subtree of one (a prefix covers its subtree). */
export function isPathCovered(path: string, covers: string[]): boolean {
  return covers.some((c) => path === c || path.startsWith(`${c}.`));
}

/** Errors for numbers that are not finite or negative, and for numeric leaves no source or estimate covers. */
function leafErrors(obj: unknown, sources: SourceRef[], estimated: EstimatedField[], label: string): string[] {
  const covers = [...sources.flatMap((s) => s.fields), ...estimated.map((e) => e.field)];
  const errors: string[] = [];
  for (const leaf of numericLeafPaths(obj)) {
    if (!Number.isFinite(leaf.value) || leaf.value < 0) {
      errors.push(`${label}: ${leaf.path} must be finite and ≥ 0 (got ${leaf.value})`);
    }
    if (!isPathCovered(leaf.path, covers)) {
      errors.push(`${label}: ${leaf.path} is not covered by any sources[].fields or estimated[].field`);
    }
  }
  return errors;
}

/** Errors in the citation records themselves (empty URL, empty fields, empty reasons). */
function provenanceRecordErrors(sources: SourceRef[], estimated: EstimatedField[], label: string): string[] {
  const errors: string[] = [];
  sources.forEach((s, i) => {
    if (s.url.trim() === '') errors.push(`${label}: sources[${i}] has an empty url`);
    if (s.fields.length === 0) errors.push(`${label}: sources[${i}] (${s.url}) covers no fields`);
  });
  estimated.forEach((e, i) => {
    if (e.reason.trim() === '') errors.push(`${label}: estimated[${i}] (${e.field}) has an empty reason`);
  });
  return errors;
}

/** Errors in one tiling: positive MMA shape, ≥ 1 CTA tile, tiles aligned to the MMA shape, ctasPerUnit ≥ 1. */
function tilingErrors(t: GemmTiling, label: string): string[] {
  const errors: string[] = [];
  for (const [name, v] of [['mmaM', t.mmaM], ['mmaN', t.mmaN], ['mmaK', t.mmaK], ['ctasPerUnit', t.ctasPerUnit]] as const) {
    if (!Number.isInteger(v) || v < 1) errors.push(`${label}: ${name} must be an integer ≥ 1 (got ${v})`);
  }
  if (t.ctaTiles.length === 0) errors.push(`${label}: ctaTiles is empty`);
  t.ctaTiles.forEach((tile, i) => {
    if (tile.m <= 0 || tile.m % t.mmaM !== 0) errors.push(`${label}: ctaTiles[${i}].m = ${tile.m} is not a positive multiple of mmaM ${t.mmaM}`);
    if (tile.n <= 0 || tile.n % t.mmaN !== 0) errors.push(`${label}: ctaTiles[${i}].n = ${tile.n} is not a positive multiple of mmaN ${t.mmaN}`);
  });
  return errors;
}

/** True for integer input classes. */
function isIntegerInput(input: MathFormat): boolean {
  return input === 'int8' || input === 'int4';
}

/**
 * Physical-consistency errors of one engine (provenance is checked on the device). Every rate the engine divides
 * by (matmul peaks, vector peaks, the memory-bandwidth cap) must be > 0.
 */
function engineErrors(e: ComputeEngine, label: string): string[] {
  const errors: string[] = [];
  if (!Number.isInteger(e.computeUnits) || e.computeUnits < 1) {
    errors.push(`${label}: computeUnits must be an integer ≥ 1 (got ${e.computeUnits})`);
  }
  e.matmulPeaks.forEach((p, i) => {
    if (!(p.flopsPerSec > 0)) errors.push(`${label}: matmulPeaks.${i}.flopsPerSec must be > 0 (got ${p.flopsPerSec})`);
  });
  for (const [dtype, v] of Object.entries(e.vectorFlopsPerSec)) {
    if (v !== undefined && !(v > 0)) errors.push(`${label}: vectorFlopsPerSec.${dtype} must be > 0 (got ${v})`);
  }
  if (e.vectorFlopsPerSec.fp32 === undefined) errors.push(`${label}: vectorFlopsPerSec.fp32 is required (the last peak fallback)`);
  if (e.memBandwidthCapBytesPerSec !== undefined && !(e.memBandwidthCapBytesPerSec > 0)) {
    errors.push(`${label}: memBandwidthCapBytesPerSec must be > 0 (got ${e.memBandwidthCapBytesPerSec})`);
  }
  const keys = new Set<string>();
  for (const p of e.matmulPeaks) {
    const key = `${p.input}/${p.accumulate}/${p.sparsity}`;
    if (keys.has(key)) errors.push(`${label}: duplicate matmul peak ${key}`);
    keys.add(key);
    if (isIntegerInput(p.input) !== (p.accumulate === 'int32')) {
      errors.push(`${label}: peak ${key}: integer inputs accumulate in int32, float inputs in fp16/fp32`);
    }
    if (p.sparsity === '2:4') {
      const dense = e.matmulPeaks.find((d) => d.input === p.input && d.accumulate === p.accumulate && d.sparsity === 'dense');
      if (dense === undefined) errors.push(`${label}: sparse peak ${key} has no dense counterpart`);
      else if (p.flopsPerSec < dense.flopsPerSec) errors.push(`${label}: sparse peak ${key} is below its dense peak`);
    }
  }
  for (const input of new Set(e.matmulPeaks.map((p) => p.input))) {
    if (!isIntegerInput(input) && !e.matmulPeaks.some((p) => p.input === input && p.accumulate === 'fp32')) {
      errors.push(`${label}: float input ${input} has peaks but no fp32-accumulate entry`);
    }
  }
  for (const [key, tiling] of Object.entries(e.gemmTiling)) {
    if (tiling !== undefined) errors.push(...tilingErrors(tiling, `${label}.gemmTiling.${key}`));
  }
  return errors;
}

/**
 * Errors of a device spec; empty = valid. Checks: numbers finite and ≥ 0; every numeric leaf covered by a
 * source or an estimate; ≥ 1 engine with unique ids; per engine: sparse ≥ dense for the same key, every float
 * input has an fp32-accumulate entry, tiles are multiples of the MMA shape; `capacityOptionsBytes` non-empty
 * and positive; memory bandwidth > 0; every rate used as a divisor (matmul and vector peaks, an engine's memory
 * cap) > 0, and a vector fp32 peak present (the end of the peak fallback chain).
 */
export function validateDevice(d: DeviceSpec): string[] {
  const label = `device ${d.id}`;
  const errors = [...leafErrors(d, d.sources, d.estimated, label), ...provenanceRecordErrors(d.sources, d.estimated, label)];
  if (d.engines.length === 0) errors.push(`${label}: no engines`);
  const ids = new Set<string>();
  d.engines.forEach((e, i) => {
    if (ids.has(e.id)) errors.push(`${label}: duplicate engine id '${e.id}'`);
    ids.add(e.id);
    errors.push(...engineErrors(e, `${label}.engines.${i}`));
  });
  if (d.memory.capacityOptionsBytes.length === 0) errors.push(`${label}: memory.capacityOptionsBytes is empty`);
  if (d.memory.capacityOptionsBytes.some((c) => c <= 0)) errors.push(`${label}: memory.capacityOptionsBytes must be > 0`);
  if (!(d.memory.bandwidthBytesPerSec > 0)) errors.push(`${label}: memory.bandwidthBytesPerSec must be > 0`);
  return errors;
}

/**
 * Errors of a link spec; empty = valid. Checks: numbers finite and ≥ 0 and sourced; bandwidth > 0;
 * `hostBounce` present iff `p2p === 'host_bounce'`, with host memory bandwidth > 0.
 */
export function validateLink(l: LinkSpec, label = `link ${l.name}`): string[] {
  const errors = [...leafErrors(l, l.sources, l.estimated, label), ...provenanceRecordErrors(l.sources, l.estimated, label)];
  if (!(l.bandwidthPerDirectionBytesPerSec > 0)) errors.push(`${label}: bandwidthPerDirectionBytesPerSec must be > 0`);
  if ((l.p2p === 'host_bounce') !== (l.hostBounce !== undefined)) {
    errors.push(`${label}: hostBounce must be present iff p2p is 'host_bounce'`);
  }
  if (l.hostBounce !== undefined && !(l.hostBounce.hostMemBandwidthBytesPerSec > 0)) {
    errors.push(`${label}: hostBounce.hostMemBandwidthBytesPerSec must be > 0`);
  }
  return errors;
}

/**
 * Errors of a topology; empty = valid. `devicesPerNode` and `nodes` must be integers ≥ 1 (they describe the
 * user's configuration, so they need no citation); each present link must pass {@link validateLink}.
 * Whether a link is REQUIRED depends on the parallelism (checked by `engine/placement.ts`).
 */
export function validateTopology(t: TopologySpec): string[] {
  const label = `topology ${t.id}`;
  const errors: string[] = [];
  if (!Number.isInteger(t.devicesPerNode) || t.devicesPerNode < 1) errors.push(`${label}: devicesPerNode must be an integer ≥ 1`);
  if (!Number.isInteger(t.nodes) || t.nodes < 1) errors.push(`${label}: nodes must be an integer ≥ 1`);
  if (t.intraNode !== undefined) errors.push(...validateLink(t.intraNode, `${label}.intraNode`));
  if (t.interNode !== undefined) errors.push(...validateLink(t.interNode, `${label}.interNode`));
  return errors;
}

/**
 * Errors of applying `mods` to `base`, in order; empty = valid. Each modifier is checked against the spec as
 * modified by the ones before it (factor > 0, engineId exists, a peakScale filter matches ≥ 1 peak, provenance
 * present), and the final result must pass {@link validateDevice}.
 */
export function validateDeviceModifiers(base: DeviceSpec, mods: DeviceModifier[]): string[] {
  let spec = base;
  for (const [i, mod] of mods.entries()) {
    const errors = deviceModifierErrors(spec, mod).map((e) => `deviceModifiers[${i}]: ${e}`);
    if (errors.length > 0) return errors;
    spec = applyDeviceModifiers(spec, [mod]);
  }
  return validateDevice(spec).map((e) => `after modifiers: ${e}`);
}

/** Errors of applying `mods` to `link`, in order (as {@link validateDeviceModifiers}); empty = valid. */
export function validateLinkModifiers(link: LinkSpec, mods: LinkModifier[]): string[] {
  let current = link;
  for (const [i, mod] of mods.entries()) {
    const errors = linkModifierErrors(current, mod).map((e) => `linkModifiers[${i}]: ${e}`);
    if (errors.length > 0) return errors;
    current = applyLinkModifiers(current, [mod]);
  }
  return validateLink(current).map((e) => `after link modifiers: ${e}`);
}
