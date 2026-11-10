/**
 * Unlock axes: the independent, toggleable ways a base device can be modified in practice (VRAM mods,
 * tensor-rate unlocks, P2P drivers, power and clock limits, ...). Each axis is one UI control: a checkbox when it
 * has one option, a radio group (plus "stock") when it has several mutually exclusive options (e.g. a 48 GB
 * or a 96 GB VRAM mod). An option is just a list of typed modifiers (`hw/modifiers.ts`), so the base preset is
 * never copied. See docs/simulation-model.md §4 ("Unlock axes").
 */

import type { DeviceModifier, LinkModifier } from './modifiers';

/** What an unlock axis changes (informational; drives grouping in the UI). */
export type UnlockAxisKind =
  | 'vram'
  | 'memoryClock'
  | 'tensorRate'
  | 'interconnect'
  | 'powerClock'
  | 'other';

/** One choice on an {@link UnlockAxis}. */
export interface UnlockOption {
  /** Stable id, unique within the axis, e.g. `'48gb'`. */
  id: string;
  /** Display label, e.g. `'48 GB (clamshell 2 GB modules)'`. */
  name: string;
  /** Device modifiers this option adds, in order (each carries its own sources / estimatedReason). */
  deviceModifiers: DeviceModifier[];
  /** Modifiers for the topology's intra-node link (e.g. P2P driver), in order. */
  intraNodeLinkModifiers?: LinkModifier[];
}

/**
 * One independent unlock axis of a base device. Options of one axis are mutually exclusive; options of
 * different axes combine freely (every combination must validate: enforced by `hw/presets.test.ts`).
 */
export interface UnlockAxis {
  /** Stable id, unique per device, e.g. `'vram-mod'`. */
  id: string;
  /** Display name, e.g. `'VRAM mod'`. */
  name: string;
  /** What it changes. */
  kind: UnlockAxisKind;
  /** What the unlock is, who does it, caveats; shown in the UI. Mention when its numbers are community-measured. */
  description: string;
  /** Mutually exclusive choices (≥ 1). "Stock" (no option) is always implied and not listed. */
  options: UnlockOption[];
}

/** A selection on a device's axes: axis id → option id. Axes absent from the map stay stock. */
export type UnlockSelection = Record<string, string>;

/** The modifiers a selection resolves to, ready for `HardwareStack.deviceModifiers` / `linkModifiers.intraNode`. */
export interface ResolvedUnlocks {
  /** Device modifiers of the chosen options, in axis order. */
  deviceModifiers: DeviceModifier[];
  /** Intra-node link modifiers of the chosen options, in axis order. */
  intraNodeLinkModifiers: LinkModifier[];
}

/**
 * Resolves a selection to modifiers, concatenating the chosen options' modifiers in the order the axes are
 * listed. Throws on an unknown axis or option id (a UI bug, not a user error).
 */
export function resolveUnlocks(axes: UnlockAxis[], selection: UnlockSelection): ResolvedUnlocks {
  for (const axisId of Object.keys(selection)) {
    if (!axes.some((a) => a.id === axisId)) throw new Error(`unknown unlock axis '${axisId}'`);
  }
  const deviceModifiers: DeviceModifier[] = [];
  const intraNodeLinkModifiers: LinkModifier[] = [];
  for (const axis of axes) {
    const optionId = selection[axis.id];
    if (optionId === undefined) continue;
    const option = axis.options.find((o) => o.id === optionId);
    if (option === undefined)
      throw new Error(`unknown option '${optionId}' on unlock axis '${axis.id}'`);
    deviceModifiers.push(...option.deviceModifiers);
    intraNodeLinkModifiers.push(...(option.intraNodeLinkModifiers ?? []));
  }
  return { deviceModifiers, intraNodeLinkModifiers };
}

/**
 * Every selection over `axes`: each axis stock or one of its options (∏ (options + 1) selections, the empty
 * selection included). Used by tests to check that unlocks combine cleanly in any combination.
 */
export function allUnlockSelections(axes: UnlockAxis[]): UnlockSelection[] {
  let selections: UnlockSelection[] = [{}];
  for (const axis of axes) {
    const next: UnlockSelection[] = [];
    for (const s of selections) {
      next.push(s);
      for (const o of axis.options) next.push({ ...s, [axis.id]: o.id });
    }
    selections = next;
  }
  return selections;
}

/** Structural errors of a device's axes: duplicate ids, empty option lists, duplicate option ids. */
export function unlockAxesErrors(axes: UnlockAxis[]): string[] {
  const errors: string[] = [];
  const seen = new Set<string>();
  for (const axis of axes) {
    if (seen.has(axis.id)) errors.push(`duplicate unlock axis id '${axis.id}'`);
    seen.add(axis.id);
    if (axis.options.length === 0) errors.push(`unlock axis '${axis.id}' has no options`);
    const optionIds = new Set<string>();
    for (const o of axis.options) {
      if (optionIds.has(o.id))
        errors.push(`unlock axis '${axis.id}': duplicate option id '${o.id}'`);
      optionIds.add(o.id);
      if (o.deviceModifiers.length === 0 && (o.intraNodeLinkModifiers ?? []).length === 0) {
        errors.push(`unlock axis '${axis.id}' option '${o.id}' has no modifiers`);
      }
    }
  }
  return errors;
}
