/**
 * The persistent UI state: exactly what the URL holds (see `url.ts`), plus the reducer that changes it.
 * Pure TypeScript (no React, no DOM), so it runs in tests, the app and the worker alike.
 *
 * Derived things (the resolved device, the topology, the SimInput) are computed from this state by
 * `hardware.ts` and `buildInput.ts`; nothing derived is stored here. Absent optional fields mean "the
 * default" (model default dtype, the device's default link, stock unlocks), so the URL can omit them.
 */

import type { Dtype } from '../../sim/ir/dtype';
import type { ParallelConfig } from '../../sim/ir/model';
import type { SimOptions, Workload } from '../../sim/engine/types';
import type { UnlockSelection } from '../../sim/hw/unlocks';
import { DEFAULT_SIM_OPTIONS } from '../../sim/engine/simulate';
import type { CablingId } from '../../sim/hw/presets/catalog';
import { DEFAULT_MODEL_ID } from './modelShortlist';
import { devicesPerNodeForLocalLink, nearestCount, normalizeCabling, normalizeLocalLink, setupOptionsOf } from './hardware';

/** Which model to simulate: a Hugging Face repo (the default one is bundled, so it needs no network). */
export type ModelChoice = {
  kind: 'hf';
  /** Repo id `owner/name`. */
  id: string;
  /** Branch, tag or commit; absent = `main`. Never `'main'` itself (normalized away). */
  revision?: string;
};

/** User overrides of the model's default numeric formats; an absent field keeps the model default. */
export interface QuantOverrides {
  /** Weight storage dtype of every linear module, or `'exl3'` (rules: `quant.ts`, `applyQuantOverride`). */
  weights?: Dtype | 'exl3';
  /** EXL3 only: average bits per weight of the linear layers; absent = the checkpoint's `bits`, else 4. Unit: bits/weight. */
  bpw?: number;
  /** EXL3 only: LM-head bits per weight; absent = the checkpoint's `head_bits`, else 6. Unit: bits/weight. */
  headBpw?: number;
  /** GEMM-input / activation dtype. */
  activations?: Dtype;
  /** KV-cache dtype (also the compressed-KV and indexer-key caches). */
  kvCache?: Dtype;
}

/** Spec edits: absolute values (SI units, as the spec field defines them) at `numericLeafPaths` paths. */
export type SpecEdits = Record<string, number>;

/** The hardware choice: a device preset, its unlocks, the device and node counts, the links, and spec edits. */
export interface HardwareChoice {
  /** `DeviceSpec.id` of a preset in `DEVICE_PRESETS`. */
  deviceId: string;
  /** Unlock selection on the device's axes (axis id → option id); absent axes stay stock. */
  unlocks: UnlockSelection;
  /** Engine id; absent = the device's first engine. */
  engineId?: string;
  /** Index into the (unlocked) device's `memory.capacityOptionsBytes`; 0 = its default. */
  capacityIndex: number;
  /** Devices per node, integer ≥ 1. Count. */
  devicesPerNode: number;
  /** Nodes, integer ≥ 1. Count. */
  nodes: number;
  /** `linkPresetId` of the intra-node link; absent = the device's default (`defaultLinks`). */
  intraLinkId?: string;
  /** `linkPresetId` of the inter-node link; absent = the device's default. */
  interLinkId?: string;
  /**
   * How the machines are cabled (`CablingId`: switch, full mesh, full mesh with 2 cables per pair, ring, daisy
   * chain); absent = the default for the network link and counts (`defaultCabling`). Kept only while the link and
   * counts allow it and it differs from the default (`normalizeCabling`).
   */
  cabling?: CablingId;
  /** Edits on the device spec, applied after the unlocks (the user's number wins). */
  deviceEdits: SpecEdits;
  /** Edits on the intra-node link, applied after the unlock link modifiers. */
  intraEdits: SpecEdits;
  /** Edits on the inter-node link. */
  interEdits: SpecEdits;
}

/** Everything the user chose; the URL encodes exactly this. */
export interface AppState {
  /** Model choice. */
  model: ModelChoice;
  /** Numeric-format overrides. */
  quant: QuantOverrides;
  /** Hardware choice. */
  hw: HardwareChoice;
  /** Parallelism degrees (each an integer ≥ 1 once valid; the state may hold invalid values while editing). */
  parallel: ParallelConfig;
  /** Batch size, prefill length (prompt tokens from an empty KV cache) and decode context (tokens already cached). */
  workload: Workload;
  /** Assumption switches that differ from `DEFAULT_SIM_OPTIONS` (a default value is never stored). */
  options: Partial<SimOptions>;
}

/** Device of the default state. */
export const DEFAULT_DEVICE_ID = 'rtx-5090';

/**
 * The default hardware choice: one stock RTX 5090. The default model (Qwen3.8-27B, about 54GB of bf16
 * weights) does not fit its 32GB, so the page opens on the "does not fit" verdict; that is intended.
 */
export const DEFAULT_HW: HardwareChoice = {
  deviceId: DEFAULT_DEVICE_ID,
  unlocks: {},
  capacityIndex: 0,
  devicesPerNode: 1,
  nodes: 1,
  deviceEdits: {},
  intraEdits: {},
  interEdits: {},
};

/** One stock device with no links; the same as {@link DEFAULT_HW} (kept as a name for the base of `setDevice`). */
export const SINGLE_DEVICE_HW: HardwareChoice = DEFAULT_HW;

/**
 * The default state: the default model (`DEFAULT_MODEL_ID`, bundled so the first paint has numbers without
 * any network), model-default formats, parallel 1/1/1/1, batch 1, prefill length 8192 and decode context
 * 8192 tokens (8k). The default hardware is set by `DEFAULT_HW`.
 */
export const DEFAULT_STATE: AppState = {
  model: { kind: 'hf', id: DEFAULT_MODEL_ID },
  quant: {},
  hw: DEFAULT_HW,
  parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
  workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
  options: {},
};

/** Which spec an edit targets: the device, the intra-node link or the inter-node link. */
export type EditTarget = 'device' | 'intra' | 'inter';

/** Every way the UI changes the state. */
export type AppAction =
  /** Replace the whole state (URL restore, popstate). */
  | { type: 'replace'; state: AppState }
  /** Pick a model; clears the quant overrides (they refer to the previous model's defaults). */
  | { type: 'setModel'; model: ModelChoice }
  /** Set (dtype) or clear (undefined) one quant override. */
  | { type: 'setQuant'; field: 'weights' | 'activations' | 'kvCache'; dtype: Dtype | 'exl3' | undefined }
  /** Set one EXL3 bits-per-weight override (`bpw`: linear layers, `headBpw`: LM head). */
  | { type: 'setBpw'; field: 'bpw' | 'headBpw'; value: number }
  /** Clear every quant override. */
  | { type: 'resetQuant' }
  /** Replace the whole hardware choice (a popular setup, see `popularSetupChoice`). */
  | { type: 'setHardware'; hw: HardwareChoice }
  /**
   * Pick a device; resets unlocks, engine, capacity, links, cabling and every edit. Counts are kept when the device
   * offers them, else they snap to the nearest offered count (`setupOptionsFor`).
   */
  | { type: 'setDevice'; deviceId: string }
  /** Choose (optionId) or clear (undefined: stock) one unlock axis; resets the capacity index. */
  | { type: 'setUnlock'; axisId: string; optionId: string | undefined }
  /** Choose an engine (undefined: the default). */
  | { type: 'setEngine'; engineId: string | undefined }
  /** Choose a capacity option index. */
  | { type: 'setCapacity'; index: number }
  /**
   * Change devices per node and/or nodes. A level's link and its edits are kept while that level still exists
   * (count > 1), else dropped; the cabling is kept while still allowed (`normalizeCabling`).
   */
  | { type: 'setCounts'; devicesPerNode?: number; nodes?: number }
  /**
   * Pick a link preset for one level (undefined: the device default); clears that level's edits. The cabling is
   * kept while the new link allows it.
   */
  | { type: 'setLink'; level: 'intra' | 'inter'; linkId: string | undefined }
  /** Choose the cabling between machines (undefined: the default); a default value is stored as absent. */
  | { type: 'setCabling'; cabling: CablingId | undefined }
  /** Set (value, SI) or clear (undefined) one spec edit. */
  | { type: 'setEdit'; target: EditTarget; path: string; value: number | undefined }
  /** Clear the edits of one target, or of all three when `target` is absent. */
  | { type: 'resetEdits'; target?: EditTarget }
  /** Change some parallel degrees. */
  | { type: 'setParallel'; parallel: Partial<ParallelConfig> }
  /** Change some workload fields. */
  | { type: 'setWorkload'; workload: Partial<Workload> }
  /** Change assumption switches; a value equal to the default (or undefined) is removed. */
  | { type: 'setOptions'; options: Partial<SimOptions> };

/** The `HardwareChoice` field holding the edits of `target`. */
export function editsKey(target: EditTarget): 'deviceEdits' | 'intraEdits' | 'interEdits' {
  return target === 'device' ? 'deviceEdits' : target === 'intra' ? 'intraEdits' : 'interEdits';
}

/** `record` without `key` (a new object; the input is not mutated). */
function without<T extends object, K extends keyof T>(record: T, key: K): T {
  const copy = { ...record };
  delete copy[key];
  return copy;
}

/** Options with default values (and undefined) removed, so equal choices encode equally. */
export function normalizeOptions(options: Partial<SimOptions>): Partial<SimOptions> {
  const out: Partial<SimOptions> = {};
  for (const key of Object.keys(options) as (keyof SimOptions)[]) {
    const value = options[key];
    if (value !== undefined && value !== DEFAULT_SIM_OPTIONS[key]) Object.assign(out, { [key]: value });
  }
  return out;
}

/** Pure reducer: the next state for `action`. Never mutates `state`. */
export function reduce(state: AppState, action: AppAction): AppState {
  const hw = state.hw;
  switch (action.type) {
    case 'replace':
      return action.state;
    case 'setModel': {
      const model: ModelChoice =
        action.model.kind === 'hf' && (action.model.revision === undefined || action.model.revision === 'main')
          ? { kind: 'hf', id: action.model.id }
          : action.model;
      return { ...state, model, quant: {} };
    }
    case 'setQuant': {
      let quant: QuantOverrides =
        action.dtype === undefined ? without(state.quant, action.field) : { ...state.quant, [action.field]: action.dtype };
      // The bpw overrides belong to EXL3 weights; leaving EXL3 drops them.
      if (action.field === 'weights' && action.dtype !== 'exl3') quant = without(without(quant, 'bpw'), 'headBpw');
      return { ...state, quant };
    }
    case 'setBpw':
      return { ...state, quant: { ...state.quant, weights: 'exl3', [action.field]: action.value } };
    case 'resetQuant':
      return { ...state, quant: {} };
    case 'setHardware':
      return { ...state, hw: action.hw };
    case 'setDevice': {
      const options = setupOptionsOf({ deviceId: action.deviceId });
      const snap = (offered: number[], n: number) => (offered.includes(n) ? n : nearestCount(offered, n));
      return {
        ...state,
        hw: {
          ...SINGLE_DEVICE_HW,
          deviceId: action.deviceId,
          devicesPerNode: snap(options.devicesPerNodeCounts, hw.devicesPerNode),
          nodes: snap(options.nodeCounts, hw.nodes),
        },
      };
    }
    case 'setUnlock': {
      const unlocks =
        action.optionId === undefined ? without(hw.unlocks, action.axisId) : { ...hw.unlocks, [action.axisId]: action.optionId };
      return { ...state, hw: { ...hw, unlocks, capacityIndex: 0 } };
    }
    case 'setEngine': {
      const next = action.engineId === undefined ? without(hw, 'engineId') : { ...hw, engineId: action.engineId };
      return { ...state, hw: next };
    }
    case 'setCapacity':
      return { ...state, hw: { ...hw, capacityIndex: action.index } };
    case 'setCounts': {
      const devicesPerNode = action.devicesPerNode ?? hw.devicesPerNode;
      const nodes = action.nodes ?? hw.nodes;
      let next: HardwareChoice = { ...hw, devicesPerNode, nodes };
      if (devicesPerNode <= 1) next = { ...without(next, 'intraLinkId'), intraEdits: {} };
      if (nodes <= 1) next = { ...without(next, 'interLinkId'), interEdits: {} };
      return { ...state, hw: normalizeCabling(normalizeLocalLink(next)) };
    }
    case 'setLink': {
      const idKey = action.level === 'intra' ? 'intraLinkId' : 'interLinkId';
      const base = action.linkId === undefined ? without(hw, idKey) : { ...hw, [idKey]: action.linkId };
      // A local link restricted to other counts (the 2-card NVLink bridge) moves the count to one it joins.
      const devicesPerNode =
        action.level === 'intra' && action.linkId !== undefined ? devicesPerNodeForLocalLink(hw, action.linkId) : hw.devicesPerNode;
      return { ...state, hw: normalizeCabling({ ...base, devicesPerNode, [editsKey(action.level)]: {} }) };
    }
    case 'setCabling': {
      const base = action.cabling === undefined ? without(hw, 'cabling') : { ...hw, cabling: action.cabling };
      return { ...state, hw: normalizeCabling(base) };
    }
    case 'setEdit': {
      const key = editsKey(action.target);
      const edits = action.value === undefined ? without(hw[key], action.path) : { ...hw[key], [action.path]: action.value };
      return { ...state, hw: { ...hw, [key]: edits } };
    }
    case 'resetEdits':
      return action.target === undefined
        ? { ...state, hw: { ...hw, deviceEdits: {}, intraEdits: {}, interEdits: {} } }
        : { ...state, hw: { ...hw, [editsKey(action.target)]: {} } };
    case 'setParallel':
      // DP and EP are fixed at 1 (not exposed in the UI).
      return { ...state, parallel: { ...state.parallel, ...action.parallel, dp: 1, ep: 1 } };
    case 'setWorkload':
      return { ...state, workload: { ...state.workload, ...action.workload } };
    case 'setOptions':
      return { ...state, options: normalizeOptions({ ...state.options, ...action.options }) };
    default: {
      const _never: never = action;
      throw new Error(`unknown action ${JSON.stringify(_never)}`);
    }
  }
}
