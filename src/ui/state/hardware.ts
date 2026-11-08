/**
 * Hardware resolution for the UI: the state's `HardwareChoice` → the effective device and links → the
 * `HardwareStack` the engine simulates. Also the default links per device, the topology quick picks, and
 * the editable spec fields with their display units.
 *
 * Order of application (docs/simulation-model.md §4 "Device variants and modifiers", "Unlock axes"):
 * preset → unlock modifiers → user edits. Edits are absolute values at `numericLeafPaths` paths applied
 * LAST, so the user's number wins over an unlock. An edited spec gets id `<id>-custom`, name
 * `<name> (edited)` and one `estimated` entry per edited path ("edited by the user"), so it still passes
 * `validateDevice` / `validateLink` and is honestly marked.
 */

import type { DeviceSpec, EstimatedField, HardwareStack, LinkSpec, TopologySpec } from '../../sim/hw/types';
import type { LinkModifier } from '../../sim/hw/modifiers';
import { applyDeviceModifiers, applyLinkModifiers, describeDeviceModifier, describeLinkModifier } from '../../sim/hw/modifiers';
import { resolveUnlocks } from '../../sim/hw/unlocks';
import { numericLeafPaths } from '../../sim/hw/validate';
import { DEVICE_PRESETS, SYSTEM_PRESETS, deviceUnlocks, type SystemPreset } from '../../sim/hw/presets/index';
import {
  ETHERNET_10G_TCP, IB_NDR_400, IB_XDR_800, LINK_PRESETS, NVLINK_4, NVLINK_5, PCIE_5_X16, PCIE_5_X16_DIRECT,
  SPARK_CONNECTX_7, TB5_RDMA, USB4_HALO_RDMA, XGMI_MI300X_8_GPU,
} from '../../sim/hw/presets/links';
import { TOPOLOGY_PRESETS, linkPresetId, nodeTopology } from '../../sim/hw/presets/topologies';
import type { HardwareChoice, SpecEdits } from './appState';

/** The reason recorded in `estimated[]` for an edited field. */
export const EDITED_REASON = 'edited by the user';

/** The device preset with this id, or undefined. */
export function findDevice(id: string): DeviceSpec | undefined {
  return DEVICE_PRESETS.find((d) => d.id === id);
}

/** The link preset with this `linkPresetId`, or undefined. */
export function findLink(id: string): LinkSpec | undefined {
  return LINK_PRESETS.find((l) => linkPresetId(l) === id);
}

/** Default links of a device: the ones the UI picks until the user chooses another preset. */
export interface DefaultLinks {
  /** Intra-node link (used when devices per node > 1); undefined for single-SoC machines. */
  intraNode?: LinkSpec;
  /** Inter-node link (used when nodes > 1). */
  interNode: LinkSpec;
}

/**
 * Hand-picked defaults where the named topologies do not say (or say several things): datacenter scale-up
 * fabrics, the SoC transports. Keyed by device id; `intraNode: null` means "no intra-node link" (one SoC per node).
 */
const DEFAULT_LINK_TABLE: Record<string, { intraNode: LinkSpec | null; interNode: LinkSpec }> = {
  'h100-sxm': { intraNode: NVLINK_4, interNode: IB_NDR_400 },
  'h200-sxm': { intraNode: NVLINK_4, interNode: IB_NDR_400 },
  b200: { intraNode: NVLINK_5, interNode: IB_NDR_400 },
  b300: { intraNode: NVLINK_5, interNode: IB_XDR_800 },
  'instinct-mi300x': { intraNode: XGMI_MI300X_8_GPU, interNode: IB_NDR_400 },
  'ryzen-ai-max-plus-395': { intraNode: null, interNode: USB4_HALO_RDMA },
  'dgx-spark': { intraNode: null, interNode: SPARK_CONNECTX_7 },
};

/**
 * Default links of `device`. In order: the hand table above (datacenter fabrics, Strix Halo USB4 RDMA, Spark
 * ConnectX-7); Apple → Thunderbolt 5 RDMA between nodes and no intra-node link; else the links of the first
 * named topology preset whose id contains the device id (GeForce PCIe generation, Arc Pro direct P2P); else
 * PCIe 5.0 x16 (host bounce for GeForce-class cards, direct P2P for datacenter accelerators) inside a node and
 * 10GbE between nodes.
 */
export function defaultLinks(device: DeviceSpec): DefaultLinks {
  const row = DEFAULT_LINK_TABLE[device.id];
  if (row !== undefined) return { ...(row.intraNode !== null ? { intraNode: row.intraNode } : {}), interNode: row.interNode };
  if (device.vendor === 'Apple') return { interNode: TB5_RDMA };
  const presets = TOPOLOGY_PRESETS.filter((t) => t.id.includes(device.id));
  const intraNode = presets.find((t) => t.intraNode !== undefined)?.intraNode;
  const interNode = presets.find((t) => t.interNode !== undefined)?.interNode;
  const datacenter = device.memory.capacityOptionsBytes[0] >= 64e9 && device.memory.kind === 'dedicated';
  return {
    intraNode: intraNode ?? (datacenter ? PCIE_5_X16_DIRECT : PCIE_5_X16),
    interNode: interNode ?? (datacenter ? IB_NDR_400 : ETHERNET_10G_TCP),
  };
}

/** Named topology presets for a device (quick-pick chips): every preset whose id contains the device id. */
export function topologyPresetsFor(device: Pick<DeviceSpec, 'id'>): TopologySpec[] {
  const hgx = device.id.match(/^(h100|b200|b300)/)?.[1];
  return TOPOLOGY_PRESETS.filter(
    (t) => t.id.includes(device.id) || (hgx !== undefined && t.id.startsWith(`hgx-${hgx}-`)) || (device.id === 'instinct-mi300x' && t.id === 'mi300x-xgmi-8'),
  );
}

/**
 * The hardware choice of a multi-device setup: its device (stock, no unlocks, first engine), the index of its memory
 * option (0 when it names none), its device and node counts, and its links as explicit preset ids (only at the
 * levels that exist). No edits.
 */
export function systemChoice(p: SystemPreset): HardwareChoice {
  const capacityIndex = p.memCapacityBytes === undefined ? 0 : p.device.memory.capacityOptionsBytes.indexOf(p.memCapacityBytes);
  if (capacityIndex < 0) throw new Error(`setup '${p.id}': ${p.memCapacityBytes} B is not a memory option of ${p.device.id}`);
  const { devicesPerNode, nodes, intraNode, interNode } = p.topology;
  return {
    deviceId: p.device.id,
    unlocks: {},
    capacityIndex,
    devicesPerNode,
    nodes,
    ...(devicesPerNode > 1 && intraNode !== undefined ? { intraLinkId: linkPresetId(intraNode) } : {}),
    ...(nodes > 1 && interNode !== undefined ? { interLinkId: linkPresetId(interNode) } : {}),
    deviceEdits: {},
    intraEdits: {},
    interEdits: {},
  };
}

/**
 * The multi-device setup a hardware choice is, or undefined: same device, memory option, counts and effective
 * links (an absent link id means the device default, `defaultLinks`), with no unlocks, engine choice or edits.
 * So picking a setup and then changing anything (a count, a link, a spec number) shows the plain device again.
 */
export function matchingSystem(hw: HardwareChoice): SystemPreset | undefined {
  const stock =
    Object.keys(hw.unlocks).length === 0 &&
    hw.engineId === undefined &&
    Object.keys(hw.deviceEdits).length === 0 &&
    Object.keys(hw.intraEdits).length === 0 &&
    Object.keys(hw.interEdits).length === 0;
  if (!stock) return undefined;
  const device = findDevice(hw.deviceId);
  if (device === undefined) return undefined;
  const defaults = defaultLinks(device);
  const intraId = hw.devicesPerNode > 1 ? (hw.intraLinkId ?? (defaults.intraNode && linkPresetId(defaults.intraNode))) : undefined;
  const interId = hw.nodes > 1 ? (hw.interLinkId ?? linkPresetId(defaults.interNode)) : undefined;
  return SYSTEM_PRESETS.find((p) => {
    const c = systemChoice(p);
    return (
      c.deviceId === hw.deviceId &&
      c.capacityIndex === hw.capacityIndex &&
      c.devicesPerNode === hw.devicesPerNode &&
      c.nodes === hw.nodes &&
      c.intraLinkId === intraId &&
      c.interLinkId === interId
    );
  });
}

/** A deep copy of a plain-data spec (presets are JSON-safe: numbers, strings, booleans, arrays, objects). */
function clonePlain<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** Sets `value` at dotted `path` in `obj` (mutates the copy it is given). */
function setAtPath(obj: unknown, path: string, value: number): void {
  const keys = path.split('.');
  let node = obj as Record<string, unknown>;
  for (const key of keys.slice(0, -1)) node = node[key] as Record<string, unknown>;
  node[keys[keys.length - 1]] = value;
}

/** Result of {@link applyEdits}. */
export interface EditedSpec<T> {
  /** The spec with the edits applied (the input itself when there are none). */
  spec: T;
  /** Edited paths that are not numeric leaves of the spec (ignored). */
  unknownPaths: string[];
}

/**
 * Applies absolute-value edits at numeric-leaf paths to a copy of `spec`, marking each edited path in
 * `estimated` with {@link EDITED_REASON}. Paths that are not numeric leaves are ignored and reported. The
 * caller renames the spec (devices: `<id>-custom`, `<name> (edited)`).
 */
export function applyEdits<T extends { estimated: EstimatedField[] }>(spec: T, edits: SpecEdits): EditedSpec<T> {
  const leaves = new Set(numericLeafPaths(spec).map((l) => l.path));
  const paths = Object.keys(edits).sort();
  const known = paths.filter((p) => leaves.has(p));
  const unknownPaths = paths.filter((p) => !leaves.has(p));
  if (known.length === 0) return { spec, unknownPaths };
  const copy = clonePlain(spec);
  for (const path of known) setAtPath(copy, path, edits[path]);
  copy.estimated = [...copy.estimated, ...known.map((field) => ({ field, reason: EDITED_REASON }))];
  return { spec: copy, unknownPaths };
}

/** The device a hardware choice resolves to. */
export interface ResolvedDevice {
  /** The preset, unmodified. */
  base: DeviceSpec;
  /** Base + unlock modifiers (before edits): what the spec editor shows as "stock for this selection". */
  unlocked: DeviceSpec;
  /** Base + unlocks + edits: the device actually simulated. */
  device: DeviceSpec;
  /** Intra-node link modifiers of the chosen unlocks (e.g. the P2P driver). */
  intraNodeLinkModifiers: LinkModifier[];
  /** True when at least one device edit applied. */
  edited: boolean;
  /** One "variant: …" line per unlock modifier. */
  notes: string[];
}

/**
 * Resolves the device of a hardware choice: preset → unlock modifiers (`resolveUnlocks`,
 * `applyDeviceModifiers`) → edits. Throws on an unknown device, axis or option id (the URL decoder and the
 * reducer only produce known ids, so this is a UI bug).
 */
export function resolveDevice(hw: Pick<HardwareChoice, 'deviceId' | 'unlocks' | 'deviceEdits'>): ResolvedDevice {
  const base = findDevice(hw.deviceId);
  if (base === undefined) throw new Error(`unknown device '${hw.deviceId}'`);
  const unlocks = resolveUnlocks(deviceUnlocks(base), hw.unlocks);
  const unlocked = unlocks.deviceModifiers.length > 0 ? applyDeviceModifiers(base, unlocks.deviceModifiers) : base;
  const { spec } = applyEdits(unlocked, hw.deviceEdits);
  const edited = spec !== unlocked;
  const device = edited ? { ...spec, id: `${base.id}-custom`, name: `${base.name} (edited)` } : spec;
  return {
    base,
    unlocked,
    device,
    intraNodeLinkModifiers: unlocks.intraNodeLinkModifiers,
    edited,
    notes: unlocks.deviceModifiers.map(describeDeviceModifier),
  };
}

/** One resolved link level. */
export interface ResolvedLink {
  /** The chosen (or default) preset, unmodified. */
  preset: LinkSpec;
  /** Preset + unlock link modifiers (intra-node only), before edits. */
  unlocked: LinkSpec;
  /** Preset + modifiers + edits: the link actually simulated. */
  link: LinkSpec;
  /** True when at least one edit applied. */
  edited: boolean;
  /** One "link variant: …" line per unlock modifier applied. */
  notes: string[];
}

/** Resolves one link: preset → modifiers → edits (renamed `<name> (edited)` when edited). */
function resolveLink(preset: LinkSpec, mods: LinkModifier[], edits: SpecEdits): ResolvedLink {
  const unlocked = mods.length > 0 ? applyLinkModifiers(preset, mods) : preset;
  const { spec } = applyEdits(unlocked, edits);
  const edited = spec !== unlocked;
  return {
    preset,
    unlocked,
    link: edited ? { ...spec, name: `${preset.name} (edited)` } : spec,
    edited,
    notes: mods.map((m) => `intraNode ${describeLinkModifier(m)}`),
  };
}

/** The links of a hardware choice, present only at the levels that exist (devicesPerNode > 1, nodes > 1). */
export interface ResolvedLinks {
  /** Intra-node link, when devicesPerNode > 1 and the device has one. */
  intra?: ResolvedLink;
  /** Inter-node link, when nodes > 1. */
  inter?: ResolvedLink;
}

/**
 * Resolves the links of a choice: the chosen preset (`intraLinkId` / `interLinkId`) or the device default,
 * the unlocks' intra-node link modifiers (only on a host-bounce link: they enable direct P2P), then the edits.
 * Unknown link ids fall back to the default.
 */
export function resolveLinks(hw: HardwareChoice, resolved: ResolvedDevice): ResolvedLinks {
  const defaults = defaultLinks(resolved.base);
  const out: ResolvedLinks = {};
  if (hw.devicesPerNode > 1) {
    const preset = (hw.intraLinkId !== undefined ? findLink(hw.intraLinkId) : undefined) ?? defaults.intraNode;
    if (preset !== undefined) {
      const mods = preset.p2p === 'host_bounce' ? resolved.intraNodeLinkModifiers : [];
      out.intra = resolveLink(preset, mods, hw.intraEdits);
    }
  }
  if (hw.nodes > 1) {
    const preset = (hw.interLinkId !== undefined ? findLink(hw.interLinkId) : undefined) ?? defaults.interNode;
    out.inter = resolveLink(preset, [], hw.interEdits);
  }
  return out;
}

/** Everything the UI derives from a hardware choice. */
export interface ResolvedHardware {
  /** The resolved device. */
  device: ResolvedDevice;
  /** The resolved links. */
  links: ResolvedLinks;
  /** The topology built from the counts and the resolved links (`nodeTopology`). */
  topology: TopologySpec;
  /** The stack handed to the engine: already-resolved specs, so no modifiers. */
  stack: HardwareStack;
  /** Capacity used, per device: `capacityOptionsBytes[capacityIndex]`, or the default when out of range. Unit: bytes. */
  memCapacityBytes: number;
  /** True when any device or link edit applied (the UI says "edited" wherever the rig is named). */
  edited: boolean;
  /** Modifier description lines (the engine sees no modifiers, so the UI shows these as notes). */
  notes: string[];
}

/**
 * Resolves a hardware choice to the engine's `HardwareStack`. The device and links are passed already
 * resolved (unlocks and edits applied), with `deviceModifiers` and `linkModifiers` empty, so the engine
 * validates exactly the numbers the user sees. Throws on unknown device / unlock ids (see {@link resolveDevice}).
 */
export function resolveHardware(hw: HardwareChoice): ResolvedHardware {
  const device = resolveDevice(hw);
  const links = resolveLinks(hw, device);
  const topology = nodeTopology({
    devicesPerNode: hw.devicesPerNode,
    nodes: hw.nodes,
    device: device.device,
    ...(links.intra !== undefined ? { intraNode: links.intra.link } : {}),
    ...(links.inter !== undefined ? { interNode: links.inter.link } : {}),
  });
  const capacities = device.device.memory.capacityOptionsBytes;
  const memCapacityBytes = capacities[hw.capacityIndex] ?? capacities[0];
  const stack: HardwareStack = {
    device: device.device,
    topology,
    memCapacityBytes,
    ...(hw.engineId !== undefined ? { engineId: hw.engineId } : {}),
  };
  return {
    device,
    links,
    topology,
    stack,
    memCapacityBytes,
    edited: device.edited || (links.intra?.edited ?? false) || (links.inter?.edited ?? false),
    notes: [...device.notes, ...(links.intra?.notes ?? [])],
  };
}

// ---------------------------------------------------------------------------------------------------------
// Editable fields and display units

/** A display unit for a spec number: `display = si / scale`. */
export interface DisplayUnit {
  /** Unit label shown next to the field, e.g. `'TFLOP/s'`. */
  label: string;
  /** SI value per display unit (e.g. 1e12 for TFLOP/s, 1e-6 for µs). */
  scale: number;
}

/** Display units used by the spec editor. */
export const UNITS = {
  tflops: { label: 'TFLOP/s', scale: 1e12 },
  tops: { label: 'TOPS', scale: 1e12 },
  gbps: { label: 'GB/s', scale: 1e9 },
  gb: { label: 'GB', scale: 1e9 },
  mb: { label: 'MB', scale: 1e6 },
  us: { label: 'µs', scale: 1e-6 },
  count: { label: '', scale: 1 },
  elements: { label: 'elements', scale: 1 },
} as const satisfies Record<string, DisplayUnit>;

/** SI value → display value. */
export function toDisplay(si: number, unit: DisplayUnit): number {
  return si / unit.scale;
}

/** Display value → SI value. */
export function fromDisplay(display: number, unit: DisplayUnit): number {
  return display * unit.scale;
}

/** Groups of the spec editor; `tiling` is the advanced group (collapsed by default). */
export type FieldGroup = 'memory' | 'matmul' | 'vector' | 'tiling' | 'link';

/** One editable number of a spec. */
export interface EditableField {
  /** Dotted `numericLeafPaths` path, the key in `SpecEdits`. */
  path: string;
  /** Human label, e.g. `'bf16, fp32 accumulate, dense'`. */
  label: string;
  /** Editor group. */
  group: FieldGroup;
  /** Display unit. */
  unit: DisplayUnit;
  /** Current value of the spec given, SI (as the field defines it). */
  value: number;
}

/**
 * The editable numbers of a device: every `numericLeafPaths` leaf, grouped into memory, matmul peaks, vector
 * peaks and tiling (advanced), with human labels and display units. Engine-specific labels carry the engine
 * name when the device has more than one engine.
 */
export function editableFields(device: DeviceSpec): EditableField[] {
  const multiEngine = device.engines.length > 1;
  const fields: EditableField[] = [];
  for (const { path, value } of numericLeafPaths(device)) {
    const parts = path.split('.');
    const field = (label: string, group: FieldGroup, unit: DisplayUnit) => fields.push({ path, label, group, unit, value });
    if (path === 'memory.bandwidthBytesPerSec') field('Memory bandwidth', 'memory', UNITS.gbps);
    else if (parts[0] === 'memory' && parts[1] === 'capacityOptionsBytes') {
      field(device.memory.capacityOptionsBytes.length > 1 ? `Capacity option ${Number(parts[2]) + 1}` : 'Capacity', 'memory', UNITS.gb);
    } else if (path === 'l2Bytes') field('Last-level cache (not used by the model)', 'tiling', UNITS.mb);
    else if (parts[0] === 'engines') {
      const engine = device.engines[Number(parts[1])];
      const prefix = multiEngine ? `${engine.name}: ` : '';
      const rest = parts.slice(2);
      if (rest[0] === 'matmulPeaks') {
        const peak = engine.matmulPeaks[Number(rest[1])];
        const integer = peak.accumulate === 'int32';
        field(`${prefix}${peak.input}, ${peak.accumulate} accumulate, ${peak.sparsity === 'dense' ? 'dense' : '2:4 sparse'}`, 'matmul', integer ? UNITS.tops : UNITS.tflops);
      } else if (rest[0] === 'vectorFlopsPerSec') field(`${prefix}vector ${rest[1]}`, 'vector', UNITS.tflops);
      else if (rest[0] === 'memBandwidthCapBytesPerSec') field(`${prefix}engine memory-bandwidth cap`, 'memory', UNITS.gbps);
      else if (rest[0] === 'computeUnits') field(`${prefix}compute units`, 'tiling', UNITS.count);
      else field(`${prefix}${rest.join(' ')}`, 'tiling', UNITS.elements);
    } else field(path, 'tiling', UNITS.count);
  }
  return fields;
}

/** The editable numbers of a link: bandwidth (GB/s per direction), latency (µs), host staging. */
export function editableLinkFields(link: LinkSpec): EditableField[] {
  const labels: Record<string, [string, DisplayUnit]> = {
    bandwidthPerDirectionBytesPerSec: ['Bandwidth per direction, per device', UNITS.gbps],
    latencySec: ['Latency per message', UNITS.us],
    'hostBounce.hostMemBandwidthBytesPerSec': ['Host memory bandwidth for staging (shared by the node)', UNITS.gbps],
    'hostBounce.extraLatencySec': ['Extra latency of the host copy', UNITS.us],
  };
  return numericLeafPaths(link).map(({ path, value }) => {
    const [label, unit] = labels[path] ?? [path, UNITS.count];
    return { path, label, group: 'link' as const, unit, value };
  });
}
