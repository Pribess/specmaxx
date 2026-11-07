/**
 * Hardware preset registry: every device, link and topology preset of the library, the device groups the UI
 * shows, and the unlock axes per device. The numbers live in the vendor files; this file only collects them.
 * See docs/simulation-model.md §4 ("Preset library", "Unlock axes", "Links and topologies").
 *
 * Files:
 * - `nvidia.ts`: GeForce RTX 3090 / 4090 / 5090 and their community unlock axes.
 * - `nvidia-datacenter.ts`: DGX Spark (GB10), H100 SXM, H200 SXM, B200, B300.
 * - `apple.ts`: M4 Max, M3 Ultra, M5 Max, M5 Ultra (binned GPU-core SKUs as separate presets).
 * - `amd.ts`: Radeon AI PRO R9700, RX 9070 XT, Ryzen AI Max+ 395 (Strix Halo), Instinct MI300X / MI325X / MI355X / MI455X.
 * - `intel.ts`: Arc Pro B70, B65, B60, Arc B580.
 * - `links.ts`: interconnect `LinkSpec`s (PCIe, NVLink, InfiniBand, ConnectX-7, xGMI, Thunderbolt, USB4, Ethernet).
 * - `topologies.ts`: multi-device `TopologySpec` builders and named presets.
 */

import type { DeviceSpec, TopologySpec } from '../types';
import type { UnlockAxis } from '../unlocks';
import {
  INSTINCT_MI300X,
  INSTINCT_MI325X,
  INSTINCT_MI355X,
  INSTINCT_MI455X,
  RADEON_AI_PRO_R9700,
  RADEON_RX_9070_XT,
  RYZEN_AI_MAX_PLUS_395,
} from './amd';
import { APPLE_DEVICE_PRESETS } from './apple';
import { INTEL_DEVICE_PRESETS } from './intel';
import { NVIDIA_CONSUMER_DEVICE_PRESETS, RTX_3090, RTX_3090_UNLOCKS, RTX_4090, RTX_4090_UNLOCKS, RTX_5090, RTX_5090_UNLOCKS } from './nvidia';
import { B200, B300, DGX_SPARK, H100_SXM, H200_SXM } from './nvidia-datacenter';

export * from './nvidia';
export * from './nvidia-datacenter';
export * from './apple';
export * from './amd';
export * from './intel';
export * from './links';
export * from './topologies';

/** A named group of device presets, one section of the UI's device picker. */
export interface DevicePresetGroup {
  /** Display name of the group, e.g. `'NVIDIA GeForce'`. */
  name: string;
  /** The group's devices, in display order. */
  devices: DeviceSpec[];
}

/**
 * The device presets by group, in display order: consumer / prosumer hardware first (GeForce, Apple, AMD Radeon
 * and Strix Halo, Intel Arc, DGX Spark), datacenter accelerators last. Every vendor array's device appears in
 * exactly one group (enforced by presets.test.ts).
 */
export const DEVICE_PRESET_GROUPS: DevicePresetGroup[] = [
  { name: 'NVIDIA GeForce', devices: NVIDIA_CONSUMER_DEVICE_PRESETS },
  { name: 'Apple silicon', devices: APPLE_DEVICE_PRESETS },
  { name: 'AMD Radeon / Ryzen AI', devices: [RADEON_AI_PRO_R9700, RADEON_RX_9070_XT, RYZEN_AI_MAX_PLUS_395] },
  { name: 'Intel Arc', devices: INTEL_DEVICE_PRESETS },
  { name: 'NVIDIA DGX Spark', devices: [DGX_SPARK] },
  {
    name: 'Datacenter',
    devices: [H100_SXM, H200_SXM, B200, B300, INSTINCT_MI300X, INSTINCT_MI325X, INSTINCT_MI355X, INSTINCT_MI455X],
  },
];

/** Every device preset, in display order (the groups flattened). Each must pass `validateDevice` (presets.test.ts). */
export const DEVICE_PRESETS: DeviceSpec[] = DEVICE_PRESET_GROUPS.flatMap((g) => g.devices);

/**
 * Unlock axes per device, keyed by `DeviceSpec.id`. A device without an entry has no unlocks. Every selection
 * of every device's axes must validate (presets.test.ts). See docs/simulation-model.md §4 ("Unlock axes").
 */
export const DEVICE_UNLOCKS: Record<string, UnlockAxis[]> = {
  [RTX_3090.id]: RTX_3090_UNLOCKS,
  [RTX_4090.id]: RTX_4090_UNLOCKS,
  [RTX_5090.id]: RTX_5090_UNLOCKS,
};

/** The unlock axes of `device` (empty when it has none). */
export function deviceUnlocks(device: Pick<DeviceSpec, 'id'>): UnlockAxis[] {
  return DEVICE_UNLOCKS[device.id] ?? [];
}

/**
 * A topology of exactly one device: 1 device per node × 1 node, no links (any parallelism > 1 is a
 * validation error). Its numbers are configuration counts, not vendor numbers, so it cites none.
 */
export function singleDevice(device: DeviceSpec): TopologySpec {
  return { id: `single-${device.id}`, name: `1 × ${device.name}`, devicesPerNode: 1, nodes: 1, sources: [] };
}
