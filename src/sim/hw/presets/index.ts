/**
 * Hardware preset registry: every device, link and topology preset of the library, the device groups the UI
 * shows, and the unlock axes per device. The numbers live in the vendor files; this file only collects them.
 * See docs/simulation-model.md §4 ("Preset library", "Unlock axes", "Links and topologies").
 *
 * Files:
 * - `nvidia.ts`: GeForce RTX 3090 / 4090 / 5090 and their community unlock axes.
 * - `nvidia-cmp.ts`: the CMP 170HX (GA100 mining card) and its community unlock axes.
 * - `nvidia-datacenter.ts`: DGX Spark (GB10), H100 SXM, H200 SXM, B200, B300.
 * - `apple.ts`: M5 Ultra, M5 Max, M4 Max, M3 Ultra, M3 Max (binned GPU-core SKUs as separate presets).
 * - `amd.ts`: Radeon AI PRO R9700, RX 9070 XT, Ryzen AI Max+ 395 (Strix Halo), Instinct MI300X / MI325X / MI355X / MI455X.
 * - `intel.ts`: Arc Pro B70, B65, B60, Arc B580.
 * - `links.ts`: interconnect `LinkSpec`s (PCIe, NVLink, InfiniBand, ConnectX-7, xGMI, Thunderbolt, USB4, Ethernet),
 *   and the device-to-host PCIe link builder `hostPcieLink`.
 * - `storage.ts`: storage tiers for weight offloading: host RAM (`HOST_RAM_PRESETS`) and SSDs (`SSD_PRESETS`).
 * - `topologies.ts`: multi-device `TopologySpec` builders (with the inter-node fabric shape) and named presets.
 * - `catalog.ts`: the setup catalog (device × count × link × cabling, validated), `composeTopology`, `setupLabel`,
 *   and `POPULAR_SETUPS`, the curated multi-device setups for one pick.
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
import { CMP_170HX, CMP_170HX_UNLOCKS, NVIDIA_CMP_DEVICE_PRESETS } from './nvidia-cmp';
import {
  A100_SXM4_80GB,
  A100_SXM4_40GB,
  A100_PCIE_80GB,
  A100_PCIE_40GB,
  V100_SXM2,
  V100_PCIE,
  B200,
  B300,
  GB200,
  GB300,
  DGX_SPARK,
  H100_SXM,
  H200_SXM,
} from './nvidia-datacenter';
import {
  NVIDIA_PRO_DEVICE_PRESETS,
  RTX_PRO_6000_WS,
  RTX_PRO_6000_MAX_Q,
  RTX_PRO_6000_WS_UNLOCKS,
  RTX_PRO_6000_MAX_Q_UNLOCKS,
} from './nvidia-pro';

export * from './nvidia';
export * from './nvidia-cmp';
export * from './nvidia-pro';
export * from './nvidia-datacenter';
export * from './apple';
export * from './amd';
export * from './intel';
export * from './links';
export * from './storage';
export * from './rack-scale-sources';
export * from './topologies';
export * from './catalog';

/** A named group of device presets, one section of the UI's device picker. */
export interface DevicePresetGroup {
  /** Display name of the group, e.g. `'NVIDIA GeForce'`. */
  name: string;
  /** The group's devices, in display order. */
  devices: DeviceSpec[];
}

/**
 * The device presets by group, in display order: consumer / prosumer hardware first (GeForce, the CMP mining card, Apple, AMD Radeon
 * and Strix Halo, Intel Arc, DGX Spark), datacenter accelerators last. Every vendor array's device appears in
 * exactly one group (enforced by presets.test.ts).
 *
 * Order within a group: newest generation first, then by tier (Ultra before Max, more cores first; within a
 * vendor family the higher-end part first).
 */
export const DEVICE_PRESET_GROUPS: DevicePresetGroup[] = [
  { name: 'NVIDIA GeForce', devices: NVIDIA_CONSUMER_DEVICE_PRESETS },
  { name: 'NVIDIA CMP (mining)', devices: NVIDIA_CMP_DEVICE_PRESETS },
  { name: 'NVIDIA RTX PRO', devices: NVIDIA_PRO_DEVICE_PRESETS },
  { name: 'Apple silicon', devices: APPLE_DEVICE_PRESETS },
  { name: 'AMD Radeon / Ryzen AI', devices: [RADEON_AI_PRO_R9700, RADEON_RX_9070_XT, RYZEN_AI_MAX_PLUS_395] },
  { name: 'Intel Arc', devices: INTEL_DEVICE_PRESETS },
  { name: 'NVIDIA DGX Spark', devices: [DGX_SPARK] },
  {
    name: 'Datacenter',
    devices: [
      GB300,
      B300,
      GB200,
      B200,
      H200_SXM,
      H100_SXM,
      A100_SXM4_80GB,
      A100_SXM4_40GB,
      A100_PCIE_80GB,
      A100_PCIE_40GB,
      V100_SXM2,
      V100_PCIE,
      INSTINCT_MI455X,
      INSTINCT_MI355X,
      INSTINCT_MI325X,
      INSTINCT_MI300X,
    ],
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
  [RTX_PRO_6000_WS.id]: RTX_PRO_6000_WS_UNLOCKS,
  [RTX_PRO_6000_MAX_Q.id]: RTX_PRO_6000_MAX_Q_UNLOCKS,
  [CMP_170HX.id]: CMP_170HX_UNLOCKS,
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
