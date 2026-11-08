/**
 * Multi-device setups: complete rigs offered as one pick in the UI's device picker (device + memory option +
 * topology). Plain data built from the device presets and the named topologies in `topologies.ts`; every link
 * number cites its source on the `LinkSpec`, and the counts are user configurations (no vendor claim).
 * See docs/simulation-model.md §4 ("Links and topologies").
 */
import type { DeviceSpec, TopologySpec } from '../types';
import { RADEON_AI_PRO_R9700, RYZEN_AI_MAX_PLUS_395 } from './amd';
import { M5_ULTRA_80C } from './apple';
import { RTX_3090, RTX_5090 } from './nvidia';
import { DGX_SPARK } from './nvidia-datacenter';
import {
  DGX_SPARK_4,
  DGX_SPARK_PAIR,
  M5_ULTRA_TB5_CLUSTERS,
  R9700_PCIE_4,
  RTX_3090_NVLINK_2,
  RTX_5090_PCIE_2,
  RTX_5090_PCIE_4,
  STRIX_HALO_USB4_RDMA_CLUSTERS,
} from './topologies';

/** A complete multi-device rig offered as one pick: the device, its memory option and the topology. */
export interface SystemPreset {
  /** Stable id (the UI's picker value), e.g. `'rtx-5090-x2-pcie5'`. */
  id: string;
  /** Display name, e.g. `'RTX 5090 ×2 (PCIe)'`. */
  name: string;
  /** The device every slot of the topology holds (stock preset, no unlocks). */
  device: DeviceSpec;
  /**
   * Memory capacity per device, one of `device.memory.capacityOptionsBytes`; absent = option 0 (the device
   * default). Unit: bytes.
   */
  memCapacityBytes?: number;
  /** Device and node counts and the links between them (a named preset from `topologies.ts`). */
  topology: TopologySpec;
}

/** 1 GB in SI bytes (the presets record vendor GB as 1e9 B). Unit: bytes. */
const GB = 1e9;

/**
 * The multi-device setups, in display order (the user's list): GeForce PCIe boxes, the RTX 3090 NVLink pair,
 * DGX Spark clusters, the M5 Ultra Thunderbolt 5 pair, the Strix Halo pair and the R9700 workstation.
 */
export const SYSTEM_PRESETS: SystemPreset[] = [
  /** Two RTX 5090s on PCIe 5.0 x16, host-staged (stock GeForce has no P2P). 2 × 32GB. */
  {
    id: 'rtx-5090-x2-pcie5',
    name: 'RTX 5090 ×2 (PCIe)',
    device: RTX_5090,
    topology: RTX_5090_PCIE_2,
  },
  /** Four RTX 5090s on PCIe 5.0 x16, host-staged; four x16 slots need a workstation platform. 4 × 32GB. */
  {
    id: 'rtx-5090-x4-pcie5',
    name: 'RTX 5090 ×4 (PCIe)',
    device: RTX_5090,
    topology: RTX_5090_PCIE_4,
  },
  /** Two RTX 3090s joined by the two-card NVLink bridge (56.25 GB/s per direction). 2 × 24GB. */
  {
    id: 'rtx-3090-x2-nvlink',
    name: 'RTX 3090 ×2 (NVLink bridge)',
    device: RTX_3090,
    topology: RTX_3090_NVLINK_2,
  },
  /** Two DGX Sparks, 128GB each, cabled back to back over ConnectX-7 200GbE RoCE. */
  {
    id: 'dgx-spark-x2',
    name: 'DGX Spark ×2 (ConnectX-7)',
    device: DGX_SPARK,
    memCapacityBytes: 128 * GB,
    topology: DGX_SPARK_PAIR,
  },
  /** Four DGX Sparks, 128GB each, through a non-oversubscribed 200GbE switch. */
  {
    id: 'dgx-spark-x4',
    name: 'DGX Spark ×4 (ConnectX-7 switch)',
    device: DGX_SPARK,
    memCapacityBytes: 128 * GB,
    topology: DGX_SPARK_4,
  },
  /** Two M5 Ultra (80-core GPU) Mac Studios with 256GB each, linked by Thunderbolt 5 RDMA. */
  {
    id: 'm5-ultra-80c-256gb-x2-tb5',
    name: 'M5 Ultra 256GB ×2 (Thunderbolt 5)',
    device: M5_ULTRA_80C,
    memCapacityBytes: 256 * GB,
    topology: M5_ULTRA_TB5_CLUSTERS[0],
  },
  /**
   * Two Strix Halo (Ryzen AI Max+ 395) machines at their largest option, 128GB each, over USB4 native RDMA:
   * the device's default inter-node link (`defaultLinks`), the fastest cited Halo-to-Halo path without add-in
   * NICs (20 Gb/s negotiated, 8 µs; needs the experimental driver, see `USB4_HALO_RDMA`).
   */
  {
    id: 'strix-halo-128gb-x2-usb4',
    name: 'Strix Halo 128GB ×2 (USB4)',
    device: RYZEN_AI_MAX_PLUS_395,
    memCapacityBytes: 128 * GB,
    topology: STRIX_HALO_USB4_RDMA_CLUSTERS[0],
  },
  /** Four Radeon AI PRO R9700s on PCIe 5.0 x16 each, host-staged, in one workstation. 4 × 32GB. */
  {
    id: 'r9700-x4-pcie5',
    name: 'R9700 ×4 (PCIe)',
    device: RADEON_AI_PRO_R9700,
    topology: R9700_PCIE_4,
  },
];
