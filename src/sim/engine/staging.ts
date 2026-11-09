/**
 * Inter-node host staging: a device with its own (dedicated) memory whose inter-node transport cannot reach that
 * memory directly (no GPUDirect RDMA on the NIC, the GPU, or because the transport is TCP) moves every message
 * through pinned host memory: the GPU writes the message into a host buffer over PCIe, the NIC (or the kernel's
 * socket stack) sends it from there, and the receiver's GPU reads it back from host memory over PCIe.
 * See docs/simulation-model.md §6 ("Inter-node host staging").
 *
 * Mechanism modelled (NCCL's network transport without GPUDirect, commit 3e46c089f4ae): the buffers are pinned host
 * memory mapped into the GPU (`src/transport/net.cc:936`, `:1121`, `ncclCudaHostCalloc`); the GPU kernel writes and
 * polls them directly, with no `cudaMemcpy` and no stream synchronization, and a message larger than one buffer step
 * pipelines GPU copy, wire and GPU copy step by step. The α-β cost this adds is applied in `effectiveLink`
 * (`collectives.ts`); this file decides WHEN it applies and gathers its parameters.
 *
 * Unified-memory devices never stage (the NIC reads the shared pool); single-node topologies have nothing to stage.
 */

import { hostPcieLink, HOST_STAGING_STEP_BYTES } from '../hw/presets/links';
import { DEFAULT_HOST_RAM } from '../hw/presets/storage';
import type { DeviceSpec, HardwareStack, LinkSpec, TopologySpec } from '../hw/types';
import type { ParallelConfig } from '../ir/model';

/**
 * The parameters of host staging on one inter-node collective path. Plain data; every number comes from a cited
 * preset (the device's PCIe host link, the node's host RAM, NCCL's buffer step), none is a free parameter.
 */
export interface HostStaging {
  /**
   * B_pcie: the device's PCIe host link, per direction (GPU → pinned host buffer on the send side, host buffer → GPU
   * on the receive side). Unit: B/s.
   */
  hostLinkBandwidthBytesPerSec: number;
  /** α_pcie: one PCIe crossing (write plus flag, or poll plus read), the host link's `latencySec`. Unit: seconds. */
  hostLinkLatencySec: number;
  /**
   * B_host: the node's host DRAM peak, read + write combined, shared by all staging streams of the node. Unit: B/s.
   */
  hostMemBandwidthBytesPerSec: number;
  /** c: one pipeline step of the staging buffer (`HOST_STAGING_STEP_BYTES`, NCCL 4 MiB / 8 = 512 KiB). Unit: bytes. */
  stepBytes: number;
  /**
   * n_stg: devices of one node that stage at the same time, sharing B_host: min(devicesPerNode, tp·dp). Count.
   */
  activeDevicesPerNode: number;
}

/**
 * True when inter-node messages of `device` on `topology` go through host memory: the device has dedicated memory,
 * the topology has more than one node and an inter-node link, and that link or the device lacks GPUDirect RDMA
 * (`gpuDirectRdma === false` on either; absent = capable, the convention before staging was modelled). TCP links
 * set `false`, so every dedicated GPU on them stages; an RDMA link skips staging only when both ends support it.
 *
 * Formula: staged ⇔ memory = dedicated ∧ nodes > 1 ∧ (link.gpuDirectRdma = false ∨ device.gpuDirectRdma = false)
 */
export function needsHostStaging(device: DeviceSpec, topology: TopologySpec): boolean {
  const link = topology.interNode;
  if (device.memory.kind !== 'dedicated' || topology.nodes <= 1 || link === undefined) return false;
  return link.gpuDirectRdma === false || device.gpuDirectRdma === false;
}

/** Result of {@link resolveHostStaging}: the staging parameters (absent = no staging) and the notes they produced. */
export interface ResolvedHostStaging {
  /** Parameters for the inter-node paths; undefined when {@link needsHostStaging} is false. */
  staging: HostStaging | undefined;
  /** Info lines: that messages are staged, and any default taken for the host link. */
  warnings: string[];
}

/**
 * The host staging of a run (docs §6 "Inter-node host staging"), or none. B_pcie and α_pcie come from
 * `hardware.host.link`, else from `hostPcieLink(device.hostInterface)` (the stack sends `host` only when something
 * is offloaded, so the device's own PCIe slot is the usual source), else from PCIe 5.0 x16 with a warning (an SXM
 * module on a TCP link, which no catalogued setup has). B_host = `hardware.host.ram` ?? `DEFAULT_HOST_RAM`.
 * n_stg = min(devicesPerNode, tp·dp). `device` and `topology` are the effective ones (after modifiers).
 */
export function resolveHostStaging(
  device: DeviceSpec,
  topology: TopologySpec,
  hardware: HardwareStack,
  parallel: ParallelConfig,
): ResolvedHostStaging {
  if (!needsHostStaging(device, topology) || topology.interNode === undefined) {
    return { staging: undefined, warnings: [] };
  }
  const warnings: string[] = [];
  let hostLink: LinkSpec;
  if (hardware.host?.link !== undefined) {
    hostLink = hardware.host.link;
  } else if (device.hostInterface !== undefined) {
    hostLink = hostPcieLink(device.hostInterface.generation, device.hostInterface.lanes);
  } else {
    hostLink = hostPcieLink(5, 16);
    warnings.push(
      `inter-node: no host link known for ${device.name}; staging assumes ${hostLink.name}`,
    );
  }
  const hostRam = hardware.host?.ram ?? DEFAULT_HOST_RAM;
  warnings.push(
    `inter-node: ${device.name} has no GPUDirect path over ${topology.interNode.name}; each message is staged through host memory`,
  );
  const staging: HostStaging = {
    hostLinkBandwidthBytesPerSec: hostLink.bandwidthPerDirectionBytesPerSec,
    hostLinkLatencySec: hostLink.latencySec,
    hostMemBandwidthBytesPerSec: hostRam.bandwidthBytesPerSec,
    stepBytes: HOST_STAGING_STEP_BYTES,
    // Formula: n_stg = min(devicesPerNode, tp·dp)
    activeDevicesPerNode: Math.min(topology.devicesPerNode, parallel.tp * parallel.dp),
  };
  return { staging, warnings };
}
