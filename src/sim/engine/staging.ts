/**
 * Inter-node host staging, in two independent parts that ride the same α-β pipeline (`stagedLink`, collectives.ts):
 *
 * 1. PCIe staging: a device with its own (dedicated) memory whose inter-node transport cannot reach that memory
 *    directly (no GPUDirect RDMA on the NIC, the GPU, or because the transport is TCP) moves every message through
 *    pinned host memory: the GPU writes the message into a host buffer over PCIe, the NIC (or the kernel's socket
 *    stack) sends it from there, and the receiver's GPU reads it back from host memory over PCIe.
 *    See docs/simulation-model.md §6 ("Inter-node host staging").
 * 2. The TCP socket copy: on a `transport: 'tcp'` link, `send()` copies each message from the user buffer into kernel
 *    socket buffers and `recv()` copies it out again (a CPU memcpy in host DRAM at each end); RDMA is zero-copy.
 *    This applies to every device, unified memory included (Mac, Strix Halo, DGX Spark over TCP), and on a dedicated
 *    GPU it comes on top of the PCIe copy (a third, physically separate copy, not a double count).
 *    See docs/simulation-model.md §6 ("TCP socket copy").
 *
 * Mechanism of part 1 (NCCL's network transport without GPUDirect, commit 3e46c089f4ae): the buffers are pinned host
 * memory mapped into the GPU (`src/transport/net.cc:936`, `:1121`, `ncclCudaHostCalloc`); the GPU kernel writes and
 * polls them directly, with no `cudaMemcpy` and no stream synchronization, and a message larger than one buffer step
 * pipelines GPU copy, wire and GPU copy step by step. Part 2: NCCL's socket proxy then calls plain `send()` /
 * `recv()` on that host buffer (`src/os/linux.cc:351-355`, no `MSG_ZEROCOPY`). The α-β cost is applied in
 * `effectiveLink` (`collectives.ts`); this file decides WHEN each part applies and gathers its parameters.
 *
 * Unified-memory devices never PCIe-stage (the NIC reads the shared pool); over TCP they pay only the socket copy.
 * Single-node topologies have nothing to stage.
 */

import { hostPcieLink, HOST_STAGING_STEP_BYTES, SOCKET_COPY_STEP_BYTES } from '../hw/presets/links';
import { DEFAULT_HOST_RAM } from '../hw/presets/storage';
import type { DeviceSpec, HardwareStack, LinkSpec, TopologySpec } from '../hw/types';
import type { ParallelConfig } from '../ir/model';

/**
 * The parameters of host staging on one inter-node collective path: PCIe staging, the TCP socket copy, or both.
 * Plain data; every number comes from a cited preset (the device's PCIe host link, the node's host RAM or the unified
 * pool, NCCL's buffer step, Linux's TCP segment size), none is a free parameter.
 */
export interface HostStaging {
  /**
   * PCIe staging of a dedicated-memory device (GPU ↔ pinned host buffer, one copy at each end); absent on unified
   * memory and on GPUDirect pairs (no PCIe crossing).
   */
  pcie?: {
    /**
     * B_pcie: the device's PCIe host link, per direction (GPU → pinned host buffer on the send side, host buffer →
     * GPU on the receive side). Unit: B/s.
     */
    bandwidthBytesPerSec: number;
    /** α_pcie: one PCIe crossing (write plus flag, or poll plus read), the host link's `latencySec`. Unit: seconds. */
    latencySec: number;
  };
  /**
   * True on a `transport: 'tcp'` link: each end copies the message between the user buffer and kernel socket buffers
   * with a CPU memcpy in B_host (send: user → kernel, receive: kernel → user), one read plus one write per byte.
   * A flag, no unit.
   */
  socketCopy: boolean;
  /**
   * B_host: the DRAM the copies and the NIC use, read + write combined, shared by all staging streams of the node:
   * host RAM for a dedicated-memory device, the unified pool's peak for a unified-memory one. Unit: B/s.
   */
  hostMemBandwidthBytesPerSec: number;
  /**
   * c: one pipeline step. `HOST_STAGING_STEP_BYTES` (NCCL 4 MiB / 8 = 512 KiB) when `pcie` is set, else
   * `SOCKET_COPY_STEP_BYTES` (64 KiB, the kernel's TCP segment). Unit: bytes.
   */
  stepBytes: number;
  /**
   * n_stg: devices of one node that stage at the same time, sharing B_host: min(devicesPerNode, tp·dp). Count.
   */
  activeDevicesPerNode: number;
}

/**
 * True when inter-node messages of `device` on `topology` go through host memory over PCIe: the device has dedicated
 * memory, the topology has more than one node and an inter-node link, and that link or the device lacks GPUDirect
 * RDMA (`gpuDirectRdma === false` on either; absent = capable, the convention before staging was modelled). TCP
 * links set `false`, so every dedicated GPU on them stages; an RDMA link skips staging only when both ends support it.
 *
 * Formula: staged ⇔ memory = dedicated ∧ nodes > 1 ∧ (link.gpuDirectRdma = false ∨ device.gpuDirectRdma = false)
 */
export function needsHostStaging(device: DeviceSpec, topology: TopologySpec): boolean {
  const link = topology.interNode;
  if (device.memory.kind !== 'dedicated' || topology.nodes <= 1 || link === undefined) return false;
  return link.gpuDirectRdma === false || device.gpuDirectRdma === false;
}

/**
 * True when inter-node messages on `topology` pay the TCP socket copy (docs §6 "TCP socket copy"): more than one
 * node, and the inter-node link is `transport: 'tcp'`. Independent of the device (unified memory pays it too).
 *
 * Formula: socketCopy ⇔ nodes > 1 ∧ interNode.transport = 'tcp'
 */
export function needsSocketCopy(topology: TopologySpec): boolean {
  return topology.nodes > 1 && topology.interNode?.transport === 'tcp';
}

/** Result of {@link resolveHostStaging}: the staging parameters (absent = no staging) and the notes they produced. */
export interface ResolvedHostStaging {
  /** Parameters for the inter-node paths; undefined when neither PCIe staging nor the socket copy applies. */
  staging: HostStaging | undefined;
  /** Info lines: that messages are PCIe-staged, and any default taken for the host link (none for the copy alone). */
  warnings: string[];
}

/**
 * The host staging of a run (docs §6 "Inter-node host staging", "TCP socket copy"), or none.
 * - pcie ⇔ {@link needsHostStaging}. B_pcie and α_pcie come from `hardware.host.link`, else from
 *   `hostPcieLink(device.hostInterface)` (the stack sends `host` only when something is offloaded, so the device's own
 *   PCIe slot is the usual source), else from PCIe 5.0 x16 with a warning (an SXM module on a TCP link, which no
 *   catalogued setup has).
 * - socketCopy ⇔ {@link needsSocketCopy}.
 * - Neither ⇒ `staging: undefined`, exactly the paths of before.
 * B_host = the device's unified pool peak (`device.memory.bandwidthBytesPerSec`) on unified memory, else
 * `hardware.host.ram` ?? `DEFAULT_HOST_RAM`. c = `HOST_STAGING_STEP_BYTES` with pcie, else `SOCKET_COPY_STEP_BYTES`.
 * n_stg = min(devicesPerNode, tp·dp). `device` and `topology` are the effective ones (after modifiers).
 */
export function resolveHostStaging(
  device: DeviceSpec,
  topology: TopologySpec,
  hardware: HardwareStack,
  parallel: ParallelConfig,
): ResolvedHostStaging {
  const pcieStaged = needsHostStaging(device, topology);
  const socketCopy = needsSocketCopy(topology);
  if ((!pcieStaged && !socketCopy) || topology.interNode === undefined) {
    return { staging: undefined, warnings: [] };
  }
  const warnings: string[] = [];
  let pcie: HostStaging['pcie'];
  if (pcieStaged) {
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
    warnings.push(
      `inter-node: ${device.name} has no GPUDirect path over ${topology.interNode.name}; each message is staged through host memory`,
    );
    pcie = {
      bandwidthBytesPerSec: hostLink.bandwidthPerDirectionBytesPerSec,
      latencySec: hostLink.latencySec,
    };
  }
  // Formula: B_host = B_mem (unified pool) on unified memory, else the node's host RAM
  const hostMemBandwidthBytesPerSec =
    device.memory.kind === 'unified'
      ? device.memory.bandwidthBytesPerSec
      : (hardware.host?.ram ?? DEFAULT_HOST_RAM).bandwidthBytesPerSec;
  const staging: HostStaging = {
    ...(pcie === undefined ? {} : { pcie }),
    socketCopy,
    hostMemBandwidthBytesPerSec,
    // Formula: c = c_pcie (512 KiB) with PCIe staging, else c_sock (64 KiB)
    stepBytes: pcie === undefined ? SOCKET_COPY_STEP_BYTES : HOST_STAGING_STEP_BYTES,
    // Formula: n_stg = min(devicesPerNode, tp·dp)
    activeDevicesPerNode: Math.min(topology.devicesPerNode, parallel.tp * parallel.dp),
  };
  return { staging, warnings };
}
