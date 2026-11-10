/**
 * Multi-NIC nodes: how a node's inter-node NICs (rails) are shared by its devices.
 * See docs/simulation-model.md §6 ("Multi-NIC nodes").
 *
 * Convention: on a switched inter-node network, `TopologySpec.interNode` describes ONE NIC, per direction, and
 * `TopologySpec.nicsPerNode` says how many the node has (absent = one per device, the convention of every topology
 * before multi-NIC nodes). The collective formulas take a per-device, per-direction bandwidth B, so placement hands
 * them {@link interNodeLinkPerDevice}: the `interNode` link with B replaced by the device's share of the node's NICs.
 *
 * Assumptions (documented approximations):
 * - all devices of a node communicate at once (every collective of the rank layout does) and share the NICs evenly
 *   (rail-affine: each device uses the NICs assigned to it, never another device's idle NIC, i.e. no PXN), so a lone
 *   PP sender on a node with shared NICs is pessimistic;
 * - the device's own PCIe attachment to its NIC is not a cap; only `allPortsCapPerDirectionBytesPerSec` is.
 */

import type { LinkSpec, TopologySpec } from './types';

/**
 * N_nic: inter-node NICs per node: `topo.nicsPerNode`, or `topo.devicesPerNode` when absent (one NIC per device).
 * Count.
 */
export function nicsPerNodeOf(topo: TopologySpec): number {
  return topo.nicsPerNode ?? topo.devicesPerNode;
}

/**
 * B_dev: the inter-node bandwidth one device gets, per direction, when every device of its node communicates at
 * once and the node's NICs are shared evenly: N_nic NICs of B_nic each, divided by the dpn devices, capped by the
 * link's all-ports cap C. With one NIC per device (`nicsPerNode` absent) this is B_nic itself. Throws if the
 * topology has no `interNode` link. Unit: B/s, per device, per direction.
 */
export function interNodeBandwidthPerDeviceBytesPerSec(topo: TopologySpec): number {
  const link = topo.interNode;
  if (link === undefined) throw new Error(`topology ${topo.id} has no interNode link`);
  // Formula: B_dev = min(N_nic · B_nic / dpn, C)      (C = allPortsCap ?? ∞)
  // See docs/simulation-model.md §6 ("Multi-NIC nodes")
  const nics = nicsPerNodeOf(topo);
  const nicBytesPerSec = link.bandwidthPerDirectionBytesPerSec;
  const capBytesPerSec = link.allPortsCapPerDirectionBytesPerSec ?? Infinity;
  const nodeBytesPerSec = nics * nicBytesPerSec;
  const shareBytesPerSec = nodeBytesPerSec / topo.devicesPerNode;
  return Math.min(shareBytesPerSec, capBytesPerSec);
}

/**
 * The inter-node link as one device sees it: `topo.interNode` with `bandwidthPerDirectionBytesPerSec` = B_dev
 * ({@link interNodeBandwidthPerDeviceBytesPerSec}); every other field (latency, transport, capabilities, sources)
 * unchanged. Returns `topo.interNode` itself (the same object, not a copy) when `nicsPerNode` is absent or equals
 * `devicesPerNode`, or when B_dev equals B_nic, so a topology without multi-NIC nodes costs exactly as before.
 * `undefined` when the topology has no `interNode` link.
 */
export function interNodeLinkPerDevice(topo: TopologySpec): LinkSpec | undefined {
  const link = topo.interNode;
  if (link === undefined) return undefined;
  if (topo.nicsPerNode === undefined || topo.nicsPerNode === topo.devicesPerNode) return link;
  const perDeviceBytesPerSec = interNodeBandwidthPerDeviceBytesPerSec(topo);
  if (perDeviceBytesPerSec === link.bandwidthPerDirectionBytesPerSec) return link;
  return { ...link, bandwidthPerDirectionBytesPerSec: perDeviceBytesPerSec };
}

/**
 * Errors of a NIC count `nics` (N_nic, count) on nodes of `devicesPerNode` (dpn) devices; empty = valid. N_nic must be
 * an integer ≥ 1 and rail-symmetric: it divides dpn (each NIC serves dpn/N_nic devices) or is a multiple of it (each
 * device has N_nic/dpn NICs), so every device gets the same share. The topology-level rules (needs `interNode`,
 * `nodes > 1`, a switched fabric) are `validateTopology`'s.
 */
export function nicCountErrors(nics: number, devicesPerNode: number): string[] {
  if (!Number.isInteger(nics) || nics < 1)
    return [`nicsPerNode must be an integer ≥ 1 (got ${nics})`];
  if (nics % devicesPerNode !== 0 && devicesPerNode % nics !== 0)
    return [
      `nicsPerNode (${nics}) must divide devicesPerNode (${devicesPerNode}) or be a multiple of it, so every device gets the same NIC share`,
    ];
  return [];
}
