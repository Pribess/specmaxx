/**
 * Hardware spec schema: devices (compute engines + memory), interconnect links, topologies, and the stack a
 * simulation runs on. Plain data; presets live in `hw/presets/`, variants are expressed as modifiers
 * (`hw/modifiers.ts`). See docs/simulation-model.md §4.
 *
 * Conventions:
 * - All numbers are SI decimal (see `hw/units.ts`).
 * - Compute peaks and memory bandwidth are PER DEVICE.
 * - Link bandwidth is PER DEVICE, PER DIRECTION (unidirectional injection rate), never the bidirectional sum.
 * - Sparse peaks count dense-equivalent FLOP (the vendor convention: a 2:4 peak is ~2× the dense one).
 * - Every number in a preset is covered by a `sources[].fields` path or an `estimated[]` entry.
 */

import type { AccumulateType, MathFormat } from '../ir/dtype';
import type { DeviceModifier, LinkModifier } from './modifiers';

/** A citation for spec numbers. */
export interface SourceRef {
  /** Where the numbers come from (vendor datasheet, whitepaper, spec). */
  url: string;
  /** Human title of the source document. */
  title: string;
  /**
   * Dotted field paths this source covers, relative to the spec object it sits on (e.g.
   * `'memory.bandwidthBytesPerSec'`, `'engines.0.matmulPeaks'`). A path covers its whole subtree.
   */
  fields: string[];
  /** Optional note (page, table, how a number was derived from the quoted one). */
  note?: string;
  /**
   * What kind of evidence the source is. Absent means `'vendor'`.
   * - `'vendor'`: a datasheet, whitepaper, ISA manual or official spec page of the vendor.
   * - `'review'`: an independent technical review or database that measured or compiled the number
   *   (Chips and Cheese, TechPowerUp, ServeTheHome, ...).
   * - `'community'`: a community measurement or report (forum post, GitHub issue, benchmark repo, seller
   *   listing), i.e. NOT a datasheet number. The UI should flag numbers that rest on these.
   */
  kind?: 'vendor' | 'review' | 'community';
}

/** A spec number no vendor publishes, with the reason for the chosen value. */
export interface EstimatedField {
  /** Dotted field path (covers its subtree), as in {@link SourceRef.fields}. */
  field: string;
  /** Why this value was chosen and from what. */
  reason: string;
}

/**
 * One matmul peak of an engine, keyed by (input, accumulate, sparsity).
 * Per device. Unit: FLOP/s for float inputs, OP/s for integer inputs. Sparse (`'2:4'`) entries count
 * dense-equivalent FLOP/s (vendor convention).
 */
export interface MatmulPeak {
  /** Matmul input class. */
  input: MathFormat;
  /** Accumulator type. */
  accumulate: AccumulateType;
  /** `'dense'`, or `'2:4'` structured sparsity of one operand. */
  sparsity: 'dense' | '2:4';
  /** Peak rate, per device. Unit: FLOP/s (OP/s for integer inputs). */
  flopsPerSec: number;
}

/**
 * Tensor-core tiling parameters for the shape-efficiency model (docs §4, `hw/efficiency.ts`). Dimensions in
 * elements of the GEMM's m, n, k.
 */
export interface GemmTiling {
  /** M granularity of one MMA instruction (rows are padded up to it). Unit: elements. */
  mmaM: number;
  /** N granularity of one MMA instruction. Unit: elements. */
  mmaN: number;
  /** K granularity of one MMA instruction. Unit: elements. */
  mmaK: number;
  /** Candidate CTA (thread-block) output tiles; the model takes the fastest. Each m/n a multiple of mmaM/mmaN. Unit: elements. */
  ctaTiles: { m: number; n: number }[];
  /** Concurrent CTAs per compute unit (wave size = computeUnits · ctasPerUnit). Count. */
  ctasPerUnit: number;
}

/** One matrix engine of a device (GPU tensor cores, an NPU, a TPU MXU, CPU matrix units). A run uses exactly one. */
export interface ComputeEngine {
  /** Stable id, unique within the device (e.g. `'gpu'`, `'npu'`). */
  id: string;
  /** Display name. */
  name: string;
  /** Engine family (informational). */
  kind: 'gpu' | 'npu' | 'tpu' | 'cpu' | 'other';
  /** Compute units (SMs, CUs, cores): the wave granularity of the efficiency model. Count. */
  computeUnits: number;
  /** Matmul peaks; every float input listed must have an `fp32`-accumulate entry. Per device. */
  matmulPeaks: MatmulPeak[];
  /** Non-tensor (vector/SIMT) peaks by dtype, per device. Unit: FLOP/s. */
  vectorFlopsPerSec: Partial<Record<'fp64' | 'fp32' | 'fp16' | 'bf16', number>>;
  /** Tiling parameters: `default`, optionally overridden per input format (MMA k differs by format). */
  gemmTiling: { default: GemmTiling } & Partial<Record<MathFormat, GemmTiling>>;
  /**
   * If this engine cannot reach the device memory peak (e.g. an NPU on a narrower fabric), its own cap on
   * memory bandwidth, per device. Unit: B/s (read + write combined, like {@link MemorySpec.bandwidthBytesPerSec}).
   */
  memBandwidthCapBytesPerSec?: number;
}

/**
 * Device memory. `'dedicated'` = memory private to the accelerator (GDDR, HBM); `'unified'` = one pool shared
 * by CPU and accelerators (Apple M-series, DGX Spark, Strix Halo), so the OS takes part of it.
 */
export interface MemorySpec {
  /** Dedicated or unified. */
  kind: 'dedicated' | 'unified';
  /** Memory technology, e.g. `'GDDR7'`, `'HBM3'`, `'LPDDR5X'` (informational). */
  technology: string;
  /** Capacity of each SKU configuration, per device; `[0]` is the default. Non-empty. Unit: bytes. */
  capacityOptionsBytes: number[];
  /** Peak memory bandwidth per device, read + write combined, as vendors quote it. Unit: B/s. */
  bandwidthBytesPerSec: number;
}

/** One accelerator device (or one SoC). All numbers per device. */
export interface DeviceSpec {
  /** Stable id, e.g. `'rtx-5090'`. */
  id: string;
  /** Display name. */
  name: string;
  /** Vendor name, e.g. `'NVIDIA'`. */
  vendor: string;
  /** Matrix engines (≥ 1); `engines[0]` is the default. */
  engines: ComputeEngine[];
  /** Device memory. */
  memory: MemorySpec;
  /** Last-level cache size. Recorded for display; NOT used by the v1 model. Unit: bytes. */
  l2Bytes?: number;
  /** Citations covering the numeric fields. */
  sources: SourceRef[];
  /** Numbers no vendor publishes, with reasons. */
  estimated: EstimatedField[];
  /** Set by `applyDeviceModifiers`: the modifiers this spec was derived with (provenance). Absent on base presets. */
  appliedModifiers?: DeviceModifier[];
}

/**
 * One interconnect level between devices. Bandwidth is PER DEVICE, PER DIRECTION (the unidirectional rate one
 * device can inject), in B/s; latency is the per-message/per-step α of the alpha-beta model.
 */
export interface LinkSpec {
  /** Link family (informational). */
  kind:
    | 'nvlink'
    | 'nvlink_bridge'
    | 'xgmi'
    | 'pcie'
    | 'infiniband'
    | 'roce'
    | 'ethernet'
    | 'thunderbolt'
    | 'usb4'
    | 'ici'
    | 'other';
  /** Display name, e.g. `'PCIe 5.0 x16'`, `'ConnectX-7 200GbE'`. */
  name: string;
  /**
   * Per-device, per-direction bandwidth. On a multi-port inter-node fabric shape (`full_mesh`, `ring`, `chain` of
   * {@link FabricShape}) it is the bandwidth of ONE port (one cable), per direction. Unit: B/s.
   */
  bandwidthPerDirectionBytesPerSec: number;
  /** Per-message (per algorithm step) latency α. Unit: seconds. */
  latencySec: number;
  /** `'direct'` = device-to-device transfers; `'host_bounce'` = staged through host memory (e.g. GeForce without P2P). */
  p2p: 'direct' | 'host_bounce';
  /** Host staging parameters; present iff `p2p === 'host_bounce'`. */
  hostBounce?: {
    /** Host memory bandwidth available to staging, shared by all devices of the node. Unit: B/s. */
    hostMemBandwidthBytesPerSec: number;
    /** Extra latency per message added by the host copy. Unit: seconds. */
    extraLatencySec: number;
  };
  /**
   * Cap on the sum over all ports of this link that one device drives at once, per direction (egress and ingress
   * each), e.g. the NIC's host PCIe attachment (DGX Spark: two ConnectX-7 ports share ~25 GB/s). Used only by the
   * multi-port fabric shapes (`full_mesh`, `ring`, `chain` of {@link FabricShape}); bytes a node relays count
   * against it. Absent = the ports add up (e.g. Apple Thunderbolt 5: one controller per port). Must be ≥
   * `bandwidthPerDirectionBytesPerSec`. Unit: B/s.
   */
  allPortsCapPerDirectionBytesPerSec?: number;
  /**
   * Extra latency a node adds when it relays a message to the next hop on a switchless multi-hop path (`ring`,
   * `chain`), on top of that hop's `latencySec`. Absent = 0 (cut-through relay: the hardware floor). Unit: seconds.
   */
  forwardingLatencySec?: number;
  /** Citations covering the numeric fields. */
  sources: SourceRef[];
  /** Numbers no vendor publishes, with reasons. */
  estimated: EstimatedField[];
  /** Set by `applyLinkModifiers`: the modifiers this link was derived with (provenance). */
  appliedModifiers?: LinkModifier[];
}

/**
 * Shape of the network between nodes (docs/simulation-model.md §6 "Fabric shapes"; geometry in `hw/fabric.ts`).
 *
 * Notation: N = nodes in the group; B_L = `interNode.bandwidthPerDirectionBytesPerSec` = ONE port / one cable, per
 * direction; C = `interNode.allPortsCapPerDirectionBytesPerSec` ?? ∞ = the most one node can send (and, separately,
 * receive) over all its ports at once, relayed bytes included.
 *
 * Every shape except `'switched'` needs `devicesPerNode = 1` (`validateTopology`): one SoC or box per node, so a
 * node is a device.
 */
export type FabricShape =
  /**
   * Each node has ONE port into a switch. Ports per node: 1. Injection per node: B_L (C does not apply: one port).
   * Hops between any two nodes: 1 (the switch traversal is inside `latencySec`). An absent
   * `TopologySpec.interNodeFabric` means this, non-blocking: the convention of every topology before fabric shapes.
   */
  | {
      kind: 'switched';
      /**
       * Oversubscription of the switch tier, finite ≥ 1 (absent or 1 = non-blocking). Every inter-node flow gets
       * B_L / oversubscription (worst case: every flow crosses the oversubscribed tier). Dimensionless.
       */
      oversubscription?: number;
    }
  /**
   * Every pair of nodes cabled directly with `cablesPerPair` cables, striped as one link of k·B_L (e.g. JACCL over
   * Thunderbolt 5). Ports per node: (N−1)·k. Per pair k·B_L; per node min((N−1)·k·B_L, C). Hops: 1. N = 2 is
   * "direct cable(s)".
   */
  | {
      kind: 'full_mesh';
      /** Cables between each pair, integer ≥ 1 (absent = 1). Count. */
      cablesPerPair?: number;
    }
  /**
   * N ≥ 3 nodes, node i cabled to i ± 1 (mod N). Ports per node: 2. Per link B_L; per node min(2·B_L, C). Hops: up
   * to ⌊N/2⌋ (shortest direction). Each relaying node adds `forwardingLatencySec`, and the bytes it relays count
   * against its C (store-and-forward through the host: no switching inside these NICs). At N = 3 the cabling equals
   * a `full_mesh` of 3.
   */
  | { kind: 'ring' }
  /**
   * Daisy chain 0 – 1 – … – (N−1), the ends not joined. Ports per node: 2 inside, 1 at the ends. Per link B_L; per
   * node min(2·B_L, C) inside, B_L at the ends. Hops: up to N − 1; relays as for `'ring'`.
   */
  | { kind: 'chain' };

/**
 * How devices are connected. A "node" is one scale-up domain (an NVLink island, a PCIe box, one Mac); devices
 * in a node talk over `intraNode`, nodes over `interNode`.
 */
export interface TopologySpec {
  /** Stable id. */
  id: string;
  /** Display name. */
  name: string;
  /** Devices per node. Count. */
  devicesPerNode: number;
  /** Nodes. Count. */
  nodes: number;
  /** Link between devices of one node; required when a parallel group spans > 1 device of a node. */
  intraNode?: LinkSpec;
  /** Link between nodes; required when a parallel group spans > 1 node. */
  interNode?: LinkSpec;
  /**
   * Shape of the inter-node network; absent = `{ kind: 'switched' }`, non-blocking (the pre-fabric convention).
   * Allowed only when `interNode` is present and `nodes > 1`; shapes other than `'switched'` need
   * `devicesPerNode = 1`.
   */
  interNodeFabric?: FabricShape;
  /** Citations for the topology itself. */
  sources: SourceRef[];
}

/** The hardware a simulation runs on: a device type replicated over a topology, plus variant modifiers. */
export interface HardwareStack {
  /** Base device spec (before `deviceModifiers`). */
  device: DeviceSpec;
  /** Topology (devices per node, nodes, links). */
  topology: TopologySpec;
  /** Variant modifiers applied to `device` before simulation, in order. */
  deviceModifiers?: DeviceModifier[];
  /** Modifiers applied to the topology's links before simulation (e.g. enabling P2P over PCIe), in order. */
  linkModifiers?: { intraNode?: LinkModifier[]; interNode?: LinkModifier[] };
  /** Engine to run on; default: `engines[0].id` of the modified device. */
  engineId?: string;
  /** Memory capacity per device; must be one of the modified `memory.capacityOptionsBytes`; default `[0]`. Unit: bytes. */
  memCapacityBytes?: number;
}
