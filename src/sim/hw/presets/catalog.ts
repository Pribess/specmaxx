/**
 * The hardware setup catalog: what each device can be composed into (GPUs per machine, machines, the local and
 * network links it actually has, and the cabling its port count allows), the composition of a `TopologySpec` from
 * such a choice, its SKU-exact label, and a short list of popular setups. Replaces the flat enumeration of every
 * count × link combination the UI used to list. Plain data and small pure functions; the geometry rules come from
 * `hw/fabric.ts`. See docs/simulation-model.md §4 ("Setup catalog", "Popular setups").
 *
 * Conventions: counts are integers; `portsPerNode` is the number of USABLE ports of a link per machine (a port that
 * exists but cannot carry the link, like the M3 Ultra Studio's Thunderbolt port next to Ethernet for RDMA, is not
 * counted); every port count cites the machine's spec page (the device presets are per chip, so the machine that
 * supplies the ports is named in the note).
 */

import { fabricShapeErrors, portsNeeded } from '../fabric';
import type { DeviceSpec, FabricShape, HostInterfaceSpec, LinkSpec, SourceRef, TopologySpec } from '../types';
import {
  AMD_DEVICE_PRESETS, INSTINCT_MI300X, INSTINCT_MI325X, INSTINCT_MI355X, INSTINCT_MI455X, RADEON_AI_PRO_R9700,
  RADEON_RX_9070_XT, RYZEN_AI_MAX_PLUS_395,
} from './amd';
import {
  APPLE_DEVICE_PRESETS, M3_MAX_30C, M3_MAX_40C, M3_ULTRA_60C, M3_ULTRA_80C, M4_MAX_32C, M4_MAX_40C, M5_MAX_32C,
  M5_MAX_40C, M5_ULTRA_64C, M5_ULTRA_80C,
} from './apple';
import { ARC_B580, ARC_PRO_B60, ARC_PRO_B65, ARC_PRO_B70, INTEL_DEVICE_PRESETS } from './intel';
import {
  ETHERNET_10G_TCP, ETHERNET_5G_TCP, IB_NDR_400, IB_XDR_800, LEGACY_LINK_IDS, LINK_PRESETS, NVLINK_4, NVLINK_5,
  A100_NVLINK_BRIDGE,
  IB_HDR_200,
  NVLINK_3,
  NVLINK_2_HCM_8,
  NVLINK_2_MESH_4,
  NVLINK_2_PAIR,
  PCIE_3_X16_DIRECT,
  PCIE_3_X16,
  PCIE_1_X16, PCIE_1_X4, PCIE_2_X16, PCIE_2_X4,
  PCIE_4_X16, PCIE_4_X16_DIRECT, PCIE_4_X8, PCIE_4_X8_DIRECT, PCIE_5_X16, PCIE_5_X16_DIRECT, PCIE_5_X8,
  PCIE_5_X8_DIRECT, RTX_3090_NVLINK, SPARK_CONNECTX_7, TB4_TCP, TB4_TCP_20G, TB5_RDMA, TB5_TCP, USB4_HALO_RDMA, USB4_HALO_TCP,
  XGMI_MI300X_8_GPU, XGMI_MI300X_PER_PEER,
  NVLINK_5_NVL72,
  NVLINK_5_NVL4,
  XGMI_MI325X_8_GPU,
  XGMI_MI325X_PER_PEER,
  XGMI_MI355X_8_GPU,
  XGMI_MI355X_PER_PEER,
  UALOE_MI455X,
} from './links';
import { NVIDIA_CONSUMER_DEVICE_PRESETS, RTX_3090, RTX_4090, RTX_5090 } from './nvidia';
import { CMP_170HX, NVIDIA_CMP_DEVICE_PRESETS } from './nvidia-cmp';
import {
  A100_SXM4_80GB,
  A100_SXM4_40GB,
  A100_PCIE_80GB,
  A100_PCIE_40GB,
  V100_SXM2,
  V100_PCIE,
  GB200,
  GB300,
  B200, B300, DGX_SPARK, H100_SXM, H200_SXM, NVIDIA_DATACENTER_DEVICE_PRESETS } from './nvidia-datacenter';
import { GB200_NVL72_URL, GB300_NVL72_URL } from './rack-scale-sources';
import {
  NVIDIA_PRO_DEVICE_PRESETS,
  RTX_PRO_6000_WS,
  RTX_PRO_6000_MAX_Q,
  RTX_PRO_5000,
  RTX_PRO_4000_SFF,
} from './nvidia-pro';
import { linkPresetId, nodeTopology } from './topologies';

// ---------------------------------------------------------------------------------------------------------------
// Types

/** One network a machine of this device can join: the per-port link, its usable port count, and switch / striping support. */
export interface NetworkOption {
  /** The link of ONE port (one cable), per direction; its `linkPresetId` is the UI / URL value. */
  link: LinkSpec;
  /** Short name for setup labels and the hero line, e.g. `'TB5'`, `'ConnectX-7'`, `'USB4'`, `'10GbE'`. */
  shortName: string;
  /** USABLE ports of this link per machine (e.g. M3 Ultra Mac Studio, TB5 RDMA: 5 of 6). Count. */
  portsPerNode: number;
  /** Whether a switch exists for the link (Ethernet, RoCE, InfiniBand: yes; Thunderbolt, USB4: no). */
  switchable: boolean;
  /** Most parallel cables one pair of machines can stripe as one link (2 for TB5 RDMA: JACCL stripes; 1 for ConnectX-7: no gain; 1 for USB4). Count. */
  maxCablesPerPair: number;
  /**
   * Most machines a switchless shape (mesh, ring, chain) is offered for, when the vendor only supports direct cabling
   * up to a size (DGX Spark: 4, NVIDIA's playbook). Absent = any count the ports allow. Count.
   */
  maxSwitchlessNodes?: number;
  /** Cites `portsPerNode` (the machine's spec page) and the switch / striping facts. */
  sources: SourceRef[];
}

/** One link between the devices of one machine (intra-node), e.g. PCIe 5.0 x16, NVLink 5. */
export interface LocalLinkOption {
  /** The link. */
  link: LinkSpec;
  /** Short name, e.g. `'PCIe 5.0 x16'`, `'NVLink 5'`. */
  shortName: string;
  /**
   * PCIe links only: the generation and electrical lanes the link runs at. It is offered only while the card's
   * EFFECTIVE host interface (`DeviceSpec.hostInterface` after unlock modifiers) is at least as new and as wide
   * ({@link localLinkFits}). Absent on non-PCIe links. Integers, no unit.
   */
  pcie?: HostInterfaceSpec;
  /**
   * Devices per machine this link supports; absent = every count the device offers. The RTX 3090 NVLink bridge
   * joins exactly two cards (`[2]`): no four-card NVLink island exists for it. Counts.
   */
  devicesPerNodeCounts?: number[];
}

/** Everything a device can be composed into. */
export interface DeviceSetupOptions {
  /** `DeviceSpec.id`. */
  deviceId: string;
  /** The exact SKU, short: `'M5 Ultra 80-core GPU'`, `'RTX 5090'`, `'DGX Spark'`, `'Strix Halo 395'`. */
  shortName: string;
  /** Devices per machine offered: `[1]` for SoCs, `[1, 2, 4, 8]` for GPUs. Counts. */
  devicesPerNodeCounts: number[];
  /** Machine counts offered, each with ≥ 1 allowed cabling on the default network, e.g. `[1, 2, 3, 4, 8]`. Counts. */
  nodeCounts: number[];
  /** Intra-node links the device has; `[0]` is the default. Empty for SoCs (one device per machine). */
  localLinks: LocalLinkOption[];
  /** Networks the machine has; `[0]` is the default. Non-empty. */
  networks: NetworkOption[];
}

/** UI / URL token of a cabling; maps 1:1 to a {@link FabricShape} (`mesh2` = full mesh with 2 cables per pair). */
export type CablingId = 'switched' | 'mesh' | 'mesh2' | 'ring' | 'chain';

/** Every cabling, in display order. */
export const CABLING_IDS: readonly CablingId[] = ['switched', 'mesh', 'mesh2', 'ring', 'chain'];

/** A composed hardware choice: the components the UI and its URL store (no preset id). */
export interface SetupChoice {
  /** `DeviceSpec.id`. */
  deviceId: string;
  /** Memory capacity per device, one of the device's `capacityOptionsBytes`; absent = option 0. Unit: bytes. */
  memCapacityBytes?: number;
  /** Devices per machine, integer ≥ 1. Count. */
  devicesPerNode: number;
  /** Machines, integer ≥ 1. Count. */
  nodes: number;
  /** `linkPresetId` of the intra-node link; absent = the device's default local link. Used when devicesPerNode > 1. */
  localLinkId?: string;
  /** `linkPresetId` of the inter-node link; absent = the device's default network. Used when nodes > 1. */
  networkLinkId?: string;
  /** Cabling between machines; absent = `defaultCabling` of the network. Used when nodes > 1. */
  cabling?: CablingId;
  /**
   * The device's EFFECTIVE PCIe host interface (after unlock modifiers, e.g. the CMP 170HX's Gen2 patch and capacitor
   * mod); absent = the preset's stock `hostInterface`. Bounds the PCIe local links offered and accepted
   * ({@link localLinkFits}).
   */
  hostInterface?: HostInterfaceSpec;
}

/** A curated popular setup, picked in one go. */
export interface PopularSetup {
  /** Stable id (a picker value), e.g. `'dgx-spark-x4-ring'`. */
  id: string;
  /** Display name = `setupLabel(choice)`, e.g. `'DGX Spark 128GB ×4 · ConnectX-7 ring'`. */
  name: string;
  /** The components it fills in. */
  choice: SetupChoice;
  /** Who runs it (vendor playbook, community report). */
  sources: SourceRef[];
  /** One line of context (what the setup is known for). */
  note?: string;
}

// ---------------------------------------------------------------------------------------------------------------
// Cabling ↔ fabric shape

/** The fabric shape of a cabling. */
export function fabricOf(c: CablingId): FabricShape {
  switch (c) {
    case 'switched':
      return { kind: 'switched' };
    case 'mesh':
      return { kind: 'full_mesh' };
    case 'mesh2':
      return { kind: 'full_mesh', cablesPerPair: 2 };
    case 'ring':
      return { kind: 'ring' };
    case 'chain':
      return { kind: 'chain' };
  }
}

/**
 * The cabling of a fabric shape (absent = `'switched'`); undefined for a shape the UI has no token for (an
 * oversubscribed switch, a mesh with 3+ cables per pair).
 */
export function cablingOf(fabric: FabricShape | undefined): CablingId | undefined {
  if (fabric === undefined) return 'switched';
  switch (fabric.kind) {
    case 'switched':
      return fabric.oversubscription === undefined || fabric.oversubscription === 1 ? 'switched' : undefined;
    case 'full_mesh': {
      const k = fabric.cablesPerPair ?? 1;
      return k === 1 ? 'mesh' : k === 2 ? 'mesh2' : undefined;
    }
    case 'ring':
      return 'ring';
    case 'chain':
      return 'chain';
  }
}

/**
 * Words for a cabling in a setup label, after the network's short name: `'switch'`, `'mesh'`, `'mesh, 2 cables'`
 * (2 machines) / `'mesh, 2 cables per pair'`, `'ring'`, `'chain'`.
 */
export function cablingLabel(c: CablingId, nodes: number): string {
  switch (c) {
    case 'switched':
      return 'switch';
    case 'mesh':
      return 'mesh';
    case 'mesh2':
      return nodes === 2 ? 'mesh, 2 cables' : 'mesh, 2 cables per pair';
    case 'ring':
      return 'ring';
    case 'chain':
      return 'chain';
  }
}

/**
 * The cablings a network allows for `nodes` machines of `devicesPerNode` devices, in display order. Empty when
 * nodes = 1. An uncatalogued network (a link from an old URL the device does not list) allows only `'switched'`.
 * Rules (`fabricShapeErrors` with the network's ports and switch): a switch only where one exists; a mesh needs
 * (nodes − 1) · cables ports and `mesh2` a network that stripes 2 cables; a ring needs 4+ machines (at 3 it is the
 * mesh) and 2 ports; a chain 3+ machines; machines with several devices network through a switch only; a vendor
 * limit on direct cabling (`maxSwitchlessNodes`) drops the switchless shapes above it.
 */
export function allowedCablings(net: NetworkOption | undefined, nodes: number, devicesPerNode: number): CablingId[] {
  if (nodes <= 1) return [];
  if (net === undefined) return ['switched'];
  const ctx = { nodes, devicesPerNode, portsPerNode: net.portsPerNode, switchable: net.switchable };
  return CABLING_IDS.filter((c) => {
    if (c === 'ring' && nodes < 4) return false;
    if (c === 'chain' && nodes < 3) return false;
    if (c === 'mesh2' && net.maxCablesPerPair < 2) return false;
    if (c !== 'switched' && net.maxSwitchlessNodes !== undefined && nodes > net.maxSwitchlessNodes) return false;
    return fabricShapeErrors(fabricOf(c), ctx).length === 0;
  });
}

/**
 * The cabling a network gets until the user picks one: a switch where one exists, else a mesh where the ports allow,
 * else a ring, else a chain; undefined when nothing is allowed (or nodes = 1).
 */
export function defaultCabling(net: NetworkOption | undefined, nodes: number, devicesPerNode: number): CablingId | undefined {
  const allowed = allowedCablings(net, nodes, devicesPerNode);
  for (const c of ['switched', 'mesh', 'ring', 'chain'] as const) if (allowed.includes(c)) return c;
  return allowed[0];
}

// ---------------------------------------------------------------------------------------------------------------
// Network and local-link options (the data)

const SPARK_PAGE_URL = 'https://www.nvidia.com/en-us/products/workstations/dgx-spark/';
const SPARK_PLAYBOOK_URL = 'https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/README.md';
const STUDIO_2025_URL = 'https://support.apple.com/en-us/122211';
const STUDIO_2026_URL = 'https://support.apple.com/en-us/128107';
const MBP_M3_URL = 'https://support.apple.com/en-us/117736';
const EXO_README_URL = 'https://github.com/exo-explore/exo/blob/main/README.md';
const TN3205_URL = 'https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt';
const MLX_DISTRIBUTED_URL = 'https://ml-explore.github.io/mlx/build/html/usage/distributed.html';
const APPLE_MLX_RESEARCH_URL = 'https://machinelearning.apple.com/research/mlx-distributed-inference';
const GEERLING_URL = 'https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/';
const AMD_395_URL = 'https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html';
const FRAMEWORK_DESKTOP_URL = 'https://web.archive.org/web/2026/https://frame.work/desktop?tab=specs';
const HALO_RDMA_URL = 'https://forum.level1techs.com/t/ryzen-ai-halo-usb4-clustering-with-rdma-testing-notes/253117';

/** The port-count field every network source covers. */
const PORTS = ['portsPerNode'];

/**
 * DGX Spark ConnectX-7: two QSFP ports per Spark, both sharing the 25 GB/s host attachment (the link's all-ports
 * cap). A 200GbE switch exists; a second cable between a pair adds nothing (1 cable per pair); NVIDIA scripts direct
 * cabling up to four Sparks (2 direct, 3 triangle, 4 ring), more go through a switch.
 */
const SPARK_CX7: NetworkOption = {
  link: SPARK_CONNECTX_7,
  shortName: 'ConnectX-7',
  portsPerNode: 2,
  switchable: true,
  maxCablesPerPair: 1,
  maxSwitchlessNodes: 4,
  sources: [
    { url: SPARK_PAGE_URL, title: 'NVIDIA DGX Spark specifications (FAQ)', fields: PORTS, note: 'Two QSFP ports, "each port up to 200 Gb/s"; a 200GbE RoCE v2 switch for clusters.' },
    {
      url: SPARK_PLAYBOOK_URL, title: 'NVIDIA DGX Spark playbook: connect multiple Sparks', fields: ['maxCablesPerPair', 'maxSwitchlessNodes'],
      note: '"Using two cables will not increase performance" (the ports share the host link). Direct cabling scripted for two, three (triangle) and four (ring) Sparks; the playbook says to adapt it by hand beyond four.',
    },
  ],
};

/** Thunderbolt 5 RDMA with `ports` usable RDMA ports on `machine`; no switch exists; JACCL stripes two cables per pair. */
function tb5Rdma(ports: number, machine: string, url: string, note: string): NetworkOption {
  return {
    link: TB5_RDMA,
    shortName: 'TB5',
    portsPerNode: ports,
    switchable: false,
    maxCablesPerPair: 2,
    sources: [
      { url, title: `Apple ${machine} technical specifications`, fields: PORTS, note },
      { url: TN3205_URL, title: 'Apple TN3205: RDMA over Thunderbolt', fields: ['switchable'], note: 'Point-to-point RDMA interfaces; Thunderbolt has no switch. A full mesh is recommended, a ring when there are more Macs than ports.' },
      { url: MLX_DISTRIBUTED_URL, title: 'MLX distributed: multiple cables between Macs', fields: ['maxCablesPerPair'], kind: 'community', note: 'JACCL stripes large messages over all cables between a pair; Apple used two cables per pair for its 2- and 4-node M5 Ultra meshes (Apple ML Research, 2026-10-15).' },
    ],
  };
}

/** Thunderbolt bridge over TCP (`TB5_TCP` on TB5 Macs, `TB4_TCP` on the M3 Max) with `ports` Thunderbolt ports on `machine`. */
function tbTcp(link: LinkSpec, ports: number, machine: string, url: string, note: string): NetworkOption {
  return {
    link,
    shortName: 'TB TCP',
    portsPerNode: ports,
    switchable: false,
    maxCablesPerPair: 1,
    sources: [
      { url, title: `Apple ${machine} technical specifications`, fields: PORTS, note },
      { url: TN3205_URL, title: 'Apple TN3205: the Thunderbolt bridge', fields: ['switchable'], note: 'IP over Thunderbolt is point-to-point; the bridge forwards in software "like a physical Ethernet hub". No switch exists.' },
    ],
  };
}

/** 10 Gigabit Ethernet through a switch, one port, on `machine`. */
function ethernet10G(machine: string, url: string, note: string): NetworkOption {
  return {
    link: ETHERNET_10G_TCP, shortName: '10GbE', portsPerNode: 1, switchable: true, maxCablesPerPair: 1,
    sources: [{ url, title: `${machine} specifications`, fields: PORTS, note }],
  };
}

/** The M3 Ultra Mac Studio (2025): 6 TB5 ports, 5 of them RDMA-capable; 10GbE. */
const M3_ULTRA_STUDIO_NETWORKS: NetworkOption[] = [
  tb5Rdma(5, 'Mac Studio (2025, M3 Ultra)', STUDIO_2025_URL, 'Six Thunderbolt 5 ports (four rear, two front). Five carry RDMA: exo README, "On a Mac Studio, you cannot use the Thunderbolt 5 port next to the Ethernet port" (en2); Geerling, "Apple said all five TB5 ports are RDMA-enabled".'),
  tbTcp(TB5_TCP, 6, 'Mac Studio (2025, M3 Ultra)', STUDIO_2025_URL, 'Six Thunderbolt 5 ports; the bridge (TCP) works on all of them.'),
  ethernet10G('Apple Mac Studio (2025)', STUDIO_2025_URL, 'One 10Gb Ethernet port.'),
];
/** The M4 Max Mac Studio (2025): 4 rear TB5 ports (the front ports are USB-C 10 Gb/s), 3 RDMA-capable; 10GbE. The MacBook Pro M4 Max has 3 TB5 ports. */
const M4_MAX_STUDIO_NETWORKS: NetworkOption[] = [
  tb5Rdma(3, 'Mac Studio (2025, M4 Max)', STUDIO_2025_URL, 'Four rear Thunderbolt 5 ports (the front USB-C ports are 10 Gb/s, not Thunderbolt); the rear port next to Ethernet carries no RDMA (exo README, same chassis as the M3 Ultra), so 3 usable. The MacBook Pro M4 Max (121553) has three TB5 ports.'),
  tbTcp(TB5_TCP, 4, 'Mac Studio (2025, M4 Max)', STUDIO_2025_URL, 'Four rear Thunderbolt 5 ports; the bridge (TCP) works on all of them.'),
  ethernet10G('Apple Mac Studio (2025)', STUDIO_2025_URL, 'One 10Gb Ethernet port.'),
];
/** The M5 Max Mac Studio (2026): 4 rear TB5 ports; whether the en2 RDMA restriction remains is unknown, so 3 (safe). */
const M5_MAX_STUDIO_NETWORKS: NetworkOption[] = [
  tb5Rdma(3, 'Mac Studio (2026, M5 Max)', STUDIO_2026_URL, 'Four rear Thunderbolt 5 ports. ESTIMATED usable count: whether the 2025 Studio\'s "no RDMA on the port next to Ethernet" rule still applies is unknown, so the safe 3 is used. The MacBook Pro M5 Max (126318) has three TB5 ports.'),
  tbTcp(TB5_TCP, 4, 'Mac Studio (2026, M5 Max)', STUDIO_2026_URL, 'Four rear Thunderbolt 5 ports; the bridge (TCP) works on all of them.'),
  ethernet10G('Apple Mac Studio (2026)', STUDIO_2026_URL, 'One 10Gb Ethernet port.'),
];
/** The M5 Ultra Mac Studio (2026): 6 TB5 ports, all inferred RDMA-capable from Apple's 4-node mesh with two cables per pair (6 cables per Mac). */
const M5_ULTRA_STUDIO_NETWORKS: NetworkOption[] = [
  tb5Rdma(6, 'Mac Studio (2026, M5 Ultra)', STUDIO_2026_URL, 'Six Thunderbolt 5 ports (four rear, two front). INFERRED all six carry RDMA: Apple\'s 4-node setup has "each pair of machines linked peer-to-peer by two Thunderbolt cables", i.e. 3 peers × 2 = 6 RDMA cables per Mac (Apple ML Research, 2026-10-15).'),
  tbTcp(TB5_TCP, 6, 'Mac Studio (2026, M5 Ultra)', STUDIO_2026_URL, 'Six Thunderbolt 5 ports; the bridge (TCP) works on all of them.'),
  ethernet10G('Apple Mac Studio (2026)', STUDIO_2026_URL, 'One 10Gb Ethernet port.'),
];
/**
 * The M3 Max MacBook Pro (2023): three Thunderbolt 4 ports, no TB5, so TCP over the bridge only; no Ethernet. Default
 * the bonded 40 Gb/s Mac ↔ Mac bridge; the 20 Gb/s negotiated variant for a host pair that links one lane.
 */
const M3_MAX_MBP_NETWORKS: NetworkOption[] = [
  tbTcp(TB4_TCP, 3, 'MacBook Pro (14-inch, Nov 2023)', MBP_M3_URL, 'Three Thunderbolt 4 (USB-C) ports at 40 Gb/s. No Thunderbolt 5, so no RDMA over Thunderbolt (TN3205 needs TB5): the TB4 bridge (TCP) is the only Thunderbolt link.'),
  { ...tbTcp(TB4_TCP_20G, 3, 'MacBook Pro (14-inch, Nov 2023)', MBP_M3_URL, 'Three Thunderbolt 4 (USB-C) ports; the same bridge when the pair negotiates one 20 Gb/s lane (seen Mac ↔ Linux).'), shortName: 'TB TCP 20G' },
];

/** Strix Halo (Ryzen AI Max+ 395) boxes: two native USB4 ports (RDMA experimental, TCP stock), 5GbE on the Framework Desktop, 10GbE with an add-in NIC. */
const STRIX_HALO_NETWORKS: NetworkOption[] = [
  {
    link: USB4_HALO_RDMA, shortName: 'USB4', portsPerNode: 2, switchable: false, maxCablesPerPair: 1,
    sources: [
      { url: AMD_395_URL, title: 'AMD Ryzen AI Max+ 395 product page', fields: PORTS, note: '2 × USB4 40 Gb/s (the SoC); the Framework Desktop and GMKtec EVO-X2 expose both. Host-to-host negotiates one 20 Gb/s lane.' },
      { url: HALO_RDMA_URL, title: 'Strix Halo USB4 clustering RDMA testing notes', fields: ['switchable', 'maxCablesPerPair'], kind: 'community', note: 'Point-to-point USB4 host links, no switch; one lane per cable, cables not bonded. Four boxes with two ports each form a ring.' },
    ],
  },
  {
    link: USB4_HALO_TCP, shortName: 'USB4 TCP', portsPerNode: 2, switchable: false, maxCablesPerPair: 1,
    sources: [
      { url: AMD_395_URL, title: 'AMD Ryzen AI Max+ 395 product page', fields: PORTS, note: '2 × USB4 40 Gb/s (the SoC); stock thunderbolt-net TCP over them.' },
      { url: HALO_RDMA_URL, title: 'Strix Halo USB4 clustering testing notes', fields: ['switchable', 'maxCablesPerPair'], kind: 'community', note: 'Point-to-point USB4 host links, no switch; one lane per cable.' },
    ],
  },
  {
    link: ETHERNET_5G_TCP, shortName: '5GbE', portsPerNode: 1, switchable: true, maxCablesPerPair: 1,
    sources: [{ url: FRAMEWORK_DESKTOP_URL, title: 'Framework Desktop specifications', fields: PORTS, note: 'One 5 Gigabit Ethernet port (RTL8126); the GMKtec EVO-X2 has 2.5GbE instead (not modeled).' }],
  },
  ethernet10G('Framework Desktop (add-in NIC)', FRAMEWORK_DESKTOP_URL, 'PCIe x4 slot for an add-in NIC (Framework suggests 50 Gb/s RoCE NICs; 10GbE is the modeled option). One port, through a switch.'),
];

/** 10GbE through a switch for a PCIe-card box (a typical desktop or workstation NIC; the box, not the card, cites it). */
const DESKTOP_10GBE: NetworkOption = {
  link: ETHERNET_10G_TCP, shortName: '10GbE', portsPerNode: 1, switchable: true, maxCablesPerPair: 1,
  sources: [{ url: HALO_RDMA_URL, title: 'Strix Halo Ethernet NIC configurations and tests (10GbE RTL812X)', fields: PORTS, kind: 'community', note: 'One 10GbE NIC per machine is user configuration (the link numbers are the 10GbE preset\'s); any count of machines through a switch.' }],
};
/** DGX A100 HDR200 per GPU: one usable NIC port per GPU network shard, switched. */
const IB_HDR: NetworkOption = {
  link: IB_HDR_200,
  shortName: 'IB HDR',
  portsPerNode: 1,
  switchable: true,
  maxCablesPerPair: 1,
  sources: [
    {
      url: 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf',
      title: 'DGX A100 datasheet',
      fields: PORTS,
      note: 'Eight single-port ConnectX-6 HDR200 NICs for eight GPUs, one port per GPU shard; switched InfiniBand.',
    },
  ],
};
/** InfiniBand NDR400, one NIC per GPU, through a switch (DGX H100 / B200 nodes). */
const IB_NDR: NetworkOption = {
  link: IB_NDR_400, shortName: 'IB NDR', portsPerNode: 1, switchable: true, maxCablesPerPair: 1,
  sources: [{ url: 'https://www.nvidia.com/en-us/data-center/dgx-b200/', title: 'DGX B200 specifications', fields: PORTS, note: 'One ConnectX-7 400 Gb/s NIC per GPU; the fabric is switched (NVIDIA Quantum).' }],
};
/** InfiniBand XDR800, one NIC per GPU, through a switch (DGX B300 nodes). */
const IB_XDR: NetworkOption = {
  link: IB_XDR_800, shortName: 'IB XDR', portsPerNode: 1, switchable: true, maxCablesPerPair: 1,
  sources: [{ url: 'https://www.nvidia.com/en-us/data-center/dgx-b300/', title: 'DGX B300 specifications', fields: PORTS, note: 'One ConnectX-8 800 Gb/s NIC per GPU; the fabric is switched.' }],
};

/** A PCIe local link option of generation `generation` and `lanes` electrical lanes (see `LocalLinkOption.pcie`). */
function pcieOption(link: LinkSpec, shortName: string, generation: HostInterfaceSpec['generation'], lanes: HostInterfaceSpec['lanes']): LocalLinkOption {
  return { link, shortName, pcie: { generation, lanes } };
}
const PCIE5_HOST = pcieOption(PCIE_5_X16, 'PCIe 5.0 x16', 5, 16);
const PCIE5_X8_HOST = pcieOption(PCIE_5_X8, 'PCIe 5.0 x8', 5, 8);
const PCIE5_DIRECT = pcieOption(PCIE_5_X16_DIRECT, 'PCIe 5.0 x16 P2P', 5, 16);
const PCIE5_X8_DIRECT_OPT = pcieOption(PCIE_5_X8_DIRECT, 'PCIe 5.0 x8 P2P', 5, 8);
const PCIE4_HOST = pcieOption(PCIE_4_X16, 'PCIe 4.0 x16', 4, 16);
const PCIE4_X8_HOST = pcieOption(PCIE_4_X8, 'PCIe 4.0 x8', 4, 8);
const PCIE4_DIRECT = pcieOption(PCIE_4_X16_DIRECT, 'PCIe 4.0 x16 P2P', 4, 16);
const PCIE4_X8_DIRECT_OPT = pcieOption(PCIE_4_X8_DIRECT, 'PCIe 4.0 x8 P2P', 4, 8);
const PCIE1_X4_HOST = pcieOption(PCIE_1_X4, 'PCIe 1.0 x4', 1, 4);
const PCIE1_X16_HOST = pcieOption(PCIE_1_X16, 'PCIe 1.0 x16', 1, 16);
const PCIE2_X4_HOST = pcieOption(PCIE_2_X4, 'PCIe 2.0 x4', 2, 4);
const PCIE2_X16_HOST = pcieOption(PCIE_2_X16, 'PCIe 2.0 x16', 2, 16);

/** GPUs per machine and machines offered for a PCIe card or a datacenter accelerator. */
const GPU_COUNTS = [1, 2, 4, 8];
const GPU_BOX_NODES = [1, 2, 4];
const DATACENTER_NODES = [1, 2, 4, 8, 16, 18, 32];
/** Configurable active GPUs within a 72-GPU switched domain; 64 permits EP64 for 256 experts. */
const RACK_GPU_COUNTS = [1, 2, 4, 8, 16, 32, 36, 64, 72];
/** Machines offered for a Mac (people run 2–4; 8 as a ring where the ports run out) and for Strix Halo boxes. */
const MAC_NODES = [1, 2, 3, 4, 8];
const HALO_NODES = [1, 2, 3, 4];
/** DGX Spark: 2–4 direct (NVIDIA playbook), 8 through a switch. */
const SPARK_NODES = [1, 2, 3, 4, 8];

/** A PCIe card: local PCIe links (default first) and 10GbE between machines. */
function pcieCard(device: DeviceSpec, shortName: string, localLinks: LocalLinkOption[]): DeviceSetupOptions {
  return { deviceId: device.id, shortName, devicesPerNodeCounts: GPU_COUNTS, nodeCounts: GPU_BOX_NODES, localLinks, networks: [DESKTOP_10GBE] };
}
/** A datacenter accelerator: its scale-up fabric first, then switched InfiniBand. */
function datacenter(device: DeviceSpec, shortName: string, localLinks: LocalLinkOption[], networks: NetworkOption[]): DeviceSetupOptions {
  return { deviceId: device.id, shortName, devicesPerNodeCounts: GPU_COUNTS, nodeCounts: DATACENTER_NODES, localLinks, networks };
}
/** Grace Blackwell: NVL72 switched domain, with a separately selected four-GPU bridge/tray option. */
function rackScale(device: DeviceSpec, shortName: string): DeviceSetupOptions {
  const ultra = device.id === 'gb300';
  const network = (base: NetworkOption): NetworkOption => ({
    ...base,
    sources: [
      {
        url: `https://www.nvidia.com/en-us/data-center/dgx-${device.id}/`,
        title: `DGX ${shortName} NIC specifications`,
        fields: PORTS,
        note: `72 single-port ConnectX-${ultra ? '8 800G' : '7 400G'} adapters for 72 GPUs; one NIC per GPU shard.`,
      },
    ],
  });
  return {
    deviceId: device.id,
    shortName,
    devicesPerNodeCounts: RACK_GPU_COUNTS,
    nodeCounts: [1, 2, 4, 8, 16, 18],
    localLinks: [
      { link: NVLINK_5_NVL72, shortName: 'NVL72 NVLink' },
      // NVIDIA publishes NVL4 for GB200, not GB300: do not invent an Ultra NVL4 product.
      ...(!ultra
        ? [{ link: NVLINK_5_NVL4, shortName: 'NVL4 NVLink bridge', devicesPerNodeCounts: [2, 4] }]
        : []),
    ],
    networks: ultra ? [network(IB_XDR), IB_NDR] : [network(IB_NDR), IB_XDR],
  };
}
/** An Apple SoC: one per Mac, Thunderbolt between Macs. */
function mac(device: DeviceSpec, networks: NetworkOption[], nodeCounts = MAC_NODES): DeviceSetupOptions {
  // 'Apple M5 Ultra (80-core GPU)' → 'M5 Ultra 80-core GPU'.
  const shortName = device.name.replace(/^Apple /, '').replace(/ \((.*)\)$/, ' $1');
  return { deviceId: device.id, shortName, devicesPerNodeCounts: [1], nodeCounts, localLinks: [], networks };
}

/**
 * The catalog, by device id. Defaults (the first local link that fits the stock `hostInterface`, `networks[0]`)
 * reproduce the UI's pre-catalog `defaultLinks` for every device that existed (catalog.test.ts pins them): GeForce PCIe
 * generation host-bounced, Arc Pro B60 direct x8, Arc B580 PCIe 4.0 x8 host-bounced (its Gen4 x8 edge), other cards
 * PCIe 5.0 x16 host-bounced, datacenter NVLink / xGMI / UALoE + InfiniBand, Macs TB5 RDMA,
 * Strix Halo USB4 RDMA, Spark ConnectX-7. Deliberate corrections: MI325X/MI355X xGMI and MI455X UALoE
 * replace the old PCIe placeholders. All other existing defaults are unchanged. PCIe links are offered
 * only while they fit the EFFECTIVE host interface ({@link allowedLocalLinks}); catalog.test.ts checks every device
 * under every unlock selection. The CMP 170HX lists its four links fastest first, so the default follows its PCIe
 * unlocks: stock Gen1 x4 only; capacitor mod Gen1 x16 / x4; Gen2 patch Gen2 x4 / Gen1 x4; both all four (Gen2 x16).
 */
const CATALOG: DeviceSetupOptions[] = [
  pcieCard(RTX_5090, 'RTX 5090', [PCIE5_HOST, PCIE5_X8_HOST, PCIE5_DIRECT, PCIE5_X8_DIRECT_OPT]),
  pcieCard(RTX_4090, 'RTX 4090', [PCIE4_HOST, PCIE4_X8_HOST, PCIE4_DIRECT, PCIE4_X8_DIRECT_OPT]),
  pcieCard(RTX_3090, 'RTX 3090', [PCIE4_HOST, PCIE4_X8_HOST, PCIE4_DIRECT, PCIE4_X8_DIRECT_OPT, { link: RTX_3090_NVLINK, shortName: 'NVLink bridge', devicesPerNodeCounts: [2] }]),
  pcieCard(RTX_PRO_6000_WS, 'RTX PRO 6000 WS', [
    PCIE5_DIRECT,
    PCIE5_X8_DIRECT_OPT,
    PCIE5_HOST,
    PCIE5_X8_HOST,
  ]),
  pcieCard(RTX_PRO_6000_MAX_Q, 'RTX PRO 6000 Max-Q', [
    PCIE5_DIRECT,
    PCIE5_X8_DIRECT_OPT,
    PCIE5_HOST,
    PCIE5_X8_HOST,
  ]),
  pcieCard(RTX_PRO_5000, 'RTX PRO 5000', [
    PCIE5_DIRECT,
    PCIE5_X8_DIRECT_OPT,
    PCIE5_HOST,
    PCIE5_X8_HOST,
  ]),
  pcieCard(RTX_PRO_4000_SFF, 'RTX PRO 4000 SFF', [PCIE5_X8_DIRECT_OPT, PCIE5_X8_HOST]),
  // Fastest first; each is offered only under the unlocks that give it (host bounce only: P2P is not modelled).
  pcieCard(CMP_170HX, 'CMP 170HX', [PCIE2_X16_HOST, PCIE1_X16_HOST, PCIE2_X4_HOST, PCIE1_X4_HOST]),
  mac(M5_ULTRA_80C, M5_ULTRA_STUDIO_NETWORKS),
  mac(M5_ULTRA_64C, M5_ULTRA_STUDIO_NETWORKS),
  mac(M5_MAX_40C, M5_MAX_STUDIO_NETWORKS),
  mac(M5_MAX_32C, M5_MAX_STUDIO_NETWORKS),
  mac(M4_MAX_40C, M4_MAX_STUDIO_NETWORKS),
  mac(M4_MAX_32C, M4_MAX_STUDIO_NETWORKS),
  mac(M3_ULTRA_80C, M3_ULTRA_STUDIO_NETWORKS),
  mac(M3_ULTRA_60C, M3_ULTRA_STUDIO_NETWORKS),
  mac(M3_MAX_40C, M3_MAX_MBP_NETWORKS, [1, 2, 3, 4]),
  mac(M3_MAX_30C, M3_MAX_MBP_NETWORKS, [1, 2, 3, 4]),
  pcieCard(RADEON_AI_PRO_R9700, 'R9700', [PCIE5_HOST, PCIE5_X8_HOST, PCIE5_DIRECT, PCIE4_DIRECT]),
  pcieCard(RADEON_RX_9070_XT, 'RX 9070 XT', [PCIE5_HOST, PCIE5_X8_HOST, PCIE5_DIRECT]),
  { deviceId: RYZEN_AI_MAX_PLUS_395.id, shortName: 'Strix Halo 395', devicesPerNodeCounts: [1], nodeCounts: HALO_NODES, localLinks: [], networks: STRIX_HALO_NETWORKS },
  pcieCard(ARC_PRO_B70, 'Arc Pro B70', [PCIE5_HOST, PCIE5_X8_HOST, PCIE5_DIRECT, PCIE5_X8_DIRECT_OPT]),
  pcieCard(ARC_PRO_B65, 'Arc Pro B65', [PCIE5_HOST, PCIE5_X8_HOST, PCIE5_DIRECT, PCIE5_X8_DIRECT_OPT, PCIE4_DIRECT]),
  // PCIe 5.0 x8 electrical (hostInterface): no x16 link.
  pcieCard(ARC_PRO_B60, 'Arc Pro B60', [PCIE5_X8_DIRECT_OPT, PCIE5_X8_HOST]),
  // PCIe 4.0 x8 (hostInterface): Gen4 x8 links only.
  pcieCard(ARC_B580, 'Arc B580', [PCIE4_X8_HOST, PCIE4_X8_DIRECT_OPT]),
  { deviceId: DGX_SPARK.id, shortName: 'DGX Spark', devicesPerNodeCounts: [1], nodeCounts: SPARK_NODES, localLinks: [], networks: [SPARK_CX7] },
  rackScale(GB300, 'GB300'),
  rackScale(GB200, 'GB200'),
  datacenter(B300, 'B300', [{ link: NVLINK_5, shortName: 'NVLink 5' }], [IB_XDR, IB_NDR]),
  datacenter(B200, 'B200', [{ link: NVLINK_5, shortName: 'NVLink 5' }], [IB_NDR, IB_XDR]),
  datacenter(H200_SXM, 'H200 SXM', [{ link: NVLINK_4, shortName: 'NVLink 4' }], [IB_NDR, IB_XDR]),
  datacenter(H100_SXM, 'H100 SXM', [{ link: NVLINK_4, shortName: 'NVLink 4' }], [IB_NDR, IB_XDR]),
  datacenter(A100_SXM4_80GB, 'A100 SXM4', [{ link: NVLINK_3, shortName: 'NVLink 3' }], [IB_HDR]),
  datacenter(A100_SXM4_40GB, 'A100 SXM4', [{ link: NVLINK_3, shortName: 'NVLink 3' }], [IB_HDR]),
  pcieCard(A100_PCIE_80GB, 'A100 PCIe', [
    PCIE4_DIRECT,
    PCIE4_HOST,
    { link: A100_NVLINK_BRIDGE, shortName: 'NVLink bridge', devicesPerNodeCounts: [2] },
  ]),
  pcieCard(A100_PCIE_40GB, 'A100 PCIe', [
    PCIE4_DIRECT,
    PCIE4_HOST,
    { link: A100_NVLINK_BRIDGE, shortName: 'NVLink bridge', devicesPerNodeCounts: [2] },
  ]),
  // Count-restricted defaults select the pair, full mesh, or hybrid cube-mesh for 2, 4, or 8 GPUs.
  {
    deviceId: V100_SXM2.id,
    shortName: 'V100 SXM2',
    devicesPerNodeCounts: [1, 2, 4, 8],
    nodeCounts: [1],
    localLinks: [
      { link: NVLINK_2_PAIR, shortName: 'NVLink 2 pair', devicesPerNodeCounts: [2] },
      { link: NVLINK_2_MESH_4, shortName: 'NVLink 2 mesh', devicesPerNodeCounts: [4] },
      { link: NVLINK_2_HCM_8, shortName: 'NVLink 2 cube-mesh', devicesPerNodeCounts: [8] },
    ],
    // Required catalog placeholder only: nodeCounts is [1], so no network is offered or costed.
    networks: [DESKTOP_10GBE],
  },
  {
    ...pcieCard(V100_PCIE, 'V100 PCIe', [
      pcieOption(PCIE_3_X16_DIRECT, 'PCIe 3.0 x16 P2P', 3, 16),
      pcieOption(PCIE_3_X16, 'PCIe 3.0 x16', 3, 16),
    ]),
    nodeCounts: [1],
  },
  { ...datacenter(INSTINCT_MI455X, 'MI455X', [{ link: UALOE_MI455X, shortName: 'UALoE' }], [IB_NDR, IB_XDR]), devicesPerNodeCounts: RACK_GPU_COUNTS },
  datacenter(INSTINCT_MI355X, 'MI355X', [{ link: XGMI_MI355X_8_GPU, shortName: 'xGMI' }, { link: XGMI_MI355X_PER_PEER, shortName: 'xGMI (one peer)' }], [IB_NDR, IB_XDR]),
  datacenter(INSTINCT_MI325X, 'MI325X', [{ link: XGMI_MI325X_8_GPU, shortName: 'xGMI' }, { link: XGMI_MI325X_PER_PEER, shortName: 'xGMI (one peer)' }], [IB_NDR, IB_XDR]),
  datacenter(INSTINCT_MI300X, 'MI300X', [{ link: XGMI_MI300X_8_GPU, shortName: 'xGMI' }, { link: XGMI_MI300X_PER_PEER, shortName: 'xGMI (one peer)' }], [IB_NDR, IB_XDR]),
];

/** Every device preset (the vendor arrays; `presets/index.ts` groups the same devices for the picker). */
const ALL_DEVICES: DeviceSpec[] = [
  ...NVIDIA_PRO_DEVICE_PRESETS,
  ...NVIDIA_CONSUMER_DEVICE_PRESETS, ...NVIDIA_CMP_DEVICE_PRESETS, ...NVIDIA_DATACENTER_DEVICE_PRESETS, ...APPLE_DEVICE_PRESETS, ...AMD_DEVICE_PRESETS,
  ...INTEL_DEVICE_PRESETS,
];

const CATALOG_BY_ID = new Map(CATALOG.map((o) => [o.deviceId, o]));

/** The setup options of a device; throws on an id that is not a device preset (every preset has an entry). */
export function setupOptionsFor(deviceId: string): DeviceSetupOptions {
  const options = CATALOG_BY_ID.get(deviceId);
  if (options === undefined) throw new Error(`no setup options for device '${deviceId}'`);
  return options;
}

/** The catalog's device ids, in catalog order (every device preset). */
export function catalogDeviceIds(): string[] {
  return CATALOG.map((o) => o.deviceId);
}

// ---------------------------------------------------------------------------------------------------------------
// Resolution of a choice

/** The device preset with this id, or undefined. */
function deviceById(id: string): DeviceSpec | undefined {
  return ALL_DEVICES.find((d) => d.id === id);
}

/** The link preset with this `linkPresetId` (or a legacy id), or undefined. */
export function linkById(id: string): LinkSpec | undefined {
  const current = LEGACY_LINK_IDS[id] ?? id;
  return LINK_PRESETS.find((l) => linkPresetId(l) === current);
}

/** The network option of a device for a link id (absent id = the default); undefined when the device does not list the link. */
export function networkOf(options: DeviceSetupOptions, networkLinkId: string | undefined): NetworkOption | undefined {
  if (networkLinkId === undefined) return options.networks[0];
  const current = LEGACY_LINK_IDS[networkLinkId] ?? networkLinkId;
  return options.networks.find((n) => linkPresetId(n.link) === current);
}

/** The local-link option of a device for a link id (absent id = the default, if any); undefined when the device does not list the link. */
export function localLinkOf(options: DeviceSetupOptions, localLinkId: string | undefined): LocalLinkOption | undefined {
  if (localLinkId === undefined) return options.localLinks[0];
  const current = LEGACY_LINK_IDS[localLinkId] ?? localLinkId;
  return options.localLinks.find((l) => linkPresetId(l.link) === current);
}

/** Whether a local link supports `devicesPerNode` devices in one machine (`devicesPerNodeCounts` absent = any count). */
export function localLinkAllows(opt: LocalLinkOption, devicesPerNode: number): boolean {
  return opt.devicesPerNodeCounts === undefined || opt.devicesPerNodeCounts.includes(devicesPerNode);
}

/**
 * Whether a local link runs on a card with host interface `hostInterface`: a non-PCIe link always; a PCIe link
 * (`LocalLinkOption.pcie`) when its generation and lanes are ≤ the interface's. An unknown interface (OAM modules,
 * which carry none) allows every link.
 */
export function localLinkFits(opt: LocalLinkOption, hostInterface: HostInterfaceSpec | undefined): boolean {
  return (
    opt.pcie === undefined ||
    hostInterface === undefined ||
    (opt.pcie.generation <= hostInterface.generation && opt.pcie.lanes <= hostInterface.lanes)
  );
}

/**
 * The local links of a device that support `devicesPerNode` devices in one machine and fit its effective host
 * interface (`hostInterface`, after unlocks; absent = the preset's stock one), in catalog order (`[0]` is the default
 * for that count and interface, when any). The UI's local-link picker lists these; a link dropped by a count change
 * (the RTX 3090 NVLink bridge at 4 cards) or an unlock change (the CMP 170HX's PCIe mods) falls back to the first.
 */
export function allowedLocalLinks(
  options: DeviceSetupOptions,
  devicesPerNode: number,
  hostInterface: HostInterfaceSpec | undefined = deviceById(options.deviceId)?.hostInterface,
): LocalLinkOption[] {
  return options.localLinks.filter((l) => localLinkAllows(l, devicesPerNode) && localLinkFits(l, hostInterface));
}

/** A resolved network: the link in use and its catalog option (undefined when the device does not list the link). */
interface ResolvedNetwork {
  /** The link actually used (the chosen preset, else the device's default network link). */
  link: LinkSpec;
  /** The catalog entry of that link, if the device lists it. */
  net?: NetworkOption;
}

/**
 * The network of a choice: the chosen link (with its catalog entry when the device lists it, else uncatalogued), or
 * the device's default network when no id is given or the id is not a link preset.
 */
export function resolveNetwork(options: DeviceSetupOptions, networkLinkId: string | undefined): ResolvedNetwork {
  const chosen = networkLinkId !== undefined ? linkById(networkLinkId) : undefined;
  if (chosen === undefined) return { link: options.networks[0].link, net: options.networks[0] };
  const net = networkOf(options, networkLinkId);
  return net !== undefined ? { link: chosen, net } : { link: chosen };
}

/**
 * The local link of a choice: the chosen preset, else the device's default local link for that count and host
 * interface (the first of {@link allowedLocalLinks}; undefined for SoCs). A chosen link that does not support the
 * count or the interface is still returned (`setupErrors` reports it; the UI falls back before composing).
 */
export function resolveLocalLink(
  options: DeviceSetupOptions,
  localLinkId: string | undefined,
  devicesPerNode = 2,
  hostInterface?: HostInterfaceSpec,
): LinkSpec | undefined {
  const chosen = localLinkId !== undefined ? linkById(localLinkId) : undefined;
  const iface = hostInterface ?? deviceById(options.deviceId)?.hostInterface;
  return chosen ?? allowedLocalLinks(options, devicesPerNode, iface)[0]?.link ?? options.localLinks[0]?.link;
}

/**
 * Errors of a setup choice; empty = valid. Checks: a known device; a memory option of that device; integer counts
 * ≥ 1; link ids that are link presets (a link the device does not list is allowed: old URLs, shown as "other", a
 * switch being its only cabling); with several devices per machine, a catalogued local link that supports that count
 * (the RTX 3090 NVLink bridge joins exactly two cards) and fits the effective host interface (`c.hostInterface`, else
 * the stock one: no PCIe 2.0 x16 on a stock CMP 170HX); with machines > 1, the cabling (or the default) against the
 * network's ports and switch (`fabricShapeErrors`), `mesh2` only where the network stripes two cables, no switchless
 * shape beyond the vendor's direct-cabling limit.
 */
export function setupErrors(c: SetupChoice): string[] {
  const errors: string[] = [];
  const device = deviceById(c.deviceId);
  if (device === undefined) return [`unknown device '${c.deviceId}'`];
  const options = setupOptionsFor(c.deviceId);
  if (c.memCapacityBytes !== undefined && !device.memory.capacityOptionsBytes.includes(c.memCapacityBytes)) {
    errors.push(`${c.memCapacityBytes} B is not a memory option of ${device.id}`);
  }
  for (const [label, n] of [['devicesPerNode', c.devicesPerNode], ['nodes', c.nodes]] as const) {
    if (!(Number.isInteger(n) && n >= 1)) errors.push(`${label} must be an integer ≥ 1 (got ${n})`);
  }
  if (c.localLinkId !== undefined && linkById(c.localLinkId) === undefined) errors.push(`unknown link '${c.localLinkId}'`);
  if (c.devicesPerNode > 1 && c.localLinkId !== undefined) {
    const opt = localLinkOf(options, c.localLinkId);
    if (opt !== undefined && !localLinkAllows(opt, c.devicesPerNode)) {
      errors.push(`${opt.shortName} joins exactly ${opt.devicesPerNodeCounts!.join(' or ')} devices per machine (got ${c.devicesPerNode})`);
    }
    const iface = c.hostInterface ?? device.hostInterface;
    if (opt !== undefined && !localLinkFits(opt, iface)) {
      errors.push(`${opt.shortName} exceeds the card's PCIe ${iface!.generation}.0 x${iface!.lanes} host interface`);
    }
  }
  if (c.networkLinkId !== undefined && linkById(c.networkLinkId) === undefined) errors.push(`unknown link '${c.networkLinkId}'`);
  if (c.nodes > 1) {
    const { net } = resolveNetwork(options, c.networkLinkId);
    const cabling = c.cabling ?? defaultCabling(net, c.nodes, c.devicesPerNode);
    if (cabling === undefined) errors.push(`no cabling connects ${c.nodes} machines of ${device.id} over this network`);
    else if (net === undefined) {
      if (cabling !== 'switched') errors.push(`link '${c.networkLinkId}' is not catalogued for ${device.id}: only a switch is modeled for it`);
    } else {
      const shape = fabricOf(cabling);
      errors.push(...fabricShapeErrors(shape, { nodes: c.nodes, devicesPerNode: c.devicesPerNode, portsPerNode: net.portsPerNode, switchable: net.switchable }));
      if (cabling === 'mesh2' && net.maxCablesPerPair < 2) errors.push(`${net.shortName} does not stripe two cables between a pair`);
      if (cabling !== 'switched' && net.maxSwitchlessNodes !== undefined && c.nodes > net.maxSwitchlessNodes) {
        errors.push(`${net.shortName}: direct cabling is supported for up to ${net.maxSwitchlessNodes} machines (got ${c.nodes}); more go through a switch`);
      }
    }
  }
  return errors;
}

/**
 * The topology of a setup choice (`nodeTopology`): the counts, the chosen or default links at the levels that exist,
 * and the fabric of the chosen or default cabling (`interNodeFabric` is omitted for a switch). The links are the
 * stock presets; the UI substitutes edited links by building the topology itself with the same arguments. Throws on
 * an unknown device.
 */
export function composeTopology(c: SetupChoice): TopologySpec {
  const device = deviceById(c.deviceId);
  if (device === undefined) throw new Error(`unknown device '${c.deviceId}'`);
  const options = setupOptionsFor(c.deviceId);
  const local = resolveLocalLink(options, c.localLinkId, c.devicesPerNode, c.hostInterface);
  const { link: network, net } = resolveNetwork(options, c.networkLinkId);
  const cabling = c.nodes > 1 ? (c.cabling ?? defaultCabling(net, c.nodes, c.devicesPerNode)) : undefined;
  return nodeTopology({
    devicesPerNode: c.devicesPerNode,
    nodes: c.nodes,
    device,
    ...(local !== undefined ? { intraNode: local } : {}),
    interNode: network,
    ...(cabling !== undefined ? { fabric: fabricOf(cabling) } : {}),
  });
}

/** `'256GB'` for 256e9 B (vendor GB, SI); fractional sizes keep up to one decimal. */
function gbLabel(bytes: number): string {
  const gb = bytes / 1e9;
  return `${Number.isInteger(gb) ? gb : gb.toFixed(1)}GB`;
}

/**
 * The SKU-exact label of a setup choice, e.g. `'M5 Ultra 80-core GPU, 256GB ×2 · TB5 mesh'`,
 * `'RTX 5090 32GB ×2 · PCIe 5.0 x16'`, `'DGX Spark 128GB ×4 · ConnectX-7 ring'`, `'RTX 5090 32GB'` for one card,
 * `'H100 SXM 80GB ×8 × 2 machines · NVLink 4 / IB NDR'` for several GPUs in several machines (a switch is implied
 * there, so no cabling word). A link the device does not list is named by its preset name. Throws on an unknown device.
 */
export function setupLabel(c: SetupChoice): string {
  const device = deviceById(c.deviceId);
  if (device === undefined) throw new Error(`unknown device '${c.deviceId}'`);
  const options = setupOptionsFor(c.deviceId);
  const mem = gbLabel(c.memCapacityBytes ?? device.memory.capacityOptionsBytes[0]);
  const base = `${options.shortName}${options.shortName.endsWith('GPU') ? ',' : ''} ${mem}`;
  const localName = (): string => {
    const link = resolveLocalLink(options, c.localLinkId, c.devicesPerNode, c.hostInterface);
    if (link === undefined) return 'no local link';
    return options.localLinks.find((l) => l.link === link)?.shortName ?? link.name;
  };
  const { link: networkLink, net } = resolveNetwork(options, c.networkLinkId);
  const netName = net?.shortName ?? networkLink.name;
  const cabling = c.cabling ?? defaultCabling(net, c.nodes, c.devicesPerNode) ?? 'switched';
  if (c.nodes <= 1) return c.devicesPerNode > 1 ? `${base} ×${c.devicesPerNode} · ${localName()}` : base;
  if (c.devicesPerNode <= 1) return `${base} ×${c.nodes} · ${netName} ${cablingLabel(cabling, c.nodes)}`;
  const word = cabling === 'switched' ? '' : ` ${cablingLabel(cabling, c.nodes)}`;
  return `${base} ×${c.devicesPerNode} × ${c.nodes} machines · ${localName()} / ${netName}${word}`;
}

// ---------------------------------------------------------------------------------------------------------------
// Popular setups

const GB = 1e9;
const EXO_1050_URL = 'https://github.com/exo-explore/exo/issues/1050';
const SPARKRING_URL = 'https://forums.developer.nvidia.com/t/glm-5-2-on-4x-gb10s-switchless-ring-22-tps-coding/378451/1';
const SPARK_SWITCH_URL = 'https://forums.developer.nvidia.com/t/dgx-spark-4-8-cluster-switch-options-post-mikrotik-crs-804-mikrotik-crs-812-market-wipe-out/384594/33';
const SPARK_TWO_URL = 'https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/README.md';
const SPARK_THREE_URL = 'https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-three-sparks/README.md';
const HN_M5_ULTRA_URL = 'https://news.ycombinator.com/item?id=50108006';

/** A community or vendor report of a setup (no numeric field: `fields` is empty). */
function report(url: string, title: string, kind: SourceRef['kind'], note?: string): SourceRef {
  return { url, title, fields: [], ...(kind !== undefined ? { kind } : {}), ...(note !== undefined ? { note } : {}) };
}

/** A popular setup; its name is its label. */
function popular(id: string, choice: SetupChoice, sources: SourceRef[], note?: string): PopularSetup {
  return { id, name: setupLabel(choice), choice, sources, ...(note !== undefined ? { note } : {}) };
}

/**
 * The curated popular setups (≤ 12), grouped by vendor: GeForce boxes, Grace Blackwell racks, DGX Spark clusters (NVIDIA's playbooks),
 * Mac Studio Thunderbolt 5 meshes (Geerling / exo, Apple ML Research, the user's example), the Strix Halo USB4 ring.
 * Every entry passes `setupErrors` (catalog.test.ts). Ids of the setups that existed before the catalog are kept.
 */
export const POPULAR_SETUPS: PopularSetup[] = [
  popular(
    'gb300-nvl72',
    { deviceId: GB300.id, devicesPerNode: 72, nodes: 1, localLinkId: linkPresetId(NVLINK_5_NVL72) },
    [report(GB300_NVL72_URL, 'NVIDIA GB300 NVL72', 'vendor')],
  ),
  popular(
    'gb200-nvl72',
    { deviceId: GB200.id, devicesPerNode: 72, nodes: 1, localLinkId: linkPresetId(NVLINK_5_NVL72) },
    [report(GB200_NVL72_URL, 'NVIDIA GB200 NVL72', 'vendor')],
  ),
  popular('rtx-5090-x2-pcie5', { deviceId: RTX_5090.id, devicesPerNode: 2, nodes: 1, localLinkId: linkPresetId(PCIE_5_X16) }, [
    report('https://github.com/tinygrad/open-gpu-kernel-modules/issues/29', 'RTX 5090 P2P bandwidth and latency matrices', 'community', 'Two cards on Gen5 x16, host-staged without the P2P patch.'),
  ], 'Two RTX 5090s on PCIe 5.0 x16, host-staged (stock GeForce has no P2P).'),
  popular('rtx-3090-x2-nvlink', { deviceId: RTX_3090.id, devicesPerNode: 2, nodes: 1, localLinkId: linkPresetId(RTX_3090_NVLINK) }, [
    report(RTX_3090_NVLINK.sources[0].url, 'NVIDIA Ampere GA102 architecture whitepaper', 'vendor', 'The two-card NVLink bridge, 56.25 GB/s per direction.'),
  ], 'Two RTX 3090s joined by the two-card NVLink bridge.'),
  popular('dgx-spark-x2', { deviceId: DGX_SPARK.id, memCapacityBytes: 128 * GB, devicesPerNode: 1, nodes: 2, networkLinkId: linkPresetId(SPARK_CONNECTX_7), cabling: 'mesh' }, [
    report(SPARK_TWO_URL, 'NVIDIA DGX Spark playbook: connect two Sparks', 'vendor', 'One QSFP cable back to back; a second cable adds no bandwidth.'),
  ], 'Two DGX Sparks cabled back to back over ConnectX-7 200GbE RoCE.'),
  popular('dgx-spark-x3-mesh', { deviceId: DGX_SPARK.id, memCapacityBytes: 128 * GB, devicesPerNode: 1, nodes: 3, networkLinkId: linkPresetId(SPARK_CONNECTX_7), cabling: 'mesh' }, [
    report(SPARK_THREE_URL, 'NVIDIA DGX Spark playbook: connect three Sparks', 'vendor', 'The triangle: both QSFP ports of every Spark, no switch.'),
    report('https://forums.developer.nvidia.com/t/three-node-spark-clusters-without-a-switch-are-now-supported-in-spark-vllm-docker-and-sparkrun/365296/1', 'Three-node Spark clusters without a switch (spark-vllm-docker, sparkrun)', 'community'),
  ], 'Three DGX Sparks cabled as a triangle, no switch.'),
  popular('dgx-spark-x4-ring', { deviceId: DGX_SPARK.id, memCapacityBytes: 128 * GB, devicesPerNode: 1, nodes: 4, networkLinkId: linkPresetId(SPARK_CONNECTX_7), cabling: 'ring' }, [
    report(SPARK_PLAYBOOK_URL, 'NVIDIA DGX Spark playbook: four-device direct ring', 'vendor', '`setup-cluster.sh --topology ring`: each Spark cabled to its two neighbours.'),
    report(SPARKRING_URL, 'GLM-5.2 on 4× GB10 switchless ring (SparkRing)', 'community', 'Host-relayed RoCE collectives on the 4-Spark ring.'),
  ], 'Four DGX Sparks in a switchless ring; the opposite Spark is relayed by a neighbour.'),
  popular('dgx-spark-x4', { deviceId: DGX_SPARK.id, memCapacityBytes: 128 * GB, devicesPerNode: 1, nodes: 4, networkLinkId: linkPresetId(SPARK_CONNECTX_7), cabling: 'switched' }, [
    report(SPARK_PLAYBOOK_URL, 'NVIDIA DGX Spark playbook: switched clusters', 'vendor', 'A 200GbE RoCE v2 switch (MikroTik CRS804 / CRS812 named), one port per Spark.'),
    report(SPARK_SWITCH_URL, 'DGX Spark 4–8 cluster switch options', 'community'),
  ], 'Four DGX Sparks through a non-oversubscribed 200GbE switch.'),
  popular('m3-ultra-80c-512gb-x4-tb5-mesh', { deviceId: M3_ULTRA_80C.id, memCapacityBytes: 512 * GB, devicesPerNode: 1, nodes: 4, networkLinkId: linkPresetId(TB5_RDMA), cabling: 'mesh' }, [
    report(GEERLING_URL, 'Jeff Geerling: 1.5 TB of VRAM on Mac Studios, RDMA over Thunderbolt 5', 'community', 'Four M3 Ultra Studios (2 × 512 GB + 2 × 256 GB), six TB5 cables, exo.'),
    report(EXO_1050_URL, 'exo #1050: 4 × Mac Studio M3 Ultra 512 GB full mesh', 'community', 'A-B, A-C, A-D, B-C, B-D, C-D cabling.'),
    report(EXO_README_URL, 'exo README: Mac Studio RDMA cluster benchmarks', 'community'),
  ], 'Four M3 Ultra Mac Studios in a Thunderbolt 5 RDMA full mesh (six cables).'),
  popular('m5-ultra-80c-256gb-x2-tb5', { deviceId: M5_ULTRA_80C.id, memCapacityBytes: 256 * GB, devicesPerNode: 1, nodes: 2, networkLinkId: linkPresetId(TB5_RDMA), cabling: 'mesh' }, [
    report(HN_M5_ULTRA_URL, 'Hacker News: M5 Ultra Mac Studio pricing and availability', 'community', 'The 256 GB M5 Ultra is the configuration people quote (~$11k); the 512 GB option was not yet shipping in October 2026. Weak source: the user\'s example.'),
  ], 'Two M5 Ultra Mac Studios with 256GB each over one Thunderbolt 5 RDMA cable.'),
  popular('m5-ultra-80c-512gb-x4-tb5-mesh2', { deviceId: M5_ULTRA_80C.id, memCapacityBytes: 512 * GB, devicesPerNode: 1, nodes: 4, networkLinkId: linkPresetId(TB5_RDMA), cabling: 'mesh2' }, [
    report(APPLE_MLX_RESEARCH_URL, 'Apple ML Research: distributed inference with MLX on Mac Studio (2026-10-15)', 'vendor', 'Four M5 Ultra 512 GB Studios, "each pair of machines linked peer-to-peer by two Thunderbolt cables", TP = 4.'),
  ], "Apple's four-Studio M5 Ultra mesh, two Thunderbolt 5 cables per pair."),
  popular('strix-halo-128gb-x4-usb4-ring', { deviceId: RYZEN_AI_MAX_PLUS_395.id, memCapacityBytes: 128 * GB, devicesPerNode: 1, nodes: 4, networkLinkId: linkPresetId(USB4_HALO_RDMA), cabling: 'ring' }, [
    report(HALO_RDMA_URL, 'Strix Halo USB4 clustering with RDMA: testing notes', 'community', 'Two native USB4 ports per box: four boxes cable as a ring (experimental usb4_rdma driver).'),
  ], 'Four Strix Halo boxes in a USB4 ring over their two native ports.'),
];

/** The popular setup with this id, or undefined. */
export function popularSetupById(id: string): PopularSetup | undefined {
  return POPULAR_SETUPS.find((p) => p.id === id);
}

/** Ports a cabling needs per machine for `nodes` (re-exported geometry, for UI hints). */
export function portsForCabling(c: CablingId, nodes: number): number {
  return portsNeeded(fabricOf(c), nodes);
}
