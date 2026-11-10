/**
 * Local-AI and reference datacenter topologies. Counts are user configurations, not vendor peak claims.
 * Link bandwidths are per device per direction. Shared PCIe uplinks and nonuniform routing are not modeled;
 * see docs/simulation-model.md §6. Builders permit swapping a lane width, host DRAM spec or device SKU, and take
 * the inter-node fabric shape (switched / full mesh / ring / chain, `FabricShape`). The UI composes topologies from
 * `presets/catalog.ts` (device × count × link × cabling); the named constants here serve tests and the sanity harness.
 */
import { isNonBlockingSwitched } from '../fabric';
import {
  BLACKWELL_DATASHEET_URL,
  GB200_NVL72_URL,
  GB300_NVL72_URL,
  MI325X_LINK_DATASHEET_URL,
  MI355X_PLATFORM_URL,
  HELIOS_URL,
} from './rack-scale-sources';
import type { DeviceSpec, FabricShape, LinkSpec, TopologySpec } from '../types';
import { RADEON_AI_PRO_R9700, RYZEN_AI_MAX_PLUS_395 } from './amd';
import { M3_ULTRA_80C, M4_MAX_40C, M5_MAX_40C, M5_ULTRA_80C } from './apple';
import { ARC_PRO_B60 } from './intel';
import { RTX_3090, RTX_4090, RTX_5090 } from './nvidia';
import { A100_PCIE_80GB, DGX_SPARK, V100_PCIE, V100_SXM2 } from './nvidia-datacenter';
import {
  ETHERNET_5G_TCP,
  ETHERNET_10G_TCP,
  IB_NDR_400,
  IB_XDR_800,
  NVLINK_4,
  NVLINK_5,
  NVLINK_5_NVL72,
  NVLINK_5_NVL4,
  XGMI_MI325X_8_GPU,
  XGMI_MI355X_8_GPU,
  UALOE_MI455X,
  A100_NVLINK_BRIDGE,
  IB_HDR_200,
  NVLINK_3,
  NVLINK_2_HCM_8,
  NVLINK_2_MESH_4,
  PCIE_3_X16_DIRECT,
  PCIE_4_X16,
  PCIE_5_X16,
  PCIE_5_X8_DIRECT,
  RTX_3090_NVLINK,
  SPARK_CONNECTX_7,
  TB4_TCP,
  TB5_RDMA,
  USB4_HALO_RDMA,
  USB4_HALO_TCP,
  XGMI_MI300X_8_GPU,
} from './links';

/**
 * Stable, readable id of a link preset: its name slugged (lowercase, runs of other characters → `-`), e.g.
 * `'PCIe 5.0 x16 (host bounce)'` → `'pcie-5-0-x16-host-bounce'`. Names distinguish lane width, transport and
 * staging mode, so the ids are unique over `LINK_PRESETS` (topologies.test.ts). Used in builder ids and by the
 * UI's URL (`LinkSpec` has no id field).
 */
export function linkPresetId(link: Pick<LinkSpec, 'name'>): string {
  return link.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

/**
 * Id suffix of an inter-node fabric shape, appended to a builder id: `''` for the default (absent, or switched
 * non-blocking), `'-switched-o<k>'` for an oversubscribed switch, `'-mesh'` / `'-mesh<k>'` (k cables per pair > 1),
 * `'-ring'`, `'-chain'`. Keeps every pre-fabric id unchanged.
 */
export function fabricIdSuffix(fabric: FabricShape | undefined): string {
  if (fabric === undefined || isNonBlockingSwitched(fabric)) return '';
  switch (fabric.kind) {
    case 'switched':
      return `-switched-o${fabric.oversubscription}`;
    case 'full_mesh':
      return (fabric.cablesPerPair ?? 1) > 1 ? `-mesh${fabric.cablesPerPair}` : '-mesh';
    case 'ring':
      return '-ring';
    case 'chain':
      return '-chain';
  }
}

/** Words for an inter-node fabric shape in a topology name, e.g. `'ring'`, `'full mesh, 2 cables per pair'`; undefined for the default. */
export function fabricWord(fabric: FabricShape | undefined): string | undefined {
  if (fabric === undefined || isNonBlockingSwitched(fabric)) return undefined;
  switch (fabric.kind) {
    case 'switched':
      return `switch, ${fabric.oversubscription}:1 oversubscribed`;
    case 'full_mesh':
      return (fabric.cablesPerPair ?? 1) > 1
        ? `full mesh, ${fabric.cablesPerPair} cables per pair`
        : 'full mesh';
    case 'ring':
      return 'ring';
    case 'chain':
      return 'chain';
  }
}

/** Arguments of {@link nodeTopology}. */
export interface NodeTopologyArgs {
  /** Devices per node (scale-up domain), integer ≥ 1. Count. */
  devicesPerNode: number;
  /** Nodes, integer ≥ 1. Count. */
  nodes: number;
  /** Link between the devices of one node; kept only when `devicesPerNode > 1`. */
  intraNode?: LinkSpec;
  /** Link between nodes; kept only when `nodes > 1`. */
  interNode?: LinkSpec;
  /**
   * Shape of the inter-node network; kept only when `nodes > 1` and `interNode` is present, and omitted from the
   * result when it is the default (absent or non-blocking switched), so pre-fabric topologies are unchanged.
   */
  fabric?: FabricShape;
  /** Device that supplies the display name and the id part; omitted: a generic "device". */
  device?: Pick<DeviceSpec, 'id' | 'name'>;
}

/**
 * A general `devicesPerNode × nodes` topology, for configurations the named presets do not cover (the UI composes
 * device count, node count, link presets and cabling freely). A link is present only when that level exists:
 * `intraNode` when `devicesPerNode > 1`, `interNode` when `nodes > 1`; a link given for an absent level is dropped. A
 * level that exists but has no link is left without one, so a parallel group that needs it fails placement
 * (`engine/placement.ts`). The fabric shape follows the inter-node link (dropped with it, and when default). Counts are
 * user configuration and cite no source. The id ends in `fabricIdSuffix`; the name carries `fabricWord`.
 */
export function nodeTopology(args: NodeTopologyArgs): TopologySpec {
  const { devicesPerNode, nodes } = args;
  const intraNode = devicesPerNode > 1 ? args.intraNode : undefined;
  const interNode = nodes > 1 ? args.interNode : undefined;
  const fabric =
    interNode !== undefined && !isNonBlockingSwitched(args.fabric) ? args.fabric : undefined;
  const deviceId = args.device?.id ?? 'device';
  const deviceName = args.device?.name ?? 'device';
  const idParts = [`node-${devicesPerNode}x${nodes}`, deviceId];
  if (intraNode !== undefined) idParts.push(`intra-${linkPresetId(intraNode)}`);
  if (interNode !== undefined) idParts.push(`inter-${linkPresetId(interNode)}`);
  const inNode = `${devicesPerNode} × ${deviceName}${intraNode !== undefined ? ` / ${intraNode.name}` : ''}`;
  const shape = fabricWord(fabric);
  const network =
    interNode !== undefined
      ? ` / ${interNode.name}${shape !== undefined ? ` (${shape})` : ''}`
      : '';
  const name = nodes > 1 ? `${nodes} nodes × (${inNode})${network}` : inNode;
  return {
    id: idParts.join('-') + fabricIdSuffix(fabric),
    name,
    devicesPerNode,
    nodes,
    ...(intraNode !== undefined ? { intraNode } : {}),
    ...(interNode !== undefined ? { interNode } : {}),
    ...(fabric !== undefined ? { interNodeFabric: fabric } : {}),
    sources: [],
  };
}

/**
 * One PCIe box with `devices` identical devices (count), one node, and a per-device PCIe link.
 * `link` defaults to stock host-bounced Gen5 x16. Use x8 for bifurcated slots; the model does not include
 * shared switch/root-complex uplink contention. Device identity only supplies the display name and id.
 */
export function pcieBox(
  device: Pick<DeviceSpec, 'id' | 'name'>,
  devices: number,
  link: LinkSpec = PCIE_5_X16,
): TopologySpec {
  return {
    id: `pcie-${devices}-${device.id}-${linkPresetId(link)}`,
    name: `${devices} × ${device.name} / ${link.name}`,
    devicesPerNode: devices,
    nodes: 1,
    intraNode: link,
    sources: [],
  };
}

/**
 * `nodes` single-SoC machines (count), connected over `link` with its per-device injection budget, cabled as
 * `fabric` (absent = a non-blocking switch, the pre-fabric convention; a switch only exists for Ethernet and RoCE).
 * Point-to-point links (Thunderbolt, USB4, back-to-back ConnectX-7) take `full_mesh` while the ports allow, else
 * `ring` or `chain`; relays on a ring or chain go through the host (see `FabricShape`). Which shapes a machine's
 * port count allows is the catalog's business (`presets/catalog.ts`); `validateTopology` checks only the structure.
 */
export function socCluster(
  device: Pick<DeviceSpec, 'id' | 'name'>,
  nodes: number,
  link: LinkSpec,
  fabric?: FabricShape,
): TopologySpec {
  const shape = nodes > 1 && !isNonBlockingSwitched(fabric) ? fabric : undefined;
  const word = fabricWord(shape);
  return {
    id: `cluster-${nodes}-${device.id}-${linkPresetId(link)}${fabricIdSuffix(shape)}`,
    name: `${nodes} × ${device.name} / ${link.name}${word !== undefined ? ` (${word})` : ''}`,
    devicesPerNode: 1,
    nodes,
    interNode: link,
    ...(shape !== undefined ? { interNodeFabric: shape } : {}),
    sources: [],
  };
}

/**
 * Eight-GPU HGX node(s). `nodes` is a configuration count; `interNode` assumes one full-rate NIC per GPU,
 * not one NIC shared by the entire node (hierarchical collective formulas run eight network shards in parallel).
 */
export function hgxCluster(
  generation: 'a100' | 'h100' | 'b200' | 'b300',
  nodes = 1,
  interNode?: LinkSpec,
): TopologySpec {
  const intraNode = generation === 'a100' ? NVLINK_3 : generation === 'h100' ? NVLINK_4 : NVLINK_5;
  const network =
    interNode ??
    (generation === 'a100' ? IB_HDR_200 : generation === 'b300' ? IB_XDR_800 : IB_NDR_400);
  const productUrl =
    generation === 'a100'
      ? 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf'
      : generation === 'h100'
        ? 'https://www.nvidia.com/en-us/data-center/dgx-h100/'
        : `https://www.nvidia.com/en-us/data-center/dgx-${generation}/`;
  return {
    id: `hgx-${generation}-${nodes}-nodes-${linkPresetId(network)}`,
    name: `${nodes} × HGX ${generation.toUpperCase()} (8 GPUs/node)`,
    devicesPerNode: 8,
    nodes,
    intraNode,
    ...(nodes > 1 ? { interNode: network } : {}),
    sources: [
      {
        url: productUrl,
        title: `NVIDIA DGX ${generation.toUpperCase()} eight-GPU system`,
        fields: ['devicesPerNode'],
        note: 'Eight GPUs in an NVSwitch scale-up island; node count is user-selected.',
      },
    ],
  };
}

/** Two stock RTX 5090s, Gen5 x16 each; host staging shares the assumed desktop DRAM. */
export const RTX_5090_PCIE_2 = pcieBox(RTX_5090, 2);
/** Four stock RTX 5090s, Gen5 x16 each; lane provisioning and shared uplinks are host-dependent. */
export const RTX_5090_PCIE_4 = pcieBox(RTX_5090, 4);
/** Two stock RTX 4090s, Gen4 x16 each, host-staged. */
export const RTX_4090_PCIE_2 = pcieBox(RTX_4090, 2, PCIE_4_X16);
/** Four stock RTX 4090s, Gen4 x16 each, host-staged. */
export const RTX_4090_PCIE_4 = pcieBox(RTX_4090, 4, PCIE_4_X16);
/** Two stock RTX 3090s, Gen4 x16 each, host-staged (no NVLink bridge). */
export const RTX_3090_PCIE_2 = pcieBox(RTX_3090, 2, PCIE_4_X16);
/** Four stock RTX 3090s, Gen4 x16 each, host-staged (no NVLink bridge). */
export const RTX_3090_PCIE_4 = pcieBox(RTX_3090, 4, PCIE_4_X16);
/** Two RTX 3090s connected by the supported two-card NVLink bridge, 56.25 GB/s per direction. */
export const RTX_3090_NVLINK_2: TopologySpec = {
  id: 'rtx-3090-nvlink-2',
  name: '2 × RTX 3090 / NVLink bridge',
  devicesPerNode: 2,
  nodes: 1,
  intraNode: RTX_3090_NVLINK,
  sources: [
    {
      url: RTX_3090_NVLINK.sources[0]!.url,
      title: 'GA102 two-GPU NVLink bridge',
      fields: ['devicesPerNode'],
      note: 'RTX 3090 NVLink supports a pair, not a four-card NVLink island.',
    },
  ],
};
/** The NVIDIA playbook that scripts 2-, 3- and 4-Spark direct cabling (`setup-cluster.sh --topology ring`) and names the 200GbE switches. */
const SPARK_PLAYBOOK_URL =
  'https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/README.md';
/** Cites the cabling of a DGX Spark cluster (the counts are user configuration; the link numbers live on `SPARK_CONNECTX_7`). */
function sparkCabling(note: string): TopologySpec['sources'] {
  return [
    {
      url: SPARK_PLAYBOOK_URL,
      title: 'NVIDIA DGX Spark playbook: connect multiple Sparks',
      fields: ['interNodeFabric'],
      note,
    },
  ];
}
/**
 * Two DGX Sparks, one device per node, one QSFP cable back to back: ConnectX-7 200GbE RoCE at 25 GB/s each
 * direction. A two-node full mesh with one cable is the same cost as a switch, so the shape stays the default (the
 * pre-fabric id and results are unchanged). A second cable adds nothing (the all-ports cap, `SPARK_CONNECTX_7`).
 */
export const DGX_SPARK_PAIR = socCluster(DGX_SPARK, 2, SPARK_CONNECTX_7);
/**
 * Three DGX Sparks cabled as a triangle (NVIDIA's "three-node ring"): every pair direct, two ports per Spark, one
 * hop everywhere, i.e. a full mesh. Both ports share the 25 GB/s cap.
 */
export const DGX_SPARK_3: TopologySpec = {
  ...socCluster(DGX_SPARK, 3, SPARK_CONNECTX_7, { kind: 'full_mesh' }),
  sources: sparkCabling(
    '"Three-device" topology: each Spark cabled to both others (QSFP port 0 and 1), no switch.',
  ),
};
/**
 * Four DGX Sparks through a 200GbE switch (e.g. MikroTik CRS804 / CRS812, named in the playbook), assumed
 * non-oversubscribed: every Spark keeps its full 25 GB/s and every pair is one hop. One QSFP port per Spark. The
 * pre-fabric preset (default shape, results unchanged).
 */
export const DGX_SPARK_4 = socCluster(DGX_SPARK, 4, SPARK_CONNECTX_7);
/**
 * Four DGX Sparks in a switchless ring: each Spark's two QSFP ports go to its two neighbours (4 cables), so the
 * opposite Spark is two hops away and a neighbour relays through its host (RoCE cannot route through a Spark in
 * hardware; `SPARK_CONNECTX_7.forwardingLatencySec`). Both ports share the 25 GB/s cap, so a ring all-reduce costs
 * exactly what the switch does; latency-bound and non-neighbour traffic pays the relay.
 */
export const DGX_SPARK_4_RING: TopologySpec = {
  ...socCluster(DGX_SPARK, 4, SPARK_CONNECTX_7, { kind: 'ring' }),
  sources: sparkCabling(
    '"Four-device direct ring": `setup-cluster.sh --topology ring`; port 0 to the previous Spark, port 1 to the next.',
  ),
};

/** Thunderbolt and USB4 are point-to-point: no switch exists, so N Macs or Halo boxes are a full mesh while the ports allow (`presets/catalog.ts`). */
const MESH: FabricShape = { kind: 'full_mesh' };
/** M3 Ultra Mac Studio RDMA cluster, 2/3/4 nodes, full mesh of TB5 cables (5 RDMA-capable ports: up to 6 nodes). */
export const M3_ULTRA_TB5_CLUSTERS = [2, 3, 4].map((n) =>
  socCluster(M3_ULTRA_80C, n, TB5_RDMA, MESH),
);
/** M4 Max Mac Studio RDMA cluster, 2/3/4 nodes, full mesh of TB5 cables (3 RDMA-capable ports: up to 4 nodes). */
export const M4_MAX_TB5_CLUSTERS = [2, 3, 4].map((n) => socCluster(M4_MAX_40C, n, TB5_RDMA, MESH));
/** M5 Max RDMA cluster, 2/3/4 nodes, full mesh of TB5 cables (3 RDMA-capable ports assumed: up to 4 nodes). */
export const M5_MAX_TB5_CLUSTERS = [2, 3, 4].map((n) => socCluster(M5_MAX_40C, n, TB5_RDMA, MESH));
/** M5 Ultra RDMA cluster, 2/3/4 nodes, full mesh of TB5 cables (6 ports inferred usable: up to 7 nodes, or 4 with two cables per pair). */
export const M5_ULTRA_TB5_CLUSTERS = [2, 3, 4].map((n) =>
  socCluster(M5_ULTRA_80C, n, TB5_RDMA, MESH),
);
/** Mac TCP bridge fallback, 2/3/4 nodes with TB4-compatible cables in a full mesh, nominal 5 GB/s per direction. */
export const MAC_TB4_TCP_CLUSTERS = [2, 3, 4].map((n) =>
  socCluster(M3_ULTRA_80C, n, TB4_TCP, MESH),
);
/** The shape of N Strix Halo boxes over their two native USB4 ports: 2 direct, 3 a triangle (mesh), 4 a ring (2 hops at most). */
function haloUsb4Shape(nodes: number): FabricShape {
  return nodes <= 3 ? MESH : { kind: 'ring' };
}
/** Strix Halo USB4 TCP clusters, 2/3/4 nodes over the two native USB4 ports (20 Gb/s negotiated); four boxes form a ring. */
export const STRIX_HALO_USB4_TCP_CLUSTERS = [2, 3, 4].map((n) =>
  socCluster(RYZEN_AI_MAX_PLUS_395, n, USB4_HALO_TCP, haloUsb4Shape(n)),
);
/** Strix Halo native USB4 RDMA clusters, 2/3/4 nodes (experimental driver); four boxes form a ring. */
export const STRIX_HALO_USB4_RDMA_CLUSTERS = [2, 3, 4].map((n) =>
  socCluster(RYZEN_AI_MAX_PLUS_395, n, USB4_HALO_RDMA, haloUsb4Shape(n)),
);
/** Strix Halo 5GbE clusters, 2/3/4 nodes, non-oversubscribed switch assumed. */
export const STRIX_HALO_5GBE_CLUSTERS = [2, 3, 4].map((n) =>
  socCluster(RYZEN_AI_MAX_PLUS_395, n, ETHERNET_5G_TCP),
);
/** Strix Halo 10GbE clusters, 2/3/4 nodes; assumes add-in NICs and a non-oversubscribed switch. */
export const STRIX_HALO_10GBE_CLUSTERS = [2, 3, 4].map((n) =>
  socCluster(RYZEN_AI_MAX_PLUS_395, n, ETHERNET_10G_TCP),
);

/** Two Intel Arc Pro B60 GPUs, Gen5 x8 direct P2P on compatible xe driver/host routing. */
export const ARC_PRO_B60_PCIE_2 = pcieBox(ARC_PRO_B60, 2, PCIE_5_X8_DIRECT);
/** Four Intel Arc Pro B60 GPUs, Gen5 x8 each, direct P2P on compatible xe driver/host routing. */
export const ARC_PRO_B60_PCIE_4 = pcieBox(ARC_PRO_B60, 4, PCIE_5_X8_DIRECT);
/** AMD's R9700 page, "Bus Type PCIe® 5.0 x16" (the interface each card gets in `R9700_PCIE_4`). */
const R9700_PRODUCT_URL =
  'https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html';
/**
 * Four Radeon AI PRO R9700s in one workstation, PCIe 5.0 x16 each, host-staged (no cited ROCm source for direct
 * Radeon peer-to-peer over PCIe, so the stock host-bounce link is used). Four full x16 Gen5 slots need a
 * workstation platform (e.g. Threadripper PRO / Xeon W lane counts); a desktop board bifurcates to x8. Shared
 * root-complex uplinks are not modeled (docs §6).
 */
export const R9700_PCIE_4: TopologySpec = {
  ...pcieBox(RADEON_AI_PRO_R9700, 4, PCIE_5_X16),
  sources: [
    {
      url: R9700_PRODUCT_URL,
      title: 'AMD Radeon AI PRO R9700 product page',
      fields: ['intraNode'],
      note: '"Bus Type PCIe® 5.0 x16": the card\'s link; the count of four is user configuration.',
    },
  ],
};
/** Eight A100 SXM4 GPUs on HGX/DGX NVSwitch: 300 GB/s per device per direction. */
export const HGX_A100_8 = hgxCluster('a100');
/** Two eight-GPU HGX A100 nodes, one HDR200 NIC (25 GB/s per direction) per GPU. */
export const HGX_A100_16_HDR = hgxCluster('a100', 2);
/** Two matching A100 PCIe 80GB cards, all three NVLink bridges: 300 GB/s per direction. */
export const A100_PCIE_NVLINK_2: TopologySpec = {
  ...pcieBox(A100_PCIE_80GB, 2, A100_NVLINK_BRIDGE),
  sources: [
    {
      url: A100_NVLINK_BRIDGE.sources[0]!.url,
      title: 'A100 PCIe bridge pair',
      fields: ['devicesPerNode'],
      note: 'One adjacent matching-capacity A100, all three bridges; exactly two GPUs.',
    },
  ],
};
/** Four V100 SXM2s, fully connected NV2 board: aggregate 150 GB/s per GPU per direction (TP2 only 50). */
export const V100_SXM2_NVLINK_4: TopologySpec = {
  ...pcieBox(V100_SXM2, 4, NVLINK_2_MESH_4),
  sources: [
    {
      url: 'https://github.com/NVIDIA/nccl-tests/issues/251',
      title: 'Four V100 SXM2 NV2 mesh',
      kind: 'community',
      fields: ['devicesPerNode'],
      note: 'Four GPUs, NV2 to every other GPU; not a DGX-1 quad.',
    },
  ],
};
/** Eight V100 SXM2 DGX-1 hybrid cube-mesh; full-node injection 150 GB/s, not uniform peer bandwidth. */
export const V100_SXM2_NVLINK_8: TopologySpec = {
  ...pcieBox(V100_SXM2, 8, NVLINK_2_HCM_8),
  sources: [
    {
      url: NVLINK_2_HCM_8.sources[0]!.url,
      title: 'Volta whitepaper DGX-1 Figure 12',
      fields: ['devicesPerNode'],
      note: 'Eight GPUs in a hybrid cube-mesh, not all-to-all. Uniform link model is optimistic for subgroups and non-neighbours.',
    },
  ],
};
/** Two genuine V100 PCIe cards, Gen3 x16 direct P2P; no NVLink. */
export const V100_PCIE_2 = pcieBox(V100_PCIE, 2, PCIE_3_X16_DIRECT);
/** Four genuine V100 PCIe cards, Gen3 x16 direct P2P; host routing and lane provisioning assumed. */
export const V100_PCIE_4 = pcieBox(V100_PCIE, 4, PCIE_3_X16_DIRECT);
/** One HGX H100 eight-GPU NVLink4 scale-up node, 450 GB/s per direction per GPU. */
export const HGX_H100_8 = hgxCluster('h100');
/** Two HGX H100 nodes, NVLink4 inside and NDR400 per GPU between nodes. */
export const HGX_H100_16_NDR = hgxCluster('h100', 2);
/** One HGX B200 eight-GPU NVLink5 scale-up node, 900 GB/s per direction per GPU. */
export const HGX_B200_8 = hgxCluster('b200');
/** Two HGX B200 nodes, NVLink5 inside and NDR400 per GPU between nodes. */
export const HGX_B200_16_NDR = hgxCluster('b200', 2);
/** One HGX B300 eight-GPU NVLink5 scale-up node. */
export const HGX_B300_8 = hgxCluster('b300');
/** Two HGX B300 nodes, NVLink5 inside and XDR800 per GPU between nodes. */
export const HGX_B300_16_XDR = hgxCluster('b300', 2);
/** Eight MI300X GPUs in a full xGMI mesh; aggregate 448 GB/s per GPU, optimistic for subgroups/point-to-point. */
export const MI300X_XGMI_8: TopologySpec = {
  id: 'mi300x-xgmi-8',
  name: '8 × MI300X / xGMI full mesh',
  devicesPerNode: 8,
  nodes: 1,
  intraNode: XGMI_MI300X_8_GPU,
  sources: [
    {
      url: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf',
      title: 'AMD Instinct MI300X platform datasheet',
      fields: ['devicesPerNode'],
      note: 'Eight GPUs, seven direct peer links per GPU. The engine does not reduce mesh injection for subgroups.',
    },
  ],
};

/**
 * NVL72 racks: 72 GPUs per switched scale-up domain. racks is a positive integer count.
 * NIC injection is per GPU per direction: GB200 NDR400, GB300 XDR800 (docs §4, §6).
 * See docs/simulation-model.md §4 "Scale-up domains".
 */
export function nvl72Rack(gen: 'gb200' | 'gb300', racks = 1): TopologySpec {
  if (!Number.isInteger(racks) || racks < 1) throw new Error('racks must be a positive integer');
  return {
    ...nodeTopology({
      device: { id: gen, name: gen.toUpperCase() },
      devicesPerNode: 72,
      nodes: racks,
      intraNode: NVLINK_5_NVL72,
      interNode: gen === 'gb200' ? IB_NDR_400 : IB_XDR_800,
    }),
    sources: [
      {
        url: gen === 'gb200' ? GB200_NVL72_URL : GB300_NVL72_URL,
        title: 'NVIDIA NVL72 specifications',
        fields: ['devicesPerNode'],
        note: '72 GPUs in one NVLink Switch domain.',
      },
    ],
  };
}
/**
 * GB200 NVL4 boards: four GPUs joined by a bridge per tray, NDR400 per GPU between boards.
 * trays is a positive integer count; no inter-board NVLink is assumed; internal mesh is estimated (link provenance).
 * See docs/simulation-model.md §4 "Scale-up domains".
 */
export function nvl4Trays(gen: 'gb200', trays = 1): TopologySpec {
  if (!Number.isInteger(trays) || trays < 1) throw new Error('trays must be a positive integer');
  return {
    ...nodeTopology({
      device: { id: gen, name: 'GB200 NVL4' },
      devicesPerNode: 4,
      nodes: trays,
      intraNode: NVLINK_5_NVL4,
      interNode: IB_NDR_400,
    }),
    sources: [
      {
        url: BLACKWELL_DATASHEET_URL,
        title: 'Blackwell datasheet OCT25 pp. 4, 5, 8',
        fields: ['devicesPerNode'],
        note: 'Four GPUs joined by NVLink bridge. NVIDIA projections assume 400G/GPU network; no inter-board NVLink specified.',
      },
    ],
  };
}
/** GB200 NVL72: 72 GPUs, one NVLink Switch domain. */
export const GB200_NVL72 = nvl72Rack('gb200');
/** GB300 NVL72: 72 GPUs, one NVLink Switch domain. */
export const GB300_NVL72 = nvl72Rack('gb300');
/** GB200: 18 NVL4 boards, 72 GPUs total, NDR400 per GPU between boards. */
export const GB200_NVL4_X18 = nvl4Trays('gb200', 18);
/** MI325X eight-GPU full mesh, aggregate 448 GB/s per GPU per direction, optimistic for subgroups.
 * See docs/simulation-model.md §4 "Scale-up domains".
 */
export const MI325X_XGMI_8: TopologySpec = {
  ...MI300X_XGMI_8,
  id: 'mi325x-xgmi-8',
  name: '8 × MI325X / xGMI full mesh',
  intraNode: XGMI_MI325X_8_GPU,
  sources: [
    {
      url: MI325X_LINK_DATASHEET_URL,
      title: 'MI325X datasheet',
      fields: ['devicesPerNode'],
      note: 'Eight accelerators with seven direct GPU-peer links.',
    },
  ],
};
/** MI355X eight-GPU full mesh, aggregate 504 GB/s per GPU per direction, optimistic for subgroups.
 * See docs/simulation-model.md §4 "Scale-up domains".
 */
export const MI355X_XGMI_8: TopologySpec = {
  ...MI300X_XGMI_8,
  id: 'mi355x-xgmi-8',
  name: '8 × MI355X / xGMI full mesh',
  intraNode: XGMI_MI355X_8_GPU,
  sources: [
    {
      url: MI355X_PLATFORM_URL,
      title: 'MI355X platform brochure',
      fields: ['devicesPerNode'],
      note: 'Eight accelerators with seven direct GPU-peer links.',
    },
  ],
};
/** Helios: 72 MI455X GPUs in a switched UALoE scale-up domain, 1.8 TB/s injection per GPU per direction.
 * See docs/simulation-model.md §4 "Scale-up domains".
 */
export const MI455X_HELIOS_72: TopologySpec = {
  id: 'mi455x-helios-72',
  name: '72 × MI455X / Helios UALoE',
  devicesPerNode: 72,
  nodes: 1,
  intraNode: UALOE_MI455X,
  sources: [
    {
      url: HELIOS_URL,
      title: 'AMD Helios solution brief',
      fields: ['devicesPerNode'],
      note: '18 four-GPU trays, six switch trays; single-switch-hop UALoE fabric.',
    },
  ],
};

/** All named multi-device topology presets; each is independently schema-valid. */
export const TOPOLOGY_PRESETS: TopologySpec[] = [
  GB200_NVL72,
  GB300_NVL72,
  GB200_NVL4_X18,
  MI325X_XGMI_8,
  MI355X_XGMI_8,
  MI455X_HELIOS_72,
  HGX_A100_8,
  HGX_A100_16_HDR,
  A100_PCIE_NVLINK_2,
  V100_SXM2_NVLINK_4,
  V100_SXM2_NVLINK_8,
  V100_PCIE_2,
  V100_PCIE_4,
  RTX_5090_PCIE_2,
  RTX_5090_PCIE_4,
  RTX_4090_PCIE_2,
  RTX_4090_PCIE_4,
  RTX_3090_PCIE_2,
  RTX_3090_PCIE_4,
  RTX_3090_NVLINK_2,
  DGX_SPARK_PAIR,
  DGX_SPARK_3,
  DGX_SPARK_4,
  DGX_SPARK_4_RING,
  ARC_PRO_B60_PCIE_2,
  ARC_PRO_B60_PCIE_4,
  R9700_PCIE_4,
  ...M3_ULTRA_TB5_CLUSTERS,
  ...M4_MAX_TB5_CLUSTERS,
  ...M5_MAX_TB5_CLUSTERS,
  ...M5_ULTRA_TB5_CLUSTERS,
  ...MAC_TB4_TCP_CLUSTERS,
  ...STRIX_HALO_USB4_TCP_CLUSTERS,
  ...STRIX_HALO_USB4_RDMA_CLUSTERS,
  ...STRIX_HALO_5GBE_CLUSTERS,
  ...STRIX_HALO_10GBE_CLUSTERS,
  HGX_H100_8,
  HGX_H100_16_NDR,
  HGX_B200_8,
  HGX_B200_16_NDR,
  HGX_B300_8,
  HGX_B300_16_XDR,
  MI300X_XGMI_8,
];
