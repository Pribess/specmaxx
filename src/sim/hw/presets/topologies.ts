/**
 * Local-AI and reference datacenter topologies. Counts are user configurations, not vendor peak claims.
 * Link bandwidths are per device per direction. Shared PCIe uplinks and nonuniform routing are not modeled;
 * see docs/simulation-model.md §6. Builders permit swapping a lane width, host DRAM spec or device SKU.
 */
import type { DeviceSpec, LinkSpec, TopologySpec } from '../types';
import { RADEON_AI_PRO_R9700, RYZEN_AI_MAX_PLUS_395 } from './amd';
import { M3_ULTRA_80C, M4_MAX_40C, M5_MAX_40C, M5_ULTRA_80C } from './apple';
import { ARC_PRO_B60 } from './intel';
import { RTX_3090, RTX_4090, RTX_5090 } from './nvidia';
import { DGX_SPARK } from './nvidia-datacenter';
import {
  ETHERNET_5G_TCP, ETHERNET_10G_TCP, IB_NDR_400, IB_XDR_800, NVLINK_4, NVLINK_5,
  PCIE_4_X16, PCIE_5_X16, PCIE_5_X8_DIRECT, RTX_3090_NVLINK, SPARK_CONNECTX_7,
  TB4_TCP, TB5_RDMA, USB4_HALO_RDMA, USB4_HALO_TCP, XGMI_MI300X_8_GPU,
} from './links';

/**
 * Stable, readable id of a link preset: its name slugged (lowercase, runs of other characters → `-`), e.g.
 * `'PCIe 5.0 x16 (host bounce)'` → `'pcie-5-0-x16-host-bounce'`. Names distinguish lane width, transport and
 * staging mode, so the ids are unique over `LINK_PRESETS` (topologies.test.ts). Used in builder ids and by the
 * UI's URL (`LinkSpec` has no id field).
 */
export function linkPresetId(link: Pick<LinkSpec, 'name'>): string {
  return link.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
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
  /** Device that supplies the display name and the id part; omitted: a generic "device". */
  device?: Pick<DeviceSpec, 'id' | 'name'>;
}

/**
 * A general `devicesPerNode × nodes` topology, for configurations the named presets do not cover (the UI composes
 * device count, node count and link presets freely). A link is present only when that level exists: `intraNode`
 * when `devicesPerNode > 1`, `interNode` when `nodes > 1`; a link given for an absent level is dropped. A level
 * that exists but has no link is left without one, so a parallel group that needs it fails placement
 * (`engine/placement.ts`). Counts are user configuration and cite no source.
 */
export function nodeTopology(args: NodeTopologyArgs): TopologySpec {
  const { devicesPerNode, nodes } = args;
  const intraNode = devicesPerNode > 1 ? args.intraNode : undefined;
  const interNode = nodes > 1 ? args.interNode : undefined;
  const deviceId = args.device?.id ?? 'device';
  const deviceName = args.device?.name ?? 'device';
  const idParts = [`node-${devicesPerNode}x${nodes}`, deviceId];
  if (intraNode !== undefined) idParts.push(`intra-${linkPresetId(intraNode)}`);
  if (interNode !== undefined) idParts.push(`inter-${linkPresetId(interNode)}`);
  const inNode = `${devicesPerNode} × ${deviceName}${intraNode !== undefined ? ` / ${intraNode.name}` : ''}`;
  const name = nodes > 1 ? `${nodes} nodes × (${inNode})${interNode !== undefined ? ` / ${interNode.name}` : ''}` : inNode;
  return {
    id: idParts.join('-'),
    name,
    devicesPerNode,
    nodes,
    ...(intraNode !== undefined ? { intraNode } : {}),
    ...(interNode !== undefined ? { interNode } : {}),
    sources: [],
  };
}

/**
 * One PCIe box with `devices` identical devices (count), one node, and a per-device PCIe link.
 * `link` defaults to stock host-bounced Gen5 x16. Use x8 for bifurcated slots; the model does not include
 * shared switch/root-complex uplink contention. Device identity only supplies the display name and id.
 */
export function pcieBox(device: Pick<DeviceSpec, 'id' | 'name'>, devices: number, link: LinkSpec = PCIE_5_X16): TopologySpec {
  return {
    id: `pcie-${devices}-${device.id}-${linkPresetId(link)}`,
    name: `${devices} × ${device.name} / ${link.name}`,
    devicesPerNode: devices, nodes: 1, intraNode: link, sources: [],
  };
}

/**
 * `nodes` single-SoC machines (count), connected over `link` with its per-device injection budget.
 * No automatic port aggregation: one link-rate budget per device. Ethernet assumes a non-oversubscribed
 * switch; point-to-point TB/USB needs suitable mesh/ring cabling and routing support. Four Halo machines
 * with two USB4 ports each can form a ring, not a full mesh: direct all-to-all latency is optimistic there.
 */
export function socCluster(device: Pick<DeviceSpec, 'id' | 'name'>, nodes: number, link: LinkSpec): TopologySpec {
  return {
    id: `cluster-${nodes}-${device.id}-${linkPresetId(link)}`,
    name: `${nodes} × ${device.name} / ${link.name}`,
    devicesPerNode: 1, nodes, interNode: link, sources: [],
  };
}

/**
 * Eight-GPU HGX node(s). `nodes` is a configuration count; `interNode` assumes one full-rate NIC per GPU,
 * not one NIC shared by the entire node (hierarchical collective formulas run eight network shards in parallel).
 */
export function hgxCluster(generation: 'h100' | 'b200' | 'b300', nodes = 1, interNode?: LinkSpec): TopologySpec {
  const intraNode = generation === 'h100' ? NVLINK_4 : NVLINK_5;
  const network = interNode ?? (generation === 'b300' ? IB_XDR_800 : IB_NDR_400);
  const productUrl = generation === 'h100'
    ? 'https://www.nvidia.com/en-us/data-center/dgx-h100/'
    : `https://www.nvidia.com/en-us/data-center/dgx-${generation}/`;
  return {
    id: `hgx-${generation}-${nodes}-nodes-${linkPresetId(network)}`,
    name: `${nodes} × HGX ${generation.toUpperCase()} (8 GPUs/node)`,
    devicesPerNode: 8, nodes, intraNode,
    ...(nodes > 1 ? { interNode: network } : {}),
    sources: [{
      url: productUrl, title: `NVIDIA DGX ${generation.toUpperCase()} eight-GPU system`,
      fields: ['devicesPerNode'], note: 'Eight GPUs in an NVSwitch scale-up island; node count is user-selected.',
    }],
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
  id: 'rtx-3090-nvlink-2', name: '2 × RTX 3090 / NVLink bridge',
  devicesPerNode: 2, nodes: 1, intraNode: RTX_3090_NVLINK,
  sources: [{
    url: RTX_3090_NVLINK.sources[0]!.url, title: 'GA102 two-GPU NVLink bridge', fields: ['devicesPerNode'],
    note: 'RTX 3090 NVLink supports a pair, not a four-card NVLink island.',
  }],
};
/** Two DGX Sparks, one device per node, linked by ConnectX-7 200GbE RoCE at 25 GB/s each direction. */
export const DGX_SPARK_PAIR = socCluster(DGX_SPARK, 2, SPARK_CONNECTX_7);
/**
 * Four DGX Sparks, one device per node, each injecting at its ConnectX-7 200GbE RoCE budget (25 GB/s per
 * direction). Two Sparks can be cabled back to back; four need a 200GbE switch, assumed non-oversubscribed (every
 * Spark keeps its full 25 GB/s through it). The link numbers cite `SPARK_CONNECTX_7`; the count is user
 * configuration.
 */
export const DGX_SPARK_4 = socCluster(DGX_SPARK, 4, SPARK_CONNECTX_7);

/** M3 Ultra Mac Studio RDMA cluster, 2/3/4 nodes, one TB5 per-device injection budget. */
export const M3_ULTRA_TB5_CLUSTERS = [2, 3, 4].map((n) => socCluster(M3_ULTRA_80C, n, TB5_RDMA));
/** M4 Max Mac Studio RDMA cluster, 2/3/4 nodes; TB5 ports are required (not TB4-only machines). */
export const M4_MAX_TB5_CLUSTERS = [2, 3, 4].map((n) => socCluster(M4_MAX_40C, n, TB5_RDMA));
/** M5 Max RDMA cluster, 2/3/4 nodes, one TB5 per-device injection budget. */
export const M5_MAX_TB5_CLUSTERS = [2, 3, 4].map((n) => socCluster(M5_MAX_40C, n, TB5_RDMA));
/** M5 Ultra RDMA cluster, 2/3/4 nodes, one TB5 per-device injection budget. */
export const M5_ULTRA_TB5_CLUSTERS = [2, 3, 4].map((n) => socCluster(M5_ULTRA_80C, n, TB5_RDMA));
/** Mac TCP bridge fallback, 2/3/4 nodes with TB4-compatible cables, nominal 5 GB/s per direction. */
export const MAC_TB4_TCP_CLUSTERS = [2, 3, 4].map((n) => socCluster(M3_ULTRA_80C, n, TB4_TCP));
/** Strix Halo USB4 TCP clusters, 2/3/4 nodes; negotiated 20 Gb/s, with the four-node ring caveat above. */
export const STRIX_HALO_USB4_TCP_CLUSTERS = [2, 3, 4].map((n) => socCluster(RYZEN_AI_MAX_PLUS_395, n, USB4_HALO_TCP));
/** Strix Halo native USB4 RDMA clusters, 2/3/4 nodes; experimental driver required. */
export const STRIX_HALO_USB4_RDMA_CLUSTERS = [2, 3, 4].map((n) => socCluster(RYZEN_AI_MAX_PLUS_395, n, USB4_HALO_RDMA));
/** Strix Halo 5GbE clusters, 2/3/4 nodes, non-oversubscribed switch assumed. */
export const STRIX_HALO_5GBE_CLUSTERS = [2, 3, 4].map((n) => socCluster(RYZEN_AI_MAX_PLUS_395, n, ETHERNET_5G_TCP));
/** Strix Halo 10GbE clusters, 2/3/4 nodes; assumes add-in NICs and a non-oversubscribed switch. */
export const STRIX_HALO_10GBE_CLUSTERS = [2, 3, 4].map((n) => socCluster(RYZEN_AI_MAX_PLUS_395, n, ETHERNET_10G_TCP));

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
  sources: [{
    url: R9700_PRODUCT_URL,
    title: 'AMD Radeon AI PRO R9700 product page',
    fields: ['intraNode'],
    note: '"Bus Type PCIe® 5.0 x16": the card\'s link; the count of four is user configuration.',
  }],
};
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
  id: 'mi300x-xgmi-8', name: '8 × MI300X / xGMI full mesh',
  devicesPerNode: 8, nodes: 1, intraNode: XGMI_MI300X_8_GPU,
  sources: [{
    url: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf',
    title: 'AMD Instinct MI300X platform datasheet', fields: ['devicesPerNode'],
    note: 'Eight GPUs, seven direct peer links per GPU. The engine does not reduce mesh injection for subgroups.',
  }],
};

/** All named multi-device topology presets; each is independently schema-valid. */
export const TOPOLOGY_PRESETS: TopologySpec[] = [
  RTX_5090_PCIE_2, RTX_5090_PCIE_4, RTX_4090_PCIE_2, RTX_4090_PCIE_4, RTX_3090_PCIE_2, RTX_3090_PCIE_4,
  RTX_3090_NVLINK_2, DGX_SPARK_PAIR, DGX_SPARK_4, ARC_PRO_B60_PCIE_2, ARC_PRO_B60_PCIE_4, R9700_PCIE_4,
  ...M3_ULTRA_TB5_CLUSTERS, ...M4_MAX_TB5_CLUSTERS, ...M5_MAX_TB5_CLUSTERS, ...M5_ULTRA_TB5_CLUSTERS,
  ...MAC_TB4_TCP_CLUSTERS, ...STRIX_HALO_USB4_TCP_CLUSTERS, ...STRIX_HALO_USB4_RDMA_CLUSTERS,
  ...STRIX_HALO_5GBE_CLUSTERS, ...STRIX_HALO_10GBE_CLUSTERS,
  HGX_H100_8, HGX_H100_16_NDR, HGX_B200_8, HGX_B200_16_NDR, HGX_B300_8, HGX_B300_16_XDR, MI300X_XGMI_8,
];
