/**
 * Local-AI and reference datacenter topologies. Counts are user configurations, not vendor peak claims.
 * Link bandwidths are per device per direction. Shared PCIe uplinks and nonuniform routing are not modeled;
 * see docs/simulation-model.md §6. Builders permit swapping a lane width, host DRAM spec or device SKU.
 */
import type { DeviceSpec, LinkSpec, TopologySpec } from '../types';
import { RYZEN_AI_MAX_PLUS_395 } from './amd';
import { M3_ULTRA_80C, M4_MAX_40C, M5_MAX_40C, M5_ULTRA_80C } from './apple';
import { ARC_PRO_B60 } from './intel';
import { RTX_3090, RTX_4090, RTX_5090 } from './nvidia';
import { DGX_SPARK } from './nvidia-datacenter';
import {
  ETHERNET_5G_TCP, ETHERNET_10G_TCP, IB_NDR_400, IB_XDR_800, NVLINK_4, NVLINK_5,
  PCIE_4_X16, PCIE_5_X16, PCIE_5_X8_DIRECT, RTX_3090_NVLINK, SPARK_CONNECTX_7,
  TB4_TCP, TB5_RDMA, USB4_HALO_RDMA, USB4_HALO_TCP, XGMI_MI300X_8_GPU,
} from './links';

/** Readable link-name slug for builder ids; names distinguish lane width, transport and staging mode. */
function linkId(link: LinkSpec): string {
  return link.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

/**
 * One PCIe box with `devices` identical devices (count), one node, and a per-device PCIe link.
 * `link` defaults to stock host-bounced Gen5 x16. Use x8 for bifurcated slots; the model does not include
 * shared switch/root-complex uplink contention. Device identity only supplies the display name and id.
 */
export function pcieBox(device: Pick<DeviceSpec, 'id' | 'name'>, devices: number, link: LinkSpec = PCIE_5_X16): TopologySpec {
  return {
    id: `pcie-${devices}-${device.id}-${linkId(link)}`,
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
    id: `cluster-${nodes}-${device.id}-${linkId(link)}`,
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
    id: `hgx-${generation}-${nodes}-nodes-${linkId(network)}`,
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
  RTX_3090_NVLINK_2, DGX_SPARK_PAIR, ARC_PRO_B60_PCIE_2, ARC_PRO_B60_PCIE_4,
  ...M3_ULTRA_TB5_CLUSTERS, ...M4_MAX_TB5_CLUSTERS, ...M5_MAX_TB5_CLUSTERS, ...M5_ULTRA_TB5_CLUSTERS,
  ...MAC_TB4_TCP_CLUSTERS, ...STRIX_HALO_USB4_TCP_CLUSTERS, ...STRIX_HALO_USB4_RDMA_CLUSTERS,
  ...STRIX_HALO_5GBE_CLUSTERS, ...STRIX_HALO_10GBE_CLUSTERS,
  HGX_H100_8, HGX_H100_16_NDR, HGX_B200_8, HGX_B200_16_NDR, HGX_B300_8, HGX_B300_16_XDR, MI300X_XGMI_8,
];
