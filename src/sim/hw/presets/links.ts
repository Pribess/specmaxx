/**
 * Interconnect ceilings: SI B/s per device per direction, never bidirectional sums.
 * Latencies are one-way hardware/verbs proxies, not NCCL/RCCL launch floors. Unpublished or transferred
 * latency assumptions are explicitly estimated. See docs/simulation-model.md §4 ("Links and topologies"),
 * §6 ("Collectives"), and engine/collectives.ts.
 */
import type { LinkSpec, SourceRef } from '../types';

const BANDWIDTH_FIELD = 'bandwidthPerDirectionBytesPerSec';
const PCIE_SIG_URL = 'https://pcisig.com/faq?field_category_value%5B%5D=pci_express_5.0&keys=';
const GEFORCE_P2P_URL = 'https://github.com/tinygrad/open-gpu-kernel-modules/issues/29';
const GB200_LATENCY_URL = 'https://arxiv.org/pdf/2607.16100';
const HALO_TCP_URL = 'https://forum.level1techs.com/t/benchmarking-usb4-performance-on-strix-halo/245299';
const HALO_RDMA_URL = 'https://forum.level1techs.com/t/ryzen-ai-halo-usb4-clustering-with-rdma-testing-notes/253117';
const TB5_BRIEF_URL =
  'https://www.thunderbolttechnology.net/sites/default/files/Thunderbolt_5_TechBrief_2023_09_12.pdf';

/** PCIe line-rate ceiling after 128b/130b encoding, before packet overhead. Gen4/5 rate is GT/s per lane. */
function pcieLink(generation: 4 | 5, lanes: 8 | 16, direct: boolean): LinkSpec {
  const transfersPerSec = (generation === 5 ? 32 : 16) * 1e9;
  // B = transfers/s × lanes × coding efficiency / bits per byte.
  const bandwidthPerDirectionBytesPerSec = transfersPerSec * lanes * (128 / 130) / 8;
  const sources: SourceRef[] = [
    {
      url: PCIE_SIG_URL,
      title: 'PCI-SIG PCI Express 5.0 FAQ',
      fields: [BANDWIDTH_FIELD],
      note: 'Gen5 32 GT/s; x32 raw 128 GB/s each direction. Gen4 is half Gen5. ' +
        'Derived ceiling = GT/s × lanes × 128/130 ÷ 8, excluding TLP/DLLP overhead.',
    },
    {
      url: GEFORCE_P2P_URL,
      title: 'RTX 5090 P2P bandwidth and latency matrices',
      kind: 'community',
      fields: direct ? ['p2p'] : ['p2p', 'hostBounce.extraLatencySec'],
      note: 'Patched Gen5 x16 P2P write 0.49–0.59 µs; disabled 12.56–12.74 µs. ' +
        'Host-bounce extra latency rounded from 12.6 − 0.5 ≈ 12 µs. Measured host-bounce one-way ' +
        '31.6–32.4 GB/s is below the single-transfer formula prediction (min(63,96/2)=48 GB/s); ' +
        'for a two-device collective this engine additionally divides host bandwidth by both active devices. ' +
        'Driver staging throughput is not a hardware ceiling and is not fitted here.',
    }
  ];
  return {
    kind: 'pcie',
    name: `PCIe ${generation}.0 x${lanes} (${direct ? 'direct P2P' : 'host bounce'})`,
    bandwidthPerDirectionBytesPerSec,
    latencySec: 1e-6,
    p2p: direct ? 'direct' : 'host_bounce',
    ...(!direct ? {
      hostBounce: {
        hostMemBandwidthBytesPerSec: 6000e6 * 2 * 8, // DDR5-6000 × 2 channels × 8 B = 96 GB/s.
        extraLatencySec: 12e-6,
      }
    } : {}),
    sources,
    estimated: [
      {
        field: 'latencySec',
        reason: 'Rounded 1 µs PCIe DMA proxy from 0.5 µs Gen5 P2P writes; extrapolated to Gen4 and x8. ' +
          'Not a measured latency for every host/switch topology.',
      },
      ...(!direct ? [
        {
          field: 'hostBounce.hostMemBandwidthBytesPerSec',
          reason: 'Assumed desktop dual-channel DDR5-6000: 6000e6 transfers/s × 2 channels × 8 bytes = 96e9 B/s. ' +
            'Shared read+write peak, not a measured platform; replace for the actual host.',
        }
      ] : []),
    ],
  };
}

/** Stock GeForce PCIe Gen4 x16, host-staged; approximately 31.51 GB/s per PCIe hop per direction. */
export const PCIE_4_X16 = pcieLink(4, 16, false);
/** Stock GeForce PCIe Gen4 x8, host-staged; approximately 15.75 GB/s per PCIe hop per direction. */
export const PCIE_4_X8 = pcieLink(4, 8, false);
/** Stock GeForce PCIe Gen5 x16, host-staged; approximately 63.02 GB/s per PCIe hop per direction. */
export const PCIE_5_X16 = pcieLink(5, 16, false);
/** Stock GeForce PCIe Gen5 x8, host-staged; approximately 31.51 GB/s per PCIe hop per direction. */
export const PCIE_5_X8 = pcieLink(5, 8, false);
/** PCIe Gen4 x16 direct P2P, only for compatible devices/drivers and host routing. Unit: B/s per direction. */
export const PCIE_4_X16_DIRECT = pcieLink(4, 16, true);
/** PCIe Gen4 x8 direct P2P, only for compatible devices/drivers and host routing. Unit: B/s per direction. */
export const PCIE_4_X8_DIRECT = pcieLink(4, 8, true);
/** PCIe Gen5 x16 direct P2P, only for compatible devices/drivers and host routing. Unit: B/s per direction. */
export const PCIE_5_X16_DIRECT = pcieLink(5, 16, true);
/** PCIe Gen5 x8 direct P2P, e.g. Intel B60 on xe with P2P DMA support; routing/ACS must permit it. */
export const PCIE_5_X8_DIRECT: LinkSpec = {
  ...pcieLink(5, 8, true),
  sources: [
    ...pcieLink(5, 8, true).sources,
    {
      url: 'https://github.com/torvalds/linux/commit/2df55d9e66a2',
      title: 'drm/xe: Support pcie p2p dma as a fast interconnect',
      fields: ['p2p'],
      note: 'Stock xe driver support, not a GeForce unlock; requires compatible kernel and host topology.',
    }
  ],
};

/** RTX 3090 two-card NVLink bridge: 112.5 GB/s bidirectional = 56.25 GB/s per direction. */
export const RTX_3090_NVLINK: LinkSpec = {
  kind: 'nvlink_bridge',
  name: 'RTX 3090 NVLink bridge',
  bandwidthPerDirectionBytesPerSec: 56.25e9,
  latencySec: 1e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/amp' +
        'ere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf',
      title: 'NVIDIA Ampere GA102 architecture whitepaper',
      fields: [BANDWIDTH_FIELD],
      note: 'Four x4 links, each 14.0625 GB/s in each direction: 4 × 14.0625 = 56.25 GB/s, not 112.5 per direction.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: 'No bridge latency published; 1 µs proxy transferred from PCIe P2P.'
    }
  ],
};

/** H100/H200 NVLink4 switched injection ceiling: 900 GB/s bidirectional / 2 = 450 GB/s per direction. */
export const NVLINK_4: LinkSpec = {
  kind: 'nvlink',
  name: 'NVLink4 / NVSwitch',
  bandwidthPerDirectionBytesPerSec: 18 * 25e9,
  latencySec: 0.8e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/',
      title: 'NVIDIA Hopper architecture in depth',
      fields: [BANDWIDTH_FIELD],
      note: '18 links × 25 GB/s each direction = 450 GB/s each direction; total 900 GB/s.',
    },
    {
      url: GB200_LATENCY_URL,
      title: 'Every µs Matters: GB200 remote-store measurements',
      kind: 'review',
      fields: ['latencySec'],
      note: 'GB200 remote store 0.792 µs, rounded to 0.8 µs; not H100-specific.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: 'NVLink4 latency unpublished; use NVLink5 remote-store 0.8 µs proxy. ' +
        'H100 NCCL floor ~23 µs includes software and is intentionally not used.'
    }
  ],
};

/** B200/B300 NVLink5 switched injection ceiling: 1.8 TB/s bidirectional / 2 = 900 GB/s per direction. */
export const NVLINK_5: LinkSpec = {
  kind: 'nvlink',
  name: 'NVLink5 / NVSwitch',
  bandwidthPerDirectionBytesPerSec: 18 * 50e9,
  latencySec: 0.8e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf',
      title: 'NVIDIA Blackwell architecture technical overview',
      fields: [BANDWIDTH_FIELD],
      note: '18 fifth-generation links provide 1.8 TB/s total, 900 GB/s in each direction.',
    },
    {
      url: GB200_LATENCY_URL,
      title: 'Every µs Matters: GB200 remote-store measurements',
      kind: 'review',
      fields: ['latencySec'],
      note: 'Remote-store latency 0.792 µs rounded to 0.8 µs including NVSwitch. ' +
        'NCCL ring floor 11 µs and LL-kernel 2.37 µs are software references, not link latency.',
    }
  ],
  estimated: [],
};

/** ConnectX-7 NDR400: one 400 Gb/s NIC per GPU, 50 GB/s injection per direction. */
export const IB_NDR_400: LinkSpec = {
  kind: 'infiniband',
  name: 'InfiniBand NDR400 (one NIC per GPU)',
  bandwidthPerDirectionBytesPerSec: 400e9 / 8,
  latencySec: 1e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://www.nvidia.com/en-us/data-center/dgx-b200/',
      title: 'DGX B200 specifications',
      fields: [BANDWIDTH_FIELD],
      note: '8 ConnectX-7 adapters, up to 400 Gb/s each; 400e9 / 8 = 50e9 B/s per GPU.',
    },
    {
      url: 'https://hoti.org/assets/slides/2025_08_22_day3_Tutorial_Track1_HPNetworks.pdf',
      title: 'Hot Interconnects 2025 high-performance networks tutorial, slide 124',
      kind: 'review',
      fields: ['latencySec'],
      note: 'ib_write_lat 0.94 µs rounded to 1 µs. Adapter unspecified on this slide.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: '1 µs verbs-floor proxy; tutorial does not identify the adapter on slide 124.'
    }
  ],
};

/** ConnectX-8 XDR800: one 800 Gb/s NIC per GPU, 100 GB/s injection per direction. */
export const IB_XDR_800: LinkSpec = {
  kind: 'infiniband',
  name: 'InfiniBand XDR800 (one NIC per GPU)',
  bandwidthPerDirectionBytesPerSec: 800e9 / 8,
  latencySec: 1e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://networking-docs.nvidia.com/connectx8hw/specifications',
      title: 'ConnectX-8 hardware specifications',
      fields: [BANDWIDTH_FIELD],
      note: 'XDR: 4 lanes × 200 Gb/s = 800 Gb/s, or 100 GB/s per direction.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: 'No XDR latency found; assume the NDR 1 µs verbs floor.'
    }
  ],
};

/** DGX Spark pair, one ConnectX-7 200GbE cable using both 100G rails: 25 GB/s per direction. */
export const SPARK_CONNECTX_7: LinkSpec = {
  kind: 'roce',
  name: 'DGX Spark ConnectX-7 200GbE RoCE',
  bandwidthPerDirectionBytesPerSec: 200e9 / 8,
  latencySec: 1.5e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://www.nvidia.com/en-us/products/workstations/dgx-spark/',
      title: 'NVIDIA DGX Spark specifications',
      fields: [BANDWIDTH_FIELD],
      note: 'NVIDIA says both "each port up to 200 Gb/s" and "NIC @ 200 Gbps". ' +
        'Use the conservative total: 200 Gb/s = 25 GB/s per direction, not 400 Gb/s across two ports. ' +
        'Adding a second cable does not double the modeled bandwidth.',
    },
    {
      url: 'https://contact.alessandrosangiorgi.net/posts/dgx-spark-roce-link-validation/',
      title: 'DGX Spark RoCE link validation',
      kind: 'community',
      fields: ['latencySec'],
      note: 'ib_write_lat minimum run 1.45 µs rounded to 1.5 µs; another run 2.01 µs. ' +
        'NCCL small-message 24–43 µs includes software overhead and is not used.',
    }
  ],
  estimated: [],
};

/** MI300X per-peer xGMI link: 128 GB/s bidirectional / 2 = 64 GB/s per direction. */
export const XGMI_MI300X_PER_PEER: LinkSpec = {
  kind: 'xgmi',
  name: 'MI300X xGMI (one peer)',
  bandwidthPerDirectionBytesPerSec: 64e9,
  latencySec: 1.5e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html',
      title: 'AMD MI300X RCCL xGMI bandwidth',
      fields: [BANDWIDTH_FIELD],
      note: 'Raw bandwidth up to 64 GB/s for each point-to-point link; seven peers per GPU in the 8-GPU mesh.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: 'No xGMI latency published; 1.5 µs assumes NVLink-like remote store with margin.'
    }
  ],
};

/**
 * Eight-GPU MI300X full-mesh injection ceiling: seven 64 GB/s outgoing peer links = 448 GB/s per GPU.
 * The engine uses B as aggregate injection in ring/all-to-all formulas, so use the aggregate rather than
 * one peer for the full-node theoretical ceiling (assumes striped/multi-ring traffic). It has no peer-count
 * cap: subgroups and send/recv on this topology are optimistic; use XGMI_MI300X_PER_PEER for a pair.
 */
export const XGMI_MI300X_8_GPU: LinkSpec = {
  ...XGMI_MI300X_PER_PEER,
  name: 'MI300X xGMI (8-GPU aggregate injection)',
  bandwidthPerDirectionBytesPerSec: 7 * 64e9,
  sources: XGMI_MI300X_PER_PEER.sources.map((s) => ({
    ...s,
    note: `${s.note} Full mesh injection = 7 × 64 GB/s = 448 GB/s per direction; requires all seven peers.`,
  })),
};

/** Mac TB5 RDMA: one port at 80 Gb/s each direction, not the asymmetric display-only 120 Gb/s mode. */
export const TB5_RDMA: LinkSpec = {
  kind: 'thunderbolt',
  name: 'Thunderbolt 5 RDMA (macOS 26.2+)',
  bandwidthPerDirectionBytesPerSec: 80e9 / 8,
  latencySec: 50e-6,
  p2p: 'direct',
  sources: [
    {
      url: TB5_BRIEF_URL,
      title: 'Intel Thunderbolt 5 technology brief',
      fields: [BANDWIDTH_FIELD],
      note: '80 Gb/s each direction = 10 GB/s. 120 Gb/s Bandwidth Boost is asymmetric and display-oriented.',
    },
    {
      url: 'https://developer.apple.com/documentation/technotes/tn3' +
        '205-low-latency-communication-with-rdma-over-thunderbolt',
      title: 'Apple TN3205: RDMA over Thunderbolt',
      fields: ['p2p'],
      note: 'Apple silicon Macs with TB5, starting macOS 26.2; point-to-point RDMA interfaces.',
    },
    {
      url: 'https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/',
      title: 'Mac Studio RDMA over Thunderbolt 5',
      kind: 'community',
      fields: ['latencySec'],
      note: 'Reported memory-access latency <50 µs, an upper bound, not a hardware/verbs floor. ' +
        'Application throughput ~50–60 Gb/s; other file-transfer tests 3.5–3.8 GB/s.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: 'Hardware floor unknown; 50 µs is a conservative proxy from a reported <50 µs upper bound, not a ' +
        'verified one-way floor.'
    }
  ],
};

/** Mac TB4 TCP bridge: nominal 40 Gb/s per-direction ceiling; real bridge traffic is substantially lower. */
export const TB4_TCP: LinkSpec = {
  kind: 'thunderbolt',
  name: 'Thunderbolt 4 bridge (TCP)',
  bandwidthPerDirectionBytesPerSec: 40e9 / 8,
  latencySec: 70e-6,
  p2p: 'direct',
  sources: [
    {
      url: TB5_BRIEF_URL,
      title: 'Intel Thunderbolt technology comparison',
      fields: [BANDWIDTH_FIELD],
      note: 'Thunderbolt 4 nominal 40 Gb/s = 5 GB/s; theoretical single-port ceiling, not achieved TCP goodput.',
    },
    {
      url: 'https://sebastianalexis.com/blog/thunderbolt-networking-is-underused/',
      title: 'Thunderbolt networking is underused',
      kind: 'community',
      fields: ['latencySec'],
      note: 'TB bridge ping minimum RTT 0.142 ms; one-way estimate 0.142 / 2 ≈ 0.070 ms. ' +
        'Measured TCP one-way 15.9–18.8 Gb/s, below the nominal rate.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: '70 µs approximates half the 142 µs minimum ICMP RTT, not a measured one-way hardware latency.'
    }
  ],
};

/** Strix Halo USB4 native RDMA: AMD host-to-host negotiates one 20 Gb/s lane, i.e. 2.5 GB/s each way. */
export const USB4_HALO_RDMA: LinkSpec = {
  kind: 'usb4',
  name: 'Strix Halo USB4 native RDMA (20 Gb/s negotiated)',
  bandwidthPerDirectionBytesPerSec: 20e9 / 8,
  latencySec: 8e-6,
  p2p: 'direct',
  sources: [
    {
      url: HALO_RDMA_URL,
      title: 'Strix Halo USB4 clustering RDMA testing notes',
      kind: 'community',
      fields: [BANDWIDTH_FIELD, 'latencySec'],
      note: 'AMD host-to-host single lane: rx_speed/tx_speed 20 Gb/s, not 40 Gb/s each way. ' +
        'Native usb4_rdma ib_write_lat avg 8.33 µs, min 7.39 µs, rounded to 8 µs. Requires experimental driver.',
    }
  ],
  estimated: [],
};

/** Strix Halo stock thunderbolt-net USB4 TCP: same 20 Gb/s negotiated line rate, estimated 10 µs one-way. */
export const USB4_HALO_TCP: LinkSpec = {
  ...USB4_HALO_RDMA,
  name: 'Strix Halo USB4 TCP (20 Gb/s negotiated)',
  latencySec: 10e-6,
  sources: [
    ...USB4_HALO_RDMA.sources.map((s) => ({
      ...s,
      fields: [BANDWIDTH_FIELD],
    })),
    {
      url: HALO_TCP_URL,
      title: 'Strix Halo USB4 TCP ping-pong benchmark',
      kind: 'community',
      fields: ['latencySec'],
      note: 'Tuned 1-byte TCP ping-pong RTT 17.4–20.3 µs: approximately 10 µs one-way.',
    }
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: 'Half the tuned TCP ping-pong RTT; software-inclusive proxy because hardware-only floor is unknown.'
    }
  ],
};

/** Ethernet TCP ceiling in B/s, with the Strix Halo 5GbE one-way latency proxy. */
function ethernetTcp(gigabitsPerSec: 5 | 10): LinkSpec {
  return {
    kind: 'ethernet',
    name: `${gigabitsPerSec}GbE TCP`,
    bandwidthPerDirectionBytesPerSec: gigabitsPerSec * 1e9 / 8,
    latencySec: 10e-6,
    p2p: 'direct',
    sources: [
      {
        url: gigabitsPerSec === 5 ? HALO_TCP_URL : HALO_RDMA_URL,
        title: 'Strix Halo Ethernet NIC configurations and tests',
        kind: 'community',
        fields: [BANDWIDTH_FIELD],
        note: `${gigabitsPerSec}GbE configured NIC rate ÷ 8 = ${gigabitsPerSec / 8} GB/s per direction; ` +
          'theoretical line-rate ceiling, not TCP goodput. The 10GbE RTL812X test reached 9.42 Gb/s.',
      },
      {
        url: HALO_TCP_URL,
        title: 'Strix Halo Ethernet TCP ping-pong benchmark',
        kind: 'community',
        fields: ['latencySec'],
        note: '5GbE direct RTT 20.8–22.8 µs, minimum 19.9 µs. Switched 10GbE RTT 31.8–40.3 µs is not this floor.',
      }
    ],
    estimated: [
      {
        field: 'latencySec',
        reason: '10 µs = half the ~20 µs 5GbE direct TCP RTT; extrapolated to 10GbE, not a switched-network ' +
          'measurement.'
      }
    ],
  };
}

/** 5GbE TCP, 0.625 GB/s per direction; assumes a non-oversubscribed network. */
export const ETHERNET_5G_TCP = ethernetTcp(5);
/** 10GbE TCP, 1.25 GB/s per direction; assumes a non-oversubscribed network. */
export const ETHERNET_10G_TCP = ethernetTcp(10);

/** All stock host-bounced GeForce PCIe lane configurations; unlock modifiers can switch each to direct P2P. */
export const HOST_BOUNCE_PCIE_LINKS: LinkSpec[] = [PCIE_4_X16, PCIE_4_X8, PCIE_5_X16, PCIE_5_X8];
/** All link presets, including per-peer and full-node xGMI conventions. Every entry must validate. */
export const LINK_PRESETS: LinkSpec[] = [
  ...HOST_BOUNCE_PCIE_LINKS, PCIE_4_X16_DIRECT, PCIE_4_X8_DIRECT, PCIE_5_X16_DIRECT, PCIE_5_X8_DIRECT,
  RTX_3090_NVLINK, NVLINK_4, NVLINK_5, IB_NDR_400, IB_XDR_800, SPARK_CONNECTX_7,
  XGMI_MI300X_PER_PEER, XGMI_MI300X_8_GPU, TB5_RDMA, TB4_TCP,
  USB4_HALO_RDMA, USB4_HALO_TCP, ETHERNET_5G_TCP, ETHERNET_10G_TCP,
];
