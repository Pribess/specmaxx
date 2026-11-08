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
const SPARK_CLUSTERING_URL = 'https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html';
const SPARK_RING_FORUM_URL =
  'https://forums.developer.nvidia.com/t/dgx-spark-networking-open-source-switchless-roce-collectives-nccl-profiles-and-fabric-tooling-for-multi-spark-inference/385049/1';
const TN3205_URL = 'https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt';
const MLX_JACCL_PR_URL = 'https://github.com/ml-explore/mlx/pull/2808';
const MLX_MULTI_CABLE_PR_URL = 'https://github.com/ml-explore/mlx/pull/3900';

/**
 * Old `linkPresetId`s → the current id, for URLs written before a link was renamed. Empty so far: no link preset has
 * been renamed (a link's id is the slug of its name, and the UI's URL stores it, so renames are avoided). The URL
 * decoder (`src/ui/state/url.ts`) consults it.
 */
export const LEGACY_LINK_IDS: Record<string, string> = {};

/** Per-lane signalling rate by PCIe generation (PCI-SIG: Gen3 8, Gen4 16, Gen5 32 GT/s). Unit: GT/s. */
const PCIE_GT_PER_LANE: Record<3 | 4 | 5, number> = { 3: 8, 4: 16, 5: 32 };

/**
 * PCIe line-rate ceiling per direction after 128b/130b encoding (Gen3, Gen4 and Gen5 all use it), before TLP/DLLP
 * packet overhead. Unit: B/s per direction.
 *
 * Formula: B = GT/s per lane × 1e9 × lanes × 128/130 / 8 bits per byte.
 * E.g. Gen5 x16 = 32e9 × 16 × 128/130 / 8 = 63.015e9 B/s; Gen4 x16 = 31.508e9; Gen3 x16 = 15.754e9.
 */
export function pcieBandwidthPerDirectionBytesPerSec(generation: 3 | 4 | 5, lanes: 4 | 8 | 16): number {
  const transfersPerSec = PCIE_GT_PER_LANE[generation] * 1e9;
  return transfersPerSec * lanes * (128 / 130) / 8;
}

/** PCIe line-rate ceiling after 128b/130b encoding, before packet overhead. Gen4/5 rate is GT/s per lane. */
function pcieLink(generation: 4 | 5, lanes: 8 | 16, direct: boolean): LinkSpec {
  // B = transfers/s × lanes × coding efficiency / bits per byte.
  const bandwidthPerDirectionBytesPerSec = pcieBandwidthPerDirectionBytesPerSec(generation, lanes);
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

/** nvbandwidth issue: achieved host↔device copy-engine rates (not presets; the `linkBandwidth` modifier can derate). */
const NVBANDWIDTH_ACHIEVED_URL = 'https://github.com/NVIDIA/nvbandwidth/issues/53';

/**
 * A device's PCIe link to host memory (`HostSpec.link`): the path offloaded weights take from host RAM (or from an
 * NVMe SSD staged through host RAM) into device memory. Per device, per direction; `p2p: 'direct'` because the
 * device DMAs to and from host memory itself (no staging hop of its own). Used by the engine when
 * `HardwareStack.host.link` is absent and the device has a cited `DeviceSpec.hostInterface`.
 *
 * Bandwidth: {@link pcieBandwidthPerDirectionBytesPerSec} (theoretical line rate; Gen5 x16 = 63.015e9 B/s, the same
 * number as `PCIE_5_X16`). Latency α = 1 µs, estimated (the same DMA proxy as the device-to-device PCIe links).
 * See docs/simulation-model.md §4 ("Host and storage tiers").
 */
export function hostPcieLink(generation: 3 | 4 | 5, lanes: 4 | 8 | 16): LinkSpec {
  const bandwidthPerDirectionBytesPerSec = pcieBandwidthPerDirectionBytesPerSec(generation, lanes);
  return {
    kind: 'pcie',
    name: `PCIe ${generation}.0 x${lanes} (host link)`,
    bandwidthPerDirectionBytesPerSec,
    latencySec: 1e-6,
    p2p: 'direct',
    sources: [
      {
        url: PCIE_SIG_URL,
        title: 'PCI-SIG PCI Express 5.0 FAQ',
        fields: [BANDWIDTH_FIELD],
        note:
          'Gen5 32 GT/s per lane (x32 raw 128 GB/s each direction); Gen4 16 GT/s and Gen3 8 GT/s are each half the next ' +
          'generation, all three with 128b/130b encoding. Derived ceiling = GT/s × lanes × 128/130 ÷ 8, excluding ' +
          `TLP/DLLP overhead. Achieved host copies are lower (nvbandwidth memcpy_ce ≈ 24.8 GB/s H2D on Gen4 x16, ${NVBANDWIDTH_ACHIEVED_URL}); ` +
          'they are not hardware ceilings and are not fitted here.',
      },
    ],
    estimated: [
      {
        field: 'latencySec',
        reason:
          'Rounded 1 µs PCIe DMA proxy from 0.5 µs Gen5 P2P writes (the device-to-device PCIe links use the same value); ' +
          'extrapolated to every generation and width. Not a measured host-memory read latency.',
      },
    ],
  };
}

/**
 * Every host link {@link hostPcieLink} builds, for a "Host link" picker: Gen3 / Gen4 / Gen5 × x4 / x8 / x16, newest
 * and widest first.
 */
export const HOST_PCIE_LINKS: LinkSpec[] = ([5, 4, 3] as const).flatMap((g) =>
  ([16, 8, 4] as const).map((lanes) => hostPcieLink(g, lanes)),
);

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

/**
 * DGX Spark ConnectX-7 200GbE RoCE, one QSFP port: 25 GB/s per direction. The NIC has two QSFP ports, but they
 * share the SoC attachment (two PCIe Gen5 x4 links, ~200 Gb/s together), so the all-ports cap equals one port: a
 * second cable adds no bandwidth, and bytes a Spark relays on a switchless ring count against the same 25 GB/s.
 */
export const SPARK_CONNECTX_7: LinkSpec = {
  kind: 'roce',
  name: 'DGX Spark ConnectX-7 200GbE RoCE',
  bandwidthPerDirectionBytesPerSec: 200e9 / 8,
  latencySec: 1.5e-6,
  allPortsCapPerDirectionBytesPerSec: 200e9 / 8,
  forwardingLatencySec: 4.5e-6,
  p2p: 'direct',
  sources: [
    {
      url: 'https://www.nvidia.com/en-us/products/workstations/dgx-spark/',
      title: 'NVIDIA DGX Spark specifications',
      fields: [BANDWIDTH_FIELD],
      note: 'NVIDIA says both "each port up to 200 Gb/s" and "NIC @ 200 Gbps". ' +
        'One port = 200 Gb/s = 25 GB/s per direction; the all-ports cap below says why two ports do not add up.',
    },
    {
      url: SPARK_CLUSTERING_URL,
      title: 'NVIDIA DGX Spark clustering documentation',
      fields: ['allPortsCapPerDirectionBytesPerSec'],
      note: 'The ConnectX-7 connects to GB10 over "two independent PCIe Gen 5 x4 links"; both QSFP ports share them, ' +
        'so a node injects ~200 Gb/s in total (measured 185–190 Gb/s RoCE over both x4 links, ServeTheHome; ' +
        '"both links to neighbor node will aggregate to 200G", NVIDIA forum). Cap = 25 GB/s per direction = one port. ' +
        'NVIDIA\'s multi-Spark playbook: "Using two cables will not increase performance."',
    },
    {
      url: SPARK_RING_FORUM_URL,
      title: 'dgx-spark-networking: switchless RoCE collectives on a 4-Spark ring (measurements)',
      kind: 'community',
      fields: ['forwardingLatencySec'],
      note: '10 KiB one-shot all-reduce: 10.9 µs on 2 direct nodes vs 16.9 µs on a 4-node ring where the opposite rank ' +
        'is relayed by a neighbour\'s host: the 2-hop path costs ~6 µs more than the 1-hop one. RoCE cannot route through ' +
        'a Spark in hardware (RC QPs fail across L3 hops), so relays are host store-and-forward.',
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
  estimated: [
    {
      field: 'forwardingLatencySec',
      reason: 'ESTIMATED, relay handling only (the engine charges α = 1.5 µs for every hop crossed, so this excludes ' +
        'the second wire hop): 4.5 µs = (16.9 − 10.9 µs) measured 2-hop vs 1-hop one-shot all-reduce − 1.5 µs for the ' +
        'extra hop\'s wire latency; it is the relay host\'s completion poll, re-post and PCIe round trip. A 2-hop path ' +
        'then costs 2 × 1.5 + 4.5 = 7.5 µs (+6 µs over direct, the measurement), a 3-hop one 3 × 1.5 + 2 × 4.5 = 13.5 µs.',
    },
  ],
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

/**
 * Mac TB5 RDMA: one port at 80 Gb/s each direction, not the asymmetric display-only 120 Gb/s mode. No all-ports
 * cap: Apple Thunderbolt controllers have a single port each (TN3205), and cables between two Macs scale about
 * linearly to four (~36.5 GB/s, MLX PR #3900); the SoC-side ceiling for five or six concurrent ports is unmeasured.
 */
export const TB5_RDMA: LinkSpec = {
  kind: 'thunderbolt',
  name: 'Thunderbolt 5 RDMA (macOS 26.2+)',
  bandwidthPerDirectionBytesPerSec: 80e9 / 8,
  latencySec: 50e-6,
  forwardingLatencySec: 3e-6,
  p2p: 'direct',
  sources: [
    {
      url: TB5_BRIEF_URL,
      title: 'Intel Thunderbolt 5 technology brief',
      fields: [BANDWIDTH_FIELD],
      note: '80 Gb/s each direction = 10 GB/s. 120 Gb/s Bandwidth Boost is asymmetric and display-oriented.',
    },
    {
      url: TN3205_URL,
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
    },
    {
      url: TN3205_URL,
      title: 'Apple TN3205: RDMA over Thunderbolt cannot route; the application forwards',
      fields: ['forwardingLatencySec'],
      note: '"RDMA over Thunderbolt can\'t route data … the application is responsible for forwarding": a relay on a ' +
        'ring is a receive completion plus a re-posted two-sided send on the next port (no remote writes).',
    },
    {
      url: MLX_MULTI_CABLE_PR_URL,
      title: 'MLX PR #3900: multiple Thunderbolt cables between two Macs',
      kind: 'community',
      fields: ['forwardingLatencySec'],
      note: '1 → 4 cables between two Macs: ~9.25 → ~36.5 GB/s all_sum bandwidth (~3.95×), so ports add up and no ' +
        'all-ports cap is set. ibv_uc_pingpong round trip ~14 µs (~7 µs one way, the measured per-hop wire + verbs time) on the same stack.',
    },
    {
      url: MLX_JACCL_PR_URL,
      title: 'MLX PR #2808: JACCL (RDMA over TB5) 4-way all-reduce timings',
      kind: 'community',
      fields: ['forwardingLatencySec'],
      note: 'JACCL 4-way mesh all-reduce ~9–12 µs at 1–8 KB on M3 Ultra Mac Studios, vs ~290–300 µs for the TCP ring backend.',
    },
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: 'Hardware floor unknown; 50 µs is a conservative proxy from a reported <50 µs upper bound, not a ' +
        'verified one-way floor.'
    },
    {
      field: 'forwardingLatencySec',
      reason: 'ESTIMATED, relay handling only: the engine charges α for every hop crossed (the ~7 µs measured one-way ' +
        'transfer; 50 µs in this preset), so the relay adds just the host turnaround between the incoming completion and ' +
        'the outgoing post of a two-sided send: ~3 µs, the difference between JACCL\'s 9–12 µs small 4-way all-reduce and ' +
        'its ~7 µs one-way transfer, the software step of a user-space verbs relay. Physically a 2-hop relay is ' +
        '2 × 7 + 3 = 17 µs and a 3-hop one 3 × 7 + 2 × 3 = 27 µs; with this preset\'s 50 µs α, 103 and 156 µs.',
    },
  ],
};

/** The Thunderbolt bridge forwards in software: TN3205 (the source of the TCP links' forwarding latency). */
const TB_BRIDGE_FORWARDING_SOURCE: SourceRef = {
  url: MLX_JACCL_PR_URL,
  title: 'MLX PR #2808: TCP ring backend 4-way all-reduce timings',
  kind: 'community',
  fields: ['forwardingLatencySec'],
  note: 'MLX `ring` (TCP over Thunderbolt) 4-way all-reduce ≈ 290–300 µs at 1–32 KB, i.e. ~48 µs per ring step, each ' +
    'step a kernel TCP receive plus send through the host: the per-hop cost is almost all software. TN3205: the ' +
    'Thunderbolt bridge forwards in software, "like a physical Ethernet hub".',
};
/** Why a TCP relay adds ~20 µs on top of the per-hop α of the Thunderbolt bridge. */
const TB_TCP_FORWARDING_REASON =
  'ESTIMATED, relay handling only: α (70 µs, half the ICMP RTT) already holds each hop\'s send stack, wire and receive ' +
  'stack, so the relay adds the user-space turnaround between the receive completing and the next send being issued ' +
  '(wakeup, copy, syscall): ~20 µs, about a third of the ~48–70 µs one-way TCP time, which is itself almost all ' +
  'software; no direct measurement of a Thunderbolt bridge relay exists. A 2-hop relay is 2 × 70 + 20 = 160 µs, a ' +
  '3-hop one 3 × 70 + 2 × 20 = 250 µs.';

/**
 * Mac Thunderbolt bridge over TCP (IP over Thunderbolt) on Thunderbolt 4 cables and ports: nominal 40 Gb/s per
 * direction; real bridge traffic is substantially lower. The Thunderbolt link of Macs without Thunderbolt 5 (the M3
 * Max MacBook Pro); TB5 Macs use `TB5_TCP`. Older URLs naming this link on a TB5 Mac still resolve to it.
 */
export const TB4_TCP: LinkSpec = {
  kind: 'thunderbolt',
  name: 'Thunderbolt 4 bridge (TCP)',
  bandwidthPerDirectionBytesPerSec: 40e9 / 8,
  latencySec: 70e-6,
  forwardingLatencySec: 20e-6,
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
    },
    TB_BRIDGE_FORWARDING_SOURCE,
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: '70 µs approximates half the 142 µs minimum ICMP RTT, not a measured one-way hardware latency.'
    },
    {
      field: 'forwardingLatencySec',
      reason: TB_TCP_FORWARDING_REASON,
    },
  ],
};

/**
 * Mac Thunderbolt bridge over TCP on Thunderbolt 5 ports and cables: the 80 Gb/s per-direction line rate (the same
 * hardware ceiling as `TB5_RDMA`; the 120 Gb/s Bandwidth Boost is asymmetric and display-oriented). Achieved TCP
 * goodput over the bridge is lower and not cleanly measured, so this is the hardware ceiling, not a goodput claim.
 * Latency and relay handling are the bridge's software costs, transferred from the TB4 bridge (same macOS stack).
 */
export const TB5_TCP: LinkSpec = {
  kind: 'thunderbolt',
  name: 'Thunderbolt 5 bridge (TCP)',
  bandwidthPerDirectionBytesPerSec: 80e9 / 8,
  latencySec: 70e-6,
  forwardingLatencySec: 20e-6,
  p2p: 'direct',
  sources: [
    {
      url: TB5_BRIEF_URL,
      title: 'Intel Thunderbolt 5 technology brief',
      fields: [BANDWIDTH_FIELD],
      note: '80 Gb/s each direction = 10 GB/s, the line rate the bridge can at most carry; 120 Gb/s Bandwidth Boost is ' +
        'asymmetric and display-oriented. Not an achieved TCP goodput (none cleanly measured on TB5).',
    },
    {
      url: 'https://sebastianalexis.com/blog/thunderbolt-networking-is-underused/',
      title: 'Thunderbolt networking is underused',
      kind: 'community',
      fields: ['latencySec'],
      note: 'Thunderbolt bridge ping minimum RTT 0.142 ms (TB4 Macs); one-way estimate ≈ 70 µs. The bridge is the same ' +
        'macOS software path on TB5.',
    },
    TB_BRIDGE_FORWARDING_SOURCE,
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: '70 µs transferred from the TB4 bridge (half the 142 µs minimum ICMP RTT): the bridge\'s latency is the ' +
        'software stack, not the wire; no TB5-specific bridge RTT found.',
    },
    {
      field: 'forwardingLatencySec',
      reason: TB_TCP_FORWARDING_REASON,
    },
  ],
};

/**
 * Strix Halo USB4 native RDMA: AMD host-to-host negotiates one 20 Gb/s lane, i.e. 2.5 GB/s each way. Two native
 * USB4 ports per SoC (AMD product page), one controller each: no all-ports cap. A relay on a 4-node ring goes through
 * the host (no USB4 routing): ~3 µs of handling on top of each hop's α.
 */
export const USB4_HALO_RDMA: LinkSpec = {
  kind: 'usb4',
  name: 'Strix Halo USB4 native RDMA (20 Gb/s negotiated)',
  bandwidthPerDirectionBytesPerSec: 20e9 / 8,
  latencySec: 8e-6,
  forwardingLatencySec: 3e-6,
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
  estimated: [
    {
      field: 'forwardingLatencySec',
      reason: 'ESTIMATED, relay handling only: α (8 µs, ib_write_lat) is charged for every hop crossed, so the relay adds ' +
        'just the host turnaround between the incoming completion and the re-posted send, ~3 µs as for Thunderbolt 5 RDMA ' +
        '(a user-space verbs relay; the Linux usb4_rdma path is experimental and unmeasured for relays). A 2-hop relay is ' +
        '2 × 8 + 3 = 19 µs, a 3-hop one 3 × 8 + 2 × 3 = 30 µs.',
    },
  ],
};

/**
 * Strix Halo stock thunderbolt-net USB4 TCP: same 20 Gb/s negotiated line rate, estimated 10 µs one-way, and an
 * estimated 5 µs of relay handling per intermediate node (Linux IP forwarding through the host on a 4-node ring).
 */
export const USB4_HALO_TCP: LinkSpec = {
  ...USB4_HALO_RDMA,
  name: 'Strix Halo USB4 TCP (20 Gb/s negotiated)',
  latencySec: 10e-6,
  forwardingLatencySec: 5e-6,
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
    },
    {
      field: 'forwardingLatencySec',
      reason: 'ESTIMATED, relay handling only: α (10 µs, half the tuned ping-pong RTT) already holds each hop\'s send ' +
        'stack, wire and receive stack, so the relay adds the kernel\'s IP-forwarding turnaround (or a user-space ' +
        'receive-to-send on a tuned stack), ~5 µs, half the one-way time; no relay measurement exists. A 2-hop relay is ' +
        '2 × 10 + 5 = 25 µs, a 3-hop one 3 × 10 + 2 × 5 = 40 µs.',
    },
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
  XGMI_MI300X_PER_PEER, XGMI_MI300X_8_GPU, TB5_RDMA, TB5_TCP, TB4_TCP,
  USB4_HALO_RDMA, USB4_HALO_TCP, ETHERNET_5G_TCP, ETHERNET_10G_TCP,
];
