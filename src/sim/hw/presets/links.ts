/**
 * Interconnect ceilings: SI B/s per device per direction, never bidirectional sums.
 * Latencies are one-way hardware/verbs proxies, not NCCL/RCCL launch floors. Unpublished or transferred
 * latency assumptions are explicitly estimated. Inter-node α has ONE definition for every transport (RDMA and TCP
 * alike): the one-way small-message floor as a collective library sees it = half the average (or median) round trip
 * of a tuned small-message ping-pong over that transport. An ICMP ping or the minimum sample is never the α; the
 * per-step time of a small all-reduce is only an upper bound. On one physical medium RDMA ≤ TCP (`LINK_MEDIA`).
 * See docs/simulation-model.md §4 ("Links and topologies"), §6 ("Collectives"), and engine/collectives.ts.
 */
import type { LinkSpec, SourceRef } from '../types';

const BANDWIDTH_FIELD = 'bandwidthPerDirectionBytesPerSec';
const PCIE_SIG_URL = 'https://pcisig.com/faq?field_category_value%5B%5D=pci_express_5.0&keys=';
const GEFORCE_P2P_URL = 'https://github.com/tinygrad/open-gpu-kernel-modules/issues/29';
const GB200_LATENCY_URL = 'https://arxiv.org/pdf/2607.16100';
/** Level1Techs: tuned TCP ping-pong over Strix Halo USB4 and its built-in 5GbE (direct and switched). */
export const HALO_TCP_URL = 'https://forum.level1techs.com/t/benchmarking-usb4-performance-on-strix-halo/245299';
const HALO_RDMA_URL = 'https://forum.level1techs.com/t/ryzen-ai-halo-usb4-clustering-with-rdma-testing-notes/253117';
const TB5_BRIEF_URL =
  'https://www.thunderbolttechnology.net/sites/default/files/Thunderbolt_5_TechBrief_2023_09_12.pdf';
const SPARK_CLUSTERING_URL = 'https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html';
const SPARK_RING_FORUM_URL =
  'https://forums.developer.nvidia.com/t/dgx-spark-networking-open-source-switchless-roce-collectives-nccl-profiles-and-fabric-tooling-for-multi-spark-inference/385049/1';
const TN3205_URL = 'https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt';
/** Anemll on X: `ibv_uc_pingpong` over Thunderbolt 5 RDMA, M4 Pro ↔ M3 Ultra (screenshot of the run). */
const TB5_RDMA_PINGPONG_URL = 'https://x.com/anemll/status/1993182652204187929';
/** Hellas: ibverbs over Thunderbolt / USB4 on Linux, `ib_write_lat` at 64 B. */
const HELLAS_IBVERBS_URL = 'https://blog.hellas.ai/blog/thunderbolt-ibverbs/';
/** MLX PR #2808: JACCL (RDMA over Thunderbolt 5) vs the TCP ring, 4-Mac all-reduce timings. */
export const MLX_JACCL_PR_URL = 'https://github.com/ml-explore/mlx/pull/2808';
const MLX_MULTI_CABLE_PR_URL = 'https://github.com/ml-explore/mlx/pull/3900';
/** NCCL source pinned at commit 3e46c089f4ae (2026-10-08), for the host-staging citations. */
const NCCL_SRC_URL = 'https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src';
const GDRCOPY_URL = 'https://github.com/NVIDIA/gdrcopy';
const CUDA_GPUDIRECT_RDMA_URL = 'https://docs.nvidia.com/cuda/gpudirect-rdma/index.html';

/**
 * c: one pipeline step of a host-staged inter-node message, the unit in which a GPU copies into (and out of) the
 * pinned host buffer while the NIC moves the previous step. NCCL's per-channel network FIFO is `NCCL_BUFFSIZE` =
 * 4 MiB split into `NCCL_STEPS` = 8 slots: 4 MiB / 8 = 512 KiB. Unit: bytes.
 * Sources: NCCL `src/init.cc:1006` (DEFAULT_BUFFSIZE 4 MiB; env docs
 * https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) and `src/include/device.h:26` (NCCL_STEPS 8),
 * commit 3e46c089f4ae. See docs/simulation-model.md §6 ("Inter-node host staging") and `engine/staging.ts`.
 */
export const HOST_STAGING_STEP_BYTES = (4 * 1024 * 1024) / 8;

/**
 * c_sock: the step in which a TCP sender's kernel copies a message from the user buffer into socket buffers while
 * the NIC sends the previous step (and the receiver's `recv()` copies the previous step out). Linux builds TCP
 * segments of at most `GSO_LEGACY_MAX_SIZE` = 65536 bytes for the NIC (TSO / GRO coalesce to 64 KiB), so a send is
 * copied and transmitted in 64 KiB pieces. Unit: bytes.
 * Source: Linux `include/linux/netdevice.h:2333` (v6.12,
 * https://github.com/torvalds/linux/blob/v6.12/include/linux/netdevice.h#L2333). ESTIMATED for macOS (XNU's
 * socket buffer chunking is not documented); the result is insensitive to it: the copy fill is at most
 * 2·c_sock / r_sock ≈ 0.3–1 µs per message on the presets. Used only when no PCIe staging step applies (unified
 * memory over TCP). See docs/simulation-model.md §6 ("TCP socket copy") and `engine/staging.ts`.
 */
export const SOCKET_COPY_STEP_BYTES = 64 * 1024;

/**
 * Why TCP links have no GPUDirect path, and why they are `transport: 'tcp'`: socket transports take host pointers
 * only, so a GPU with its own memory stages every message through pinned host memory (`gpuDirectRdma: false`), and
 * every TCP message is copied by the kernel at both ends (`send()` / `recv()`, no `MSG_ZEROCOPY`).
 */
const TCP_HOST_POINTERS_SOURCE: SourceRef = {
  url: `${NCCL_SRC_URL}/transport/net_socket.cc#L57`,
  title: 'NCCL socket transport: ptrSupport = NCCL_PTR_HOST',
  fields: ['gpuDirectRdma', 'transport'],
  note: 'The socket plugin reports host-pointer support only, so NCCL places the network buffers in pinned host ' +
    'memory (net.cc:936, :1121, ncclCudaHostCalloc). Its socket progress calls plain send() / recv() on that buffer ' +
    '(MSG_DONTWAIT | MSG_NOSIGNAL, no MSG_ZEROCOPY: src/os/linux.cc:351-355), so the kernel copies every message in ' +
    'and out. Gloo\'s CUDA all-reduce likewise copies to a pinned host workspace (https://github.com/pytorch/gloo, ' +
    'gloo/cuda_allreduce_ring.cc:82,102); MLX\'s ring is host sockets.',
};

/** InfiniBand to a data-center GPU: the NIC DMAs into device memory (GPUDirect RDMA / AMD PeerDirect). */
const IB_SOURCE: SourceRef = {
  url: CUDA_GPUDIRECT_RDMA_URL,
  title: 'NVIDIA CUDA: GPUDirect RDMA',
  fields: ['gpuDirectRdma', 'transport'],
  note: '"GPUDirect RDMA is available on both Tesla and Quadro GPUs"; AMD Instinct via PeerDirect (ROCm GPU-enabled ' +
    'MPI, https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html). Caveat: by default NCCL uses it only when ' +
    'the GPU and NIC sit under the same PCIe switch (PXB; src/graph/paths.cc:511-536), which HGX/DGX boards meet ' +
    '(DGX B200: 8 ConnectX-7, one per GPU).',
};

/** DGX Spark: no GPUDirect RDMA; irrelevant here because the GB10 memory is unified (nothing to stage over PCIe). */
const SPARK_SOURCE: SourceRef = {
  url: 'https://forums.developer.nvidia.com/t/enabling-gpu-direct-rdma-for-dgx-spark-clustering/352051/2',
  title: 'NVIDIA forum: GPUDirect RDMA on DGX Spark',
  fields: ['gpuDirectRdma', 'transport'],
  note: 'NVIDIA: "the GPUDirect RDMA technology is not supported" on Spark; buffers go through cudaHostAlloc + ' +
    'ib_reg_mr (verbs, RoCE: zero-copy from host memory). No staging in the engine: the Spark is unified memory.',
};

/** RDMA over Thunderbolt 5 (macOS) and USB4 (Strix Halo): host/unified memory only, SoCs only. */
const SOC_RDMA_SOURCE: SourceRef = {
  url: TN3205_URL,
  title: 'Apple TN3205: RDMA over Thunderbolt (host memory regions)',
  fields: ['gpuDirectRdma', 'transport'],
  note: 'RDMA into host (unified) memory only; no discrete-GPU memory path exists on these links (also the Strix ' +
    `Halo usb4_rdma notes, ${HALO_RDMA_URL}). No effect in the engine: only unified-memory SoCs use them.`,
};

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
 * number as `PCIE_5_X16`). Latency α = 1 µs, estimated (the same DMA proxy as the device-to-device PCIe links). The
 * inter-node host staging (`engine/staging.ts`) charges this α once per end of a staged message (α_stage = 2·α).
 * See docs/simulation-model.md §4 ("Host and storage tiers") and §6 ("Inter-node host staging").
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
      {
        url: GDRCOPY_URL,
        title: 'NVIDIA GDRCopy README: gdrcopy_pplat and the cudaMemcpy overhead',
        fields: ['latencySec'],
        note: 'CPU ↔ GPU ping-pong through mapped memory on an A40: "Round-trip latency per iteration is 1.08762 us", ' +
          'i.e. one PCIe crossing (a write plus a flag poll) is about 0.5–1 µs. Contrast, the copy path Gloo takes: ' +
          '"currently a cudaMemcpy can incur in a 6-7us overhead".',
      },
      {
        url: 'https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/neugebauer2018understanding.pdf',
        title: 'Neugebauer et al., "Understanding PCIe performance for end host networking", SIGCOMM 2018',
        kind: 'review',
        fields: ['latencySec'],
        note: 'pcie-bench: 64 B DMA read from host memory min 520 ns, median 547 ns (Xeon E5, Gen3); a NIC\'s 128 B ' +
          'round trip ≈ 1000 ns, ≈ 900 ns of it PCIe.',
      },
    ],
    estimated: [
      {
        field: 'latencySec',
        reason:
          'Rounded 1 µs PCIe DMA proxy from 0.5 µs Gen5 P2P writes (the device-to-device PCIe links use the same value); ' +
          'extrapolated to every generation and width. Not a measured host-memory read latency. Consistent with a ' +
          'mapped-memory crossing (GDRCopy round trip 1.09 µs, pcie-bench 64 B DMA read 547 ns median); used as the ' +
          'per-end floor of inter-node host staging (no cudaMemcpy, no stream sync: NCCL\'s network transport without ' +
          'GPUDirect has the GPU write and poll pinned host buffers, net.cc:936, :1121).',
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
  gpuDirectRdma: true,
  transport: 'rdma',
  sources: [
    IB_SOURCE,
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
  gpuDirectRdma: true,
  transport: 'rdma',
  sources: [
    IB_SOURCE,
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
  gpuDirectRdma: false,
  transport: 'rdma',
  sources: [
    SPARK_SOURCE,
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
      note: 'ib_write_lat 2 B, t_typical (median) 1.45 µs, rounded to 1.5 µs; another run 2.01 µs. ' +
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
  latencySec: 7e-6,
  forwardingLatencySec: 3e-6,
  p2p: 'direct',
  gpuDirectRdma: false,
  transport: 'rdma',
  sources: [
    SOC_RDMA_SOURCE,
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
      url: TB5_RDMA_PINGPONG_URL,
      title: 'Anemll: ibv_uc_pingpong over Thunderbolt 5 RDMA, M4 Pro ↔ M3 Ultra',
      kind: 'community',
      fields: ['latencySec'],
      note: '"1000 iters in 0.01 seconds = 13.72 usec/iter": one iteration is one round trip, so 6.86 µs one way, ' +
        'rounded to 7 µs; an average over 1000 round trips, as the α definition takes ' +
        '(screenshot https://pbs.twimg.com/media/G6k0RngagAAIBuy.jpg?name=orig).',
    },
    {
      url: 'https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/',
      title: 'Mac Studio RDMA over Thunderbolt 5',
      kind: 'community',
      fields: ['latencySec'],
      note: 'Context only, not the α: "lowering memory access latency from 300us down to <50us" reads off the ' +
        'all-reduce plot (JACCL vs the TCP ring), a collective upper bound, not a one-way link floor. ' +
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
        'all-ports cap is set.',
    },
    {
      url: MLX_JACCL_PR_URL,
      title: 'MLX PR #2808: JACCL (RDMA over TB5) 4-way all-reduce timings',
      kind: 'community',
      fields: ['latencySec', 'forwardingLatencySec'],
      note: 'JACCL 4-way mesh all-reduce ~9–10 µs at 1–2 KB (~9–12 µs up to 8 KB) on M3 Ultra Mac Studios, vs ' +
        '~290–300 µs for the TCP ring backend. A small all-reduce is one step of the mesh algorithm ' +
        '(mlx/distributed/jaccl/lib/jaccl/mesh_impl.h:416), so ~9.5 µs is an upper bound on α, consistent with 7 µs.',
    },
  ],
  estimated: [
    {
      field: 'forwardingLatencySec',
      reason: 'ESTIMATED, relay handling only: the engine charges α (7 µs, the measured one-way ping-pong) for every hop ' +
        'crossed, so the relay adds just the host turnaround between the incoming completion and the outgoing post of a ' +
        'two-sided send: ~3 µs, the difference between JACCL\'s ~9.5 µs small all-reduce step and the 7 µs one-way ' +
        'transfer, the software step of a user-space verbs relay. A 2-hop relay is 2 × 7 + 3 = 17 µs and a 3-hop one ' +
        '3 × 7 + 2 × 3 = 27 µs.',
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
/** The latency source of the Thunderbolt bridge's TCP α: MLX's TCP ring over Thunderbolt, Mac to Mac. */
const TB_TCP_LATENCY_SOURCE: SourceRef = {
  url: MLX_JACCL_PR_URL,
  title: 'MLX PR #2808: TCP ring backend 4-way all-reduce timings (M3 Ultra Mac Studios over Thunderbolt)',
  kind: 'community',
  fields: ['latencySec'],
  note: 'MLX `ring` 4-way all-reduce ≈ 290–300 µs at 1–32 KB: 6 ring steps (2·(N − 1)) of ≈ 48 µs each, reduction ' +
    'and dispatch included, so one bridge hop is at most ≈ 48 µs one way. JACCL (RDMA over TB5) on the same Macs: 9–12 µs.',
};
/** Why the Thunderbolt bridge's TCP α is 45 µs (estimated). */
const TB_TCP_LATENCY_REASON =
  '45 µs, ESTIMATED: no tuned TCP ping-pong over a Thunderbolt bridge was found, so the floor sits just below the ' +
  'measured upper bound, MLX\'s TCP ring over Thunderbolt (a 4-Mac all-reduce in 290–300 µs, flat over 1–32 KB = ' +
  '2(N − 1) = 6 ring steps of ≈ 48 µs, each including thread hand-offs and the reduction; MLX PR 2808). The ICMP ' +
  'bridge ping (0.142 ms RTT, MacBook Air ↔ Linux PC, untuned) is not a floor: on Strix Halo USB4 untuned ICMP ' +
  '(0.119 ms minimum) overstated the tuned TCP ping-pong (14.1 µs minimum RTT) about 8×. RDMA over the same TB5 ' +
  'cable measures 7 µs one way (TB5_RDMA), so TCP stays the slower transport, as it must (LINK_MEDIA).';
/** Why a TCP relay adds ~20 µs on top of the per-hop α of the Thunderbolt bridge. */
const TB_TCP_FORWARDING_REASON =
  'ESTIMATED, relay handling only: α (45 µs, below MLX\'s ≈ 48 µs TCP ring step) already holds each hop\'s send ' +
  'stack, wire and receive stack, so the relay adds the user-space turnaround between the receive completing and the ' +
  'next send being issued (wakeup, copy, syscall): ~20 µs, under half the ≈ 45–48 µs one-way TCP time, which is ' +
  'itself almost all software; no direct measurement of a Thunderbolt bridge relay exists. A 2-hop relay is ' +
  '2 × 45 + 20 = 110 µs, a 3-hop one 3 × 45 + 2 × 20 = 175 µs.';

/**
 * Mac Thunderbolt bridge over TCP (IP over Thunderbolt) on Thunderbolt 4 cables and ports: nominal 40 Gb/s per
 * direction; real bridge traffic is substantially lower. The Thunderbolt link of Macs without Thunderbolt 5 (the M3
 * Max MacBook Pro); TB5 Macs use `TB5_TCP`. Older URLs naming this link on a TB5 Mac still resolve to it.
 */
export const TB4_TCP: LinkSpec = {
  kind: 'thunderbolt',
  name: 'Thunderbolt 4 bridge (TCP)',
  bandwidthPerDirectionBytesPerSec: 40e9 / 8,
  latencySec: 45e-6,
  forwardingLatencySec: 20e-6,
  p2p: 'direct',
  gpuDirectRdma: false,
  transport: 'tcp',
  sources: [
    TCP_HOST_POINTERS_SOURCE,
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
      fields: [BANDWIDTH_FIELD],
      note: 'Measured TCP one-way 15.9–18.8 Gb/s, below the nominal rate. Context for the latency: TB bridge ping ' +
        'minimum RTT 0.142 ms (MacBook Air ↔ Linux PC, untuned ICMP), not used as the floor.',
    },
    TB_TCP_LATENCY_SOURCE,
    TB_BRIDGE_FORWARDING_SOURCE,
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: TB_TCP_LATENCY_REASON,
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
 * Latency and relay handling are the bridge's software costs, the same as the TB4 bridge (same macOS stack).
 */
export const TB5_TCP: LinkSpec = {
  kind: 'thunderbolt',
  name: 'Thunderbolt 5 bridge (TCP)',
  bandwidthPerDirectionBytesPerSec: 80e9 / 8,
  latencySec: 45e-6,
  forwardingLatencySec: 20e-6,
  p2p: 'direct',
  gpuDirectRdma: false,
  transport: 'tcp',
  sources: [
    TCP_HOST_POINTERS_SOURCE,
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
      note: 'Context only, not the floor: Thunderbolt bridge ping minimum RTT 0.142 ms (MacBook Air ↔ Linux PC, ' +
        'untuned ICMP). The bridge is the same macOS software path on TB5; the α rests on MLX PR 2808 below.',
    },
    TB_TCP_LATENCY_SOURCE,
    TB_BRIDGE_FORWARDING_SOURCE,
  ],
  estimated: [
    {
      field: 'latencySec',
      reason: TB_TCP_LATENCY_REASON + ' The TB5 bridge runs the same macOS software path as TB4 (MLX\'s measurement is ' +
        'M3 Ultra Mac Studios, TB5 hardware).',
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
  gpuDirectRdma: false,
  transport: 'rdma',
  sources: [
    SOC_RDMA_SOURCE,
    {
      url: HALO_RDMA_URL,
      title: 'Strix Halo USB4 clustering RDMA testing notes',
      kind: 'community',
      fields: [BANDWIDTH_FIELD, 'latencySec'],
      note: 'AMD host-to-host single lane: rx_speed/tx_speed 20 Gb/s, not 40 Gb/s each way. ' +
        'Native usb4_rdma "ib_write_lat 2B (n=10000) ... avg 8.33us (min 7.39, 99% 9.98)" (post 22): α = the average, ' +
        'rounded to 8 µs. Requires experimental driver.',
    },
    {
      url: HELLAS_IBVERBS_URL,
      title: 'Hellas: ibverbs over Thunderbolt / USB4',
      kind: 'community',
      fields: ['latencySec'],
      note: '~7 µs one-way ib_write_lat at 64 B, a cross-check. Caveat: these verbs are emulated over the same USB4 DMA ' +
        'rings as thunderbolt-net TCP, so RDMA ≈ TCP here is plausible (8 vs 9 µs on averages).',
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
 * Strix Halo stock thunderbolt-net USB4 TCP: same 20 Gb/s negotiated line rate, 9 µs one-way (half the tuned
 * 1-byte TCP ping-pong average), and an estimated 5 µs of relay handling per intermediate node (Linux IP forwarding
 * through the host on a 4-node ring).
 */
export const USB4_HALO_TCP: LinkSpec = {
  ...USB4_HALO_RDMA,
  name: 'Strix Halo USB4 TCP (20 Gb/s negotiated)',
  latencySec: 9e-6,
  forwardingLatencySec: 5e-6,
  gpuDirectRdma: false,
  transport: 'tcp',
  sources: [
    TCP_HOST_POINTERS_SOURCE,
    ...USB4_HALO_RDMA.sources.filter((s) => s.url === HALO_RDMA_URL).map((s) => ({
      ...s,
      fields: [BANDWIDTH_FIELD],
    })),
    {
      url: HALO_TCP_URL,
      title: 'Strix Halo USB4 TCP ping-pong benchmark',
      kind: 'community',
      fields: ['latencySec'],
      note: 'Tuned 1-byte TCP ping-pong (Python), post 3: RTT average Base 20.25 / Partial 17.35 / Full 19.56 µs ' +
        '(minimum 15.15 / 14.13 / 17.41 µs). α = half the best-tuned average, 17.35 / 2 = 8.7 µs, rounded to 9 µs. ' +
        'RDMA ≤ TCP holds on averages (8.33 vs 8.7 µs); on minima it would invert (7.39 vs 7.07 µs), which is why α ' +
        'never uses the minimum.',
    }
  ],
  estimated: [
    {
      field: 'forwardingLatencySec',
      reason: 'ESTIMATED, relay handling only: α (9 µs, half the tuned ping-pong RTT) already holds each hop\'s send ' +
        'stack, wire and receive stack, so the relay adds the kernel\'s IP-forwarding turnaround (or a user-space ' +
        'receive-to-send on a tuned stack), ~5 µs, about half the one-way time; no relay measurement exists. A 2-hop ' +
        'relay is 2 × 9 + 5 = 23 µs, a 3-hop one 3 × 9 + 2 × 5 = 37 µs.',
    },
  ],
};

/** Cross-checks of the 10GbE TCP floor (no direct 10GbE desktop measurement exists). */
const TEN_GBE_LATENCY_CROSS_CHECK: SourceRef = {
  url: 'https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-belay.pdf',
  title: 'Belay et al., "IX: A Protected Dataplane Operating System", OSDI 2014',
  kind: 'review',
  fields: ['latencySec'],
  note: 'Linux 10GbE one-way 24 µs (NetPIPE 64 B, epoll, Intel 82599 through a Broadcom Trident+ switch) vs 5.7 µs ' +
    'kernel-bypass; "3µs latency across a pair of 10 GbE NICs". 10GBASE-T PHY latency is 2–4 µs ' +
    '(https://en.wikipedia.org/wiki/10_Gigabit_Ethernet); a tuned UDP echo through a switch averages 30.7 µs RTT ' +
    '(https://blog.cloudflare.com/how-to-achieve-low-latency/). Upper cross-checks for the estimated 10 µs direct floor.',
};
/** Ethernet TCP ceiling in B/s, with the Strix Halo 5GbE one-way latency proxy. */
function ethernetTcp(gigabitsPerSec: 5 | 10): LinkSpec {
  return {
    kind: 'ethernet',
    name: `${gigabitsPerSec}GbE TCP`,
    bandwidthPerDirectionBytesPerSec: gigabitsPerSec * 1e9 / 8,
    latencySec: 10e-6,
    p2p: 'direct',
    gpuDirectRdma: false,
    transport: 'tcp',
    sources: [
      TCP_HOST_POINTERS_SOURCE,
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
        note: '5GbE direct RTT average 20.8–22.8 µs, minimum 19.9 µs (tuned 1-byte ping-pong, Framework Desktop ' +
          'built-in 5GbE). Through a 10G switch (Unifi Pro XG 8) the same port measured RTT average 31.8–40.3 µs ' +
          '(minimum 29.4–32.3 µs): a switch adds ≈ 5 µs one way. α is the direct-cable figure.',
      },
      ...(gigabitsPerSec === 10 ? [TEN_GBE_LATENCY_CROSS_CHECK] : []),
    ],
    estimated: [
      {
        field: 'latencySec',
        reason: gigabitsPerSec === 5
          ? '10 µs ≈ half the 20.8–22.8 µs average 5GbE direct TCP RTT (direct cable; a switch adds ≈ 5 µs one way).'
          : '10 µs carried over from the 5GbE direct measurement (half its ~20 µs tuned TCP RTT): no 10GbE desktop ' +
            'ping-pong was found. The floor is the kernel stack plus PHY, not line rate (a 1-byte frame serializes in < 0.1 ' +
            'µs); 10GBASE-T PHYs add 2–4 µs, like NBASE-T. Cross-checks: ≈ 15 µs switched (5GbE through a 10G switch); ' +
            'IX (OSDI\'14) Linux 10GbE one-way 24 µs through a switch with epoll, untuned.'
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

/**
 * Link presets on one physical medium (the same ports, cable and controller / NIC), split by transport. RDMA is
 * zero-copy and kernel-bypass: the NIC moves the registered buffer itself, while TCP adds the kernel's socket stack
 * (syscalls, copies, protocol processing) on top of the same wire. So on one medium RDMA's one-way floor `latencySec`
 * can never exceed TCP's, and neither can its relay handling `forwardingLatencySec` (a user-space verbs re-post vs a
 * kernel receive and send). `links.test.ts` checks both for every pair, and that every preset with a `transport` is
 * listed here exactly once, so a new preset joins the check. Data only: the engine never reads it.
 */
export interface LinkMedium {
  /** The physical medium, e.g. `'thunderbolt-5'` (informational, unique). */
  medium: string;
  /** Presets over this medium with `transport: 'rdma'`. */
  rdma: LinkSpec[];
  /** Presets over this medium with `transport: 'tcp'`. */
  tcp: LinkSpec[];
}

/** Every inter-node preset by physical medium (see {@link LinkMedium}): the RDMA ≤ TCP latency invariant's pairs. */
export const LINK_MEDIA: LinkMedium[] = [
  { medium: 'thunderbolt-5', rdma: [TB5_RDMA], tcp: [TB5_TCP] },
  { medium: 'thunderbolt-4', rdma: [], tcp: [TB4_TCP] },
  { medium: 'strix-halo-usb4', rdma: [USB4_HALO_RDMA], tcp: [USB4_HALO_TCP] },
  { medium: 'connectx-7-dgx-spark', rdma: [SPARK_CONNECTX_7], tcp: [] },
  { medium: 'infiniband-ndr', rdma: [IB_NDR_400], tcp: [] },
  { medium: 'infiniband-xdr', rdma: [IB_XDR_800], tcp: [] },
  { medium: '5gbe', rdma: [], tcp: [ETHERNET_5G_TCP] },
  { medium: '10gbe', rdma: [], tcp: [ETHERNET_10G_TCP] },
];
