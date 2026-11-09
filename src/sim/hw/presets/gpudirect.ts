/**
 * Citations for `DeviceSpec.gpuDirectRdma`: whether an RDMA NIC may DMA straight into a dedicated-memory device
 * (NVIDIA GPUDirect RDMA, AMD PeerDirect). `false` makes every inter-node message of the device go through host
 * memory (`engine/staging.ts`, docs/simulation-model.md §6 "Inter-node host staging"). Unified-memory devices leave
 * the field absent. The vendor preset files attach these records to their devices.
 */
import type { EstimatedField, SourceRef } from '../types';

/** GeForce: no GPUDirect RDMA (NVIDIA's own forum answers). */
export const GEFORCE_NO_GPUDIRECT_SOURCES: SourceRef[] = [
  {
    url: 'https://forums.developer.nvidia.com/t/is-gpu-direct-rdma-supported-on-newly-released-rtx-30-series/154509/5',
    title: 'NVIDIA forum: GPUDirect RDMA on GeForce RTX 30 series',
    kind: 'community',
    fields: ['gpuDirectRdma'],
    note: 'NVIDIA moderator: "RDMA is not supported on geforce. RDMA is supported on Quadro and Tesla".',
  },
  {
    url: 'https://forums.developer.nvidia.com/t/gpudirect-storage-requirement-questions/157984/2',
    title: 'NVIDIA forum: GPUDirect Storage requirements',
    kind: 'community',
    fields: ['gpuDirectRdma'],
    note:
      '"The GeForce line doesn\'t currently support GPUDirect RDMA". The tinygrad P2P patch unlocks GPU-to-GPU ' +
      'P2P only, not NIC RDMA.',
  },
];

/** NVIDIA data-center GPUs: GPUDirect RDMA. */
export const NVIDIA_DC_GPUDIRECT_SOURCE: SourceRef = {
  url: 'https://docs.nvidia.com/cuda/gpudirect-rdma/index.html',
  title: 'NVIDIA CUDA: GPUDirect RDMA',
  fields: ['gpuDirectRdma'],
  note:
    '"GPUDirect RDMA is available on both Tesla and Quadro GPUs" (data-center parts); the GDRCopy README also ' +
    'requires "NVIDIA Data Center GPU or NVIDIA RTX GPU". HGX boards place one NIC per GPU under a shared PCIe switch, ' +
    'which NCCL needs by default (PXB, src/graph/paths.cc:511-536).',
};

/** AMD Instinct: PeerDirect (the AMD counterpart of GPUDirect RDMA). */
export const AMD_PEERDIRECT_SOURCE: SourceRef = {
  url: 'https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html',
  title: 'ROCm: GPU-enabled MPI (PeerDirect)',
  fields: ['gpuDirectRdma'],
  note:
    '"The AMD kernel driver exposes remote direct memory access (RDMA) through PeerDirect interfaces. This allows ' +
    'network interface cards (NICs) to directly read and write to RDMA-capable GPU device memory".',
};

/** Consumer Radeon and Intel Arc cards: no published NIC-RDMA path into device memory. */
export const NO_RDMA_SOURCE_ESTIMATE: EstimatedField = {
  field: 'gpuDirectRdma',
  reason:
    "ESTIMATED false: no vendor or community source documents NIC RDMA into this card's memory. Moot on the " +
    'TCP links these cards ship with in the catalog, which stage every message through host memory anyway.',
};
