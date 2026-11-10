/** Vendor references for rack-scale devices, links and pooled host memory (docs/simulation-model.md §4 "Scale-up domains"). */

/** NVIDIA Blackwell datasheet OCT25, per-GPU GB200 NVL72 / NVL4 table, p. 8. */
export const BLACKWELL_DATASHEET_URL =
  'https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf';
/** NVIDIA GB200 NVL72 specifications: 72 GPUs, 36 Grace CPUs, 17 TB CPU memory. */
export const GB200_NVL72_URL = 'https://www.nvidia.com/en-us/data-center/gb200-nvl72/';
/** NVIDIA GB300 NVL72 specifications: 72 GPUs, 36 Grace CPUs, 17 TB CPU memory. */
export const GB300_NVL72_URL = 'https://www.nvidia.com/en-us/data-center/gb300-nvl72/';
/** NVIDIA Grace tuning guide: 480 GB modules supply 384 GB/s (512 GB/s applies to smaller modules). */
export const GRACE_TUNING_URL = 'https://docs.nvidia.com/grace-perf-tuning-guide/index.html';
/** NVIDIA Blackwell Ultra architecture, Table 1: NVLink-C2C 900 GB/s bidirectional in the per-GPU comparison. */
export const BLACKWELL_ULTRA_BLOG_URL =
  'https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/';
/** AMD MI325X datasheet: seven 128 GB/s bidirectional GPU-peer links. */
export const MI325X_LINK_DATASHEET_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf';
/** AMD MI355X platform brochure: seven 144 GB/s bidirectional GPU-peer links. */
export const MI355X_PLATFORM_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-miI355x-platform-brochure.pdf';
/** AMD CDNA4 whitepaper, text p. 13 and Table 2: 72 GB/s per direction per link, 1,008 GB/s bidirectional aggregate. */
export const CDNA4_LINK_WHITEPAPER_URL =
  'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf';
/** AMD MI455X product specifications: 3.6 TB/s bidirectional UALoE scale-up per GPU. */
export const MI455X_SCALE_UP_URL =
  'https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html';
/** AMD Helios solution brief: 72 GPUs on a switched single-hop UALoE scale-up fabric. */
export const HELIOS_URL =
  'https://www.amd.com/content/dam/amd/en/documents/products/accelerators/instinct/amd-instinct-helios-blueprint-brochure.pdf';
