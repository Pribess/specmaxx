/**
 * Storage-tier presets for weight offloading: host DRAM configurations (`HostMemorySpec`) and NVMe SSDs
 * (`NvmeSpec`). Datasheet ceilings, never achieved fractions: these are the tiers' theoretical limits, as the rest of
 * the library. Every number is covered by a `sources[].fields` path or an `estimated[]` entry.
 * See docs/simulation-model.md §4 ("Host and storage tiers") and design doc docs/design/offload-specdec-latency.md §2.2.
 *
 * Conventions:
 * - Host RAM bandwidth is the theoretical peak of the whole node, read + write combined: MT/s × 8 B × channels
 *   (one 64-bit DDR5 channel moves 8 B per transfer).
 * - Host RAM capacity is a configuration choice, not a vendor number: every capacity is marked estimated.
 * - SSD rates are the datasheet's "up to" numbers; random reads are 4 KiB (`readGranularityBytes`).
 * - SI decimal units throughout (1 GB = 1e9 B, as both Samsung and Sandisk define it in their footnotes).
 */

import type { EstimatedField, HostMemorySpec, NvmeSpec, SourceRef } from '../types';
import { gb, gbPerSec } from '../units';
import { pcieBandwidthPerDirectionBytesPerSec } from './links';

// ---------------------------------------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------------------------------------

/** AMD Ryzen 9 9950X spec page: 2 memory channels, "2x1R DDR5-5600" max memory speed. */
const RYZEN_9950X_URL =
  'https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html';
/** Chips and Cheese, Ryzen 9 9950X review (DDR5-6000 EXPO: "theoretical 96 GB/s", "just over 70 ns"). */
const CNC_9950X_URL = 'https://chipsandcheese.com/p/amds-ryzen-9950x-zen-5-on-desktop';
/** Intel ark, Core Ultra 9 285K (Arrow Lake): "Up to DDR5 6400 MT/s", 2 memory channels. */
const ARK_285K_URL =
  'https://www.intel.com/content/www/us/en/products/sku/241060/intel-core-ultra-9-processor-285k-36m-cache-up-to-5-70-ghz/specifications.html';
/** Chips and Cheese, Arrow Lake: "Over 100 ns of DRAM load-to-use latency". */
const CNC_ARROW_LAKE_URL = 'https://chipsandcheese.com/p/examining-intels-arrow-lake-at-the';
/** AMD Ryzen Threadripper PRO 9995WX spec page: 8 memory channels, "Up to 6400 MT/s". */
const TR_9995WX_URL =
  'https://www.amd.com/en/products/processors/workstations/ryzen-threadripper/9000-wx-series/amd-ryzen-threadripper-pro-9995wx.html';
/** AMD EPYC 9654 spec page: 12 channels, "Up to 4800 MT/s", "Per Socket Mem BW 460.8 GB/s". */
const EPYC_9654_URL =
  'https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9654.html';
/** AMD EPYC 9575F spec page: 12 channels, "Up to 6400 MT/s", "Per Socket Mem BW 614 GB/s". */
const EPYC_9575F_URL =
  'https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9575f.html';

/** JEDEC DDR5 SDRAM standard JESD79-5: 64-bit data bus per channel (2 × 32-bit subchannels), 4800 MT/s is the baseline bin. */
const JEDEC_DDR5_URL = 'https://www.jedec.org/standards-documents/docs/jesd79-5b';
/** Intel ark, Core i9-12900K (Alder Lake): "Up to DDR5 4800 MT/s", 2 memory channels. */
const ARK_12900K_URL =
  'https://www.intel.com/content/www/us/en/products/sku/134599/intel-core-i912900k-processor-30m-cache-up-to-5-20-ghz/specifications.html';
/** Intel ark, Xeon w5-2465X (Sapphire Rapids, W-2400 class): DDR5-4800, 4 memory channels. */
const ARK_W5_2465X_URL =
  'https://www.intel.com/content/www/us/en/products/sku/233415/intel-xeon-w52465x-processor-33-75m-cache-3-10-ghz/specifications.html';
/** AMD launch release for Threadripper 7000 / PRO 7000 WX: 4-channel (TRX50) and 8-channel (WRX90) DDR5-5200 platforms. */
const AMD_TR7000_URL =
  'https://www.globenewswire.com/news-release/2023/10/19/2763278/0/en/AMD-Introduces-New-AMD-Ryzen-Threadripper-7000-Series-Processors-and-Ryzen-Threadripper-PRO-7000-WX-Series-Processors-for-the-Ultimate-Workstation.html';
/** Tom's Hardware, Threadripper 7000 / PRO 7000 WX launch: eight-channel DDR5-4800 on Intel's Xeon w9-3475X. */
const TOMS_TR7000_URL =
  'https://www.tomshardware.com/news/amd-announces-threadripper-hedt-and-pro-7000-wx-series-processors-96-cores-and-192-threads-for-desktops-and-workstations';

/** JEDEC DDR4 SDRAM standard JESD79-4: 64-bit data bus per channel, transfer rates 1600 to 3200 MT/s (higher bins are vendor XMP/DOCP overclocks). */
const JEDEC_DDR4_URL = 'https://www.jedec.org/standards-documents/docs/jesd79-4a';
/** AMD Ryzen 9 5950X spec page (AM4, Zen 3): 2 memory channels, max memory speed DDR4-3200. */
const RYZEN_5950X_URL =
  'https://www.amd.com/en/products/processors/desktops/ryzen/5000-series/amd-ryzen-9-5950x.html';
/** AMD Ryzen Threadripper 3970X specifications (TRX40): 4 memory channels, up to 3200 MT/s. */
const TR_3970X_URL =
  'https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen-threadripper/ryzen-threadripper-3000-series/amd-ryzen-threadripper-3970x.html';
/** AMD EPYC 7763 spec page (Milan): 8 channels, DDR4-3200, "Per Socket Mem BW 204.8 GB/s". */
const EPYC_7763_URL =
  'https://www.amd.com/en/products/processors/server/epyc/7003-series/amd-epyc-7763.html';
/** Intel ark, Xeon Gold 6248 (Cascade Lake-SP): DDR4-2933, 6 memory channels. */
const ARK_GOLD_6248_URL =
  'https://www.intel.com/content/www/us/en/products/sku/192446/intel-xeon-gold-6248-processor-27-5m-cache-2-50-ghz/specifications.html';
/** Intel ark, Xeon E5-2680 v4 (Broadwell-EP): DDR4 up to 2400, 4 memory channels, "76.8 GB/s". */
const ARK_E5_2680_V4_URL =
  'https://www.intel.com/content/www/us/en/products/sku/91754/intel-xeon-processor-e52680-v4-35m-cache-2-40-ghz/specifications.html';

/** Samsung 990 PRO datasheet Rev 1.0 (1 TB / 2 TB). */
const SAMSUNG_990_PRO_URL =
  'https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_NVMe_SSD_990_PRO_Datasheet_Rev.1.0.pdf';
/** Sandisk WD_BLACK SN850X datasheet (July 2025). */
const SN850X_URL =
  'https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/product/internal-drives/wd-black-ssd/data-sheet-wd-black-sn850x-nvme-ssd.pdf';
/** Samsung 9100 PRO datasheet Rev 2.0 (1 / 2 / 4 / 8 TB). */
const SAMSUNG_9100_PRO_URL =
  'https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_NVMe_SSD_9100_PRO_Datasheet_Rev.2.0.pdf';
/** Crucial T705 2TB product page (spec table; footnote 2: "CrystalDiskMark® with a queue depth of 512"). */
const CRUCIAL_T705_URL = 'https://www.crucial.com/ssd/t705/ct2000t705ssd3';
/** Crucial T705 product flyer (Micron): "up to 14,500/12,700 MB/s". */
const CRUCIAL_T705_FLYER_URL =
  'https://assets.micron.com/adobe/assets/urn%3Aaaid%3Aaem%3A8b31ab31-5c1c-4dbe-ad26-5bdc076d5ff3/renditions/original/as/crucial-t705-b2c-product-flyer-en.pdf';
/** CDRLabs T705 2TB review, Iometer: 1,411,981 random-read IOPS at 16 threads × QD32. */
const CDRLABS_T705_URL =
  'https://www.cdrlabs.com/reviews/crucial-t705-2tb-pcie-gen5-nvme-solid-state-drive/performance-anvil-s-storage-utilities-and-iometer.html';
/** r/mac crowd-sourced AmorphousDiskMark medians per Mac SSD (community data). */
const MAC_SSD_REDDIT_URL =
  'https://www.reddit.com/r/mac/comments/1rpsc1n/mac_ssd_speeds_overview_updated_with_m5_promax/';
/** Alizadeh et al., "LLM in a flash" (arXiv 2312.11514) §2.2: M1 Max 1 TB "speeds exceeding 6 GiB/s" linear read. */
const LLM_IN_A_FLASH_URL = 'https://arxiv.org/pdf/2312.11514';
/** StorageReview DGX Spark review: internal Gen5 x4 2242 SSD, GDSIO sequential read peak 11.4 GiB/s. */
const STORAGEREVIEW_SPARK_URL =
  'https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops';
/** NVIDIA DGX Spark hardware overview: "1 TB or 4 TB NVMe M.2 with self-encryption". */
const DGX_SPARK_HW_URL = 'https://docs.nvidia.com/dgx/dgx-spark/hardware.html';
/** Framework Desktop specs: two NVMe PCIe 4.0 x4 M.2 2280 sockets. */
const FRAMEWORK_DESKTOP_URL = 'https://frame.work/desktop?tab=specs';

// ---------------------------------------------------------------------------------------------------------
// Shared estimates
// ---------------------------------------------------------------------------------------------------------

/** Host RAM: a random read moves one 64 B cache line (x86-64 and Arm server cores alike). */
const CACHE_LINE_ESTIMATE: EstimatedField = {
  field: 'readGranularityBytes',
  reason:
    'A DRAM random read moves one 64 B cache line on every current x86-64 and Arm server/desktop core.',
};

/** Host RAM: memory-level parallelism the host sustains for random reads. */
const HOST_MLP_ESTIMATE: EstimatedField = {
  field: 'maxOutstanding',
  reason:
    'Random host-RAM reads in flight (memory-level parallelism) across the cores issuing them; no vendor publishes ' +
    "it. 64 is the order of a desktop part's outstanding L2 misses summed over a few cores; it bounds only " +
    "latency-bound row gathers (Little's law), never large expert reads.",
};

/** NVMe: a read moves at least one 4 KiB LBA / NAND page. */
const PAGE_ESTIMATE: EstimatedField = {
  field: 'readGranularityBytes',
  reason:
    'The smallest read an NVMe SSD serves is one 4 KiB logical block (the 4K LBA format every datasheet rates its ' +
    'random reads at), so a 320 B row read costs 4096 B of drive bandwidth.',
};

/** A host-RAM capacity: a configuration choice, not a vendor number. */
function capacityEstimate(config: string): EstimatedField {
  return {
    field: 'capacityBytes',
    reason: `Configuration choice, not a vendor number: ${config}. Set it to the installed amount.`,
  };
}

// ---------------------------------------------------------------------------------------------------------
// Host RAM
// ---------------------------------------------------------------------------------------------------------

/** Bytes one DDR5 channel moves per transfer (64-bit data bus, ECC bits excluded). Unit: bytes. */
const DDR5_BYTES_PER_TRANSFER_PER_CHANNEL = 8;

/**
 * Theoretical DDR5 bandwidth of a node, read + write combined. Unit: B/s.
 *
 * Formula: B = MT/s × 1e6 × 8 B × channels. E.g. DDR5-6000 × 2 channels = 6000e6 × 8 × 2 = 96e9 B/s.
 */
export function ddr5BandwidthBytesPerSec(megaTransfersPerSec: number, channels: number): number {
  const transfersPerSec = megaTransfersPerSec * 1e6;
  return transfersPerSec * DDR5_BYTES_PER_TRANSFER_PER_CHANNEL * channels;
}

/** The varying parameters of one DDR5 host-RAM preset (everything else is shared). */
interface Ddr5Params {
  /** Data rate. Unit: MT/s. */
  megaTransfersPerSec: number;
  /** Memory channels. Count. */
  channels: number;
  /** Installed capacity (an estimate). Unit: bytes. */
  capacityBytes: number;
  /** Unloaded load-to-use latency. Unit: seconds. */
  latencySec: number;
  /** Display-name suffix naming the platform class. */
  platform: string;
  sources: SourceRef[];
  estimated: EstimatedField[];
}

/** A DDR5 host-RAM preset `ddr5-<MT/s>-<channels>ch`. */
function ddr5(p: Ddr5Params): HostMemorySpec {
  return {
    kind: 'host_ram',
    id: `ddr5-${p.megaTransfersPerSec}-${p.channels}ch`,
    name: `DDR5-${p.megaTransfersPerSec}, ${p.channels} channels (${p.platform})`,
    capacityBytes: p.capacityBytes,
    bandwidthBytesPerSec: ddr5BandwidthBytesPerSec(p.megaTransfersPerSec, p.channels),
    latencySec: p.latencySec,
    channels: p.channels,
    readGranularityBytes: 64,
    maxOutstanding: 64,
    sources: p.sources,
    estimated: [...p.estimated, CACHE_LINE_ESTIMATE, HOST_MLP_ESTIMATE],
  };
}

/** Desktop dual-channel DDR5-5600 (AMD's rated speed for 2 DIMMs on Ryzen 9000): 5600e6 × 8 × 2 = 89.6e9 B/s. */
export const DDR5_5600_2CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 5600,
  channels: 2,
  capacityBytes: gb(96),
  latencySec: 70e-9,
  platform: 'desktop',
  sources: [
    {
      url: RYZEN_9950X_URL,
      title: 'AMD Ryzen 9 9950X specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"Memory Channels 2"; "Max Memory Speed 2x1R DDR5-5600 2x2R DDR5-5600". Bandwidth = 5600e6 × 8 B × 2.',
    },
  ],
  estimated: [
    capacityEstimate('2 × 48 GB DIMMs'),
    {
      field: 'latencySec',
      reason:
        `Chips and Cheese measured "just over 70 ns" on the 9950X with DDR5-6000 (${CNC_9950X_URL}); at 5600 MT/s ` +
        'the latency is about the same (it is dominated by the controller and fabric, not the data rate).',
    },
  ],
});

/**
 * Desktop dual-channel DDR5-6000 (EXPO, the speed AMD's reviewer guide recommends): 6000e6 × 8 × 2 = 96e9 B/s. The
 * DEFAULT host RAM, and the same 96 GB/s the host-bounce PCIe links assume for staging.
 */
export const DDR5_6000_2CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 6000,
  channels: 2,
  capacityBytes: gb(96),
  latencySec: 70e-9,
  platform: 'desktop',
  sources: [
    {
      url: CNC_9950X_URL,
      title: "Chips and Cheese: AMD's Ryzen 9950X, Zen 5 on desktop",
      kind: 'review',
      fields: ['bandwidthBytesPerSec', 'latencySec'],
      note:
        'Tested at "6000 MT/s, since AMD\'s reviewer guide suggests that\'s the sweet spot"; "the Ryzen 9 9950X\'s ' +
        'theoretical 96 GB/s"; "With just over 70 ns of memory latency, the Ryzen 9 9950X with DDR5-6000".',
    },
    {
      url: RYZEN_9950X_URL,
      title: 'AMD Ryzen 9 9950X specifications',
      fields: ['channels'],
      note: '"Memory Channels 2". 6000 MT/s is an EXPO overclock above the rated DDR5-5600.',
    },
  ],
  estimated: [capacityEstimate('2 × 48 GB DIMMs')],
});

/** Desktop dual-channel DDR5-6400 (Intel Arrow Lake's rated speed): 6400e6 × 8 × 2 = 102.4e9 B/s. */
export const DDR5_6400_2CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 6400,
  channels: 2,
  capacityBytes: gb(96),
  latencySec: 100e-9,
  platform: 'desktop',
  sources: [
    {
      url: ARK_285K_URL,
      title: 'Intel ark: Intel Core Ultra 9 Processor 285K specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"Up to DDR5 6400 MT/s"; "Max # of Memory Channels 2". Bandwidth = 6400e6 × 8 B × 2.',
    },
  ],
  estimated: [
    capacityEstimate('2 × 48 GB DIMMs'),
    {
      field: 'latencySec',
      reason:
        `Chips and Cheese, Arrow Lake (${CNC_ARROW_LAKE_URL}): "Over 100 ns of DRAM load-to-use latency"; rounded ` +
        'to 100 ns (a floor of the measurement).',
    },
  ],
});

/** JEDEC baseline DDR5 data-rate source every 4800 MT/s preset shares. */
const JEDEC_DDR5_SOURCE: SourceRef = {
  url: JEDEC_DDR5_URL,
  title: 'JEDEC JESD79-5 DDR5 SDRAM standard',
  fields: ['bandwidthBytesPerSec'],
  note: 'Per-device standard: 64-bit data bus per channel ⇒ 8 B per transfer; bandwidth = MT/s × 1e6 × 8 B × channels. 4800 MT/s is the baseline bin.',
};

/** Workstation 8-channel DDR5-6400 (Threadripper PRO 9000 WX): 6400e6 × 8 × 8 = 409.6e9 B/s. */
export const DDR5_6400_8CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 6400,
  channels: 8,
  capacityBytes: gb(256),
  latencySec: 100e-9,
  platform: 'workstation',
  sources: [
    {
      url: TR_9995WX_URL,
      title: 'AMD Ryzen Threadripper PRO 9995WX specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"Memory Channels 8"; "System Memory Specification Up to 6400 MT/s". Bandwidth = 6400e6 × 8 B × 8.',
    },
  ],
  estimated: [
    capacityEstimate('8 × 32 GB RDIMMs'),
    {
      field: 'latencySec',
      reason:
        'No measurement of this platform found; RDIMMs and the larger IO die add latency over desktop DDR5 (70 ns), ' +
        'so 100 ns is assumed.',
    },
  ],
});

/** Server 12-channel DDR5-4800 (one EPYC 9004 socket): 4800e6 × 8 × 12 = 460.8e9 B/s, as AMD quotes it. */
export const DDR5_4800_12CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 4800,
  channels: 12,
  capacityBytes: gb(384),
  latencySec: 110e-9,
  platform: 'server, 1 socket',
  sources: [
    {
      url: EPYC_9654_URL,
      title: 'AMD EPYC 9654 specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"Memory Channels 12"; "Up to 4800 MT/s"; "Per Socket Mem BW 460.8 GB/s" (= 4800e6 × 8 B × 12).',
    },
  ],
  estimated: [
    capacityEstimate('12 × 32 GB RDIMMs, one socket'),
    {
      field: 'latencySec',
      reason:
        'No vendor latency; server DDR5 with RDIMMs and a large IO die is assumed 110 ns unloaded.',
    },
  ],
});

/** Server 12-channel DDR5-6000 (one EPYC 9005 socket below its 6400 maximum): 6000e6 × 8 × 12 = 576e9 B/s. */
export const DDR5_6000_12CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 6000,
  channels: 12,
  capacityBytes: gb(768),
  latencySec: 110e-9,
  platform: 'server, 1 socket',
  sources: [
    {
      url: EPYC_9575F_URL,
      title: 'AMD EPYC 9575F specifications',
      fields: ['channels'],
      note: '"Memory Channels 12"; "Up to 6400 MT/s"; "Per Socket Mem BW 614 GB/s" (= 6400e6 × 8 B × 12 = 614.4e9).',
    },
  ],
  estimated: [
    capacityEstimate('12 × 64 GB RDIMMs, one socket'),
    {
      field: 'bandwidthBytesPerSec',
      reason:
        `The EPYC 9575F page (${EPYC_9575F_URL}) rates 12 channels up to 6400 MT/s (614 GB/s); this preset assumes ` +
        'DIMMs at 6000 MT/s: 6000e6 × 8 B × 12 = 576e9 B/s.',
    },
    {
      field: 'latencySec',
      reason:
        'No vendor latency; server DDR5 with RDIMMs and a large IO die is assumed 110 ns unloaded.',
    },
  ],
});

/**
 * Workstation 8-channel DDR5-4800 (Threadripper PRO 7000 WX on WRX90, Xeon W-3400, Sapphire Rapids 1S): 4800e6 × 8 × 8 = 307.2e9 B/s.
 * Threadripper PRO 7000 WX is rated up to DDR5-5200; Xeon W-3400 rates DDR5-4800.
 */
export const DDR5_4800_8CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 4800,
  channels: 8,
  capacityBytes: gb(256),
  latencySec: 100e-9,
  platform: 'workstation',
  sources: [
    JEDEC_DDR5_SOURCE,
    {
      url: AMD_TR7000_URL,
      title: 'AMD Threadripper PRO 7000 WX launch release',
      fields: ['channels'],
      note: 'Eight-channel DDR5-5200 RDIMM (WRX90); this preset assumes DIMMs run at 4800 MT/s.',
    },
    {
      url: TOMS_TR7000_URL,
      title: "Tom's Hardware: Threadripper 7000 launch",
      fields: ['channels'],
      note: 'Eight-channel DDR5-4800 on Intel Xeon w9-3475X (W-3400).',
    },
  ],
  estimated: [
    capacityEstimate('8 × 32 GB RDIMMs'),
    {
      field: 'latencySec',
      reason:
        'No measurement of this platform found; RDIMMs and the larger memory fabric add latency over desktop DDR5 (70 ns), so 100 ns is assumed (as DDR5-6400 8-channel).',
    },
  ],
});

/**
 * HEDT quad-channel DDR5-4800 (Threadripper 7000 on TRX50, Xeon W-2400): 4800e6 × 8 × 4 = 153.6e9 B/s.
 * Threadripper 7000 is rated up to DDR5-5200; Xeon W-2400 rates DDR5-4800.
 */
export const DDR5_4800_4CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 4800,
  channels: 4,
  capacityBytes: gb(128),
  latencySec: 100e-9,
  platform: 'HEDT',
  sources: [
    JEDEC_DDR5_SOURCE,
    {
      url: ARK_W5_2465X_URL,
      title: 'Intel ark: Intel Xeon w5-2465X specifications',
      fields: ['channels'],
      note: '"Max # of Memory Channels 4"; DDR5-4800.',
    },
    {
      url: AMD_TR7000_URL,
      title: 'AMD Threadripper 7000 launch release',
      fields: ['channels'],
      note: 'Four-channel DDR5-5200 RDIMM (TRX50); this preset assumes DIMMs run at 4800 MT/s.',
    },
  ],
  estimated: [
    capacityEstimate('4 × 32 GB RDIMMs'),
    {
      field: 'latencySec',
      reason:
        'No measurement of this platform found; RDIMMs and the larger memory fabric add latency over desktop DDR5 (70 ns), so 100 ns is assumed (as DDR5-6400 8-channel).',
    },
  ],
});

/**
 * Desktop dual-channel DDR5-4800 (JEDEC baseline; Intel 12th gen and early AM5 stock speed): 4800e6 × 8 × 2 = 76.8e9 B/s.
 * 4-DIMM AM5 / LGA1700 setups often run at or below this speed, because two DIMMs per channel force a lower rate.
 */
export const DDR5_4800_2CH: HostMemorySpec = ddr5({
  megaTransfersPerSec: 4800,
  channels: 2,
  capacityBytes: gb(64),
  latencySec: 70e-9,
  platform: 'desktop',
  sources: [
    JEDEC_DDR5_SOURCE,
    {
      url: ARK_12900K_URL,
      title: 'Intel ark: Intel Core i9-12900K specifications',
      fields: ['channels'],
      note: '"Up to DDR5 4800 MT/s"; "Max # of Memory Channels 2".',
    },
  ],
  estimated: [
    capacityEstimate('2 × 32 GB DIMMs'),
    {
      field: 'latencySec',
      reason:
        'No measurement of this platform found; assumed the same as DDR5-6000 desktop (70 ns), since controller and fabric dominate, not the data rate.',
    },
  ],
});

// ---------------------------------------------------------------------------------------------------------
// Host RAM: DDR4
//
// Platform PCIe is NOT carried by a host-RAM preset (`HostMemorySpec` has no PCIe field; the GPU-host link is
// chosen separately, see links.ts). Pair these presets with a PCIe 3.0/4.0 link, not the PCIe 5.0 default:
// AM4 is PCIe 4.0 on X570/B550 and 3.0 on older boards; Threadripper 3000 and EPYC 7002/7003 are PCIe 4.0;
// Xeon Scalable 2nd gen and Xeon E5 v4 are PCIe 3.0.
// ---------------------------------------------------------------------------------------------------------

/** Bytes one DDR4 channel moves per transfer (64-bit data bus, ECC bits excluded). Unit: bytes. */
const DDR4_BYTES_PER_TRANSFER_PER_CHANNEL = 8;

/**
 * Theoretical DDR4 bandwidth of a node, read + write combined. Unit: B/s.
 *
 * Formula: B = MT/s × 1e6 × 8 B × channels. E.g. DDR4-3200 × 2 channels = 3200e6 × 8 × 2 = 51.2e9 B/s.
 */
export function ddr4BandwidthBytesPerSec(megaTransfersPerSec: number, channels: number): number {
  const transfersPerSec = megaTransfersPerSec * 1e6;
  return transfersPerSec * DDR4_BYTES_PER_TRANSFER_PER_CHANNEL * channels;
}

/** A DDR4 host-RAM preset `ddr4-<MT/s>-<channels>ch`; same parameters and conventions as {@link ddr5}. */
function ddr4(p: Ddr5Params): HostMemorySpec {
  return {
    kind: 'host_ram',
    id: `ddr4-${p.megaTransfersPerSec}-${p.channels}ch`,
    name: `DDR4-${p.megaTransfersPerSec}, ${p.channels} channels (${p.platform})`,
    capacityBytes: p.capacityBytes,
    bandwidthBytesPerSec: ddr4BandwidthBytesPerSec(p.megaTransfersPerSec, p.channels),
    latencySec: p.latencySec,
    channels: p.channels,
    readGranularityBytes: 64,
    maxOutstanding: 64,
    sources: p.sources,
    estimated: [...p.estimated, CACHE_LINE_ESTIMATE, HOST_MLP_ESTIMATE],
  };
}

/** Latency estimate shared by the DDR4 presets: an assumption (no vendor number, no cited measurement). */
function ddr4LatencyEstimate(ns: number, platform: string): EstimatedField {
  return {
    field: 'latencySec',
    reason:
      `No vendor latency and no measurement cited: ${ns} ns unloaded load-to-use is an ASSUMED figure for ` +
      `${platform} (typical of DDR4 platforms; real values vary with timings and fabric clock).`,
  };
}

/** The DDR4 data-rate source every preset shares (the transfer rate itself is a JEDEC / XMP number). */
const JEDEC_DDR4_SOURCE: SourceRef = {
  url: JEDEC_DDR4_URL,
  title: 'JEDEC JESD79-4 DDR4 SDRAM standard',
  fields: ['bandwidthBytesPerSec'],
  note: 'Per-device standard: 64-bit data bus per channel ⇒ 8 B per transfer; bandwidth = MT/s × 1e6 × 8 B × channels. It does not specify how many channels a platform has.',
};

/**
 * Desktop dual-channel DDR4-2666 (the rated speed of Intel 8th/9th gen and Ryzen 2000-class platforms):
 * 2666e6 × 8 × 2 = 42.656e9 B/s. PCIe 3.0 on these boards.
 */
export const DDR4_2666_2CH: HostMemorySpec = ddr4({
  megaTransfersPerSec: 2666,
  channels: 2,
  capacityBytes: gb(32),
  latencySec: 80e-9,
  platform: 'desktop',
  sources: [JEDEC_DDR4_SOURCE],
  estimated: [
    capacityEstimate('2 × 16 GB DIMMs'),
    {
      field: 'channels',
      reason:
        'Configuration assumption: a mainstream desktop platform with two memory channels (dual-channel is the ' +
        'standard layout of Intel Core 8th/9th gen and AMD Ryzen 2000 desktops); no platform page is cited.',
    },
    ddr4LatencyEstimate(80, 'a dual-channel desktop at 2666 MT/s'),
  ],
});

/**
 * Desktop dual-channel DDR4-3200 (AM4 Ryzen 5000 rated speed, JEDEC top bin): 3200e6 × 8 × 2 = 51.2e9 B/s. PCIe 4.0
 * on X570/B550, 3.0 on older AM4 boards.
 */
export const DDR4_3200_2CH: HostMemorySpec = ddr4({
  megaTransfersPerSec: 3200,
  channels: 2,
  capacityBytes: gb(64),
  latencySec: 75e-9,
  platform: 'AM4 desktop',
  sources: [
    {
      url: RYZEN_5950X_URL,
      title: 'AMD Ryzen 9 5950X specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"Memory Channels 2"; "Max Memory Speed 2x1R DDR4-3200". Bandwidth = 3200e6 × 8 B × 2 = 51.2e9.',
    },
  ],
  estimated: [
    capacityEstimate('2 × 32 GB DIMMs'),
    ddr4LatencyEstimate(75, 'AM4 Zen 3 at 3200 MT/s'),
  ],
});

/**
 * Desktop dual-channel DDR4-3600 (the AM4 sweet spot: 1:1 with the 1800 MHz fabric clock): 3600e6 × 8 × 2 = 57.6e9 B/s.
 * PCIe 4.0 on X570/B550.
 */
export const DDR4_3600_2CH: HostMemorySpec = ddr4({
  megaTransfersPerSec: 3600,
  channels: 2,
  capacityBytes: gb(64),
  latencySec: 70e-9,
  platform: 'AM4 desktop',
  sources: [
    {
      url: RYZEN_5950X_URL,
      title: 'AMD Ryzen 9 5950X specifications',
      fields: ['channels'],
      note: 'Rated DDR4-3200; 3600 MT/s is an XMP/DOCP overclock that most Zen 3 chips run at FCLK 1800 MHz (1:1).',
    },
    JEDEC_DDR4_SOURCE,
  ],
  estimated: [
    capacityEstimate('2 × 32 GB DIMMs'),
    {
      field: 'bandwidthBytesPerSec',
      reason:
        'Overclock speed, above the 3200 MT/s AMD rates: 3600e6 × 8 B × 2 = 57.6e9 B/s theoretical.',
    },
    ddr4LatencyEstimate(70, 'AM4 Zen 3 at 3600 MT/s, CL16'),
  ],
});

/**
 * HEDT / workstation quad-channel DDR4-3200 (Threadripper 3000 on TRX40; X299 and Xeon W run DDR4-2933 on 4-6 channels):
 * 3200e6 × 8 × 4 = 102.4e9 B/s. PCIe 4.0 on TRX40.
 */
export const DDR4_3200_4CH: HostMemorySpec = ddr4({
  megaTransfersPerSec: 3200,
  channels: 4,
  capacityBytes: gb(128),
  latencySec: 90e-9,
  platform: 'HEDT',
  sources: [
    {
      url: TR_3970X_URL,
      title: 'AMD Ryzen Threadripper 3970X specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"4 memory channels"; "Up to 3200 MT/s". Bandwidth = 3200e6 × 8 B × 4 = 102.4e9.',
    },
  ],
  estimated: [
    capacityEstimate('4 × 32 GB DIMMs'),
    ddr4LatencyEstimate(90, 'Threadripper 3000 (chiplet IO die)'),
  ],
});

/** Server 8-channel DDR4-3200 (EPYC 7002/7003, Threadripper PRO 3000WX): 3200e6 × 8 × 8 = 204.8e9 B/s. PCIe 4.0. */
export const DDR4_3200_8CH: HostMemorySpec = ddr4({
  megaTransfersPerSec: 3200,
  channels: 8,
  capacityBytes: gb(256),
  latencySec: 115e-9,
  platform: 'server, EPYC Rome/Milan',
  sources: [
    {
      url: EPYC_7763_URL,
      title: 'AMD EPYC 7763 specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"Memory Channels 8"; "DDR4-3200"; "Per Socket Mem BW 204.8 GB/s" (= 3200e6 × 8 B × 8).',
    },
  ],
  estimated: [
    capacityEstimate('8 × 32 GB RDIMMs, one socket'),
    ddr4LatencyEstimate(115, 'EPYC 7002/7003 with RDIMMs'),
  ],
});

/** Server 6-channel DDR4-2933 (Xeon Scalable 2nd gen, Cascade Lake-SP): 2933e6 × 8 × 6 = 140.784e9 B/s. PCIe 3.0. */
export const DDR4_2933_6CH: HostMemorySpec = ddr4({
  megaTransfersPerSec: 2933,
  channels: 6,
  capacityBytes: gb(192),
  latencySec: 95e-9,
  platform: 'server, Xeon Scalable 2nd gen',
  sources: [
    {
      url: ARK_GOLD_6248_URL,
      title: 'Intel ark: Intel Xeon Gold 6248 specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"DDR4-2933"; "Max # of Memory Channels 6". Bandwidth = 2933e6 × 8 B × 6.',
    },
  ],
  estimated: [
    capacityEstimate('6 × 32 GB RDIMMs, one socket'),
    ddr4LatencyEstimate(95, 'Xeon Scalable 2nd gen RDIMMs'),
  ],
});

/** Server 4-channel DDR4-2400 (Xeon E5 v4, e.g. E5-2680 v4; E5 v3 tops out at DDR4-2133): 2400e6 × 8 × 4 = 76.8e9 B/s, as Intel quotes it. PCIe 3.0. */
export const DDR4_2400_4CH: HostMemorySpec = ddr4({
  megaTransfersPerSec: 2400,
  channels: 4,
  capacityBytes: gb(128),
  latencySec: 85e-9,
  platform: 'server, Xeon E5 v4',
  sources: [
    {
      url: ARK_E5_2680_V4_URL,
      title: 'Intel ark: Intel Xeon E5-2680 v4 specifications',
      fields: ['bandwidthBytesPerSec', 'channels'],
      note: '"Max # of Memory Channels 4"; "Max Memory Bandwidth 76.8 GB/s" (= 2400e6 × 8 B × 4); DDR4 up to 2400.',
    },
  ],
  estimated: [
    capacityEstimate('4 × 32 GB RDIMMs, one socket'),
    ddr4LatencyEstimate(85, 'Xeon E5 v4 RDIMMs'),
  ],
});

/** Every host-RAM preset: DDR5 first, then DDR4; desktop first, then by bandwidth. */
export const HOST_RAM_PRESETS: HostMemorySpec[] = [
  DDR5_5600_2CH,
  DDR5_6000_2CH,
  DDR5_6400_2CH,
  DDR5_6400_8CH,
  DDR5_4800_12CH,
  DDR5_6000_12CH,
  DDR5_4800_8CH,
  DDR5_4800_4CH,
  DDR5_4800_2CH,
  DDR4_2666_2CH,
  DDR4_3200_2CH,
  DDR4_3600_2CH,
  DDR4_3200_4CH,
  DDR4_3200_8CH,
  DDR4_2933_6CH,
  DDR4_2400_4CH,
];

/** The default host RAM when the stack names none: desktop DDR5-6000, 2 channels. */
export const DEFAULT_HOST_RAM: HostMemorySpec = DDR5_6000_2CH;

// ---------------------------------------------------------------------------------------------------------
// NVMe SSDs
// ---------------------------------------------------------------------------------------------------------

/**
 * Samsung 990 PRO 2 TB (PCIe 4.0 x4): 7,450 MB/s sequential read; 1,400K IOPS random read at QD32 × 16 threads;
 * 22K IOPS at QD1 ⇒ 1/22,000 = 45.5 µs per random read. The DEFAULT SSD.
 */
export const NVME_990_PRO_2TB: NvmeSpec = {
  kind: 'nvme',
  id: 'nvme-990-pro-2tb',
  name: 'Samsung 990 PRO 2 TB (PCIe 4.0 x4)',
  capacityBytes: 2e12,
  seqReadBandwidthBytesPerSec: gbPerSec(7.45),
  sustainedRead: { fraction: 0.646, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  // 1 / QD1 IOPS = 1 / 22,000 = 45.45e-6 s.
  randomReadLatencySec: 1 / 22_000,
  randomRead: { iops: 1_400_000, queueDepth: 32, threads: 16 },
  // queueDepth × threads of the rated condition = 32 × 16.
  maxOutstanding: 32 * 16,
  readGranularityBytes: 4096,
  sources: [
    {
      url: 'https://www.tomshardware.com/reviews/samsung-990-pro-ssd-review/2',
      title: 'Low-QD large-read comparison',
      kind: 'review',
      fields: ['sustainedRead'],
      note: '2 TB CDM SEQ1M Q1T1 / Q8T1: 4,618 / 7,153 MB/s = 0.646. Condition: 1 MiB, QD1, 1 thread; divide by the same run deep-queue rate.',
    },
    {
      url: SAMSUNG_990_PRO_URL,
      title: 'Samsung SSD 990 PRO datasheet Rev 1.0, Technical specifications',
      fields: [
        'capacityBytes',
        'seqReadBandwidthBytesPerSec',
        'randomReadLatencySec',
        'randomRead',
        'maxOutstanding',
      ],
      note:
        '2TB column: "Sequential Read 7,450 MB/s"; "QD 1 Thread 1 Ran. Read 22K IOPS"; "QD 32 Thread 16 Ran. Read ' +
        '1400K IOPS"; "Interface PCIe Gen 4.0 x4"; footnote 1: "1GB = 1,000,000,000 bytes". Latency = 1/22K IOPS; ' +
        'maxOutstanding = 32 × 16.',
    },
  ],
  estimated: [PAGE_ESTIMATE],
};

/** Sandisk WD_BLACK SN850X 2 TB (PCIe 4.0 x4): 7,300 MB/s sequential read; 1,200K IOPS random read. */
export const NVME_SN850X_2TB: NvmeSpec = {
  kind: 'nvme',
  id: 'nvme-sn850x-2tb',
  name: 'WD_BLACK SN850X 2 TB (PCIe 4.0 x4)',
  capacityBytes: 2e12,
  seqReadBandwidthBytesPerSec: gbPerSec(7.3),
  sustainedRead: { fraction: 0.747, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  randomReadLatencySec: 50e-6,
  randomRead: { iops: 1_200_000, queueDepth: 32, threads: 16 },
  maxOutstanding: 32 * 16,
  readGranularityBytes: 4096,
  sources: [
    {
      url: 'https://www.tomshardware.com/reviews/samsung-990-pro-ssd-review/2',
      title: 'Low-QD large-read comparison',
      kind: 'review',
      fields: ['sustainedRead'],
      note: '2 TB SN850X comparison: CDM SEQ1M Q1T1 / Q8T1: 5,214 / 6,982 MB/s = 0.747. Condition: 1 MiB, QD1, 1 thread; divide by the same run deep-queue rate.',
    },
    {
      url: SN850X_URL,
      title: 'Sandisk WD_BLACK SN850X NVMe SSD datasheet (July 2025), Product specifications',
      fields: ['capacityBytes', 'seqReadBandwidthBytesPerSec', 'randomRead.iops'],
      note:
        '2TB column: "Sequential Read [up to] 7,300MB/s", "Random Read [up to] 1,200K IOPS"; "PCIe Gen4 16GT/s, up ' +
        'to 4 lanes"; "1 MB/s = 1 million bytes per second"; "1TB = 1 trillion bytes". No QD1 figure or test condition.',
    },
  ],
  estimated: [
    {
      field: 'randomReadLatencySec',
      reason:
        'The datasheet gives no QD1 figure; 50 µs assumed for a Gen4 TLC drive (the 990 PRO measures 45 µs).',
    },
    {
      field: 'randomRead.queueDepth',
      reason:
        'The datasheet states no test condition; the client-SSD convention QD32 × 16 threads is assumed.',
    },
    {
      field: 'randomRead.threads',
      reason:
        'The datasheet states no test condition; the client-SSD convention QD32 × 16 threads is assumed.',
    },
    {
      field: 'maxOutstanding',
      reason: 'queueDepth × threads of the assumed rated condition: 32 × 16 = 512.',
    },
    PAGE_ESTIMATE,
  ],
};

/** Samsung 9100 PRO 4 TB (PCIe 5.0 x4): 14,800 MB/s sequential read; 2,200K IOPS random read at QD256 × 32 threads. */
export const NVME_9100_PRO_4TB: NvmeSpec = {
  kind: 'nvme',
  id: 'nvme-9100-pro-4tb',
  name: 'Samsung 9100 PRO 4 TB (PCIe 5.0 x4)',
  capacityBytes: 4e12,
  seqReadBandwidthBytesPerSec: gbPerSec(14.8),
  sustainedRead: { fraction: 0.72, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  randomReadLatencySec: 45e-6,
  randomRead: { iops: 2_200_000, queueDepth: 256, threads: 32 },
  // queueDepth × threads of the rated condition = 256 × 32.
  maxOutstanding: 256 * 32,
  readGranularityBytes: 4096,
  sources: [
    {
      url: 'https://www.tomshardware.com/pc-components/ssds/samsung-9100-pro-ssd-review/3',
      title: 'Low-QD large-read comparison',
      kind: 'review',
      fields: ['sustainedRead'],
      note: '4 TB with heatsink: CDM SEQ1M Q1T1 / Q8T1: 10,287 / 14,295 MB/s = 0.720. Condition: 1 MiB, QD1, 1 thread; divide by the same run deep-queue rate.',
    },
    {
      url: SAMSUNG_9100_PRO_URL,
      title: 'Samsung SSD 9100 PRO datasheet Rev 2.0, Technical specifications',
      fields: ['capacityBytes', 'seqReadBandwidthBytesPerSec', 'randomRead', 'maxOutstanding'],
      note:
        '4TB column: "Sequential Read 14,800 MB/s"; "QD 256 Thread 32 Ran. Read 2,200 KIOPS"; "Interface PCIe 5.0 ' +
        'x4"; footnote 1: "1GB = 1,000,000,000 bytes". maxOutstanding = 256 × 32. No QD1 figure.',
    },
  ],
  estimated: [
    {
      field: 'randomReadLatencySec',
      reason:
        "The datasheet gives no QD1 figure; taken equal to the 990 PRO's 45 µs (same vendor, same NAND class).",
    },
    PAGE_ESTIMATE,
  ],
};

/** Crucial T705 2 TB (PCIe 5.0 x4): 14,500 MB/s sequential read; 1,550K IOPS random read at queue depth 512. */
export const NVME_T705_2TB: NvmeSpec = {
  kind: 'nvme',
  id: 'nvme-t705-2tb',
  name: 'Crucial T705 2 TB (PCIe 5.0 x4)',
  capacityBytes: 2e12,
  seqReadBandwidthBytesPerSec: gbPerSec(14.5),
  sustainedRead: { fraction: 0.646, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  randomReadLatencySec: 45e-6,
  randomRead: { iops: 1_550_000, queueDepth: 512, threads: 1 },
  maxOutstanding: 512,
  readGranularityBytes: 4096,
  sources: [
    {
      url: 'https://www.cdrlabs.com/reviews/crucial-t705-2tb-pcie-gen5-nvme-solid-state-drive/print.html',
      title: 'Low-QD large-read comparison',
      kind: 'review',
      fields: ['sustainedRead'],
      note: '2 TB CDM Real World SEQ1M Q1T1 / Peak SEQ1M Q8T1: 9,453.93 / 14,631.35 MB/s = 0.646. Condition: 1 MiB, QD1, 1 thread; divide by the same run deep-queue rate.',
    },
    {
      url: CRUCIAL_T705_URL,
      title: 'Crucial T705 2TB PCIe Gen5 NVMe M.2 SSD (CT2000T705SSD3) product page',
      fields: ['capacityBytes', 'randomRead', 'maxOutstanding'],
      note:
        'Spec table: 14,500 MB/s sequential read, 1,550K IOPS random read (2TB). Footnote 2: "Typical I/O performance ' +
        'as measured using CrystalDiskMark® with a queue depth of 512"; footnote 6: "1GB equals 1 billion bytes". The ' +
        'split of the 512 over threads is not stated: recorded as 512 × 1, maxOutstanding = 512.',
    },
    {
      url: CRUCIAL_T705_FLYER_URL,
      title: 'Crucial T705 Gen5 SSD product flyer',
      fields: ['seqReadBandwidthBytesPerSec'],
      note: '"Sequential reads/writes up to 14,500/12,700MB/s".',
    },
    {
      url: CDRLABS_T705_URL,
      title: 'CDRLabs: Crucial T705 2TB review, Iometer',
      kind: 'review',
      fields: ['randomRead.iops'],
      note: '"With sixteen threads and the queue depth set to 32, it reached 1,411,981 random read IOPS" (91% of rated).',
    },
  ],
  estimated: [
    {
      field: 'randomReadLatencySec',
      reason:
        "No QD1 figure published; taken equal to the 990 PRO's 45 µs (a Gen5 controller does not shorten NAND reads).",
    },
    PAGE_ESTIMATE,
  ],
};

/**
 * Apple M3 / M4 Max internal SSD, 1 TB configuration: about 6.4 GB/s sequential read (crowd medians). Apple
 * publishes no SSD figures, so these rest on community measurements (the UI flags `kind: 'community'`).
 */
export const APPLE_SSD_M3_M4_MAX_1TB: NvmeSpec = {
  kind: 'nvme',
  id: 'apple-ssd-m3-m4-max-1tb',
  name: 'Apple M3 / M4 Max internal SSD, 1 TB',
  capacityBytes: 1e12,
  seqReadBandwidthBytesPerSec: gbPerSec(6.4),
  sustainedRead: { fraction: 0.51, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  // 4096 B / 60e6 B/s (crowd-median random 4 KiB QD1 ≈ 60 MB/s) = 68e-6 s.
  randomReadLatencySec: 4096 / 60e6,
  randomRead: { iops: 250_000, queueDepth: 64, threads: 1 },
  maxOutstanding: 64,
  readGranularityBytes: 4096,
  sources: [
    {
      url: 'https://docs.google.com/spreadsheets/d/1Mu8n3438TM2orWivg_Y8YGlbd8VLcy3c-xBZulH_1zM/edit?usp=sharing',
      title: 'Low-QD large-read comparison',
      kind: 'community',
      fields: ['sustainedRead'],
      note: 'AmorphousDiskMark SEQ1M QD1 / QD8, 1 TB: M3 Max 3,169 / 6,396; M4 Max 3,363 / 6,444 MB/s; mean 0.509 ≈ 0.51 (n=2). Condition: 1 MiB, QD1, 1 thread; divide by the same run deep-queue rate.',
    },
    {
      url: MAC_SSD_REDDIT_URL,
      title:
        'r/mac: Mac SSD speeds overview (AmorphousDiskMark crowd medians, updated with M5 Pro/Max)',
      kind: 'community',
      fields: ['seqReadBandwidthBytesPerSec'],
      note:
        'Crowd medians over 115 AmorphousDiskMark runs of M3 Max / M4 Max 1 TB machines: sequential read ≈ 6.4 GB/s, ' +
        'random 4 KiB QD1 ≈ 60 MB/s; 2.5e5 IOPS at QD64 derived from the same data (all as transcribed in ' +
        'docs/design/offload-specdec-latency.md §2.2; Reddit blocks automated fetches, so not re-read here).',
    },
    {
      url: LLM_IN_A_FLASH_URL,
      title: 'Alizadeh et al., "LLM in a flash" (arXiv 2312.11514), §2.2 Read Throughput',
      fields: ['seqReadBandwidthBytesPerSec'],
      note: '"benchmarks on an Apple MacBook M1 Max with 1TB flash memory demonstrate speeds exceeding 6 GiB/s for a 1GiB linear read" (consistent).',
    },
  ],
  estimated: [
    {
      field: 'capacityBytes',
      reason: 'Configuration choice: the 1 TB SSD option. Set it to the installed amount.',
    },
    {
      field: 'randomReadLatencySec',
      reason: `Derived from the crowd-median random 4 KiB QD1 ≈ 60 MB/s (${MAC_SSD_REDDIT_URL}): 4096 B / 60e6 B/s = 68 µs.`,
    },
    {
      field: 'randomRead',
      reason:
        `Derived from the crowd-median random 4 KiB QD64 rate (${MAC_SSD_REDDIT_URL}), ≈ 1 GB/s: 1e9 / 4096 ≈ ` +
        "250K IOPS at QD64 × 1 thread (AmorphousDiskMark's default random test).",
    },
    {
      field: 'maxOutstanding',
      reason: 'The queue depth of the measured condition (AmorphousDiskMark random 4 KiB, QD64).',
    },
    PAGE_ESTIMATE,
  ],
};

/** Apple M5 Pro / Max internal SSD, 2 TB configuration: about 13.1 GB/s sequential read (few crowd runs, n = 1–3). */
export const APPLE_SSD_M5_PRO_MAX: NvmeSpec = {
  kind: 'nvme',
  id: 'apple-ssd-m5-pro-max',
  name: 'Apple M5 Pro / Max internal SSD, 2 TB',
  capacityBytes: 2e12,
  seqReadBandwidthBytesPerSec: gbPerSec(13.1),
  sustainedRead: { fraction: 0.84, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  randomReadLatencySec: 65e-6,
  randomRead: { iops: 270_000, queueDepth: 64, threads: 1 },
  maxOutstanding: 64,
  readGranularityBytes: 4096,
  sources: [
    {
      url: 'https://docs.google.com/spreadsheets/d/1Mu8n3438TM2orWivg_Y8YGlbd8VLcy3c-xBZulH_1zM/edit?usp=sharing',
      title: 'Low-QD large-read comparison',
      kind: 'community',
      fields: ['sustainedRead'],
      note: 'AmorphousDiskMark SEQ1M QD1 / QD8 median 0.839 ≈ 0.84 (7 M5 Pro/Max, 1–8 TB). 2 TB: 11,157 / 13,302 and 11,384 / 13,147 MB/s. Condition: 1 MiB, QD1, 1 thread; divide by the same run deep-queue rate.',
    },
    {
      url: MAC_SSD_REDDIT_URL,
      title:
        'r/mac: Mac SSD speeds overview (AmorphousDiskMark crowd medians, updated with M5 Pro/Max)',
      kind: 'community',
      fields: ['seqReadBandwidthBytesPerSec'],
      note:
        'M5 Pro / M5 Max rows (n = 1–3 runs): sequential read ≈ 13.1 GB/s (as transcribed in ' +
        'docs/design/offload-specdec-latency.md §2.2; Reddit blocks automated fetches, so not re-read here). Low confidence.',
    },
  ],
  estimated: [
    {
      field: 'capacityBytes',
      reason: 'Configuration choice: a 2 TB SSD option. Set it to the installed amount.',
    },
    {
      field: 'randomReadLatencySec',
      reason: `Derived from the random 4 KiB QD1 rows (${MAC_SSD_REDDIT_URL}), n = 1–3: ≈ 4096 B / 63e6 B/s ≈ 65 µs.`,
    },
    {
      field: 'randomRead',
      reason: `Derived from the random 4 KiB QD64 rows (${MAC_SSD_REDDIT_URL}), n = 1–3: ≈ 1.1 GB/s / 4096 ≈ 270K IOPS at QD64 × 1.`,
    },
    {
      field: 'maxOutstanding',
      reason: 'The queue depth of the measured condition (AmorphousDiskMark random 4 KiB, QD64).',
    },
    PAGE_ESTIMATE,
  ],
};

/**
 * NVIDIA DGX Spark internal SSD, 4 TB (PCIe 5.0 x4, M.2 2242): GDSIO sequential read peak 11.4 GiB/s =
 * 12.24e9 B/s (StorageReview). The random-read figures are estimates: no published measurement.
 */
export const DGX_SPARK_SSD_4TB: NvmeSpec = {
  kind: 'nvme',
  id: 'dgx-spark-ssd-4tb',
  name: 'DGX Spark internal SSD, 4 TB (PCIe 5.0 x4, M.2 2242)',
  capacityBytes: 4e12,
  // 11.4 GiB/s × 2^30 B/GiB = 12.24e9 B/s.
  seqReadBandwidthBytesPerSec: 11.4 * 2 ** 30,
  sustainedRead: { fraction: 0.65, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  randomReadLatencySec: 50e-6,
  randomRead: { iops: 1_500_000, queueDepth: 32, threads: 16 },
  maxOutstanding: 32 * 16,
  readGranularityBytes: 4096,
  sources: [
    {
      url: STORAGEREVIEW_SPARK_URL,
      title: 'StorageReview: NVIDIA DGX Spark review, GDSIO internal 4 TB M.2',
      kind: 'review',
      fields: ['seqReadBandwidthBytesPerSec'],
      note:
        '"the tiny 2242 form factor M.2 SSD connected with Gen5 x4"; "we see the highest throughput at 1M block size ' +
        'with 16 threads, achieving 11.4 GiB/s" (GPU Direct Storage, sequential read). 11.4 × 2^30 = 12.24e9 B/s.',
    },
    {
      url: DGX_SPARK_HW_URL,
      title: 'NVIDIA DGX Spark user guide: hardware overview',
      fields: ['capacityBytes'],
      note: '"Storage: 1 TB or 4 TB NVMe M.2 with self-encryption" (the 4 TB Founders Edition).',
    },
  ],
  estimated: [
    {
      field: 'sustainedRead',
      reason:
        'No CPU-path low-QD data. Borrowed Gen5 peer fraction (T705 0.646, 9100 PRO 0.720). Closest GDSIO 1 MiB 1-thread / 16-thread = 2.2 / 11.4 GiB/s = 0.193 includes cuFile overhead and is not a drive-only ratio. https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops Condition: 1 MiB, QD1, 1 thread.',
    },
    {
      field: 'randomReadLatencySec',
      reason:
        'No published QD1 figure; a Gen5 2242 drive is assumed between the 990 PRO (45 µs) and the Apple SSDs (65 µs).',
    },
    {
      field: 'randomRead',
      reason:
        'StorageReview published no random-read figure; 1.5M IOPS at QD32 × 16 is assumed between the 990 PRO (1.4M) ' +
        'and the 9100 PRO (2.2M).',
    },
    {
      field: 'maxOutstanding',
      reason: 'queueDepth × threads of the assumed rated condition: 32 × 16 = 512.',
    },
    PAGE_ESTIMATE,
  ],
};

/**
 * A generic PCIe 4.0 x4 M.2 SSD, for Strix Halo boxes (Framework Desktop: two PCIe 4.0 x4 M.2 sockets). The slot
 * ceiling 16e9 × 4 × 128/130 / 8 × 256/280 = 7.20e9 B/s (line rate × TLP payload efficiency, links.ts
 * `PCIE_PAYLOAD_EFFICIENCY`); the drive itself is assumed at 7.0e9 B/s, so the drive binds.
 */
export const NVME_GEN4_X4_GENERIC: NvmeSpec = {
  kind: 'nvme',
  id: 'nvme-gen4-x4-generic',
  name: 'Generic PCIe 4.0 x4 NVMe SSD, 2 TB',
  capacityBytes: 2e12,
  seqReadBandwidthBytesPerSec: gbPerSec(7.0),
  sustainedRead: { fraction: 0.666, blockBytes: 1_048_576, queueDepth: 1, threads: 1 },
  randomReadLatencySec: 50e-6,
  randomRead: { iops: 1_000_000, queueDepth: 32, threads: 16 },
  maxOutstanding: 32 * 16,
  readGranularityBytes: 4096,
  linkBandwidthBytesPerSec: pcieBandwidthPerDirectionBytesPerSec(4, 4),
  sources: [
    {
      url: FRAMEWORK_DESKTOP_URL,
      title: 'Framework Desktop (Ryzen AI Max) specifications',
      fields: ['linkBandwidthBytesPerSec'],
      note:
        'Storage: two "NVMe PCIe 4.0 x4" M.2 2280 sockets. Slot ceiling = 16 GT/s × 4 × 128/130 ÷ 8 × 256/280 ' +
        '(TLP payload efficiency, Intel AN 855: https://cdrdv2-public.intel.com/666287/an855-683504-666287.pdf) = 7.20e9 B/s.',
    },
  ],
  estimated: [
    {
      field: 'sustainedRead',
      reason:
        'Median CDM SEQ1M Q1T1 / Q8T1 fraction of 19 Gen4 x4 drives (2/4 TB), IQR 0.608–0.736; comparison charts in this review and the 9100 PRO review pages 2–3. https://www.tomshardware.com/reviews/samsung-990-pro-ssd-review/2 Condition: 1 MiB, QD1, 1 thread.',
    },
    {
      field: 'capacityBytes',
      reason: 'Configuration choice: a 2 TB drive. Set it to the installed amount.',
    },
    {
      field: 'seqReadBandwidthBytesPerSec',
      reason:
        'A generic Gen4 x4 drive: 7.0 GB/s, just below the 7.3–7.45 GB/s of the best Gen4 drives (SN850X, 990 PRO).',
    },
    {
      field: 'randomReadLatencySec',
      reason: 'Generic Gen4 TLC drive: 50 µs assumed (the 990 PRO measures 45 µs).',
    },
    {
      field: 'randomRead',
      reason:
        'Generic Gen4 drive: 1.0M IOPS at QD32 × 16 assumed (high-end Gen4 drives rate 1.2–1.4M).',
    },
    {
      field: 'maxOutstanding',
      reason: 'queueDepth × threads of the assumed rated condition: 32 × 16 = 512.',
    },
    PAGE_ESTIMATE,
  ],
};

/** Every SSD preset: desktop drives first (default first), then the internal SSDs of unified-memory machines. */
export const SSD_PRESETS: NvmeSpec[] = [
  NVME_990_PRO_2TB,
  NVME_SN850X_2TB,
  NVME_9100_PRO_4TB,
  NVME_T705_2TB,
  NVME_GEN4_X4_GENERIC,
  APPLE_SSD_M3_M4_MAX_1TB,
  APPLE_SSD_M5_PRO_MAX,
  DGX_SPARK_SSD_4TB,
];

/** The default SSD when the stack names none: Samsung 990 PRO 2 TB. */
export const DEFAULT_SSD: NvmeSpec = NVME_990_PRO_2TB;
