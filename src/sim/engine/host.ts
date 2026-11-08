/**
 * Resolving weight offloading for one run: the deployment's `OffloadConfig` → the placement lowering applies
 * (`OffloadLowering`, every default filled in), and the stack's `HardwareStack.host` → the host link, host RAM and
 * SSD the transfers use (defaults with a warning). `hardware.host` is read only when something is off-device.
 * See docs/simulation-model.md §12 ("Storage tiers", "Placement") and design doc §2.1–2.2.
 */

import {
  APPLE_SSD_M3_M4_MAX_1TB,
  APPLE_SSD_M5_PRO_MAX,
  DEFAULT_HOST_RAM,
  DEFAULT_SSD,
  DGX_SPARK_SSD_4TB,
  NVME_GEN4_X4_GENERIC,
} from '../hw/presets/storage';
import { hostPcieLink } from '../hw/presets/links';
import type { DeviceSpec, HardwareStack, HostMemorySpec, LinkSpec, NvmeSpec } from '../hw/types';
import type { ModelSpec } from '../ir/model';
import type { OffloadLowering } from '../lower/lower';
import { hitRatio, resolveQueueDepth, type TransferPath } from './transfer';
import type { OffloadConfig, OffloadTier } from './types';

/** The resolved placement and the notes it produced. */
export interface ResolvedOffload {
  /** What lowering applies; undefined when nothing ends up off the device (identity). */
  offload: OffloadLowering | undefined;
  /** Human-readable notes (approximations, placements that are device memory). */
  warnings: string[];
}

/**
 * Resolve the deployment's offload placement (design doc §2.1): the routed-expert layer range defaults to every
 * layer, h(f) is evaluated from the hit-ratio model, the table scale layout defaults to `'separate'` on host RAM and
 * `'pinned_host'` on NVMe. On unified memory host RAM IS the device pool, so a `'host_ram'` placement is dropped
 * (no transfer, weights stay counted on the device). Lookup tables at `'device'` are no placement.
 */
export function resolveOffload(
  model: ModelSpec,
  config: OffloadConfig | undefined,
  memoryKind: 'dedicated' | 'unified',
  ep: number,
): ResolvedOffload {
  const warnings: string[] = [];
  if (config === undefined) return { offload: undefined, warnings };
  const onDevice = (tier: OffloadTier): boolean => {
    if (memoryKind === 'unified' && tier === 'host_ram') {
      warnings.push("offload: on unified memory host RAM is the device memory; the 'host_ram' placement keeps the weights on the device");
      return true;
    }
    return false;
  };
  const offload: OffloadLowering = {};
  const experts = config.routedExperts;
  if (experts !== undefined && !onDevice(experts.tier)) {
    const f = experts.residentFraction;
    offload.routedExperts = {
      tier: experts.tier,
      layers: experts.layers ?? { from: 0, to: model.layers.length },
      residentFraction: f,
      hitRatio: hitRatio(experts.hitRatio, f),
      includeDraft: experts.includeDraft ?? false,
    };
    if (ep > 1) {
      warnings.push(
        'offload: ep > 1 — needed experts per device use the with-replacement approximation U_dev = E_loc·(1 − (1 − 1/E_loc)^A)',
      );
    }
    const skewed = experts.hitRatio.kind === 'skewed_classic' || experts.hitRatio.kind === 'skewed_fine_grained';
    if (skewed) {
      const thrashing = model.layers.some((l) =>
        l.blocks.some((b) => b.kind === 'moe' && f * (b.experts / ep) > 0 && f * (b.experts / ep) < b.topK),
      );
      if (thrashing) {
        warnings.push(
          'offload: fewer resident experts per layer than top-k — an LRU cache thrashes there (h ≈ 0); the skewed hit ratio is optimistic',
        );
      }
    }
  }
  const dense = config.denseLayers;
  if (dense !== undefined && !onDevice(dense.tier)) {
    offload.denseLayers = { tier: dense.tier, layers: dense.layers };
  }
  const tables = config.lookupTables;
  if (tables !== undefined && tables.tier !== 'device' && !onDevice(tables.tier)) {
    offload.lookupTables = {
      tier: tables.tier,
      scaleLayout: tables.scaleLayout ?? (tables.tier === 'host_ram' ? 'separate' : 'pinned_host'),
    };
  }
  const any = offload.routedExperts !== undefined || offload.denseLayers !== undefined || offload.lookupTables !== undefined;
  return { offload: any ? offload : undefined, warnings: [...new Set(warnings)] };
}

/** The off-device tiers a resolved placement uses. */
export function offloadTiers(offload: OffloadLowering): Set<OffloadTier> {
  const tiers = new Set<OffloadTier>();
  if (offload.routedExperts !== undefined) tiers.add(offload.routedExperts.tier);
  if (offload.denseLayers !== undefined) tiers.add(offload.denseLayers.tier);
  if (offload.lookupTables !== undefined) tiers.add(offload.lookupTables.tier);
  return tiers;
}

/**
 * The SSD a device uses when the stack names none: the device's own internal drive on unified-memory systems
 * (Apple M5 → `apple-ssd-m5-pro-max`, other Apple → `apple-ssd-m3-m4-max-1tb`, DGX Spark → `dgx-spark-ssd-4tb`,
 * Ryzen AI Max (Strix Halo) → `nvme-gen4-x4-generic`), else the Samsung 990 PRO 2 TB.
 */
export function defaultSsdFor(device: DeviceSpec): NvmeSpec {
  if (device.memory.kind === 'unified') {
    if (device.id.startsWith('apple-m5')) return APPLE_SSD_M5_PRO_MAX;
    if (device.id.startsWith('apple-')) return APPLE_SSD_M3_M4_MAX_1TB;
    if (device.id.startsWith('dgx-spark')) return DGX_SPARK_SSD_4TB;
    if (device.id.startsWith('ryzen-ai-max')) return NVME_GEN4_X4_GENERIC;
  }
  return DEFAULT_SSD;
}

/** The host side a run's transfers use, with every default filled in. */
export interface ResolvedHost {
  /** The device's host link (dedicated memory only). */
  link?: LinkSpec;
  /** The node's host RAM (dedicated memory only: a tier, and the staging hop of NVMe reads). */
  ram?: HostMemorySpec;
  /** The node's SSD (when something is placed on `'nvme'`). */
  ssd?: NvmeSpec;
  /** One line per default taken, and the SSD's missing random-read rating. */
  warnings: string[];
}

/**
 * Fill the host side for the tiers a placement uses (design doc §2.2). Dedicated memory: `host.link` absent →
 * `hostPcieLink(device.hostInterface)`; `host.ram` absent → `DDR5_6000_2CH` (needed for host-RAM groups and as the
 * staging hop of NVMe reads). `host.ssd` absent with an NVMe group → {@link defaultSsdFor}. Each default warns.
 * Validation guarantees a dedicated device has a link or a `hostInterface`.
 */
export function resolveHost(hardware: HardwareStack, device: DeviceSpec, tiers: Set<OffloadTier>): ResolvedHost {
  const warnings: string[] = [];
  const out: ResolvedHost = { warnings };
  const host = hardware.host;
  if (device.memory.kind === 'dedicated') {
    if (host?.link !== undefined) {
      out.link = host.link;
    } else {
      const iface = device.hostInterface;
      if (iface === undefined) throw new Error(`offload: no host link known for ${device.id}; set hardware.host.link`);
      out.link = hostPcieLink(iface.generation, iface.lanes);
      warnings.push(`offload: no host link given; using ${out.link.name} from ${device.name}'s host interface`);
    }
    if (host?.ram !== undefined) {
      out.ram = host.ram;
    } else {
      out.ram = DEFAULT_HOST_RAM;
      warnings.push(`offload: no host RAM given; using ${DEFAULT_HOST_RAM.name}`);
    }
  }
  if (tiers.has('nvme')) {
    if (host?.ssd !== undefined) {
      out.ssd = host.ssd;
    } else {
      out.ssd = defaultSsdFor(device);
      warnings.push(`offload: no SSD given; using ${out.ssd.name}`);
    }
    if (out.ssd.randomRead === undefined) {
      warnings.push(`offload: no random-read IOPS for ${out.ssd.id}; request rate from latency × queue depth`);
    }
  }
  return out;
}

/**
 * The transfer path of `tier` on one device (design doc §2.3): host RAM → link → device; NVMe → host RAM → link →
 * device on dedicated memory (the staging term uses host RAM's bandwidth); NVMe → unified memory directly (no link,
 * no staging). `queueDepth` is the deployment's request (capped by the tier); `nStream` the tier's sharing.
 */
export function transferPath(tier: OffloadTier, host: ResolvedHost, queueDepth: number | undefined, nStream: number): TransferPath {
  const spec = tier === 'host_ram' ? host.ram : host.ssd;
  if (spec === undefined) throw new Error(`offload: no ${tier === 'host_ram' ? 'host RAM' : 'SSD'} resolved for a transfer`);
  const path: TransferPath = { tier: spec, queueDepth: resolveQueueDepth(spec, queueDepth), nStream };
  if (host.link !== undefined) {
    path.link = { latencySec: host.link.latencySec, bandwidthPerDirectionBytesPerSec: host.link.bandwidthPerDirectionBytesPerSec };
  }
  if (tier === 'nvme' && host.ram !== undefined) path.hostStagingBandwidthBytesPerSec = host.ram.bandwidthBytesPerSec;
  return path;
}
