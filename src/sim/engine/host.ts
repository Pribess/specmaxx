/**
 * Resolving weight offloading for one run: the deployment's `OffloadConfig` → the placement lowering applies
 * (`OffloadLowering`, every default filled in), and the stack's `HardwareStack.host` → the host link, host RAM and
 * SSD the transfers use (defaults with a warning). `hardware.host` is read only when something is off-device.
 * See docs/simulation-model.md §12 ("Storage tiers", "Placement") and design doc §2.1–2.2.
 */

import {
  APPLE_SSD_M3_M4_MAX_1TB,
  APPLE_SSD_M5_PRO_MAX,
  defaultHostRamFor,
  DEFAULT_SSD,
  DGX_SPARK_SSD_4TB,
  NVME_GEN4_X4_GENERIC,
} from '../hw/presets/storage';
import { deviceHostLink } from '../hw/presets/links';
import type { DeviceSpec, HardwareStack, HostMemorySpec, LinkSpec, NvmeSpec } from '../hw/types';
import type { ModelSpec } from '../ir/model';
import type { ExpertPlacement, OffloadLowering, TablePlacement } from '../lower/lower';
import { expertRules, lookupTableRules } from './offloadRules';
import { DEFAULT_PREFETCH_ACCURACY, hitRatio, resolveQueueDepth, type TransferPath } from './transfer';
import type { LayerRange, OffloadConfig, OffloadTier } from './types';

/** The resolved placement and the notes it produced. */
export interface ResolvedOffload {
  /** What lowering applies; undefined when nothing ends up off the device (identity). */
  offload: OffloadLowering | undefined;
  /** Human-readable notes (approximations, placements that are device memory). */
  warnings: string[];
}

/**
 * Resolve the deployment's offload placement (design doc §2.1, §2.11): each routed-expert and lookup-table rule
 * becomes one placement (a lone rule's layer range defaults to every layer; h(f) is evaluated per expert rule from its
 * own hit-ratio model and f), sorted by `layers.from` so the order the user listed the rules in never changes a result.
 * The prefetch accuracy defaults to `DEFAULT_PREFETCH_ACCURACY`; a table rule's scale layout defaults to `'separate'`
 * on host RAM and `'pinned_host'` on NVMe (also on unified memory, where the pinned scales sit in the device pool and
 * count in the device bytes: `hostRamIsDevice`). On unified memory host RAM IS the device pool, so each `'host_ram'`
 * rule is dropped (no transfer, its layers' weights stay counted on the device) and only `'nvme'` rules remain. Table
 * rules at `'device'` are no placement. The token-embedding rule (`config.embeddings`) becomes `offload.embeddings`
 * when its tier is off the device and not dropped the same way (docs/simulation-model.md §12 "Token embedding
 * off-device"). Nothing left off-device → `offload: undefined` (identity).
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
  const offload: OffloadLowering = { prefetchAccuracy: config.prefetchAccuracy ?? DEFAULT_PREFETCH_ACCURACY };
  const expertPlacements: ExpertPlacement[] = [];
  const skewedRules: { range: LayerRange; f: number }[] = [];
  for (const rule of expertRules(config)) {
    if (onDevice(rule.tier)) continue;
    const f = rule.residentFraction;
    const layers = rule.layers ?? { from: 0, to: model.layers.length };
    expertPlacements.push({
      tier: rule.tier,
      layers,
      residentFraction: f,
      hitRatio: hitRatio(rule.hitRatio, f),
      includeDraft: rule.includeDraft ?? false,
    });
    if (rule.hitRatio.kind === 'skewed_classic' || rule.hitRatio.kind === 'skewed_fine_grained') {
      skewedRules.push({ range: layers, f });
    }
  }
  if (expertPlacements.length > 0) {
    offload.routedExperts = expertPlacements.sort((a, b) => a.layers.from - b.layers.from);
    if (ep > 1) {
      warnings.push(
        'offload: with EP > 1, the experts each GPU must fetch per step are estimated as if every routed pick were independent (U_dev = E_loc·(1 − (1 − 1/E_loc)^A))',
      );
    }
    // A skewed rule whose layers keep fewer resident experts than top-k: LRU thrashing, per rule over its own range.
    const thrashing = skewedRules.some(({ range, f }) =>
      model.layers.some(
        (l, i) =>
          i >= range.from &&
          i < range.to &&
          l.blocks.some((b) => b.kind === 'moe' && f * (b.experts / ep) > 0 && f * (b.experts / ep) < b.topK),
      ),
    );
    if (thrashing) {
      warnings.push(
        'offload: fewer resident experts per layer than top-k — an LRU cache thrashes there (h ≈ 0); the skewed hit ratio is optimistic',
      );
    }
  }
  const tablePlacements: TablePlacement[] = [];
  for (const rule of lookupTableRules(config)) {
    const tier = rule.tier;
    if (tier === 'device' || onDevice(tier)) continue;
    tablePlacements.push({
      tier,
      layers: rule.layers ?? { from: 0, to: model.layers.length },
      scaleLayout: rule.scaleLayout ?? (tier === 'host_ram' ? 'separate' : 'pinned_host'),
    });
  }
  if (tablePlacements.length > 0) {
    offload.lookupTables = tablePlacements.sort((a, b) => a.layers.from - b.layers.from);
  }
  const embeddingTier = config.embeddings?.tier;
  if (embeddingTier !== undefined && embeddingTier !== 'device' && !onDevice(embeddingTier)) {
    offload.embeddings = { tier: embeddingTier };
  }
  const any =
    offload.routedExperts !== undefined || offload.lookupTables !== undefined || offload.embeddings !== undefined;
  if (any && memoryKind === 'unified') {
    offload.hostRamIsDevice = true;
    if (offload.lookupTables?.some((t) => t.scaleLayout === 'pinned_host') === true) {
      warnings.push('offload: unified memory — pinned table scales stay in the unified pool and count in device memory');
    }
  }
  return { offload: any ? offload : undefined, warnings: [...new Set(warnings)] };
}

/** The off-device tiers a resolved placement uses (the union over every rule of every group). */
export function offloadTiers(offload: OffloadLowering): Set<OffloadTier> {
  const tiers = new Set<OffloadTier>();
  for (const rule of offload.routedExperts ?? []) tiers.add(rule.tier);
  for (const rule of offload.lookupTables ?? []) tiers.add(rule.tier);
  if (offload.embeddings !== undefined) tiers.add(offload.embeddings.tier);
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
 * `deviceHostLink(device)` (non-PCIe attachment before PCIe); `host.ram` absent → size-aware Grace RAM on GB200/GB300,
 * else unchanged `DDR5_6000_2CH` (needed for host-RAM groups and as the
 * staging hop of NVMe reads). `host.ssd` absent with an NVMe group → {@link defaultSsdFor}. Each default warns.
 * Validation guarantees a dedicated device has a link, `hostLink`, or a `hostInterface` (docs/simulation-model.md §4).
 */
export function resolveHost(hardware: HardwareStack, device: DeviceSpec, tiers: Set<OffloadTier>): ResolvedHost {
  const warnings: string[] = [];
  const out: ResolvedHost = { warnings };
  const host = hardware.host;
  if (device.memory.kind === 'dedicated') {
    if (host?.link !== undefined) {
      out.link = host.link;
    } else {
      const link = deviceHostLink(device);
      if (link === undefined) throw new Error(`offload: no host link known for ${device.id}; set hardware.host.link`);
      out.link = link;
      warnings.push(`offload: no host link given; using ${out.link.name} from ${device.name}'s host interface`);
    }
    if (host?.ram !== undefined) {
      out.ram = host.ram;
    } else {
      out.ram = defaultHostRamFor(device, hardware.topology.devicesPerNode);
      warnings.push(`offload: no host RAM given; using ${out.ram.name}`);
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
 * no staging). `queueDepth` is the deployment's request (capped by the tier); `nStream` the tier's sharing (n_host for
 * the host-RAM tier); `hostStagingNStream` the host RAM sharing n_host of the NVMe staging hop (absent = `nStream`;
 * docs/simulation-model.md §12 "Sharing"). `ssdReadRate` opts into the SSD's large-read fraction; absent/datasheet
 * leaves all B/s ceilings unchanged.
 */
export function transferPath(
  tier: OffloadTier,
  host: ResolvedHost,
  queueDepth: number | undefined,
  nStream: number,
  ssdReadRate?: OffloadConfig['ssdReadRate'],
  hostStagingNStream?: number,
): TransferPath {
  const spec = tier === 'host_ram' ? host.ram : host.ssd;
  if (spec === undefined)
    throw new Error(
      `offload: no ${tier === 'host_ram' ? 'host RAM' : 'SSD'} resolved for a transfer`,
    );
  const path: TransferPath = {
    tier: spec,
    queueDepth: resolveQueueDepth(spec, queueDepth),
    nStream,
  };
  if (spec.kind === 'nvme' && ssdReadRate === 'sustained' && spec.sustainedRead !== undefined) {
    path.nvmeSustainedRead = spec.sustainedRead;
  }
  if (host.link !== undefined) {
    path.link = { latencySec: host.link.latencySec, bandwidthPerDirectionBytesPerSec: host.link.bandwidthPerDirectionBytesPerSec };
  }
  if (tier === 'nvme' && host.ram !== undefined) {
    path.hostStagingBandwidthBytesPerSec = host.ram.bandwidthBytesPerSec;
    if (hostStagingNStream !== undefined) path.hostStagingNStream = hostStagingNStream;
  }
  return path;
}
