/**
 * The deployment-scenario choices of the UI state (design doc docs/design/offload-specdec-latency.md §6.1, §6.2):
 * weight offloading (`OffloadChoice`), speculative decoding (`SpecDecChoice`) and the node's host (`HostChoice`),
 * plus the normalizers that keep a default value out of the state, so equal choices encode to equal URLs.
 *
 * Every knob is off by default: `{}` everywhere. With all of them off, `buildSimInput` emits no `deployment` and no
 * `hardware.host`, so the results are bit-identical to before (design doc §0). Pure TypeScript.
 */

import { SPEC_USE_CASES, type SpecUseCase } from '../../sim/engine/specdec';
import { DEFAULT_PREFETCH_ACCURACY } from '../../sim/engine/transfer';

/** An off-device storage tier: the node's host RAM or its NVMe SSD (engine `OffloadTier`). */
export type TierChoice = 'host_ram' | 'nvme';

/**
 * Expert hit-ratio model h(f) (engine `HitRatioModel`): `'uniform'` (h = f, the default), the skewed fits
 * `'skewed_classic'` (h = f^0.55, ≤ 64 experts) and `'skewed_fine_grained'` (h = f^0.2, ≥ 128 experts), or a custom h in
 * [0, 1] (dimensionless).
 */
export type HitRatioChoice = 'uniform' | 'skewed_classic' | 'skewed_fine_grained' | number;

/** Layout of a block-scaled table's scales (engine `lookupTables.scaleLayout`). */
export type ScaleLayoutChoice = 'separate' | 'pinned_host' | 'interleaved';

/** Weight offloading choices; `{}` = everything on the device. */
export interface OffloadChoice {
  /** Routed experts of MoE layers kept on a tier; absent = on the device. */
  experts?: {
    /** Tier holding the experts. */
    tier: TierChoice;
    /** First offloaded layer index (layers ≥ it are offloaded, llama.cpp `--n-cpu-moe`); absent = 0 (all). Zero-based. */
    fromLayer?: number;
    /** Fraction f of each offloaded layer's experts kept resident on the device, in (0, 1]; absent = 0. Dimensionless. */
    residentFraction?: number;
    /** Hit-ratio model; absent = `'uniform'`. */
    hitRatio?: HitRatioChoice;
  };
  /** Whole decoder layers streamed every step (llama.cpp `-ngl N`); absent = every layer on the device. */
  dense?: {
    /** Tier holding the streamed layers. */
    tier: TierChoice;
    /** Layers kept on the device (layers `[0, n)`); the rest `[n, L)` stream. Integer ≥ 0. Count. */
    layersOnDevice: number;
  };
  /** n-gram lookup tables (Qwen PLE, DeepSeek Engram) on a tier; absent = on the device. */
  tables?: {
    /** Tier holding the tables. */
    tier: TierChoice;
    /** Scale layout; absent = the tier's default (`'separate'` on host RAM, `'pinned_host'` on SSD). */
    scaleLayout?: ScaleLayoutChoice;
  };
  /** Outstanding read requests per device; absent = the tier's rated maximum. Integer ≥ 1. Count. */
  queueDepth?: number;
  /** SSD large-read rate: rated sequential speed (default) or the preset's measured low-queue-depth fraction. */
  ssdReadRate?: 'datasheet' | 'sustained';
  /**
   * Prefetch accuracy p (engine `OffloadConfig.prefetchAccuracy`): the predictor's recall, the share of the needed
   * experts it picks one layer ahead; absent = the default 0.8. In [0, 1]. Dimensionless.
   */
  prefetchAccuracy?: number;
}

/** Speculative-decoding method (engine `SpecDecodeConfig.method`). */
export type SpecDecMethod = 'mtp' | 'dflash2' | 'dspark';

/** Use case that pre-fills the acceptance ratio (design doc §5.1). `'agentic'` = agentic code / editing. */
export type SpecDecUseCase = SpecUseCase;

/** Every use case, in picker order (the engine's list). */
export const SPEC_DEC_USE_CASES: readonly SpecDecUseCase[] = SPEC_USE_CASES;

/** The id of the labelled hypothetical drafter in {@link SpecDecChoice.drafter} (URL `sdr=hyp`). */
export const HYPOTHETICAL_DRAFTER = 'hyp';

/** Speculative-decoding choices; absent in the state = off. */
export interface SpecDecChoice {
  /** The method. */
  method: SpecDecMethod;
  /**
   * Block methods only: a drafter repo id `owner/name`, or {@link HYPOTHETICAL_DRAFTER}; absent = the known drafter of
   * the target, else the hypothetical one.
   */
  drafter?: string;
  /** Draft tokens per step k, integer in 1..16; absent = the method's default (1 for single-step MTP heads). Count. */
  k?: number;
  /** Use case for the preset acceptance; absent = `'prose'`. Ignored while `alpha` is set. */
  useCase?: SpecDecUseCase;
  /** User acceptance α in [0, 0.99] (the use case becomes "custom"); absent = the preset. Dimensionless. */
  alpha?: number;
}

/** The deployment-scenario choices; `{}` = every knob off. */
export interface DeploymentChoice {
  /** Weight offloading; absent or `{}` = off. */
  offload?: OffloadChoice;
  /** Speculative decoding; absent = off. */
  specDec?: SpecDecChoice;
}

/** A PCIe host link override: generation and electrical lanes (engine `HostInterfaceSpec`). */
export interface HostLinkChoice {
  /** PCIe generation. */
  generation: 3 | 4 | 5;
  /** Electrical lanes. Count. */
  lanes: 4 | 8 | 16;
}

/** The node's host: RAM and SSD presets and the host-link override; `{}` = the defaults. */
export interface HostChoice {
  /** `HostMemorySpec.id` of a preset in `HOST_RAM_PRESETS`; absent = DDR5-6000 × 2 channels. */
  ramId?: string;
  /** Installed host-RAM capacity override; absent = the preset's. Unit: bytes (whole GB, 1e9 B, in the URL). */
  ramCapacityBytes?: number;
  /** `NvmeSpec.id` of a preset in `SSD_PRESETS`; absent = the device's own SSD (unified memory) or the 990 PRO. */
  ssdId?: string;
  /** Host-link override; absent = the device's `hostInterface`. */
  link?: HostLinkChoice;
}

/** True when the choice moves any weight group off the device. */
export function offloadsAnything(offload: OffloadChoice | undefined): boolean {
  return (
    offload !== undefined &&
    (offload.experts !== undefined || offload.dense !== undefined || offload.tables !== undefined)
  );
}

/** True when any offloaded weight group uses the SSD. */
export function offloadsToSsd(offload: OffloadChoice | undefined): boolean {
  return [offload?.experts, offload?.dense, offload?.tables].some(
    (group) => group?.tier === 'nvme',
  );
}

/** The default scale layout of a tier (design doc §2.10): `'separate'` on host RAM, `'pinned_host'` on SSD. */
export function defaultScaleLayout(tier: TierChoice): ScaleLayoutChoice {
  return tier === 'host_ram' ? 'separate' : 'pinned_host';
}

/** `record` without its undefined fields (a new object). */
function compact<T extends object>(record: T): T {
  const out = {} as T;
  for (const key of Object.keys(record) as (keyof T)[])
    if (record[key] !== undefined) out[key] = record[key];
  return out;
}

/**
 * The offload choice with defaults removed (fromLayer 0, f 0, uniform h, the tier's scale layout) and a queue depth
 * and a prefetch accuracy (without the default 0.8) kept only while something is offloaded; `{}` when nothing is.
 */
export function normalizeOffload(o: OffloadChoice): OffloadChoice {
  const experts =
    o.experts &&
    compact({
      tier: o.experts.tier,
      fromLayer: o.experts.fromLayer === 0 ? undefined : o.experts.fromLayer,
      residentFraction: o.experts.residentFraction === 0 ? undefined : o.experts.residentFraction,
      hitRatio: o.experts.hitRatio === 'uniform' ? undefined : o.experts.hitRatio,
    });
  const tables =
    o.tables &&
    compact({
      tier: o.tables.tier,
      scaleLayout:
        o.tables.scaleLayout === defaultScaleLayout(o.tables.tier)
          ? undefined
          : o.tables.scaleLayout,
    });
  const out = compact({ experts, dense: o.dense && { ...o.dense }, tables });
  const prefetchAccuracy =
    o.prefetchAccuracy === DEFAULT_PREFETCH_ACCURACY ? undefined : o.prefetchAccuracy;
  return offloadsAnything(out)
    ? compact({
        ...out,
        queueDepth: o.queueDepth,
        ssdReadRate: offloadsToSsd(out) && o.ssdReadRate === 'sustained' ? 'sustained' : undefined,
        prefetchAccuracy,
      })
    : {};
}

/** The spec-dec choice with defaults removed (use case `'prose'`; the use case while α is set). */
export function normalizeSpecDec(s: SpecDecChoice): SpecDecChoice {
  return compact({
    method: s.method,
    drafter: s.method === 'mtp' ? undefined : s.drafter,
    k: s.k,
    useCase: s.alpha !== undefined || s.useCase === 'prose' ? undefined : s.useCase,
    alpha: s.alpha,
  });
}

/** The deployment choice with defaults removed; `{}` when every knob is off. */
export function normalizeDeployment(d: DeploymentChoice): DeploymentChoice {
  const offload = d.offload && normalizeOffload(d.offload);
  return compact({
    offload: offload && offloadsAnything(offload) ? offload : undefined,
    specDec: d.specDec && normalizeSpecDec(d.specDec),
  });
}

/** The host choice without undefined fields; `{}` = every default. */
export function normalizeHost(h: HostChoice): HostChoice {
  return compact({ ...h, link: h.link && { ...h.link } });
}
