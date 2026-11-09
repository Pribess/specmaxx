import type { TierMemory } from '../../sim/engine/types';
import {
  formatBytes,
  HIGH_MEMORY_UTILISATION,
  memoryState,
  memoryVerdict,
  type MemoryState,
} from '../state';
import type { MemorySummary, StageMemory } from '../state/summary';

export { HIGH_MEMORY_UTILISATION, memoryState, type MemoryState };

/**
 * Per-device utilisation rounded to 0.1 percentage points. Clamp warnings above the threshold and
 * values below full capacity below 100%, so rounding cannot contradict the physical verdict.
 */
export function memoryUtilisation(stage: StageMemory): string {
  const utilisation = stage.totalBytes / stage.capacityBytes;
  let percent = Math.round(utilisation * 1000) / 10;
  if (memoryState(stage) === 'warning') {
    percent = Math.max(HIGH_MEMORY_UTILISATION * 100 + 0.1, percent);
  }
  if (utilisation < 1) percent = Math.min(99.9, percent);
  return `${percent}%`;
}

/** Stand-in for "a tier overflows" so the red icon renders without a failing stage. */
const DANGER_STAGE: StageMemory = {
  stage: 0,
  weightBytes: 0,
  sequenceStateBytes: 0,
  activationPeakBytes: 0,
  totalBytes: 1,
  capacityBytes: 0,
  fits: false,
};

/** Accessible triangular warning sign; comfortable stages render no icon. */
export function MemoryWarningIcon({
  stage,
  label,
}: {
  /** Per-device memory sizes in bytes and the physical fit verdict. */
  stage: StageMemory;
  /** Overrides the accessible label (e.g. to name a failing tier). */
  label?: string;
}) {
  const state = memoryState(stage);
  if (state === 'fits') return null;
  return (
    <svg
      className={`memory-warning-icon memory-${state}`}
      viewBox="0 0 24 24"
      width="20"
      height="20"
      role="img"
      aria-label={
        label ??
        (state === 'danger'
          ? 'Warning: does not fit'
          : `Warning: ${memoryUtilisation(stage)} of memory used`)
      }
    >
      <path d="M12 3 22 21H2Z" fill="currentColor" />
      <path d="M12 9v5" stroke="var(--paper)" strokeWidth="2" strokeLinecap="round" />
      <circle cx="12" cy="17.5" r="1.2" fill="var(--paper)" />
    </svg>
  );
}

const TIER_NAME: Record<TierMemory['tier'], string> = { host_ram: 'Host RAM', nvme: 'SSD' };

/** Short names of the failing tiers with their sizes, e.g. "SSD 2.2GB of 2.0GB". */
function failingTierText(tiers: TierMemory[]): string {
  return tiers
    .map((t) => `${TIER_NAME[t.tier]} ${formatBytes(t.bytes)} of ${formatBytes(t.capacityBytes)}`)
    .join(', ');
}

/**
 * Links a headline to the memory report: a red triangle when a stage or an off-device tier does not fit (the label
 * names the failing tier), an amber one for the fullest stage over the headroom threshold.
 */
export function HeroMemoryWarning({
  memory,
}: {
  /** The memory report: stages and off-device tiers. */
  memory: Pick<MemorySummary, 'stages' | 'tiers'>;
}) {
  const verdict = memoryVerdict(memory);
  if (verdict.state === 'fits') return null;
  const tierLabel =
    verdict.failingTiers.length > 0
      ? `Does not fit: ${failingTierText(verdict.failingTiers)}`
      : undefined;
  return (
    <a className="hero-memory-warning" href="#memory-fit" title={tierLabel ?? 'View memory fit'}>
      <MemoryWarningIcon
        stage={
          verdict.state === 'danger' && verdict.worstStage?.fits !== false
            ? DANGER_STAGE
            : (verdict.worstStage ?? DANGER_STAGE)
        }
        label={tierLabel}
      />
    </a>
  );
}
