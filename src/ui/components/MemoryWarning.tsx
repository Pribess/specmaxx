import type { StageMemory } from '../state/summary';

/** Fraction of per-device capacity above which a fitting stage has little headroom (dimensionless). */
export const HIGH_MEMORY_UTILISATION = 0.9;

/** Memory severity, ordered conceptually from comfortable fit to capacity overflow. */
export type MemoryState = 'fits' | 'warning' | 'danger';

/** Classifies one stage using its physical fit verdict and total/capacity byte ratio. */
export function memoryState(stage: StageMemory): MemoryState {
  if (!stage.fits) return 'danger';
  return stage.totalBytes / stage.capacityBytes > HIGH_MEMORY_UTILISATION ? 'warning' : 'fits';
}

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

/** Accessible triangular warning sign; comfortable stages render no icon. */
export function MemoryWarningIcon({
  stage,
}: {
  /** Per-device memory sizes in bytes and the physical fit verdict. */
  stage: StageMemory;
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
        state === 'danger'
          ? 'Warning: does not fit'
          : `Warning: ${memoryUtilisation(stage)} of memory used`
      }
    >
      <path d="M12 3 22 21H2Z" fill="currentColor" />
      <path d="M12 9v5" stroke="var(--paper)" strokeWidth="2" strokeLinecap="round" />
      <circle cx="12" cy="17.5" r="1.2" fill="var(--paper)" />
    </svg>
  );
}

/** Links a headline to the memory report, showing the worst stage (overflow before high utilisation). */
export function HeroMemoryWarning({
  stages,
}: {
  /** All pipeline stages, each reporting one device's memory sizes in bytes. */
  stages: StageMemory[];
}) {
  const stage =
    stages.find((s) => memoryState(s) === 'danger') ??
    stages
      .filter((s) => memoryState(s) === 'warning')
      .sort((a, b) => b.totalBytes / b.capacityBytes - a.totalBytes / a.capacityBytes)[0];
  if (!stage) return null;
  return (
    <a className="hero-memory-warning" href="#memory-fit" title="View memory fit">
      <MemoryWarningIcon stage={stage} />
    </a>
  );
}
