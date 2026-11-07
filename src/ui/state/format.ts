/**
 * Number formatting for the UI (en-US separators, SI decimal units).
 */

/** Formats with `digits` fraction digits and thousands separators. */
function fixed(x: number, digits: number): string {
  return x.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/**
 * A rate in tokens/s, without the unit: ≥ 1000 → no decimals (`25,710`), ≥ 10 → one (`231.4`), else two
 * (`3.27`). Non-finite → `'—'`.
 */
export function formatTokensPerSec(x: number): string {
  if (!Number.isFinite(x)) return '—';
  return fixed(x, x >= 1000 ? 0 : x >= 10 ? 1 : 2);
}

/** A compact rate for tight spaces: `25.7k`, `1.2M`, `231`, `3.3`. */
export function formatTokensPerSecCompact(x: number): string {
  if (!Number.isFinite(x)) return '—';
  if (x >= 1e6) return `${fixed(x / 1e6, 1)}M`;
  if (x >= 1e3) return `${fixed(x / 1e3, 1)}k`;
  return fixed(x, x >= 100 ? 0 : 1);
}

/** Bytes in SI units: `19.1 GB`, `512 MB`, `1.50 TB`, `0 B`. */
export function formatBytes(bytes: number): string {
  const units: [number, string][] = [[1e12, 'TB'], [1e9, 'GB'], [1e6, 'MB'], [1e3, 'kB']];
  for (const [scale, unit] of units) {
    if (Math.abs(bytes) >= scale) {
      const v = bytes / scale;
      return `${fixed(v, v >= 100 ? 0 : 1)} ${unit}`;
    }
  }
  return `${fixed(bytes, 0)} B`;
}

/**
 * A duration in the largest unit that shows it as ≥ 0.1 (`1.23 s`, `0.91 ms`, `12.5 µs`, `850 ns`), with two
 * decimals below 10, one below 100, none above.
 */
export function formatSeconds(sec: number): string {
  const units: [number, string][] = [[1, 's'], [1e-3, 'ms'], [1e-6, 'µs'], [1e-9, 'ns']];
  for (const [scale, unit] of units) {
    if (Math.abs(sec) >= 0.1 * scale || unit === 'ns') {
      const v = sec / scale;
      return `${fixed(v, v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${unit}`;
    }
  }
  return `${sec} s`;
}

/** A parameter count: `27.4 B`, `350 M`, `1.02 T`. */
export function formatParams(n: number): string {
  if (n >= 1e12) return `${fixed(n / 1e12, 2)} T`;
  if (n >= 1e9) return `${fixed(n / 1e9, 1)} B`;
  if (n >= 1e6) return `${fixed(n / 1e6, 0)} M`;
  return fixed(n, 0);
}

/** A share in [0, 1] as a percentage: `21%`, `0.4%`. */
export function formatShare(share: number): string {
  const pct = share * 100;
  return `${fixed(pct, pct >= 10 || pct === 0 ? 0 : 1)}%`;
}

/** A compute rate: `209.5 TFLOP/s`, `1.04 PFLOP/s`, `850 GFLOP/s`. */
export function formatFlopsPerSec(x: number): string {
  if (x >= 1e15) return `${fixed(x / 1e15, 2)} PFLOP/s`;
  if (x >= 1e12) return `${fixed(x / 1e12, 1)} TFLOP/s`;
  return `${fixed(x / 1e9, 0)} GFLOP/s`;
}

/** A bandwidth: `1.79 TB/s`, `64.0 GB/s`. */
export function formatBytesPerSec(x: number): string {
  if (x >= 1e12) return `${fixed(x / 1e12, 2)} TB/s`;
  return `${fixed(x / 1e9, 1)} GB/s`;
}
