/**
 * Number formatting for the UI (en-US separators, SI decimal units).
 */

/** Formats with `digits` fraction digits and thousands separators. */
function fixed(x: number, digits: number): string {
  return x.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
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

/** Bytes in SI units: `19.1GB`, `512MB`, `1.50TB`, `0B`. */
export function formatBytes(bytes: number): string {
  const units: [number, string][] = [
    [1e12, 'TB'],
    [1e9, 'GB'],
    [1e6, 'MB'],
    [1e3, 'kB'],
  ];
  for (const [scale, unit] of units) {
    if (Math.abs(bytes) >= scale) {
      const v = bytes / scale;
      return `${fixed(v, v >= 100 ? 0 : 1)}${unit}`;
    }
  }
  return `${fixed(bytes, 0)}B`;
}

/**
 * A duration in the largest unit that shows it as ≥ 0.1 (`1.23s`, `0.91ms`, `12.5µs`, `850ns`), with two
 * decimals below 10, one below 100, none above.
 */
export function formatSeconds(sec: number): string {
  const units: [number, string][] = [
    [1, 's'],
    [1e-3, 'ms'],
    [1e-6, 'µs'],
    [1e-9, 'ns'],
  ];
  for (const [scale, unit] of units) {
    if (Math.abs(sec) >= 0.1 * scale || unit === 'ns') {
      const v = sec / scale;
      return `${fixed(v, v >= 100 ? 0 : v >= 10 ? 1 : 2)}${unit}`;
    }
  }
  return `${sec}s`;
}

/** A parameter count: `27.4B`, `350M`, `1.02T`. */
export function formatParams(n: number): string {
  if (n >= 1e12) return `${fixed(n / 1e12, 2)}T`;
  if (n >= 1e9) return `${fixed(n / 1e9, 1).replace(/\.0$/, '')}B`;
  if (n >= 1e6) return `${fixed(n / 1e6, 0)}M`;
  return fixed(n, 0);
}

/** A share in [0, 1] as a percentage: `21%`, `0.4%`. */
export function formatShare(share: number): string {
  const pct = share * 100;
  return `${fixed(pct, pct >= 10 || pct === 0 ? 0 : 1)}%`;
}

/** A compute rate: `209.5TFLOP/s`, `1.04PFLOP/s`, `850GFLOP/s`. */
export function formatFlopsPerSec(x: number): string {
  if (x >= 1e15) return `${fixed(x / 1e15, 2)}PFLOP/s`;
  if (x >= 1e12) return `${fixed(x / 1e12, 1)}TFLOP/s`;
  return `${fixed(x / 1e9, 0)}GFLOP/s`;
}

/** A bandwidth: `1.79TB/s`, `64.0GB/s`. */
export function formatBytesPerSec(x: number): string {
  if (x >= 1e12) return `${fixed(x / 1e12, 2)}TB/s`;
  return `${fixed(x / 1e9, 1)}GB/s`;
}

/** Context tokens, using binary 1k = 1024 and 1M = 1,048,576; other values stay exact. */
export function formatContextTokens(n: number): string {
  if (n >= 1048576 && n % 1048576 === 0) return `${n / 1048576}M`;
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`;
  return fixed(n, 0);
}

/** Aggregate or per-sequence tokens/s quoted at its evaluated context length in tokens. */
export function formatRateAtContext(tokensPerSec: number, contextTokens: number): string {
  return `${formatTokensPerSec(tokensPerSec)} tok/s @ ${formatContextTokens(contextTokens)}`;
}
