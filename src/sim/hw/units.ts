/**
 * Unit helpers for writing hardware numbers as vendors quote them. All are SI DECIMAL (powers of 10, not 2):
 * 1 GB = 1e9 B, 1 TB/s = 1e12 B/s, 1 TFLOP/s = 1e12 FLOP/s. Binary sizes (GiB, KiB) must be spelled out
 * explicitly at the use site.
 */

/** TFLOP/s (or TOP/s) → FLOP/s (or OP/s). SI decimal: ×1e12. */
export function tflops(x: number): number {
  return x * 1e12;
}

/** GB/s → B/s. SI decimal: ×1e9. */
export function gbPerSec(x: number): number {
  return x * 1e9;
}

/** TB/s → B/s. SI decimal: ×1e12. */
export function tbPerSec(x: number): number {
  return x * 1e12;
}

/** GB → bytes. SI decimal: ×1e9. */
export function gb(x: number): number {
  return x * 1e9;
}

/** MB → bytes. SI decimal: ×1e6. */
export function mb(x: number): number {
  return x * 1e6;
}

/** Microseconds → seconds. SI decimal: ×1e-6. */
export function usec(x: number): number {
  return x * 1e-6;
}
