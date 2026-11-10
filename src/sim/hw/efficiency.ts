/**
 * Shape-aware tensor-core time: how much of the peak a GEMM of a given shape can use, from MMA padding and
 * wave quantization alone (deterministic; no calibration).
 * See docs/simulation-model.md §4 ("Shape efficiency").
 *
 * Not modeled (documented): split-K / stream-K (pessimistic for few-tile shapes, which are memory-bound
 * anyway) and shared-memory bandwidth penalties of small tiles.
 */

import type { GemmTiling } from './types';

/** A (batched) GEMM shape: `batch` independent `m×k · k×n` products. Unit: elements (counts). */
export interface GemmShape {
  /** Independent GEMMs (grouped experts, attention heads × sequences). Count. */
  batch: number;
  /** Rows of the output (usually tokens). Unit: elements. */
  m: number;
  /** Columns of the output. Unit: elements. */
  n: number;
  /** Reduction length. Unit: elements. */
  k: number;
}

/** Result of {@link gemmComputeSec}. */
export interface GemmTimeResult {
  /** Tensor compute time of the GEMM on one device. Unit: seconds. */
  computeSec: number;
  /** Useful FLOP / (peak · computeSec), in (0, 1]. Dimensionless. */
  efficiency: number;
  /** The CTA tile that won (after shrinking to the padded dims), on the (A, B) sides below. Unit: elements. */
  tile: { m: number; n: number };
  /** True if the winning layout put the GEMM's n on the MMA M side (swap-AB). */
  swapped: boolean;
  /** CTA tiles of the winning layout, over all batch entries. Count. */
  tiles: number;
  /** Waves of the winning layout: ceil(tiles / concurrent CTA slots). Count. */
  waves: number;
}

/** Smallest multiple of `step` that is ≥ `x` (elements). */
function roundUp(x: number, step: number): number {
  return Math.ceil(x / step) * step;
}

/**
 * Tensor compute time of a (batched) GEMM on one device, per docs §4 "Shape efficiency".
 *
 * `peakFlopsPerSec` is the device's matmul peak for the op's key (FLOP/s, per device); `computeUnits` the
 * number of SMs/CUs; `tiling` the engine's tiling parameters for the op's input class.
 * Both layouts (m on the MMA M side, or swapped) and every candidate CTA tile are tried; the fastest wins
 * (ties keep the unswapped layout and the earlier tile). A GEMM with no work returns 0 s and efficiency 1.
 */
export function gemmComputeSec(
  shape: GemmShape,
  peakFlopsPerSec: number,
  computeUnits: number,
  tiling: GemmTiling,
): GemmTimeResult {
  const { batch, m, n, k } = shape;
  // Formula: F_useful = 2 · batch · m · n · k
  // See docs/simulation-model.md §4 ("Shape efficiency")
  const usefulFlops = 2 * batch * m * n * k;
  if (usefulFlops === 0) {
    return {
      computeSec: 0,
      efficiency: 1,
      tile: { m: 0, n: 0 },
      swapped: false,
      tiles: 0,
      waves: 0,
    };
  }

  // Formula: P = computeUnits · ctasPerUnit   (concurrent CTA slots per wave)
  const slots = computeUnits * tiling.ctasPerUnit;
  // Formula: peak per slot = peak / P
  const peakPerSlotFlopsPerSec = peakFlopsPerSec / slots;
  // Formula: K' = roundUp(k, mmaK)
  const paddedK = roundUp(k, tiling.mmaK);

  let best: GemmTimeResult | null = null;
  const layouts: { a: number; b: number; swapped: boolean }[] = [
    { a: m, b: n, swapped: false },
    { a: n, b: m, swapped: true }, // swap-AB: the small token dim may sit on the MMA N side
  ];
  for (const layout of layouts) {
    // Formula: A' = roundUp(A, mmaM);  B' = roundUp(B, mmaN)
    const paddedA = roundUp(layout.a, tiling.mmaM);
    const paddedB = roundUp(layout.b, tiling.mmaN);
    for (const t of tiling.ctaTiles) {
      // Formula: tm = min(t.m, A');  tn = min(t.n, B')   (kernels shrink tiles to small dims)
      const tileM = Math.min(t.m, paddedA);
      const tileN = Math.min(t.n, paddedB);
      // Formula: tiles = batch · ceil(A'/tm) · ceil(B'/tn);  waves = ceil(tiles / P)
      const tiles = batch * Math.ceil(paddedA / tileM) * Math.ceil(paddedB / tileN);
      const waves = Math.ceil(tiles / slots);
      // Formula: sec = waves · 2·tm·tn·K' / (peak / P)   (every slot runs full padded tiles at its share of the peak)
      const sec = (waves * 2 * tileM * tileN * paddedK) / peakPerSlotFlopsPerSec;
      if (best === null || sec < best.computeSec) {
        best = {
          computeSec: sec,
          efficiency: 0,
          tile: { m: tileM, n: tileN },
          swapped: layout.swapped,
          tiles,
          waves,
        };
      }
    }
  }
  if (best === null) throw new Error('gemmTiling.ctaTiles is empty');

  // Formula: efficiency = (F_useful / peak) / computeSec
  const idealSec = usefulFlops / peakFlopsPerSec;
  return { ...best, efficiency: idealSec / best.computeSec };
}

/**
 * Compute time of a GEMM run as a GEMV kernel (SIMD dot products on the vector ALUs, no MMA instruction), per docs
 * §4 "GEMV path": the useful FLOP `usefulFlops` (FLOP, `2·batch·m·n·k`) at the vector peak
 * `vectorPeakFlopsPerSec` (FLOP/s, per device) of the GEMM's math format. No MMA padding and no wave quantization,
 * the same rule as elementwise ops (one output per lane, every lane busy). Only used for `m ≤
 * ComputeEngine.gemvMaxRows`, where the op is memory-bound in practice. Unit: seconds.
 */
export function gemvComputeSec(usefulFlops: number, vectorPeakFlopsPerSec: number): number {
  // Formula: t_gemv = F / vectorPeak(math)
  // See docs/simulation-model.md §4 ("GEMV path")
  return usefulFlops / vectorPeakFlopsPerSec;
}
