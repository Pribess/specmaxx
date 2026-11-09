/**
 * Engine dispatch overlay (review F1b): an informational figure next to the theoretical-maximum decode rate, never
 * folded into it. It is the project's one engine-overhead exception, opt-in only (`SimInput.overlays.engineDispatch`).
 * See docs/simulation-model.md §7 ("Engine dispatch overlay") and docs/modeling-method.md.
 *
 * Model: every IR op of a decode step is one kernel (no fusion), and each kernel costs a fixed dispatch time τ that
 * does not overlap the device's work (at batch 1 the device waits for the next launch). Per stage s:
 *   n_s  = Σ over the stage's layers of the number of IR ops   (kernels; PP sends and the token return not counted)
 *   t_s' = t_s + n_s · τ
 * The pipeline rule (docs §7 "Pipeline and DP") then runs on the t_s' unchanged.
 */

import type { StageResult } from './types';

/**
 * Default dispatch cost τ per kernel: 2 µs, inside the measured 1.3–2.9 µs range of docs/review/practicality-review.md
 * F1. Sources: Hazy Research, "Look Ma, No Bubbles!" (2025-05-27,
 * https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles): kernel launch ≈ 2.1 µs with CUDA streams, ≈ 1.3 µs
 * with CUDA graphs; mpk-apple research notes
 * (https://github.com/jiazhihao/mpk-apple/blob/main/docs/research/apple-inference-systems.md): MLX ≈ 2.75 µs per tiny
 * dependent kernel. Engine and runtime overhead, not hardware. Unit: seconds.
 */
export const DEFAULT_DISPATCH_SEC_PER_KERNEL = 2e-6;

/** Kernels one stage dispatches per microbatch step: its IR ops, one kernel each (no fusion). Count. */
export function stageKernelCount(stage: StageResult): number {
  // Formula: n_s = Σ_layers |ops|
  return stage.layers.reduce((n, layer) => n + layer.ops.length, 0);
}

/**
 * The stage with the dispatch cost of its kernels added to its time: `t_s' = t_s + n_s · τ` (`perKernelSec` = τ,
 * seconds). Every other field is the stage's own (the per-op times, comm and transfer terms do not change).
 */
export function withDispatch(stage: StageResult, perKernelSec: number): StageResult {
  // Formula: t_s' = t_s + n_s·τ
  // See docs/simulation-model.md §7 ("Engine dispatch overlay")
  const dispatchSec = stageKernelCount(stage) * perKernelSec;
  return { ...stage, sec: stage.sec + dispatchSec };
}
