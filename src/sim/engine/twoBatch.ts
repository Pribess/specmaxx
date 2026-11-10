/**
 * Two-batch overlap (TBO, `DeploymentConfig.twoBatchOverlap`): each microbatch runs as two halves, lowered
 * separately, so one half's EP all-to-alls can run while the other half computes. The halves are real lowerings at
 * half the batch, so their costs come out of the per-op model: GEMMs at half the rows (shape efficiency, GEMV path),
 * every weight read once per half, the expected distinct experts per half, α paid per half, the activation peak of
 * the larger half. See docs/simulation-model.md §7 ("Two-batch overlap").
 */

import type { LoweredLayer, LoweredModel, LoweringContext } from '../lower/lower';

/** The warning of a TBO run whose microbatch cannot be split (one sequence, one token per step). */
export const TWO_BATCH_NOTHING_TO_SPLIT_WARNING =
  'two-batch overlap: one sequence per microbatch, nothing to split';

/** The two halves of one microbatch for two-batch overlap. */
export interface TwoBatchSplit {
  /** Lowering context of the first half (A). */
  a: LoweringContext;
  /** Lowering context of the second half (B). */
  b: LoweringContext;
  /**
   * True for a causal token split of one sequence's prompt (A = the earlier tokens, B = the later ones); false for a
   * sequence split.
   */
  tokenSplit: boolean;
}

/**
 * The two halves of a microbatch of b = `ctx.seqs` sequences, each `qLen` new tokens over `kvLen` keys:
 * - b ≥ 2, sequence split: A = ceil(b/2), B = floor(b/2) sequences, same qLen and kvLen;
 * - b = 1 and prefill with qLen ≥ 2, causal token split of the q new tokens (the later half attends to the earlier):
 *     A = { qLen: ceil(q/2), kvLen: kvLen − floor(q/2) },  B = { qLen: floor(q/2), kvLen }
 *   (with a prefix cache q = P_new and kvLen = P, so A also sees the cached prefix);
 * - otherwise (one sequence, one token per step: decode, or a one-token prefill): undefined, no split.
 * Counts and tokens.
 */
export function twoBatchHalves(ctx: LoweringContext): TwoBatchSplit | undefined {
  const b = ctx.seqs;
  if (b >= 2) {
    // Formula: seqs_A = ceil(b/2);  seqs_B = floor(b/2)
    return {
      a: { ...ctx, seqs: Math.ceil(b / 2) },
      b: { ...ctx, seqs: Math.floor(b / 2) },
      tokenSplit: false,
    };
  }
  const q = ctx.qLen;
  if (ctx.phase !== 'prefill' || q < 2) return undefined;
  // Formula: q_A = ceil(q/2), kv_A = kvLen − floor(q/2);  q_B = floor(q/2), kv_B = kvLen
  return {
    a: { ...ctx, qLen: Math.ceil(q / 2), kvLen: ctx.kvLen - Math.floor(q / 2) },
    b: { ...ctx, qLen: Math.floor(q / 2) },
    tokenSplit: true,
  };
}

/**
 * One lowered microbatch made of the two halves: per stage and per layer, the ops of half A followed by those of
 * half B (`ops = [...A.ops, ...B.ops]`), so the stage sums count both halves and the layer list stays the full
 * batch's. Exceptions:
 * - token split: the `head` layer is half B's alone (prefill computes the logits of each sequence's last position,
 *   which only B holds; `lower/head.ts: headTokens`);
 * - the stage sends (`sendToNext`, `sendToFirst`) are the full batch's: the activations are handed over once per
 *   microbatch.
 * `full` is the full-batch lowering of the same step; A and B must have its stages and layer names.
 */
export function mergeTwoBatch(
  full: LoweredModel,
  a: LoweredModel,
  b: LoweredModel,
  tokenSplit: boolean,
): LoweredModel {
  const stages = full.stages.map((fullStage, s) => {
    const stageA = a.stages[s];
    const stageB = b.stages[s];
    if (
      stageA.layers.length !== fullStage.layers.length ||
      stageB.layers.length !== fullStage.layers.length
    ) {
      throw new Error(`two-batch overlap: stage ${s} of the halves has other layers`);
    }
    const layers: LoweredLayer[] = fullStage.layers.map((layer, i) => {
      const layerA = stageA.layers[i];
      const layerB = stageB.layers[i];
      if (layerA.name !== layer.name || layerB.name !== layer.name) {
        throw new Error(`two-batch overlap: layer ${layer.name} differs between the halves`);
      }
      // Formula: ops = ops_A ++ ops_B   (token split: head = ops_B)
      const ops = tokenSplit && layer.name === 'head' ? layerB.ops : [...layerA.ops, ...layerB.ops];
      return { name: layer.name, layerIndex: layer.layerIndex, ops };
    });
    return { ...fullStage, layers };
  });
  return { stages };
}
