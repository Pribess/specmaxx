/**
 * Lowering of norm and residual blocks (memory-bound elementwise ops over the residual stream).
 * Both are replicated on every TP rank (no sequence parallelism). See docs/simulation-model.md §5
 * "Norms and residuals".
 */

import type { Op } from '../ir/ops';
import type { NormBlock, ResidualBlock } from '../ir/model';
import { actBpe, elementwise, normOp, stepTokens } from './common';
import type { LoweringContext } from './lower';

/** A norm block → one elementwise norm over `T·dim` (read and write `T·dim·bpe(act)`, gamma `dim·bpe(hp)`). */
export function lowerNorm(block: NormBlock, ctx: LoweringContext): Op[] {
  return [normOp(block.name, block.norm, stepTokens(ctx), block.dim, ctx)];
}

/** A residual block → one `residual_add` over `T·dim`: read `2·T·dim·bpe(act)`, write `T·dim·bpe(act)`. */
export function lowerResidual(block: ResidualBlock, ctx: LoweringContext): Op[] {
  const elems = stepTokens(ctx) * block.dim;
  // Formula: read = 2·T·dim·bpe(act); write = T·dim·bpe(act)
  // See docs/simulation-model.md §5 "Norms and residuals"
  return [
    elementwise(
      block.name,
      'residual_add',
      { elems, activationReadBytes: 2 * elems * actBpe(ctx), activationWriteBytes: elems * actBpe(ctx) },
      ctx,
    ),
  ];
}
