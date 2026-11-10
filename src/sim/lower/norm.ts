/**
 * Lowering of norm, residual and residual-mix blocks (memory-bound elementwise ops over the residual stream).
 * All are replicated on every TP rank (no sequence parallelism). See docs/simulation-model.md §5
 * "Norms and residuals" and "Attention Residuals".
 */

import type { Op } from '../ir/ops';
import type { NormBlock, ResidualBlock, ResidualMixBlock } from '../ir/model';
import { actBpe, elementwise, hpBpe, normOp, stepTokens } from './common';
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
      {
        elems,
        activationReadBytes: 2 * elems * actBpe(ctx),
        activationWriteBytes: elems * actBpe(ctx),
      },
      ctx,
    ),
  ];
}

/**
 * A residual-mix block (Attention Residuals) → one `residual_mix` over `T·dim` output elements: read the C candidate
 * vectors `C·T·dim·bpe(act)`, write `T·dim·bpe(act)`, weights `2·dim·bpe(hp)` (score norm γ and score projection),
 * `6·C` FLOP per output element (see `ELEMENTWISE_FLOPS_PER_ELEM`). Replicated on every TP rank, like a norm.
 */
export function lowerResidualMix(block: ResidualMixBlock, ctx: LoweringContext): Op[] {
  const C = block.candidates;
  const elems = stepTokens(ctx) * block.dim;
  // Formula: elems = T·dim; flopsPerElem = 6·C; read = C·T·dim·bpe(act); write = T·dim·bpe(act); weights = 2·dim·bpe(hp)
  // See docs/simulation-model.md §5 "Attention Residuals"
  return [
    elementwise(
      block.name,
      'residual_mix',
      {
        elems,
        flopsPerElem: 6 * C,
        activationReadBytes: C * elems * actBpe(ctx),
        activationWriteBytes: elems * actBpe(ctx),
        weightBytes: 2 * block.dim * hpBpe(ctx),
      },
      ctx,
    ),
  ];
}
