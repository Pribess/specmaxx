/**
 * Lowering of the short depthwise causal conv plus residual on a branch output (`short_conv`, Inkling `use_sconv`):
 * one `causal_conv1d` over T·dim with the residual add fused. REPLICATED on every TP rank (it runs on the reduced
 * branch output). See docs/simulation-model.md §5 "Short convs".
 */

import type { ShortConvBlock } from '../ir/model';
import type { Op } from '../ir/ops';
import { actBpe, causalConvOp, convStateTokens, stepTokens } from './common';
import type { LoweringContext } from './lower';

/** short_conv ops: `causal_conv1d` over T·dim, 2·K + 1 FLOP per element (taps + fused residual add). */
export function lowerShortConv(block: ShortConvBlock, ctx: LoweringContext): Op[] {
  // Formula: conv over T·dim: flopsPerElem = 2·K + 1; taps dim·K·bpe(hp); state seqs·(K−1)·dim·bpe(act)
  //          (state written k + 1 times in a speculative verify: causalConvOp applies stateSnapshots)
  // See docs/simulation-model.md §5 "Short convs"
  return [
    causalConvOp(
      block.name,
      stepTokens(ctx),
      block.dim,
      { kernel: block.kernel, residual: true },
      ctx,
    ),
  ];
}

/** Per-sequence conv rolling state on one device: `(K−1)·dim·bpe(act)`. Unit: bytes. */
export function shortConvSequenceStateBytes(block: ShortConvBlock, ctx: LoweringContext): number {
  // Formula: S = (K−1)·dim·bpe(act)
  // See docs/simulation-model.md §8 "Memory fit"
  return convStateTokens({ kernel: block.kernel }) * block.dim * actBpe(ctx);
}
