/**
 * Lowering of hyper-connection steps (`hc_pre`, `hc_post`) over N residual streams of width D: Qwen4Exp "Gated
 * Residual" and DeepSeek/GLM mHC. REPLICATED on every TP rank, like norms (every rank holds all T tokens'
 * streams). Hyper-connections keep no per-sequence state. See docs/simulation-model.md §5 "Hyper-connections".
 *
 * Notation: N streams, D = dim, ND = N·D, T = seqs·qLen.
 */

import type { HcPostBlock, HcPreBlock } from '../ir/model';
import type { Op } from '../ir/ops';
import { actBpe, elementwise, hpBpe, linearGemm, stepTokens } from './common';
import type { LoweringContext } from './lower';

/**
 * hc_pre ops, in order:
 * 1. the stream norm over T·ND: with a learned γ (`normGamma`, Qwen) read and written, γ ND·bpe(hp); else (mHC)
 *    the RMS statistic only, read once;
 * 2. each projection as a GEMM (m = T, module `residual_mixer`);
 * 3. `sinkhorn` over T·N² elements, 4·iters FLOP each, if iters > 0;
 * 4. `hc_collapse`: read T·ND streams, write T·D, 2·N FLOP per output element; it carries the `vectorParams`
 *    weights (mHC base, scale). The mixing weights the projections produced are assumed consumed on chip.
 */
export function lowerHcPre(block: HcPreBlock, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  const N = block.streams;
  const D = block.dim;
  const act = actBpe(ctx);
  const streamElems = T * N * D;
  const ops: Op[] = [];

  // Formula: norm over T·N·D: γ ? (read + write T·N·D, γ N·D) : read T·N·D
  // See docs/simulation-model.md §5 "Hyper-connections"
  ops.push(
    elementwise(
      `${block.name}.norm`,
      'rmsnorm',
      {
        elems: streamElems,
        activationReadBytes: streamElems * act,
        activationWriteBytes: block.normGamma ? streamElems * act : 0,
        weightBytes: block.normGamma ? N * D * hpBpe(ctx) : 0,
      },
      ctx,
    ),
  );
  // Formula: projections: m = T, n = out, k = in (module residual_mixer)
  for (const p of block.projections) {
    ops.push(
      linearGemm(`${block.name}.${p.name}`, { m: T, n: p.outputDim, k: p.inputDim }, ctx, {
        module: 'residual_mixer',
      }),
    );
  }
  if (block.sinkhornIters > 0) {
    // Formula: sinkhorn: elems = T·N², flopsPerElem = 4·iters; read = write = T·N²·bpe(act)
    const elems = T * N * N;
    ops.push(
      elementwise(
        `${block.name}.sinkhorn`,
        'sinkhorn',
        {
          elems,
          flopsPerElem: 4 * block.sinkhornIters,
          activationReadBytes: elems * act,
          activationWriteBytes: elems * act,
        },
        ctx,
      ),
    );
  }
  // Formula: hc_collapse: elems = T·D, flopsPerElem = 2·N; read T·N·D, write T·D; weights vectorParams·bpe(hp)
  ops.push(
    elementwise(
      `${block.name}.collapse`,
      'hc_collapse',
      {
        elems: T * D,
        flopsPerElem: 2 * N,
        activationReadBytes: streamElems * act,
        activationWriteBytes: T * D * act,
        weightBytes: block.vectorParams * hpBpe(ctx),
      },
      ctx,
    ),
  );
  return ops;
}

/**
 * hc_post ops: one `hc_expand` over T·ND output elements, reading the streams T·ND and the sublayer output T·D,
 * writing T·ND; FLOP per element 2N + 2 when the streams are mixed (mHC: N comb terms + post·out), else 2 (Qwen:
 * res_i + w_i·out).
 */
export function lowerHcPost(block: HcPostBlock, ctx: LoweringContext): Op[] {
  const T = stepTokens(ctx);
  const N = block.streams;
  const D = block.dim;
  const act = actBpe(ctx);
  const streamElems = T * N * D;
  // Formula: hc_expand: elems = T·N·D, flopsPerElem = mixes ? 2N+2 : 2; read T·N·D + T·D, write T·N·D
  // See docs/simulation-model.md §5 "Hyper-connections"
  return [
    elementwise(
      `${block.name}.expand`,
      'hc_expand',
      {
        elems: streamElems,
        flopsPerElem: block.mixesStreams ? 2 * N + 2 : 2,
        activationReadBytes: (streamElems + T * D) * act,
        activationWriteBytes: streamElems * act,
      },
      ctx,
    ),
  ];
}
