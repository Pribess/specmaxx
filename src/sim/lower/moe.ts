/**
 * Lowering of mixture-of-experts blocks with balanced ("uniform") routing and two expert-parallel modes.
 * See docs/simulation-model.md §5 "MoE modes A/B".
 *
 * - **Mode A** (`tp % ep == 0`, includes ep = 1): the EP group lies inside the TP group. Every TP rank holds all
 *   T tokens; each device runs its `Eloc = E/ep` experts, each expert sharded `etp = tp/ep` ways; one TP
 *   all-reduce combines everything (routed and shared).
 * - **Mode B** (`ep % tp == 0`, `ep > tp`, `(tp·dp) % ep == 0`): the EP group spans `p = ep/tp` replicas. Each
 *   device dispatches its token slice `Ts = ceil(T/tp)` with all-to-alls; experts are unsharded (`etp = 1`).
 *
 * Latent MoE (`latentDim` l set, Muse Spark, Nemotron): a replicated `h → l` projection runs after the router and
 * before the dispatch, the routed experts, dispatch/combine and the weighted sum work at width l, and a replicated
 * `l → h` projection follows the weighted sum (in mode A on the TP partial sums, valid by linearity, so the single
 * all-reduce stays at h). Router and shared experts stay at h. Without `latentDim` nothing changes (l = h, no
 * projections).
 *
 * Routing is balanced: the busiest device gets the ceiling of its average share (`A` assignments), spread over
 * `a = min(Eloc, A)` active experts with `mE = ceil(A/a)` tokens each.
 *
 * Offloaded experts (deployment `offload.routedExperts`, design doc §2.4–2.5): a `TransferOp` of the expected missed
 * experts goes in front of the block; the grouped GEMMs keep their `a` active experts (device-memory bytes unchanged).
 */

import { gemmWeightBytes } from '../ir/metrics';
import { expertHitsMisses, uniqueExperts, uniqueLocalExperts } from '../engine/transfer';
import type { MoeBlock, ParallelConfig } from '../ir/model';
import type { CollectiveOp, Op, TransferOp } from '../ir/ops';
import {
  actBpe,
  elementwise,
  highPrecision,
  hpBpe,
  linearGemm,
  moduleQuant,
  reduceBpe,
  reduceDtype,
  shardWidth,
  stepTokens,
  tpAllReduce,
} from './common';
import { activationOp, lowerDenseFfn, type DenseFfnShape } from './mlp';
import type { LoweringContext, OffloadLowering } from './lower';
import { expertPlacementAt, prefetchedFraction } from './offload';

/** The expert-parallel layout of one MoE block under a parallel config. */
export interface MoeLayout {
  /** `'A'`: ep divides tp (experts inside the TP group); `'B'`: tp divides ep, ep > tp (EP across replicas). */
  mode: 'A' | 'B';
  /** Routed experts per device `E/ep`. Count. */
  localExperts: number;
  /** Ways each expert's intermediate width is sharded: `max(1, tp/ep)`. Count. */
  expertTp: number;
  /** Replicas pooled into one EP group: `max(1, ep/tp)`. Count. */
  pooledReplicas: number;
}

/**
 * Validation errors of a MoE block's expert parallelism (empty = valid): `experts % ep == 0` and either
 * `tp % ep == 0` (mode A) or (`ep % tp == 0` and `(tp·dp) % ep == 0`) (mode B).
 */
export function moeParallelErrors(block: MoeBlock, parallel: ParallelConfig): string[] {
  const { tp, dp, ep } = parallel;
  const errors: string[] = [];
  if (block.experts % ep !== 0) errors.push(`${block.name}: experts (${block.experts}) must be divisible by ep (${ep})`);
  const modeA = tp % ep === 0;
  const modeB = ep % tp === 0 && (tp * dp) % ep === 0;
  if (!modeA && !modeB) {
    errors.push(`${block.name}: ep (${ep}) must divide tp (${tp}), or be a multiple of tp that divides tp·dp (${tp * dp})`);
  }
  return errors;
}

/** The layout of a MoE block; throws on an invalid expert parallelism (see {@link moeParallelErrors}). */
export function moeLayout(block: MoeBlock, parallel: ParallelConfig): MoeLayout {
  const errors = moeParallelErrors(block, parallel);
  if (errors.length > 0) throw new Error(errors.join('\n'));
  const { tp, ep } = parallel;
  // Formula: Eloc = E/ep; etp = max(1, tp/ep); p = max(1, ep/tp)
  // See docs/simulation-model.md §5 "MoE modes A/B"
  return {
    mode: tp % ep === 0 ? 'A' : 'B',
    localExperts: block.experts / ep,
    expertTp: Math.max(1, tp / ep),
    pooledReplicas: Math.max(1, ep / tp),
  };
}

/** Balanced routing on the busiest device: active experts `a` and tokens per active expert `mE`. */
export interface MoeRouting {
  /** Expert assignments (token, expert) on this device. Count. */
  assignments: number;
  /** Active local experts `a = min(Eloc, A)`. Count. */
  activeExperts: number;
  /** Tokens per active expert `mE = ceil(A/a)`. Unit: tokens. */
  tokensPerExpert: number;
}

/**
 * Balanced routing of `tokens` routed tokens (mode A: T; mode B: Ts) on one device.
 * Mode A: `A = ceil(T·k/ep)`; mode B: `A = Ts·k`. Then `a = min(Eloc, A)`, `mE = ceil(A/a)`.
 */
export function moeRouting(block: MoeBlock, layout: MoeLayout, tokens: number, parallel: ParallelConfig): MoeRouting {
  // Formula: A = mode A ? ceil(T·k/ep) : Ts·k; a = min(Eloc, A); mE = ceil(A/a)
  // See docs/simulation-model.md §5 "MoE modes A/B"
  const assignments =
    layout.mode === 'A' ? Math.ceil((tokens * block.topK) / parallel.ep) : tokens * block.topK;
  const activeExperts = Math.min(layout.localExperts, assignments);
  const tokensPerExpert = Math.ceil(assignments / activeExperts);
  return { assignments, activeExperts, tokensPerExpert };
}

/** Shared experts of a MoE block as one dense gated MLP of width `S·fs` (no bias). */
function sharedFfnShape(block: MoeBlock): DenseFfnShape {
  return {
    hidden: block.hidden,
    intermediate: block.sharedExperts * block.sharedIntermediate,
    gated: block.gated,
    activation: block.activation,
    bias: false,
  };
}

/** An expert-parallel all-to-all of `messageBytes` per device over the EP group. */
function epAllToAll(name: string, messageBytes: number, dtype: CollectiveOp['dtype'], ep: number): CollectiveOp {
  return { kind: 'collective', name, collective: 'all_to_all', group: 'ep', groupSize: ep, messageBytes, dtype };
}

/**
 * Width l the routed experts of a MoE block read and write: `latentDim` for a latent MoE, else the hidden h. It is
 * also the width of the EP dispatch/combine messages and of the weighted sum. Unit: elements.
 */
export function moeExpertWidth(block: MoeBlock): number {
  return block.latentDim ?? block.hidden;
}

/**
 * Stored bytes of a latent MoE's two projections on one device: `h → l` (`n l, k h`) plus `l → h` (`n h, k l`)
 * in the `'moe_latent'` format, replicated (the same on every TP rank, never offloaded); 0 without `latentDim`.
 * Computed per GEMM like the ops of {@link lowerMoe}, so resident and streamed bytes agree. Unit: bytes.
 */
export function moeLatentWeightBytes(block: MoeBlock, ctx: LoweringContext): number {
  if (block.latentDim === undefined) return 0;
  const h = block.hidden;
  const l = block.latentDim;
  const fmt = moduleQuant(ctx, 'moe_latent');
  // Formula: W_latent = gemmWeightBytes(1, l, h, w_latent) + gemmWeightBytes(1, h, l, w_latent)  ≈ 2·h·l·bpe(w_latent)
  // See docs/simulation-model.md §5 "MoE modes A/B"
  return gemmWeightBytes(1, l, h, fmt.weights, fmt.sparsity) + gemmWeightBytes(1, h, l, fmt.weights, fmt.sparsity);
}

/**
 * Stored bytes of ONE routed expert on one device: its (gated ? 3 : 2) matrices of `l × fe/etp` in the routed-expert
 * format (`gemmWeightBytes`, incl. 2:4 metadata and side vectors), l = {@link moeExpertWidth}. Unit: bytes.
 */
export function expertWeightBytes(block: MoeBlock, ctx: LoweringContext): number {
  const layout = moeLayout(block, ctx.parallel);
  const routedFmt = moduleQuant(ctx, 'routed_experts');
  const feLocal = block.expertIntermediate / layout.expertTp;
  // Formula: W_expert = gemmWeightBytes(gated ? 3 : 2, l, fe/etp, w_routed),  l = latentDim ?? h
  // See docs/simulation-model.md §12 ("Expert hit ratio")
  return gemmWeightBytes(block.gated ? 3 : 2, moeExpertWidth(block), feLocal, routedFmt.weights, routedFmt.sparsity);
}

/**
 * Expected needed experts of one MoE layer on one device in a step: at ep = 1 the exact
 * `U = E·(1 − (1 − k/E)^T)` over the T routed tokens; at ep > 1 the per-device form
 * `U_dev = E_loc·(1 − (1 − 1/E_loc)^A)` over its A assignments (design doc §2.9). Unit: experts (real).
 */
export function neededExperts(block: MoeBlock, layout: MoeLayout, routing: MoeRouting, routedTokens: number, ep: number): number {
  return ep === 1
    ? uniqueExperts(block.experts, block.topK, routedTokens)
    : uniqueLocalExperts(layout.localExperts, routing.assignments);
}

/**
 * The expert fetch of an offloaded MoE layer (design doc §2.5): `reads [{ M, W_expert }]`, `usefulBytes = M·W_expert`,
 * with `M` the expected misses of {@link expertHitsMisses} over the `E_loc` local experts. Undefined when `M = 0`
 * (every needed expert resident): no op, no latency, no staging.
 */
export function expertFetchOp(
  block: MoeBlock,
  layout: MoeLayout,
  routing: MoeRouting,
  routedTokens: number,
  ctx: LoweringContext,
  placement: NonNullable<OffloadLowering['routedExperts']>,
): TransferOp | undefined {
  const U = neededExperts(block, layout, routing, routedTokens, ctx.parallel.ep);
  const { misses, residentExperts } = expertHitsMisses(U, placement.hitRatio, placement.residentFraction, layout.localExperts);
  if (!(misses > 0)) return undefined;
  const W_expert = expertWeightBytes(block, ctx);
  // Formula: reads = [{ M, W_expert }];  usefulBytes = M·W_expert
  // See docs/simulation-model.md §12 ("Placement")
  return {
    kind: 'transfer',
    name: 'moe.expert_fetch',
    group: 'routed_experts',
    source: placement.tier,
    reads: [{ count: misses, bytesEach: W_expert }],
    usefulBytes: misses * W_expert,
    // Formula: q = max(p, M/(E_loc − C))   (docs §12 "Expert prefetch")
    prefetchedFraction: prefetchedFraction(ctx.offload?.prefetchAccuracy ?? 0, {
      group: 'routed_experts',
      misses,
      nonResidentExperts: layout.localExperts - residentExperts,
    }),
  };
}

/**
 * MoE ops. Mode A: router → topk → [latent_in] → gate_up → act → down → combine → [latent_out] → [shared MLP /tp]
 * → TP all-reduce. Mode B: router (Ts) → topk → [latent_in] → a2a dispatch → gate_up → act → down → a2a combine →
 * combine → [latent_out] → [shared MLP, full] → TP all-gather. The bracketed latent projections exist only with
 * `latentDim`. With the layer's experts offloaded (`layerIndex` in the placement's range), the expert fetch comes first.
 */
export function lowerMoe(block: MoeBlock, ctx: LoweringContext, layerIndex?: number): Op[] {
  const { tp, ep } = ctx.parallel;
  const layout = moeLayout(block, ctx.parallel);
  const T = stepTokens(ctx);
  const h = block.hidden;
  // l: width of the routed experts, the dispatch/combine messages and the weighted sum (= h without latentDim)
  const l = moeExpertWidth(block);
  const E = block.experts;
  const act = actBpe(ctx);
  // Formula: Ts = mode A ? T : ceil(T/tp)    (tokens this device routes)
  const routedTokens = layout.mode === 'A' ? T : Math.ceil(T / tp);
  const routing = moeRouting(block, layout, routedTokens, ctx.parallel);
  const a = routing.activeExperts;
  const mE = routing.tokensPerExpert;
  const feLocal = shardWidth(block.expertIntermediate, layout.expertTp, `${block.name} expertIntermediate`);
  const ops: Op[] = [];
  const placement = expertPlacementAt(ctx, layerIndex);
  if (placement !== undefined) {
    const fetch = expertFetchOp(block, layout, routing, routedTokens, ctx, placement);
    if (fetch !== undefined) ops.push(fetch);
  }

  // Router: m = Ts, n = E, k = h in high precision, replicated; then top-k over Ts·E logits
  // (reads the logits, writes k int32 indices + k weights per token).
  const hp = { dtype: highPrecision(ctx) };
  ops.push(linearGemm('moe.router', { m: routedTokens, n: E, k: h }, ctx, { format: { weights: hp } }));
  const logits = routedTokens * E;
  ops.push(
    elementwise(
      'moe.router_topk',
      'router_topk',
      {
        elems: logits,
        activationReadBytes: logits * act,
        activationWriteBytes: routedTokens * block.topK * (4 + act),
      },
      ctx,
    ),
  );

  if (block.latentDim !== undefined) {
    // Latent input projection h → l, replicated (every TP rank computes all of it, as vLLM ReplicatedLinear).
    // Formula: latent_in: m = Ts, n = l, k = h
    // See docs/simulation-model.md §5 "MoE modes A/B"
    ops.push(linearGemm('moe.latent_in', { m: routedTokens, n: l, k: h }, ctx, { module: 'moe_latent' }));
  }

  if (layout.mode === 'B') {
    // Formula: dispatch S = Ts·k·l·bpe(act)
    // See docs/simulation-model.md §5 "MoE modes A/B"
    ops.push(epAllToAll('moe.dispatch', routedTokens * block.topK * l * act, ctx.quant.activations, ep));
  }

  // Grouped expert GEMMs over a active experts, mE tokens each.
  // Formula: gate_up: batch = a, m = mE, n = (gated ? 2 : 1)·fe/etp, k = l; down: batch = a, m = mE, n = l, k = fe/etp
  const upWidth = (block.gated ? 2 : 1) * feLocal;
  ops.push(linearGemm('moe.gate_up', { m: mE, n: upWidth, k: l }, ctx, { module: 'routed_experts', batch: a, outParts: block.gated ? [feLocal, feLocal] : [feLocal] }));
  ops.push(activationOp('moe.act', block, a * mE, feLocal, ctx));
  ops.push(linearGemm('moe.down', { m: mE, n: l, k: feLocal }, ctx, { module: 'routed_experts', batch: a }));

  if (layout.mode === 'B') {
    // Formula: combine S = Ts·k·l·reduceBpe
    ops.push(epAllToAll('moe.combine_a2a', routedTokens * block.topK * l * reduceBpe(ctx), reduceDtype(ctx), ep));
  }

  // Weighted sum of the expert outputs: mode A over a·mE rows into a T·l partial-sum buffer;
  // mode B over the Ts·k returned rows into Ts·l.
  // Formula: mode A: elems = a·mE·l, write T·l·reduceBpe; mode B: elems = Ts·k·l, write Ts·l·bpe(act)
  const combineRows = layout.mode === 'A' ? a * mE : routedTokens * block.topK;
  const combineElems = combineRows * l;
  const combineWrite = layout.mode === 'A' ? T * l * reduceBpe(ctx) : routedTokens * l * act;
  ops.push(
    elementwise(
      'moe.combine',
      'moe_combine',
      { elems: combineElems, activationReadBytes: combineElems * act, activationWriteBytes: combineWrite },
      ctx,
    ),
  );

  if (block.latentDim !== undefined) {
    // Latent output projection l → h, replicated. In mode A it maps each rank's T·l partial sum to a T·h partial
    // sum: Σ_r W·p_r = W·Σ_r p_r (linearity), so the one all-reduce below stays at h.
    // Formula: latent_out: m = Ts, n = h, k = l
    // See docs/simulation-model.md §5 "MoE modes A/B"
    ops.push(linearGemm('moe.latent_out', { m: routedTokens, n: h, k: l }, ctx, { module: 'moe_latent' }));
  }

  if (block.sharedExperts > 0) {
    // Mode A: TP-sharded by tp (reduced by the one all-reduce below); mode B: unsharded on the Ts slice.
    const shards = layout.mode === 'A' ? tp : 1;
    ops.push(...lowerDenseFfn('moe.shared', sharedFfnShape(block), routedTokens, shards, 'shared_experts', ctx));
  }

  if (layout.mode === 'A') {
    ops.push(...tpAllReduce('moe.tp_all_reduce', T, h, ctx));
  } else if (tp > 1) {
    // Formula: all_gather output S = T·h·bpe(act)  (replaces the all-reduce)
    ops.push({
      kind: 'collective',
      name: 'moe.tp_all_gather',
      collective: 'all_gather',
      group: 'tp',
      groupSize: tp,
      messageBytes: T * h * act,
      dtype: ctx.quant.activations,
    });
  }
  return ops;
}

/**
 * Resident MoE weights on one device: ALL local experts (not only the active ones), the router, the latent
 * projections (if any, replicated) and the shared experts. With the layer's experts offloaded (`layerIndex` in the
 * placement's range), only the `C = f·E_loc` resident experts count (the full set lives on the tier, design doc
 * §2.7); router, latent projections and shared experts stay resident. Unit: bytes.
 */
export function moeResidentWeightBytes(block: MoeBlock, ctx: LoweringContext, layerIndex?: number): number {
  const layout = moeLayout(block, ctx.parallel);
  const h = block.hidden;
  const l = moeExpertWidth(block);
  const routedFmt = moduleQuant(ctx, 'routed_experts');
  const sharedFmt = moduleQuant(ctx, 'shared_experts');
  const matrices = block.gated ? 3 : 2;
  const feLocal = block.expertIntermediate / layout.expertTp;
  const sharedShards = layout.mode === 'A' ? ctx.parallel.tp : 1;
  const sharedLocal = (block.sharedExperts * block.sharedIntermediate) / sharedShards;
  // Formula: W = Eloc·(gated ? 3 : 2)·l·fe/etp·bpe(w_routed) + E·h·bpe(hp) + [latent] 2·h·l·bpe(w_latent)
  //            + (gated ? 3 : 2)·h·S·fs/(mode A ? tp : 1)·bpe(w_shared),   l = latentDim ?? h
  // Formula (offloaded layer, f < 1): the routed part becomes C·W_expert, C = f·Eloc
  // See docs/simulation-model.md §8 "Memory fit", §12 ("Memory tiers")
  const placement = expertPlacementAt(ctx, layerIndex);
  const experts =
    placement === undefined || placement.residentFraction >= 1
      ? gemmWeightBytes(layout.localExperts * matrices, l, feLocal, routedFmt.weights, routedFmt.sparsity)
      : placement.residentFraction * layout.localExperts * expertWeightBytes(block, ctx);
  const router = block.experts * h * hpBpe(ctx);
  const latent = moeLatentWeightBytes(block, ctx);
  const shared = gemmWeightBytes(matrices, h, sharedLocal, sharedFmt.weights, sharedFmt.sparsity);
  return experts + router + latent + shared;
}

/**
 * Bytes of a MoE block's FULL local expert set on one device (all `E_loc` experts): what an offloaded layer places on
 * its tier per device (design doc §2.7). Unit: bytes.
 */
export function moeLocalExpertBytes(block: MoeBlock, ctx: LoweringContext): number {
  const layout = moeLayout(block, ctx.parallel);
  const routedFmt = moduleQuant(ctx, 'routed_experts');
  const feLocal = block.expertIntermediate / layout.expertTp;
  // Formula: W_experts = Eloc·(gated ? 3 : 2)·l·fe/etp·bpe(w_routed),  l = latentDim ?? h
  return gemmWeightBytes(layout.localExperts * (block.gated ? 3 : 2), moeExpertWidth(block), feLocal, routedFmt.weights, routedFmt.sparsity);
}
