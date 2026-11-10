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
 * projections). With `latentNorm` (Kimi K3; docs §5 "Latent MoE") an RMSNorm over l runs before the `l → h`
 * projection: in mode A the TP ranks' partial expert sums are all-reduced at l first (`moe.latent_all_reduce`, the
 * norm needs them complete), and `l → h` is then row-parallel over `k = l/tp`, its partial sums joined by the one TP
 * all-reduce with the shared experts; in mode B it runs unsharded on Ts.
 *
 * Routing is balanced in load and uniform in expert choice. A device gets the ceiling of its average share (`A`
 * assignments), the busiest device `A_max` (ρ times the unrounded mean, within `[A, T_g·min(k, Eloc)]`; see
 * {@link moeRouting}) with the EP load imbalance ρ ({@link epImbalance}; 1 at ep = 1). Compute runs on a balanced
 * grid of `a = min(Eloc, A_max)` experts with `mE = ceil(A_max/a)` tokens each;
 * weights are read for the `U` expected distinct experts under iid-uniform top-k routing ({@link neededExperts}: an
 * EXPECTATION, not a bound), `U ≤ a` (`GemmOp.distinctWeightMatrices`).
 *
 * Offloaded experts (deployment `offload.routedExperts`, design doc §2.4–2.5): a `TransferOp` of the expected missed
 * experts among those `U` goes in front of the block; the grouped GEMMs keep their shape and their `U` weight reads
 * (device-memory bytes unchanged).
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
  normOp,
  reduceBpe,
  reduceDtype,
  shardWidth,
  stepTokens,
  tpAllReduce,
} from './common';
import { activationOp, lowerDenseFfn, type DenseFfnShape } from './mlp';
import type { ExpertPlacement, LoweringContext } from './lower';
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

/**
 * Routing of one MoE layer on the busiest device: the compute grid (`a` experts × `mE` tokens) and the expected
 * distinct experts `U` whose weights are read.
 */
export interface MoeRouting {
  /** Expert assignments (token, expert) of a device's balanced share `A`: mode A `ceil(T·k/ep)`, mode B `Ts·k`. Count. */
  assignments: number;
  /** EP load imbalance ρ applied to the busiest device: {@link epImbalance}, 1 at ep = 1. Dimensionless, ≥ 1. */
  imbalance: number;
  /**
   * Assignments on the busiest device: `A_max = clamp(ceil(ρ·T_g·k/ep), A, T_g·min(k, Eloc))` ({@link moeRouting};
   * = A at ep = 1). Count.
   */
  busiestAssignments: number;
  /** Experts of the balanced compute grid `a = min(Eloc, A_max)` (the grouped GEMMs' batch). Count. */
  activeExperts: number;
  /** Tokens per grid expert `mE = ceil(A_max/a)` (the grouped GEMMs' m; a ceiling). Unit: tokens. */
  tokensPerExpert: number;
  /**
   * Expected distinct local experts `U` the balanced share `A` touches under iid-uniform routing ({@link neededExperts};
   * real, `U ≤ min(Eloc, A) ≤ a`; at ep = 1, `U = k` for one token and `U → Eloc` in prefill): the expert weight
   * matrices read. An expectation, not a bound. Unit: experts.
   */
  distinctExperts: number;
}

/**
 * Default EP load imbalance ρ = max/mean routed tokens per device ({@link epImbalance}) when `ParallelConfig.epImbalance`
 * is not given. Dimensionless.
 *
 * Measured max/mean token load per EP rank, DeepSeek-V3/R1 (256 experts, top-8), averaged over layers and steps:
 * - with an expert-placement load balancer (EPLB, https://github.com/deepseek-ai/EPLB): 1.109 at EP32 (9 slots per
 *   rank: 1 redundant expert per rank, 32 in total) and 1.115 at EP36 (8 slots per rank, 288 slots for 256 experts)
 *   (TensorRT-LLM wide-EP blog, Table 1, reported as (max − mean)/mean 0.109 / 0.115:
 *   https://nvidia.github.io/TensorRT-LLM/1.0.0rc3/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html);
 * - without one: 2.56 at EP32 (same table, 1.564); ≈ 1.47 at 8 GPUs, rising with the GPU count (SGLang
 *   "balancedness" mean/max ≈ 0.68, LMSYS large-scale EP blog, https://lmsys.org/blog/2025-05-05-large-scale-ep/,
 *   figure values as read in Aleph Alpha's DeepSeek inference model, Fig. 12).
 * 1.2 takes the balanced case, which the engines that run EP use, rounded up from 1.11 because those figures need
 * redundant expert copies (+12.5% expert memory), which the default does not store (`redundantExpertsPerRank` 0).
 * ρ is the user's input: setting `redundantExpertsPerRank` adds the copies' memory but does not lower ρ. An
 * estimate, not a constant of the hardware.
 */
export const DEFAULT_EP_IMBALANCE = 1.2;

/**
 * EP load imbalance ρ of a parallel config: the busiest device's routed (token, expert) assignments over the mean
 * device's. 1 at ep = 1 (one device holds every expert, nothing to balance); else `parallel.epImbalance`, defaulting
 * to {@link DEFAULT_EP_IMBALANCE}. Dimensionless, ≥ 1.
 */
export function epImbalance(parallel: ParallelConfig): number {
  return parallel.ep > 1 ? (parallel.epImbalance ?? DEFAULT_EP_IMBALANCE) : 1;
}

/**
 * Routing of `tokens` routed tokens (mode A: T; mode B: Ts) on the busiest device.
 * Mode A: `A = ceil(T·k/ep)`; mode B: `A = Ts·k`. Then `A_max` (below), `a = min(Eloc, A_max)`,
 * `mE = ceil(A_max/a)`, and `U` = {@link neededExperts} of A.
 *
 * `A_max = min(T_g·min(k, Eloc), max(A, ceil(ρ·T_g·k/ep)))`, with `T_g` the distinct tokens routed through the EP
 * group (mode A: T, every TP rank holds all of them; mode B: `ep·Ts`, the ep devices' slices) and `T_g·k/ep` the
 * unrounded mean share:
 * - ρ multiplies the unrounded mean, so a fractional mean (T·k < ep) is not first rounded up and then scaled again;
 * - never below the balanced share `A` (ρ ≥ 1; A is already the ceiling of the mean);
 * - never above what the group's tokens can route to one device: each token picks k distinct experts, of which at
 *   most `min(k, Eloc)` live on this device (this also implies `≤ T_g·k`). E.g. E 8, top-2, T 1, ep 8 (mode A):
 *   mean 0.25, ρ·mean 0.3 → 1, cap 1·min(2, 1) = 1: one assignment, not 2.
 * `A ≤` the cap always (`T_g·k/ep ≤ T_g·min(k, Eloc)` since `k ≤ E`), so the clamp is well-formed.
 *
 * ρ scales the busiest device's assignments (the grid: compute, activations, combine; in mode B, with
 * `parallel.epImbalanceScalesAllToAll`, also the dispatch and combine all-to-all bytes, see {@link a2aRowsOf})
 * but not `U`: EP imbalance comes from hot experts drawing more
 * tokens (the EPLB motivation), so the busiest device runs more rows through the same experts rather than reading
 * more distinct expert weights; in prefill `U = Eloc` either way.
 */
export function moeRouting(block: MoeBlock, layout: MoeLayout, tokens: number, parallel: ParallelConfig): MoeRouting {
  // Formula: A = mode A ? ceil(T·k/ep) : Ts·k;  T_g = mode A ? T : ep·Ts;
  //          A_max = min(T_g·min(k, Eloc), max(A, ceil(ρ·T_g·k/ep)));  a = min(Eloc, A_max);  mE = ceil(A_max/a);
  //          U = neededExperts(A)
  // See docs/simulation-model.md §5 "MoE modes A/B", "Expected distinct experts", "EP load imbalance"
  const k = block.topK;
  const { ep } = parallel;
  const assignments = layout.mode === 'A' ? Math.ceil((tokens * k) / ep) : tokens * k;
  const imbalance = epImbalance(parallel);
  const groupTokens = layout.mode === 'A' ? tokens : ep * tokens;
  const meanAssignments = (groupTokens * k) / ep;
  const routableCap = groupTokens * Math.min(k, layout.localExperts);
  // − 1e-9: a float product landing a hair above a whole count (1.1·50 = 55.00000000000001) must not round up a step
  const scaled = Math.ceil(imbalance * meanAssignments - 1e-9);
  const busiestAssignments = imbalance === 1 ? assignments : Math.min(routableCap, Math.max(assignments, scaled));
  const activeExperts = Math.min(layout.localExperts, busiestAssignments);
  const tokensPerExpert = Math.ceil(busiestAssignments / activeExperts);
  const distinctExperts = neededExperts(block, layout, assignments, tokens, ep);
  return { assignments, imbalance, busiestAssignments, activeExperts, tokensPerExpert, distinctExperts };
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

/**
 * Rows (token copies) a mode-B dispatch / combine all-to-all carries per device: the busiest device's `A_max` when
 * `parallel.epImbalanceScalesAllToAll` is true (it receives the A_max assignments routed to its experts and sends
 * them back, and the step waits for it), else (default) the mean volume `Ts·k` (the device's own token copies), as
 * before the switch existed. ρ scales the expert rows either way. Equal to `Ts·k` at ρ = 1. Unit: rows (token,
 * expert) assignments.
 */
export function a2aRowsOf(routing: MoeRouting, parallel: ParallelConfig): number {
  // Formula: R_a2a = epImbalanceScalesAllToAll ? A_max : Ts·k
  // See docs/simulation-model.md §5 "EP load imbalance"
  return parallel.epImbalanceScalesAllToAll === true ? routing.busiestAssignments : routing.assignments;
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
 * Ways each routed expert's output is split across the expert-TP group after the expert-output norm: `etp` when
 * the block has `expertOutputNorm` and its experts are TP-sharded (`etp > 1`, mode A), else 1. With a split, the
 * down partial sums are reduce-scattered so each rank normalizes, combines and projects an `l/etp` shard. Count.
 */
export function expertOutputShards(block: MoeBlock, layout: MoeLayout): number {
  // Formula: s = (expertOutputNorm ∧ etp > 1) ? etp : 1
  return block.expertOutputNorm === true && layout.expertTp > 1 ? layout.expertTp : 1;
}

/**
 * Ways the latent MoE's `l → h` projection (`latent_out`) is k-split on one device. With `latentNorm`: tp in mode A
 * (row-parallel after the all-reduce at l, see {@link lowerMoe}), 1 in mode B (unsharded on Ts). Otherwise
 * {@link expertOutputShards}: 1 (replicated) unless a split expert-output norm leaves each rank an l-shard. Count.
 */
export function latentOutShards(block: MoeBlock, layout: MoeLayout, tp: number): number {
  // Formula: s_out = latentNorm ? (mode A ? tp : 1) : expertOutputShards
  if (block.latentNorm === true) return layout.mode === 'A' ? tp : 1;
  return expertOutputShards(block, layout);
}

/**
 * Stored bytes of a latent MoE's projections on one device: `h → l` (`n l, k h`), replicated, plus `l → h`
 * (`n h, k l/s`, s = {@link latentOutShards}: replicated when s = 1, else row-split like the l-shards it reads),
 * in the `'moe_latent'` format, plus the latent norm γ (`l·bpe(hp)`) with `latentNorm`; never offloaded; 0 without
 * `latentDim`. Computed per GEMM like the ops of {@link lowerMoe}, so resident and streamed bytes agree. Unit: bytes.
 */
export function moeLatentWeightBytes(block: MoeBlock, ctx: LoweringContext): number {
  if (block.latentDim === undefined) return 0;
  const h = block.hidden;
  const l = block.latentDim;
  const s = latentOutShards(block, moeLayout(block, ctx.parallel), ctx.parallel.tp);
  const fmt = moduleQuant(ctx, 'moe_latent');
  // Formula: W_latent = gemmWeightBytes(1, l, h, w_latent) + gemmWeightBytes(1, h, l/s, w_latent) + [latentNorm] l·bpe(hp)
  //          ≈ (1 + 1/s)·h·l·bpe(w_latent) + [latentNorm] l·bpe(hp),   s = latentOutShards
  // See docs/simulation-model.md §5 "MoE modes A/B", "Latent MoE"
  const latentIn = gemmWeightBytes(1, l, h, fmt.weights, fmt.sparsity);
  const latentOut = gemmWeightBytes(1, h, l / s, fmt.weights, fmt.sparsity);
  const norm = block.latentNorm === true ? l * hpBpe(ctx) : 0;
  return latentIn + latentOut + norm;
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
 * Expected distinct experts one device touches in one MoE layer in a step, under iid-uniform top-k routing: what
 * the grouped expert GEMMs read (`MoeRouting.distinctExperts`) and what an offloaded layer needs on the device
 * ({@link expertFetchOp}). At ep = 1 the exact `U = E·(1 − (1 − k/E)^T)` over the T routed tokens; at ep > 1 the
 * per-device form `U_dev = E_loc·(1 − (1 − 1/E_loc)^A)` over its A assignments (design doc §2.9). An EXPECTATION,
 * not a bound: skewed or correlated real routing can touch fewer experts, and a batch can touch more than the
 * mean. Unit: experts (real).
 */
export function neededExperts(block: MoeBlock, layout: MoeLayout, assignments: number, routedTokens: number, ep: number): number {
  // Formula: ep = 1: U = E·(1 − (1 − k/E)^T);  ep > 1: U_dev = E_loc·(1 − (1 − 1/E_loc)^A)
  // See docs/simulation-model.md §5 "Expected distinct experts", §12 ("Expert hit ratio")
  return ep === 1
    ? uniqueExperts(block.experts, block.topK, routedTokens)
    : uniqueLocalExperts(layout.localExperts, assignments);
}

/**
 * The expert fetch of an offloaded MoE layer (design doc §2.5): `reads [{ M, W_expert }]`, `usefulBytes = M·W_expert`,
 * with `M` the expected misses of {@link expertHitsMisses} among the routing's `U` needed experts
 * ({@link neededExperts}) over the `E_loc` local experts. Undefined when `M = 0` (every needed expert resident): no
 * op, no latency, no staging.
 */
export function expertFetchOp(
  block: MoeBlock,
  layout: MoeLayout,
  routing: MoeRouting,
  ctx: LoweringContext,
  placement: ExpertPlacement,
): TransferOp | undefined {
  const U = routing.distinctExperts;
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
 * MoE ops. Mode A: router → topk → [latent_in] → gate_up → act → down → [expert-output norm] → combine →
 * [latent_all_reduce → latent_norm] → [latent_out] → [shared MLP /tp] → TP all-reduce. Mode B: router (Ts) → topk →
 * [latent_in] → a2a dispatch → gate_up → act → down → [expert-output norm] → a2a combine → combine → [latent_norm] →
 * [latent_out] → [shared MLP, full] → TP all-gather. The latent projections exist only with `latentDim`; the
 * expert-output norm only with `expertOutputNorm` (with TP-sharded experts it is preceded by a reduce-scatter of
 * the down partial sums and a sum-of-squares all-reduce, and the combine and latent_out then work on `l/etp`
 * shards); the latent norm and its all-reduce only with `latentNorm`. With the layer's experts offloaded
 * (`layerIndex` in the placement's range), the expert fetch comes first.
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
    const fetch = expertFetchOp(block, layout, routing, ctx, placement);
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

  // Mode B all-to-alls: rows R_a2a = epImbalanceScalesAllToAll ? A_max : Ts·k. Switched on, they carry the busiest
  // device's A_max rows (it sends its own Ts·k token copies but receives the A_max assignments routed to its experts,
  // and the step waits for it); off (default) they carry the mean volume Ts·k, as before the switch existed.
  const a2aRows = a2aRowsOf(routing, ctx.parallel);
  if (layout.mode === 'B') {
    // Formula: dispatch S = R_a2a·l·bpe(act)
    // See docs/simulation-model.md §5 "MoE modes A/B", "EP load imbalance"
    ops.push(epAllToAll('moe.dispatch', a2aRows * l * act, ctx.quant.activations, ep));
  }

  // Grouped expert GEMMs on the balanced grid of a experts × mE tokens (FLOP, tiles, activations), reading the
  // weights of the U ≤ a expected distinct experts only (distinctWeightMatrices).
  // Formula: gate_up: batch = a, m = mE, n = (gated ? 2 : 1)·fe/etp, k = l; down: batch = a, m = mE, n = l, k = fe/etp;
  //          weights read for U of the a entries
  // See docs/simulation-model.md §5 "Expected distinct experts"
  const U = routing.distinctExperts;
  const upWidth = (block.gated ? 2 : 1) * feLocal;
  const outParts = block.gated ? [feLocal, feLocal] : [feLocal];
  ops.push(linearGemm('moe.gate_up', { m: mE, n: upWidth, k: l }, ctx, { module: 'routed_experts', batch: a, distinctWeightMatrices: U, outParts }));
  // Formula: R = a·mE   (token-expert rows of the grid on this device)
  const expertRows = a * mE;
  ops.push(activationOp('moe.act', block, expertRows, feLocal, ctx));
  ops.push(linearGemm('moe.down', { m: mE, n: l, k: feLocal }, ctx, { module: 'routed_experts', batch: a, distinctWeightMatrices: U }));

  // Expert-output norm (expertOutputNorm) over the R = a·mE token-expert rows of this device. With etp > 1 the
  // down outputs are partial sums over fe/etp: reduce-scatter them over the etp ranks sharing the experts (each
  // keeps an l/etp shard), then all-reduce the per-row partial sum of squares (R fp32 scalars) for the RMS.
  // Both run on the expert-TP group ('etp': the etp ranks of the TP group holding these experts, strided by ep;
  // its links come from engine/placement.ts: expertTpGroupSpan).
  // Formula: s = (expertOutputNorm ∧ etp > 1) ? etp : 1;  l_s = l/s
  // Formula: reduce_scatter S = R·l·reduceBpe;  sumsq all_reduce S = R·4;  rmsnorm elems = R·l_s
  // See docs/simulation-model.md §5 "MoE modes A/B"
  const s = expertOutputShards(block, layout);
  const lShard = shardWidth(l, s, `${block.name} expert output width`);
  // Without latentDim (l = h) a split leaves each rank an h/etp column slice of the T·h partial-sum buffer (zero
  // elsewhere), which the closing all-reduce at h sums like any partial sum.
  if (s > 1) {
    ops.push({
      kind: 'collective',
      name: 'moe.expert_out_reduce_scatter',
      collective: 'reduce_scatter',
      group: 'etp',
      groupSize: s,
      messageBytes: expertRows * l * reduceBpe(ctx),
      dtype: reduceDtype(ctx),
    });
    ops.push({
      kind: 'collective',
      name: 'moe.expert_norm_sumsq_all_reduce',
      collective: 'all_reduce',
      group: 'etp',
      groupSize: s,
      messageBytes: expertRows * 4,
      dtype: 'fp32',
    });
  }
  if (block.expertOutputNorm === true) ops.push(normOp('moe.expert_norm', 'rmsnorm', expertRows, lShard, ctx));

  if (layout.mode === 'B') {
    // Formula: combine S = R_a2a·l·reduceBpe   (R_a2a = a2aRowsOf: A_max when switched on, else Ts·k)
    ops.push(epAllToAll('moe.combine_a2a', a2aRows * l * reduceBpe(ctx), reduceDtype(ctx), ep));
  }

  // Weighted sum of the expert outputs: mode A over the R = a·mE expert rows into a T·l_s partial-sum buffer;
  // mode B over the Ts·k returned rows into Ts·l (l_s = l/s, s = 1 except a split expert-output norm).
  // Formula: mode A: elems = R·l_s, write T·l_s·reduceBpe; mode B: elems = Ts·k·l, write Ts·l·bpe(act)
  const combineRows = layout.mode === 'A' ? expertRows : routedTokens * block.topK;
  const combineElems = combineRows * lShard;
  const combineWrite = layout.mode === 'A' ? T * lShard * reduceBpe(ctx) : routedTokens * lShard * act;
  ops.push(
    elementwise(
      'moe.combine',
      'moe_combine',
      { elems: combineElems, activationReadBytes: combineElems * act, activationWriteBytes: combineWrite },
      ctx,
    ),
  );

  if (block.latentDim !== undefined) {
    if (block.latentNorm === true) {
      // The norm needs the complete expert sum. Mode A: the combined T·l holds each rank's partial sums (its
      // experts, its fe shard), so one all-reduce of T·l comes first; mode B: the device's Ts·l rows are complete
      // after the combine a2a. The l → h projection is then row-parallel in mode A (k = l/tp; its partial h sums
      // join moe.tp_all_reduce below) and unsharded on Ts in mode B. (latentNorm excludes expertOutputNorm, so the
      // combine above wrote full l-wide rows.)
      // Formula: [mode A] all_reduce S = T·l·reduceBpe; rmsnorm over Ts·l (γ l·bpe(hp))
      // See docs/simulation-model.md §5 "Latent MoE"
      if (layout.mode === 'A') ops.push(...tpAllReduce('moe.latent_all_reduce', T, l, ctx));
      ops.push(normOp('moe.latent_norm', 'rmsnorm', routedTokens, l, ctx));
    }
    // Latent output projection l → h. Without latentNorm, in mode A it maps each rank's T·l_s partial sum to a T·h
    // partial sum: Σ_r W_r·p_r = W·Σ_r p_r (linearity), so the one all-reduce below stays at h. Replicated (k = l)
    // when s_out = 1; a k-split over the rank's l/s_out shard (row-split weights) otherwise.
    // Formula: latent_out: m = Ts, n = h, k = l/s_out,   s_out = latentOutShards
    // See docs/simulation-model.md §5 "MoE modes A/B"
    const sOut = latentOutShards(block, layout, tp);
    const kOut = shardWidth(l, sOut, `${block.name} latentDim`);
    ops.push(linearGemm('moe.latent_out', { m: routedTokens, n: h, k: kOut }, ctx, { module: 'moe_latent' }));
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
 * Redundant expert slots per device of a parallel config (EPLB replicas): `parallel.redundantExpertsPerRank` at
 * ep > 1, else 0 (one device holds every expert, nothing to replicate). Count per device per MoE layer.
 */
export function redundantExperts(parallel: ParallelConfig): number {
  return parallel.ep > 1 ? (parallel.redundantExpertsPerRank ?? 0) : 0;
}

/**
 * Resident MoE weights on one device: ALL local experts (not only the active ones), the router, the latent
 * projections and latent norm (if any; see {@link moeLatentWeightBytes}), the expert-output norm γ (if any) and the
 * shared experts. With the layer's experts offloaded (`layerIndex` in the placement's range), only the
 * `C = f·E_loc` resident experts count (the full set lives on the tier, design doc §2.7); everything else stays
 * resident. With ep > 1 and `parallel.redundantExpertsPerRank` r > 0 (EPLB replicas), `r` full copies of one routed
 * expert ({@link expertWeightBytes}) are added on every device, always resident (also on an offloaded layer: the
 * replicas are the hot experts, kept on the device; a simplification). Unit: bytes.
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
  // Formula: W = Eloc·(gated ? 3 : 2)·l·fe/etp·bpe(w_routed) + E·h·bpe(hp) + [latent] W_latent
  //            + [expertOutputNorm] (l/s)·bpe(hp) + (gated ? 3 : 2)·h·S·fs/(mode A ? tp : 1)·bpe(w_shared),
  //   l = latentDim ?? h, s = expertOutputShards (the γ slice the norm op reads)
  // Formula (offloaded layer, f < 1): the routed part becomes C·W_expert, C = f·Eloc
  // Formula (EPLB replicas, ep > 1): + r·W_expert,  r = redundantExpertsPerRank (default 0)
  // See docs/simulation-model.md §5 "Redundant experts", §8 "Memory fit", §12 ("Memory tiers")
  const placement = expertPlacementAt(ctx, layerIndex);
  const experts =
    placement === undefined || placement.residentFraction >= 1
      ? gemmWeightBytes(layout.localExperts * matrices, l, feLocal, routedFmt.weights, routedFmt.sparsity)
      : placement.residentFraction * layout.localExperts * expertWeightBytes(block, ctx);
  const router = block.experts * h * hpBpe(ctx);
  const latent = moeLatentWeightBytes(block, ctx);
  const norm = block.expertOutputNorm === true ? (l / expertOutputShards(block, layout)) * hpBpe(ctx) : 0;
  const shared = gemmWeightBytes(matrices, h, sharedLocal, sharedFmt.weights, sharedFmt.sparsity);
  const replicas = redundantExperts(ctx.parallel);
  const redundant = replicas > 0 ? replicas * expertWeightBytes(block, ctx) : 0;
  // The replicas are added last, so with none the sum is the one from before they existed, bit for bit.
  return experts + router + latent + norm + shared + redundant;
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
