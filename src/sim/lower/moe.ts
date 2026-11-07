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
 * Routing is balanced: the busiest device gets the ceiling of its average share (`A` assignments), spread over
 * `a = min(Eloc, A)` active experts with `mE = ceil(A/a)` tokens each.
 */

import { gemmWeightBytes } from '../ir/metrics';
import type { MoeBlock, ParallelConfig } from '../ir/model';
import type { CollectiveOp, Op } from '../ir/ops';
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
import type { LoweringContext } from './lower';

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
 * MoE ops. Mode A: router → topk → gate_up → act → down → combine → [shared MLP /tp] → TP all-reduce.
 * Mode B: router (Ts) → topk → a2a dispatch → gate_up → act → down → a2a combine → combine → [shared MLP, full]
 * → TP all-gather.
 */
export function lowerMoe(block: MoeBlock, ctx: LoweringContext): Op[] {
  const { tp, ep } = ctx.parallel;
  const layout = moeLayout(block, ctx.parallel);
  const T = stepTokens(ctx);
  const h = block.hidden;
  const E = block.experts;
  const act = actBpe(ctx);
  // Formula: Ts = mode A ? T : ceil(T/tp)    (tokens this device routes)
  const routedTokens = layout.mode === 'A' ? T : Math.ceil(T / tp);
  const routing = moeRouting(block, layout, routedTokens, ctx.parallel);
  const a = routing.activeExperts;
  const mE = routing.tokensPerExpert;
  const feLocal = shardWidth(block.expertIntermediate, layout.expertTp, `${block.name} expertIntermediate`);
  const ops: Op[] = [];

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

  if (layout.mode === 'B') {
    // Formula: dispatch S = Ts·k·h·bpe(act)
    // See docs/simulation-model.md §5 "MoE modes A/B"
    ops.push(epAllToAll('moe.dispatch', routedTokens * block.topK * h * act, ctx.quant.activations, ep));
  }

  // Grouped expert GEMMs over a active experts, mE tokens each.
  // Formula: gate_up: batch = a, m = mE, n = (gated ? 2 : 1)·fe/etp, k = h; down: batch = a, m = mE, n = h, k = fe/etp
  const upWidth = (block.gated ? 2 : 1) * feLocal;
  ops.push(linearGemm('moe.gate_up', { m: mE, n: upWidth, k: h }, ctx, { module: 'routed_experts', batch: a }));
  ops.push(activationOp('moe.act', block, a * mE, feLocal, ctx));
  ops.push(linearGemm('moe.down', { m: mE, n: h, k: feLocal }, ctx, { module: 'routed_experts', batch: a }));

  if (layout.mode === 'B') {
    // Formula: combine S = Ts·k·h·reduceBpe
    ops.push(epAllToAll('moe.combine_a2a', routedTokens * block.topK * h * reduceBpe(ctx), reduceDtype(ctx), ep));
  }

  // Weighted sum of the expert outputs: mode A over a·mE rows into a T·h partial-sum buffer;
  // mode B over the Ts·k returned rows into Ts·h.
  // Formula: mode A: elems = a·mE·h, write T·h·reduceBpe; mode B: elems = Ts·k·h, write Ts·h·bpe(act)
  const combineRows = layout.mode === 'A' ? a * mE : routedTokens * block.topK;
  const combineElems = combineRows * h;
  const combineWrite = layout.mode === 'A' ? T * h * reduceBpe(ctx) : routedTokens * h * act;
  ops.push(
    elementwise(
      'moe.combine',
      'moe_combine',
      { elems: combineElems, activationReadBytes: combineElems * act, activationWriteBytes: combineWrite },
      ctx,
    ),
  );

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
 * Resident MoE weights on one device: ALL local experts (not only the active ones), the router and the shared
 * experts. Unit: bytes.
 */
export function moeResidentWeightBytes(block: MoeBlock, ctx: LoweringContext): number {
  const layout = moeLayout(block, ctx.parallel);
  const h = block.hidden;
  const routedFmt = moduleQuant(ctx, 'routed_experts');
  const sharedFmt = moduleQuant(ctx, 'shared_experts');
  const matrices = block.gated ? 3 : 2;
  const feLocal = block.expertIntermediate / layout.expertTp;
  const sharedShards = layout.mode === 'A' ? ctx.parallel.tp : 1;
  const sharedLocal = (block.sharedExperts * block.sharedIntermediate) / sharedShards;
  // Formula: W = Eloc·(gated ? 3 : 2)·h·fe/etp·bpe(w_routed) + E·h·bpe(hp)
  //            + (gated ? 3 : 2)·h·S·fs/(mode A ? tp : 1)·bpe(w_shared)
  // See docs/simulation-model.md §8 "Memory fit"
  const experts = gemmWeightBytes(layout.localExperts * matrices, h, feLocal, routedFmt.weights, routedFmt.sparsity);
  const router = block.experts * h * hpBpe(ctx);
  const shared = gemmWeightBytes(matrices, h, sharedLocal, sharedFmt.weights, sharedFmt.sparsity);
  return experts + router + shared;
}
