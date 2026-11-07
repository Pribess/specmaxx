/**
 * Parameter counts of model specs: the LOGICAL, unsharded number of weights (elements), independent of dtype and
 * parallelism. Used to sanity-check mapped models against published sizes (the HF mapper task's ±2% check).
 * Unit of every result: parameters (elements). See docs/simulation-model.md §2 (Model IR).
 *
 * Notation: h hidden, hd head dim, rq/rkv MLA ranks, dn/dr/dv MLA head dims, E experts, k top-k,
 * fe/fs routed/shared expert widths, S shared experts.
 */

import type { BlockSpec, IndexerSpec, ModelSpec } from './model';

/** Weights of a DSA indexer whose query projection reads `qInputDim` features (h for GQA, rq for MLA). Unit: params. */
function indexerParams(ix: IndexerSpec, hidden: number, qInputDim: number): number {
  // Formula: P_idx = qIn·heads_I·hd_I (wq) + h·hd_I (wk) + h·heads_I (head weights) + 2·hd_I (key LayerNorm γ, β)
  // See docs/simulation-model.md §2 "Parameter counts"
  return qInputDim * ix.heads * ix.headDim + hidden * ix.headDim + hidden * ix.heads + 2 * ix.headDim;
}

/** Weights of a (gated) MLP of width `f` over `h`, with optional biases. Unit: params. */
function mlpParams(hidden: number, f: number, gated: boolean, bias: boolean): number {
  const matrices = gated ? 3 : 2;
  // Formula: P_mlp = (gated ? 3 : 2)·h·f  [+ bias: (gated ? 2 : 1)·f + h]
  // See docs/simulation-model.md §2 "Parameter counts"
  const biasParams = bias ? (gated ? 2 : 1) * f + hidden : 0;
  return matrices * hidden * f + biasParams;
}

/**
 * Logical (unsharded) parameters of one block, with every routed expert counted. Unit: params.
 */
export function blockParams(block: BlockSpec): number {
  switch (block.kind) {
    case 'norm':
      // Formula: rmsnorm γ: dim; layernorm γ, β: 2·dim
      return block.norm === 'layernorm' ? 2 * block.dim : block.dim;
    case 'residual':
      return 0;
    case 'gqa_attention': {
      const h = block.hidden;
      const hd = block.headDim;
      const qkvOut = (block.heads * (block.outputGate ? 2 : 1) + 2 * block.kvHeads) * hd;
      // Formula: P = h·(heads + 2·kvHeads)·hd [+ h·heads·hd gate] + heads·hd·h (o) + biases + 2·hd (q/k norm) + P_idx
      // See docs/simulation-model.md §2 "Parameter counts"
      const qkv = h * qkvOut;
      const o = block.heads * hd * h;
      const qkvBias = block.qkvBias ? qkvOut : 0;
      const outBias = block.outBias ? h : 0;
      const qkNorm = block.qkNorm ? 2 * hd : 0;
      const indexer = block.sparse === undefined ? 0 : indexerParams(block.sparse, h, h);
      return qkv + o + qkvBias + outBias + qkNorm + indexer;
    }
    case 'mla_attention': {
      const h = block.hidden;
      const rq = block.qLoraRank;
      const rkv = block.kvLoraRank;
      const dn = block.qkNopeDim;
      const dr = block.qkRopeDim;
      const dv = block.vHeadDim;
      const H = block.heads;
      // Formula: q = rq > 0 ? h·rq + rq + rq·H·(dn+dr) : h·H·(dn+dr)
      // Formula: kv = h·(rkv+dr) + rkv + rkv·H·(dn+dv); o = H·dv·h; + P_idx (q input rq or h)
      // See docs/simulation-model.md §2 "Parameter counts"
      const q = rq > 0 ? h * rq + rq + rq * H * (dn + dr) : h * H * (dn + dr);
      const kv = h * (rkv + dr) + rkv + rkv * H * (dn + dv);
      const o = H * dv * h;
      const indexer = block.sparse === undefined ? 0 : indexerParams(block.sparse, h, rq > 0 ? rq : h);
      return q + kv + o + indexer;
    }
    case 'linear_attention': {
      const h = block.hidden;
      const qkDims = 2 * block.qkHeads * block.keyDim;
      const vDims = block.vHeads * block.valueDim;
      // Formula: in = h·(2·qkH·dk + vH·dv·(gate ? 2 : 1) + 2·vH); conv = (2·qkH·dk + vH·dv)·K
      // Formula: + 2·vH (A_log, dt_bias) + dv (out norm) + vH·dv·h (out_proj)
      // See docs/simulation-model.md §2 "Parameter counts"
      const inProj = h * (qkDims + vDims * (block.outputGate ? 2 : 1) + 2 * block.vHeads);
      const conv = (qkDims + vDims) * block.convKernel;
      const decay = 2 * block.vHeads;
      const outNorm = block.valueDim;
      const outProj = vDims * h;
      return inProj + conv + decay + outNorm + outProj;
    }
    case 'mlp':
      return mlpParams(block.hidden, block.intermediate, block.gated, block.bias);
    case 'moe': {
      // Formula: P = E·(gated ? 3 : 2)·h·fe + E·h (router) + S·(gated ? 3 : 2)·h·fs
      // See docs/simulation-model.md §2 "Parameter counts"
      const routed = block.experts * mlpParams(block.hidden, block.expertIntermediate, block.gated, false);
      const router = block.experts * block.hidden;
      const shared = block.sharedExperts * mlpParams(block.hidden, block.sharedIntermediate, block.gated, false);
      return routed + router + shared;
    }
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Parameters of one block that a single token uses: like {@link blockParams}, but a MoE block counts only its
 * top-k routed experts (plus the router and the shared experts). Unit: params.
 */
export function activeBlockParams(block: BlockSpec): number {
  if (block.kind !== 'moe') return blockParams(block);
  // Formula: P_active = k·(gated ? 3 : 2)·h·fe + E·h (router) + S·(gated ? 3 : 2)·h·fs
  // See docs/simulation-model.md §2 "Parameter counts"
  const routed = block.topK * mlpParams(block.hidden, block.expertIntermediate, block.gated, false);
  const router = block.experts * block.hidden;
  const shared = block.sharedExperts * mlpParams(block.hidden, block.sharedIntermediate, block.gated, false);
  return routed + router + shared;
}

/** Embedding + LM head (+ final norm) parameters: vocab·h, twice unless tied. Unit: params. */
function outerParams(model: ModelSpec): number {
  // Formula: P_outer = vocab·h·(tied ? 1 : 2) + finalNorm (h, or 2·h for layernorm)
  const embed = model.vocab * model.hidden;
  const head = model.tieEmbeddings ? 0 : embed;
  const finalNorm = model.finalNorm === 'layernorm' ? 2 * model.hidden : model.hidden;
  return embed + head + finalNorm;
}

/** Logical unsharded total parameters of a model (all experts; a tied embedding counted once). Unit: params. */
export function countParams(model: ModelSpec): number {
  let total = outerParams(model);
  for (const layer of model.layers) for (const block of layer.blocks) total += blockParams(block);
  return total;
}

/**
 * Parameters touched per token: {@link countParams} with every MoE block counting only top-k routed experts
 * (embedding and head included, as vendors usually quote "active parameters"). Unit: params.
 */
export function activeParamsPerToken(model: ModelSpec): number {
  let total = outerParams(model);
  for (const layer of model.layers) for (const block of layer.blocks) total += activeBlockParams(block);
  return total;
}
