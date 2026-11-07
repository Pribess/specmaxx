/**
 * Parameter counts of model specs: the LOGICAL, unsharded number of weights (elements), independent of dtype and
 * parallelism. Used to sanity-check mapped models against published sizes (the HF mapper task's ±2% check).
 * Unit of every result: parameters (elements). See docs/simulation-model.md §2 (Model IR) "Parameter counts".
 *
 * Notation: h hidden, hd head dim, rq/rkv MLA ranks, dn/dr/dv MLA head dims, E experts, k top-k,
 * fe/fs routed/shared expert widths, S shared experts, N residual streams, D stream width.
 */

import type { BlockSpec, IndexerSpec, ModelSpec, ProjectionSpec, SparseSelection } from './model';

/** Weights of a list of dense projections: `Σ inputDim·outputDim`. Unit: params. */
function projectionParams(projections: ProjectionSpec[]): number {
  // Formula: P = Σ in·out
  return projections.reduce((sum, p) => sum + p.inputDim * p.outputDim, 0);
}

/**
 * Weights of an indexer whose query projection reads `qInputDim` features (h for GQA, rq for MLA and CSA).
 * The key norm is counted as a LayerNorm (γ, β = 2·hd_I); RMSNorm keys (DeepSeek-V4.1) are hd_I smaller, below
 * 1e-5 of a layer. Unit: params.
 */
function indexerParams(ix: IndexerSpec, hidden: number, qInputDim: number): number {
  const hdI = ix.headDim;
  const keyIn = ix.keyInputDim ?? hidden;
  const pool = ix.keyPoolTokens ?? 1;
  // Formula: P_idx = qIn·H_I·hd_I (wq)
  //               + [ownsKeys] (kIn·hd_I (wk) + 2·hd_I (key norm γ, β))
  //               + [keyPoolGate] (h·hd_I (gate) + p·hd_I (APE))
  //               + [headWeights] h·H_I (weights_proj)
  // See docs/simulation-model.md §2 "Parameter counts"
  const query = qInputDim * ix.heads * hdI;
  const keys = ix.ownsKeys === false ? 0 : keyIn * hdI + 2 * hdI;
  const poolGate = ix.keyPoolGate === true ? hidden * hdI + pool * hdI : 0;
  const headWeights = ix.headWeights === false ? 0 : hidden * ix.heads;
  return query + keys + poolGate + headWeights;
}

/** Weights of a sparse selection: its indexer's, or 0 for a reused top-k. Unit: params. */
function sparseParams(sel: SparseSelection | undefined, hidden: number, qInputDim: number): number {
  if (sel === undefined || sel.kind === 'reuse') return 0;
  return indexerParams(sel.indexer, hidden, qInputDim);
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
 * Logical (unsharded) parameters of one block, with every routed expert and every lookup-table row counted.
 * Unit: params.
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
      // Formula: + [relativeBias] h·heads·relDim (r_proj) + relDim·extent (profile bank)
      // Formula: + [kvConvKernel] 2·kvHeads·hd·K (K and V conv taps)
      // See docs/simulation-model.md §2 "Parameter counts"
      const qkv = h * qkvOut;
      const o = block.heads * hd * h;
      const qkvBias = block.qkvBias ? qkvOut : 0;
      const outBias = block.outBias ? h : 0;
      const qkNorm = block.qkNorm ? 2 * hd : 0;
      const indexer = sparseParams(block.sparse, h, h);
      const rb = block.relativeBias;
      const relBias = rb === undefined ? 0 : h * block.heads * rb.relDim + rb.relDim * rb.extentTokens;
      const kvConv = block.kvConvKernel === undefined ? 0 : 2 * block.kvHeads * hd * block.kvConvKernel;
      return qkv + o + qkvBias + outBias + qkNorm + indexer + relBias + kvConv;
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
      // Formula: kv = [own latent] (h·(rkv+dr) + rkv) + rkv·H·(dn+dv); o = H·dv·h; + P_idx (q input rq or h)
      // See docs/simulation-model.md §2 "Parameter counts"
      const q = rq > 0 ? h * rq + rq + rq * H * (dn + dr) : h * H * (dn + dr);
      const kvA = block.kvCacheFromLayer === undefined ? h * (rkv + dr) + rkv : 0;
      const kvB = rkv * H * (dn + dv);
      const o = H * dv * h;
      const indexer = sparseParams(block.sparse, h, rq > 0 ? rq : h);
      return q + kvA + kvB + o + indexer;
    }
    case 'csa_attention': {
      const h = block.hidden;
      const rq = block.qLoraRank;
      const H = block.heads;
      const hd = block.headDim;
      const G = block.outGroups;
      const Ro = block.outLoraRank;
      // Formula: q = h·rq + rq + rq·H·hd (wq_a, q_norm, wq_b); kv = h·hd + hd (wkv, kv_norm)
      // Formula: o = G·R_o·(H·hd/G) (wo_a, block-diagonal) + G·R_o·h (wo_b)
      // Formula: compressor (owner only): r > 1: 2·h·hd + hd (wkv, wgate, norm); r = 1: h·hd + hd
      // Formula: + P_idx of the global selection (q input rq)
      // See docs/simulation-model.md §2 "Parameter counts"
      const q = h * rq + rq + rq * H * hd;
      const kv = h * hd + hd;
      const o = G * Ro * ((H * hd) / G) + G * Ro * h;
      const g = block.global;
      let compressor = 0;
      if (g !== undefined && g.kvCacheFromLayer === undefined) {
        compressor = g.compressRatio > 1 ? 2 * h * hd + hd : h * hd + hd;
      }
      const indexer = g === undefined ? 0 : sparseParams(g.selection, h, rq);
      return q + kv + o + compressor + indexer;
    }
    case 'linear_attention': {
      const h = block.hidden;
      const qkDims = 2 * block.qkHeads * block.keyDim;
      const vDims = block.vHeads * block.valueDim;
      const conv = (qkDims + vDims) * block.convKernel;
      const outNorm = block.valueDim;
      const outProj = vDims * h;
      if (block.variant === 'kda') {
        const r = block.gateRank ?? 0;
        // Formula: KDA = h·(2·qkH·dk + vH·dv) (q, k, v) + h·vH (β) + h·r + r·vH·dk (forget gate)
        // Formula:     + h·r + r·vH·dv (output gate) + (2·qkH·dk + vH·dv)·K (convs) + vH (A_log)
        // Formula:     + vH·dk (dt_bias) + dv (o_norm) + vH·dv·h (o_proj)
        // See docs/simulation-model.md §2 "Parameter counts"
        const qkv = h * (qkDims + vDims);
        const beta = h * block.vHeads;
        const forgetGate = h * r + r * block.vHeads * block.keyDim;
        const outGate = h * r + r * vDims;
        const aLog = block.vHeads;
        const dtBias = block.vHeads * block.keyDim;
        return qkv + beta + forgetGate + outGate + conv + aLog + dtBias + outNorm + outProj;
      }
      // Formula: in = h·(2·qkH·dk + vH·dv·(gate ? 2 : 1) + 2·vH); conv = (2·qkH·dk + vH·dv)·K
      // Formula: + 2·vH (A_log, dt_bias) + dv (out norm) + vH·dv·h (out_proj)
      // See docs/simulation-model.md §2 "Parameter counts"
      const inProj = h * (qkDims + vDims * (block.outputGate ? 2 : 1) + 2 * block.vHeads);
      const decay = 2 * block.vHeads;
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
    case 'hc_pre': {
      // Formula: P_hc_pre = [normGamma] N·D + Σ in·out + vectorParams
      // See docs/simulation-model.md §2 "Parameter counts"
      const gamma = block.normGamma ? block.streams * block.dim : 0;
      return gamma + projectionParams(block.projections) + block.vectorParams;
    }
    case 'hc_post':
      return 0;
    case 'ngram_embedding': {
      // Formula: P_ngram = tableRows·rowDim + Σ in·out + [conv] outputDim·K + vectorParams
      // See docs/simulation-model.md §2 "Parameter counts"
      const table = block.tableRows * block.rowDim;
      const conv = block.conv === undefined ? 0 : block.outputDim * block.conv.kernel;
      return table + projectionParams(block.projections) + conv + block.vectorParams;
    }
    case 'short_conv':
      // Formula: P_sconv = dim·K (depthwise taps)
      return block.dim * block.kernel;
    default: {
      const _never: never = block;
      throw new Error(`unknown block ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * Parameters of one block that a single token uses: like {@link blockParams}, but a MoE block counts only its
 * top-k routed experts (plus the router and the shared experts), and an n-gram block only the table rows a token
 * gathers instead of the whole table. Unit: params.
 */
export function activeBlockParams(block: BlockSpec): number {
  switch (block.kind) {
    case 'moe': {
      // Formula: P_active = k·(gated ? 3 : 2)·h·fe + E·h (router) + S·(gated ? 3 : 2)·h·fs
      // See docs/simulation-model.md §2 "Parameter counts"
      const routed = block.topK * mlpParams(block.hidden, block.expertIntermediate, block.gated, false);
      const router = block.experts * block.hidden;
      const shared = block.sharedExperts * mlpParams(block.hidden, block.sharedIntermediate, block.gated, false);
      return routed + router + shared;
    }
    case 'ngram_embedding':
      // Formula: P_active = P_ngram − tableRows·rowDim + rowsPerToken·rowDim
      // See docs/simulation-model.md §2 "Parameter counts"
      return blockParams(block) - block.tableRows * block.rowDim + block.rowsPerToken * block.rowDim;
    default:
      return blockParams(block);
  }
}

/** Embedding + LM head (+ final norm) parameters: vocab·h, twice unless tied. Unit: params. */
function outerParams(model: ModelSpec): number {
  // Formula: P_outer = vocab·h·(tied ? 1 : 2) + finalNorm (none 0, rmsnorm h, layernorm 2·h)
  const embed = model.vocab * model.hidden;
  const head = model.tieEmbeddings ? 0 : embed;
  const finalNormVectors = model.finalNorm === 'none' ? 0 : model.finalNorm === 'layernorm' ? 2 : 1;
  return embed + head + finalNormVectors * model.hidden;
}

/** Logical unsharded total parameters of a model (all experts; a tied embedding counted once). Unit: params. */
export function countParams(model: ModelSpec): number {
  let total = outerParams(model);
  for (const layer of model.layers) for (const block of layer.blocks) total += blockParams(block);
  return total;
}

/**
 * Parameters touched per token: {@link countParams} with every MoE block counting only top-k routed experts and
 * every n-gram table only its gathered rows (embedding and head included, as vendors usually quote "active
 * parameters"). Unit: params.
 */
export function activeParamsPerToken(model: ModelSpec): number {
  let total = outerParams(model);
  for (const layer of model.layers) for (const block of layer.blocks) total += activeBlockParams(block);
  return total;
}
