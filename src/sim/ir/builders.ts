/**
 * Builder API for model specs: the follow-up Hugging Face config mappers use only these functions, so they stay
 * architecture-only (no tokens, no parallelism). A decoder layer is a pre-norm block sequence. With a plain
 * residual (the default):
 *
 *   [ngram] → norm → mixer → [short_conv] → [post-norm] → residual → norm → ffn → [short_conv] → [post-norm] → residual
 *
 * With hyper-connections (`residual.kind: 'hyper'`, N residual streams), each residual add becomes an
 * `hc_pre … hc_post` pair and the norms are kept only with `subNorms`:
 *
 *   [ngram] → hc_pre → [norm] → mixer → [short_conv] → [post-norm] → hc_post
 *           → hc_pre → [norm] → ffn → [short_conv] → [post-norm] → hc_post
 *
 * A one-sublayer layer ({@link singleBlockLayer}, Nemotron-H: each layer is ONE Mamba-2, attention or MoE block)
 * is `norm → mixer|ffn → residual`.
 *
 * Defaults (when a field is omitted): activation `'silu'`, gated `true`, norm `'rmsnorm'`, rope `true`,
 * convKernel 4, outputGate `false`, residual `{ kind: 'plain' }`, finalNorm `'rmsnorm'`, residualStreams 1;
 * every other boolean `false`, every other count 0, every other optional field absent.
 * See docs/simulation-model.md §2 (Model IR).
 */

import type {
  BlockSpec,
  CsaAttentionBlock,
  GqaAttentionBlock,
  HcPreBlock,
  LayerSpec,
  LinearAttentionBlock,
  MlaAttentionBlock,
  MlpBlock,
  ModelSpec,
  MoeBlock,
  NgramEmbeddingBlock,
  SparseSelection,
} from './model';

/** Token-mixer parameters of one decoder layer (dims are logical, unsharded). */
export type MixerParams =
  /** MHA/GQA/MQA attention. Maps to a {@link GqaAttentionBlock}. */
  | {
      kind: 'gqa';
      /** Query heads. Count. */
      heads: number;
      /** Key/value heads. Count. */
      kvHeads: number;
      /** Head dimension. Unit: elements. */
      headDim: number;
      /** Per-head RMS q/k-norm (default false). */
      qkNorm?: boolean;
      /** Biases on q/k/v projections (default false). */
      qkvBias?: boolean;
      /** Bias on the output projection (default false). */
      outBias?: boolean;
      /** Rotary embeddings on q/k (default true). */
      rope?: boolean;
      /** Gated attention output `sigmoid(gate) ⊙ out` (default false). */
      outputGate?: boolean;
      /** Sliding window; absent = full attention. Unit: tokens. */
      slidingWindow?: number;
      /** Top-k sparse attention (own indexer or reused top-k); absent = dense. */
      sparse?: SparseSelection;
      /** Learned relative-position bias (see {@link GqaAttentionBlock.relativeBias}); absent = none. */
      relativeBias?: GqaAttentionBlock['relativeBias'];
      /** Short conv width on K and V (see {@link GqaAttentionBlock.kvConvKernel}); absent = none. Unit: taps. */
      kvConvKernel?: number;
    }
  /** Multi-head latent attention. Maps to an {@link MlaAttentionBlock}; all dims in elements. */
  | {
      kind: 'mla';
      /** Attention heads. Count. */
      heads: number;
      /** Query compression rank rq (0 = none). Unit: elements. */
      qLoraRank: number;
      /** KV latent rank rkv. Unit: elements. */
      kvLoraRank: number;
      /** Per-head non-RoPE q/k dim dn. Unit: elements. */
      qkNopeDim: number;
      /** Per-head RoPE q/k dim dr (0 for NoPE layers). Unit: elements. */
      qkRopeDim: number;
      /** Per-head value dim dv. Unit: elements. */
      vHeadDim: number;
      /** DSA top-k sparse attention (own indexer or reused top-k); absent = dense. */
      sparse?: SparseSelection;
      /** Earlier layer whose latent cache this layer reads (see {@link MlaAttentionBlock.kvCacheFromLayer}). Index. */
      kvCacheFromLayer?: number;
    }
  /**
   * DeepSeek-V4.1 compressed sparse attention: every field of a {@link CsaAttentionBlock} except kind, name and
   * hidden (heads, headDim, ropeDim, qLoraRank, outGroups, outLoraRank, windowTokens, global?).
   */
  | ({ kind: 'csa' } & Omit<CsaAttentionBlock, 'kind' | 'name' | 'hidden'>)
  /** Linear attention / SSM mixer. Maps to a {@link LinearAttentionBlock}. */
  | {
      kind: 'linear';
      /** Recurrence variant. */
      variant: LinearAttentionBlock['variant'];
      /** q/k heads (Mamba2: B/C groups). Count. */
      qkHeads: number;
      /** Value (state) heads. Count. */
      vHeads: number;
      /** State key dim dk. Unit: elements. */
      keyDim: number;
      /** State value dim dv. Unit: elements. */
      valueDim: number;
      /** Causal conv width (default 4; 0 = no conv). Unit: taps. */
      convKernel?: number;
      /** z-gated output norm (default false; must be true for kda; true for mamba2). */
      outputGate?: boolean;
      /** conv1d bias over the conv channels (see {@link LinearAttentionBlock.convBias}; default false). */
      convBias?: boolean;
      /** Chunk length of the chunked prefill form (see {@link LinearAttentionBlock.chunkTokens}; default 64). Unit: tokens. */
      chunkTokens?: number;
      /** Low-rank gate rank r, set iff variant is 'kda'. Unit: elements. */
      gateRank?: number;
    };

/** Feed-forward parameters of one decoder layer (dims are logical, unsharded). */
export type FfnParams =
  /** Dense MLP. Maps to an {@link MlpBlock}. */
  | {
      kind: 'dense';
      /** Intermediate width. Unit: elements. */
      intermediate: number;
      /** Activation (default 'silu'; 'relu2' only with gated: false). */
      activation?: 'silu' | 'gelu' | 'relu' | 'relu2';
      /** GLU-style (default true). */
      gated?: boolean;
      /** Projection biases (default false). */
      bias?: boolean;
    }
  /** Mixture of experts. Maps to a {@link MoeBlock}. */
  | {
      kind: 'moe';
      /** Routed experts E. Count. */
      experts: number;
      /** Routed experts per token k. Count. */
      topK: number;
      /** Intermediate width of one routed expert. Unit: elements. */
      expertIntermediate: number;
      /** Latent width l of the routed experts (see {@link MoeBlock.latentDim}); default none (experts at hidden). Unit: elements. */
      latentDim?: number;
      /** Shared experts (default 0). Count. */
      sharedExperts?: number;
      /** Intermediate width of EACH shared expert (default 0). Unit: elements. */
      sharedIntermediate?: number;
      /** Activation (default 'silu'; 'relu2' only with gated: false). */
      activation?: 'silu' | 'gelu' | 'relu' | 'relu2';
      /** GLU-style experts (default true). */
      gated?: boolean;
    };

/** The mixing parameters of one {@link HcPreBlock} (the same for every hyper-connection of a model). */
export type HcPreParams = Pick<HcPreBlock, 'normGamma' | 'projections' | 'sinkhornIters' | 'vectorParams'>;

/** How the sublayers of a decoder layer write back into the residual stream. */
export type ResidualParams =
  /** One stream, `x = x + f(x)` after each sublayer (residual blocks). */
  | { kind: 'plain' }
  /** N streams with hyper-connections: an `hc_pre … hc_post` pair around each sublayer. */
  | {
      kind: 'hyper';
      /** Residual streams N; must equal the model's `residualStreams`. Count. */
      streams: number;
      /** Mixing parameters of each sublayer's `hc_pre` (Qwen: down/up/inject; DeepSeek/GLM: one `fn` projection). */
      pre: HcPreParams;
      /** True if `hc_post` mixes the streams with the N×N comb matrix (mHC); false: per-stream inject (Qwen). */
      mixesStreams: boolean;
      /** True if each sublayer still has its own pre-norm after `hc_pre` (DeepSeek, GLM); false: none (Qwen). */
      subNorms: boolean;
    };

/** Parameters of one pre-norm decoder layer. */
export interface DecoderLayerParams {
  /** Width of one residual stream. Unit: elements. */
  hidden: number;
  /** Token mixer (attention or linear attention). */
  mixer: MixerParams;
  /** Feed-forward block (dense or MoE). */
  ffn: FfnParams;
  /** Norm function of every norm in the layer (default 'rmsnorm'). */
  norm?: 'rmsnorm' | 'layernorm';
  /** Extra norms after the mixer and after the ffn, before each residual add (Gemma 2/3 style; default false). */
  postNorms?: boolean;
  /** Residual-stream form (default `{ kind: 'plain' }`). */
  residual?: ResidualParams;
  /**
   * A hashed n-gram embedding placed first in the layer (Qwen PLE at the start of layer index 1, DeepSeek Engram
   * at the start of layers 1 and 14); absent = none. Name default 'ngram'.
   */
  ngram?: Omit<NgramEmbeddingBlock, 'kind' | 'name'> & { name?: string };
  /** A short conv plus residual after the mixer and after the ffn (Inkling `use_sconv`); absent = none. */
  shortConv?: {
    /** Taps. Unit: tokens. */
    kernel: number;
  };
  /** Layer display name (default 'layer'). */
  name?: string;
}

/** Builds the mixer block of a layer from its params, applying the defaults. */
function mixerBlock(hidden: number, m: MixerParams): BlockSpec {
  switch (m.kind) {
    case 'gqa': {
      if (m.slidingWindow !== undefined && m.sparse !== undefined) {
        throw new Error('gqa mixer: slidingWindow and sparse are mutually exclusive');
      }
      const block: GqaAttentionBlock = {
        kind: 'gqa_attention',
        name: 'attn',
        hidden,
        heads: m.heads,
        kvHeads: m.kvHeads,
        headDim: m.headDim,
        qkvBias: m.qkvBias ?? false,
        outBias: m.outBias ?? false,
        qkNorm: m.qkNorm ?? false,
        rope: m.rope ?? true,
        causal: true,
        outputGate: m.outputGate ?? false,
        ...(m.slidingWindow !== undefined ? { slidingWindow: m.slidingWindow } : {}),
        ...(m.sparse !== undefined ? { sparse: m.sparse } : {}),
        ...(m.relativeBias !== undefined ? { relativeBias: m.relativeBias } : {}),
        ...(m.kvConvKernel !== undefined ? { kvConvKernel: m.kvConvKernel } : {}),
      };
      return block;
    }
    case 'mla': {
      const block: MlaAttentionBlock = {
        kind: 'mla_attention',
        name: 'attn',
        hidden,
        heads: m.heads,
        qLoraRank: m.qLoraRank,
        kvLoraRank: m.kvLoraRank,
        qkNopeDim: m.qkNopeDim,
        qkRopeDim: m.qkRopeDim,
        vHeadDim: m.vHeadDim,
        ...(m.sparse !== undefined ? { sparse: m.sparse } : {}),
        ...(m.kvCacheFromLayer !== undefined ? { kvCacheFromLayer: m.kvCacheFromLayer } : {}),
      };
      return block;
    }
    case 'csa': {
      const { kind: _kind, ...fields } = m;
      void _kind;
      const block: CsaAttentionBlock = { kind: 'csa_attention', name: 'attn', hidden, ...fields };
      return block;
    }
    case 'linear': {
      const block: LinearAttentionBlock = {
        kind: 'linear_attention',
        name: 'linear_attn',
        variant: m.variant,
        hidden,
        qkHeads: m.qkHeads,
        vHeads: m.vHeads,
        keyDim: m.keyDim,
        valueDim: m.valueDim,
        convKernel: m.convKernel ?? 4,
        outputGate: m.outputGate ?? false,
        ...(m.gateRank !== undefined ? { gateRank: m.gateRank } : {}),
        ...(m.convBias !== undefined ? { convBias: m.convBias } : {}),
        ...(m.chunkTokens !== undefined ? { chunkTokens: m.chunkTokens } : {}),
      };
      return block;
    }
    default: {
      const _never: never = m;
      throw new Error(`unknown mixer ${JSON.stringify(_never)}`);
    }
  }
}

/** Builds the feed-forward block of a layer from its params, applying the defaults. */
function ffnBlock(hidden: number, f: FfnParams): BlockSpec {
  switch (f.kind) {
    case 'dense': {
      const block: MlpBlock = {
        kind: 'mlp',
        name: 'mlp',
        hidden,
        intermediate: f.intermediate,
        gated: f.gated ?? true,
        activation: f.activation ?? 'silu',
        bias: f.bias ?? false,
      };
      return block;
    }
    case 'moe': {
      const block: MoeBlock = {
        kind: 'moe',
        name: 'moe',
        hidden,
        experts: f.experts,
        topK: f.topK,
        expertIntermediate: f.expertIntermediate,
        gated: f.gated ?? true,
        activation: f.activation ?? 'silu',
        sharedExperts: f.sharedExperts ?? 0,
        sharedIntermediate: f.sharedIntermediate ?? 0,
        // Only when given, so a block without it is deep-equal to one built before the field existed.
        ...(f.latentDim === undefined ? {} : { latentDim: f.latentDim }),
      };
      return block;
    }
    default: {
      const _never: never = f;
      throw new Error(`unknown ffn ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * One sublayer (`prefix` 'mixer' or 'ffn') with its residual handling: plain `norm → branch → [short_conv] →
 * [post-norm] → residual`, or hyper `hc_pre → [norm] → branch → [short_conv] → [post-norm] → hc_post`.
 */
function sublayerBlocks(
  p: Pick<DecoderLayerParams, 'hidden' | 'norm' | 'residual' | 'shortConv' | 'postNorms'>,
  prefix: 'mixer' | 'ffn',
  branch: BlockSpec,
): BlockSpec[] {
  const norm = p.norm ?? 'rmsnorm';
  const h = p.hidden;
  const residual = p.residual ?? { kind: 'plain' };
  const preNormName = prefix === 'mixer' ? 'input_norm' : 'ffn_norm';
  const blocks: BlockSpec[] = [];
  if (residual.kind === 'hyper') {
    blocks.push({ kind: 'hc_pre', name: `${prefix}_hc_pre`, streams: residual.streams, dim: h, ...residual.pre });
    if (residual.subNorms) blocks.push({ kind: 'norm', name: preNormName, norm, dim: h });
  } else {
    blocks.push({ kind: 'norm', name: preNormName, norm, dim: h });
  }
  blocks.push(branch);
  if (p.shortConv !== undefined) {
    blocks.push({ kind: 'short_conv', name: `${prefix}_short_conv`, dim: h, kernel: p.shortConv.kernel });
  }
  if (p.postNorms === true) blocks.push({ kind: 'norm', name: `${prefix}_post_norm`, norm, dim: h });
  if (residual.kind === 'hyper') {
    blocks.push({
      kind: 'hc_post',
      name: `${prefix}_hc_post`,
      streams: residual.streams,
      dim: h,
      mixesStreams: residual.mixesStreams,
    });
  } else {
    blocks.push({ kind: 'residual', name: `${prefix}_residual`, dim: h });
  }
  return blocks;
}

/** Pre-norm decoder layer: [ngram], then the mixer and the ffn sublayers (see the module header for the layout). */
export function decoderLayer(p: DecoderLayerParams): LayerSpec {
  const h = p.hidden;
  const blocks: BlockSpec[] = [];
  if (p.ngram !== undefined) {
    const { name, ...fields } = p.ngram;
    const ngram: NgramEmbeddingBlock = { kind: 'ngram_embedding', name: name ?? 'ngram', ...fields };
    blocks.push(ngram);
  }
  blocks.push(...sublayerBlocks(p, 'mixer', mixerBlock(h, p.mixer)));
  blocks.push(...sublayerBlocks(p, 'ffn', ffnBlock(h, p.ffn)));
  return { name: p.name ?? 'layer', blocks };
}

/**
 * Parameters of a one-sublayer layer: exactly one of `mixer` (attention or linear attention / Mamba-2) or `ffn`
 * (dense or MoE), with a plain residual.
 */
export type SingleBlockLayerParams = {
  /** Width of the residual stream. Unit: elements. */
  hidden: number;
  /** Norm function of the pre-norm (default 'rmsnorm'). */
  norm?: 'rmsnorm' | 'layernorm';
  /** Layer display name (default 'layer'). */
  name?: string;
} & ({ mixer: MixerParams; ffn?: never } | { ffn: FfnParams; mixer?: never });

/**
 * A layer with ONE sublayer, `norm → mixer|ffn → residual` (Nemotron-H `hybrid_override_pattern`: each layer is a
 * Mamba-2, attention or MoE block). Same blocks and names as that sublayer of {@link decoderLayer}.
 */
export function singleBlockLayer(p: SingleBlockLayerParams): LayerSpec {
  const branch = p.mixer !== undefined ? mixerBlock(p.hidden, p.mixer) : ffnBlock(p.hidden, p.ffn);
  const blocks = sublayerBlocks(p, p.mixer !== undefined ? 'mixer' : 'ffn', branch);
  return { name: p.name ?? 'layer', blocks };
}

/**
 * The final hyper-connection collapse of N streams into one before the LM head, for `decoderModel`'s
 * `beforeHead` (Qwen `hyper_connection_mixer`: γ + down/up, no inject; DeepSeek `hc_pre` and GLM `hc_head`: no
 * projections, no params).
 */
export function hyperCollapse(p: {
  /** Residual streams N. Count. */
  streams: number;
  /** Width of one stream D. Unit: elements. */
  dim: number;
  /** Mixing parameters of the collapse. */
  pre: HcPreParams;
}): HcPreBlock {
  return { kind: 'hc_pre', name: 'final_hc_pre', streams: p.streams, dim: p.dim, ...p.pre };
}

/** A decoder-only model from its layers (finalNorm default 'rmsnorm', residualStreams default 1). */
export function decoderModel(p: {
  /** Display name. */
  name: string;
  /** Width of one residual stream. Unit: elements. */
  hidden: number;
  /** Vocabulary size. Count. */
  vocab: number;
  /** LM head shares the embedding matrix. */
  tieEmbeddings: boolean;
  /** Layers in execution order. */
  layers: LayerSpec[];
  /** Norm before the LM head (default 'rmsnorm'; 'none' for models whose final collapse normalizes). */
  finalNorm?: 'rmsnorm' | 'layernorm' | 'none';
  /** Parallel residual streams N (default 1, omitted from the spec). Count. */
  residualStreams?: number;
  /**
   * Blocks run after the last layer and before the final norm, e.g. the final {@link hyperCollapse}. They are
   * appended to the LAST layer's blocks, so they are lowered (and pipeline-staged) as part of the last layer.
   */
  beforeHead?: BlockSpec[];
  /** Where the dims came from. */
  source?: string;
}): ModelSpec {
  let layers = p.layers;
  if (p.beforeHead !== undefined && p.beforeHead.length > 0) {
    const last = layers.at(-1);
    if (last === undefined) throw new Error('decoderModel: beforeHead needs at least one layer');
    layers = [...layers.slice(0, -1), { ...last, blocks: [...last.blocks, ...p.beforeHead] }];
  }
  return {
    name: p.name,
    hidden: p.hidden,
    vocab: p.vocab,
    tieEmbeddings: p.tieEmbeddings,
    layers,
    finalNorm: p.finalNorm ?? 'rmsnorm',
    ...(p.residualStreams !== undefined ? { residualStreams: p.residualStreams } : {}),
    ...(p.source !== undefined ? { source: p.source } : {}),
  };
}

/** `n` layers built by `make(i)` for i = 0..n−1 (heterogeneous stacks pick per index). */
export function repeatLayers(n: number, make: (i: number) => LayerSpec): LayerSpec[] {
  const layers: LayerSpec[] = [];
  for (let i = 0; i < n; i++) layers.push(make(i));
  return layers;
}
