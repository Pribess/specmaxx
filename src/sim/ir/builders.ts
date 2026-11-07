/**
 * Builder API for model specs: the follow-up Hugging Face config mappers use only these functions, so they stay
 * architecture-only (no tokens, no parallelism). A decoder layer is a pre-norm block sequence:
 * norm → mixer → [post-norm] → residual → norm → ffn → [post-norm] → residual.
 *
 * Defaults (when a field is omitted): activation `'silu'`, gated `true`, norm `'rmsnorm'`, rope `true`,
 * convKernel 4, outputGate `false`; every other boolean `false` and every other count 0.
 * See docs/simulation-model.md §2 (Model IR).
 */

import type {
  BlockSpec,
  GqaAttentionBlock,
  IndexerSpec,
  LayerSpec,
  LinearAttentionBlock,
  MlaAttentionBlock,
  MlpBlock,
  ModelSpec,
  MoeBlock,
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
      /** Top-k sparse attention indexer; absent = dense. */
      sparse?: IndexerSpec;
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
      /** Per-head RoPE q/k dim dr. Unit: elements. */
      qkRopeDim: number;
      /** Per-head value dim dv. Unit: elements. */
      vHeadDim: number;
      /** DSA top-k sparse attention; absent = dense. */
      sparse?: IndexerSpec;
    }
  /** Linear attention / SSM mixer. Maps to a {@link LinearAttentionBlock}. */
  | {
      kind: 'linear';
      /** Recurrence variant. */
      variant: 'gated_delta_net' | 'mamba2' | 'linear';
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
      /** z-gated output norm (default false). */
      outputGate?: boolean;
    };

/** Feed-forward parameters of one decoder layer (dims are logical, unsharded). */
export type FfnParams =
  /** Dense MLP. Maps to an {@link MlpBlock}. */
  | {
      kind: 'dense';
      /** Intermediate width. Unit: elements. */
      intermediate: number;
      /** Activation (default 'silu'). */
      activation?: 'silu' | 'gelu' | 'relu';
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
      /** Shared experts (default 0). Count. */
      sharedExperts?: number;
      /** Intermediate width of EACH shared expert (default 0). Unit: elements. */
      sharedIntermediate?: number;
      /** Activation (default 'silu'). */
      activation?: 'silu' | 'gelu' | 'relu';
      /** GLU-style experts (default true). */
      gated?: boolean;
    };

/** Parameters of one pre-norm decoder layer. */
export interface DecoderLayerParams {
  /** Residual-stream width. Unit: elements. */
  hidden: number;
  /** Token mixer (attention or linear attention). */
  mixer: MixerParams;
  /** Feed-forward block (dense or MoE). */
  ffn: FfnParams;
  /** Norm function of every norm in the layer (default 'rmsnorm'). */
  norm?: 'rmsnorm' | 'layernorm';
  /** Extra norms after the mixer and after the ffn, before each residual add (Gemma 2/3 style; default false). */
  postNorms?: boolean;
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
      };
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
      };
      return block;
    }
    default: {
      const _never: never = f;
      throw new Error(`unknown ffn ${JSON.stringify(_never)}`);
    }
  }
}

/** Pre-norm decoder layer: norm, mixer, [postNorm], residual, norm, ffn, [postNorm], residual. */
export function decoderLayer(p: DecoderLayerParams): LayerSpec {
  const norm = p.norm ?? 'rmsnorm';
  const h = p.hidden;
  const blocks: BlockSpec[] = [{ kind: 'norm', name: 'input_norm', norm, dim: h }, mixerBlock(h, p.mixer)];
  if (p.postNorms === true) blocks.push({ kind: 'norm', name: 'mixer_post_norm', norm, dim: h });
  blocks.push({ kind: 'residual', name: 'mixer_residual', dim: h });
  blocks.push({ kind: 'norm', name: 'ffn_norm', norm, dim: h });
  blocks.push(ffnBlock(h, p.ffn));
  if (p.postNorms === true) blocks.push({ kind: 'norm', name: 'ffn_post_norm', norm, dim: h });
  blocks.push({ kind: 'residual', name: 'ffn_residual', dim: h });
  return { name: p.name ?? 'layer', blocks };
}

/** A decoder-only model from its layers (finalNorm default 'rmsnorm'). */
export function decoderModel(p: {
  /** Display name. */
  name: string;
  /** Residual-stream width. Unit: elements. */
  hidden: number;
  /** Vocabulary size. Count. */
  vocab: number;
  /** LM head shares the embedding matrix. */
  tieEmbeddings: boolean;
  /** Layers in execution order. */
  layers: LayerSpec[];
  /** Norm before the LM head (default 'rmsnorm'). */
  finalNorm?: 'rmsnorm' | 'layernorm';
  /** Where the dims came from. */
  source?: string;
}): ModelSpec {
  return {
    name: p.name,
    hidden: p.hidden,
    vocab: p.vocab,
    tieEmbeddings: p.tieEmbeddings,
    layers: p.layers,
    finalNorm: p.finalNorm ?? 'rmsnorm',
    ...(p.source !== undefined ? { source: p.source } : {}),
  };
}

/** `n` layers built by `make(i)` for i = 0..n−1 (heterogeneous stacks pick per index). */
export function repeatLayers(n: number, make: (i: number) => LayerSpec): LayerSpec[] {
  const layers: LayerSpec[] = [];
  for (let i = 0; i < n; i++) layers.push(make(i));
  return layers;
}
