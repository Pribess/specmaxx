import { describe, expect, it } from 'vitest';
import { decoderLayer, decoderModel, type HcPreParams, type MixerParams } from './builders';
import type { IndexerSpec, ModelSpec } from './model';
import { modelSpecErrors } from './validateModel';
import { REF_DENSE_8B, TINY_HYBRID, TINY_MLA_SPARSE } from '../models/reference';

const INDEXER: IndexerSpec = { heads: 2, headDim: 16, topK: 32 };

/** An MLA mixer of a tiny test model (rkv 64, dr 16), with optional sparse selection and latent source. */
function mla(extra: Partial<Extract<MixerParams, { kind: 'mla' }>> = {}): MixerParams {
  return { kind: 'mla', heads: 4, qLoraRank: 64, kvLoraRank: 64, qkNopeDim: 32, qkRopeDim: 16, vHeadDim: 32, ...extra };
}

/** A tiny CSA mixer (hd 64, window 16) with an optional global branch. */
function csa(global?: Extract<MixerParams, { kind: 'csa' }>['global']): MixerParams {
  return {
    kind: 'csa',
    heads: 4,
    headDim: 64,
    ropeDim: 16,
    qLoraRank: 64,
    outGroups: 2,
    outLoraRank: 32,
    windowTokens: 16,
    ...(global !== undefined ? { global } : {}),
  };
}

/** A model of one layer per mixer (hidden 128, dense ffn 256). */
function model(mixers: MixerParams[], extra: Partial<ModelSpec> = {}): ModelSpec {
  return {
    ...decoderModel({
      name: 'test',
      hidden: 128,
      vocab: 100,
      tieEmbeddings: true,
      layers: mixers.map((mixer, i) =>
        decoderLayer({ name: `layer.${i}`, hidden: 128, mixer, ffn: { kind: 'dense', intermediate: 256 } }),
      ),
    }),
    ...extra,
  };
}

describe('modelSpecErrors', () => {
  it('accepts the reference models', () => {
    for (const m of [REF_DENSE_8B, TINY_HYBRID, TINY_MLA_SPARSE]) expect(modelSpecErrors(m)).toEqual([]);
  });

  it('MLA kvCacheFromLayer: earlier owner with the same rkv and dr (Mistral Large 4)', () => {
    expect(modelSpecErrors(model([mla(), mla({ kvCacheFromLayer: 0 })]))).toEqual([]);
    expect(modelSpecErrors(model([mla({ kvCacheFromLayer: 1 }), mla()]))[0]).toMatch(/layer 0: .*earlier layer/);
    expect(modelSpecErrors(model([mla(), mla({ kvCacheFromLayer: 0, qkRopeDim: 0 })]))[0]).toMatch(/layer 1: .*rkv/);
    // a consumer cannot be a source
    const chain = model([mla(), mla({ kvCacheFromLayer: 0 }), mla({ kvCacheFromLayer: 1 })]);
    expect(modelSpecErrors(chain)[0]).toMatch(/layer 2: .*does not own/);
    const gqa: MixerParams = { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 32 };
    expect(modelSpecErrors(model([gqa, mla({ kvCacheFromLayer: 0 })]))[0]).toMatch(/no mla_attention/);
  });

  it("'reuse' needs an earlier indexer of the same kind (GLM-5.3 shared indexer)", () => {
    const own = mla({ sparse: { kind: 'indexer', indexer: INDEXER } });
    const reuse = mla({ sparse: { kind: 'reuse', topK: 32 } });
    expect(modelSpecErrors(model([own, reuse, reuse]))).toEqual([]);
    expect(modelSpecErrors(model([reuse, own]))[0]).toMatch(/layer 0: .*no earlier layer runs an indexer/);
    const gqaOwn: MixerParams = { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 32, sparse: { kind: 'indexer', indexer: INDEXER } };
    expect(modelSpecErrors(model([gqaOwn, reuse]))[0]).toMatch(/layer 1: .*not compatible/);
  });

  it('CSA: compressed-cache source and reuse need the same compress ratio (DeepSeek-V4.1)', () => {
    const ownIx = { kind: 'indexer', indexer: { ...INDEXER, keyPoolTokens: 2, keyInputDim: 64 } } as const;
    const layers = [
      csa(),
      csa({ compressRatio: 2, selection: ownIx }),
      csa({ compressRatio: 2, kvCacheFromLayer: 1, selection: { kind: 'reuse', topK: 32 } }),
      csa({ compressRatio: 1, selection: { kind: 'indexer', indexer: { ...INDEXER, keyInputDim: 64 } } }),
      csa({
        compressRatio: 1,
        kvCacheFromLayer: 3,
        selection: { kind: 'indexer', indexer: { ...INDEXER, ownsKeys: false, maxScoredKeys: 64 } },
      }),
    ];
    expect(modelSpecErrors(model(layers))).toEqual([]);
    const badRatio = [...layers.slice(0, 2), csa({ compressRatio: 1, kvCacheFromLayer: 1, selection: { kind: 'reuse', topK: 32 } })];
    const errors = modelSpecErrors(model(badRatio));
    expect(errors.some((e) => /layer 2: global.kvCacheFromLayer 1 has compress ratio/.test(e))).toBe(true);
    expect(errors.some((e) => /layer 2: reuses the top-k of layer 1/.test(e))).toBe(true);
    expect(modelSpecErrors(model([csa(), csa({ compressRatio: 2, kvCacheFromLayer: 0, selection: ownIx })]))[0]).toMatch(
      /no csa global branch/,
    );
  });

  it('ownsKeys false needs an earlier key owner with the same key shape', () => {
    const foreign = mla({ sparse: { kind: 'indexer', indexer: { ...INDEXER, ownsKeys: false } } });
    expect(modelSpecErrors(model([foreign]))[0]).toMatch(/no earlier layer owns indexer keys/);
    const owner = mla({ sparse: { kind: 'indexer', indexer: { ...INDEXER, headDim: 32 } } });
    expect(modelSpecErrors(model([owner, foreign]))[0]).toMatch(/headDim or keyPoolTokens differ/);
  });

  it('gateRank iff kda, and kda needs outputGate', () => {
    const kda = (extra: object): MixerParams => ({
      kind: 'linear',
      variant: 'kda',
      qkHeads: 2,
      vHeads: 2,
      keyDim: 32,
      valueDim: 32,
      outputGate: true,
      gateRank: 32,
      ...extra,
    });
    expect(modelSpecErrors(model([kda({})]))).toEqual([]);
    expect(modelSpecErrors(model([kda({ gateRank: undefined })]))[0]).toMatch(/gateRank/);
    expect(modelSpecErrors(model([kda({ variant: 'gated_delta_net' })]))[0]).toMatch(/gateRank/);
    expect(modelSpecErrors(model([kda({ outputGate: false })]))[0]).toMatch(/outputGate/);
  });

  it('hc blocks and n-gram width must match residualStreams · hidden', () => {
    const pre: HcPreParams = { normGamma: false, projections: [], sinkhornIters: 20, vectorParams: 0 };
    const layer = decoderLayer({
      hidden: 128,
      mixer: { kind: 'gqa', heads: 4, kvHeads: 2, headDim: 32 },
      ffn: { kind: 'dense', intermediate: 256 },
      residual: { kind: 'hyper', streams: 4, pre, mixesStreams: true, subNorms: true },
      ngram: { tableRows: 10, rowDim: 8, rowsPerToken: 2, projections: [], outputDim: 512, vectorParams: 0 },
    });
    const base = { name: 'hc', hidden: 128, vocab: 10, tieEmbeddings: true, layers: [layer] };
    expect(modelSpecErrors(decoderModel({ ...base, residualStreams: 4 }))).toEqual([]);
    const errors = modelSpecErrors(decoderModel(base));
    expect(errors.filter((e) => /residualStreams 1 × hidden 128/.test(e))).toHaveLength(4);
    expect(errors.some((e) => /outputDim 512 must equal residualStreams·hidden 128/.test(e))).toBe(true);
  });
});
