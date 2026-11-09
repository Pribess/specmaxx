// DeepSeek-V4-style CSA additions on a tiny made-up block (inline, not in models/reference.ts, whose exports the
// identity pin covers): the overlapping compressor and indexer key pools (r = 4, `compressorOverlap`,
// `keyPoolOverlap`) and the attend-all entry selection (`{ kind: 'all' }`, r = 8 here; r = 128 in DeepSeek-V4).
// bf16 everywhere (2 B/element). See docs/simulation-model.md §5 "CSA".

import { describe, expect, it } from 'vitest';
import { decoderLayer, decoderModel, repeatLayers, type MixerParams } from '../ir/builders';
import { attentionPairs, opBytes } from '../ir/metrics';
import type { CsaAttentionBlock, ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import type { AttentionOp, ElementwiseOp, GemmOp, Op } from '../ir/ops';
import { blockParams } from '../ir/params';
import { crossLayerReferences, modelSpecErrors } from '../ir/validateModel';
import {
  blockResidentWeightBytes,
  blockSequenceStateBytes,
  lowerModel,
  type LoweringContext,
} from './lower';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

/** h 128, H 4, hd 32, dr 8, rq 32, G 2 × R_o 16, W 16 (the TINY_CSA dims). */
const BASE: Omit<Extract<MixerParams, { kind: 'csa' }>, 'global'> = {
  kind: 'csa',
  heads: 4,
  headDim: 32,
  ropeDim: 8,
  qLoraRank: 32,
  outGroups: 2,
  outLoraRank: 16,
  windowTokens: 16,
};

/**
 * Layer 0: window only. Layer 1: r = 4, overlapping compressor + own indexer (2 heads × 16, top-4 entries, keys
 * pooled 4 tokens from overlapping windows with a gate, from the hidden state). Layer 2: r = 8, own compressor,
 * attends every compressed entry.
 */
const TINY_V4: ModelSpec = decoderModel({
  name: 'TINY_CSA_V4',
  hidden: 128,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture (DeepSeek-V4-style)',
  layers: repeatLayers(3, (i) => {
    const mixers: MixerParams[] = [
      BASE,
      {
        ...BASE,
        global: {
          compressRatio: 4,
          compressorOverlap: true,
          selection: {
            kind: 'indexer',
            indexer: {
              heads: 2,
              headDim: 16,
              topK: 4,
              keyPoolTokens: 4,
              keyPoolGate: true,
              keyPoolOverlap: true,
            },
          },
        },
      },
      { ...BASE, global: { compressRatio: 8, selection: { kind: 'all' } } },
    ];
    return decoderLayer({
      name: `layer.${i}`,
      hidden: 128,
      mixer: mixers[i],
      ffn: { kind: 'dense', intermediate: 256 },
    });
  }),
});

/** Prefill: 2 sequences of 64 tokens (T = 128). Decode: 2 sequences at a 256-token context (T = 2). */
function ctx(phase: 'prefill' | 'decode'): LoweringContext {
  const parallel: ParallelConfig = { tp: 1, pp: 1, dp: 1, ep: 1 };
  return {
    phase,
    seqs: 2,
    qLen: phase === 'prefill' ? 64 : 1,
    kvLen: phase === 'prefill' ? 64 : 256,
    quant: BF16,
    parallel,
  };
}

function layerOps(i: number, c: LoweringContext): Op[] {
  const layer = lowerModel(TINY_V4, c)
    .stages.flatMap((s) => s.layers)
    .find((l) => l.layerIndex === i);
  if (layer === undefined) throw new Error(`no layer ${i}`);
  return layer.ops;
}
const byName = (ops: Op[], name: string): Op => {
  const op = ops.find((o) => o.name === name);
  if (op === undefined) throw new Error(`no op ${name} in ${ops.map((o) => o.name).join(', ')}`);
  return op;
};
const csaBlock = (i: number): CsaAttentionBlock => {
  const b = TINY_V4.layers[i].blocks.find((x) => x.kind === 'csa_attention');
  if (b?.kind !== 'csa_attention') throw new Error('no csa block');
  return b;
};

describe('DeepSeek-V4 CSA additions (tiny block: h 128, H 4, hd 32, dr 8, rq 32, G 2 × R_o 16, W 16)', () => {
  it('is a valid model with no cross-layer references; resident weights = params · 2 B', () => {
    expect(modelSpecErrors(TINY_V4)).toEqual([]);
    expect(crossLayerReferences(TINY_V4)).toEqual([]);
    for (const i of [0, 1, 2])
      expect(blockResidentWeightBytes(csaBlock(i), ctx('decode'))).toBe(
        2 * blockParams(csaBlock(i)),
      );
  });

  it('r = 4 overlap: ops in order, coff = 2 projections, pooled-key ops', () => {
    const ops = layerOps(1, ctx('decode'));
    expect(
      ops
        .map((o) => o.name)
        .filter((n) => n.startsWith('csa.') || n.startsWith('idx.') || n === 'attn.core'),
    ).toEqual([
      'csa.q_a',
      'csa.q_norm',
      'csa.q_b',
      'csa.q_rope',
      'csa.wkv',
      'csa.kv_norm',
      'csa.k_rope',
      'csa.wkv_c',
      'csa.wgate_c',
      'csa.kv_pool',
      'csa.compress_norm',
      'csa.compress_rope',
      'idx.q',
      'idx.k',
      'idx.k_norm',
      'idx.pool_gate',
      'idx.kv_pool',
      'idx.w',
      'idx.score',
      'attn.core',
      'csa.o_inverse_rope',
      'csa.wo_a',
      'csa.wo_b',
    ]);
    // wkv_c, wgate_c: n = coff·hd = 2·32 = 64
    expect((byName(ops, 'csa.wkv_c') as GemmOp).n).toBe(64);
    expect((byName(ops, 'csa.wgate_c') as GemmOp).n).toBe(64);
    // kv_pool over T·coff·hd = 2·64 = 128 elements: read 2·128·2 = 512 B, write (T·hd/r)·2 = (2·32/4)·2 = 32 B
    const pool = byName(ops, 'csa.kv_pool') as ElementwiseOp;
    expect([pool.elems, pool.activationReadBytes, pool.activationWriteBytes]).toEqual([
      128, 512, 32,
    ]);
    // idx.k, idx.pool_gate: n = coff·hd_I = 2·16 = 32, k = h = 128 (keys from the hidden state)
    expect([(byName(ops, 'idx.k') as GemmOp).n, (byName(ops, 'idx.k') as GemmOp).k]).toEqual([
      32, 128,
    ]);
    expect((byName(ops, 'idx.pool_gate') as GemmOp).n).toBe(32);
    // idx.kv_pool: elems T·coff·hd_I = 64; write (T·hd_I/p)·2 = (2·16/4)·2 = 16 B; APE p·coff·hd_I·2 = 4·32·2 = 256 B
    const kpool = byName(ops, 'idx.kv_pool') as ElementwiseOp;
    expect([kpool.elems, kpool.activationWriteBytes, kpool.weightBytes]).toEqual([64, 16, 256]);
    // The attention keeps the indexer's top-4 entries.
    expect((byName(ops, 'attn.core') as AttentionOp).secondaryKv?.kvSelection).toEqual({
      kind: 'topk',
      topK: 4,
    });
  });

  it('r = 4 overlap: params by hand', () => {
    // q = h·rq + rq + rq·H·hd = 4096 + 32 + 4096 = 8224; kv = h·hd + hd = 4128;
    // o = G·R_o·(H·hd/G) + G·R_o·h = 2048 + 4096 = 6144; compressor 2·h·coff·hd + hd = 2·128·2·32 + 32 = 16,416;
    // indexer: rq·H_I·hd_I 1024 + (h·coff·hd_I + 2·hd_I) 4128 + (h·coff·hd_I + p·coff·hd_I) 4224 + h·H_I 256 = 9632
    expect(blockParams(csaBlock(1))).toBe(8224 + 4128 + 6144 + 16_416 + 9632);
    // r = 8 'all': compressor 2·h·hd + hd = 8224, no indexer
    expect(blockParams(csaBlock(2))).toBe(8224 + 4128 + 6144 + 8224);
  });

  it("r = 8 'all': no indexer ops, full selection over the entries", () => {
    const ops = layerOps(2, ctx('decode'));
    expect(ops.map((o) => o.name).filter((n) => n.startsWith('idx.'))).toEqual([]);
    expect((byName(ops, 'csa.wkv_c') as GemmOp).n).toBe(32);
    const att = byName(ops, 'attn.core') as AttentionOp;
    expect(att.secondaryKv?.kvSelection).toEqual({ kind: 'full' });
    // decode at kvLen 256: min(W, kvLen) + kvLen/r = 16 + 32 = 48 pairs
    expect(attentionPairs(att)).toBe(16 + 256 / 8);
    // reads: 2 seqs·(16 window tokens + 32 entries)·32·2 B = 6144; writes 2·(32 + 32/8)·2 = 144
    expect(opBytes(att).kvReadBytes).toBe(6144);
    expect(opBytes(att).kvWriteBytes).toBe(144);
    // prefill (qLen 64, no past): window 16·17/2 + 48·16 = 904; entries (64·65/2)/8 = 260
    const pre = byName(layerOps(2, ctx('prefill')), 'attn.core') as AttentionOp;
    expect(attentionPairs(pre)).toBe(904 + 260);
  });

  it('sequence state: doubled fp32 pooling tail with overlap, no keys for all', () => {
    const state = (i: number) => blockSequenceStateBytes(csaBlock(i), ctx('decode'), 256);
    // window min(256, 16)·32·2 = 1024
    expect(state(0)).toBe(1024);
    // + entries ceil(256/4)·32·2 = 4096 + tail 2·(coff·r)·(coff·hd)·4 = 2·8·64·4 = 4096 + keys ceil(256/4)·16·2 = 2048
    expect(state(1)).toBe(1024 + 4096 + 4096 + 2048);
    // + entries ceil(256/8)·32·2 = 2048 + tail 2·8·32·4 = 2048; no indexer keys
    expect(state(2)).toBe(1024 + 2048 + 2048);
  });
});
