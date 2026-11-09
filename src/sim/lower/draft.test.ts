import { describe, expect, it } from 'vitest';
import { opBytes, opFlops } from '../ir/metrics';
import type { BlockDraftSpec, QuantConfig } from '../ir/model';
import type { AttentionOp, ElementwiseOp, EmbeddingOp, GemmOp, Op, RecurrentOp } from '../ir/ops';
import { draftContextRows, expectedTokens } from '../engine/specdec';
import { parseDrafterConfig } from '../models/hf/drafter';
import { REF_DENSE_8B, TINY_HYBRID } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import {
  blockDrafterParams,
  draftResidentWeightBytes,
  draftSequenceStateBytes,
  lowerDraftDecode,
  lowerDraftPrefill,
} from './draft';
import {
  blockResidentWeightBytes,
  lowerBlock,
  lowerModel,
  type LoweredLayer,
  type LoweringContext,
} from './lower';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const P1 = { tp: 1, pp: 1, dp: 1, ep: 1 };

const RAW = import.meta.glob<string>('../models/hf/__fixtures__/*.json', {
  query: '?raw',
  import: 'default',
  eager: true,
});

/** A drafter config fixture parsed into its spec. */
function drafter(file: string, repoId: string): BlockDraftSpec {
  const raw = RAW[`../models/hf/__fixtures__/${file}`];
  if (raw === undefined) throw new Error(`missing fixture ${file}`);
  const r = parseDrafterConfig(JSON.parse(raw) as Record<string, unknown>, repoId);
  if (r.kind !== 'ok') throw new Error(r.errors.join('; '));
  return r.spec;
}
const DFLASH2 = drafter('incoai__Qwen3.8-27B-DFlash2.config.json', 'incoai/Qwen3.8-27B-DFlash2');
const DSPARK = drafter(
  'deepseek-ai__dspark_qwen3_8b_block7.config.json',
  'deepseek-ai/dspark_qwen3_8b_block7',
);
const MTP = REF_DENSE_8B_MTP.draft!;

/** The verify context of k drafts at decode context N (qLen k + 1, kvLen N + k + 1). */
function verifyCtx(N: number, k: number, seqs = 1): LoweringContext {
  return { phase: 'decode', seqs, qLen: k + 1, kvLen: N + k + 1, quant: BF16, parallel: P1 };
}
function prefillCtx(P: number, seqs = 1): LoweringContext {
  return { phase: 'prefill', seqs, qLen: P, kvLen: P, quant: BF16, parallel: P1 };
}
const ops = (layers: LoweredLayer[]): Op[] => layers.flatMap((l) => l.ops);
const names = (layers: LoweredLayer[]) => ops(layers).map((op) => op.name);
function find<T extends Op>(list: Op[], name: string): T {
  const op = list.find((o) => o.name === name);
  if (op === undefined) throw new Error(`no op ${name} in ${list.map((o) => o.name).join(', ')}`);
  return op as T;
}
const all = <T extends Op>(list: Op[], name: string) => list.filter((o) => o.name === name) as T[];
const shape = (g: GemmOp) => [g.m, g.n, g.k];
const sum = (list: Op[], f: (op: Op) => number) => list.reduce((s, op) => s + f(op), 0);

describe('MTP decode ops (U-9, design doc §3.3)', () => {
  it('k 1, m_ctx 2: one catch-up pass embed(2) → enorm, hnorm → eh_proj(2, 4096, 8192) → layer(qLen 2) → final_norm(1 row) → lm_head(m 1)', () => {
    const layers = lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 1), 2, 1);
    expect(layers.map((l) => l.name)).toEqual(['draft.0']);
    const list = ops(layers);
    expect(names(layers).every((n) => n.startsWith('draft.'))).toBe(true);
    expect(find<EmbeddingOp>(list, 'draft.embed').tokens).toBe(2);
    expect(find<ElementwiseOp>(list, 'draft.enorm').elems).toBe(2 * 4096);
    expect(find<ElementwiseOp>(list, 'draft.hnorm').elems).toBe(2 * 4096);
    expect(shape(find<GemmOp>(list, 'draft.eh_proj'))).toEqual([2, 4096, 8192]);
    const core = find<AttentionOp>(list, 'draft.attn.core');
    expect([core.qLen, core.kvLen, core.causal]).toEqual([2, 8194, true]);
    expect(core.kvWriteTokens).toBeUndefined();
    expect(shape(find<GemmOp>(list, 'draft.attn.qkv_proj'))[0]).toBe(2);
    expect(shape(find<GemmOp>(list, 'draft.mlp.gate_up'))[0]).toBe(2);
    expect(find<ElementwiseOp>(list, 'draft.final_norm').elems).toBe(4096);
    expect(shape(find<GemmOp>(list, 'draft.lm_head'))).toEqual([1, 151936, 4096]);
  });
  it('the module layer ops are the target layer ops at qLen m_ctx, renamed draft.*', () => {
    const ctx = verifyCtx(8192, 1);
    const list = ops(lowerDraftDecode(REF_DENSE_8B, MTP, ctx, 2, 1));
    const layerCtx: LoweringContext = { ...ctx, qLen: 2 };
    const expected =
      MTP.kind === 'mtp' ? MTP.layers[0].blocks.flatMap((b) => lowerBlock(b, layerCtx)) : [];
    const actual = list.filter(
      (op) =>
        ![
          'draft.embed',
          'draft.enorm',
          'draft.hnorm',
          'draft.eh_proj',
          'draft.final_norm',
          'draft.lm_head',
        ].includes(op.name),
    );
    expect(actual).toEqual(expected.map((op) => ({ ...op, name: `draft.${op.name}` })));
  });
  it('W2 draft bytes: embed 16,384 + eh_proj 67,108,864 + layer weights + KV 8194·4096 + 2·4096 + lm_head 1,244,659,712', () => {
    const list = ops(lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 1), 2, 1));
    expect(opBytes(find(list, 'draft.embed')).weightBytes).toBe(16_384);
    expect(opBytes(find(list, 'draft.eh_proj')).weightBytes).toBe(67_108_864);
    expect(opBytes(find(list, 'draft.lm_head')).weightBytes).toBe(1_244_659_712);
    const core = opBytes(find(list, 'draft.attn.core'));
    expect(core.kvReadBytes).toBe(8194 * 4096);
    expect(core.kvWriteBytes).toBe(2 * 4096);
    const layerGemms = list.filter(
      (op) =>
        op.kind === 'gemm' && (op.name.startsWith('draft.attn') || op.name.startsWith('draft.mlp')),
    );
    const gemmWeights = sum(layerGemms, (op) => opBytes(op).weightBytes);
    expect(gemmWeights).toBe(385_875_968);
  });
  it('k 3: the catch-up pass (qLen m_ctx) + two one-row passes, all over N + k + 1 keys', () => {
    const layers = lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 3), 3, 3);
    expect(layers.map((l) => l.name)).toEqual(['draft.0', 'draft.1', 'draft.2']);
    const cores = layers.map((l) => find<AttentionOp>(l.ops, 'draft.attn.core'));
    expect(cores.map((c) => [c.qLen, c.kvLen])).toEqual([
      [3, 8196],
      [1, 8196],
      [1, 8196],
    ]);
    expect(layers.map((l) => shape(find<GemmOp>(l.ops, 'draft.eh_proj'))[0])).toEqual([3, 1, 1]);
    expect(layers.map((l) => shape(find<GemmOp>(l.ops, 'draft.lm_head'))[0])).toEqual([1, 1, 1]);
  });
  it('batch 8: rows scale with b (eh_proj m 16, lm_head m 8)', () => {
    const list = ops(lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 1, 8), 2, 1));
    expect(shape(find<GemmOp>(list, 'draft.eh_proj'))[0]).toBe(16);
    expect(shape(find<GemmOp>(list, 'draft.lm_head'))[0]).toBe(8);
  });
  it('rejects a non-verify context and an m_ctx outside [1, k + 1]', () => {
    expect(() =>
      lowerDraftDecode(REF_DENSE_8B, MTP, { ...verifyCtx(8192, 1), qLen: 1 }, 1, 1),
    ).toThrow(/verify context/);
    expect(() => lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 1), 3, 1)).toThrow(/m_ctx/);
    expect(() => lowerDraftDecode(REF_DENSE_8B, MTP, prefillCtx(8192), 1, 1)).toThrow(/decode/);
  });
});

describe('MTP cache advance (U-18)', () => {
  it('alphas [1] (E[τ] 2): qLen 2, KV write 2·4096 B, eh_proj m 2, lm_head m 1', () => {
    const m = draftContextRows(expectedTokens({ kind: 'per_position', alphas: [1] }, 1));
    expect(m).toBe(2);
    const list = ops(lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 1), m, 1));
    const core = find<AttentionOp>(list, 'draft.attn.core');
    expect(core.qLen).toBe(2);
    expect(opBytes(core).kvWriteBytes).toBe(8_192);
    expect(find<GemmOp>(list, 'draft.eh_proj').m).toBe(2);
    expect(find<GemmOp>(list, 'draft.lm_head').m).toBe(1);
  });
  it('alphas [0] (E[τ] 1): qLen 1, KV write 4,096 B; the previous tentative row is overwritten, not charged', () => {
    const m = draftContextRows(expectedTokens({ kind: 'per_position', alphas: [0] }, 1));
    expect(m).toBe(1);
    const core = find<AttentionOp>(
      ops(lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 1), m, 1)),
      'draft.attn.core',
    );
    expect(core.qLen).toBe(1);
    expect(opBytes(core).kvWriteBytes).toBe(4_096);
    // one op writes the cache: no separate charge-back of the overwritten row
    expect(
      ops(lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(8192, 1), m, 1)).filter(
        (op) => opBytes(op).kvWriteBytes > 0,
      ),
    ).toHaveLength(1);
  });
  it('three all-accepted steps advance the committed length by 2 each', () => {
    const k = 1;
    const m = draftContextRows(expectedTokens({ kind: 'per_position', alphas: [1] }, k));
    let committed = 8192;
    const kvLens: number[] = [];
    for (let step = 0; step < 3; step++) {
      const core = find<AttentionOp>(
        ops(lowerDraftDecode(REF_DENSE_8B, MTP, verifyCtx(committed, k), m, k)),
        'draft.attn.core',
      );
      kvLens.push(core.kvLen);
      committed += m; // the module's cache now holds every committed token
    }
    expect(kvLens).toEqual([8194, 8196, 8198]);
  });
});

describe('DFlash 2 decode ops (U-9, Qwen3.8-27B-DFlash2 config, k 7, m_ctx 5)', () => {
  const N = 8192;
  const layers = lowerDraftDecode(REF_DENSE_8B, DFLASH2, verifyCtx(N, 7), 5, 7);
  const list = ops(layers);
  it('layers draft.input, draft.0..4, draft.head', () =>
    expect(layers.map((l) => l.name)).toEqual([
      'draft.input',
      'draft.0',
      'draft.1',
      'draft.2',
      'draft.3',
      'draft.4',
      'draft.head',
    ]));
  it('fc (5, 5120, 25600), hidden_norm over 5 rows, noise_embed of 8 rows', () => {
    expect(shape(find<GemmOp>(list, 'draft.fc'))).toEqual([5, 5120, 25600]);
    expect(find<ElementwiseOp>(list, 'draft.hidden_norm').elems).toBe(5 * 5120);
    expect(find<EmbeddingOp>(list, 'draft.noise_embed').tokens).toBe(8);
  });
  it('per layer: attn_conv.kernel_proj (8, 1280, 5120), q_proj (8, 4096, 5120), k/v_proj (13, 1024, 5120), mlp.gate_up (8, 34816, 5120)', () => {
    const layer = layers[1].ops;
    expect(shape(find<GemmOp>(layer, 'draft.attn_conv.kernel_proj'))).toEqual([8, 1280, 5120]);
    expect(shape(find<GemmOp>(layer, 'draft.mlp_conv.kernel_proj'))).toEqual([8, 1280, 5120]);
    expect(shape(find<GemmOp>(layer, 'draft.q_proj'))).toEqual([8, 4096, 5120]);
    expect(shape(find<GemmOp>(layer, 'draft.k_proj'))).toEqual([13, 1024, 5120]);
    expect(shape(find<GemmOp>(layer, 'draft.v_proj'))).toEqual([13, 1024, 5120]);
    expect(shape(find<GemmOp>(layer, 'draft.o_proj'))).toEqual([8, 5120, 4096]);
    expect(shape(find<GemmOp>(layer, 'draft.mlp.gate_up'))).toEqual([8, 34816, 5120]);
    expect(shape(find<GemmOp>(layer, 'draft.mlp.down'))).toEqual([8, 5120, 17408]);
    // qk_norm and rope over (8·32 + 13·8)·128 elements
    expect(find<ElementwiseOp>(layer, 'draft.qk_norm').elems).toBe((8 * 32 + 13 * 8) * 128);
    expect(find<ElementwiseOp>(layer, 'draft.attn_conv.prepare')).toMatchObject({
      fn: 'causal_conv1d',
      elems: 8 * 5120,
      flopsPerElem: 8,
    });
  });
  it('attn.core: qLen 8, kvLen min(N, 2048) + 13, window 2048, non-causal, kvWriteTokens 5', () => {
    const core = find<AttentionOp>(layers[1].ops, 'draft.attn.core');
    expect(core).toMatchObject({
      qLen: 8,
      kvLen: 2048 + 13,
      causal: false,
      kvWriteTokens: 5,
      heads: 32,
      kvHeads: 8,
    });
    expect(core.kvSelection).toEqual({ kind: 'window', windowTokens: 2048 });
    expect(opBytes(core).kvWriteBytes).toBe(5 * 2 * 8 * 128 * 2);
    // a short context: min(N, W) = N
    const short = find<AttentionOp>(
      ops(lowerDraftDecode(REF_DENSE_8B, DFLASH2, verifyCtx(100, 7), 5, 7)),
      'draft.attn.core',
    );
    expect(short.kvLen).toBe(100 + 13);
  });
  it('head: lm_head (7, 248320, 5120), selector.hidden_proj (7, 256, 5120), selector.codebooks 7·17 = 119 rows, topk over 7·V logits', () => {
    const head = layers.at(-1)!.ops;
    expect(shape(find<GemmOp>(head, 'draft.lm_head'))).toEqual([7, 248320, 5120]);
    expect(shape(find<GemmOp>(head, 'draft.selector.hidden_proj'))).toEqual([7, 256, 5120]);
    expect(find<EmbeddingOp>(head, 'draft.selector.codebooks')).toMatchObject({
      tokens: 119,
      hidden: 256,
    });
    expect(find<ElementwiseOp>(head, 'draft.selector.topk')).toMatchObject({
      fn: 'router_topk',
      elems: 7 * 248320,
    });
    expect(find<ElementwiseOp>(head, 'draft.selector.score')).toMatchObject({
      fn: 'candidate_score',
      elems: 7 * 16 * 256,
      flopsPerElem: 2,
    });
    expect(find<ElementwiseOp>(head, 'draft.norm').elems).toBe(8 * 5120);
    expect(names(layers).some((n) => n.startsWith('draft.markov'))).toBe(false);
  });
  it('k must fit the block (k + 1 ≤ blockSize 8)', () =>
    expect(() => lowerDraftDecode(REF_DENSE_8B, DFLASH2, verifyCtx(N, 8), 5, 8)).toThrow(
      /blockSize/,
    ));
  it('tp 2 adds 2·L + 1 = 11 all-reduces and shards heads and the MLP', () => {
    const tp2 = ops(
      lowerDraftDecode(
        REF_DENSE_8B,
        DFLASH2,
        { ...verifyCtx(N, 7), parallel: { ...P1, tp: 2 } },
        5,
        7,
      ),
    );
    expect(
      tp2.filter((op) => op.kind === 'collective' && op.collective === 'all_reduce'),
    ).toHaveLength(11);
    expect(
      tp2.filter((op) => op.kind === 'collective' && op.collective === 'all_gather'),
    ).toHaveLength(1);
    expect(find<GemmOp>(tp2, 'draft.q_proj').n).toBe(2048);
    expect(find<GemmOp>(tp2, 'draft.mlp.gate_up').n).toBe(17408);
    expect(find<GemmOp>(tp2, 'draft.lm_head').n).toBe(124160);
  });
});

describe('DSpark decode ops (U-9, dspark_qwen3_8b_block7 config, k 7)', () => {
  const layers = lowerDraftDecode(REF_DENSE_8B, DSPARK, verifyCtx(8192, 7), 5, 7);
  const list = ops(layers);
  it('+ 7 × markov.W2 (1, 151936, 256), each with a W1 gather of 1 row and a bias add', () => {
    const w2 = all<GemmOp>(list, 'draft.markov.W2');
    expect(w2.map(shape)).toEqual(Array(7).fill([1, 151936, 256]));
    expect(all<EmbeddingOp>(list, 'draft.markov.W1').map((o) => [o.tokens, o.hidden])).toEqual(
      Array(7).fill([1, 256]),
    );
    expect(all(list, 'draft.markov.add')).toHaveLength(7);
  });
  it('no convs, no selector', () => {
    expect(names(layers).some((n) => n.includes('conv') || n.includes('selector'))).toBe(false);
  });
  it('block of k = 7 positions (anchor + 6 masks), logits at all 7, full attention over N + m + 7 keys', () => {
    expect(find<EmbeddingOp>(list, 'draft.noise_embed').tokens).toBe(7);
    expect(shape(find<GemmOp>(list, 'draft.k_proj'))).toEqual([12, 1024, 4096]);
    expect(shape(find<GemmOp>(list, 'draft.lm_head'))).toEqual([7, 151936, 4096]);
    const core = find<AttentionOp>(list, 'draft.attn.core');
    expect(core).toMatchObject({ qLen: 7, kvLen: 8192 + 5 + 7, causal: false, kvWriteTokens: 5 });
    expect(core.kvSelection).toEqual({ kind: 'full' });
  });
});

describe('drafter parameter counts (U-10, design doc §3.3)', () => {
  it('DFlash 2 drafter: 1,924,404,480 (model card 1.924 B), ± 0.1%', () => {
    const p = blockDrafterParams(DFLASH2);
    expect(p).toBe(1_924_404_480);
    expect(Math.abs(p / 1.924e9 - 1)).toBeLessThan(1e-3);
  });
  it('DSpark drafter: 1,126,417,664 + its own embedding/head copies 2·151,936·4096 = 2,371,077,376 (safetensors 2.371 B), ± 1%', () => {
    const p = blockDrafterParams(DSPARK);
    expect(p).toBe(1_126_417_664);
    const withCopies = p + 2 * 151_936 * 4096;
    expect(withCopies).toBe(2_371_077_376);
    expect(Math.abs(withCopies / 2.371e9 - 1)).toBeLessThan(1e-2);
  });
});

describe('draft memory', () => {
  it('block drafters in bf16: resident bytes = 2 · params (shared embedding and head excluded)', () => {
    const ctx = verifyCtx(8192, 7);
    expect(draftResidentWeightBytes(REF_DENSE_8B, DFLASH2, ctx)).toBe(2 * 1_924_404_480);
    expect(draftResidentWeightBytes(REF_DENSE_8B, DSPARK, ctx)).toBe(2 * 1_126_417_664);
  });
  it('MTP: the module layer + eh_proj (2·4096·4096·2 B) + three norms (3·4096·2 B)', () => {
    const ctx = verifyCtx(8192, 1);
    const layer = MTP.kind === 'mtp' ? MTP.layers[0] : undefined;
    const layerBytes = layer!.blocks.reduce((s, b) => s + blockResidentWeightBytes(b, ctx), 0);
    expect(layerBytes).toBe(2 * 192_946_432);
    expect(draftResidentWeightBytes(REF_DENSE_8B, MTP, ctx)).toBe(
      layerBytes + 67_108_864 + 3 * 4096 * 2,
    );
  });
  it('draft KV per sequence: DFlash 2 5·min(ctx, 2048)·2·8·128·2 B; MTP its layer over the full context (8194·4096 B)', () => {
    const ctx = verifyCtx(8192, 1);
    expect(draftSequenceStateBytes(DFLASH2, ctx, 8194)).toBe(5 * 2048 * 2 * 8 * 128 * 2);
    expect(draftSequenceStateBytes(DFLASH2, ctx, 1000)).toBe(5 * 1000 * 2 * 8 * 128 * 2);
    expect(draftSequenceStateBytes(MTP, ctx, 8194)).toBe(8194 * 4096);
    expect(draftSequenceStateBytes(DSPARK, ctx, 8194)).toBe(5 * 8194 * 2 * 8 * 128 * 2);
  });
});

describe('spec-enabled prefill (U-15, design doc §3.4)', () => {
  it('MTP: the module forward over the prompt (eh_proj m 8192, attn.core writing 8192 KV tokens, lm_head m 1)', () => {
    const ctx = prefillCtx(8192);
    const layers = lowerDraftPrefill(REF_DENSE_8B, MTP, ctx);
    const list = ops(layers);
    expect(shape(find<GemmOp>(list, 'draft.eh_proj'))).toEqual([8192, 4096, 8192]);
    const core = find<AttentionOp>(list, 'draft.attn.core');
    expect([core.qLen, core.kvLen]).toEqual([8192, 8192]);
    expect(opBytes(core).kvWriteBytes).toBe(8192 * 4096);
    expect(shape(find<GemmOp>(list, 'draft.lm_head'))).toEqual([1, 151936, 4096]);
    // the layer part equals an independently lowered module at qLen = kvLen = 8192
    const expected =
      MTP.kind === 'mtp' ? MTP.layers[0].blocks.flatMap((b) => lowerBlock(b, ctx)) : [];
    const layerOps = list.filter(
      (op) =>
        ![
          'draft.embed',
          'draft.enorm',
          'draft.hnorm',
          'draft.eh_proj',
          'draft.final_norm',
          'draft.lm_head',
        ].includes(op.name),
    );
    expect(layerOps).toEqual(expected.map((op) => ({ ...op, name: `draft.${op.name}` })));
  });
  it('block drafter: exactly { fc, hidden_norm } ∪ 5 × { k_proj, v_proj, qk_norm, rope, kv_cache_write }', () => {
    const layers = lowerDraftPrefill(REF_DENSE_8B, DFLASH2, prefillCtx(8192));
    const counts: Record<string, number> = {};
    for (const n of names(layers)) counts[n] = (counts[n] ?? 0) + 1;
    expect(counts).toEqual({
      'draft.fc': 1,
      'draft.hidden_norm': 1,
      'draft.k_proj': 5,
      'draft.v_proj': 5,
      'draft.qk_norm': 5,
      'draft.rope': 5,
      'draft.kv_cache_write': 5,
    });
    const list = ops(layers);
    expect(shape(find<GemmOp>(list, 'draft.k_proj'))).toEqual([8192, 1024, 5120]);
    expect(shape(find<GemmOp>(list, 'draft.fc'))).toEqual([8192, 5120, 25600]);
    expect(find<ElementwiseOp>(list, 'draft.qk_norm').elems).toBe(8192 * 8 * 128);
    const write = find<ElementwiseOp>(list, 'draft.kv_cache_write');
    expect(write).toMatchObject({
      fn: 'kv_cache_write',
      elems: 4_194_304,
      activationReadBytes: 8_388_608,
      activationWriteBytes: 0,
      kvWriteBytes: 8_388_608,
    });
    expect(opBytes(write).kvWriteBytes).toBe(8_388_608);
    expect(opFlops(write)).toBe(4_194_304);
  });
  it('DSpark (full attention) stores every prompt token: 8192·2·8·128 elements', () =>
    expect(
      find<ElementwiseOp>(
        ops(lowerDraftPrefill(REF_DENSE_8B, DSPARK, prefillCtx(8192))),
        'draft.kv_cache_write',
      ).elems,
    ).toBe(8192 * 2 * 8 * 128));
});

describe('hybrid verify snapshots (U-14, lowering half)', () => {
  const N = 1024;
  const k = 3;
  const plain: LoweringContext = {
    phase: 'decode',
    seqs: 2,
    qLen: 1,
    kvLen: N + 1,
    quant: BF16,
    parallel: P1,
  };
  const verify: LoweringContext = {
    ...plain,
    qLen: k + 1,
    kvLen: N + k + 1,
    speculative: {
      draft: MTP,
      draftLength: k,
      verifyTokens: k + 1,
      contextRows: 2,
      stateSnapshots: k + 1,
    },
  };
  const opsOf = (ctx: LoweringContext) =>
    lowerModel(TINY_HYBRID, ctx).stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
  it('TINY_HYBRID k 3: every GDN RecurrentOp has stateSnapshots 4 and 4× the state write bytes; reads stay 1×', () => {
    const base = opsOf(plain).filter((op): op is RecurrentOp => op.kind === 'recurrent');
    const ver = opsOf(verify).filter((op): op is RecurrentOp => op.kind === 'recurrent');
    expect(ver.length).toBeGreaterThan(0);
    expect(ver.map((op) => op.stateSnapshots)).toEqual(ver.map(() => 4));
    ver.forEach((op, i) => {
      expect(opBytes(op).stateWriteBytes).toBe(4 * opBytes(base[i]).stateWriteBytes);
      expect(opBytes(op).stateReadBytes).toBe(opBytes(base[i]).stateReadBytes);
    });
  });
  it('conv rolling states (causal_conv1d) write 4× too', () => {
    const conv = (ctx: LoweringContext) =>
      opsOf(ctx).filter(
        (op): op is ElementwiseOp => op.kind === 'elementwise' && op.fn === 'causal_conv1d',
      );
    const base = conv(plain);
    const ver = conv(verify);
    expect(ver.length).toBeGreaterThan(0);
    ver.forEach((op, i) => {
      expect(op.stateWriteBytes).toBe(4 * base[i].stateWriteBytes);
      expect(op.stateReadBytes).toBe(base[i].stateReadBytes);
    });
  });
  it('without speculative decoding the ops carry no stateSnapshots field (identity)', () => {
    expect(opsOf(plain).some((op) => op.kind === 'recurrent' && 'stateSnapshots' in op)).toBe(
      false,
    );
    expect(
      opsOf({
        ...plain,
        phase: 'prefill',
        qLen: 64,
        kvLen: 64,
        speculative: verify.speculative,
      }).some((op) => op.kind === 'recurrent' && 'stateSnapshots' in op),
    ).toBe(false);
  });
  it('the MTP draft of a hybrid keeps one snapshot (only its final state is kept)', () => {
    const hybridMtp = { kind: 'mtp' as const, layers: [TINY_HYBRID.layers[0]], hiddenProj: true };
    const list = ops(lowerDraftDecode(TINY_HYBRID, hybridMtp, verify, 2, k));
    expect(list.some((op) => op.kind === 'recurrent' && 'stateSnapshots' in op)).toBe(false);
  });
});

describe('G1 sampling exchange on draft forwards, not just the target head', () => {
  it('MTP tp2 b2 k3: three 2·2·8=32 B all-gathers; prefill has one 32 B gather', () => {
    const ctx = { ...verifyCtx(1024, 3, 2), parallel: { ...P1, tp: 2 } };
    const decode = ops(lowerDraftDecode(REF_DENSE_8B, MTP, ctx, 2, 3));
    const prefill = ops(
      lowerDraftPrefill(REF_DENSE_8B, MTP, { ...ctx, phase: 'prefill', qLen: 128, kvLen: 128 }),
    );
    const exchanges = (list: Op[]) => list.filter((op) => op.name === 'draft.sample_all_gather');
    expect(exchanges(decode)).toHaveLength(3);
    expect(exchanges(prefill)).toHaveLength(1);
    for (const op of [...exchanges(decode), ...exchanges(prefill)])
      expect(op).toMatchObject({
        kind: 'collective',
        collective: 'all_gather',
        group: 'tp',
        groupSize: 2,
        messageBytes: 32,
      });
  });
  it.each([
    ['DFlash 2', DFLASH2],
    ['DSpark', DSPARK],
  ] as const)(
    '%s tp2 b2 k7: block head gathers 2·7·2·8=224 B once, no prefill head',
    (_name, draft) => {
      const ctx = { ...verifyCtx(1024, 7, 2), parallel: { ...P1, tp: 2 } };
      const list = ops(lowerDraftDecode(REF_DENSE_8B, draft, ctx, 2, 7));
      const head = list.findIndex((op) => op.name === 'draft.lm_head');
      expect(list[head + 1]).toMatchObject({ name: 'draft.sample_all_gather', messageBytes: 224 });
      const prefill = ops(
        lowerDraftPrefill(REF_DENSE_8B, draft, { ...ctx, phase: 'prefill', qLen: 128, kvLen: 128 }),
      );
      expect(prefill.some((op) => op.name === 'draft.sample_all_gather')).toBe(false);
    },
  );
});
