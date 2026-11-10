import { describe, expect, it } from 'vitest';
import { REF_DENSE_8B } from '../models/reference';
import type { LoweringContext } from './lower';
import { lmHeadResidentWeightBytes, lmHeadVocabShard, lowerHead } from './head';

const ctx: LoweringContext = {
  phase: 'decode',
  seqs: 2,
  qLen: 3,
  kvLen: 1024,
  quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
  parallel: { tp: 4, pp: 1, dp: 1, ep: 1 },
};

describe('G1 sampled-token exchange after lm_head', () => {
  it.each(['prefill', 'decode'] as const)(
    '%s: output is Tout·4·8 bytes (64 prefill, 192 decode)',
    (phase) => {
      // Each rank contributes one fp32 score + one uint32 token ID per output position.
      const ops = lowerHead(REF_DENSE_8B, { ...ctx, phase });
      expect(ops.at(-2)?.name).toBe('lm_head');
      expect(ops.at(-1)).toMatchObject({
        kind: 'collective',
        collective: 'all_gather',
        group: 'tp',
        groupSize: 4,
        messageBytes: phase === 'prefill' ? 64 : 192,
      });
    },
  );
  it('tp1 has no exchange, even without a final norm', () => {
    const ops = lowerHead(
      { ...REF_DENSE_8B, finalNorm: 'none' },
      { ...ctx, parallel: { ...ctx.parallel, tp: 1 } },
    );
    expect(ops.map((op) => op.name)).toEqual(['lm_head']);
  });
  it('tp4 still exchanges when finalNorm is none', () => {
    expect(lowerHead({ ...REF_DENSE_8B, finalNorm: 'none' }, ctx).at(-1)?.kind).toBe('collective');
  });
});

describe("LM head over the EP group (denseTpGroup 'ep', tp 1, G = ep)", () => {
  const V = REF_DENSE_8B.vocab; // 151936
  const h = REF_DENSE_8B.hidden; // 4096
  const G = 6; // ceil(151936/6) = 25323 (not exact: the last shard is short, as with tp)
  const epCtx: LoweringContext = {
    ...ctx,
    parallel: { tp: 1, pp: 1, dp: G, ep: G, denseTpGroup: 'ep' },
  };

  it('lmHeadVocabShard = ceil(V/G); ceil(V/tp) without the field', () => {
    expect(lmHeadVocabShard(REF_DENSE_8B, epCtx)).toBe(Math.ceil(V / G));
    expect(lmHeadVocabShard(REF_DENSE_8B, ctx)).toBe(V / 4);
  });
  it.each(['prefill', 'decode'] as const)(
    '%s: final_norm(Tout) → all_gather S = G·Tout·h·2 → lm_head m = G·Tout → all_to_all S = G·Tout·8',
    (phase) => {
      const Tout = phase === 'prefill' ? 2 : 6; // seqs, or seqs·qLen
      const ops = lowerHead(REF_DENSE_8B, { ...epCtx, phase });
      expect(ops.map((o) => o.name)).toEqual([
        'final_norm',
        'head.dense_all_gather',
        'lm_head',
        'head.sample_all_to_all',
      ]);
      expect(ops[0]).toMatchObject({ elems: Tout * h });
      expect(ops[1]).toMatchObject({
        kind: 'collective',
        collective: 'all_gather',
        group: 'ep',
        groupSize: G,
        messageBytes: G * Tout * h * 2,
        dtype: 'bf16',
      });
      expect(ops[2]).toMatchObject({ m: G * Tout, n: Math.ceil(V / G), k: h });
      expect(ops[3]).toMatchObject({
        collective: 'all_to_all',
        group: 'ep',
        groupSize: G,
        messageBytes: G * Tout * 8,
        dtype: 'fp32',
      });
    },
  );
  it('resident head = ceil(V/G)·h·bpe + final-norm γ', () => {
    expect(lmHeadResidentWeightBytes(REF_DENSE_8B, epCtx)).toBe(Math.ceil(V / G) * h * 2 + h * 2);
  });
  it("'ep' at ep 1 is the tp1 head (no collective)", () => {
    const one: LoweringContext = {
      ...ctx,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1, denseTpGroup: 'ep' },
    };
    const tp1: LoweringContext = { ...ctx, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 } };
    expect(lowerHead(REF_DENSE_8B, one)).toEqual(lowerHead(REF_DENSE_8B, tp1));
  });
});
