import { describe, expect, it } from 'vitest';
import { REF_DENSE_8B } from '../models/reference';
import type { LoweringContext } from './lower';
import { lowerHead } from './head';

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
