import { describe, expect, it } from 'vitest';
import type { MtpDraftSpec } from '../ir/model';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import {
  ACCEPTANCE_PRESETS,
  DEFAULT_DRAFT_LENGTH,
  SPEC_USE_CASES,
  defaultDraftLength,
  draftContextRows,
  expectedTokens,
  mtpTailAcceptance,
  perPositionAlphas,
  presetAcceptance,
  presetMethod,
} from './specdec';

/** |actual − expected| ≤ rel·|expected|. */
function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const MTP_DRAFT = REF_DENSE_8B_MTP.draft as MtpDraftSpec;
const MULTI_STEP: MtpDraftSpec = { ...MTP_DRAFT, maxReliableDepth: undefined };

describe('expectedTokens (U-7, design doc §3.5)', () => {
  it('iid α .85, k 1 → 1 + .85 = 1.85', () =>
    expectClose(expectedTokens({ kind: 'iid', alpha: 0.85 }, 1), 1.85));
  it('iid α .79, k 7 → (1 − .79^8)/.21 = 4.03948…', () =>
    expectClose(expectedTokens({ kind: 'iid', alpha: 0.79 }, 7), (1 - 0.79 ** 8) / 0.21));
  it('iid α .79, k 7 is about 4.0395', () =>
    expect(expectedTokens({ kind: 'iid', alpha: 0.79 }, 7)).toBeCloseTo(4.0395, 4));
  it('per-position DFlash 2 blog vector [.854 … .729] → 4.2746 (blog: 4.27)', () => {
    const alphas = [0.854, 0.803, 0.794, 0.783, 0.775, 0.759, 0.729];
    // 1 + .854 + .685762 + .544495 + .426340 + .330413 + .250784 + .182822 = 4.274616
    expect(expectedTokens({ kind: 'per_position', alphas }, 7)).toBeCloseTo(4.274616, 5);
  });
  it('per-position α_j = 1 is legal: alphas [1] at k 1 → 2 = k + 1', () =>
    expect(expectedTokens({ kind: 'per_position', alphas: [1] }, 1)).toBe(2));
  it('alphas [0] → 1 (only the bonus token)', () =>
    expect(expectedTokens({ kind: 'per_position', alphas: [0] }, 3)).toBe(1));
  it('a constant per-position vector equals the iid closed form', () =>
    expectClose(
      expectedTokens({ kind: 'per_position', alphas: [0.8] }, 5),
      expectedTokens({ kind: 'iid', alpha: 0.8 }, 5),
      1e-12,
    ));
  it('a short vector is extended with its last value', () => {
    expect(perPositionAlphas([0.9, 0.5], 4)).toEqual([0.9, 0.5, 0.5, 0.5]);
    // 1 + .9 + .45 + .225 + .1125
    expectClose(expectedTokens({ kind: 'per_position', alphas: [0.9, 0.5] }, 4), 2.6875);
  });
  it('k 0 → 1', () => expect(expectedTokens({ kind: 'iid', alpha: 0.5 }, 0)).toBe(1));
  it('rejects iid α = 1, α < 0, an empty vector, α_j > 1 and a fractional k', () => {
    expect(() => expectedTokens({ kind: 'iid', alpha: 1 }, 1)).toThrow(/\[0, 1\)/);
    expect(() => expectedTokens({ kind: 'iid', alpha: -0.1 }, 1)).toThrow();
    expect(() => expectedTokens({ kind: 'per_position', alphas: [] }, 1)).toThrow(/at least one/);
    expect(() => expectedTokens({ kind: 'per_position', alphas: [1.2] }, 1)).toThrow(/\[0, 1\]/);
    expect(() => expectedTokens({ kind: 'iid', alpha: 0.5 }, 1.5)).toThrow();
  });
});

describe('acceptance presets (design doc §5.1)', () => {
  it('bold table values', () => {
    expect(ACCEPTANCE_PRESETS.mtp_multi_step.prose.alpha).toBe(0.75);
    expect(ACCEPTANCE_PRESETS.mtp_multi_step.code.alpha).toBe(0.82);
    expect(ACCEPTANCE_PRESETS.mtp_multi_step.math.alpha).toBe(0.85);
    expect(ACCEPTANCE_PRESETS.mtp_single_step.prose.alpha).toBe(0.85);
    expect(ACCEPTANCE_PRESETS.dflash2.prose.alpha).toBe(0.79);
    expect(ACCEPTANCE_PRESETS.dflash2.code.alpha).toBe(0.83);
    expect(ACCEPTANCE_PRESETS.dflash2.math.alpha).toBe(0.87);
    expect(ACCEPTANCE_PRESETS.dspark.prose.alpha).toBe(0.76);
    expect(ACCEPTANCE_PRESETS.dspark.code.alpha).toBe(0.84);
    expect(ACCEPTANCE_PRESETS.dspark.math.alpha).toBe(0.88);
  });
  it('extras: MTP reasoning 0.70, DFlash 2 agentic 0.65; missing pairs are the prose value (+0.02 for block reasoning), marked interpolated', () => {
    expect(ACCEPTANCE_PRESETS.mtp_multi_step.reasoning).toEqual({
      alpha: 0.7,
      interpolated: false,
    });
    expect(ACCEPTANCE_PRESETS.dflash2.agentic).toEqual({ alpha: 0.65, interpolated: false });
    expect(ACCEPTANCE_PRESETS.mtp_multi_step.agentic).toEqual({ alpha: 0.75, interpolated: true });
    expect(ACCEPTANCE_PRESETS.mtp_single_step.reasoning).toEqual({
      alpha: 0.85,
      interpolated: true,
    });
    expect(ACCEPTANCE_PRESETS.mtp_single_step.agentic).toEqual({ alpha: 0.85, interpolated: true });
    expect(ACCEPTANCE_PRESETS.dflash2.reasoning).toEqual({ alpha: 0.81, interpolated: true });
    expect(ACCEPTANCE_PRESETS.dspark.reasoning).toEqual({ alpha: 0.78, interpolated: true });
    expect(ACCEPTANCE_PRESETS.dspark.agentic).toEqual({ alpha: 0.76, interpolated: true });
  });
  it('every method × use case is a legal iid α', () => {
    for (const row of Object.values(ACCEPTANCE_PRESETS)) {
      for (const useCase of SPEC_USE_CASES) {
        expect(row[useCase].alpha).toBeGreaterThanOrEqual(0);
        expect(row[useCase].alpha).toBeLessThan(1);
      }
    }
  });
  it('default k: MTP multi-step 3, single-step 1, DFlash 2 7, DSpark 7', () =>
    expect(DEFAULT_DRAFT_LENGTH).toEqual({
      mtp_multi_step: 3,
      mtp_single_step: 1,
      dflash2: 7,
      dspark: 7,
    }));
  it('presetMethod splits MTP by maxReliableDepth', () => {
    expect(presetMethod('mtp', MTP_DRAFT)).toBe('mtp_single_step');
    expect(presetMethod('mtp', MULTI_STEP)).toBe('mtp_multi_step');
    expect(presetMethod('mtp')).toBe('mtp_multi_step');
    expect(presetMethod('dspark')).toBe('dspark');
  });
  it('defaultDraftLength caps block drafters at blockSize − 1', () => {
    const block = { kind: 'block_drafter', method: 'dspark', blockSize: 6 } as const;
    expect(defaultDraftLength('dspark', block as never)).toBe(5);
    expect(
      defaultDraftLength('dflash2', { ...block, method: 'dflash2', blockSize: 8 } as never),
    ).toBe(7);
    expect(defaultDraftLength('mtp', MTP_DRAFT)).toBe(1);
    expect(defaultDraftLength('mtp', MULTI_STEP)).toBe(3);
  });
});

describe('single-step MTP tail (design doc §3.5)', () => {
  it('[α, .30, .10, .05, …] past a depth-1 head', () =>
    expect(mtpTailAcceptance(0.85, 1)).toEqual({
      kind: 'per_position',
      alphas: [0.85, 0.3, 0.1, 0.05],
    }));
  it('presetAcceptance keeps iid within the trained depth and switches to the tail past it', () => {
    expect(presetAcceptance('mtp', 'prose', 1, MTP_DRAFT)).toEqual({ kind: 'iid', alpha: 0.85 });
    const tail = presetAcceptance('mtp', 'prose', 3, MTP_DRAFT);
    expect(tail).toEqual({ kind: 'per_position', alphas: [0.85, 0.3, 0.1, 0.05] });
    // 1 + .85 + .85·.3 + .85·.3·.1 = 2.1305
    expectClose(expectedTokens(tail, 3), 2.1305);
    // the tail extends with .05: k 5 adds .85·.3·.1·.05 and its ·.05
    expectClose(expectedTokens(tail, 5), 2.1305 + 0.001275 + 0.00006375);
  });
  it('multi-step heads keep iid at any k', () =>
    expect(presetAcceptance('mtp', 'code', 3, MULTI_STEP)).toEqual({ kind: 'iid', alpha: 0.82 }));
});

describe('draftContextRows: m_ctx = ⌈E[τ]⌉ (design doc §3.3)', () => {
  it('1.85 → 2, 4.04 → 5, exactly 2 → 2, 1 → 1', () => {
    expect(draftContextRows(1.85)).toBe(2);
    expect(draftContextRows(4.0394)).toBe(5);
    expect(draftContextRows(2)).toBe(2);
    expect(draftContextRows(1)).toBe(1);
  });
  it('absorbs float error just above an integer', () =>
    expect(draftContextRows(2.0000000000000004)).toBe(2));
});
