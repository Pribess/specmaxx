import { describe, expect, it } from 'vitest';
import type { MtpDraftSpec } from '../ir/model';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import type { AcceptanceModel } from './types';
import {
  ACCEPTANCE_PRESETS,
  breakEvenBracket,
  DEFAULT_DRAFT_LENGTH,
  SPEC_USE_CASES,
  defaultDraftLength,
  draftContextRows,
  expectedTokens,
  mtpTailAcceptance,
  perPositionAlphas,
  presetAcceptance,
  presetMethod,
  fittedDraftLengths,
  iidAlphaForTokens,
  presetDraftLengthExtrapolation,
  pooledDraftLengths,
  acceptanceRuns,
  FIT_TOLERANCE_ALPHA,
} from './specdec';

/** |actual − expected| ≤ rel·|expected|. */
function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const MTP_DRAFT = REF_DENSE_8B_MTP.draft as MtpDraftSpec;
const MULTI_STEP: MtpDraftSpec = { ...MTP_DRAFT, maxReliableDepth: undefined };

describe('breakEvenBracket: first adjacent pair with speedup ≥ 1 then < 1', () => {
  const pts = (pairs: [number, number][]) => pairs.map(([batch, speedup]) => ({ batch, speedup }));
  it('a crossing between 32 and 64 → [32, 64]', () => {
    expect(
      breakEvenBracket(
        pts([
          [1, 2.4],
          [16, 1.9],
          [32, 1.3],
          [64, 0.97],
          [128, 0.79],
        ]),
      ),
    ).toEqual([32, 64]);
  });
  it('exactly 1 still pays: [1, 1.0], [2, 0.9] → [1, 2]', () => {
    expect(
      breakEvenBracket(
        pts([
          [1, 1],
          [2, 0.9],
        ]),
      ),
    ).toEqual([1, 2]);
  });
  it('pays everywhere → undefined', () => {
    expect(
      breakEvenBracket(
        pts([
          [1, 2],
          [2, 1.5],
          [4, 1.01],
        ]),
      ),
    ).toBeUndefined();
  });
  it('never pays → undefined (nothing to bracket)', () => {
    expect(
      breakEvenBracket(
        pts([
          [1, 0.9],
          [2, 0.8],
        ]),
      ),
    ).toBeUndefined();
    expect(breakEvenBracket([])).toBeUndefined();
  });
  it('non-monotone: the first crossing wins, and input order does not matter', () => {
    const wiggle = pts([
      [8, 0.98],
      [1, 1.5],
      [2, 1.2],
      [4, 0.99],
      [16, 1.02],
      [32, 0.9],
    ]);
    expect(breakEvenBracket(wiggle)).toEqual([2, 4]);
  });
});

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
  it('extras: MTP reasoning 0.68, DFlash 2 agentic 0.65 (borrowed v1); missing pairs carry the prose value over (+0.02 for block reasoning)', () => {
    expect(ACCEPTANCE_PRESETS.mtp_multi_step.reasoning).toMatchObject({
      alpha: 0.68,
      basis: 'measured',
    });
    expect(ACCEPTANCE_PRESETS.dflash2.agentic).toMatchObject({ alpha: 0.65, basis: 'borrowed' });
    expect(ACCEPTANCE_PRESETS.dflash2.agentic.measurements[0]).toMatchObject({
      methodVersion: 'DFlash v1',
      draftLength: 15,
      meanAcceptedTokens: 2.82,
    });
    const carried = [
      ['mtp_multi_step', 'agentic', 0],
      ['mtp_single_step', 'code', 0],
      ['mtp_single_step', 'math', 0.02],
      ['mtp_single_step', 'reasoning', 0],
      ['mtp_single_step', 'agentic', 0],
      ['dflash2', 'reasoning', 0.02],
      ['dspark', 'reasoning', 0.02],
      ['dspark', 'agentic', 0],
    ] as const;
    for (const [method, useCase, offset] of carried) {
      expect(ACCEPTANCE_PRESETS[method][useCase]).toMatchObject({
        basis: 'carried_over',
        carriedFrom: { useCase: 'prose', alphaOffset: offset },
      });
    }
  });
  it('every preset names its sources, method version, target, benchmark, k and τ', () => {
    for (const row of Object.values(ACCEPTANCE_PRESETS)) {
      for (const useCase of SPEC_USE_CASES) {
        const preset = row[useCase];
        expect(preset.measurements.length).toBeGreaterThan(0);
        // every preset has a fitted k, except pooled ones (fitted at none, by definition)
        expect(fittedDraftLengths(preset).length > 0).toBe(preset.basis !== 'pooled');
        for (const m of [...preset.measurements, ...(preset.otherReports ?? [])]) {
          expect(m.url).toMatch(/^https:\/\//);
          expect(m.location).not.toBe('');
          expect(m.methodVersion).not.toBe('');
          expect(m.target).not.toBe('');
          expect(m.benchmark).not.toBe('');
          expect(Number.isInteger(m.draftLength) && m.draftLength >= 1).toBe(true);
          expect(m.meanAcceptedTokens).toBeGreaterThanOrEqual(1);
          expect(m.meanAcceptedTokens).toBeLessThan(m.draftLength + 1);
        }
      }
    }
  });
  it('measured and borrowed α reproduce every fitted run (± FIT_TOLERANCE_ALPHA); pooled α lies between its runs and reproduces none', () => {
    for (const row of Object.values(ACCEPTANCE_PRESETS)) {
      for (const useCase of SPEC_USE_CASES) {
        const preset = row[useCase];
        if (preset.basis === 'carried_over') continue;
        const runs = acceptanceRuns(preset.measurements).map((r) => r.alphaIid);
        if (preset.basis === 'pooled') {
          expect(runs.length).toBeGreaterThan(1);
          expect(preset.alpha).toBeGreaterThan(Math.min(...runs));
          expect(preset.alpha).toBeLessThan(Math.max(...runs));
          for (const a of runs)
            expect(Math.abs(preset.alpha - a)).toBeGreaterThan(FIT_TOLERANCE_ALPHA);
        } else {
          for (const a of runs)
            expect(Math.abs(preset.alpha - a)).toBeLessThanOrEqual(FIT_TOLERANCE_ALPHA);
        }
      }
    }
  });
  it('no cited report would reproduce α (else it belongs in the fitted runs)', () => {
    for (const row of Object.values(ACCEPTANCE_PRESETS)) {
      for (const useCase of SPEC_USE_CASES) {
        const preset = row[useCase];
        if (preset.basis === 'carried_over') continue;
        for (const run of acceptanceRuns(preset.otherReports ?? [])) {
          expect(Math.abs(preset.alpha - run.alphaIid)).toBeGreaterThan(FIT_TOLERANCE_ALPHA);
        }
      }
    }
  });
  it('acceptanceRuns averages a source table’s benchmarks and targets: DSpark paper MT-Bench 3.64/3.72/3.70 → 3.687, α .7585', () => {
    const [run] = acceptanceRuns(ACCEPTANCE_PRESETS.dspark.prose.measurements);
    expect(run.draftLength).toBe(7);
    expectClose(run.meanAcceptedTokens, (3.64 + 3.72 + 3.7) / 3, 1e-12);
    expect(run.alphaIid).toBeCloseTo(0.7585, 4);
    // DFlash 2 code: Qwen3.8 card (k 7) and Muse card (k 15) are two runs
    expect(
      acceptanceRuns(ACCEPTANCE_PRESETS.dflash2.code.measurements).map((r) => r.draftLength),
    ).toEqual([7, 15]);
  });
  it('carried-over α is the source use case α plus its offset, with that use case’s measurements', () => {
    for (const row of Object.values(ACCEPTANCE_PRESETS)) {
      for (const useCase of SPEC_USE_CASES) {
        const preset = row[useCase];
        if (preset.basis !== 'carried_over') {
          expect(preset.carriedFrom).toBeUndefined();
          continue;
        }
        const from = row[preset.carriedFrom!.useCase];
        expectClose(preset.alpha, from.alpha + preset.carriedFrom!.alphaOffset, 1e-12);
        expect(preset.measurements).toBe(from.measurements);
      }
    }
  });
  it('single-source fits reproduce their τ: DFlash v1 SWE-bench Lite 2.82 @ k 15 → α 0.65 gives 2.854', () => {
    expect(iidAlphaForTokens(2.82, 15)).toBeCloseTo(0.6457, 4);
    expect(expectedTokens({ kind: 'iid', alpha: 0.65 }, 15)).toBeCloseTo(2.854, 3);
    // DSpark paper MT-Bench τ 3.70 @ k 7 (Qwen3-14B) → α_iid .7597; the 0.76 preset gives 3.703
    expect(iidAlphaForTokens(3.7, 7)).toBeCloseTo(0.7597, 4);
    expect(expectedTokens({ kind: 'iid', alpha: 0.76 }, 7)).toBeCloseTo(3.703, 3);
    // llama.cpp PR 27210 reasoning τ 2.47 @ k 3 → α_iid .6835 (preset 0.68)
    expect(iidAlphaForTokens(2.47, 3)).toBeCloseTo(0.6835, 4);
  });
  it('iidAlphaForTokens inverts expectedTokens and rejects τ outside [1, k + 1)', () => {
    expectClose(iidAlphaForTokens(expectedTokens({ kind: 'iid', alpha: 0.8 }, 7), 7), 0.8, 1e-9);
    expect(iidAlphaForTokens(1, 3)).toBeCloseTo(0, 9);
    expect(() => iidAlphaForTokens(4, 3)).toThrow(/\[1, k \+ 1\)/);
    expect(() => iidAlphaForTokens(0.5, 3)).toThrow();
    expect(() => iidAlphaForTokens(2, 0)).toThrow();
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

describe('k extrapolation of a preset α (practicality review F10)', () => {
  it('fitted k per preset: only the runs α reproduces count; pooled presets have none', () => {
    const fitted = Object.fromEntries(
      Object.entries(ACCEPTANCE_PRESETS).map(([method, row]) => [
        method,
        Object.fromEntries(SPEC_USE_CASES.map((u) => [u, fittedDraftLengths(row[u])])),
      ]),
    );
    expect(fitted).toEqual({
      mtp_multi_step: { prose: [15], code: [], math: [7], reasoning: [3], agentic: [15] },
      mtp_single_step: { prose: [1], code: [1], math: [1], reasoning: [1], agentic: [1] },
      dflash2: { prose: [7], code: [7, 15], math: [], reasoning: [7], agentic: [15] },
      dspark: { prose: [7], code: [], math: [], reasoning: [7], agentic: [7] },
    });
    expect(pooledDraftLengths(ACCEPTANCE_PRESETS.dspark.code)).toEqual([7, 15]);
    expect(pooledDraftLengths(ACCEPTANCE_PRESETS.mtp_multi_step.code)).toEqual([3, 7, 15]);
    expect(pooledDraftLengths(ACCEPTANCE_PRESETS.dspark.prose)).toEqual([]);
  });
  it('warns with the fitted k when k is none of them, not when it is one; a cited report at k does not count', () => {
    expect(presetDraftLengthExtrapolation('dflash2', 'agentic', 7)).toEqual([15]);
    expect(presetDraftLengthExtrapolation('dflash2', 'agentic', 15)).toBeUndefined();
    expect(presetDraftLengthExtrapolation('dflash2', 'prose', 7)).toBeUndefined();
    // Muse and the blog report DFlash 2 prose at k 15, but 0.79 is fitted to the k 7 card only
    expect(presetDraftLengthExtrapolation('dflash2', 'prose', 15)).toEqual([7]);
    // Inco's k 15 DSpark retrain (.796) is cited, not fitted: E[τ](0.76, 15) = 4.12 vs 4.77
    expect(presetDraftLengthExtrapolation('dspark', 'prose', 15)).toEqual([7]);
    expect(presetDraftLengthExtrapolation('dspark', 'prose', 5)).toEqual([7]);
    expect(presetDraftLengthExtrapolation('mtp', 'math', 3, MULTI_STEP)).toEqual([7]);
    expect(presetDraftLengthExtrapolation('mtp', 'prose', 3, MULTI_STEP)).toEqual([15]);
    expect(presetDraftLengthExtrapolation('mtp', 'prose', 15, MULTI_STEP)).toBeUndefined();
  });
  it('pooled presets warn when k is none of their pooled runs’ k', () => {
    expect(presetDraftLengthExtrapolation('dspark', 'code', 5)).toEqual([7, 15]);
    expect(presetDraftLengthExtrapolation('dspark', 'code', 7)).toBeUndefined();
    expect(presetDraftLengthExtrapolation('dflash2', 'math', 3)).toEqual([7, 15]);
    expect(presetDraftLengthExtrapolation('mtp', 'code', 3, MULTI_STEP)).toBeUndefined();
    expect(presetDraftLengthExtrapolation('mtp', 'code', 5, MULTI_STEP)).toEqual([3, 7, 15]);
  });
  it('no warning for a user α, nor for a single-step head past its depth (the tail rule applies)', () => {
    expect(presetDraftLengthExtrapolation('dflash2', 'agentic', 7, undefined, 0.7)).toBeUndefined();
    expect(presetDraftLengthExtrapolation('mtp', 'prose', 1, MTP_DRAFT)).toBeUndefined();
    expect(presetDraftLengthExtrapolation('mtp', 'prose', 3, MTP_DRAFT)).toBeUndefined();
  });
  it('F10 probe: an α fitted at k 7 to a decay curve 0.92 − 0.06/position overstates k 15 and understates k 3', () => {
    // α_j = 0.92 − 0.06·(j − 1), j = 1..15 (illustrative, not measured)
    const decay: AcceptanceModel = {
      kind: 'per_position',
      alphas: Array.from({ length: 15 }, (_, j) => 0.92 - 0.06 * j),
    };
    const alpha = iidAlphaForTokens(expectedTokens(decay, 7), 7);
    expect(expectedTokens({ kind: 'iid', alpha }, 15)).toBeCloseTo(5.36, 2);
    expect(expectedTokens(decay, 15)).toBeCloseTo(4.53, 2);
    expect(expectedTokens({ kind: 'iid', alpha }, 3)).toBeCloseTo(3.05, 2);
    expect(expectedTokens(decay, 3)).toBeCloseTo(3.34, 2);
  });
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
