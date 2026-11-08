/**
 * Speculative decoding: accepted tokens per step and the researched acceptance defaults (design doc §3.5, §5.1).
 *
 * Pure math and data, no lowering. A step drafts `k` tokens and verifies `k + 1` positions in one target pass; it
 * commits τ tokens, the accepted prefix of the drafts plus the target's bonus token (1 ≤ τ ≤ k + 1). The engine
 * multiplies the step's sequence rate by E[τ] (see docs/simulation-model.md §13 "Accepted tokens").
 *
 * Acceptance α is the probability that one draft token is accepted (chain verification, T ≈ 1 sampling). The presets
 * are α_iid values DERIVED from reported mean acceptance lengths τ at the reported k (design doc §5.1): the constant
 * per-position α that reproduces τ under {@link expectedTokens}. They are user-facing defaults, quoted with the
 * context they were measured at by their sources; α depends on the workload, so the UI lets the user override it.
 */

import type { DraftSpec, ModelSpec } from '../ir/model';
import type { SpeculativeLowering } from '../lower/lower';
import type { AcceptanceModel, SpecDecodeConfig } from './types';

/**
 * Expected tokens committed per sequence per step, E[τ], including the bonus token. Count (real, in [1, k + 1]).
 *
 * - `iid` (Leviathan et al. 2023, arXiv 2211.17192 Eq. 1): `E[τ] = (1 − α^(k+1)) / (1 − α)`, α in [0, 1).
 * - `per_position` (DSpark arXiv 2607.05147 §3.2.2): `E[τ] = 1 + Σ_{i=1..k} Π_{j=1..i} α_j`, α_j in [0, 1] the
 *   acceptance of draft j given drafts 1..j−1 were accepted; a vector shorter than k is extended with its last value.
 *
 * `k` is the draft length (an integer ≥ 0; k = 0 gives 1). Throws on α outside its range or an empty vector.
 */
export function expectedTokens(acceptance: AcceptanceModel, k: number): number {
  if (!Number.isInteger(k) || k < 0)
    throw new Error(`draft length k must be a non-negative integer, got ${k}`);
  switch (acceptance.kind) {
    case 'iid': {
      const alpha = acceptance.alpha;
      if (!(alpha >= 0 && alpha < 1))
        throw new Error(`iid acceptance α must be in [0, 1), got ${alpha}`);
      // Formula: E[τ] = (1 − α^(k+1)) / (1 − α)
      // See docs/simulation-model.md §13 "Accepted tokens"
      return (1 - alpha ** (k + 1)) / (1 - alpha);
    }
    case 'per_position': {
      const alphas = perPositionAlphas(acceptance.alphas, k);
      // Formula: E[τ] = 1 + Σ_{i=1..k} Π_{j=1..i} α_j
      // See docs/simulation-model.md §13 "Accepted tokens"
      let expected = 1;
      let allAccepted = 1;
      for (let i = 0; i < k; i++) {
        allAccepted *= alphas[i];
        expected += allAccepted;
      }
      return expected;
    }
    default: {
      const _never: never = acceptance;
      throw new Error(`unknown acceptance ${JSON.stringify(_never)}`);
    }
  }
}

/**
 * The per-position acceptances α_1..α_k of a `per_position` vector: entries past its end repeat its last value.
 * Dimensionless, each in [0, 1]. Throws on an empty vector or an entry outside [0, 1].
 */
export function perPositionAlphas(alphas: readonly number[], k: number): number[] {
  if (alphas.length === 0) throw new Error('per-position acceptance needs at least one α');
  for (const a of alphas) {
    if (!(a >= 0 && a <= 1))
      throw new Error(`per-position acceptance α_j must be in [0, 1], got ${a}`);
  }
  const last = alphas[alphas.length - 1];
  return Array.from({ length: k }, (_, i) => (i < alphas.length ? alphas[i] : last));
}

/** Workload class an acceptance preset is quoted for (design doc §5.1 columns and extras). */
export type SpecUseCase = 'prose' | 'code' | 'math' | 'reasoning' | 'agentic';

/** Every {@link SpecUseCase}, in display order. */
export const SPEC_USE_CASES: readonly SpecUseCase[] = [
  'prose',
  'code',
  'math',
  'reasoning',
  'agentic',
];

/**
 * Row of the acceptance table (design doc §5.1). MTP splits into heads trained for several draft positions
 * (Qwen3.5+/3.8, MiMo; `MtpDraftSpec.maxReliableDepth` absent) and single-step heads (DeepSeek-V3 / GLM
 * `num_nextn_predict_layers`; `maxReliableDepth` set), whose acceptance collapses past their depth.
 */
export type AcceptancePresetMethod = 'mtp_multi_step' | 'mtp_single_step' | 'dflash2' | 'dspark';

/** One researched default acceptance. */
export interface AcceptancePreset {
  /** Constant per-position acceptance α_iid (derived from a reported τ at the reported k). Dimensionless, [0, 1). */
  alpha: number;
  /**
   * True if no source measured this method × use case and the value is carried over (another use case's value,
   * or the prose value + an offset), design doc §5.1 "(I)". Low confidence; shown as such.
   */
  interpolated: boolean;
}

/**
 * Default acceptance α per method × use case (design doc §5.1; T ≈ 1 sampling, chain verification). Sources:
 * - MTP multi-step: Qwen3.8-27B card k=7 T=1 (https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), Qwen3.5-4B blog
 *   k=15 (https://inco.ai/blog/dflash2/), llama.cpp PR 27210 k=3 (https://github.com/ggml-org/llama.cpp/pull/27210;
 *   reasoning 0.69 → 0.70), PR 22673 (https://github.com/ggml-org/llama.cpp/pull/22673). Agentic: prose (I).
 * - MTP single-step: DeepSeek-V3 report §5.4.3, 85–90% (https://arxiv.org/abs/2412.19437); FastMTP Table 1
 *   (https://arxiv.org/abs/2509.18362). Code and math interpolated; reasoning and agentic: prose (I).
 * - DFlash 2: Qwen3.8-27B card k=7; Muse-Glimmer-30B card k=15 (https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2);
 *   Qwen3.5-4B blog k=15. Reasoning: prose + 0.02 (I). Agentic: DFlash SWE-bench Lite τ 2.82 at k=15 → 0.65
 *   (DDTree, https://arxiv.org/abs/2604.12989 Table 1).
 * - DSpark: DSpark paper Table 1 k=7 (https://arxiv.org/abs/2607.05147), midpoint with Inco's and RadixArk's
 *   retrains. Reasoning: prose + 0.02 (I); agentic: prose (I).
 * Dimensionless.
 */
export const ACCEPTANCE_PRESETS: Record<
  AcceptancePresetMethod,
  Record<SpecUseCase, AcceptancePreset>
> = {
  mtp_multi_step: {
    prose: { alpha: 0.75, interpolated: false },
    code: { alpha: 0.82, interpolated: false },
    math: { alpha: 0.85, interpolated: false },
    reasoning: { alpha: 0.7, interpolated: false },
    agentic: { alpha: 0.75, interpolated: true },
  },
  mtp_single_step: {
    prose: { alpha: 0.85, interpolated: false },
    code: { alpha: 0.85, interpolated: true },
    math: { alpha: 0.87, interpolated: true },
    reasoning: { alpha: 0.85, interpolated: true },
    agentic: { alpha: 0.85, interpolated: true },
  },
  dflash2: {
    prose: { alpha: 0.79, interpolated: false },
    code: { alpha: 0.83, interpolated: false },
    math: { alpha: 0.87, interpolated: false },
    reasoning: { alpha: 0.81, interpolated: true },
    agentic: { alpha: 0.65, interpolated: false },
  },
  dspark: {
    prose: { alpha: 0.76, interpolated: false },
    code: { alpha: 0.84, interpolated: false },
    math: { alpha: 0.88, interpolated: false },
    reasoning: { alpha: 0.78, interpolated: true },
    agentic: { alpha: 0.76, interpolated: true },
  },
};

/**
 * Default draft length k per method (design doc §5.1): MTP multi-step 3, single-step 1, DFlash 2 7 (block 8),
 * DSpark 7 (the paper's block 8). Block drafters are further capped at `blockSize − 1` by {@link defaultDraftLength}.
 * Count.
 */
export const DEFAULT_DRAFT_LENGTH: Record<AcceptancePresetMethod, number> = {
  mtp_multi_step: 3,
  mtp_single_step: 1,
  dflash2: 7,
  dspark: 7,
};

/**
 * Per-position acceptance of positions past a single-step head's trained depth, after its first `depth`
 * positions (design doc §3.5): 0.30, 0.10, 0.05, then 0.05 onwards. Interpolated between FastMTP's vanilla
 * (~0.10 at position 2, arXiv 2509.18362) and fine-tuned (0.55) recursive reuse; low confidence. Dimensionless.
 */
export const MTP_TAIL_ALPHAS: readonly number[] = [0.3, 0.1, 0.05];

/** The acceptance-table row of a configured method and drafter (MTP: single-step iff `maxReliableDepth` is set). */
export function presetMethod(
  method: SpecDecodeConfig['method'],
  draft?: DraftSpec,
): AcceptancePresetMethod {
  switch (method) {
    case 'mtp':
      return draft?.kind === 'mtp' && draft.maxReliableDepth !== undefined
        ? 'mtp_single_step'
        : 'mtp_multi_step';
    case 'dflash2':
    case 'dspark':
      return method;
    default: {
      const _never: never = method;
      throw new Error(`unknown speculative method ${String(_never)}`);
    }
  }
}

/**
 * Default draft length k for a method and drafter: {@link DEFAULT_DRAFT_LENGTH}, capped at `blockSize − 1` for a
 * block drafter (e.g. DeepSeek-V4.1's in-checkpoint DSpark: `dspark_block_size` 5 → blockSize 6 → k 5). Count, ≥ 1.
 */
export function defaultDraftLength(method: SpecDecodeConfig['method'], draft?: DraftSpec): number {
  const k = DEFAULT_DRAFT_LENGTH[presetMethod(method, draft)];
  // Formula: k = min(k_default, blockSize − 1) for block drafters
  if (draft?.kind === 'block_drafter') return Math.max(1, Math.min(k, draft.blockSize - 1));
  return k;
}

/**
 * Acceptance of a single-step MTP head drafted past its trained depth: α for the first `maxReliableDepth`
 * positions, then {@link MTP_TAIL_ALPHAS} (`[α, 0.30, 0.10, 0.05, …]` for a depth-1 head; design doc §3.5).
 */
export function mtpTailAcceptance(alpha: number, maxReliableDepth: number): AcceptanceModel {
  if (!Number.isInteger(maxReliableDepth) || maxReliableDepth < 1) {
    throw new Error(`maxReliableDepth must be a positive integer, got ${maxReliableDepth}`);
  }
  return {
    kind: 'per_position',
    alphas: [...Array<number>(maxReliableDepth).fill(alpha), ...MTP_TAIL_ALPHAS],
  };
}

/**
 * The preset acceptance for a method, use case and draft length: the table's α as `iid`, except for a single-step
 * MTP head with `k > maxReliableDepth`, which gets the collapsing tail of {@link mtpTailAcceptance} (the engine
 * also warns in that case).
 */
export function presetAcceptance(
  method: SpecDecodeConfig['method'],
  useCase: SpecUseCase,
  k: number,
  draft?: DraftSpec,
): AcceptanceModel {
  const { alpha } = ACCEPTANCE_PRESETS[presetMethod(method, draft)][useCase];
  if (
    method === 'mtp' &&
    draft?.kind === 'mtp' &&
    draft.maxReliableDepth !== undefined &&
    k > draft.maxReliableDepth
  ) {
    return mtpTailAcceptance(alpha, draft.maxReliableDepth);
  }
  return { kind: 'iid', alpha };
}

/**
 * Newly committed tokens of the previous step that the drafter ingests per sequence, `m_ctx = ⌈E[τ]⌉` (design doc
 * §3.3): GEMM shapes need a whole row count, and rounding up is conservative by at most one activation row. A tiny
 * tolerance absorbs float error in E[τ] (e.g. 2.0000000000000004 → 2). Unit: tokens.
 */
export function draftContextRows(expected: number): number {
  // Formula: m_ctx = ⌈E[τ]⌉
  // See docs/simulation-model.md §13 ("Draft ops")
  return Math.ceil(expected - 1e-9);
}

/** A resolved speculative-decoding run: what lowering applies, E[τ], and the notes it produced. */
export interface ResolvedSpeculative {
  /** The verify / draft shape lowering applies. */
  lowering: SpeculativeLowering;
  /** E[τ], expected tokens committed per sequence per step (incl. the bonus token). Count (real, in [1, k + 1]). */
  expectedTokensPerStep: number;
  /** Human-readable notes (a single-step head drafted past its trained depth). */
  warnings: string[];
}

/**
 * Resolve the deployment's speculative decoding (design doc §3.3, §3.5): the drafter (`model.draft` for MTP, the
 * configured block drafter otherwise), k, the verify width `k + 1`, `m_ctx = ⌈E[τ]⌉` and the `k + 1` state
 * snapshots of the verify. Undefined when the deployment has none. Validation has already checked the config.
 */
export function resolveSpeculative(model: ModelSpec, config: SpecDecodeConfig | undefined): ResolvedSpeculative | undefined {
  if (config === undefined) return undefined;
  const draft = config.method === 'mtp' ? model.draft : config.drafter?.spec;
  if (draft === undefined) throw new Error(`speculative decoding '${config.method}': no drafter`);
  const k = config.draftLength;
  const expected = expectedTokens(config.acceptance, k);
  const warnings: string[] = [];
  if (draft.kind === 'mtp' && draft.maxReliableDepth !== undefined && k > draft.maxReliableDepth) {
    warnings.push(
      `speculative decoding: k = ${k} drafts past the MTP head's trained depth (${draft.maxReliableDepth}); ` +
        'acceptance collapses beyond it',
    );
  }
  return {
    lowering: {
      draft,
      draftLength: k,
      // Formula: verify width = k + 1;  m_ctx = ⌈E[τ]⌉;  snapshots = k + 1
      // See docs/simulation-model.md §13 "Verify"
      verifyTokens: k + 1,
      contextRows: draftContextRows(expected),
      stateSnapshots: k + 1,
    },
    expectedTokensPerStep: expected,
    warnings,
  };
}
