/**
 * Speculative decoding: accepted tokens per step and the researched acceptance defaults (design doc §3.5, §5.1).
 *
 * Pure math and data, no lowering. A step drafts `k` tokens and verifies `k + 1` positions in one target pass; it
 * commits τ tokens, the accepted prefix of the drafts plus the target's bonus token (1 ≤ τ ≤ k + 1). The engine
 * multiplies the step's sequence rate by E[τ] (see docs/simulation-model.md §13 "Accepted tokens").
 *
 * Acceptance α is the probability that one draft token is accepted (chain verification). The presets are α_iid values
 * FITTED to reported mean acceptance lengths τ at the reported k (design doc §5.1): the constant per-position α that
 * reproduces τ under {@link expectedTokens}. Each preset records its measurements (source, method version, target, k,
 * temperature, τ) and whether it was measured for that method × use case, borrowed or carried over. They are
 * user-facing defaults; α depends on the workload and on k, so the UI lets the user override α and flags a k the
 * preset was not fitted at ({@link presetDraftLengthExtrapolation}).
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

/**
 * One published mean acceptance length τ: where it comes from, what was run, and the value. Plain record of a
 * source; the presets below fit α to these (design doc §5.1, docs/simulation-model.md §13 "Acceptance defaults").
 */
export interface AcceptanceMeasurement {
  /** Source URL: paper, model card, blog post or PR comment. */
  url: string;
  /** Where in the source the number is: table, section, comment and row. */
  location: string;
  /** The speculative method and version actually measured, e.g. `'DFlash 2'`, `'DFlash v1'`, `'DSpark (Markov head)'`. */
  methodVersion: string;
  /** Target model, with its thinking mode where the source states it. */
  target: string;
  /** The drafter checkpoint, or who trained it, when it is not the method's own release. */
  drafter?: string;
  /** Benchmark or workload the τ is measured on. */
  benchmark: string;
  /** Draft length k of the run: tokens drafted per verify step (the verify checks k + 1 positions). Count. */
  draftLength: number;
  /** Sampling temperature of the run (0 = greedy); absent when the source does not state it. Dimensionless. */
  temperature?: number;
  /** Mean tokens committed per verify step τ, bonus token included, as reported. Count (real, in [1, k + 1]). */
  meanAcceptedTokens: number;
}

/**
 * How a preset's α relates to its `measurements`:
 * - `measured`: α reproduces the τ of every run in `measurements` (a run = one source table at one k; its τ is the
 *   mean over the use case's benchmarks and targets), i.e. |α − α_iid(τ, k)| ≤ {@link FIT_TOLERANCE_ALPHA}. Their k
 *   are the k the α was fitted at.
 * - `borrowed`: the same, but the runs are of another version of the method (DFlash v1 for DFlash 2): a proxy.
 * - `pooled`: α is a rounded value between the runs in `measurements` and reproduces none of them: it is fitted at no
 *   k. Low confidence.
 * - `carried_over`: no source measured this method × use case; α is another use case's α plus an offset (design doc
 *   §5.1 "(I)"), and `measurements` are that use case's. Low confidence.
 */
export type AcceptanceBasis = 'measured' | 'borrowed' | 'pooled' | 'carried_over';

/**
 * How close α must be to a run's α_iid(τ, k) for the run to count as fitted: α is quoted to 0.01 (± 0.005) and τ to
 * 0.01 (about ± 0.001 in α). Dimensionless.
 */
export const FIT_TOLERANCE_ALPHA = 0.006;

/** One researched default acceptance, with its provenance. */
export interface AcceptancePreset {
  /** Constant per-position acceptance α_iid used by {@link expectedTokens}. Dimensionless, [0, 1). */
  alpha: number;
  /** How α relates to `measurements`. */
  basis: AcceptanceBasis;
  /** `carried_over` only: the use case α is taken from and the offset added to it: α = α(useCase) + alphaOffset. */
  carriedFrom?: { useCase: SpecUseCase; alphaOffset: number };
  /**
   * The reported τ behind α (non-empty): the runs it reproduces (`measured`, `borrowed`; their k are the fitted k,
   * {@link fittedDraftLengths}), the runs it lies between (`pooled`), or the source use case's (`carried_over`).
   */
  measurements: readonly AcceptanceMeasurement[];
  /**
   * Reports of this method × use case that α does not reproduce and is not fitted to (other k, other targets,
   * community retrains). Cited for context; they never count as a fitted k.
   */
  otherReports?: readonly AcceptanceMeasurement[];
  /** Caveats a reader of the table needs: mixed temperatures, how far the cited reports are from α. */
  note?: string;
}

/** Shared setup of the Qwen3.8-27B DFlash 2 model card's "Acceptance Length" table (SGLang, H200, 4096 new tokens). */
const QWEN38_CARD = {
  url: 'https://huggingface.co/incoai/Qwen3.8-27B-DFlash2',
  location: 'model card, "Acceptance Length" table',
  target: 'Qwen3.8-27B (thinking, xhigh effort)',
  draftLength: 7,
  temperature: 1,
} as const;

/** Shared setup of the Muse-Glimmer-30B DFlash 2 model card's "Acceptance Length" table. */
const MUSE_CARD = {
  url: 'https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2',
  location: 'model card, "Acceptance Length" table',
  target: 'Muse-Glimmer-30B (high reasoning strength)',
  draftLength: 15,
  temperature: 1,
} as const;

/**
 * Shared setup of the DFlash 2 blog's matched Qwen3.5-4B comparison, Table 3: thinking enabled, temperature 1.0,
 * top-p 0.95, top-k 20, presence penalty 1.5, rejection sampling; k 15 (Figure 5 shows 15 draft positions). Figure 5
 * also gives per-position conditional acceptance on MATH-500 for MTP, DFlash, DSpark and DFlash 2 (recorded, not
 * used: the presets stay i.i.d.).
 */
const INCO_BLOG_QWEN35 = {
  url: 'https://inco.ai/blog/dflash2/',
  location: 'Table 3 ("Putting It Together")',
  target: 'Qwen3.5-4B (thinking)',
  draftLength: 15,
  temperature: 1,
} as const;

/** Shared setup of llama.cpp PR 27210's author run (Qwen3.8-27B Q8_0, 2× R9700, config C1 = fixed MTP depth 3, greedy). */
const LLAMACPP_27210 = {
  url: 'https://github.com/ggml-org/llama.cpp/pull/27210#issuecomment-5385811671',
  location: 'comment, "Draft acceptance" table, C1 (τ = 1 + 3 × ratio)',
  methodVersion: 'MTP, built-in multi-step head (Qwen3.8)',
  draftLength: 3,
  temperature: 0,
} as const;

/** Shared setup of the DSpark paper's Table 1 (DeepSeek-trained 5-layer drafters, non-thinking mode, fixed block). */
const DSPARK_PAPER = {
  url: 'https://arxiv.org/abs/2607.05147',
  location: 'Table 1',
  methodVersion: 'DSpark (Markov head)',
  drafter: 'DeepSeek-trained, 5 layers',
  draftLength: 7,
  temperature: 1,
} as const;

/** Shared setup of DeepSeek-V3's MTP acceptance (technical report §5.4.3, one next-token module, k 1). */
const DEEPSEEK_V3 = {
  url: 'https://arxiv.org/abs/2412.19437',
  location: '§5.4.3: second-token acceptance 85–90% (low end; τ = 1 + 0.85)',
  methodVersion: 'MTP, single-step head (DeepSeek-V3)',
  target: 'DeepSeek-V3',
  draftLength: 1,
} as const;

/** Shared setup of FastMTP's Table 1 vanilla MTP row at K = 1 (MiMo-RL-7B, SGLang, greedy). */
const FASTMTP_VANILLA = {
  url: 'https://arxiv.org/abs/2509.18362',
  location: 'Table 1, K = 1, Vanilla MTP',
  methodVersion: 'MTP, vanilla single-step head (MiMo)',
  target: 'MiMo-RL-7B',
  draftLength: 1,
  temperature: 0,
} as const;

const MTP_QWEN38 = 'MTP, built-in multi-step head (Qwen3.8)';
const MTP_QWEN35 = 'MTP, built-in multi-step head (Qwen3.5)';
const LLAMACPP_TARGET_OFF = 'Qwen3.8-27B Q8_0 (thinking off)';

const MTP_MULTI_PROSE: readonly AcceptanceMeasurement[] = [
  {
    ...INCO_BLOG_QWEN35,
    methodVersion: MTP_QWEN35,
    benchmark: 'MT-Bench',
    meanAcceptedTokens: 3.9,
  },
];

const MTP_SINGLE_PROSE: readonly AcceptanceMeasurement[] = [
  { ...DEEPSEEK_V3, benchmark: 'various generation topics', meanAcceptedTokens: 1.85 },
];

const DFLASH2_PROSE: readonly AcceptanceMeasurement[] = [
  { ...QWEN38_CARD, methodVersion: 'DFlash 2', benchmark: 'MT-Bench', meanAcceptedTokens: 4.1 },
];

/** DSpark paper Table 1 rows of one benchmark for its three Qwen3 targets (non-thinking), τ in target order 4B, 8B, 14B. */
function dsparkPaperRows(
  benchmark: string,
  tau4b: number,
  tau8b: number,
  tau14b: number,
): AcceptanceMeasurement[] {
  return [
    { ...DSPARK_PAPER, target: 'Qwen3-4B (non-thinking)', benchmark, meanAcceptedTokens: tau4b },
    { ...DSPARK_PAPER, target: 'Qwen3-8B (non-thinking)', benchmark, meanAcceptedTokens: tau8b },
    { ...DSPARK_PAPER, target: 'Qwen3-14B (non-thinking)', benchmark, meanAcceptedTokens: tau14b },
  ];
}

/** Inco's matched DSpark retrain on Qwen3.5-4B (DFlash 2 blog Table 3), one benchmark. */
function dsparkIncoRow(benchmark: string, tau: number): AcceptanceMeasurement {
  return {
    ...INCO_BLOG_QWEN35,
    methodVersion: 'DSpark (Markov head)',
    drafter: 'Inco-trained, matched setup',
    benchmark,
    meanAcceptedTokens: tau,
  };
}

/** Community DSpark drafters measured on the DFlash 2 model cards (k 7 on Qwen3.8-27B, k 15 on Muse-Glimmer-30B). */
function dsparkCommunityRows(
  benchmark: string,
  tauQwen38: number,
  tauMuse: number,
): AcceptanceMeasurement[] {
  return [
    {
      ...QWEN38_CARD,
      methodVersion: 'DSpark (Markov head)',
      drafter: 'RadixArk/Qwen3.8-27B-DSpark (community)',
      benchmark,
      meanAcceptedTokens: tauQwen38,
    },
    {
      ...MUSE_CARD,
      methodVersion: 'DSpark (Markov head)',
      drafter: 'DaoCloud/Muse-Glimmer-30B-DSpark (community)',
      benchmark,
      meanAcceptedTokens: tauMuse,
    },
  ];
}

const DSPARK_PROSE: readonly AcceptanceMeasurement[] = dsparkPaperRows('MT-Bench', 3.64, 3.72, 3.7);

/**
 * Default acceptance α per method × use case, with provenance (design doc §5.1; chain verification). "Prose" is
 * MT-Bench, "code" HumanEval + MBPP, "math" GSM8K + MATH-500. The comments give each run's α_iid (τ averaged over
 * the use case's benchmarks and the run's targets, then {@link iidAlphaForTokens}); the full table with every τ is in
 * docs/simulation-model.md §13 "Acceptance defaults". Dimensionless.
 */
export const ACCEPTANCE_PRESETS: Record<
  AcceptancePresetMethod,
  Record<SpecUseCase, AcceptancePreset>
> = {
  mtp_multi_step: {
    // fitted: blog k 15 .746; cited: card k 7 .763, llama.cpp greedy k 3 .735
    prose: {
      alpha: 0.75,
      basis: 'measured',
      measurements: MTP_MULTI_PROSE,
      otherReports: [
        {
          ...QWEN38_CARD,
          methodVersion: MTP_QWEN38,
          benchmark: 'MT-Bench',
          meanAcceptedTokens: 3.74,
        },
        {
          ...LLAMACPP_27210,
          target: LLAMACPP_TARGET_OFF,
          benchmark: 'essay prompt (prose lane), ratio 0.557',
          meanAcceptedTokens: 2.671,
        },
      ],
      note: 'At k 3, 0.75 gives E[τ] 2.73 against the greedy llama.cpp run’s 2.67.',
    },
    // pooled: card k 7 .782, blog k 15 .782, llama.cpp greedy k 3 .883
    code: {
      alpha: 0.82,
      basis: 'pooled',
      measurements: [
        {
          ...QWEN38_CARD,
          methodVersion: MTP_QWEN38,
          benchmark: 'HumanEval',
          meanAcceptedTokens: 3.91,
        },
        { ...QWEN38_CARD, methodVersion: MTP_QWEN38, benchmark: 'MBPP', meanAcceptedTokens: 3.99 },
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: MTP_QWEN35,
          benchmark: 'HumanEval',
          meanAcceptedTokens: 4.84,
        },
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: MTP_QWEN35,
          benchmark: 'MBPP',
          meanAcceptedTokens: 4.16,
        },
        {
          ...LLAMACPP_27210,
          target: LLAMACPP_TARGET_OFF,
          benchmark: 'C++ LRU store prompt (code lane), ratio 0.784',
          meanAcceptedTokens: 3.352,
        },
      ],
      note: 'Mixes temperatures: the T = 1 card and blog give .78, the greedy llama.cpp run .88; 0.82 sits between.',
    },
    // fitted: card k 7 .851; cited: blog k 15 .802
    math: {
      alpha: 0.85,
      basis: 'measured',
      measurements: [
        { ...QWEN38_CARD, methodVersion: MTP_QWEN38, benchmark: 'GSM8K', meanAcceptedTokens: 5.02 },
        {
          ...QWEN38_CARD,
          methodVersion: MTP_QWEN38,
          benchmark: 'MATH-500',
          meanAcceptedTokens: 4.72,
        },
      ],
      otherReports: [
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: MTP_QWEN35,
          benchmark: 'GSM8K',
          meanAcceptedTokens: 4.78,
        },
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: MTP_QWEN35,
          benchmark: 'MATH-500',
          meanAcceptedTokens: 5.04,
        },
      ],
      note: 'The blog’s Figure 5 gives this head’s per-position acceptance on MATH-500 (84.6% falling to about 78%).',
    },
    // fitted: llama.cpp greedy k 3 .684
    reasoning: {
      alpha: 0.68,
      basis: 'measured',
      measurements: [
        {
          ...LLAMACPP_27210,
          target: 'Qwen3.8-27B Q8_0 (thinking on)',
          benchmark: 'essay prompt, all thinking tokens (reasoning lane), ratio 0.490',
          meanAcceptedTokens: 2.47,
        },
      ],
      note: 'One greedy run; T = 1 is usually lower.',
    },
    agentic: {
      alpha: 0.75,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0 },
      measurements: MTP_MULTI_PROSE,
    },
  },
  mtp_single_step: {
    // fitted: DeepSeek-V3 k 1 .85
    prose: {
      alpha: 0.85,
      basis: 'measured',
      measurements: MTP_SINGLE_PROSE,
      otherReports: [{ ...FASTMTP_VANILLA, benchmark: 'MT-Bench', meanAcceptedTokens: 1.72 }],
      note: 'DeepSeek-V3 reports one rate across topics, not prose alone.',
    },
    code: {
      alpha: 0.85,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0 },
      measurements: MTP_SINGLE_PROSE,
      otherReports: [{ ...FASTMTP_VANILLA, benchmark: 'code', meanAcceptedTokens: 1.73 }],
    },
    math: {
      alpha: 0.87,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0.02 },
      measurements: MTP_SINGLE_PROSE,
      otherReports: [{ ...FASTMTP_VANILLA, benchmark: 'math', meanAcceptedTokens: 1.83 }],
      note: 'The +0.02 follows FastMTP, where math is above chat.',
    },
    reasoning: {
      alpha: 0.85,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0 },
      measurements: MTP_SINGLE_PROSE,
    },
    agentic: {
      alpha: 0.85,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0 },
      measurements: MTP_SINGLE_PROSE,
    },
  },
  dflash2: {
    // fitted: card k 7 .795; cited: Muse k 15 .778, blog k 15 .815
    prose: {
      alpha: 0.79,
      basis: 'measured',
      measurements: DFLASH2_PROSE,
      otherReports: [
        {
          ...MUSE_CARD,
          methodVersion: 'DFlash 2',
          benchmark: 'MT-Bench',
          meanAcceptedTokens: 4.42,
        },
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: 'DFlash 2',
          benchmark: 'MT-Bench',
          meanAcceptedTokens: 5.2,
        },
      ],
    },
    // fitted: card k 7 .832, Muse k 15 .826; cited: blog k 15 .839
    code: {
      alpha: 0.83,
      basis: 'measured',
      measurements: [
        {
          ...QWEN38_CARD,
          methodVersion: 'DFlash 2',
          benchmark: 'HumanEval',
          meanAcceptedTokens: 4.39,
        },
        { ...QWEN38_CARD, methodVersion: 'DFlash 2', benchmark: 'MBPP', meanAcceptedTokens: 4.79 },
        {
          ...MUSE_CARD,
          methodVersion: 'DFlash 2',
          benchmark: 'HumanEval',
          meanAcceptedTokens: 5.66,
        },
        { ...MUSE_CARD, methodVersion: 'DFlash 2', benchmark: 'MBPP', meanAcceptedTokens: 5.3 },
      ],
      otherReports: [
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: 'DFlash 2',
          benchmark: 'HumanEval',
          meanAcceptedTokens: 6.28,
        },
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: 'DFlash 2',
          benchmark: 'MBPP',
          meanAcceptedTokens: 5.41,
        },
      ],
    },
    // pooled: card k 7 .882, Muse k 15 .862, blog k 15 .859
    math: {
      alpha: 0.87,
      basis: 'pooled',
      measurements: [
        { ...QWEN38_CARD, methodVersion: 'DFlash 2', benchmark: 'GSM8K', meanAcceptedTokens: 5.46 },
        {
          ...QWEN38_CARD,
          methodVersion: 'DFlash 2',
          benchmark: 'MATH-500',
          meanAcceptedTokens: 5.28,
        },
        { ...MUSE_CARD, methodVersion: 'DFlash 2', benchmark: 'GSM8K', meanAcceptedTokens: 6.57 },
        {
          ...MUSE_CARD,
          methodVersion: 'DFlash 2',
          benchmark: 'MATH-500',
          meanAcceptedTokens: 6.56,
        },
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: 'DFlash 2',
          benchmark: 'GSM8K',
          meanAcceptedTokens: 6.2,
        },
        {
          ...INCO_BLOG_QWEN35,
          methodVersion: 'DFlash 2',
          benchmark: 'MATH-500',
          meanAcceptedTokens: 6.76,
        },
      ],
      note: 'The blog’s Figure 5 gives DFlash 2’s per-position acceptance on MATH-500: flat near 85–86% over 15 positions.',
    },
    reasoning: {
      alpha: 0.81,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0.02 },
      measurements: DFLASH2_PROSE,
    },
    // fitted (DFlash v1, not DFlash 2): DDTree k 15 .646
    agentic: {
      alpha: 0.65,
      basis: 'borrowed',
      measurements: [
        {
          url: 'https://arxiv.org/abs/2604.12989',
          location: 'DDTree Table 1, T = 1, DFlash column',
          methodVersion: 'DFlash v1',
          target: 'Qwen3-8B',
          drafter: 'z-lab DFlash checkpoint (block 16)',
          benchmark: 'SWE-bench Lite',
          draftLength: 15,
          temperature: 1,
          meanAcceptedTokens: 2.82,
        },
      ],
      note: 'No DFlash 2 agentic measurement exists; DFlash 2 is above v1 elsewhere, so this proxy is likely low.',
    },
  },
  dspark: {
    // fitted: paper k 7 .759 (Qwen3-4B/8B/14B .754/.762/.760); cited: Inco k 15 .796, community .684 (k 7), .723 (k 15)
    prose: {
      alpha: 0.76,
      basis: 'measured',
      measurements: DSPARK_PROSE,
      otherReports: [
        dsparkIncoRow('MT-Bench', 4.77),
        ...dsparkCommunityRows('MT-Bench', 3.01, 3.59),
      ],
      note:
        'Fitted to the paper (non-thinking Qwen3, k 7). At k 15, 0.76 gives E[τ] 4.12 against Inco’s retrain’s 4.77; ' +
        'at k 7 it gives 3.70 against the community Qwen3.8-27B thinking retrain’s 3.01 (+23%).',
    },
    // pooled: paper k 7 .879, Inco k 15 .822
    code: {
      alpha: 0.84,
      basis: 'pooled',
      measurements: [
        ...dsparkPaperRows('HumanEval', 5.38, 5.52, 5.43),
        ...dsparkPaperRows('MBPP', 5.13, 5.16, 5.26),
        dsparkIncoRow('HumanEval', 5.8),
        dsparkIncoRow('MBPP', 4.96),
      ],
      otherReports: [
        ...dsparkCommunityRows('HumanEval', 3.3, 4.33),
        ...dsparkCommunityRows('MBPP', 3.51, 4.02),
      ],
    },
    // pooled: paper k 7 .913, Inco k 15 .843
    math: {
      alpha: 0.88,
      basis: 'pooled',
      measurements: [
        ...dsparkPaperRows('GSM8K', 6.11, 6.17, 6.21),
        ...dsparkPaperRows('MATH-500', 5.7, 5.78, 5.74),
        dsparkIncoRow('GSM8K', 5.69),
        dsparkIncoRow('MATH-500', 6.2),
      ],
      otherReports: [
        ...dsparkCommunityRows('GSM8K', 4.36, 5.45),
        ...dsparkCommunityRows('MATH-500', 3.92, 5.01),
      ],
      note: 'The blog’s Figure 5 gives Inco’s DSpark per-position acceptance on MATH-500 (87.2% falling to 79.9%).',
    },
    reasoning: {
      alpha: 0.78,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0.02 },
      measurements: DSPARK_PROSE,
    },
    agentic: {
      alpha: 0.76,
      basis: 'carried_over',
      carriedFrom: { useCase: 'prose', alphaOffset: 0 },
      measurements: DSPARK_PROSE,
    },
  },
};

/**
 * The constant α that reproduces a mean acceptance length τ at draft length k under the i.i.d. model: the inverse
 * of {@link expectedTokens}, (1 − α^(k+1)) / (1 − α) = τ, solved by bisection (E[τ] rises monotonically in α).
 * `meanAcceptedTokens` is a count in [1, k + 1), `k` an integer ≥ 1. Dimensionless, [0, 1).
 */
export function iidAlphaForTokens(meanAcceptedTokens: number, k: number): number {
  if (!Number.isInteger(k) || k < 1)
    throw new Error(`draft length k must be a positive integer, got ${k}`);
  if (!(meanAcceptedTokens >= 1 && meanAcceptedTokens < k + 1)) {
    throw new Error(`τ must be in [1, k + 1) = [1, ${k + 1}), got ${meanAcceptedTokens}`);
  }
  let low = 0;
  let high = 1;
  for (let i = 0; i < 60; i++) {
    const mid = (low + high) / 2;
    if (expectedTokens({ kind: 'iid', alpha: mid }, k) < meanAcceptedTokens) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}

/** One run of measurements: one source table at one k, its τ averaged over benchmarks and targets. */
export interface AcceptanceRun {
  /** Source URL of the run. */
  url: string;
  /** Where in the source. */
  location: string;
  /** Draft length k of the run. Count. */
  draftLength: number;
  /** Mean of the run's τ over its benchmarks and targets. Count (real, in [1, k + 1)). */
  meanAcceptedTokens: number;
  /** α_iid(meanAcceptedTokens, draftLength), {@link iidAlphaForTokens}. Dimensionless. */
  alphaIid: number;
}

/**
 * Groups measurements into runs (same URL, location, method version, drafter and k) and gives each run's mean τ and
 * α_iid: a use case such as "code" (HumanEval + MBPP) is one run per source table.
 */
export function acceptanceRuns(measurements: readonly AcceptanceMeasurement[]): AcceptanceRun[] {
  const runs = new Map<string, AcceptanceMeasurement[]>();
  for (const m of measurements) {
    const key = [m.url, m.location, m.methodVersion, m.drafter ?? '', m.draftLength].join('|');
    runs.set(key, [...(runs.get(key) ?? []), m]);
  }
  return [...runs.values()].map((group) => {
    const { url, location, draftLength } = group[0];
    // Formula: τ_run = mean of the run's τ;  α_iid = (1 − α^(k+1)) / (1 − α) solved for τ_run
    const meanAcceptedTokens =
      group.reduce((sum, m) => sum + m.meanAcceptedTokens, 0) / group.length;
    return {
      url,
      location,
      draftLength,
      meanAcceptedTokens,
      alphaIid: iidAlphaForTokens(meanAcceptedTokens, draftLength),
    };
  });
}

/**
 * The draft lengths k a preset's α was fitted at, ascending, without repeats: the k of the runs it reproduces
 * (`measured`, `borrowed`; for `carried_over` those of the use case it was carried from). Empty for `pooled`, whose α
 * reproduces no run. Reports in `otherReports` never count. Count.
 */
export function fittedDraftLengths(preset: AcceptancePreset): number[] {
  if (preset.basis === 'pooled') return [];
  return [...new Set(preset.measurements.map((m) => m.draftLength))].sort((a, b) => a - b);
}

/** The draft lengths of the runs a `pooled` preset lies between (empty otherwise), ascending. Count. */
export function pooledDraftLengths(preset: AcceptancePreset): number[] {
  if (preset.basis !== 'pooled') return [];
  return [...new Set(preset.measurements.map((m) => m.draftLength))].sort((a, b) => a - b);
}

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
 * The draft lengths a preset α was fitted at ({@link fittedDraftLengths}; for a `pooled` preset, the k of the runs
 * it is pooled from, {@link pooledDraftLengths}) when it is used at a k none of them equals, else undefined. The
 * i.i.d. model then carries α to a k no source measured, and the real per-position acceptance may differ
 * (practicality review F10: an α fitted at k 7 to a decaying curve gives E[τ] +18% at k 15). A cited report at k
 * (`otherReports`) does not clear the warning. Also undefined when the user entered α (`userAlpha`), and for a
 * single-step MTP head past its depth, where {@link presetAcceptance} swaps in the collapsing tail (the engine warns
 * there). `k` is a count.
 */
export function presetDraftLengthExtrapolation(
  method: SpecDecodeConfig['method'],
  useCase: SpecUseCase,
  k: number,
  draft?: DraftSpec,
  userAlpha?: number,
): number[] | undefined {
  if (userAlpha !== undefined) return undefined;
  if (presetAcceptance(method, useCase, k, draft).kind !== 'iid') return undefined;
  const preset = ACCEPTANCE_PRESETS[presetMethod(method, draft)][useCase];
  const measuredAt =
    preset.basis === 'pooled' ? pooledDraftLengths(preset) : fittedDraftLengths(preset);
  return measuredAt.includes(k) ? undefined : measuredAt;
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
