/**
 * Types of the Hugging Face config → model IR mapping (`mapHfModel`).
 *
 * The mapping is pure: the browser side (`src/hf/`) fetches the repo files and hands them over as
 * {@link HfRepoFiles}; this module never fetches. Pipeline (see `index.ts`):
 * unwrap `text_config` → dispatch on architecture / model type → family build → quant defaults → param check.
 */

import type { ModelSpec, QuantConfig } from '../../ir/model';

/** A parsed JSON object (a config.json or one of its sub-objects). */
export type JsonObject = Record<string, unknown>;

/**
 * The subset of the Hub's `GET /api/models/<id>` response that the mapping reads.
 * Every field is optional on the wire; the Hub omits fields it has no data for.
 */
export interface HfApiModelInfo {
  /** Repo ID, e.g. `'Qwen/Qwen3.8-27B'`. */
  id: string;
  /** Commit SHA of the revision the info describes. */
  sha?: string;
  /** `false`, or the gating mode (`'auto'` / `'manual'`) for gated repos. */
  gated?: boolean | string;
  /** True for private repos. */
  private?: boolean;
  /** Hub tags (library, model type, quant format, license, …). */
  tags?: string[];
  /** Model-card front matter; only `base_model` is read (the repo a quantized repo was made from). */
  cardData?: { base_model?: string | string[] };
  /**
   * Parameter counts from the safetensors headers, by safetensors dtype key (`BF16`, `F8_E4M3`, `U8`, …).
   * The Hub counts TENSOR ELEMENTS: a packed int4 word (`U8`, `I32`, …) counts as one element, not as the
   * 2 or 8 weights it holds. `total` is the sum over dtypes. Unit: elements. Absent for non-safetensors repos
   * (e.g. Mistral consolidated checkpoints).
   */
  safetensors?: { parameters: Record<string, number>; total: number };
  /** Files of the repo (only the names are read, to know which optional JSON files exist). */
  siblings?: { rfilename: string }[];
}

/** Everything the mapping needs from one Hub repo, already fetched and JSON-parsed. */
export interface HfRepoFiles {
  /** Repo ID the user asked for, e.g. `'mistralai/Mistral-Large-4-1T-A52B-NVFP4'`. */
  id: string;
  /** Revision the files were read at (branch name or commit SHA). */
  revision: string;
  /** Parsed config.json (or the base model's when the repo has none; see `configFrom`). */
  config: unknown;
  /** Repo the config came from when it is not `id` (e.g. a Mistral NVFP4 repo → its `cardData.base_model`). */
  configFrom?: string;
  /** Parsed `/api/models/<id>` response, if fetched. */
  api?: HfApiModelInfo;
  /** Parsed `hf_quant_config.json` (NVIDIA ModelOpt), if the repo has one. */
  hfQuantConfig?: unknown;
  /** Parsed `params.json` (Mistral consolidated format), if the repo has one. */
  paramsJson?: unknown;
  /** Parsed `quantization_config.json` (some AutoRound / Quark exports), if the repo has one. */
  quantizationConfigJson?: unknown;
}

/**
 * The config split into the language-model part and the wrapper (see `textConfig.ts`).
 * For a flat (text-only) config, `text === top`.
 */
export interface HfConfigView {
  /** The top-level config object. */
  top: JsonObject;
  /** The language-model config: `top.text_config` when present, else `top`. */
  text: JsonObject;
  /** True when `text` came from `top.text_config`. */
  nested: boolean;
  /** `top.model_type`, or null. */
  topModelType: string | null;
  /** `top.architectures` (empty when absent). */
  topArchitectures: string[];
  /** `text.model_type` when nested (absent for MiniMax / Inkling text configs), else null. */
  textModelType: string | null;
  /** `text.architectures` when nested (empty when absent). */
  textArchitectures: string[];
  /** `quantization_config` from the top level, else from `text_config`; undefined when neither has one. */
  quantizationConfig: unknown;
  /** `vision_config` from the top level, else from `text_config`; undefined when absent. */
  visionConfig: JsonObject | undefined;
  /** Human-readable notes about the unwrap (e.g. "language model only; vision/audio towers excluded"). */
  notes: string[];
}

/** Numeric-format defaults derived from the repo's quantization metadata (see `quant/index.ts`). */
export interface QuantDefaults {
  /** Formats of weights, activations and caches to simulate with. */
  quant: QuantConfig;
  /** Human-readable scheme label, e.g. `'bf16 (unquantized)'`, `'NVFP4 experts, FP8 attention'`. */
  scheme: string;
  /** Which file / key the scheme was read from. */
  source:
    | 'none'
    | 'quantization_config'
    | 'hf_quant_config.json'
    | 'params.json'
    | 'quantization (mlx)'
    | 'quantization_config.json';
  /** Human-readable notes (approximations, unrecognized fields). */
  notes: string[];
  /**
   * EXL3 repos only: the average and LM-head bits per weight of the checkpoint (`bits` / `head_bits`), which
   * prefill the UI's bpw inputs. Unit: bits/weight.
   */
  exl3?: { bpw: number; headBpw: number };
  /**
   * QuantConfig fields the repo's quant metadata sets explicitly (e.g. `'kvCache'` from `kv_cache_scheme` /
   * `kv_cache_quant_algo` / Mistral `dsa.fp8_kv_cache`, `'indexerKeyCache'`, `'lookupTables'`).
   * Family `quantHints` never override these. Empty when the metadata sets none.
   */
  explicit: (keyof QuantConfig)[];
}

/** Parameters the IR leaves out, estimated so the total can be checked against the checkpoint. */
export interface ExcludedParams {
  /** Multi-token-prediction (MTP / next-n / DSpark draft) layers in the checkpoint. Unit: params. */
  mtpParamsEstimate: number;
  /** Vision (and audio) towers and projectors in the checkpoint. Unit: params. */
  visionParamsEstimate: number;
  /** Names of the excluded parts, for display (e.g. `['MTP layer', 'vision tower']`). */
  parts: string[];
}

/**
 * Result of comparing the IR's parameter count with the checkpoint's `safetensors.total`
 * (see `paramCheck.ts` for the rules).
 */
export interface ParamCheck {
  /**
   * - `ok`: within ±{@link PARAM_CHECK_TOLERANCE} of the expected count.
   * - `mismatch`: outside the tolerance (a warning, not an error).
   * - `packed`: the checkpoint stores packed quantized words, so its element count is not a parameter count.
   * - `unavailable`: the repo publishes no safetensors totals.
   */
  status: 'ok' | 'mismatch' | 'packed' | 'unavailable';
  /** `safetensors.total` of the repo, when present. Unit: elements (≈ params for unpacked checkpoints). */
  expected?: number;
  /** IR count + excluded estimates: `countParams(model) + mtp + vision`. Unit: params. */
  actual: number;
  /** `(actual − expected) / expected`, when `expected` is present. Unit: dimensionless fraction. */
  relError?: number;
  /** Human-readable explanation of the status. */
  note: string;
}

/** What a family builder returns on success. */
export interface FamilyBuild {
  /** The language model as IR (MTP and vision/audio excluded; n-gram tables included). */
  model: ModelSpec;
  /** Human-readable notes: approximations, exclusions, ignored config fields. */
  notes: string[];
  /** Estimated MTP params in the checkpoint (0 when none). Unit: params. */
  mtpParamsEstimate: number;
  /** Estimated vision/audio params in the checkpoint (0 when none). Unit: params. */
  visionParamsEstimate: number;
  /** Names of the excluded parts, for display (e.g. `['MTP layer', 'vision tower']`). */
  excludedParts: string[];
  /**
   * Architecture-native formats that override the derived quant defaults (e.g. DeepSeek's mxfp8 / nvfp4 KV
   * formats, `recurrentState` from `mamba_ssm_dtype`). Merged over `QuantDefaults.quant`.
   */
  quantHints?: Partial<QuantConfig>;
}

/** What a family builder returns: a model, a config it cannot read, or an architecture it does not cover. */
export type FamilyBuildResult =
  | ({ kind: 'ok' } & FamilyBuild)
  | { kind: 'invalid_config'; errors: string[] }
  | { kind: 'unsupported'; reason: string };

/** One supported model family: which configs it claims and how it builds the IR. */
export interface FamilyEntry {
  /** Family name for display and results, e.g. `'qwen'`. */
  family: string;
  /** `model_type` values (top-level or `text_config`) this family handles. */
  modelTypes: string[];
  /** `architectures[0]` values (top-level or `text_config`) this family handles. */
  architectures: string[];
  /**
   * Build the IR. `text` is the language-model config (`text_config` or the flat config), `top` the
   * top-level config, `files` the whole repo (for params.json, api info, …). Never throws for bad configs:
   * returns `invalid_config` with path-qualified errors.
   */
  build(text: JsonObject, top: JsonObject, files: HfRepoFiles): FamilyBuildResult;
}

/** Result of {@link mapHfModel}. */
export type HfMapResult =
  | {
      kind: 'ok';
      /** The language model as IR. */
      model: ModelSpec;
      /** Family that built it, e.g. `'qwen'`. */
      family: string;
      /** The config's `model_type` (top-level, else text_config), or null; shown as the detected architecture. */
      modelType: string | null;
      /** The config's `architectures[0]` (top-level, else text_config), or null. */
      architecture: string | null;
      /** Notes from unwrap, family build, quant derivation and param check, in that order. */
      notes: string[];
      /** What the IR leaves out of the checkpoint. */
      excluded: ExcludedParams;
      /** Quant defaults, with the family's `quantHints` merged in. */
      quant: QuantDefaults;
      /** IR vs checkpoint parameter count. */
      paramCheck: ParamCheck;
    }
  | {
      kind: 'unsupported';
      /** The config's `model_type` (top-level, else text_config), or null. */
      modelType: string | null;
      /** The config's `architectures[0]` (top-level, else text_config), or null. */
      architecture: string | null;
      /** Supported model types, for the error message. */
      supported: string[];
      /** Why, when a family matched but cannot build it (e.g. a layer type the family does not cover). */
      reason?: string;
    }
  | {
      kind: 'invalid_config';
      /** Path-qualified errors, e.g. `'text_config.hidden_size: missing number'`. */
      errors: string[];
    };

/** Relative tolerance of the `ok` param-check status: ±2%. Unit: dimensionless fraction. */
export const PARAM_CHECK_TOLERANCE = 0.02;
