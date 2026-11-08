/**
 * Block drafters for speculative decoding (design doc §3.1–3.2): a drafter repo's `config.json` → `BlockDraftSpec`,
 * the known target → drafter repos, and the labelled hypothetical stand-in.
 *
 * Pure (no fetching, like the rest of `models/hf`): the browser side fetches
 * `https://huggingface.co/<repoId>/resolve/main/config.json` and hands the parsed JSON over.
 *
 * Config fields read (Qwen3-architecture drafters; both released families share them):
 * - `architectures[0]`: `DFlash2DraftModel` → DFlash 2 (z-lab `dflash/model.py`), `Qwen3DSparkModel` → DSpark
 *   (DeepSpec `deepspec/modeling/dspark`). Anything else is rejected.
 * - `hidden_size`, `vocab_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`
 *   (default hidden/heads), `intermediate_size`, `hidden_act` (gated MLP, `Qwen3MLP`).
 * - Window: `sliding_window` when every `layer_types` entry is `sliding_attention` (else full attention; mixed
 *   layer types are rejected, the spec has one window).
 * - `dflash_config.*`, else the top level: `target_layer_ids` (default DFlash's `build_target_layer_ids(
 *   num_target_layers, num_hidden_layers)`), `block_size` (default 16, as `DFlashDraftModel`), DFlash 2
 *   `conv_kernel_size`, `conv_group_size`, `selector_top_k`, `selector_rank`; DSpark `markov_rank`.
 *
 * `BlockDraftSpec.blockSize` is the largest verify width k + 1 the drafter supports. DFlash's `block_size` already
 * counts the anchor (k ≤ block_size − 1); DSpark's counts draft positions (DeepSpec `common.py`: "number of draft
 * positions per anchor", k ≤ block_size), so a DSpark config maps to `blockSize = block_size + 1`.
 */

import type { BlockDraftSpec, DraftSpec, LayerSpec, ModelSpec, MtpDraftSpec } from '../../ir/model';
import { createReader, isJsonObject, type ConfigReader } from './reader';
import type { JsonObject } from './types';

/**
 * The checkpoint's MTP draft (design doc §3.2) from the module layer an HF family already builds for its param
 * estimate: `count` copies named `mtp.<j>` (modules in chain order), with `eh_proj`. `singleStep`: DeepSeek-V3-style
 * `num_nextn_predict_layers` modules, each trained for one draft position, so `maxReliableDepth = count`; false for
 * heads trained for several positions (Qwen3.5+/3.8 `mtp_num_hidden_layers`). Undefined when `count` is 0. A module
 * layer is costed exactly like a target layer of the same blocks.
 */
export function mtpDraft(
  moduleLayer: (j: number) => LayerSpec,
  count: number,
  singleStep: boolean,
): MtpDraftSpec | undefined {
  if (count <= 0) return undefined;
  const layers = Array.from({ length: count }, (_, j) => ({ ...moduleLayer(j), name: `mtp.${j}` }));
  return {
    kind: 'mtp',
    layers,
    hiddenProj: true,
    ...(singleStep ? { maxReliableDepth: count } : {}),
  };
}

/**
 * The model with its draft attached as `model.draft`; the model itself when there is none, so a checkpoint without
 * a drafter keeps exactly the spec it had.
 */
export function withDraft(model: ModelSpec, draft: DraftSpec | undefined): ModelSpec {
  return draft === undefined ? model : { ...model, draft };
}

/** Result of {@link parseDrafterConfig}. */
export type DrafterParseResult =
  | {
      kind: 'ok';
      /** The drafter's shape. */
      spec: BlockDraftSpec;
      /** Human-readable notes (defaults applied, fields not modelled). */
      notes: string[];
    }
  | {
      kind: 'invalid_config';
      /** Path-qualified errors, e.g. `'dflash_config.block_size: expected non-negative integer'`. */
      errors: string[];
    };

/** The `architectures[0]` values of the supported drafters and their method. */
const DRAFTER_ARCHITECTURES: Record<string, BlockDraftSpec['method']> = {
  DFlash2DraftModel: 'dflash2',
  Qwen3DSparkModel: 'dspark',
};

/** Default `block_size` of `DFlashDraftModel` when the config has none. Count. */
const DFLASH_DEFAULT_BLOCK_SIZE = 16;

/**
 * DFlash's `build_target_layer_ids(num_target_layers, num_draft_layers)`: the middle layer for one draft layer,
 * else `round(1 + i·(L − 4)/(n − 1))` for i = 0..n−1, spread over layers 1..L−3. Zero-based layer indices.
 * Source: https://github.com/z-lab/dflash/blob/main/dflash/model.py
 */
export function buildTargetLayerIds(numTargetLayers: number, numDraftLayers: number): number[] {
  if (numDraftLayers === 1) return [Math.floor(numTargetLayers / 2)];
  const start = 1;
  const end = numTargetLayers - 3;
  const span = end - start;
  // Formula: id_i = round(1 + i·(L − 4)/(n − 1)); Python round() is half-to-even, matched for .5 below
  // See docs/hf-loader.md ("Drafters")
  return Array.from({ length: numDraftLayers }, (_, i) =>
    pythonRound(start + (i * span) / (numDraftLayers - 1)),
  );
}

/** Python 3 `round()` of a float to an integer: half to even. */
function pythonRound(x: number): number {
  const floor = Math.floor(x);
  const diff = x - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** The MLP activation of a `hidden_act` string. */
function readActivation(r: ConfigReader): BlockDraftSpec['activation'] {
  const act = r.optStr('hidden_act') ?? 'silu';
  if (act === 'silu' || act === 'swish') return 'silu';
  if (act.startsWith('gelu')) return 'gelu';
  if (act === 'relu') return 'relu';
  r.errors.push(`${r.path ? `${r.path}.` : ''}hidden_act: unsupported activation '${act}'`);
  return 'silu';
}

/**
 * Parse a block drafter's `config.json` into a {@link BlockDraftSpec} (`source` = `repoId`, plus `@revision`
 * when given). Never throws: returns `invalid_config` with path-qualified errors.
 */
export function parseDrafterConfig(
  config: JsonObject,
  repoId: string,
  revision?: string,
): DrafterParseResult {
  const r = createReader(config);
  const notes: string[] = [];
  const arch = r.optStrArray('architectures')?.[0];
  const method = arch === undefined ? undefined : DRAFTER_ARCHITECTURES[arch];
  if (method === undefined) {
    return {
      kind: 'invalid_config',
      errors: [
        `architectures: expected one of ${Object.keys(DRAFTER_ARCHITECTURES).join(', ')}, got ${arch ?? 'none'}`,
      ],
    };
  }
  const hidden = r.int('hidden_size');
  const vocab = r.int('vocab_size');
  const layers = r.int('num_hidden_layers');
  const heads = r.int('num_attention_heads');
  const kvHeads = r.int('num_key_value_heads');
  const headDim = r.optInt('head_dim') ?? (heads > 0 ? hidden / heads : 0);
  const intermediate = r.int('intermediate_size');
  const activation = readActivation(r);

  // DFlash keeps its drafter fields under `dflash_config`; DSpark (and older DFlash) at the top level.
  const d = isJsonObject(config.dflash_config)
    ? createReader(config.dflash_config, 'dflash_config', r.errors)
    : r;
  const pick = (key: string) => (d.has(key) ? d : r);

  const layerTypes = r.optStrArray('layer_types');
  const sliding =
    layerTypes !== undefined &&
    layerTypes.length > 0 &&
    layerTypes.every((t) => t === 'sliding_attention');
  if (layerTypes !== undefined && !sliding && !layerTypes.every((t) => t === 'full_attention')) {
    r.errors.push('layer_types: mixed sliding and full attention layers are not supported');
  }
  if (layerTypes !== undefined && layerTypes.length !== layers)
    r.errors.push('layer_types: length must equal num_hidden_layers');
  const window = sliding ? r.int('sliding_window') : undefined;

  let targetLayerIds = pick('target_layer_ids').optNumArray('target_layer_ids');
  if (targetLayerIds === undefined) {
    const numTarget = r.int('num_target_layers');
    targetLayerIds = buildTargetLayerIds(numTarget, layers);
    notes.push(
      `target_layer_ids absent: DFlash's build_target_layer_ids(${numTarget}, ${layers}) used`,
    );
  }
  if (targetLayerIds.length === 0 || targetLayerIds.some((id) => !Number.isInteger(id) || id < 0)) {
    r.errors.push('target_layer_ids: expected non-empty zero-based layer indices');
  }
  const blockSizeIn = pick('block_size').optInt('block_size');
  if (blockSizeIn === undefined && method === 'dflash2')
    notes.push(`block_size absent: DFlash default ${DFLASH_DEFAULT_BLOCK_SIZE} used`);
  if (blockSizeIn === undefined && method === 'dspark')
    r.errors.push('block_size: missing non-negative integer');

  const spec: BlockDraftSpec = {
    kind: 'block_drafter',
    method,
    hidden,
    vocab,
    layers,
    heads,
    kvHeads,
    headDim,
    intermediate,
    gated: true,
    activation,
    ...(window !== undefined ? { windowTokens: window } : {}),
    targetLayerIds,
    // Formula: blockSize = DFlash: block_size; DSpark: block_size + 1 (its block_size counts drafts only)
    // See docs/hf-loader.md ("Drafters"), docs/simulation-model.md §13 ("Draft spec")
    blockSize:
      method === 'dspark' ? (blockSizeIn ?? 0) + 1 : (blockSizeIn ?? DFLASH_DEFAULT_BLOCK_SIZE),
    source: revision === undefined ? repoId : `${repoId}@${revision}`,
  };
  if (method === 'dflash2') {
    spec.conv = {
      kernel: pick('conv_kernel_size').int('conv_kernel_size'),
      groupSize: pick('conv_group_size').int('conv_group_size'),
    };
    spec.selector = {
      topK: pick('selector_top_k').int('selector_top_k'),
      rank: pick('selector_rank').int('selector_rank'),
    };
    if (spec.conv.groupSize > 0 && hidden % spec.conv.groupSize !== 0)
      r.errors.push('conv_group_size: must divide hidden_size');
  } else {
    spec.markovRank = pick('markov_rank').int('markov_rank');
    const head = r.optStr('markov_head_type');
    if (head !== undefined && head !== 'vanilla')
      notes.push(`markov_head_type '${head}' costed as the vanilla W1·W2 head`);
    if (r.bool('enable_confidence_head', false))
      notes.push('confidence head not costed (< 0.1% of params; fixed k)');
  }
  if (spec.blockSize < 2)
    r.errors.push('block_size: the drafter must allow at least one draft token');
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };
  return { kind: 'ok', spec, notes };
}

/** A released block drafter for a target checkpoint. */
export interface KnownDrafter {
  /** Hugging Face repo id of the target model, e.g. `'Qwen/Qwen3.8-27B'`. */
  targetId: string;
  /** Hugging Face repo id of the drafter. */
  repoId: string;
  /** Drafter method. */
  method: BlockDraftSpec['method'];
}

/**
 * Released drafters for known targets (design doc §3.1). Sources: the drafter configs
 * https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json (64 target layers, h 5120: Qwen3.8-27B)
 * and https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/blob/main/config.json (36 target layers, h 4096:
 * Qwen3-8B).
 */
export const KNOWN_DRAFTERS: readonly KnownDrafter[] = [
  { targetId: 'Qwen/Qwen3.8-27B', repoId: 'incoai/Qwen3.8-27B-DFlash2', method: 'dflash2' },
  { targetId: 'Qwen/Qwen3-8B', repoId: 'deepseek-ai/dspark_qwen3_8b_block7', method: 'dspark' },
];

/** The known drafters of a target repo id (case-insensitive match). */
export function knownDrafters(targetId: string): KnownDrafter[] {
  const id = targetId.toLowerCase();
  return KNOWN_DRAFTERS.filter((d) => d.targetId.toLowerCase() === id);
}

/** Default dimensions of the hypothetical drafter (design doc §3.2): the released drafters' common shape. */
export const HYPOTHETICAL_DRAFTER_DEFAULTS = {
  /** Drafter layers. Count. */
  layers: 5,
  /** Query heads. Count. */
  heads: 32,
  /** KV heads. Count. */
  kvHeads: 8,
  /** Head dimension. Unit: elements. */
  headDim: 128,
  /** Sliding window. Unit: tokens. */
  windowTokens: 2048,
  /** Largest verify width k + 1 (DFlash 2 block 8; DSpark block 7 + 1). Count. */
  blockSize: 8,
  /** DFlash 2 conv taps and group size (`conv_kernel_size` 2, `conv_group_size` 16). */
  conv: { kernel: 2, groupSize: 16 },
  /** DFlash 2 selector candidates and rank (`selector_top_k` 16, `selector_rank` 256). */
  selector: { topK: 16, rank: 256 },
  /** DSpark Markov rank (`markov_rank` 256). Unit: elements. */
  markovRank: 256,
} as const;

/**
 * The target's FFN width for the hypothetical drafter: its first dense MLP's intermediate width, else its first
 * MoE's `topK · expertIntermediate` (the active routed width); 0 when it has neither (the caller then uses 4·h).
 * Unit: elements.
 */
function targetFfnWidth(layers: LayerSpec[]): number {
  const blocks = layers.flatMap((l) => l.blocks);
  const mlp = blocks.find((b) => b.kind === 'mlp');
  if (mlp !== undefined && mlp.kind === 'mlp') return mlp.intermediate;
  const moe = blocks.find((b) => b.kind === 'moe');
  if (moe !== undefined && moe.kind === 'moe') return moe.topK * moe.expertIntermediate;
  return 0;
}

/**
 * The labelled hypothetical block drafter for a target without a released one (design doc §3.2; `source`
 * `'hypothetical'`): {@link HYPOTHETICAL_DRAFTER_DEFAULTS} with the target's width h and vocabulary, FFN = the
 * target's dense FFN width (or `topK · fe`), target layers from DFlash's `build_target_layer_ids`.
 */
export function hypotheticalDrafter(
  target: ModelSpec,
  method: BlockDraftSpec['method'],
): BlockDraftSpec {
  const d = HYPOTHETICAL_DRAFTER_DEFAULTS;
  const width = targetFfnWidth(target.layers);
  return {
    kind: 'block_drafter',
    method,
    hidden: target.hidden,
    vocab: target.vocab,
    layers: d.layers,
    heads: d.heads,
    kvHeads: d.kvHeads,
    headDim: d.headDim,
    intermediate: width > 0 ? width : 4 * target.hidden,
    gated: true,
    activation: 'silu',
    windowTokens: d.windowTokens,
    targetLayerIds: buildTargetLayerIds(target.layers.length, d.layers),
    blockSize: d.blockSize,
    ...(method === 'dflash2'
      ? { conv: { ...d.conv }, selector: { ...d.selector } }
      : { markovRank: d.markovRank }),
    source: 'hypothetical',
  };
}
