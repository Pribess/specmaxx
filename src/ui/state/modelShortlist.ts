/**
 * The model quick-select list of the model picker (the user's shortlist) and the default model. Any other
 * Hugging Face repo id can still be typed or pasted.
 */

/** Hugging Face repo id of the default model; its config is bundled (`bundledModel.ts`) for an offline first paint. */
export const DEFAULT_MODEL_ID = 'Qwen/Qwen3.8-27B';

/** One shortlist entry. */
export interface ShortlistModel {
  /** Hugging Face repo id `owner/name`. */
  id: string;
  /** Display name. */
  name: string;
}

/**
 * The user's shortlist, in their order. Repo ids are the ones the loader's fixtures were taken from
 * (`src/sim/models/__fixtures__/`).
 */
export const MODEL_SHORTLIST: ShortlistModel[] = [
  { id: 'Qwen/Qwen3.8-Flash-Next', name: 'Qwen3.8-Flash-Next' },
  { id: 'Qwen/Qwen3.8-27B', name: 'Qwen3.8-27B' },
  { id: 'deepseek-ai/DeepSeek-V4.1-Flash', name: 'DeepSeek-V4.1-Flash' },
  { id: 'deepseek-ai/DeepSeek-V4-Flash-0731', name: 'DeepSeek-V4-Flash-0731' },
  { id: 'zai-org/GLM-5.3-Flash', name: 'GLM-5.3 Flash' },
  { id: 'zai-org/GLM-5.3', name: 'GLM-5.3' },
  { id: 'mistralai/Mistral-Large-4-1T-A52B', name: 'Mistral Large 4' },
  // "Inkling 2 Small" is left out until its weights are on HF: its model card gives 594B total / 17B active
  // (48 layers, 8 of 384 experts), while `thinkingmachines/Inkling-Small` is a different, 276B / 12B model.
  // Add it here once the matching repo is published.
  { id: 'MiniMaxAI/MiniMax-M3', name: 'Minimax M3' },
];
