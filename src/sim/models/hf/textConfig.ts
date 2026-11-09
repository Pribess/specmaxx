/**
 * Split a config.json into its language-model part and its multimodal wrapper.
 *
 * Multimodal checkpoints nest the language model under `text_config` (NVIDIA's Nemotron-H omni wrappers:
 * `llm_config`) and keep `architectures`, `model_type`, `vision_config` (and sometimes `quantization_config`)
 * at the top level. Text-only checkpoints are flat. The simulator models the language model only, so vision and
 * audio towers are excluded (their parameters are estimated separately for the param check).
 *
 * Quirks of the shortlist this handles:
 * - MiniMax-M3 and Inkling text configs have no `model_type`; dispatch then uses the top-level one.
 * - Qwen4Exp's `vision_config.model_type` equals the top-level `model_type` (`'qwen4_exp'`), so the vision
 *   type must never be used for dispatch.
 */

import { isJsonObject } from './reader';
import type { HfConfigView, JsonObject } from './types';

/** Result of {@link unwrapTextConfig}: the view, or why the config cannot be read at all. */
export type UnwrapResult = { kind: 'ok'; view: HfConfigView } | { kind: 'invalid_config'; errors: string[] };

/** A string field of `o`, or null. */
function strOrNull(o: JsonObject, key: string): string | null {
  const v = o[key];
  return typeof v === 'string' ? v : null;
}

/** The string entries of an array field of `o` (empty when absent or not an array). */
function strList(o: JsonObject, key: string): string[] {
  const v = o[key];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

/**
 * Unwrap `text_config`, else `llm_config` (when present), and record the wrapper's identity and quantization
 * metadata.
 */
export function unwrapTextConfig(config: unknown): UnwrapResult {
  if (!isJsonObject(config)) return { kind: 'invalid_config', errors: ['config.json: expected a JSON object'] };
  const top = config;
  const notes: string[] = [];

  const textKey = top.text_config !== undefined && top.text_config !== null ? 'text_config' : 'llm_config';
  const rawText = top[textKey];
  if (rawText !== undefined && rawText !== null && !isJsonObject(rawText)) {
    return { kind: 'invalid_config', errors: [`${textKey}: expected object`] };
  }
  const nested = isJsonObject(rawText);
  const text: JsonObject = nested ? rawText : top;

  const hasTower = ['vision_config', 'audio_config'].some((k) => isJsonObject(top[k]) || isJsonObject(text[k]));
  if (nested || hasTower) notes.push('language model only; vision/audio towers excluded');

  const quantizationConfig = top.quantization_config ?? (nested ? text.quantization_config : undefined);
  const visionRaw = isJsonObject(top.vision_config) ? top.vision_config : text.vision_config;
  const visionConfig = isJsonObject(visionRaw) ? visionRaw : undefined;

  return {
    kind: 'ok',
    view: {
      top,
      text,
      nested,
      topModelType: strOrNull(top, 'model_type'),
      topArchitectures: strList(top, 'architectures'),
      textModelType: nested ? strOrNull(text, 'model_type') : null,
      textArchitectures: nested ? strList(text, 'architectures') : [],
      quantizationConfig: quantizationConfig ?? undefined,
      visionConfig,
      notes,
    },
  };
}
