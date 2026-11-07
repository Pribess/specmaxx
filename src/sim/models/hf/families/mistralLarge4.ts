/**
 * Family `mistral_large4`: Mistral Large 4 (`MistralLarge4ForConditionalGeneration`; language model under
 * `text_config`, Pixtral vision tower beside it).
 *
 * Every layer: pre-norm MLA without q compression (no `q_lora_rank` key → `qLoraRank` 0) with DSA top-k sparse
 * attention, then a dense MLP (`first_k_dense_replace` 1, `intermediate_size`) or a MoE with one shared expert
 * of width `moe_intermediate_size · n_shared_experts`. Per-layer variations, all from config lists
 * (`configuration_mistral_large4.py:153-218`, `modeling_mistral_large4.py:335-373`):
 * - `no_rope_layers[i] == 0` → NoPE layer: `qkNopeDim = qk_nope_head_dim + qk_rope_head_dim`, `qkRopeDim` 0
 *   (absent list: NoPE iff `(i + 1) % no_rope_layer_interval == 0`, interval default 4).
 * - `indexer_types[i] == 'shared'` → no indexer, reuse the previous full layer's top-k.
 * - `kv_source_layers[i] != i` → the layer reads layer `kv_source_layers[i]`'s latent cache: no `kv_a` and no
 *   cache of its own, but its own `kv_b` (and its own indexer when `indexer_types[i]` is 'full').
 *
 * Excluded: the DSpark MTP draft (described only in `params.json` `mtp`; estimated) and the vision tower
 * (generic estimate). Approximations: the indexer's `q_input_norm` LayerNorm (2·h) and the ssmax per-head
 * scalars (2·H, 2·H_I) are not counted (12,480 params per indexer layer, 128 per shared layer); the logit
 * softcap, LeRoPE and yarn are elementwise.
 */

import type { FfnParams } from '../../../ir/builders';
import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import { estimateVisionParams } from '../paramCheck';
import { createReader, isJsonObject } from '../reader';
import type { FamilyBuildResult, FamilyEntry, HfRepoFiles, JsonObject } from '../types';
import {
  dsaSelection,
  layerParams,
  readActivation,
  readDenseFfn,
  readDsaIndexer,
  readIndexerSchedule,
  readMlaMixer,
  readMlpSchedule,
  readMoeFfn,
  structuralErrors,
} from './mlaCommon';

/**
 * Params of the DSpark MTP draft from `params.json` `mtp` (absent from config.json), for the param check only:
 * `n_layers` dense MLA layers (RoPE, no indexer, dense FFN of `intermediate_size`), the target projection
 * (`target_activation_layers`·h → h) and the autoregressive head's embedding and output (2·vocab·`markov_rank`).
 * Its small joint projection, norms and mask embedding (≈ 11 M) are left out. Unit: params.
 */
function dsparkParamsEstimate(params: unknown, mtpLayerParams: number, hidden: number, vocab: number): number {
  if (!isJsonObject(params) || !isJsonObject(params.mtp)) return 0;
  const mtp = params.mtp;
  const layers = typeof mtp.n_layers === 'number' ? mtp.n_layers : 0;
  const targets = typeof mtp.target_activation_layers === 'string' ? mtp.target_activation_layers.split(',').length : 0;
  const dspark = isJsonObject(mtp.dspark) ? mtp.dspark : {};
  const rank = typeof dspark.markov_rank === 'number' ? dspark.markov_rank : 0;
  // Formula: P_dspark = n_layers·P_layer + targets·h·h + 2·vocab·markov_rank
  return layers * mtpLayerParams + targets * hidden * hidden + 2 * vocab * rank;
}

/** Build the Mistral Large 4 IR from its text config (and `params.json` for the MTP estimate). */
export function buildMistralLarge4(text: JsonObject, top: JsonObject, files: Pick<HfRepoFiles, 'id' | 'paramsJson'>): FamilyBuildResult {
  const r = createReader(text, text === top ? '' : 'text_config');
  const at = (key: string) => (r.path ? `${r.path}.${key}` : key);
  const hidden = r.int('hidden_size');
  const n = r.int('num_hidden_layers');
  const vocab = r.int('vocab_size');
  const tie = r.bool('tie_word_embeddings', false);
  const act = readActivation(r);
  const ropeMixer = readMlaMixer(r, true);
  const nopeMixer = readMlaMixer(r, false);
  const indexer = readDsaIndexer(r);
  const indexerModes = readIndexerSchedule(r, n);
  const mlpTypes = readMlpSchedule(r, n, 1);
  const moe = readMoeFfn(r, act);
  const dense: FfnParams = readDenseFfn(r, act);

  const interval = r.optInt('no_rope_layer_interval') ?? 4;
  const noRope = r.optNumArray('no_rope_layers') ?? Array.from({ length: n }, (_, i) => ((i + 1) % interval !== 0 ? 1 : 0));
  if (noRope.length < n) r.errors.push(`${at('no_rope_layers')}: ${noRope.length} entries for ${n} layers`);
  const kvSources = r.optNumArray('kv_source_layers') ?? Array.from({ length: n }, (_, i) => i);
  if (kvSources.length < n) r.errors.push(`${at('kv_source_layers')}: ${kvSources.length} entries for ${n} layers`);
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

  const usesRope = (i: number) => noRope[i] !== 0;
  const layers = repeatLayers(n, (i) => {
    const src = kvSources[i];
    return decoderLayer({
      name: `layer.${i}`,
      hidden,
      mixer: {
        ...(usesRope(i) ? ropeMixer : nopeMixer),
        sparse: dsaSelection(indexerModes[i], indexer),
        ...(src !== i ? { kvCacheFromLayer: src } : {}),
      },
      ffn: mlpTypes[i] === 'dense' ? dense : moe,
    });
  });
  const model = decoderModel({ name: files.id, hidden, vocab, tieEmbeddings: tie, layers, source: files.id });
  const errors = structuralErrors(model);
  if (errors.length > 0) return { kind: 'invalid_config', errors };

  const mtpLayer = decoderLayer({ hidden, mixer: ropeMixer, ffn: dense });
  const mtp = dsparkParamsEstimate(files.paramsJson, layerParams(mtpLayer), hidden, vocab);
  const vision = isJsonObject(top.vision_config) ? top.vision_config : undefined;
  const visionParams = estimateVisionParams(vision, hidden);

  const nope = layers.filter((_, i) => !usesRope(i)).length;
  const shared = indexerModes.filter((m) => m === 'shared').length;
  const kvConsumers = kvSources.slice(0, n).filter((s, i) => s !== i).length;
  const notes = [
    `MLA without q-LoRA + DSA: ${nope} NoPE layers, ${shared} layers reuse the previous top-k, ${kvConsumers} layers read an earlier layer's latent cache`,
    'indexer q_input_norm and ssmax scalars not counted (12,480 params per indexer layer)',
  ];
  const excludedParts: string[] = [];
  if (mtp > 0) excludedParts.push('DSpark MTP draft (params.json)');
  if (visionParams > 0) excludedParts.push('vision tower');
  return { kind: 'ok', model, notes, mtpParamsEstimate: mtp, visionParamsEstimate: visionParams, excludedParts };
}

/** Family entry for Mistral Large 4 (MLA + DSA with shared indexers and shared KV caches + MoE). */
export const mistralLarge4Family: FamilyEntry = {
  family: 'mistral_large4',
  modelTypes: ['mistral_large4', 'mistral_large4_text'],
  architectures: ['MistralLarge4ForConditionalGeneration'],
  build: (text, top, files) => buildMistralLarge4(text, top, files),
};
