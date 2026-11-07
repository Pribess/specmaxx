/**
 * Family `glm_moe_dsa`: GLM-5 / GLM-5.3 (`GlmMoeDsaForCausalLM`, a DeepSeek-V3.2 derivative; flat config).
 *
 * Every layer: pre-norm MLA (q-LoRA, RoPE) with DSA top-k sparse attention, then a dense MLP (the first
 * `first_k_dense_replace` / `mlp_layer_types` 'dense' layers) or a MoE with one shared expert. Layers whose
 * `indexer_types[i]` is 'shared' run no indexer and reuse the top-k of the last 'full' layer
 * (`modular_glm_moe_dsa.py:199-257`); GLM-5.3: 21 full, 57 shared, derived from `index_topk_freq` 4 and
 * `index_skip_topk_offset` 3 when `indexer_types` is absent.
 *
 * Excluded: the `num_nextn_predict_layers` MTP layer (estimated for the param check as one full-indexer MoE
 * layer + `eh_proj` + 3 norms). Approximations: the router bias (E params per MoE layer) is not counted.
 */

import { decoderLayer, decoderModel, repeatLayers } from '../../../ir/builders';
import type { FamilyEntry, FamilyBuildResult, JsonObject } from '../types';
import { createReader } from '../reader';
import {
  dsaSelection,
  mtpParamsEstimate,
  readActivation,
  readDenseFfn,
  readDsaIndexer,
  readIndexerSchedule,
  readMlaMixer,
  readMlpSchedule,
  readMoeFfn,
  structuralErrors,
} from './mlaCommon';

/** Build the GLM-5.3 IR from its (flat) config. */
export function buildGlmMoeDsa(text: JsonObject, top: JsonObject, repoId: string): FamilyBuildResult {
  const r = createReader(text, text === top ? '' : 'text_config');
  const hidden = r.int('hidden_size');
  const n = r.int('num_hidden_layers');
  const vocab = r.int('vocab_size');
  const tie = r.bool('tie_word_embeddings', false);
  const act = readActivation(r);
  const mixer = readMlaMixer(r);
  const indexer = readDsaIndexer(r);
  const indexerModes = readIndexerSchedule(r, n);
  const mlpTypes = readMlpSchedule(r, n, 3);
  const moe = readMoeFfn(r, act);
  const dense = readDenseFfn(r, act);
  const mtpLayers = r.optInt('num_nextn_predict_layers') ?? 0;
  if (r.errors.length > 0) return { kind: 'invalid_config', errors: r.errors };

  const layers = repeatLayers(n, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden,
      mixer: { ...mixer, sparse: dsaSelection(indexerModes[i], indexer) },
      ffn: mlpTypes[i] === 'dense' ? dense : moe,
    }),
  );
  const model = decoderModel({ name: repoId, hidden, vocab, tieEmbeddings: tie, layers, source: repoId });
  const errors = structuralErrors(model);
  if (errors.length > 0) return { kind: 'invalid_config', errors };

  // The MTP layer is a full-indexer MoE layer (checkpoint `layers.<n>.*`, with its own indexer weights).
  const mtpLayer = decoderLayer({ hidden, mixer: { ...mixer, sparse: dsaSelection('full', indexer) }, ffn: moe });
  const full = indexerModes.filter((m) => m === 'full').length;
  const notes = [
    `MLA + DSA: ${full} layers run an indexer, ${n - full} reuse the previous full layer's top-k`,
    `${mlpTypes.filter((t) => t === 'dense').length} dense FFN layers, ${mlpTypes.filter((t) => t === 'sparse').length} MoE layers`,
    'router bias (e_score_correction_bias) not counted (E params per MoE layer)',
  ];
  if (mtpLayers > 0) notes.push(`${mtpLayers} MTP layer(s) excluded from the IR`);
  return {
    kind: 'ok',
    model,
    notes,
    mtpParamsEstimate: mtpParamsEstimate(mtpLayer, hidden, mtpLayers),
    visionParamsEstimate: 0,
    excludedParts: mtpLayers > 0 ? ['MTP layer'] : [],
  };
}

/** Family entry for GLM-5 / GLM-5.3 (MLA + DSA with shared indexers + MoE). */
export const glmMoeDsaFamily: FamilyEntry = {
  family: 'glm_moe_dsa',
  modelTypes: ['glm_moe_dsa'],
  architectures: ['GlmMoeDsaForCausalLM'],
  build: (text, top, files) => buildGlmMoeDsa(text, top, files.id),
};
