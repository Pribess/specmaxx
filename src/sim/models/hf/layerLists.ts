/**
 * Layer-index lists in configs (`linear_attn_config.kda_layers` / `full_attn_layers`) come 0-based in some
 * families (GLM-5.3-Flash) and 1-based in others (Kimi K3, `configuration_kimi_k3.py: (layer_idx + 1) in kda_layers`).
 * One rule decides which, shared by the family parser (`families/kimiLinear.ts`) and the quant probes
 * (`quant/moduleClass.ts`) so their layer kinds always agree.
 */

/**
 * The offset to subtract from listed layer indices to get 0-based ones: 1 when the lists are 1-based, else 0.
 *
 * Rule: a 1-based set of lists names the last layer `numLayers` (and never 0); a 0-based set never names
 * `numLayers`. Check the UNION of every list given, since the last layer may be in either list (both Kimi
 * checkpoints end in an MLA layer, so 93 is only in `full_attn_layers`).
 *
 * Formula: offset = (⋃ lists) ∋ numLayers ? 1 : 0.
 *
 * @param lists Every layer-index list of the config that shares one numbering (e.g. `kda_layers` and
 *   `full_attn_layers`); missing lists may be passed as `undefined`.
 * @param numLayers Number of decoder layers (`num_hidden_layers`).
 * @returns 1 for 1-based lists, 0 for 0-based ones. Unit: layer indices.
 */
export function layerListOffset(
  lists: readonly (readonly number[] | undefined)[],
  numLayers: number,
): 0 | 1 {
  return lists.some((l) => l?.includes(numLayers)) ? 1 : 0;
}
