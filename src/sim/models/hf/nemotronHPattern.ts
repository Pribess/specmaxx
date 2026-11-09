/**
 * Layer roles of a Nemotron-H (`nemotron_h`) config, shared by the quant probes (`quant/moduleClass.ts`) and the
 * family builder.
 *
 * A Nemotron-H decoder layer holds exactly one sublayer (a Mamba-2 mixer, a MoE, a full-attention block or a
 * dense MLP), named by one entry of the layer-role list. Three spellings exist, all mapping to the same roles
 * (transformers `configuration_nemotron_h.py`, `_pattern_to_list` and `remap_legacy_layer_types`):
 * - `layers_block_type` (alias `layer_types`): a list of `'mamba' | 'moe' | 'attention' | 'mlp'` (the names in
 *   the published configs), or transformers' current `'linear_attention' | 'moe' | 'full_attention' | 'mlp'`;
 * - `hybrid_override_pattern`: a legacy string, one character per layer: `M` mamba, `E` moe, `*` attention,
 *   `-` mlp.
 * MTP layers have their own list (`mtp_layers_block_type` / `mtp_hybrid_override_pattern`); they are not in the
 * IR, so the family reads them only to estimate their parameters.
 */

import type { JsonObject } from './types';

/** One Nemotron-H layer role: Mamba-2 mixer, MoE, full attention, or dense MLP. */
export type HybridBlockType = 'mamba' | 'moe' | 'attention' | 'mlp';

/**
 * List entry → role. Legacy names first, then transformers' `layer_types` names. A Map, not an object literal,
 * so inherited keys ('constructor', '__proto__', 'toString') are not roles.
 */
const BLOCK_NAMES: ReadonlyMap<string, HybridBlockType> = new Map<string, HybridBlockType>([
  ['mamba', 'mamba'],
  ['moe', 'moe'],
  ['attention', 'attention'],
  ['mlp', 'mlp'],
  ['linear_attention', 'mamba'],
  ['full_attention', 'attention'],
]);

/** `hybrid_override_pattern` character → role (a Map for the same reason). */
const PATTERN_CHARS: ReadonlyMap<string, HybridBlockType> = new Map<string, HybridBlockType>([
  ['M', 'mamba'],
  ['E', 'moe'],
  ['*', 'attention'],
  ['-', 'mlp'],
]);

/**
 * Layer roles from `layers_block_type` (or its alias `layer_types`) or the legacy `hybrid_override_pattern`
 * string (M mamba, E moe, * attention, - mlp). Accepts the legacy names 'mamba'/'attention' and transformers'
 * 'linear_attention'/'full_attention'. undefined when none is present; { error } on an unknown entry.
 *
 * `key`: `'layers'` (default) reads the decoder stack; `'mtp'` reads `mtp_layers_block_type` /
 * `mtp_hybrid_override_pattern`. The list wins over the pattern when both are present (as in transformers).
 * One entry per layer; the list's length is the layer count.
 */
export function nemotronBlockTypes(
  cfg: JsonObject,
  key: 'layers' | 'mtp' = 'layers',
): HybridBlockType[] | { error: string } | undefined {
  const listKeys = key === 'mtp' ? ['mtp_layers_block_type'] : ['layers_block_type', 'layer_types'];
  const patternKey = key === 'mtp' ? 'mtp_hybrid_override_pattern' : 'hybrid_override_pattern';

  for (const k of listKeys) {
    const v = cfg[k];
    if (v === undefined || v === null) continue;
    if (!Array.isArray(v)) return { error: `${k}: expected a list of layer types` };
    const out: HybridBlockType[] = [];
    for (const [i, entry] of v.entries()) {
      const role = typeof entry === 'string' ? BLOCK_NAMES.get(entry) : undefined;
      if (role === undefined)
        return { error: `${k}[${i}]: unknown layer type ${JSON.stringify(entry)}` };
      out.push(role);
    }
    return out;
  }

  const pattern = cfg[patternKey];
  if (pattern === undefined || pattern === null) return undefined;
  if (typeof pattern !== 'string') return { error: `${patternKey}: expected a string` };
  const out: HybridBlockType[] = [];
  for (const [i, ch] of [...pattern].entries()) {
    const role = PATTERN_CHARS.get(ch);
    if (role === undefined)
      return { error: `${patternKey}[${i}]: unknown layer character '${ch}'` };
    out.push(role);
  }
  return out;
}
