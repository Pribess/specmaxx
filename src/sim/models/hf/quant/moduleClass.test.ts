import { describe, expect, it } from 'vitest';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { mapHfModel } from '../index';
import { isJsonObject } from '../reader';
import type { JsonObject } from '../types';
import { deriveQuantDefaults } from './index';
import { buildProbes, layerLayout } from './moduleClass';

/** The language-model config of a fixture repo (`text_config` or the flat config). */
function textOf(id: string): { top: JsonObject; text: JsonObject } {
  const top = fixtureRepoFiles(id).config as JsonObject;
  return { top, text: isJsonObject(top.text_config) ? top.text_config : top };
}

/** Attention kind per layer of the IR the family builds: `linear` for a linear-attention mixer, else `full`. */
function irAttentionKinds(id: string): ('full' | 'linear')[] {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r.model.layers.map((l) =>
    l.blocks.some((b) => b.kind === 'linear_attention') ? 'linear' : 'full',
  );
}

describe('layerLayout: agrees with the IR on every layer', () => {
  // Kimi lists its layers 1-based and ends in an MLA layer (93 only in `full_attn_layers`); GLM-5.3-Flash is 0-based.
  for (const id of ['moonshotai/Kimi-K3', 'moonshotai/Kimi-K3.1-Pro', 'zai-org/GLM-5.3-Flash']) {
    it(id, () => {
      const { text } = textOf(id);
      const ir = irAttentionKinds(id);
      expect(ir.filter((k) => k === 'linear').length).toBeGreaterThan(0);
      expect(layerLayout(text).attention).toEqual(ir);
    });
  }

  it('Kimi K3: KDA probes at layers 0–2, MLA probes at layer 3 (1-based lists, last layer MLA)', () => {
    const { top, text } = textOf('moonshotai/Kimi-K3');
    const { probes } = buildProbes(top, text, '');
    const leavesAt = (i: number) => probes.filter((p) => p.layer === i).map((p) => p.leaf);
    for (const i of [0, 1, 2]) {
      expect(leavesAt(i)).toContain('self_attn.f_a_proj');
      expect(leavesAt(i)).not.toContain('self_attn.kv_a_proj_with_mqa');
    }
    for (const i of [3, 91, 92]) {
      expect(leavesAt(i)).toContain('self_attn.kv_a_proj_with_mqa');
      expect(leavesAt(i)).not.toContain('self_attn.f_a_proj');
    }
  });
});

describe('layer-specific quant targets on Kimi K3', () => {
  it('an ignore naming layer 0 hits its KDA projections, not MLA ones', () => {
    const { top, text } = textOf('moonshotai/Kimi-K3');
    const qc = text.quantization_config as JsonObject;
    // Quantize attention too, except layer 0 (a KDA layer: 1-based `kda_layers` lists 1).
    const ignore = (qc.ignore as string[]).filter((p) => p !== 're:.*self_attn.*');
    const quantizedText: JsonObject = {
      ...text,
      quantization_config: { ...qc, ignore: [...ignore, 're:.*layers\\.0\\.self_attn\\..*'] },
    };
    const d = deriveQuantDefaults({
      top: { ...top, text_config: quantizedText },
      text: quantizedText,
    });
    const noteOf = (cls: string) => d.notes.filter((n) => n.startsWith(`${cls}:`)).join('\n');
    expect(noteOf('linear_attention')).toMatch(/in 68 of 69 layers; layer 0 is bf16/);
    expect(noteOf('attention')).not.toMatch(/layer 0/);
  });
});
