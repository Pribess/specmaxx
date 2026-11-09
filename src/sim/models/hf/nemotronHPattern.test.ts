import { describe, expect, it } from 'vitest';
import { fixtureRepoFiles } from '../__fixtures__/fixtures.testutil';
import { nemotronBlockTypes } from './nemotronHPattern';
import { unwrapTextConfig } from './textConfig';
import type { JsonObject } from './types';

/** Language-model config of a fixture. */
function textOf(id: string): JsonObject {
  const r = unwrapTextConfig(fixtureRepoFiles(id).config);
  if (r.kind !== 'ok') throw new Error(r.errors.join('; '));
  return r.view.text;
}

describe('nemotronBlockTypes', () => {
  it('gives the same roles for the list, its alias, the new names and the pattern string', () => {
    const want = ['mamba', 'moe', 'attention', 'mlp'];
    expect(nemotronBlockTypes({ layers_block_type: ['mamba', 'moe', 'attention', 'mlp'] })).toEqual(
      want,
    );
    expect(nemotronBlockTypes({ layer_types: ['mamba', 'moe', 'attention', 'mlp'] })).toEqual(want);
    expect(
      nemotronBlockTypes({
        layers_block_type: ['linear_attention', 'moe', 'full_attention', 'mlp'],
      }),
    ).toEqual(want);
    expect(nemotronBlockTypes({ hybrid_override_pattern: 'ME*-' })).toEqual(want);
  });

  it('prefers the list over the pattern, and reads the MTP keys with key mtp', () => {
    expect(
      nemotronBlockTypes({ layers_block_type: ['moe'], hybrid_override_pattern: 'M' }),
    ).toEqual(['moe']);
    expect(nemotronBlockTypes({ mtp_layers_block_type: ['attention', 'moe'] }, 'mtp')).toEqual([
      'attention',
      'moe',
    ]);
    expect(nemotronBlockTypes({ mtp_hybrid_override_pattern: '*E' }, 'mtp')).toEqual([
      'attention',
      'moe',
    ]);
    expect(nemotronBlockTypes({ layers_block_type: ['moe'] }, 'mtp')).toBeUndefined();
    expect(nemotronBlockTypes({})).toBeUndefined();
  });

  it('reports unknown entries as errors', () => {
    expect(nemotronBlockTypes({ layers_block_type: ['mamba', 'conv'] })).toEqual({
      error: 'layers_block_type[1]: unknown layer type "conv"',
    });
    expect(nemotronBlockTypes({ hybrid_override_pattern: 'MX' })).toEqual({
      error: "hybrid_override_pattern[1]: unknown layer character 'X'",
    });
    expect(nemotronBlockTypes({ layers_block_type: 'ME' })).toEqual({
      error: 'layers_block_type: expected a list of layer types',
    });
  });

  it('rejects inherited object keys as layer types', () => {
    for (const name of ['constructor', '__proto__', 'toString']) {
      expect(nemotronBlockTypes({ layers_block_type: [name] })).toEqual({
        error: `layers_block_type[0]: unknown layer type "${name}"`,
      });
      expect(nemotronBlockTypes({ mtp_layers_block_type: [name] }, 'mtp')).toEqual({
        error: `mtp_layers_block_type[0]: unknown layer type "${name}"`,
      });
    }
  });

  it('reads the shortlist configs', () => {
    const lightning = nemotronBlockTypes(
      textOf('nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'),
    );
    const zuper = nemotronBlockTypes(textOf('nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'));
    if (!Array.isArray(lightning) || !Array.isArray(zuper)) throw new Error('expected lists');
    const count = (l: string[], t: string) => l.filter((x) => x === t).length;
    expect([
      lightning.length,
      count(lightning, 'mamba'),
      count(lightning, 'moe'),
      count(lightning, 'attention'),
    ]).toEqual([52, 23, 23, 6]);
    expect([
      zuper.length,
      count(zuper, 'mamba'),
      count(zuper, 'moe'),
      count(zuper, 'attention'),
    ]).toEqual([88, 40, 40, 8]);
    expect(
      nemotronBlockTypes(textOf('nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'), 'mtp'),
    ).toEqual(['attention', 'moe']);
  });
});
