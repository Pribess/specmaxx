import { describe, expect, it } from 'vitest';
import { createReader } from './reader';

describe('createReader', () => {
  const cfg = {
    hidden_size: 4096,
    rope_theta: 1e6,
    tie_word_embeddings: false,
    model_type: 'x',
    layer_types: ['a', 'b'],
    ids: [1, 2, 3],
    nullish: null,
    bad_num: '12',
    frac: 1.5,
    sub: { n: 3 },
  };

  it('reads well-typed values without errors', () => {
    const r = createReader(cfg, 'text_config');
    expect(r.num('hidden_size')).toBe(4096);
    expect(r.int('hidden_size')).toBe(4096);
    expect(r.num('rope_theta')).toBe(1e6);
    expect(r.bool('tie_word_embeddings')).toBe(false);
    expect(r.str('model_type')).toBe('x');
    expect(r.strArray('layer_types')).toEqual(['a', 'b']);
    expect(r.numArray('ids')).toEqual([1, 2, 3]);
    expect(r.obj('sub').int('n')).toBe(3);
    expect(r.optNum('missing')).toBeUndefined();
    expect(r.optNum('nullish')).toBeUndefined();
    expect(r.optObj('missing')).toBeUndefined();
    expect(r.bool('missing', true)).toBe(true);
    expect(r.has('hidden_size')).toBe(true);
    expect(r.has('nullish')).toBe(false);
    expect(r.errors).toEqual([]);
  });

  it('collects path-qualified errors and returns placeholders instead of throwing', () => {
    const r = createReader(cfg, 'text_config');
    expect(r.num('num_hidden_layers')).toBe(0);
    expect(r.num('bad_num')).toBe(0);
    expect(r.int('frac')).toBe(0);
    expect(r.optNum('bad_num')).toBeUndefined();
    expect(r.str('hidden_size')).toBe('');
    expect(r.numArray('layer_types')).toEqual([]);
    expect(r.bool('missing')).toBe(false);
    const sub = r.obj('missing_obj');
    expect(sub.num('x')).toBe(0);
    expect(r.errors).toEqual([
      'text_config.num_hidden_layers: missing number',
      'text_config.bad_num: expected number',
      'text_config.frac: expected non-negative integer',
      'text_config.bad_num: expected number',
      'text_config.hidden_size: expected string',
      'text_config.layer_types: expected number array',
      'text_config.missing: missing boolean',
      'text_config.missing_obj: missing object',
      'text_config.missing_obj.x: missing number',
    ]);
  });

  it('shares the error list with sub-readers and names root paths without a prefix', () => {
    const r = createReader({ a: { b: 'no' } });
    r.obj('a').num('b');
    expect(r.errors).toEqual(['a.b: expected number']);
  });

  it('reports a non-object root', () => {
    expect(createReader(42).errors).toEqual(['(root): missing object']);
  });
});
