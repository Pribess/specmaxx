import { expect, it } from 'vitest';
import { sameModel } from './sameModel';
it('compares mapped plain data structurally, including optional fields and array order', () => {
  expect(sameModel({ a: [1, { x: 2 }], b: undefined }, { b: undefined, a: [1, { x: 2 }] })).toBe(
    true,
  );
  expect(sameModel({ a: [1, 2] }, { a: [2, 1] })).toBe(false);
  expect(sameModel({ a: undefined }, {})).toBe(false);
  expect(sameModel([], {})).toBe(false);
  expect(sameModel(null, {})).toBe(false);
});
