import { describe, expect, it } from 'vitest';
import type { DeviceModifier, LinkModifier } from './modifiers';
import { allUnlockSelections, resolveUnlocks, unlockAxesErrors } from './unlocks';
import type { UnlockAxis, UnlockOption } from './unlocks';

/** A distinguishable device modifier (the note names it, so order is visible in the result). */
function mod(note: string): DeviceModifier {
  return { kind: 'clockScale', factor: 0.9, note, sources: [], estimatedReason: 'test' };
}

const P2P: LinkModifier = {
  kind: 'p2p',
  p2p: 'direct',
  note: 'p2p',
  sources: [],
  estimatedReason: 'test',
};

function option(
  id: string,
  deviceModifiers: DeviceModifier[],
  intraNodeLinkModifiers?: LinkModifier[],
): UnlockOption {
  return {
    id,
    name: id,
    deviceModifiers,
    ...(intraNodeLinkModifiers !== undefined ? { intraNodeLinkModifiers } : {}),
  };
}

function axis(id: string, options: UnlockOption[]): UnlockAxis {
  return { id, name: id, kind: 'other', description: id, options };
}

/** Three axes: a checkbox (1 option), a radio (2 options), and a link-only checkbox. */
const AXES: UnlockAxis[] = [
  axis('a', [option('on', [mod('a1'), mod('a2')])]),
  axis('b', [option('x', [mod('bx')]), option('y', [mod('by')])]),
  axis('c', [option('p2p', [], [P2P])]),
];

describe('resolveUnlocks', () => {
  it('concatenates the chosen options in AXIS order, not selection-key order', () => {
    const r = resolveUnlocks(AXES, { c: 'p2p', b: 'y', a: 'on' });
    expect(r.deviceModifiers.map((m) => m.note)).toEqual(['a1', 'a2', 'by']);
    expect(r.intraNodeLinkModifiers).toEqual([P2P]);
  });

  it('the empty selection is stock: no modifiers', () => {
    expect(resolveUnlocks(AXES, {})).toEqual({ deviceModifiers: [], intraNodeLinkModifiers: [] });
  });

  it('throws on an unknown axis id or option id', () => {
    expect(() => resolveUnlocks(AXES, { z: 'on' })).toThrow("unknown unlock axis 'z'");
    expect(() => resolveUnlocks(AXES, { b: 'w' })).toThrow("unknown option 'w' on unlock axis 'b'");
  });
});

describe('allUnlockSelections', () => {
  it('yields ∏ (options + 1) distinct selections, stock included', () => {
    const s = allUnlockSelections(AXES);
    expect(s.length).toBe((1 + 1) * (2 + 1) * (1 + 1)); // 12
    expect(new Set(s.map((x) => JSON.stringify(Object.entries(x).sort()))).size).toBe(s.length);
    expect(s).toContainEqual({});
    expect(s).toContainEqual({ a: 'on', b: 'y', c: 'p2p' });
    // Every selection resolves (no unknown ids).
    for (const x of s) expect(() => resolveUnlocks(AXES, x)).not.toThrow();
  });

  it('no axes: one (stock) selection', () => {
    expect(allUnlockSelections([])).toEqual([{}]);
  });
});

describe('unlockAxesErrors', () => {
  it('well-formed axes: no errors', () => {
    expect(unlockAxesErrors(AXES)).toEqual([]);
  });

  it('duplicate axis id, empty options, duplicate option id, option without modifiers', () => {
    const errors = unlockAxesErrors([
      axis('a', [option('on', [mod('a')])]),
      axis('a', []),
      axis('b', [option('x', [mod('x')]), option('x', [mod('x2')])]),
      axis('c', [option('noop', [], [])]),
    ]);
    expect(errors).toEqual([
      "duplicate unlock axis id 'a'",
      "unlock axis 'a' has no options",
      "unlock axis 'b': duplicate option id 'x'",
      "unlock axis 'c' option 'noop' has no modifiers",
    ]);
  });
});
