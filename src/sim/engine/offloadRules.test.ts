import { describe, expect, it } from 'vitest';
import { expertRules, lookupTableRules } from './offloadRules';
import type { LookupTableRule, RoutedExpertRule } from './types';

const ram: RoutedExpertRule = {
  tier: 'host_ram',
  layers: { from: 0, to: 24 },
  residentFraction: 0.1,
  hitRatio: { kind: 'uniform' },
};
const ssd: RoutedExpertRule = { ...ram, tier: 'nvme', layers: { from: 24, to: 48 } };
const table: LookupTableRule = { tier: 'nvme', layers: { from: 1, to: 2 } };

describe('expertRules', () => {
  it('absent config or field → []', () => {
    expect(expertRules(undefined)).toEqual([]);
    expect(expertRules({})).toEqual([]);
  });
  it('one object → a one-rule list holding it', () => {
    const rules = expertRules({ routedExperts: ram });
    expect(rules).toHaveLength(1);
    expect(rules[0]).toBe(ram);
  });
  it('a list passes through unchanged and unsorted', () => {
    const list = [ssd, ram];
    expect(expertRules({ routedExperts: list })).toBe(list);
  });
  it('[] → []', () => {
    expect(expertRules({ routedExperts: [] })).toEqual([]);
  });
});

describe('lookupTableRules', () => {
  it('absent config or field → []', () => {
    expect(lookupTableRules(undefined)).toEqual([]);
    expect(lookupTableRules({ routedExperts: ram })).toEqual([]);
  });
  it('one object → a one-rule list holding it (a device rule too)', () => {
    const rules = lookupTableRules({ lookupTables: table });
    expect(rules).toHaveLength(1);
    expect(rules[0]).toBe(table);
    expect(lookupTableRules({ lookupTables: { tier: 'device' } })).toEqual([{ tier: 'device' }]);
  });
  it('a list passes through; [] → []', () => {
    const list = [table, { tier: 'host_ram' as const, layers: { from: 14, to: 15 } }];
    expect(lookupTableRules({ lookupTables: list })).toBe(list);
    expect(lookupTableRules({ lookupTables: [] })).toEqual([]);
  });
});
