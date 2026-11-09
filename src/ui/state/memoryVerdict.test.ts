import { expect, it } from 'vitest';
import type { TierMemory } from '../../sim/engine/types';
import { memoryFits, memoryVerdict } from './memoryVerdict';
import type { StageMemory } from './summary';

const stage = (used: number): StageMemory => ({
  stage: 0,
  weightBytes: used * 100,
  sequenceStateBytes: 0,
  activationPeakBytes: 0,
  totalBytes: used * 100,
  capacityBytes: 100,
  fits: used <= 1,
});
const tier = (fits: boolean): TierMemory => ({
  tier: 'nvme',
  bytes: fits ? 1 : 3,
  breakdown: { lookupTables: fits ? 1 : 3, routedExperts: 0, draftExperts: 0, tokenEmbeddings: 0 },
  capacityBytes: 2,
  fits,
});

it('a stage that fits with a tier that does not is a danger, naming the tier', () => {
  const v = memoryVerdict({ stages: [stage(0.5)], tiers: [tier(false)] });
  expect(v.fits).toBe(false);
  expect(v.state).toBe('danger');
  expect(v.failingTiers.map((t) => t.tier)).toEqual(['nvme']);
  expect(memoryFits({ stages: [stage(0.5)], tiers: [tier(false)] })).toBe(false);
});

it('everything fitting is fits; no tiers is fine', () => {
  expect(memoryVerdict({ stages: [stage(0.5)], tiers: [tier(true)] })).toMatchObject({
    fits: true,
    state: 'fits',
    failingTiers: [],
  });
  expect(memoryVerdict({ stages: [stage(0.5)] }).fits).toBe(true);
});

it('keeps the warning band and the failing stage', () => {
  expect(memoryVerdict({ stages: [stage(0.95)] })).toMatchObject({ fits: true, state: 'warning' });
  const v = memoryVerdict({ stages: [stage(1.2)] });
  expect(v.state).toBe('danger');
  expect(v.worstStage?.fits).toBe(false);
});
