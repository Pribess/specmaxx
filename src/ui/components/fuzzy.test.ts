import { describe, expect, it } from 'vitest';
import { fuzzyFilter, groupOptions, stepIndex, type PickerOption } from './fuzzy';
import { placeFloating } from './floating';

const DEVICES: PickerOption[] = [
  { value: 'device:rtx-5090', label: 'RTX 5090', group: 'NVIDIA' },
  { value: 'device:rtx-4090', label: 'RTX 4090', group: 'NVIDIA' },
  { value: 'device:m5-max-40', label: 'M5 Max 40C', group: 'Apple' },
  { value: 'device:mi300x', label: 'MI300X', group: 'Datacenter', keywords: ['cdna3'] },
];

describe('fuzzyFilter', () => {
  it('returns every option in order for an empty or blank query', () => {
    expect(fuzzyFilter(DEVICES, '')).toEqual(DEVICES);
    expect(fuzzyFilter(DEVICES, '   ')).toEqual(DEVICES);
  });

  it('matches case-insensitively on the label', () => {
    expect(fuzzyFilter(DEVICES, 'rtx').map((o) => o.label)).toEqual(['RTX 5090', 'RTX 4090']);
    expect(fuzzyFilter(DEVICES, 'MI300').map((o) => o.label)).toEqual(['MI300X']);
  });

  it('matches a subsequence, so "r5" finds RTX 5090', () => {
    expect(fuzzyFilter(DEVICES, 'r5').map((o) => o.label)).toEqual(['RTX 5090']);
  });

  it('ranks a contiguous match above a scattered one', () => {
    const opts: PickerOption[] = [
      { value: 'a', label: 'x a b' },
      { value: 'b', label: 'ab' },
    ];
    expect(fuzzyFilter(opts, 'ab').map((o) => o.value)).toEqual(['b', 'a']);
  });

  it('ranks a prefix match above a later contiguous match', () => {
    const opts: PickerOption[] = [
      { value: 'late', label: 'Qwen 5090' },
      { value: 'early', label: '5090 Qwen' },
    ];
    expect(fuzzyFilter(opts, '5090').map((o) => o.value)).toEqual(['early', 'late']);
  });

  it('also matches the value and keywords, below label matches', () => {
    expect(fuzzyFilter(DEVICES, 'cdna3').map((o) => o.value)).toEqual(['device:mi300x']);
    expect(fuzzyFilter(DEVICES, 'device:m5').map((o) => o.value)).toEqual(['device:m5-max-40']);
  });

  it('requires every whitespace-separated token to match', () => {
    expect(fuzzyFilter(DEVICES, 'nvidia 4090').length).toBe(0);
    expect(fuzzyFilter(DEVICES, 'rtx 4090').map((o) => o.label)).toEqual(['RTX 4090']);
  });

  it('returns nothing when no option matches', () => {
    expect(fuzzyFilter(DEVICES, 'zzz')).toEqual([]);
  });

  it('scatter-matches only labels, so "m5" does not match a system id through its prefix letters', () => {
    const opts: PickerOption[] = [
      { value: 'system:rtx-5090-x2-pcie5', label: 'RTX 5090 ×2 (PCIe)' },
      { value: 'device:m5-max', label: 'M5 Max 40C' },
    ];
    expect(fuzzyFilter(opts, 'm5').map((o) => o.label)).toEqual(['M5 Max 40C']);
    expect(fuzzyFilter(opts, 'r5').map((o) => o.label)).toEqual(['RTX 5090 ×2 (PCIe)']);
  });
});

describe('groupOptions', () => {
  it('buckets by group in first-appearance order and keeps order inside a group', () => {
    const groups = groupOptions([
      { value: '1', label: 'a', group: 'B' },
      { value: '2', label: 'b', group: 'A' },
      { value: '3', label: 'c', group: 'B' },
    ]);
    expect(groups.map((g) => g.group)).toEqual(['B', 'A']);
    expect(groups[0].options.map((o) => o.value)).toEqual(['1', '3']);
  });

  it('keeps ungrouped options in their own bucket', () => {
    const groups = groupOptions([{ value: '1', label: 'a' }]);
    expect(groups).toHaveLength(1);
    expect(groups[0].group).toBeUndefined();
  });
});

describe('stepIndex', () => {
  it('moves by one and clamps at both ends (no wrap)', () => {
    expect(stepIndex(0, 3, 'next')).toBe(1);
    expect(stepIndex(2, 3, 'next')).toBe(2);
    expect(stepIndex(0, 3, 'prev')).toBe(0);
    expect(stepIndex(2, 3, 'prev')).toBe(1);
  });

  it('jumps to the ends and pages by 10 rows', () => {
    expect(stepIndex(4, 30, 'first')).toBe(0);
    expect(stepIndex(4, 30, 'last')).toBe(29);
    expect(stepIndex(4, 30, 'pageDown')).toBe(14);
    expect(stepIndex(25, 30, 'pageDown')).toBe(29);
    expect(stepIndex(4, 30, 'pageUp')).toBe(0);
  });

  it('starts from none (-1) at the first option going forward and the last going back', () => {
    expect(stepIndex(-1, 3, 'next')).toBe(0);
    expect(stepIndex(-1, 3, 'prev')).toBe(2);
  });

  it('returns -1 for an empty list', () => {
    expect(stepIndex(-1, 0, 'next')).toBe(-1);
  });
});

describe('placeFloating', () => {
  const viewport = { width: 400, height: 800 };
  const panel = { width: 300, height: 200 };

  it('opens below the anchor when there is room', () => {
    const p = placeFloating({ top: 100, left: 20, bottom: 142, width: 300 }, panel, viewport, {
      preferred: 'below',
      maxHeight: 360,
    });
    expect(p.side).toBe('below');
    expect(p.top).toBe(142 + 6);
    expect(p.left).toBe(20);
    expect(p.maxHeight).toBe(360);
  });

  it('flips above when the space below is too small and the space above is larger', () => {
    const p = placeFloating({ top: 600, left: 20, bottom: 642, width: 300 }, panel, viewport, {
      preferred: 'below',
      maxHeight: 360,
    });
    expect(p.side).toBe('above');
    expect(p.top).toBe(600 - 6 - 200);
  });

  it('caps the height to the room on the chosen side', () => {
    const p = placeFloating(
      { top: 300, left: 20, bottom: 342, width: 300 },
      { width: 300, height: 900 },
      viewport,
      {
        preferred: 'below',
        maxHeight: 2000,
      },
    );
    expect(p.side).toBe('below');
    expect(p.maxHeight).toBe(800 - 342 - 6 - 8);
  });

  it('clamps horizontally into the viewport', () => {
    const p = placeFloating({ top: 100, left: 300, bottom: 142, width: 90 }, panel, viewport, {
      preferred: 'below',
      maxHeight: 360,
    });
    expect(p.left).toBe(400 - 8 - 300);
    expect(p.width).toBe(300);
  });

  it('never makes the panel wider than the viewport minus margins', () => {
    const p = placeFloating(
      { top: 100, left: 0, bottom: 142, width: 300 },
      { width: 900, height: 50 },
      { width: 320, height: 800 },
      {
        preferred: 'below',
        maxHeight: 360,
      },
    );
    expect(p.width).toBe(320 - 16);
  });
});
