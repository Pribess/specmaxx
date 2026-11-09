// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import {
  ETHERNET_10G_TCP,
  LINK_PRESETS,
  linkPresetId,
  NVLINK_4,
  TB4_TCP,
} from '../../sim/hw/presets';
import { LINK_HELP_IDS, networkLinkHelp } from './linkHelp';

afterEach(cleanup);

function lines(node: ReturnType<typeof networkLinkHelp>): HTMLElement[] {
  const { container } = render(<div>{node}</div>);
  return [...container.querySelectorAll<HTMLElement>('.help-line')];
}

describe('networkLinkHelp', () => {
  it('has an entry for every TCP link preset, keyed by linkPresetId', () => {
    const tcp = LINK_PRESETS.filter((l) => l.name.includes('TCP')).map(linkPresetId);
    expect(new Set(LINK_HELP_IDS)).toEqual(new Set(tcp));
    for (const id of LINK_HELP_IDS)
      expect(
        LINK_PRESETS.some((l) => linkPresetId(l) === id),
        id,
      ).toBe(true);
  });

  it('formats α from the link, links its sources, and adds the staging line for a GPU (≤ 3 lines)', () => {
    const ls = lines(networkLinkHelp(ETHERNET_10G_TCP, true));
    expect(ls).toHaveLength(3);
    expect(ls[0].textContent).toMatch(
      /^Latency 10 µs: one-way TCP floor between two machines over a direct cable/,
    );
    expect(ls[1].textContent).toMatch(/50–130 µs for a small all-reduce/);
    expect(ls[2].textContent).toBe(
      'GPUs with their own memory stage every message through host memory: about 2 µs more per message.',
    );
    const links = ls[1].querySelectorAll('a');
    expect(links).toHaveLength(2);
    for (const a of links) {
      expect(a.getAttribute('target')).toBe('_blank');
      expect(a.getAttribute('rel')).toBe('noreferrer');
    }
    cleanup();
    expect(
      lines(networkLinkHelp({ ...TB4_TCP, latencySec: 12.5e-6 }, false))[0].textContent,
    ).toMatch(/^Latency 12\.5 µs/);
    cleanup();
    expect(lines(networkLinkHelp(TB4_TCP, false))).toHaveLength(2);
    for (const link of LINK_PRESETS.filter((l) => LINK_HELP_IDS.includes(linkPresetId(l)))) {
      cleanup();
      expect(lines(networkLinkHelp(link, true)).length, link.name).toBeLessThanOrEqual(3);
    }
  });

  it('is undefined for links without their own help', () => {
    expect(networkLinkHelp(NVLINK_4, false)).toBeUndefined();
  });
});
