// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { App } from '../App';
import { inlineRunner } from '../sim/runner';
import { loadHardwareProvenance } from '../state/provenance';
import { mapHfModel } from '../../sim/models/hf';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';

vi.mock('../state/provenance', async () => {
  const { resolveHardware } = await import('../state/hardware');
  const { DEVICE_UNLOCKS } = await import('../../sim/hw/presets');
  return {
    loadHardwareProvenance: vi.fn(async () => ({
      DEVICE_UNLOCKS,
      resolveHardware: (...args: Parameters<typeof resolveHardware>) => {
        const resolved = resolveHardware(...args);
        const device = { ...resolved.device.device, l2Bytes: 123 };
        // Deliberately missing source: the full editor must flag this, never trust a stripped marker.
        device.sources = device.sources.filter((source) => !source.fields.includes('l2Bytes'));
        device.estimated = device.estimated.filter((entry) => entry.field !== 'l2Bytes');
        return { ...resolved, device: { ...resolved.device, device } };
      },
    })),
  };
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it('does not fetch sources on first paint; the editor flags uncovered numbers after the sidecar loads', async () => {
  history.replaceState(null, '', '/');
  const { container } = render(
    <App runner={inlineRunner} loader={async (id) => mapHfModel(fixtureRepoFiles(id))} />,
  );
  expect(loadHardwareProvenance).not.toHaveBeenCalled();
  const advanced = container.querySelector('details.advanced') as HTMLDetailsElement;
  advanced.open = true;
  fireEvent(advanced, new Event('toggle'));
  expect(await screen.findByText(/l2Bytes is not covered by any sources/)).toBeTruthy();
  expect(loadHardwareProvenance).toHaveBeenCalled();
});
