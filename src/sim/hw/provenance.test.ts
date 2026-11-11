import { beforeAll, describe, expect, it } from 'vitest';
import { build } from 'vite';
import { COMMUNITY_UNLOCKS } from 'virtual:community-unlocks';
import { deviceModifierErrors } from './modifiers';
import { stripProvenance } from '../../../vite/stripProvenance';
import { strip, hasStrippedProvenance } from './provenance';
import { numericLeafPaths, validateDevice, validateLink } from './validate';
import { DEVICE_PRESETS, DEVICE_UNLOCKS, POPULAR_SETUPS } from './presets';
import { LINK_PRESETS, HOST_PCIE_LINKS, hostPcieLink } from './presets/links';
import { HOST_RAM_PRESETS, SSD_PRESETS } from './presets/storage';
import { NVIDIA_SM_SFU, AMD_CDNA4_SFU, APPLE_GPU_SFU } from './presets/sfu';
import { resolveHardware } from '../../ui/state/hardware';
import { loadHardwareProvenance } from '../../ui/state/provenance';
import { DEFAULT_STATE, reduce } from '../../ui/state/appState';
import { buildSimInput, buildDisaggregatedInput } from '../../ui/state/buildInput';
import { BUNDLED_DEFAULT_MODEL } from '../../ui/state/bundledModel';
import { inlineRunner } from '../../ui/sim/runner';

const full = {
  DEVICE_PRESETS,
  DEVICE_UNLOCKS,
  LINK_PRESETS,
  HOST_PCIE_LINKS,
  HOST_RAM_PRESETS,
  SSD_PRESETS,
  POPULAR_SETUPS,
  SFU_RATES: [NVIDIA_SM_SFU, AMD_CDNA4_SFU, APPLE_GPU_SFU],
};
type Bundled = typeof full & {
  resolveHardware: typeof resolveHardware;
  hostPcieLink: typeof hostPcieLink;
  inlineRunner: typeof inlineRunner;
  loadHardwareProvenance: typeof loadHardwareProvenance;
  COMMUNITY_UNLOCKS: Set<string>;
};
let bundled: Bundled;

beforeAll(async () => {
  const entry = 'virtual:provenance-test-entry';
  const result = await build({
    configFile: false,
    logLevel: 'silent',
    plugins: [
      stripProvenance(),
      {
        name: 'provenance-test-entry',
        resolveId(id) {
          if (id === entry) return `\0${entry}`;
        },
        load(id) {
          if (id !== `\0${entry}`) return;
          return `export { DEVICE_PRESETS, DEVICE_UNLOCKS, POPULAR_SETUPS } from '/src/sim/hw/presets/index.ts';
export { LINK_PRESETS, HOST_PCIE_LINKS, hostPcieLink } from '/src/sim/hw/presets/links.ts';
export { HOST_RAM_PRESETS, SSD_PRESETS } from '/src/sim/hw/presets/storage.ts';
import { NVIDIA_SM_SFU, AMD_CDNA4_SFU, APPLE_GPU_SFU } from '/src/sim/hw/presets/sfu.ts';
export const SFU_RATES = [NVIDIA_SM_SFU, AMD_CDNA4_SFU, APPLE_GPU_SFU];
export { resolveHardware } from '/src/ui/state/hardware.ts';
export { inlineRunner } from '/src/ui/sim/runner.ts';
export { COMMUNITY_UNLOCKS } from 'virtual:community-unlocks';
export { loadHardwareProvenance } from '/src/ui/state/provenance.ts';`;
        },
      },
    ],
    build: {
      write: false,
      minify: false,
      rollupOptions: {
        input: entry,
        preserveEntrySignatures: 'strict',
        output: { format: 'es', codeSplitting: false },
      },
    },
  });
  if (Array.isArray(result) || !('output' in result)) throw new Error('expected one Rollup output');
  const chunk = result.output.find((item) => item.type === 'chunk');
  if (!chunk || chunk.type !== 'chunk') throw new Error('missing bundle');
  bundled = (await import(
    /* @vite-ignore */ `data:text/javascript;base64,${Buffer.from(chunk.code).toString('base64')}`
  )) as Bundled;
});

describe('build-time preset provenance', () => {
  it('dev and generated build community flags match every original option', () => {
    const expected: string[] = [];
    for (const [device, axes] of Object.entries(DEVICE_UNLOCKS))
      for (const axis of axes)
        for (const option of axis.options)
          if (
            [...option.deviceModifiers, ...(option.intraNodeLinkModifiers ?? [])].some((mod) =>
              mod.sources.some((source) => source.kind === 'community'),
            )
          )
            expected.push(`${device}/${axis.id}/${option.id}`);
    expect([...COMMUNITY_UNLOCKS].sort()).toEqual(expected.sort());
    expect([...bundled.COMMUNITY_UNLOCKS].sort()).toEqual(expected.sort());
  });

  it('stripped modifiers still reject empty user-visible notes', () => {
    const mod = strip({
      kind: 'memoryCapacity' as const,
      capacityOptionsBytes: [32e9],
      sources: [],
      note: '',
    });
    expect(deviceModifierErrors(DEVICE_PRESETS[0], mod)).toContain(
      'modifier memoryCapacity: note is empty',
    );
  });
  it('the module actually bundled equals pure strip(full), including factories and unlocks', () => {
    for (const key of Object.keys(full) as (keyof typeof full)[])
      expect(bundled[key], key).toEqual(strip(full[key]));
    for (const generation of [1, 2, 3, 4, 5] as const)
      for (const lanes of [4, 8, 16] as const)
        expect(bundled.hostPcieLink(generation, lanes)).toEqual(
          strip(hostPcieLink(generation, lanes)),
        );
  });

  it('changes only citation payloads and their explicit availability marker; every numeric leaf is identical', () => {
    function withoutCitations(value: unknown): unknown {
      if (Array.isArray(value)) return value.map(withoutCitations);
      if (!value || typeof value !== 'object') return value;
      return Object.fromEntries(
        Object.entries(value)
          .filter(
            ([key]) =>
              ![
                'sources',
                'estimated',
                'estimatedReason',
                'source',
                'reason',
                'provenanceStripped',
              ].includes(key),
          )
          .map(([key, child]) => [key, withoutCitations(child)]),
      );
    }
    expect(withoutCitations(strip(full))).toEqual(withoutCitations(full));
    expect(numericLeafPaths(strip(full))).toEqual(numericLeafPaths(full));
    expect(DEVICE_PRESETS.every((device) => !hasStrippedProvenance(device))).toBe(true);
    expect(DEVICE_PRESETS.every((device) => device.sources.length > 0)).toBe(true);
  });

  it('stripped presets still reject physically invalid numbers, while custom specs still need sources', () => {
    const device = strip(DEVICE_PRESETS[0]);
    expect(validateDevice(device)).toEqual([]);
    expect(
      validateDevice({ ...device, memory: { ...device.memory, bandwidthBytesPerSec: -1 } }),
    ).not.toEqual([]);
    const { provenanceStripped: _marker, ...custom } = device as typeof device & {
      provenanceStripped: true;
    };
    void _marker;
    expect(validateDevice(custom).some((error) => error.includes('not covered'))).toBe(true);
    expect(validateLink({ ...strip(LINK_PRESETS[0]), latencySec: NaN })).not.toEqual([]);
  });

  it('the lazy sidecar restores sources for edits/unlocks and flags an uncovered editor field', async () => {
    // Test both dev/Vitest's originals and the production full-query graph.
    const sidecar = await bundled.loadHardwareProvenance();
    expect(await loadHardwareProvenance()).toEqual(expect.objectContaining({ DEVICE_UNLOCKS }));
    const state = reduce(DEFAULT_STATE, {
      type: 'setEdit',
      target: 'device',
      path: 'memory.bandwidthBytesPerSec',
      value: 123e9,
    });
    const resolved = sidecar.resolveHardware(state.hw);
    expect(resolved).toEqual(resolveHardware(state.hw));
    expect(
      resolved.device.device.estimated.some((entry) => entry.reason === 'edited by the user'),
    ).toBe(true);
    const device = { ...resolved.device.device, l2Bytes: 123 };
    device.sources = device.sources.filter(
      (source) => !source.fields.some((field) => field === 'l2Bytes'),
    );
    device.estimated = device.estimated.filter((entry) => entry.field !== 'l2Bytes');
    expect(validateDevice(device).some((error) => error.includes('l2Bytes is not covered'))).toBe(
      true,
    );
    expect(sidecar.DEVICE_UNLOCKS).toEqual(DEVICE_UNLOCKS);
  });

  it('the worker runner gives identical default and PD + offload results with bundled stripped hardware', async () => {
    const pd = reduce(reduce(DEFAULT_STATE, { type: 'setDisaggregation', on: true }), {
      type: 'setOffload',
      offload: { embeddings: { tier: 'host_ram' } },
    });
    for (const state of [DEFAULT_STATE, pd]) {
      const built = buildSimInput(state, BUNDLED_DEFAULT_MODEL);
      if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
      const disaggregated = buildDisaggregatedInput(state, BUNDLED_DEFAULT_MODEL);
      if (disaggregated && disaggregated.kind !== 'ok')
        throw new Error(JSON.stringify(disaggregated));
      const input = {
        ...built.input,
        hardware: bundled.resolveHardware(state.hw, state === pd).stack,
      };
      const fullPd =
        disaggregated?.kind === 'ok'
          ? { input: disaggregated.input, decodeNotes: disaggregated.decode.notes }
          : undefined;
      const strippedPd = fullPd ? strip(fullPd) : undefined;
      expect(await bundled.inlineRunner.run(input, built.notes, strippedPd)).toEqual(
        await inlineRunner.run(built.input, built.notes, fullPd),
      );
    }
  });
});
