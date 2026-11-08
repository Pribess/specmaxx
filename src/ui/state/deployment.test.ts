import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf';
import type { HfApiModelInfo } from '../../sim/models/hf/types';
import flashApi from '../../sim/models/__fixtures__/Qwen__Qwen3.8-Flash-Next.api.json?raw';
import flashConfig from '../../sim/models/__fixtures__/Qwen__Qwen3.8-Flash-Next.config.json?raw';
import {
  DDR5_6400_8CH,
  DEFAULT_HOST_RAM,
  DEFAULT_SSD,
  DGX_SPARK_SSD_4TB,
  hostPcieLink,
} from '../../sim/hw/presets';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { BUNDLED_DEFAULT_MODEL } from './bundledModel';
import { buildSimInput, loadedFromHf, type LoadedModel } from './buildInput';
import { normalizeDeployment, normalizeOffload } from './deployment';
import { drafterSource, lookupTableParams, tablesOffDeviceByDesign } from './deploymentInput';
import { defaultSsdFor, editableFields, findDevice, resolveHardware } from './hardware';

const apply = (s: AppState, ...actions: Parameters<typeof reduce>[1][]) =>
  actions.reduce(reduce, s);

/** Qwen3.8-Flash-Next (512 experts, PLE table off-device by design), mapped from the committed fixtures. */
function flashNext(): LoadedModel {
  const id = 'Qwen/Qwen3.8-Flash-Next';
  const r = mapHfModel({
    id,
    revision: 'main',
    config: JSON.parse(flashConfig) as unknown,
    api: JSON.parse(flashApi) as HfApiModelInfo,
  });
  if (r.kind !== 'ok') throw new Error(r.kind);
  return loadedFromHf({ kind: 'hf', id }, r);
}
const FLASH = flashNext();
const FLASH_STATE: AppState = {
  ...DEFAULT_STATE,
  model: { kind: 'hf', id: 'Qwen/Qwen3.8-Flash-Next' },
};

describe('deployment normalizers', () => {
  it('drop default values, and drop a queue depth when nothing is offloaded', () => {
    expect(
      normalizeOffload({
        experts: { tier: 'host_ram', fromLayer: 0, residentFraction: 0, hitRatio: 'uniform' },
        tables: { tier: 'nvme', scaleLayout: 'pinned_host' },
      }),
    ).toEqual({ experts: { tier: 'host_ram' }, tables: { tier: 'nvme' } });
    expect(normalizeOffload({ queueDepth: 4 })).toEqual({});
    expect(
      normalizeDeployment({
        offload: {},
        specDec: { method: 'mtp', drafter: 'a/b', useCase: 'prose' },
      }),
    ).toEqual({ specDec: { method: 'mtp' } });
    expect(
      normalizeDeployment({ specDec: { method: 'dflash2', useCase: 'code', alpha: 0.8 } }),
    ).toEqual({ specDec: { method: 'dflash2', alpha: 0.8 } });
  });
});

describe('reduce: deployment', () => {
  it('turning offload on prefills host-RAM tables for a model whose table is off-device by design, once', () => {
    const on = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { experts: { tier: 'host_ram' } },
      tablesOffDeviceByDesign: true,
    });
    expect(on.deployment.offload).toEqual({
      experts: { tier: 'host_ram' },
      tables: { tier: 'host_ram' },
    });
    // The user moves the tables back to the device; changing the experts later does not prefill again.
    const back = apply(
      on,
      {
        type: 'setOffload',
        offload: { experts: { tier: 'host_ram' } },
        tablesOffDeviceByDesign: true,
      },
      { type: 'setOffload', offload: { experts: { tier: 'nvme' } }, tablesOffDeviceByDesign: true },
    );
    expect(back.deployment.offload).toEqual({ experts: { tier: 'nvme' } });
    // Without the flag nothing is prefilled.
    expect(
      reduce(DEFAULT_STATE, { type: 'setOffload', offload: { experts: { tier: 'host_ram' } } })
        .deployment.offload,
    ).toEqual({
      experts: { tier: 'host_ram' },
    });
  });

  it('turning everything off leaves deployment {}', () => {
    const s = apply(
      DEFAULT_STATE,
      { type: 'setOffload', offload: { experts: { tier: 'host_ram' }, queueDepth: 8 } },
      { type: 'setSpecDec', specDec: { method: 'mtp' } },
      { type: 'setOffload', offload: { queueDepth: 8 } },
      { type: 'setSpecDec', specDec: undefined },
    );
    expect(s.deployment).toEqual({});
  });

  it('changing the model drops the drafter repo; changing the device keeps host RAM and SSD but not the host link', () => {
    const s = apply(
      DEFAULT_STATE,
      {
        type: 'setSpecDec',
        specDec: { method: 'dflash2', drafter: 'incoai/Qwen3.8-27B-DFlash2', k: 5 },
      },
      {
        type: 'setHost',
        host: { ramId: 'ddr5-6400-8ch', ssdId: 'nvme-t705-2tb', link: { generation: 4, lanes: 8 } },
      },
      { type: 'setModel', model: { kind: 'hf', id: 'Qwen/Qwen3-8B' } },
      { type: 'setDevice', deviceId: 'rtx-4090' },
    );
    expect(s.deployment.specDec).toEqual({ method: 'dflash2', k: 5 });
    expect(s.hw.host).toEqual({ ramId: 'ddr5-6400-8ch', ssdId: 'nvme-t705-2tb' });
    expect(reduce(s, { type: 'setHost', host: {} }).hw.host).toBeUndefined();
  });
});

describe('hardware: host', () => {
  it('excludes hostInterface from the editable device fields', () => {
    const fields = editableFields(findDevice('rtx-5090')!);
    expect(fields.some((f) => f.path.startsWith('hostInterface'))).toBe(false);
    expect(fields.length).toBeGreaterThan(0);
  });

  it('builds stack.host only when something is off-device', () => {
    expect(resolveHardware(DEFAULT_STATE.hw).stack.host).toBeUndefined();
    expect(resolveHardware(DEFAULT_STATE.hw, true).stack.host).toEqual({
      link: hostPcieLink(5, 16),
      ram: DEFAULT_HOST_RAM,
      ssd: DEFAULT_SSD,
    });
    const hw = {
      ...DEFAULT_STATE.hw,
      host: {
        ramId: DDR5_6400_8CH.id,
        ramCapacityBytes: 512e9,
        link: { generation: 4 as const, lanes: 8 as const },
      },
    };
    const host = resolveHardware(hw, true).stack.host!;
    expect(host.link).toEqual(hostPcieLink(4, 8));
    expect(host.ram?.capacityBytes).toBe(512e9);
    expect(host.ram?.bandwidthBytesPerSec).toBe(DDR5_6400_8CH.bandwidthBytesPerSec);
  });

  it("on unified memory gives only the SSD, by default the machine's own", () => {
    const spark = { ...DEFAULT_STATE.hw, deviceId: 'dgx-spark' };
    expect(defaultSsdFor(findDevice('dgx-spark')!)).toBe(DGX_SPARK_SSD_4TB);
    expect(defaultSsdFor(findDevice('rtx-5090')!)).toBe(DEFAULT_SSD);
    expect(resolveHardware(spark, true).stack.host).toEqual({ ssd: DGX_SPARK_SSD_4TB });
  });

  it('gives no host link for a device without a host interface', () => {
    const h100 = { ...DEFAULT_STATE.hw, deviceId: 'h100-sxm' };
    expect(resolveHardware(h100, true).stack.host?.link).toBeUndefined();
  });
});

describe('buildSimInput: deployment', () => {
  it('omits deployment and host with every knob off', () => {
    const built = buildSimInput(FLASH_STATE, FLASH);
    expect(built.kind).toBe('ok');
    if (built.kind !== 'ok') return;
    expect('deployment' in built.input).toBe(false);
    expect(built.input.hardware.host).toBeUndefined();
    // A table placement on a model without tables emits nothing either (a warning).
    const noTables = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { tables: { tier: 'host_ram' } },
    });
    const b2 = buildSimInput(noTables, BUNDLED_DEFAULT_MODEL);
    expect(b2.kind === 'ok' && 'deployment' in b2.input).toBe(false);
    expect(b2.issues.some((i) => i.field === 'offload' && i.level === 'warning')).toBe(true);
  });

  it('emits the offload config and the host when experts are offloaded', () => {
    const s = reduce(FLASH_STATE, {
      type: 'setOffload',
      offload: {
        experts: {
          tier: 'host_ram',
          fromLayer: 4,
          residentFraction: 0.25,
          hitRatio: 'skewed_fine_grained',
        },
        queueDepth: 16,
      },
    });
    const built = buildSimInput(s, FLASH);
    expect(built.issues.filter((i) => i.level === 'error')).toEqual([]);
    if (built.kind !== 'ok') throw new Error('not ok');
    expect(built.input.deployment).toEqual({
      offload: {
        routedExperts: {
          tier: 'host_ram',
          layers: { from: 4, to: FLASH.model.layers.length },
          residentFraction: 0.25,
          hitRatio: { kind: 'skewed_fine_grained' },
        },
        queueDepth: 16,
      },
    });
    expect(built.input.hardware.host?.ram).toBe(DEFAULT_HOST_RAM);
  });

  it('dense-layer offload streams layers [n, L)', () => {
    const s = reduce(DEFAULT_STATE, {
      type: 'setOffload',
      offload: { dense: { tier: 'nvme', layersOnDevice: 40 } },
    });
    const built = buildSimInput(s, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    expect(built.input.deployment?.offload?.denseLayers).toEqual({
      tier: 'nvme',
      layers: { from: 40, to: BUNDLED_DEFAULT_MODEL.model.layers.length },
    });
  });

  it('MTP on the default model uses its head, the default k and the prose preset', () => {
    const s = reduce(DEFAULT_STATE, { type: 'setSpecDec', specDec: { method: 'mtp' } });
    const built = buildSimInput(s, BUNDLED_DEFAULT_MODEL);
    if (BUNDLED_DEFAULT_MODEL.model.draft?.kind !== 'mtp') {
      expect(built.issues.some((i) => i.field === 'specDec' && i.level === 'error')).toBe(true);
      return;
    }
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    const sd = built.input.deployment?.speculative;
    expect(sd?.method).toBe('mtp');
    expect(sd?.drafter).toBeUndefined();
    expect(sd?.draftLength).toBeGreaterThanOrEqual(1);
    expect(sd?.acceptance.kind).toBeDefined();
    const custom = buildSimInput(
      reduce(s, { type: 'setSpecDec', specDec: { method: 'mtp', k: 2, alpha: 0.6 } }),
      BUNDLED_DEFAULT_MODEL,
    );
    if (custom.kind !== 'ok') throw new Error(JSON.stringify(custom.issues));
    expect(custom.input.deployment?.speculative).toMatchObject({
      draftLength: 2,
      acceptance: { kind: 'iid', alpha: 0.6 },
    });
  });

  it('a block method waits for its drafter repo, then uses the loaded spec; the hypothetical one needs no fetch', () => {
    const s = reduce(DEFAULT_STATE, { type: 'setSpecDec', specDec: { method: 'dflash2' } });
    const source = drafterSource(s.deployment.specDec, BUNDLED_DEFAULT_MODEL.model, s.model.id);
    expect(source).toEqual({ kind: 'repo', repoId: 'incoai/Qwen3.8-27B-DFlash2', known: true });
    const loading = buildSimInput(s, BUNDLED_DEFAULT_MODEL, {
      kind: 'loading',
      repoId: 'incoai/Qwen3.8-27B-DFlash2',
    });
    expect(loading.kind).toBe('issues');
    expect(
      loading.issues.some((i) => i.field === 'specDec' && /Loading drafter/.test(i.message)),
    ).toBe(true);
    const failed = buildSimInput(s, BUNDLED_DEFAULT_MODEL, {
      kind: 'error',
      repoId: 'incoai/Qwen3.8-27B-DFlash2',
      message: 'gone (404)',
    });
    expect(failed.issues.some((i) => /gone \(404\)/.test(i.message))).toBe(true);
    const hyp = reduce(s, { type: 'setSpecDec', specDec: { method: 'dflash2', drafter: 'hyp' } });
    expect(
      drafterSource(hyp.deployment.specDec, BUNDLED_DEFAULT_MODEL.model, hyp.model.id),
    ).toEqual({ kind: 'hypothetical' });
    const built = buildSimInput(hyp, BUNDLED_DEFAULT_MODEL);
    if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
    expect(built.input.deployment?.speculative?.drafter?.kind).toBe('hypothetical');
    expect('host' in built.input.hardware).toBe(false);
  });
});

describe('lookup tables', () => {
  it('counts the PLE table params of Qwen3.8-Flash-Next and flags it off-device by design', () => {
    expect(lookupTableParams(FLASH.model) / 1e9).toBeCloseTo(51.2, 1);
    const flagged = {
      ...FLASH.model,
      layers: FLASH.model.layers.map((l) => ({
        ...l,
        blocks: l.blocks.map((b) =>
          b.kind === 'ngram_embedding' ? { ...b, offDeviceByDesign: true } : b,
        ),
      })),
    };
    expect(tablesOffDeviceByDesign(flagged)).toBe(true);
    expect(tablesOffDeviceByDesign(BUNDLED_DEFAULT_MODEL.model)).toBe(false);
    expect(lookupTableParams(BUNDLED_DEFAULT_MODEL.model)).toBe(0);
  });
});
