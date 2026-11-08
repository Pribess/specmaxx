import { describe, expect, it } from 'vitest';
import { checkPlacement } from '../../sim/engine/placement';
import { blockShardingErrors } from '../../sim/engine/validate';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../../sim/ir/model';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { nodeTopology } from '../../sim/hw/presets/topologies';
import { PCIE_5_X16 } from '../../sim/hw/presets/links';
import type { Workload } from '../../sim/engine/types';
import { SINGLE_DEVICE_HW, type HardwareChoice } from './appState';
import { resolveHardware } from './hardware';
import { chooseDegrees, fieldIssues, placementField, shardingField, validDegrees, type FieldIssue } from './parallelism';

function load(repo: string): { model: ModelSpec; quant: QuantConfig } {
  const r = mapHfModel(fixtureRepoFiles(repo));
  if (r.kind !== 'ok') throw new Error(`${repo}: ${r.kind}`);
  return { model: r.model, quant: r.quant.quant };
}

const QWEN_27B = load('Qwen/Qwen3.8-27B');
const FLASH_NEXT = load('Qwen/Qwen3.8-Flash-Next');
const BOX_8 = nodeTopology({ devicesPerNode: 8, nodes: 1, intraNode: PCIE_5_X16 });
const ONE = { tp: 1, pp: 1, dp: 1, ep: 1 };
const LOAD: Workload = { batch: 1, promptLen: 1024, decodeContextLen: 1024 };

const issues = (m: { model: ModelSpec; quant: QuantConfig }, parallel: Partial<ParallelConfig>, workload: Partial<Workload> = {}, topo = BOX_8) =>
  fieldIssues(m.model, m.quant, { ...ONE, ...parallel }, { ...LOAD, ...workload }, topo);
const errorsOf = (list: FieldIssue[], field: FieldIssue['field']) => list.filter((i) => i.level === 'error' && i.field === field);

describe('fieldIssues', () => {
  it('is clean when TP × PP equals the device count', () => {
    expect(issues(QWEN_27B, { tp: 8 })).toEqual([]);
    expect(issues(QWEN_27B, { tp: 4, pp: 2 })).toEqual([]);
  });

  it('TP × PP below or above the device count is an error on devices, mentioning only TP and PP', () => {
    const under = errorsOf(issues(QWEN_27B, { tp: 2 }), 'devices');
    expect(under).toHaveLength(1);
    expect(under[0].message).toBe('TP × PP = 2, but the rig has 8 devices. They must be equal.');
    expect(errorsOf(issues(QWEN_27B, { tp: 4, pp: 4 }), 'devices')[0].message).toMatch(/TP × PP = 16/);
    for (const m of [under[0].message]) expect(m).not.toMatch(/\b(DP|EP)\b/);
  });

  it('TP 3 on Qwen3.8-27B: sharding and placement errors on tp, deduplicated over layers', () => {
    const tp = errorsOf(issues(QWEN_27B, { tp: 3 }), 'tp');
    expect(tp.length).toBeGreaterThan(1);
    expect(tp.some((i) => /kvHeads \(4\) and tp \(3\)/.test(i.message))).toBe(true);
    expect(tp.some((i) => i.message.startsWith('tp: '))).toBe(true);
    expect(new Set(tp.map((i) => i.message)).size).toBe(tp.length);
    expect(tp.every((i) => !/^layer\./.test(i.message))).toBe(true);
  });

  it('whole numbers ≥ 1 first, nothing else then', () => {
    const list = issues(QWEN_27B, { tp: 0, pp: 1.5 }, { batch: NaN });
    expect(list.map((i) => i.field).sort()).toEqual(['batch', 'pp', 'tp']);
  });

  it('world size and pp ≤ layers; DP and EP are ignored', () => {
    expect(errorsOf(issues(QWEN_27B, { tp: 4, pp: 4 }), 'devices')).toHaveLength(1);
    const many = nodeTopology({ devicesPerNode: 128, nodes: 1, intraNode: PCIE_5_X16 });
    expect(errorsOf(issues(QWEN_27B, { pp: 65 }, {}, many), 'pp').some((i) => /exceeds the 64 layers/.test(i.message))).toBe(true);
    expect(issues(QWEN_27B, { tp: 8, dp: 4, ep: 2 }, { batch: 3 }).filter((i) => i.level === 'error')).toEqual([]);
    expect(issues(FLASH_NEXT, { tp: 8, ep: 3 }).filter((i) => i.level === 'error')).toEqual([]);
  });

  it('a missing link is a placement error on the degree that needs it', () => {
    const noLink = nodeTopology({ devicesPerNode: 2, nodes: 1 });
    expect(errorsOf(issues(QWEN_27B, { tp: 2 }, {}, noLink), 'tp')[0].message).toMatch(/^tp: .*no intraNode link/);
    expect(errorsOf(issues(QWEN_27B, { pp: 2 }, {}, noLink), 'pp')[0].message).toMatch(/^pp: /);
  });
});

describe('validDegrees', () => {
  it('Qwen3.8-27B on 8 devices', () => {
    const v = (field: 'tp' | 'pp', p: Partial<ParallelConfig> = {}) =>
      validDegrees(field, QWEN_27B.model, QWEN_27B.quant, { ...ONE, ...p }, LOAD, BOX_8);
    expect(v('tp')).toEqual([1, 2, 4, 8]);
    expect(v('pp')).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(v('tp', { pp: 2 })).toEqual([1, 2, 4, 8]);
  });
});

describe('engine message wording this classification relies on', () => {
  it('blockShardingErrors: "<layer>: …", EP rules name "ep (", TP rules name "tp (" or a width split', () => {
    const tpErrors = blockShardingErrors({ ...QWEN_27B, parallel: { ...ONE, tp: 3 } });
    expect(tpErrors.length).toBeGreaterThan(0);
    for (const e of tpErrors) {
      expect(e).toMatch(/^layer\.\d+: /);
      expect(shardingField(e)).toBe('tp');
    }
    const epErrors = blockShardingErrors({ ...FLASH_NEXT, parallel: { ...ONE, tp: 2, ep: 3 } });
    expect(epErrors.length).toBeGreaterThan(0);
    for (const e of epErrors) expect(shardingField(e)).toBe('ep');
  });

  it('checkPlacement: errors prefixed tp: / ep: / pp:, or the world-size message', () => {
    const noLink = nodeTopology({ devicesPerNode: 2, nodes: 1 });
    expect(checkPlacement({ ...ONE, tp: 2 }, noLink).errors[0]).toMatch(/^tp: /);
    expect(checkPlacement({ ...ONE, ep: 2 }, noLink).errors[0]).toMatch(/^ep: /);
    expect(checkPlacement({ ...ONE, pp: 2 }, noLink).errors[0]).toMatch(/^pp: /);
    expect(checkPlacement({ ...ONE, tp: 4 }, noLink).errors[0]).toMatch(/^tp·pp·dp = 4 devices needed/);
    expect(checkPlacement({ ...ONE, tp: 1 }, BOX_8).warnings[0]).toMatch(/devices are unused/);
    expect(placementField('tp: x')).toBe('tp');
    expect(placementField('tp·pp·dp = 4 devices needed')).toBe('devices');
  });
});

describe('chooseDegrees', () => {
  const RIGS: Record<string, Partial<HardwareChoice>> = {
    'dgx-spark-x4-switch': { deviceId: 'dgx-spark', nodes: 4 },
    'dgx-spark-x4-ring': { deviceId: 'dgx-spark', nodes: 4, cabling: 'ring' },
    'r9700-x4-pcie5': { deviceId: 'radeon-ai-pro-r9700', devicesPerNode: 4 },
  };
  const rig = (id: string) => resolveHardware({ ...SINGLE_DEVICE_HW, ...RIGS[id] }).topology;
  it.each(Object.keys(RIGS))('%s: TP × PP uses all 4 devices, with no errors', (id) => {
    const topology = rig(id);
    for (const m of [QWEN_27B, FLASH_NEXT]) {
      const { tp, pp } = chooseDegrees(m.model, m.quant, LOAD, topology);
      expect(tp * pp).toBe(4);
      expect(fieldIssues(m.model, m.quant, { tp, pp, dp: 1, ep: 1 }, LOAD, topology).filter((i) => i.level === 'error')).toEqual([]);
    }
  });
  it('one device: 1 × 1', () => {
    expect(chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, nodeTopology({ devicesPerNode: 1, nodes: 1 }))).toEqual({ tp: 1, pp: 1 });
  });
  it('3 devices on a model whose heads do not divide by 3: pipeline only', () => {
    expect(chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, nodeTopology({ devicesPerNode: 3, nodes: 1, intraNode: PCIE_5_X16 }))).toEqual({ tp: 1, pp: 3 });
  });
  it('no valid split: TP = PP = 1', () => {
    expect(chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, nodeTopology({ devicesPerNode: 1000, nodes: 1, intraNode: PCIE_5_X16 }))).toEqual({ tp: 1, pp: 1 });
  });
  it.each([1e6, 2 ** 54, 2 ** 60, Infinity, NaN])('a huge or unsafe device count (%s) returns promptly with TP = PP = 1', (n) => {
    const start = Date.now();
    const topology = nodeTopology({ devicesPerNode: n, nodes: 1, intraNode: PCIE_5_X16 });
    const { tp, pp } = chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, topology);
    expect(Date.now() - start).toBeLessThan(1000);
    expect(tp).toBe(1);
    expect(pp).toBe(1);
  });
});
