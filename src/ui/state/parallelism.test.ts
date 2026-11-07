import { describe, expect, it } from 'vitest';
import { checkPlacement } from '../../sim/engine/placement';
import { blockShardingErrors } from '../../sim/engine/validate';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../../sim/ir/model';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { nodeTopology } from '../../sim/hw/presets/topologies';
import { PCIE_5_X16 } from '../../sim/hw/presets/links';
import type { Workload } from '../../sim/engine/types';
import { fieldIssues, placementField, shardingField, validDegrees, type FieldIssue } from './parallelism';

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
  it('is clean for a valid input, with a warning for unused devices', () => {
    const list = issues(QWEN_27B, { tp: 2 });
    expect(list.filter((i) => i.level === 'error')).toEqual([]);
    expect(list).toContainEqual({ field: 'devices', level: 'warning', message: '6 of 8 devices are unused (TP × PP × DP = 2).' });
    expect(issues(QWEN_27B, { tp: 8 })).toEqual([]);
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

  it('world size, batch % dp, pp ≤ layers, EP needs MoE', () => {
    expect(errorsOf(issues(QWEN_27B, { tp: 4, dp: 4 }), 'devices')).toHaveLength(1);
    expect(errorsOf(issues(QWEN_27B, { dp: 2 }, { batch: 3 }), 'batch')).toHaveLength(1);
    const many = nodeTopology({ devicesPerNode: 128, nodes: 1, intraNode: PCIE_5_X16 });
    expect(errorsOf(issues(QWEN_27B, { pp: 65 }, {}, many), 'pp').some((i) => /exceeds the 64 layers/.test(i.message))).toBe(true);
    expect(errorsOf(issues(QWEN_27B, { ep: 2, tp: 2 }), 'ep')[0].message).toMatch(/dense/);
  });

  it('EP errors on the MoE model go to ep, multi-rule messages split', () => {
    const ep = errorsOf(issues(FLASH_NEXT, { tp: 2, ep: 3 }), 'ep');
    expect(ep.some((i) => /experts \(512\) must be divisible by ep \(3\)/.test(i.message))).toBe(true);
    expect(ep.some((i) => /^moe: ep \(3\) must divide tp/.test(i.message))).toBe(true);
    expect(ep.every((i) => !i.message.includes('\n'))).toBe(true);
    expect(issues(FLASH_NEXT, { tp: 2, dp: 4, ep: 8 }, { batch: 4 })).toEqual([]);
  });

  it('a missing link is a placement error on the degree that needs it', () => {
    const noLink = nodeTopology({ devicesPerNode: 2, nodes: 1 });
    expect(errorsOf(issues(QWEN_27B, { tp: 2 }, {}, noLink), 'tp')[0].message).toMatch(/^tp: .*no intraNode link/);
    expect(errorsOf(issues(QWEN_27B, { pp: 2 }, {}, noLink), 'pp')[0].message).toMatch(/^pp: /);
  });
});

describe('validDegrees', () => {
  it('Qwen3.8-27B on 8 devices', () => {
    const v = (field: keyof ParallelConfig, p: Partial<ParallelConfig> = {}) =>
      validDegrees(field, QWEN_27B.model, QWEN_27B.quant, { ...ONE, ...p }, LOAD, BOX_8);
    expect(v('tp')).toEqual([1, 2, 4, 8]);
    expect(v('ep')).toEqual([1]);
    expect(v('pp')).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(v('tp', { dp: 2 })).toEqual([1, 2, 4]);
  });

  it('Qwen3.8-Flash-Next: EP at the current TP and DP', () => {
    const v = (field: keyof ParallelConfig, p: Partial<ParallelConfig>) =>
      validDegrees(field, FLASH_NEXT.model, FLASH_NEXT.quant, { ...ONE, ...p }, { ...LOAD, batch: 4 }, BOX_8);
    expect(v('ep', { tp: 2, dp: 4 })).toEqual([1, 2, 4, 8]);
    expect(v('ep', { tp: 2, dp: 1 })).toEqual([1, 2]);
    // TP is checked at ep = 1, so the current EP does not narrow it.
    expect(v('tp', { ep: 8 })).toEqual(v('tp', { ep: 1 }));
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
