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
/** A MoE model with an expert-output norm (expert-TP collectives when tp/ep > 1). */
const MUSE_SPARK = load('unsloth/Muse-Spark-Open-816B-A42B');
const BOX_8 = nodeTopology({ devicesPerNode: 8, nodes: 1, intraNode: PCIE_5_X16 });
const ONE = { tp: 1, pp: 1, dp: 1, ep: 1 };
const LOAD: Workload = { batch: 1, promptLen: 1024, decodeContextLen: 1024 };

const issues = (m: { model: ModelSpec; quant: QuantConfig }, parallel: Partial<ParallelConfig>, workload: Partial<Workload> = {}, topo = BOX_8) =>
  fieldIssues(m.model, m.quant, { ...ONE, ...parallel }, { ...LOAD, ...workload }, topo);
const errorsOf = (list: FieldIssue[], field: FieldIssue['field']) => list.filter((i) => i.level === 'error' && i.field === field);

describe('fieldIssues', () => {
  it('is clean when TP × PP × DP equals the device count', () => {
    expect(issues(QWEN_27B, { tp: 8 })).toEqual([]);
    expect(issues(QWEN_27B, { tp: 4, pp: 2 })).toEqual([]);
    expect(issues(QWEN_27B, { tp: 4, dp: 2 }, { batch: 2 })).toEqual([]);
    expect(issues(QWEN_27B, { tp: 2, pp: 2, dp: 2 }, { batch: 4 })).toEqual([]);
    expect(issues(QWEN_27B, { dp: 8 }, { batch: 8 })).toEqual([]);
  });

  it('TP × PP × DP below or above the device count is an error on devices', () => {
    const under = errorsOf(issues(QWEN_27B, { tp: 2 }), 'devices');
    expect(under).toHaveLength(1);
    expect(under[0].message).toBe('TP × PP × DP = 2, but the rig has 8 devices. They must be equal.');
    expect(errorsOf(issues(QWEN_27B, { tp: 4, pp: 4 }), 'devices')[0].message).toMatch(/TP × PP × DP = 16/);
    // DP counts devices: TP 4 × DP 2 fills the 8, TP 4 × DP 4 is too many, TP 2 × DP 2 too few.
    expect(errorsOf(issues(QWEN_27B, { tp: 4, dp: 4 }, { batch: 4 }), 'devices')[0].message).toBe(
      'TP × PP × DP = 16, but the rig has 8 devices. They must be equal.',
    );
    expect(errorsOf(issues(QWEN_27B, { tp: 2, dp: 2 }, { batch: 2 }), 'devices')[0].message).toMatch(/TP × PP × DP = 4,/);
    const one = nodeTopology({ devicesPerNode: 1, nodes: 1 });
    expect(errorsOf(issues(QWEN_27B, { dp: 2 }, { batch: 2 }, one), 'devices')[0].message).toBe(
      'TP × PP × DP = 2, but the rig has 1 device. They must be equal.',
    );
  });

  it('the batch must split evenly over DP: an error on batch', () => {
    const odd = issues(QWEN_27B, { tp: 4, dp: 2 }, { batch: 3 }).filter((i) => i.level === 'error');
    expect(odd).toEqual([{ field: 'batch', level: 'error', message: 'Batch 3 does not split evenly over DP 2. Use a multiple of 2.' }]);
    expect(issues(QWEN_27B, { tp: 4, dp: 2 }, { batch: 6 })).toEqual([]);
    expect(errorsOf(issues(QWEN_27B, { tp: 2, dp: 4 }, { batch: 2 }), 'batch')[0].message).toMatch(/^Batch 2 does not split evenly over DP 4/);
    // DP 1 accepts every batch.
    expect(issues(QWEN_27B, { tp: 8 }, { batch: 7 })).toEqual([]);
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
    const list = issues(QWEN_27B, { tp: 0, pp: 1.5, dp: 0 }, { batch: NaN });
    expect(list.map((i) => i.field).sort()).toEqual(['batch', 'dp', 'pp', 'tp']);
    expect(errorsOf(list, 'dp')[0].message).toBe('DP must be a whole number ≥ 1.');
  });

  it('world size and pp ≤ layers; EP is ignored on a dense model', () => {
    expect(errorsOf(issues(QWEN_27B, { tp: 4, pp: 4 }), 'devices')).toHaveLength(1);
    const many = nodeTopology({ devicesPerNode: 128, nodes: 1, intraNode: PCIE_5_X16 });
    expect(errorsOf(issues(QWEN_27B, { pp: 65 }, {}, many), 'pp').some((i) => /exceeds the 64 layers/.test(i.message))).toBe(true);
    expect(issues(QWEN_27B, { tp: 2, dp: 4, ep: 2 }, { batch: 4 }).filter((i) => i.level === 'error')).toEqual([]);
    expect(issues(QWEN_27B, { tp: 8, ep: 0 }).filter((i) => i.level === 'error')).toEqual([]);
  });

  it('EP on a MoE model: a whole number ≥ 1 that divides TP and the experts, errors on ep in UI words', () => {
    expect(issues(FLASH_NEXT, { tp: 8, ep: 8 })).toEqual([]);
    expect(issues(FLASH_NEXT, { tp: 2, ep: 2, dp: 4 }, { batch: 4 })).toEqual([]);
    expect(errorsOf(issues(FLASH_NEXT, { tp: 8, ep: 0 }), 'ep')[0].message).toBe('EP must be a whole number ≥ 1.');
    const three = issues(FLASH_NEXT, { tp: 8, ep: 3 }).filter((i) => i.level === 'error');
    expect(three.every((i) => i.field === 'ep')).toBe(true);
    expect(three.map((i) => i.message).slice(0, 2)).toEqual(['EP 3 must divide the 512 experts.', 'EP 3 must divide TP 8.']);
    // Plus the engine's placement rule for the EP group (3 does not tile a node of 8), on ep too.
    expect(three[2].message).toMatch(/^ep: a group of 3 devices does not tile/);
    // TP 1 × PP 2 on 2 devices: EP 2 needs a TP group of 2. The error is on EP, without any DP wording.
    const two = nodeTopology({ devicesPerNode: 2, nodes: 1, intraNode: PCIE_5_X16 });
    const onEp = issues(FLASH_NEXT, { pp: 2, ep: 2 }, {}, two).filter((i) => i.level === 'error');
    expect(onEp).toEqual([{ field: 'ep', level: 'error', message: 'EP 2 must divide TP 1.' }]);
    for (const i of onEp) expect(i.message).not.toMatch(/\bdp\b|DP/);
  });

  it('with DP > 1, EP may also be a multiple of TP that divides TP × DP (mode B), and the message says so', () => {
    // TP 2 × DP 4 on 8 devices: EP 4 and EP 8 span 2 and 4 replicas.
    expect(issues(FLASH_NEXT, { tp: 2, dp: 4, ep: 4 }, { batch: 4 })).toEqual([]);
    expect(issues(FLASH_NEXT, { tp: 2, dp: 4, ep: 8 }, { batch: 4 })).toEqual([]);
    const two = nodeTopology({ devicesPerNode: 2, nodes: 1, intraNode: PCIE_5_X16 });
    expect(issues(FLASH_NEXT, { dp: 2, ep: 2 }, { batch: 2 }, two)).toEqual([]);
    // EP 3 at TP 2 × DP 4: neither rule holds; the message names both.
    const three = errorsOf(issues(FLASH_NEXT, { tp: 2, dp: 4, ep: 3 }, { batch: 4 }), 'ep').map((i) => i.message);
    expect(three).toContain('EP 3 must divide TP 2, or be a multiple of TP 2 that divides TP × DP = 8.');
  });

  it('EP does not change the device count: it uses the TP and DP devices', () => {
    // TP 2 × EP 2 on 4 devices still leaves 2 unused: the device-count error.
    const four = nodeTopology({ devicesPerNode: 4, nodes: 1, intraNode: PCIE_5_X16 });
    const list = issues(FLASH_NEXT, { tp: 2, ep: 2 }, {}, four).filter((i) => i.level === 'error');
    expect(list).toEqual([{ field: 'devices', level: 'error', message: 'TP × PP × DP = 2, but the rig has 4 devices. They must be equal.' }]);
    expect(issues(FLASH_NEXT, { tp: 4, ep: 2 }, {}, four)).toEqual([]);
    expect(issues(FLASH_NEXT, { tp: 2, pp: 2, ep: 2 }, {}, four)).toEqual([]);
    expect(issues(FLASH_NEXT, { tp: 2, dp: 2, ep: 4 }, { batch: 2 }, four)).toEqual([]);
  });

  it('an expert-TP group the topology cannot model is an error on ep (expert-output-norm models)', () => {
    // DGX Spark ×4 on a ring, TP 4 EP 2: the 2 ranks sharing experts sit 2 nodes apart, which a ring does not model.
    const ring = resolveHardware({ ...SINGLE_DEVICE_HW, deviceId: 'dgx-spark', nodes: 4, cabling: 'ring' }).topology;
    const ep = errorsOf(issues(MUSE_SPARK, { tp: 4, ep: 2 }, {}, ring), 'ep');
    expect(ep).toHaveLength(1);
    expect(ep[0].message).toMatch(/^etp: expert-TP group .* ring fabric is not modeled/);
    expect(issues(MUSE_SPARK, { tp: 4, ep: 4 }, {}, ring).filter((i) => i.level === 'error')).toEqual([]);
    expect(issues(MUSE_SPARK, { tp: 4, ep: 1 }, {}, ring).filter((i) => i.level === 'error')).toEqual([]);
  });

  it('a missing link is a placement error on the degree that needs it', () => {
    const noLink = nodeTopology({ devicesPerNode: 2, nodes: 1 });
    expect(errorsOf(issues(QWEN_27B, { tp: 2 }, {}, noLink), 'tp')[0].message).toMatch(/^tp: .*no intraNode link/);
    expect(errorsOf(issues(QWEN_27B, { pp: 2 }, {}, noLink), 'pp')[0].message).toMatch(/^pp: /);
  });
});

describe('validDegrees', () => {
  it('Qwen3.8-27B on 8 devices', () => {
    const v = (field: 'tp' | 'pp' | 'dp' | 'ep', p: Partial<ParallelConfig> = {}) =>
      validDegrees(field, QWEN_27B.model, QWEN_27B.quant, { ...ONE, ...p }, LOAD, BOX_8);
    expect(v('tp')).toEqual([1, 2, 4, 8]);
    expect(v('pp')).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(v('tp', { pp: 2 })).toEqual([1, 2, 4, 8]);
    // Every DP is valid on its own: the device count and the batch split are shown on their own fields.
    expect(v('dp')).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(v('tp', { dp: 2 })).toEqual([1, 2, 4, 8]);
    // A dense model has one valid EP, whatever the state holds.
    expect(v('ep')).toEqual([1]);
    expect(v('ep', { ep: 4 })).toEqual([1]);
    expect(v('ep', { tp: 2, dp: 4 })).toEqual([1]);
  });

  it('Qwen3.8-Flash-Next (512 experts): valid EP are the divisors of TP', () => {
    const v = (field: 'tp' | 'pp' | 'ep', p: Partial<ParallelConfig>, n: number) =>
      validDegrees(field, FLASH_NEXT.model, FLASH_NEXT.quant, { ...ONE, ...p }, LOAD,
        nodeTopology({ devicesPerNode: n, nodes: 1, intraNode: PCIE_5_X16 }));
    expect(v('ep', { tp: 2 }, 2)).toEqual([1, 2]);
    expect(v('ep', { tp: 1, pp: 2 }, 2)).toEqual([1]);
    expect(v('ep', { tp: 4 }, 4)).toEqual([1, 2, 4]);
    expect(v('ep', { tp: 2, pp: 2 }, 4)).toEqual([1, 2]);
    // TP is listed as if EP were 1 (EP is chosen after TP); a TP the current EP does not divide shows the error on EP.
    expect(v('tp', { ep: 4 }, 4)).toEqual(v('tp', {}, 4));
  });

  it('Qwen3.8-Flash-Next with DP > 1: valid EP adds the mode-B multiples of TP that divide TP × DP', () => {
    const v = (p: Partial<ParallelConfig>, n: number) =>
      validDegrees('ep', FLASH_NEXT.model, FLASH_NEXT.quant, { ...ONE, ...p }, { ...LOAD, batch: 8 },
        nodeTopology({ devicesPerNode: n, nodes: 1, intraNode: PCIE_5_X16 }));
    // DP 1: only the divisors of TP (mode A).
    expect(v({ tp: 2 }, 2)).toEqual([1, 2]);
    expect(v({ dp: 1 }, 1)).toEqual([1]);
    // TP 1 × DP 2: EP 2 spans the two replicas.
    expect(v({ dp: 2 }, 2)).toEqual([1, 2]);
    // TP 2 × DP 2: 1, 2 (mode A) and 4 (mode B).
    expect(v({ tp: 2, dp: 2 }, 4)).toEqual([1, 2, 4]);
    // TP 2 × DP 4: 1, 2, then 4 and 8 (2 and 4 replicas); 6 is a multiple of TP that does not divide 8.
    expect(v({ tp: 2, dp: 4 }, 8)).toEqual([1, 2, 4, 8]);
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
  it('DGX Spark ×4 ring: an EP the ring cannot place falls back to 1 (Muse Spark, expert-output norm)', () => {
    expect(chooseDegrees(MUSE_SPARK.model, MUSE_SPARK.quant, LOAD, rig('dgx-spark-x4-ring'), { ep: 2 })).toEqual({ tp: 4, pp: 1, dp: 1, ep: 1 });
    expect(chooseDegrees(MUSE_SPARK.model, MUSE_SPARK.quant, LOAD, rig('dgx-spark-x4-ring'), { ep: 4 })).toEqual({ tp: 4, pp: 1, dp: 1, ep: 4 });
  });
  it.each(Object.keys(RIGS))('%s: TP × PP uses all 4 devices at DP 1, with no errors', (id) => {
    const topology = rig(id);
    for (const m of [QWEN_27B, FLASH_NEXT]) {
      const { tp, pp, dp, ep } = chooseDegrees(m.model, m.quant, LOAD, topology);
      expect(ep).toBe(1);
      expect(dp).toBe(1);
      expect(tp * pp).toBe(4);
      expect(fieldIssues(m.model, m.quant, { tp, pp, dp: 1, ep: 1 }, LOAD, topology).filter((i) => i.level === 'error')).toEqual([]);
    }
  });
  it('one device: 1 × 1', () => {
    expect(chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, nodeTopology({ devicesPerNode: 1, nodes: 1 }))).toEqual({ tp: 1, pp: 1, dp: 1, ep: 1 });
  });
  it('3 devices on a model whose heads do not divide by 3: pipeline only', () => {
    expect(chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, nodeTopology({ devicesPerNode: 3, nodes: 1, intraNode: PCIE_5_X16 }))).toEqual({ tp: 1, pp: 3, dp: 1, ep: 1 });
  });
  it('no valid split: TP = PP = 1', () => {
    expect(chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, nodeTopology({ devicesPerNode: 1000, nodes: 1, intraNode: PCIE_5_X16 }))).toEqual({ tp: 1, pp: 1, dp: 1, ep: 1 });
  });
  it('never raises EP; keeps a user-set EP while it stays valid, else falls back to 1', () => {
    const box = (n: number) => nodeTopology({ devicesPerNode: n, nodes: 1, intraNode: PCIE_5_X16 });
    const pick = (n: number, ep?: number) => chooseDegrees(FLASH_NEXT.model, FLASH_NEXT.quant, LOAD, box(n), { ep });
    expect(pick(4)).toEqual({ tp: 4, pp: 1, dp: 1, ep: 1 });
    expect(pick(2, 2)).toEqual({ tp: 2, pp: 1, dp: 1, ep: 2 });
    expect(pick(4, 2)).toEqual({ tp: 4, pp: 1, dp: 1, ep: 2 });
    expect(pick(2, 4)).toEqual({ tp: 2, pp: 1, dp: 1, ep: 1 });
    expect(pick(1, 2)).toEqual({ tp: 1, pp: 1, dp: 1, ep: 1 });
    // A dense model never gets EP > 1.
    expect(chooseDegrees(QWEN_27B.model, QWEN_27B.quant, LOAD, box(4), { ep: 2 }).ep).toBe(1);
  });
  it('never raises DP; keeps a user-set DP while it divides the devices and the batch, else falls back to 1', () => {
    const box = (n: number) => nodeTopology({ devicesPerNode: n, nodes: 1, intraNode: PCIE_5_X16 });
    const pick = (n: number, current: { dp?: number; ep?: number }, batch = 4) =>
      chooseDegrees(QWEN_27B.model, QWEN_27B.quant, { ...LOAD, batch }, box(n), current);
    expect(pick(4, {})).toEqual({ tp: 4, pp: 1, dp: 1, ep: 1 });
    // DP 2 kept: TP × PP fills the rest (4 devices: TP 2; 8 devices: TP 4).
    expect(pick(4, { dp: 2 })).toEqual({ tp: 2, pp: 1, dp: 2, ep: 1 });
    expect(pick(8, { dp: 2 })).toEqual({ tp: 4, pp: 1, dp: 2, ep: 1 });
    expect(pick(2, { dp: 2 })).toEqual({ tp: 1, pp: 1, dp: 2, ep: 1 });
    // DP 2 on 3 devices does not divide; DP 2 with batch 3 cannot split it: both fall back to DP 1.
    expect(pick(3, { dp: 2 })).toEqual({ tp: 1, pp: 3, dp: 1, ep: 1 });
    expect(pick(4, { dp: 2 }, 3)).toEqual({ tp: 4, pp: 1, dp: 1, ep: 1 });
    // DP and a mode-B EP are kept together on a MoE model (TP 1 × DP 2, EP 2 spans both replicas).
    expect(chooseDegrees(FLASH_NEXT.model, FLASH_NEXT.quant, { ...LOAD, batch: 2 }, box(2), { dp: 2, ep: 2 })).toEqual({ tp: 1, pp: 1, dp: 2, ep: 2 });
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
