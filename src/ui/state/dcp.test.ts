/**
 * Decode context parallelism (DCP) in the UI state: the reducer, the URL key `dcp`, the per-field checks and the
 * "Valid DCP" hint, the degree choice on hardware changes, and the state → SimInput step. Two mapped fixtures:
 * Kimi-K3 (MLA: one latent KV head, so DCP may be any divisor of TP) and Qwen3.8-27B (GQA with 4 KV heads, so at
 * TP 8 each KV head has 2 copies and DCP may be 1 or 2), both on one 8 × B200 node.
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import type { ModelSpec, ParallelConfig } from '../../sim/ir/model';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { buildSimInput, loadedFromHf, type LoadedModel } from './buildInput';
import { resolveHardware } from './hardware';
import {
  chooseDegrees,
  effectiveParallel,
  fieldIssues,
  placementField,
  shardingField,
  uiShardingMessage,
  validDegrees,
} from './parallelism';
import { decodeState, encodeState } from './url';

function load(id: string): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id }, r);
}

const KIMI = load('moonshotai/Kimi-K3');
const QWEN = load('Qwen/Qwen3.8-27B');
const NODE = 'd=b200&n=8&tp=8';

/** The state of a query (asserts nothing was dropped). */
function decoded(query: string): AppState {
  const { state, dropped } = decodeState(query);
  expect(dropped).toEqual([]);
  return state;
}
const QWEN_STATE = decoded(NODE);
const TOPOLOGY = resolveHardware(QWEN_STATE.hw).topology;
const WORKLOAD = QWEN_STATE.workload;
const TP8: ParallelConfig = { tp: 8, pp: 1, dp: 1, ep: 1 };

/** The DCP field's messages for a model and parallel config. */
function dcpMessages(
  loaded: LoadedModel,
  parallel: ParallelConfig,
  model: ModelSpec = loaded.model,
): string[] {
  return fieldIssues(model, loaded.quant.quant, parallel, WORKLOAD, TOPOLOGY)
    .filter((i) => i.field === 'dcp')
    .map((i) => i.message);
}

describe('reducer', () => {
  it('stores DCP > 1, removes 1 and undefined, and keeps an invalid value while editing', () => {
    const on = reduce(QWEN_STATE, { type: 'setParallel', parallel: { dcp: 2 } });
    expect(on.parallel).toEqual({ ...TP8, dcp: 2 });
    for (const off of [1, undefined]) {
      const s = reduce(on, { type: 'setParallel', parallel: { dcp: off } });
      expect(s).toEqual(QWEN_STATE);
      expect('dcp' in s.parallel).toBe(false);
    }
    expect(reduce(on, { type: 'setParallel', parallel: { dcp: 2.5 } }).parallel.dcp).toBe(2.5);
    expect(reduce(on, { type: 'setParallel', parallel: { dcp: 0 } }).parallel.dcp).toBe(0);
    // Other degrees leave DCP as it is.
    expect(reduce(on, { type: 'setParallel', parallel: { tp: 4, pp: 2 } }).parallel.dcp).toBe(2);
  });

  it('entering DP attention + EP drops DCP; leaving it keeps DCP absent', () => {
    const on = reduce(QWEN_STATE, { type: 'setParallel', parallel: { dcp: 2 } });
    const dep = reduce(on, { type: 'setParallelLayout', layout: 'dpAttentionEp' });
    expect(dep.parallel).toEqual({ tp: 1, pp: 1, dp: 8, ep: 8 });
    expect(reduce(dep, { type: 'setParallel', parallel: { dcp: 2 } }).parallel).toEqual(
      dep.parallel,
    );
    const custom = reduce(dep, { type: 'setParallelLayout', layout: 'custom' });
    expect('dcp' in custom.parallel).toBe(false);
  });
});

describe('URL', () => {
  it('round-trips dcp right after the degrees', () => {
    const s = decoded(`${NODE}&dcp=2`);
    expect(s.parallel).toEqual({ ...TP8, dcp: 2 });
    expect(encodeState(s)).toBe('d=b200&n=8&tp=8&dcp=2');
    expect(decoded(encodeState(s))).toEqual(s);
  });

  it('dcp=1 decodes as off (not a dropped value); without dcp the URL and state are as before', () => {
    expect(decoded(`${NODE}&dcp=1`)).toEqual(QWEN_STATE);
    expect(encodeState(QWEN_STATE)).toBe('d=b200&n=8&tp=8');
    expect('dcp' in QWEN_STATE.parallel).toBe(false);
  });

  it('drops an invalid dcp and any dcp with lay=dep', () => {
    expect(decodeState(`${NODE}&dcp=0`).dropped).toEqual(['dcp=0']);
    expect(decodeState(`${NODE}&dcp=x`).dropped).toEqual(['dcp=x']);
    const dep = decodeState('m=zai-org/GLM-5.3-Flash&d=b200&n=8&dp=8&ep=8&lay=dep&dcp=2');
    expect(dep.dropped).toEqual(['dcp=2']);
    expect(dep.state.parallelLayout).toBe('dpAttentionEp');
    expect('dcp' in dep.state.parallel).toBe(false);
  });
});

describe('effectiveParallel', () => {
  it('keeps DCP on every model, except in DP attention + EP', () => {
    const p = { ...TP8, dcp: 2 };
    expect(effectiveParallel(p, QWEN.model)).toEqual(p);
    expect(effectiveParallel(p, KIMI.model)).toEqual(p);
    expect(effectiveParallel(p, KIMI.model, 'dpAttentionEp')).toEqual(TP8);
    expect(effectiveParallel(TP8, QWEN.model)).toEqual(TP8);
  });
});

describe('per-field checks and hints', () => {
  it('MLA: DCP may be any divisor of TP', () => {
    expect(validDegrees('dcp', KIMI.model, KIMI.quant.quant, TP8, WORKLOAD, TOPOLOGY)).toEqual([
      1, 2, 4, 8,
    ]);
    expect(dcpMessages(KIMI, { ...TP8, dcp: 8 })).toEqual([]);
    expect(dcpMessages(KIMI, { ...TP8, dcp: 3 })).toEqual(['DCP 3 must divide TP 8.']);
    // TP 4: only 1, 2 and 4.
    const tp4 = { tp: 4, pp: 2, dp: 1, ep: 1 };
    expect(validDegrees('dcp', KIMI.model, KIMI.quant.quant, tp4, WORKLOAD, TOPOLOGY)).toEqual([
      1, 2, 4,
    ]);
  });

  it('GQA: DCP must also divide TP ÷ KV heads (4 KV heads at TP 8: 2 copies each)', () => {
    expect(validDegrees('dcp', QWEN.model, QWEN.quant.quant, TP8, WORKLOAD, TOPOLOGY)).toEqual([
      1, 2,
    ]);
    expect(dcpMessages(QWEN, { ...TP8, dcp: 2 })).toEqual([]);
    expect(dcpMessages(QWEN, { ...TP8, dcp: 4 })).toEqual([
      'DCP 4 must divide TP ÷ KV heads = 2: only the GPUs holding copies of the same KV head can split it.',
    ]);
    // TP 4 = KV heads: no copies, so no DCP.
    const tp4 = { tp: 4, pp: 2, dp: 1, ep: 1 };
    expect(validDegrees('dcp', QWEN.model, QWEN.quant.quant, tp4, WORKLOAD, TOPOLOGY)).toEqual([1]);
  });

  it('a whole number ≥ 1; a bad DCP does not change the TP hint', () => {
    expect(dcpMessages(QWEN, { ...TP8, dcp: 0 })).toEqual(['DCP must be a whole number ≥ 1.']);
    expect(dcpMessages(QWEN, { ...TP8, dcp: 2.5 })).toEqual(['DCP must be a whole number ≥ 1.']);
    const quant = QWEN.quant.quant;
    expect(validDegrees('tp', QWEN.model, quant, { ...TP8, dcp: 4 }, WORKLOAD, TOPOLOGY)).toEqual(
      validDegrees('tp', QWEN.model, quant, TP8, WORKLOAD, TOPOLOGY),
    );
  });

  it('a model whose attention cannot be split by token gets the DCP message on the DCP field', () => {
    // Kimi-K3 with its MLA blocks made sparse (DSA top-k): the engine rejects DCP there.
    const sparse: ModelSpec = {
      ...KIMI.model,
      layers: KIMI.model.layers.map((l) => ({
        ...l,
        blocks: l.blocks.map((b) =>
          b.kind === 'mla_attention' ? { ...b, sparse: { topK: 2048 } } : b,
        ),
      })) as ModelSpec['layers'],
    };
    expect(dcpMessages(KIMI, { ...TP8, dcp: 2 }, sparse)).toEqual([
      'DCP is not supported with sparse attention.',
    ]);
  });

  it('attributes and rewords the engine DCP messages by their wording', () => {
    const cases: [string, string][] = [
      ['parallel.dcp (3) must divide tp (8)', 'DCP 3 must divide TP 8.'],
      [
        'attn dcp (4) must divide tp / kvHeads (2): KV heads are replicated on 2 ranks and DCP shards that copy',
        'DCP 4 must divide TP ÷ KV heads = 2: only the GPUs holding copies of the same KV head can split it.',
      ],
      [
        'attn dcp (2) > 1 is not supported with sparse (top-k) attention',
        'DCP is not supported with sparse attention.',
      ],
      [
        'attn dcp (2) > 1 is not supported with compressed sparse attention (CSA)',
        'DCP is not supported with compressed sparse attention.',
      ],
      [
        'attn dcp (2) > 1 is not supported with a relative-position bias or K/V short convs',
        'DCP is not supported with relative-position bias or K/V convs.',
      ],
      [
        'dcp (2) > 1 but the model has no full-attention (MLA or GQA) block to shard',
        'DCP needs a model with MLA or full-attention GQA layers.',
      ],
    ];
    for (const [engine, ui] of cases) {
      expect(shardingField(engine)).toBe('dcp');
      expect(uiShardingMessage(engine)).toBe(ui);
    }
    // The EP rule is not mistaken for a DCP one, nor the other way round.
    expect(shardingField('moe: ep (3) must divide tp (2)')).toBe('ep');
    expect(uiShardingMessage('ep (3) must divide tp (2)')).toBe('EP 3 must divide TP 2.');
    expect(placementField('dcp: group of 16 spans 2 nodes')).toBe('dcp');
  });

  it('chooseDegrees keeps DCP only when the chosen TP allows it', () => {
    const quant = QWEN.quant.quant;
    // 8 devices: TP 8, so DCP 2 is kept.
    expect(chooseDegrees(QWEN.model, quant, WORKLOAD, TOPOLOGY, { dcp: 2 })).toMatchObject({
      tp: 8,
      dcp: 2,
    });
    // 4 devices: TP 4 = KV heads, so DCP is turned off (an explicit undefined for the reducer).
    const four = resolveHardware(decoded('d=b200&n=4').hw).topology;
    const chosen = chooseDegrees(QWEN.model, quant, WORKLOAD, four, { dcp: 2 });
    expect(chosen.tp).toBe(4);
    expect('dcp' in chosen && chosen.dcp === undefined).toBe(true);
    // Without a current DCP the result has no dcp key, as before.
    expect('dcp' in chooseDegrees(QWEN.model, quant, WORKLOAD, TOPOLOGY, {})).toBe(false);
  });
});

describe('buildSimInput', () => {
  it('with DCP off, the input is exactly the one without the dcp key (and has none)', () => {
    const base = buildSimInput(QWEN_STATE, QWEN);
    expect(base.kind).toBe('ok');
    expect(buildSimInput(decodeState(`${NODE}&dcp=1`).state, QWEN)).toEqual(base);
    expect(
      buildSimInput({ ...QWEN_STATE, parallel: { ...QWEN_STATE.parallel, dcp: 1 } }, QWEN),
    ).toEqual(base);
    if (base.kind === 'ok') {
      expect(base.input.parallel).toEqual(TP8);
      expect('dcp' in base.input.parallel).toBe(false);
    }
    const kimi = { ...decoded(`m=moonshotai/Kimi-K3&${NODE}`) };
    const def = buildSimInput({ ...DEFAULT_STATE, model: kimi.model }, KIMI);
    if (def.kind === 'ok') expect('dcp' in def.input.parallel).toBe(false);
  });

  it('DCP > 1 reaches the engine as parallel.dcp; an invalid one stops on the DCP field', () => {
    const on = buildSimInput(decoded(`${NODE}&dcp=2`), QWEN);
    expect(on.kind).toBe('ok');
    if (on.kind === 'ok') expect(on.input.parallel).toEqual({ ...TP8, dcp: 2 });
    const bad = buildSimInput(decoded(`${NODE}&dcp=4`), QWEN);
    expect(bad.kind).toBe('issues');
    expect(bad.issues.map((i) => i.field)).toEqual(['dcp']);
  });
});
