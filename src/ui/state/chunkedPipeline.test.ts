/**
 * Chunked pipeline prefill (`ParallelConfig.chunkedPipelinePrefill`) in the UI state: the reducer (stored only as
 * true with PP > 1), the URL key `cpp` (and `dec.cpp` for the decode pool), and the state → SimInput step (emitted
 * only when on with PP > 1, so off gives exactly the input from before the switch), and the batch sweep (its decode
 * rates and memory fit match a full run with the switch on, without costing the chunked prefill). Qwen3.8-27B on one
 * 8 × B200 node at TP 2, PP 4, and on 4 × RTX 5090 at PP 4.
 */
import { describe, expect, it } from 'vitest';
import { mapHfModel } from '../../sim/models/hf/index';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import type { ParallelConfig } from '../../sim/ir/model';
import { DEFAULT_STATE, reduce, type AppState } from './appState';
import { buildSimInput, loadedFromHf, type LoadedModel } from './buildInput';
import { decodeState, encodeState } from './url';
import { batchSweep } from './sweep';
import { memoryFits } from './memoryVerdict';
import { simulate } from '../../sim/engine/simulate';

function load(id: string): LoadedModel {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${r.kind}`);
  return loadedFromHf({ kind: 'hf', id }, r);
}

const QWEN = load('Qwen/Qwen3.8-27B');
const NODE = 'd=b200&n=8&tp=2&pp=4';
const TP2PP4: ParallelConfig = { tp: 2, pp: 4, dp: 1, ep: 1 };

/** The state of a query (asserts nothing was dropped). */
function decoded(query: string): AppState {
  const { state, dropped } = decodeState(query);
  expect(dropped).toEqual([]);
  return state;
}
const BASE = decoded(NODE);

describe('reducer', () => {
  it('stores true with PP > 1; false and undefined remove it', () => {
    const on = reduce(BASE, { type: 'setParallel', parallel: { chunkedPipelinePrefill: true } });
    expect(on.parallel).toEqual({ ...TP2PP4, chunkedPipelinePrefill: true });
    for (const off of [false, undefined]) {
      const s = reduce(on, { type: 'setParallel', parallel: { chunkedPipelinePrefill: off } });
      expect(s).toEqual(BASE);
      expect('chunkedPipelinePrefill' in s.parallel).toBe(false);
    }
    // Other degrees leave it as it is while PP stays above 1.
    expect(
      reduce(on, { type: 'setParallel', parallel: { tp: 4, pp: 2 } }).parallel
        .chunkedPipelinePrefill,
    ).toBe(true);
  });

  it('PP 1 drops it (nothing to pipeline), and it cannot be turned on at PP 1', () => {
    const on = reduce(BASE, { type: 'setParallel', parallel: { chunkedPipelinePrefill: true } });
    const pp1 = reduce(on, { type: 'setParallel', parallel: { pp: 1, tp: 8 } });
    expect('chunkedPipelinePrefill' in pp1.parallel).toBe(false);
    const tryOn = reduce(DEFAULT_STATE, {
      type: 'setParallel',
      parallel: { chunkedPipelinePrefill: true },
    });
    expect(tryOn).toEqual(DEFAULT_STATE);
  });
});

describe('URL', () => {
  it('round-trips cpp=1 right after the degrees', () => {
    const s = decoded(`${NODE}&cpp=1`);
    expect(s.parallel).toEqual({ ...TP2PP4, chunkedPipelinePrefill: true });
    expect(encodeState(s)).toBe('d=b200&n=8&tp=2&pp=4&cpp=1');
    expect(decoded(encodeState(s))).toEqual(s);
  });

  it('without cpp the URL and state are as before; the default state encodes to nothing', () => {
    expect(encodeState(BASE)).toBe(NODE);
    expect('chunkedPipelinePrefill' in BASE.parallel).toBe(false);
    expect(encodeState(DEFAULT_STATE)).toBe('');
  });

  it('drops any other value and cpp at PP 1', () => {
    expect(decodeState(`${NODE}&cpp=0`).dropped).toEqual(['cpp=0']);
    expect(decodeState(`${NODE}&cpp=on`).dropped).toEqual(['cpp=on']);
    const pp1 = decodeState('d=b200&n=8&tp=8&cpp=1');
    expect(pp1.dropped).toEqual(['cpp=1']);
    expect('chunkedPipelinePrefill' in pp1.state.parallel).toBe(false);
  });

  it('the decode pool keeps its own copy as dec.cpp', () => {
    const s = decoded(`${NODE}&cpp=1&pd=1&dec.pp=2&dec.cpp=1`);
    expect(s.disaggregation?.decodePool.parallel.chunkedPipelinePrefill).toBe(true);
    expect(encodeState(s)).toContain('&dec.pp=2&dec.cpp=1');
    expect(decoded(encodeState(s))).toEqual(s);
  });
});

describe('buildSimInput', () => {
  it('off: the input is exactly the one without the key (and has none)', () => {
    const base = buildSimInput(BASE, QWEN);
    expect(base.kind).toBe('ok');
    if (base.kind === 'ok') {
      expect(base.input.parallel).toEqual(TP2PP4);
      expect('chunkedPipelinePrefill' in base.input.parallel).toBe(false);
    }
    // A stray false (not normalized) changes nothing either.
    expect(
      buildSimInput(
        { ...BASE, parallel: { ...BASE.parallel, chunkedPipelinePrefill: false } },
        QWEN,
      ),
    ).toEqual(base);
  });

  it('on with PP > 1 reaches the engine; at PP 1 it never does', () => {
    const on = buildSimInput(decoded(`${NODE}&cpp=1`), QWEN);
    expect(on.kind).toBe('ok');
    if (on.kind === 'ok')
      expect(on.input.parallel).toEqual({ ...TP2PP4, chunkedPipelinePrefill: true });
    const pp1State: AppState = {
      ...decoded('d=b200&n=8&tp=8'),
    };
    const pp1 = buildSimInput(
      { ...pp1State, parallel: { ...pp1State.parallel, chunkedPipelinePrefill: true } },
      QWEN,
    );
    expect(pp1).toEqual(buildSimInput(pp1State, QWEN));
  });
});

describe('batchSweep', () => {
  it('has the same decode rates as without the switch', () => {
    const off = buildSimInput(decoded(`${NODE}&p=65536`), QWEN);
    const on = buildSimInput(decoded(`${NODE}&p=65536&cpp=1`), QWEN);
    if (off.kind !== 'ok' || on.kind !== 'ok') throw new Error('invalid test state');
    expect(on.input.parallel.chunkedPipelinePrefill).toBe(true);
    const rates = (input: typeof on.input) =>
      batchSweep(input).map((p) => [p.batch, p.decodeTokensPerSec, p.perSeqTokensPerSec]);
    expect(rates(on.input)).toEqual(rates(off.input));
  });

  it('fits exactly where a full run with the switch on fits, also with several sequences per microbatch', () => {
    // 4 × RTX 5090 at PP 4, P = 8k, C = 4k: from batch 8 on a microbatch holds b_mb > 1 sequences, and the chunked
    // schedule's activation peak (b_mb × C tokens) is larger than the whole-prompt one; at batch 64 that decides the
    // fit (with the switch off the sweep said it fits).
    const built = buildSimInput(decoded('d=rtx-5090&n=4&pp=4&p=8192&pc=4096&cpp=1'), QWEN);
    if (built.kind !== 'ok') throw new Error('invalid test state');
    const sweep = batchSweep(built.input);
    expect(sweep.map((p) => p.batch)).toEqual([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]);
    for (const point of sweep) {
      const full = simulate({
        ...built.input,
        workload: { ...built.input.workload, batch: point.batch },
      });
      expect(full.prefill.chunkedPipeline).toBeDefined();
      expect([point.batch, point.fits]).toEqual([
        point.batch,
        memoryFits({ stages: full.memory.perStage, tiers: full.memory.tiers }),
      ]);
      expect(point.decodeTokensPerSec).toBe(full.decode.tokensPerSec);
    }
    expect(sweep.find((p) => p.batch === 64)?.fits).toBe(false);
  });
});
