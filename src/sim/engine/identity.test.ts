/**
 * Deployment identity pin (design/offload-specdec-latency.md §8.2 ID-1/ID-2).
 * Originally generated on 68ae0f4; deliberately regenerated for audit G1/G2/G3/G4/G7/G8, for the chunked-prefill
 * activation peak of the memory fit (review F11: only W2/batch8's memory leaves changed, 8 × 8192 tokens > the 8192 chunk),
 * and for the MoE routing review fixes (docs/review/practicality-review.md F5, F12): only MoE cases moved (178 of 323 keys).
 * ep = 1: decode +0.02 … +13.1 % (expert weights of U = E·(1 − (1 − k/E)^T) expected distinct experts at batch 4
 * instead of min(E, T·k); batch 1 unchanged, U(1) = k), prefill 0 … +0.02 % (U a hair below E at 1024 tokens);
 * ep > 1 (ep2dp2, G-e): prefill −0.1 … −12.4 % (EP load imbalance ρ = 1.2 on the busiest device's expert rows),
 * decode −0.03 … +8.4 %. Every dense case is unchanged bit for bit.
 * JSON round-trips finite IEEE-754 doubles exactly. Numeric leaves match within MAX_ULPS units in the last place (32),
 * not a relative tolerance: V8's Math.pow / exp / log differ in the last bits across Node majors and CPU architectures
 * (CI runs x64 Linux, the pin was made on arm64 macOS). Strings, booleans, zeros and signs match exactly.
 * New result fields intentionally do not enter this projection. Never regenerate to hide a regression.
 * Regenerate deliberately: UPDATE_DEPLOYMENT_IDENTITY=1 bun run test src/sim/engine/identity.test.ts
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { ModelSpec, QuantConfig } from '../ir/model';
import { countParams } from '../ir/params';
import {
  fixtureRepoFiles,
  listFixturePaths,
  listFixtureRepos,
} from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import * as references from '../models/reference';
import { simulate } from './simulate';
import type { PhaseResult, SimInput, SimResult } from './types';
import { validateInput } from './validate';

const { REF_DENSE_8B, REF_MOE_30B_A3B, TINY_HYBRID, TINY_MLA_SHARED } = references;
const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const P1 = { tp: 1, pp: 1, dp: 1, ep: 1 };
const GPU = { device: RTX_5090, topology: singleDevice(RTX_5090) };
const CLUSTER = { device: SYNTH_GPU, topology: SYNTH_CLUSTER };
const UPDATE =
  import.meta.env.UPDATE_DEPLOYMENT_IDENTITY === '1' ||
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
    ?.UPDATE_DEPLOYMENT_IDENTITY === '1';
const RAW = import.meta.glob<string>('./__fixtures__/identity.json', {
  query: '?raw',
  import: 'default',
  eager: true,
});
const FIXTURE_URL = new URL('./__fixtures__/identity.json', import.meta.url);

type Leaf = number | string | boolean;
type Pin = Record<string, Leaf>;
type Fixture = Record<string, Pin>;

function input(over: Partial<SimInput> = {}): SimInput {
  return {
    model: REF_DENSE_8B,
    quant: BF16,
    parallel: P1,
    workload: { batch: 1, promptLen: 4096, decodeContextLen: 1024 },
    hardware: GPU,
    ...over,
  };
}

function hf(repo: string, dir = ''): Pick<SimInput, 'model' | 'quant'> {
  const mapped = mapHfModel(fixtureRepoFiles(repo, { dir }));
  if (mapped.kind !== 'ok') throw new Error(`${repo}: ${JSON.stringify(mapped)}`);
  return { model: mapped.model, quant: mapped.quant.quant };
}

/** Explicit inputs, not results recomputed from each other: preserve the original e2e workloads. */
function cases(): Record<string, SimInput> {
  const out: Record<string, SimInput> = {};
  const add = (key: string, over: Partial<SimInput> = {}) => {
    out[key] = input(over);
  };
  add('G-a');
  add('G-b', { hardware: CLUSTER, parallel: { ...P1, tp: 4 } });
  add('G-c', {
    hardware: CLUSTER,
    parallel: { ...P1, tp: 2, pp: 2 },
    workload: { batch: 2, promptLen: 4096, decodeContextLen: 1024 },
  });
  add('G-d', {
    hardware: CLUSTER,
    parallel: { ...P1, pp: 2 },
    workload: { batch: 4, promptLen: 4096, decodeContextLen: 1024 },
  });
  add('G-e', {
    model: TINY_HYBRID,
    hardware: CLUSTER,
    parallel: { ...P1, ep: 2, dp: 2 },
    workload: { batch: 4, promptLen: 512, decodeContextLen: 1024 },
  });
  add('G-f', {
    model: TINY_HYBRID,
    hardware: CLUSTER,
    parallel: { ...P1, tp: 2, pp: 2 },
    workload: { batch: 2, promptLen: 512, decodeContextLen: 1024 },
  });
  add('G-g', { hardware: CLUSTER, parallel: { ...P1, tp: 8, pp: 2 } });
  add('G-i', { hardware: CLUSTER, parallel: { ...P1, tp: 16 } });
  for (const [key, repo] of [
    ['G-h1', 'Qwen/Qwen3.8-Flash-Next'],
    ['G-h2', 'deepseek-ai/DeepSeek-V4.1-Flash'],
  ]) {
    add(key, { ...hf(repo), workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 } });
  }
  for (const key of ['G-a', 'G-c', 'G-f'])
    out[`${key}/overlap-none`] = { ...out[key], options: { overlap: 'none' } };
  add('W1', {
    model: REF_MOE_30B_A3B,
    quant: {
      ...BF16,
      modules: { routed_experts: { weights: { dtype: 'int4', extraBitsPerElem: 0.5 } } },
    },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
  });
  add('W2', { workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 } });
  add('W2/batch8', { workload: { batch: 8, promptLen: 8192, decodeContextLen: 8192 } });
  add('T9');
  add('T9/fusion-ideal', { options: { elementwiseFusion: 'ideal' } });
  for (const batch of [1, 8, 64])
    add(`T10/b${batch}`, { workload: { batch, promptLen: 128, decodeContextLen: 1024 } });
  add('T10/no-fit', { workload: { batch: 64, promptLen: 128, decodeContextLen: 8192 } });
  for (const tp of [1, 8])
    for (const overlap of ['ideal', 'none'] as const)
      add(`T11/tp${tp}/${overlap}`, {
        hardware: tp === 1 ? { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU) } : CLUSTER,
        parallel: { ...P1, tp },
        workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
        options: { overlap },
      });
  add('T15/sparse', { quant: { ...BF16, weightSparsity: '2:4' } });
  add('T15/nvfp4', {
    quant: { weights: { dtype: 'nvfp4' }, activations: 'nvfp4', kvCache: { dtype: 'bf16' } },
  });
  for (const model of [TINY_HYBRID, references.TINY_MLA_SPARSE])
    for (const tp of [1, 2])
      add(`T18/${model.name}/tp${tp}`, {
        model,
        hardware: CLUSTER,
        parallel: { ...P1, tp },
        workload: { batch: 2, promptLen: 256, decodeContextLen: 1000 },
      });
  add('T19/modifier', {
    hardware: {
      ...GPU,
      deviceModifiers: [
        {
          kind: 'memoryBandwidth',
          bandwidthBytesPerSec: 2e12,
          note: 'synthetic memory overclock',
          sources: [
            {
              url: 'synthetic://test',
              title: 'Synthetic',
              fields: ['memory.bandwidthBytesPerSec'],
            },
          ],
        },
      ],
    },
  });
  for (const model of [
    references.TINY_CSA,
    references.TINY_HC_NGRAM,
    references.TINY_KDA,
    TINY_MLA_SHARED,
    references.TINY_INKLING,
  ])
    add(`shortlist/${model.name}`, {
      model,
      hardware: CLUSTER,
      parallel: { ...P1, tp: 2, pp: 2 },
      workload: { batch: 2, promptLen: 256, decodeContextLen: 512 },
    });
  add('shortlist/cross-stage-warning', {
    model: TINY_MLA_SHARED,
    hardware: CLUSTER,
    parallel: { ...P1, pp: 3 },
  });
  const models: [string, ModelSpec, QuantConfig][] = Object.entries(references).map(
    ([key, model]) => [key, model, BF16],
  );
  // Include quant subdirectories too, not just the architecture fixtures.
  const dirs = [
    ...new Set(
      listFixturePaths()
        .map((p) => p.slice(2, p.lastIndexOf('/') + 1))
        .filter(Boolean),
    ),
  ];
  for (const dir of ['', ...dirs.map((d) => d.replace(/\/$/, ''))])
    for (const repo of listFixtureRepos(dir)) {
      const m = hf(repo, dir);
      models.push([`hf/${dir}/${repo}`, m.model, m.quant]);
      add(`hf5090/${dir}/${repo}`, m);
    }
  for (const [key, model, quant] of models) {
    const variants = [
      ['single', { device: SYNTH_GPU, topology: singleDevice(SYNTH_GPU) }, P1],
      ['tp2', CLUSTER, { ...P1, tp: 2 }],
      ['pp2', CLUSTER, { ...P1, pp: 2 }],
      ['ep2dp2', CLUSTER, { ...P1, ep: 2, dp: 2 }],
    ] as const;
    for (const [variant, hardware, parallel] of variants) {
      const candidate = input({
        model,
        quant,
        hardware,
        parallel,
        workload: { batch: 4, promptLen: 256, decodeContextLen: 1024 },
      });
      if (validateInput(candidate).length === 0) out[`models/${key}/${variant}`] = candidate;
    }
  }
  return out;
}

/** Select pre-existing fields only; units follow the result names (seconds, tokens/s, bytes or counts). */
function pin(r: SimResult): Pin {
  const out: Pin = {};
  function phase(p: PhaseResult): void {
    for (const key of [
      'cycleSec',
      'latencySec',
      'tokensPerSec',
      'perSequenceTokensPerSec',
      'microbatches',
    ] as const)
      out[`${p.phase}.${key}`] = p[key];
    for (const [i, s] of p.stages.entries()) {
      const prefix = `${p.phase}.stage${i}`;
      for (const key of [
        'sec',
        'computeSec',
        'commLatencySec',
        'commBandwidthSec',
        'exposedCommSec',
      ] as const)
        out[`${prefix}.${key}`] = s[key];
      out[`${prefix}.sendToNext.sec`] = s.sendToNext?.sec ?? 0;
      for (const [j, l] of s.layers.entries()) {
        out[`${prefix}.layer${j}.${l.name}.sec`] = l.sec;
      }
    }
  }
  phase(r.prefill);
  phase(r.decode);
  for (const [i, s] of r.memory.perStage.entries())
    for (const key of [
      'weightBytes',
      'sequenceStateBytes',
      'activationPeakBytes',
      'totalBytes',
      'fits',
    ] as const)
      out[`memory.stage${i}.${key}`] = s[key];
  out['memory.maxSequencesPerReplica'] = Number.isFinite(r.memory.maxSequencesPerReplica)
    ? r.memory.maxSequencesPerReplica
    : String(r.memory.maxSequencesPerReplica);
  out['memory.fits'] = r.memory.fits;
  out.warnings = JSON.stringify(r.warnings);
  return out;
}

/**
 * Most units in the last place (ULP) a numeric leaf may move and still match the pin (see the file header). V8's `**`
 * (Math.pow), exp and log are not correctly rounded and differ in the last bits between Node majors (22 vs 24) and
 * between arm64 and x64 builds of one Node; summed over the layers that reached up to 18 ULP (≈ 2e-15 relative, Node
 * 22 vs the Node 24 arm64 pin). 32 ULP is ≈ 7e-15 relative, far below what any intended model change moves.
 */
const MAX_ULPS = 32;
const BITS = new DataView(new ArrayBuffer(8));

/** IEEE-754 bit pattern of |x| as an integer: adjacent doubles of one sign differ by exactly 1. */
function magnitudeBits(x: number): bigint {
  BITS.setFloat64(0, Math.abs(x));
  return BITS.getBigUint64(0);
}

/**
 * True if two leaves match: strings and booleans exactly; numbers bit for bit (Object.is, so +0 ≠ −0 and NaN = NaN),
 * or both finite and non-zero with the same sign and at most {@link MAX_ULPS} representable doubles apart.
 */
function leavesMatch(expected: Leaf | undefined, actual: Leaf | undefined): boolean {
  if (Object.is(expected, actual)) return true;
  if (typeof expected !== 'number' || typeof actual !== 'number') return false;
  if (!Number.isFinite(expected) || !Number.isFinite(actual)) return false;
  if (expected === 0 || actual === 0 || Math.sign(expected) !== Math.sign(actual)) return false;
  const ulps = magnitudeBits(expected) - magnitudeBits(actual);
  return (ulps < 0n ? -ulps : ulps) <= BigInt(MAX_ULPS);
}

function mismatches(expected: Pin, actual: Pin): string[] {
  return [...new Set([...Object.keys(expected), ...Object.keys(actual)])]
    .filter((key) => !leavesMatch(expected[key], actual[key]))
    .map((key) => `${key}: expected ${String(expected[key])}, got ${String(actual[key])}`);
}

const inputs = cases();

/** §8.3 as printed in the design: decode cycle/rate, prefill cycle/TTFT/rate, per-stage total bytes, fit. */
const DESIGN_GOLDENS: [string, number, number, number, number, number, number[], boolean][] = [
  [
    'G-a',
    0.008538328428571402,
    117.118942936623,
    0.31770528850866675,
    0.31770528850866675,
    12892.451426373611,
    [17287440384],
    true,
  ],
  [
    'G-b',
    0.004123760767999988,
    242.49709337164026,
    0.1800527418072269,
    0.1800527418072269,
    22748.889902411895,
    [4347488256],
    true,
  ],
  [
    'G-c',
    0.008421233535999988,
    237.4948980396027,
    0.34323770569445416,
    0.5146386971656812,
    23866.84173705675,
    [4548502528, 4548510720],
    true,
  ],
  [
    'G-d',
    0.01670985011199998,
    239.3797654191672,
    1.3158195622538693,
    1.9732903249488039,
    12451.555266389125,
    [10002670592, 10002678784],
    true,
  ],
  // G-e moved on purpose with the MoE routing review fixes (docs/review/practicality-review.md F5, F12): ep 2 now
  // applies the EP load imbalance ρ = 1.2 to the busiest device's expert rows (prefill 8,530,004 → 8,485,274 tok/s,
  // activation peak 2,097,152 → 2,519,040 B, ≈ ×1.2 from the expert GEMM's rows), and decode reads the expected distinct
  // experts U_dev instead of min(E_loc, A) (188,600.7 → 188,746.5 tok/s). Design §8.3 prints the same numbers.
  [
    'G-e',
    0.000021192448000000002,
    188746.48176557987,
    0.00024135932787548635,
    0.00024135932787548635,
    8485273.877861196,
    [10179008],
    true,
  ],
  [
    'G-f',
    0.000031463856,
    63564.999788964196,
    0.00018910956775097272,
    0.0002818100477509727,
    5414850.301749119,
    [3323008, 3958592],
    true,
  ],
  [
    'G-g',
    0.0024470846560000017,
    408.6495322293416,
    0.04824354397980686,
    0.09707363224761373,
    84902.55197077664,
    [1162519552, 1162527744],
    true,
  ],
  [
    'G-i',
    0.002922674992000001,
    342.15231003694157,
    0.060721280187806884,
    0.060721280187806884,
    67455.75829974837,
    [1238329344],
    true,
  ],
  [
    'G-h1',
    0.008356467181537718,
    119.66779480799501,
    0.6093556337025076,
    0.6093556337025076,
    13443.709300305578,
    [354839488512],
    false,
  ],
  [
    'G-h2',
    0.008979504281438205,
    111.3647222227099,
    0.8779083600783636,
    0.8779083600783636,
    9331.270064757975,
    [502483014272],
    false,
  ],
];

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

describe('deployment identity, knobs absent (after latency audit G1–G8, G6 excluded)', () => {
  if (UPDATE) {
    it('writes the baseline fixture, deliberately', async () => {
      const fs = (await import(/* @vite-ignore */ 'node:fs' as string)) as {
        mkdirSync(p: URL, o: { recursive: boolean }): void;
        writeFileSync(p: URL, d: string): void;
      };
      const fixture: Fixture = Object.fromEntries(
        Object.entries(inputs).map(([key, value]) => [key, pin(simulate(value))]),
      );
      fs.mkdirSync(new URL('./', FIXTURE_URL), { recursive: true });
      fs.writeFileSync(FIXTURE_URL, `${JSON.stringify(fixture, null, 2)}\n`);
    });
    return;
  }
  const raw = Object.values(RAW)[0];
  if (!raw)
    throw new Error(
      'Missing identity.json; generate on the unchanged engine with UPDATE_DEPLOYMENT_IDENTITY=1',
    );
  const fixture = JSON.parse(raw) as Fixture;
  it('pins the exact case set', () =>
    expect(Object.keys(inputs).sort()).toEqual(Object.keys(fixture).sort()));
  it('allows only a few ULP on numbers, nothing on signs, zeros, strings or booleans', () => {
    const ulpsAway = (x: number, n: number): number => {
      BITS.setFloat64(0, x);
      BITS.setBigUint64(0, BITS.getBigUint64(0) + BigInt(n));
      return BITS.getFloat64(0);
    };
    expect(mismatches({ x: 1 }, { x: 1 + Number.EPSILON })).toEqual([]);
    expect(mismatches({ x: 0.0175 }, { x: ulpsAway(0.0175, MAX_ULPS) })).toEqual([]);
    expect(mismatches({ x: -227.7 }, { x: ulpsAway(-227.7, -MAX_ULPS) })).toEqual([]);
    expect(mismatches({ x: 0.0175 }, { x: ulpsAway(0.0175, MAX_ULPS + 1) })).toHaveLength(1);
    expect(mismatches({ x: 227.7 }, { x: 227.7 * (1 + 1e-12) })).toHaveLength(1);
    expect(mismatches({ x: 0 }, { x: -0 })).toHaveLength(1);
    expect(mismatches({ x: 0 }, { x: Number.MIN_VALUE })).toHaveLength(1);
    expect(mismatches({ x: 1 }, { x: -1 })).toHaveLength(1);
    expect(mismatches({ x: 4096 }, { x: 4097 })).toHaveLength(1);
    expect(mismatches({ x: 'Infinity' }, { x: Infinity })).toHaveLength(1);
    expect(mismatches({ x: true }, { x: 1 })).toHaveLength(1);
    expect(mismatches({ x: '[]' }, { x: '[] ' })).toHaveLength(1);
    expect(mismatches({ x: 1 }, {})).toHaveLength(1);
  });
  it.each(Object.entries(inputs))(
    '%s: original numeric fields (within a few ULP) and warnings (exactly) match the pin',
    (key, value) => {
      expect(mismatches(fixture[key], pin(simulate(value)))).toEqual([]);
    },
  );
  it.each(DESIGN_GOLDENS)(
    '%s: design §8.3 numbers still match (relative 1e-9)',
    (key, dc, dt, pc, pt, pr, memory, fits) => {
      const r = simulate(inputs[key]);
      for (const [actual, expected] of [
        [r.decode.cycleSec, dc],
        [r.decode.tokensPerSec, dt],
        [r.prefill.cycleSec, pc],
        [r.prefill.latencySec, pt],
        [r.prefill.tokensPerSec, pr],
      ])
        expectClose(actual, expected);
      expect(r.memory.perStage.map((s) => s.totalBytes)).toEqual(memory);
      expect(r.memory.fits).toBe(fits);
    },
  );
  it('W1/W2 rounded baseline cycles and W1 memory still match the design', () => {
    const w1 = simulate(inputs.W1);
    expectClose(w1.decode.cycleSec, 2.46673e-3, 1e-6);
    expectClose(w1.prefill.cycleSec, 377.58e-3, 1e-6);
    expect(w1.memory.perStage[0]).toMatchObject({
      weightBytes: 19389640704,
      sequenceStateBytes: 805404672,
      activationPeakBytes: 469762048,
    });
    expectClose(simulate(inputs.W2).decode.cycleSec, 9.128152e-3, 1e-6);
    expectClose(simulate(inputs['W2/batch8']).decode.cycleSec, 13.897e-3, 1e-5);
  });
  it('REF_MOE_30B_A3B has exactly 30,532,122,624 parameters', () =>
    expect(countParams(REF_MOE_30B_A3B)).toBe(30_532_122_624));
  it.each(Object.entries(inputs))(
    'ID-1 %s: deployment {} and a prefetch accuracy with nothing offloaded reproduce the pin and deep-equal absent deployment',
    (key, value) => {
      const absent = simulate(value);
      const empty = simulate({ ...value, deployment: {} });
      const offloadIdeal = simulate({
        ...value,
        deployment: { offload: { prefetchAccuracy: 1 } },
      });
      for (const r of [empty, offloadIdeal]) {
        expect(mismatches(fixture[key], pin(r))).toEqual([]);
        expect(r).toEqual(absent);
      }
      // With nothing offloaded, prefetch accuracy 0 changes nothing either (xfer_s = 0).
      expect(simulate({ ...value, deployment: { offload: { prefetchAccuracy: 0 } } })).toEqual(
        absent,
      );
      // The new fields are derived sums or absent: no transfer, no spec-dec, zero deployment memory.
      for (const phase of [absent.prefill, absent.decode]) {
        expect(phase.transfer).toBeUndefined();
        expect(phase.speculative).toBeUndefined();
        for (const s of phase.stages) expect([s.transferSec, s.exposedTransferSec]).toEqual([0, 0]);
      }
      expect(absent.memory.tiers).toEqual([]);
      for (const s of absent.memory.perStage)
        expect([s.stagingBytes, s.draftBytes, s.speculativeTransientBytes]).toEqual([0, 0, 0]);
    },
  );
  it.each(Object.entries(inputs).filter(([key]) => key.startsWith('models/')))(
    'ID-2 %s: absent deployment deep-equals {}, lookupTables tier device and embeddings tier device',
    (_key, value) => {
      const absent = simulate(value);
      expect(simulate({ ...value, deployment: {} })).toEqual(absent);
      expect(simulate({ ...value, deployment: { offload: {} } })).toEqual(absent);
      expect(
        simulate({ ...value, deployment: { offload: { lookupTables: { tier: 'device' } } } }),
      ).toEqual(absent);
      expect(
        simulate({ ...value, deployment: { offload: { embeddings: { tier: 'device' } } } }),
      ).toEqual(absent);
    },
  );
  it.each(Object.entries(inputs))(
    'ID-3 %s: empty placement rule lists (routedExperts [], lookupTables [], embeddings absent or device) deep-equal absent deployment',
    (_key, value) => {
      const absent = simulate(value);
      expect(
        simulate({ ...value, deployment: { offload: { routedExperts: [], lookupTables: [] } } }),
      ).toEqual(absent);
      expect(
        simulate({
          ...value,
          deployment: { offload: { routedExperts: [], lookupTables: [], embeddings: { tier: 'device' } } },
        }),
      ).toEqual(absent);
    },
  );
});
