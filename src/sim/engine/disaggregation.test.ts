/**
 * Prefill/decode disaggregation: pools, request rates, balance, system throughput, and the KV transfer wired in.
 * The KV transfer formulas themselves are pinned in kvTransfer.test.ts.
 * See docs/simulation-model.md §14 and docs/design/prefill-decode-disaggregation.md §8.
 *
 * What is pinned here:
 * - ID-PD-1: with both pools equal to one SimInput, the decode pool's result IS `simulate(input)` and the prefill
 *   pool's prefill phase IS `simulate(input).prefill` (dense pp 1, dense pp 2, MoE DP attention + EP);
 * - the prefill pool's memory fit is at context P exactly (`decodeContextLen = max(1, P − 1)`);
 * - W-PD1 (design note §8.1, commit-A part): hand numbers for r*, the device share, the per-device rate at balance
 *   and the system at 1:2 and 1:3;
 * - the 'balanced' tolerance, and the validation messages;
 * - ID-PD-4: `kvTransfer` absent gives no `kvTransfer` key; an infinitely fast, zero-latency link gives a result
 *   equal to it; a slow link lengthens the prefill cycle; MTP's drafter cache is moved; DCP in either pool changes
 *   neither the bytes nor the ingress bound.
 */
import { describe, expect, it } from 'vitest';
import { B200, composeTopology } from '../hw/presets';
import { decoderLayer, decoderModel, repeatLayers } from '../ir/builders';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { REF_DENSE_8B, REF_MOE_30B_A3B } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import {
  balancedDeployment,
  balancedPrefillPerDecode,
  decodePoolRates,
  disaggregationErrors,
  poolInput,
  prefillPoolRates,
  simulateDisaggregated,
  systemThroughput,
  type DisaggregatedInput,
  type PoolRates,
  type PoolSpec,
} from './disaggregation';
import { simulate } from './simulate';
import type { SimInput } from './types';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** `nodes` machines of 8 × B200 (NVLink 5 inside, the catalog's default network between them). */
function b200(nodes: number): SimInput['hardware'] {
  return {
    device: B200,
    topology: composeTopology({ deviceId: 'b200', devicesPerNode: 8, nodes }),
  };
}

function single(
  model: SimInput['model'],
  parallel: Partial<ParallelConfig>,
  batch: number,
  promptLen = 4096,
  decodeContextLen = 4096,
): SimInput {
  return {
    model,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    workload: { batch, promptLen, decodeContextLen },
    hardware: b200(1),
  };
}

/** The disaggregated input with both pools equal to `input`'s hardware, parallel and batch. */
function samePools(input: SimInput, outputLen = 1024): DisaggregatedInput {
  const { hardware, parallel, ...shared } = input;
  const pool: PoolSpec = { hardware, parallel, batch: input.workload.batch };
  return {
    shared,
    prefillPool: pool,
    decodePool: pool,
    config: { outputLen, prefillInstances: 1, decodeInstances: 1 },
  };
}

describe('ID-PD-1: equal pools reproduce simulate() bit for bit', () => {
  const cases: [string, SimInput][] = [
    ['dense tp 2, pp 1', single(REF_DENSE_8B, { tp: 2 }, 8)],
    ['dense tp 2, pp 2', single(REF_DENSE_8B, { tp: 2, pp: 2 }, 8)],
    ['MoE DP attention + EP 8', single(REF_MOE_30B_A3B, { dp: 8, ep: 8 }, 64)],
    ['dense, P ≠ N (P 2048, N 8192)', single(REF_DENSE_8B, { tp: 2 }, 8, 2048, 8192)],
  ];
  for (const [name, input] of cases) {
    it(name, () => {
      const reference = simulate(input);
      const result = simulateDisaggregated(samePools(input));
      // The decode pool's run is the input itself.
      expect(result.decodePool).toEqual(reference);
      // The prefill pool runs with decodeContextLen = max(1, P − 1); its prefill phase does not read N.
      expect(result.prefillPool.prefill).toEqual(reference.prefill);
      expect(result.prefill.tokensPerSec).toBe(reference.prefill.tokensPerSec);
      expect(result.decode.tokensPerSec).toBe(reference.decode.tokensPerSec);
      expect(result.prefill.devices).toBe(reference.devicesUsed);
    });
  }
});

describe('pools', () => {
  it('the prefill pool fits at context P: 4 seqs · 147,456 B/token · 4096 = 2,415,919,104 B on one B200', () => {
    // REF_DENSE_8B bf16 tp 1: 36 layers · 2·8·128·2 B = 147,456 B per token per sequence.
    const input = single(REF_DENSE_8B, {}, 4, 4096, 8192);
    const result = simulateDisaggregated(samePools(input));
    expect(result.prefillPool.memory.perStage[0].sequenceStateBytes).toBe(4 * 147_456 * 4096);
    // The decode pool holds N + 1 = 8193 tokens.
    expect(result.decodePool.memory.perStage[0].sequenceStateBytes).toBe(4 * 147_456 * 8193);
  });

  it('poolInput: the pool sets hardware, parallel and batch; prefill sets N = max(1, P − 1), decode keeps N', () => {
    const { shared } = samePools(single(REF_DENSE_8B, {}, 1, 4096, 8192));
    const pool: PoolSpec = {
      hardware: b200(4),
      parallel: { tp: 1, pp: 1, dp: 32, ep: 1 },
      batch: 64,
    };
    const p = poolInput(shared, pool, 'prefill');
    expect(p.hardware).toBe(pool.hardware);
    expect(p.parallel).toBe(pool.parallel);
    expect(p.workload).toEqual({ batch: 64, promptLen: 4096, decodeContextLen: 4095 });
    expect(poolInput(shared, pool, 'decode').workload).toEqual({
      batch: 64,
      promptLen: 4096,
      decodeContextLen: 8192,
    });
    const one = poolInput(
      { ...shared, workload: { ...shared.workload, promptLen: 1 } },
      pool,
      'prefill',
    );
    expect(one.workload.decodeContextLen).toBe(1);
  });

  it('different pools: DeepSeek-shaped 32 : 144 devices (prefill dp 32, decode dp 144 on 18 nodes)', () => {
    const input: DisaggregatedInput = {
      shared: {
        model: REF_DENSE_8B,
        quant: BF16,
        workload: { batch: 1, promptLen: 4096, decodeContextLen: 4608 },
      },
      prefillPool: { hardware: b200(4), parallel: { tp: 1, pp: 1, dp: 32, ep: 1 }, batch: 32 },
      decodePool: {
        hardware: b200(18),
        parallel: { tp: 1, pp: 1, dp: 144, ep: 1 },
        batch: 144 * 16,
      },
      config: { outputLen: 1024, prefillInstances: 1, decodeInstances: 1 },
    };
    const result = simulateDisaggregated(input);
    expect(result.prefill.devices).toBe(32);
    expect(result.decode.devices).toBe(144);
    expect(result.system.devices).toBe(176);
    expect(result.prefill.contextTokens).toBe(4096);
    expect(result.decode.contextTokens).toBe(4608);
    // Each pool is an ordinary simulate() run.
    expect(result.decodePool).toEqual(
      simulate(poolInput(input.shared, input.decodePool, 'decode')),
    );
    expect(result.prefillPool).toEqual(
      simulate(poolInput(input.shared, input.prefillPool, 'prefill')),
    );
  });
});

describe('W-PD1 (design note §8.1): REF_DENSE_8B, P 8192, O 1024, N 8704; given pool rates', () => {
  // Prefill instance: D_P = 8, R_P = 8·8192 / 0.5 s = 131,072 tok/s @ 8k ⇒ λ_P = 16 req/s.
  const prefill: PoolRates = {
    devices: 8,
    tokensPerSec: 131_072,
    contextTokens: 8192,
    requestsPerSec: 16,
  };
  // Decode instance: D_D = 8, R_D = 256 / 0.04 s = 6,400 tok/s @ 8,704 ⇒ λ_D = 6400 / 1024 = 6.25 req/s.
  const decode: PoolRates = {
    devices: 8,
    tokensPerSec: 6400,
    contextTokens: 8704,
    requestsPerSec: 6.25,
  };
  const P = 8192;
  const O = 1024;

  it('request rates: λ_P = 131,072 / 8192 = 16, λ_D = 6400 / 1024 = 6.25', () => {
    const fake = (prefillTps: number, decodeTps: number) =>
      ({
        devicesUsed: 8,
        prefill: { tokensPerSec: prefillTps, cycleSec: 0.5 },
        decode: { tokensPerSec: decodeTps },
      }) as Parameters<typeof prefillPoolRates>[0];
    expect(prefillPoolRates(fake(131_072, 0), P)).toEqual(prefill);
    expect(decodePoolRates(fake(0, 6400), 8704, O)).toEqual(decode);
  });

  it('r* = 6.25 / 16 = 0.390625 (P:D 1 : 2.56); share 3.125 / 11.125; 6400 / 11.125 = 575.2809 tok/s/GPU', () => {
    expect(balancedPrefillPerDecode(16, 6.25)).toBe(0.390625);
    const balance = balancedDeployment(prefill, decode);
    expect(balance.prefillPerDecode).toBe(0.390625);
    expectClose(balance.prefillDeviceShare, 3.125 / 11.125);
    expectClose(balance.outputTokensPerSecPerDevice, 6400 / 11.125);
  });

  it('1:2 ⇒ λ_sys = min(16, 12.5) = 12.5, G 24: 12,800 out, 102,400 in, 533.333/GPU, decode-bound', () => {
    const s = systemThroughput({
      prefill,
      decode,
      promptLen: P,
      outputLen: O,
      prefillInstances: 1,
      decodeInstances: 2,
    });
    expect(s).toEqual({
      prefillInstances: 1,
      decodeInstances: 2,
      devices: 24,
      requestsPerSec: 12.5,
      inputTokensPerSec: 102_400,
      outputTokensPerSec: 12_800,
      outputTokensPerSecPerDevice: 12_800 / 24,
      bottleneck: 'decode',
      prefillUtilization: 0.78125,
      decodeUtilization: 1,
    });
  });

  it('1:3 ⇒ λ_sys = min(16, 18.75) = 16, G 32: 16,384 out, 512/GPU, prefill-bound, util_D 16/18.75', () => {
    const s = systemThroughput({
      prefill,
      decode,
      promptLen: P,
      outputLen: O,
      prefillInstances: 1,
      decodeInstances: 3,
    });
    expect(s.requestsPerSec).toBe(16);
    expect(s.devices).toBe(32);
    expect(s.outputTokensPerSec).toBe(16_384);
    expect(s.outputTokensPerSecPerDevice).toBe(512);
    expect(s.bottleneck).toBe('prefill');
    expect(s.prefillUtilization).toBe(1);
    expectClose(s.decodeUtilization, 16 / 18.75);
  });

  it("a ratio at r* is 'balanced': λ_P 10, λ_D 25 ⇒ r* 2.5; at 5:2 both sides carry 50 req/s", () => {
    const p: PoolRates = {
      devices: 32,
      tokensPerSec: 10 * P,
      contextTokens: P,
      requestsPerSec: 10,
    };
    const d: PoolRates = {
      devices: 144,
      tokensPerSec: 25 * O,
      contextTokens: 8704,
      requestsPerSec: 25,
    };
    const balance = balancedDeployment(p, d);
    expect(balance.prefillPerDecode).toBe(2.5);
    expectClose(balance.prefillDeviceShare, 80 / 224);
    expectClose(balance.outputTokensPerSecPerDevice, (25 * O) / 224);
    const s = systemThroughput({
      prefill: p,
      decode: d,
      promptLen: P,
      outputLen: O,
      prefillInstances: 5,
      decodeInstances: 2,
    });
    expect(s.bottleneck).toBe('balanced');
    expect(s.requestsPerSec).toBe(50);
    // Within the 1e-9 relative tolerance: a capacity off by 1e-12 relative still counts as balanced.
    const nudged = { ...d, requestsPerSec: 25 * (1 + 1e-12) };
    expect(
      systemThroughput({
        prefill: p,
        decode: nudged,
        promptLen: P,
        outputLen: O,
        prefillInstances: 5,
        decodeInstances: 2,
      }).bottleneck,
    ).toBe('balanced');
    const off = { ...d, requestsPerSec: 25 * (1 + 1e-6) };
    expect(
      systemThroughput({
        prefill: p,
        decode: off,
        promptLen: P,
        outputLen: O,
        prefillInstances: 5,
        decodeInstances: 2,
      }).bottleneck,
    ).toBe('prefill');
  });
});

describe('disaggregationErrors', () => {
  const base = samePools(single(REF_DENSE_8B, { tp: 2 }, 8));

  it('a valid input has no errors', () => {
    expect(disaggregationErrors(base)).toEqual([]);
  });

  it('outputLen 0 and non-integer or zero instance counts are rejected', () => {
    const errors = disaggregationErrors({
      ...base,
      config: { outputLen: 0, prefillInstances: 1.5, decodeInstances: 0 },
    });
    expect(errors).toEqual([
      'disaggregation.outputLen must be an integer ≥ 1 (got 0)',
      'disaggregation.prefillInstances must be an integer ≥ 1 (got 1.5)',
      'disaggregation.decodeInstances must be an integer ≥ 1 (got 0)',
    ]);
  });

  it('each pool is validated on its own, with a prefix', () => {
    const bad: PoolSpec = {
      ...base.decodePool,
      parallel: { tp: 1, pp: 1, dp: 3, ep: 1 },
      batch: 8,
    };
    const errors = disaggregationErrors({ ...base, decodePool: bad });
    expect(errors.length).toBeGreaterThan(0);
    expect(errors.every((e) => e.startsWith('decode pool: '))).toBe(true);
    const prefixed = disaggregationErrors({ ...base, prefillPool: bad });
    expect(prefixed.every((e) => e.startsWith('prefill pool: '))).toBe(true);
    expect(() => simulateDisaggregated({ ...base, decodePool: bad })).toThrow(/^decode pool: /);
  });
});

/** Made-up dense MLA model (2 layers, rkv 64 + dr 16), valid at tp 8 and dcp 8. */
const TINY_MLA_DENSE: ModelSpec = decoderModel({
  name: 'TINY_MLA_DENSE (test)',
  hidden: 512,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture',
  layers: repeatLayers(2, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 512,
      mixer: {
        kind: 'mla',
        heads: 8,
        qLoraRank: 128,
        kvLoraRank: 64,
        qkNopeDim: 32,
        qkRopeDim: 16,
        vHeadDim: 32,
      },
      ffn: { kind: 'dense', intermediate: 1024 },
    }),
  ),
});

describe('KV transfer in simulateDisaggregated', () => {
  const base = samePools(single(REF_DENSE_8B, { tp: 2 }, 8));

  it('ID-PD-4: absent ⇒ no kvTransfer key; B = ∞ and α = 0 ⇒ everything else equal, both overlap modes', () => {
    const without = simulateDisaggregated(base);
    expect('kvTransfer' in without).toBe(false);
    for (const overlap of ['layerwise', 'none'] as const) {
      const link = { name: 'ideal', bandwidthPerDirectionBytesPerSec: Infinity, latencySec: 0 };
      const { kvTransfer, ...rest } = simulateDisaggregated({
        ...base,
        config: { ...base.config, kvTransfer: { link, overlap } },
      });
      expect(rest).toEqual(without);
      expect(kvTransfer?.effectivePrefillCycleSec).toBe(without.prefillPool.prefill.cycleSec);
      expect(kvTransfer?.ingressRequestsPerSec).toBe(Infinity);
    }
  });

  it('a 1 MB/s link: cycle_eff = t_bw = B_r·S/tp / B, R_P,eff = R_P·cycle_P/t_bw, prefill-bound', () => {
    const link = { name: 'slow', bandwidthPerDirectionBytesPerSec: 1e6, latencySec: 1e-6 };
    const r = simulateDisaggregated({
      ...base,
      config: { ...base.config, decodeInstances: 2, kvTransfer: { link, overlap: 'layerwise' } },
    });
    // tp 2, dp 1, batch 8: B_r = 8, Y = 8 · 147,456 · 4096 / 2 B.
    const Y = (8 * 147_456 * 4096) / 2;
    expect(r.kvTransfer?.batchBytesPerDevice).toBe(Y);
    expectClose(r.kvTransfer?.effectivePrefillCycleSec ?? 0, Y / 1e6);
    const R_P = r.prefillPool.prefill.tokensPerSec;
    expectClose(r.prefill.tokensPerSec, (R_P * r.prefillPool.prefill.cycleSec) / (Y / 1e6));
    // Sender and receiver are symmetric here (2 devices each, same link): at 1:1 the prefill side and the ingress
    // tie at 2·1e6 / S requests/s ('balanced'); at 1:2 the decode side has twice the ingress, so prefill binds.
    expect(r.system.bottleneck).toBe('prefill');
    expectClose(r.system.requestsPerSec, r.kvTransfer?.ingressRequestsPerSec ?? NaN);
  });

  it('spec-dec MTP: the drafter cache moves too, S_tot = 37 · 4096 B · 4096 tokens', () => {
    const input = samePools({
      ...single(REF_DENSE_8B_MTP, {}, 1),
      deployment: {
        speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.8 } },
      },
    });
    const r = simulateDisaggregated({
      ...input,
      config: {
        ...input.config,
        kvTransfer: {
          link: { name: 'ib', bandwidthPerDirectionBytesPerSec: 50e9, latencySec: 1e-6 },
          overlap: 'layerwise',
        },
      },
    });
    expect(r.kvTransfer?.requestBytes).toBe(37 * 4096 * 4096);
  });

  it('DCP: decode dcp 1 vs 8 and prefill dcp 1 vs 8 leave the transfer unchanged; D = tp·pp·dp', () => {
    const link = { name: 'ib', bandwidthPerDirectionBytesPerSec: 50e9, latencySec: 1e-6 };
    const shared: DisaggregatedInput['shared'] = {
      model: TINY_MLA_DENSE,
      quant: BF16,
      workload: { batch: 1, promptLen: 4096, decodeContextLen: 4096 },
    };
    const pool = (parallel: Partial<ParallelConfig>, batch: number): PoolSpec => ({
      hardware: b200(1),
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
      batch,
    });
    const run = (prefillPool: PoolSpec, decodePool: PoolSpec) =>
      simulateDisaggregated({
        shared,
        prefillPool,
        decodePool,
        config: {
          outputLen: 1024,
          prefillInstances: 1,
          decodeInstances: 1,
          kvTransfer: { link, overlap: 'layerwise' },
        },
      });
    // Prefill layout matching the decode one (tp 8) and differing from it (tp 1 · dp 8).
    for (const prefill of [pool({ tp: 8 }, 8), pool({ dp: 8 }, 8)]) {
      const d1 = run(prefill, pool({ tp: 8, dcp: 1 }, 16));
      const d8 = run(prefill, pool({ tp: 8, dcp: 8 }, 16));
      expect(d8.kvTransfer).toEqual(d1.kvTransfer);
      expect(d8.kvTransfer?.requestBytes).toBe(320 * 4096);
      expect(d8.decode.devices).toBe(8);
      expect(d8.decodePool).toEqual(
        simulate(poolInput(shared, pool({ tp: 8, dcp: 8 }, 16), 'decode')),
      );
    }
    const p1 = run(pool({ tp: 8 }, 8), pool({ tp: 8 }, 16));
    const p8 = run(pool({ tp: 8, dcp: 8 }, 8), pool({ tp: 8 }, 16));
    expect(p8.kvTransfer).toEqual(p1.kvTransfer);
    expect(p8.prefill.devices).toBe(8);
  });

  it('rejects a link with bandwidth ≤ 0 or a negative latency', () => {
    const link = { name: 'bad', bandwidthPerDirectionBytesPerSec: 0, latencySec: -1 };
    expect(
      disaggregationErrors({
        ...base,
        config: { ...base.config, kvTransfer: { link, overlap: 'none' } },
      }),
    ).toEqual([
      'disaggregation.kvTransfer.link bandwidth must be > 0 B/s (got 0)',
      'disaggregation.kvTransfer.link latency must be ≥ 0 s (got -1)',
    ]);
  });
});
