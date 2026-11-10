/**
 * KV transfer of prefill/decode disaggregation (design note §4, §8): the bytes per request and per stage, the busiest
 * sender's time, the exposed TTFT (layer-wise flow shop vs not overlapped, α once), the lengthened prefill cycle, and
 * the decode side's ingress bound, with the W-PD1 KV rows as hand numbers. See docs/simulation-model.md §14
 * "KV transfer".
 */
import { describe, expect, it } from 'vitest';
import { decoderLayer, decoderModel, repeatLayers } from '../ir/builders';
import type { ModelSpec, QuantConfig } from '../ir/model';
import { REF_DENSE_8B, TINY_HYBRID } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import {
  balancedDeployment,
  prefillPoolRates,
  systemThroughput,
  type PoolRates,
} from './disaggregation';
import {
  DEFAULT_KV_TRANSFER_LINK,
  effectivePrefillCycleSec,
  exposedTransferSec,
  kvTransferTime,
  stageRequestKvBytes,
  type KvTransferConfig,
} from './kvTransfer';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
const FP8_KV: QuantConfig = { ...BF16, kvCache: { dtype: 'fp8_e4m3' } };

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

/** REF_DENSE_8B bf16: 36 layers · 2·8 KV heads·128·2 B = 147,456 B per token. */
const DENSE_8B_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2;
const P = 8192;
/** S_tot(8192) = 147,456 · 8192 = 1,207,959,552 B. */
const S = DENSE_8B_BYTES_PER_TOKEN * P;

/** Made-up dense MLA model: 2 layers, rkv 64 + dr 16 = 80 latent elements per token per layer. */
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

/** Made-up GQA model, 2 layers of 2 KV heads × 64, the second with a 1024-token sliding window. */
const TINY_SWA: ModelSpec = decoderModel({
  name: 'TINY_SWA (test)',
  hidden: 256,
  vocab: 1000,
  tieEmbeddings: false,
  source: 'made-up test fixture',
  layers: repeatLayers(2, (i) =>
    decoderLayer({
      name: `layer.${i}`,
      hidden: 256,
      mixer: {
        kind: 'gqa',
        heads: 4,
        kvHeads: 2,
        headDim: 64,
        ...(i === 1 ? { slidingWindow: 1024 } : {}),
      },
      ffn: { kind: 'dense', intermediate: 512 },
    }),
  ),
});

describe('stageRequestKvBytes: S_s(P), unreplicated', () => {
  it('REF_DENSE_8B bf16 @ 8k: one stage of 147,456 B/token · 8192 = 1,207,959,552 B', () => {
    expect(stageRequestKvBytes(REF_DENSE_8B, BF16, 1, P)).toEqual([1_207_959_552]);
  });

  it('FP8 (e4m3, no block scale) KV halves it: 603,979,776 B', () => {
    expect(stageRequestKvBytes(REF_DENSE_8B, FP8_KV, 1, P)).toEqual([603_979_776]);
  });

  it('uneven PP: pp 5 splits 36 layers 7, 7, 7, 7, 8 ⇒ stage 4 holds 8 · 4096 · 8192 = 268,435,456 B', () => {
    const perLayer = 2 * 8 * 128 * 2 * P;
    expect(stageRequestKvBytes(REF_DENSE_8B, BF16, 5, P)).toEqual(
      [7, 7, 7, 7, 8].map((n) => n * perLayer),
    );
  });

  it('MLA: (rkv 64 + dr 16) · 2 B · 2 layers = 320 B/token, never sharded', () => {
    expect(stageRequestKvBytes(TINY_MLA_DENSE, BF16, 1, P)).toEqual([320 * P]);
  });

  it('sliding window: the windowed layer is capped at 1024 tokens (2·2·64·2 B = 512 B/token/layer)', () => {
    expect(stageRequestKvBytes(TINY_SWA, BF16, 1, P)).toEqual([512 * P + 512 * 1024]);
  });

  it('hybrid: only the attention layer grows with P (GQA 1 KV head × 64: 256 B/token); the recurrent state is fixed', () => {
    const [s4k] = stageRequestKvBytes(TINY_HYBRID, BF16, 1, 4096);
    const [s8k] = stageRequestKvBytes(TINY_HYBRID, BF16, 1, P);
    expect(s8k - s4k).toBe(256 * 4096);
    expect(s4k - 256 * 4096).toBeGreaterThan(0);
  });

  it('spec-dec MTP: the drafter cache is transferred too, 37 GQA layers · 4096 B · 8192', () => {
    expect(stageRequestKvBytes(REF_DENSE_8B_MTP, BF16, 1, P, REF_DENSE_8B_MTP.draft)).toEqual([
      37 * 4096 * P,
    ]);
  });
});

describe('exposedTransferSec / effectivePrefillCycleSec', () => {
  it('layer-wise, link keeps up (a ≥ β_c): exposed = α + t_bw/L; cycle_eff = cycle_P', () => {
    const args = {
      prefillLatencySec: 0.5,
      bandwidthSec: 0.36,
      latencySec: 1e-6,
      layers: 36,
    } as const;
    expectClose(exposedTransferSec({ ...args, overlap: 'layerwise' }), 1e-6 + 0.01);
    expect(
      effectivePrefillCycleSec({
        prefillCycleSec: 0.5,
        bandwidthSec: 0.36,
        latencySec: 1e-6,
        overlap: 'layerwise',
      }),
    ).toBe(0.5);
  });

  it('layer-wise, link-bound (β_c > a): exposed = α + t_bw − t_pf·(L−1)/L; cycle_eff = t_bw', () => {
    const args = { prefillLatencySec: 0.5, bandwidthSec: 2, latencySec: 1e-6, layers: 36 } as const;
    expectClose(exposedTransferSec({ ...args, overlap: 'layerwise' }), 1e-6 + 2 - (0.5 * 35) / 36);
    expect(
      effectivePrefillCycleSec({
        prefillCycleSec: 0.5,
        bandwidthSec: 2,
        latencySec: 1e-6,
        overlap: 'layerwise',
      }),
    ).toBe(2);
  });

  it('not overlapped: exposed = α + t_bw; cycle_eff = cycle_P + α + t_bw (wait for arrival)', () => {
    expect(
      exposedTransferSec({
        prefillLatencySec: 0.5,
        bandwidthSec: 0.25,
        latencySec: 0.125,
        layers: 36,
        overlap: 'none',
      }),
    ).toBe(0.375);
    expect(
      effectivePrefillCycleSec({
        prefillCycleSec: 0.5,
        bandwidthSec: 0.25,
        latencySec: 0.125,
        overlap: 'none',
      }),
    ).toBe(0.875);
  });

  it('L = 1: layer-wise equals not overlapped', () => {
    const args = {
      prefillLatencySec: 0.5,
      bandwidthSec: 0.3,
      latencySec: 1e-6,
      layers: 1,
    } as const;
    expectClose(
      exposedTransferSec({ ...args, overlap: 'layerwise' }),
      exposedTransferSec({ ...args, overlap: 'none' }),
    );
  });

  it('B → ∞, α = 0: cycle_eff === cycle_P in both modes', () => {
    for (const overlap of ['layerwise', 'none'] as const)
      expect(
        effectivePrefillCycleSec({ prefillCycleSec: 0.5, bandwidthSec: 0, latencySec: 0, overlap }),
      ).toBe(0.5);
  });
});

describe('kvTransferTime: busiest sender, per request, ingress', () => {
  it('uneven PP (pp 5, tp 1, B_r 1) at 50e9 B/s: Y = 268,435,456 B, t_bw = 5.36870912 ms, t_req = 1 µs + t_bw', () => {
    const r = kvTransferTime({
      stageRequestBytes: stageRequestKvBytes(REF_DENSE_8B, BF16, 5, P),
      promptLen: P,
      replicaBatch: 1,
      prefillTp: 1,
      decodeDevices: 8,
      layers: 36,
      link: DEFAULT_KV_TRANSFER_LINK,
      overlap: 'layerwise',
      prefillLatencySec: 0.5,
      prefillCycleSec: 0.5,
    });
    expect(r.requestBytes).toBe(S);
    expect(r.batchBytesPerDevice).toBe(268_435_456);
    expectClose(r.bandwidthSec, 0.00536870912);
    expectClose(r.requestSec, 1e-6 + 0.00536870912);
  });

  it('tp splits each stage across its ranks: tp 4, B_r 2 ⇒ Y = 2·S/4', () => {
    const r = kvTransferTime({
      stageRequestBytes: [S],
      promptLen: P,
      replicaBatch: 2,
      prefillTp: 4,
      decodeDevices: 8,
      layers: 36,
      link: DEFAULT_KV_TRANSFER_LINK,
      overlap: 'none',
      prefillLatencySec: 0.5,
      prefillCycleSec: 0.5,
    });
    expect(r.batchBytesPerDevice).toBe(S / 2);
    expectClose(r.requestSec, 1e-6 + S / (4 * 50e9));
  });
});

describe('W-PD1 KV rows (design note §8.1): λ_P 16, λ_D 6.25, D_P = D_D = 8, t_pf = cycle_P = 0.5 s, L 36', () => {
  const decode: PoolRates = {
    devices: 8,
    tokensPerSec: 6400,
    contextTokens: 8704,
    requestsPerSec: 6.25,
  };
  /** The prefill pool's result, reduced to what the formulas read. */
  const prefillResult = {
    devicesUsed: 8,
    prefill: { tokensPerSec: 131_072, cycleSec: 0.5, latencySec: 0.5 },
  } as unknown as Parameters<typeof prefillPoolRates>[0];

  function row(bandwidth: number, overlap: KvTransferConfig['overlap'], decodeDevices = 8) {
    const kv = kvTransferTime({
      stageRequestBytes: [S],
      promptLen: P,
      replicaBatch: 1,
      prefillTp: 1,
      decodeDevices,
      layers: 36,
      link: { name: 'test', bandwidthPerDirectionBytesPerSec: bandwidth, latencySec: 1e-6 },
      overlap,
      prefillLatencySec: 0.5,
      prefillCycleSec: 0.5,
    });
    const prefill = prefillPoolRates(prefillResult, P, kv.effectivePrefillCycleSec);
    const d = { ...decode, devices: decodeDevices };
    const balance = balancedDeployment(prefill, d, kv.ingressRequestsPerSec);
    const system = (nP: number, nD: number) =>
      systemThroughput({
        prefill,
        decode: d,
        ingressRequestsPerSec: kv.ingressRequestsPerSec,
        promptLen: P,
        outputLen: 1024,
        prefillInstances: nP,
        decodeInstances: nD,
      });
    return { kv, prefill, balance, system };
  }

  it('bytes: 147,456 B/token, 1,207,959,552 B per request; at 50 GB/s t_bw = 24.159 ms, t_req = t_bw + 1 µs', () => {
    const { kv } = row(50e9, 'layerwise');
    expect(kv.bytesPerToken).toBe(DENSE_8B_BYTES_PER_TOKEN);
    expect(kv.batchBytesPerDevice).toBe(S);
    expectClose(kv.bandwidthSec, 0.02415919104);
    expectClose(kv.requestSec, 0.02416019104);
    expectClose(kv.ingressRequestsPerSec, (8 * 50e9) / S);
  });

  it('layer-wise, 50 GB/s: exposed 0.67209 ms, cycle 0.5, λ_P 16, r* 0.390625, 575.28 /GPU, 1:2 533.33 decode-bound', () => {
    const { kv, prefill, balance, system } = row(50e9, 'layerwise');
    expectClose(kv.exposedSec, 0.00067208864);
    expect(kv.effectivePrefillCycleSec).toBe(0.5);
    expect(prefill.requestsPerSec).toBe(16);
    expect(balance.prefillPerDecode).toBe(0.390625);
    expectClose(balance.outputTokensPerSecPerDevice, 575.2808988764045);
    expectClose(system(1, 2).outputTokensPerSecPerDevice, 12_800 / 24);
    expect(system(1, 2).bottleneck).toBe('decode');
  });

  it('not overlapped, 50 GB/s: exposed 24.16 ms, cycle 0.52416, λ_P 15.26251, r* 0.40950, 567.58 /GPU', () => {
    const { kv, prefill, balance, system } = row(50e9, 'none');
    expectClose(kv.exposedSec, 0.02416019104);
    expectClose(kv.effectivePrefillCycleSec, 0.52416019104);
    expectClose(prefill.requestsPerSec, 15.262509699805683);
    expectClose(balance.prefillPerDecode, 0.40950014925);
    expectClose(balance.outputTokensPerSecPerDevice, 567.5770949195592);
    expect(system(1, 2).bottleneck).toBe('decode');
  });

  it('layer-wise, 1 GB/s (link-bound): exposed 0.72185 s, cycle 1.20796, λ_P 6.62274, r* 0.94372, 411.58 /GPU, 1:2 282.57 prefill-bound', () => {
    const { kv, prefill, balance, system } = row(1e9, 'layerwise');
    expectClose(kv.exposedSec, 0.7218494408888887);
    expectClose(kv.effectivePrefillCycleSec, 1.207959552);
    expectClose(prefill.requestsPerSec, 6.622738308376737);
    // Ingress 8e9 / S = 6.62274 ≥ λ_D 6.25: not binding.
    expectClose(kv.ingressRequestsPerSec, 6.622738308376736);
    expectClose(balance.prefillPerDecode, 0.9437184);
    expectClose(balance.outputTokensPerSecPerDevice, 411.5822538902755);
    expectClose(system(1, 2).outputTokensPerSecPerDevice, 282.5701678240741);
    expect(system(1, 2).bottleneck).toBe('prefill');
  });

  it('not overlapped, 1 GB/s: exposed 1.20796 s, cycle 1.70796, λ_P 4.68395, r* 1.33434, 342.71 /GPU, 1:2 199.85', () => {
    const { kv, prefill, balance, system } = row(1e9, 'none');
    expectClose(kv.exposedSec, 1.207960552);
    expectClose(kv.effectivePrefillCycleSec, 1.707960552);
    expectClose(prefill.requestsPerSec, 4.683948929986763);
    expectClose(balance.prefillPerDecode, 1.3343441812499999);
    expectClose(balance.outputTokensPerSecPerDevice, 342.708674421616);
    expectClose(system(1, 2).outputTokensPerSecPerDevice, 199.84848767943524);
  });

  it('ingress binds with D_D = 4 at 1 GB/s: λ_recv,1 3.31137 < λ_D, r* 0.5, 423.86 /GPU; 1:1 kvIngress, 1:2 balanced', () => {
    const { kv, balance, system } = row(1e9, 'layerwise', 4);
    expectClose(kv.ingressRequestsPerSec, 3.311369154188368);
    expectClose(balance.prefillPerDecode, 0.5);
    expectClose(balance.prefillDeviceShare, 0.5);
    expectClose(balance.outputTokensPerSecPerDevice, 423.8552517361111);
    const one = system(1, 1);
    expect(one.bottleneck).toBe('kvIngress');
    expectClose(one.requestsPerSec, 3.311369154188368);
    expectClose(one.outputTokensPerSecPerDevice, 282.5701678240741);
    expectClose(one.decodeUtilization, 3.311369154188368 / 6.25);
    const two = system(1, 2);
    expect(two.bottleneck).toBe('balanced');
    expectClose(two.outputTokensPerSecPerDevice, 423.8552517361111);
  });
});
