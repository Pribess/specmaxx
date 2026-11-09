/**
 * The attention softmax term (docs/simulation-model.md §7 "Attention softmax"): decode stays KV-bandwidth-bound on
 * every preset, the review's probe (docs/review/practicality-review.md F6) stays within bands, and the bound stays
 * above published FlashAttention-3 / -4 forward throughput.
 */
import { describe, expect, it } from 'vitest';
import { resolveQuant, simpleDecoder } from '../__sanity__/harness';
import { GPT_OSS_120B, QWEN3_235B_A22B } from '../__sanity__/models';
import { applyDeviceModifiers } from '../hw/modifiers';
import { DEVICE_PRESETS, deviceUnlocks, singleDevice } from '../hw/presets/index';
import { resolveUnlocks } from '../hw/unlocks';
import { B200, H100_SXM } from '../hw/presets/nvidia-datacenter';
import type { ComputeEngine, DeviceSpec } from '../hw/types';
import type { ModelSpec } from '../ir/model';
import type { AttentionOp } from '../ir/ops';
import { engineMemBandwidthBytesPerSec, opTime, type OpTimeContext } from './opTime';
import { simulate } from './simulate';
import type { SimOptions } from './types';

/** Meta-Llama-3-8B (meta-llama/Meta-Llama-3-8B config.json): h 4096, 32 layers, GQA 32/8, hd 128, ffn 14336. */
const LLAMA3_8B = simpleDecoder({
  name: 'Llama-3-8B',
  source: 'https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json',
  hidden: 4096,
  vocab: 128256,
  tieEmbeddings: false,
  layers: 32,
  mixer: { kind: 'gqa', heads: 32, kvHeads: 8, headDim: 128 },
  ffn: { kind: 'dense', intermediate: 14336 },
});

function ctxOf(device: DeviceSpec, engine: ComputeEngine): OpTimeContext {
  return {
    engine,
    memBandwidthBytesPerSec: engineMemBandwidthBytesPerSec(
      engine,
      device.memory.bandwidthBytesPerSec,
    ),
    collectivePaths: { tp: { kind: 'none' }, ep: { kind: 'none' }, pp: { kind: 'none' } },
    collectiveAlgorithm: 'best',
    elementwiseFusion: 'none',
    warnings: [],
  };
}

/** The device with `sfuOpsPerSec` removed from every engine: the model before the softmax term. */
function withoutSfu(device: DeviceSpec): DeviceSpec {
  return {
    ...device,
    engines: device.engines.map((e) => {
      const { sfuOpsPerSec: _sfu, ...rest } = e;
      void _sfu;
      return rest;
    }),
  };
}

function attention(p: {
  heads: number;
  kvHeads: number;
  qk: number;
  v: number;
  kvElemsPerToken: number;
  kv: 'bf16' | 'fp8_e4m3';
  qLen: number;
  kvLen: number;
  seqs: number;
  causal?: boolean;
  math?: 'bf16' | 'fp8';
}): AttentionOp {
  return {
    kind: 'attention',
    name: 'attn.core',
    seqs: p.seqs,
    heads: p.heads,
    kvHeads: p.kvHeads,
    qkHeadDim: p.qk,
    vHeadDim: p.v,
    qLen: p.qLen,
    kvLen: p.kvLen,
    causal: p.causal ?? true,
    kvSelection: { kind: 'full' },
    kvRead: { format: { dtype: p.kv }, elemsPerToken: p.kvElemsPerToken },
    kvWrite: { format: { dtype: p.kv }, elemsPerToken: p.kvElemsPerToken },
    math: p.math ?? 'bf16',
    accumulate: 'fp32',
    activationDtype: 'bf16',
  };
}

describe('decode attention stays KV-bandwidth-bound on every preset engine', () => {
  const shapes = {
    'GQA hd 128, 32/8, bf16 KV': {
      heads: 32,
      kvHeads: 8,
      qk: 128,
      v: 128,
      kvElemsPerToken: 2 * 8 * 128,
      kv: 'bf16',
    },
    'gpt-oss hd 64, 64/8, fp8 KV': {
      heads: 64,
      kvHeads: 8,
      qk: 64,
      v: 64,
      kvElemsPerToken: 2 * 8 * 64,
      kv: 'fp8_e4m3',
    },
    'absorbed MLA 128 heads, 576/512, fp8 latent': {
      heads: 128,
      kvHeads: 1,
      qk: 576,
      v: 512,
      kvElemsPerToken: 576,
      kv: 'fp8_e4m3',
    },
  } as const;
  /**
   * Presets whose stock vector FP32 peak is the FMA rate of an issue-throttled card, checked on the variant whose FP32
   * peak is the unthrottled FADD / FMUL path the softmax uses (max, subtract, sum: no FMA). CMP 170HX: FMA fused to
   * 1/32 (0.39 TFLOP/s), FADD / FMUL not (6.32 TFLOP/s, the `locked-workarounds` unlock).
   */
  const SOFTMAX_VARIANT: Record<string, Record<string, string>> = { 'cmp-170hx': { 'compute-path': 'locked-workarounds' } };
  for (const preset of DEVICE_PRESETS) {
    it(preset.id, () => {
      const selection = SOFTMAX_VARIANT[preset.id];
      const device =
        selection === undefined
          ? preset
          : applyDeviceModifiers(preset, resolveUnlocks(deviceUnlocks(preset), selection).deviceModifiers);
      const stripped = withoutSfu(device);
      device.engines.forEach((engine, i) => {
        for (const [name, shape] of Object.entries(shapes)) {
          for (const kvLen of [4097, 131073]) {
            for (const seqs of [1, 64]) {
              const op = attention({ ...shape, qLen: 1, kvLen, seqs });
              const r = opTime(op, ctxOf(device, engine));
              const before = opTime(op, ctxOf(stripped, stripped.engines[i]));
              const label = `${engine.id} ${name} kv ${kvLen} seqs ${seqs}`;
              expect(r.softmaxSec, label).toBeDefined();
              expect(r.softmaxSec!, label).toBeLessThan(r.memorySec);
              expect(r.sec, label).toBe(before.sec);
              expect(r.bound, label).toBe(before.bound);
            }
          }
        }
      });
    });
  }
});

describe('review probe F6 (bands; docs/review/practicality-review.md, docs/simulation-model.md §7)', () => {
  /** Single-device prefill tok/s @ promptLen (per sequence), batch 1, default options unless given. */
  function prefill(
    model: ModelSpec,
    quant: 'fp8' | 'bf16' | 'mxfp4-wo',
    device: DeviceSpec,
    promptLen: number,
    options: Partial<SimOptions> = {},
  ): number {
    return simulate({
      model,
      quant: resolveQuant(quant, { activations: 'bf16' }),
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen, decodeContextLen: 1024 },
      hardware: {
        device,
        topology: singleDevice(device),
        memCapacityBytes: Math.max(...device.memory.capacityOptionsBytes),
      },
      options,
    }).prefill.perSequenceTokensPerSec;
  }
  const ratio = (
    model: ModelSpec,
    quant: 'fp8' | 'bf16' | 'mxfp4-wo',
    device: DeviceSpec,
    P: number,
  ): number => prefill(model, quant, device, P) / prefill(model, quant, withoutSfu(device), P);

  it('B200 Llama-3-8B fp8: −4 % @ pp8k, −10 % @ pp32k, −16 % @ pp128k (measured 0.960 / 0.899 / 0.845)', () => {
    const r8 = ratio(LLAMA3_8B, 'fp8', B200, 8192);
    const r32 = ratio(LLAMA3_8B, 'fp8', B200, 32768);
    const r128 = ratio(LLAMA3_8B, 'fp8', B200, 131072);
    expect(r8).toBeGreaterThanOrEqual(0.93);
    expect(r8).toBeLessThanOrEqual(0.99);
    expect(r32).toBeGreaterThanOrEqual(0.85);
    expect(r32).toBeLessThanOrEqual(0.94);
    expect(r128).toBeGreaterThanOrEqual(0.78);
    expect(r128).toBeLessThanOrEqual(0.9);
  });

  it('B200 gpt-oss-120b (hd 64): −48 % @ pp128k (measured 0.519)', () => {
    const r = ratio(GPT_OSS_120B, 'mxfp4-wo', B200, 131072);
    expect(r).toBeGreaterThanOrEqual(0.45);
    expect(r).toBeLessThanOrEqual(0.6);
  });

  it('H100 Llama-3-8B bf16: unchanged (softmax overlaps the MMAs) at pp8k / 32k / 128k', () => {
    for (const P of [8192, 32768, 131072]) {
      expect(prefill(LLAMA3_8B, 'bf16', H100_SXM, P)).toBe(
        prefill(LLAMA3_8B, 'bf16', withoutSfu(H100_SXM), P),
      );
    }
  });

  it('decode tok/s is unchanged on B200 and H100 at 8k and 128k', () => {
    for (const device of [B200, H100_SXM]) {
      for (const N of [8192, 131072]) {
        const run = (d: DeviceSpec) =>
          simulate({
            model: LLAMA3_8B,
            quant: resolveQuant('fp8', { activations: 'bf16' }),
            parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
            workload: { batch: 1, promptLen: 512, decodeContextLen: N },
            hardware: { device: d, topology: singleDevice(d) },
          }).decode.tokensPerSec;
        expect(run(device)).toBe(run(withoutSfu(device)));
      }
    }
  });
});

describe('F14: Apple MoE decode gains from the GEMV path (review probe +3.3 %)', () => {
  it('M3 Ultra 80c Qwen3-235B-A22B mlx-4bit decode @ 1k: ×1.01–1.06 (measured 1.030)', () => {
    const device = DEVICE_PRESETS.find((d) => d.id === 'apple-m3-ultra-80c')!;
    const { gemvMaxRows: _rows, ...mmaOnly } = device.engines[0];
    void _rows;
    const run = (d: DeviceSpec) =>
      simulate({
        model: QWEN3_235B_A22B,
        quant: resolveQuant('mlx-4bit', { activations: 'bf16' }),
        parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
        workload: { batch: 1, promptLen: 512, decodeContextLen: 1024 },
        hardware: { device: d, topology: singleDevice(d), memCapacityBytes: 512e9 },
      }).decode.tokensPerSec;
    const r = run(device) / run({ ...device, engines: [mmaOnly] });
    expect(r).toBeGreaterThanOrEqual(1.01);
    expect(r).toBeLessThanOrEqual(1.06);
  });
});

describe('the bound stays above published FlashAttention forward throughput (docs §7 "Attention softmax")', () => {
  /** Attention-only bound: F / sec of one causal prefill op (FA benchmark shape: 16k tokens, batch 2, 16 × hd 128 or 32 × hd 64 heads). */
  function boundTflops(device: DeviceSpec, hd: 64 | 128, math: 'bf16' | 'fp8', P: number): number {
    const heads = hd === 128 ? 16 : 32;
    const op = attention({
      heads,
      kvHeads: heads,
      qk: hd,
      v: hd,
      kvElemsPerToken: 2 * heads * hd,
      kv: 'bf16',
      qLen: P,
      kvLen: P,
      seqs: 2,
      math,
    });
    const r = opTime(op, ctxOf(device, device.engines[0]));
    return r.flops / r.sec / 1e12;
  }
  it('H100 ≥ FlashAttention-3 (bf16 497 / 648, fp8 613 / 1008 TFLOP/s at hd 64 / 128)', () => {
    expect(boundTflops(H100_SXM, 64, 'bf16', 16384)).toBeGreaterThan(497);
    expect(boundTflops(H100_SXM, 128, 'bf16', 16384)).toBeGreaterThan(648);
    expect(boundTflops(H100_SXM, 64, 'fp8', 16384)).toBeGreaterThan(613);
    expect(boundTflops(H100_SXM, 128, 'fp8', 16384)).toBeGreaterThan(1008);
  });
  it('B200 ≥ FlashAttention-4 (bf16 hd 128: 1613 TFLOP/s @ 32k)', () => {
    expect(boundTflops(B200, 128, 'bf16', 32768)).toBeGreaterThan(1613);
  });
});
