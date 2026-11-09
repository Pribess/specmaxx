/**
 * End to end: Nemotron-H fixtures → `mapHfModel` → `simulate`, decode @ 8k, batch 1, one device (tp = 1).
 * Each rate is compared with the bandwidth bound `memBW / streamedBytes`, where the streamed bytes are derived by
 * hand and asserted equal to the sum over the lowered decode ops.
 */

import { describe, expect, it } from 'vitest';
import { simulate } from '../../../engine/simulate';
import type { Workload } from '../../../engine/types';
import { singleDevice } from '../../../hw/presets';
import { B200, H200_SXM } from '../../../hw/presets/nvidia-datacenter';
import type { DeviceSpec } from '../../../hw/types';
import { bytesPerElement } from '../../../ir/dtype';
import { opBytes } from '../../../ir/metrics';
import type { ParallelConfig } from '../../../ir/model';
import { lowerModel } from '../../../lower/lower';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { mapHfModel } from '../index';

const P1: ParallelConfig = { tp: 1, pp: 1, dp: 1, ep: 1 };
/** Batch 1, 8192-token prompt, decode at an 8192-token context (the new token attends to 8193 keys). */
const WORKLOAD: Workload = { batch: 1, promptLen: 8192, decodeContextLen: 8192 };

/** Asserts lo·ref ≤ actual ≤ hi·ref. */
function expectWithin(actual: number, ref: number, lo: number, hi: number): void {
  expect(actual).toBeGreaterThanOrEqual(lo * ref);
  expect(actual).toBeLessThanOrEqual(hi * ref);
}

/** Simulate `id` and sum the bytes of its lowered decode step by kind. */
function run(id: string, device: DeviceSpec) {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  const hardware = { device, topology: singleDevice(device) };
  const sim = simulate({
    model: r.model,
    quant: r.quant.quant,
    parallel: P1,
    workload: WORKLOAD,
    hardware,
  });
  const ops = lowerModel(r.model, {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 8193,
    quant: r.quant.quant,
    parallel: P1,
  }).stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
  const sum = { gemmWeights: 0, otherWeights: 0, kvRead: 0, stateRead: 0, stateWrite: 0 };
  for (const op of ops) {
    const b = opBytes(op);
    if (op.kind === 'gemm') sum.gemmWeights += b.weightBytes;
    else sum.otherWeights += b.weightBytes;
    sum.kvRead += b.kvReadBytes;
    sum.stateRead += b.stateReadBytes;
    sum.stateWrite += b.stateWriteBytes;
  }
  return { r, sim, sum, quant: r.quant.quant, memBW: device.memory.bandwidthBytesPerSec };
}

describe('Nemotron-3.5 Lightning 30B-A3B, bf16, 1 × H200 SXM (4.8 TB/s)', () => {
  const { sim, sum, memBW } = run('nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16', H200_SXM);

  it('decode @ 8k sits just below the weights + KV + SSM-state bandwidth bound (≈ 726 tok/s)', () => {
    // GEMM weights (bf16, 2 B), per token at batch 1 (top-6 of 128 experts → 6 distinct experts):
    //   Mamba 23 × (in_proj 2688·10,304 + out_proj 4096·2688)                    = 23 × 38,707,200
    //   MoE   23 × (router 128·2688 + 6 experts · 2·2688·1856 + shared 2·2688·3712)
    //   attn   6 × (qkv 2688·4608 + o 4096·2688)
    //   lm_head 131,072 · 2688
    const mamba = 23 * (2688 * 10_304 + 4096 * 2688);
    const moe = 23 * (128 * 2688 + 6 * 2 * 2688 * 1856 + 2 * 2688 * 3712);
    const attn = 6 * (2688 * 4608 + 4096 * 2688);
    const head = 131_072 * 2688;
    expect(sum.gemmWeights).toBe(2 * (mamba + moe + attn + head));
    // Small per-layer vectors (norm γ, conv taps + bias, out-norm γ, embedding row) are < 0.2 % of the bytes.
    const streamed =
      sum.gemmWeights +
      sum.otherWeights +
      sum.kvRead + // 6 layers × 8193 tokens × (2·2·128 elements × 2 B) = 50,337,792 B
      sum.stateRead +
      sum.stateWrite; // 2 × 49,082,368 B (fp32 SSM state + conv state, read and written)
    expect(sum.kvRead).toBe(6 * 8193 * 1024);
    expect(sum.stateRead).toBe(49_082_368);
    expect(sum.stateWrite).toBe(49_082_368);
    // ≈ 6.60 GB per token → 4.8e12 / 6.604e9 ≈ 726.8 tok/s; the simulator reaches 99.5 % of it (the rest is the
    // non-bandwidth-bound kernels: the ~60 MB of activations, the conv and the norms).
    const bound = memBW / streamed;
    expect(bound).toBeCloseTo(726.8, 0);
    expectWithin(sim.decode.tokensPerSec, bound, 0.97, 1.0);
    // "3B active" bf16 = 6.45 GB of weights alone would give 743.5 tok/s; the SSM state adds 1.5 % to the traffic.
    expect(sim.decode.tokensPerSec).toBeLessThan(memBW / (2 * 3_227_751_872));
  });

  it('the model fits one H200 without warnings', () => {
    expect(sim.warnings).toEqual([]);
    expect(sim.memory.fits).toBe(true);
  });
});

describe('Nemotron-3.5 Super-VL 120B-A12B, NVFP4 + FP8, 1 × B200 (7.7 TB/s)', () => {
  const { sim, sum, quant, memBW } = run(
    'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4',
    B200,
  );

  it('decode @ 8k sits just below the weights + KV + SSM-state bandwidth bound (≈ 760 tok/s)', () => {
    const nvfp4 = bytesPerElement({ dtype: 'nvfp4' }); // 0.5 B + 1 fp8 scale per 16 elements = 0.5625 B
    expect(nvfp4).toBe(0.5625);
    // GEMM weights per token at batch 1 (top-22 of 512 experts → 22 distinct experts), by format:
    //   NVFP4: routed 40 × 22 · 2·1024·2688; shared 40 × 2·4096·5376
    //   FP8:   Mamba 40 × (in_proj 4096·18,560 + out_proj 8192·4096); attn 8 × (qkv 4096·4608 + o 4096·4096);
    //          lm_head 131,072·4096
    //   bf16:  router 40 × 512·4096; latent in/out 40 × 2·4096·1024
    const routed = 40 * 22 * 2 * 1024 * 2688;
    const shared = 40 * 2 * 4096 * 5376;
    const fp8 =
      40 * (4096 * 18_560 + 8192 * 4096) + 8 * (4096 * 4608 + 4096 * 4096) + 131_072 * 4096;
    const bf16 = 40 * 512 * 4096 + 40 * 2 * 4096 * 1024;
    expect(quant.modules?.moe_latent?.weights.dtype).toBe('bf16');
    expect(sum.gemmWeights).toBe((routed + shared) * nvfp4 + fp8 * 1 + bf16 * 2);
    // Per-token state: 8 × 8193 × (2·2·128 fp8 elements) = 33,558,528 B KV; 2 × 170,229,760 B SSM + conv state.
    expect(sum.kvRead).toBe(8 * 8193 * 512);
    expect(sum.stateRead).toBe(170_229_760);
    expect(sum.stateWrite).toBe(170_229_760);
    const streamed =
      sum.gemmWeights + sum.otherWeights + sum.kvRead + sum.stateRead + sum.stateWrite;
    // ≈ 10.14 GB per token → 7.7e12 / 10.14e9 ≈ 759 tok/s upper bound (the SSM + conv state, read and written,
    // is 2 × 170,229,760 B ≈ 0.34 GB of it, 3.4 % of the traffic).
    const bound = memBW / streamed;
    expect(bound).toBeGreaterThan(755);
    expect(bound).toBeLessThan(765);
    expectWithin(sim.decode.tokensPerSec, bound, 0.97, 1.0);
  });

  it('the NVFP4 checkpoint fits one B200 without warnings', () => {
    expect(sim.warnings).toEqual([]);
    expect(sim.memory.fits).toBe(true);
  });
});
