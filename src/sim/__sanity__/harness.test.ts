/**
 * Smoke test of the sanity-check harness: one hand-written model, one HF fixture, and multi-device topologies run
 * through `simulateReport`; the rates are printed (for docs/sanity-check.md) and must be finite and positive.
 */

import { describe, expect, it } from 'vitest';
import { DGX_SPARK_PAIR, HGX_H100_8, pcieBox } from '../hw/presets/topologies';
import { PCIE_4_X16 } from '../hw/presets/links';
import { RTX_3090 } from '../hw/presets/nvidia';
import { bitsPerElement } from '../ir/dtype';
import { REF_DENSE_8B, REF_MOE_30B_A3B } from '../models/reference';
import { sanityRepoFiles } from './configs.testutil';
import { bpwFormat, resolveQuant, simulateReport, simpleDecoder, type SanityInput, type SanityResult } from './harness';

/** Prints one line per result and checks the rates. */
function expectSane(label: string, r: SanityResult): void {
  console.log(
    `${label}: prefill ${r.prefillTokensPerSec.toFixed(1)} tok/s total, ${r.prefillPerSequenceTokensPerSec.toFixed(1)} tok/s/seq ` +
      `(${r.prefillBound.dominant}), ` +
      `decode ${r.decodePerSequenceTokensPerSec.toFixed(2)} tok/s/seq, ${r.decodeTokensPerSec.toFixed(2)} tok/s total ` +
      `(${r.decodeBound.dominant}) @ ctx ${r.decodeContextLen}, fits ${r.fits}` +
      (r.sim.warnings.length > 0 ? `\n  warnings: ${r.sim.warnings.join(' | ')}` : ''),
  );
  for (const v of [r.prefillTokensPerSec, r.prefillPerSequenceTokensPerSec, r.decodeTokensPerSec, r.decodePerSequenceTokensPerSec]) {
    expect(Number.isFinite(v) && v > 0).toBe(true);
  }
}

const PP512_TG128 = { batch: 1, promptLen: 512, genLen: 128 };

describe('sanity harness smoke', () => {
  it('Qwen3-8B shape (REF_DENSE_8B) Q4_K_M on one RTX 4090, pp512 / tg128', () => {
    const r = simulateReport({ model: { kind: 'spec', spec: REF_DENSE_8B }, quant: 'q4_k_m', hardware: { deviceId: 'rtx-4090' }, workload: PP512_TG128 });
    expectSane('REF_DENSE_8B q4_k_m 1×4090', r);
    expect(r.decodeBound.dominant).toBe('memory');
    expect(r.decodeContextLen).toBe(64);
    expect(r.fits).toBe(true);
  });

  it('Qwen/Qwen3.8-27B (HF fixture) MLX 4-bit on an M3 Ultra', () => {
    const r = simulateReport({
      model: { kind: 'hf', files: sanityRepoFiles('Qwen/Qwen3.8-27B') },
      quant: 'mlx-4bit',
      hardware: { deviceId: 'apple-m3-ultra-80c' },
      workload: PP512_TG128,
    });
    expectSane('Qwen3.8-27B mlx-4bit M3 Ultra 80c', r);
    // The preset replaces the repo's bf16 weights with MLX affine int4 g64 (4.5 bpw).
    expect(r.quant.weights).toEqual({ dtype: 'int4', extraBitsPerElem: 0.5 });
  });

  it('REF_DENSE_8B Q8_0 on 2 × RTX 3090 over PCIe 4 x16: tp 2 and pp 2 (layer split)', () => {
    const base: SanityInput = {
      model: { kind: 'spec', spec: REF_DENSE_8B },
      quant: 'q8_0',
      hardware: { deviceId: 'rtx-3090', topology: pcieBox(RTX_3090, 2, PCIE_4_X16) },
      workload: PP512_TG128,
    };
    const tp = simulateReport(base);
    expectSane('REF_DENSE_8B q8_0 2×3090 PCIe tp2', tp);
    const pp = simulateReport({ ...base, parallel: { tp: 1, pp: 2 } });
    expectSane('REF_DENSE_8B q8_0 2×3090 PCIe pp2', pp);
    const p2p = simulateReport({ ...base, hardware: { ...base.hardware, unlocks: { 'pcie-p2p': 'direct' } } });
    expectSane('REF_DENSE_8B q8_0 2×3090 PCIe tp2 + P2P', p2p);
  });

  it('presets by id: DGX Spark pair, HGX H100 ×8 with FP8, a hand-written MoE', () => {
    expectSane(
      'REF_DENSE_8B nvfp4 2×DGX Spark tp2',
      simulateReport({
        model: { kind: 'spec', spec: REF_DENSE_8B },
        quant: 'nvfp4',
        hardware: { deviceId: 'dgx-spark', topology: DGX_SPARK_PAIR.id },
        workload: PP512_TG128,
      }),
    );
    const moe = simpleDecoder({
      name: 'Qwen3-30B-A3B shape',
      source: 'https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json',
      hidden: 2048,
      vocab: 151936,
      tieEmbeddings: false,
      layers: 48,
      mixer: { kind: 'gqa', heads: 32, kvHeads: 4, headDim: 128, qkNorm: true },
      ffn: { kind: 'moe', experts: 128, topK: 8, expertIntermediate: 768 },
    });
    expectSane(
      'Qwen3-30B-A3B fp8 8×H100 tp8 ep8 batch 64',
      simulateReport({
        model: { kind: 'spec', spec: moe },
        quant: 'fp8',
        hardware: { deviceId: 'h100-sxm', topology: HGX_H100_8 },
        parallel: { ep: 8 },
        workload: { batch: 64, promptLen: 1024, genLen: 1024 },
      }),
    );
  });

  it('quant presets: bpw formats and W8A8 modules', () => {
    expect(bitsPerElement(bpwFormat(4.85))).toBeCloseTo(4.85, 10);
    expect(bitsPerElement(bpwFormat(8.5))).toBe(8.5);
    const fp8 = resolveQuant('fp8');
    expect(fp8.modules?.routed_experts).toEqual({ weights: { dtype: 'fp8_e4m3' }, gemmInput: 'fp8_e4m3' });
    expect(fp8.lmHead).toEqual({ dtype: 'fp16' });
    expect(resolveQuant('q4_k_m').lmHead).toEqual(bpwFormat((210 * 8) / 256));
    expect(() => resolveQuant('native')).toThrow();
  });

  it('deployment: absent and {} give identical results; offloaded experts make decode transfer-bound; a checkpoint drafter resolves', () => {
    const base: SanityInput = { model: { kind: 'spec', spec: REF_DENSE_8B }, quant: 'q4_k_m', hardware: { deviceId: 'rtx-4090' }, workload: PP512_TG128 };
    const off = simulateReport(base);
    const empty = simulateReport({ ...base, deployment: {} });
    expect(empty.decodePerSequenceTokensPerSec).toBe(off.decodePerSequenceTokensPerSec);
    expect(empty.prefillPerSequenceTokensPerSec).toBe(off.prefillPerSequenceTokensPerSec);
    // A MoE model with every routed expert in host RAM (nothing cached) over the 4090's PCIe 4.0 x16: decode becomes
    // transfer-bound. (Whole-layer streaming of a dense model, the former case here, was removed.)
    const moe: SanityInput = { ...base, model: { kind: 'spec', spec: REF_MOE_30B_A3B } };
    const moeOff = simulateReport(moe);
    const streamed = simulateReport({
      ...moe,
      deployment: { offload: { routedExperts: { tier: 'host_ram', residentFraction: 0, hitRatio: { kind: 'uniform' } } } },
    });
    expect(streamed.decodeBound.dominant).toBe('transfer');
    expect(streamed.decodePerSequenceTokensPerSec).toBeLessThan(moeOff.decodePerSequenceTokensPerSec);
    // DeepSeek-V4.1-Flash ships its DSpark drafter in the checkpoint; REF_DENSE_8B has none.
    const flash = simulateReport({
      model: { kind: 'hf', files: sanityRepoFiles('deepseek-ai/DeepSeek-V4.1-Flash') },
      quant: 'native',
      hardware: { deviceId: 'b200' },
      workload: PP512_TG128,
      deployment: { speculative: { method: 'dspark', drafter: { kind: 'checkpoint' }, draftLength: 5, acceptance: { kind: 'iid', alpha: 0.7 } } },
    });
    expect(flash.sim.decode.speculative?.expectedTokensPerStep).toBeCloseTo((1 - 0.7 ** 6) / 0.3, 12);
    expect(() =>
      simulateReport({ ...base, deployment: { speculative: { method: 'dspark', drafter: { kind: 'checkpoint' }, draftLength: 5, acceptance: { kind: 'iid', alpha: 0.7 } } } }),
    ).toThrow(/no in-checkpoint block drafter/);
  });
});
