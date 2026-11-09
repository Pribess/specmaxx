/**
 * Kimi K3 / K3.1-Pro end to end: fixture → `mapHfModel` → `simulate` decode @ 8k on 2 × HGX B200 (16 GPUs, 180 GB
 * each; the ~1.6 TB of MXFP4 experts + BF16 rest need more than one node) as tp 8 × pp 2 with ep 8, batch 1.
 *
 * Sanity against the bandwidth bound, by hand (Kimi K3):
 * - Active weights per token: 16 of 896 experts × 3 · 3584 · 3072 × 92 layers = 48.6 G params in MXFP4 (0.53125 B,
 *   e2m1 + 8/32 scale) = 25.8 GB; the dense rest (105.36 G active − 48.6 G routed − 1.17 G embedding) = 55.6 G params
 *   in BF16 = 111 GB; so ≈ 137 GB stream per token, ≈ 17.1 GB per device when split 8 ways over the 2 stages.
 * - Replicated per-rank reads add to that: `moe.latent_in` (92 · 7168 · 3584 · 2 B = 4.7 GB, read in full by every
 *   rank), the routers (92 · 896 · 7168 · 2 B = 1.2 GB) and MLA `q_a` / `kv_a` (24 · 7168 · 2112 · 2 B = 0.7 GB), so
 *   the two stages read ≈ 23.0 GB of weights per rank per token (`stage weightBytes` below: 11.30 + 11.75 GB).
 * - At 7.7 TB/s per B200 that is 2.99 ms of weight reads per token, a 334 tok/s bound for a pp-2 pipeline at batch 1
 *   (both stages run in series for one sequence). The simulated 225 tok/s @ 8k sits below it: the rest is the TP
 *   all-reduces (two per layer, plus the latent all-reduce per MoE layer) and the pipeline hop over the NVLink /
 *   NDR fabric at batch 1 (the hop carries the running sum plus 4 Attention-Residuals summaries, 5 · 7168 · 2 B =
 *   71,680 B per token; that costs 0.06 tok/s against a plain 14,336 B hop: 225.11 → 225.05).
 */
import { describe, expect, it } from 'vitest';
import { simulate } from '../../../engine/simulate';
import { B200 } from '../../../hw/presets/nvidia-datacenter';
import { HGX_B200_16_NDR } from '../../../hw/presets/topologies';
import { bytesPerElement } from '../../../ir/dtype';
import { opBytes } from '../../../ir/metrics';
import type { ParallelConfig } from '../../../ir/model';
import { lowerModel, ppBoundaryResidualVectors } from '../../../lower/lower';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { mapHfModel } from '../index';

const PARALLEL: ParallelConfig = { tp: 8, pp: 2, dp: 1, ep: 8 };
const CTX = 8192;

function mapOk(id: string) {
  const r = mapHfModel(fixtureRepoFiles(id));
  if (r.kind !== 'ok') throw new Error(`${id}: ${JSON.stringify(r)}`);
  return r;
}

describe('Kimi K3 / K3.1-Pro decode @ 8k on 2 × HGX B200 (tp 8, pp 2, ep 8, batch 1)', () => {
  it.each([
    ['moonshotai/Kimi-K3', 225.1],
    ['moonshotai/Kimi-K3.1-Pro', 221.6],
  ])(
    '%s: fits, and decodes at %f tok/s @ 8k, below the per-token weight-read bound',
    (id, quoted) => {
      const r = mapOk(id);
      const q = r.quant.quant;
      expect(bytesPerElement(q.weights)).toBeCloseTo(0.53125, 6); // mxfp4: 4 bit + 8/32 scale bits
      const s = simulate({
        model: r.model,
        quant: q,
        parallel: PARALLEL,
        workload: { batch: 1, promptLen: CTX, decodeContextLen: CTX },
        hardware: { device: B200, topology: HGX_B200_16_NDR },
      });
      expect(s.warnings).toEqual([]);
      expect(s.memory.fits).toBe(true);
      // ≈ 100 GB per device of the 180 GB: the 1.6 TB checkpoint over 16 devices plus the caches.
      for (const stage of s.memory.perStage) expect(stage.totalBytes / 1e9).toBeGreaterThan(95);
      expect(s.memory.perStage.every((st) => st.totalBytes / 1e9 < 110)).toBe(true);

      // Per-token weight (+ cache) reads of each stage on one rank, from the lowered decode step at kvLen 8193.
      const lowered = lowerModel(r.model, {
        phase: 'decode',
        seqs: 1,
        qLen: 1,
        kvLen: CTX + 1,
        quant: q,
        parallel: PARALLEL,
      });
      const stageBytes = lowered.stages.map((st) =>
        st.layers
          .flatMap((l) => l.ops)
          .reduce((sum, op) => sum + opBytes(op).weightBytes + opBytes(op).kvReadBytes, 0),
      );
      expect(stageBytes).toHaveLength(2);
      const perRank = stageBytes.reduce((a, b) => a + b, 0);
      // 23.0 GB per rank per token (K3: 11.30 + 11.75 GB; the KV / latent reads at 8k are 0.2 GB of it).
      expect(perRank / 1e9).toBeGreaterThan(22.5);
      expect(perRank / 1e9).toBeLessThan(24);
      // Formula: bound = memBW / Σ_stages bytes   (a batch-1 pipeline runs its stages in series)
      const bound = B200.memory.bandwidthBytesPerSec / perRank;
      expect(bound).toBeGreaterThan(325);
      expect(s.decode.tokensPerSec).toBeLessThan(bound);
      expect(s.decode.tokensPerSec).toBeGreaterThan(0.6 * bound);
      expect(Math.abs(s.decode.tokensPerSec / quoted - 1)).toBeLessThan(0.005);
      expect(s.prefill.tokensPerSec).toBeGreaterThan(50_000);
    },
  );
});

describe('Kimi K3: PP boundaries carry the Attention-Residuals summaries', () => {
  const m = mapOk('moonshotai/Kimi-K3').model;
  const h = 7168;
  // Formula: N_live(i) = 1 + ceil(i / B), B = attn_res_block_size 12; S = T·N_live·h·bpe(act)
  it('N_live = 1 + ceil(i/12) at every boundary: pp 2 before layer 46 → 5; pp 4 before 23 / 46 / 69 → 3 / 5 / 7', () => {
    expect(ppBoundaryResidualVectors(m, 46)).toBe(5);
    expect([23, 46, 69].map((i) => ppBoundaryResidualVectors(m, i))).toEqual([3, 5, 7]);
    for (let i = 1; i < m.layers.length; i++)
      expect(ppBoundaryResidualVectors(m, i)).toBe(1 + Math.ceil(i / 12));
  });

  it.each([
    ['decode', 4, 1, 4],
    ['prefill', 1, 512, 512],
  ] as const)(
    'pp 2 %s: the send is T · 5 · h · 2 B = 71,680 B per token in BF16',
    (phase, seqs, qLen, T) => {
      const lowered = lowerModel(m, {
        phase,
        seqs,
        qLen,
        kvLen: qLen,
        quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
        parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
      });
      expect(lowered.stages[0].layers.at(-1)?.layerIndex).toBe(45);
      const send = lowered.stages[0].sendToNext;
      expect(send?.messageBytes).toBe(T * 5 * h * 2);
      expect((send?.messageBytes ?? 0) / T).toBe(71_680); // a plain residual would send 14,336 B
    },
  );
});
