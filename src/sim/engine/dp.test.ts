/**
 * Data parallelism (DP) next to tensor parallelism (TP), end to end through `simulate`, as the UI now offers it.
 *
 * What DP is in the engine. DP runs `dp` independent replicas of the whole model (each its own TP × PP group), so
 * the devices used are `tp·pp·dp` (placement.ts rank layout `rank = tpIdx + tp·(dpIdx + dp·ppIdx)`). The global
 * batch splits evenly: each replica serves `B_r = batch / dp` sequences (validate.ts rejects a batch DP does not
 * divide). The replicas never talk to each other (no collective crosses them, except a MoE mode-B EP group), so
 * one replica at `B_r` runs exactly like a single model instance at `B_r`, and:
 *   aggregate tok/s(dp, batch) = dp · tok/s(one replica at batch/dp)        (pipeline.ts: tokens/s = dp·B_r/cycle)
 *   per-sequence tok/s(dp, batch) = per-sequence tok/s(one replica at batch/dp)
 * The memory fit is per replica: every device of a replica holds its 1/tp shard of the whole model, so DP 2 on
 * 2 GPUs puts the full weights on each GPU where TP 2 puts half.
 *
 * DP 2 against TP 2 on 2 GPUs, dense 8B (REF_DENSE_8B, bf16, decode @ 4k). Per sequence, DP is always slower: a
 * sequence runs on one GPU that reads all 16.4 GB of weights per token, where TP spreads it over both GPUs (each
 * reads 8.2 GB) for the price of two all-reduces per layer. Aggregate decode depends on what bounds the step:
 * - while decode is memory bound (small batch), TP wins on a fast link: both GPUs read half the weights once per
 *   step for the whole batch, DP's GPUs each read all of them for half the batch, so TP moves half the bytes per
 *   token (here 2 × RTX 5090 over PCIe 5.0 P2P, batch 2: TP ≈ 424 tok/s, DP ≈ 228);
 * - DP wins where TP's all-reduces cost more than the weight reads they save: prefill (compute bound, the
 *   all-reduces only add), decode over a slow network (2 machines over 10 GbE TCP, batch ≥ 32), and decode at a
 *   batch large enough to be compute bound (PCIe P2P, batch 512).
 * See docs/simulation-model.md §7 "Pipeline and DP" and docs/modeling-method.md "Multiple devices".
 */
import { describe, expect, it } from 'vitest';
import { ETHERNET_10G_TCP, PCIE_5_X16_DIRECT } from '../hw/presets/links';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH } from '../hw/presets/storage';
import { nodeTopology } from '../hw/presets/topologies';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_DENSE_8B } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import { simulate } from './simulate';
import type { DeploymentConfig, SimInput, SimResult } from './types';
import { validateInput } from './validate';

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };
const DENSE = { model: REF_DENSE_8B, quant: BF16 };

function load(repo: string): { model: ModelSpec; quant: QuantConfig } {
  const r = mapHfModel(fixtureRepoFiles(repo));
  if (r.kind !== 'ok') throw new Error(`${repo}: ${r.kind}`);
  return { model: r.model, quant: r.quant.quant };
}
const FLASH_NEXT = load('Qwen/Qwen3.8-Flash-Next');

/** n RTX 5090s in one box over direct PCIe 5.0 x16 P2P. */
function box(n: number): SimInput['hardware'] {
  return {
    device: RTX_5090,
    topology: nodeTopology({ devicesPerNode: n, nodes: 1, intraNode: PCIE_5_X16_DIRECT, device: RTX_5090 }),
  };
}

/** n machines of one RTX 5090 each over 10 GbE TCP. */
function overEthernet(n: number): SimInput['hardware'] {
  return {
    device: RTX_5090,
    topology: nodeTopology({ devicesPerNode: 1, nodes: n, interNode: ETHERNET_10G_TCP, device: RTX_5090 }),
  };
}

function run(
  m: { model: ModelSpec; quant: QuantConfig },
  parallel: Partial<ParallelConfig>,
  batch: number,
  hardware: SimInput['hardware'],
  deployment?: DeploymentConfig,
): SimResult {
  const input: SimInput = {
    model: m.model,
    quant: m.quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    workload: { batch, promptLen: 4096, decodeContextLen: 4096 },
    hardware,
    ...(deployment !== undefined ? { deployment } : {}),
  };
  expect(validateInput(input)).toEqual([]);
  return simulate(input);
}

describe('DP 2 on 2 GPUs: two independent replicas', () => {
  it.each([2, 8, 32])('batch %i: aggregate = 2 × one GPU at batch/2; per sequence the same', (batch) => {
    const dp = run(DENSE, { dp: 2 }, batch, box(2));
    const one = run(DENSE, {}, batch / 2, box(1));
    expect(dp.devicesUsed).toBe(2);
    for (const phase of ['prefill', 'decode'] as const) {
      expect(dp[phase].tokensPerSec).toBeCloseTo(2 * one[phase].tokensPerSec, 6);
      expect(dp[phase].perSequenceTokensPerSec).toBeCloseTo(one[phase].perSequenceTokensPerSec, 9);
      // No collective: the replicas never talk.
      expect(dp[phase].stages.flatMap((s) => s.layers.flatMap((l) => l.ops)).some((o) => o.op.kind === 'collective')).toBe(false);
    }
  });

  it('the memory fit is per replica: each GPU holds the whole model under DP, half under TP', () => {
    const dp = run(DENSE, { dp: 2 }, 2, box(2));
    const tp = run(DENSE, { tp: 2 }, 2, box(2));
    const one = run(DENSE, {}, 1, box(1));
    expect(dp.memory.perStage[0].weightBytes).toBe(one.memory.perStage[0].weightBytes);
    expect(tp.memory.perStage[0].weightBytes).toBeLessThan(0.51 * dp.memory.perStage[0].weightBytes);
    // Each replica holds B_r = batch / dp sequences; the most it could hold is the single GPU's.
    expect(dp.memory.perStage[0].sequenceStateBytes).toBe(one.memory.perStage[0].sequenceStateBytes);
    expect(dp.memory.maxSequencesPerReplica).toBe(one.memory.maxSequencesPerReplica);
  });

  it('a batch DP does not divide is rejected', () => {
    const input: SimInput = {
      model: DENSE.model,
      quant: DENSE.quant,
      parallel: { tp: 1, pp: 1, dp: 2, ep: 1 },
      workload: { batch: 3, promptLen: 4096, decodeContextLen: 4096 },
      hardware: box(2),
    };
    expect(validateInput(input)).toContain('workload.batch (3) must be divisible by dp (2)');
  });
});

describe('DP 2 against TP 2 on 2 GPUs (dense 8B, bf16)', () => {
  // Per sequence, DP is always slower: one GPU reads all the weights for each token (header).
  it.each([2, 8, 32, 128])('batch %i, PCIe P2P: DP is slower per sequence', (batch) => {
    const dp = run(DENSE, { dp: 2 }, batch, box(2));
    const tp = run(DENSE, { tp: 2 }, batch, box(2));
    expect(dp.decode.perSequenceTokensPerSec).toBeLessThan(tp.decode.perSequenceTokensPerSec);
  });

  it('PCIe P2P: TP has the higher aggregate decode while it is memory bound, DP once it is compute bound', () => {
    // Memory bound: TP reads half the weights per GPU for the whole batch, DP all of them for half the batch.
    for (const batch of [2, 8, 32]) {
      const dp = run(DENSE, { dp: 2 }, batch, box(2));
      const tp = run(DENSE, { tp: 2 }, batch, box(2));
      expect(tp.decode.tokensPerSec).toBeGreaterThan(dp.decode.tokensPerSec);
    }
    // Compute bound (batch 512): the weight reads no longer bound the step; TP still pays its all-reduces.
    const dp = run(DENSE, { dp: 2 }, 512, box(2));
    const tp = run(DENSE, { tp: 2 }, 512, box(2));
    expect(dp.decode.tokensPerSec).toBeGreaterThan(tp.decode.tokensPerSec);
    // Prefill is compute bound at any batch: DP avoids the all-reduces.
    expect(dp.prefill.tokensPerSec).toBeGreaterThan(tp.prefill.tokensPerSec);
  });

  it('2 machines over 10 GbE TCP: DP has the higher aggregate decode from batch 32, and higher prefill', () => {
    for (const batch of [32, 128]) {
      const dp = run(DENSE, { dp: 2 }, batch, overEthernet(2));
      const tp = run(DENSE, { tp: 2 }, batch, overEthernet(2));
      expect(dp.decode.tokensPerSec).toBeGreaterThan(tp.decode.tokensPerSec);
      expect(dp.decode.perSequenceTokensPerSec).toBeGreaterThan(tp.decode.perSequenceTokensPerSec);
      expect(dp.prefill.tokensPerSec).toBeGreaterThan(5 * tp.prefill.tokensPerSec);
    }
  });
});

describe('DP with speculative decoding, offload and MoE', () => {
  it('MTP spec-dec at DP 2: each replica runs the draft and verify of its own sequences', () => {
    const mtp: DeploymentConfig = {
      speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.8 } },
    };
    const m = { model: REF_DENSE_8B_MTP, quant: BF16 };
    const dp = run(m, { dp: 2 }, 4, box(2), mtp);
    const one = run(m, {}, 2, box(1), mtp);
    expect(dp.decode.speculative?.speedup).toBeCloseTo(one.decode.speculative?.speedup ?? 0, 9);
    expect(dp.decode.tokensPerSec).toBeCloseTo(2 * one.decode.tokensPerSec, 6);
  });

  it('routed experts in host RAM at DP 2: the node holds both replicas\u2019 offloaded experts', () => {
    const offload: DeploymentConfig = {
      offload: { routedExperts: { tier: 'host_ram', residentFraction: 0.5, hitRatio: { kind: 'uniform' } } },
    };
    const host = { ram: DDR5_6000_2CH };
    const dp = run(FLASH_NEXT, { dp: 2 }, 2, { ...box(2), host }, offload);
    const one = run(FLASH_NEXT, {}, 1, { ...box(1), host }, offload);
    const ram = (r: SimResult) => r.memory.tiers.find((t) => t.tier === 'host_ram')?.bytes ?? 0;
    expect(ram(one)).toBeGreaterThan(0);
    expect(ram(dp)).toBe(2 * ram(one));
    expect(dp.decode.tokensPerSec).toBeLessThanOrEqual(2 * one.decode.tokensPerSec * (1 + 1e-9));
  });

  it('MoE mode B: EP 2 at TP 1, DP 2 spans both replicas, with an all-to-all of tokens', () => {
    const r = run(FLASH_NEXT, { dp: 2, ep: 2 }, 2, box(2));
    const ops = r.decode.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
    expect(ops.some((o) => o.op.kind === 'collective' && o.op.collective === 'all_to_all')).toBe(true);
    expect(r.devicesUsed).toBe(2);
  });
});
