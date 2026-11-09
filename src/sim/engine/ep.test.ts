/**
 * Expert parallelism (EP) next to tensor parallelism (TP) at DP = 1: what EP changes in
 * the results, end to end through `simulate`, on 2 and 4 GPUs.
 *
 * Why EP looks like TP here. At DP = 1 a valid ep divides tp (lower/moe.ts `moeParallelErrors`; the other branch,
 * ep a multiple of tp spanning tp·dp ranks, needs DP > 1), so the engine runs its MoE mode A: the EP group lies
 * inside the TP group and adds no devices (devices = TP × PP). After attention every TP rank already holds all T
 * tokens, so no token moves: each rank runs the experts it holds (E/ep whole experts, each sliced etp = tp/ep ways)
 * on the tokens routed to them, and the expert outputs are partial sums of the layer output, summed by the one
 * `moe.tp_all_reduce` of T·h that TP alone needs as well. The all-to-all dispatch and combine exist only in mode B
 * (EP across DP replicas, i.e. DP attention; dp.test.ts). Hence, EP = TP against EP = 1:
 * - communication is identical (same collectives, same sizes, no all_to_all);
 * - expert FLOP in prefill rises by the EP load imbalance ρ (DEFAULT_EP_IMBALANCE 1.2): the busiest device holds
 *   ⌈ρ·T·k/ep⌉ token-expert rows instead of the mean T·k/ep, while TP splits every expert evenly;
 * - but each device now reads and writes only its own token-expert rows (TP: every device reads the input and
 *   writes a partial output for all T·k rows, on its fe/tp slice), so where the expert GEMMs are memory bound (an
 *   RTX 5090 at m ≈ 80 tokens per expert, bf16) prefill gets faster with EP, not slower;
 * - decode reads whole experts: the busiest device needs ⌈T·k/ep⌉ of them where TP reads a 1/tp slice of the
 *   ~T·k distinct experts, so decode moves by the rounding (+/− a few %, e.g. k 10 on 4 GPUs: 3 whole experts for a
 *   mean of 2.5);
 * - on models with an expert-output norm (Muse Spark) etp > 1 needs a reduce-scatter and a sum-of-squares
 *   all-reduce over the etp ranks sharing each expert, which EP = TP (etp = 1) removes: there EP cuts comm.
 * See docs/simulation-model.md §5 "MoE modes A/B" and docs/modeling-method.md "Multiple devices".
 */
import { describe, expect, it } from 'vitest';
import { PCIE_5_X16_DIRECT } from '../hw/presets/links';
import { RTX_5090 } from '../hw/presets/nvidia';
import { nodeTopology } from '../hw/presets/topologies';
import { DDR5_6000_2CH } from '../hw/presets/storage';
import type { CollectiveOp } from '../ir/ops';
import type { MoeBlock, ModelSpec, QuantConfig } from '../ir/model';
import { DEFAULT_EP_IMBALANCE, moeLayout, moeRouting } from '../lower/moe';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { simulate } from './simulate';
import type { DeploymentConfig, PhaseResult, SimInput, SimResult } from './types';
import { validateInput } from './validate';

function load(repo: string): { model: ModelSpec; quant: QuantConfig } {
  const r = mapHfModel(fixtureRepoFiles(repo));
  if (r.kind !== 'ok') throw new Error(`${repo}: ${r.kind}`);
  return { model: r.model, quant: r.quant.quant };
}

const FLASH_NEXT = load('Qwen/Qwen3.8-Flash-Next');
const MUSE_SPARK = load('unsloth/Muse-Spark-Open-816B-A42B');
const V41_FLASH = load('deepseek-ai/DeepSeek-V4.1-Flash');

/** n RTX 5090s in one box over direct PCIe 5.0 x16 P2P. */
function box(n: number): SimInput['hardware'] {
  return {
    device: RTX_5090,
    topology: nodeTopology({
      devicesPerNode: n,
      nodes: 1,
      intraNode: PCIE_5_X16_DIRECT,
      device: RTX_5090,
    }),
  };
}

/** Prompt 4096 tokens, batch 1, decode at a 4096-token context. */
const WORKLOAD = { batch: 1, promptLen: 4096, decodeContextLen: 4096 };

function run(
  m: { model: ModelSpec; quant: QuantConfig },
  tp: number,
  ep: number,
  deployment?: DeploymentConfig,
  hardware: SimInput['hardware'] = box(tp),
): SimResult {
  const input: SimInput = {
    model: m.model,
    quant: m.quant,
    parallel: { tp, pp: 1, dp: 1, ep },
    workload: WORKLOAD,
    hardware,
    ...(deployment !== undefined ? { deployment } : {}),
  };
  expect(validateInput(input)).toEqual([]);
  return simulate(input);
}

/** Every collective of a phase, in execution order: name, pattern, group, size and per-device bytes. */
function collectives(
  phase: PhaseResult,
): Pick<CollectiveOp, 'name' | 'collective' | 'group' | 'groupSize' | 'messageBytes'>[] {
  return phase.stages
    .flatMap((s) => s.layers.flatMap((l) => l.ops))
    .flatMap((o) => (o.op.kind === 'collective' ? [o.op] : []))
    .map(({ name, collective, group, groupSize, messageBytes }) => ({
      name,
      collective,
      group,
      groupSize,
      messageBytes,
    }));
}

/** Σ FLOP of the routed-expert GEMMs (`moe.gate_up`, `moe.down`) of a phase, on one device. Unit: FLOP. */
function expertFlops(phase: PhaseResult): number {
  return phase.stages
    .flatMap((s) => s.layers.flatMap((l) => l.ops))
    .filter((o) => o.op.name === 'moe.gate_up' || o.op.name === 'moe.down')
    .reduce((sum, o) => sum + o.flops, 0);
}

/** The first MoE block of the model. */
function firstMoe(model: ModelSpec): MoeBlock {
  for (const layer of model.layers) for (const b of layer.blocks) if (b.kind === 'moe') return b;
  throw new Error('no MoE block');
}

describe.each([2, 4])(
  'Qwen3.8-Flash-Next on %i × RTX 5090 (PCIe 5.0 P2P), prompt 4096, batch 1',
  (n) => {
    const tpOnly = run(FLASH_NEXT, n, 1);
    const withEp = run(FLASH_NEXT, n, n);

    it('EP = TP adds no devices: both use TP × PP', () => {
      expect(tpOnly.devicesUsed).toBe(n);
      expect(withEp.devicesUsed).toBe(n);
    });

    it.each(['prefill', 'decode'] as const)(
      '%s: the same collectives, the same bytes, and no all-to-all',
      (phase) => {
        const a = collectives(tpOnly[phase]);
        const b = collectives(withEp[phase]);
        expect(b).toEqual(a);
        expect(a.some((c) => c.name === 'moe.tp_all_reduce')).toBe(true);
        expect(b.some((c) => c.collective === 'all_to_all')).toBe(false);
        expect(withEp[phase].comm.latencySec).toBe(tpOnly[phase].comm.latencySec);
        expect(withEp[phase].comm.bandwidthSec).toBe(tpOnly[phase].comm.bandwidthSec);
      },
    );

    /**
     * Routing of the first MoE block for a phase of `withEp` at this ep, and its expert-TP split: mode A, with T the
     * tokens of the step read from the router GEMM (m = T).
     */
    const routing = (phase: 'prefill' | 'decode', ep: number) => {
      const block = firstMoe(FLASH_NEXT.model);
      const router = withEp[phase].stages[0].layers
        .flatMap((l) => l.ops)
        .find((o) => o.op.name === 'moe.router');
      if (router?.op.kind !== 'gemm') throw new Error('no router GEMM');
      const parallel = { tp: n, pp: 1, dp: 1, ep };
      const layout = moeLayout(block, parallel);
      return { ...moeRouting(block, layout, router.op.m, parallel), etp: layout.expertTp };
    };

    it("prefill expert FLOP rises by the busiest device's share A_max over the mean (≈ ρ = 1.2)", () => {
      // TP: every device runs the a·mE grid on fe/tp; EP: the a·mE grid of A_max = ⌈ρ·T·k/ep⌉ rows on full fe.
      // Ratio = (a·mE/etp)_EP / (a·mE/etp)_TP.
      const grid = (ep: number) => {
        const r = routing('prefill', ep);
        return (r.activeExperts * r.tokensPerExpert) / r.etp;
      };
      const ratio = expertFlops(withEp.prefill) / expertFlops(tpOnly.prefill);
      expect(ratio).toBeCloseTo(grid(n) / grid(1), 9);
      expect(ratio).toBeGreaterThan(1.15);
      expect(ratio).toBeLessThanOrEqual(DEFAULT_EP_IMBALANCE + 0.01);
    });

    it('prefill: each device reads and writes only its own expert rows, so the memory-bound experts get faster', () => {
      const activationBytes = (r: SimResult) =>
        r.prefill.stages
          .flatMap((s) => s.layers.flatMap((l) => l.ops))
          .filter((o) => ['moe.gate_up', 'moe.down', 'moe.combine'].includes(o.op.name))
          .reduce((sum, o) => sum + o.bytes.activationReadBytes + o.bytes.activationWriteBytes, 0);
      expect(activationBytes(withEp)).toBeLessThan(activationBytes(tpOnly));
      expect(withEp.prefill.tokensPerSec).toBeGreaterThan(tpOnly.prefill.tokensPerSec);
    });

    it('decode: expert weight bytes follow whole experts on the busiest device (U_EP) against 1/tp slices (U_TP/tp)', () => {
      const weightBytes = (r: SimResult) =>
        r.decode.stages
          .flatMap((s) => s.layers.flatMap((l) => l.ops))
          .filter((o) => o.op.name === 'moe.gate_up' || o.op.name === 'moe.down')
          .reduce((sum, o) => sum + o.bytes.weightBytes, 0);
      const read = (ep: number) => {
        const r = routing('decode', ep);
        return r.distinctExperts / r.etp;
      };
      expect(weightBytes(withEp) / weightBytes(tpOnly)).toBeCloseTo(read(n) / read(1), 9);
      // Decode moves by that rounding only: within 5 % (2 GPUs: ≈ 0 %; 4 GPUs, k 10: 3 experts for a mean of 2.5).
      const rel = withEp.decode.tokensPerSec / tpOnly.decode.tokensPerSec - 1;
      expect(Math.abs(rel)).toBeLessThan(0.05);
    });
  },
);

describe('Muse Spark (expert-output norm) on 4 × RTX 5090: EP removes the expert-TP collectives', () => {
  const tpOnly = run(MUSE_SPARK, 4, 1);
  const withEp = run(MUSE_SPARK, 4, 4);
  const ETP = ['moe.expert_out_reduce_scatter', 'moe.expert_norm_sumsq_all_reduce'];

  it.each(['prefill', 'decode'] as const)(
    '%s: EP 1 has the etp collectives, EP 4 none, so comm is lower',
    (phase) => {
      expect(collectives(tpOnly[phase]).filter((c) => ETP.includes(c.name)).length).toBeGreaterThan(
        0,
      );
      expect(collectives(withEp[phase]).filter((c) => ETP.includes(c.name))).toEqual([]);
      expect(collectives(withEp[phase]).some((c) => c.collective === 'all_to_all')).toBe(false);
      const total = (r: SimResult) => r[phase].comm.latencySec + r[phase].comm.bandwidthSec;
      expect(total(withEp)).toBeLessThan(total(tpOnly));
    },
  );
});

describe('speculative decoding and offload with EP = 2', () => {
  it('MTP on Qwen3.8-Flash-Next, TP 2 EP 2: the draft MoE layers lower and speed up decode', () => {
    expect(FLASH_NEXT.model.draft?.kind).toBe('mtp');
    const r = run(FLASH_NEXT, 2, 2, {
      speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.8 } },
    });
    expect(r.decode.speculative?.speedup).toBeGreaterThan(1);
    const draftOps = r.decode.stages
      .flatMap((s) => s.layers)
      .filter((l) => l.name.startsWith('draft.'))
      .flatMap((l) => l.ops);
    expect(draftOps.some((o) => o.op.name === 'draft.moe.gate_up')).toBe(true);
  });

  it.each([2, 4])(
    'DSpark on DeepSeek-V4.1-Flash (its own 128-expert drafter MoE), TP = EP = %i',
    (n) => {
      const draft = V41_FLASH.model.draft;
      if (draft?.kind !== 'block_drafter') throw new Error('no DSpark drafter');
      const draftExperts = (draft.moeLayers ?? [])
        .flatMap((l) => l.blocks)
        .flatMap((b) => (b.kind === 'moe' ? [b.experts] : []));
      expect(draftExperts.length).toBeGreaterThan(0);
      for (const e of draftExperts) expect(e % n).toBe(0);
      const r = run(V41_FLASH, n, n, {
        speculative: {
          method: 'dspark',
          drafter: { kind: 'hypothetical', spec: draft },
          draftLength: 3,
          acceptance: { kind: 'iid', alpha: 0.8 },
        },
      });
      expect(r.decode.speculative?.expectedTokensPerStep).toBeGreaterThan(1);
    },
  );

  it('a drafter MoE whose experts EP does not divide is a validation error, not a throw while lowering', () => {
    // Synthetic: the MTP layer's experts set to 510 (divisible by 2, not by 4).
    const draft = structuredClone(FLASH_NEXT.model.draft);
    if (draft?.kind !== 'mtp') throw new Error('no MTP head');
    for (const layer of draft.layers)
      for (const b of layer.blocks) if (b.kind === 'moe') b.experts = 510;
    const input = (ep: number): SimInput => ({
      model: { ...FLASH_NEXT.model, draft },
      quant: FLASH_NEXT.quant,
      parallel: { tp: 4, pp: 1, dp: 1, ep },
      workload: WORKLOAD,
      hardware: box(4),
      deployment: {
        speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.8 } },
      },
    });
    const errors = validateInput(input(4));
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(
      /^deployment\.speculative drafter mtp\.0: moe: experts \(510\) must be divisible by ep \(4\)$/,
    );
    expect(validateInput(input(2))).toEqual([]);
  });

  it('routed experts in host RAM on Qwen3.8-Flash-Next, TP 2 EP 2: simulates, with the EP approximation note', () => {
    const hardware = { ...box(2), host: { ram: DDR5_6000_2CH } };
    const r = run(
      FLASH_NEXT,
      2,
      2,
      {
        offload: {
          routedExperts: { tier: 'host_ram', residentFraction: 0.5, hitRatio: { kind: 'uniform' } },
        },
      },
      hardware,
    );
    expect(r.decode.transfer?.bytes).toBeGreaterThan(0);
    expect(r.warnings.some((w) => /^offload: with EP > 1/.test(w))).toBe(true);
  });
});
