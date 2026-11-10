/**
 * Decode context parallelism (DCP) end to end through `simulate`: identity at d = 1 on every block family, the
 * Kimi K2 TP8 DCP8 case at 128k (more sequences fit, a batch that fails at d = 1 fits, decode is faster, the DCP
 * collectives exist only in decode and run on the right links), and the validation rules.
 * See docs/simulation-model.md §5 "Decode context parallelism (DCP)".
 */
import { describe, expect, it } from 'vitest';
import { B200, composeTopology } from '../hw/presets';
import { decoderLayer, decoderModel, repeatLayers } from '../ir/builders';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import type { CollectiveOp, Op } from '../ir/ops';
import { DCP_BLOCK_DRAFTER_ERROR } from '../lower/dcp';
import {
  REF_DENSE_8B,
  REF_MOE_30B_A3B,
  TINY_CSA,
  TINY_HYBRID,
  TINY_INKLING,
  TINY_MLA_SPARSE,
} from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import { collectiveTime } from './collectives';
import { checkPlacement, collectivePath, groupSpan, hasInterNodeTraffic } from './placement';
import { DEFAULT_SIM_OPTIONS, simulate } from './simulate';
import type { DeploymentConfig, PhaseResult, SimInput, SimResult } from './types';
import { dcpErrors, validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};
/** FP8 weights and KV cache, bf16 activations: lets Kimi K2's ~1.03 T parameters fit one 8×B200 node. */
const FP8: QuantConfig = {
  weights: { dtype: 'fp8_e4m3' },
  activations: 'bf16',
  kvCache: { dtype: 'fp8_e4m3' },
};

/**
 * Kimi K2 shape, built by hand (no loader): h 7168, 61 layers (layer 0 dense, f 18432; 60 MoE: 384 routed experts,
 * top-8, fe 2048, 1 shared expert of 2048), MLA 64 heads, rq 1536, rkv 512, dn 128, dr 64, dv 128, untied vocab
 * 163840. Source: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json (dims checked against
 * the file).
 */
function kimiK2(): ModelSpec {
  return decoderModel({
    name: 'Kimi K2 shape (hand-built test model)',
    hidden: 7168,
    vocab: 163840,
    tieEmbeddings: false,
    source:
      'https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json (dims copied by hand)',
    layers: repeatLayers(61, (i) =>
      decoderLayer({
        name: `layer.${i}`,
        hidden: 7168,
        mixer: {
          kind: 'mla',
          heads: 64,
          qLoraRank: 1536,
          kvLoraRank: 512,
          qkNopeDim: 128,
          qkRopeDim: 64,
          vHeadDim: 128,
        },
        ffn:
          i < 1
            ? { kind: 'dense', intermediate: 18432 }
            : {
                kind: 'moe',
                experts: 384,
                topK: 8,
                expertIntermediate: 2048,
                sharedExperts: 1,
                sharedIntermediate: 2048,
              },
      }),
    ),
  });
}
const K2 = kimiK2();

/** A small dense MLA model (made-up dims): h 512, 2 layers, MLA 8 heads (rq 128, rkv 64, dn 32, dr 16, dv 32). */
const TINY_MLA_DENSE: ModelSpec = decoderModel({
  name: 'TINY_MLA_DENSE',
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

/** `nodes` machines of 8 × B200 (NVLink 5 inside, the catalog's default network between them). */
function b200(nodes: number): SimInput['hardware'] {
  return {
    device: B200,
    topology: composeTopology({ deviceId: 'b200', devicesPerNode: 8, nodes }),
  };
}

interface InputOptions {
  quant?: QuantConfig;
  batch?: number;
  promptLen?: number;
  decodeContextLen?: number;
  deployment?: DeploymentConfig;
}

function input(
  model: ModelSpec,
  parallel: Partial<ParallelConfig>,
  hardware: SimInput['hardware'],
  o: InputOptions = {},
): SimInput {
  return {
    model,
    quant: o.quant ?? BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    workload: {
      batch: o.batch ?? 8,
      promptLen: o.promptLen ?? 1024,
      decodeContextLen: o.decodeContextLen ?? 2048,
    },
    hardware,
    ...(o.deployment === undefined ? {} : { deployment: o.deployment }),
  };
}

function run(i: SimInput): SimResult {
  expect(validateInput(i)).toEqual([]);
  return simulate(i);
}

const allOps = (p: PhaseResult): Op[] =>
  p.stages.flatMap((s) => s.layers.flatMap((l) => l.ops.map((r) => r.op)));
const dcpOps = (p: PhaseResult) =>
  p.stages
    .flatMap((s) => s.layers.flatMap((l) => l.ops))
    .filter((r) => r.op.kind === 'collective' && r.op.group === 'dcp');

const MTP: DeploymentConfig = {
  speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.8 } },
};

describe('identity: dcp absent = dcp 1, bit for bit', () => {
  const cases: [string, ModelSpec, number[], DeploymentConfig | undefined][] = [
    ['REF_DENSE_8B (GQA)', REF_DENSE_8B, [1, 2, 4], undefined],
    ['REF_MOE_30B_A3B (GQA + MoE)', REF_MOE_30B_A3B, [1, 2, 4], undefined],
    ['TINY_MLA_DENSE (MLA)', TINY_MLA_DENSE, [1, 2, 4], undefined],
    ['TINY_HYBRID (GatedDeltaNet + GQA + MoE)', TINY_HYBRID, [1, 2, 4], undefined],
    ['REF_DENSE_8B_MTP (spec-dec MTP)', REF_DENSE_8B_MTP, [1, 2, 4], MTP],
  ];
  it.each(cases)('%s at tp 1/2/4', (_name, model, tps, deployment) => {
    for (const tp of tps) {
      const ep = model === REF_MOE_30B_A3B || model === TINY_HYBRID ? tp : 1;
      const absent = run(input(model, { tp, ep, dp: 8 / tp }, b200(1), { deployment }));
      const one = run(input(model, { tp, ep, dp: 8 / tp, dcp: 1 }, b200(1), { deployment }));
      expect(one).toStrictEqual(absent);
      for (const phase of [one.prefill, one.decode]) {
        const ops = allOps(phase);
        expect(ops.some((o) => 'kvTokenShards' in o)).toBe(false);
        expect(ops.some((o) => o.kind === 'collective' && o.group === 'dcp')).toBe(false);
      }
    }
  });
});

describe('Kimi K2 on one 8×B200 node, TP8 (EP8) vs TP8 DCP8 at 128k', () => {
  const hw = b200(1);
  const long = { quant: FP8, promptLen: 4096, decodeContextLen: 131072 };
  const par = { tp: 8, ep: 8 };
  // Batch 1 runs are only to read the capacity; the activation peak barely depends on the batch here.
  const probe1 = run(input(K2, par, hw, { ...long, batch: 1 }));
  const probe8 = run(input(K2, { ...par, dcp: 8 }, hw, { ...long, batch: 1 }));
  const m1 = probe1.memory.maxSequencesPerReplica;
  const m8 = probe8.memory.maxSequencesPerReplica;
  const ctxTokens = 131072 + 1; // max(promptLen, decodeContextLen + 1)

  it('per-device latent state is ceil(ctx/8)/ctx of TP8 alone (TP8 replicates the latent 8×)', () => {
    // Formula: S_dev = L·N_rank·(rkv + dr)·bpe(kv), N_rank = ceil(ctx/d)
    expect(probe1.memory.perStage[0].sequenceStateBytes).toBe(61 * ctxTokens * 576 * 1);
    expect(probe8.memory.perStage[0].sequenceStateBytes).toBe(
      61 * Math.ceil(ctxTokens / 8) * 576 * 1,
    );
  });
  it('about 8× more sequences fit (at least one fits at d = 1)', () => {
    expect(m1).toBeGreaterThanOrEqual(1);
    // The ceil makes a rank's share a hair over 1/8, so floor() may lose one sequence against exactly 8·m1.
    expect(m8).toBeGreaterThanOrEqual(8 * m1 - 1);
    expect(m8).toBeGreaterThan(7 * m1);
  });
  it('a batch that does not fit at d = 1 fits at d = 8', () => {
    const batch = m1 + 1;
    expect(simulate(input(K2, par, hw, { ...long, batch })).memory.fits).toBe(false);
    expect(run(input(K2, { ...par, dcp: 8 }, hw, { ...long, batch })).memory.fits).toBe(true);
  });

  const batch = Math.max(m1, 1);
  const d1 = run(input(K2, par, hw, { ...long, batch }));
  const d8 = run(input(K2, { ...par, dcp: 8 }, hw, { ...long, batch }));
  it('decode tok/s @ 128k is higher with DCP8 at the same batch', () => {
    expect(d8.decode.tokensPerSec).toBeGreaterThan(d1.decode.tokensPerSec);
  });
  it('the DCP collectives appear in every decode layer and never in prefill; prefill is unchanged', () => {
    for (let i = 0; i < 61; i++) {
      const names = d8.decode.stages[0].layers
        .find((l) => l.name === `layer.${i}`)
        ?.ops.map((r) => r.op.name);
      expect(names).toContain('attn.dcp_q_all_gather');
      expect(names).toContain('attn.dcp_out_reduce_scatter');
    }
    expect(dcpOps(d8.prefill)).toEqual([]);
    expect(d8.prefill).toStrictEqual(d1.prefill);
  });
  it('the DCP group of 8 runs on the intra-node link, costed exactly by collectiveTime', () => {
    const path = collectivePath(groupSpan(8, hw.topology), hw.topology);
    const ops = dcpOps(d8.decode);
    expect(ops).toHaveLength(2 * 61);
    for (const res of ops) {
      const op = res.op as CollectiveOp;
      const t = collectiveTime(
        op.collective,
        op.messageBytes,
        op.groupSize,
        path,
        DEFAULT_SIM_OPTIONS.collectiveAlgorithm,
      );
      expect(res.linkLevel).toBe('intra');
      expect(res.commLatencySec).toBe(t.latencySec);
      expect(res.commBandwidthSec).toBe(t.bandwidthSec);
    }
  });
  it('at a short context (1k) the DCP collectives still pay their α terms', () => {
    const short = run(
      input(K2, { ...par, dcp: 8 }, hw, {
        quant: FP8,
        batch: 8,
        promptLen: 1024,
        decodeContextLen: 1024,
      }),
    );
    for (const res of dcpOps(short.decode)) expect(res.commLatencySec).toBeGreaterThan(0);
  });
});

describe('Kimi K2 on two 8×B200 nodes, TP16 DCP16: the DCP group goes hierarchical', () => {
  const hw = b200(2);
  const r = run(
    input(K2, { tp: 16, ep: 16, dcp: 16 }, hw, {
      quant: FP8,
      batch: 8,
      promptLen: 4096,
      decodeContextLen: 131072,
    }),
  );
  it('every DCP collective runs on the hierarchical path of 2 nodes × 8', () => {
    const path = collectivePath(groupSpan(16, hw.topology), hw.topology);
    expect(path).toMatchObject({ kind: 'hierarchical', devicesPerNode: 8, nodes: 2 });
    const ops = dcpOps(r.decode);
    expect(ops.length).toBe(2 * 61);
    for (const res of ops) {
      const op = res.op as CollectiveOp;
      const t = collectiveTime(
        op.collective,
        op.messageBytes,
        op.groupSize,
        path,
        DEFAULT_SIM_OPTIONS.collectiveAlgorithm,
      );
      expect(res.linkLevel).toBe('hierarchical');
      expect(res.commLatencySec).toBeCloseTo(t.latencySec, 15);
      expect(res.commBandwidthSec).toBeCloseTo(t.bandwidthSec, 15);
    }
  });
});

describe('MTP with DCP', () => {
  it('REF_DENSE_8B_MTP at tp 16 (r = 2) dcp 2: valid, the draft layer shards its KV too', () => {
    const r = run(input(REF_DENSE_8B_MTP, { tp: 16, dcp: 2 }, b200(2), { deployment: MTP }));
    const names = allOps(r.decode).map((o) => o.name);
    expect(names).toContain('attn.dcp_q_all_gather');
    expect(names).toContain('draft.attn.dcp_q_all_gather');
    const base = run(input(REF_DENSE_8B_MTP, { tp: 16 }, b200(2), { deployment: MTP }));
    expect(r.memory.perStage[0].draftBytes).toBeLessThan(base.memory.perStage[0].draftBytes);
  });
});

describe('validation', () => {
  const hw = b200(1);
  it('dcp must be an integer ≥ 1', () => {
    expect(validateInput(input(K2, { tp: 8, dcp: 0 }, hw))).toContain(
      'parallel.dcp must be an integer ≥ 1 (got 0)',
    );
    expect(validateInput(input(K2, { tp: 8, dcp: 2.5 }, hw))).toContain(
      'parallel.dcp must be an integer ≥ 1 (got 2.5)',
    );
  });
  it('dcp must divide tp (also tp = 1, e.g. the DP attention + EP layout)', () => {
    expect(validateInput(input(TINY_MLA_DENSE, { tp: 4, dp: 2, dcp: 8 }, hw))).toContain(
      'parallel.dcp (8) must divide tp (4)',
    );
    expect(validateInput(input(TINY_MLA_DENSE, { tp: 1, dp: 8, dcp: 2 }, hw))).toContain(
      'parallel.dcp (2) must divide tp (1)',
    );
    expect(
      validateInput(
        input(REF_MOE_30B_A3B, { tp: 1, dp: 8, ep: 8, denseTpGroup: 'ep', dcp: 2 }, hw),
      ),
    ).toContain('parallel.dcp (2) must divide tp (1)');
  });
  it('GQA: d must divide tp / kvHeads (layer-prefixed, from lowering)', () => {
    const errors = validateInput(input(REF_MOE_30B_A3B, { tp: 8, ep: 8, dcp: 4 }, hw));
    expect(errors).toContain(
      'layer.0: attn dcp (4) must divide tp / kvHeads (2): KV heads are replicated on 2 ranks and DCP shards that copy',
    );
    expect(validateInput(input(REF_MOE_30B_A3B, { tp: 8, ep: 8, dcp: 2 }, hw))).toEqual([]);
    expect(validateInput(input(REF_DENSE_8B, { tp: 8, dcp: 2 }, hw))).toContain(
      'layer.0: attn dcp (2) must divide tp / kvHeads (1): KV heads are replicated on 1 ranks and DCP shards that copy',
    );
  });
  it('unsupported blocks: sparse MLA, CSA, relative-position bias / K/V convs', () => {
    expect(validateInput(input(TINY_MLA_SPARSE, { tp: 2, dp: 4, dcp: 2 }, hw))).toContain(
      'layer.0: attn dcp (2) > 1 is not supported with sparse (top-k) attention',
    );
    expect(validateInput(input(TINY_CSA, { tp: 2, dp: 4, dcp: 2 }, hw))).toContain(
      'layer.0: attn dcp (2) > 1 is not supported with compressed sparse attention (CSA)',
    );
    expect(validateInput(input(TINY_INKLING, { tp: 2, dp: 4, dcp: 2 }, hw))).toContain(
      'layer.1: attn dcp (2) > 1 is not supported with a relative-position bias or K/V short convs',
    );
  });
  it('a model with nothing to shard', () => {
    const windowOnly: ModelSpec = decoderModel({
      name: 'window-only',
      hidden: 256,
      vocab: 1000,
      tieEmbeddings: false,
      source: 'made-up test fixture',
      layers: repeatLayers(2, (i) =>
        decoderLayer({
          name: `layer.${i}`,
          hidden: 256,
          mixer: { kind: 'gqa', heads: 4, kvHeads: 1, headDim: 64, slidingWindow: 128 },
          ffn: { kind: 'dense', intermediate: 512 },
        }),
      ),
    });
    expect(validateInput(input(windowOnly, { tp: 2, dp: 4, dcp: 2 }, hw))).toEqual([
      'dcp (2) > 1 but the model has no full-attention (MLA or GQA) block to shard',
    ]);
  });
  it('a block drafter is rejected; MTP is not', () => {
    const block: DeploymentConfig = {
      speculative: { method: 'dflash2', draftLength: 3, acceptance: { kind: 'iid', alpha: 0.7 } },
    };
    expect(
      dcpErrors({
        model: TINY_MLA_DENSE,
        parallel: { tp: 2, pp: 1, dp: 1, ep: 1, dcp: 2 },
        deployment: block,
      }),
    ).toEqual([DCP_BLOCK_DRAFTER_ERROR]);
    expect(DCP_BLOCK_DRAFTER_ERROR).toBe(
      'parallel.dcp > 1 with a block drafter (dflash2, dspark) is not supported yet',
    );
    expect(
      dcpErrors({
        model: TINY_MLA_DENSE,
        parallel: { tp: 2, pp: 1, dp: 1, ep: 1, dcp: 2 },
        deployment: MTP,
      }),
    ).toEqual([]);
  });
  it('dcpErrors ignores absent, 1 and non-integer degrees (validateInput reports the latter)', () => {
    const p = { tp: 3, pp: 1, dp: 1, ep: 1 };
    for (const dcp of [undefined, 1, 0, 2.5]) {
      expect(
        dcpErrors({
          model: TINY_HYBRID,
          parallel: { ...p, ...(dcp === undefined ? {} : { dcp }) },
        }),
      ).toEqual([]);
    }
  });
  it('placement: a DCP group that does not tile the node is reported with the dcp: prefix', () => {
    const hw6 = {
      device: B200,
      topology: composeTopology({ deviceId: 'b200', devicesPerNode: 8, nodes: 1 }),
    };
    const errors = validateInput(input(TINY_MLA_DENSE, { tp: 8, dcp: 8 }, hw6));
    expect(errors).toEqual([]);
    const bad = validateInput(input(K2, { tp: 6, dcp: 3 }, hw6));
    expect(bad.some((e) => e.startsWith('dcp: '))).toBe(true);
  });
  it('placement: d | tp is not enough (dpn 4 × 3 nodes, tp 12 hierarchical, d 6 straddles nodes)', () => {
    const topo = composeTopology({ deviceId: 'b200', devicesPerNode: 4, nodes: 3 });
    const par = { tp: 12, pp: 1, dp: 1, ep: 1 };
    expect(checkPlacement(par, topo).errors).toEqual([]);
    expect(checkPlacement({ ...par, dcp: 6 }, topo).errors).toEqual([
      'dcp: a group of 6 devices does not tile nodes of 4 devices (needs size | 4 or 4 | size)',
    ]);
    // d 4 (one node) and d 12 (all three, hierarchical) place.
    expect(checkPlacement({ ...par, dcp: 4 }, topo).errors).toEqual([]);
    expect(checkPlacement({ ...par, dcp: 12 }, topo).errors).toEqual([]);
    // A node-crossing DCP group lies in a node-crossing TP group: inter-node traffic is already reported via TP.
    expect(hasInterNodeTraffic({ ...par, dcp: 12 }, topo)).toBe(true);
    expect(hasInterNodeTraffic({ ...par, dcp: 12 }, topo)).toBe(hasInterNodeTraffic(par, topo));
  });
});
