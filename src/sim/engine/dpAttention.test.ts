/**
 * DP attention + EP (the vLLM DEP / SGLang `--enable-dp-attention` layout): tp = 1, dp = ep = N, end to end
 * through `simulate`, and the opt-in `ParallelConfig.denseTpGroup: 'ep'` that shards the dense FFN layers and the
 * LM head over the whole EP group. See docs/simulation-model.md §5 "DP attention + EP".
 *
 * What is pinned here:
 * - identity: `denseTpGroup` absent, `'tp'`, or `'ep'` at ep = 1 give the same SimResult (the old numbers);
 * - MLA latent KV in DEP is ONE copy per DP rank: each device holds only its own B/N sequences' latent
 *   (`(B/32)·ctx·61·576·bpe` for DeepSeek-V3 at DEP32), and attention TP 8 (tp8·dp4·ep32) holds 8× that;
 * - `'ep'`: the dense layers' gather → sharded GEMMs → scatter, the head's gather → `ceil(V/G)` → all-to-all, the
 *   weights they save, the MoE layers unchanged op for op, the tied-head dedupe, the validation rules;
 * - the group-`'ep'` all_gather / reduce_scatter on a 2-node (hierarchical) group cost exactly `collectiveTime`.
 */
import { describe, expect, it } from 'vitest';
import { B200, composeTopology } from '../hw/presets';
import { decoderLayer, decoderModel, repeatLayers } from '../ir/builders';
import type { ModelSpec, ParallelConfig, QuantConfig } from '../ir/model';
import type { CollectiveOp, Op } from '../ir/ops';
import { REF_DENSE_8B, REF_MOE_30B_A3B } from '../models/reference';
import { collectiveTime } from './collectives';
import { collectivePath, groupSpan } from './placement';
import { DEFAULT_SIM_OPTIONS, simulate } from './simulate';
import type { DeploymentConfig, PhaseResult, SimInput, SimResult } from './types';
import { DENSE_TP_EP_NEEDS_TP1_ERROR, DENSE_TP_EP_SPEC_DEC_ERROR, validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

/**
 * DeepSeek-V3 shape, built by hand (no loader): h 7168, 61 layers (3 dense, f 18432; 58 MoE: 256 routed experts,
 * top-8, 1 shared expert, fe 2048), MLA 128 heads, rq 1536, rkv 512, dn 128, dr 64, dv 128, untied vocab 129280.
 * Source: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json (dims copied by hand; MTP head left
 * out). `denseIntermediate` overrides f of the 3 dense layers (to test the f % G rule).
 */
function deepseekV3(denseIntermediate = 18432): ModelSpec {
  return decoderModel({
    name: 'DeepSeek-V3 shape (hand-built test model)',
    hidden: 7168,
    vocab: 129280,
    tieEmbeddings: false,
    source:
      'https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json (dims copied by hand)',
    layers: repeatLayers(61, (i) =>
      decoderLayer({
        name: `layer.${i}`,
        hidden: 7168,
        mixer: {
          kind: 'mla',
          heads: 128,
          qLoraRank: 1536,
          kvLoraRank: 512,
          qkNopeDim: 128,
          qkRopeDim: 64,
          vHeadDim: 128,
        },
        ffn:
          i < 3
            ? { kind: 'dense', intermediate: denseIntermediate }
            : {
                kind: 'moe',
                experts: 256,
                topK: 8,
                expertIntermediate: 2048,
                sharedExperts: 1,
                sharedIntermediate: 2048,
              },
      }),
    ),
  });
}
const DSV3 = deepseekV3();

/** `nodes` machines of 8 × B200 (NVLink 5 inside, the catalog's default network between them). */
function b200(nodes: number): SimInput['hardware'] {
  return {
    device: B200,
    topology: composeTopology({ deviceId: 'b200', devicesPerNode: 8, nodes }),
  };
}

function input(
  model: ModelSpec,
  parallel: Partial<ParallelConfig>,
  hardware: SimInput['hardware'],
  batch = 64,
  deployment?: DeploymentConfig,
): SimInput {
  return {
    model,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1, ...parallel },
    workload: { batch, promptLen: 4096, decodeContextLen: 4096 },
    hardware,
    ...(deployment === undefined ? {} : { deployment }),
  };
}

function run(i: SimInput): SimResult {
  expect(validateInput(i)).toEqual([]);
  return simulate(i);
}

/** Ops of the named layer of the (single) stage. */
function layerOps(p: PhaseResult, name: string): Op[] {
  const layer = p.stages.flatMap((s) => s.layers).find((l) => l.name === name);
  if (layer === undefined) throw new Error(`no layer ${name}`);
  return layer.ops.map((r) => r.op);
}

function opNamed(ops: Op[], name: string): Op {
  const op = ops.find((o) => o.name === name);
  if (op === undefined) throw new Error(`no op ${name}`);
  return op;
}

describe('identity: denseTpGroup absent = before', () => {
  it.each([
    ['tp1 dp8 ep8', { tp: 1, dp: 8, ep: 8 }],
    ['tp2 dp4 ep8', { tp: 2, dp: 4, ep: 8 }],
    ['tp2 dp2 ep2 (pinned mode-B shape)', { tp: 2, dp: 2, ep: 2 }],
  ])('MoE mode B %s: absent and "tp" give equal SimResults', (_label, parallel) => {
    const absent = run(input(REF_MOE_30B_A3B, parallel, b200(1)));
    const tp = run(input(REF_MOE_30B_A3B, { ...parallel, denseTpGroup: 'tp' }, b200(1)));
    expect(tp).toEqual(absent);
  });
  it('"ep" at ep = 1 equals absent (dense model and MoE model, tp 1 dp 4)', () => {
    for (const model of [REF_DENSE_8B, REF_MOE_30B_A3B]) {
      const absent = run(input(model, { dp: 4 }, b200(1)));
      const ep1 = run(input(model, { dp: 4, denseTpGroup: 'ep' }, b200(1)));
      expect(ep1).toEqual(absent);
    }
  });
});

describe('DeepSeek-V3 DEP32: MLA latent KV is one copy per DP rank', () => {
  const B = 64;
  const dep = run(input(DSV3, { tp: 1, dp: 32, ep: 32 }, b200(4), B));
  const tp8 = run(input(DSV3, { tp: 8, dp: 4, ep: 32 }, b200(4), B));
  // The cache holds max(prompt, decodeContextLen + 1) tokens after the decode step (docs §8 "Memory fit").
  const ctxTokens = 4096 + 1;
  const latentElems = 512 + 64; // rkv + dr
  const bpe = 2;

  it('tp1·dp32·ep32: sequence state = (B/32)·ctx·61·576·bpe per device', () => {
    // Formula: S_dev = B_r·ctx·L·(rkv + dr)·bpe(kv), B_r = B/dp = 2
    expect(dep.memory.perStage[0].sequenceStateBytes).toBe(
      (B / 32) * ctxTokens * 61 * latentElems * bpe,
    );
  });
  it('tp8·dp4·ep32 holds exactly 8× that (the latent replicated on every attention-TP rank)', () => {
    expect(tp8.memory.perStage[0].sequenceStateBytes).toBe(
      8 * dep.memory.perStage[0].sequenceStateBytes,
    );
  });
  it('decode attention kvRead per device follows the same 8× ratio', () => {
    const kvRead = (r: SimResult) =>
      r.decode.stages[0].layers
        .flatMap((l) => l.ops)
        .filter((o) => o.op.kind === 'attention')
        .reduce((sum, o) => sum + o.bytes.kvReadBytes, 0);
    // Formula: kvRead = B_r·kvLen·L·576·bpe   (absorbed MLA reads only the latent; kvLen = N + 1 incl. the new token)
    expect(kvRead(dep)).toBe((B / 32) * ctxTokens * 61 * latentElems * bpe);
    expect(kvRead(tp8)).toBe(8 * kvRead(dep));
  });
});

describe("DeepSeek-V3 DEP32 with denseTpGroup 'ep' (dense FFN + LM head TP over the EP group)", () => {
  const B = 64;
  const G = 32;
  const T = B / G; // decode: B_r sequences × 1 token per device
  const h = 7168;
  const f = 18432;
  const V = 129280;
  const dp = run(input(DSV3, { tp: 1, dp: G, ep: G }, b200(4), B));
  const ep = run(input(DSV3, { tp: 1, dp: G, ep: G, denseTpGroup: 'ep' }, b200(4), B));

  it('dense layers 0–2: all_gather(ep) S = G·T·h·bpe → gate_up m = G·T, n = 2f/G → down k = f/G → reduce_scatter(ep)', () => {
    for (const layer of ['layer.0', 'layer.1', 'layer.2']) {
      const ops = layerOps(ep.decode, layer);
      expect(opNamed(ops, 'mlp.dense_all_gather')).toMatchObject({
        kind: 'collective',
        collective: 'all_gather',
        group: 'ep',
        groupSize: G,
        messageBytes: G * T * h * 2,
      });
      expect(opNamed(ops, 'mlp.gate_up')).toMatchObject({ m: G * T, n: (2 * f) / G, k: h });
      expect(opNamed(ops, 'mlp.down')).toMatchObject({ m: G * T, n: h, k: f / G });
      expect(opNamed(ops, 'mlp.dense_reduce_scatter')).toMatchObject({
        collective: 'reduce_scatter',
        group: 'ep',
        groupSize: G,
        messageBytes: G * T * h * 2,
      });
      // DP (absent): whole MLP on own T rows, no collective.
      const dpOps = layerOps(dp.decode, layer);
      expect(opNamed(dpOps, 'mlp.gate_up')).toMatchObject({ m: T, n: 2 * f, k: h });
      expect(dpOps.some((o) => o.name.startsWith('mlp.') && o.kind === 'collective')).toBe(false);
    }
  });
  it('MoE layers 3–60 are unchanged op for op', () => {
    for (let i = 3; i < 61; i++)
      expect(layerOps(ep.decode, `layer.${i}`)).toEqual(layerOps(dp.decode, `layer.${i}`));
    for (let i = 3; i < 61; i++)
      expect(layerOps(ep.prefill, `layer.${i}`)).toEqual(layerOps(dp.prefill, `layer.${i}`));
  });
  it('head: final_norm(T) → all_gather(ep) S = G·T·h·bpe → lm_head m = G·T, n = ceil(V/G) → all_to_all(ep) S = G·T·8', () => {
    const ops = layerOps(ep.decode, 'head');
    expect(ops.map((o) => o.name)).toEqual([
      'final_norm',
      'head.dense_all_gather',
      'lm_head',
      'head.sample_all_to_all',
    ]);
    expect(opNamed(ops, 'final_norm')).toMatchObject({ elems: T * h });
    expect(opNamed(ops, 'head.dense_all_gather')).toMatchObject({
      collective: 'all_gather',
      group: 'ep',
      messageBytes: G * T * h * 2,
    });
    expect(opNamed(ops, 'lm_head')).toMatchObject({ m: G * T, n: Math.ceil(V / G), k: h });
    expect(opNamed(ops, 'head.sample_all_to_all')).toMatchObject({
      collective: 'all_to_all',
      group: 'ep',
      groupSize: G,
      messageBytes: G * T * 8,
    });
    expect(layerOps(dp.decode, 'head').map((o) => o.name)).toEqual(['final_norm', 'lm_head']);
  });
  it('resident weights shrink by the dense (1 − 1/G) share and the head (V − ceil(V/G)) rows', () => {
    // Formula: ΔW = 3 layers·3·h·f·bpe·(1 − 1/G) + (V − ceil(V/G))·h·bpe
    const saved = 3 * 3 * h * f * 2 * (1 - 1 / G) + (V - Math.ceil(V / G)) * h * 2;
    expect(dp.memory.perStage[0].weightBytes - ep.memory.perStage[0].weightBytes).toBe(saved);
  });
  it('the latent KV is unchanged (attention stays data-parallel)', () => {
    expect(ep.memory.perStage[0].sequenceStateBytes).toBe(dp.memory.perStage[0].sequenceStateBytes);
  });
  it('prefill: the gather widens the dense GEMM to G·T rows (T = B_r·prompt) and the head to G·B_r rows', () => {
    const Tp = (B / G) * 4096;
    expect(opNamed(layerOps(ep.prefill, 'layer.0'), 'mlp.gate_up')).toMatchObject({ m: G * Tp });
    expect(opNamed(layerOps(ep.prefill, 'head'), 'lm_head')).toMatchObject({ m: G * (B / G) });
  });
});

describe("tied LM head with denseTpGroup 'ep'", () => {
  const tied: ModelSpec = { ...REF_MOE_30B_A3B, tieEmbeddings: true };
  const untied = REF_MOE_30B_A3B;
  const par = { tp: 1, dp: 8, ep: 8 };
  const weight = (m: ModelSpec, p: Partial<ParallelConfig>) =>
    run(input(m, p, b200(1))).memory.perStage[0].weightBytes;
  it('tied: the head shard lives inside the whole (tp 1) embedding table, so weights equal DP', () => {
    expect(weight(tied, { ...par, denseTpGroup: 'ep' })).toBe(weight(tied, par));
  });
  it('untied: the head drops to ceil(V/8) rows', () => {
    const V = 151936;
    const h = 2048;
    expect(weight(untied, par) - weight(untied, { ...par, denseTpGroup: 'ep' })).toBe(
      (V - Math.ceil(V / 8)) * h * 2,
    );
  });
});

describe("validation of denseTpGroup 'ep'", () => {
  it('rejects tp > 1', () => {
    expect(
      validateInput(input(DSV3, { tp: 8, dp: 4, ep: 32, denseTpGroup: 'ep' }, b200(4))),
    ).toContain(DENSE_TP_EP_NEEDS_TP1_ERROR);
  });
  it('rejects speculative decoding over a real group (ep > 1), not at ep 1', () => {
    const spec: DeploymentConfig = {
      speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.8 } },
    };
    expect(
      validateInput(input(DSV3, { dp: 32, ep: 32, denseTpGroup: 'ep' }, b200(4), 64, spec)),
    ).toContain(DENSE_TP_EP_SPEC_DEC_ERROR);
    expect(
      validateInput(input(DSV3, { dp: 32, ep: 1, denseTpGroup: 'ep' }, b200(4), 64, spec)),
    ).not.toContain(DENSE_TP_EP_SPEC_DEC_ERROR);
  });
  it('rejects a dense width G does not divide, naming the layer and the group', () => {
    const errors = validateInput(
      input(deepseekV3(18000), { dp: 32, ep: 32, denseTpGroup: 'ep' }, b200(4)),
    );
    expect(errors).toContain(
      'layer.0: mlp intermediate (18000) must be divisible by the dense TP group (32)',
    );
    // The same width is fine with dense DP (absent).
    expect(validateInput(input(deepseekV3(18000), { dp: 32, ep: 32 }, b200(4)))).toEqual([]);
  });
  it('rejects an unknown value', () => {
    const bad = input(DSV3, { dp: 32, ep: 32, denseTpGroup: 'dp' as unknown as 'ep' }, b200(4));
    expect(validateInput(bad)).toContain("parallel.denseTpGroup must be 'tp' or 'ep' (got dp)");
  });
});

describe("two nodes: group-'ep' all_gather and reduce_scatter cost on the hierarchical EP path", () => {
  const hw = b200(2);
  const r = run(input(DSV3, { dp: 16, ep: 16, denseTpGroup: 'ep' }, hw, 32));
  const path = collectivePath(groupSpan(16, hw.topology), hw.topology);
  it('the EP group of 16 spans 2 nodes × 8 (hierarchical path)', () => {
    expect(path).toMatchObject({ kind: 'hierarchical', devicesPerNode: 8, nodes: 2 });
  });
  it.each([
    'mlp.dense_all_gather',
    'mlp.dense_reduce_scatter',
    'head.dense_all_gather',
    'head.sample_all_to_all',
  ])('%s: α and β equal collectiveTime on that path', (name) => {
    for (const phase of [r.prefill, r.decode]) {
      const res = phase.stages[0].layers.flatMap((l) => l.ops).find((o) => o.op.name === name);
      if (res === undefined) throw new Error(`no op ${name}`);
      const op = res.op as CollectiveOp;
      const t = collectiveTime(
        op.collective,
        op.messageBytes,
        op.groupSize,
        path,
        DEFAULT_SIM_OPTIONS.collectiveAlgorithm,
      );
      expect(res.linkLevel).toBe('hierarchical');
      expect(t.totalSec).toBeGreaterThan(0);
      expect(res.commLatencySec).toBeCloseTo(t.latencySec, 15);
      expect(res.commBandwidthSec).toBeCloseTo(t.bandwidthSec, 15);
    }
  });
});
