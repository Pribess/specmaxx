/**
 * Speculative decoding end to end through `simulate` (design doc §3.3–3.5, §3.8, §8.2 U-8, U-14, U-18, §8.5 W2):
 * the verify over k + 1 positions, the draft layers on the last stage, the E[τ] rate, the draft memory and the
 * spec-dec × offload interplay. Exact (rel 1e-6) on the hand-derived terms, within 0.1% on the quoted tok/s.
 */
import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { ModelSpec, QuantConfig } from '../ir/model';
import type { Op } from '../ir/ops';
import { linearAttentionSequenceStateBytes } from '../lower/linearAttention';
import type { LoweringContext } from '../lower/lower';
import { REF_MOE_30B_A3B, TINY_HYBRID } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import { stageSnapshotStateBytesPerSeq } from './memory';
import { simulate } from './simulate';
import type { AcceptanceModel, DeploymentConfig, OpResult, SimInput, SimResult } from './types';
import { validateInput } from './validate';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const BF16: QuantConfig = { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } };
const GPU = { device: RTX_5090, topology: singleDevice(RTX_5090) };
const B_LINK = (32e9 * 16 * 128) / 130 / 8;

function run(
  model: ModelSpec,
  deployment: DeploymentConfig | undefined,
  over: Partial<SimInput> = {},
): SimResult {
  const input: SimInput = {
    model,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: GPU,
    ...over,
  };
  if (deployment !== undefined) input.deployment = deployment;
  return simulate(input);
}

const mtp = (k: number, acceptance: AcceptanceModel): DeploymentConfig => ({
  speculative: { method: 'mtp', draftLength: k, acceptance },
});

/** The decode ops of the target (verify) and of the drafter, separately. */
function splitOps(r: SimResult): { verify: OpResult[]; draft: OpResult[] } {
  const layers = r.decode.stages.flatMap((s) => s.layers);
  return {
    verify: layers.filter((l) => !l.name.startsWith('draft.')).flatMap((l) => l.ops),
    draft: layers.filter((l) => l.name.startsWith('draft.')).flatMap((l) => l.ops),
  };
}

function opNamed(ops: OpResult[], name: string): Op {
  const found = ops.find((o) => o.op.name === name);
  if (found === undefined) throw new Error(`no op ${name}`);
  return found.op;
}

describe('W2 MTP k = 1 on REF_DENSE_8B, one RTX 5090, bf16, b 1, decode @ 8k (design doc §8.5)', () => {
  const r = run(REF_DENSE_8B_MTP, mtp(1, { kind: 'iid', alpha: 0.85 }));
  const base = run(REF_DENSE_8B_MTP, undefined);
  it('baseline decode 9.128152 ms (109.55 tok/s @ 8k)', () => {
    expectClose(base.decode.cycleSec, 9.128152e-3, 1e-6);
  });
  it('verifySec 9.135339 ms, draftSec 0.966748 ms, E[τ] 1.85, 183.13 tok/s @ 8k, speedup ×1.67', () => {
    const s = r.decode.speculative;
    expect(s).toBeDefined();
    expectClose(s?.verifySec ?? 0, 9.135339e-3, 1e-6);
    expectClose(s?.draftSec ?? 0, 9.66748e-4, 1e-6);
    expectClose(s?.expectedTokensPerStep ?? 0, 1.85, 1e-12);
    expectClose(r.decode.cycleSec, 10.102087e-3, 1e-6);
    expectClose(r.decode.tokensPerSec, 183.13, 1e-3);
    expectClose(s?.speedup ?? 0, 1.6717, 1e-3);
    expectClose(s?.speedup ?? 0, r.decode.tokensPerSec / base.decode.tokensPerSec, 1e-12);
    expectClose(r.decode.perSequenceTokensPerSec, 1.85 / r.decode.cycleSec, 1e-12);
    expectClose(r.decode.latencySec, r.decode.cycleSec / 1.85, 1e-12);
  });
  it.each([
    [0.75, 173.2, 1.58],
    [0.82, 180.2, 1.64],
  ])('α %d: %d tok/s @ 8k (×%d)', (alpha, rate, speedup) => {
    const x = run(REF_DENSE_8B_MTP, mtp(1, { kind: 'iid', alpha }));
    expectClose(x.decode.tokensPerSec, rate, 1e-3);
    expectClose(x.decode.speculative?.speedup ?? 0, speedup, 5e-3);
  });
  it('batch 8: the verify (m = 16 < 117) is still weight-bound, 13.955 ms vs 13.897 ms baseline: the gain holds', () => {
    const over = { workload: { batch: 8, promptLen: 8192, decodeContextLen: 8192 } };
    const b8 = run(REF_DENSE_8B_MTP, mtp(1, { kind: 'iid', alpha: 0.85 }), over);
    const b8base = run(REF_DENSE_8B_MTP, undefined, over);
    expectClose(b8.decode.speculative?.verifySec ?? 0, 13.955e-3, 1e-4);
    expectClose(b8base.decode.cycleSec, 13.897e-3, 1e-4);
    expect(b8.decode.speculative?.speedup ?? 0).toBeGreaterThan(1.6);
  });
  it('memory: draft weights 453,026,304 B (module + eh_proj + 3 norms) + draft KV at 8194 tokens 33,562,624 B', () => {
    expect(r.memory.perStage[0].draftBytes).toBe(453_026_304 + 8194 * 4096);
    expect(base.memory.perStage[0].draftBytes).toBe(0);
    // The verify appends k + 1 = 2 rows: the target's KV is sized for N + k + 1 = 8194 tokens.
    expect(r.memory.perStage[0].sequenceStateBytes - base.memory.perStage[0].sequenceStateBytes).toBe(36 * 4096);
    expect(r.memory.perStage[0].speculativeTransientBytes).toBe(0); // dense: no recurrent state
  });
  it('prefill gets the drafter: one MTP forward over the prompt, slower than without', () => {
    const last = r.prefill.stages[0].layers.at(-1);
    expect(last?.name).toBe('draft.0');
    expect(r.prefill.cycleSec).toBeGreaterThan(base.prefill.cycleSec);
  });
});

describe('U-8 verify lowering: REF_DENSE_8B k 1 @ N 8192', () => {
  const { verify } = splitOps(run(REF_DENSE_8B_MTP, mtp(1, { kind: 'iid', alpha: 0.85 })));
  it('every GEMM m = 2, attn.core qLen 2 kvLen 8194, lm_head m = 2', () => {
    const gemms = verify.flatMap((o) => (o.op.kind === 'gemm' ? [o.op] : []));
    expect(gemms.length).toBeGreaterThan(0);
    expect(gemms.every((g) => g.m === 2)).toBe(true);
    const cores = verify.flatMap((o) => (o.op.kind === 'attention' ? [o.op] : []));
    expect(cores.every((a) => a.qLen === 2 && a.kvLen === 8194)).toBe(true);
    expect(cores.length).toBe(36);
    const head = opNamed(verify, 'lm_head');
    expect(head.kind === 'gemm' && head.m).toBe(2);
  });
});

describe('U-18 MTP cache advance (REF_DENSE_8B k 1, explicit per-position acceptance)', () => {
  it('alphas [1] (E[τ] = 2): the catch-up pass has qLen 2, writes 2·4096 B of KV, eh_proj m 2, lm_head m 1', () => {
    const { draft } = splitOps(run(REF_DENSE_8B_MTP, mtp(1, { kind: 'per_position', alphas: [1] })));
    const core = draft.find((o) => o.op.name === 'draft.attn.core');
    expect(core?.op.kind === 'attention' && core.op.qLen).toBe(2);
    expect(core?.bytes.kvWriteBytes).toBe(8_192);
    const eh = opNamed(draft, 'draft.eh_proj');
    expect(eh.kind === 'gemm' && eh.m).toBe(2);
    const head = opNamed(draft, 'draft.lm_head');
    expect(head.kind === 'gemm' && head.m).toBe(1);
  });
  it('alphas [0] (E[τ] = 1): qLen 1, 4,096 B of KV; the previous tentative row is overwritten (no extra write)', () => {
    const { draft } = splitOps(run(REF_DENSE_8B_MTP, mtp(1, { kind: 'per_position', alphas: [0] })));
    const core = draft.find((o) => o.op.name === 'draft.attn.core');
    expect(core?.op.kind === 'attention' && core.op.qLen).toBe(1);
    expect(core?.bytes.kvWriteBytes).toBe(4_096);
  });
  it('three consecutive steps advance the committed length by E[τ] = 2 each: kvLen = N + 2i + k + 1', () => {
    const N = 8192;
    for (let i = 0; i < 3; i++) {
      // Stepping helper: step i starts with N + 2i committed tokens.
      const over = { workload: { batch: 1, promptLen: 8192, decodeContextLen: N + 2 * i } };
      const { draft } = splitOps(run(REF_DENSE_8B_MTP, mtp(1, { kind: 'per_position', alphas: [1] }), over));
      const core = opNamed(draft, 'draft.attn.core');
      expect(core.kind === 'attention' && core.kvLen).toBe(N + 2 * i + 2);
    }
  });
});

describe('U-14 hybrid verify: TINY_HYBRID with an MTP module, k 3, B_r 2', () => {
  const model: ModelSpec = {
    ...TINY_HYBRID,
    draft: { kind: 'mtp', layers: [TINY_HYBRID.layers[3]], hiddenProj: true },
  };
  const over = { workload: { batch: 2, promptLen: 128, decodeContextLen: 256 } };
  const r = run(model, mtp(3, { kind: 'iid', alpha: 0.75 }), over);
  const base = run(model, undefined, over);
  const verifyOps = splitOps(r).verify;
  const plainOps = splitOps(base).verify;
  it('each GDN RecurrentOp has stateSnapshots 4 and 4× the decode step state write; conv state writes 4×', () => {
    const rec = verifyOps.filter((o) => o.op.kind === 'recurrent');
    const plainRec = plainOps.filter((o) => o.op.kind === 'recurrent');
    expect(rec.length).toBe(3);
    rec.forEach((o, i) => {
      expect(o.op.kind === 'recurrent' && o.op.stateSnapshots).toBe(4);
      expect(o.bytes.stateWriteBytes).toBe(4 * plainRec[i].bytes.stateWriteBytes);
      expect(o.bytes.stateReadBytes).toBe(plainRec[i].bytes.stateReadBytes);
    });
    const conv = verifyOps.filter((o) => o.op.kind === 'elementwise' && o.bytes.stateWriteBytes > 0);
    const plainConv = plainOps.filter((o) => o.op.kind === 'elementwise' && o.bytes.stateWriteBytes > 0);
    expect(conv.length).toBe(plainConv.length);
    conv.forEach((o, i) => expect(o.bytes.stateWriteBytes).toBe(4 * plainConv[i].bytes.stateWriteBytes));
  });
  it('speculativeTransientBytes = 4 · state_s · B_r (state_s = the three GDN layers’ recurrent + conv state)', () => {
    const ctx: LoweringContext = { phase: 'decode', seqs: 2, qLen: 1, kvLen: 257, quant: BF16, parallel: { tp: 1, pp: 1, dp: 1, ep: 1 } };
    const gdnState = TINY_HYBRID.layers
      .flatMap((l) => l.blocks)
      .reduce((sum, b) => sum + (b.kind === 'linear_attention' ? linearAttentionSequenceStateBytes(b, ctx) : 0), 0);
    expect(stageSnapshotStateBytesPerSeq(TINY_HYBRID, ctx, 0)).toBe(gdnState);
    expect(r.memory.perStage[0].speculativeTransientBytes).toBe(4 * gdnState * 2);
    expect(base.memory.perStage[0].speculativeTransientBytes).toBe(0);
  });
});

describe('§3.8 spec-dec × offload: W1 (experts in DDR5, f = h = .5) with an MTP module k = 1', () => {
  const W_EXPERT = 2_654_208;
  const W1_QUANT: QuantConfig = { ...BF16, modules: { routed_experts: { weights: { dtype: 'int4', extraBitsPerElem: 0.5 } } } };
  const model: ModelSpec = {
    ...REF_MOE_30B_A3B,
    draft: { kind: 'mtp', layers: [REF_MOE_30B_A3B.layers[0]], hiddenProj: true, maxReliableDepth: 1 },
  };
  const deployment = (includeDraft: boolean): DeploymentConfig => ({
    offload: { routedExperts: { tier: 'host_ram', residentFraction: 0.5, hitRatio: { kind: 'uniform' }, includeDraft } },
    speculative: { method: 'mtp', draftLength: 1, acceptance: { kind: 'iid', alpha: 0.85 } },
  });
  const over = { quant: W1_QUANT };
  it('the verify needs U(2) = 15.5 experts per layer: M = 7.75 (1.94× W1’s M = 4), transfer 48·(1.07 µs + 7.75·W/B_link)', () => {
    const r = run(model, deployment(false), over);
    const target = r.decode.stages[0].layers.filter((l) => !l.name.startsWith('draft.'));
    const fetches = target.flatMap((l) => l.ops).filter((o) => o.op.kind === 'transfer');
    expect(fetches.length).toBe(48);
    for (const f of fetches) {
      expect(f.op.kind === 'transfer' && f.op.reads[0].count).toBeCloseTo(7.75, 12);
      expectClose(f.sec, 1.07e-6 + (7.75 * W_EXPERT) / B_LINK, 1e-9);
    }
    expectClose(7.75 / 4, 1.9375);
    // includeDraft false: the module's own experts stay resident, no draft fetch.
    expect(splitOps(r).draft.some((o) => o.op.kind === 'transfer')).toBe(false);
    expect(r.memory.tiers[0].bytes).toBe(16_307_453_952);
  });
  it('includeDraft: the module’s MoE layer streams its misses too, and its 128 experts join the host-RAM tier', () => {
    const r = run(model, deployment(true), over);
    const fetch = splitOps(r).draft.filter((o) => o.op.kind === 'transfer');
    expect(fetch.length).toBe(1);
    expect(fetch[0].op.name).toBe('draft.moe.expert_fetch');
    expect(r.memory.tiers[0].bytes).toBe(16_307_453_952 + 128 * W_EXPERT);
    const off = run(model, deployment(false), over);
    expect(off.memory.perStage[0].draftBytes - r.memory.perStage[0].draftBytes).toBe(64 * W_EXPERT);
  });
});

describe('spec-dec off is today: validation and identity of the knob', () => {
  it('deployment {} or speculative absent equals no deployment, bit for bit', () => {
    const a = run(REF_DENSE_8B_MTP, undefined);
    const b = run(REF_DENSE_8B_MTP, {});
    expect(b).toEqual(a);
    expect(a.decode.speculative).toBeUndefined();
  });
  it('pp > 1 is rejected with the §3.6 message', () => {
    const input: SimInput = {
      model: REF_DENSE_8B_MTP,
      quant: BF16,
      parallel: { tp: 1, pp: 2, dp: 1, ep: 1 },
      workload: { batch: 2, promptLen: 128, decodeContextLen: 128 },
      hardware: GPU,
      deployment: mtp(1, { kind: 'iid', alpha: 0.8 }),
    };
    expect(validateInput(input)).toContain('speculative decoding with pp > 1 is not supported yet');
  });
  it('a single-step head drafted past its depth warns', () => {
    const r = run(REF_DENSE_8B_MTP, mtp(3, { kind: 'iid', alpha: 0.85 }));
    expect(r.warnings.some((w) => w.includes("past the MTP head's trained depth"))).toBe(true);
  });
});
