/**
 * Speculative decoding with pipeline parallelism (docs/simulation-model.md §13 "Pipeline parallelism"): the verify's
 * k + 1 rows flow through the stages with today's PP math, a block drafter's target features ride on the stage sends
 * (`draft.ts: ppBoundaryDraftFeatureVectors`), the last stage keeps its own embedding copy
 * (`memory.ts: draftEmbeddingCopyBytes`), and the §7 decode cycle rule is unchanged. Hand-derived numbers in the
 * test titles.
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_CLUSTER, SYNTH_GPU } from '../__fixtures__/synthetic';
import type { BlockDraftSpec, ModelSpec, QuantConfig } from '../ir/model';
import { ppBoundaryDraftFeatureVectors } from '../lower/draft';
import type { LoweringContext } from '../lower/lower';
import { REF_DENSE_8B } from '../models/reference';
import { REF_DENSE_8B_MTP } from '../models/referenceDraft';
import { draftEmbeddingCopyBytes } from './memory';
import { simulate } from './simulate';
import type { DeploymentConfig, SimInput, SimResult } from './types';
import { SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR, validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

/** REF_DENSE_8B dims: h 4096, V 151,936, 36 layers (pp 2: stage 0 = layers [0, 18)). */
const H = 4096;
const V = 151_936;

/** A DFlash-2-shaped hypothetical block drafter whose feature layers span both pp-2 stages: 1, 9, 17 | 25, 33. */
const DRAFTER: BlockDraftSpec = {
  kind: 'block_drafter',
  method: 'dflash2',
  hidden: H,
  vocab: V,
  layers: 5,
  heads: 32,
  kvHeads: 8,
  headDim: 128,
  intermediate: 12_288,
  gated: true,
  activation: 'silu',
  windowTokens: 2048,
  targetLayerIds: [1, 9, 17, 25, 33],
  blockSize: 8,
  source: 'hypothetical',
};

const dflash = (k: number): DeploymentConfig => ({
  speculative: {
    method: 'dflash2',
    drafter: { kind: 'hypothetical', spec: DRAFTER },
    draftLength: k,
    acceptance: { kind: 'iid', alpha: 0.8 },
  },
});
const mtp = (k: number): DeploymentConfig => ({
  speculative: { method: 'mtp', draftLength: k, acceptance: { kind: 'iid', alpha: 0.8 } },
});

const P = 128;

function input(
  model: ModelSpec,
  pp: number,
  deployment: DeploymentConfig | undefined,
  batch = 4,
): SimInput {
  const i: SimInput = {
    model,
    quant: BF16,
    parallel: { tp: 1, pp, dp: 1, ep: 1 },
    workload: { batch, promptLen: P, decodeContextLen: 256 },
    hardware: { device: SYNTH_GPU, topology: SYNTH_CLUSTER },
  };
  if (deployment !== undefined) i.deployment = deployment;
  return i;
}

function stageLayerNames(r: SimResult, s: number): string[] {
  return r.decode.stages[s].layers.map((l) => l.name);
}

describe('ppBoundaryDraftFeatureVectors: F = |{ i ∈ targetLayerIds : i < boundary − 1 }|', () => {
  it('ids [1, 9, 17, 25, 33]: boundary 18 → 2 (17 is the send itself), 0 → 0, 36 → 5, 37 → 5', () => {
    expect(ppBoundaryDraftFeatureVectors(DRAFTER, 18)).toBe(2);
    expect(ppBoundaryDraftFeatureVectors(DRAFTER, 0)).toBe(0);
    expect(ppBoundaryDraftFeatureVectors(DRAFTER, 36)).toBe(5);
    expect(ppBoundaryDraftFeatureVectors(DRAFTER, 37)).toBe(5);
  });
  it('boundary 10 → 1 (id 9 is the last layer before it), 11 → 2 (id 9 now an extra vector), 9 → 1', () => {
    expect(ppBoundaryDraftFeatureVectors(DRAFTER, 10)).toBe(1);
    expect(ppBoundaryDraftFeatureVectors(DRAFTER, 11)).toBe(2);
    expect(ppBoundaryDraftFeatureVectors(DRAFTER, 9)).toBe(1);
  });
  it('MTP: 0 at every boundary (it reads the last stage only)', () => {
    const draft = REF_DENSE_8B_MTP.draft;
    if (draft === undefined) throw new Error('no draft');
    expect(ppBoundaryDraftFeatureVectors(draft, 18)).toBe(0);
  });
});

describe('draftEmbeddingCopyBytes: W_copy = [pp > 1 ∧ ¬(tied ∧ same format)]·ceil(V/tp)·h·bpe(emb)', () => {
  const ctx = (pp: number, tp = 1): LoweringContext => ({
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 2,
    quant: BF16,
    parallel: { tp, pp, dp: 1, ep: 1 },
  });
  it('pp 1 → 0', () => {
    expect(draftEmbeddingCopyBytes(REF_DENSE_8B, ctx(1))).toBe(0);
  });
  it('pp 2, untied, tp 1 → 151,936·4096·2 = 1,244,659,712 B; tp 2 → half', () => {
    expect(draftEmbeddingCopyBytes(REF_DENSE_8B, ctx(2))).toBe(V * H * 2);
    expect(draftEmbeddingCopyBytes(REF_DENSE_8B, ctx(2))).toBe(1_244_659_712);
    expect(draftEmbeddingCopyBytes(REF_DENSE_8B, ctx(2, 2))).toBe((V / 2) * H * 2);
  });
  it('pp 2, tied head in the same format → 0 (the last stage already holds the matrix)', () => {
    expect(draftEmbeddingCopyBytes({ ...REF_DENSE_8B, tieEmbeddings: true }, ctx(2))).toBe(0);
  });
  it('pp 2, tied head stored in another format → the full copy', () => {
    const c = ctx(2);
    c.quant = { ...BF16, lmHead: { dtype: 'fp8_e4m3' } };
    expect(draftEmbeddingCopyBytes({ ...REF_DENSE_8B, tieEmbeddings: true }, c)).toBe(V * H * 2);
  });
});

describe('block drafter (DFlash 2, k 3) at pp 2 on REF_DENSE_8B, batch 4 (b_mb 2), P 128, N 256', () => {
  const r = simulate(input(REF_DENSE_8B, 2, dflash(3)));
  const plain = simulate(input(REF_DENSE_8B, 2, undefined));
  // Boundary 18: ids 1 and 9 travel as extra vectors; id 17 is the stage's own output, the activation send itself.
  const F0 = 2;
  it('validates (pp > 1 is no longer rejected)', () => {
    expect(validateInput(input(REF_DENSE_8B, 2, dflash(3)))).toEqual([]);
  });
  it(`decode send S_0 = b_mb·(k+1)·(1 + F_0)·h·2 = 2·4·3·4096·2 = 196,608 B`, () => {
    expect(r.decode.stages[0].sendToNext?.op).toMatchObject({
      kind: 'collective',
      messageBytes: 2 * 4 * (1 + F0) * H * 2,
    });
    expect(2 * 4 * (1 + F0) * H * 2).toBe(196_608);
  });
  it('prefill send S_0 = b_mb·P·(1 + F_0)·h·2 = 2·128·3·4096·2 = 6,291,456 B', () => {
    const op = r.prefill.stages[0].sendToNext?.op;
    expect(op?.kind === 'collective' ? op.messageBytes : NaN).toBe(6_291_456);
    expect(2 * P * (1 + F0) * H * 2).toBe(6_291_456);
  });
  it('loop-back b_mb·(k+1)·4 = 2·4·4 = 32 B (§5, unchanged)', () => {
    const op = r.decode.loopBack?.op;
    expect(op?.kind === 'collective' ? op.messageBytes : NaN).toBe(32);
  });
  it('the draft.* layers sit on the last stage only', () => {
    expect(stageLayerNames(r, 0).some((n) => n.startsWith('draft.'))).toBe(false);
    expect(stageLayerNames(r, 1).filter((n) => n.startsWith('draft.')).length).toBeGreaterThan(0);
    expect(stageLayerNames(r, 1).at(-1)).toBe('draft.head');
  });
  it('the last stage draftBytes grows by exactly ceil(V/tp)·h·2 = 1,244,659,712 B against the same spec at pp 1', () => {
    const pp1 = simulate(input(REF_DENSE_8B, 1, dflash(3)));
    expect(r.memory.perStage[1].draftBytes - pp1.memory.perStage[0].draftBytes).toBe(V * H * 2);
    expect(r.memory.perStage[0].draftBytes).toBe(0);
  });
  it('the spec-off comparison sees no feature bytes: speedup uses the plain pp-2 decode', () => {
    const s = r.decode.speculative;
    expect(s).toBeDefined();
    expect(s?.offTokensPerSec).toBe(plain.decode.tokensPerSec);
    expect(s?.offCycleSec).toBe(plain.decode.cycleSec);
  });
});

describe('MTP (k 1) at pp 2 on REF_DENSE_8B_MTP, batch 4', () => {
  const r = simulate(input(REF_DENSE_8B_MTP, 2, mtp(1)));
  it('decode send = the plain verify send (F = 0): 2·2·1·4096·2 = 32,768 B', () => {
    const op = r.decode.stages[0].sendToNext?.op;
    expect(op?.kind === 'collective' ? op.messageBytes : NaN).toBe(2 * 2 * H * 2);
  });
  it('a tied model with the same format adds no embedding copy (draftBytes pp 2 = pp 1)', () => {
    const tied: ModelSpec = { ...REF_DENSE_8B_MTP, tieEmbeddings: true };
    const pp2 = simulate(input(tied, 2, mtp(1)));
    const pp1 = simulate(input(tied, 1, mtp(1)));
    expect(pp2.memory.perStage[1].draftBytes).toBe(pp1.memory.perStage[0].draftBytes);
  });
  it('cycle rule (§7): cycle = max(m·b, Σ t_s + Σ p2p + return); tokens/s = dp·B_r·E[τ]/cycle', () => {
    const d = r.decode;
    const m = d.microbatches;
    expect(m).toBe(2);
    const loop = d.loopBack;
    expect(loop).toBeDefined();
    const b = Math.max(
      ...d.stages.map((st) => st.sec),
      ...d.stages.map((st) => st.sendToNext?.commBandwidthSec ?? 0),
      loop?.commBandwidthSec ?? 0,
    );
    const path =
      d.stages.reduce((s, st) => s + st.sec, 0) +
      d.stages.reduce((s, st) => s + (st.sendToNext?.sec ?? 0), 0) +
      (loop?.sec ?? 0);
    expect(d.cycleSec).toBe(Math.max(m * b, path));
    const E = d.speculative?.expectedTokensPerStep ?? NaN;
    expect(E).toBeCloseTo(1.8, 12);
    expect(d.tokensPerSec).toBe((1 * 4 * E) / d.cycleSec);
    expect(d.speculative).toBeDefined();
  });
});

describe('validation of spec-dec at pp > 1', () => {
  it('pp 2 + spec-dec is valid', () => {
    expect(validateInput(input(REF_DENSE_8B_MTP, 2, mtp(1)))).toEqual([]);
  });
  it('pp 2 + spec-dec + token embedding off-device: the new exact error', () => {
    const i = input(REF_DENSE_8B_MTP, 2, {
      ...mtp(1),
      offload: { embeddings: { tier: 'host_ram' } },
    });
    expect(validateInput(i)).toContain(SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR);
    expect(SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR).toBe(
      'speculative decoding with pp > 1 and the token embedding off-device is not supported',
    );
  });
  it('pp 1 + spec-dec + token embedding off-device: no such error', () => {
    const i = input(REF_DENSE_8B_MTP, 1, {
      ...mtp(1),
      offload: { embeddings: { tier: 'host_ram' } },
    });
    expect(validateInput(i)).not.toContain(SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR);
  });
});
