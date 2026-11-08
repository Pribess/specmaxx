import { describe, expect, it } from 'vitest';
import { countParams } from '../../ir/params';
import type { ModelSpec } from '../../ir/model';
import { defaultDraftLength } from '../../engine/specdec';
import { lowerDraftDecode, lowerDraftPrefill } from '../../lower/draft';
import type { LoweringContext } from '../../lower/lower';
import { fixtureRepoFiles } from '../__fixtures__/fixtures.testutil';
import { REF_DENSE_8B, REF_MOE_30B_A3B } from '../reference';
import { layerParams } from './families/mlaCommon';
import {
  KNOWN_DRAFTERS,
  buildTargetLayerIds,
  hypotheticalDrafter,
  knownDrafters,
  mtpDraft,
  parseDrafterConfig,
  withDraft,
} from './drafter';
import { mapHfModel } from './index';
import type { JsonObject } from './types';

const RAW = import.meta.glob<string>('./__fixtures__/*.json', {
  query: '?raw',
  import: 'default',
  eager: true,
});
const config = (file: string) => JSON.parse(RAW[`./__fixtures__/${file}`]!) as JsonObject;
/** A shallow copy of `obj` without `keys`. */
function omit<T extends object>(obj: T, ...keys: string[]): T {
  const copy = { ...obj } as Record<string, unknown>;
  for (const key of keys) delete copy[key];
  return copy as unknown as T;
}
const DFLASH2_CONFIG = config('incoai__Qwen3.8-27B-DFlash2.config.json');
const DSPARK_CONFIG = config('deepseek-ai__dspark_qwen3_8b_block7.config.json');

describe('parseDrafterConfig (released configs, fetched from huggingface.co)', () => {
  it('incoai/Qwen3.8-27B-DFlash2: 5 layers, h 5120, FFN 17408, 32/8 heads of 128, window 2048, dflash_config fields', () => {
    const r = parseDrafterConfig(DFLASH2_CONFIG, 'incoai/Qwen3.8-27B-DFlash2', 'abc123');
    expect(r.kind).toBe('ok');
    if (r.kind !== 'ok') return;
    expect(r.spec).toEqual({
      kind: 'block_drafter',
      method: 'dflash2',
      hidden: 5120,
      vocab: 248320,
      layers: 5,
      heads: 32,
      kvHeads: 8,
      headDim: 128,
      intermediate: 17408,
      gated: true,
      activation: 'silu',
      windowTokens: 2048,
      targetLayerIds: [5, 19, 33, 47, 61],
      blockSize: 8,
      conv: { kernel: 2, groupSize: 16 },
      selector: { topK: 16, rank: 256 },
      source: 'incoai/Qwen3.8-27B-DFlash2@abc123',
    });
  });
  it('deepseek-ai/dspark_qwen3_8b_block7: full attention, top-level target ids, markov rank 256, blockSize = block_size 7 + 1', () => {
    const r = parseDrafterConfig(DSPARK_CONFIG, 'deepseek-ai/dspark_qwen3_8b_block7');
    expect(r.kind).toBe('ok');
    if (r.kind !== 'ok') return;
    expect(r.spec).toMatchObject({
      method: 'dspark',
      hidden: 4096,
      vocab: 151936,
      layers: 5,
      heads: 32,
      kvHeads: 8,
      headDim: 128,
      intermediate: 12288,
      targetLayerIds: [1, 9, 17, 25, 33],
      blockSize: 8,
      markovRank: 256,
      source: 'deepseek-ai/dspark_qwen3_8b_block7',
    });
    expect(r.spec.windowTokens).toBeUndefined();
    expect(r.spec.conv).toBeUndefined();
    expect(r.notes.some((n) => n.includes('confidence head'))).toBe(true);
  });
  it('rejects other architectures (DFlash v1, a target model)', () => {
    const v1 = parseDrafterConfig(
      { ...DFLASH2_CONFIG, architectures: ['DFlashDraftModel'] },
      'x/y',
    );
    expect(v1.kind).toBe('invalid_config');
    expect(parseDrafterConfig({ architectures: ['Qwen3ForCausalLM'] }, 'x/y').kind).toBe(
      'invalid_config',
    );
  });
  it('reports missing fields with paths', () => {
    const rest = omit(DFLASH2_CONFIG, 'dflash_config', 'hidden_size');
    const r = parseDrafterConfig({ ...rest, dflash_config: { block_size: 8 } }, 'x/y');
    expect(r.kind).toBe('invalid_config');
    if (r.kind !== 'invalid_config') return;
    expect(r.errors).toEqual(
      expect.arrayContaining([
        'hidden_size: missing non-negative integer',
        'conv_kernel_size: missing non-negative integer',
      ]),
    );
  });
  it('rejects mixed layer types and falls back to build_target_layer_ids without target_layer_ids', () => {
    const mixed = parseDrafterConfig(
      {
        ...DSPARK_CONFIG,
        layer_types: [
          'full_attention',
          'sliding_attention',
          'full_attention',
          'full_attention',
          'full_attention',
        ],
      },
      'x/y',
    );
    expect(mixed.kind).toBe('invalid_config');
    const noIds = omit(DSPARK_CONFIG, 'target_layer_ids');
    const r = parseDrafterConfig(noIds, 'x/y');
    expect(r.kind === 'ok' && r.spec.targetLayerIds).toEqual([1, 9, 17, 25, 33]);
  });
});

describe('buildTargetLayerIds (DFlash build_target_layer_ids)', () => {
  it('36 target layers, 5 draft layers → [1, 9, 17, 25, 33] (the DSpark config)', () =>
    expect(buildTargetLayerIds(36, 5)).toEqual([1, 9, 17, 25, 33]));
  it('64 / 5 → [1, 16, 31, 46, 61]; one draft layer → the middle', () => {
    expect(buildTargetLayerIds(64, 5)).toEqual([1, 16, 31, 46, 61]);
    expect(buildTargetLayerIds(36, 1)).toEqual([18]);
  });
  it('rounds half to even like Python (48 / 5: 1 + 11·i → exact; 40 / 3: [1, 19, 37])', () => {
    expect(buildTargetLayerIds(48, 5)).toEqual([1, 12, 23, 34, 45]);
    expect(buildTargetLayerIds(40, 3)).toEqual([1, 19, 37]);
    // 6 / 3: span 2, i = 1 → 2.0; 7 / 5: span 3, 1 + 0.75·i → 1, 2 (1.75), 2 (2.5 → even), 3 (3.25), 4
    expect(buildTargetLayerIds(7, 5)).toEqual([1, 2, 2, 3, 4]);
  });
});

describe('known drafters and the hypothetical drafter', () => {
  it('maps Qwen3.8-27B and Qwen3-8B to their released drafters (case-insensitive)', () => {
    expect(knownDrafters('qwen/qwen3.8-27b').map((d) => d.repoId)).toEqual([
      'incoai/Qwen3.8-27B-DFlash2',
    ]);
    expect(knownDrafters('Qwen/Qwen3-8B')).toEqual([
      { targetId: 'Qwen/Qwen3-8B', repoId: 'deepseek-ai/dspark_qwen3_8b_block7', method: 'dspark' },
    ]);
    expect(knownDrafters('org/unknown')).toEqual([]);
    expect(KNOWN_DRAFTERS).toHaveLength(2);
  });
  it('hypothetical DFlash 2 on REF_DENSE_8B: target h and V, FFN 12288, 5 layers of 32/8 × 128, window 2048, ids from build_target_layer_ids', () => {
    expect(hypotheticalDrafter(REF_DENSE_8B, 'dflash2')).toEqual({
      kind: 'block_drafter',
      method: 'dflash2',
      hidden: 4096,
      vocab: 151936,
      layers: 5,
      heads: 32,
      kvHeads: 8,
      headDim: 128,
      intermediate: 12288,
      gated: true,
      activation: 'silu',
      windowTokens: 2048,
      targetLayerIds: [1, 9, 17, 25, 33],
      blockSize: 8,
      conv: { kernel: 2, groupSize: 16 },
      selector: { topK: 16, rank: 256 },
      source: 'hypothetical',
    });
  });
  it('hypothetical DSpark on a MoE target: FFN = topK · expert width, Markov rank 256', () => {
    const d = hypotheticalDrafter(REF_MOE_30B_A3B, 'dspark');
    expect(d.intermediate).toBe(8 * 768);
    expect(d.markovRank).toBe(256);
    expect(d.selector).toBeUndefined();
  });
  it('withDraft keeps a model without a draft as the same object', () => {
    expect(withDraft(REF_DENSE_8B, undefined)).toBe(REF_DENSE_8B);
    expect(mtpDraft(() => REF_DENSE_8B.layers[0], 0, true)).toBeUndefined();
  });
});

/** The HF map of a fixture repo (throws unless ok). */
function mapped(repoId: string) {
  const r = mapHfModel(fixtureRepoFiles(repoId));
  if (r.kind !== 'ok') throw new Error(`${repoId}: ${JSON.stringify(r)}`);
  return r;
}

describe('HF families keep their drafter as model.draft (spec-dec off: never counted)', () => {
  const cases: [string, string, (m: ModelSpec) => void][] = [
    [
      'Qwen/Qwen3.8-27B',
      'mtp',
      (m) => {
        expect(m.draft).toMatchObject({ kind: 'mtp', hiddenProj: true });
        expect(m.draft && 'maxReliableDepth' in m.draft).toBe(false);
      },
    ],
    [
      'zai-org/GLM-5.3',
      'mtp',
      (m) => expect(m.draft).toMatchObject({ kind: 'mtp', maxReliableDepth: 1 }),
    ],
    [
      'zai-org/GLM-5.3-Flash',
      'mtp',
      (m) => expect(m.draft).toMatchObject({ kind: 'mtp', maxReliableDepth: 1 }),
    ],
    [
      'thinkingmachines/Inkling-Small',
      'mtp',
      (m) => expect(m.draft).toMatchObject({ kind: 'mtp', maxReliableDepth: 8 }),
    ],
    [
      'deepseek-ai/DeepSeek-V4.1-Flash',
      'block_drafter',
      (m) =>
        expect(m.draft).toMatchObject({
          kind: 'block_drafter',
          method: 'dspark',
          layers: 3,
          windowTokens: 128,
          targetLayerIds: [37, 38, 39],
          blockSize: 6,
          markovRank: 256,
        }),
    ],
  ];
  it.each(cases)('%s carries a %s draft', (repoId, _kind, check) => {
    const r = mapped(repoId);
    check(r.model);
    // countParams and the param check never see the draft
    const withoutDraft = omit(r.model, 'draft');
    expect(countParams(r.model)).toBe(countParams(withoutDraft));
  });
  it('DeepSeek-V4.1-Flash DSpark experts stream only with includeDraft (design doc §3.8)', () => {
    const r = mapped('deepseek-ai/DeepSeek-V4.1-Flash');
    const draft = r.model.draft!;
    const ctx = (includeDraft: boolean): LoweringContext => ({
      phase: 'decode',
      seqs: 1,
      qLen: 6,
      kvLen: 8198,
      quant: r.quant.quant,
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      offload: {
        routedExperts: {
          tier: 'host_ram',
          layers: { from: 0, to: 40 },
          residentFraction: 0.5,
          hitRatio: 0.5,
          includeDraft,
        },
      },
    });
    const fetches = (includeDraft: boolean) =>
      lowerDraftDecode(r.model, draft, ctx(includeDraft), 2, 5)
        .flatMap((l) => l.ops)
        .filter((op) => op.kind === 'transfer');
    expect(fetches(false)).toHaveLength(0);
    expect(fetches(true)).toHaveLength(3);
    expect(fetches(true).every((op) => op.name.startsWith('draft.'))).toBe(true);
  });
  it('Qwen3.8-27B MTP: one gated-attention module layer, no recurrent state', () => {
    const d = mapped('Qwen/Qwen3.8-27B').model.draft;
    expect(d?.kind === 'mtp' && d.layers.map((l) => l.name)).toEqual(['mtp.0']);
    expect(d?.kind === 'mtp' && d.layers[0].blocks.some((b) => b.kind === 'linear_attention')).toBe(
      false,
    );
  });
  it('GLM-5.3 MTP params equal the param-check estimate (layer + eh_proj 2h² + 3 norms)', () => {
    const r = mapped('zai-org/GLM-5.3');
    const d = r.model.draft;
    if (d?.kind !== 'mtp') throw new Error('no MTP draft');
    const h = r.model.hidden;
    expect(d.layers.reduce((s, l) => s + layerParams(l) + 2 * h * h + 3 * h, 0)).toBe(
      r.excluded.mtpParamsEstimate,
    );
  });
  it('DeepSeek-V4.1-Flash DSpark: 3 MoE stages (128 experts, top-3, window-only CSA)', () => {
    const d = mapped('deepseek-ai/DeepSeek-V4.1-Flash').model.draft;
    if (d?.kind !== 'block_drafter' || d.moeLayers === undefined) throw new Error('no MoE drafter');
    expect(d.moeLayers.map((l) => l.name)).toEqual(['mtp.0', 'mtp.1', 'mtp.2']);
    const moe = d.moeLayers[0].blocks.find((b) => b.kind === 'moe');
    expect(moe).toMatchObject({ experts: 128, topK: 3 });
    const csa = d.moeLayers[0].blocks.find((b) => b.kind === 'csa_attention');
    expect(csa?.kind === 'csa_attention' && csa.global).toBeUndefined();
  });
  it('Mistral Large 4 and MiniMax keep no draft (Mistral DSpark deferred: params.json is underspecified)', () => {
    expect(mapped('mistralai/Mistral-Large-4-1T-A52B').model.draft).toBeUndefined();
    expect(mapped('MiniMaxAI/MiniMax-M3').model.draft).toBeUndefined();
  });
  it.each(cases.map(([id]) => id))(
    '%s: the draft lowers for decode (default k) and prefill',
    (repoId) => {
      const r = mapped(repoId);
      const draft = r.model.draft!;
      const method = draft.kind === 'mtp' ? 'mtp' : draft.method;
      const k = defaultDraftLength(method, draft);
      const base: LoweringContext = {
        phase: 'decode',
        seqs: 1,
        qLen: k + 1,
        kvLen: 8192 + k + 1,
        quant: r.quant.quant,
        parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      };
      const decode = lowerDraftDecode(r.model, draft, base, Math.min(2, k + 1), k);
      expect(decode.flatMap((l) => l.ops).every((op) => op.name.startsWith('draft.'))).toBe(true);
      const prefill = lowerDraftPrefill(r.model, draft, {
        ...base,
        phase: 'prefill',
        qLen: 4096,
        kvLen: 4096,
      });
      expect(prefill.flatMap((l) => l.ops).length).toBeGreaterThan(0);
    },
  );
});
