// DeepSeek-V4-Flash-0731 (`deepseek_v4`), built by the DeepSeek-V4.1 builder with the V4 profile
// (`deepseekV41.ts: deepseekV4Family`). Targets are derived from the safetensors counts of the real checkpoint
// (`__fixtures__/deepseek-ai__DeepSeek-V4-Flash-0731.api.json`) and the shapes of its inference/model.py; the tech
// report and the release blog quote 284B total / 13B activated (NVIDIA NGC: 304B incl. the MTP stages), which
// matches: main 284.33B, active 13.27B excluding the embedding, 304.18B incl. DSpark MTP (= safetensors.total).

import { describe, expect, it } from 'vitest';
import { bytesPerElement } from '../../../ir/dtype';
import type { CsaAttentionBlock, LayerSpec, ModelSpec } from '../../../ir/model';
import { activeParamsPerToken, countParams } from '../../../ir/params';
import { simulate } from '../../../engine/simulate';
import { singleDevice } from '../../../hw/presets';
import { B200 } from '../../../hw/presets/nvidia-datacenter';
import { fixtureRepoFiles } from '../../__fixtures__/fixtures.testutil';
import { mapHfModel } from '../index';
import { checkParams } from '../paramCheck';
import { unwrapTextConfig } from '../textConfig';
import type { FamilyBuild } from '../types';
import { deepseekV4Family } from './deepseekV41';
import { layerParams } from './mlaCommon';

const ID = 'deepseek-ai/DeepSeek-V4-Flash-0731';

/** Build the fixture with the V4 family; fails the test unless the build is ok. */
function build(): FamilyBuild {
  const files = fixtureRepoFiles(ID);
  const u = unwrapTextConfig(files.config);
  if (u.kind !== 'ok') throw new Error('unwrap failed');
  const r = deepseekV4Family.build(u.view.text, u.view.top, files);
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  return r;
}

const csaOf = (layer: LayerSpec): CsaAttentionBlock => {
  const b = layer.blocks.find((x) => x.kind === 'csa_attention');
  if (b?.kind !== 'csa_attention') throw new Error(`${layer.name}: no csa block`);
  return b;
};

/** The CSA role of a layer, from its block. */
function role(layer: LayerSpec): string {
  const g = csaOf(layer).global;
  if (g === undefined) return 'swa';
  if (g.selection.kind !== 'indexer') return g.selection.kind;
  return g.selection.indexer.ownsKeys === false ? 'reindex' : 'full';
}

describe('deepseek_v4: DeepSeek-V4-Flash-0731', () => {
  const b = build();
  const m: ModelSpec = b.model;

  it('reads the dims and the mHC stream count', () => {
    expect(m.layers).toHaveLength(43);
    expect(m.hidden).toBe(4096);
    expect(m.vocab).toBe(129280);
    expect(m.tieEmbeddings).toBe(false);
    expect(m.residualStreams).toBe(4);
  });

  it('assigns the CSA roles from compress_ratios alone (no kv_source / index_source lists)', () => {
    const roles = m.layers.map(role);
    const count = (r: string) => roles.filter((x) => x === r).length;
    expect([count('swa'), count('full'), count('all'), count('reuse'), count('reindex')]).toEqual([
      2, 21, 20, 0, 0,
    ]);
    expect([roles[0], roles[1], roles[2], roles[3], roles[42]]).toEqual([
      'swa',
      'swa',
      'full',
      'all',
      'full',
    ]);
    // No layer reads another's compressed cache.
    expect(m.layers.some((l) => csaOf(l).global?.kvCacheFromLayer !== undefined)).toBe(false);
  });

  it('r = 4: overlapping compressor + own indexer with its own overlapping key compressor; r = 128: attend all', () => {
    expect(csaOf(m.layers[2]).global).toEqual({
      compressRatio: 4,
      compressorOverlap: true,
      selection: {
        kind: 'indexer',
        indexer: {
          heads: 64,
          headDim: 128,
          topK: 512,
          keyPoolTokens: 4,
          keyPoolGate: true,
          keyPoolOverlap: true,
          ownsKeys: true,
        },
      },
    });
    expect(csaOf(m.layers[3]).global).toEqual({ compressRatio: 128, selection: { kind: 'all' } });
  });

  it('has no Engram tables', () => {
    expect(m.layers.some((l) => l.blocks.some((x) => x.kind === 'ngram_embedding'))).toBe(false);
  });

  // Per-layer params of the checkpoint (inference/model.py shapes; the I8 fp4 bytes count one element per weight,
  // scales excluded), less what the IR does not count: attn_sink (64), the router bias (256, not in hash layers
  // 0-2), tid2eid in hash layers (129,280·6 = 775,680), the compressor APE (r = 4: 4·1024 = 4096; r = 128:
  // 128·512 = 65,536). The r = 4 indexer's key RMSNorm (128) is counted as a LayerNorm (+128).
  it.each([
    [0, 6_576_416_310, 0, 0], // SWA-only, hash-routed
    [1, 6_576_416_310, 0, 0], // SWA-only, hash-routed
    [2, 6_595_558_582, 4096, 128], // r = 4, hash-routed
    [3, 6_580_676_662, 65_536, 0], // r = 128
    [42, 6_595_558_582, 4096, 128], // r = 4
  ])('layer %i matches the checkpoint shapes', (i, header, ape, keyNormDelta) => {
    expect(layerParams(m.layers[i])).toBe(header - ape + keyNormDelta);
  });

  it('total and active params match the header-derived targets', () => {
    // 304,180,418,494 (safetensors.total) − MTP 19,845,850,983 = 284,334,567,511. The IR gives 284,330,767,890;
    // the 3,799,621 gap: APE 1,396,736 + sinks 2,752 + router biases 10,240 + tid2eid 2,327,040 + hc_head 65,541
    // − LayerNorm 2,688. Tech report: 284B total.
    expect(countParams(m)).toBe(284_330_767_890);
    expect(Math.abs(countParams(m) / 284_334_567_511 - 1)).toBeLessThan(0.005);
    // Active per token 13,798,159,890; excluding the embedding (vocab·h) 13,268,629,010. Tech report: 13B activated.
    expect(activeParamsPerToken(m)).toBe(13_798_159_890);
    const active = activeParamsPerToken(m) - m.vocab * m.hidden;
    expect(Math.abs(active / 13_268_629_010 - 1)).toBeLessThan(0.015);
  });

  it('counts 3 DSpark MTP stages (config says 1) and passes the param check', () => {
    // Checkpoint MTP 19,845,850,983; the estimate misses 66,501 (sinks, biases, the MTP hc_head).
    expect(b.mtpParamsEstimate).toBe(19_845_784_482);
    expect(b.excludedParts).toEqual(['DSpark MTP stages']);
    expect(
      b.notes.some((n) =>
        /num_nextn_predict_layers 1, but compress_ratios lists 3 MTP stages.*using 3/.test(n),
      ),
    ).toBe(true);
    expect(b.notes.some((n) => /layers 0–2 route by token id/.test(n))).toBe(true);
    // IR 284,330,767,890 + MTP 19,845,784,482 = 304,176,552,372 vs 304,180,418,494: −1.27e-5.
    const check = checkParams(m, b, fixtureRepoFiles(ID).api, 'fp8');
    expect(check.status).toBe('ok');
    expect(Math.abs(check.relError ?? 1)).toBeLessThan(1e-4);
  });

  it('hooks the DSpark stages up as the block drafter', () => {
    const d = m.draft;
    expect(d?.kind).toBe('block_drafter');
    if (d?.kind !== 'block_drafter') return;
    expect([d.method, d.layers, d.blockSize, d.markovRank, d.targetLayerIds]).toEqual([
      'dspark',
      3,
      6,
      256,
      [40, 41, 42],
    ]);
    for (const layer of d.moeLayers ?? []) {
      expect(layer.blocks.find((x) => x.kind === 'moe')).toMatchObject({ experts: 256, topK: 6 });
      expect(csaOf(layer).global).toBeUndefined();
    }
  });

  it('hints the native (QAT) cache formats', () => {
    // fp8 non-RoPE dims + one ue8m0 scale per 64 + bf16 RoPE dims: (64·8 + 7·8)/512 = 1.109375 extra bits.
    const kv = { dtype: 'fp8_e4m3', extraBitsPerElem: 1.109375 };
    expect(b.quantHints).toEqual({
      kvCache: kv,
      compressedKvCache: kv,
      indexerKeyCache: { dtype: 'mxfp4' },
    });
    expect(512 * bytesPerElement({ dtype: 'fp8_e4m3', extraBitsPerElem: 1.109375 })).toBe(583);
  });
});

describe('DeepSeek-V4-Flash-0731 decode @ 8k on 1 × B200 (mapped default quant, tp 1, batch 1)', () => {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(JSON.stringify(r));
  const q = r.quant.quant;
  const sim = simulate({
    model: r.model,
    quant: q,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: { device: B200, topology: singleDevice(B200) },
  });
  const ops = sim.decode.stages.flatMap((s) => s.layers.flatMap((l) => l.ops));
  const hcFn = ops.filter((o) => o.op.name.endsWith('hc_pre.hc.fn'));

  // Hand bandwidth bound (bytes one decode step reads, batch 1, the top-6 experts per layer):
  // - routed experts 43·6·(3·4096·2048) = 43·6·25,165,824 at mxfp4 (4.25 bits)          ≈ 3.449 GB
  // - fp8 (128×128 ue8m0): per layer wq_a, wq_b, wkv, wo_a, wo_b + shared expert 132,120,576 · 43
  //   + 21 indexer wq_b 8,388,608 = 5,857,345,536                                     ≈ 5.858 GB
  // - bf16: head 529,530,880 + routers 43·1,048,576 + compressors 21·8,388,608 + 20·4,194,304
  //   + indexer key compressors 21·2,097,152 + weights_proj 21·262,144 = 884,211,712 · 2 B  ≈ 1.768 GB
  // - fp32 mHC fn weights 2·43·24·16,384 = 33,816,576 · 4 B                              ≈ 0.135 GB
  // - KV read: 43 windows of 128 + 21·512 top-k entries + 20·64 entries (8193/128) at 583 B, and 21·2048
  //   indexer keys (8193/4) at 68 B                                                        ≈ 13 MB
  // Σ ≈ 11.22 GB → 7.7e12 / 11.22e9 ≈ 686 tok/s @ 8k.
  const fmt = (m: 'routed_experts' | 'attention') => q.modules?.[m]?.weights ?? q.weights;
  const weightBytes =
    43 * 6 * 25_165_824 * bytesPerElement(fmt('routed_experts')) +
    5_857_345_536 * bytesPerElement(fmt('attention')) +
    884_211_712 * 2 +
    33_816_576 * 4;
  const kvBytes =
    (43 * 128 + 21 * 512 + 20 * Math.floor(8193 / 128)) * 583 + 21 * Math.ceil(8193 / 4) * 68;
  const boundTokPerSec = B200.memory.bandwidthBytesPerSec / (weightBytes + kvBytes);

  it('fits: weights ≈ 156 GB < 180 GB', () => {
    expect(sim.memory.fits).toBe(true);
    expect(sim.memory.perStage[0].weightBytes).toBeLessThan(160e9);
    expect(boundTokPerSec).toBeGreaterThan(680);
    expect(boundTokPerSec).toBeLessThan(690);
  });

  it('decode stays under the bandwidth bound; everything but the mHC fn GEMMs is within 0.9× of it', () => {
    // The sim gives ≈ 138 tok/s @ 8k against the ≈ 683-686 tok/s @ 8k bound. The gap is the mHC `hc_pre.hc.fn`
    // GEMMs (fp32, m 1, n 24, k 16,384: 786 kFLOP each): B200 has no fp32 tensor peak, so they run on the vector
    // fp32 peak (75 TFLOP/s) with the GEMM tile/wave efficiency (≈ 1.58e-4) → ≈ 66 µs each, 2 per layer: 5.69 ms of
    // the 7.22 ms step. DeepSeek-V4.1-Flash on 1 × B200 shows the same (engine behaviour, not this family).
    // Replacing their time with their memory time, the rest of the step runs at 0.9-1.0× the hand bound.
    expect(sim.decode.tokensPerSec).toBeLessThanOrEqual(boundTokPerSec);
    expect(hcFn).toHaveLength(2 * 43);
    for (const o of hcFn) {
      expect(o.bound).toBe('compute');
      expect(o.peakFlopsPerSec).toBe(75e12);
    }
    const hcSec = hcFn.reduce((s, o) => s + o.sec, 0);
    expect(hcSec / sim.decode.latencySec).toBeGreaterThan(0.7);
    const restSec = sim.decode.latencySec - hcSec + hcFn.reduce((s, o) => s + o.memorySec, 0);
    const restTokPerSec = 1 / restSec;
    expect(restTokPerSec).toBeLessThanOrEqual(boundTokPerSec);
    expect(restTokPerSec).toBeGreaterThanOrEqual(0.9 * boundTokPerSec);
  });
});
