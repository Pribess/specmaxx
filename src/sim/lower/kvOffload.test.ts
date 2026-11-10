/**
 * KV-cache tiering at the lowering level (docs/simulation-model.md §12 "KV cache off-device"): bytes per token, the
 * offloadable state, the fetch ops' tier byte accounting (full and top-k, decode and prefill, DCP), and the prefetch
 * share q.
 */
import { describe, expect, it } from 'vitest';
import type { GqaAttentionBlock, MlaAttentionBlock, QuantConfig } from '../ir/model';
import type { AttentionOp, Op, TransferOp } from '../ir/ops';
import { TINY_CSA, TINY_HYBRID, TINY_MLA_SPARSE } from '../models/reference';
import { DEFAULT_PREFETCH_ACCURACY } from '../engine/transfer';
import {
  kvBytesPerToken,
  kvFetchOps,
  kvOffloadableBlock,
  kvOffloadableStateBytes,
  modelHasCsaBlock,
  modelHasOffloadableKv,
  modelHasTopKKv,
} from './kvOffload';
import {
  blockSequenceStateBytes,
  lowerBlock,
  type KvCachePlacement,
  type LoweringContext,
} from './lower';
import { prefetchedFraction } from './offload';

function expectClose(actual: number, expected: number, rel = 1e-12): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

/** Qwen3-8B-shaped full-attention GQA block: 32 heads, 8 KV heads, hd 128, h 4096. kvB = 2·8·128·2 = 4096 B. */
const GQA: GqaAttentionBlock = {
  kind: 'gqa_attention',
  name: 'attn',
  hidden: 4096,
  heads: 32,
  kvHeads: 8,
  headDim: 128,
  qkvBias: false,
  outBias: false,
  qkNorm: true,
  rope: true,
  causal: true,
  outputGate: false,
};
const GQA_KVB = 2 * 8 * 128 * 2;

/** Kimi-K2-shaped MLA block, with a top-2048 DSA indexer. kvB = (512 + 64)·2 = 1152 B. */
const MLA_TOPK: MlaAttentionBlock = {
  kind: 'mla_attention',
  name: 'attn',
  hidden: 7168,
  heads: 64,
  qLoraRank: 1536,
  kvLoraRank: 512,
  qkNopeDim: 128,
  qkRopeDim: 64,
  vHeadDim: 128,
  sparse: { kind: 'indexer', indexer: { heads: 64, headDim: 128, topK: 2048 } },
};
const MLA_KVB = (512 + 64) * 2;

function placement(over: Partial<KvCachePlacement> = {}): KvCachePlacement {
  return {
    deviceFraction: 0.25,
    ramShare: 1,
    nvmeShare: 0,
    remotePoolShare: 0,
    hitRatio: 0.25,
    prefixDeviceFraction: 0.25,
    ...over,
  };
}

/** Decode @ N (qLen 1, kvLen N + 1), b sequences, with a KV placement and prefetch accuracy p. */
function decode(
  N: number,
  b: number,
  kv: KvCachePlacement | undefined,
  p = 0,
  extra: Partial<LoweringContext> = {},
): LoweringContext {
  return {
    phase: 'decode',
    seqs: b,
    qLen: 1,
    kvLen: N + 1,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    ...(kv === undefined ? {} : { offload: { prefetchAccuracy: p, kvCache: kv } }),
    ...extra,
  };
}

const fetches = (ops: Op[]): TransferOp[] =>
  ops.flatMap((o) => (o.kind === 'transfer' && o.group === 'kv_cache' ? [o] : []));
const attentionOf = (ops: Op[]): AttentionOp => {
  const a = ops.find((o): o is AttentionOp => o.kind === 'attention');
  if (a === undefined) throw new Error('no attention op');
  return a;
};

describe('kvBytesPerToken and the offloadable state', () => {
  it('GQA 2·KVl·hd·bpe = 4096 B (tp 1), 2048 B at tp 2; MLA (rkv + dr)·bpe = 1152 B', () => {
    expect(kvBytesPerToken(GQA, decode(1, 1, undefined))).toBe(GQA_KVB);
    expect(
      kvBytesPerToken(
        GQA,
        decode(1, 1, undefined, 0, { parallel: { tp: 2, pp: 1, dp: 1, ep: 1 } }),
      ),
    ).toBe(GQA_KVB / 2);
    expect(kvBytesPerToken(MLA_TOPK, decode(1, 1, undefined))).toBe(MLA_KVB);
  });

  it('offloadable state = N·kvB for full GQA and MLA (indexer keys excluded); 0 for a sliding window', () => {
    const ctx = decode(1, 1, undefined);
    expect(kvOffloadableStateBytes(GQA, ctx, 8192)).toBe(8192 * GQA_KVB);
    expect(blockSequenceStateBytes(GQA, ctx, 8192)).toBe(8192 * GQA_KVB);
    // MLA with an indexer: the state also holds the indexer keys (128·2 B per token), which stay on the device.
    expect(kvOffloadableStateBytes(MLA_TOPK, ctx, 8192)).toBe(8192 * MLA_KVB);
    expect(blockSequenceStateBytes(MLA_TOPK, ctx, 8192)).toBe(8192 * (MLA_KVB + 128 * 2));
    const windowed: GqaAttentionBlock = { ...GQA, slidingWindow: 1024 };
    expect(kvOffloadableBlock(windowed)).toBe(false);
    expect(kvOffloadableStateBytes(windowed, ctx, 8192)).toBe(0);
    expect(kvOffloadableStateBytes({ ...MLA_TOPK, kvCacheFromLayer: 0 }, ctx, 8192)).toBe(0);
  });

  it('DCP d = 4: this rank stores ceil(ctx/4) tokens', () => {
    const ctx = decode(1, 1, undefined, 0, { parallel: { tp: 4, pp: 1, dp: 1, ep: 1, dcp: 4 } });
    const MLA_FULL: MlaAttentionBlock = { ...MLA_TOPK, sparse: undefined };
    expect(kvOffloadableStateBytes(MLA_FULL, ctx, 8193)).toBe(Math.ceil(8193 / 4) * MLA_KVB);
  });

  it('model helpers: offloadable / top-k / CSA', () => {
    expect(modelHasOffloadableKv(TINY_HYBRID)).toBe(true);
    expect(modelHasTopKKv(TINY_HYBRID)).toBe(false);
    expect(modelHasTopKKv(TINY_MLA_SPARSE)).toBe(true);
    expect(modelHasCsaBlock(TINY_CSA)).toBe(true);
    expect(modelHasOffloadableKv(TINY_CSA)).toBe(false);
  });
});

describe('kvFetchOps: tier byte accounting', () => {
  it('full GQA decode, f 0.25, b 4, N 8192: one span per sequence, 4·0.75·8192·kvB bytes from host RAM', () => {
    const ops = lowerBlock(GQA, decode(8192, 4, placement()));
    const [op, ...rest] = fetches(ops);
    expect(rest).toHaveLength(0);
    // Formula: M = past − f·past = 0.75·8192 = 6144 tokens per sequence; reads [{ b, M·kvB }]
    expect(op).toEqual({
      kind: 'transfer',
      name: 'attn.kv_fetch',
      group: 'kv_cache',
      source: 'host_ram',
      reads: [{ count: 4, bytesEach: 6144 * GQA_KVB }],
      usefulBytes: 4 * 0.75 * 8192 * GQA_KVB,
      prefetchedFraction: 0,
    });
    // Placed right before the attention core.
    const i = ops.indexOf(op);
    expect(ops[i + 1].kind).toBe('attention');
  });

  it('the attention op is unchanged by the placement (device memory bytes identical)', () => {
    const on = attentionOf(lowerBlock(GQA, decode(8192, 4, undefined)));
    const off = attentionOf(lowerBlock(GQA, decode(8192, 4, placement())));
    expect(off).toEqual(on);
    expect(
      lowerBlock(GQA, decode(8192, 4, undefined)).filter((o) => o.kind === 'transfer'),
    ).toEqual([]);
  });

  it('RAM/SSD split 0.4 : 0.6 sums to the whole off-device part', () => {
    const whole = fetches(lowerBlock(GQA, decode(8192, 4, placement())))[0].usefulBytes;
    const split = fetches(
      lowerBlock(GQA, decode(8192, 4, placement({ ramShare: 0.4, nvmeShare: 0.6 }))),
    );
    expect(split.map((o) => [o.name, o.source])).toEqual([
      ['attn.kv_fetch', 'host_ram'],
      ['attn.kv_fetch_ssd', 'nvme'],
    ]);
    expectClose(split[0].usefulBytes, 0.4 * whole);
    expectClose(split[1].usefulBytes, 0.6 * whole);
    expectClose(split[0].usefulBytes + split[1].usefulBytes, whole);
  });

  it('RAM/SSD/pool split 0.2 : 0.3 : 0.5: a third op reads the remote pool, the three sum to the whole', () => {
    const whole = fetches(lowerBlock(GQA, decode(8192, 4, placement())))[0].usefulBytes;
    const split = fetches(
      lowerBlock(
        GQA,
        decode(8192, 4, placement({ ramShare: 0.2, nvmeShare: 0.3, remotePoolShare: 0.5 })),
      ),
    );
    expect(split.map((o) => [o.name, o.source])).toEqual([
      ['attn.kv_fetch', 'host_ram'],
      ['attn.kv_fetch_ssd', 'nvme'],
      ['attn.kv_fetch_pool', 'remote_pool'],
    ]);
    expectClose(split[2].usefulBytes, 0.5 * whole);
    // Formula: M_pool = 0.5·6144 = 3072 tokens per sequence; reads [{ b, M_pool·kvB }]
    expect(split[2].reads).toEqual([{ count: 4, bytesEach: 0.5 * 6144 * GQA_KVB }]);
    expect(split[2].prefetchedFraction).toBe(0);
    expectClose(split[0].usefulBytes + split[1].usefulBytes + split[2].usefulBytes, whole);
  });

  it('the pool alone (remotePoolShare 1): one op per layer, scattered rows for top-k decode', () => {
    const only = placement({ ramShare: 0, nvmeShare: 0, remotePoolShare: 1 });
    const [full, ...rest] = fetches(lowerBlock(GQA, decode(8192, 4, only)));
    expect(rest).toHaveLength(0);
    expect(full).toMatchObject({
      name: 'attn.kv_fetch_pool',
      source: 'remote_pool',
      reads: [{ count: 4, bytesEach: 6144 * GQA_KVB }],
      usefulBytes: 4 * 6144 * GQA_KVB,
    });
    // Top-k MLA: N 32768, f 0.25, U 2048, M 1536 per sequence (as the host-RAM case), one row per token.
    const [topk] = fetches(lowerBlock(MLA_TOPK, decode(32768, 2, only)));
    expect(topk.source).toBe('remote_pool');
    expect(topk.reads).toEqual([{ count: 2 * 1536, bytesEach: MLA_KVB }]);
    expect(topk.usefulBytes).toBe(2 * 1536 * MLA_KVB);
  });

  it('f = 1 or no past tokens: no op', () => {
    expect(fetches(lowerBlock(GQA, decode(8192, 4, placement({ deviceFraction: 1 }))))).toEqual([]);
    const prefillNoHit: LoweringContext = {
      ...decode(0, 1, placement()),
      phase: 'prefill',
      qLen: 4096,
      kvLen: 4096,
    };
    expect(fetches(lowerBlock(GQA, prefillNoHit))).toEqual([]);
  });

  it('a sliding-window block emits no fetch', () => {
    expect(
      fetches(lowerBlock({ ...GQA, slidingWindow: 1024 }, decode(8192, 4, placement()))),
    ).toEqual([]);
  });

  it('top-k MLA decode: U = topK, uniform h = f, scattered rows (count b·M, kvB each)', () => {
    // N 32768, f 0.25: N_res = 8192; U = 2048; hits = clamp(2048·0.25, max(0, 2048 + 8192 − 32768), min(2048, 8192))
    //   = 512; M = 1536 per sequence.
    const [op] = fetches(lowerBlock(MLA_TOPK, decode(32768, 2, placement())));
    expect(op.reads).toEqual([{ count: 2 * 1536, bytesEach: MLA_KVB }]);
    expect(op.usefulBytes).toBe(2 * 1536 * MLA_KVB);
  });

  it('top-k: a custom h is clamped to the physical bounds', () => {
    // N 4096, f 0.75: N_res = 3072; U = 2048; lo = max(0, 2048 + 3072 − 4096) = 1024, hi = 2048.
    // h 0.1 → U·h = 204.8 < lo → hits = 1024, M = 1024.
    const low = fetches(
      lowerBlock(MLA_TOPK, decode(4096, 1, placement({ deviceFraction: 0.75, hitRatio: 0.1 }))),
    );
    expect(low[0].usefulBytes).toBe(1024 * MLA_KVB);
    // h 0.9 → hits = 1843.2, M = 204.8 (inside the bounds).
    const high = fetches(
      lowerBlock(MLA_TOPK, decode(4096, 1, placement({ deviceFraction: 0.75, hitRatio: 0.9 }))),
    );
    expectClose(high[0].usefulBytes, (2048 - 2048 * 0.9) * MLA_KVB);
    // Uniform h = f 0.75: M = 2048·0.25 = 512.
    const uniform = fetches(
      lowerBlock(MLA_TOPK, decode(4096, 1, placement({ deviceFraction: 0.75, hitRatio: 0.75 }))),
    );
    expect(uniform[0].usefulBytes).toBe(512 * MLA_KVB);
  });

  it('DCP d 4 (MLA, decode): this rank fetches its ceil(N/4) share, before the Q all-gather', () => {
    const MLA_FULL: MlaAttentionBlock = { ...MLA_TOPK, sparse: undefined };
    const ctx = decode(8192, 2, placement(), 0, {
      parallel: { tp: 4, pp: 1, dp: 1, ep: 1, dcp: 4 },
    });
    const ops = lowerBlock(MLA_FULL, ctx);
    const [op] = fetches(ops);
    // past = 8192 → per rank 2048, M = 0.75·2048 = 1536 per sequence.
    expect(op.reads).toEqual([{ count: 2, bytesEach: 1536 * MLA_KVB }]);
    expect(ops[ops.indexOf(op) + 1]).toMatchObject({ kind: 'collective', group: 'dcp' });
  });

  it('prefill with a prefix-cache hit: (1 − f_pre)·floor(P·h_pc) tokens per sequence, a span each, q = 1', () => {
    // P 8192, h_pc 0.5 → cached 4096; f_pre 0.5 → 2048 missed tokens per sequence.
    const ctx: LoweringContext = {
      ...decode(0, 3, placement({ prefixDeviceFraction: 0.5 }), 0.8),
      phase: 'prefill',
      qLen: 4096,
      kvLen: 8192,
    };
    const [op] = fetches(lowerBlock(GQA, ctx));
    expect(op.reads).toEqual([{ count: 3, bytesEach: 2048 * GQA_KVB }]);
    expect(op.prefetchedFraction).toBe(1);
    // MLA dense prefill (MHA form): the fetch comes before kv_b, which reads every latent.
    const MLA_FULL: MlaAttentionBlock = { ...MLA_TOPK, sparse: undefined };
    const mlaOps = lowerBlock(MLA_FULL, ctx);
    const mlaFetch = fetches(mlaOps)[0];
    expect(mlaFetch.usefulBytes).toBe(3 * 2048 * MLA_KVB);
    expect(mlaOps[mlaOps.indexOf(mlaFetch) + 1]).toMatchObject({ name: 'mla.kv_b' });
  });

  it('an SSD top-k read is one row per token (the tier then pays its 4 KiB granularity per read)', () => {
    const [op] = fetches(
      lowerBlock(MLA_TOPK, decode(32768, 1, placement({ ramShare: 0, nvmeShare: 1 }))),
    );
    expect(op).toMatchObject({ source: 'nvme', name: 'attn.kv_fetch_ssd' });
    expect(op.reads[0].bytesEach).toBe(MLA_KVB);
    expect(op.reads[0].bytesEach).toBeLessThan(4096);
  });
});

describe('prefetchedFraction for the KV cache', () => {
  it('p = 0 → 0; full / prefill → 1; top-k → max(p, M/(past − N_res))', () => {
    expect(
      prefetchedFraction(0, { group: 'kv_cache', topK: false, misses: 10, nonResidentTokens: 10 }),
    ).toBe(0);
    expect(
      prefetchedFraction(0.3, { group: 'kv_cache', topK: false, misses: 1, nonResidentTokens: 10 }),
    ).toBe(1);
    expect(
      prefetchedFraction(0.3, { group: 'kv_cache', topK: true, misses: 1, nonResidentTokens: 10 }),
    ).toBe(0.3);
    expect(
      prefetchedFraction(0.3, { group: 'kv_cache', topK: true, misses: 8, nonResidentTokens: 10 }),
    ).toBe(0.8);
  });

  it('full decode op at the default p gets q = 1; top-k decode at batch 1 gets q = p', () => {
    const p = DEFAULT_PREFETCH_ACCURACY;
    expect(fetches(lowerBlock(GQA, decode(8192, 1, placement(), p)))[0].prefetchedFraction).toBe(1);
    // M = 1536 of 24576 non-resident tokens: 0.0625 < p.
    expect(
      fetches(lowerBlock(MLA_TOPK, decode(32768, 1, placement(), p)))[0].prefetchedFraction,
    ).toBe(p);
  });
});

it('kvFetchOps with no placement is empty', () => {
  const ctx = decode(8192, 1, undefined);
  expect(kvFetchOps(GQA, ctx, attentionOf(lowerBlock(GQA, ctx)))).toEqual([]);
});
