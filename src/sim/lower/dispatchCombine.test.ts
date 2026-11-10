/**
 * MoE dispatch / combine precision (`ParallelConfig.dispatchFormat` / `combineFormat`, docs §5 "Dispatch and combine
 * precision"): bytes per element and per token for each wire format, block scales included, and the identity at the
 * defaults (absent = the activation dtype for dispatch, `max(bpe(act), 2)` for combine).
 */
import { describe, expect, it } from 'vitest';
import {
  A2A_FORMATS,
  A2A_TENSOR_FORMATS,
  a2aBytesPerElement,
  a2aWireDtype,
  isA2aFormat,
  type A2aFormat,
} from '../ir/a2aFormat';
import { bytesPerElement, type Dtype } from '../ir/dtype';
import type { MoeBlock, ParallelConfig } from '../ir/model';
import type { CollectiveOp, Op } from '../ir/ops';
import type { LoweringContext } from './lower';
import { combineBpe, combineDtype, dispatchBpe, dispatchDtype, lowerMoe } from './moe';

/** DeepSeek-V3's hidden width h: the unit of the per-token byte counts below. */
const H = 7168;
const BLOCK: MoeBlock = {
  kind: 'moe',
  name: 'moe',
  hidden: H,
  experts: 256,
  topK: 8,
  expertIntermediate: 2048,
  gated: true,
  activation: 'silu',
  sharedExperts: 1,
  sharedIntermediate: 2048,
};

function ctx(parallel: Partial<ParallelConfig>, activations: Dtype = 'bf16'): LoweringContext {
  return {
    phase: 'decode',
    seqs: 8,
    qLen: 1,
    kvLen: 100,
    quant: { weights: { dtype: 'bf16' }, activations, kvCache: { dtype: 'bf16' } },
    // mode B: tp 1, dp = ep = 8 (SGLang / vLLM DEP)
    parallel: { tp: 1, pp: 1, dp: 8, ep: 8, ...parallel },
  };
}

const a2a = (ops: Op[]): CollectiveOp[] =>
  ops.filter((o): o is CollectiveOp => o.kind === 'collective' && o.collective === 'all_to_all');

describe('wire format bytes per element (block scales included)', () => {
  it('bits: bf16 16, fp8 8, fp8 + fp32/128 8.25, mxfp8 8.25, nvfp4 4.5, mxfp4 4.25', () => {
    const bits = Object.fromEntries(A2A_FORMATS.map((f) => [f, a2aBytesPerElement(f) * 8]));
    expect(bits).toEqual({
      bf16: 16,
      fp8_e4m3: 8,
      fp8_block128: 8.25,
      mxfp8: 8.25,
      nvfp4: 4.5,
      mxfp4: 4.25,
    });
  });

  it('is the shared helper: bytesPerElement of the storage format', () => {
    for (const f of A2A_FORMATS)
      expect(a2aBytesPerElement(f)).toBe(bytesPerElement(A2A_TENSOR_FORMATS[f]));
  });

  it('wire dtype tags and the type guard', () => {
    expect(A2A_FORMATS.map(a2aWireDtype)).toEqual([
      'bf16',
      'fp8_e4m3',
      'fp8_e4m3',
      'mxfp8',
      'nvfp4',
      'mxfp4',
    ]);
    expect(A2A_FORMATS.every(isA2aFormat)).toBe(true);
    expect([isA2aFormat('fp8'), isA2aFormat(undefined), isA2aFormat(3)]).toEqual([
      false,
      false,
      false,
    ]);
  });
});

describe('bytes per token at h = 7168 (one routed copy)', () => {
  const perToken = (f: A2aFormat) => H * a2aBytesPerElement(f);

  it("FP8: 7168 B plain; 7392 B with DeepEP's fp32 scale per 128 values (56 scales = 224 B)", () => {
    expect(perToken('fp8_e4m3')).toBe(7168);
    // Formula: 7168 + 7168/128 · 4 B = 7168 + 224
    expect(perToken('fp8_block128')).toBe(7392);
  });

  it('NVFP4: 4032 B = 3584 B of e2m1 values + 448 B of e4m3 scales (one per 16 values)', () => {
    expect(perToken('nvfp4')).toBe(4032);
    expect(perToken('mxfp4')).toBe(3808); // 3584 + 224 (one e8m0 per 32)
    expect(perToken('bf16')).toBe(14336);
  });

  it('AMD MoRI: FP4 (NVFP4-sized) dispatch + FP8 combine = 11,200 B per token', () => {
    // https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html quotes 11,200 B/token
    expect(perToken('nvfp4') + perToken('fp8_e4m3')).toBe(11_200);
  });
});

describe('lowering: dispatch / combine bytes follow the formats', () => {
  // T 8, ep 8: Ts = 8, A = 64 rows per all-to-all; ρ off the all-to-all (default), so R_a2a = Ts·k = 64
  const ROWS = 8 * 8;
  const bytes = (p: Partial<ParallelConfig>, act: Dtype = 'bf16') =>
    a2a(lowerMoe(BLOCK, ctx(p, act))).map((c) => c.messageBytes);

  it('defaults: dispatch = rows·h·bpe(act), combine = rows·h·max(bpe(act), 2)', () => {
    expect(bytes({})).toEqual([ROWS * H * 2, ROWS * H * 2]); // bf16
    expect(bytes({}, 'fp8_e4m3')).toEqual([ROWS * H * 1, ROWS * H * 2]); // fp8 act: dispatch 1 B, combine floored at 2 B
    expect(bytes({}, 'nvfp4')).toEqual([ROWS * H * 0.5625, ROWS * H * 2]);
  });

  it('absent = the equivalent explicit format: bf16, and (fp8 act) fp8 plain', () => {
    expect(bytes({ dispatchFormat: 'bf16', combineFormat: 'bf16' })).toEqual(bytes({}));
    expect(bytes({ dispatchFormat: 'fp8_e4m3' }, 'fp8_e4m3')).toEqual(bytes({}, 'fp8_e4m3'));
  });

  it('FP8 dispatch with BF16 combine (DeepSeek-V3): dispatch halves (+3% scales), combine unchanged', () => {
    expect(bytes({ dispatchFormat: 'fp8_block128', combineFormat: 'bf16' })).toEqual([
      ROWS * 7392,
      ROWS * 14336,
    ]);
  });

  it('NVFP4 dispatch (GB200): 4032 B per row, 3.56× less than bf16', () => {
    const [dispatch, combine] = bytes({ dispatchFormat: 'nvfp4' });
    expect(dispatch).toBe(ROWS * 4032);
    expect(combine).toBe(ROWS * 14336);
    expect((ROWS * 14336) / dispatch).toBeCloseTo(3.5556, 3);
  });

  it('FP4 dispatch + FP8 combine (MoRI): 4032 + 7168 B per row', () => {
    expect(bytes({ dispatchFormat: 'nvfp4', combineFormat: 'fp8_e4m3' })).toEqual([
      ROWS * 4032,
      ROWS * 7168,
    ]);
  });

  it('the formats scale with the latent width l, and with ρ on the all-to-all (A_max rows)', () => {
    const latent: MoeBlock = { ...BLOCK, latentDim: 1024 };
    const c = a2a(lowerMoe(latent, ctx({ dispatchFormat: 'nvfp4', combineFormat: 'fp8_e4m3' })));
    expect(c.map((x) => x.messageBytes)).toEqual([ROWS * 1024 * 0.5625, ROWS * 1024]);
    // ρ on the all-to-all: Ts 8, k 8, ep 8 → mean share 64 rows; A_max = ceil(1.2·64) = 77 rows instead of 64
    const scaled = a2a(
      lowerMoe(BLOCK, ctx({ dispatchFormat: 'nvfp4', epImbalanceScalesAllToAll: true })),
    );
    expect(scaled[0].messageBytes).toBe(77 * H * 0.5625);
  });

  it('wire dtype tags follow the format; the defaults keep the activation / reduce dtype', () => {
    expect(a2a(lowerMoe(BLOCK, ctx({}))).map((c) => c.dtype)).toEqual(['bf16', 'bf16']);
    expect(a2a(lowerMoe(BLOCK, ctx({}, 'fp8_e4m3'))).map((c) => c.dtype)).toEqual([
      'fp8_e4m3',
      'bf16',
    ]);
    expect(
      a2a(lowerMoe(BLOCK, ctx({ dispatchFormat: 'nvfp4', combineFormat: 'mxfp8' }))).map(
        (c) => c.dtype,
      ),
    ).toEqual(['nvfp4', 'mxfp8']);
  });

  it('only the two all-to-alls change; every other op is identical', () => {
    const strip = (ops: Op[]) =>
      ops.filter((o) => !(o.kind === 'collective' && o.collective === 'all_to_all'));
    expect(
      strip(lowerMoe(BLOCK, ctx({ dispatchFormat: 'nvfp4', combineFormat: 'fp8_e4m3' }))),
    ).toEqual(strip(lowerMoe(BLOCK, ctx({}))));
  });

  it('mode A (no all-to-all): the formats change nothing', () => {
    const modeA = (p: Partial<ParallelConfig>) =>
      lowerMoe(BLOCK, ctx({ tp: 8, dp: 1, ep: 8, ...p }));
    expect(modeA({ dispatchFormat: 'nvfp4', combineFormat: 'fp8_e4m3' })).toEqual(modeA({}));
  });

  it('bpe helpers: default = actBpe / reduceBpe', () => {
    expect([dispatchBpe(ctx({})), combineBpe(ctx({}))]).toEqual([2, 2]);
    expect([dispatchBpe(ctx({}, 'fp8_e4m3')), combineBpe(ctx({}, 'fp8_e4m3'))]).toEqual([1, 2]);
    expect([dispatchDtype(ctx({}, 'fp8_e4m3')), combineDtype(ctx({}, 'fp8_e4m3'))]).toEqual([
      'fp8_e4m3',
      'bf16',
    ]);
    expect([
      dispatchBpe(ctx({ dispatchFormat: 'nvfp4' })),
      combineBpe(ctx({ combineFormat: 'mxfp4' })),
    ]).toEqual([0.5625, 0.53125]);
  });
});
