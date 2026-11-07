import { describe, expect, it } from 'vitest';
import { DTYPE_INFO, bitsPerElement, bytesPerElement, mathFormatOf, resolveAccumulate, resolveGemmMath } from './dtype';

describe('DTYPE_INFO', () => {
  it('stores element bits and block-scale overhead', () => {
    expect(DTYPE_INFO.bf16.elementBits).toBe(16);
    expect(DTYPE_INFO.tf32.elementBits).toBe(32); // tf32 is stored as a full fp32 word
    expect(DTYPE_INFO.nvfp4.blockScaleBitsPerElem).toBe(0.5); // e4m3 scale (8 bit) per 16 elements
    expect(DTYPE_INFO.mxfp4.blockScaleBitsPerElem).toBe(0.25); // e8m0 scale (8 bit) per 32 elements
    expect(DTYPE_INFO.int32.math).toBeNull();
  });
});

describe('bytesPerElement', () => {
  it('matches the T1 derivations', () => {
    expect(bytesPerElement({ dtype: 'bf16' })).toBe(2); // 16 / 8
    expect(bytesPerElement({ dtype: 'fp8_e4m3' })).toBe(1); // 8 / 8
    expect(bytesPerElement({ dtype: 'nvfp4' })).toBe(0.5625); // (4 + 0.5) / 8
    expect(bytesPerElement({ dtype: 'mxfp4' })).toBe(0.53125); // (4 + 0.25) / 8
    // GPTQ int4 g128: (16-bit scale + 4-bit zero) / 128 = 0.15625 bit/elem → (4 + 0.15625) / 8
    expect(bytesPerElement({ dtype: 'int4', extraBitsPerElem: 0.15625 })).toBe(0.51953125);
    expect(bitsPerElement({ dtype: 'mxfp8' })).toBe(8.25); // 8 + 8/32
  });
});

describe('math and accumulate', () => {
  it('maps dtypes to math formats', () => {
    expect(mathFormatOf('fp8_e5m2')).toBe('fp8');
    expect(() => mathFormatOf('int32')).toThrow();
  });

  it.each([
    ['int4', 'bf16', 'bf16'],
    ['int8', 'int8', 'int8'],
    ['int4', 'fp8_e4m3', 'fp8'],
    ['nvfp4', 'nvfp4', 'fp4'],
    ['mxfp4', 'bf16', 'bf16'],
    ['bf16', 'fp16', 'fp16'], // tie: the activation wins
  ] as const)('resolveGemmMath(%s, %s) = %s', (weight, activation, expected) => {
    expect(resolveGemmMath(weight, activation)).toBe(expected);
  });

  it('resolves the accumulator', () => {
    expect(resolveAccumulate('bf16', 'fp16')).toBe('fp32'); // bf16 always accumulates in fp32
    expect(resolveAccumulate('int8', 'fp32')).toBe('int32');
    expect(resolveAccumulate('int8', 'fp16')).toBe('int32');
    expect(resolveAccumulate('fp16', 'fp16')).toBe('fp16');
    expect(resolveAccumulate('fp8', 'fp32')).toBe('fp32');
  });
});
