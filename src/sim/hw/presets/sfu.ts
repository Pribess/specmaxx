/**
 * Special-function-unit (SFU) exp2 rates of every preset family, in one file so each per-clock rate, its source and
 * its reasoning can be checked and adjusted in one place. Feeds `ComputeEngine.sfuOpsPerSec`, the softmax term of
 * attention (docs/simulation-model.md §4 "SFU rate", §7 "Attention softmax").
 *
 * Rule: `sfuOpsPerSec = compute units × exp2 results per clock per unit × clock`, with the SAME clock the preset's
 * vector FP32 peak uses (so a `clockScale` modifier moves both together). Prefer a vendor figure. Otherwise the
 * value is an estimate, and an estimate takes the UPPER end of the plausible range: the simulation is an upper
 * bound, and an SFU rate that is too low could push it below a real kernel.
 */

import type { EstimatedField, SourceRef } from '../types';

/** The exp2 rate of one family per compute unit per clock, with its provenance. */
export interface SfuRate {
  /** FP32 exp2 results per clock per compute unit (SM, CU, Xe core, Apple GPU core). Unit: results/(clock · unit). */
  exp2PerClockPerUnit: number;
  /** Citation of the per-clock rate (vendor or review). Absent = an estimate; `reason` then says why this value. */
  source?: Omit<SourceRef, 'fields' | 'note'>;
  /** The quote behind the rate, or the reasoning of an estimate. */
  reason: string;
}

/**
 * Device SFU rate from its per-unit rate: units × exp2 results/(clock · unit) × clock (Hz). Unit: results/s, per
 * device.
 */
export function sfuOpsPerSec(
  computeUnits: number,
  exp2PerClockPerUnit: number,
  clockHz: number,
): number {
  // Formula: sfuOpsPerSec = units · exp2/(clk·unit) · clock
  // See docs/simulation-model.md §4 ("SFU rate")
  return computeUnits * exp2PerClockPerUnit * clockHz;
}

/**
 * The clock a vector FP32 peak implies: `vectorFp32 / (units · FP32 FLOP/(clock · unit))`. Unit: Hz. Used where a
 * preset quotes its vector peak but no clock (NVIDIA datacenter), so the SFU runs at the vector peak's clock.
 */
export function vectorPeakClockHz(
  vectorFp32FlopsPerSec: number,
  computeUnits: number,
  fp32FlopPerClockPerUnit: number,
): number {
  // Formula: clock = vectorFp32 / (units · FLOP/(clk·unit))
  return vectorFp32FlopsPerSec / (computeUnits * fp32FlopPerClockPerUnit);
}

/** The SFU rate of one engine and the provenance of its `sfuOpsPerSec` field (spread into the device's lists). */
export interface EngineSfu {
  /** Device SFU rate. Unit: exp2 results/s. */
  sfuOpsPerSec: number;
  /** `sources[]` entries for the field (one when the per-clock rate is cited, else none). */
  sources: SourceRef[];
  /** `estimated[]` entries for the field (one when the per-clock rate is an estimate, else none). */
  estimated: EstimatedField[];
}

/**
 * The SFU rate of an engine with `computeUnits` units (named `unitName` in the note, e.g. `'SM'`) at `clockHz`, and
 * the provenance of `field` (e.g. `'engines.0.sfuOpsPerSec'`): a `sources[]` entry when the family's per-clock rate
 * is cited, else an `estimated[]` entry, each stating this device's arithmetic.
 */
export function engineSfu(
  rate: SfuRate,
  computeUnits: number,
  unitName: string,
  clockHz: number,
  field: string,
): EngineSfu {
  const value = sfuOpsPerSec(computeUnits, rate.exp2PerClockPerUnit, clockHz);
  const derivation =
    `${computeUnits} ${unitName} × ${rate.exp2PerClockPerUnit} exp2/clk × ${(clockHz / 1e9).toFixed(3)} GHz ` +
    `(the clock of the vector FP32 peak) = ${(value / 1e12).toFixed(2)}e12 exp2/s.`;
  const note = `${rate.reason} ${derivation}`;
  return rate.source !== undefined
    ? { sfuOpsPerSec: value, sources: [{ ...rate.source, fields: [field], note }], estimated: [] }
    : { sfuOpsPerSec: value, sources: [], estimated: [{ field, reason: note }] };
}

/**
 * {@link engineSfu} at the clock a vector FP32 peak implies, for presets that quote the vector peak but no clock
 * (NVIDIA): `clock = vectorFp32 / (units · fp32FlopPerClockPerUnit)`, so the SFU runs at the vector peak's clock.
 */
export function engineSfuAtVectorClock(
  rate: SfuRate,
  computeUnits: number,
  unitName: string,
  vectorFp32FlopsPerSec: number,
  fp32FlopPerClockPerUnit: number,
  field: string,
): EngineSfu {
  const clockHz = vectorPeakClockHz(vectorFp32FlopsPerSec, computeUnits, fp32FlopPerClockPerUnit);
  return engineSfu(rate, computeUnits, unitName, clockHz, field);
}

/** FP32 FLOP per clock per SM of every NVIDIA SM in the presets (Ampere to Blackwell): 128 FP32 lanes × 2 (FMA). */
const NVIDIA_FP32_FLOP_PER_CLOCK_PER_SM = 128 * 2;

/**
 * The SFU rate of an NVIDIA engine 0 with `sms` SMs and vector FP32 peak `vectorFp32FlopsPerSec`, at that peak's clock
 * (`vectorFp32 / (SMs · 256)`), with the provenance of `engines.0.sfuOpsPerSec`.
 */
export function nvidiaSfu(rate: SfuRate, sms: number, vectorFp32FlopsPerSec: number): EngineSfu {
  // Formula: clock = vectorFp32 / (SMs · 256 FLOP/clk);  sfuOpsPerSec = SMs · exp2/(clk·SM) · clock
  return engineSfuAtVectorClock(
    rate,
    sms,
    'SM',
    vectorFp32FlopsPerSec,
    NVIDIA_FP32_FLOP_PER_CLOCK_PER_SM,
    'engines.0.sfuOpsPerSec',
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Per-family rates
// ---------------------------------------------------------------------------------------------------------------

/** CUDA C++ Programming Guide 12.9.1, "Throughput of Native Arithmetic Instructions" (results per clock per SM). */
const CUDA_GUIDE_THROUGHPUT_URL =
  'https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions';
/** NVIDIA developer blog: Inside NVIDIA Blackwell Ultra (Table 2: SFU EX2 TeraExponentials/s per generation). */
const BLACKWELL_ULTRA_BLOG_URL =
  'https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/';
/** AMD "RDNA Architecture" public deck (GPUOpen), p. 9: transcendental unit. */
const AMD_RDNA_DECK_URL = 'https://gpuopen.com/download/RDNA_Architecture_public.pdf';

/** The CUDA guide's exp2f row, quoted by every sourced NVIDIA rate below. */
const CUDA_EXP2_ROW =
  'CUDA guide table row "32-bit floating-point reciprocal, reciprocal square root, base-2 logarithm (__log2f), base 2 ' +
  'exponential (exp2f), sine (__sinf), cosine (__cosf)": 16 results per clock per SM for compute capability 7.x–9.0, ' +
  '10.0 and 12.0 (4 MUFU lanes per SM sub-partition).';

/** NVIDIA SMs of Ampere (cc 8.6), Ada (8.9), Hopper (9.0), GeForce Blackwell (12.0): 16 MUFU exp2/clk per SM. */
export const NVIDIA_SM_SFU: SfuRate = {
  exp2PerClockPerUnit: 16,
  source: {
    url: CUDA_GUIDE_THROUGHPUT_URL,
    title: 'CUDA C++ Programming Guide 12.9.1: throughput of native arithmetic instructions',
  },
  reason:
    CUDA_EXP2_ROW +
    ' The Blackwell Ultra blog\'s Table 2 quotes Hopper "SFU EX2 4.5 TeraExponentials/s", a full-die figure (≈ 144 SMs); ' +
    'the per-SKU rate below is used instead.',
};

/** NVIDIA Blackwell datacenter (B200, cc 10.0): 16 exp2/clk per SM, the same as Hopper. */
export const NVIDIA_BLACKWELL_DC_SFU: SfuRate = {
  exp2PerClockPerUnit: 16,
  source: {
    url: CUDA_GUIDE_THROUGHPUT_URL,
    title: 'CUDA C++ Programming Guide 12.9.1: throughput of native arithmetic instructions',
  },
  reason:
    CUDA_EXP2_ROW +
    ' FlashAttention-4 (https://arxiv.org/html/2603.05451v1) agrees: "16 ops/clock/SM, same as Hopper". The Blackwell ' +
    'Ultra blog quotes Blackwell at 5 TeraExponentials/s, a full-die figure (≈ 160 SMs); the per-SKU rate is used.',
};

/** NVIDIA Blackwell Ultra (B300, cc 10.3): twice the B200's exp2 throughput per SM per clock. */
export const NVIDIA_BLACKWELL_ULTRA_SFU: SfuRate = {
  exp2PerClockPerUnit: 32,
  source: {
    url: BLACKWELL_ULTRA_BLOG_URL,
    title: 'NVIDIA developer blog: Inside NVIDIA Blackwell Ultra, Table 2',
  },
  reason:
    'Blackwell Ultra has "SFU throughput doubled" for attention (Table 2: 10.7 TeraExponentials/s, a full-die ' +
    'figure); FlashAttention-4 (https://arxiv.org/html/2603.05451v1) gives "32 ops/clock/SM". The per-SKU rate is used.',
};

/** NVIDIA GB10 (DGX Spark, cc 12.1): not in the CUDA table; the cc 12.0 rate of the same SM generation. Estimate. */
export const NVIDIA_GB10_SFU: SfuRate = {
  exp2PerClockPerUnit: 16,
  reason:
    `Estimate: compute capability 12.1 is not listed in the CUDA guide throughput table (${CUDA_GUIDE_THROUGHPUT_URL}); ` +
    'GB10 has the same SM generation as cc 12.0, which the table gives 16 exp2 results per clock per SM.',
};

/** AMD CDNA3 (MI300X, MI325X): transcendentals at a quarter of the 64-lane CU rate. Estimate from a sourced ratio. */
export const AMD_CDNA3_SFU: SfuRate = {
  exp2PerClockPerUnit: 16,
  reason:
    'Estimate: AMD publishes no CDNA3 transcendental rate. AMD\'s RDNA deck p. 9 says "Transcendental instructions are ' +
    `¼ rate (like GCN)" (${AMD_RDNA_DECK_URL}), and LLVM's AMDGPU SISchedule.td models transcendentals at quarter ` +
    'rate on GCN/CDNA: 64 lanes / 4 = 16 results per clock per CU. Packed FP32 does not apply to transcendentals.',
};

/** AMD CDNA4 (MI355X): twice the CDNA3 transcendental rate. */
export const AMD_CDNA4_SFU: SfuRate = {
  exp2PerClockPerUnit: 32,
  source: {
    url: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf',
    title: 'AMD CDNA 4 architecture whitepaper, p. 9',
  },
  reason:
    'CDNA 4: "transcendental rates increased by 2x" over CDNA 3, i.e. 2 × 16 = 32 results per clock per CU (the base ' +
    `rate of 16 is the quarter-rate estimate of the CDNA3 entry, ${AMD_RDNA_DECK_URL}).`,
};

/** AMD MI455X (CDNA5): no source; the upper-bound rule. Estimate. */
export const AMD_MI455X_SFU: SfuRate = {
  exp2PerClockPerUnit: 64,
  reason:
    'Estimate, no source: AMD publishes no CDNA5 transcendental rate, and the preset\'s 256 "compute units" may be WGPs ' +
    '(2 CUs each; the vector FP32 315 TFLOP/s at 2.4 GHz is 512 FLOP per clock per unit, 2× CDNA4). 64 per clock per ' +
    "unit keeps CDNA4's 1/8 ratio of exp2 to vector FP32 FLOP; 32 would put head-dim-64 attention exactly at the " +
    'softmax knee, so the upper end is taken.',
};

/** AMD RDNA 3.5 / RDNA4 (Strix Halo, R9700, RX 9070 XT): a quarter-rate transcendental unit per SIMD32. Estimate. */
export const AMD_RDNA_SFU: SfuRate = {
  exp2PerClockPerUnit: 16,
  reason:
    `Estimate: AMD's RDNA deck p. 9 (${AMD_RDNA_DECK_URL}) gives each SIMD32 a co-issued transcendental unit at ¼ ` +
    'rate: 32 lanes / 4 = 8 results per clock per SIMD, 2 SIMDs per CU = 16 per clock per CU. Not derived from the ' +
    'vector ratio: these presets count single-issue FP32.',
};

/** Intel Xe2 (Battlemage): an extended-math unit per vector engine. Estimate. */
export const INTEL_XE2_SFU: SfuRate = {
  exp2PerClockPerUnit: 32,
  reason:
    'Estimate, no source: Intel publishes no Xe2 extended-math rate. Each Xe core has 8 XVEs; the plausible range is ' +
    '16–32 exp2 per clock per Xe core (2–4 per XVE), and the upper end, 8 × 4 = 32, is taken.',
};

/** Apple GPU cores (M3, M4, M5): exp2 at a quarter of the 128 FP32 lanes. Estimate from a community measurement. */
export const APPLE_GPU_SFU: SfuRate = {
  exp2PerClockPerUnit: 32,
  reason:
    'Estimate: Apple publishes no rate. philipturner/metal-benchmarks (https://github.com/philipturner/metal-benchmarks) ' +
    'measures "F32 Exp2" at 32 per clock per core ("SFUs 32/core") on M1 / A15; the same is assumed for M3 and later. ' +
    'Apple attention is not SFU-bound for any rate above 1/16 of the lanes, so the result is insensitive to it.',
};
