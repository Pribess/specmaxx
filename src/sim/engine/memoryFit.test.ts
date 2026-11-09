/**
 * Memory fit: GPU-usable capacity on unified memory (review F9), the chunked-prefill activation peak and the runtime
 * reserve (review F11). See docs/simulation-model.md §8 and docs/review/practicality-review.md.
 */
import { describe, expect, it } from 'vitest';
import { SYNTH_UNIFIED } from '../__fixtures__/synthetic';
import { simpleDecoder, simulateReport, type SanityInput } from '../__sanity__/harness';
import { LLAMA3_70B } from '../__sanity__/models';
import { DEVICE_PRESETS, singleDevice } from '../hw/presets';
import { MACOS_GPU_USABLE_FRACTION } from '../hw/presets/apple';
import { RYZEN_AI_MAX_PLUS_395 } from '../hw/presets/amd';
import { DGX_SPARK } from '../hw/presets/nvidia-datacenter';
import { RTX_5090 } from '../hw/presets/nvidia';
import type { DeviceSpec } from '../hw/types';
import { validateDevice } from '../hw/validate';
import type { QuantConfig } from '../ir/model';
import { TINY_DENSE } from '../models/reference';
import {
  DEFAULT_PREFILL_CHUNK_TOKENS,
  prefillChunkShape,
  resolveGpuUsableFraction,
  unifiedMemoryWarnings,
} from './memory';
import { simulate } from './simulate';
import type { SimInput } from './types';
import { validateInput } from './validate';

const BF16: QuantConfig = {
  weights: { dtype: 'bf16' },
  activations: 'bf16',
  kvCache: { dtype: 'bf16' },
};

/** SYNTH_UNIFIED with a default GPU-usable fraction of 0.75. */
const UNIFIED_75: DeviceSpec = {
  ...SYNTH_UNIFIED,
  memory: { ...SYNTH_UNIFIED.memory, gpuUsableFraction: 0.75 },
};

function input(
  over: Partial<SimInput> = {},
  hardware: Partial<SimInput['hardware']> = {},
): SimInput {
  return {
    model: TINY_DENSE,
    quant: BF16,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 128, decodeContextLen: 256 },
    hardware: { device: UNIFIED_75, topology: singleDevice(UNIFIED_75), ...hardware },
    ...over,
  };
}

describe('prefillChunkShape', () => {
  it('a microbatch within the budget is one step of whole prompts (the rate lowering)', () => {
    expect(prefillChunkShape(1000, 8, 8192)).toEqual({ seqs: 8, qLen: 1000, kvLen: 1000 });
    expect(prefillChunkShape(8192, 1, 8192)).toEqual({ seqs: 1, qLen: 8192, kvLen: 8192 });
  });
  it('whole prompts up to the budget when they are shorter than it', () => {
    expect(prefillChunkShape(3000, 8, 8192)).toEqual({ seqs: 2, qLen: 3000, kvLen: 3000 });
  });
  it('a prompt longer than the budget: its last chunk, which sees the whole prompt', () => {
    expect(prefillChunkShape(131072, 4, 8192)).toEqual({ seqs: 1, qLen: 8192, kvLen: 131072 });
  });
});

describe('resolveGpuUsableFraction', () => {
  it('dedicated memory is always the whole device, override or not', () => {
    expect(resolveGpuUsableFraction({ kind: 'dedicated' }, 0.5)).toBe(1);
  });
  it('unified memory: override, else the device default, else 1', () => {
    expect(resolveGpuUsableFraction({ kind: 'unified', gpuUsableFraction: 0.75 }, 0.9)).toBe(0.9);
    expect(resolveGpuUsableFraction({ kind: 'unified', gpuUsableFraction: 0.75 }, undefined)).toBe(
      0.75,
    );
    expect(resolveGpuUsableFraction({ kind: 'unified' }, undefined)).toBe(1);
  });
});

describe('GPU-usable capacity (F9)', () => {
  it('the check uses f · capacity; the installed capacity is reported beside it', () => {
    const r = simulate(input({}, { memCapacityBytes: 64e9 }));
    expect(r.memory.gpuUsableFraction).toBe(0.75);
    expect(r.memory.perStage[0].capacityBytes).toBe(48e9);
    expect(r.memory.perStage[0].physicalCapacityBytes).toBe(64e9);
    expect(r.warnings).toContain(
      'memory: unified memory — the GPU may use 75% of it (48.0 GB of 64.0 GB); the rest stays with the OS and CPU',
    );
  });

  it('maxSequencesPerReplica is sized against the usable capacity', () => {
    const limited = simulate(input()).memory;
    const raised = simulate(input({}, { gpuUsableFraction: 1 })).memory;
    const s = limited.perStage[0];
    const perSeq = s.sequenceStateBytes; // batch 1
    expect(limited.maxSequencesPerReplica).toBe(
      Math.floor((0.75 * 32e9 - s.weightBytes - s.activationPeakBytes) / perSeq),
    );
    expect(raised.maxSequencesPerReplica).toBeGreaterThan(limited.maxSequencesPerReplica);
  });

  it('a stage above the limit but within the installed memory: does not fit, with the raised-limit warning', () => {
    // TINY_DENSE needs a few MB: shrink the capacity options so the weights land between 75% and 100%.
    const base = simulate(
      input({}, { device: SYNTH_UNIFIED, topology: singleDevice(SYNTH_UNIFIED) }),
    );
    const total = base.memory.perStage[0].totalBytes;
    const capacity = Math.ceil(total / 0.9);
    const device: DeviceSpec = {
      ...UNIFIED_75,
      memory: { ...UNIFIED_75.memory, capacityOptionsBytes: [capacity] },
    };
    const r = simulate(input({}, { device, topology: singleDevice(device) }));
    expect(r.memory.fits).toBe(false);
    expect(r.warnings.some((w) => w.includes('fits only with a raised limit'))).toBe(true);
    const raised = simulate(
      input({}, { device, topology: singleDevice(device), gpuUsableFraction: 0.95 }),
    );
    expect(raised.memory.fits).toBe(true);
    expect(unifiedMemoryWarnings(raised.memory)).toHaveLength(1);
    // The rates do not depend on the limit.
    expect(raised.decode.tokensPerSec).toBe(r.decode.tokensPerSec);
    expect(raised.prefill.tokensPerSec).toBe(r.prefill.tokensPerSec);
  });

  it('dedicated memory ignores the override', () => {
    const r = simulate(
      input({}, { device: RTX_5090, topology: singleDevice(RTX_5090), gpuUsableFraction: 0.5 }),
    );
    expect(r.memory.gpuUsableFraction).toBe(1);
    expect(r.memory.perStage[0].capacityBytes).toBe(r.memory.perStage[0].physicalCapacityBytes);
  });

  it('validation: the fraction is in (0, 1] and only on unified memory', () => {
    expect(validateInput(input({}, { gpuUsableFraction: 0 })).join()).toMatch(/gpuUsableFraction/);
    expect(validateInput(input({}, { gpuUsableFraction: 1.2 })).join()).toMatch(
      /gpuUsableFraction/,
    );
    const bad: DeviceSpec = { ...RTX_5090, memory: { ...RTX_5090.memory, gpuUsableFraction: 0.9 } };
    expect(validateDevice(bad).join()).toMatch(/unified memory only/);
  });

  it('presets: every Mac 0.75 (all have ≥ 36 GB), DGX Spark 119.7/128, Strix Halo 0.5 (Linux GTT default)', () => {
    const macs = DEVICE_PRESETS.filter((d) => d.vendor === 'Apple');
    expect(macs.length).toBeGreaterThan(0);
    for (const d of macs) {
      expect(d.memory.gpuUsableFraction).toBe(MACOS_GPU_USABLE_FRACTION);
      // The 3/4 rule holds from 36 GB; smaller Macs get about 2/3.
      expect(Math.min(...d.memory.capacityOptionsBytes)).toBeGreaterThanOrEqual(36e9);
    }
    expect(DGX_SPARK.memory.gpuUsableFraction).toBeCloseTo(0.935, 3);
    expect(RYZEN_AI_MAX_PLUS_395.memory.gpuUsableFraction).toBe(0.5);
    for (const d of DEVICE_PRESETS) {
      if (d.memory.kind === 'dedicated') expect(d.memory.gpuUsableFraction).toBeUndefined();
    }
  });

  it('review probe: Llama-3-70B MLX 8-bit on an M4 Max 128 GB @ 64k needs 96.5 GB: fits only with a raised limit', () => {
    const probe = (gpuUsableFraction?: number) => {
      const report: SanityInput = {
        model: { kind: 'spec', spec: LLAMA3_70B },
        quant: 'mlx-8bit',
        hardware: { deviceId: 'apple-m4-max-40c', memCapacityGb: 128 },
        workload: { promptLen: 512, genLen: 1, contextDepth: 65536 },
      };
      const r = simulateReport(report);
      if (gpuUsableFraction === undefined) return r.sim;
      return simulate({
        ...rInput(report),
        hardware: { ...rInput(report).hardware, gpuUsableFraction },
      });
    };
    const byDefault = probe();
    expect(byDefault.memory.perStage[0].totalBytes / 1e9).toBeCloseTo(96.5, 1);
    expect(byDefault.memory.perStage[0].capacityBytes).toBe(96e9);
    expect(byDefault.memory.fits).toBe(false);
    expect(probe(0.85).memory.fits).toBe(true);
  });
});

/** The SimInput simulateReport builds (re-derived from its result, which keeps the effective device). */
function rInput(report: SanityInput): SimInput {
  const r = simulateReport(report);
  return {
    model: LLAMA3_70B,
    quant: r.quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: {
      batch: 1,
      promptLen: report.workload.promptLen,
      decodeContextLen: r.decodeContextLen,
    },
    hardware: {
      device: r.sim.effectiveDevice,
      topology: singleDevice(r.sim.effectiveDevice),
      memCapacityBytes: 128e9,
    },
  };
}

describe('runtime reserve (F11)', () => {
  it('adds exactly the reserve to every stage total and takes it from the free bytes', () => {
    const dedicated = { device: RTX_5090, topology: singleDevice(RTX_5090) };
    const base = simulate(input({}, dedicated)).memory;
    const reserved = simulate(input({}, { ...dedicated, runtimeReserveBytes: 1e9 })).memory;
    const b = base.perStage[0];
    const s = reserved.perStage[0];
    expect(b.runtimeReserveBytes).toBe(0);
    expect(s.runtimeReserveBytes).toBe(1e9);
    expect(s.totalBytes).toBe(b.totalBytes + 1e9);
    expect(reserved.maxSequencesPerReplica).toBe(
      Math.floor(
        (s.capacityBytes - s.weightBytes - s.activationPeakBytes - 1e9) / s.sequenceStateBytes,
      ),
    );
  });

  it('validation: finite and ≥ 0', () => {
    expect(validateInput(input({}, { runtimeReserveBytes: -1 })).join()).toMatch(
      /runtimeReserveBytes/,
    );
  });
});

describe('chunked-prefill activation peak (F11)', () => {
  const dedicated = { device: RTX_5090, topology: singleDevice(RTX_5090) };
  const long = (prefillChunkTokens?: number): SimInput =>
    input(
      {
        workload: {
          batch: 1,
          promptLen: 65536,
          decodeContextLen: 256,
          ...(prefillChunkTokens === undefined ? {} : { prefillChunkTokens }),
        },
      },
      dedicated,
    );

  it('the default chunk is 8192 tokens', () => {
    expect(DEFAULT_PREFILL_CHUNK_TOKENS).toBe(8192);
  });

  it('a 64k prompt: the peak is the last 8k chunk, 1/8 of the one-pass peak; rates unchanged', () => {
    const chunked = simulate(long());
    const onePass = simulate(long(65536));
    const ratio =
      chunked.memory.perStage[0].activationPeakBytes /
      onePass.memory.perStage[0].activationPeakBytes;
    expectNear(ratio, 8192 / 65536, 1e-3);
    expect(chunked.prefill.tokensPerSec).toBe(onePass.prefill.tokensPerSec);
    expect(chunked.decode.tokensPerSec).toBe(onePass.decode.tokensPerSec);
    expect(chunked.memory.perStage[0].sequenceStateBytes).toBe(
      onePass.memory.perStage[0].sequenceStateBytes,
    );
  });

  it('a prompt within the budget is unchanged (bit for bit)', () => {
    const a = simulate(input({}, dedicated)).memory;
    const b = simulate(
      input(
        { workload: { batch: 1, promptLen: 128, decodeContextLen: 256, prefillChunkTokens: 128 } },
        dedicated,
      ),
    ).memory;
    expect(a).toEqual(b);
  });

  it('validation: an integer ≥ 1', () => {
    expect(validateInput(long(0)).join()).toMatch(/prefillChunkTokens/);
  });

  it('review probe: Llama-3-8B Q4_K_M on a 4090 with a 128k prompt fits (33.4 → 22.8 GB of 24 GB)', () => {
    const llama3_8b = simpleDecoder({
      name: 'Llama-3-8B',
      source: 'https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json',
      hidden: 4096,
      vocab: 128256,
      tieEmbeddings: false,
      layers: 32,
      mixer: { kind: 'gqa', heads: 32, kvHeads: 8, headDim: 128 },
      ffn: { kind: 'dense', intermediate: 14336 },
    });
    const r = simulateReport({
      model: { kind: 'spec', spec: llama3_8b },
      quant: 'q4_k_m',
      hardware: { deviceId: 'rtx-4090' },
      workload: { promptLen: 131072, genLen: 1 },
    });
    const s = r.sim.memory.perStage[0];
    expect(r.fits).toBe(true);
    expect(s.totalBytes / 1e9).toBeCloseTo(22.84, 1);
    expect(s.activationPeakBytes / 1e9).toBeLessThan(1);
  });
});

function expectNear(actual: number, expected: number, rel: number): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}
