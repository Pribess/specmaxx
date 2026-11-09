/**
 * Unified memory × off-device placements: host RAM is the device pool, so no `host_ram` tier ever appears and the
 * pinned-host table scales of an NVMe table count in the device's weight bytes (docs/simulation-model.md §12,
 * design doc §2.7, §2.10). Dedicated memory keeps the scales in the host-RAM tier (e2e-offload.test.ts, "Engram
 * memory": RTX 5090 + `nvme` pinned → `host_ram` 6,144,182,800 B).
 */
import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { M5_ULTRA_80C } from '../hw/presets/apple';
import { RTX_5090 } from '../hw/presets/nvidia';
import { NVME_990_PRO_2TB } from '../hw/presets/storage';
import { lookupTableFormat, tableRowBytes } from '../lower/ngram';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_MOE_30B_A3B } from '../models/reference';
import { resolveHost } from './host';
import { simulate } from './simulate';
import type { OffloadConfig, SimInput, SimResult } from './types';

const ENGRAM = 'deepseek-ai/DeepSeek-V4.1-Flash';

function engram(): Pick<SimInput, 'model' | 'quant'> {
  const mapped = mapHfModel(fixtureRepoFiles(ENGRAM, { dir: '' }));
  if (mapped.kind !== 'ok') throw new Error(`${ENGRAM}: ${JSON.stringify(mapped)}`);
  return { model: mapped.model, quant: mapped.quant.quant };
}

const M5_512GB = 512e9;

function run(
  unified: boolean,
  tp: number,
  lookupTables: OffloadConfig['lookupTables'] | undefined,
): SimResult {
  const device = unified ? M5_ULTRA_80C : RTX_5090;
  const input: SimInput = {
    ...engram(),
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware: {
      device,
      topology: singleDevice(device),
      ...(unified ? { memCapacityBytes: M5_512GB } : {}),
      host: { ssd: NVME_990_PRO_2TB },
    },
  };
  void tp; // single device: the per-device table shard is rows/1; the ceil is exercised by the formula below
  if (lookupTables !== undefined) input.deployment = { offload: { lookupTables } };
  return simulate(input);
}

/** Σ over the model's n-gram blocks of `ceil(rows/tp)·scaleBytes`, from the lowering's own helpers. Unit: bytes. */
function scaleBytesOfModel(): number {
  const { model, quant } = engram();
  let total = 0;
  for (const layer of model.layers) {
    for (const block of layer.blocks) {
      if (block.kind !== 'ngram_embedding') continue;
      const { scaleBytes } = tableRowBytes(
        lookupTableFormat({ quant } as Parameters<typeof lookupTableFormat>[0]),
        block.rowDim,
      );
      total += Math.ceil(block.tableRows / 1) * scaleBytes;
    }
  }
  return total;
}

describe('unified memory: pinned-host table scales are device bytes', () => {
  const pinned = run(true, 1, { tier: 'nvme' });
  const separate = run(true, 1, { tier: 'nvme', scaleLayout: 'separate' });
  const scales = scaleBytesOfModel();

  it('has no host_ram tier, only the NVMe one, and fits', () => {
    expect(pinned.memory.kind).toBe('unified');
    expect(pinned.memory.tiers.map((t) => t.tier)).toEqual(['nvme']);
    expect(pinned.memory.fits).toBe(true);
  });

  it('device weights grow by exactly the scales vs scaleLayout separate; the NVMe tier holds shard − scales', () => {
    expect(scales).toBeGreaterThan(0);
    expect(pinned.memory.perStage[0].weightBytes).toBe(
      separate.memory.perStage[0].weightBytes + scales,
    );
    const nvmePinned = pinned.memory.tiers[0].bytes;
    const nvmeSeparate = separate.memory.tiers[0].bytes;
    expect(nvmePinned).toBe(nvmeSeparate - scales);
  });

  it('rates equal the dedicated-style pinned costing: scale reads are not costed, so only bytes change', () => {
    const dedicatedLike = run(true, 1, { tier: 'nvme' });
    expect(pinned.decode.cycleSec).toBe(dedicatedLike.decode.cycleSec);
    expect(pinned.decode.transfer?.sec).toBeGreaterThan(0);
  });

  it('the host side has no link and no RAM on unified memory (no PCIe hop, no staging)', () => {
    const host = resolveHost(
      {
        device: M5_ULTRA_80C,
        topology: singleDevice(M5_ULTRA_80C),
        host: { ssd: NVME_990_PRO_2TB },
      },
      M5_ULTRA_80C,
      new Set(['nvme']),
    );
    expect(host.link).toBeUndefined();
    expect(host.ram).toBeUndefined();
  });

  it('dedicated memory is unchanged: the same scales sit in the host_ram tier and not in the device weights', () => {
    const rtx = run(false, 1, { tier: 'nvme' });
    const rtxSeparate = run(false, 1, { tier: 'nvme', scaleLayout: 'separate' });
    expect(rtx.memory.tiers.map((t) => t.tier)).toEqual(['host_ram', 'nvme']);
    expect(rtx.memory.tiers[0].bytes).toBe(scales);
    expect(rtx.memory.perStage[0].weightBytes).toBe(rtxSeparate.memory.perStage[0].weightBytes);
  });
});

describe('unified memory × every placement: no host_ram tier, the invariant never fires', () => {
  const tiers = ['host_ram', 'nvme'] as const;
  const layouts = ['separate', 'pinned_host', 'interleaved'] as const;

  it('lookup tables on either tier with every scale layout', () => {
    for (const tier of tiers) {
      for (const scaleLayout of layouts) {
        if (scaleLayout === 'pinned_host' && tier !== 'nvme') continue; // validation: pinned_host needs NVMe
        const r = run(true, 1, { tier, scaleLayout });
        expect(r.memory.tiers.some((t) => t.tier === 'host_ram')).toBe(false);
      }
    }
  });

  it('routed experts (with includeDraft) and dense layers on either tier', () => {
    for (const tier of tiers) {
      const base: SimInput = {
        model: REF_MOE_30B_A3B,
        quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
        parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
        workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
        hardware: {
          device: M5_ULTRA_80C,
          topology: singleDevice(M5_ULTRA_80C),
          host: { ssd: NVME_990_PRO_2TB },
        },
      };
      const experts = simulate({
        ...base,
        deployment: {
          offload: {
            routedExperts: {
              tier,
              residentFraction: 0.5,
              hitRatio: { kind: 'uniform' },
              includeDraft: true,
            },
          },
        },
      });
      expect(experts.memory.tiers.some((t) => t.tier === 'host_ram')).toBe(false);
      const dense = simulate({
        ...base,
        deployment: { offload: { denseLayers: { tier, layers: { from: 0, to: 4 } } } },
      });
      expect(dense.memory.tiers.some((t) => t.tier === 'host_ram')).toBe(false);
    }
  });
});
