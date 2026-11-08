import { describe, expect, it } from 'vitest';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_6000_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import type { ModelSpec, QuantConfig } from '../ir/model';
import type { TransferOp } from '../ir/ops';
import { stageTierBytes } from '../engine/memory';
import { simulate } from '../engine/simulate';
import { transferTime } from '../engine/transfer';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { lowerModel, type LoweringContext, type OffloadLowering } from './lower';
import { tableRowBytes } from './ngram';

function hf(repo: string): { model: ModelSpec; quant: QuantConfig } {
  const mapped = mapHfModel(fixtureRepoFiles(repo, { dir: '' }));
  if (mapped.kind !== 'ok') throw new Error(`${repo}: ${JSON.stringify(mapped)}`);
  return { model: mapped.model, quant: mapped.quant.quant };
}

const PLE = hf('Qwen/Qwen3.8-Flash-Next');
const ENGRAM = hf('deepseek-ai/DeepSeek-V4.1-Flash');

function decodeCtx(quant: QuantConfig, offload?: OffloadLowering): LoweringContext {
  const ctx: LoweringContext = {
    phase: 'decode',
    seqs: 1,
    qLen: 1,
    kvLen: 8193,
    quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
  };
  if (offload !== undefined) ctx.offload = offload;
  return ctx;
}

/** Transfer ops of a decode step, with their layer index. */
function fetches(m: { model: ModelSpec; quant: QuantConfig }, offload?: OffloadLowering): { layer: number | null; op: TransferOp }[] {
  return lowerModel(m.model, decodeCtx(m.quant, offload)).stages.flatMap((s) =>
    s.layers.flatMap((l) => l.ops.flatMap((op) => (op.kind === 'transfer' ? [{ layer: l.layerIndex, op }] : []))),
  );
}

const ramPath = { tier: DDR5_6000_2CH, queueDepth: 64, nStream: 1 };
const ssdPath = { tier: NVME_990_PRO_2TB, queueDepth: 512, nStream: 1 };

describe('U-16 n-gram tables off-device: read layouts, tier bytes and PCIe bytes', () => {
  it('lookupTables absent ⇒ no transfer op', () => {
    // Both checkpoints ship their tables to live off-device (model cards); the flag drives the UI hint only.
    for (const m of [PLE, ENGRAM]) {
      const tables = m.model.layers.flatMap((l) => l.blocks).filter((b) => b.kind === 'ngram_embedding');
      expect(tables.length).toBeGreaterThan(0);
      expect(tables.every((b) => b.kind === 'ngram_embedding' && b.offDeviceByDesign === true)).toBe(true);
    }
    expect(fetches(PLE)).toEqual([]);
    expect(fetches(ENGRAM)).toEqual([]);
  });

  it('PLE (config ple_layer_ids [2], one-based) sits at layer index 1: one 320 B row read, 16 rows per token', () => {
    const f = fetches(PLE, { lookupTables: { tier: 'host_ram', scaleLayout: 'separate' } });
    expect(f.map((x) => x.layer)).toEqual([1]);
    expect(f[0].op.reads).toEqual([{ count: 16, bytesEach: 320 }]); // bf16, no scales: one read whatever the layout
    expect(f[0].op.usefulBytes).toBe(5_120);
  });

  it('Engram (engram_layer_ids [1, 14], zero-based) at indices 1 and 14; useful 264 B per row (256 value + 8 scale)', () => {
    expect(tableRowBytes({ dtype: 'mxfp8' }, 256)).toEqual({ valueBytes: 256, scaleBytes: 8, rowBytes: 264 });
    const f = fetches(ENGRAM, { lookupTables: { tier: 'host_ram', scaleLayout: 'separate' } });
    expect(f.map((x) => x.layer)).toEqual([1, 14]);
  });

  it('Engram separate on DDR5: reads [{24, 256}, {24, 8}], bytes_tier 24·256 + 24·64 = 7,680, usefulBytes 6,336', () => {
    const [{ op }] = fetches(ENGRAM, { lookupTables: { tier: 'host_ram', scaleLayout: 'separate' } });
    expect(op.reads).toEqual([
      { count: 24, bytesEach: 256 },
      { count: 24, bytesEach: 8 },
    ]);
    expect(op.usefulBytes).toBe(6_336);
    expect(transferTime(op.reads, op.usefulBytes, ramPath).tierBytes).toBe(7_680);
  });

  it('Engram pinned_host on NVMe: reads [{24, 256}], bytes_tier 98,304, usefulBytes 6,336; host RAM holds the scales', () => {
    const offload: OffloadLowering = { lookupTables: { tier: 'nvme', scaleLayout: 'pinned_host' } };
    const [{ op }] = fetches(ENGRAM, offload);
    expect(op.source).toBe('nvme');
    expect(op.reads).toEqual([{ count: 24, bytesEach: 256 }]);
    expect(op.usefulBytes).toBe(6_336);
    expect(transferTime(op.reads, op.usefulBytes, ssdPath).tierBytes).toBe(98_304);
    // Scales: 384,006,168·8 = 3,072,049,344 B (table 1) + 384,016,682·8 = 3,072,133,456 B (table 2).
    const tiers = stageTierBytes(ENGRAM.model, decodeCtx(ENGRAM.quant, offload), 0);
    expect(tiers.host_ram).toBe(3_072_049_344 + 3_072_133_456);
    expect(tiers.nvme).toBe((384_006_168 + 384_016_682) * 256);
  });

  it('Engram interleaved: reads [{24, 264}], bytes_tier 98,304 on NVMe / 7,680 on DDR5, usefulBytes 6,336', () => {
    const [{ op }] = fetches(ENGRAM, { lookupTables: { tier: 'nvme', scaleLayout: 'interleaved' } });
    expect(op.reads).toEqual([{ count: 24, bytesEach: 264 }]);
    expect(op.usefulBytes).toBe(6_336);
    expect(transferTime(op.reads, op.usefulBytes, ssdPath).tierBytes).toBe(98_304);
    expect(transferTime(op.reads, op.usefulBytes, ramPath).tierBytes).toBe(7_680);
  });

  it('the default scale layout is separate on host RAM and pinned_host on NVMe (through simulate)', () => {
    const run = (tier: 'host_ram' | 'nvme') =>
      simulate({
        ...ENGRAM,
        parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
        workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
        hardware: { device: RTX_5090, topology: singleDevice(RTX_5090) },
        deployment: { offload: { lookupTables: { tier } } },
      });
    const reads = (tier: 'host_ram' | 'nvme') => {
      const op = run(tier).decode.stages[0].layers.flatMap((l) => l.ops).find((o) => o.op.kind === 'transfer')?.op;
      return op?.kind === 'transfer' ? op.reads.length : 0;
    };
    expect(reads('host_ram')).toBe(2);
    expect(reads('nvme')).toBe(1);
  });
});
