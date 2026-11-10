/**
 * The token embedding off-device (docs/simulation-model.md §12 "Token embedding off-device"): a 1-gram lookup table
 * in host RAM or on the SSD. End to end through `simulate` on a dense shortlist model.
 */
import { describe, expect, it } from 'vitest';
import { M5_ULTRA_80C } from '../hw/presets/apple';
import { singleDevice } from '../hw/presets';
import { RTX_5090 } from '../hw/presets/nvidia';
import { DDR5_5600_2CH, NVME_990_PRO_2TB } from '../hw/presets/storage';
import type { TransferOp } from '../ir/ops';
import { fixtureRepoFiles } from '../models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../models/hf/index';
import { REF_DENSE_8B } from '../models/reference';
import { simulate } from './simulate';
import type { HardwareStack } from '../hw/types';
import type { EmbeddingRule, SimInput, SimResult } from './types';
import { TIED_EMBEDDINGS_OFFLOAD_ERROR } from './validate';

function expectClose(actual: number, expected: number, rel = 1e-9): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rel * Math.abs(expected));
}

function hf(repo: string): Pick<SimInput, 'model' | 'quant'> {
  const mapped = mapHfModel(fixtureRepoFiles(repo, { dir: '' }));
  if (mapped.kind !== 'ok') throw new Error(`${repo}: ${JSON.stringify(mapped)}`);
  return { model: mapped.model, quant: mapped.quant.quant };
}

/** PCIe 5.0 x16 per direction, 32e9·16·128/130/8 × 256/280 (TLP payload efficiency, links.ts). Unit: B/s. */
const B_LINK = ((32e9 * 16 * 128) / 130 / 8) * (256 / 280);
const GPU: HardwareStack = {
  device: RTX_5090,
  topology: singleDevice(RTX_5090),
  host: { ram: DDR5_5600_2CH, ssd: NVME_990_PRO_2TB },
};
const M5: HardwareStack = {
  device: M5_ULTRA_80C,
  topology: singleDevice(M5_ULTRA_80C),
  host: { ssd: NVME_990_PRO_2TB },
};

/**
 * Qwen/Qwen3.8-27B (fixture): untied, V = 248,320, h = 5120, embeddings bf16, so one row is
 * rowBytes = h·2 = 10,240 B and the table shard (tp 1) is V·h·2 = 2,542,796,800 B.
 */
const QWEN = 'Qwen/Qwen3.8-27B';
const ROW_BYTES = 5120 * 2;
const SHARD_BYTES = 248_320 * 5120 * 2;

function qwen(
  embeddings: EmbeddingRule | undefined,
  hardware = GPU,
  extra: Partial<SimInput> = {},
): SimResult {
  const { model, quant } = hf(QWEN);
  return simulate({
    model,
    quant,
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
    hardware,
    ...(embeddings === undefined ? {} : { deployment: { offload: { embeddings } } }),
    ...extra,
  });
}

/** Every transfer op of a phase, over every stage and layer. */
function transfers(r: SimResult, phase: 'prefill' | 'decode'): TransferOp[] {
  return r[phase].stages.flatMap((s) =>
    s.layers.flatMap((l) => l.ops.flatMap((o) => (o.op.kind === 'transfer' ? [o.op] : []))),
  );
}

/**
 * Worked example, RTX 5090 + DDR5-5600 ×2 + 990 PRO, b 1, P 8192, decode @ 8k:
 * - decode reads one row per step, q = 0 (the row is the token just sampled; nothing to hide it behind):
 *   host RAM t = α_ram + α_link + max(tier 10,240/89.6e9, link 10,240/B_link) = 70 ns + 1 µs + 0.178 µs = 1.248 µs;
 *   SSD t = α_ssd 45.5 µs + 1 µs + a 12 KiB read ≈ 48.1 µs.
 * - the decode step itself is ≈ 29.09 ms (34.377 tok/s @ 8k, bandwidth bound on the weights + KV), so host RAM moves
 *   decode @ 8k by 4.3e-5 (34.376 tok/s @ 8k) and the SSD by 0.17 % (34.321 tok/s @ 8k): both well under 1 %.
 * - prefill gathers the whole prompt: 8192 rows × 10,240 B = 83,886,080 B over the link, q = 1 (ids known), so
 *   t = 1.07 µs + 83,886,080/B_link = 1.457 ms (SSD 13.56 ms), all hidden behind the 2.07 s prefill step: prefill
 *   stays 3953 tok/s @ 8k.
 */
describe(`${QWEN} bf16 on one RTX 5090 + DDR5-5600 ×2 + 990 PRO: token embedding on the device / in host RAM / on the SSD, b 1, P 8192, decode @ 8k`, () => {
  const base = qwen(undefined);
  const ram = qwen({ tier: 'host_ram' });
  const ssd = qwen({ tier: 'nvme' });

  it('the fixture is untied with V 248,320 and h 5120', () => {
    const { model } = hf(QWEN);
    expect(model.tieEmbeddings).toBe(false);
    expect(model.vocab).toBe(248_320);
    expect(model.hidden).toBe(5120);
  });

  it('decode @ 8k changes by less than 1 %: 34.377 → 34.376 (host RAM) / 34.321 (SSD) tok/s @ 8k', () => {
    const b = base.decode.tokensPerSec;
    expectClose(b, 34.3774, 1e-5);
    expectClose(ram.decode.tokensPerSec, 34.376, 1e-5);
    expectClose(ssd.decode.tokensPerSec, 34.3207, 1e-5);
    const dRam = Math.abs(ram.decode.tokensPerSec - b) / b;
    const dSsd = Math.abs(ssd.decode.tokensPerSec - b) / b;
    expect(dRam).toBeGreaterThan(0);
    expect(dRam).toBeLessThan(1e-4);
    expect(dSsd).toBeGreaterThan(dRam);
    expect(dSsd).toBeLessThan(0.01);
  });

  it('decode: one row from the tier per step, not prefetched, fully exposed', () => {
    const [op] = transfers(ram, 'decode');
    expect(transfers(ram, 'decode')).toHaveLength(1);
    expect(op).toMatchObject({
      name: 'embed.row_fetch',
      group: 'token_embedding',
      source: 'host_ram',
      reads: [{ count: 1, bytesEach: ROW_BYTES }],
      usefulBytes: ROW_BYTES,
      prefetchedFraction: 0,
    });
    // Formula: t = α_ram + α_link + max(tier, link) = 70 ns + 1 µs + 10,240 B / B_link
    expectClose(ram.decode.transfer?.sec ?? 0, 70e-9 + 1e-6 + ROW_BYTES / B_LINK);
    expectClose(ram.decode.transfer?.exposedSec ?? 0, ram.decode.transfer?.sec ?? 0, 1e-12);
    expect(transfers(ssd, 'decode')[0].source).toBe('nvme');
  });

  it('prefill gathers the prompt: 8192 rows, 83,886,080 B and 8192 requests, prefetched (q = 1)', () => {
    const ops = transfers(ram, 'prefill');
    expect(ops).toHaveLength(1);
    expect(ops[0].prefetchedFraction).toBe(1);
    expect(ram.prefill.transfer?.bytes).toBe(83_886_080);
    expect(ram.prefill.transfer?.requests).toBe(8192);
    // Formula: t = α_ram + α_link + 83,886,080 B / B_link   (link bound), hidden: q = 1 and core ≫ t
    expectClose(ram.prefill.transfer?.sec ?? 0, 70e-9 + 1e-6 + 83_886_080 / B_LINK);
    expect(ram.prefill.transfer?.exposedSec).toBe(0);
    expect(ram.prefill.tokensPerSec).toBe(base.prefill.tokensPerSec);
  });

  it('stage-0 device weights drop by exactly 2,542,796,800 B, and the tier holds the same bytes', () => {
    for (const r of [ram, ssd]) {
      expect(base.memory.perStage[0].weightBytes - r.memory.perStage[0].weightBytes).toBe(
        SHARD_BYTES,
      );
    }
    expect(ram.memory.tiers).toEqual([
      {
        tier: 'host_ram',
        bytes: SHARD_BYTES,
        breakdown: {
          lookupTables: 0,
          routedExperts: 0,
          draftExperts: 0,
          tokenEmbeddings: SHARD_BYTES,
          kvCache: 0,
        },
        capacityBytes: DDR5_5600_2CH.capacityBytes,
        fits: true,
      },
    ]);
    expect(ssd.memory.tiers.map((t) => [t.tier, t.breakdown.tokenEmbeddings])).toEqual([
      ['nvme', SHARD_BYTES],
    ]);
  });

  it('staging is the prefill chunk double-buffered: 2·8192·10,240 B, also for a 32k prompt (not 2·32768·…)', () => {
    expect(base.memory.perStage[0].stagingBytes).toBe(0);
    expect(ram.memory.perStage[0].stagingBytes).toBe(2 * 8192 * ROW_BYTES);
    const long = (e: EmbeddingRule | undefined) =>
      qwen(e, GPU, { workload: { batch: 1, promptLen: 32_768, decodeContextLen: 32_768 } });
    const longRam = long({ tier: 'host_ram' });
    // Formula: staging = 2·C·rowBytes, C = 8192 (default prefill chunk) = 167,772,160 B ≪ the 2.54 GB freed
    expect(longRam.memory.perStage[0].stagingBytes).toBe(2 * 8192 * ROW_BYTES);
    // The prefill rate still pays all 32k rows: the costed step is the whole prompt.
    expect(longRam.prefill.transfer?.requests).toBe(32_768);
    const longBase = long(undefined);
    expect(longBase.memory.perStage[0].totalBytes - longRam.memory.perStage[0].totalBytes).toBe(
      SHARD_BYTES - 2 * 8192 * ROW_BYTES,
    );
  });
});

describe('unified memory (M5 Ultra): host RAM is the device pool', () => {
  const base = qwen(undefined, M5);
  it("a 'host_ram' embedding equals the default except the unified note", () => {
    const ram = qwen({ tier: 'host_ram' }, M5);
    expect(ram.warnings).toContain(
      "offload: on unified memory host RAM is the device memory; the 'host_ram' placement keeps the weights on the device",
    );
    expect({ ...ram, warnings: [] }).toEqual({ ...base, warnings: [] });
  });
  it("an 'nvme' embedding moves the shard to the SSD tier", () => {
    const ssd = qwen({ tier: 'nvme' }, M5);
    expect(base.memory.perStage[0].weightBytes - ssd.memory.perStage[0].weightBytes).toBe(
      SHARD_BYTES,
    );
    expect(ssd.memory.tiers.map((t) => [t.tier, t.bytes])).toEqual([['nvme', SHARD_BYTES]]);
  });
});

describe('a tied model keeps its embedding on the device', () => {
  const tied = { ...REF_DENSE_8B, tieEmbeddings: true };
  const input = (tier: EmbeddingRule['tier']): SimInput => ({
    model: tied,
    quant: { weights: { dtype: 'bf16' }, activations: 'bf16', kvCache: { dtype: 'bf16' } },
    parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
    workload: { batch: 1, promptLen: 1024, decodeContextLen: 1024 },
    hardware: GPU,
    deployment: { offload: { embeddings: { tier } } },
  });
  it.each(['host_ram', 'nvme'] as const)(
    '%s is rejected with the tied-embeddings reason',
    (tier) => {
      expect(() => simulate(input(tier))).toThrow(TIED_EMBEDDINGS_OFFLOAD_ERROR);
    },
  );
  it("'device' is accepted", () => {
    expect(() => simulate(input('device'))).not.toThrow();
  });
});

describe('MTP spec-dec: the drafter gathers from the same off-device table', () => {
  const spec = {
    deployment: {
      offload: { embeddings: { tier: 'host_ram' as const } },
      speculative: {
        method: 'mtp' as const,
        draftLength: 1,
        acceptance: { kind: 'iid' as const, alpha: 0.8 },
      },
    },
  };
  const r = qwen(undefined, GPU, spec);
  it('decode: draft.embed.row_fetch with q = 0; prefill: q = 1', () => {
    const draftDecode = transfers(r, 'decode').filter((op) => op.name === 'draft.embed.row_fetch');
    expect(draftDecode.length).toBeGreaterThan(0);
    for (const op of draftDecode)
      expect(op).toMatchObject({
        group: 'token_embedding',
        source: 'host_ram',
        prefetchedFraction: 0,
      });
    const draftPrefill = transfers(r, 'prefill').filter(
      (op) => op.name === 'draft.embed.row_fetch',
    );
    expect(draftPrefill).toHaveLength(1);
    expect(draftPrefill[0].prefetchedFraction).toBe(1);
  });
});

/**
 * With "Overlap across tiers" (docs/simulation-model.md §12): Qwen3.8-Flash-Next fp8, every layer's experts and the
 * PLE table on the SSD (f = 0, p 0.8), b 1, P 8192, and the token embedding in host RAM. Decode: the embedding row
 * has q = 0, so the SSD is the only prefetched source and the stage stays serial: decode transfer = the SSD-only
 * transfer + the one row, exactly. Prefill: the embedding rows (q = 1) are a host RAM stream next to the SSD stream,
 * so the stage is overlapped and its transfer is below the serial sum; staging becomes
 * max_all + max_ram + max_ssd = the SSD-only 2·max_ssd + the 8192 embedding rows.
 */
describe('Qwen/Qwen3.8-Flash-Next fp8, experts on the SSD + token embedding in host RAM: overlap across tiers', () => {
  const run = (embeddings: EmbeddingRule | undefined): SimResult => {
    const { model, quant } = hf('Qwen/Qwen3.8-Flash-Next');
    return simulate({
      model,
      quant: { ...quant, weights: { dtype: 'fp8_e4m3' } },
      parallel: { tp: 1, pp: 1, dp: 1, ep: 1 },
      workload: { batch: 1, promptLen: 8192, decodeContextLen: 8192 },
      hardware: GPU,
      deployment: {
        offload: {
          routedExperts: { tier: 'nvme', residentFraction: 0, hitRatio: { kind: 'uniform' } },
          lookupTables: { tier: 'nvme' },
          ...(embeddings === undefined ? {} : { embeddings }),
        },
      },
    });
  };
  const ssdOnly = run(undefined);
  const both = run({ tier: 'host_ram' });
  const embedOp = (r: SimResult, phase: 'prefill' | 'decode') => {
    const ops = r[phase].stages[0].layers
      .flatMap((l) => l.ops)
      .filter((o) => o.op.name === 'embed.row_fetch');
    expect(ops).toHaveLength(1);
    return ops[0];
  };

  it('decode stays serial: transfer = SSD-only transfer + the row (q = 0)', () => {
    const row = embedOp(both, 'decode');
    expect(row.op.kind === 'transfer' && row.op.prefetchedFraction).toBe(0);
    expectClose(
      both.decode.transfer?.sec ?? 0,
      (ssdOnly.decode.transfer?.sec ?? 0) + row.sec,
      1e-12,
    );
    const dDecode =
      (ssdOnly.decode.tokensPerSec - both.decode.tokensPerSec) / ssdOnly.decode.tokensPerSec;
    expect(dDecode).toBeGreaterThan(0);
    expect(dDecode).toBeLessThan(1e-4);
  });

  it('prefill overlaps the embedding rows with the SSD stream: below the serial sum', () => {
    const row = embedOp(both, 'prefill');
    expect(row.op.kind === 'transfer' && row.op.prefetchedFraction).toBe(1);
    const serial = (ssdOnly.prefill.transfer?.sec ?? 0) + row.sec;
    expect(both.prefill.transfer?.sec ?? 0).toBeLessThan(serial);
    expect(both.prefill.transfer?.sec ?? 0).toBeGreaterThanOrEqual(
      ssdOnly.prefill.transfer?.sec ?? 0,
    );
  });

  it('staging: max_all + max_ram + max_ssd = the SSD-only 2·max_ssd + the 8192 rows', () => {
    const row = embedOp(both, 'prefill');
    if (row.op.kind !== 'transfer') throw new Error('not a transfer');
    expect(both.memory.perStage[0].stagingBytes - ssdOnly.memory.perStage[0].stagingBytes).toBe(
      row.op.usefulBytes,
    );
  });
});
