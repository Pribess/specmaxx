import { expect, it } from 'vitest';
import { simulate } from '../../sim/engine/simulate';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../../sim/models/hf';
import { buildSimInput, loadedFromHf } from './buildInput';
import { memoryVerdict } from './memoryVerdict';
import { summarize } from './summary';
import { batchSweep } from './sweep';
import { decodeState } from './url';

const ID = 'deepseek-ai/DeepSeek-V4.1-Flash';
const REPRO = `m=${ID}&d=apple-m5-ultra-80c&eng=gpu-na&cap=1&nodes=2&xl=thunderbolt-5-rdma-macos-26-2&hr=ddr5-5600-2ch&tp=2&olt=ssd`;

function loaded() {
  const r = mapHfModel(fixtureRepoFiles(ID));
  if (r.kind !== 'ok') throw new Error(r.kind);
  return loadedFromHf({ kind: 'hf', id: ID }, r);
}

function inputOf(search: string) {
  const { state } = decodeState(search);
  const built = buildSimInput(state, loaded());
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built));
  return built.input;
}

it('repro: fits at batch 1, no host_ram tier, scales counted in the device bytes', () => {
  const input = inputOf(REPRO);
  const r = simulate(input);
  const s = summarize(r, input);
  expect(s.memory.kind).toBe('unified');
  expect(s.memory.fits).toBe(true);
  expect((s.memory.tiers ?? []).map((t) => t.tier)).toEqual(['nvme']);
  const noScales = simulate({
    ...input,
    deployment: {
      ...input.deployment,
      offload: {
        ...input.deployment?.offload,
        lookupTables: { tier: 'nvme', scaleLayout: 'separate' },
      },
    },
  });
  expect(r.memory.perStage[0].weightBytes).toBeGreaterThan(noScales.memory.perStage[0].weightBytes);
});

it('the sweep fits equals the main verdict at every batch, and is not all hollow', () => {
  // A 256k decode context, so the KV cache (not the prefill activations, bounded by the 8192-token chunk) makes the
  // large batches overflow.
  const input = inputOf(`${REPRO}&c=262144`);
  const points = batchSweep(input);
  for (const p of points) {
    const withBatch = { ...input, workload: { ...input.workload, batch: p.batch } };
    const s = summarize(simulate(withBatch), withBatch);
    expect(p.fits).toBe(s.memory.fits);
    expect(p.fits).toBe(memoryVerdict(s.memory).fits);
  }
  expect(points[0].fits).toBe(true);
  expect(points.some((p) => !p.fits)).toBe(true);
});

it('hr is ignored on unified memory', () => {
  const withHr = decodeState(REPRO);
  expect(withHr.dropped.join(' ')).not.toContain('hr');
  const without = REPRO.replace('&hr=ddr5-5600-2ch', '');
  expect(inputOf(REPRO)).toEqual(inputOf(without));
});

it('a dedicated GPU with olt=ssd still keeps the scales in the host_ram tier', () => {
  const input = inputOf(
    REPRO.replace('d=apple-m5-ultra-80c', 'd=rtx-5090')
      .replace('&cap=1', '')
      .replace(/&nodes=2&xl=[^&]+/, '')
      .replace('&tp=2', '&tp=1'),
  );
  const s = summarize(simulate(input), input);
  const host = (s.memory.tiers ?? []).find((t) => t.tier === 'host_ram');
  expect(host?.breakdown.lookupTables).toBeGreaterThan(0);
});
