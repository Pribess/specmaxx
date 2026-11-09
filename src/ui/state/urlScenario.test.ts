/**
 * A user URL reproduced end to end (offline, from the fixture): Qwen3.8-Flash-Next NVFP4 on two single-RTX-5090
 * nodes over 10GbE TCP, TP 2, 70 % of the routed experts and the lookup tables in host RAM. It pins the decode and
 * prefill results @ 8k and the effect of each modelling change, undone one at a time (`variants`):
 * - inter-node host staging (docs §6 "Inter-node host staging"): undone by giving the GPU and the link GPUDirect;
 * - the TCP latency recheck (docs §6): undone by restoring the link's old α (10GbE kept 10 µs, so no change);
 * - the expert-prefetch accuracy (docs §12 "Expert prefetch"): undone by perfect prediction (the former 'ideal').
 * Set SCENARIO_REPORT=1 to print the rows.
 */

import { describe, expect, it } from 'vitest';
import { simulate } from '../../sim/engine/simulate';
import type { PhaseResult, SimInput } from '../../sim/engine/types';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../../sim/models/hf/index';
import { buildSimInput, loadedFromHf } from './buildInput';
import { decodeState } from './url';

const URL_QUERY =
  'm=Qwen/Qwen3.8-Flash-Next&w=nvfp4&kv=nvfp4&nodes=2&hr=ddr5-5600-2ch&tp=2&ox=ram&oxf=0.7&olt=ram';

function scenarioInput(): SimInput {
  const { state } = decodeState(URL_QUERY);
  if (state.model.kind !== 'hf') throw new Error('expected an HF model');
  const r = mapHfModel(fixtureRepoFiles(state.model.id));
  if (r.kind !== 'ok') throw new Error(r.kind);
  const built = buildSimInput(state, loadedFromHf(state.model, r));
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  return built.input;
}

/** One row of the report: a phase's headline numbers (tok/s; the rest in seconds). */
interface Row {
  decodeTokPerSec: number;
  prefillTokPerSec: number;
  cycleSec: number;
  computeSec: number;
  commLatencySec: number;
  commBandwidthSec: number;
  exposedCommSec: number;
  transferSec: number;
  exposedTransferSec: number;
}

function row(input: SimInput): Row {
  const r = simulate(input);
  const d: PhaseResult = r.decode;
  return {
    decodeTokPerSec: d.tokensPerSec,
    prefillTokPerSec: r.prefill.tokensPerSec,
    cycleSec: d.cycleSec,
    computeSec: d.stages.reduce((s, st) => s + st.computeSec, 0),
    commLatencySec: d.comm.latencySec,
    commBandwidthSec: d.comm.bandwidthSec,
    exposedCommSec: d.comm.exposedSec,
    transferSec: d.transfer?.sec ?? 0,
    exposedTransferSec: d.transfer?.exposedSec ?? 0,
  };
}

/** The input with one modelling change undone (deep copies; `input` is not mutated). */
const UNDO = {
  /** Host staging off: GPUDirect on both the device and the inter-node link (the convention before staging). */
  staging(input: SimInput): SimInput {
    const { device, topology } = input.hardware;
    if (topology.interNode === undefined) throw new Error('expected an inter-node link');
    return {
      ...input,
      hardware: {
        ...input.hardware,
        device: { ...device, gpuDirectRdma: true },
        topology: { ...topology, interNode: { ...topology.interNode, gpuDirectRdma: true } },
      },
    };
  },
  /** The inter-node link's α before the TCP latency recheck (10GbE: 10 µs, unchanged). Unit: seconds. */
  latency(input: SimInput): SimInput {
    const { topology } = input.hardware;
    if (topology.interNode === undefined) throw new Error('expected an inter-node link');
    return {
      ...input,
      hardware: {
        ...input.hardware,
        topology: { ...topology, interNode: { ...topology.interNode, latencySec: 1e-5 } },
      },
    };
  },
  /** Perfect expert prediction, p = 1 (the former offload overlap 'ideal'). */
  prefetch(input: SimInput): SimInput {
    const offload = input.deployment?.offload;
    if (offload === undefined) throw new Error('expected an offload config');
    return {
      ...input,
      deployment: { ...input.deployment, offload: { ...offload, prefetchAccuracy: 1 } },
    };
  },
};

/** The report rows: every change undone, then each one added back in turn. */
function variants(
  input: SimInput,
): Record<'before' | 'staging' | 'latency' | 'prefetch', SimInput> {
  return {
    before: UNDO.staging(UNDO.latency(UNDO.prefetch(input))),
    staging: UNDO.latency(UNDO.prefetch(input)),
    latency: UNDO.prefetch(input),
    prefetch: input,
  };
}

/** |actual − expected| ≤ rel·|expected|. */
function expectRel(actual: number, expected: number, label: string, rel = 1e-3): void {
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    rel * Math.abs(expected),
  );
}

/** Pinned rows (decode and prefill @ 8k; seconds per decode token for the rest). */
const ROW_BEFORE: Row = {
  decodeTokPerSec: 192.9954,
  prefillTokPerSec: 2489.7393,
  cycleSec: 5.181471e-3,
  computeSec: 2.905715e-3,
  commLatencySec: 1.97e-3,
  commBandwidthSec: 0.4014144e-3,
  exposedCommSec: 1.97e-3,
  transferSec: 3.211471e-3,
  exposedTransferSec: 0.3057558e-3,
};
/** +staging: ≈ 205 inter-node message steps × (2 µs + a negligible fill) = +0.41 ms of α. */
const ROW_STAGING: Row = {
  ...ROW_BEFORE,
  decodeTokPerSec: 178.8462,
  prefillTokPerSec: 2486.9763,
  cycleSec: 5.591396e-3,
  commLatencySec: 2.379925e-3,
  exposedCommSec: 2.379925e-3,
};
/** +prefetch accuracy 0.8 (the default): the copies of mispredicted experts are on the critical path. */
const ROW_PREFETCH: Row = {
  ...ROW_STAGING,
  decodeTokPerSec: 162.0132,
  cycleSec: 6.172338e-3,
  exposedTransferSec: 0.8866978e-3,
};
const PINNED: Record<keyof ReturnType<typeof variants>, Row> = {
  before: ROW_BEFORE,
  staging: ROW_STAGING,
  latency: ROW_STAGING, // 10GbE keeps 10 µs
  prefetch: ROW_PREFETCH,
};

describe('URL scenario: Qwen3.8-Flash-Next on 2 × RTX 5090 over 10GbE TCP', () => {
  it('decodes the URL to the expected rig', () => {
    const input = scenarioInput();
    expect(input.hardware.topology.id).toBe('node-1x2-rtx-5090-inter-10gbe-tcp');
  });

  it('pins the rows: before → +staging → +latency → +prefetch', () => {
    const rows = Object.fromEntries(
      Object.entries(variants(scenarioInput())).map(([k, v]) => [k, row(v)]),
    );
    if (process.env.SCENARIO_REPORT) console.log(JSON.stringify(rows, null, 2));
    for (const [name, pinned] of Object.entries(PINNED)) {
      for (const [field, value] of Object.entries(pinned)) {
        expectRel(rows[name][field as keyof Row], value, `${name}.${field}`);
      }
    }
  });

  it('the latency recheck leaves 10GbE unchanged, so its row equals the staging row', () => {
    const v = variants(scenarioInput());
    expect(row(v.latency)).toEqual(row(v.staging));
  });
});
