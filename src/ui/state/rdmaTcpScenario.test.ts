/**
 * A user URL reproduced end to end (offline, from the fixture): DeepSeek-V4.1-Flash on two M5 Ultra (64-core GPU)
 * Macs, TP 2, lookup tables on SSD, over Thunderbolt 5, with and without DSpark speculative decoding (code). The
 * user reported that decode over TB5 RDMA was SLOWER than over TB5 TCP, because the RDMA preset carried a 50 µs α
 * (an all-reduce upper bound) while TCP carried an estimated 45 µs floor. With one α definition for both
 * transports (docs §4 "Link latencies") RDMA must decode at least as fast as TCP over the same cable.
 *
 * The rows are decomposed like `urlScenario.test.ts`: `tcpNoCopy` is the TCP link with its socket copy undone
 * (docs §6 "TCP socket copy"; `transport: 'rdma'` on the same link numbers), so the α re-sourcing and the copy term
 * are reported separately. Set SCENARIO_REPORT=1 to print the rows.
 */

import { describe, expect, it } from 'vitest';
import { simulate } from '../../sim/engine/simulate';
import type { SimInput } from '../../sim/engine/types';
import { linkPresetId } from '../../sim/hw/presets/topologies';
import { fixtureRepoFiles } from '../../sim/models/__fixtures__/fixtures.testutil';
import { mapHfModel } from '../../sim/models/hf/index';
import { buildSimInput, loadedFromHf } from './buildInput';
import { decodeState } from './url';

const ID = 'deepseek-ai/DeepSeek-V4.1-Flash';
const BASE = `m=${ID}&d=apple-m5-ultra-64c&cap=1&nodes=2&tp=2&olt=ssd`;
const DSPARK = '&sd=dspark&sdu=code';
const TCP = '&xl=thunderbolt-5-bridge-tcp';

/** The four URLs of the report: the default network (TB5 RDMA) and TB5 TCP, each with and without DSpark. */
const QUERIES = {
  rdmaDspark: BASE + DSPARK,
  tcpDspark: BASE + TCP + DSPARK,
  rdma: BASE,
  tcp: BASE + TCP,
} as const;

type Case = keyof typeof QUERIES;

function inputOf(query: string): SimInput {
  const { state } = decodeState(query);
  if (state.model.kind !== 'hf') throw new Error('expected an HF model');
  const r = mapHfModel(fixtureRepoFiles(state.model.id));
  if (r.kind !== 'ok') throw new Error(r.kind);
  const built = buildSimInput(state, loadedFromHf(state.model, r));
  if (built.kind !== 'ok') throw new Error(JSON.stringify(built.issues));
  return built.input;
}

/** The input with the TCP socket copy undone: the same link numbers, marked as RDMA (zero-copy). */
function withoutSocketCopy(input: SimInput): SimInput {
  const { topology } = input.hardware;
  if (topology.interNode === undefined) throw new Error('expected an inter-node link');
  return {
    ...input,
    hardware: {
      ...input.hardware,
      topology: {
        ...topology,
        interNode: { ...topology.interNode, transport: 'rdma' } as typeof topology.interNode,
      },
    },
  };
}

/** One row: throughput (tok/s) at its context (tokens), and the decode communication terms (seconds per token). */
interface Row {
  link: string;
  decodeTokPerSec: number;
  decodeContextLen: number;
  prefillTokPerSec: number;
  promptLen: number;
  commLatencySec: number;
  commBandwidthSec: number;
}

function row(input: SimInput): Row {
  const r = simulate(input);
  return {
    link:
      input.hardware.topology.interNode === undefined
        ? '(none)'
        : linkPresetId(input.hardware.topology.interNode),
    decodeTokPerSec: r.decode.tokensPerSec,
    decodeContextLen: input.workload.decodeContextLen,
    prefillTokPerSec: r.prefill.tokensPerSec,
    promptLen: input.workload.promptLen,
    commLatencySec: r.decode.comm.latencySec,
    commBandwidthSec: r.decode.comm.bandwidthSec,
  };
}

function rows(): Record<Case | 'tcpDsparkNoCopy' | 'tcpNoCopy', Row> {
  const inputs = Object.fromEntries(
    Object.entries(QUERIES).map(([k, q]) => [k, inputOf(q)]),
  ) as Record<Case, SimInput>;
  return {
    rdmaDspark: row(inputs.rdmaDspark),
    tcpDsparkNoCopy: row(withoutSocketCopy(inputs.tcpDspark)),
    tcpDspark: row(inputs.tcpDspark),
    rdma: row(inputs.rdma),
    tcpNoCopy: row(withoutSocketCopy(inputs.tcp)),
    tcp: row(inputs.tcp),
  };
}

/** |actual − expected| ≤ rel·|expected|. */
function expectRel(actual: number, expected: number, label: string, rel = 1e-3): void {
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    rel * Math.abs(expected),
  );
}

/**
 * Pinned decode rows, all @ 8k context (decodeContextLen = promptLen = 8192 tokens); comm in seconds per decode token.
 * Before this fix: RDMA + DSpark 145.2 tok/s (comm α 9.10 ms), TCP + DSpark 150.1 (8.19 ms); without spec-dec RDMA
 * 63.3 (8.35 ms), TCP 66.8 (7.51 ms). TCP's rows did not move with the α fix (45 µs kept); the copy adds ≈ 0.1 %.
 * The DSpark rows then moved with the expected-unique MoE routing (docs §5 "Expected distinct experts"): the 6-token
 * verify reads U(6) = 256·(1 − (1 − 6/256)⁶) = 33.96 routed experts per layer instead of min(E, 6·6) = 36 (−5.7 %
 * expert bytes): RDMA + DSpark 201.80 → 206.40 tok/s, TCP + DSpark 150.02 → 152.55 (no-copy 150.12 → 152.65); the
 * batch-1 rows without spec-dec (U(1) = k) and every prefill (U = E) are unchanged.
 */
const PINNED: Record<keyof ReturnType<typeof rows>, Partial<Row>> = {
  /** TB5 RDMA, α 50 → 7 µs: each of the ≈ 167 (182 with DSpark) TP all-reduce steps per token pays 43 µs less α. */
  rdmaDspark: { decodeTokPerSec: 206.3956, prefillTokPerSec: 5154.225, commLatencySec: 1.274e-3 },
  tcpDsparkNoCopy: {
    decodeTokPerSec: 152.6515,
    prefillTokPerSec: 5133.7273,
    commLatencySec: 8.19e-3,
  },
  /** + the socket copy: +18.3 µs of α per token (DSpark's verify steps carry larger messages). */
  tcpDspark: {
    decodeTokPerSec: 152.5465,
    prefillTokPerSec: 5133.6106,
    commLatencySec: 8.208275e-3,
  },
  rdma: { decodeTokPerSec: 115.9919, prefillTokPerSec: 5198.503, commLatencySec: 1.169e-3 },
  tcpNoCopy: { decodeTokPerSec: 66.81235, prefillTokPerSec: 5177.6523, commLatencySec: 7.515e-3 },
  /** + the socket copy: +2.8 µs of α per token. */
  tcp: { decodeTokPerSec: 66.79964, prefillTokPerSec: 5177.5336, commLatencySec: 7.517847e-3 },
};

describe('TB5 RDMA vs TCP repro (DeepSeek-V4.1-Flash, 2 × M5 Ultra 64c, TP 2)', () => {
  it('report (SCENARIO_REPORT=1)', () => {
    const r = rows();
    if (process.env.SCENARIO_REPORT === '1') console.log(JSON.stringify(r, null, 2));
    expect(r.rdma.link).toBe('thunderbolt-5-rdma-macos-26-2');
    expect(r.tcp.link).toBe('thunderbolt-5-bridge-tcp');
  });

  it('RDMA decodes at least as fast as TCP over the same cable, with and without DSpark', () => {
    const r = rows();
    expect(r.rdmaDspark.decodeTokPerSec).toBeGreaterThanOrEqual(r.tcpDspark.decodeTokPerSec);
    expect(r.rdma.decodeTokPerSec).toBeGreaterThanOrEqual(r.tcp.decodeTokPerSec);
    expect(r.rdmaDspark.commLatencySec).toBeLessThan(r.tcpDspark.commLatencySec);
    expect(r.rdma.commLatencySec).toBeLessThan(r.tcp.commLatencySec);
  });

  it('the socket copy only adds α (β unchanged) and moves decode by under 0.2 %', () => {
    const r = rows();
    for (const [withCopy, without] of [
      [r.tcpDspark, r.tcpDsparkNoCopy],
      [r.tcp, r.tcpNoCopy],
    ] as const) {
      expect(withCopy.commLatencySec).toBeGreaterThan(without.commLatencySec);
      expect(withCopy.commBandwidthSec).toBe(without.commBandwidthSec);
      expect(1 - withCopy.decodeTokPerSec / without.decodeTokPerSec).toBeLessThan(2e-3);
    }
  });

  it('pins the rows (decode and prefill @ 8k)', () => {
    const r = rows();
    for (const [name, pinned] of Object.entries(PINNED)) {
      const actual = r[name as keyof typeof r];
      expect(actual.decodeContextLen, name).toBe(8192);
      expect(actual.promptLen, name).toBe(8192);
      for (const [field, value] of Object.entries(pinned)) {
        expectRel(actual[field as keyof Row] as number, value as number, `${name}.${field}`);
      }
    }
  });
});
