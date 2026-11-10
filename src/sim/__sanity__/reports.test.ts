/**
 * The sanity check (docs/sanity-check.md): every community report in `reports.ts` is simulated, and each asserted
 * metric must satisfy `simulated ≥ 0.9 × reported` (the simulator is a theoretical upper bound). Known exceptions
 * must STILL fail (so the list stays honest); excluded metrics are only printed. The ratio report ÷ sim is printed
 * for every metric, with a summary of its distribution. Reports that ran with speculative decoding or offloading are
 * simulated with the same knobs on (`input.deployment`), so their decode is asserted against the spec-dec / offload
 * bound (docs/simulation-model.md §12–§13).
 */

import { describe, expect, it } from 'vitest';
import { sanityDrafterSpec, sanityRepoFiles } from './configs.testutil';
import {
  simulateReport,
  type SanityDeployment,
  type SanityModel,
  type SanityResult,
} from './harness';
import {
  REPORTS,
  UPPER_BOUND_OPTIONS,
  type BenchmarkReport,
  type ReportDeployment,
  type ReportModel,
  type ReportedRate,
} from './reports';

/** Required margin: the simulated rate must be at least this fraction of the reported one. Dimensionless. */
const MARGIN = 0.9;

/** The harness model of a report model (HF fixtures are loaded here: the loader needs vite). */
function toSanityModel(m: ReportModel): SanityModel {
  return m.kind === 'spec'
    ? m
    : { kind: 'hf', files: sanityRepoFiles(m.repoId, m.dir ? { dir: m.dir } : {}) };
}

/** The harness deployment of a report's: a drafter repo id becomes the parsed drafter fixture. */
function toSanityDeployment(d: ReportDeployment): SanityDeployment {
  const s = d.speculative;
  if (s?.drafter?.kind !== 'repo') return d as SanityDeployment;
  const { repoId } = s.drafter;
  return {
    ...d,
    speculative: { ...s, drafter: { kind: 'repo', repoId, spec: sanityDrafterSpec(repoId) } },
  };
}

/** Simulates one report with the shared upper-bound options. */
function simulate(r: BenchmarkReport): SanityResult {
  const { deployment, ...input } = r.input;
  return simulateReport({
    ...input,
    model: toSanityModel(r.input.model),
    options: { ...UPPER_BOUND_OPTIONS, ...r.input.options },
    ...(deployment ? { deployment: toSanityDeployment(deployment) } : {}),
  });
}

/** The simulated rate matching a reported one (same phase, same per-sequence / aggregate kind). Unit: tokens/s. */
function simulatedRate(sim: SanityResult, phase: 'prefill' | 'decode', rate: ReportedRate): number {
  if (phase === 'prefill')
    return rate.kind === 'perSequence'
      ? sim.prefillPerSequenceTokensPerSec
      : sim.prefillTokensPerSec;
  return rate.kind === 'perSequence' ? sim.decodePerSequenceTokensPerSec : sim.decodeTokensPerSec;
}

/** One compared metric: reported, simulated, ratio = reported / simulated. */
interface Row {
  id: string;
  phase: 'prefill' | 'decode';
  reported: number;
  simulated: number;
  ratio: number;
  mode: 'assert' | 'exception' | 'excluded';
  bound: string;
}

const SIMS = new Map(REPORTS.map((r) => [r.id, simulate(r)]));
const ROWS: Row[] = REPORTS.flatMap((r) => {
  const sim = SIMS.get(r.id)!;
  return (['prefill', 'decode'] as const).flatMap((phase) => {
    const rate = r[phase];
    if (rate === undefined) return [];
    const simulated = simulatedRate(sim, phase, rate);
    const bound = phase === 'prefill' ? sim.prefillBound.dominant : sim.decodeBound.dominant;
    return [
      {
        id: r.id,
        phase,
        reported: rate.tokensPerSec,
        simulated,
        ratio: rate.tokensPerSec / simulated,
        mode: rate.check?.mode ?? 'assert',
        bound,
      },
    ];
  });
});

/** Quantile `q` ∈ [0, 1] of sorted values (linear interpolation). */
function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

describe('community reports vs simulated theoretical maxima', () => {
  it('prints the comparison table and the ratio distribution', () => {
    const lines = ROWS.map(
      (r) =>
        `${r.id.padEnd(7)} ${r.phase.padEnd(7)} reported ${r.reported.toFixed(1).padStart(9)}  sim ${r.simulated.toFixed(1).padStart(10)}` +
        `  ratio ${r.ratio.toFixed(3)}  ${r.bound}${r.mode === 'assert' ? '' : `  [${r.mode}]`}`,
    );
    const stats = (['prefill', 'decode'] as const).map((phase) => {
      const v = ROWS.filter((r) => r.phase === phase && r.mode !== 'excluded')
        .map((r) => r.ratio)
        .sort((a, b) => a - b);
      const buckets = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, Infinity];
      const hist = buckets
        .slice(0, -1)
        .map(
          (lo, i) =>
            `[${lo}, ${buckets[i + 1]}): ${v.filter((x) => x >= lo && x < buckets[i + 1]).length}`,
        );
      return (
        `${phase}: n ${v.length}, min ${v[0].toFixed(3)}, p25 ${quantile(v, 0.25).toFixed(3)}, median ${quantile(v, 0.5).toFixed(3)}, ` +
        `p75 ${quantile(v, 0.75).toFixed(3)}, max ${v[v.length - 1].toFixed(3)} | ${hist.join(' ')}`
      );
    });
    console.log([...lines, ...stats].join('\n'));
    expect(ROWS.length).toBeGreaterThanOrEqual(30);
  });

  it.each(REPORTS.map((r) => [r.id, r] as const))('%s: simulated ≥ 0.9 × reported', (_id, r) => {
    const sim = SIMS.get(r.id)!;
    for (const phase of ['prefill', 'decode'] as const) {
      const rate = r[phase];
      if (rate === undefined) continue;
      const simulated = simulatedRate(sim, phase, rate);
      expect(Number.isFinite(simulated) && simulated > 0, `${r.id} ${phase}: simulated rate`).toBe(
        true,
      );
      const holds = simulated >= MARGIN * rate.tokensPerSec;
      const check = rate.check ?? { mode: 'assert' as const };
      if (check.mode === 'assert')
        expect(
          holds,
          `${r.id} ${phase}: sim ${simulated.toFixed(1)} vs reported ${rate.tokensPerSec}`,
        ).toBe(true);
      if (check.mode === 'exception')
        expect(holds, `${r.id} ${phase}: known exception now passes (${check.reason})`).toBe(false);
    }
  });

  it.each(REPORTS.filter((r) => r.weightFileBytes !== undefined).map((r) => [r.id, r] as const))(
    '%s: simulated resident weight bytes are within 8% of the weight file',
    (_id, r) => {
      const sim = SIMS.get(r.id)!;
      const tp = r.input.parallel?.tp ?? 1;
      // Formula: total weights = Σ_stages (per-device weight bytes × tp)  (each stage's tp devices are symmetric)
      const totalWeightBytes = sim.sim.memory.perStage.reduce(
        (sum, s) => sum + s.weightBytes * tp,
        0,
      );
      const fileBytes = r.weightFileBytes!;
      expect(
        Math.abs(totalWeightBytes / fileBytes - 1),
        `${r.id}: sim ${totalWeightBytes / 1e9} GB vs file ${fileBytes / 1e9} GB`,
      ).toBeLessThan(0.08);
    },
  );

  it.each(REPORTS.map((r) => [r.id, r] as const))(
    '%s: decode is simulated at the recorded context',
    (_id, r) => {
      // The recorded depth and the simulated one agree, and decode runs at depth + ceil(genLen / 2).
      expect(r.input.workload.contextDepth ?? 0).toBe(r.contextDepth);
      expect(SIMS.get(r.id)!.decodeContextLen).toBe(
        Math.max(1, r.contextDepth + Math.ceil(r.genLen / 2)),
      );
      // Only llama-bench's tg test (and llama-sweep-bench's, at each N_KV) starts from an empty cache; everywhere else
      // generation follows the prompt.
      const emptyCacheTg =
        r.engine.includes('llama-bench') || r.engine.includes('llama-sweep-bench');
      if (!emptyCacheTg)
        expect(r.contextDepth, `${r.id}: decode should follow the prompt`).toBeGreaterThanOrEqual(
          r.promptLen,
        );
    },
  );

  it('web-26 (mlx_lm.benchmark -p 2048 -g 128) decodes after its 2048-token prompt: context 2048 + 64', () => {
    expect(SIMS.get('web-26')!.decodeContextLen).toBe(2112);
  });

  it('spec-dec rows run with the knobs on: lm-09 MTP k 8, lm-36 DSpark k 5 at the reported E[τ] 2.85, lm-39 DFlash 2 k 7', () => {
    const spec = (id: string) => SIMS.get(id)!.sim.decode.speculative;
    expect(spec('lm-09')).toBeDefined();
    // Formula: E[τ] = (1 − α^(k+1)) / (1 − α) at α 0.75, k 8 = 3.6997
    expect(spec('lm-09')!.expectedTokensPerStep).toBeCloseTo((1 - 0.75 ** 9) / 0.25, 9);
    expect(spec('lm-36')!.expectedTokensPerStep).toBeCloseTo(2.85, 9);
    // Formula: E[τ] at α 0.79, k 7 = (1 − 0.79^8) / 0.21 = 4.0395
    expect(spec('lm-39')!.expectedTokensPerStep).toBeCloseTo((1 - 0.79 ** 8) / 0.21, 9);
    for (const id of ['lm-09', 'lm-36', 'lm-39']) expect(spec(id)!.speedup, id).toBeGreaterThan(1);
  });

  it('offload rows stream experts: web-52 and web-53 have decode transfers', () => {
    for (const id of ['web-52', 'web-53']) {
      const t = SIMS.get(id)!.sim.decode.transfer;
      expect(t, id).toBeDefined();
      expect(t!.bytes, id).toBeGreaterThan(0);
    }
  });
});
