import { useState } from 'react';
import { HelpIcon } from './HelpIcon';
import { Num } from './Num';
import {
  BREAKDOWN_BOUND_LABELS,
  formatSeconds,
  formatShare,
  formatContextTokens,
  type BreakdownBound,
  type BreakdownRow,
  type ResultSummary,
} from '../state';

/** Bounds in legend and bar order; `transfer` only shows when the result has offload transfers. */
const BOUNDS: BreakdownBound[] = ['compute', 'memory', 'commLatency', 'commBandwidth', 'transfer'];

/** CSS class of a bound (bar segment `.<class>`, text `.<class>-text`). */
const BOUND_CLASS: Record<BreakdownBound, string> = {
  compute: 'compute',
  memory: 'memory',
  commLatency: 'comm-latency',
  commBandwidth: 'comm-bandwidth',
  transfer: 'transfer',
};

const VISIBLE_ROWS = 12;

export function BreakdownTable({ result }: { result: ResultSummary }) {
  const [phase, setPhase] = useState<'prefill' | 'decode'>('decode');
  const [group, setGroup] = useState<'byOp' | 'byKind' | 'byLayer'>('byOp');
  const [all, setAll] = useState(false);
  const rows = result.breakdown[phase][group];
  const sections = group === 'byLayer' ? result.breakdown[phase].sections : undefined;
  const maxShare = Math.max(...rows.map((row) => row.share), 0);
  const hasTransfer = (['prefill', 'decode'] as const).some((p) =>
    result.breakdown[p].byKind.some((r) => r.boundSec.transfer > 0),
  );
  const bounds = BOUNDS.filter((b) => b !== 'transfer' || hasTransfer);
  const cut = (list: BreakdownRow[]) => list.slice(0, all ? undefined : VISIBLE_ROWS);
  // Without sections one body holds every row; a speculative decode by layer gets one body per section, each cut
  // on its own so the short draft section stays visible under a long verify section.
  const bodies: { section?: BreakdownRow; rows: BreakdownRow[] }[] =
    sections === undefined
      ? [{ rows: cut(rows) }]
      : sections
          .map((section) => ({
            section,
            rows: cut(rows.filter((r) => r.section === section.key)),
          }))
          .filter((body) => body.rows.length > 0);
  const shown = bodies.reduce((n, body) => n + body.rows.length, 0);
  return (
    <section className="breakdown">
      <div className="section-head">
        <h2>
          Bottleneck estimation{' '}
          <HelpIcon term="Bottleneck estimation">
            Summed op time before overlap, across all stages. Comm latency: the per-message latency
            terms of collectives and stage sends. Comm bandwidth: their bytes over the link
            bandwidth. Offload transfer: reads of off-device weights over the host link.
          </HelpIcon>
        </h2>
        <div className="toggles">
          <div className="segmented" aria-label="Phase">
            {(['decode', 'prefill'] as const).map((p) => (
              <button aria-pressed={phase === p} onClick={() => setPhase(p)} key={p}>
                {p === 'decode' ? 'Decode' : 'Prefill'} @{' '}
                {formatContextTokens(result[p].contextTokens)}
              </button>
            ))}
          </div>
          <div className="segmented" aria-label="Group by">
            {(['byOp', 'byKind', 'byLayer'] as const).map((g, i) => (
              <button
                aria-pressed={group === g}
                onClick={() => {
                  setGroup(g);
                  setAll(false);
                }}
                key={g}
              >
                {['Op', 'Kind', 'Layer'][i]}
              </button>
            ))}
          </div>
        </div>
      </div>
      <ul className="bound-legend" aria-label="Limiting resource">
        {bounds.map((bound) => (
          <li className={`${BOUND_CLASS[bound]}-text`} key={bound}>
            <span aria-hidden="true" />
            {BREAKDOWN_BOUND_LABELS[bound]}
          </li>
        ))}
      </ul>
      <table>
        <thead>
          <tr>
            <th scope="col">{group === 'byLayer' ? 'Layer' : 'Operation'}</th>
            <th scope="col">Time</th>
            <th scope="col">Share</th>
            <th scope="col">Bound</th>
          </tr>
        </thead>
        {bodies.map((body) => (
          <tbody key={body.section?.key ?? 'rows'}>
            {body.section && (
              <tr className="breakdown-section" data-testid="breakdown-section">
                <th scope="rowgroup">{body.section.key}</th>
                <td>
                  <Num>{formatSeconds(body.section.sec)}</Num>
                </td>
                <td>
                  <Num>{formatShare(body.section.share)}</Num>
                </td>
                <td />
              </tr>
            )}
            {body.rows.map((r) => {
              const bound = bounds.reduce((a, b) => (r.boundSec[a] >= r.boundSec[b] ? a : b));
              return (
                <tr key={r.key}>
                  <th scope="row">{r.key}</th>
                  <td>
                    <Num>{formatSeconds(r.sec)}</Num>
                  </td>
                  <td>
                    <Num>{formatShare(r.share)}</Num>
                  </td>
                  <td>
                    <div className="bound-cell">
                      <div className="bound-track">
                        <div
                          className="bound-bar"
                          style={{ width: `${maxShare ? (r.share / maxShare) * 100 : 0}%` }}
                        >
                          {bounds.map((b) => (
                            <span
                              className={BOUND_CLASS[b]}
                              key={b}
                              style={{ width: `${r.sec ? (r.boundSec[b] / r.sec) * 100 : 0}%` }}
                            />
                          ))}
                        </div>
                      </div>
                      <span className={`${BOUND_CLASS[bound]}-text`}>
                        {BREAKDOWN_BOUND_LABELS[bound]}
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        ))}
      </table>
      {(all || shown < rows.length) && rows.length > VISIBLE_ROWS && (
        <button className="quiet" onClick={() => setAll(!all)}>
          {all ? 'Show fewer' : `Show all ${rows.length} rows`}
        </button>
      )}
    </section>
  );
}
