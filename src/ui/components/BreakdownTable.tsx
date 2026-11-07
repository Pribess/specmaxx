import { useState } from 'react';
import { formatSeconds, formatShare, type ResultSummary } from '../state';

export function BreakdownTable({ result }: { result: ResultSummary }) {
  const [phase, setPhase] = useState<'prefill' | 'decode'>('decode');
  const [group, setGroup] = useState<'byOp' | 'byKind' | 'byLayer'>('byOp');
  const [all, setAll] = useState(false);
  const rows = result.breakdown[phase][group];
  const maxShare = Math.max(...rows.map((row) => row.share), 0);
  return (
    <section className="breakdown">
      <div className="section-head">
        <h2>Where the time goes</h2>
        <div className="toggles">
          <div className="segmented" aria-label="Phase">
            {(['decode', 'prefill'] as const).map((p) => (
              <button aria-pressed={phase === p} onClick={() => setPhase(p)} key={p}>
                {p === 'decode' ? 'Decode' : 'Prefill'}
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
      <p className="hint">
        Summed op time before overlap, across all stages. Colour shows the limiting resource.
      </p>
      <table>
        <thead>
          <tr>
            <th scope="col">{group === 'byLayer' ? 'Layer' : 'Operation'}</th>
            <th scope="col">Time</th>
            <th scope="col">Share</th>
            <th scope="col">Bound</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, all ? undefined : 12).map((r) => {
            const bound = (['compute', 'memory', 'comm'] as const).reduce((a, b) =>
              r.boundSec[a] >= r.boundSec[b] ? a : b,
            );
            return (
              <tr key={r.key}>
                <th scope="row">{r.key}</th>
                <td>{formatSeconds(r.sec)}</td>
                <td>{formatShare(r.share)}</td>
                <td>
                  <div className="bound-cell">
                    <div className="bound-track">
                      <div
                        className="bound-bar"
                        style={{ width: `${maxShare ? (r.share / maxShare) * 100 : 0}%` }}
                      >
                        {(['compute', 'memory', 'comm'] as const).map((b) => (
                          <span
                            className={b}
                            key={b}
                            style={{ width: `${r.sec ? (r.boundSec[b] / r.sec) * 100 : 0}%` }}
                          />
                        ))}
                      </div>
                    </div>
                    <span className={`${bound}-text`}>{bound}</span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {rows.length > 12 && (
        <button className="quiet" onClick={() => setAll(!all)}>
          {all ? 'Show fewer' : `Show all ${rows.length} rows`}
        </button>
      )}
    </section>
  );
}
