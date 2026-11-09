import {
  formatContextTokens,
  formatTokensPerSec,
  formatTokensPerSecCompact,
  type SweepPoint,
} from '../state';
import { ChartTooltip, useChartPoint } from './ChartTooltip';

const PLOT_LEFT = 60;
const PLOT_RIGHT = 630;
const AXIS_RIGHT = 640;

/** Marker half-size of the per-stream series (square), in viewBox units. */
const SQUARE = 5;

export function BatchSweepChart({ points }: { points: SweepPoint[] }) {
  const maxAgg = Math.max(...points.map((p) => p.decodeTokensPerSec), 1);
  const maxStream = Math.max(...points.map((p) => p.perSeqTokensPerSec), 1);
  const x = (n: number) =>
    PLOT_LEFT +
    (Math.log2(n) / Math.log2(Math.max(...points.map((p) => p.batch), 2))) *
      (PLOT_RIGHT - PLOT_LEFT - 10);
  /** Left axis: aggregate tok/s. */
  const y = (n: number) => 155 - (n / maxAgg) * 115;
  /** Right axis: per-stream tok/s, its own scale. */
  const yStream = (n: number) => 155 - (n / maxStream) * 115;
  const chart = useChartPoint(
    points.map((p) => x(p.batch)),
    700,
  );
  const label = (p: SweepPoint) =>
    `batch ${p.batch} · ${formatTokensPerSec(p.decodeTokensPerSec)} tok/s aggregate · ${formatTokensPerSec(p.perSeqTokensPerSec)} per stream @ ${formatContextTokens(p.contextTokens)}${p.fits ? '' : ' · does not fit in memory'}`;
  const active = chart.active === null ? undefined : points[chart.active];
  const context = formatContextTokens(points[0]?.contextTokens ?? 0);
  return (
    <section className="sweep">
      <h2>Decode throughput by batch size</h2>
      <div className="interactive-chart" data-chart="sweep">
        <svg
          viewBox="0 0 700 210"
          role="group"
          aria-label="Decode throughput by batch size"
          {...chart.chartProps}
        >
          <path className="axis" d={`M${PLOT_LEFT} 30V155H${AXIS_RIGHT}M${AXIS_RIGHT} 30V155`} />
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <path className="grid" d={`M${PLOT_LEFT} ${y(maxAgg * f)}H${AXIS_RIGHT}`} />
              <text x={PLOT_LEFT - 8} y={y(maxAgg * f) + 4} textAnchor="end">
                {formatTokensPerSecCompact(maxAgg * f)}
              </text>
              <text
                className="sweep-right-tick"
                x={AXIS_RIGHT + 8}
                y={yStream(maxStream * f) + 4}
                textAnchor="start"
              >
                {formatTokensPerSecCompact(maxStream * f)}
              </text>
            </g>
          ))}
          <text x={PLOT_LEFT} y="18">
            aggregate tok/s @ {context}
          </text>
          <text className="sweep-right-title" x={AXIS_RIGHT + 50} y="18" textAnchor="end">
            per-stream tok/s @ {context}
          </text>
          <polyline
            className="sweep-line"
            points={points.map((p) => `${x(p.batch)},${y(p.decodeTokensPerSec)}`).join(' ')}
          />
          <polyline
            className="sweep-line-stream"
            points={points.map((p) => `${x(p.batch)},${yStream(p.perSeqTokensPerSec)}`).join(' ')}
          />
          {active && <path className="chart-crosshair" d={`M${x(active.batch)} 30V155`} />}
          {points.map((p, index) => (
            <g key={p.batch}>
              <rect
                className={`sweep-point-stream ${p.fits ? 'fits' : ''} ${chart.active === index ? 'is-active' : ''}`}
                x={x(p.batch) - SQUARE}
                y={yStream(p.perSeqTokensPerSec) - SQUARE}
                width={SQUARE * 2}
                height={SQUARE * 2}
              />
              <circle
                {...chart.pointProps(index, label(p))}
                className={`sweep-point ${p.fits ? 'fits' : ''} ${chart.active === index ? 'is-active' : ''}`}
                cx={x(p.batch)}
                cy={y(p.decodeTokensPerSec)}
                r="6"
              />
              <text x={x(p.batch)} y="177" textAnchor="middle">
                {p.batch}
              </text>
            </g>
          ))}
          <text x="335" y="205" textAnchor="middle">
            Batch size (sequences)
          </text>
        </svg>
        <ul className="sweep-legend" aria-label="Legend">
          <li>
            <svg viewBox="0 0 28 12" aria-hidden="true">
              <path className="sweep-line" d="M0 6H28" />
              <circle className="sweep-point fits" cx="14" cy="6" r="4" />
            </svg>
            Aggregate
          </li>
          <li>
            <svg viewBox="0 0 28 12" aria-hidden="true">
              <path className="sweep-line-stream" d="M0 6H28" />
              <rect className="sweep-point-stream fits" x="9" y="1" width="10" height="10" />
            </svg>
            Per stream
          </li>
        </ul>
        {active && <ChartTooltip id={chart.id}>{label(active)}</ChartTooltip>}
      </div>
    </section>
  );
}
