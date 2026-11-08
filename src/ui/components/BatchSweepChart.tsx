import {
  formatContextTokens,
  formatRateAtContext,
  formatTokensPerSecCompact,
  type SweepPoint,
} from '../state';
import { ChartTooltip, useChartPoint } from './ChartTooltip';

export function BatchSweepChart({ points }: { points: SweepPoint[] }) {
  const max = Math.max(...points.map((p) => p.decodeTokensPerSec), 1);
  const x = (n: number) =>
    60 + (Math.log2(n) / Math.log2(Math.max(...points.map((p) => p.batch), 2))) * 600;
  const y = (n: number) => 155 - (n / max) * 115;
  const chart = useChartPoint(
    points.map((p) => x(p.batch)),
    700,
  );
  const label = (p: SweepPoint) =>
    `batch ${p.batch} · ${formatRateAtContext(p.decodeTokensPerSec, p.contextTokens)}${p.fits ? '' : ' · does not fit in memory'}`;
  const active = chart.active === null ? undefined : points[chart.active];
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
          <path className="axis" d="M60 30V155H678" />
          {[0, max / 2, max].map((value) => (
            <g key={value}>
              <path className="grid" d={`M60 ${y(value)}H678`} />
              <text x="52" y={y(value) + 4} textAnchor="end">
                {formatTokensPerSecCompact(value)}
              </text>
            </g>
          ))}
          <text x="60" y="18">
            aggregate tok/s @ {formatContextTokens(points[0]?.contextTokens ?? 0)}
          </text>
          <polyline
            className="sweep-line"
            points={points.map((p) => `${x(p.batch)},${y(p.decodeTokensPerSec)}`).join(' ')}
          />
          {active && (
            <path
              className="chart-crosshair"
              d={`M${x(active.batch)} 30V155M60 ${y(active.decodeTokensPerSec)}H678`}
            />
          )}
          {points.map((p, index) => (
            <g key={p.batch}>
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
          <text x="360" y="205" textAnchor="middle">
            Batch size (sequences)
          </text>
        </svg>
        {active && <ChartTooltip id={chart.id}>{label(active)}</ChartTooltip>}
      </div>
    </section>
  );
}
