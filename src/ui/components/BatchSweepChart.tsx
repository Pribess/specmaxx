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
/** Outer edges of the axis titles (left title starts here, right title ends here). */
const TITLE_LEFT = 10;
const TITLE_RIGHT = 690;
/** Width of the series sample drawn next to each axis title, and its gap to the text. */
const SAMPLE_W = 24;
const SAMPLE_GAP = 6;

/** Marker half-size of the per-stream series (square), in viewBox units. */
const SQUARE = 5;
/** Baseline of the break-even label, inside the plot below the axis titles; and its gap to the marker line. */
const BREAK_EVEN_LABEL_Y = 46;
const BREAK_EVEN_LABEL_GAP = 6;

/** Text of the break-even marker (also its accessible title). */
export function breakEvenText(batch: number): string {
  return `break-even batch ${batch}`;
}

export function BatchSweepChart({
  points,
  breakEvenBatch,
}: {
  points: SweepPoint[];
  /** Speculative decoding on: the smallest batch where it stops paying (sequences); absent = no marker. */
  breakEvenBatch?: number;
}) {
  const offPoints = points.flatMap((p) =>
    p.offDecodeTokensPerSec === undefined ? [] : [{ batch: p.batch, off: p.offDecodeTokensPerSec }],
  );
  const hasOff = offPoints.length > 0;
  const maxAgg = Math.max(
    ...points.map((p) => p.decodeTokensPerSec),
    ...offPoints.map((p) => p.off),
    1,
  );
  const maxStream = Math.max(...points.map((p) => p.perSeqTokensPerSec), 1);
  const x = (n: number) =>
    PLOT_LEFT +
    (Math.log2(n) / Math.log2(Math.max(...points.map((p) => p.batch), 2))) *
      (PLOT_RIGHT - PLOT_LEFT - 10);
  /** Left axis: per-stream tok/s. */
  const yStream = (n: number) => 155 - (n / maxStream) * 115;
  /** Right axis: aggregate tok/s, its own scale. */
  const y = (n: number) => 155 - (n / maxAgg) * 115;
  const chart = useChartPoint(
    points.map((p) => x(p.batch)),
    700,
  );
  const specText = (p: SweepPoint) =>
    p.offDecodeTokensPerSec === undefined || p.speedup === undefined
      ? ''
      : ` · ${formatTokensPerSec(p.offDecodeTokensPerSec)} without spec-dec · ×${p.speedup.toFixed(2)}`;
  const label = (p: SweepPoint) =>
    `batch ${p.batch} · ${formatTokensPerSec(p.perSeqTokensPerSec)} tok/s per stream · ${formatTokensPerSec(p.decodeTokensPerSec)} aggregate${specText(p)} @ ${formatContextTokens(p.contextTokens)}${p.fits ? '' : ' · does not fit in memory'}`;
  const active = chart.active === null ? undefined : points[chart.active];
  const context = formatContextTokens(points[0]?.contextTokens ?? 0);
  const breakEvenX = breakEvenBatch === undefined ? undefined : x(breakEvenBatch);
  // The label sits on the side of the marker with more room.
  const breakEvenLeft = breakEvenX !== undefined && breakEvenX > (PLOT_LEFT + AXIS_RIGHT) / 2;
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
              <path className="grid" d={`M${PLOT_LEFT} ${yStream(maxStream * f)}H${AXIS_RIGHT}`} />
              <text
                className="sweep-stream-tick"
                x={PLOT_LEFT - 8}
                y={yStream(maxStream * f) + 4}
                textAnchor="end"
              >
                {formatTokensPerSecCompact(maxStream * f)}
              </text>
              <text
                className="sweep-agg-tick"
                x={AXIS_RIGHT + 8}
                y={y(maxAgg * f) + 4}
                textAnchor="start"
              >
                {formatTokensPerSecCompact(maxAgg * f)}
              </text>
            </g>
          ))}
          <g className="sweep-title-sample" aria-hidden="true">
            <path className="sweep-line-stream" d={`M${TITLE_LEFT} 13.5h${SAMPLE_W}`} />
            <rect
              className="sweep-point-stream fits"
              x={TITLE_LEFT + SAMPLE_W / 2 - 4}
              y="9.5"
              width="8"
              height="8"
            />
          </g>
          <text
            className="sweep-stream-title"
            x={TITLE_LEFT + SAMPLE_W + SAMPLE_GAP}
            y="18"
            textAnchor="start"
          >
            per stream, tok/s @ {context}
          </text>
          <text
            className="sweep-agg-title"
            x={TITLE_RIGHT - SAMPLE_W - SAMPLE_GAP}
            y="18"
            textAnchor="end"
          >
            aggregate, tok/s @ {context}
          </text>
          <g className="sweep-title-sample" aria-hidden="true">
            <path className="sweep-line" d={`M${TITLE_RIGHT - SAMPLE_W} 13.5h${SAMPLE_W}`} />
            <circle className="sweep-point fits" cx={TITLE_RIGHT - SAMPLE_W / 2} cy="13.5" r="4" />
          </g>
          {breakEvenBatch !== undefined && breakEvenX !== undefined && (
            <g className="sweep-break-even" data-testid="specdec-break-even">
              <title>{breakEvenText(breakEvenBatch)}</title>
              <path className="sweep-break-even-line" d={`M${breakEvenX} 30V155`} />
              <text
                className="sweep-break-even-label"
                x={breakEvenX + (breakEvenLeft ? -BREAK_EVEN_LABEL_GAP : BREAK_EVEN_LABEL_GAP)}
                y={BREAK_EVEN_LABEL_Y}
                textAnchor={breakEvenLeft ? 'end' : 'start'}
              >
                {breakEvenText(breakEvenBatch)}
              </text>
            </g>
          )}
          {hasOff && (
            <polyline
              className="sweep-line-off"
              points={offPoints.map((p) => `${x(p.batch)},${y(p.off)}`).join(' ')}
            />
          )}
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
              <path className="sweep-line-stream" d="M0 6H28" />
              <rect className="sweep-point-stream fits" x="9" y="1" width="10" height="10" />
            </svg>
            Per stream (left axis)
          </li>
          <li>
            <svg viewBox="0 0 28 12" aria-hidden="true">
              <path className="sweep-line" d="M0 6H28" />
              <circle className="sweep-point fits" cx="14" cy="6" r="4" />
            </svg>
            Aggregate (right axis)
          </li>
          {hasOff && (
            <li data-testid="sweep-legend-off">
              <svg viewBox="0 0 28 12" aria-hidden="true">
                <path className="sweep-line-off" d="M0 6H28" />
              </svg>
              Aggregate, spec-dec off (right axis)
            </li>
          )}
        </ul>
        {active && <ChartTooltip id={chart.id}>{label(active)}</ChartTooltip>}
      </div>
    </section>
  );
}
