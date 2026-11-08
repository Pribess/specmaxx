import { useId } from 'react';
import { ChartTooltip, useChartPoint } from './ChartTooltip';
import { formatFlopsPerSec, formatContextTokens, type RooflineSummary } from '../state';

export function RooflineChart({
  roof,
  contexts,
}: {
  roof: RooflineSummary;
  contexts: { prefill: number; decode: number };
}) {
  const id = useId();
  const maxI = Math.max(roof.ridgeFlopPerByte * 10, roof.prefill.intensityFlopPerByte * 2, 1000);
  const maxF = Math.max(roof.peakFlopsPerSec * 2, 1e12);
  const minF = Math.min(roof.memBandwidthBytesPerSec * 0.1, roof.decode.attainedFlopsPerSec / 3);
  const x = (i: number) => 48 + ((Math.log10(Math.max(0.1, i)) + 1) / (Math.log10(maxI) + 1)) * 390;
  const y = (f: number) =>
    218 -
    ((Math.log10(Math.max(minF, f)) - Math.log10(minF)) / (Math.log10(maxF) - Math.log10(minF))) *
      188;
  const ridge = Math.min(maxI, Math.max(0.1, roof.ridgeFlopPerByte));
  const phases = ['prefill', 'decode'] as const;
  const chart = useChartPoint(
    phases.map((phase) => x(roof[phase].intensityFlopPerByte)),
    470,
  );
  const label = (phase: (typeof phases)[number]) =>
    `${phase === 'prefill' ? 'Prefill' : 'Decode'} · ${roof[phase].intensityFlopPerByte.toFixed(1)} FLOP/B · ${formatFlopsPerSec(roof[phase].attainedFlopsPerSec)} @ ${formatContextTokens(contexts[phase])}`;
  const active = chart.active === null ? undefined : phases[chart.active];
  const contextLabel =
    contexts.prefill === contexts.decode
      ? formatContextTokens(contexts.decode)
      : `${formatContextTokens(contexts.prefill)} prefill / ${formatContextTokens(contexts.decode)} decode`;
  return (
    <figure className="roofline interactive-chart" data-chart="roofline">
      <svg
        viewBox="0 0 470 280"
        role="group"
        aria-labelledby={`${id}-title ${id}-desc`}
        {...chart.chartProps}
      >
        <title id={`${id}-title`}>The roofline of this device</title>
        <desc id={`${id}-desc`}>
          Logarithmic axes. Decode: {roof.decode.intensityFlopPerByte.toFixed(1)} FLOP per byte at{' '}
          {formatFlopsPerSec(roof.decode.attainedFlopsPerSec)}. Prefill:{' '}
          {roof.prefill.intensityFlopPerByte.toFixed(1)} FLOP per byte at{' '}
          {formatFlopsPerSec(roof.prefill.attainedFlopsPerSec)}. Compute bound:{' '}
          {formatFlopsPerSec(roof.peakFlopsPerSec)}.
        </desc>
        <path className="axis" d="M48 22V218H447" />
        {[1, 10, 100, 1000, 10000]
          .filter((i) => i <= maxI)
          .map((i) => (
            <g key={i}>
              <path className="grid" d={`M${x(i)} 30V218`} />
              <text x={x(i)} y="238" textAnchor="middle">
                {i.toLocaleString()}
              </text>
            </g>
          ))}
        {[1e11, 1e12, 1e13, 1e14, 1e15, 1e16]
          .filter((f) => f > minF && f < maxF)
          .map((f) => (
            <g key={f}>
              <path className="grid" d={`M48 ${y(f)}H438`} />
              <text x="40" y={y(f) + 4} textAnchor="end">
                {f / 1e12}
              </text>
            </g>
          ))}
        <path
          className="roof-memory"
          d={`M${x(0.1)} ${y(roof.memBandwidthBytesPerSec * 0.1)}L${x(ridge)} ${y(roof.peakFlopsPerSec)}`}
        />
        <path className="roof-compute" d={`M${x(ridge)} ${y(roof.peakFlopsPerSec)}H${x(maxI)}`} />
        <text className="compute-text" x="438" y={y(roof.peakFlopsPerSec) - 12} textAnchor="end">
          Compute bound
        </text>
        <text className="memory-text" x="62" y="140">
          Bandwidth bound
        </text>
        <text x="48" y="14">
          TFLOP/s @ {contextLabel}
        </text>
        {active && (
          <path
            className="chart-crosshair"
            d={`M${x(roof[active].intensityFlopPerByte)} 22V218M48 ${y(roof[active].attainedFlopsPerSec)}H447`}
          />
        )}
        {phases.map((phase, index) => {
          const px = x(roof[phase].intensityFlopPerByte);
          const py = y(roof[phase].attainedFlopsPerSec);
          return (
            <g key={phase}>
              <circle
                {...chart.pointProps(index, label(phase))}
                className={`point ${phase} ${chart.active === index ? 'is-active' : ''}`}
                cx={px}
                cy={py}
                r="6"
              />
              <text
                x={px + (px > 380 ? -10 : 10)}
                y={py + 17}
                textAnchor={px > 380 ? 'end' : 'start'}
              >
                {phase === 'prefill' ? 'Prefill' : 'Decode'} @{' '}
                {formatContextTokens(contexts[phase])}
              </text>
            </g>
          );
        })}
        <text x="245" y="268" textAnchor="middle">
          Arithmetic intensity (FLOP / byte)
        </text>
      </svg>
      {active && <ChartTooltip id={chart.id}>{label(active)}</ChartTooltip>}
    </figure>
  );
}
