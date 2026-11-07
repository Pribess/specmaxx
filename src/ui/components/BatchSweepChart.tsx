import { formatTokensPerSec, formatTokensPerSecCompact, type SweepPoint } from '../state';

export function BatchSweepChart({ points }: { points: SweepPoint[] }) {
  const max = Math.max(...points.map((p) => p.decodeTokensPerSec), 1);
  const x = (n: number) =>
    48 + (Math.log2(n) / Math.log2(Math.max(...points.map((p) => p.batch), 2))) * 620;
  const y = (n: number) => 155 - (n / max) * 125;
  return (
    <section className="sweep">
      <h2>Decode throughput by batch size</h2>
      <svg
        viewBox="0 0 700 200"
        role="img"
        aria-label="Decode tokens per second by batch size. Hollow points do not fit in memory."
      >
        <path className="axis" d="M48 20V155H678" />
        <polyline
          className="sweep-line"
          points={points.map((p) => `${x(p.batch)},${y(p.decodeTokensPerSec)}`).join(' ')}
        />
        {points.map((p) => (
          <g key={p.batch}>
            <circle
              className={`sweep-point ${p.fits ? 'fits' : ''}`}
              cx={x(p.batch)}
              cy={y(p.decodeTokensPerSec)}
              r="4"
            >
              <title>
                Batch {p.batch}: {formatTokensPerSec(p.decodeTokensPerSec)} tok/s,{' '}
                {p.fits ? 'fits' : 'does not fit'}
              </title>
            </circle>
            <text x={x(p.batch)} y="177" textAnchor="middle">
              {p.batch}
            </text>
          </g>
        ))}
        {[0, max / 2, max].map((value) => (
          <g key={value}>
            <path className="grid" d={`M48 ${y(value)}H678`} />
            <text x="40" y={y(value) + 4} textAnchor="end">
              {formatTokensPerSecCompact(value)}
            </text>
          </g>
        ))}
        <text x="360" y="198" textAnchor="middle">
          Batch size
        </text>
      </svg>
      <p className="hint">Aggregate tok/s. Hollow points do not fit in memory.</p>
    </section>
  );
}
