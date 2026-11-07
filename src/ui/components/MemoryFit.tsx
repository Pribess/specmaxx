import type { CSSProperties } from 'react';
import { formatBytes, type ResultSummary } from '../state';

/** Per-device capacity and its three resident/transient allocations. */
export function MemoryFit({ result }: { result: ResultSummary }) {
  return (
    <section className="memory">
      <h2>Memory per device</h2>
      {result.memory.stages.map((stage) => {
        const segments = [
          { key: 'weights', label: 'Weights', bytes: stage.weightBytes },
          { key: 'state', label: 'KV + state', bytes: stage.sequenceStateBytes },
          { key: 'activations', label: 'Activations', bytes: stage.activationPeakBytes },
        ];
        return (
          <div className="memory-stage" key={stage.stage}>
            <p data-testid="memory-verdict" className={!stage.fits ? 'error' : ''}>
              {result.memory.stages.length > 1 && `Stage ${stage.stage + 1}. `}
              {stage.fits ? 'Fits' : 'Does not fit'}: {formatBytes(stage.totalBytes)} of{' '}
              {formatBytes(stage.capacityBytes)}
            </p>
            <div
              className={`capacity-bar ${stage.fits ? '' : 'overflow'}`}
              role="img"
              aria-label={segments
                .map((segment) => `${segment.label} ${formatBytes(segment.bytes)}`)
                .join(', ')}
              style={
                {
                  '--capacity': `${Math.min(100, (stage.capacityBytes / stage.totalBytes) * 100)}%`,
                } as CSSProperties
              }
            >
              {segments.map((segment) => (
                <span
                  key={segment.key}
                  className={`allocation-${segment.key}`}
                  style={{
                    width: `${(segment.bytes / Math.max(stage.capacityBytes, stage.totalBytes)) * 100}%`,
                  }}
                />
              ))}
            </div>
            <ul
              className="memory-legend"
              data-testid="memory-legend"
              aria-label="Memory allocations"
            >
              {segments.map((segment) => (
                <li key={segment.key}>
                  <span className={`memory-swatch allocation-${segment.key}`} aria-hidden="true" />
                  {segment.label} <strong>{(segment.bytes / 1e9).toFixed(2)} GB</strong>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
      <p className="hint">
        Up to{' '}
        {Number.isFinite(result.memory.maxSequencesPerReplica)
          ? result.memory.maxSequencesPerReplica.toLocaleString()
          : 'unlimited'}{' '}
        sequences per replica at this context.
        {result.memory.kind === 'unified' && ' Unified memory: the OS share is not reserved.'}
      </p>
      {!result.memory.fits && (
        <p className="error">
          Throughput is a mathematical ceiling only; this workload cannot run in the selected
          memory.
        </p>
      )}
    </section>
  );
}
