import type { CSSProperties } from 'react';
import { formatBytes, formatContextTokens, type ResultSummary } from '../state';
import { ChartTooltip, useChartPoint } from './ChartTooltip';
import { HelpIcon } from './HelpIcon';
import { memoryState, memoryUtilisation, MemoryWarningIcon } from './MemoryWarning';

/** Per-device capacity and its three resident/transient allocations. */
export function MemoryFit({ result }: { result: ResultSummary }) {
  const allocations = result.memory.stages.flatMap((stage) => [
    { stage: stage.stage, key: 'weights', label: 'Weights', bytes: stage.weightBytes },
    { stage: stage.stage, key: 'state', label: 'KV + state', bytes: stage.sequenceStateBytes },
    {
      stage: stage.stage,
      key: 'activations',
      label: 'Activations',
      bytes: stage.activationPeakBytes,
    },
  ]);
  const chart = useChartPoint(
    allocations.map((_, i) => i),
    allocations.length,
  );
  const active = chart.active === null ? undefined : allocations[chart.active];
  const context = formatContextTokens(
    Math.max(result.prefill.contextTokens, result.decode.contextTokens),
  );
  const label = (segment: (typeof allocations)[number]) =>
    `Stage ${segment.stage + 1} · ${segment.label} · ${formatBytes(segment.bytes)} @ ${context}`;
  return (
    <section className="memory interactive-chart" data-chart="memory" id="memory-fit" tabIndex={-1}>
      <h2>
        Memory per device{' '}
        <HelpIcon term="Memory fit">
          Memory is sized for the larger prefill or decode workload. Throughput is a mathematical
          ceiling only when the workload does not fit. Unified memory does not reserve the OS share.
        </HelpIcon>
      </h2>
      {result.memory.stages.map((stage, stageIndex) => {
        const segments = allocations.slice(stageIndex * 3, stageIndex * 3 + 3);
        const state = memoryState(stage);
        return (
          <div className="memory-stage" key={stage.stage}>
            <div data-testid="memory-verdict" className={`memory-${state}`}>
              <MemoryWarningIcon stage={stage} />
              {result.memory.stages.length > 1 && `Stage ${stage.stage + 1}. `}
              {stage.fits ? 'Fits' : 'Does not fit'}: {formatBytes(stage.totalBytes)} of{' '}
              {formatBytes(stage.capacityBytes)}
              {state === 'warning' && (
                <>
                  {' '}
                  ({memoryUtilisation(stage)}){' '}
                  <HelpIcon term="High memory utilisation">
                    There is little headroom for the OS, the runtime, and fragmentation. Unified
                    memory shares this pool with the OS.
                  </HelpIcon>
                </>
              )}
            </div>
            <div
              className={`capacity-bar memory-${state} ${stage.fits ? '' : 'overflow'}`}
              role="group"
              aria-label={`Stage ${stage.stage + 1} memory allocations`}
              style={
                {
                  '--capacity': `${Math.min(100, (stage.capacityBytes / stage.totalBytes) * 100)}%`,
                } as CSSProperties
              }
            >
              {segments.map((segment, index) => (
                <span
                  {...chart.pointProps(stageIndex * 3 + index, label(segment))}
                  key={segment.key}
                  className={`allocation-${segment.key} ${chart.active === stageIndex * 3 + index ? 'is-active' : ''}`}
                  style={{
                    width: `${(segment.bytes / Math.max(stage.capacityBytes, stage.totalBytes)) * 100}%`,
                  }}
                  onPointerLeave={() => chart.chartProps.onPointerLeave()}
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
                  {segment.label} <strong>{formatBytes(segment.bytes)}</strong>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
      <p className="memory-limit">
        max{' '}
        {Number.isFinite(result.memory.maxSequencesPerReplica)
          ? result.memory.maxSequencesPerReplica.toLocaleString()
          : 'unlimited'}{' '}
        sequences per replica @ {context}
      </p>
      {active && <ChartTooltip id={chart.id}>{label(active)}</ChartTooltip>}
    </section>
  );
}
