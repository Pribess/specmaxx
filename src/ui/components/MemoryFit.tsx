import type { CSSProperties } from 'react';
import {
  formatBytes,
  formatContextTokens,
  type OffloadPlacement,
  type ResultSummary,
  type StageMemory,
} from '../state';
import type { TierMemory } from '../../sim/engine/types';
import { ChartTooltip, useChartPoint } from './ChartTooltip';
import { HelpIcon } from './HelpIcon';
import { helpLines } from './helpLines';
import { Num } from './Num';
import {
  HIGH_MEMORY_UTILISATION,
  memoryState,
  memoryUtilisation,
  MemoryWarningIcon,
  type MemoryState,
} from './MemoryWarning';

/** One coloured part of a capacity bar. Unit of `bytes`: bytes. */
interface Segment {
  key: string;
  label: string;
  bytes: number;
}

/** Display name of a storage tier. */
const TIER_LABEL: Record<TierMemory['tier'], string> = { host_ram: 'Host RAM', nvme: 'SSD' };

/**
 * The allocations of one stage's device: weights (the n-gram table split out, hatched, when it stays on the
 * device), KV + state, activations, then staging, draft, verify snapshots and the runtime reserve when present.
 */
function stageSegments(stage: StageMemory): Segment[] {
  const table = stage.lookupTableBytes ?? 0;
  const optional: Segment[] = [
    { key: 'table', label: 'n-gram table', bytes: table },
    { key: 'staging', label: 'Staging', bytes: stage.stagingBytes ?? 0 },
    { key: 'draft', label: 'Draft', bytes: stage.draftBytes ?? 0 },
    { key: 'snapshots', label: 'Verify state', bytes: stage.speculativeTransientBytes ?? 0 },
    { key: 'reserve', label: 'Runtime reserve', bytes: stage.runtimeReserveBytes ?? 0 },
  ];
  return [
    { key: 'weights', label: 'Weights', bytes: stage.weightBytes - table },
    optional[0],
    { key: 'state', label: 'KV + state', bytes: stage.sequenceStateBytes },
    { key: 'activations', label: 'Activations', bytes: stage.activationPeakBytes },
    ...optional.slice(1),
  ].filter((s) => s.bytes > 0 || ['weights', 'state', 'activations'].includes(s.key));
}

/**
 * Legend order, labels and key suffixes of what fills an off-device tier ({@link TierMemory.breakdown}), with the
 * offload group whose placements give the layer spans (none for draft experts).
 */
const TIER_CATEGORIES = [
  { field: 'lookupTables', key: 'tier-tables', label: 'Lookup tables', group: 'lookup_tables' },
  { field: 'routedExperts', key: 'tier-experts', label: 'MoE experts', group: 'routed_experts' },
  { field: 'draftExperts', key: 'tier-draft', label: 'Draft experts', group: undefined },
] as const;

/** A layer range `[from, to)` as the inclusive span users read, e.g. `0–23`. */
function layerSpan(layers: { from: number; to: number }): string {
  return layers.to - layers.from === 1 ? `${layers.from}` : `${layers.from}–${layers.to - 1}`;
}

/**
 * The label of a tier's category with the layer spans of the group's placements on that tier, e.g. `MoE experts,
 * layers 0–23` (a host-RAM + SSD split); the plain label when no placement gives a range (every layer).
 */
function categoryLabel(
  category: (typeof TIER_CATEGORIES)[number],
  tier: TierMemory['tier'],
  placements: readonly OffloadPlacement[],
): string {
  const onTier = placements.filter((p) => p.group === category.group && p.tier === tier);
  const spans = onTier.flatMap((p) => (p.layers ? [layerSpan(p.layers)] : []));
  if (spans.length === 0 || spans.length < onTier.length) return category.label;
  return `${category.label}, ${spans.length === 1 && !spans[0].includes('–') ? 'layer' : 'layers'} ${spans.join(', ')}`;
}

/** The non-empty categories of an off-device tier as bar segments, labelled with their layer spans. */
function tierSegments(tier: TierMemory, placements: readonly OffloadPlacement[]): Segment[] {
  return TIER_CATEGORIES.map((c) => ({
    key: c.key,
    label: categoryLabel(c, tier.tier, placements),
    bytes: tier.breakdown[c.field],
  })).filter((s) => s.bytes > 0);
}

/** Fit state of an off-device tier, with the same thresholds as a device. */
function tierState(tier: TierMemory): MemoryState {
  if (!tier.fits) return 'danger';
  return tier.bytes / tier.capacityBytes > HIGH_MEMORY_UTILISATION ? 'warning' : 'fits';
}

/**
 * Per-device capacity and its allocations, one bar per pipeline stage, then one bar per off-device tier (host RAM,
 * SSD).
 * `onPlaceTablesOnHost`, when given, adds a button that moves an on-device n-gram table to host RAM.
 */
export function MemoryFit({
  result,
  onPlaceTablesOnHost,
}: {
  result: ResultSummary;
  onPlaceTablesOnHost?: () => void;
}) {
  const unified = result.memory.kind === 'unified';
  // On unified memory no host_ram tier exists (host RAM is the device pool; the sim enforces it).
  const tierBars = result.memory.tiers ?? [];
  const bars = [
    ...result.memory.stages.map((stage) => ({
      key: `stage-${stage.stage}`,
      name: `Stage ${stage.stage + 1}`,
      segments: stageSegments(stage),
    })),
    ...tierBars.map((tier) => ({
      key: `tier-${tier.tier}`,
      name: TIER_LABEL[tier.tier],
      segments: tierSegments(tier, result.offload ?? []),
    })),
  ];
  const offsets = bars.map((_, i) =>
    bars.slice(0, i).reduce((n, bar) => n + bar.segments.length, 0),
  );
  const points = bars.flatMap((bar) => bar.segments.map((segment) => ({ bar, segment })));
  const chart = useChartPoint(
    points.map((_, i) => i),
    points.length,
  );
  const active = chart.active === null ? undefined : points[chart.active];
  const context = formatContextTokens(
    Math.max(result.prefill.contextTokens, result.decode.contextTokens),
  );
  const label = ({ bar, segment }: (typeof points)[number]) =>
    `${bar.name} · ${segment.label} · ${formatBytes(segment.bytes)} @ ${context}`;
  const renderBar = (
    barIndex: number,
    ariaLabel: string,
    state: MemoryState,
    fits: boolean,
    totalBytes: number,
    capacityBytes: number,
  ) => (
    <div
      className={`capacity-bar memory-${state} ${fits ? '' : 'overflow'}`}
      role="group"
      aria-label={ariaLabel}
      style={
        {
          '--capacity': `${Math.min(100, (capacityBytes / totalBytes) * 100)}%`,
        } as CSSProperties
      }
    >
      {bars[barIndex].segments.map((segment, index) => {
        const point = offsets[barIndex] + index;
        return (
          <span
            {...chart.pointProps(point, label(points[point]))}
            key={segment.key}
            className={`allocation-${segment.key} ${chart.active === point ? 'is-active' : ''}`}
            style={{
              width: `${(segment.bytes / Math.max(capacityBytes, totalBytes)) * 100}%`,
            }}
            onPointerLeave={() => chart.chartProps.onPointerLeave()}
          />
        );
      })}
    </div>
  );
  return (
    <section className="memory interactive-chart" data-chart="memory" id="memory-fit" tabIndex={-1}>
      <h2>
        Memory per device{' '}
        <HelpIcon term="Memory fit">
          {helpLines([
            'Memory needed per device, sized for the larger of prefill and decode.',
            'If it does not fit, the speed shown is only a theoretical ceiling.',
            'Host RAM and SSD bars are per machine.',
          ])}
        </HelpIcon>
      </h2>
      {result.memory.stages.map((stage, stageIndex) => {
        const segments = bars[stageIndex].segments;
        const state = memoryState(stage);
        return (
          <div className="memory-stage" key={stage.stage}>
            <div data-testid="memory-verdict" className={`memory-${state}`}>
              <MemoryWarningIcon stage={stage} />
              {result.memory.stages.length > 1 && `Stage ${stage.stage + 1}. `}
              {stage.fits ? 'Fits' : 'Does not fit'}: {formatBytes(stage.totalBytes)} of{' '}
              {formatBytes(stage.capacityBytes)}
              {stage.physicalCapacityBytes !== undefined &&
                stage.physicalCapacityBytes > stage.capacityBytes && (
                  <>
                    {' '}
                    GPU limit ({formatBytes(stage.physicalCapacityBytes)} installed){' '}
                    <HelpIcon term="GPU memory limit">
                      {helpLines([
                        'The OS lets the GPU use only part of unified memory.',
                        'Raised the limit? Set it under Hardware → Advanced.',
                      ])}
                    </HelpIcon>
                  </>
                )}
              {state === 'warning' && (
                <>
                  {' '}
                  ({memoryUtilisation(stage)}){' '}
                  <HelpIcon term="High memory utilisation">
                    {helpLines([
                      'Little memory is left for the OS and the runtime.',
                      'Unified memory is shared with the OS.',
                    ])}
                  </HelpIcon>
                </>
              )}
            </div>
            {renderBar(
              stageIndex,
              `Stage ${stage.stage + 1} memory allocations`,
              state,
              stage.fits,
              stage.totalBytes,
              stage.capacityBytes,
            )}
            <ul
              className="memory-legend"
              data-testid="memory-legend"
              aria-label="Memory allocations"
            >
              {segments.map((segment) => (
                <li key={segment.key}>
                  <span className={`memory-swatch allocation-${segment.key}`} aria-hidden="true" />
                  {segment.label}{' '}
                  <strong>
                    <Num>{formatBytes(segment.bytes)}</Num>
                  </strong>
                  {segment.key === 'table' && onPlaceTablesOnHost && (
                    <button
                      type="button"
                      className="quiet legend-action"
                      onClick={onPlaceTablesOnHost}
                    >
                      {unified ? 'Place on SSD' : 'Place on host RAM'}
                    </button>
                  )}
                </li>
              ))}
            </ul>
          </div>
        );
      })}
      {tierBars.map((tier, tierIndex) => {
        const state = tierState(tier);
        return (
          <div className="memory-stage memory-tier" key={tier.tier} data-testid="memory-tier">
            <div className={`memory-${state}`}>
              {TIER_LABEL[tier.tier]} {formatBytes(tier.bytes)} / {formatBytes(tier.capacityBytes)}
              {!tier.fits && ': does not fit'}
            </div>
            {renderBar(
              result.memory.stages.length + tierIndex,
              `${TIER_LABEL[tier.tier]} allocations`,
              state,
              tier.fits,
              tier.bytes,
              tier.capacityBytes,
            )}
            <ul
              className="memory-legend"
              data-testid="memory-tier-legend"
              aria-label={`${TIER_LABEL[tier.tier]} contents`}
            >
              {bars[result.memory.stages.length + tierIndex].segments.map((segment) => (
                <li key={segment.key}>
                  <span className={`memory-swatch allocation-${segment.key}`} aria-hidden="true" />
                  {segment.label}{' '}
                  <strong>
                    <Num>{formatBytes(segment.bytes)}</Num>
                  </strong>
                  {segment.key === 'tier-experts' && (
                    <HelpIcon term="MoE experts">
                      {helpLines([
                        'The tier always stores every expert.',
                        'The resident share is only a cached copy, so this number stays the same.',
                      ])}
                    </HelpIcon>
                  )}
                </li>
              ))}
              <li>
                Node total{' '}
                <strong>
                  <Num>{formatBytes(tier.bytes)}</Num>
                </strong>
              </li>
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
