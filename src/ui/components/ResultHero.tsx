import type { Ref } from 'react';
import {
  formatTokensPerSec,
  formatContextTokens,
  formatSeconds,
  type ResultSummary,
  type AppState,
  type LoadedModel,
  type BuildResult,
  type OffloadPlacement,
  cablingName,
} from '../state';
import { RooflineChart } from './RooflineChart';
import { MemoryFit } from './MemoryFit';
import { HeroMemoryWarning } from './MemoryWarning';
import { HelpIcon } from './HelpIcon';
import { Num } from './Num';

/** Display name of an off-device tier. */
export function tierName(tier: 'host_ram' | 'nvme'): string {
  return tier === 'host_ram' ? 'host RAM' : 'SSD';
}

/** A layer range `[from, to)` as the inclusive span users read, e.g. `24–35`. */
function layerSpan(layers: { from: number; to: number }): string {
  return layers.to - layers.from === 1 ? `${layers.from}` : `${layers.from}–${layers.to - 1}`;
}

/** The chip text of one off-device placement, e.g. `experts: host RAM, h 0.50`. */
export function offloadChipText(p: OffloadPlacement): string {
  switch (p.group) {
    case 'routed_experts':
      return `experts${p.layers ? ` ${layerSpan(p.layers)}` : ''}: ${tierName(p.tier)}, h ${(p.hitRatio ?? 0).toFixed(2)}`;
    case 'dense_layers':
      return `layers${p.layers ? ` ${layerSpan(p.layers)}` : ''}: ${tierName(p.tier)}`;
    case 'lookup_tables':
      return `n-gram tables: ${tierName(p.tier)}`;
  }
}

const METHOD_NAMES = { mtp: 'MTP', dflash2: 'DFlash 2', dspark: 'DSpark' } as const;

export function ResultHero({
  result,
  state,
  loaded,
  built,
  pending,
  onPlaceTablesOnHost,
  roofline = true,
  figuresRef,
}: {
  result?: ResultSummary;
  state: AppState;
  loaded: LoadedModel | null;
  built: BuildResult;
  pending: boolean;
  /** Moves an on-device n-gram lookup table to host RAM; shown as a button in the memory card when given. */
  onPlaceTablesOnHost?: () => void;
  /** Show the roofline chart beside the figures (false: the stacked layout places it further down). */
  roofline?: boolean;
  /** The element holding the two figures, watched to show the mini result bar once it scrolls away. */
  figuresRef?: Ref<HTMLDivElement>;
}) {
  const link = built.hardware?.links.intra?.link;
  const intraName = link
    ? `${link.name.replace(/ \(host bounce\)/, '')}${link.p2p === 'direct' ? ' with direct P2P' : ' with host staging'}`
    : undefined;
  // Between machines: the catalog's short name of the network (else the link's name), and the cabling when it is
  // not a switch, e.g. "over ConnectX-7 (ring)".
  const inter = built.hardware?.links.inter?.link;
  const cabling = built.hardware?.cabling;
  const interName = inter
    ? `${built.hardware?.network?.shortName ?? inter.name}${cabling !== undefined && cabling !== 'switched' ? ` (${cablingName(cabling, state.hw.nodes).toLowerCase()})` : ''}`
    : undefined;
  const linkName = [intraName, interName].filter((n) => n !== undefined).join(' and ');
  const offload = result?.offload ?? [];
  const spec = result?.decode.speculative;
  // The count and "×" stay on one line with the device name's first word (never "on 2 ×" / "NVIDIA …").
  const [deviceFirst, ...deviceRest] = (
    built.hardware?.device.device.name ?? state.hw.deviceId
  ).split(' ');
  return (
    <section className={`hero ${pending ? 'pending' : ''}`} aria-busy={pending}>
      <h1 className="rig">
        {loaded?.label ?? 'Loading model'} on{' '}
        <span className="keep-together">
          {state.hw.devicesPerNode * state.hw.nodes} × {deviceFirst}
        </span>
        {deviceRest.length > 0 && ` ${deviceRest.join(' ')}`}
        {built.hardware?.edited && !built.hardware.device.edited ? ' (edited)' : ''}
        {linkName ? ` over ${linkName}` : ''}, with{' '}
        <span className="keep-together">TP {state.parallel.tp}</span>
      </h1>
      {result ? (
        <>
          <div className={`hero-grid${roofline ? '' : ' no-roofline'}`}>
            <div className="answers" ref={figuresRef}>
              {(['prefill', 'decode'] as const).map((phase) => (
                <div className="answer" key={phase}>
                  <h2>{phase === 'prefill' ? 'Prefill' : 'Decode'}</h2>
                  <div className="answer-line">
                    <strong data-testid={`${phase}-tps`}>
                      <Num>{formatTokensPerSec(result[phase].tokensPerSec)}</Num>
                    </strong>
                    <HeroMemoryWarning memory={result.memory} />
                    <span className="answer-unit">
                      tok/s @ {formatContextTokens(result[phase].contextTokens)}
                    </span>
                    {phase === 'decode' && spec && (
                      <span className="spec-dec" data-testid="spec-dec">
                        (spec-dec ×{spec.speedup.toFixed(2)}){' '}
                        <HelpIcon term="Speculative decoding speedup">
                          {result.speculative ? `${METHOD_NAMES[result.speculative.method]}, ` : ''}
                          {result.speculative ? `k = ${result.speculative.draftLength}, ` : ''}
                          E[τ] = {spec.expectedTokensPerStep.toFixed(2)} tokens accepted per step
                          (incl. the bonus token). Step: draft {formatSeconds(spec.draftSec)} +
                          verify {formatSeconds(spec.verifySec)}. Speedup over decode without
                          speculative decoding @ {formatContextTokens(result.decode.contextTokens)}.
                        </HelpIcon>
                        {result.speculative?.hypothetical && (
                          <span className="result-chip">hypothetical drafter</span>
                        )}
                      </span>
                    )}
                  </div>
                  <small
                    aria-hidden={state.workload.batch === 1}
                    style={{
                      display: 'block',
                      height: '1.2em',
                      lineHeight: '1.2',
                      visibility: state.workload.batch > 1 ? 'visible' : 'hidden',
                    }}
                  >
                    {state.workload.batch > 1
                      ? `${formatTokensPerSec(result[phase].perSequenceTokensPerSec)} per sequence`
                      : ''}
                  </small>
                </div>
              ))}
              {offload.length > 0 && (
                <ul className="result-chips" aria-label="Offloaded weights">
                  {offload.map((p, i) => (
                    // The "?" shares the last chip's item, so it never wraps onto a line of its own.
                    <li className="result-chip-item" key={p.group}>
                      <span className="result-chip">{offloadChipText(p)}</span>
                      {i === offload.length - 1 && (
                        <HelpIcon term="Offloaded weights">
                          Weight groups kept in host RAM or on an SSD and read over the host link
                          each step. h: probability that a needed expert is already in device
                          memory, per unique expert per layer per step at batch 1.
                        </HelpIcon>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            {roofline && (
              <RooflineChart
                roof={result.roofline}
                contexts={{
                  prefill: result.prefill.contextTokens,
                  decode: result.decode.contextTokens,
                }}
              />
            )}
          </div>
          <p className="theory">
            This is a theoretical maximum. <a href="./about.html">Read how the estimate works</a>
          </p>
          <MemoryFit result={result} onPlaceTablesOnHost={onPlaceTablesOnHost} />
        </>
      ) : (
        <div className="empty-result">
          {built.kind === 'issues'
            ? 'Resolve the input errors to calculate a ceiling.'
            : 'Calculating the hardware ceiling…'}
        </div>
      )}
    </section>
  );
}
