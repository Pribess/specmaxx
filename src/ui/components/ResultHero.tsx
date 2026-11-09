import type { Ref } from 'react';
import {
  formatTokensPerSec,
  formatContextTokens,
  formatSeconds,
  type ResultSummary,
  type AppState,
  type LoadedModel,
  type BuildResult,
  type ResolvedHardware,
  type OffloadPlacement,
  cablingName,
} from '../state';
import { RooflineChart } from './RooflineChart';
import { MemoryFit } from './MemoryFit';
import { HeroMemoryWarning } from './MemoryWarning';
import { HelpIcon } from './HelpIcon';
import { helpLines } from './helpLines';
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

/**
 * The device name for the rig line. Apple devices and devices sold in several memory sizes get the chosen
 * per-device size in their parentheses, e.g. `Apple M5 Ultra (64-core GPU, 256GB memory)`; the size is
 * `memCapacityBytes` (so a spec-editor edit shows), written as the memory picker does.
 */
export function rigDeviceName(hardware: ResolvedHardware | undefined, fallback = ''): string {
  if (!hardware) return fallback;
  const { device, memCapacityBytes } = hardware;
  // A spec edit renames the device `<name> (edited)`; the memory goes into the name's own parentheses.
  const edited = device.device.name.endsWith(' (edited)') ? ' (edited)' : '';
  const name = device.device.name.slice(0, device.device.name.length - edited.length);
  const multi = device.device.memory.capacityOptionsBytes.length > 1;
  if (!multi && device.device.vendor !== 'Apple') return device.device.name;
  const mem = `${memCapacityBytes / 1e9}GB memory`;
  return (name.endsWith(')') ? `${name.slice(0, -1)}, ${mem})` : `${name} (${mem})`) + edited;
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
  const [deviceFirst, ...deviceRest] = rigDeviceName(built.hardware, state.hw.deviceId).split(' ');
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
                          {helpLines([
                            ...(result.speculative
                              ? [
                                  `${METHOD_NAMES[result.speculative.method]}, draft length k = ${result.speculative.draftLength}.`,
                                ]
                              : []),
                            `${spec.expectedTokensPerStep.toFixed(2)} tokens accepted per step on average (incl. the bonus token).`,
                            `One step: draft ${formatSeconds(spec.draftSec)} + verify ${formatSeconds(spec.verifySec)}.`,
                          ])}
                        </HelpIcon>
                        {result.speculative?.hypothetical && (
                          <span className="result-chip">hypothetical drafter</span>
                        )}
                      </span>
                    )}
                  </div>
                  {phase === 'prefill' && result.prefill.withoutOverlap && (
                    <div className="what-if" data-testid="prefill-no-overlap">
                      <span>Without overlap: </span>
                      <span className="what-if-value">
                        <span>
                          <Num>
                            {formatTokensPerSec(result.prefill.withoutOverlap.tokensPerSec)}
                          </Num>
                        </span>
                        <HelpIcon term="Prefill without overlap">
                          {helpLines([
                            'Prefill when sending data between devices does not overlap with compute.',
                            'The headline assumes it fully overlaps; most engines do not.',
                          ])}
                        </HelpIcon>
                      </span>
                    </div>
                  )}
                  {phase === 'decode' && result.decode.withDispatch && (
                    <div className="what-if" data-testid="decode-dispatch">
                      <span>With engine dispatch: </span>
                      <span className="what-if-value">
                        <span>
                          <Num>{formatTokensPerSec(result.decode.withDispatch.tokensPerSec)}</Num>
                        </span>
                        <HelpIcon term="With engine dispatch">
                          {helpLines([
                            `Decode if the engine spends ${formatSeconds(result.decode.withDispatch.perKernelSec)} launching each GPU kernel: ${result.decode.withDispatch.kernelsPerStep} per step, one per operation.`,
                            'That is engine overhead, not a hardware limit, so the headline leaves it out.',
                          ])}
                        </HelpIcon>
                      </span>
                    </div>
                  )}
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
                          {helpLines([
                            'Weight groups kept in host RAM or on an SSD.',
                            'They are read over the host link every step.',
                            'h is how often a needed expert is already in device memory.',
                          ])}
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
