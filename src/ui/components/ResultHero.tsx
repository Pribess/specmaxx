import { Fragment, type Ref } from 'react';
import {
  formatTokensPerSec,
  type DisaggregatedSummary,
  effectivePrefillPhase,
  formatContextTokens,
  formatSeconds,
  type ResultSummary,
  type AppState,
  type LoadedModel,
  type BuildResult,
  type ResolvedHardware,
  type OffloadPlacement,
  cablingName,
  hasMoe,
} from '../state';
import { RooflineChart } from './RooflineChart';
import { MemoryFit } from './MemoryFit';
import { HeroMemoryWarning } from './MemoryWarning';
import { HelpIcon } from './HelpIcon';
import { helpLines } from './helpLines';
import { Num } from './Num';
import { SystemAnswer } from './SystemAnswer';

/** The parallel layout of a pool in words: `DP attention + EP 32`, `TP 8, DCP 8`, `TP 4, DP 2`. */
export function parallelText(state: AppState, moe: boolean): string {
  const p = state.parallel;
  if (moe && state.parallelLayout === 'dpAttentionEp') return `DP attention + EP ${p.ep}`;
  return [
    `TP ${p.tp}`,
    ...(p.dcp !== undefined && p.dcp > 1 ? [`DCP ${p.dcp}`] : []),
    ...(p.dp > 1 ? [`DP ${p.dp}`] : []),
  ].join(', ');
}

/**
 * The rig line of a disaggregated deployment: both pools' device sets and layouts, e.g. `GLM-5.3-Flash on 32 × H100
 * SXM prefill and 144 × H100 SXM decode, with DP attention + EP 32 / 144`.
 */
function PdRig({
  label,
  moe,
  pools,
}: {
  label: string;
  moe: boolean;
  pools: { name: 'prefill' | 'decode'; state: AppState; built: BuildResult }[];
}) {
  const dep = moe && pools.every((p) => p.state.parallelLayout === 'dpAttentionEp');
  return (
    <h1 className="rig" data-testid="pd-rig">
      {label} on{' '}
      {pools.map((p, i) => {
        const [first, ...rest] = rigDeviceName(p.built.hardware, p.state.hw.deviceId).split(' ');
        return (
          <Fragment key={p.name}>
            {i > 0 && ' and '}
            <span className="keep-together">
              {p.state.hw.devicesPerNode * p.state.hw.nodes} × {first}
            </span>
            {rest.length > 0 && ` ${rest.join(' ')}`} {p.name}
          </Fragment>
        );
      })}
      , with{' '}
      {dep ? (
        <span className="keep-together">
          DP attention + EP {pools.map((p) => p.state.parallel.ep).join(' / ')}
        </span>
      ) : (
        pools.map((p) => `${parallelText(p.state, moe)} ${p.name}`).join(', ')
      )}
    </h1>
  );
}

/** Display name of an off-device tier. */
export function tierName(tier: 'host_ram' | 'nvme'): string {
  return tier === 'host_ram' ? 'host RAM' : 'SSD';
}

/** A layer range `[from, to)` as the inclusive span users read, e.g. `24–35`. */
function layerSpan(layers: { from: number; to: number }): string {
  return layers.to - layers.from === 1 ? `${layers.from}` : `${layers.from}–${layers.to - 1}`;
}

/** The chip text of one off-device placement, e.g. `MoE experts: host RAM, h 0.50`, `n-gram tables 0–23: SSD`. */
export function offloadChipText(p: OffloadPlacement): string {
  const span = p.layers ? ` ${layerSpan(p.layers)}` : '';
  switch (p.group) {
    case 'routed_experts':
      return `MoE experts${span}: ${tierName(p.tier)}, h ${(p.hitRatio ?? 0).toFixed(2)}`;
    case 'lookup_tables':
      return `n-gram tables${span}: ${tierName(p.tier)}`;
    case 'token_embeddings':
      return `Token embeddings: ${tierName(p.tier)}`;
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

/**
 * The spec-dec chip: `spec-dec: off`, `spec-dec: MTP, k=3`, or `spec-dec: DSpark, k=7 · code` (block methods name the use case
 * preset, prose by default, but not a custom acceptance), plus ` · hypothetical drafter` when that applies.
 */
export function specDecChipText(
  speculative: ResultSummary['speculative'],
  choice?: { useCase?: string; alpha?: number },
): string {
  if (!speculative) return 'spec-dec: off';
  const parts = [`${METHOD_NAMES[speculative.method]}, k=${speculative.draftLength}`];
  // MTP keeps its compact form; a block method names its preset (prose when none is chosen), not a custom α.
  if (speculative.method !== 'mtp' && choice && choice.alpha === undefined)
    parts.push(choice.useCase ?? 'prose');
  if (speculative.hypothetical) parts.push('hypothetical drafter');
  return `spec-dec: ${parts.join(' · ')}`;
}

export function ResultHero({
  result,
  state,
  loaded,
  built,
  pending,
  onPlaceTablesOnHost,
  roofline = true,
  figuresRef,
  pd,
}: {
  /** The summary the memory card, roofline and chips show (with disaggregation on: the edited pool's). */
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
  /**
   * With disaggregation on: Prefill comes from the prefill pool and Decode from the decode pool, a System answer is
   * added, and the rig line names both pools. `phases`: what the roofline shows (the edited pool's phases).
   */
  pd?: {
    summary: DisaggregatedSummary;
    decodeState: AppState;
    decodeBuilt: BuildResult;
    phases: readonly ('prefill' | 'decode')[];
  };
}) {
  const prefillSrc = pd?.summary.prefillPool ?? result;
  const decodeSrc = pd?.summary.decodePool ?? result;
  const decodeBatch = pd?.decodeState.workload.batch ?? state.workload.batch;
  const effPrefill = pd ? effectivePrefillPhase(pd.summary) : undefined;
  const moe = loaded !== null && hasMoe(loaded.model);
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
  const chips = [
    { key: 'spec-dec', text: specDecChipText(result?.speculative, state.deployment.specDec) },
    ...offload.map((p) => ({ key: `${p.group}-${p.layers?.from ?? 0}`, text: offloadChipText(p) })),
  ];
  // The count and "×" stay on one line with the device name's first word (never "on 2 ×" / "NVIDIA …").
  const [deviceFirst, ...deviceRest] = rigDeviceName(built.hardware, state.hw.deviceId).split(' ');
  return (
    <section className={`hero ${pending ? 'pending' : ''}`} aria-busy={pending}>
      {pd ? (
        <PdRig
          label={loaded?.label ?? 'Loading model'}
          moe={moe}
          pools={[
            { name: 'prefill', state, built },
            { name: 'decode', state: pd.decodeState, built: pd.decodeBuilt },
          ]}
        />
      ) : (
        <h1 className="rig">
          {loaded?.label ?? 'Loading model'} on{' '}
          <span className="keep-together">
            {state.hw.devicesPerNode * state.hw.nodes} × {deviceFirst}
          </span>
          {deviceRest.length > 0 && ` ${deviceRest.join(' ')}`}
          {built.hardware?.edited && !built.hardware.device.edited ? ' (edited)' : ''}
          {linkName ? ` over ${linkName}` : ''}, with{' '}
          {moe && state.parallelLayout === 'dpAttentionEp' ? (
            <span className="keep-together">DP attention + EP {state.parallel.ep}</span>
          ) : (
            <>
              <span className="keep-together">TP {state.parallel.tp}</span>
              {state.parallel.dcp !== undefined && state.parallel.dcp > 1 && (
                <>
                  , <span className="keep-together">DCP {state.parallel.dcp}</span>
                </>
              )}
              {state.parallel.dp > 1 && (
                <>
                  , <span className="keep-together">DP {state.parallel.dp}</span>
                </>
              )}
            </>
          )}
        </h1>
      )}
      {result && prefillSrc && decodeSrc ? (
        <>
          <div className={`hero-grid${roofline ? '' : ' no-roofline'}`}>
            <div className="answers" ref={figuresRef}>
              {(['prefill', 'decode'] as const).map((phase) => {
                const src = phase === 'prefill' ? prefillSrc : decodeSrc;
                const batch = phase === 'prefill' ? state.workload.batch : decodeBatch;
                const spec = src.decode.speculative;
                // With disaggregation on, the prefill figure is the pool's rate with the KV transfer (R_P,eff).
                const fig = phase === 'prefill' && effPrefill ? effPrefill : src[phase];
                return (
                  <div className="answer" key={phase}>
                    <h2>{phase === 'prefill' ? 'Prefill' : 'Decode'}</h2>
                    <div className="answer-line">
                      <strong data-testid={`${phase}-tps`}>
                        <Num>{formatTokensPerSec(fig.tokensPerSec)}</Num>
                      </strong>
                      <HeroMemoryWarning memory={src.memory} />
                      <span className="answer-unit">
                        tok/s @ {formatContextTokens(fig.contextTokens)}
                      </span>
                      {phase === 'decode' && spec && (
                        <span className="spec-dec" data-testid="spec-dec">
                          (spec-dec ×{spec.speedup.toFixed(2)}){' '}
                          <HelpIcon term="Speculative decoding speedup">
                            {helpLines([
                              ...(src.speculative
                                ? [
                                    `${METHOD_NAMES[src.speculative.method]}, k=${src.speculative.draftLength}.`,
                                  ]
                                : []),
                              `${spec.expectedTokensPerStep.toFixed(2)} tokens accepted per step on average (incl. the bonus token).`,
                              `One step: draft ${formatSeconds(spec.draftSec)} + verify ${formatSeconds(spec.verifySec)}.`,
                            ])}
                          </HelpIcon>
                        </span>
                      )}
                    </div>
                    {phase === 'prefill' && fig.computedTokensPerSec !== undefined && (
                      <div className="what-if" data-testid="prefill-computed">
                        <span>Computed: </span>
                        <span className="what-if-value">
                          <span>
                            <span data-testid="prefill-computed-tps">
                              <Num>{formatTokensPerSec(fig.computedTokensPerSec)}</Num>
                            </span>
                            <span className="answer-unit">
                              {' '}
                              tok/s @ {formatContextTokens(src.prefill.contextTokens)}
                            </span>
                          </span>
                          <HelpIcon term="Computed prefill rate">
                            {helpLines([
                              'The headline counts every prompt token, cached ones included.',
                              'This counts only the tokens prefill actually computed.',
                            ])}
                          </HelpIcon>
                        </span>
                      </div>
                    )}
                    {phase === 'prefill' &&
                      effPrefill &&
                      effPrefill.computeOnlyTokensPerSec !== effPrefill.tokensPerSec && (
                        <div className="what-if" data-testid="prefill-no-kv">
                          <span>Without KV transfer: </span>
                          <span className="what-if-value">
                            <span>
                              <Num>{formatTokensPerSec(effPrefill.computeOnlyTokensPerSec)}</Num>
                            </span>
                            <HelpIcon term="Prefill without KV transfer">
                              {helpLines([
                                'The prefill pool when sending the KV cache to decode costs no time.',
                                'The headline includes the KV link: a slow or not overlapped link stretches each prefill batch.',
                              ])}
                            </HelpIcon>
                          </span>
                        </div>
                      )}
                    {phase === 'prefill' && src.prefill.withoutOverlap && (
                      <div className="what-if" data-testid="prefill-no-overlap">
                        <span>Without overlap: </span>
                        <span className="what-if-value">
                          <span>
                            <Num>{formatTokensPerSec(src.prefill.withoutOverlap.tokensPerSec)}</Num>
                          </span>
                          <HelpIcon term="Prefill without overlap">
                            {helpLines([
                              'Prefill when sending data between devices does not overlap with compute.',
                              'The headline hides as much as the overlap settings allow; most engines hide less.',
                            ])}
                          </HelpIcon>
                        </span>
                      </div>
                    )}
                    {phase === 'decode' && src.decode.withDispatch && (
                      <div className="what-if" data-testid="decode-dispatch">
                        <span>With engine dispatch: </span>
                        <span className="what-if-value">
                          <span>
                            <Num>{formatTokensPerSec(src.decode.withDispatch.tokensPerSec)}</Num>
                          </span>
                          <HelpIcon term="With engine dispatch">
                            {helpLines([
                              `Decode if the engine spends ${formatSeconds(src.decode.withDispatch.perKernelSec)} launching each GPU kernel: ${src.decode.withDispatch.kernelsPerStep} per step, one per operation.`,
                              'That is engine overhead, not a hardware limit, so the headline leaves it out.',
                            ])}
                          </HelpIcon>
                        </span>
                      </div>
                    )}
                    <small
                      aria-hidden={batch === 1}
                      style={{
                        display: 'block',
                        height: '1.2em',
                        lineHeight: '1.2',
                        visibility: batch > 1 ? 'visible' : 'hidden',
                      }}
                    >
                      {batch > 1
                        ? `${formatTokensPerSec(fig.perSequenceTokensPerSec)} per sequence`
                        : ''}
                    </small>
                  </div>
                );
              })}
              {pd && <SystemAnswer summary={pd.summary.system} />}
              <ul className="result-chips" aria-label="Result settings">
                {chips.map((chip, i) => (
                  // The "?" shares the last chip's item, so it never wraps onto a line of its own.
                  <li className="result-chip-item" key={chip.key}>
                    <span className="result-chip">{chip.text}</span>
                    {i === chips.length - 1 && (
                      <HelpIcon term="Result settings">
                        {helpLines([
                          'Settings that change these numbers.',
                          'Offloaded weights are fetched from host RAM or SSD every step.',
                          'Spec-dec shows whether decode uses a draft model, and which.',
                        ])}
                      </HelpIcon>
                    )}
                  </li>
                ))}
              </ul>
            </div>
            {roofline && (
              <RooflineChart
                roof={result.roofline}
                contexts={{
                  prefill: result.prefill.contextTokens,
                  decode: result.decode.contextTokens,
                }}
                phases={pd?.phases}
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
