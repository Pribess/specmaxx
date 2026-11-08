import {
  formatTokensPerSec,
  formatContextTokens,
  type ResultSummary,
  type AppState,
  type LoadedModel,
  type BuildResult,
  cablingName,
} from '../state';
import { RooflineChart } from './RooflineChart';
import { MemoryFit } from './MemoryFit';
import { HeroMemoryWarning } from './MemoryWarning';

export function ResultHero({
  result,
  state,
  loaded,
  built,
  pending,
}: {
  result?: ResultSummary;
  state: AppState;
  loaded: LoadedModel | null;
  built: BuildResult;
  pending: boolean;
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
  return (
    <section className={`hero ${pending ? 'pending' : ''}`} aria-busy={pending}>
      <p className="rig">
        {loaded?.label ?? 'Loading model'} on {state.hw.devicesPerNode * state.hw.nodes} ×{' '}
        {built.hardware?.device.device.name ?? state.hw.deviceId}
        {built.hardware?.edited && !built.hardware.device.edited ? ' (edited)' : ''}
        {linkName ? ` over ${linkName}` : ''}, with{' '}
        <span className="keep-together">TP {state.parallel.tp}.</span>
      </p>
      {result ? (
        <>
          <div className="hero-grid">
            <div className="answers">
              {(['prefill', 'decode'] as const).map((phase) => (
                <div className="answer" key={phase}>
                  <h1>{phase === 'prefill' ? 'Prefill' : 'Decode'}</h1>
                  <p>
                    <strong data-testid={`${phase}-tps`}>
                      {formatTokensPerSec(result[phase].tokensPerSec)}
                    </strong>
                    <HeroMemoryWarning stages={result.memory.stages} />
                    <span>tok/s @ {formatContextTokens(result[phase].contextTokens)}</span>
                  </p>
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
            </div>
            <RooflineChart
              roof={result.roofline}
              contexts={{
                prefill: result.prefill.contextTokens,
                decode: result.decode.contextTokens,
              }}
            />
          </div>
          <p className="theory">
            This is a theoretical maximum. <a href="./about.html">Read how the estimate works</a>
          </p>
          <MemoryFit result={result} />
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
