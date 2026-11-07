import {
  formatTokensPerSec,
  type ResultSummary,
  type AppState,
  type LoadedModel,
  type BuildResult,
} from '../state';
import { RooflineChart } from './RooflineChart';
import { MemoryFit } from './MemoryFit';

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
  const linkName = link
    ? `${link.name.replace(/ \(host bounce\)/, '')}${link.p2p === 'direct' ? ' with direct P2P' : ' with host staging'}`
    : undefined;
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
                    <span>tok/s</span>
                  </p>
                  <small>
                    {formatTokensPerSec(result[phase].perSequenceTokensPerSec)} per sequence
                  </small>
                </div>
              ))}
            </div>
            <RooflineChart roof={result.roofline} />
          </div>
          <p className="theory">
            Theoretical maximum: excludes engine overheads.{' '}
            <a href="./about.html">How the estimate works</a>
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
