import {
  formatContextTokens,
  formatTokensPerSec,
  formatWorkloadQuote,
  effectivePrefillPhase,
  memoryVerdict,
  type DisaggregatedSummary,
  type ResultSummary,
} from '../state';
import { Num } from './Num';

/**
 * The mini result bar: prefill and decode rates, each with its context, and the memory fit, in one slim row. It
 * appears while the hero figures are scrolled out of view (pinned to the top of the output pane on the split
 * layout, of the window on the stacked one), so the numbers stay in sight while inputs are edited. It repeats
 * what the hero says, so it is hidden from assistive technology. With disaggregation on (`pd`), Prefill is the
 * prefill pool's, Decode the decode pool's, a System cell follows, and the fit is both pools' worst.
 */
export function MiniResult({
  result,
  visible,
  pd,
}: {
  result: ResultSummary;
  visible: boolean;
  pd?: DisaggregatedSummary;
}) {
  const src = { prefill: pd?.prefillPool ?? result, decode: pd?.decodePool ?? result };
  const verdicts = [
    memoryVerdict(src.prefill.memory).state,
    memoryVerdict(src.decode.memory).state,
  ];
  // With disaggregation on, the text names the pool(s) that do not fit.
  const failing = (['prefill', 'decode'] as const).filter((_, i) => verdicts[i] === 'danger');
  const dangerText =
    pd === undefined
      ? 'Does not fit'
      : failing.length === 2
        ? 'Both pools do not fit'
        : `${failing[0] === 'prefill' ? 'Prefill' : 'Decode'} pool does not fit`;
  const fit = verdicts.includes('danger')
    ? 'danger'
    : verdicts.includes('warning')
      ? 'warning'
      : verdicts[0];
  const spec = src.decode.decode.speculative;
  // With disaggregation on, the prefill figure is the pool's rate with the KV transfer (R_P,eff).
  const phases = {
    prefill: pd ? effectivePrefillPhase(pd) : result.prefill,
    decode: src.decode.decode,
  };
  const s = pd?.system;
  return (
    <div
      className={`compact-results memory-${fit}`}
      data-testid="mini-result"
      data-visible={visible || undefined}
      aria-hidden="true"
    >
      {(['prefill', 'decode'] as const).map((phase) => (
        <span className="mini-cell" key={phase}>
          <span className="mini-label">{phase === 'prefill' ? 'Prefill' : 'Decode'}</span>
          <strong className="mini-value">
            <Num>{formatTokensPerSec(phases[phase].tokensPerSec)}</Num>
          </strong>
          <span className="mini-unit">
            tok/s @ {formatContextTokens(phases[phase].contextTokens)}
            {phase === 'decode' && spec && ` ×${spec.speedup.toFixed(2)}`}
          </span>
        </span>
      ))}
      {s && (
        <span className="mini-cell" data-testid="mini-system">
          <span className="mini-label">System</span>
          <strong className="mini-value">
            <Num>{formatTokensPerSec(s.system.outputTokensPerSecPerDevice)}</Num>
          </strong>
          <span className="mini-unit">
            tok/s/GPU {formatWorkloadQuote(s.promptLen, s.outputLen, s.decodeContextLen)}
          </span>
        </span>
      )}
      <span className="mini-fit">
        {fit === 'danger' ? dangerText : fit === 'warning' ? 'Tight fit' : 'Fits'}
      </span>
    </div>
  );
}
