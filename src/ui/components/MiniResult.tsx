import { formatContextTokens, formatTokensPerSec, memoryVerdict, type ResultSummary } from '../state';
import { Num } from './Num';

/**
 * The mini result bar: prefill and decode rates, each with its context, and the memory fit, in one slim row. It
 * appears while the hero figures are scrolled out of view (pinned to the top of the output pane on the split
 * layout, of the window on the stacked one), so the numbers stay in sight while inputs are edited. It repeats
 * what the hero says, so it is hidden from assistive technology.
 */
export function MiniResult({ result, visible }: { result: ResultSummary; visible: boolean }) {
  const fit = memoryVerdict(result.memory).state;
  const spec = result.decode.speculative;
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
            <Num>{formatTokensPerSec(result[phase].tokensPerSec)}</Num>
          </strong>
          <span className="mini-unit">
            tok/s @ {formatContextTokens(result[phase].contextTokens)}
            {phase === 'decode' && spec && ` ×${spec.speedup.toFixed(2)}`}
          </span>
        </span>
      ))}
      <span className="mini-fit">
        {fit === 'danger' ? 'Does not fit' : fit === 'warning' ? 'Tight fit' : 'Fits'}
      </span>
    </div>
  );
}
