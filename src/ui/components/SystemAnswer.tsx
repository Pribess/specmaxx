import { BALANCE_REL_TOLERANCE } from '../../sim/engine/disaggregation';
import {
  formatBytes,
  formatContextTokens,
  formatRatio,
  formatSeconds,
  formatShare,
  formatTokensPerSec,
  formatWorkloadQuote,
  type DisaggregationSummary,
} from '../state';
import { HelpIcon } from './HelpIcon';
import { helpLines } from './helpLines';
import { Num } from './Num';

/** A device count for a ratio: whole numbers as is, else one decimal (`20.5`). */
function deviceCount(n: number): string {
  return Number.isInteger(n) ? n.toLocaleString('en-US') : n.toFixed(1);
}

/**
 * The balanced ratio as devices, the smaller instance side at 1: one prefill instance's devices to the decode
 * devices it feeds (`8 : 20.5 GPUs`), or the reverse when prefill needs more than one instance per decode one.
 */
export function balancedDevicesText(s: DisaggregationSummary): string {
  const r = s.balance.prefillPerDecode;
  const dp = s.prefill.devices;
  const dd = s.decode.devices;
  if (!Number.isFinite(r) || r <= 0) return '—';
  return r <= 1
    ? `${deviceCount(dp)} : ${deviceCount(dd / r)} GPUs`
    : `${deviceCount(r * dp)} : ${deviceCount(dd)} GPUs`;
}

/**
 * The bottleneck line: which side limits the system at the user's ratio, and how busy each pool's compute is (always
 * the actual utilizations, also when balanced).
 */
export function bottleneckText(s: DisaggregationSummary): string {
  const util = `prefill ${formatShare(s.system.prefillUtilization)}, decode ${formatShare(s.system.decodeUtilization)} used`;
  switch (s.system.bottleneck) {
    case 'balanced':
      // Balance means prefill capacity equals the decode side's effective capacity; with decode compute not fully
      // used, the decode side is held by its KV ingress, not its compute.
      return s.system.decodeUtilization < 1 - BALANCE_REL_TOLERANCE
        ? `Balanced at KV ingress (${util})`
        : `Balanced (${util})`;
    case 'prefill':
      return `Bottleneck: prefill pool (${util})`;
    case 'decode':
      return `Bottleneck: decode pool (${util})`;
    case 'kvIngress':
      return `Bottleneck: KV ingress at decode (${util})`;
  }
}

/**
 * The third hero answer of a disaggregated deployment: output tok/s per GPU of the whole system at the user's
 * instance ratio, the balanced ratio, the bottleneck and the KV transfer, each with its context quote.
 */
export function SystemAnswer({ summary: s }: { summary: DisaggregationSummary }) {
  const quote = formatWorkloadQuote(s.promptLen, s.outputLen, s.decodeContextLen);
  const kv = s.kvTransfer;
  return (
    <div className="answer answer-system">
      <h2>
        System{' '}
        <HelpIcon term="System throughput">
          {helpLines([
            'Output tokens per second per GPU of the whole deployment, at your instance counts.',
            'Every output token is made at the decode rate @ the decode context.',
            'No queueing, latency targets or request routing.',
          ])}
        </HelpIcon>
      </h2>
      <div className="answer-line" data-testid="pd-system-tps">
        <strong>
          <Num>{formatTokensPerSec(s.system.outputTokensPerSecPerDevice)}</Num>
        </strong>
        <span className="answer-unit">tok/s/GPU {quote}</span>
      </div>
      <div className="what-if">
        <span>Balanced P:D</span>
        <span data-testid="pd-balanced">
          <Num>{formatRatio(s.balance.prefillPerDecode)}</Num>
        </span>
        <span className="what-if-value">
          <span className="answer-quote">{quote}</span>
          <HelpIcon term="Balanced P:D">
            {helpLines([
              'Prefill to decode instances at which neither pool waits for the other.',
              `As GPUs: ${balancedDevicesText(s)}.`,
              `At this ratio: ${formatTokensPerSec(s.balance.outputTokensPerSecPerDevice)} tok/s/GPU ${quote}.`,
            ])}
          </HelpIcon>
        </span>
      </div>
      <div className="what-if" data-testid="pd-bottleneck">
        {bottleneckText(s)}
      </div>
      {kv && (
        <div className="what-if" data-testid="pd-kv">
          <span>
            KV {formatBytes(kv.bytesPerToken)}/token, {formatBytes(kv.requestBytes)} per request @{' '}
            {formatContextTokens(s.promptLen)}
          </span>
          <HelpIcon term="KV transfer">
            {helpLines([
              `One request alone takes ${formatSeconds(kv.requestSec)} to send.`,
              `It adds ${formatSeconds(kv.exposedSec)} to the time to first token.`,
              `Prefill cycle ${formatSeconds(kv.prefillCycleSec)}, with the transfer ${formatSeconds(kv.effectivePrefillCycleSec)}.`,
            ])}
          </HelpIcon>
        </div>
      )}
      <small>
        {s.system.prefillInstances} : {s.system.decodeInstances} instances,{' '}
        {s.system.devices.toLocaleString('en-US')} GPUs
      </small>
    </div>
  );
}
