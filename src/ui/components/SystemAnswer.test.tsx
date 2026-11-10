// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import {
  balancedDeployment,
  systemThroughput,
  type PoolRates,
} from '../../sim/engine/disaggregation';
import type { DisaggregationSummary } from '../state';
import { SystemAnswer, bottleneckText } from './SystemAnswer';

afterEach(cleanup);

/**
 * The design note's W-PD1 ingress-bound row (kvTransfer.test.ts, "ingress binds with D_D = 4 at 1 GB/s"): a prefill
 * instance at λ_P,eff 6.62274 req/s on 8 devices, a decode instance at λ_D 6.25 req/s on 4 devices whose KV ingress
 * caps it at λ_recv,1 = 4e9 / 1,207,959,552 = 3.31137 req/s. At 1 : 2 prefill and ingress tie (balanced), but decode
 * compute is only 3.31137 / 6.25 = 53% used.
 */
function ingressBound(prefillInstances: number, decodeInstances: number): DisaggregationSummary {
  const P = 8192;
  const lambdaP = 6.622738308376737;
  const ingress = 4e9 / 1_207_959_552;
  const prefill: PoolRates = {
    devices: 8,
    tokensPerSec: lambdaP * P,
    contextTokens: P,
    requestsPerSec: lambdaP,
  };
  const decode: PoolRates = {
    devices: 4,
    tokensPerSec: 6400,
    contextTokens: 8704,
    requestsPerSec: 6.25,
  };
  return {
    promptLen: P,
    outputLen: 1024,
    decodeContextLen: 8704,
    prefill,
    decode,
    balance: balancedDeployment(prefill, decode, ingress),
    system: systemThroughput({
      prefill,
      decode,
      ingressRequestsPerSec: ingress,
      promptLen: P,
      outputLen: 1024,
      prefillInstances,
      decodeInstances,
    }),
  };
}

describe('SystemAnswer', () => {
  it('a balance held by KV ingress names it and shows decode at 53%, never "fully used"', () => {
    const s = ingressBound(1, 2);
    expect(s.system.bottleneck).toBe('balanced');
    expect(bottleneckText(s)).toBe('Balanced at KV ingress (prefill 100%, decode 53% used)');
    render(<SystemAnswer summary={s} />);
    const line = screen.getByTestId('pd-bottleneck').textContent ?? '';
    expect(line).toBe('Balanced at KV ingress (prefill 100%, decode 53% used)');
    expect(line).not.toMatch(/fully/);
    expect(screen.getByTestId('pd-system-tps').textContent).toBe(
      '423.9tok/s/GPU @ 8k/1k, decode @ 8,704',
    );
    expect(screen.getByTestId('pd-balanced').textContent).toBe('1 : 2.00');
  });

  it('at 1 : 1 the same rig is bound by KV ingress', () => {
    expect(bottleneckText(ingressBound(1, 1))).toBe(
      'Bottleneck: KV ingress at decode (prefill 50%, decode 53% used)',
    );
  });

  it('a compute balance shows both utilizations at 100%', () => {
    const prefill: PoolRates = {
      devices: 8,
      tokensPerSec: 16 * 8192,
      contextTokens: 8192,
      requestsPerSec: 16,
    };
    const decode: PoolRates = {
      devices: 8,
      tokensPerSec: 8 * 1024,
      contextTokens: 8192,
      requestsPerSec: 8,
    };
    const s: DisaggregationSummary = {
      promptLen: 8192,
      outputLen: 1024,
      decodeContextLen: 8192,
      prefill,
      decode,
      balance: balancedDeployment(prefill, decode),
      system: systemThroughput({
        prefill,
        decode,
        promptLen: 8192,
        outputLen: 1024,
        prefillInstances: 1,
        decodeInstances: 2,
      }),
    };
    expect(s.system.bottleneck).toBe('balanced');
    expect(bottleneckText(s)).toBe('Balanced (prefill 100%, decode 100% used)');
  });
});
