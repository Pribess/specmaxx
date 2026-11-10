/**
 * Prefill/decode disaggregation: prefill and decode run on two separate device pools, each with its own hardware,
 * parallel layout and batch (DeepSeek: prefill 32 GPUs, decode 144). An opt-in deployment scenario, built on top of
 * `simulate()` without changing it: each pool is one ordinary `simulate()` run, and a few closed-form functions
 * combine the prefill pool's prefill rate and the decode pool's decode rate into request rates, the balanced P:D
 * ratio and the system throughput at the user's ratio.
 *
 * Off (no `DisaggregatedInput` built) nothing here runs, so every single-pool result is unchanged by construction.
 *
 * Pools, request rates, balance and system throughput, plus the optional KV transfer (`config.kvTransfer`, see
 * `kvTransfer.ts`): it lengthens the prefill cycle (R_P,eff) and caps the decode side by its ingress (λ_D,eff).
 * Absent, the prefill cycle is unchanged (cycle_eff = cycle_P) and there is no ingress bound.
 *
 * See docs/simulation-model.md §14 "Prefill/decode disaggregation"; design docs/design/prefill-decode-disaggregation.md.
 */

import type { HardwareStack } from '../hw/types';
import type { ParallelConfig } from '../ir/model';
import {
  kvTransferTime,
  stageRequestKvBytes,
  type KvTransferConfig,
  type KvTransferResult,
} from './kvTransfer';
import { simulate } from './simulate';
import { resolveSpeculative } from './specdec';
import type { SimInput, SimResult } from './types';
import { validateInput } from './validate';

/** One instance of a pool: the device set one `simulate()` run models. */
export interface PoolSpec {
  /** Hardware of ONE instance (device, node count, links, host). */
  hardware: HardwareStack;
  /** Parallel layout of ONE instance (tp, pp, dp, ep, dcp, …); its devices are tp·pp·dp. */
  parallel: ParallelConfig;
  /** GLOBAL concurrent sequences of one instance step; must divide by `parallel.dp`. Count. */
  batch: number;
}

/** The disaggregation knobs that are not a pool. */
export interface DisaggregationConfig {
  /** Output tokens per request O (OSL): tokens the decode pool generates per request. Integer ≥ 1. Unit: tokens. */
  outputLen: number;
  /** The user's ratio n_P : n_D, prefill side: prefill instances in the system. Integer ≥ 1. Count. */
  prefillInstances: number;
  /** The user's ratio n_P : n_D, decode side: decode instances in the system. Integer ≥ 1. Count. */
  decodeInstances: number;
  /**
   * The KV transfer from prefill to decode (link and overlap). Absent = no transfer cost: the prefill cycle is the
   * pool's own and the decode side has no ingress bound (results identical to running without it).
   */
  kvTransfer?: KvTransferConfig;
}

/** A disaggregated deployment: what both pools share, the two pools, and the knobs. */
export interface DisaggregatedInput {
  /**
   * Everything both pools share: model, quant, workload (P, N, prefix cache, prefill chunk), options, deployment and
   * overlays. `workload.batch` is ignored: each pool has its own batch.
   */
  shared: Omit<SimInput, 'hardware' | 'parallel'>;
  /** The pool that runs prefill (its run's prefill phase is used). */
  prefillPool: PoolSpec;
  /** The pool that runs decode (its run's decode phase is used). */
  decodePool: PoolSpec;
  /** Output length and instance ratio. */
  config: DisaggregationConfig;
}

/** The rates of ONE instance of a pool, for the phase that pool runs. */
export interface PoolRates {
  /** Devices of one instance, D = tp·pp·dp (`SimResult.devicesUsed`). Count. */
  devices: number;
  /**
   * Prefill pool: prompt tokens/s R_P,eff @ P (cached prefix tokens included), i.e. R_P·cycle_P / cycle_eff with a
   * KV transfer and R_P without one; decode pool: generated tokens/s R_D @ N. Aggregate over the instance's DP
   * replicas. Unit: tokens/s.
   */
  tokensPerSec: number;
  /** The context the rate is quoted at: P for the prefill pool, N for the decode pool. Unit: tokens. */
  contextTokens: number;
  /**
   * Requests one instance completes per second: λ_P,eff = R_P,eff / P, or λ_D = R_D / O (decode compute only; the
   * ingress bound is separate, {@link KvTransferResult.ingressRequestsPerSec}). Unit: requests/s.
   */
  requestsPerSec: number;
}

/** The instance ratio at which neither pool idles, and what it yields. */
export interface BalanceResult {
  /**
   * r* = n_P / n_D = λ_D,eff / λ_P,eff: prefill instances per decode instance at balance (real), with
   * λ_D,eff = min(λ_D, λ_recv,1). Dimensionless.
   */
  prefillPerDecode: number;
  /** Share of all devices that are prefill devices at balance, r*·D_P / (r*·D_P + D_D), in (0, 1). Dimensionless. */
  prefillDeviceShare: number;
  /** Output tokens/s per device at balance, λ_D,eff·O / (D_D + r*·D_P). Unit: tokens/s per device. */
  outputTokensPerSecPerDevice: number;
}

/**
 * What limits the system at the user's ratio: the prefill side, the decode compute, the decode side's KV ingress
 * (only with a KV transfer), or 'balanced' when prefill and the decode side are equal.
 */
export type DisaggregationBottleneck = 'prefill' | 'decode' | 'kvIngress' | 'balanced';

/** The whole system at the user's integer ratio n_P : n_D. */
export interface SystemResult {
  /** n_P. Count. */
  prefillInstances: number;
  /** n_D. Count. */
  decodeInstances: number;
  /** All devices, G = n_P·D_P + n_D·D_D. Count. */
  devices: number;
  /**
   * Requests the system completes per second, λ_sys = min(n_P·λ_P,eff, n_D·λ_D, n_D·λ_recv,1) (no ingress term
   * without a KV transfer). Unit: requests/s.
   */
  requestsPerSec: number;
  /** Prompt tokens prefilled per second, λ_sys·P. Unit: tokens/s. */
  inputTokensPerSec: number;
  /** Output tokens generated per second, λ_sys·O. Unit: tokens/s. */
  outputTokensPerSec: number;
  /** Output tokens/s per device, λ_sys·O / G. Unit: tokens/s per device. */
  outputTokensPerSecPerDevice: number;
  /**
   * The smallest capacity of the three, or 'balanced' when the prefill side equals the decode side
   * (n_D·min(λ_D, λ_recv,1)) within {@link BALANCE_REL_TOLERANCE}.
   */
  bottleneck: DisaggregationBottleneck;
  /** Busy fraction of the prefill pool, λ_sys / (n_P·λ_P,eff), in (0, 1]. Dimensionless. */
  prefillUtilization: number;
  /** Busy fraction of the decode pool's compute, λ_sys / (n_D·λ_D), in (0, 1]. Dimensionless. */
  decodeUtilization: number;
}

/** The result of a disaggregated deployment. */
export interface DisaggregatedResult {
  /**
   * The prefill pool's full `simulate()` result, run with `decodeContextLen = max(1, P − 1)` so its memory fit is at
   * context P (see {@link poolInput}); only its prefill phase is a headline, its decode phase is never shown.
   */
  prefillPool: SimResult;
  /** The decode pool's full `simulate()` result (the shared workload with the decode pool's batch). */
  decodePool: SimResult;
  /** One prefill instance: D_P, R_P @ P, λ_P. */
  prefill: PoolRates;
  /** One decode instance: D_D, R_D @ N, λ_D. */
  decode: PoolRates;
  /** The balanced ratio r* and the per-device output rate at it. */
  balance: BalanceResult;
  /** The system at the user's ratio. */
  system: SystemResult;
  /** The KV transfer; absent when `config.kvTransfer` is. */
  kvTransfer?: KvTransferResult;
}

/**
 * Relative tolerance under which the two sides' capacities count as equal ('balanced'), so that a ratio set to r*
 * is not reported as bound by floating-point noise. Dimensionless.
 */
export const BALANCE_REL_TOLERANCE = 1e-9;

/**
 * The `SimInput` of one pool: the shared input with the pool's hardware, parallel layout and batch. The prefill
 * pool holds only the prompt, so its run uses `decodeContextLen = max(1, P − 1)`: the decode `kvLen` is then P and the
 * memory fit's context `max(P, kvLen)` is exactly P (P + k with spec-dec, conservative). Its prefill phase does not
 * depend on N. The decode pool's run keeps the shared N.
 */
export function poolInput(
  shared: DisaggregatedInput['shared'],
  pool: PoolSpec,
  role: 'prefill' | 'decode',
): SimInput {
  const P = shared.workload.promptLen;
  // Formula: N_prefill-pool = max(1, P − 1)  ⇒  fit context = max(P, N + 1) = P
  // See docs/simulation-model.md §14 "Pools"
  const decodeContextLen =
    role === 'prefill' ? Math.max(1, P - 1) : shared.workload.decodeContextLen;
  return {
    ...shared,
    hardware: pool.hardware,
    parallel: pool.parallel,
    workload: { ...shared.workload, batch: pool.batch, decodeContextLen },
  };
}

/**
 * One prefill instance's rates from its pool's result: R_P = prefill tok/s @ P and, with a KV transfer, the rate
 * over the lengthened cycle R_P,eff = R_P·cycle_P / cycle_eff; λ_P,eff = R_P,eff / P.
 */
export function prefillPoolRates(
  result: SimResult,
  promptLen: number,
  /** cycle_eff from the KV transfer; absent = cycle_P (no transfer). Unit: seconds. */
  effectivePrefillCycleSec?: number,
): PoolRates {
  const R_P = result.prefill.tokensPerSec;
  const cycle_P = result.prefill.cycleSec;
  const cycle_eff = effectivePrefillCycleSec ?? cycle_P;
  // Formula: R_P,eff = R_P · (cycle_P / cycle_eff);  λ_P,eff = R_P,eff / P
  // See docs/simulation-model.md §14 "Request rates", "KV transfer"
  const R_P_eff = R_P * (cycle_P / cycle_eff);
  const lambda_P = R_P_eff / promptLen;
  return {
    devices: result.devicesUsed,
    tokensPerSec: R_P_eff,
    contextTokens: promptLen,
    requestsPerSec: lambda_P,
  };
}

/**
 * One decode instance's rates from its pool's result: R_D = decode tok/s @ N, λ_D = R_D / O. Every output token is
 * taken at the rate @ N (a fixed decode context, not an average over P..P+O).
 */
export function decodePoolRates(
  result: SimResult,
  decodeContextLen: number,
  outputLen: number,
): PoolRates {
  const R_D = result.decode.tokensPerSec;
  // Formula: λ_D = R_D / O
  // See docs/simulation-model.md §14 "Request rates"
  const lambda_D = R_D / outputLen;
  return {
    devices: result.devicesUsed,
    tokensPerSec: R_D,
    contextTokens: decodeContextLen,
    requestsPerSec: lambda_D,
  };
}

/**
 * Balanced ratio r* = n_P / n_D = λ_D / λ_P: prefill instances per decode instance at which both pools are busy all
 * the time (n_P·λ_P = n_D·λ_D). A real number. Dimensionless.
 */
export function balancedPrefillPerDecode(
  prefillRequestsPerSec: number,
  decodeRequestsPerSec: number,
): number {
  // Formula: r* = λ_D / λ_P
  // See docs/simulation-model.md §14 "Balanced ratio"
  return decodeRequestsPerSec / prefillRequestsPerSec;
}

/**
 * The balance point: r*, the prefill share of the devices, and the output rate per device. At balance one decode
 * instance serves λ_D,eff = min(λ_D, λ_recv,1) requests/s, i.e. R_D·(λ_D,eff / λ_D) output tok/s, on its D_D devices
 * plus r*·D_P prefill devices.
 */
export function balancedDeployment(
  prefill: PoolRates,
  decode: PoolRates,
  /** λ_recv,1, one decode instance's KV ingress; absent = no bound (no KV transfer). Unit: requests/s. */
  ingressRequestsPerSec = Infinity,
): BalanceResult {
  // Formula: λ_D,eff = min(λ_D, λ_recv,1)
  const lambda_D_eff = Math.min(decode.requestsPerSec, ingressRequestsPerSec);
  const r = balancedPrefillPerDecode(prefill.requestsPerSec, lambda_D_eff);
  const prefillDevicesPerDecodeInstance = r * prefill.devices;
  const devicesPerDecodeInstance = decode.devices + prefillDevicesPerDecodeInstance;
  // Formula: out per decode instance = λ_D,eff·O = R_D · (λ_D,eff / λ_D)
  const outputTokensPerSec = decode.tokensPerSec * (lambda_D_eff / decode.requestsPerSec);
  // Formula: share = r*·D_P / (r*·D_P + D_D);  out/device = λ_D,eff·O / (D_D + r*·D_P)
  // See docs/simulation-model.md §14 "Balanced ratio"
  return {
    prefillPerDecode: r,
    prefillDeviceShare: prefillDevicesPerDecodeInstance / devicesPerDecodeInstance,
    outputTokensPerSecPerDevice: outputTokensPerSec / devicesPerDecodeInstance,
  };
}

/**
 * The system at the user's integer ratio n_P : n_D. The request rate is the smallest of the sides' capacities
 * (prefill, decode compute, and with a KV transfer the decode side's ingress); the other sides idle for the rest.
 */
export function systemThroughput(args: {
  /** One prefill instance's rates (λ_P,eff). */
  prefill: PoolRates;
  /** One decode instance's rates (λ_D, compute). */
  decode: PoolRates;
  /** λ_recv,1, one decode instance's KV ingress; absent = no bound (no KV transfer). Unit: requests/s. */
  ingressRequestsPerSec?: number;
  /** P. Unit: tokens. */
  promptLen: number;
  /** O. Unit: tokens. */
  outputLen: number;
  /** n_P. Count. */
  prefillInstances: number;
  /** n_D. Count. */
  decodeInstances: number;
}): SystemResult {
  const { prefill, decode, promptLen: P, outputLen: O } = args;
  const n_P = args.prefillInstances;
  const n_D = args.decodeInstances;
  const prefillCapacity = n_P * prefill.requestsPerSec;
  const decodeCapacity = n_D * decode.requestsPerSec;
  const ingressCapacity = n_D * (args.ingressRequestsPerSec ?? Infinity);
  // Formula: λ_sys = min(n_P·λ_P,eff, n_D·λ_D, n_D·λ_recv,1);  G = n_P·D_P + n_D·D_D
  // See docs/simulation-model.md §14 "System throughput"
  const decodeSideCapacity = Math.min(decodeCapacity, ingressCapacity);
  const lambda_sys = Math.min(prefillCapacity, decodeSideCapacity);
  const G = n_P * prefill.devices + n_D * decode.devices;
  const outputTokensPerSec = lambda_sys * O;
  const balanced =
    Math.abs(prefillCapacity - decodeSideCapacity) <=
    BALANCE_REL_TOLERANCE * Math.max(prefillCapacity, decodeSideCapacity);
  const bottleneck: DisaggregationBottleneck = balanced
    ? 'balanced'
    : prefillCapacity < decodeSideCapacity
      ? 'prefill'
      : ingressCapacity <= decodeCapacity
        ? 'kvIngress'
        : 'decode';
  return {
    prefillInstances: n_P,
    decodeInstances: n_D,
    devices: G,
    requestsPerSec: lambda_sys,
    // Formula: in = λ_sys·P;  out = λ_sys·O;  out/device = λ_sys·O / G
    inputTokensPerSec: lambda_sys * P,
    outputTokensPerSec,
    outputTokensPerSecPerDevice: outputTokensPerSec / G,
    bottleneck,
    // Formula: util_P = λ_sys / (n_P·λ_P,eff);  util_D = λ_sys / (n_D·λ_D)
    prefillUtilization: lambda_sys / prefillCapacity,
    decodeUtilization: lambda_sys / decodeCapacity,
  };
}

/** True for an integer ≥ 1. */
function isPositiveInteger(x: number): boolean {
  return Number.isInteger(x) && x >= 1;
}

/**
 * Validation errors of a disaggregated input: the config's ranges, then each pool's `validateInput` errors prefixed
 * "prefill pool: " / "decode pool: ". Empty = valid.
 */
export function disaggregationErrors(input: DisaggregatedInput): string[] {
  const { config } = input;
  const errors: string[] = [];
  if (!isPositiveInteger(config.outputLen))
    errors.push(`disaggregation.outputLen must be an integer ≥ 1 (got ${config.outputLen})`);
  if (!isPositiveInteger(config.prefillInstances))
    errors.push(
      `disaggregation.prefillInstances must be an integer ≥ 1 (got ${config.prefillInstances})`,
    );
  if (!isPositiveInteger(config.decodeInstances))
    errors.push(
      `disaggregation.decodeInstances must be an integer ≥ 1 (got ${config.decodeInstances})`,
    );
  const link = config.kvTransfer?.link;
  if (link !== undefined) {
    const B = link.bandwidthPerDirectionBytesPerSec;
    if (!(B > 0))
      errors.push(`disaggregation.kvTransfer.link bandwidth must be > 0 B/s (got ${B})`);
    if (!(link.latencySec >= 0))
      errors.push(`disaggregation.kvTransfer.link latency must be ≥ 0 s (got ${link.latencySec})`);
  }
  for (const role of ['prefill', 'decode'] as const) {
    const pool = role === 'prefill' ? input.prefillPool : input.decodePool;
    for (const e of validateInput(poolInput(input.shared, pool, role)))
      errors.push(`${role} pool: ${e}`);
  }
  return errors;
}

/**
 * Simulates a disaggregated deployment: one `simulate()` per pool, the KV transfer when configured, then the request
 * rates, the balanced ratio and the system at the user's ratio. Throws `Error(errors.join('\n'))` on an invalid
 * input, like `simulate()`.
 */
export function simulateDisaggregated(input: DisaggregatedInput): DisaggregatedResult {
  const errors = disaggregationErrors(input);
  if (errors.length > 0) throw new Error(errors.join('\n'));
  const { shared, config } = input;
  const P = shared.workload.promptLen;
  const N = shared.workload.decodeContextLen;
  const prefillPool = simulate(poolInput(shared, input.prefillPool, 'prefill'));
  const decodePool = simulate(poolInput(shared, input.decodePool, 'decode'));
  const kvTransfer =
    config.kvTransfer === undefined
      ? undefined
      : poolKvTransfer(input, config.kvTransfer, prefillPool, decodePool);
  const prefill = prefillPoolRates(prefillPool, P, kvTransfer?.effectivePrefillCycleSec);
  const decode = decodePoolRates(decodePool, N, config.outputLen);
  const ingress = kvTransfer?.ingressRequestsPerSec;
  return {
    prefillPool,
    decodePool,
    prefill,
    decode,
    balance: balancedDeployment(prefill, decode, ingress),
    system: systemThroughput({
      prefill,
      decode,
      ...(ingress === undefined ? {} : { ingressRequestsPerSec: ingress }),
      promptLen: P,
      outputLen: config.outputLen,
      prefillInstances: config.prefillInstances,
      decodeInstances: config.decodeInstances,
    }),
    ...(kvTransfer === undefined ? {} : { kvTransfer }),
  };
}

/**
 * The KV transfer of this deployment, from the two pools' runs: the per-stage bytes on the prefill pool's PP split
 * (drafter included under spec-dec), its per-replica batch B_r = batch / dp and tp, its prefill latency and cycle,
 * and the decode instance's D_D receivers.
 */
function poolKvTransfer(
  input: DisaggregatedInput,
  config: KvTransferConfig,
  prefillPool: SimResult,
  decodePool: SimResult,
): KvTransferResult {
  const { shared } = input;
  const parallel = input.prefillPool.parallel;
  const draft = resolveSpeculative(shared.model, shared.deployment?.speculative)?.lowering.draft;
  return kvTransferTime({
    stageRequestBytes: stageRequestKvBytes(
      shared.model,
      shared.quant,
      parallel.pp,
      shared.workload.promptLen,
      draft,
    ),
    promptLen: shared.workload.promptLen,
    // Formula: B_r = B_P / dp
    replicaBatch: input.prefillPool.batch / parallel.dp,
    prefillTp: parallel.tp,
    decodeDevices: decodePool.devicesUsed,
    layers: shared.model.layers.length,
    link: config.link,
    overlap: config.overlap,
    prefillLatencySec: prefillPool.prefill.latencySec,
    prefillCycleSec: prefillPool.prefill.cycleSec,
  });
}
