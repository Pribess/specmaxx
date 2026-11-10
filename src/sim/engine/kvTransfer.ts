/**
 * KV transfer of prefill/decode disaggregation: a finished prompt's cache (KV, MLA latents, indexer keys, recurrent
 * state, and the drafter's own cache under spec-dec) moves from the prefill instance that computed it to a decode
 * instance, over one RDMA link per device. This file computes the bytes, the sender's link time per prefill cycle,
 * the TTFT the transfer adds, the lengthened prefill cycle, and the decode side's ingress bound.
 *
 * Only bandwidth and one latency α per transfer: no RDMA software stack, memory registration or re-sharding of the
 * received cache into the decode pool's layout. See docs/simulation-model.md §14 "KV transfer"; design
 * docs/design/prefill-decode-disaggregation.md §4.
 */

import { IB_NDR_400 } from '../hw/presets/links';
import type { LinkSpec } from '../hw/types';
import type { DraftSpec, ModelSpec, QuantConfig } from '../ir/model';
import { draftSequenceStateBytes } from '../lower/draft';
import type { LoweringContext } from '../lower/lower';
import { stageSequenceStateBytesPerSeq } from './memory';

/** The P→D link and how the transfer overlaps the prefill compute. */
export interface KvTransferConfig {
  /**
   * The link of ONE device, per direction, the same at both ends (sender NIC on a prefill device, receiver NIC on a
   * decode device). Only `bandwidthPerDirectionBytesPerSec` (B_link, B/s) and `latencySec` (α, s) are read.
   */
  link: Pick<LinkSpec, 'name' | 'bandwidthPerDirectionBytesPerSec' | 'latencySec'>;
  /**
   * `'layerwise'` (default in the UI): each layer's cache leaves as soon as that layer finishes, while the next
   * layers compute. `'none'`: the transfer starts after the whole prefill, and the instance waits for the last byte
   * to arrive before its next batch (wait-for-arrival).
   */
  overlap: 'layerwise' | 'none';
}

/**
 * Default P→D link: InfiniBand NDR400, one 400 Gb/s NIC per GPU (50e9 B/s per direction, α = 1 µs), as on
 * DeepSeek's H800 nodes and Mooncake's 8 × 400 Gb nodes. The sourced preset of `hw/presets/links.ts`, reused as is.
 */
export const DEFAULT_KV_TRANSFER_LINK: KvTransferConfig['link'] = IB_NDR_400;

/** The KV transfer of one disaggregated deployment. */
export interface KvTransferResult {
  /** The link used (as configured). */
  link: KvTransferConfig['link'];
  /** The overlap assumption used. */
  overlap: 'layerwise' | 'none';
  /** S_s(P): one request's unreplicated cache held by prefill stage s (drafter on the last stage). Unit: bytes. */
  stageRequestBytes: number[];
  /** S_tot(P) = Σ_s S_s(P): one request's whole cache. Unit: bytes per request. */
  requestBytes: number;
  /** S_tot(P) / P, for display. Unit: bytes per prompt token. */
  bytesPerToken: number;
  /** t_req = α + max_s S_s(P) / (tp·B_link): one request alone on its replica's devices. Unit: seconds. */
  requestSec: number;
  /** Y = max_s B_r·S_s(P) / tp: what the busiest prefill device sends per prefill cycle. Unit: bytes per device. */
  batchBytesPerDevice: number;
  /** t_bw = Y / B_link: the busiest sender's link occupancy per prefill cycle (no α). Unit: seconds. */
  bandwidthSec: number;
  /**
   * The TTFT the transfer adds for the last request of a batch (display-only approximation, never a rate input):
   * layer-wise, the flow-shop tail; not overlapped, α + t_bw. Unit: seconds.
   */
  exposedSec: number;
  /** cycle_P: the prefill pool's own cycle, without the transfer. Unit: seconds. */
  prefillCycleSec: number;
  /** cycle_eff: the prefill cycle with the transfer (`max(cycle_P, t_bw)` or `cycle_P + α + t_bw`). Unit: seconds. */
  effectivePrefillCycleSec: number;
  /** λ_recv,1 = D_D·B_link / S_tot(P): requests one decode instance can take in per second. Unit: requests/s. */
  ingressRequestsPerSec: number;
}

/**
 * S_s(P) for every prefill stage s: the cache one request holds at P tokens on stage s, unreplicated (as if tp = 1,
 * no DCP), at the shared KV dtype. With spec-dec the drafter's own cache (MTP modules' layer state, or a block
 * drafter's KV) is added to the last stage, where the drafter runs (spec-dec forces pp = 1). Uneven PP splits
 * (`stageLayerRange`) give uneven stages. Unit: bytes per request, one entry per stage.
 */
export function stageRequestKvBytes(
  model: ModelSpec,
  quant: QuantConfig,
  prefillPp: number,
  promptLen: number,
  draft?: DraftSpec,
): number[] {
  // One sequence, tp = dp = ep = 1 and no dcp: the state functions then count every byte once.
  const ctx1: LoweringContext = {
    phase: 'prefill',
    seqs: 1,
    qLen: promptLen,
    kvLen: promptLen,
    quant,
    parallel: { tp: 1, pp: prefillPp, dp: 1, ep: 1 },
  };
  const bytes: number[] = [];
  for (let s = 0; s < prefillPp; s++) {
    const targetBytes = stageSequenceStateBytesPerSeq(model, ctx1, s, promptLen);
    const isLast = s === prefillPp - 1;
    const draftBytes =
      isLast && draft !== undefined ? draftSequenceStateBytes(draft, ctx1, promptLen) : 0;
    // Formula: S_s(P) = state_s(P; tp 1) + [s = pp − 1] S_draft(P)
    // See docs/simulation-model.md §14 "KV transfer"
    bytes.push(targetBytes + draftBytes);
  }
  return bytes;
}

/**
 * The TTFT the transfer adds (display only). Layer-wise: L equal layers of compute a = t_pf / L, each followed by
 * an equal chunk of link occupancy β_c = t_bw / L that may start once its layer is done and the previous chunk has
 * left; chunks are pipelined writes, so α is paid once, on the last. Not overlapped: α + t_bw after the prefill.
 * Unit: seconds.
 */
export function exposedTransferSec(args: {
  /** t_pf: prefill latency of the batch (`PhaseResult.latencySec`). Unit: seconds. */
  prefillLatencySec: number;
  /** t_bw. Unit: seconds. */
  bandwidthSec: number;
  /** α. Unit: seconds. */
  latencySec: number;
  /** L, the model's layers (one chunk each). Count. */
  layers: number;
  overlap: 'layerwise' | 'none';
}): number {
  const { prefillLatencySec: t_pf, bandwidthSec: t_bw, latencySec: alpha, layers: L } = args;
  // Formula (none): exposed = α + t_bw
  if (args.overlap === 'none') return alpha + t_bw;
  const a = t_pf / L;
  const beta_c = t_bw / L;
  // Formula (layer-wise): T = a + β_c + (L−1)·max(a, β_c) + α;  exposed = T − t_pf
  //   = α + t_bw / L if a ≥ β_c;  = α + t_bw − t_pf·(L−1)/L if β_c > a
  // See docs/simulation-model.md §14 "KV transfer"
  const lastChunkArrivalSec = a + beta_c + (L - 1) * Math.max(a, beta_c) + alpha;
  return lastChunkArrivalSec - t_pf;
}

/**
 * The prefill cycle with the transfer. Layer-wise the link is a second resource next to compute and only its
 * bandwidth occupies it (α is latency, not occupancy): `max(cycle_P, t_bw)`. Not overlapped (wait-for-arrival):
 * `cycle_P + α + t_bw`. Unit: seconds.
 */
export function effectivePrefillCycleSec(args: {
  /** cycle_P. Unit: seconds. */
  prefillCycleSec: number;
  /** t_bw. Unit: seconds. */
  bandwidthSec: number;
  /** α. Unit: seconds. */
  latencySec: number;
  overlap: 'layerwise' | 'none';
}): number {
  const { prefillCycleSec: cycle_P, bandwidthSec: t_bw, latencySec: alpha } = args;
  // Formula: cycle_eff = max(cycle_P, t_bw) (layer-wise);  cycle_P + α + t_bw (none)
  // See docs/simulation-model.md §14 "KV transfer"
  return args.overlap === 'layerwise' ? Math.max(cycle_P, t_bw) : cycle_P + alpha + t_bw;
}

/**
 * The KV transfer of one deployment: bytes, the busiest sender's time per prefill cycle, the exposed TTFT, the
 * lengthened prefill cycle and the decode side's ingress bound (receivers spread evenly over the D_D devices).
 */
export function kvTransferTime(args: {
  /** S_s(P) per prefill stage ({@link stageRequestKvBytes}). Unit: bytes. */
  stageRequestBytes: number[];
  /** P. Unit: tokens. */
  promptLen: number;
  /** B_r = B_P / dp: sequences one prefill replica finishes per cycle. Count. */
  replicaBatch: number;
  /** tp of the prefill pool (ranks that split one stage's bytes). Count. */
  prefillTp: number;
  /** D_D, devices of one decode instance (receivers). Count. */
  decodeDevices: number;
  /** L, the model's layers. Count. */
  layers: number;
  link: KvTransferConfig['link'];
  overlap: 'layerwise' | 'none';
  /** t_pf, the prefill pool's `prefill.latencySec`. Unit: seconds. */
  prefillLatencySec: number;
  /** cycle_P, the prefill pool's `prefill.cycleSec`. Unit: seconds. */
  prefillCycleSec: number;
}): KvTransferResult {
  const { stageRequestBytes, link, overlap } = args;
  const B_link = link.bandwidthPerDirectionBytesPerSec;
  const alpha = link.latencySec;
  const tp = args.prefillTp;
  // Formula: S_tot = Σ_s S_s;  busiest stage S_max = max_s S_s
  const requestBytes = stageRequestBytes.reduce((sum, b) => sum + b, 0);
  const busiestStageBytes = Math.max(...stageRequestBytes);
  // Formula: Y = B_r·S_max / tp;  t_bw = Y / B_link;  t_req = α + S_max / (tp·B_link)
  // See docs/simulation-model.md §14 "KV transfer"
  const batchBytesPerDevice = (args.replicaBatch * busiestStageBytes) / tp;
  const bandwidthSec = batchBytesPerDevice / B_link;
  const requestSec = alpha + busiestStageBytes / (tp * B_link);
  // Formula: λ_recv,1 = D_D·B_link / S_tot
  const ingressRequestsPerSec = (args.decodeDevices * B_link) / requestBytes;
  return {
    link,
    overlap,
    stageRequestBytes,
    requestBytes,
    bytesPerToken: requestBytes / args.promptLen,
    requestSec,
    batchBytesPerDevice,
    bandwidthSec,
    exposedSec: exposedTransferSec({
      prefillLatencySec: args.prefillLatencySec,
      bandwidthSec,
      latencySec: alpha,
      layers: args.layers,
      overlap,
    }),
    prefillCycleSec: args.prefillCycleSec,
    effectivePrefillCycleSec: effectivePrefillCycleSec({
      prefillCycleSec: args.prefillCycleSec,
      bandwidthSec,
      latencySec: alpha,
      overlap,
    }),
    ingressRequestsPerSec,
  };
}
