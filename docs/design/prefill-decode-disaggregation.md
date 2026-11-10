# Design: prefill/decode disaggregation (separate pools, KV transfer)

Status: revision 2 (after a critic review); implemented in the sim (commit A: pools, rates, balance, system;
commit B: KV transfer, §4); the UI follows. Task
[`#specmaxx/prefill-decode-disaggregation-separate-p`](`ref://task:01a1c329-b2a2-7549-9f2c-29a5a96231ae`); evidence
and sources in `docs/research/large-scale-deployment-gaps.md` §G1. Companion of `docs/simulation-model.md` (cited
as "model doc §N"; this feature becomes its new §14). Style and contract follow
`docs/design/offload-specdec-latency.md` §0.

## 0. Principle and contract

Disaggregation is an **opt-in deployment-scenario knob, off by default**. Prefill and decode run on two separate
device pools, each with its own hardware, parallel layout and batch; finished prompts move their KV cache from a
prefill instance to a decode instance over a link.

- **Strict identity.** Off ⇒ every number on every page, and the URL, are bit-identical to today.
- **Why identity holds (mechanism).** `simulate()` is not changed: no line of `engine/simulate.ts`,
  `engine/types.ts` or `engine/pipeline.ts` is edited. Disaggregation is a new pure module
  `src/sim/engine/disaggregation.ts` that calls `simulate()` once per pool and combines the two results with a few
  closed-form functions. Its result is a new type next to `SimResult`.
- **The off path, explicitly.** With `AppState.disaggregation` absent, `buildDisaggregatedInput` returns
  `undefined`, so the UI builds and runs only today's single-pool `SimInput` (the runner gets two arguments and its
  result has no `pd` key). `encodeState` emits none of the new keys, so the URL is unchanged, and an old URL
  decodes to a state without `disaggregation`. Existing fixtures, goldens and `identity.test.ts` are compared
  **without regenerating** them.
- **On ⇒ still the hardware/math upper bound** for the stated pool sizes, ratio, decode context and link. Each
  pool's rate is the ordinary single-pool theoretical maximum; the only new costs are the KV bytes crossing the
  P→D link, at the sending and at the receiving NICs.
- **Out of scope:** queueing and arrival variance, TTFT/ITL SLO scheduling, the RDMA software stack, KV-cache
  tiering (§G10, a later task), routing of requests to instances, and redistributing received KV into the decode
  pool's layout (beyond the receiver bound of §4).

Notation: P = prompt length (ISL, `workload.promptLen`), N = decode context (`workload.decodeContextLen`), O =
output tokens per request (OSL, new), L = `model.layers.length`, D_P / D_D = devices of one prefill / decode
instance (`SimResult.devicesUsed` = tp·pp·dp of that pool; DCP subdivides the TP ranks and adds no devices), B_P =
global batch of one prefill instance and B_r = B_P / dp_P its per-replica batch, n_P : n_D = the user's instance
ratio (integers ≥ 1). Times in seconds, bytes in bytes, bandwidths in B/s per direction, rates in tok/s
"@ context", request rates in requests/s.

## 1. What exists today (verified in the code)

| Area                   | Code                                                                                                                              | Fact relied on below                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single-pool simulation | `engine/simulate.ts: simulate`                                                                                                    | one device set; returns `prefill` and `decode` `PhaseResult`s, `memory`, `devicesUsed = tp·pp·dp`                                                                                                                                                                                                                                                                                                                                 |
| Phase rates            | `engine/types.ts: PhaseResult`; model doc §7 "Pipeline and DP"                                                                    | prefill `cycle = m·b`, `tokensPerSec = dp·B_r·P/cycle`, `latencySec` = TTFT of a replica batch; decode `tokensPerSec = dp·B_r/cycle`                                                                                                                                                                                                                                                                                              |
| Memory-fit context     | `simulate.ts` (`contextTokens: max(P, decodeCtx.kvLen)`, decode `kvLen = N + 1`, `N + k + 1` with spec-dec); `engine/validate.ts` | `decodeContextLen` must be an integer ≥ 1                                                                                                                                                                                                                                                                                                                                                                                         |
| Sequence state         | `engine/memory.ts: stageSequenceStateBytesPerSeq` → `lower/lower.ts: blockSequenceStateBytes`                                     | Σ over stage s's layers (`stageLayerRange`, uneven when pp ∤ L) of KV / MLA latent / indexer keys (× context, windows capped) + fixed recurrent and conv state. The per-kind functions read only `ctx.quant`, `ctx.parallel.tp` and `ctx.parallel.dcp` (GQA and linear attention shard by tp; MLA is replicated over tp; DCP keeps ceil(ctx/d) tokens per rank), so a context with tp = 1 and no dcp gives the unreplicated bytes |
| Drafter state          | `lower/draft.ts: draftSequenceStateBytes`; `engine/specdec.ts: resolveSpeculative`                                                | the drafter's per-sequence cache (MTP modules' layer state, or a block drafter's KV) on the last stage; the resolved `DraftSpec` comes from `resolveSpeculative(model, deployment.speculative)`                                                                                                                                                                                                                                   |
| DCP                    | `ir/model.ts: ParallelConfig.dcp`; model doc §5 "Decode context parallelism (DCP)"                                                | a per-`ParallelConfig` field, so per pool; must divide tp; adds no devices                                                                                                                                                                                                                                                                                                                                                        |
| RDMA links             | `hw/presets/links.ts: IB_HDR_200, IB_NDR_400, IB_XDR_800`                                                                         | sourced `LinkSpec`s, one NIC per GPU: 25e9 / 50e9 / 100e9 B/s per direction, `latencySec` 1e-6                                                                                                                                                                                                                                                                                                                                    |
| DP attention + EP      | model doc §5 "DP attention + EP"                                                                                                  | tp = 1, dp = ep = N: each device holds only its own `B_r` sequences' KV                                                                                                                                                                                                                                                                                                                                                           |
| EP imbalance, EPLB     | `ir/model.ts: ParallelConfig.epImbalance, redundantExpertsPerRank`                                                                | in `ParallelConfig`, so per pool for free                                                                                                                                                                                                                                                                                                                                                                                         |
| Prefix cache           | `engine/types.ts: Workload.prefixCacheHitRatio`; model doc §7 "Prefix cache"                                                      | prefill computes `P_new` tokens; `tokensPerSec` still counts all P prompt tokens                                                                                                                                                                                                                                                                                                                                                  |
| Rate formatting        | `ui/state/format.ts: formatRateAtContext, formatContextTokens, formatBytes`                                                       | "6,400 tok/s @ 8k" (en-US separators; ≥ 1000 no decimals, ≥ 10 one); context 1k = 1024, other contexts exact ("8,704"); bytes in SI units ("147kB")                                                                                                                                                                                                                                                                               |

## 2. Pools

A **pool** is the set of identical instances that run one phase. One instance is an ordinary `simulate()` device
set: one `HardwareStack`, one `ParallelConfig`, one global batch.

| Shared by both pools                                                                                                     | Per pool                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| model, quant, P, N, prefix-cache ratio, prefill chunk, `options`, `deployment` (offload placement, spec-dec), `overlays` | hardware (device, node count, links, host), parallel (tp, pp, dp, ep, dcp, layout, ρ, redundant slots), batch |

- **Prefill-pool memory fit at P.** The prefill pool's run uses `decodeContextLen = max(1, P − 1)`, so its decode
  `kvLen = P` and the fit's `max(P, kvLen)` is exactly P: a prefill instance holds only the prompt. No engine
  change is needed. With spec-dec the fit is at P + k (the verify window) and still adds the verify snapshots:
  conservative by k tokens, documented. The prefill phase does not read N (asserted in ID-PD-1). The pool's decode
  phase is never shown.
- The **decode pool's** run is the plain input. Its prefill phase is shown only in that pool's details.
- Offload: the placement (`deployment.offload`) is shared, the host RAM / SSD / host link come from each pool's
  own hardware. Validation runs per pool, so e.g. spec-dec with pp > 1, or a dcp that does not divide tp, is
  rejected for that pool with a "prefill pool: " / "decode pool: " prefix.

## 3. Output definitions and how they are quoted

Each number below carries its context; the quote format is part of the definition.

**Pool rates.**

- R_P = `prefillPool.prefill.tokensPerSec`: prompt tokens/s of **one** prefill instance (cached prefix tokens
  included). Quoted "131,072 tok/s @ 8k" (context P). λ_P = R_P / P requests/s.
- R_D = `decodePool.decode.tokensPerSec`: generated tokens/s of **one** decode instance (spec-dec acceptance
  included when on). Quoted "6,400 tok/s @ 8,704" (context N). λ_D = R_D / O requests/s.
- **Fixed decode context.** Every one of the O output tokens is taken at the rate @ N. The system figures are
  therefore the result **for the chosen decode context N**, not an exact average over the output trajectory
  P..P+O. The help text suggests N ≈ P + O/2 as a representative choice.
- With KV transfer (§4): R_P,eff = R_P · cycle_P / cycle_eff, λ_P,eff = R_P,eff / P, and the decode side is capped
  by its ingress, λ_D,eff = min(λ_D, λ_recv,1). Without it (commit A): R_P,eff = R_P and λ_D,eff = λ_D.

**Quote of the cross-pool figures.** They depend on P, O and N, so they carry all three in one compact form,
`@ <P>/<O>, decode @ <N>`, produced by one new formatter `formatWorkloadQuote(P, O, N)` (each number through
`formatContextTokens`): "@ 8k/1k, decode @ 8,704". Every system and balance figure uses exactly this suffix.

**Balanced ratio** (the instance ratio at which neither side idles):

```
r* = n_P / n_D = λ_D,eff / λ_P,eff          prefill instances per decode instance (real number)
device share of prefill = r*·D_P / (r*·D_P + D_D)
output tok/s per device at balance = λ_D,eff·O / (D_D + r*·D_P)
```

Derivation: balance means n_P·λ_P,eff = n_D·λ_D,eff. One decode instance then serves λ_D,eff requests/s, i.e.
λ_D,eff·O output tok/s (= R_D when ingress does not bind), on D_D of its own devices plus r*·D_P prefill devices.

Quoted "P:D 1 : 2.56 @ 8k/1k, decode @ 8,704" (instances, the smaller side normalized to 1, two decimals), with the
device ratio "8 : 20.5 GPUs" next to it.

**System at the user's ratio** (n_P, n_D integers ≥ 1):

```
λ_sys  = min(n_P·λ_P,eff, n_D·λ_D, λ_recv)          requests/s,  λ_recv = n_D·λ_recv,1 (§4; ∞ without transfer)
       = min(n_P·λ_P,eff, n_D·λ_D,eff)
G      = n_P·D_P + n_D·D_D                          devices
output tok/s = λ_sys·O        input tok/s = λ_sys·P
output tok/s per device = λ_sys·O / G
utilization_P = λ_sys / (n_P·λ_P,eff)    utilization_D = λ_sys / (n_D·λ_D)    (compute utilization)
bottleneck = 'balanced'  if |n_P·λ_P,eff − n_D·λ_D,eff| ≤ 1e-9·max(both)
           = 'prefill'   if n_P·λ_P,eff is the smallest
           = 'kvIngress' if λ_recv is the smallest (ties with n_D·λ_D go to 'kvIngress')
           = 'decode'    otherwise
```

Quoted "533.3 tok/s/GPU @ 8k/1k, decode @ 8,704" (output tok/s per device). Totals use the same suffix
("12,800 tok/s @ 8k/1k, decode @ 8,704"). The device word follows the existing UI ("GPU").

**Interpretation of "per pool: prefill tok/s and decode tok/s".** The hero shows the prefill pool's prefill @ P and
the decode pool's decode @ N; the details of the selected pool show both of its phases, as today.

## 4. KV transfer (commit B)

**Bytes per request, per prefill stage.** For prefill stage s (of pp_P), the stage's unreplicated per-sequence state
at P tokens, with the drafter's state on the last stage when spec-dec is on:

```
ctx₁ = { quant: shared quant, parallel: { tp: 1, pp: pp_P, dp: 1, ep: 1 } }      (no dcp)
S_s(P) = stageSequenceStateBytesPerSeq(model, ctx₁, s, P)
         + [s = pp_P − 1 ∧ spec-dec on] draftSequenceStateBytes(draft, ctx₁, P)
S_tot(P) = Σ_s S_s(P)            bytes per request;   bytes/token (display) = S_tot(P) / P
```

This covers GQA KV (windows capped), MLA latents, indexer keys, CSA, shared-KV layers (counted once), recurrent /
conv state (constant in P), each at the KV dtype with its scale overhead, and the drafter's own cache (MTP modules'
layer state or a block drafter's KV; `draft` is `resolveSpeculative(...).lowering.draft`, the one `simulate` uses;
on the prefill pool's last stage, where the drafter runs at any pp since spec-dec supports PP > 1; docs/simulation-model.md
§13 "Pipeline parallelism"). The drafter state is therefore **transferred**, not rebuilt.
All P tokens move even with a prefix-cache hit (the decode instance needs the whole cache). Quoted "147kB/token,
1.2GB per request @ 8k".

**Link.** One `LinkSpec` per device, per direction, **the same at both ends**: bandwidth B_link (B/s) and latency α
(s). Default `IB_NDR_400` (one 400 Gb/s NIC per GPU, as on DeepSeek's H800 nodes and Mooncake's 8×400 Gb nodes); the
user may pick another preset or edit both numbers. No new preset numbers are added.

**Sender: the busiest device.** In one prefill cycle each replica finishes its B_r sequences. A device of stage s
holds that stage's layers for those sequences; its tp_P ranks each send a distinct 1/tp slice of the stage's unique
bytes (sharded KV heads are distinct already; replicated MLA latents and DCP-sharded caches still let each rank
send a different slice). The busiest sender bounds the time:

```
Y    = max_s B_r·S_s(P) / tp_P          bytes per sending device per prefill cycle
t_bw = Y / B_link                       its link occupancy per cycle (bandwidth only)
t_req = α + max_s S_s(P) / (tp_P·B_link)    one request alone on its replica's devices
```

Uneven PP splits come out right: the stage with the most layers sends most. With DP attention (tp = 1, pp = 1) each
device sends exactly its own B_r sequences. Note the per-replica batch B_r = B_P / dp_P (the global batch would
count dp replicas onto one device).

**Receiver: decode-side ingress.** A decode instance's D_D devices receive every request's S_tot(P), spread evenly
over them (bandwidth only):

```
λ_recv,1 = D_D·B_link / S_tot(P)        requests/s one decode instance can ingest
λ_recv   = n_D·λ_recv,1                 the system's ingress bound, in λ_sys's min()
```

It enters the balance through λ_D,eff = min(λ_D, λ_recv,1) (§3). The even spread is optimistic for a decode layout
that differs from the prefill one; re-sharding the received KV into the decode layout (TP, DCP, a different PP split)
over the decode pool's scale-up links is not modelled beyond this bound.

**Throughput effect on the prefill pool** (the only place the transfer changes rates):

```
layer-wise overlap (default):  cycle_eff = max(cycle_P, t_bw)
not overlapped (the toggle):   cycle_eff = cycle_P + α + t_bw
```

α is latency, not occupancy (model doc §7 "Pipeline and DP" treats links the same way): writes are pipelined, so in
the overlapped case α never adds to the cycle. The not-overlapped case is a **wait-for-arrival policy**: the instance
starts its next batch only when the last byte of this batch has arrived, so the whole α + t_bw is serial.

**Exposed transfer latency (display-only approximation).** Shown as the TTFT a request gains; it never feeds a rate.
Assumptions: equal per-layer compute a = t_pf / L (t_pf = prefill `latencySec`), equal chunks β_c = t_bw / L, each
chunk an independent RDMA write that may start once its layer is done, with enough in-flight capacity that the chunk
latencies overlap. The chunks then form a two-stage flow shop and α is paid once, on the last chunk:

```
T       = a + β_c + (L−1)·max(a, β_c) + α                arrival of the last chunk, from prefill start
exposed = T − t_pf
        = α + t_bw / L                     if a ≥ β_c   (link keeps up: only the last chunk shows)
        = α + t_bw − t_pf·(L−1)/L          if β_c > a   (link-bound)
not overlapped: exposed = α + t_bw
```

At L = 1 both give the same value. `T` was checked against a brute-force event simulation of the L chunks (equal
to 1e-15 in both branches). With PP it ignores which stage a layer runs on and uses the busiest sender's t_bw.

**Approximations (also in model doc §9):** the busiest-sender and even-receiver assumptions above; no re-sharding
cost on the decode side; no RDMA-stack or memory-registration overhead. Sanity check against measurement (doc
only): Mooncake reports 87 GB/s per node on 4×200 Gb RoCE and 190 GB/s on 8×400 Gb (report §G1), below the 100 /
400 GB/s line rates this model uses, as an upper bound should be.

## 5. Interactions

- **Spec-dec:** the drafter's prefill runs on the prefill pool (already part of `simulate`'s prefill phase); draft
  and verify run on the decode pool (its decode phase). The drafter's cache is transferred with the target's (§4).
- **DCP:** per pool through `ParallelConfig.dcp`. D = tp·pp·dp either way. It changes each pool's own run (decode
  KV reads and memory per rank) but not S_tot(P), the sender's Y (each TP rank still sends 1/tp of the unique bytes)
  or λ_recv. A prefill pool without DCP feeding a DCP decode pool is allowed; the re-sharding is not modelled (§4).
- **Offload:** shared placement, per-pool host (§2). Each pool's transfers stay inside its own `simulate()` run.
- **Prefix cache:** R_P counts all P tokens, so λ_P = R_P / P is the request rate; all P tokens' KV move.
- **DP attention + EP, ρ, EPLB slots:** per pool through `ParallelConfig` (e.g. prefill DEP32, decode DEP144).
- **Overlays** (engine dispatch) and "prefill without overlap" stay informational per pool; the system numbers use
  the headline rates only.

## 6. UI surface and URL

- The existing `AppState` fields `hw`, `parallel` (including `dcp`), `parallelLayout` and `workload.batch` **are the
  prefill pool**. The decode pool is a copy of those four in `AppState.disaggregation.decodePool` (absent = off;
  never `{}`).
- A pool view `poolView(state, 'decode')` swaps the decode pool's fields into an `AppState`, so the Hardware and
  Parallelism panels, the batch field and `buildSimInput` are reused unchanged. One action
  `{ type: 'inPool', pool: 'decode', action }` runs the ordinary reducer on the view and writes back only the four
  pool fields.
- A "Disaggregation" panel: Off | On; when on, "Pool: Prefill | Decode" (which pool the panels edit; React state,
  not in the URL), "Output length", "Prefill instances", "Decode instances"; commit B adds the KV link picker
  (shadcn Popover + Command), bandwidth and latency fields and "KV transfer: Overlapped | Not overlapped". Each with
  a "?".
- Hero: Prefill (prefill pool, @ P), Decode (decode pool, @ N), and "System" "533.3 tok/s/GPU @ 8k/1k, decode @
  8,704" with "Balanced P:D 1 : 2.56 @ 8k/1k, decode @ 8,704" and the bottleneck; commit B adds "KV 147kB/token,
  1.2GB per request @ 8k". The rig line names both pools ("8 × H100 prefill, 8 × H100 decode").
- URL: `pd=1`, `osl` (default 1024), `pdr=<nP>:<nD>` (default 1:1), `dec.<key>` for each hardware/parallel/batch
  key of the decode pool (including `dcp`); commit B: `kvl`, `kvx`, `kvo=none`. These keys without `pd=1` are
  dropped.

## 7. Type sketches (`src/sim/engine/disaggregation.ts`, `kvTransfer.ts`; JSDoc with units on every field)

```ts
/** One instance of a pool: its hardware, parallel layout (dcp included) and global batch. */
export interface PoolSpec {
  hardware: HardwareStack;
  parallel: ParallelConfig;
  /** GLOBAL sequences of one instance step; divisible by parallel.dp. Count. */
  batch: number;
}
export interface DisaggregationConfig {
  /** Output tokens per request O (OSL), integer ≥ 1. Tokens. */
  outputLen: number;
  /** The user's ratio n_P : n_D, in instances, integers ≥ 1. Count. */
  prefillInstances: number;
  decodeInstances: number;
  /** Commit B. Absent = no transfer cost (cycle_eff = cycle_P, no ingress bound). */
  kvTransfer?: KvTransferConfig;
}
export interface DisaggregatedInput {
  /** What both pools share; `workload.batch` is ignored (each pool has its own). */
  shared: Omit<SimInput, 'hardware' | 'parallel'>;
  prefillPool: PoolSpec;
  decodePool: PoolSpec;
  config: DisaggregationConfig;
}
/** One instance's rates: devices D, tokens/s (R_P,eff @ P or R_D @ N), requests/s (λ_P,eff or λ_D,eff). */
export interface PoolRates {
  devices: number;
  tokensPerSec: number;
  requestsPerSec: number;
}
export interface DisaggregatedResult {
  prefillPool: SimResult; // run with decodeContextLen = max(1, P − 1): fit at P
  decodePool: SimResult;
  prefill: PoolRates;
  decode: PoolRates;
  balance: {
    prefillPerDecode: number;
    prefillDeviceShare: number;
    outputTokensPerSecPerDevice: number;
  };
  system: {
    prefillInstances: number;
    decodeInstances: number;
    devices: number;
    requestsPerSec: number;
    inputTokensPerSec: number;
    outputTokensPerSec: number;
    outputTokensPerSecPerDevice: number;
    bottleneck: 'prefill' | 'decode' | 'kvIngress' | 'balanced';
    prefillUtilization: number;
    decodeUtilization: number;
  };
  kvTransfer?: KvTransferResult;
}
export function poolInput(
  shared: DisaggregatedInput['shared'],
  pool: PoolSpec,
  role: 'prefill' | 'decode',
): SimInput;
export function balancedPrefillPerDecode(
  prefillRequestsPerSec: number,
  decodeRequestsPerSec: number,
): number;
export function systemThroughput(args: {
  prefill: PoolRates;
  decode: PoolRates; // λ_D (compute)
  ingressRequestsPerSecPerDecodeInstance?: number; // λ_recv,1; absent = ∞
  outputLen: number;
  promptLen: number;
  prefillInstances: number;
  decodeInstances: number;
}): DisaggregatedResult['system'];
export function disaggregationErrors(input: DisaggregatedInput): string[]; // "prefill pool: " / "decode pool: " prefixes
export function simulateDisaggregated(input: DisaggregatedInput): DisaggregatedResult; // throws like simulate()

// kvTransfer.ts (commit B)
export interface KvTransferConfig {
  /** P→D link per device per direction, the same at both ends; only bandwidth and α are read. */
  link: Pick<LinkSpec, 'name' | 'bandwidthPerDirectionBytesPerSec' | 'latencySec'>;
  overlap: 'layerwise' | 'none';
}
/** S_s(P) per prefill stage (drafter state on the last stage), bytes per request. */
export function stageRequestKvBytes(
  model: ModelSpec,
  quant: QuantConfig,
  prefillPp: number,
  promptLen: number,
  draft?: DraftSpec,
): number[];
export interface KvTransferResult {
  stageRequestBytes: number[]; // S_s(P), bytes
  requestBytes: number; // S_tot(P), bytes
  bytesPerToken: number; // S_tot(P)/P, bytes per token
  requestSec: number; // t_req, seconds
  batchBytesPerDevice: number; // Y (busiest sender), bytes per device per prefill cycle
  bandwidthSec: number; // t_bw = Y / B_link, seconds
  exposedSec: number; // TTFT added, seconds (display-only approximation)
  prefillCycleSec: number; // cycle_eff, seconds
  ingressRequestsPerSec: number; // λ_recv,1 of one decode instance, requests/s
  overlap: 'layerwise' | 'none';
}
export function kvTransferTime(args: {
  stageRequestBytes: number[];
  replicaBatch: number; // B_r of the prefill pool
  prefillTp: number;
  decodeDevices: number; // D_D
  layers: number;
  link: KvTransferConfig['link'];
  overlap: 'layerwise' | 'none';
  prefillLatencySec: number;
  prefillCycleSec: number;
}): KvTransferResult;
```

Changes against the planning sketch: `linkSec` (= L·α + t_bw) became `bandwidthSec` (= t_bw); `requestKvBytes`
became per-stage `stageRequestKvBytes` with the drafter; the ingress bound and `'kvIngress'` are new.

## 8. Identity and tests

- **ID-PD-1:** both pools equal to one `SimInput`'s hardware / parallel / batch ⇒ `decodePool` `toEqual`
  `simulate(input)`, and `prefillPool.prefill` `toEqual` `simulate(input).prefill` (the prefill phase does not read
  N). Also a MoE DP attention + EP case and a pp 2 case. `prefillPool.memory` equals `simulate` at
  `decodeContextLen = P − 1` and its fit context is P.
- **ID-PD-2:** `identity.test.ts` and every existing golden pass without regeneration (no engine file changes).
- **ID-PD-3 (UI):** every existing fixture state has no `disaggregation`; `encodeState` output is unchanged; the
  runner result has no `pd` key without the third argument.
- **ID-PD-4:** `kvTransfer` absent ⇒ the commit-A result; B_link → ∞ and α = 0 ⇒ cycle_eff === cycle_P and
  λ_recv = ∞.
- **KV bytes:** W-PD1; an MLA model ((rkv + dr)·bpe·L·P); a sliding-window layer (capped); a hybrid (recurrent state
  constant in P); FP8 KV (half of bf16 plus scale overhead); MTP spec-dec (S_tot grows by the MTP layer's state).
- **Uneven PP:** REF_DENSE_8B, P = 8192, prefill pp 5 (stages of 7, 7, 7, 7, 8 layers), tp 1, dp 1, B_P = 1:
  S_4 = 8·4096·8192 = 268,435,456 B is the busiest, so Y = 268,435,456 B and t_bw = 0.00536870912 s at 50e9 B/s
  (an even split would give 241,591,910.4 B); t_req = 1e-6 + 0.00536870912 s.
- **DCP:** an MLA fixture with a tp 8 decode pool at dcp 1 and dcp 8, against a prefill pool with the same layout
  and with a different one (tp 1 · dp 8): S_tot, Y and λ_recv are unchanged by dcp, D = tp·pp·dp in every case,
  and the decode pool's own run equals `simulate` with that dcp.
- **Errors:** outputLen 0, instances 0 or non-integer, pool batch not divisible by dp, dcp not dividing tp in either
  pool (prefixed messages).
- **e2e:** DeepSeek's 32:144 shape (prefill DEP32 on 4 nodes, decode DEP144 on 18 nodes); it asserts the layout and
  the quote formats, not DeepSeek's measured numbers.

### 8.1 Worked example W-PD1 (closed-form; pool rates given, not simulated)

Model REF_DENSE_8B (36 layers, 8 KV heads × 128, bf16 KV), P = 8192, O = 1024, N = 8704 (= P + O/2). Every quote
below ends in "@ 8k/1k, decode @ 8,704" (abbreviated "@ …").

S_tot(P) (pp 1, so one stage): per token per layer 2·8·128·2 B = 4096 B; ×36 = **147,456 B/token**; ×8192 =
**1,207,959,552 B** per request ("147kB/token, 1.2GB per request @ 8k").

Prefill instance: tp 1, pp 1, dp 8 ⇒ D_P = 8, B_P = 8, B_r = 1, given cycle_P = t_pf = 0.5 s.
R_P = 8·8192 / 0.5 = **131,072 tok/s @ 8k**, λ_P = 16 req/s.
Decode instance: dp 8 ⇒ D_D = 8, batch 256, given cycle_D = 0.04 s. R_D = 256 / 0.04 = **6,400 tok/s @ 8,704**,
λ_D = 6400 / 1024 = 6.25 req/s.

No transfer (commit A):

| Output                                     | Value                                                                        | Quote                               |
| ------------------------------------------ | ---------------------------------------------------------------------------- | ----------------------------------- |
| r* = 6.25 / 16                             | 0.390625 (1/r* = 2.56)                                                       | "P:D 1 : 2.56 @ …", "8 : 20.5 GPUs" |
| prefill device share = 3.125 / 11.125      | 0.280899                                                                     |                                     |
| at balance = 6400 / (8 + 3.125)            | 575.2809                                                                     | "575.3 tok/s/GPU @ …"               |
| 1:2 ⇒ λ_sys = min(16, 12.5) = 12.5, G = 24 | 12,800 out tok/s; 102,400 in; 533.333 /GPU; decode; util_P 0.78125, util_D 1 | "533.3 tok/s/GPU @ …"               |
| 1:3 ⇒ λ_sys = min(16, 18.75) = 16, G = 32  | 16,384 out tok/s; 512 /GPU; prefill; util_D 0.853333                         | "512.0 tok/s/GPU @ …"               |

KV transfer, L = 36, α = 1e-6 s. Y = B_r·S_tot / tp = 1,207,959,552 B. At 50e9 B/s (`IB_NDR_400`): t_bw =
0.02415919104 s, t_req = 0.02416019104 s, a = 0.5/36 = 0.0138889 s ≥ β_c = 0.00067108864 s; λ_recv,1 =
8·50e9 / S_tot = 331.137 req/s (not binding). At 1e9 B/s: t_bw = 1.207959552 s, λ_recv,1 = 8e9 / S_tot =
6.6227383 req/s ≥ λ_D = 6.25 (not binding, but close).

| Case                        | exposed (s)   | cycle_eff (s) | λ_P,eff    | λ_D,eff   | r*        | at balance /GPU | 1:2 /GPU           |
| --------------------------- | ------------- | ------------- | ---------- | --------- | --------- | --------------- | ------------------ |
| layer-wise, 50 GB/s         | 0.00067208864 | 0.5           | 16         | 6.25      | 0.390625  | 575.2809        | 533.333 (decode)   |
| none, 50 GB/s               | 0.02416019104 | 0.52416019104 | 15.2625097 | 6.25      | 0.4095001 | 567.5771        | 533.333 (decode)   |
| layer-wise, 1 GB/s          | 0.7218494409  | 1.207959552   | 6.6227383  | 6.25      | 0.9437184 | 411.5823        | 282.5702 (prefill) |
| none, 1 GB/s                | 1.207960552   | 1.707960552   | 4.6839489  | 6.25      | 1.3343442 | 342.7087        | 199.8485 (prefill) |
| layer-wise, 1 GB/s, D_D = 4 | 0.7218494409  | 1.207959552   | 6.6227383  | 3.3113692 | 0.5       | 423.8553        | see below          |

The 1 GB/s layer-wise rows take the link-bound branch: 1e-6 + 1.207959552 − 0.5·35/36 = 0.7218494409. The last row
keeps R_D = 6400 tok/s but on a 4-device decode instance (dp 4, B_r 64), so its ingress λ_recv,1 = 4e9 / S_tot =
3.3113692 req/s < λ_D and binds: r* = 3.3113692 / 6.6227383 = 0.5 ("P:D 1 : 2.00 @ …"), at balance
3.3113692·1024 / (4 + 0.5·8) = 423.8553 tok/s/GPU. At 1:1 (G = 12): λ_sys = min(6.6227, 6.25, 3.3114) = 3.3113692,
bottleneck 'kvIngress', 3,391 out tok/s, 282.5702 /GPU ("282.6 tok/s/GPU @ …"), util_D 0.529819. At 1:2 (G = 16):
prefill 6.6227383 and ingress 6.6227383 tie ⇒ 'balanced' (within the 1e-9 tolerance), 423.8553 /GPU.
Other quotes: the 1 GB/s layer-wise r* reads "P:D 1 : 1.06 @ …"; the not-overlapped 1 GB/s r* (> 1) reads
"P:D 1.33 : 1 @ …". Tests assert these with relative tolerance 1e-9 (values above rounded for reading).

## 9. Decisions (after the critic review) and open questions

1. **Fixed decode context N** for all O tokens; the results say so (§3). Follow-up if wanted: average over a few
   contexts in P..P+O.
2. **Quote** `@ P/O, decode @ N` on every cross-pool figure (§3).
3. **α once, not per layer**, and never occupancy (§4): accepted.
4. **Integer user ratios**; r* is real, shown with the smaller side at 1: accepted.
5. **Busiest sender** per PP stage with S_s split over tp; **receiver ingress bound** with even spread, the same link
   spec both ends (§4).
6. **Drafter state is transferred** (§4).
7. **Prefill fit at exactly P** via `decodeContextLen = max(1, P − 1)`; P + k with spec-dec (conservative).
8. Open: re-sharding cost when the decode layout differs from the prefill one (TP/DCP/PP) is not modelled; it would
   use the decode pool's scale-up link and belongs to a later task if the user wants it.

## 10. Implementation notes

Where `src/sim/engine/disaggregation.ts` and `kvTransfer.ts` differ from the sketch in §7:

- `KvTransferResult` also carries `link`, `overlap` and both cycles (`prefillCycleSec` = cycle_P,
  `effectivePrefillCycleSec` = cycle_eff); `ingressRequestsPerSec` is per decode instance (λ_recv,1). The exposed
  time and the cycle have their own functions (`exposedTransferSec`, `effectivePrefillCycleSec`), and the default
  link is `DEFAULT_KV_TRANSFER_LINK` (= `IB_NDR_400`, the sourced preset).
- `balancedDeployment(prefill, decode, ingressRequestsPerSec?)` and `systemThroughput({ …, ingressRequestsPerSec? })`
  take the ingress as an optional argument; absent = ∞. Without a KV transfer the ratios cycle_P / cycle_eff and
  λ_D,eff / λ_D are exactly 1, so every number is bit-identical to running without it (ID-PD-4).
- With a symmetric deployment (same devices and link on both sides) the prefill sender limit and the decode ingress
  are the same `D·B / S_tot`, so a link-bound 1:1 system is reported 'balanced' (tested).
- `PoolRates` also carries `contextTokens` (P for the prefill pool, N for the decode pool), the context its rate is
  quoted at.
- The balance and system results are named types (`BalanceResult`, `SystemResult`, `DisaggregationBottleneck`), and
  the rates have their own small functions (`prefillPoolRates`, `decodePoolRates`, `balancedDeployment`), each
  with a `// Formula:` comment and a link to model doc §14.
