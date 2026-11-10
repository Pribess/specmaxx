# Design: chunked pipeline prefill (and ring context parallelism, deferred)

Status: chunked pipeline prefill (CPP) implemented in the sim (`src/sim/engine/chunkedPipeline.ts`, wired in
`simulate.ts` step 4a); ring context parallelism (CP) not implemented, its math is in §4 for a follow-up task. Task
[`#specmaxx/advanced-low-priority-prefill-context-pa`](`ref://task:01a1c329-b2b1-73ba-8397-9881d9688842`); evidence in
`docs/research/large-scale-deployment-gaps.md` §G9. The model, formulas and code links are in
`docs/simulation-model.md` §7 "Chunked pipeline prefill" (cited as "model doc"); this note records the decision.

## 0. Contract

- **Opt-in, off by default.** `ParallelConfig.chunkedPipelinePrefill?: boolean`. Absent or false, results are bit
  for bit those of before; also with pp > 1 when the prompt is one chunk (P_new ≤ C), since the engine then takes the
  whole-prompt path. With pp = 1 it is ignored with a warning. Pinned by `identity.test.ts` ID-7 over every case of
  the deployment identity set, without regenerating its fixture.
- **On ⇒ still the hardware/math bound** of the stated schedule: each chunk is an ordinary per-op lowering; the only
  new thing is the order in which stages and links serve the chunks. No engine scheduling, no chunk mixing.

## 1. Decision: CPP first, ring CP deferred

The task asked for the cheaper option first. CPP is cheaper here, measured against the code:

|                                                            | Chunked pipeline prefill                                               | Ring CP                                                                                                                                                                                                                                                |
| ---------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Devices and rank layout                                    | none new (the PP stages)                                               | a new dimension, world = tp·pp·dp·cp: `devicesUsed`, `checkPlacement` / `groupSpan` (a group strided by tp), `ppSendPath` partners, `hasInterNodeTraffic`, `stageNStream`, multi-NIC sharing, the disaggregation sender bound, the UI's degree pickers |
| Decode                                                     | unchanged                                                              | the KV stays sharded on cp ranks: a new decode mode (batch-split GEMMs, Q all-gather, (o, lse) merge over cp) that no source specifies (vLLM: prefill CP "under active development")                                                                   |
| Serial state (linear attention, Mamba-2, KDA, short convs) | the existing `qLen < kvLen` prior-state lowering (prefix cache)        | state passes rank to rank: a serial chain                                                                                                                                                                                                              |
| Lowering                                                   | exists: a chunk is the prefix-cache shape `{ qLen: q_j, kvLen: kv_j }` | new: per-rank zigzag shards, ring KV sends per layer                                                                                                                                                                                                   |

What CPP buys in this model is **TTFT of one long prompt** (up to pp× lower at batch 1). The prefill headline
(`tokensPerSec = dp·B_r·P/cycle`, a steady stream) already gets the full pipelining credit with whole-prompt
microbatches, so it moves only by the chunking costs. Ring CP has the same property here: at a fixed device count, PP
already gets the same steady-stream throughput and the same KV split. So the UI shows the TTFT next to the headline.

**Verify mapping.** The task's checks were written for ring CP. Under CPP: "per-rank FLOPs" become per-chunk FLOPs (Σ
attention pairs over chunks = the unchunked pairs exactly; Σ GEMM FLOP equal), "ring traffic" becomes the per-chunk PP
send bytes (Σ = the unchunked send). Plus the identity tests and a 1M-token run (model doc §7 worked example; e2e
`e2e/chunked-pipeline.spec.ts`). Ring CP's traffic and FLOP tests belong to its follow-up task.

## 2. Model (summary; formulas in model doc §7 "Chunked pipeline prefill")

- Chunks: `K = ceil(P_new / C)`, `q_j = min(C, P_new − j·C)`, `kv_j = cached + j·C + q_j`; C =
  `Workload.prefillChunkTokens` (default 8192), per sequence. Chunks before the last have no LM head.
- Schedule: stages and boundary links as machines serving (microbatch, chunk) units in order; link occupancy
  `f_pp·β`, delay `f_pp·(p − β)`; TTFT = when the last unit leaves the last stage;
  `cycle = m·max(max_s Σ_j t_{s,j}, max_s Σ_j β_{s,j})`.
- Result: `PhaseResult.latencySec` / `cycleSec` chunked, `stages` the per-stage chunk sums (ops merged per position),
  `PhaseResult.chunkedPipeline = { chunkTokens, chunks, unchunked: PhaseRates }`.

Numbers at 1M (Kimi-K3 fixture, 4 × 8 B300, tp 8 pp 4 ep 8, batch 1, K = 128): TTFT 69.06 s → 20.01 s, prefill
55,086 → 52,784 tok/s @ 1M. MiniMax-M3 (GQA): 22.64 s → 7.49 s, 145,680 → 143,978 tok/s @ 1M. One `simulate` with
128 chunks takes about 0.6–0.8 s (the whole-prompt one about 25 ms), so no chunk-count cap was needed.

## 3. Approximations and known effects

- **What chunking costs** (falls out of the per-op model, not tuned): every chunk reads every weight (K weight reads),
  its GEMMs have `b_mb·q_j` rows (lower shape efficiency at small C), its attention re-reads every earlier chunk's KV,
  each PP send pays α once per chunk, offloaded weights are re-streamed per chunk. At C = 1024 on the tiny test models
  this makes CPP _slower_ (`chunkedPipeline.test.ts` comment); at C = 8192 on real models it costs 0.4–4%.
- **MLA dense prefill** (`lower/mla.ts`: MHA form, `kv_b` with m = seqs·kvLen): each chunk decompresses every visible
  key, Σ_j kv_j ≈ P·(K + 1)/2 rows. Kimi-K3 at 1M: `kv_b` 0.04 s → 2.27 s, against 45 s of attention, so CPP still
  wins on TTFT by 3.45×; the headline loses 4.2%. This is the math of that form, not tuned.
- **Fractional f_pp with m > 1:** the recursion counts the link occupancy outside the stage as f_pp·β, while the
  whole-prompt formula's bottleneck `b` counts the full β. Both agree at f_pp ∈ {0, 1}; K = 1 takes the old path.
- **Memory fit:** the activation peak is the largest chunk `{ seqs: b_mb, qLen: C, kvLen: P }` (at b_mb = 1 the step
  of before); KV per stage is the whole prompt's. The UI's batch sweep reads only decode and this fit, so it calls
  `simulate(input, { skipChunkedPrefillSchedule: true })`: the same fit and decode without the K chunk costings.
- **Offload sharing** (review P1): the K chunks of a microbatch occupy stages at the same time, so each chunk's
  transfers are costed with m·K units in flight (`simulatePhase` `unitsInFlight`): without prefetching,
  `min(m·K, |S_tier|)` stages of a node share the tier, not `min(m, |S_tier|)`. Then TTFT ≥ the node tier's time
  for all its bytes (Σ bytes / B_tier), checked in `chunkedPipeline.test.ts` (REF_MOE_30B_A3B, experts in DDR4 host
  RAM, pp 4 on one node: ≥ 36.24 s, where m = 1 sharing gave 12.04 s). The sharing also applies while the pipeline
  fills and drains, when fewer stages stream, so the TTFT is a little conservative there.
- **KV-cache tiering** (`OffloadConfig.kvCache`, model doc §12 "KV cache off-device", "Remote KV pool"). Decision:
  a chunk's past splits into the prefix-cache hit and the earlier chunks of the same prefill
  (`LoweringContext.prefixCacheTokens`), and each part is placed as the memory fit places it: the prefix at the share
  f_pre on the device, the earlier chunks as live KV at the share f_dev (the fit keeps only f_dev of each sequence's KV
  on the device). **Every chunk re-reads the off-device part of both**: Σ_j M = K·(1 − f_pre)·cached +
  (1 − f_dev)·C·K(K − 1)/2 tokens per layer, against (1 − f_pre)·cached once whole-prompt. Why: a chunk must not hold
  more of the sequence's KV on the device than the fit counts (review: retaining every earlier chunk at f_dev = 0.25
  gave a false fit), and §12 stages each layer's fetch and drops it; reading once and keeping it would need uncounted
  device memory. The re-reads share host RAM / SSD with m·K units in flight (above), the pool per device. Staging:
  the last chunk fetches the most, so the fit counts it as one more phase of `stageStagingBytes` (also on the sweep
  path, which lowers only that chunk); tier contents are unchanged. Tests: `chunkedPipeline.test.ts` ("KV-cache
  offload": the review repro at f_dev 0.25, 0.75·28·8192 tokens × 36 layers fetched at K = 8, the fit's resident KV
  unchanged, 352,321,536 B of staging and no fit, the sweep path's memory equal; the prefix in host RAM, 4× the
  whole-prompt bytes at K = 4; TTFT ≥ bytes / B_ram in both).
- Not modelled: engine chunk scheduling, mixing chunks of other requests or decode steps into a chunk, per-stage or
  adaptive chunk sizes; the drafter's own head runs on every chunk.

## 4. Ring CP math (for the follow-up task; not implemented)

n = cp degree (new devices). Layout: rank = tpIdx + tp·(cpIdx + cp·(dpIdx + dp·ppIdx)). Zigzag sharding (Meta,
[arXiv 2411.01783](https://arxiv.org/html/2411.01783v2): Llama-3-405B, 1M tokens in 77 s on 128 H100): 2n chunks of
c = P/(2n) tokens; rank i holds chunks i and 2n−1−i.

- **Pairs per rank:** step 0 (own KV) `c² + c(c+1)`; each of the n−1 ring steps exactly `2c²` (j < i: q_i and
  q_{2n−1−i} see kv_j; j > i: q_{2n−1−i} sees kv_j and kv_{2n−1−j}). Total per rank `2n·c² + c = P(P+1)/(2n)` exactly
  when 2n | P (contiguous sharding instead: the last rank ≈ (2n−1)/n × the mean). `FLOP_rank = 2·seqs·Hl·pairs_rank·
  (qk + v)`; GEMMs on P/n tokens.
- **Ring traffic:** n−1 steps per attention layer, each sending the rank's own KV chunk `S_step = (P/n)·2·KVl·hd·
  bpe(kv)` (GQA; MLA: the latent `(P/n)(rkv + dr)·bpe(kv)`). Per layer `(n−1)·(α + S_step/B)`, the ring all-gather
  of `S = P·(per-token KV)`.
- **Overlap per layer:** `a_0 + Σ_{k=1}^{n−1} max(a_k, α + S_step/B)`, a_k = the attention of step k (equal under
  zigzag).
- **Memory:** KV of P/n tokens per rank; activations of P/n tokens.
- **Composition:** TP inside a cp rank (heads), PP outside; DCP is decode-only and orthogonal.
- **Open:** the decode mode on cp ranks, serial recurrent state across ranks, MLA dense prefill, sparse attention,
  pass-Q for prefix-cache hits (Meta), the KV-transfer sender bound under disaggregation.
