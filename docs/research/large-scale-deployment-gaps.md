# Large-scale deployment options specmaxx cannot model yet

Research report. Goal: a broad, evidence-backed list of deployment options used by real frontier-scale LLM serving that specmaxx does not model (or models only partly), graded by **practicability**.

Grades: **A** = common in production / default in recipes (main UI). **B** = used but niche or model/hardware-specific (Advanced section). **C** = rare, experimental, negligible effect, or engine overhead (do not implement).

Fit is judged against the project principle: specmaxx estimates a hardware/math upper bound and excludes engine implementation overheads (paged attention, continuous batching, scheduler, kernel-launch). Opt-in deployment-scenario knobs (like spec-dec and offload) are legitimate.

## Method and caveats

- vLLM: the [`vllm-project/recipes`](https://github.com/vllm-project/recipes) repo (commit `20d870f1`, 2026-11-09; rendered at recipes.vllm.ai) was parsed. **76 recipes with ≥100B total parameters** were counted. Counts below are "of 76". Families repeat (5 Kimi-K2.x, 5 GLM-4.x/5, 5 MiniMax-M2.x, …), so counts overstate independent families. Recipe `compatible_strategies` is mostly builder boilerplate (e.g. 49/76 list PD); only explicit pool configs, default strategies, guide commands and benchmark tables are used as evidence.
- Cross-checked with SGLang/LMSYS, TensorRT-LLM, Dynamo, llm-d, DeepSeek's inference-system overview, Mooncake, Meta's context-parallel paper, and AMD/NVIDIA blogs.
- Many sources are dated 2026 and name models newer than common knowledge (Kimi-K3, DeepSeek-V4/V4.1, GLM-5.x, Qwen3.8, MiniMax-M3). They are cited as published and not independently verified.
- **Evidence is thin for**: Llama 3.1 405B / Llama 4 (no 405B vLLM recipe; only Scout; Meta's context-parallel paper is the only primary Meta source); TPU serving numbers (no primary tokens/s for a frontier MoE); production EPLB gains (DeepSeek publishes none; numbers come from SGLang/TRT-LLM experiments); default prefix-hit ratios (DeepSeek 56%, Mooncake 34–55% only); DSA cost figures (chart-derived, secondhand); Qwen3-Next speedups (secondary quotes).
- Large-scale topology (P:D ratios, EP degree, redundant experts, DBO) mostly lives in **blogs and engine docs, not in vLLM recipe flags**: recipes are mostly single-node TP/TEP command lines. The grade therefore weights SGLang/DeepSeek/TRT-LLM/llm-d production evidence too.

## Hardware targeted by the recipes

Among the 70 vLLM recipes with a hardware map (`meta.hardware`): H200 52, MI355X 49, MI300X 40, MI325X 38, B200 19, H100 17, GB200 16, B300 14, GB300 11, MI350X 10. Smaller: DGX Spark, RTX PRO 6000, Ascend, TPU Ironwood (2: gpt-oss-120b, Qwen3-Coder-480B), Xeon. GB200/GB300 are modelled in recipes as 4-GPU NVL4 trays; NVL72 appears in blogs ([DeepSeek-R1 on GB200](https://vllm.ai/blog/dsr1-gb200-part1), [V4.1-Flash on GB300 NVL72](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)). **AMD MI300X/MI325X/MI355X are targeted by more recipes than B200/GB200**, which matters for presets (see G11).

Representative layouts ([recipe YAMLs](https://github.com/vllm-project/recipes/tree/main/models)):

| Model                     | Layout / flags                                                                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepSeek-V3/R1            | 8×H200 `--tensor-parallel-size 8 --enable-expert-parallel` (TEP) or `--data-parallel-size 8 --enable-expert-parallel` (DEP); FP4 on 4×B200; `supports_dcp` |
| DeepSeek-V3.2             | TP8, DSA (sparse attention), opt-in MTP (3 tokens), NVFP4 variant with `--kv-cache-dtype fp8`                                                              |
| DeepSeek-V4-Pro (1.6T)    | `--kv-cache-dtype fp8`, DP8+EP on B300/H200; PD pools DEP+DEP; FP4 indexer KV on Blackwell; MTP-2 / DSpark-7                                               |
| Kimi-K2.5/K2.6            | INT4 8×H200; NVFP4 4×GB200 + FP8 KV; PD pools DEP; `--all2all-backend flashinfer_nvlink_one_sided`; EAGLE3                                                 |
| Kimi-K2-Thinking          | TP8 + `--decode-context-parallel-size 8`: KV tokens 715k → 5.7M (8×), output tok/s +43% (8×H200, 8k in/4k out)                                             |
| Kimi-K3 (2.8T KDA hybrid) | TEP8 prefill + DEP decode (`--data-parallel-hybrid-lb`); GB200 lane TP8×PP2 + DCP8, FP8 KV, DSpark, prefix caching, Mooncake store                         |
| Qwen3-235B / Coder-480B   | TP, or `--enable-expert-parallel` TP8, or FP8 DP8+EP                                                                                                       |
| Qwen3.8-2.4T (GDN hybrid) | GB300: NVFP4 TP8 2170 tok/s/GPU vs DEP16 + `flashinfer_nvlink_one_sided` 4300; MTP-3 raises per-user rate 130 → 307 tok/s                                  |
| GLM-4.x/5.x               | FP8 TP4/8, MTP 1–5, FP8 KV                                                                                                                                 |
| gpt-oss-120b              | 1 GPU or TP4–8, FP8 KV, EAGLE3 (7), prefix caching often disabled; TPU TP2×DP4                                                                             |
| MiniMax-M2.x/M3           | TP4/8 (+EP), FP8 KV, EAGLE3; M3 sparse attention, 1M context                                                                                               |
| Mistral Large 3 (675B)    | FP8 8×H200 / NVFP4 4×B200, TP only                                                                                                                         |

Feature frequency across the 76 recipes (regex over YAML + guides, approximate): FP8 weights offered 64; NVFP4 43; explicit `--kv-cache-dtype fp8` 44–47 (FP4 KV: 0); spec-dec 49 (41 opt-in; MTP 35, DSpark 12, DFlash 10, EAGLE 8); EP flag ≈44; DEP as recommended layout ≥10; PD with explicit pool config 14; DCP launch 4 (11 flag support, all MLA); `--all2all-backend` set 3–4 (all `flashinfer_nvlink_one_sided`; pplx 0); EPLB 0; DBO 0; prefix caching explicit 21 (13 on, 10 off); chunked prefill flag 7 (24 tune `max-num-batched-tokens`); KV offload verified 22; sparse attention ≥13; linear/hybrid ≥16; SWA ≥14; MLA ≥14; runtime LoRA 0 (1 merged).

## What specmaxx already models (inventory)

Source: `docs/simulation-model.md` (SM), `src/sim/**`.

**Modeled:** TP (column/row GEMMs, one all-reduce per block, KV-head replication below TP); PP (contiguous stages, microbatches, activation sends); DP (independent replicas, `B_r = batch/dp`); EP mode A (EP inside TP, no all-to-all) and **EP mode B** (EP across DP replicas, all-to-all dispatch/combine; this is functionally attention-DP + wide EP with tp=1, dp=ep, but not labelled so); expert-TP; scalar EP imbalance ρ (default 1.2, engine-level only, scales busiest device's rows, not weights/comm bytes); flat and two-level hierarchical collectives (ring/tree/direct), switched/mesh/ring/chain inter-node fabrics with oversubscription; host-bounce paths; global overlap switch (`ideal` = max(compute, bw) / `none`); rich dtype set (FP8, MXFP8, FP6, FP4, NVFP4, MXFP4, INT4/8, EXL3) with block-scale overhead, separate dtypes for weights/activations/**KV cache**/recurrent state/embeddings/lm-head, per-module weight formats, W4A4/W8A8 input; attention: GQA (+sliding window), MLA (absorbed decode, MHA prefill), sparse indexers (DSA/NSA/CSA-like), linear/state-space (GatedDeltaNet, KDA, Mamba-2); MoE routing with shared experts and distinct-expert-read expectation; spec-dec (MTP, DFlash, DSpark; opt-in; k≤16; iid or per-position acceptance); offload of experts/lookup tables/embeddings to host RAM/SSD with prefetch accuracy; memory fit; batch sweep; presets for A100/V100/H100/H200/B200/B300, RTX, Apple, AMD (MI300X…MI455X), Intel.

**Partial:** attention-DP+wide EP (since done as G2: the named "DP attention + EP" layout, dense FFN / LM head DP or TP over the EP group, MLA latent one copy per DP rank; still no all-to-all dedup); EP imbalance (one scalar, UI doesn't expose ρ); MI325X/MI355X/MI455X (catalog uses PCIe5 placeholders instead of xGMI, `catalog.ts:465`); NVLink domain >8 (engine's `nodeTopology` accepts any `devicesPerNode`, but the UI caps at 8 GPUs × 8 machines and there is no NVL72 preset); chunked prefill (only affects memory-fit activation peak; rate is a whole-prompt pass); spec-dec (rejected with pp>1; no trees; no small-draft-LM/ngram methods).

**Absent:** prefill/decode disaggregation; prefix caching (prefill always from empty cache); context/sequence parallelism (decode context parallelism since done as G8; prefill CP and sequence parallelism still absent); EPLB/redundant experts; two-batch overlap and per-collective overlap fractions; KV-cache offload/tiering; skewed routing for compute; LoRA; NVLS/SHARP in-network reduction; DeepEP-style dispatch dedup; rail-optimized/multi-tier fabrics, multiple NICs/GPU; FP8 collectives and dispatch precision independent of activation dtype; mixed prefill+decode batches; interleaved PP; TPU/Trainium/Gaudi/GB200/GB300 presets.

## Gap list

Effort: S ≈ a day or less, M ≈ days, L ≈ a week+.

### G1. Prefill/decode disaggregation (separate pools) — **A**, in scope as scenario knob

- **What / changes physically:** prefill and decode run on separate device pools with their own parallelism (prefill small EP, decode large EP). Removes prefill/decode interference; adds a per-request KV transfer (ISL × KV-bytes/token over the P→D link, overlapped layer-wise). Output: per-pool rates, P:D ratio for balance, aggregate system tok/s.
- **Evidence:** DeepSeek's service (prefill EP32 on 4 nodes, decode EP144 on 18 nodes, H800) [overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md); SGLang Kimi K2 on 128 H200 (4 prefill + 12 decode nodes) [LMSYS](https://lmsys.org/blog/2025-07-20-k2-large-scale-ep/); SGLang DeepSeek 96-GPU [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/); GB200 NVL72 2P+12D [LMSYS](https://lmsys.org/blog/2025-06-16-gb200-part-1/); Mooncake/Kimi [FAST'25](https://www.usenix.org/conference/fast25/presentation/qin); TRT-LLM [blog 5](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html); Dynamo planner sizes pools by profiled tok/s [design](https://docs.dynamo.nvidia.com/dynamo/v-0-9-0/design-docs/planner-design.md); llm-d well-lit path [repo](https://github.com/llm-d/llm-d); vLLM recipes: 14 explicit PD pool configs (DEP+DEP, TEP prefill + DEP decode). Caveat: vLLM docs call it experimental and say it "does not improve throughput" ([docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/)), mainly for TTFT/ITL tail control and EP stalls ([blog](https://vllm.ai/blog/large-scale-serving)).
- **Parameters:** P:D ratio 1:3 (Kimi K2/SGLang), ~1:2 (GB200 test), DeepSeek unit 32:144 GPUs. MLA KV 35,136 B/token at FP8 ([LMSYS](https://www.lmsys.org/blog/2026-02-19-gb300-longctx/)). Mooncake transfer: 87 GB/s on 4×200 Gb RoCE, 190 GB/s on 8×400 Gb ([slides](https://datatracker.ietf.org/meeting/125/materials/slides-125-irtfopen-disaggregated-architecture-for-llm-inference-mooncake-01)). Reported gains vary: 10–30% (MI300X), up to 70% (gpt-oss B200) per llm-d; 1.5–5× in wide-EP vs TP16 baseline.
- **Fit:** legitimate hardware/math knob (two device sets, KV bytes ÷ link bandwidth). Out of scope: queueing, SLO scheduling, RDMA-stack overhead (**borderline**: TTFT/ITL trade-off).
- **Effort: L.** `src/sim/engine` (new two-phase composition + transfer cost), `hw` topology (second pool + link), `ui` setup panel, result chips.
- **Status: done in the sim (UI in progress).** Two pools (each its own hardware, `ParallelConfig` and batch), each an ordinary `simulate()` run; per-instance request rates λ_P = R_P / P and λ_D = R_D / O (every output token at the decode rate @ N); the balanced ratio r* = λ_D,eff / λ_P,eff; the system at the user's integer ratio, λ_sys = min(n_P·λ_P,eff, n_D·λ_D, n_D·λ_recv,1), as output tok/s per device quoted "@ P/O, decode @ N". The prefill pool's memory fit is at context P. KV transfer: S_s(P) per prefill stage (drafter cache included under spec-dec), the busiest sender's B_r·S_s/tp over one NIC per GPU (default IB NDR400), layer-wise overlap (cycle_eff = max(cycle_P, t_bw)) or not (cycle_P + α + t_bw), a decode-side ingress bound D_D·B / S_tot. Not modelled: RDMA-stack overhead, re-sharding into the decode layout, queueing/SLOs. See `docs/simulation-model.md` §14 and `docs/design/prefill-decode-disaggregation.md`.

### G2. DP-attention + wide EP as a first-class layout (incl. dense-layer DP, DP LM head) — **A**, in scope

- **What:** attention data-parallel (no MLA latent-KV replication across TP), experts sharded over 32–144+ devices, dispatch/combine all-to-all. SGLang additionally runs dense FFNs and the LM head DP instead of TP (halves FFN comm, avoids non-128-aligned shards).
- **Evidence:** DeepSeek EP32/EP144; SGLang EP72 decode 22.3k tok/s/node (5.2× TP16) [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/); TRT-LLM wide-EP EP32 up to 1.8× per GPU vs EP8 on GB200 [NVIDIA](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/); vLLM recipes DEP ≥10 of 76 (V3/R1, V4, K2.5/K2.6/K3 decode, Qwen3.8); vLLM [EP docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/). Caveat: gain drops for small MoEs with few experts (NVIDIA).
- **Fit:** pure hardware/math. Mostly present via EP mode B; the gap is UX (name the layout, preset it), MLA KV not replicated when attention is DP, dense-FFN/LM-head DP option, EP sizes beyond 8-GPU nodes (needs G11).
- **Effort: S–M.** `src/sim/lower/moe.ts`, `lower/common.ts` (dense DP), Parallelism UI.

### G3. EPLB and redundant experts — **A** for large-EP decode (cheap), in scope

- **What:** step time is set by the hottest rank. EPLB replicates hot experts, reducing max/mean load; each redundant slot adds expert weight bytes (≈2.4 GB per rank for DeepSeek-V3 per redundant expert/rank; vLLM docs advise 32 redundant for large use).
- **Evidence:** TRT-LLM hottest rank 1.56× mean without EPLB → ~1.11× with 32 redundant (DeepSeek-R1, 32 GB200) [blog 4](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html); SGLang EPLB prefill 1.49×, decode 2.54×, "perfect EPLB" +18% [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/); DeepSeek 32 redundant experts per unit; Kimi K2/SGLang 96 redundant on 96 decode GPUs; vLLM V4.1-Flash balancedness 0.35 → 0.94, rebalance every 3000 steps ([blog](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)). **Thin evidence:** zero vLLM recipes enable it; DeepSeek publishes no gains; numbers are from two datasets and experiments.
- **Fit:** legitimate as an imbalance multiplier (extends existing ρ) + redundant-slot weight footprint. Skip rebalancing mechanics.
- **Effort: S–M.** `lower/moe.ts` (ρ also on all-to-all bytes of the hottest rank, weights += slots), UI (expose ρ and redundant experts; existing ρ default 1.2 is UI-hidden).
- **Status: done.** The UI exposes ρ ("EP imbalance", default 1.2), "Imbalance on all-to-all" (default off) and "Redundant experts" per EP rank (default 0) for MoE models at EP > 1 (URL `epi`, `epa`, `rex`). With the switch on (`ParallelConfig.epImbalanceScalesAllToAll`), ρ also scales the mode-B dispatch / combine all-to-all bytes to the busiest rank's `A_max` rows; off, they keep the mean volume, so default results are unchanged. Redundant experts add `r·W_expert` per device per MoE layer and change nothing else (ρ stays the user's input; no rebalancing model). Correction to the evidence above: the TRT-LLM table reports (max − mean)/mean, so the hottest rank is 2.56× the mean without EPLB and 1.11× with it, and its "32 redundant experts" are 288 slots for 256 experts at EP32, i.e. 1 per rank. See `docs/simulation-model.md` §5 "EP load imbalance", "Redundant experts".

### G4. Two-batch / dual-batch overlap (comm hidden behind compute) — **B**, in scope

- **What:** two microbatches alternate so one's all-to-all hides behind the other's compute; halves peak activation. Today overlap is one global `ideal`/`none` switch.
- **Evidence:** SGLang prefill +27–35% (up to +40.5%), decode +25.5% at 256 tokens/device, ≤0 below ~64–128 tokens/device [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/); DeepSeek's decode pipeline; vLLM [DBO docs](https://docs.vllm.ai/en/latest/design/dbo/) (needs DP>1+EP+DeepEP; not with PP/LoRA/hybrid). SGLang disabled TBO on NVL72 where comm is small [part 1](https://lmsys.org/blog/2025-06-16-gb200-part-1/). **0 vLLM recipes use it.**
- **Fit:** hardware/math upper bound (max vs sum) with per-collective overlap fraction; the existing `ideal` default already assumes full hiding, so the real value is making `none`/partial available per collective class and showing the loss when batch is small. **Borderline:** minimum batch size for effective overlap is an engine property.
- **Effort: S–M.** `engine/` overlap in pipeline/step composition, UI toggle.
- **Status: done.** `SimOptions.overlapByClass` sets the overlap fraction f ∈ [0, 1] of each collective class (TP: `tp`/`etp`/`dcp` groups; EP: the mode-B all-to-alls and DP-attention collectives; PP: the stage hand-off); a share f of the class's bandwidth time may hide behind the stage's compute (`core = max(compute, Σ f·bw) + Σ (1 − f)·bw`, latency never hidden), and an absent class follows the global `overlap` switch, so default results are unchanged. `DeploymentConfig.twoBatchOverlap` lowers each microbatch as two halves (sequence split, or a causal token split for one prefill sequence) and sets EP to full overlap; smaller GEMMs, a second weight read, twice the message latencies and the halved activation peak fall out of the op model, so it loses at small batch. MoE mode B only, not with offload; the minimum batch for a gain is not enforced. UI: Hardware → Advanced → "Communication overlap" (URL `ovt`, `ove`, `ovp` in percent, `tbo=1`). See `docs/simulation-model.md` §7 "Stage time and overlap", "Two-batch overlap".

### G5. Rack-scale NVLink domains and GB200/GB300 (NVL72) presets — **A**, in scope

- **What:** 72-GPU NVLink domain (130 TB/s aggregate) makes wide EP viable without IB; Grace C2C (900 GB/s bidirectional) gives fast offload. GB300: 288 GB HBM3e, faster softmax SFU.
- **Evidence:** NVIDIA [wide-EP blog](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/); SGLang [GB200](https://lmsys.org/blog/2025-06-16-gb200-part-1/)/[part 2](https://lmsys.org/blog/2025-09-25-gb200-part-2/)/[GB300](https://www.lmsys.org/blog/2026-02-19-gb300-longctx/); vLLM GB200 16 / GB300 11 recipes (as NVL4 trays), NVL72 in blogs.
- **Fit:** pure hardware data. Engine already supports arbitrary `devicesPerNode`; UI caps at 8×8, no NVL72 preset.
- **Effort: S–M.** `src/sim/hw/presets` (GB200/GB300 devices, NVL72 topology), `catalog.ts` limits (devices-per-machine, max machines), UI.
- **Status: done.** GB200 (186 GB) and GB300 (279 GB; the 288 GB above is the full-chip maximum, not the product) device presets, the NVL72 switched domain (900 GB/s per GPU per direction) and GB200 NVL4 trays (bridge mesh estimated), NVLink-C2C as the host link (450 GB/s per GPU per direction, sharing ambiguous) and the pooled Grace LPDDR5X (480 GB, 384 GB/s per Grace). The UI offers up to 72 GPUs per machine and 18 machines. EP stays limited to divisors of the expert count (EP64 on a 64-GPU domain for 256 experts; EP72/EP144 wait for G3). See `docs/simulation-model.md` §4 "Scale-up domains", "Host and storage tiers".

### G6. AMD scale-up presets (MI325X/MI355X xGMI) and other accelerators — **A** (AMD), **C** (TPU)

- **What:** AMD hardware is the most-targeted non-NVIDIA family: MI355X 49, MI300X 40, MI325X 38 recipes. Catalog uses PCIe placeholders for MI325X/MI355X/MI455X.
- **Evidence:** recipe hardware counts above; AMD MI355X + SGLang/MoRI [TCO post](https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html). TPU Ironwood: only 2 vLLM recipes, no primary frontier-MoE tokens/s found (thin) [Google](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads).
- **Fit:** data only. TPU needs a torus/ICI topology the fabric model may not express → effort L, grade C for now.
- **Effort: S** (AMD xGMI numbers/topology), **L** (TPU).
- **Status: done for AMD.** MI325X (7 × 64 GB/s per direction, as MI300X) and MI355X (7 × 72 GB/s per direction, 504 aggregate) xGMI links replace the PCIe placeholders; MI455X gets its 72-GPU switched UALoE domain (1.8 TB/s per GPU per direction). TPU remains open. See `docs/simulation-model.md` §4 "Scale-up domains".

### G7. Prefix-cache hit ratio (scenario knob) — **A**, in scope

- **What:** cached prompt prefix skips prefill FLOPs; only new tokens are computed (attention still reads cached KV). Cached KV from a lower tier costs a read (PCIe/C2C).
- **Evidence:** DeepSeek 56.3% hit (342B/608B tokens/24 h), cached tokens priced 4× lower; Mooncake 34–55% depending on capacity, prefill GPU time −36…−64% [arXiv](https://arxiv.org/html/2407.00079); SGLang HiCache hit 40→80%, TTFT −56% [LMSYS](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/); llm-d prefix-aware routing 3× throughput [llm-d](https://github.com/llm-d/llm-d); vLLM recipes: 13 enable, 10 disable explicitly (agentic/hybrid enable; benchmarks disable).
- **Fit:** clean workload parameter (0 = today's behaviour). Radix/paging mechanics and routing are out of scope.
- **Effort: S.** `Workload` field, prefill lowering (new tokens = P·(1−h), past = P·h), UI input with help text.

### G8. Decode context parallelism (DCP) — **B** (high value for MLA at long context), in scope

- **What:** shards decode KV along the token dimension across DCP ranks; removes KV duplication when TP > KV heads (MLA has 1 latent head, so TP8 holds 8 copies). Adds a small Q all-gather + partial-output reduce (with log-sum-exp).
- **Evidence:** vLLM [docs](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/); Kimi-K2-Thinking recipe (KV capacity 8×, +43% tok/s); [vLLM DCP blog](https://vllm.ai/blog/2026-08-07-decode-context-parallelism) (TP plateau ~1.9k vs DCP ~6.1k tok/s/GPU on 8×B200 K2.6); Kimi-K3 GB200 lane DCP8. 4 explicit launches, 11 recipes flag support (all MLA).
- **Fit:** pure memory-footprint + communication model.
- **Effort: M.** `lower/attention`/MLA lowering, collectives, memory fit, Parallelism UI.
- **Status: done.** `ParallelConfig.dcp` shards the decode KV of MLA (non-sparse) and full-attention GQA layers over the tokens of d consecutive TP ranks (`ceil(ctx/d)` per rank), with a Q all-gather and one (partial output + fp32 LSE) reduce-scatter per sharded layer; d must divide tp and, for GQA, tp/kvHeads (vLLM bounds dcp to [1, tp/kvHeads]). Prefill is unchanged; sliding-window layers stay unsharded; sparse/CSA/relative-bias/K-V-conv blocks and block drafters are rejected; MTP works. Kimi K2 on 8×B200 at 128k: 10 → 87 sequences per replica. See `docs/simulation-model.md` §5 "Decode context parallelism (DCP)".

### G9. Prefill context parallelism / chunked pipeline prefill (very long context) — **B/C**

- **What:** ring-attention CP shards the prompt across nodes (Meta: Llama-3-405B 1M tokens in 77 s on 128 H100, 93% parallel efficiency) [arXiv](https://arxiv.org/html/2411.01783v2). Mooncake and SGLang GB300 prefer chunked PP across nodes for 128K [arXiv](https://arxiv.org/html/2407.00079), [LMSYS](https://www.lmsys.org/blog/2026-02-19-gb300-longctx/).
- **Evidence of use:** research/blogs; **0 vLLM recipes** (prefill CP "under active development" in vLLM docs).
- **Fit:** legitimate math. Grade B for long-context scenarios, kept low priority; chunked-PP prefill overlaps with existing PP model (microbatching a single long prompt) and could be cheaper.
- **Effort: M–L.**

### G10. KV-cache tiering/offload (host RAM, SSD, remote) — **B**, in scope (reuses offload machinery)

- **What:** cold KV spills to CPU/SSD/pool; fetching costs bytes ÷ tier bandwidth, but increases effective batch/context.
- **Evidence:** 22 vLLM recipes verified (Simple CPU, Mooncake store in 5, LMCache in 3); SGLang HiCache; llm-d hierarchical offload 13.9× throughput at 250 users [blog](https://llm-d.ai/blog/kvcache-wins-you-can-see); Kimi-K3 GB200 lane Mooncake store.
- **Fit:** deployment-scenario knob like existing offload. Absent today.
- **Effort: M.** `engine` offload (new group "kv"), memory fit, UI.

### G11. Dispatch/combine precision & all-to-all efficiency — **B**, in scope (S)

- **What:** all-to-all bytes per token depend on dispatch dtype (FP8/NVFP4) and combine dtype (BF16). Achieved bandwidth is below link peak; DeepEP normal vs low-latency vs `flashinfer_nvlink_one_sided`/MoRI are implementations of the same collective.
- **Evidence:** DeepSeek FP8 dispatch/BF16 combine; SGLang NVFP4 dispatch halves traffic [part 2](https://lmsys.org/blog/2025-09-25-gb200-part-2/); vLLM GB200 blog ~4× less all-to-all volume with NVFP4 dispatch; AMD MoRI FP4 dispatch + FP8 combine, 28,672 → 11,200 B/token [AMD](https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html). Backend choice itself: only 3–4 recipes set it; pplx 0.
- **Fit:** dtype = legitimate (bytes). Backend = engine implementation → model only as a per-link achieved-bandwidth efficiency factor (**borderline**); pplx/DeepEP choice out of scope.
- **Effort: S.** `lower/moe.ts` dtypes (today dispatch=`bpe(act)`, combine=`max(bpe(act),2)`), a dispatch-dtype option.
- **Status: done.** `ParallelConfig.dispatchFormat` / `combineFormat` (`bf16`, `fp8_e4m3`, `fp8_block128` = FP8 + fp32 scale per 128 as DeepEP, `mxfp8`, `nvfp4`, `mxfp4`; block scales counted through `bytesPerElement`; absent = today's `bpe(act)` / `max(bpe(act),2)`) and `allToAllEfficiency` η ∈ (0, 1] (divides every all-to-all's bandwidth term; absent/1 = today; no backend choice). UI: Hardware → Advanced → "Expert all-to-all" for MoE models with EP > TP (URL `dsp`, `cmb`, `a2e` in percent). AMD MoRI's 11,200 B/token is reproduced (NVFP4 4,032 + FP8 7,168 at h = 7168). See `docs/simulation-model.md` §5 "Dispatch and combine precision", §6 "All-to-all efficiency".

### G12. FP8/FP4 KV cache and low-precision GEMM — **A** (FP8 KV), **B** (FP4 KV), mostly modeled

- **Evidence:** explicit `--kv-cache-dtype fp8` in 44–47/76 recipes (the most common non-parallelism knob); FP4/NVFP4 KV in 0 recipes, but NVIDIA reports ~50% KV memory cut, <1% accuracy loss [NVIDIA](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/); SGLang FP8 attention/KV 1.8× attention, NVFP4 GEMMs up to 1.9× [part 2](https://lmsys.org/blog/2025-09-25-gb200-part-2/).
- **Status:** `QuantConfig.kvCache` and the dtypes already exist. Remaining gaps: quantization-scale overhead layouts for KV, FP8 collectives, online dequant cost, FP4/MXFP4 indexer cache is present (`indexerKeyCache`). **Verify the UI exposes the KV dtype prominently** (likely only a presentation task).
- **Fit:** legitimate. **Effort: S.**

### G13. Speculative-decoding refinements — **B**, in scope

- **What:** acceptance is batch-dependent in practice; PP>1 is rejected; trees and small-draft-LM/ngram methods are absent.
- **Evidence:** 49/76 recipes (41 opt-in); Qwen3.8-2.4T guide: MTP-1 +3.4% at concurrency 1 but −9% at 128, −23% at 256; SGLang assumed 70% acceptance. EAGLE3 is in 8 recipes.
- **Fit:** an extension of the existing legit knob; pp>1 support is feasible math. Draft trees borderline (engine/verification strategy). **Effort: M.**

### G14. Sparse (DSA/CSA/MSA), SWA, linear/hybrid attention — already modeled

- Evidence of use: sparse ≥13, linear/hybrid ≥16, SWA ≥14, MLA ≥14 of 76 recipes; DSA paper [arXiv](https://arxiv.org/html/2512.02556); Qwen3-Next/MiniMax-M1 linear attention (secondary sources).
- **Status:** specmaxx lowers all of these. Open items are small: attention sinks ignored, partial-RoPE approximated, prefill ≤2048 DSA falls back to dense in SGLang (engine detail), state transfer for hybrids in PD (comes with G1). No new gap item beyond keeping models current.

### G15. Multi-tier networks, multiple NICs, in-network reduction (NVLS/SHARP) — **B**, in scope but low priority

- **What:** rail-optimized fat-trees, 1–8 NICs/GPU, oversubscription, NVLS in-switch all-reduce. Current model: two levels, one NIC/GPU, scalar oversubscription.
- **Evidence:** recipes set `NCCL_NVLS_ENABLE=1` and `NCCL_MNNVL_ENABLE=1` for GB200 PD clusters ([pd_cluster strategy](https://github.com/vllm-project/recipes/blob/main/strategies/pd_cluster.yaml)); DeepSeek/SGLang/Mooncake use multi-NIC IB/RoCE (Mooncake 4×200 / 8×400 Gb).
- **Fit:** hardware; NCCL protocol selection/LL128 software latency is engine detail (C). **Effort: M.** `hw/types.ts`, `engine/collectives.ts`.

### G16. Multi-node PP over IB — **already modeled**; interleaved PP / bubbles — **C**

- PP exists with activation sends over node-crossing links. Real use ≈2–5 recipes (Kimi-K3 TP8×PP2). Interleaved/virtual stages not used in decode serving. Spec-dec with PP is G13.

### G17. Mixed prefill+decode batches, chunked prefill rate, continuous batching, async scheduling — **C**, out of scope

- Engine/scheduler behaviour (explicitly excluded by project principle). Chunked prefill appears as a `max-num-batched-tokens` tuning knob in 24 recipes and `--async-scheduling` in 9; neither is a hardware limit. The existing prefill-chunk memory-fit input is enough.

### G18. Runtime LoRA / multi-LoRA serving — **C**

- Evidence thin: 0 recipes with runtime adapters; only S-LoRA literature ([paper](https://par.nsf.gov/biblio/10552444/media/xml)) and llm-d mentions of cache-aware LoRA routing. A rank-r adapter adds ~r/h extra FLOPs/bytes per GEMM, small next to the base model. Don't implement.

### G19. Operations / scheduling: autoscaling, routing, early rejection, KV-aware routing policy, CUDA graphs, kernel fusion, DeepEP SM allocation, encoder DP — **C**, out of scope

- Engine/operations. Trivial post-processing ($/Mtok from GPU-hour price) could be a UI nicety but is not a modeling gap (SGLang quotes $0.21/M for K2).

## Summary table

| #   | Option                                                             | What it changes                                     | Grade                                | In scope?                         | Effort | Key sources                                                                                                                                                                                                                                                                                 |
| --- | ------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------ | --------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G1  | Prefill/decode disaggregation                                      | Two device pools, P:D ratio, KV transfer bytes/link | A                                    | Yes (scenario knob; queueing out) | L      | [DeepSeek](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), [LMSYS K2](https://lmsys.org/blog/2025-07-20-k2-large-scale-ep/), [Mooncake](https://arxiv.org/html/2407.00079), 14 vLLM recipes |
| G2  | DP-attention + wide EP layout (dense DP, DP LM head)               | No MLA KV replication, EP≫8, all-to-all             | A                                    | Yes                               | S–M    | [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [NVIDIA](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/), ≥10 recipes                                                                                     |
| G3  | EPLB + redundant experts                                           | Hot-rank factor on compute/comm, +weight bytes      | A (large EP)                         | Yes (factor + slots)              | S–M    | [TRT-LLM](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html), [vLLM blog](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep) (0 recipes)                                                                              |
| G4  | Two-batch overlap (partial overlap)                                | max vs sum of compute/comm, per collective          | B                                    | Yes (borderline min-batch)        | S–M    | [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [DBO docs](https://docs.vllm.ai/en/latest/design/dbo/)                                                                                                                                                                          |
| G5  | NVL72 / GB200 / GB300 presets, scale-up domain >8                  | NVLink domain size, Grace C2C                       | A                                    | Yes                               | S–M    | [NVIDIA](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/), [LMSYS GB200](https://lmsys.org/blog/2025-06-16-gb200-part-1/)                                                                                              |
| G6  | AMD MI325X/MI355X xGMI presets (TPU later)                         | Intra-node link data                                | A (AMD) / C (TPU)                    | Yes                               | S / L  | recipe hardware counts, [AMD](https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html)                                                                                                                                                                           |
| G7  | Prefix-cache hit ratio                                             | Prefill FLOPs × (1−h), cached KV read               | A                                    | Yes                               | S      | [DeepSeek](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), [Mooncake](https://arxiv.org/html/2407.00079), [HiCache](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)                  |
| G8  | Decode context parallelism (DCP)                                   | KV sharded over tokens, small LSE reduce            | B                                    | Yes                               | M      | [vLLM docs](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), [blog](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)                                                                                                                                        |
| G9  | Prefill CP / chunked PP prefill                                    | Sequence-sharded prefill, ring KV traffic           | B/C                                  | Yes, low priority                 | M–L    | [Meta CP](https://arxiv.org/html/2411.01783v2) (0 recipes)                                                                                                                                                                                                                                  |
| G10 | KV-cache tiering (host/SSD/pool)                                   | KV capacity↑, read bytes/step from tier             | B                                    | Yes                               | M      | 22 recipes, [llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see)                                                                                                                                                                                                                         |
| G11 | Dispatch/combine dtype, all-to-all efficiency                      | All-to-all bytes, achieved BW                       | B                                    | dtype yes; backend no             | S      | [LMSYS pt2](https://lmsys.org/blog/2025-09-25-gb200-part-2/), [AMD](https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html)                                                                                                                                     |
| G12 | FP8/FP4 KV, scale overhead, FP8 collectives                        | KV bytes, GEMM peak                                 | A (FP8 KV, mostly done) / B (FP4 KV) | Yes                               | S      | 44–47 recipes, [NVIDIA](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)                                                                                                                                                 |
| G13 | Spec-dec refinements (batch-dependent acceptance, PP, trees)       | Verify shapes, acceptance                           | B                                    | Yes (trees borderline)            | M      | 49 recipes, Qwen3.8 guide                                                                                                                                                                                                                                                                   |
| G14 | Sparse/SWA/linear attention                                        | —                                                   | already modeled                      | —                                 | —      | ≥13/14/16 recipes                                                                                                                                                                                                                                                                           |
| G15 | Multi-tier fabrics, multi-NIC, NVLS/SHARP                          | Collective time on real topologies                  | B                                    | Yes (NCCL protocol no)            | M      | [pd_cluster](https://github.com/vllm-project/recipes/blob/main/strategies/pd_cluster.yaml)                                                                                                                                                                                                  |
| G16 | Interleaved PP / bubbles                                           | PP bubble                                           | C                                    | Borderline                        | M      | ≈2–5 recipes use PP                                                                                                                                                                                                                                                                         |
| G17 | Mixed batching, chunked-prefill rate, async scheduling             | Scheduler behaviour                                 | C                                    | **No** (engine)                   | —      | 24/9 recipes tune it                                                                                                                                                                                                                                                                        |
| G18 | Runtime LoRA                                                       | ~r/h extra FLOPs                                    | C                                    | Not worthwhile                    | —      | thin                                                                                                                                                                                                                                                                                        |
| G19 | Autoscaling, routing, kernels, CUDA graphs, DeepEP SMs, encoder DP | Ops/engine                                          | C                                    | **No**                            | —      | —                                                                                                                                                                                                                                                                                           |

## Proposed scope

**Main UI (A):**

1. **G5 NVL72/GB200/GB300 presets + scale-up domain beyond 8 and G6 AMD xGMI presets** — enablers; unblock wide EP at realistic sizes (S–M, S).
2. **G2 DP-attention + wide-EP layout** (label existing mode B, MLA KV not replicated under DP-attention, dense-FFN/LM-head DP) — depends on G5 for sizes >8 (S–M).
3. **G3 EPLB/imbalance + redundant experts** — extend existing ρ; expose in UI; depends on G2 (S–M).
4. **G7 Prefix-cache hit ratio** — independent, small (S).
5. **G1 PD disaggregation** — biggest item; depends on G2/G5 for realistic pool shapes, benefits from G7 (L). Split into: (a) separate P and D parallelism + P:D ratio + aggregate throughput, (b) KV transfer cost.
6. **G12 FP8/FP4 KV exposure + scale overhead** — check UI first (S).

**Advanced (B), suggested order:** G11 dispatch dtype (S) → G4 overlap model (S–M, with G2) → G8 DCP (M, for MLA at long context) → G10 KV tiering (M, reuse offload) → G13 spec-dec refinements (M) → G15 network tiers (M) → G9 prefill CP / chunked PP (M–L, last).

**Not implemented (C / out of scope):** G16 interleaved PP, G17 mixed batching / chunked-prefill rate / async scheduling, G18 LoRA, G19 ops and kernel-level items, TPU presets (revisit when primary serving numbers appear), all2all backend selection (DeepEP vs pplx vs flashinfer; only as an efficiency factor under G11).

**Dependencies:** G5 → G2 → G3 → G4; G1 depends on G2, G5 and uses G7/G10; G8 independent but pairs with G2 for MLA; G13's PP support is independent.

## Open questions

- Should EPLB default to 1.0 (ideal) or keep today's 1.2? Evidence: 1.56 without EPLB, ~1.1 with 32 redundant (TRT-LLM, one dataset).
- Is PD disaggregation a throughput feature or a latency/SLO feature? vLLM docs say latency control; DeepSeek/SGLang show large throughput gains because it enables per-phase EP/DeepEP modes. The model should report both pool rates and the balanced aggregate.
- Whether to model GB200/GB300 as NVL4 trays (matches recipes) or NVL72 (matches wide-EP blogs): provide both topologies.
