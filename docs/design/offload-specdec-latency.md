# Design: weight offloading, speculative decoding, interconnect-latency audit

Status: implemented (revision 3: the design as built). Revision 2 (after review) was implemented by
[`#specmaxx/implement-expert-offloading-speculative`](`ref://task:01a1bb13-6549-7100-b13a-254ffb2c32ec`); the sections
below are updated to the code, and §10 lists where and why the implementation deviates from revision 2. The
latency-audit follow-up now implements G1–G4, G7, G8 (§4.3); G6 remains out of scope. Companion of
`docs/simulation-model.md` (cited as "model doc §N"; offload is its §12, speculative decoding its §13). Line
numbers in historical audit §4.1 are relative to the repo root at `d4dba96` (the design's base).
The goldens in §8.3 are the deliberately regenerated post-audit values.

## 0. Principle and contract

Offloading and speculative decoding are **opt-in deployment-scenario knobs, off by default**.

- **Strict identity.** With `SimInput.deployment` absent, `{}`, or every knob Off, `simulate()` returns results
  **bit-identical to today for every model**, including the n-gram-table models (PLE, Engram): lookup tables
  default to `'device'` placement in the simulator (§2.10). The identity test (§8.2, ID-1) pins **numeric goldens
  computed from today's code** (§8.3), including TP > 1, PP > 1, EP > 1 and hierarchical cases, and compares the
  new engine against them.
- **Why identity holds (mechanism).** Every new input is optional (`SimInput.deployment?`,
  `SimOptions.offloadOverlap`, `HardwareStack.host?`, `ModelSpec.draft?`). Absent ⇒ no `TransferOp` is emitted, no
  `draft.*` layer is appended, the decode context keeps `qLen = 1`, the stage formula's new term is 0 and both
  branches reduce to today's expression, the memory fit adds 0 B, and `hardware.host` is never read.
- **The interconnect corrections were separate from the original deployment feature.** The completed audit
  follow-up applies G1–G4, G7, G8 even with every knob off; §4.3 records each measured delta and §8.3 the updated
  baseline. Deployment absent vs empty/off still remains bit-identical. G6 stays out of scope.
- **A knob on ⇒ still the hardware/math upper bound given the stated parameters** (hit ratio h, acceptance α,
  overlap assumption, queue depth). The engine adds only bytes crossing a link, requests paying a latency, and the
  GEMM/attention/state work the draft and verify passes physically require. No scheduler, paged cache, predictor
  accuracy or kernel-launch overhead is modelled.
- DP and EP leave the UI (fixed at 1, [`#specmaxx/hide-dp-and-ep-from-the-ui-fixed-at-1`](`ref://task:01a1bb22-ba75-742f-999b-30d88ffa17ea`)).
  No UI knob below depends on them. **EP = 1 is the designed case**; EP > 1 composes at engine level (§2.9, §3.7).
- **Spec-dec with PP > 1** was rejected in the first implementation and is now implemented (§3.6): the features
  ride on the stage sends and the last stage keeps an embedding copy; only pp > 1 with the token embedding
  off-device is rejected.

Notation: `T = seqs·qLen` tokens per device step (prefill `b_mb·P`, decode `b_mb`); `N` = decode context
(`decodeContextLen`); `B_r` = sequences per replica; `bpe(f)` = bytes per element (`ir/dtype.ts`). Times in
seconds, bytes in bytes, bandwidths in B/s (unidirectional unless stated), rates in tok/s "@ context". All transfer
quantities are **expected values per step** (real numbers; no integer rounding of requests or bytes; §2.3).

---

## 1. What exists today (verified in the code)

| Area                 | Code                                                                                                                               | Fact relied on below                                                                                                                                                                   |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-op time          | `engine/opTime.ts:58-62, 133-166`                                                                                                  | `t_op = max(F/(peak·eff), bytes/memBW)`; collectives `α + β` with `commLatencySec` / `commBandwidthSec`                                                                                |
| Stage time           | `engine/pipeline.ts:26-59`                                                                                                         | `'ideal'`: `t_s = lat_s + max(compute_s, bw_s)`; `'none'`: `compute_s + lat_s + bw_s + p2p_s`; exhaustive `switch` + `never`                                                           |
| Phase                | `engine/pipeline.ts:109-153`                                                                                                       | decode `cycle = max(m·b, Σt_s + Σp2p)`, `tok/s = dp·B_r/cycle`                                                                                                                         |
| MoE routing          | `lower/moe.ts:91-99, 161-166`                                                                                                      | balanced worst case: `a = min(E_loc, T·k)` experts read from device memory; no expected-unique formula (model doc §9 lists `E·(1−(1−k/E)^T)` as future work)                           |
| MoE residency        | `lower/moe.ts:215-231`                                                                                                             | all local experts resident                                                                                                                                                             |
| Recurrent step       | `ir/metrics.ts:388-400`, `lower/linearAttention.ts:141`                                                                            | decode `'step'` mode reads and writes the state **once per step**, whatever `qLen`                                                                                                     |
| n-gram tables        | `lower/ngram.ts:28-67, 74-83`                                                                                                      | gather `T·g` rows of `d` (an `EmbeddingOp`); the whole table is counted VRAM-resident (`ceil(rows/tp)·d·bpe`)                                                                          |
| Memory fit           | `engine/memory.ts:56-73, 112-142`                                                                                                  | one tier (device); the embedding shard is counted on stage 0 only, the LM head on the last stage                                                                                       |
| Hardware             | `hw/types.ts:121-218`                                                                                                              | `DeviceSpec` has no host link, host RAM or SSD; `singleDevice()` has no links; the only host number is `LinkSpec.hostBounce.hostMemBandwidthBytesPerSec` (`hw/presets/links.ts:49-54`) |
| Overlap knob pattern | `engine/types.ts:35-48`, `simulate.ts:27`, `pipeline.ts:38-53`, `ui/state/url.ts:38-42`, `ui/components/AssumptionsPanel.tsx:9-43` | the pattern `offloadOverlap` copies                                                                                                                                                    |
| MTP in the HF loader | `models/hf/families/{glmMoeDsa,glm5Next,qwen,inkling,deepseekV41,mistralLarge4}.ts`, `mlaCommon.ts:216-224`                        | the MTP `LayerSpec` is built, counted (`mtpParamsEstimate`) and discarded                                                                                                              |
| α/β split            | `engine/types.ts:81-84, 121-126`                                                                                                   | per op and per stage, not per phase, not in the UI (`ui/state/summary.ts:181-189`)                                                                                                     |

---

## 2. Offloading

### 2.1 Placement model

A **placement** says which weight groups are not in device memory and where they live. One mechanism, three group
kinds; each adds one `TransferOp` (§2.5) in front of the block that consumes the group, **only when it has bytes to
move**.

| Group kind                          | Covers                                                                 | Selector                                                                                | Resident on device                                                                                                                                     | Streamed per step                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `routed_experts`                    | `MoeBlock` routed experts of the selected layers                       | all MoE layers, or a layer range `[from, to)` (llama.cpp `--n-cpu-moe`, ik_llama `-ot`) | `C = f·E` experts per offloaded layer (pinned or LRU cache; the sim does not distinguish, the hit-ratio model does), router, shared experts, attention | the missed experts' weights (§2.4–2.5)                                                                        |
| ~~`dense_layers`~~ (removed, §2.11) | every weight of whole decoder layers `[from, to)` (dense or MoE model) | layer range                                                                             | all other layers                                                                                                                                       | the full weight bytes of each offloaded layer, every step (no hit ratio: a dense layer is read once per step) |
| `lookup_tables`                     | `NgramEmbeddingBlock` tables (Qwen PLE, DeepSeek Engram)               | all tables, or rules by layer range (§2.11)                                             | projections, gate vectors, conv taps                                                                                                                   | the `T·g` gathered rows (§2.10)                                                                               |

Decisions:

- **Dense whole-layer offload was in scope, and was later removed (§2.11)**: the same `TransferOp` with `bytes = W_layer`, `R = 1`; one `case` in
  `lowerBlock`; it gives the common consumer case (a 70B at 4.5 bpw on a 32 GB card, llama.cpp `-ngl N`) an answer.
- **Dense/attention weights and shared experts stay resident** (before the removal: unless their layer was in a `dense_layers` range).
- **Execution mode** `'transfer'` (misses stream to the device, which computes) is the first implementation;
  `'cpu_compute'` is a designed hook (§2.8).

### 2.2 Storage tiers and the host (HW spec data)

New plain-data specs in `src/sim/hw/types.ts` (JSDoc in §7.2):

- `HostMemorySpec` (`kind: 'host_ram'`): `capacityBytes`, `bandwidthBytesPerSec` (theoretical `MT/s·8 B·channels`),
  `latencySec` (unloaded load-to-use), `channels`, `readGranularityBytes` (64), `maxOutstanding` (memory-level
  parallelism the host sustains, estimated 64).
- `NvmeSpec` (`kind: 'nvme'`): `capacityBytes`, `seqReadBandwidthBytesPerSec`, `randomReadLatencySec` (QD1 4 KiB),
  `maxOutstanding` (requests the drive accepts in flight; the datasheet's QD·threads condition, or an explicit
  estimate), `randomRead?: { iops, queueDepth, threads }` (the datasheet's max-IOPS condition; **optional**: when
  absent the IOPS term of §2.3 is dropped and the request rate is bounded by Little's law alone,
  `maxOutstanding / randomReadLatencySec`, with a warning "no random-read IOPS for <id>; request rate from latency
  × queue depth"), `readGranularityBytes` (4096), `linkBandwidthBytesPerSec?` (the M.2 slot ceiling).
- `HostSpec` on `HardwareStack.host?: { link?, ram?, ssd? }` — **one host per node**, shared by every device of the
  node (§2.3 `n_stream`). `link` is the device's PCIe connection (per device, per direction; reuses `LinkSpec`).
  `DeviceSpec.hostInterface?: { generation, lanes }` is added as cited preset data (RTX 5090: Gen5 x16, whitepaper
  Table 3; RTX 4090/3090: Gen4 x16); when `host.link` is absent and something is off-device the engine builds it with
  the `pcieLink` bandwidth formula (`links.ts: hostPcieLink`), `latencySec = 1e-6`, and warns. A device without
  `hostInterface` (SXM / OAM modules, SoCs) and without `host.link` is a **validation error** ("deployment.offload: no
  host link known for <id>; set hardware.host.link"). Link modifiers are not applied to the host link. `ram` absent
  with an off-device group on a dedicated-memory device ⇒ `DDR5_6000_2CH` with a warning. On **unified memory** host
  RAM _is_ the device pool: `'host_ram'` placement = `'device'` (no transfer), `link` ignored, only `ssd` is a tier,
  and the default SSD is the device's own drive (Apple M5 → `apple-ssd-m5-pro-max`, other Apple →
  `apple-ssd-m3-m4-max-1tb`, DGX Spark → `dgx-spark-ssd-4tb`, Ryzen AI Max → `nvme-gen4-x4-generic`); dedicated-memory
  devices default to `nvme-990-pro-2tb`.

**Presets** (`src/sim/hw/presets/storage.ts`; every number in `sources[].fields` or `estimated[]`). Datasheet
ceilings by default (Q3); the final column is the opt-in 1 MiB QD1 / QD8 large-read fraction (one thread).

| id                                                     | capacity             | bandwidth                                     | latency                                                                                    | random read (iops @ QD×T)                                                                                            | gran. | source                                                                                                                                                                                                                                      | conf.                       | sustained fraction |
| ------------------------------------------------------ | -------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ------------------ |
| `ddr5-5600-2ch`                                        | 96e9 (2×48 GB, est.) | 89.6e9                                        | 70e-9 (est.: the review measured 70 ns at 6000 MT/s)                                       | —                                                                                                                    | 64    | AMD Ryzen 9 9950X spec (2 ch, DDR5-5600) https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-9-9950x.html ; latency Chips and Cheese 9950X https://chipsandcheese.com/p/amds-ryzen-9950x-zen-5-on-desktop       | high / est.                 | —                  |
| `ddr5-6000-2ch` (**default host RAM**)                 | 96e9 (est.)          | 96.0e9                                        | 70e-9                                                                                      | —                                                                                                                    | 64    | same (EXPO 6000; the review quotes "theoretical 96 GB/s"); equals `links.ts:51`                                                                                                                                                             | high                        | —                  |
| `ddr5-6400-2ch`                                        | 96e9 (est.)          | 102.4e9                                       | 100e-9 (est.)                                                                              | —                                                                                                                    | 64    | derived `6400e6·8·2`; latency Arrow Lake review https://chipsandcheese.com/p/examining-intels-arrow-lake-at-the                                                                                                                             | medium                      | —                  |
| `ddr5-6400-8ch`                                        | 256e9 (est.)         | 409.6e9                                       | 100e-9 (est.)                                                                              | —                                                                                                                    | 64    | AMD Threadripper PRO 9995WX (8 ch, DDR5-6400) https://www.amd.com/en/products/processors/workstations/ryzen-threadripper/9000-wx-series/amd-ryzen-threadripper-pro-9995wx.html                                                              | high / est.                 | —                  |
| `ddr5-4800-12ch`                                       | 384e9 (est.)         | 460.8e9                                       | 110e-9 (est.)                                                                              | —                                                                                                                    | 64    | AMD EPYC 9654 "Per Socket Mem BW 460.8 GB/s" https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9654.html                                                                                  | high                        | —                  |
| `ddr5-6000-12ch`                                       | 768e9 (est.)         | 576e9 (est.: AMD rates the part at DDR5-6400) | 110e-9 (est.)                                                                              | —                                                                                                                    | 64    | derived from EPYC 9575F (12 ch; 614 GB/s at 6400) https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9575f.html                                                                                                    | high / est.                 | —                  |
| `nvme-990-pro-2tb` (**default SSD**, dedicated memory) | 2e12                 | 7.45e9                                        | **45.45e-6** (= 1/22,000 QD1 IOPS; the worked examples use a 45e-6 variant, §8.6)          | 1.4e6 @ 32×16                                                                                                        | 4096  | Samsung 990 PRO datasheet Rev 1.0 https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_NVMe_SSD_990_PRO_Datasheet_Rev.1.0.pdf                                                                                            | high                        | .646               |
| `nvme-sn850x-2tb`                                      | 2e12                 | 7.3e9                                         | 50e-6 (est.)                                                                               | 1.2e6 @ (unstated; 32×16 assumed, in `estimated[]`)                                                                  | 4096  | Sandisk SN850X datasheet https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/product/internal-drives/wd-black-ssd/data-sheet-wd-black-sn850x-nvme-ssd.pdf                                                   | high / low                  | .747               |
| `nvme-9100-pro-4tb`                                    | 4e12                 | 14.8e9                                        | 45e-6 (est.)                                                                               | 2.2e6 @ 256×32                                                                                                       | 4096  | Samsung 9100 PRO datasheet Rev 2.0 https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_NVMe_SSD_9100_PRO_Datasheet_Rev.2.0.pdf                                                                                          | high / low                  | .720               |
| `nvme-t705-2tb`                                        | 2e12                 | 14.5e9                                        | 45e-6 (est.)                                                                               | 1.55e6 @ 512 (recorded as QD 512 × 1 thread: the vendor gives no split)                                              | 4096  | Crucial T705 page footnote https://www.crucial.com/ssd/t705/ct2000t705ssd3 ; CDRLabs 1.41M @16×32 https://www.cdrlabs.com/reviews/crucial-t705-2tb-pcie-gen5-nvme-solid-state-drive/performance-anvil-s-storage-utilities-and-iometer.html  | medium                      | .646               |
| `apple-ssd-m3-m4-max-1tb`                              | 1e12                 | 6.4e9                                         | 68e-6 (from QD1 ≈ 60 MB/s)                                                                 | 2.5e5 @ 64 (derived)                                                                                                 | 4096  | AmorphousDiskMark crowd medians (115 runs; `kind 'community'`) https://www.reddit.com/r/mac/comments/1rpsc1n/mac_ssd_speeds_overview_updated_with_m5_promax/ ; Apple ≥ 6 GiB/s linear (arXiv 2312.11514 §2)                                 | medium                      | .51 community      |
| `apple-ssd-m5-pro-max`                                 | 2e12                 | 13.1e9                                        | 65e-6 (derived)                                                                            | 2.7e5 @ 64 (derived)                                                                                                 | 4096  | same dataset, n = 1–3                                                                                                                                                                                                                       | low                         | .84 community      |
| `dgx-spark-ssd-4tb`                                    | 4e12                 | 12.2e9 (11.4 GiB/s GDSIO peak)                | 50e-6 (**est.**: Gen5 2242 class, between the 990 Pro's 45 µs and the Apple drives' 65 µs) | 1.5e6 @ 32×16 (**est.**: between the 990 PRO 1.4e6 and the 9100 PRO 2.2e6; StorageReview published no random figure) | 4096  | StorageReview https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops ; NVIDIA https://docs.nvidia.com/dgx/dgx-spark/hardware.html ; both random fields in `estimated[]` | medium (seq) / low (random) | .65 est.           |
| `nvme-gen4-x4-generic` (Strix Halo boxes)              | 2e12                 | 7.0e9, `linkBandwidth` 7.20e9 (payload)       | 50e-6 (est.)                                                                               | 1.0e6 @ 512 (est.)                                                                                                   | 4096  | Framework Desktop: 2× PCIe 4.0 x4 M.2 https://frame.work/desktop                                                                                                                                                                            | medium                      | .666 est.          |

Removed since revision 1: the quad-channel DDR5-6400 preset (no vendor citation for the channel count).
Host-link presets: `hostPcieLink(generation, lanes)` (`HOST_PCIE_LINKS`, Gen3–5 × x4/x8/x16; Gen4 x16 31.5 GB/s,
Gen5 x16 63.0 GB/s per direction; α 1 µs estimated); `DEFAULT_HOST_RAM` / `DEFAULT_SSD` alias the defaults. Achieved
copy-engine rates (≈ 25 GB/s on Gen4 x16, https://github.com/NVIDIA/nvbandwidth/issues/53 ; the ≈ 50–53 GB/s Gen5
figure is unconfirmed) are not presets; the `linkBandwidth` modifier can derate the device-to-device links.

### 2.3 Transfer path: one read model for every group

Sustained fractions are sourced individually in `storage.ts`: [Tom's 990 PRO comparison](https://www.tomshardware.com/reviews/samsung-990-pro-ssd-review/2)
(990 PRO 4618/7153, SN850X 5214/6982 MB/s), [9100 PRO 4 TB](https://www.tomshardware.com/pc-components/ssds/samsung-9100-pro-ssd-review/3)
(10287/14295), [CDRLabs T705](https://www.cdrlabs.com/reviews/crucial-t705-2tb-pcie-gen5-nvme-solid-state-drive/print.html)
(9453.93/14631.35). Generic Gen4 is the median of 19 comparison drives.
[Community AmorphousDiskMark sheet](https://docs.google.com/spreadsheets/d/1Mu8n3438TM2orWivg_Y8YGlbd8VLcy3c-xBZulH_1zM/edit?usp=sharing):
M3/M4 Max 1 TB mean (n=2), M5 Pro/Max median (n=7, 1–8 TB). Spark borrows Gen5 peers: its only
[GDSIO measurement](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops)
1-thread / 16-thread = 2.2/11.4 includes cuFile overhead, not just drive throughput.

`OffloadConfig.ssdReadRate?: 'datasheet' | 'sustained'` defaults to datasheet, preserving results bit for bit.
In sustained mode `B_i = min(f_sus·B_seq, B_slot)` for individual NVMe reads ≥ `blockBytes`;
otherwise use the original `B_tier`. Missing data warns and falls back to datasheet. Tiny rows and host RAM are
unchanged. Measurement QD does not replace deployment QD. Apply by size, not access pattern: large random reads
can saturate the drive too ([LLM in a flash §2.2](https://arxiv.org/abs/2312.11514)).
The 1 MiB QD1 scenario is pessimistic for 10–20 MB experts: CDRLabs' 2 MB QD3 measurements reach 91–95%.
It is not a tighter proven bound; thermal throttling and engine overhead are not modeled.

An off-device read is a set of **physical read classes** `{count_i, bytes_i}` (e.g. one 256 B value read and one
8 B scale read per row) from a tier, carrying `usefulBytes` of payload, then (dedicated memory) one copy over the
host link. **One rule everywhere** (`engine/transfer.ts: transferTime`):

```
R          = Σ_i count_i                                   physical read requests (pay IOPS and Little's law)
bytes_tier = Σ_i count_i · ceil(bytes_i / gran_tier) · gran_tier   granularity rounds every physical read up (a 320 B row costs a 4 KiB page)
bytes_link = usefulBytes                                   only useful payload crosses PCIe (264 B per Engram row whatever the read layout)
α_path     = α_tier + α_link                               latencies add: the first byte traverses every hop   (α_link = 0 on unified memory)
service    = max( (R − 1) · α_tier / QD ,                  Little's law: at most QD requests outstanding, each α_tier long
                  R / (IOPS_tier / n_stream) ,             the tier's request rate, shared by concurrent consumers (NVMe with `randomRead`; term absent otherwise)
                  Σ_i count_i·ceil(bytes_i/gran_tier)·gran_tier / (B_i/n_stream),   tier bandwidth by read class
                  bytes_link / B_link ,                    this device's link (not shared)
                  [ bytes_tier / (B_ram / (2 · n_host)) ]     NVMe → host → device only: host RAM written once, read once (n_host: §2.11)
                )
t_path     = α_path + service
```

- **Pipelining rule (chosen, applied everywhere):** hops are store-and-forward at request/chunk granularity but
  pipelined across requests and chunks (NVMe DMA and PCIe DMA both move 4 KiB–MB chunks; a chunk is forwarded while
  the next is read), so bandwidth/IOPS terms take the **max** (the slowest hop bounds throughput) and latencies
  **add**. The alternative (sum of hop times) would be right only for a single unchunked request, and no group has
  one: expert reads are many chunks, row gathers many requests. Revision 1 applied the sum to tables and the max to
  experts; W3/W4 are regenerated under this rule (§8.5).
- **Queue depth** `QD = min(OffloadConfig.queueDepth ?? tier.maxOutstanding, tier.maxOutstanding)`; `QD = 1`
  serializes: `t = α_tier + (R−1)·α_tier + … = R·α_tier` for latency-bound reads. Test U-11.
- **Sharing: `n_stream(tier)` = concurrent offload consumers of that tier on this device's node**, never idle or
  non-offloading ranks. Per tier, a rank is a consumer in a step if its stage holds a group placed on the tier. Which
  stages run concurrently: under `offloadOverlap 'ideal'` every offloading stage can have a prefetch in flight, so
  **all offloading stages** count; under `'none'` a stage only streams while a microbatch occupies it, and `m =
  min(pp, B_r)` microbatches circulate per replica, so **at most `m` offloading stages** stream at once. Among the
  offloading stages the engine takes the **optimistic schedule**: the `min(m, |S_tier|)` stages with the fewest
  consumers on this node (an explicit upper-bound choice; a pessimistic schedule would take the most).

  ```
  consumers(tier, s) = ranks of stage s on this node (rank = tpIdx + tp·(dpIdx + dp·ppIdx), node = floor(rank / devicesPerNode); model doc §5 "Rank layout")
                       if stage s holds a group on `tier`, else 0
  S_tier             = { s : consumers(tier, s) > 0 }                                      (offloading stages on this node)
  slots              = offloadOverlap === 'ideal' ? |S_tier| : min(m, |S_tier|)              (concurrently streaming stages)
  n_stream(tier)     = Σ of the `slots` smallest consumers(tier, s) over S_tier, at least consumers(tier, stage of this op)
  ```

  Each consumer gets `1/n_stream` of the tier's bandwidth and IOPS (and of the host-staging term); the host link is
  per device. With TP each device streams its `1/etp` shard, so the node total is the same bytes. Examples (one node,
  `tp = dp = 1` unless said): tp 4, all MoE layers offloaded ⇒ 4; `pp = 4, b = 1, 'none'`, group on stage 0 only ⇒
  **1**; the same with `'ideal'` ⇒ 1 (stage 0 is the only offloading stage); `pp = 4, b = 1, 'ideal'`, every stage
  offloaded ⇒ 4; `pp = 4, b = 1, 'none'`, every stage ⇒ 1 (serial stages); **`pp = 4, b = 2, 'none'`** (`m = 2`), every
  stage ⇒ **2**, stages {0, 1} only ⇒ 2, stage 0 only ⇒ 1; `pp = 4, b = 4, 'none'` ⇒ 4. Tests U-12, U-17.
  **Across nodes** (as built): a stage spanning several nodes takes the largest `n_stream` of its nodes; the tier
  bytes of §2.7 report the fullest node; DP replicas on one node each count their own copy (no shared host copy).

- **Expected values.** `R = M` (expected misses, a real number) and `bytes = M·s`; no `ceil`. Rounding the request
  count would bias the mean over steps; the memory staging buffer (§2.7) uses `ceil(M)` because capacity is not an
  average. `M = 0 ⇒ no TransferOp` (no α, no service, no staging).
- Documented approximation: the DMA write of streamed bytes into device memory is not charged to `memBW`
  (≤ `B_link/memBW` ≈ 3.5% on an RTX 5090; the same convention as collectives, `ir/ops.ts:308-309`).
- Paths: host RAM → device (`α_tier = lat_ram`, no host term); NVMe → host → device (`α_tier = lat_nvme`, host
  term present); NVMe → unified memory (`α_link = 0`, `B_link = ∞`, `B_tier = min(B_nvme, linkBandwidth_nvme)`).

### 2.4 Expert hit ratio and misses per layer

**Unique experts.** Under independent uniform top-k routing (the engine's balanced-routing assumption):

```
U(T) = E · (1 − (1 − k/E)^T)        expected distinct experts touched by T tokens in one layer   (MoESD, arXiv 2505.19645 Eq. 7–8)
```

E 128, k 8: `U(1) = 8`, `U(2) = 15.5`, `U(4) = 29.1230`, `U(8) = 51.6199`, `U(16) = 82.4225`, `U(32) = 111.771`,
`U(64) = 125.942`, `U(8192) = 128`. `U ≤ min(E, T·k) = a` always, so misses never exceed the experts the grouped
GEMM reads; the GEMM keeps today's `a` (device-memory bytes unchanged ⇒ identity), only the transfer uses `U`.

**Hit ratio h — definition.** `h` is the probability, at batch 1, that a routed expert a token needs is already in
device memory (pinned or cached), **per unique expert per layer per step** (= per activation at batch 1, which is what
every cited measurement reports: decode, batch 1, LRU, per layer). With `C = f·E` resident experts per layer and `U`
needed experts, the hits of a step are bounded physically: at most `min(U, C)` (no more than the needed or the
resident), at least `max(0, U + C − E)` (pigeonhole: when more than `E − C` are needed, some must be resident).
The model applies `h` inside those bounds:

```
C     = f · E
hits  = clamp( U · h ,  lo = max(0, U + C − E) ,  hi = min(U, C) )
M     = U − hits                                                 expected missed (streamed) experts per layer per step
```

Consequences (all enforced, tests U-2/U-3): `f = 0 ⇒ C = 0 ⇒ hits = 0, M = U`; `f = 1 ⇒ lo = U ⇒ M = 0` (no
transfer op, whatever h); a full sweep `U = E ⇒ lo = hi = C ⇒ hits = C exactly` (prefill with any locality model:
the resident experts are the hits, the `E − C` others are streamed once); custom `h < f` at small U: `hits = U·h`
(a pessimistic cache is allowed); custom `h > 1 − (E − C)/U`… is capped at `hi`. Endpoints of `h(f)` itself: the
preset models satisfy `h(0) = 0`, `h(1) = 1`; a custom `h > 0` with `f = 0` is a validation error ("no experts are
resident, h must be 0"), and with `f = 1` `h` is ignored (M = 0).

**Hit-ratio models** (`hitRatio`, §7.3):

| model                   | h(f)              | when                                                                             | confidence |
| ----------------------- | ----------------- | -------------------------------------------------------------------------------- | ---------- |
| `uniform` (**default**) | `h = f`           | exact for a well-balanced router on random inputs; the honest theoretical choice | exact      |
| `skewed_classic`        | `h = f^0.55`      | observed LRU locality of 2023–24 MoEs with E ≤ 64                                | medium     |
| `skewed_fine_grained`   | `h = f^0.2`       | 2026 reports on E = 128–512 models                                               | low        |
| `custom`                | user `h ∈ [0, 1]` |                                                                                  | —          |

Data behind the fits in §5.2. Below `C < k` an LRU thrashes (h ≈ 0); the engine warns when `0 < C < k` and a
skewed model is chosen.

### 2.5 Cost per step

`TransferOp` (§7.1) is emitted by lowering **before** the consuming block, with concrete numbers, **iff `bytes > 0`**:

| group                        | physical reads `{count, bytes}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `usefulBytes` (crosses the link)                                                                        | per step       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -------------- |
| routed experts               | `[{ M, W_expert }]` (`M` from §2.4 with the phase's `T`; `W_expert = (gated ? 3 : 2)·h·fe·bpe(w_routed)/etp`, `gemmWeightBytes` incl. 2:4 metadata)                                                                                                                                                                                                                                                                                                                                                                                                                               | `M·W_expert`                                                                                            | expected value |
| dense layer (removed, §2.11) | `[{ 1, W_layer }]` (`blockResidentWeightBytes` of the layer's blocks; a layer in the range streams whole: no expert or table transfer on top)                                                                                                                                                                                                                                                                                                                                                                                                                                     | `W_layer`                                                                                               | every step     |
| lookup table                 | per §2.10's scale layout, with **`valueBytes = d·elementBits/8`** (Engram 256 B, PLE 320 B) and **`scaleBytes = d·blockScaleBitsPerElem/8`** (Engram `256·0.25/8 = 8 B`, PLE 0); the logical row is `rowBytes = d·bytesPerElement(format) = valueBytes + scaleBytes` (Engram 264 B, counted **once**); rows per device `R = T·g/tp` (the table is row-parallel; `T·g` at tp 1): `interleaved` `[{ R, rowBytes }]`; `separate` `[{ R, valueBytes }, { R, scaleBytes }]`; `pinned_host` `[{ R, valueBytes }]` on the table's tier (the scale read comes from RAM and is not costed) | `R·rowBytes` (Engram 6,336 B per 24-row step, PLE 5,120 B per 16-row step at tp 1, whatever the layout) | §2.10          |

Op time (`opTime.ts`, new `case 'transfer'`): `t_path` of §2.3; result fields `transferLatencySec = α_path`,
`transferBandwidthSec = service`, `sec = sum`, `bound: 'transfer'`, FLOP 0, HBM bytes 0.

**Decode vs prefill.** Only `T` differs (`b_mb` vs `b_mb·P`). For prefill `U ≈ E` from T ≥ ~72 (E 128, k 8) or
~145 (E 256, k 8): every non-resident expert is streamed **once per prefill step** of a microbatch; the
transfer-bound prefill ceiling is `≈ P / (Σ_layers (E − C)·W_expert / B_min)` (W1).

**Batch.** `U(B)` grows sub-linearly (8 → 29.1 → 51.6 → 82.4 → 111.8 at B 1/4/8/16/32 for E 128, k 8), so streamed
bytes per token fall until `U → E`, where every step streams the `E − C` non-resident experts. Reported LRU hit ratios
fall with batch (ExpertFlow 76.6 → 58.4% at BS 4 → 16); the bounds reproduce the direction, not the measured curve.

### 2.6 Overlap: `SimOptions.offloadOverlap`

> **Superseded** by the prefetch accuracy `OffloadConfig.prefetchAccuracy` (default 0.8, URL key `opf`; see
> docs/simulation-model.md §12 "Expert prefetch"). `'ideal'` is p = 1 and `'none'` is p = 0, both bit for bit; old
> `oo=` URLs map to them. The "31–67% on Qwen2 / 512-expert models" below has no source for its 31%: the sourced
> figure for Qwen2-57B is 66.9% next-gate recall (ProMoE arXiv 2410.22134 §4.2). Kept as the original design.

Copies the `overlap` pattern (default in `DEFAULT_SIM_OPTIONS`, exhaustive `switch`, URL key `oo`, Assumptions
picker with `help`):

- `'ideal'` (**default**): the next layer's transfers are issued while this layer computes (prefetch; for tables the
  hashes depend only on token ids, so the gather can start at step start). Upper bound: the stage pays the larger
  of its compute/HBM time and its total transfer time. **Predictor accuracy is not modelled** (research systems
  report 84–96% on Mixtral, 31–67% on Qwen2 / 512-expert models; the bound assumes 100%).
- `'none'`: demand fetch, serialized with compute.

Stage formula (`pipeline.ts: stageResult`), `xfer_s = Σ_layers transfer op sec`:

```
core_s    = overlap === 'ideal' ? max(compute_s, bw_s) : compute_s + bw_s + p2p_s       (as today)
exposed_s = lat_s                                                                      (as today)
t_s       = offloadOverlap === 'ideal' ? exposed_s + max(core_s, xfer_s)
                                       : exposed_s + core_s + xfer_s
```

`xfer_s = 0` ⇒ both branches equal today's `t_s`. Transfers are not collectives (not in `lat_s`/`bw_s`);
`StageResult.transferSec` and `exposedTransferSec = t_s − (exposed_s + core_s)` are added.

### 2.7 Memory fit with tiers

```
W_s      = Σ resident block weights  (routed experts: C = f·E_loc per offloaded layer, a real number; tables: the non-table part)
staging  = (offloadOverlap === 'ideal' ? 2 : 1) · max_{phase, layer} ( ceil(M_ℓ)·W_expert,ℓ  |  usefulBytes of the table op )    (0 when nothing is streamed)
total_s  = W_s + B_r·state_s + act_s + staging  ≤ capacity_device
tier_bytes[host_ram | nvme] = Σ groups placed there (the FULL group: all experts of the offloaded layers, whole tables; the device cache is a copy)
                            + [dedicated ∧ nvme tables with scaleLayout 'pinned_host'] rows·scaleBytes on host_ram (§2.10)
                              (unified: those scales are device bytes in W_s instead)
fits = fits_device ∧ ∀ tiers (tier_bytes ≤ capacity_tier)
```

**As built (§2.11):** a stage that prefetches from both host RAM and the SSD keeps one buffer being read plus one
being filled per source, `staging = max_all buffer + max_{ram, q>0} buffer + max_{nvme, q>0} buffer` per phase
(`memory.ts: stageStagingBytes`, docs/simulation-model.md §12 "Overlap across tiers"); with one source it is the
double buffer `max (q > 0 ? 2 : 1)·buffer` (the prefetch rule that replaced `offloadOverlap`).

`MemoryReport.tiers: TierMemory[]` (§7.4); `maxSequencesPerReplica` uses `W_s + staging`. On unified memory
`host_ram` groups count in the device pool: no `host_ram` tier exists (an invariant in `stageTierBreakdown`).

### 2.8 CPU-compute mode (designed hook, rejected by validation for now)

`OffloadConfig.mode: 'cpu_compute'` (llama.cpp `-ot exps=CPU` / `--n-cpu-moe`, ktransformers at batch < 32): the
missed experts' GEMMs run on the host; only activations cross PCIe:

```
t_cpu,ℓ = max( F_miss / cpuPeak(math) ,  M·W_expert / (B_ram / n_stream) )     F_miss = 2·M·mE·(3 or 2)·h·fe
t_act,ℓ = 2 · (α_link + T·h·bpe(act) / B_link)                                  hidden states out, partial sums back
```

Hook only in the first implementation, because (1) no CPU preset carries cited matmul peaks per format and llama.cpp's
dequant kernels sit far below any AVX-512/AMX peak, so a "theoretical" CPU FLOP bound would mislead; (2) at the batch
sizes engines use it (< 32: llama.cpp's `GGML_OP_OFFLOAD_MIN_BATCH`, above which it copies weights to the GPU, i.e.
our `'transfer'` mode) the bound is host-DRAM bandwidth; (3) it needs a `HostComputeSpec` and its own §11 checklist.

### 2.9 EP > 1 (engine-level only)

Offload applies per device to its `E_loc = E/ep` experts with `A` assignments (`moeRouting.assignments`):
`U_dev = E_loc·(1 − (1 − 1/E_loc)^A)` (sampling with replacement across assignments; at ep = 1 the engine uses
§2.4's exact form), `C = f·E_loc`, `n_stream` counts the node's streaming devices. Warning emitted. No UI.

### 2.10 n-gram lookup tables (PLE, Engram)

**Why.** Qwen3.8-Flash-Next (PLE) and DeepSeek-V4.1-Flash (Engram) ship hashed n-gram tables designed to live
off-device: Qwen 320,001,536 rows × 160 bf16 = 320 B/row (the mapper's padded 320,000,000 is 0.0005% low, noted at
`qwen.ts:122`), 102.4 GB; DeepSeek two tables of 384,006,168 and 384,016,682 rows × 256 F8_E4M3 with one ue8m0 scale
per 32 elements (`embed.scale` F8_E8M0 `[rows, 8]`) = 264 B/row at mxfp8, 202.76 GB together. Today
`ngramResidentWeightBytes` counts them as VRAM: the fit for these models is 102–203 GB too pessimistic and the UI
never shows the table. Compute is already right (`lowerNgram` gathers `T·g` rows: Qwen `g = 16`, DeepSeek `g = 24`).

**Indexing (verified from the modelling code).** Qwen `ple_layer_ids` is **one-based**
(`configuration_qwen4_exp.py:45-46` "One-indexed decoder layer ids"; `modeling_qwen4_exp.py:1257`
`config.ple_layer_ids.index(layer_idx + 1)`; SGLang `qwen4_exp.py:1494`): config `[2]` ⇒ layer index 1, PLE at the
start of the layer (`modeling_qwen4_exp.py:1272`); the checkpoint has tensors only under `layers.1.ple.*`. The
mapper (`qwen.ts:90, 101`, `pleLayers.includes(i + 1)`) is correct. DeepSeek `engram_layer_ids` `[1, 14]` is
**zero-based** (`inference/model.py:1206`, `:933-934`; SGLang `deepseek_v4.py:3311`): checkpoint `layers.1.engram.*`,
`layers.14.engram.*`; `deepseekV41.ts:102` is correct. A note goes into `docs/hf-loader.md`.

**Placement knob** `lookupTables.tier: 'device' | 'host_ram' | 'nvme'`.

- **Simulator default: `'device'`** (strict identity; today's pessimistic fit stays unless the user acts).
- The mapper flags PLE and Engram blocks `offDeviceByDesign: true` (sources: the model cards). This drives the **UI
  only**:
  - **Prefill:** when the user turns any offload placement on (experts off-device, or opens the
    Lookup-tables picker) on a model with such a table, the picker pre-fills **Host RAM**.
  - **Hint state (visible before any knob is on):** the "Lookup tables" picker trigger shows its current value
    "Device" with an accent dot, and a chip button "Host RAM ↗" sits next to the trigger (as built: a button inside
    the trigger button would nest buttons); the memory card draws the table as its
    own hatched segment of the device bar labelled "n-gram table 102.4 GB"; the model summary lists "n-gram tables
    51.2 B" as a parameter line. Clicking the chip or the hatched segment sets Host RAM (one click, no prose; the
    "?" explains). This is a _hint state_ of the picker, not a different default: the simulator still runs at
    `'device'` until the click.
- Q1: should the UI _start_ these models at Host RAM? (As built: no; the hint state above, §9.)

**Cost per step** (§2.3 with the read classes of §2.5 per table: `rowBytes = d·bytesPerElement(lookup)` is the whole
logical row — 264 B for mxfp8 Engram since `bpe(mxfp8) = 1.03125` already carries the ue8m0 scales, 320 B for bf16 PLE —
split once into `valueBytes` 256 / 320 and `scaleBytes` 8 / 0; never add the scale bytes on top of `rowBytes`): host RAM:
`α = lat_ram + α_link`, `service = max((R−1)·lat_ram/QD, bytes_tier/(B_ram/n_stream), usefulBytes/B_link)`
(`bytes_tier` rounds each read to 64 B lines; the host gathers into one buffer, one DMA per table per step); NVMe:
`α = lat_nvme + α_link`, `service = max((R−1)·lat_nvme/QD, R/(IOPS/n), bytes_tier/(B_nvme/n), bytes_tier/(B_ram/2n),
usefulBytes/B_link)`; unified + SSD: no link terms.
**Scale layout** (`lookupTables.scaleLayout`, concrete defaults, Engram only; PLE has no scales): the checkpoint stores
the F8_E8M0 scales as a separate tensor (`embed.scale [rows, 8]`), so

- **`'separate'` (default for `host_ram`)**: two physical reads per row on the same tier (`256 B` value, `8 B` scale);
  on DDR5 that is 4 + 1 cache lines = 320 B of tier bytes, the same as one 320 B read, so the DDR5 numbers of W4 do not
  depend on the layout; useful bytes stay 264 B/row;
- **`'pinned_host'` (default for `nvme`)**: the value rows on the SSD (one 4 KiB page per row), the scales pinned in
  host RAM: `rows·8 B` per table is **counted in the host-RAM tier capacity** (Engram: 3,072,049,344 + 3,072,133,456 =
  6,144,182,800 B for both tables) and the 8 B RAM read per row is not costed (< 0.1% of the page read). **On unified
  memory** (Apple, Spark, Strix Halo) host RAM is the device pool, so there is no host-RAM tier: the same scales count
  in the device's weight bytes (`OffloadLowering.hostRamIsDevice`). `pinned_host` stays the default for `nvme` there
  too: rates are identical, only device bytes grow; defaulting to `interleaved` would change SSD reads for no physical
  reason, and a virtual host tier is the bug this fixes;
- **`'interleaved'`**: one 264 B read per row (a deployment that repacked the table); `requestsPerRow` of revision 2 is
  replaced by this field.
  Test U-16 checks tier bytes and PCIe bytes separately for all three layouts.
  Prefetch: `offloadOverlap` as for experts (ids known at step start; prefill knows all `P·g` rows up front). Decode is
  latency-bound (one `α` per table per step), prefill IOPS-bound on NVMe and link-bound on DDR5 (W3/W4).

**Memory fit**: the table shard `ceil(rows/tp)·d·bpe(lookup)` per device counts against its tier (summed over the
node's devices for the host tiers); projections, gate vectors and conv taps stay in `W_s`.

### 2.11 Mixed placement by layer range (host RAM + SSD)

**Why.** Host RAM often cannot hold every offloaded layer (Qwen3.8-Flash-Next fp8: 120.8 GB of experts against
96 GB of DDR5), but the SSD is ≈ 8× slower per miss. Splitting the layers keeps as many as fit in RAM and puts the
rest on the SSD.

**Placement** (`engine/types.ts: RoutedExpertRule, LookupTableRule`; `engine/offloadRules.ts`; `engine/host.ts:
resolveOffload`; `lower/offload.ts`; model doc §12 "Placement"):

- `routedExperts` and `lookupTables` take one rule or a list of rules over **disjoint** layer ranges. The single
  object is the one-rule list (pinned by `engine/e2e-mixed-offload.test.ts` "single-tier configs are unchanged" and
  `identity.test.ts` ID-3). `[]` = none. Tables live in given layers (Qwen PLE: layer 1; DeepSeek-V4.1 Engram: layers
  1 and 14), so a table split is a layer boundary between them; a table rule at `'device'` places nothing.
- Rules are sorted by `layers.from` when resolved, so list order never changes a result. `layers` may be absent only
  in a one-rule list (every layer, as before). Layer i takes the one rule containing it; uncovered layers keep that
  group on the device.
- **`residentFraction` and `hitRatio` are per rule.** The device cache is `C = f·E_loc` _per layer_, and the hit clamp
  `[max(0, U + C − E_loc), min(U, C)]` and prefetch's `M/(E_loc − C)` already use that layer's C, so a per-range f is
  exact, not an approximation. It lets a deployment spend VRAM where a hit saves the most (an SSD miss costs ≈ 8× a
  RAM miss on an RTX 5090 + 990 PRO), and the old form, which carries f and h on the group, normalises losslessly. The
  UI keeps one f / h control and writes it into both rules.
- Unified memory: each `host_ram` rule is dropped with the existing note (its layers stay device weights); the `nvme`
  rules remain. Nothing left ⇒ identity.
- `includeDraft` may be set on one rule only (validation); the drafter's experts take that rule's tier, f and h
  (`lower/offload.ts: draftExpertPlacement`, used by `lower/draft.ts` and `engine/memory.ts`).

**Whole-layer offload removed.** `denseLayers` (the `dense_layer` group of §2.1/§2.5) copied whole decoder layers
to the device every step (the Accelerate / FlexGen style). Local users don't run that: llama.cpp `-ngl N` computes
the layers it keeps off the GPU on the CPU. For a MoE model it also read every expert of those layers each step, a
trap next to the far cheaper expert offload. It is removed from the types, lowering, memory fit, validation (an input
that still carries `denseLayers` is rejected) and the UI. Partial offload of dense layers would come back as the
CPU-compute mode (§2.8, review F3), not as streaming. The one sanity row that used it (F-W08) is re-encoded and
excluded (docs/sanity-check.md).

**Cost and fit.** No formula changes: transfer ops are already per layer, so each layer's op reads from its rule's
tier (§2.3: RAM direct over PCIe, SSD with the staging term), prefetch's q is per op, and the tier bytes of §2.7 are
already summed per layer through the per-layer lookup.

**Contention: n_host** (`engine/simulate.ts: stageTransferPaths`, `engine/transfer.ts: TransferPath.hostStagingNStream`,
model doc §12 "Sharing"). Host RAM is one resource behind host-RAM reads and NVMe staging. Before, with stage A
streaming from RAM and stage B from the SSD, each got the full B_ram (granted twice). Now:

```
hostStreams(s) = streams_ram(s) ∨ streams_nvme(s)          dedicated memory; each device counted once
n_host         = n_stream rule of §2.3 over hostStreams
host RAM tier:   tier = bytes_tier / (B_ram / n_host)
NVMe tier:       tier, IOPS shared by n_nvme;  staging = bytes_tier / (B_ram / (2·n_host))
```

**Compatibility.** With one physical off-device tier, or with both tiers used only inside the same stage(s),
hostStreams = streams_tier, so n_host = n_tier and the result of an old input is unchanged: the identity pin, the
sanity reports and every single-tier test pass without expectation edits. The exception is intended: an old
single-object config whose **stages stream from different tiers** (pp > 1, e.g. tables in RAM on stage 0 and experts
on the SSD on stage 1) moves, because the double grant is gone. Pinned: Qwen3.8-Flash-Next pp 2, PLE table in RAM
(stage 0), experts of layers 24–47 on a 200 GB/s SSD (stage 1), decode @ 2k 15.86 → **8.64 tok/s @ 2k**: the expert
fetch is bound by host-RAM staging, bytes/(B_ram/4) instead of bytes/(B_ram/2) (`e2e-mixed-offload.test.ts`, "moved by
n_host"). The PCIe link is per device and never shared.

**Within one device the demand share stays serial; the prefetched share overlaps across tiers** (follow-up 1, built).
In decode, a layer's needed experts are known only after that layer's router runs, so the demand fetch (1 − q) of
layer j cannot start while layer i < j is still streaming: that share is summed in layer order. The prefetched share
q of a device that prefetches from both tiers overlaps across the two streams, bounded by the busiest stream, its
PCIe link and host RAM (`pipeline.ts: stageTransfer`). A device with one source, or at p = 0, keeps `xfer_s = Σ op
sec`, bit for bit.

**Worked example** (`engine/e2e-mixed-offload.test.ts`): Qwen3.8-Flash-Next fp8 (48 identical MoE layers, 512
experts, top-10, W_expert = 4.9152 MB), one RTX 5090, DDR5-5600 ×2 (89.6 GB/s, 96 GB), 990 PRO, PLE table on the
SSD, f = 0, b 1, decode @ 8k. Per layer M = 10 misses = 49.15 MB: a RAM layer is link-bound, 1.07 µs + 49.15 MB /
57.6 GB/s = 0.854 ms; an SSD layer drive-bound, 46.5 µs + 49.15 MB / 7.45 GB/s = 6.644 ms (staging 1.10 ms below it).
Transfer per step at p = 0 (serial): all-RAM 41.06 ms, 24/24 180.02 ms (exactly the mean), all-SSD 318.97 ms; at the
default p = 0.8 the 24/24 split overlaps to 163.61 ms (follow-up 1). Decode at p 0.8: **23.76 tok/s @ 8k** (all-RAM,
whose PLE table on the SSD also overlaps; host RAM 120.8 GB > 96 GB, does not fit), **6.07 tok/s @ 8k** (24/24; 60.4
GB in RAM, fits; serial 5.52), **3.12 tok/s @ 8k** (all-SSD). Tables split the same way: DeepSeek-V4.1-Flash with
Engram layer 1 in RAM and layer 14 on the SSD sits between both-in-RAM and both-on-SSD, its transfer exactly their
mean at p = 0 and below it at p > 0.

**Follow-up 1: overlap the prefetched share across tiers (done, as built).** Only the prefetched share q
(default p = 0.8 at b 1), fetched during the previous layer's compute, can overlap: an SSD → host RAM read of one
layer can run while another layer's host RAM → GPU copy is in flight, so only the final PCIe hop and host RAM
bandwidth are shared. The demand share (1 − q) stays serial (router-bound, above). Per device and stage:

```
xfer_s = Σ_ops (1 − q)·t_op
       + max( Σ_ram q·t_op,  Σ_ssd q·t_op,                         per-source limits (each tier's own stream)
              Σ_ops q·usefulBytes / B_link,                         the shared PCIe link
              (Σ_ram q·bytes + 2·Σ_ssd q·bytes) / (B_ram / n_host) ) host RAM: RAM reads + SSD staging (written, read)
```

What it needed (the original sketch): per-op link bytes, host-RAM bytes and per-source tier time on `OpResult`; a new
`xfer_s` in `pipeline.ts: stageResult`; a staging buffer per tier in flight; a one-source branch for bit-identity.
Estimate for the worked example: 0.2·180.0 + 0.8·159.5 ≈ 163.6 ms per step, ≈ 6.07 vs 5.52 tok/s @ 8k (≈ +10%).

**As built** (docs/simulation-model.md §12 "Overlap across tiers"; `engine/pipeline.ts: stageTransfer`,
`engine/transfer.ts: prefetchesFromBothTiers`, `TransferTime.linkBytes / hostRamBytes / hostRamSec` passed up as
`OpResult.transferResources`, `engine/memory.ts: stageStagingBytes`). Settled points:

- The overlapped branch applies only when a stage has an op with q > 0 from host RAM AND one from the SSD; every other
  stage runs the former expressions in the same float order, so one-source and p = 0 configs are bit-identical.
- `P_s = max(Σ_ram q·t_op, Σ_ssd q·t_op, Σ q·linkSec_op, Σ q·hostRamSec_op)`, `xfer_s = Σ (1 − q)·t_op + P_s`. The
  per-source sums use the full t_op (α included: one tier's ops stay serial). The host RAM term sums each op's time
  on host RAM: tier bytes (rounded to the granularity) / (B_ram / n_host) for a RAM read, / (B_ram / (2·n_host)) for
  SSD staging, the same terms `transferTime` charges.
- The hiding term keeps its form with `pbar_s = P_s / xfer_s`; p = 1 gives `t_s = lat + max(core, P_s)`. Since
  `P_s ≤ Σ q·t_op`, overlapped results are never slower than serial.
- `StageResult.transferSec` reports the effective `xfer_s`.
- Staging: one buffer being read + one being filled per source, `max_all + max_ram + max_ssd` (§2.7 note); with
  one source it reduces to the double buffer.

Computed for the worked example (p 0.8): D = 35.99 ms, P = the SSD stream 0.8·24·t_ssd + t_table = 127.62 ms (link
32.76 ms, host RAM 31.60 ms) ⇒ **163.61 ms per step** (estimate 163.6) and **6.068 tok/s @ 8k** (estimate 6.07,
serial 5.522). The rate differs slightly from 1/163.6 ms because the stage's compute (≈ 5.4 ms) is hidden only by
`pbar = P/xfer = 0.78`, and the PLE table (q = 1) adds its whole time to the SSD stream. At p 1: 159.51 ms,
6.269 tok/s @ 8k. Staging of 24/24 grows from 2 to 3 layers of experts (prefill: 7.55 GB), still fitting the 32 GB
device.

**Follow-up 2: host RAM as a cache over the SSD (three-level VRAM → RAM → SSD, design only).** Hot experts of
SSD-resident layers kept in RAM. What it needs:

- two sources per expert fetch: `TransferOp.source` is one tier today, so an op per level (`reads` from RAM for the
  level-2 hits, from the SSD for the rest), or a `sources` list;
- RAM capacity for the cache copy (`f_ram·E_loc·W_expert` per SSD layer on the host-RAM tier, on top of the SSD copy
  for an inclusive cache);
- a level-2 hit model. For an exclusive hierarchy with the same h(f) curve:
  `hits_dev = U·h(f_dev)`, `hits_ram = U·(h(f_dev + f_ram) − h(f_dev))`, `misses_ssd = U − hits_dev − hits_ram`, each
  clamped like `expertHitsMisses` (`hits_dev + hits_ram ≤ min(U, C_dev + C_ram)`).

Open questions: inclusive vs exclusive caches (capacity vs promotion traffic); whether prefetch also prefetches from
RAM (q per level); whether RAM-cache fills (SSD → RAM promotions) cost host RAM bandwidth at steady state; and the UI
(a second fraction next to f, or "RAM holds hot experts of SSD layers").
---

## 3. Speculative decoding

### 3.1 Methods (primary sources)

| method                                                        | drafter                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | draft passes per step                                                                                          | verify                   | own KV                                                                      | sources                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MTP** (DeepSeek-V3/V4, GLM-4.5+, Qwen3-Next/3.5+/3.8, MiMo) | the checkpoint's `num_nextn_predict_layers` / `mtp_num_hidden_layers` module(s): shared embedding, `eh_proj` (2h → h) over `[norm(h_prev); norm(emb(token))]`, one target-shaped decoder layer, shared LM head                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | **k sequential** passes (each: 1 layer + head)                                                                 | chain of k+1             | 1 layer, full context, per module                                           | DeepSeek-V3 arXiv 2412.19437 §2.2; Qwen3.8-27B card ("MTP trained with multiple steps")                                                                                                                                                                                                                  |
| **DFlash 2** (Inco AI / z-lab)                                | separate Qwen3-architecture drafter read from its own `config.json`; per `dflash/model.py`: `fc` (`len(target_layer_ids)·h → h`) + `hidden_norm` over the **newly committed tokens'** target features; each layer: q/k/v/o projections, q/k RMSNorm, gated MLP, with `k_proj`/`v_proj` applied to **both** the context features (`k_ctx`/`v_ctx`, appended to the draft cache) and the block hidden states; **two `GroupedDynamicCausalConv`** per layer (before attention and before the MLP), each a `kernel_projection` GEMM `h → 2·K·(h/g)` plus a 2-tap grouped conv; a `CandidateSelector` with two `V×r` codebooks and one `h → r` projection; shared frozen embedding and LM head; non-causal block attention with a sliding window | **1 parallel** pass over the block (anchor + k masks) + the selector's sequential walk over precomputed scores | chain of k+1             | per layer `2·KV·hd` elems/token over the window                             | https://github.com/z-lab/dflash/blob/main/dflash/model.py (`Qwen3DFlashAttention.forward`, `GroupedDynamicCausalConv`, `CandidateSelector`, `DFlashDraftModel.__init__`); config https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/blob/main/config.json ; blog https://inco.ai/blog/dflash2/ (no paper) |
| **DSpark** (PKU + DeepSeek)                                   | DFlash-style parallel backbone (same `fc`, KV injection, shared embed/head; config `Qwen3DSparkModel`) **+ Markov head** `B(x_{j−1},·) = W1[x_{j−1}]·W2` (`markov_rank` 256, `V×r` and `r×V`) adding a prefix-dependent bias to the block logits, sampled left to right, **+ confidence head** (`enable_confidence_head`) driving a per-request verify length; DeepSeek-V4 production: 3 MoE layers under `mtp.*`, window 128                                                                                                                                                                                                                                                                                                               | 1 parallel pass + k sequential Markov steps                                                                    | chain, ≤ k+1 per request | as DFlash (full attention in the released Qwen3 drafters; window 128 on V4) | arXiv 2607.05147 §3.1–3.2; config https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7/blob/main/config.json ; https://github.com/deepseek-ai/DeepSpec                                                                                                                                              |

Released drafter shapes (config.json, R): `incoai/Qwen3.8-27B-DFlash2`: 5 layers, h 5120, FFN 17408, 32 q / 8 kv
heads, hd 128, all `sliding_attention` window 2048, `is_causal false`, `target_layer_ids [5, 19, 33, 47, 61]` of 64,
`block_size 8`, `conv_kernel_size 2`, `conv_group_size 16`, `selector_top_k 16`, `selector_rank 256`, V 248,320.
`deepseek-ai/dspark_qwen3_8b_block7`: 5 layers, h 4096, FFN 12288, 32/8, hd 128, `full_attention`,
`target_layer_ids [1, 9, 17, 25, 33]` of 36, `block_size 7`, `markov_rank 256`, `markov_head_type vanilla`,
`num_anchors 512`, V 151,936. Note the drafter's attention is **not** the target's (Qwen3.8-27B: gated 24/4 heads,
hd 256, mostly linear attention; its drafter: 32/8, hd 128, softmax): the drafter is its own architecture.

Not modelled, and why: **trees** (a different engine feature; chain τ formulas do not apply); DSpark's **confidence
scheduler** (engine policy; fixed `k`); **context dependence of α** (DFlash v1 drafters trained at 4K lose 25–55% of τ
at 16–32K; α is a user input quoted at the context); the selector's and Markov head's **sequential walks** (latency,
not bytes); the DSpark confidence head and `num_anchors` (unverified layout; < 0.1% of params); the blog's "+2.0 M
selector params" (the code's codebooks are `2·V·r` = 127 M for V 248k; the code is primary).

### 3.2 IR: `ModelSpec.draft?: DraftSpec`, read per checkpoint

```
DraftSpec =
  | { kind: 'mtp';  layers: LayerSpec[];  hiddenProj: boolean;  maxReliableDepth?: number }
  | { kind: 'block_drafter'; method: 'dflash2' | 'dspark';
      hidden; vocab; layers: number; heads; kvHeads; headDim; intermediate; gated; activation;
      windowTokens?: number; targetLayerIds: number[]; blockSize: number;
      conv?: { kernel: number; groupSize: number };            // DFlash 2
      selector?: { topK: number; rank: number };               // DFlash 2
      markovRank?: number;                                     // DSpark
      source: string }
```

(JSDoc in §7.1.) **MTP**: the HF loader keeps the `LayerSpec` it already builds (`glmMoeDsa.ts:59-60`,
`glm5Next.ts:109-110`, `qwen.ts:107-109`, `inkling.ts:77`, `deepseekV41.ts:165-189`, `mistralLarge4.ts:40-53`) as
`draft: { kind: 'mtp', layers, hiddenProj: true }`; `mtpParamsEstimate` stays for the param check; the draft is
excluded from `countParams(model)` and the fit unless spec-dec is on. `maxReliableDepth` = the module count for
single-step heads (DeepSeek-V3-style `num_nextn_predict_layers`: GLM, Inkling; one draft position per module),
`undefined` for Qwen3.5+/3.8; several modules run as a chain (pass p uses module min(p, L)). MiniMax-M3 ships no MTP
tensors (no draft). `deepseek_v41` checkpoints with a DSpark drafter under `mtp.*` map to `kind: 'block_drafter',
method: 'dspark'` with the family's layers (V4.1-Flash: 3 MoE layers, `dspark_block_size 5` ⇒ **`blockSize 6`**, default
k 5, `windowTokens 128`; these use the MoE `LayerSpec` path: `moeLayers`). Mistral Large 4's DSpark is **deferred**:
its `params.json` lacks the drafter's head counts and dims and declares head type `rnn`.

**`blockSize` semantics (as built).** `blockSize` is the largest verify width k + 1. DFlash's `block_size` counts the
anchor (k ≤ block_size − 1); DeepSpec's DSpark `block_size` counts draft positions (k ≤ block_size), so the loader
stores `blockSize = block_size + 1` for DSpark (Qwen3-8B `block7`: blockSize 8, k 7). A DSpark pass embeds k rows
(anchor + k − 1 masks) and takes logits at all k; a DFlash 2 pass embeds k + 1 rows (anchor + k masks).

**Block drafters** are separate repos. The UI's method picker offers (a) **a drafter repo id** (`draftRepoId`, e.g.
`incoai/Qwen3.8-27B-DFlash2`), whose `config.json` the loader reads into the fields above (`hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`,
`sliding_window`/`layer_types`, `dflash_config.*` or top-level `target_layer_ids`, `block_size`, `markov_rank`,
`vocab_size`); and (b) **"hypothetical drafter"**, an explicitly labelled fallback (defaults only in v1, not
configurable: 5 layers, the target's `h` and `V`, 32/8 heads, hd 128, FFN = the target's dense FFN width or `topK·fe`,
window 2048, `targetLayerIds` from DFlash's `build_target_layer_ids`), shown with a "hypothetical" badge in the
results. The engine is pure: `SpecDecodeConfig.drafter` carries the parsed `spec` in both cases (§7.3).

### 3.3 Draft and verify ops (all through the existing IR)

Per decode step at context N; `b` = `b_mb` sequences; **`m_ctx = ⌈E[τ]⌉`** = the newly committed tokens (accepted +
bonus) of the previous step, whose target hidden states every drafter ingests: the block drafter through `fc`/`k_ctx`/
`v_ctx` (`dflash_generate`: `target_hidden[:, :produced]`, `produced = acceptance + 1`), the MTP module through its own
forward over them (below). Choice and justification: `E[τ]` is the expected count; GEMM shapes need an integer row
count, so the engine uses `⌈E[τ]⌉` for both drafters (one rule). Rounding up is conservative by at most one row of
activations per step (the weight reads, which dominate, do not depend on `m_ctx`); an expected-value form (`m = E[τ]`
real) would give fractional attention/KV-write shapes that the IR does not allow. With explicit per-position
acceptance the count is exact (`alphas [1]` at k = 1 ⇒ `E[τ] = 2`; `alphas [0]` ⇒ 1), which the tests use (U-18).

**Verify** = the target lowered with `qLen = k + 1, kvLen = N + k + 1` (head on all positions, causal triangle over
`past = N`, absorbed MLA; `simulate.ts:161-162` generalized). GEMMs `m = b·(k+1)`; attention reads `N + k + 1` keys
and writes `k+1` (rejected KV entries are overwritten next step; not charged back). MoE targets: `a = min(E_loc,
b·(k+1)·k_moe)` and the offload path uses `U(b·(k+1))` (§3.8). **Recurrent / conv state** (Qwen3.x hybrids, the
main MTP users): a `k+1`-token verify in `'step'` mode advances the state past the accepted prefix, so the engine
costs **per-position snapshots**: `RecurrentOp.stateSnapshots = k + 1` (default 1) multiplies the state **write**
bytes (`stateWriteBytes·(k+1)`; the read stays one), likewise `ElementwiseOp.stateWriteBytes·(k+1)` for the conv
rolling state; transient memory `(k+1)·state_s` per sequence on the stage (`DeviceMemory.speculativeTransientBytes`).
Test U-14.

**Draft**, appended to the last stage as layers `draft.<i>` (by `simulate.ts: withDraftLayers` after lowering, not
by `lowerModel`, to avoid an import cycle; so stage sums, overlap and collectives need no new rule _within one
stage_; PP > 1 is rejected, §3.6):

- **MTP, catch-up pass + k−1 recursive passes.** The module keeps its own KV cache over the committed history, so each
  step it must first **advance that cache by the `τ` positions the target committed** (the accepted drafts plus the
  bonus token), not by one. As in the DeepSeek-V3 MTP forward (and the SGLang/vLLM NextN workers), the module runs
  once over those positions with the target's hidden states (available from the verify pass) and the next-token
  embeddings; the output at the last position is the first draft token. Per step:
  1. **catch-up pass** (`qLen = m_ctx`): `embed` (`b·m_ctx` rows) → `norm_h`, `norm_e` (`b·m_ctx·h` each) → `eh_proj`
     GEMM (m `b·m_ctx`, n h, k 2h; module `attention`) → the module's layer(s) in decode context `qLen m_ctx, kvLen N +
     k + 1` (causal over `past = N + k + 1 − m_ctx`; KV write of `m_ctx` tokens, the default `kvWriteTokens = qLen`;
     hybrids: `stateSnapshots` default 1, since only the final state is kept) → `final_norm` and `lm_head` over the
     **last position only** (m b): the earlier positions' logits are not needed.
  2. **passes 2..k** (`qLen = 1` each): the same ops with one row, `kvLen` growing by one per pass (documented
     simplification: `N + k + 1` for all), producing draft tokens 2..k.
  3. **Tentative rows.** The KV rows the module wrote for draft positions (k−1 recursive passes, plus the catch-up
     rows beyond the accepted prefix of the _previous_ step) are **overwritten** by the next step's catch-up pass (the
     cache is cropped to the committed length, as `_crop_to` does for the block drafter); nothing is charged back.
     The shared embedding and LM head are the target's: on the same device (pp = 1) they cost no extra memory and are
     read once per pass (once-per-op). For k = 1 the step is exactly one pass of `qLen = m_ctx` (W2). With several MTP
     modules (L > 1) the passes run as a chain: pass p uses module min(p, L).
- **Block drafter (DFlash 2 / DSpark), one pass** over `B` block rows per sequence: DFlash 2 `B = k + 1` (anchor + k
  masks), DSpark `B = k` (anchor + k − 1 masks; §3.2 `blockSize`). Below, `k+1` stands for `B`:
  1. `draft.fc` GEMM (m `b·m_ctx`, n h, k `|targetLayerIds|·h`) + `draft.hidden_norm` (RMSNorm over `b·m_ctx·h`).
  2. `draft.noise_embed`: `EmbeddingOp` of `b·B` rows.
  3. per layer ℓ: `input_norm` (block rows `b·(k+1)`); DFlash 2: `attn_conv.kernel_proj` GEMM (m `b·(k+1)`, n
     `2·K·h/g`, k h) + `attn_conv` elementwise over `b·(k+1)·h` with `4·K` FLOP/elem (base + dynamic taps, `prepare`);
     `q_proj` (m `b·(k+1)`, n `H·hd`, k h); `k_proj` and `v_proj` each as **one GEMM with m = `b·(m_ctx + k + 1)`**
     (applied to the context features and the block, weights read once, as in the code's two calls); `qk_norm`
     over `(k+1)·H + (m_ctx + k + 1)·KV` heads; `rope`; `attn.core`: `seqs b, heads H, kvHeads KV, qLen k+1, kvLen
     min(N, W) + m_ctx + k + 1, causal false, kvSelection window W` (or full), `kvRead` `2·KV·hd` per key,
     **`kvWriteTokens = m_ctx`** (only the context K/V stay in the cache; new optional `AttentionOp` field, default
     `qLen`); DFlash 2: `attn_conv` `finish` elementwise (`4·K` FLOP/elem: base and dynamic taps are applied in both
     prepare and finish); `o_proj`; residual; `post_norm`; DFlash 2:
     `mlp_conv.kernel_proj` + `prepare`; `mlp.gate_up`, `act`, `down` (m `b·B`); DFlash 2: `finish`; residual.
  4. `draft.norm` over the block; **`lm_head` over `b·k` rows** (DFlash 2: the k mask positions,
     `draft_hidden[:, 1−B:]`; DSpark: all k block rows).
  5. DFlash 2 selector: `selector.topk` elementwise over `b·k·V` logits (reads the logits, 5 FLOP/elem, writes
     `16` ids + scores per row); `selector.hidden_proj` GEMM (m `b·k`, n r, k h); `selector.codebooks`:
     `EmbeddingOp` of `b·k·(1 + 16)` rows of r (predecessor + 16 successors); `selector.score` elementwise
     (`fn 'candidate_score'`) over `b·k·16·r` (2 FLOP/elem). Sequential walk: not costed.
  6. DSpark Markov head: `k` GEMMs `markov.W2` (m b, n V, k r), each preceded by an `EmbeddingOp` of b rows of r
     (`W1[x_{j−1}]`) and followed by an elementwise add over `b·V`; the confidence head is not costed.
     **DeepSeek-V4.1 MoE drafter** (`moeLayers`, approximation): the context rows are added to the m of the CSA `wkv`
     GEMM, the attention runs non-causally with `kvWriteTokens = m_ctx`, and the context rows' kv norm / RoPE are not
     added.
     All quantization follows `QuantConfig` through the module classes (`attention` for q/k/v/o and `fc`, `dense_mlp`
     for the MLP, `lmHead` for the shared head, `embeddings` for codebooks and `W1`/`W2`, `residual_mixer` for the
     conv kernel projections) — the drafters ship bf16; there is no draft-specific quant override (Q6, decided: none).

**Draft memory** (last stage; `blockResidentWeightBytes` over the draft layers + `fc`/`eh_proj` + conv kernels and
projections + selector codebooks and projection + `W1`/`W2`; the shared embedding and LM head are **not** counted
twice at pp = 1, where they are already on the device) and draft KV per sequence
(`layers · min(ctx, W) · 2·KV·hd · bpe(kv)`; MTP: full context). Under spec-dec the memory fit uses the context
`max(P, N + k + 1)` (the verify appends k + 1 rows); the per-position snapshots of §3.3 are the stage's state blocks'
`blockSequenceStateBytes` at context 0 (recurrent and conv state, which do not grow with the context).

**Parameter-count tests (U-10) against the released configs**, bf16, excluding the shared embedding/head:
`incoai/Qwen3.8-27B-DFlash2`: per layer `q 5120·4096 + k 5120·1024 + v 5120·1024 + o 4096·5120 + qk-norms 256 +
mlp 3·5120·17408 + norms 2·5120 + 2 convs·(base 2·2·5120 + proj 5120·1280) = 332,974,336`; ×5 + `fc 5·5120·5120 =
131,072,000` + norms 10,240 + selector `2·248,320·256 + 5120·256 = 128,450,560` = **1,924,404,480** (model card:
1.924 B). `deepseek-ai/dspark_qwen3_8b_block7`: per layer 192,946,432; ×5 + `fc 83,886,080` + norms 8,192 +
Markov `2·151,936·256 = 77,791,232` = 1,126,417,664; plus its own embedding and head copies `2·151,936·4096 =
1,244,659,712` (the checkpoint stores them) = **2,371,077,376** (safetensors: 2.371 B). Tolerance 0.1% (DFlash 2)
and 1% (DSpark: confidence head not modelled).

### 3.4 Draft-cache population (prefill) and maintenance

Spec-dec on changes the **prefill** phase too: the drafter's context cache must be filled from the prompt, so prefill
appends `draft.*` layers to its last stage (same `phaseResult` rules):

- MTP: the module's ordinary prefill forward over the prompt (`embed` P rows, norms, `eh_proj` m `b·P`, the layer with
  `qLen = kvLen = P` writing its KV through its attention op as usual, `lm_head` for one position per sequence like the
  target's prefill head). This is a real forward (the module attends causally over the prompt), so no special op.
- Block drafter: the context K/V are **projections of target features**, not an attention pass, so prefill emits per
  layer exactly: `k_proj`, `v_proj` (m `b·P`), `qk_norm` over the K heads (`b·P·KV·hd` elements), `rope` over the same,
  and a **write-only cache op**: `ElementwiseOp` with the new `fn: 'kv_cache_write'` — `elems = b·min(P, W)·2·KV·hd`,
  `flopsPerElem = 1` (the cast act → kv format), `activationReadBytes = elems·bpe(act)` (the projected K and V rows
  it stores), `activationWriteBytes = 0`, and `kvWriteBytes = elems·bpe(kv)` (a new optional `ElementwiseOp` field,
  default 0, mapped to `ByteBreakdown.kvWriteBytes` by `elementwiseBytes`). Only the last `min(P, W)` tokens are
  written for a windowed drafter (the rest are projected and discarded: the GEMMs still run over `b·P` rows, as the
  code does). Plus `fc` (m `b·P`) and `hidden_norm` once. **No `attn.core`, no `q_proj`/`o_proj`, no MLP, no head,
  no selector/Markov op in prefill.** Test U-15 asserts the exact op-name multiset.
- Maintenance per decode step: the `m_ctx` new context rows of §3.3 (block drafter: `fc`, `k_ctx/v_ctx` inside the
  per-layer projections and `kvWriteTokens = m_ctx`; MTP: the catch-up pass). Rejected block positions and tentative
  MTP rows leave nothing in the caches (`_crop_to`). Tests U-15, U-18.

### 3.5 Accepted tokens, effective rates, batch

```
i.i.d.:          E[τ] = (1 − α^(k+1)) / (1 − α)                      (Leviathan et al. 2023, arXiv 2211.17192 Eq. 1)
per-position:    E[τ] = 1 + Σ_{i=1..k} Π_{j=1..i} α_j                 (DSpark §3.2.2; α_j conditional on 1..j−1 accepted)
t_step        = cycle(verify stages with draft.* appended)            today's cycle rule on the combined stage times
tokensPerSec  = dp · B_r · E[τ] / t_step   @ N ;   perSequence = E[τ] / t_step ;   speedup = tokensPerSec / tokensPerSec(off)
latencySec    = t_step / E[τ]   (decode: time per output token; cycleSec stays the step time)
```

`acceptance: { kind: 'iid', alpha } | { kind: 'per_position', alphas }` (a short vector is extended with its last
value). Ranges: i.i.d. `α ∈ [0, 1)` (the closed form divides by `1 − α`); per-position `α_j ∈ [0, 1]`, so the
all-accepted endpoint is a legal input through the vector form (`alphas [1]` at k = 1 ⇒ `E[τ] = 2 = k + 1`, the
limit of the i.i.d. formula as α → 1). τ includes the bonus token (1 ≤ τ ≤ k+1).

**Batch.** A dense GEMM `m×k · k×n` is weight-bound while `2·m·n·k/peak < n·k·bpe(w)/memBW`, i.e.
`m < peak·bpe(w)/(2·memBW)` (before tiling and the activation operands): on an RTX 5090 bf16/fp32-acc,
`209.5e12·2/(2·1.792e12) ≈ 117` tokens. The verify's `m = b·(k+1)` crosses it at `b ≈ 117/(k+1)` (58 at k = 1, 14 at
k = 7); past it verify time grows ∝ (k+1) and the gain shrinks toward `E[τ]/(k+1) < 1`. The shape-efficiency
roofline reproduces this with no extra rule (the Qwen3.8-27B card's 3.4× at c1 → 1.45× at c32 is this plus
scheduling). W2 checks batch 8 at k = 1 (`m = 16 < 117`: still weight-bound).

**Single-step MTP heads and k > 1.** Reusing a one-step head recursively collapses after position 1 (FastMTP
arXiv 2509.18362: ~70% → ~10% → ~0). When `k > draft.maxReliableDepth` the engine only **warns**; switching a kept
preset acceptance to the per-position vector `[α, 0.30, 0.10, 0.05, …]` (tail interpolated between FastMTP's vanilla
0.10 and fine-tuned 0.55; low confidence; `specdec.ts: mtpTailAcceptance`) is the UI preset's job, since the engine
only sees the acceptance it is given. The UI defaults `k = 1` for these heads.

### 3.6 PP > 1: implemented

First rejected (`"speculative decoding with pp > 1 is not supported yet"`), now implemented as below
(docs/simulation-model.md §13 "Pipeline parallelism"; §10 deviation 43). The draft on the last stage needs (a) the
**token embedding**, resident on stage 0 only: either a second copy on the last stage (`+ceil(V/tp)·h·bpe`) or one
`send_recv` of `b·(k+1)` rows per step; (b) for block drafters the **target features of `targetLayerIds`**, which may
live on earlier stages: one `send_recv` of `b·m_ctx·h·bpe(act)` per feature layer on an earlier stage per step
(`α + β` each, on the decode latency path), or `(k+1)` rows at verify time; (c) the G2 loop-back of accepted ids.

Chosen: (a) the copy (`memory.ts: draftEmbeddingCopyBytes`; none for a tied head in the same format, which the
last stage already holds), as vLLM's EAGLE proposer loads its own `embed_tokens` when the PP world size is > 1; (b)
the `(k+1)` verify rows of the `F_s` features made before each boundary (the stage's own last-layer output is the
send itself and adds nothing there), carried on the existing activation
send (`draft.ts: ppBoundaryDraftFeatureVectors`, `simulate.ts: withDrafter`): one message per boundary, no extra α
and no round trip after acceptance, against more bytes than the per-layer sends of `m_ctx` rows; (c) the G2
loop-back as built (`b_mb·(k+1)·4 B`). pp > 1 with the token embedding off-device is rejected
(`SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR`). Tests: `engine/specdecPp.test.ts` (F-1).

At **pp = 1** the shared embedding and LM head are on the drafter's device: no extra memory, and their reads per
pass are already in the op list.

### 3.7 Collectives per step (for the audit) and EP/DP

Verify: one target forward ⇒ the same `2·L + 1` TP all-reduces (+ mode-B EP all-to-alls) as a decode step with
`T = b·(k+1)`. Draft: MTP `k·(2·L_draft + 1)`; block drafters `2·L_draft + 1` (+ the selector/Markov head is
replicated). These are **logical message counts**; the α/β of every message come from the collective cost functions
in `engine/collectives.ts` on the group's `CollectivePath` (whatever fabric shape and hop count the topology gives;
[`#specmaxx`](`ref://task:01a1bb67-ba01-75d8-ba32-559c0e8f81a4`) adds switched / ring / full-mesh / chain fabrics with
per-hop latency and multi-hop all-to-all and p2p there), so the draft layers inherit them by lowering through the same
`CollectiveOp`s. Spec-dec is per replica (DP unchanged); EP > 1 uses mode A/B routing with `T = b·(k+1)`, no new rule.

### 3.8 Spec-dec × offload

The verify's `U(b·(k+1))` replaces `U(b)` in §2.4 (E 128, k_moe 8, b 1: `U(1) = 8` → `U(2) = 15.5` at k = 1,
`U(8) = 51.6` at k = 7). The draft's own MoE layers (DeepSeek-V4 DSpark) follow the routed-expert placement only
with `includeDraft: true` (default false). Transfer time enters `xfer_s` of the last stage like any other.

---

## 4. Interconnect-latency audit

### 4.1 Historical audit: where α was applied before the corrections (verified), per crossing

Convention for this section and for every new cost above: the engine counts **logical messages** per layer and per
step; the per-message α and β (hop count, ring/tree/direct algorithm, host bounce, fabric shape) are whatever the cost
functions in `engine/collectives.ts` return for the group's `CollectivePath`. Nothing here assumes a switched fabric;
the parallel fabric-shape task ([`#specmaxx`](`ref://task:01a1bb67-ba01-75d8-ba32-559c0e8f81a4`)) changes those
functions, not these counts. The offload `TransferOp` is the one cost that does not go through them (host link, §2.3).

| crossing                                   | α applied at                                                                                            | per layer               | per decode step                                           | with spec-dec                                       | gap                                                                                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ----------------------- | --------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| TP all-reduce (mixer, FFN, embed, n-gram)  | `collectives.ts:82-99` (`2(n−1)α` ring / `2⌈log2 n⌉α` tree) via `tpAllReduce` `lower/common.ts:344-360` | 2 (+1 per n-gram block) | `2L + 1`                                                  | `+ k·(2·L_draft + 1)` MTP / `+ 2·L_draft + 1` block | **G7** `'best'` picks ring/tree by `totalSec` (`:116-118`), not by exposed α under `'ideal'`                                     |
| LM-head logits / sampling exchange         | **none** (`head.ts:3-5` "treated as free")                                                              | —                       | 0 (should be ≥ 1 all-gather of top-k scalars when tp > 1) | ×(1 + passes)                                       | **G1**                                                                                                                           |
| EP all-to-all (mode B: dispatch + combine) | `:148-155` one α flat; `:228-242` `α_inter` only hierarchical                                           | 2                       | `2·L_moe`                                                 | `+ 2·L_moe,draft`                                   | **G3** hierarchical drops `α_intra` (wrong with host-bounce intra, α_eff 13 µs); **G4** flat one α assumes a non-blocking fabric |
| PP send/recv forward                       | `:158-162` `α + S/B`; `pipeline.ts:118-119, 125, 139`                                                   | —                       | `pp − 1`                                                  | same, `k+1` rows                                    | **G5** the send's α/β are not in `StageResult.commLatencySec/commBandwidthSec` ⇒ no phase-level α total                          |
| PP loop-back (sampled token → stage 0)     | **none** (`lower/lower.ts:83-97`)                                                                       | —                       | 0 (should be 1)                                           | 1 (`k+1` ids)                                       | **G2**                                                                                                                           |
| PP host-bounce send β                      | `collectives.ts:258` `nActive = 1`                                                                      | —                       | —                                                         | —                                                   | **G8** `tp·dp` ranks share one host                                                                                              |
| DP                                         | no collectives (model doc §5)                                                                           | 0                       | 0                                                         | 0                                                   | none                                                                                                                             |
| cross-stage cross-layer data               | **none**, warning (`simulate.ts:56-76`)                                                                 | —                       | 0                                                         | —                                                   | **G6** (out of scope)                                                                                                            |
| offload transfer                           | new `TransferOp` (§2.3): `α_tier + α_link` once per layer batch                                         | 1 per group             | `L_off`                                                   | `+ L_off,draft`                                     | — (own fields)                                                                                                                   |

### 4.2 In this feature (display-only): the latency vs bandwidth split

`PhaseResult.comm = { latencySec, bandwidthSec, exposedSec }` = `Σ_s lat_s + Σ α_p2p`, `Σ_s bw_s + Σ β_p2p`,
`Σ_s exposedComm_s + sends on the latency path` — sums of numbers the engine already computes (`StageResult`,
`sendToNext`), so `cycleSec`/`tokensPerSec` are untouched (identity). `PhaseResult.transfer? = { sec, exposedSec,
bytes, requests }`, `speculative? = { draftSec, verifySec, expectedTokensPerStep, speedup }`. UI
(`ui/state/summary.ts: BreakdownRow.boundSec`): **"comm latency"** (Σ `op.commLatencySec`), **"comm bandwidth"**
(Σ `op.commBandwidthSec`) and **"offload transfer"** (`bound: 'transfer'`) are bound parts of each breakdown row
(bar segments and legend entries), not extra rows; transfer ops still appear as their own rows by op / kind. `byLayer`
puts **"draft"** (the `draft.*` layers) and **"verify"** (the rest of a spec-dec decode) as group header rows over
their per-layer rows. `Bound` becomes `'compute' | 'memory' | 'comm' | 'transfer'`. This closes G5 for display without
changing any rate.

### 4.3 Implemented latency-audit corrections (behaviour-changing, goldens updated)

All of G1–G4, G7, G8 are applied, per the user's §9.7 decision. G6 is excluded. The table below preserves the
**pre-implementation expected deltas**; measurements and corrections to its estimates follow it. Fixes were
applied in order G1, G2, G3, G4, G7, G8, with `UPDATE_DEPLOYMENT_IDENTITY=1` regeneration and a JSON diff
against the previous snapshot after each. No memory leaf or warning changed. "G-x" means a §8.3 golden ID.

| fix                                 | change                                                                                                                                                                 | Δ per step (formula)                                                                          | numeric example                                                                                                                                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G1 sampling exchange                | after `lm_head`, one `all_gather(tp)` of `Tout·tp·8 B` per forward                                                                                                     | `+ (n−1)·α_eff` (flat ring) or `+ AG_inter + AG_intra` α (hierarchical) per step, both phases | G-b (tp 4, α 1 µs): decode `4.12076 ms → +3 µs` (−0.07%); G-i (tp 16 hierarchical, 8×2): `+ (1·5 + 7·1) = +12 µs` on 2.91067 ms (−0.41%); 4× M3 Ultra TB5 (α 50 µs): `+150 µs` on ≈ 19.3 ms (−0.8%) |
| G2 decode loop-back                 | `send_recv` of `b·(k+1)·4 B` last → first stage, added to `Σ p2p` on the decode latency path                                                                           | `+ α_p2p + 4·b·(k+1)/B` per decode cycle (pp > 1)                                             | G-c (pp 2 intra): `8.41923 ms → +1.0 µs` (−0.01%); G-g (pp 2 inter α 5 µs): `2.43508 ms → +5 µs` (−0.21%); TB5 pp 4: `+50 µs`                                                                       |
| G3 hierarchical a2a α               | `max(α_intra,eff, α_inter,eff)` instead of `α_inter`                                                                                                                   | `+ max(0, α_intra,eff − α_inter)` per all-to-all (2 per mode-B MoE layer)                     | SYNTH_CLUSTER (1 µs < 5 µs): 0; host-bounce PCIe box (13 µs) + IB (1 µs): `+12 µs` per a2a                                                                                                          |
| G4 flat a2a on non-switched fabrics | per-hop α for ring / chain / host-bounce all-to-all (likely delivered by the fabric-shape task's `collectives.ts` changes; otherwise `(n−1)·α` on `host_bounce` links) | `+ (hops − 1)·α` per a2a                                                                      | G-e (ep 2): 0; ep 4 ring α 1.5 µs: `+3 µs` per a2a                                                                                                                                                  |
| G7 overlap-aware `'best'`           | under `overlap 'ideal'` pick by exposed α when β hides                                                                                                                 | `− (2(n−1) − 2⌈log2 n⌉)·α` per all-reduce where ring was chosen by total                      | G-b prefill (ring chosen for S 33.5 MB): `−2 µs × 73 = −146 µs` on 180.2 ms (+0.08%); decode at b 1 already picks tree (no change)                                                                  |
| G8 host-bounce send β               | `nActive = min(devicesPerNode, tp·dp)` for `send_recv`                                                                                                                 | `β_p2p × min(B, hostBW/2) / min(B, hostBW/(2·tp·dp))`                                         | SYNTH_CLUSTER (direct NVLink): 0; host-bounce Gen5 box tp 2: send β `×2` (48 → 24 GB/s cap)                                                                                                         |
| G6                                  | out of scope                                                                                                                                                           | —                                                                                             | —                                                                                                                                                                                                   |

#### Implementation and subsumption findings

- **G1**: `src/sim/lower/head.ts: samplingExchangeOps` is used after target `lowerHead` and draft
  `draft.ts: sharedLmHeadOps` forwards (MTP prefill/decode and block decode); no head means no exchange.
  It applies with or without a final norm. Full logits are not gathered.
- **G2**: `lower/lower.ts: lowerModel` emits `sendToFirst`; `engine/placement.ts: ppSendPath` resolves
  last-stage → stage-0 endpoints; `engine/pipeline.ts: phaseResult, phaseComm` adds return α+β exactly once
  under both overlap modes and return β as a link resource. `PhaseResult.loopBack` exposes its op cost.
  Spec-dec × PP remains rejected; the lowerer nevertheless uses qLen = k+1.
- **G3 not subsumed**: `collectives.ts: hierarchicalAllToAllSec` still used only inter-node α, including
  oversubscribed paths. It now retains the larger effective intra/inter α. The regression proves 13 vs 1 µs
  becomes 13 µs (+12 µs) with unchanged β.
- **G4 partially subsumed**: `fabricCollectives.ts: fabricAllToAllSec` already charges
  `lat(maxHops) = hops·α + (hops−1)·α_f` on ring/chain paths; tests verify this without another multiplier.
  Ring4 has 2 shortest-path hops: at α 1.5 µs its increase over one-α is **1.5 µs**, not the table's 3 µs.
  Chain4 has 3 hops and gives the estimated +3 µs. Flat host-bounce was not covered:
  `collectives.ts: collectiveTime` now uses n−1 serialized peer rounds, so n4 α_eff 13 µs costs 39 µs
  (+26 µs); n2 stays unchanged.
- **G7**: `engine/simulate.ts: simulatePhase` compares three assignments by exact core stage score
  `Σα + max(compute, Σβ)`: per-op min-total (mesh-direct included), uniform ring, uniform tree.
  This bounded heuristic preserves explicit algorithms and no-overlap behavior and never worsens that core
  score, but does not search all mixed assignments or optimize against the offload-transfer floor.
- **G8**: `placement.ts: ppSendPath` sets `sendActiveDevices = min(devicesPerNode, tp·dp)` on host-bounce
  paths; flat and fabric costs consume it. Tests cover TP, DP, node clamping and return β.
- Formulas/code cross-links: `docs/simulation-model.md` §5 (head/PP), §6 (collectives/host bounce), §7
  (stage selection/pipeline). Regression suites: `lower/head.test.ts`, `lower/draft.test.ts`, `engine/collectivesAudit.test.ts`,
  `engine/latencyAudit.test.ts`. Each newly fixed regression failed on the preceding implementation,
  then passed with its fix; already-subsumed ring/chain tests passed before editing.

#### Expected versus measured per fix

Times below are microseconds. Differences below 1e-9 µs are floating-point roundoff.

| Fix | Original expected example                            | Measured                                                                | Explanation                                                                                                                                            |
| --- | ---------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| G1  | G-b decode +3; G-i +12                               | +3; +12                                                                 | Match; β hides under ideal overlap.                                                                                                                    |
| G2  | G-c decode ≈+1; G-g ≈+5                              | G-c **0**; G-g **+5.00008**                                             | G-c is resource-bound (`m·b` wins), hiding the return. G-g includes 4/50 GB/s β. Original G-c estimate omitted the pipeline max, not a missing return. |
| G3  | SYNTH_CLUSTER 0; host-bounce 13 vs inter 1 ⇒ +12/a2a | 0 in goldens; +12/a2a unit test                                         | Match.                                                                                                                                                 |
| G4  | G-e 0; ring4 α1.5 ⇒ +3/a2a                           | G-e 0; ring already +1.5 over one-α; chain already +3; host-bounce4 +26 | Ring/chain subsumed; ring example assumed too many hops. Host-bounce gap fixed here.                                                                   |
| G7  | G-b prefill −146; decode 0                           | −146; 0                                                                 | Match. G-g also changes −288 cycle / −584 TTFT.                                                                                                        |
| G8  | Direct links 0; tp2 host-bounce send β ×2            | 0 in goldens; β ×2 (48→24 GB/s), DP2 ×4, node-clamped ×4                | Match; forward and return both share host bandwidth.                                                                                                   |

Every §8.3 row, per-fix measured **Δ decode cycle / prefill cycle / prefill TTFT**, in µs.
`0` means all three unchanged. Contexts P/N are those in §8.3.

| Golden | G1                  | G2               | G3  | G4  | G7              | G8  |
| ------ | ------------------- | ---------------- | --- | --- | --------------- | --- |
| G-a    | 0                   | 0                | 0   | 0   | 0               | 0   |
| G-b    | +3 / +3 / +3        | 0                | 0   | 0   | 0 / −146 / −146 | 0   |
| G-c    | +2 / +2 / +2        | 0                | 0   | 0   | 0               | 0   |
| G-d    | 0                   | 0                | 0   | 0   | 0               | 0   |
| G-e    | 0                   | 0                | 0   | 0   | 0               | 0   |
| G-f    | +0.906424 / +2 / +2 | +1.00004 / 0 / 0 | 0   | 0   | 0               | 0   |
| G-g    | +7 / +7 / +7        | +5.00008 / 0 / 0 | 0   | 0   | 0 / −288 / −584 | 0   |
| G-i    | +12 / +12 / +12     | 0                | 0   | 0   | 0               | 0   |
| G-h1   | 0                   | 0                | 0   | 0   | 0               | 0   |
| G-h2   | 0                   | 0                | 0   | 0   | 0               | 0   |

G1's per-head α is 1 µs in G-c/G-f; the resource term multiplies a stage increase by m = 2.
G-f decode switches branches, yielding +0.906424 rather than +1 µs. G-g G7 saves 8 µs per AR:
36 AR on stage 0 save 288 µs (cycle bottleneck), 37 AR on stage 1 save 296 µs.
With m = 1, TTFT saves their sum 584 µs.

Additional identity variants: G-a/overlap-none stays identical throughout. G-c/overlap-none G1:
`+2.00016 / 0 / +1.00008` µs; G-f/overlap-none G1: `0 / 0 / +1.00008` µs.
Their later fixes leave cycle/TTFT unchanged (resource branch wins); return α/β still enters comm.

Changed leaves in the full identity fixture by fix: G1 **1280**, G2 **64**, G3 **0**, G4 **0**,
G7 **139**, G8 **0**. Snapshots were retained in the implementer's scratchpad per comparison.
The separate `fabric-identity.json` pin was deliberately regenerated with `UPDATE_FABRIC_IDENTITY=1`:
96 collective leaves (host-bounce a2a α/total), 1477 simulation leaves, 120 UI leaves, 36 historical
`uiDeliberate` leaves changed. The last category also compares pre-fabric cabling defaults, so those combined
deltas are not solely this audit. Example unchanged-cabling deltas: RTX5090 TP2 decode/prefill cycle +13 µs
(sampling); RTX5090 PP2 decode +13.000083333334 µs (return). Regeneration additionally includes 84 cases
for the already-existing TB5 TCP link absent from the old pin (no preset values changed). A final regeneration
after adding draft-head coverage reproduced both knobs-off fixtures exactly.
The Spark fabric smoke test now expects slightly faster switched prefill (6988 vs ring 6987 tok/s @ 8k):
G7 can use lower-α tree when β hides; the previous equality assertion is no longer valid.

Magnitude today (audit probe, REF_DENSE_8B bf16 TP4 b 1 @1024): α is 3% of the step on 4× DGX Spark and **76%** on
4× M3 Ultra over TB5, so G1/G2 matter for TB/USB/Ethernet clusters and are invisible on NVLink.

---

## 5. Researched defaults

Conventions: τ = mean tokens committed per verify pass including the bonus token; α_iid = the constant per-position
acceptance reproducing a reported τ at the same k under §3.5 (always **derived**); "R" reported, "D" derived, "I"
interpolated. Sources disagree on DSpark vs DFlash 2 (DeepSeek's paper, non-thinking k = 7, ranks DSpark highest;
Inco's matched retrain, thinking, T = 1, k = 15, ranks DFlash 2 ≈ +0.5 τ above DSpark); a default either reproduces
named runs (their k are the fitted k) or is pooled between runs and fitted at no k. The per-entry provenance (every τ with its source, method version, target, k and temperature, and
whether the entry is measured, borrowed, pooled or carried over) is in `src/sim/engine/specdec.ts: ACCEPTANCE_PRESETS` and
the table of [simulation-model.md §13 "Acceptance defaults"](../simulation-model.md#acceptance-defaults), which
supersede the summary below where they differ (practicality review §1 rows 542/543, F10).

### 5.1 Acceptance α per method × use case (T ≈ 1 sampling, chain)

| method                                                                            | default k                             | prose / chat | code (generation) | math     | confidence                                                        | evidence (τ → α_iid)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------------- | ------------------------------------- | ------------ | ----------------- | -------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MTP, multi-step head (Qwen3.5+/3.8, MiMo 3-layer, FastMTP-style)                  | 3                                     | **0.75**     | **0.82**          | **0.85** | medium (prose, math); low (code: .76–.88, mixes greedy and T = 1) | Qwen3.8-27B card k=7 T=1 xhigh: MT-Bench 3.74 / HumanEval 3.91, MBPP 3.99 / GSM8K 5.02, MATH 4.72 → .76 / .78–.79 / .84–.86 https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 ; Qwen3.5-4B blog Table 3 (thinking, T=1) k=15: .75 / .76–.80 / .80–.81 https://inco.ai/blog/dflash2/ ; llama.cpp PR 27210 Qwen3.8-27B greedy k=3 (C1, τ = 1 + 3 × draft-acceptance ratio): prose 2.67 (.735), code 3.35 (.883), reasoning 2.47 (.684) https://github.com/ggml-org/llama.cpp/pull/27210#issuecomment-5385811671 ; PR 22673 reports accepted/drafted ratios, not τ, and is not used https://github.com/ggml-org/llama.cpp/pull/22673                                                                                          |
| MTP, single-step head (DeepSeek-V3/R1/V4 `num_nextn_predict_layers 1`, GLM-4.5/5) | 1                                     | **0.85**     | 0.85 (I)          | 0.87 (I) | high for k=1 overall; low per use case                            | DeepSeek-V3 report §5.4.3: 85–90% "across various generation topics" (R) https://arxiv.org/abs/2412.19437 ; FastMTP Table 1 MiMo vanilla K=1: MT 1.72 / code 1.73 / math 1.83 → .72/.73/.83 https://arxiv.org/abs/2509.18362 ; SGLang #3582 "~1.9 accept length for 2 draft tokens" (k ambiguous) https://github.com/sgl-project/sglang/pull/3582 ; k ≥ 2 tail α₂ ≈ 0.10 (vanilla) – 0.55 (fine-tuned)                                                                                                                                                                                                                                                                                                                     |
| DFlash 2                                                                          | 7 (block 8); 15 for block-16 drafters | **0.79**     | **0.83**          | **0.87** | medium                                                            | Qwen3.8-27B card k=7: MT 4.10 / HE 4.39, MBPP 4.79 / GSM8K 5.46, MATH 5.28 → .80 / .82–.85 / .88–.89; Muse-Glimmer-30B card k=15: .78 / .82–.83 / .86 https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2 ; Qwen3.5-4B blog Table 3 (thinking, T=1) k=15: .82 / .82–.85 / .85–.87; per-position acceptance flat (blog Fig. 5: 88.3, 85.3, 85.0 … 86.5%) so i.i.d. fits                                                                                                                                                                                                                                                                                                                                                  |
| DSpark                                                                            | 7 (paper); 5 (DeepSeek-V4 production) | **0.76**     | **0.84**          | **0.88** | low–medium                                                        | DSpark paper Table 1 (DeepSeek-trained Qwen3-4/8/14B, k=7, T=1, non-thinking): MT 3.64–3.72 / HE 5.38–5.52, MBPP 5.13–5.26 / GSM8K 6.11–6.21, MATH 5.70–5.78 → .75–.76 / .87–.89 / .90–.93 https://arxiv.org/abs/2607.05147 ; Inco-trained Qwen3.5-4B (blog Table 3, thinking, T=1) k=15: .80 / .81–.84 / .83–.85. Prose 0.76 is fitted to the paper alone (τ 3.69 at k 7); code and math are pooled between the paper and Inco's retrain. Not fitted (community retrains, lower): RadixArk Qwen3.8-27B thinking k=7 (card): MT 3.01 (.68), HE 3.30, MBPP 3.51 (.72–.74), GSM8K 4.36, MATH 3.92 (.78–.82); DaoCloud Muse-Glimmer-30B k=15. The production gain (+57–85% per user) comes from the scheduler we do not model |

Extras (low confidence, offered as such): **reasoning trace (thinking on)**: MTP 0.68 (PR 27210 greedy τ 2.47 at k 3),
DFlash 2 / DSpark = prose + 0.02 (I). **Agentic code / editing**: DFlash 2 0.65 is **DFlash v1** data, borrowed as a
proxy (DDTree arXiv 2604.12989 Table 1: Qwen3-8B, z-lab DFlash v1, SWE-bench Lite τ 2.82 at k=15, T=1; no DFlash 2
agentic measurement exists), others = prose (I). "Code" = code generation (HumanEval/MBPP class); MBPP usually
below HumanEval; T = 1 lowers τ by 5–15% vs greedy. Reference methods not in the picker: DFlash v1 (0.74 / 0.82 /
0.85, arXiv 2602.06036 Table 1), EAGLE-3 chain (0.63 / 0.81 / 0.88, DSpark Table 1 retrain).

### 5.2 Expert hit ratio h vs resident fraction f (decode, batch 1, LRU-class cache, steady state)

| f                               | 0   | 0.1                     | 0.125     | 0.25                    | 0.5             | 0.75      | 1   |
| ------------------------------- | --- | ----------------------- | --------- | ----------------------- | --------------- | --------- | --- |
| `uniform` h = f (exact)         | 0   | 0.10                    | 0.125     | 0.25                    | 0.50            | 0.75      | 1   |
| `skewed_classic` h = f^0.55     | 0   | 0.28 (I)                | 0.32      | 0.47                    | 0.68            | 0.85      | 1   |
| data, E ≤ 64                    |     |                         | 0.04–0.39 | 0.30–0.55               | 0.57–0.78       | 0.80–0.93 |     |
| `skewed_fine_grained` h = f^0.2 | 0   | 0.63                    | 0.66      | 0.76                    | 0.87 (I)        | 0.94 (I)  | 1   |
| data, E ≥ 128                   |     | 0.67–0.77 (f 0.10–0.16) | 0.67      | 0.71–0.81; 0.89 at 0.28 | > 0.90 at 0.625 | —         |     |

Sources: Mixtral-8x7B (8, 2) 0.37/0.60/0.82 at f 0.25/0.5/0.75 (Skliar et al. arXiv 2412.00099 Fig. 11; figure-read
±0.03), 0.31/0.57/0.80 (ProMoE arXiv 2410.22134 Fig. 5b), 0.302/≈0.58/0.806 (HybriMoE arXiv 2504.05897 §VI-D),
≈0.35/0.61 (Eliseev & Mazur arXiv 2312.17238 Fig. 2); Phi-3.5-MoE (16, 2) 0.55/0.78/0.93; DeepSeek-V2-Lite (64, 6)
0.50/0.72/0.89, 0.477/≈0.715/≈0.885; Qwen1.5-MoE (60, 4) 0.45/0.65/0.85; Qwen2-57B (64, 8) 0.450/≈0.715/≈0.90.
Fine-grained: Qwen3.8-Flash-Next (512, 10) 0.67 at LRU-64, 0.81 at LRU-128 (https://github.com/ggml-org/llama.cpp/pull/27861 ),
0.72–0.77 at f ≈ 0.10–0.16 and 0.89 at 0.28 in-system (https://github.com/ggml-org/llama.cpp/pull/29887 , merged);
GLM-5.2 (256) 0.73 at 0.25, 0.79 at 0.35 (https://github.com/ggml-org/llama.cpp/pull/25294 ); Qwen3-30B-A3B (128, 8)
"> 90%" at 0.625 (https://github.com/ggml-org/llama.cpp/pull/23440 ). Static pinning gains 3–5 pp over random (Fiddler
arXiv 2402.07033 App. C; PR 27861). Not found: an LRU curve for DeepSeek-V3/R1. Confidence: γ = 1 exact; γ = 0.55
medium; γ = 0.2 low (2026 developer measurements, temp 0, short runs, AI-assisted PRs; gpt-oss reported flat).

---

## 6. UI surface (design only)

Project UI rules: every picker is a shadcn Popover + Command (cmdk) picker; lists of more than ~8 items get the
fuzzy-search box; every non-obvious input has a "?" help icon; scroll areas use shadcn ScrollArea; no prose under
headings/results; standard terms ("compute bound", "bandwidth bound", "latency bound").

### 6.1 Workload → "Speculative decoding"

| knob           | control                                                                                   | default                                                                                | range                                                                                              | "?" help                                                                                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Method         | picker: Off · MTP (model's head) · DFlash 2 · DSpark                                      | Off                                                                                    | MTP only when `model.draft` exists; at pp > 1 the picker offers only "Off" (the reason in its "?") | "Draft-then-verify decoding. Only the extra memory and compute/bandwidth of the draft and verify passes are modelled, with the acceptance ratio below; no engine overhead. Not available with pipeline parallelism yet." |
| Drafter        | picker: repo id (text with validation) · Hypothetical (5 layers, 32/8 heads, window 2048) | repo id when known for the target, else Hypothetical (badge "hypothetical" in results) | DFlash 2 / DSpark only                                                                             | "The drafter's architecture from its Hugging Face config.json. 'Hypothetical' is a stand-in with fixed defaults, not a released model."                                                                                  |
| Draft length k | number                                                                                    | per method (§5.1); 1 for single-step MTP                                               | 1–16                                                                                               | "Tokens drafted per step; the verify pass runs the model over k+1 tokens. Expected accepted tokens per step = (1−α^(k+1))/(1−α)." (warning icon when k > the head's reliable depth)                                      |
| Use case       | picker: prose · code · math · reasoning · agentic code                                    | prose                                                                                  |                                                                                                    | "Pre-fills the acceptance ratio from published measurements for this method. Code = code generation."                                                                                                                    |
| Acceptance α   | number, 2 decimals                                                                        | from table                                                                             | 0.00–0.99                                                                                          | "Per-token probability that the target accepts a drafted token, assumed independent per position; measured at ~temperature 1 on short-context benchmarks." Editing sets the use case to "custom".                        |

### 6.2 Hardware → "Offload" (dedicated memory) / "Storage" (unified)

| knob                         | control                                                                                                                                                                  | default                                                                           | range          | "?" help                                                                                                                                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Routed experts               | picker: Device · Host RAM · SSD                                                                                                                                          | Device                                                                            | MoE models     | "Where the routed expert weights live. Misses stream to the GPU over PCIe at the hardware ceiling; the GPU computes."                                                                                                                             |
| Offloaded layers             | number "from layer"                                                                                                                                                      | 0 (all)                                                                           | 0–L            | "Experts of layers ≥ this index are offloaded (llama.cpp --n-cpu-moe keeps the first layers on the device)."                                                                                                                                      |
| Resident fraction f          | number field, % (as the other numeric knobs)                                                                                                                             | 0%                                                                                | 0–100          | "Share of each offloaded layer's experts kept in device memory (pinned or cache), counted in the memory fit."                                                                                                                                     |
| Hit-ratio model              | picker: Uniform (h = f) · Skewed, ≤ 64 experts · Skewed, ≥ 128 experts · Custom                                                                                          | Uniform                                                                           |                | "Probability that a needed expert is already on the device, per unique expert per layer per step at batch 1. Uniform is exact for balanced routing; skewed presets fit measured LRU locality (h = f^0.55, f^0.2). Bounded by the resident count." |
| Hit ratio h                  | number                                                                                                                                                                   | from model                                                                        | 0–1            | shown when Custom, else read-only                                                                                                                                                                                                                 |
| Queue depth                  | number (Advanced)                                                                                                                                                        | 0 = the tier's rated outstanding requests                                         | 0–tier max     | "Outstanding read requests the deployment keeps in flight; 1 serializes every request behind its latency."                                                                                                                                        |
| Dense layers on device       | number                                                                                                                                                                   | all                                                                               | 0–L            | dense models: "Layers beyond this count are streamed from the tier every step."                                                                                                                                                                   |
| Lookup tables (Advanced)     | picker: Device · Host RAM · SSD; a "Host RAM ↗" chip button next to the picker (not inside the trigger: no nested buttons) when the model's table is designed off-device | Device (hint state for PLE/Engram; pre-filled Host RAM once offload is turned on) | n-gram models  | "Where the hashed n-gram tables live. Only the rows a token needs are gathered: ~1 µs per step from RAM, latency-bound from SSD. Counted against that tier's capacity. This model ships its table to be kept off-device."                         |
| Host RAM                     | picker with fuzzy search (6 presets) + capacity                                                                                                                          | DDR5-6000 ×2 (96 GB)                                                              |                | "Host memory: theoretical channel bandwidth and capacity (datasheet values); shared by every GPU of the node."                                                                                                                                    |
| SSD                          | picker with fuzzy search (8 presets)                                                                                                                                     | Samsung 990 PRO 2TB; on unified memory the device's own drive (§2.2)              |                | "NVMe drive: sequential read, QD1 latency and max random-read IOPS from the datasheet; shared by every GPU of the node."                                                                                                                          |
| Host link                    | read-only from the device preset ("PCIe 5.0 x16, 63.0 GB/s") + picker to override gen/lanes                                                                              | from `hostInterface`                                                              | Gen3–5, x4–x16 | "GPU-to-host PCIe link per direction; with host RAM bandwidth, the streaming ceiling."                                                                                                                                                            |
| Assumptions: Offload overlap | picker ideal / none (as the other Assumptions)                                                                                                                           | ideal                                                                             |                | "ideal: the next layer's transfers overlap this layer's compute (prefetch, perfect prediction); none: fetched on demand, serialized."                                                                                                             |

### 6.3 Results

- Rate line: `decode 183.1 tok/s @ 8k (spec-dec ×1.67)`; with offload a chip `experts: host RAM, h 0.50`; the "?" on
  the rate explains E[τ] per step.
- Breakdown rows: existing op rows plus **offload transfer**, **comm latency**, **comm bandwidth**; a spec-dec decode
  adds the layer groups **verify** and **draft**.
- Memory card: the device bar (with the hatched "n-gram table" segment when a table is on the device) plus one bar
  per tier ("Host RAM 16.3 / 96 GB", "SSD 102 / 2000 GB"); the model summary lists "n-gram tables 51.2 B".
- Charts keep hover/focus/tap tooltips with units; no captions.

---

## 7. Type sketches (plain data, discriminated unions, JSDoc with units)

### 7.1 IR (`src/sim/ir/ops.ts`, `src/sim/ir/model.ts`)

```ts
/**
 * A read of weight bytes from an off-device tier into this device's memory, in front of the op that needs them.
 * Costed by the tier/link model (engine/transfer.ts, design doc §2.3); moves no HBM bytes in this model (the DMA
 * write into device memory, ≤ B_link/memBW, is a documented approximation) and has no FLOP. Per device, per
 * microbatch step; all quantities are EXPECTED values (real numbers). Emitted only when `bytes > 0`.
 */
export interface TransferOp {
  kind: 'transfer';
  /** Stable dotted name, e.g. `'moe.expert_fetch'`, `'layer.fetch'`, `'ngram.row_fetch'`. */
  name: string;
  /** What is streamed (breakdown row and tier accounting). */
  group: 'routed_experts' | 'dense_layer' | 'lookup_table'; // as built: 'dense_layer' removed (§2.11)
  /** Tier the bytes come from (`'host_ram'` on a unified-memory device is never emitted). */
  source: 'host_ram' | 'nvme';
  /**
   * Physical read classes issued this step: `count` reads of `bytesEach` bytes each (before granularity rounding).
   * Experts: `[{ M, W_expert }]`; dense layer: `[{ 1, W_layer }]`; table rows per §2.10's scale layout, e.g. separate
   * value and scale reads `[{ T·g, 256 }, { T·g, 8 }]`. `Σ count` pays the IOPS and Little's-law terms; each class is
   * rounded to the tier's granularity for the tier-bandwidth term. Counts are expected values (real). Unit: bytes.
   */
  reads: { count: number; bytesEach: number }[];
  /** Useful payload crossing the host link (independent of the read layout: 264 B per Engram row). Unit: bytes. */
  usefulBytes: number;
}
export type Op = GemmOp | AttentionOp | IndexerOp | RecurrentOp | ElementwiseOp | EmbeddingOp | CollectiveOp | TransferOp;

/** Added to AttentionOp: tokens whose K/V are appended to the cache this step (default qLen). Block drafters keep only the m_ctx context rows. Unit: tokens. */
  kvWriteTokens?: number;
/** Added to ElementwiseOp: KV-cache bytes written by a write-only cache op (`fn: 'kv_cache_write'`, §3.4); default 0. Unit: bytes. */
  kvWriteBytes?: number;
/** Added to RecurrentOp: state snapshots written this step (verify of k+1 tokens: k+1; default 1). Multiplies the state WRITE bytes. Count. */
  stateSnapshots?: number;
/** Added to NgramEmbeddingBlock: the checkpoint is designed to keep this table off-device (Qwen PLE, DeepSeek Engram); drives the UI hint/prefill only. */
  offDeviceByDesign?: boolean;
/** Added to ModelSpec: the checkpoint's own draft head(s); costed only when spec-dec is on. */
  draft?: DraftSpec;

/** A speculative-decoding drafter. Dims logical (unsharded). */
export type DraftSpec =
  | {
      kind: 'mtp';
      /** The MTP module's decoder layer(s), one target-shaped layer per module. */
      layers: LayerSpec[];
      /** True if the module projects `[norm(h_prev); norm(emb)]` (2h → h) before its layer (`eh_proj`). */
      hiddenProj: boolean;
      /** Draft positions the head was trained for; `k` beyond it collapses (warning + tail acceptance). Absent = any. Count. */
      maxReliableDepth?: number;
    }
  | {
      kind: 'block_drafter';
      /** Extras to cost: DFlash 2 (dynamic convs + candidate selector) or DSpark (Markov head). */
      method: 'dflash2' | 'dspark';
      /** Drafter width h (= the target's, since embedding and LM head are shared). Unit: elements. */
      hidden: number;
      /** Shared vocabulary V (selector codebooks and Markov matrices are V×rank). Count. */
      vocab: number;
      /** Softmax-attention layers (`num_hidden_layers`; released drafters: 5). Count. */
      layers: number;
      /** Query heads / KV heads / head dim of the DRAFTER's attention (not the target's). Count / count / elements. */
      heads: number; kvHeads: number; headDim: number;
      /** Gated MLP width of the drafter. Unit: elements. */
      intermediate: number;
      gated: boolean; activation: 'silu' | 'gelu' | 'relu';
      /** Sliding window of the drafter's attention over its context K/V (`sliding_window` when `layer_types` are sliding); absent = full. Unit: tokens. */
      windowTokens?: number;
      /** Target layers whose hidden states are concatenated into `fc`'s input (`len · h → h`). Zero-based indices. */
      targetLayerIds: number[];
      /** Block positions per draft pass = k + 1 (anchor + masks); the UI's k must satisfy k + 1 ≤ blockSize. Count. */
      blockSize: number;
      /** DFlash 2 `GroupedDynamicCausalConv` (two per layer): taps K and channels per group g. */
      conv?: { kernel: number; groupSize: number };
      /** DFlash 2 `CandidateSelector`: candidates per position (16) and codebook rank r (256). */
      selector?: { topK: number; rank: number };
      /** DSpark Markov head rank r (`W1: V×r`, `W2: r×V`). */
      markovRank?: number;
      /** DeepSeek-V4-style MoE drafter layers (used instead of the dense fields above when present). */
      moeLayers?: LayerSpec[];
      /** Where the shape came from: a repo id + revision, or `'hypothetical'`. */
      source: string;
    };
```

### 7.2 Hardware (`src/sim/hw/types.ts`)

```ts
/** Host DRAM as a storage tier. ONE per node, shared by every device of the node. */
export interface HostMemorySpec {
  kind: 'host_ram';
  id: string; name: string;
  /** Installed capacity. Unit: bytes. */
  capacityBytes: number;
  /** Theoretical peak, read + write combined: MT/s × 8 B × channels. Unit: B/s. */
  bandwidthBytesPerSec: number;
  /** Unloaded load-to-use latency (the α of a random read). Unit: seconds. */
  latencySec: number;
  /** Memory channels (informational). Count. */
  channels: number;
  /** Smallest unit a random read moves (a cache line, 64). Unit: bytes. */
  readGranularityBytes: number;
  /** Random reads the host keeps in flight (memory-level parallelism; the QD cap of Little's law). Count. */
  maxOutstanding: number;
  sources: SourceRef[]; estimated: EstimatedField[];
}
/** An NVMe SSD as a storage tier. ONE per node, shared. */
export interface NvmeSpec {
  kind: 'nvme';
  id: string; name: string;
  /** Usable capacity. Unit: bytes. */
  capacityBytes: number;
  /** Datasheet sequential read (large blocks, deep queue). Unit: B/s. */
  seqReadBandwidthBytesPerSec: number;
  /** Random 4 KiB read latency at QD1 (1 / QD1 IOPS when only IOPS is published). Unit: seconds. */
  randomReadLatencySec: number;
  /**
   * Datasheet max random-read rate and its measurement condition. OPTIONAL: when absent the IOPS term is dropped and
   * the request rate is bounded by Little's law alone (`maxOutstanding / randomReadLatencySec`), with a warning.
   * Unit: requests/s; counts.
   */
  randomRead?: { iops: number; queueDepth: number; threads: number };
  /** Outstanding requests the drive accepts (queueDepth · threads of the rated condition, or an explicit estimate in `estimated[]`). Required. Count. */
  maxOutstanding: number;
  /** Smallest unit a read moves (LBA/page: 4096); a 320 B row costs this much. Unit: bytes. */
  readGranularityBytes: number;
  /** The drive's own slot ceiling when it binds below the datasheet rate (Gen4 x4 7.88e9). Unit: B/s. */
  linkBandwidthBytesPerSec?: number;
  sources: SourceRef[]; estimated: EstimatedField[];
}
export type StorageTierSpec = HostMemorySpec | NvmeSpec;
/** The host side of a node. `link` is PER DEVICE; `ram` and `ssd` are shared by the node's devices. */
export interface HostSpec { link?: LinkSpec; ram?: HostMemorySpec; ssd?: NvmeSpec; }
/** Added to DeviceSpec (cited): the card's PCIe edge connector, used to build `HostSpec.link` when absent. */
  hostInterface?: { generation: 3 | 4 | 5; lanes: 4 | 8 | 16 };
/** Added to HardwareStack. */
  host?: HostSpec;
/** Hook (§2.8): validated, not costed in the first implementation. */
export interface HostComputeSpec { matmulPeaks: MatmulPeak[]; vectorFlopsPerSec: Partial<Record<'fp32' | 'bf16', number>>; sources: SourceRef[]; estimated: EstimatedField[]; }
```

### 7.3 Engine inputs (`src/sim/engine/types.ts`)

```ts
export interface SimOptions {
  overlap: 'ideal' | 'none';
  elementwiseFusion: 'none' | 'ideal';
  collectiveAlgorithm: 'best' | 'ring' | 'tree';
  /**
   * Overlap of off-device transfers with compute. `'ideal'` (default): the next layer's transfer hides behind this
   * layer's compute, `t_s = lat_s + max(core_s, xfer_s)`; `'none'`: `t_s = lat_s + core_s + xfer_s`. Prediction
   * accuracy is not modelled. No effect without transfers.
   */
  offloadOverlap: 'ideal' | 'none';
}
/** Deployment-scenario knobs; every field optional and off by default (absent or {} ⇒ today's results, bit for bit). */
export interface DeploymentConfig {
  offload?: OffloadConfig;
  speculative?: SpecDecodeConfig;
}
export interface SimInput {
  /* …as today… */ deployment?: DeploymentConfig;
}

export interface OffloadConfig {
  /** `'transfer'` (default): misses stream to the device; `'cpu_compute'`: designed hook, rejected by validation for now. */
  mode?: 'transfer' | 'cpu_compute';
  /** Outstanding read requests kept in flight per device; default and cap: the tier's `maxOutstanding`. 1 serializes. Count. */
  queueDepth?: number;
  // As built (§2.11): routedExperts / lookupTables are `Rule | readonly Rule[]` (RoutedExpertRule, LookupTableRule in
  // engine/types.ts), disjoint layer ranges; the single object below is the one-rule list. denseLayers was removed.
  routedExperts?: {
    tier: 'host_ram' | 'nvme';
    /** Layers whose experts are offloaded, `[from, to)` over `model.layers`; absent = all MoE layers. */
    layers?: { from: number; to: number };
    /** Fraction f of each offloaded layer's experts kept in device memory, in [0, 1]. Dimensionless. */
    residentFraction: number;
    /** Hit-ratio model h(f) (§2.4); `custom` gives h per unique expert per layer per step at batch 1; clamped to the physical bounds. */
    hitRatio:
      | { kind: 'uniform' }
      | { kind: 'skewed_classic' }
      | { kind: 'skewed_fine_grained' }
      | { kind: 'custom'; h: number };
    /** Also offload the drafter's own MoE experts (DeepSeek-V4 DSpark); default false. */
    includeDraft?: boolean;
  };
  /** Whole decoder layers `[from, to)` streamed every step. */
  denseLayers?: { tier: 'host_ram' | 'nvme'; layers: { from: number; to: number } };
  /**
   * n-gram table placement; absent = `'device'` (identity). `scaleLayout` (block-scaled tables only, §2.10): default
   * `'separate'` on host RAM, `'pinned_host'` on NVMe (scales counted in the host-RAM tier; on unified memory in the device total), `'interleaved'` for a repacked table.
   */
  lookupTables?: {
    tier: 'device' | 'host_ram' | 'nvme';
    scaleLayout?: 'separate' | 'pinned_host' | 'interleaved';
  };
}
export interface SpecDecodeConfig {
  /** `'mtp'` needs `model.draft.kind === 'mtp'`; block methods need `drafter`. Rejected when pp > 1. */
  method: 'mtp' | 'dflash2' | 'dspark';
  /** The block drafter: read from a repo, or the labelled hypothetical. Ignored for MTP. */
  drafter?:
    | { kind: 'repo'; repoId: string; spec: Extract<DraftSpec, { kind: 'block_drafter' }> } // the loader parsed the repo's config.json
    | { kind: 'hypothetical'; spec: Extract<DraftSpec, { kind: 'block_drafter' }> };
  /** Draft tokens per step k (verify width k + 1; ≤ blockSize − 1 for block drafters). Count. */
  draftLength: number;
  /**
   * Per-token acceptance. `iid`: α in [0, 1); α = 1 (every draft accepted) is expressed through the per-position form,
   * whose α_j lie in [0, 1] (α_j = 1 is a legal input: E[τ] = k + 1 is the limit of the i.i.d. formula). Dimensionless.
   */
  acceptance: { kind: 'iid'; alpha: number } | { kind: 'per_position'; alphas: number[] };
}
```

### 7.4 Results (`src/sim/engine/types.ts`)

```ts
export type Bound = 'compute' | 'memory' | 'comm' | 'transfer';
export interface OpResult {
  /* …as today… */
  /** Off-device read latency terms α_tier + α_link; 0 for other ops. Unit: seconds. */
  transferLatencySec: number;
  /** Off-device service term (max of Little's-law, IOPS, tier-, host- and link-bandwidth terms); 0 for other ops. Unit: seconds. */
  transferBandwidthSec: number;
}
export interface StageResult {
  /* …as today… */
  /** Σ transfer op times (latency + service). Unit: seconds. */
  transferSec: number;
  /** Transfer time not hidden under `offloadOverlap`: t_s − (exposed_s + core_s). Unit: seconds. */
  exposedTransferSec: number;
}
export interface PhaseResult {
  /* …as today… */
  /** Communication on the critical path: Σ α (incl. PP sends), Σ β, and the part not hidden. Sums of existing numbers. Unit: seconds. */
  comm: { latencySec: number; bandwidthSec: number; exposedSec: number };
  /** Off-device transfers on the critical path; absent when nothing is offloaded. */
  transfer?: { sec: number; exposedSec: number; bytes: number; requests: number };
  /** Decode with spec-dec: draft and verify shares of the step, E[τ], and the speedup over spec-dec off. */
  speculative?: {
    draftSec: number;
    verifySec: number;
    expectedTokensPerStep: number;
    speedup: number;
  };
}
export interface TierMemory {
  tier: 'host_ram' | 'nvme';
  /** Bytes placed on the tier by the node's devices (full groups, not the device cache). Unit: bytes. */
  bytes: number;
  capacityBytes: number;
  fits: boolean;
}
export interface DeviceMemory {
  /* …as today… */
  /** Staging buffer(s) for streamed groups (double-buffered under ideal overlap; 0 when nothing streams). Unit: bytes. */
  stagingBytes: number;
  /** Draft weights and draft KV (spec-dec on). Unit: bytes. */
  draftBytes: number;
  /** Per-position recurrent/conv state snapshots of a k+1-token verify, × B_r (spec-dec on, hybrid models). Unit: bytes. */
  speculativeTransientBytes: number;
}
export interface MemoryReport {
  /* …as today… */ tiers: TierMemory[];
}
```

---

## 8. Identity, tests, goldens, worked examples

### 8.1 Why defaults-off is identical

- `deployment` absent or `{}` ⇒ `LoweringContext.offload` and `.draft` undefined ⇒ `lowerMoe`, `lowerNgram`,
  `lowerBlock` emit no `TransferOp`; `lowerModel` appends no `draft.*` layers; `qLen = 1` in decode; no
  `kvWriteTokens` / `stateSnapshots` overrides.
- `xfer_s = 0` ⇒ both `offloadOverlap` branches equal today's `t_s`; `PhaseResult.comm/transfer/speculative` are
  derived from existing sums (no change to `cycleSec`, `tokensPerSec`); `E[τ] = 1`.
- Memory: `staging = draftBytes = speculativeTransientBytes = 0`, `tiers = []`, lookup tables at `'device'` ⇒
  `fits` and every byte count unchanged, including PLE/Engram models.
- `hardware.host` absent and nothing off-device ⇒ never read. The audit corrections now apply equally with
  deployment absent or empty; identity means deployment-off equivalence against the updated §8.3 baseline.

### 8.2 Test plan (vitest, co-located; hand-derived numbers in `it` titles and comments; `expectClose` rel 1e-9 for

exact formulas; bands for engine totals)

| id      | file                              | what                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ID-1    | `engine/identity.test.ts`         | the goldens of §8.3 (decode/prefill `cycleSec`, `tokensPerSec`, `latencySec`, per-stage `totalBytes`, `fits`) reproduced to rel 1e-9 by `simulate()` with `deployment` absent, with `deployment: {}`, and with `options: { offloadOverlap: 'ideal' }`; deep-equal of the three results                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ID-2    | same                              | every `reference.ts` model and every HF fixture on `SYNTH_GPU` (tp 1) and `SYNTH_CLUSTER` (tp 2, pp 2, ep 2 dp 2 where valid): result with `deployment` absent deep-equals `deployment: {}` and `lookupTables: { tier: 'device' }`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| U-1     | `engine/transfer.test.ts`         | `uniqueExperts(E, k, T)`: `U(1) = k`; E 128 k 8: `U(2) = 15.5`, `U(4) = 29.123046875`, `U(8) = 51.61990735…`, `U(16) = 82.4225113…`, `U(32) = 111.771035…`, `U(8192) = 128`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| U-2     | same                              | hits/misses with E 128: `(U 8, h .5, f .5) → hits 4, M 4`; `(U 128, h .87, f .5) → hits 64`; `(U 128, h .1, f .5) → hits 64` (lower bound); `(U 8, h .1, f .5) → hits 0.8, M 7.2` (custom below f allowed); `(f 0, any h ≤ 0) → M = U`; `(f 1, h .3) → M 0`; `(U 100, h .2, f .5) → lo = 36, hits 36`                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| U-3     | same                              | validation: `f 0, custom h .2` rejected; `f 1` with any h accepted and no TransferOp emitted; hit models at f .25: `.25 / .25^.55 = .46651… / .25^.2 = .75785…`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| U-4     | same                              | `transferTime` host RAM → PCIe5: reads `[{8, 2,654,208}]`: `α 1.07e-6`, `service = 21,233,664/63,015,384,615.38 = 3.36960e-4` (link-bound; DRAM term 2.2120e-4, Little 7.66e-9); NVMe → host → device, reads `[{16, 320}]`, QD 512, n 1: `bytes_tier 65,536`, `usefulBytes 5,120`, `α 46e-6`, `service = max(15·45e-6/512 = 1.318e-6, 16/1.4e6 = 1.1429e-5, 65,536/7.45e9 = 8.797e-6, 65,536/48e9 = 1.365e-6, 5,120/63.0e9 = 8.1e-8) = 1.1429e-5`; an `NvmeSpec` without `randomRead` (Spark-like, `maxOutstanding 512`, lat 50e-6): the IOPS term is absent, `service = max(15·50e-6/512, 65,536/12.2e9, …)` and a warning is emitted                                                                                                                           |
| U-5     | `engine/pipeline.test.ts`         | stage formula: `compute 2, bw 1, lat 0.5, xfer 3` ⇒ ideal/ideal 3.5, ideal/none 5.5, none/ideal 3.5 + p2p, none/none 6.5 + p2p; `xfer 0` ⇒ today's values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| U-6     | `engine/memory.test.ts`           | TINY_HYBRID experts on host, f .5: device `W` drops by half the expert bytes; `tiers[host_ram].bytes` = all expert bytes; staging `2·ceil(M_max)·W_expert` under ideal; **f 1**: `stagingBytes 0`, no `TransferOp`, `perStage[*].weightBytes / sequenceStateBytes / activationPeakBytes / fits` and every `decode`/`prefill` rate and op time equal the all-device baseline, while `tiers[host_ram].bytes` still holds the full expert allocation and `warnings` differ (the named-field contract of W1)                                                                                                                                                                                                                                                         |
| U-7     | `engine/specdec.test.ts`          | `expectedTokens`: iid α .85 k 1 → 1.85; α .79 k 7 → `(1 − .79^8)/.21 = 4.03948…`; per-position `[.854, .803, .794, .783, .775, .759, .729]` → 4.275 (DFlash 2 blog: 4.27)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| U-8     | same                              | verify lowering REF_DENSE_8B k 1 @ N 8192: every GEMM `m = 2`, `attn.core` `qLen 2, kvLen 8194`, `lm_head m = 2`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| U-9     | same                              | op lists: MTP k 1 = one catch-up pass `embed (⌈E[τ]⌉ rows), 2 norms, eh_proj (m ⌈E[τ]⌉, n 4096, k 8192), layer ops (qLen ⌈E[τ]⌉), final_norm (1 row), lm_head (m 1)`; MTP k 3 = that pass + 2 one-row passes; DFlash 2 (Qwen3.8-27B-DFlash2 config, k 7, m_ctx 5): `fc (m 5, n 5120, k 25600)`, per layer `attn_conv.kernel_proj (m 8, n 1280, k 5120)`, `q_proj (8, 4096, 5120)`, `k_proj`/`v_proj (m 13, n 1024, k 5120)`, `attn.core (qLen 8, kvLen min(N, 2048) + 13, window 2048, causal false, kvWriteTokens 5)`, `mlp_conv.kernel_proj`, `mlp.gate_up (8, 34816, 5120)`, `lm_head (m 7)`, `selector.hidden_proj (7, 256, 5120)`, `selector.codebooks 119 rows`; DSpark (dspark_qwen3_8b config): + 7 × `markov.W2 (1, 151936, 256)` and no convs/selector |
| U-10    | same                              | parameter counts (§3.3): DFlash 2 drafter 1,924,404,480 (± 0.1%); DSpark drafter 2,371,077,376 incl. its embedding/head copies (± 1%)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| U-11    | `engine/transfer.test.ts`         | QD: Qwen rows from the 990 Pro (45 µs variant), reads `[{16, 320}]`: QD 512 ⇒ 57.429 µs; **QD 1 ⇒ 46 + 15·45 = 721 µs**; QD 4 ⇒ 46 + max(168.75, 11.43, …) = 214.75 µs (the real preset, α = 1/22,000 s: QD 1 ⇒ 46.4545 + 15·45.4545 = 728.27 µs)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| U-12    | same                              | shared SSD, genuine four consumers (`tp 4`, one node, every stage offloads): Qwen rows `[{16, 320}]`: `service = max(1.318e-6, 16/(1.4e6/4) = 4.571e-5, 65,536/(7.45e9/4) = 3.519e-5, 65,536/(96e9/8) = 5.46e-6)` ⇒ `46 + 45.71 = 91.71 µs` (vs 57.43 µs alone); experts `[{8, 2,654,208}]` from SSD: alone `46 µs + 2.8502 ms`, with n 4 `46 µs + 11.4006 ms`                                                                                                                                                                                                                                                                                                                                                                                                   |
| U-13    | `engine/validate.test.ts`         | spec-dec with pp 2 valid, and with the token embedding off-device rejected (`SPEC_DEC_PP_OFFLOADED_EMBEDDING_ERROR`); `cpu_compute` rejected; `k + 1 > blockSize` rejected                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| U-14    | `engine/specdec.test.ts`          | hybrid verify: TINY_HYBRID k 3 ⇒ each GDN `RecurrentOp.stateSnapshots = 4`, state write bytes 4× the decode step's, conv `stateWriteBytes` 4×; `speculativeTransientBytes = 4 · state_s · B_r`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| U-15    | same                              | spec-enabled prefill. MTP (REF_DENSE_8B, P 8192): the prefill's `draft.*` ops equal an independently lowered module at `qLen = kvLen = 8192` (`eh_proj m 8192`, `attn.core` writing 8192 KV tokens, `lm_head m 1`). Block drafter (Qwen3.8-27B-DFlash2 config, P 8192): the `draft.*` op-name multiset is **exactly** `{ draft.fc, draft.hidden_norm } ∪ 5 × { k_proj, v_proj, qk_norm, rope, kv_cache_write }`: `k_proj`/`v_proj (m 8192, n 1024, k 5120)`, `kv_cache_write` `elems 2048·2·8·128 = 4,194,304`, `kvWriteBytes 8,388,608` (bf16), `activationReadBytes 8,388,608`, FLOP 4,194,304; **no** `attn.core`, `q_proj`, `o_proj`, `mlp.*`, `lm_head`, `selector.*`, `markov.*`                                                                           |
| U-16    | `lower/ngram.test.ts`             | PLE at index 1 (config `[2]`), Engram at 1 and 14; useful bytes 320 / 264 per row; Engram decode `T 1`: `separate` ⇒ `reads [{24, 256}, {24, 8}]`, DDR5 `bytes_tier 24·256 + 24·64 = 7,680`, `usefulBytes 6,336`; `pinned_host` on NVMe ⇒ `reads [{24, 256}]`, `bytes_tier 98,304`, `usefulBytes 6,336`, host-RAM tier gains `384,006,168·8 = 3,072,049,344 B` (table 1) and `3,072,133,456 B` (table 2); `interleaved` ⇒ `reads [{24, 264}]`, `bytes_tier 98,304` on NVMe / 7,680 on DDR5, `usefulBytes 6,336`; `lookupTables` absent ⇒ no op                                                                                                                                                                                                                   |
| U-17    | `engine/transfer.test.ts`         | `n_stream` (one node, `tp = dp = 1` unless said): (a) `b = 1, pp = 4`, `offloadOverlap 'none'`, a `dense_layers` range on stage 0 only ⇒ 1 (service equals the single-device value); (b) same with `'ideal'` ⇒ 1; (c) `pp 4, b 1, 'ideal'`, experts offloaded on every stage ⇒ 4; (d) `pp 4, b 1, 'none'`, every stage ⇒ 1; **(e) `pp 4, b 2 (m = 2), 'none'`, every stage ⇒ 2; (f) `pp 4, b 2, 'none'`, experts offloaded on stages {0, 1} only ⇒ 2; (g) `pp 4, b 2, 'none'`, stage 0 only ⇒ 1;** (h) `pp 4, b 4 (m = 4), 'none'`, every stage ⇒ 4; (i) one node, `tp 4`, every stage ⇒ 4 (= U-12)                                                                                                                                                              |
| U-18    | `engine/specdec.test.ts`          | MTP cache advance: REF_DENSE_8B k 1, `acceptance { per_position, alphas [1] }` (all accepted, `E[τ] = 2`) ⇒ the draft pass has `qLen 2`, `attn.core.kvWriteBytes = 2·4096 = 8,192 B`, `eh_proj m 2`, `lm_head m 1`; three consecutive steps advance the module's committed length by 2 each (`kvLen` of step i = N + 2i + … in the test's explicit stepping helper); `alphas [0]` (rejected, `E[τ] = 1`) ⇒ `qLen 1`, `kvWriteBytes 4,096`, the tentative row of the previous step counted as overwritten (no extra write)                                                                                                                                                                                                                                        |
| F-1     | `engine/specdecPp.test.ts` (done) | PP > 1 spec-dec with `targetLayerIds` spanning stages: features on the stage sends and the embedding copy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| S-1     | `__sanity__/reports.test.ts`      | lm-09 (MTP k 8), lm-36 (DSpark k 5 at the reported E[τ] 2.85), lm-39 (DFlash 2 k 7) with spec-dec on, and the offload rows web-52 (PR 29887, 5090, 67.8 tok/s @ h .89) and web-53 (ubergarm Qwen3-235B, 10 tok/s), `simulated ≥ 0.9 × reported`; as built, web-53's decode is a known exception (CPU-compute offload, §10) and its prefill is asserted; ratios in docs/sanity-check.md                                                                                                                                                                                                                                                                                                                                                                           |
| W-1…W-4 | `engine/e2e.test.ts`              | §8.4–8.6: exact (rel 1e-6) on the hand-derived transfer/draft terms, within **0.1%** on the quoted tok/s                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

### 8.3 Numeric goldens after the latency audit (pinned by ID-1; measured deltas in §4.3)

bf16 weights/activations/KV, fp32 accumulate, default options. `SYNTH_CLUSTER` = 8 `SYNTH_GPU` per node × 2 nodes,
intra `SYNTH_NVLINK` (100 GB/s, α 1 µs), inter `SYNTH_IB` (50 GB/s, α 5 µs) (`__fixtures__/synthetic.ts`). Workload
batch `b`, prompt `P`, decode context `N`. Cycle times in seconds, rates in tok/s, memory in bytes per device of each
stage (`totalBytes`).

| id   | config                                                                                            | decode cycle            | decode tok/s @N    | prefill cycle          | prefill TTFT          | prefill tok/s @P   | memory per stage                 | fits  |
| ---- | ------------------------------------------------------------------------------------------------- | ----------------------- | ------------------ | ---------------------- | --------------------- | ------------------ | -------------------------------- | ----- |
| G-a  | REF_DENSE_8B, RTX 5090, tp 1, b 1, P 4096, N 1024                                                 | 0.008538328428571402    | 117.118942936623   | 0.31770528850866675    | same                  | 12892.451426373611 | [17,287,440,384]                 | true  |
| G-b  | REF_DENSE_8B, cluster, tp 4, b 1, P 4096, N 1024                                                  | 0.004123760767999988    | 242.49709337164026 | 0.1800527418072269     | same                  | 22748.889902411895 | [4,347,488,256]                  | true  |
| G-c  | REF_DENSE_8B, cluster, tp 2 pp 2, b 2, P 4096, N 1024                                             | 0.008421233535999988    | 237.4948980396027  | 0.34323770569445416    | 0.5146386971656812    | 23866.84173705675  | [4,548,502,528, 4,548,510,720]   | true  |
| G-d  | REF_DENSE_8B, cluster, pp 2, b 4, P 4096, N 1024                                                  | 0.01670985011199998     | 239.3797654191672  | 1.3158195622538693     | 1.9732903249488039    | 12451.555266389125 | [10,002,670,592, 10,002,678,784] | true  |
| G-e  | TINY_HYBRID, cluster, tp 1 ep 2 dp 2, b 4, P 512, N 1024 (mode B all-to-alls; EP imbalance ρ 1.2) | 0.000021192448000000002 | 188746.48176557987 | 0.00024135932787548635 | same                  | 8485273.877861196  | [10,179,008]                     | true  |
| G-f  | TINY_HYBRID, cluster, tp 2 pp 2, b 2, P 512, N 1024                                               | 0.000031463856          | 63564.999788964196 | 0.00018910956775097272 | 0.0002818100477509727 | 5414850.301749119  | [3,323,008, 3,958,592]           | true  |
| G-g  | REF_DENSE_8B, cluster, tp 8 pp 2 (inter-node PP), b 1, P 4096, N 1024                             | 0.0024470846560000017   | 408.6495322293416  | 0.04824354397980686    | 0.09707363224761373   | 84902.55197077664  | [1,162,519,552, 1,162,527,744]   | true  |
| G-i  | REF_DENSE_8B, cluster, tp 16 (hierarchical 8×2), b 1, P 4096, N 1024                              | 0.002922674992000001    | 342.15231003694157 | 0.060721280187806884   | same                  | 67455.75829974837  | [1,238,329,344]                  | true  |
| G-h1 | Qwen/Qwen3.8-Flash-Next fixture (repo quant), RTX 5090, tp 1, b 1, P 8192, N 8192                 | 0.008356467181537718    | 119.66779480799501 | 0.6093556337025076     | same                  | 13443.709300305578 | [354,839,488,512]                | false |
| G-h2 | deepseek-ai/DeepSeek-V4.1-Flash fixture (repo quant), same                                        | 0.008979504281438205    | 111.3647222227099  | 0.8779083600783636     | same                  | 9331.270064757975  | [502,483,014,272]                | false |

α bookkeeping after the audit (`PhaseResult.comm` includes forward and return): G-b decode stage α =
`295 µs` (292 AR + 3 sampling AG), β = `11.96056 µs`; G-c stage α `147 µs`, forward α 1 µs + β 0.08192 µs,
return α 1 µs + β 0.00004 µs, total α `149 µs`. G-g stage α `445 µs` (438 AR + 7 AG), forward and return
α 5 µs each, total `455 µs`. G-i stage α `1764 µs` (1752 AR + 12 hierarchical AG).
G-e stays α `8 µs` (4 dispatch + 4 combine); G-f stage α `19 µs` (18 AR + 1 AG), forward and return α 1 µs
each, total `21 µs`. The resource branch of the pipeline maximum can hide loop-back time (§4.3).

### 8.4 Worked example W1: Qwen3-30B-A3B-class MoE on one RTX 5090, experts in DDR5 over PCIe 5.0 x16

Fixture `REF_MOE_30B_A3B` (add to `models/reference.ts`; dims from https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json ,
copy them from the file when adding): h 2048, 48 layers, GQA 32 / 4 KV heads, hd 128, q/k-norm, MoE E 128, top-k 8,
expert intermediate 768, gated SiLU, no shared experts, vocab 151,936, untied; `countParams` = 30,532,122,624. Quant:
bf16, except `modules.routed_experts = { dtype: 'int4', extraBitsPerElem: 0.5 }` (4.5 bits/weight, weight-only).
Hardware `RTX_5090`, `singleDevice`, host link PCIe 5.0 x16 (`B_link = 63,015,384,615.38 B/s`, `α_link 1e-6`),
`DDR5_6000_2CH` (`B_ram 96e9`, `lat_ram 70e-9`, `QD 64`), `n_stream 1`. Batch 1, prompt 8192, decode context 8192,
`overlap 'ideal'`, both `offloadOverlap` values, `hitRatio uniform` (h = f).

- `W_expert = 3·2048·768·4.5/8 = 2,654,208 B`; per layer `E·W_expert = 339,738,624 B`; all 48 layers
  16,307,453,952 B (≤ 96e9: host tier fits).
- Baseline (all resident, today's engine): decode cycle **2.46673 ms** (405.39 tok/s @ 8k; device traffic
  4,299,377,664 B / 1.792e12 = 2.3992 ms plus the compute-bound 128-wide router GEMMs and vector ops); prefill
  cycle **377.580 ms** (21,696 tok/s @ 8k). No collectives ⇒ `core_s` = these cycles, `lat_s = 0`.
- Decode `T = 1`: `U = 8`. Prefill `T = 8192`: `U = 128`. Hits per §2.4 with `C = 64` at f .5: decode `clamp(4, 0, 8)
  = 4`; prefill `clamp(64, 64, 64) = 64`.
- Per layer `t_path = 1.07 µs + max((M−1)·70e-9/64, M·W/96e9, M·W/63.015e9)` = link-bound.

| case         | f = h | M   | bytes/layer | service    | `t_path`/layer | `xfer = 48·t_path` | `'ideal'`: `max(core, xfer)`       | `'none'`: `core + xfer`            |
| ------------ | ----- | --- | ----------- | ---------- | -------------- | ------------------ | ---------------------------------- | ---------------------------------- |
| decode @ 8k  | 0     | 8   | 21,233,664  | 336.960 µs | 338.030 µs     | 16.2254 ms         | 16.2254 ms → **61.63 tok/s @ 8k**  | 18.6922 ms → **53.50 tok/s @ 8k**  |
| decode @ 8k  | 0.5   | 4   | 10,616,832  | 168.480 µs | 169.550 µs     | 8.1384 ms          | 8.1384 ms → **122.87 tok/s @ 8k**  | 10.6051 ms → **94.29 tok/s @ 8k**  |
| prefill @ 8k | 0     | 128 | 339,738,624 | 5.39136 ms | 5.39243 ms     | 258.837 ms         | 377.580 ms → **21,696 tok/s @ 8k** | 636.417 ms → **12,872 tok/s @ 8k** |
| prefill @ 8k | 0.5   | 64  | 169,869,312 | 2.69568 ms | 2.69675 ms     | 129.444 ms         | 377.580 ms → **21,696 tok/s @ 8k** | 507.024 ms → **16,157 tok/s @ 8k** |

Since the PCIe payload efficiency 256/280 (`src/sim/hw/presets/links.ts: PCIE_PAYLOAD_EFFICIENCY`,
simulation-model.md §4), `B_link` is 57.614e9 B/s instead of 63.015e9, so every link term above grows by 280/256.
The pinned rates (`e2e-offload.test.ts`) are now decode **56.364 / 49.484** (f 0, ideal / none) and **112.40 /
88.003 tok/s @ 8k** (f .5); prefill ideal stays **21,696** (compute bound), none **12,399** (f 0) and **15,780 tok/s
@ 8k** (f .5). The table keeps the original derivation.

Memory (f 0, ideal): device `W_s = 19,389,640,704 − 16,307,453,952 = 3,082,186,752 B`, staging `2·128·2,654,208 =
679,477,248 B` (prefill misses all 128), KV at 8193 tokens 805,404,672 B, activation peak 469,762,048 B ⇒
5,036,830,720 B ≤ 32e9; `tiers[host_ram] = 16,307,453,952 B ≤ 96e9`. f .5: `W_s = 11,235,913,728 B`, staging
`2·64·W = 339,738,624 B`. f 1: no TransferOp, staging 0, results = baseline. Spec-dec note: MTP k = 1 on top makes
the verify's `U(2) = 15.5`, i.e. 1.94× W1's decode transfer at the same h.

Test: `tokensPerSec` within 0.1% of the bold numbers; `transfer.sec = 48·(1.07e-6 + M·2,654,208/63,015,384,615.38…)`
to rel 1e-6; `memory.tiers[0].bytes` exactly 16,307,453,952. **f = 1 contract:** `decode.cycleSec`, `decode.tokensPerSec`,
`prefill.cycleSec`, `prefill.latencySec`, `prefill.tokensPerSec`, every `OpResult.sec` of both phases, and
`memory.perStage[0].{weightBytes, sequenceStateBytes, activationPeakBytes, stagingBytes (= 0), fits}` equal the
all-device baseline to rel 1e-9; `memory.tiers[0].bytes = 16,307,453,952` (the host copy) and `warnings` are the only
differences (no `TransferOp` is emitted).

### 8.5 Worked example W2: MTP k = 1 on REF_DENSE_8B, one RTX 5090, bf16, batch 1, decode @ 8k

Draft: a synthetic DeepSeek-V3-style MTP module on REF_DENSE_8B (`hiddenProj: true`, one target-shaped layer,
192,937,984 params; shared embedding/head, same device, no extra memory). α **0.85** (single-step prose default,
§5.1) ⇒ `E[τ] = 1.85`. pp = 1 (pp > 1 rejected).

- Baseline decode @ 8k: cycle **9.128152 ms** (109.55 tok/s @ 8k): bytes 16,357,649,152 B (GEMM weights
  15,136,194,560 + KV 36·8193·4096 = 1,208,156,160 + activations/norms) / 1.792e12.
- **Verify** (`qLen 2, kvLen 8194`): bytes 16,370,527,744 B (+147,456 B KV for one more token × 36 layers, + doubled
  activation rows), 39.95 GFLOP, every op memory-bound ⇒ `t_verify = 9.135339 ms`. No recurrent state (dense).
- **Draft** (one catch-up pass with `qLen = m_ctx = ⌈1.85⌉ = 2`, `kvLen 8194`: `embed` 2 rows 16,384 B; 2 norms over 2
  rows; `eh_proj` 67,108,864 B weights (m 2); layer weights 385,875,968 B with `qkv_proj`/`o_proj`/`mlp` at m 2, its
  KV read 8194·4096 = 33,562,624 B and KV write 2·4096 B; `final_norm` 1 row; `lm_head` 1,244,659,712 B at m 1; total
  1,732,412,672 B, every op memory-bound) ⇒ `t_draft = 0.966748 ms` (the shared LM head is 72% of it; the second
  row costs 0.233 µs, 0.02%).
- Step 10.102087 ms; `tokensPerSec = 1.85/0.010102087 = 183.13 tok/s @ 8k`; speedup 1.6717 ⇒ **"decode 183.1 tok/s
  @ 8k (spec-dec ×1.67)"**. α .75: 173.2 (×1.58); α .82: 180.2 (×1.64).
- Batch 8: verify `m = 16 < 117` still weight-bound (13.955 ms vs 13.897 ms baseline): the gain holds.
- Memory: draft weights `2·(192,937,984 + 2·4096·4096) = 452,984,832 B` for the GEMMs; as built **453,026,304 B**, which
  also counts the module's norm vectors (`enorm`, `hnorm`, `final_norm`, the layer's two norms and q/k norms: 20,736
  params); draft KV at 8194 tokens 33,562,624 B; prefill (U-15) adds the module over 8192 tokens to the prefill phase.

Test: `speculative.verifySec = 9.135339e-3`, `draftSec = 9.66748e-4` rel 1e-6; `expectedTokensPerStep = 1.85`;
`tokensPerSec` within 0.1% of 183.13.

### 8.6 Worked examples W3/W4: n-gram tables off-device, decode and prefill @ 8k, one RTX 5090

Fixtures as loaded by `mapHfModel` (repo quant; G-h1/G-h2 baselines). The rest of the model does not fit a 5090;
the example isolates the table terms (which the implementation must reproduce exactly) and quotes the resulting
rates. Tiers: `DDR5_6000_2CH` (`lat 70e-9, B 96e9, gran 64, QD 64`), `nvme-990-pro-2tb` **with `lat 45e-6`** (`IOPS 1.4e6,
B 7.45e9, gran 4096, QD 512`), link PCIe 5.0 x16, `n_stream 1`, scale layout per §2.10's defaults (`separate` on
DDR5: reads `[{T·g, 256}, {T·g, 8}]`, 320 B of cache lines per row; `pinned_host` on the SSD: one 256 B read per row
= one page, scales 6,144,182,800 B in host RAM), `offloadOverlap 'none'`
(`'ideal'` leaves every rate at the baseline: the gathers hide behind compute). Rule of §2.3: `t = α_tier + α_link +
max(Little, IOPS, tier, host, link)` per table.
(The preset's latency is 1/22,000 s = 45.45 µs; the tests reproduce this table, U-4, U-11 and U-12 with an explicit
45 µs variant of the preset. With the preset itself every SSD `α` is 0.4545 µs larger and the Little's-law terms
1% larger, e.g. the PLE decode path 57.883 µs instead of 57.429 µs.)

|                      | Qwen3.8-Flash-Next (PLE)                                                                                                                                                                                    | DeepSeek-V4.1-Flash (Engram)                                                                                                                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| table                | 1 at layer index 1: 320,000,000 rows × 160 bf16 = 320 B/row; 102,400,000,000 B                                                                                                                              | 2 at indices 1, 14: 384,006,168 + 384,016,682 rows × 256 mxfp8 (1.03125 B/elem) = 264 B/row (320 B in 64 B lines; one 4 KiB page); 202,758,032,400 B                                                                               |
| rows per step        | decode `R = 16`; prefill `R = 131,072` (one 320 B read per row)                                                                                                                                             | per table decode `R = 24` rows (DDR5 `separate`: 48 physical reads; SSD `pinned_host`: 24 page reads); prefill `R = 196,608` rows (393,216 / 196,608 reads)                                                                        |
| baseline             | decode 8.356467 ms (119.668 tok/s @ 8k); prefill 609.356 ms (13,443.7 tok/s @ 8k)                                                                                                                           | decode 8.979504 ms (111.365 tok/s @ 8k); prefill 877.908 ms (9,331.3 tok/s @ 8k)                                                                                                                                                   |
| **DDR5, decode**     | `α 1.07 µs`; terms: Little `15·70e-9/64 = 16 ns`, tier `5,120/96e9 = 53 ns`, link `5,120/63.015e9 = 81 ns` ⇒ **1.1513 µs** ⇒ 8.357618 ms → **119.651 tok/s**                                                | per table `α 1.07 µs` + max(Little `47·70e-9/64 = 51 ns`, tier `(24·256 + 24·64)/96e9 = 80 ns`, link `6,336/63.015e9 = 100.5 ns`) = 1.1705 µs; two tables **2.3411 µs** ⇒ 8.981845 ms → **111.336 tok/s**                          |
| **DDR5, prefill**    | terms: Little 143.4 µs, tier `41,943,040/96e9 = 436.9 µs`, link `41,943,040/63.015e9 = 665.6 µs` ⇒ **0.66667 ms** ⇒ 610.023 ms → **13,429.0 tok/s @ 8k**                                                    | per table max(Little `393,215·70e-9/64 = 430.1 µs`, tier `196,608·320/96e9 = 655.4 µs`, link `51,904,512/63.015e9 = 823.7 µs`) + 1.07 µs = 0.82475 ms; two tables **1.6495 ms** ⇒ 879.558 ms → **9,313.8 tok/s @ 8k**              |
| **990 Pro, decode**  | `α 46 µs`; terms: Little `15·45e-6/512 = 1.32 µs`, IOPS `16/1.4e6 = 11.43 µs`, tier `65,536/7.45e9 = 8.80 µs`, host `65,536/48e9 = 1.37 µs`, link 0.08 µs ⇒ **57.429 µs** ⇒ 8.413896 ms → **118.851 tok/s** | per table `46 µs` + max(2.02 µs, `24/1.4e6 = 17.14 µs`, `98,304/7.45e9 = 13.20 µs`, 2.05 µs, 0.1 µs) = 63.143 µs; two tables **126.286 µs** ⇒ 9.105790 ms → **109.820 tok/s**                                                      |
| **990 Pro, prefill** | terms: Little 11.52 ms, IOPS `131,072/1.4e6 = 93.623 ms`, tier `536,870,912/7.45e9 = 72.06 ms`, host 11.18 ms, link 0.67 ms ⇒ **93.669 ms** ⇒ 703.025 ms → **11,652.5 tok/s @ 8k**                          | per table max(17.28 ms, `196,608/1.4e6 = 140.43 ms`, 108.1 ms, 16.78 ms, 0.82 ms) + 46 µs = 140.480 ms; two tables **280.96 ms** ⇒ 1,158.87 ms → **7,069.0 tok/s @ 8k**                                                            |
| memory               | device weights 353,887,054,848 → 251,487,054,848 B; `tiers[host_ram] = 102.4e9 > 96e9` ⇒ **host tier does not fit** (128 GB or an int8 table needed); `nvme` 102.4e9 ≤ 2e12 fits                            | device 501,382,474,992 → 298,624,442,592 B; host RAM: `separate` 202,758,032,400 B > 96e9 does not fit; SSD with `pinned_host`: `tiers[nvme] = 196,613,849,600 B` (values) fits, `tiers[host_ram] = 6,144,182,800 B` (scales) fits |

The `interleaved` layout gives the same numbers as above on both tiers for these two models (320 B = 256 + 64 on DDR5;
one page either way on the SSD); `separate` on the SSD would double the Engram page reads (IOPS term 34.3 µs per table
in decode) while `usefulBytes` stays 6,336 B. Test: the transfer op's `sec` equal to the bold values to rel 1e-6;
With the PCIe payload efficiency (`B_link` 57.614e9) the DDR5 link terms grow by 280/256: PLE 1.1589 µs (decode) and
0.72907 ms (prefill), Engram 2.3599 µs and 1.8039 ms for both tables; the pinned rates move by less than 0.1%.
`tiers` bytes exact; rates within 0.1%; with `lookupTables` absent the results equal G-h1/G-h2 exactly.

---

## 9. Open questions (for the user), with the decisions the implementation took

1. **Lookup-table UI start state.** The simulator defaults to `'device'` (strict identity); the UI shows the hint chip
   and pre-fills Host RAM once offload is turned on. Should the UI _start_ PLE/Engram models at Host RAM instead?
   _Built: no; `'device'` with the hint chip._
2. **Prefill with offloaded experts under `'ideal'`:** the transfer hides behind the (large) prefill compute, so a
   transfer-bound prefill only shows under `'none'`. Acceptable as the upper bound, or should prefill always
   serialize (llama.cpp copies per ubatch synchronously)? _Built: accepted as the upper bound; the "?" help says so._
3. **Achieved vs theoretical rates.** Presets store datasheet ceilings (PCIe 5 x16 63.0 GB/s vs ≈ 50–53 GB/s measured).
   Keep the "theoretical" stance with the `linkBandwidth` modifier, or add an `achievedFraction` knob? _Built:
   theoretical (datasheet) rates._
   F13 addendum (2026-11-09 user decision): retain that default; add the opt-in sustained SSD scenario of §2.2–2.3.
4. **Hit-ratio default** `uniform` (exact lower bound) vs `skewed_classic` (typical observed)? _Built: `uniform`._
5. **CPU-compute mode**: hook only (this doc) or a first version with a user-entered host matmul peak? _Built: hook
   only; validation rejects `'cpu_compute'`._
6. **Drafter quantization**: drafters ship bf16; should a draft-specific `QuantConfig` override be offered? _Built: no
   override._
7. **Latency-audit corrections (§4.3)**: implement all of G1–G4, G7, G8 in the follow-up, or only the clear omissions
   G1/G2/G3/G8 (G4/G7 are judgments about fabrics and algorithm choice)? _Decision: apply ALL of G1–G4, G7, G8.
   Implemented in the latency-audit follow-up (§4.3); G6 remains out of scope._
8. Research gaps accepted as is: no Reddit access; no primary citation for DeepSeek-R1-from-NVMe runs; no LRU curve for
   DeepSeek-V3/R1; DFlash 2 numbers come from the vendor (no paper); Gen5 SSD QD1 latency, M3 Ultra SSD speed and the
   DSpark confidence-head layout are estimates; vLLM support for the n-gram models unverified.

Further decisions the design left open: the hypothetical drafter has its defaults only (not configurable) in v1; a
method × use-case pair without a measurement takes the prose value, marked interpolated (§5.1 "(I)"); the memory fit
under spec-dec uses the context `max(P, N + k + 1)`; the table staging term is the op's `usefulBytes`; a device with
no `hostInterface` and no `host.link` is a validation error when something is off-device.

---

## 10. Implementation deviations

Where the code differs from revision 2 of this design, and why. The sections above already read as built.

**F13 sustained SSD scenario (2026-11-09):** optional measured fractions augment, rather than replace, datasheet ceilings. See §2.2–2.3 and `src/sim/engine/sustained-read.test.ts`; no hit-ratio defaults change.

**Types and validation**

1. `SpecDecodeConfig.drafter` repo variant is `{ kind: 'repo'; repoId; spec: BlockDraftSpec }` (§7.3): the engine is
   pure and does not fetch, so the loader (`src/hf/loadDrafter.ts`) parses the repo's config and hands the spec over.
2. The full §2.6 stage formula and `PhaseResult.transfer` (`pipeline.ts: phaseTransfer`) landed with the types, ahead
   of the offload engine; no behaviour change (identity holds with `xfer_s = 0`).
3. New elementwise fns `kv_cache_write` (1 FLOP/elem) and `candidate_score` (2 FLOP/elem) in `lower/common.ts`:
   the write-only cache op of §3.4 and the DFlash 2 selector score of §3.3 needed their own entries in the FLOP table.
4. Extra validation: `scaleLayout 'pinned_host'` requires tier `'nvme'` (the scales are pinned in host RAM while the
   values sit on the SSD); a block drafter's `spec.method` must match the configured method; the host-link error
   reads "deployment.offload: no host link known for <id>; set hardware.host.link".
5. `OffloadLowering.routedExperts.hitRatio` is the evaluated number h(f), not the model: lowering needs a number, and
   resolving the model once (`engine/host.ts`) keeps the lowering free of the hit-ratio table.

**Hardware presets**

6. T705 random read is recorded as QD 512 × 1 thread: the vendor footnote gives no split.
7. SN850X: the QD 32 × 16 condition of its 1.2 M IOPS is an estimate (the datasheet does not state it).
8. DDR5-5600 latency is estimated (the review measured DDR5-6000); DDR5-6000 12-channel bandwidth is estimated (AMD
   rates the EPYC part at 6400).
9. The achieved PCIe Gen5 ≈ 50–53 GB/s copy rate is unconfirmed (only Gen4 ≈ 24.8 GB/s is, nvbandwidth #53); it is
   not a preset either way.
10. SXM / OAM modules and SoCs have no `hostInterface`, so offload on them is a validation error unless
    `hardware.host.link` is set: the design's default link exists only for cards whose PCIe edge is cited.
11. Added `HOST_PCIE_LINKS` (the host-link picker's list) and the `DEFAULT_HOST_RAM` / `DEFAULT_SSD` aliases.

**Offload engine**

12. The 990 PRO preset's QD1 latency is 1/22,000 s = 45.45 µs (the design rounded to 45 µs): the preset keeps the
    datasheet-derived value. The worked examples W3/W4 and U-4, U-11, U-12 use an explicit 45 µs variant so their
    numbers hold; the real preset gives `α = 1/22,000 + 1 µs` (§8.6).
13. n-gram table rows per device are `T·g/tp` (the table is row-parallel under TP); the design wrote `T·g`, the same
    at tp 1.
14. (Superseded: whole-layer offload was removed, §2.11.) A layer in the `denseLayers` range streams whole, with no expert or table transfer on top: its `W_layer` already
    holds every expert and the table shard.
15. `n_stream` across nodes: a stage spanning nodes takes the largest `n_stream` of its nodes; the tier bytes report
    the fullest node; DP replicas on one node each count their own copy. The design defined the sharing for one node.
16. The resident expert count `C = f·E_loc` is a real number in the memory fit (consistent with the expected-value
    misses of §2.4).
17. The unified-memory default SSD is the device's own drive (Apple, DGX Spark, Strix Halo presets), not the 990 PRO:
    a unified system streams from its internal SSD.
18. Link modifiers are not applied to the host link (they describe the device-to-device links).
19. New file `lower/offload.ts` (placement helpers); `lower/moe.ts` imports `engine/transfer.ts` for `U` and the
    hit/miss split.

**Speculative decoding**

20. DSpark block semantics: DeepSpec's `block_size` counts draft positions, so the loader stores `blockSize =
    block_size + 1`; a DSpark pass embeds k rows (anchor + k − 1 masks) and takes logits at all k, a DFlash 2 pass
    k + 1 rows. DeepSeek-V4.1: `dspark_block_size 5` ⇒ blockSize 6, default k 5 (the design said blockSize 5);
    Qwen3-8B DSpark: blockSize 8, k 7 (§3.2, §3.3).
21. The DFlash 2 conv `finish` step costs 4·K FLOP/elem, not 2·K: the code applies the base and the dynamic taps in
    both prepare and finish.
22. The draft lowering takes the target `model` (shared embedding and head shapes) and the state function takes the
    draft and the lowering context; signatures only.
23. `REF_DENSE_8B_MTP` (W2's model) lives in `models/referenceDraft.ts`: the identity test pins the exports of
    `models/reference.ts`.
24. The drafter config fixtures live in `models/hf/__fixtures__/`.
25. Several MTP modules run as a chain (pass p uses module min(p, L)); `maxReliableDepth` = the module count.
26. W2 draft weights are 453,026,304 B, not 452,984,832 B: the code also counts the module's norm vectors (20,736
    params); the design summed the GEMMs only (§8.5).
27. Mistral Large 4's DSpark drafter is deferred: its `params.json` lacks the drafter's head counts and dims and
    declares head type `rnn`, so no shape can be read honestly.
28. DeepSeek-V4.1 MoE drafter approximation: the context rows are added to the m of the CSA `wkv` GEMM; the attention
    runs non-causally with `kvWriteTokens = m_ctx`; the context rows' kv norm / RoPE are not added (§3.3).
29. Draft layers are appended in `simulate.ts: withDraftLayers`, not in `lowerModel`: the draft lowering imports
    the lowering, so `lowerModel` importing it back would be a cycle.
30. With spec-dec, the decode `latencySec` is `cycle / E[τ]` (time per output token); `cycleSec` stays the step time.
31. `verifySec` / `draftSec` are sums of layer times without overlap (display shares of the step, not a timeline).
32. The snapshot state is `blockSequenceStateBytes` at context 0 over the snapshotted block kinds (recurrent and
    conv state do not grow with the context).
33. The memory fit under spec-dec uses the context `max(P, N + k + 1)`.
34. Past a single-step head's depth the engine only warns; switching a kept preset acceptance to the tail vector is
    the UI preset's job (§3.5).

**UI**

35. Comm latency, comm bandwidth and offload transfer are bound parts of each breakdown row (bar segments and
    legend), per `BreakdownRow.boundSec`, not extra rows; transfer ops still appear as their own rows by op / kind.
36. "verify" and "draft" are group header rows over the per-layer rows.
37. The "Host RAM ↗" chip is a button next to the Lookup-tables picker (a button inside the trigger would nest
    buttons).
38. URL keys: `oo, hr, hrc, ssd, hl, ox, oxl, oxf, oxh, odl, odt, olt, ols, oqd, sd, sdr, sdk, sdu, sda`; `odt`
    (dense-layer tier) is extra.
39. Resident fraction is a number field (not a slider) and offload overlap a picker (not a switch), matching the
    existing controls.
40. At pp > 1 the method picker offers only "Off" (no disabled state), with the reason in its "?".
41. Queue depth 0 means the tier's rated maximum.

**Sanity check (§8.2 S-1)**

42. web-53 (ubergarm Qwen3-235B, 10 tok/s) is a known **exception** for decode, not an asserted row: the run computes
    the offloaded experts on the CPU from DDR5-6400 (102.4 GB/s), while the simulator's transfer mode streams them
    over the PCIe 4.0 x16 link (31.5 GB/s; 5.2 tok/s), and the CPU-compute mode that would bound it is a hook
    validation rejects (§2.8). Its prefill (transfer-bound in both modes) is asserted. lm-09 and lm-39 run with the
    method's prose acceptance preset (the records give none); lm-36 at the reported mean accepted length 2.85.

**Spec-dec at PP > 1 and the break-even batch (2026-11-10)**

43. §3.6 is implemented with the target features on the existing stage sends (all `k + 1` verify rows) instead of
    the `pp` collectives per feature layer the future-work note named: one message per boundary keeps the §7 cycle
    rule unchanged and needs no round trip after acceptance. The embedding copy is counted unless a tied head in the
    same format already sits on the last stage. `PhaseResult.speculative` gains `offTokensPerSec`, `offCycleSec` and
    `stepCostRatio` (speedup = E[τ]/stepCostRatio); the batch chart shows the spec-off rate and the break-even batch
    (docs/simulation-model.md §13 "Break-even batch"). Acceptance stays batch-independent (§13 "Not modeled").
