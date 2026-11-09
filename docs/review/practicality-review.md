# Practicality review of the simulation model

Review date: 2026-11-09, at base `39b0bec9`. This is a review only: nothing in `src/` changed, except that
`src/sim/__sanity__/reports.ts` gained 60 fresh report entries (see §2).

What it asks: are specmaxx's numbers practical? Are the default preset numbers the right kind? Do real reports fall below the
bound, and by how much? Which physics or math does the model leave out?

Inputs, produced independently:

- **§1** an audit of every hardware preset and default knob: 599 table rows.
- **§2** 105 benchmark reports (45 earlier, 60 fresh) giving 181 report ÷ sim metrics. Of these, 164 enter the
  distributions and 17 are excluded as unusable.
- **§3** a flaw hunt using throwaway simulator probes (`simulateReport` from `src/sim/__sanity__/harness.ts` plus op-list
  post-processing). The probes were deleted afterwards. F1–F14 below.

Conventions:

- Decode rates are batch 1 and per sequence unless stated.
- "@ N" after a decode rate is the number of KV tokens already cached.
- "@ ppN" after a prefill rate is the prompt length.
- A **ratio** is report ÷ sim. The sim claims an upper bound, so a ratio > 1 means the sim is too slow: it _underestimates_
  the bound and breaks its contract. A low ratio means the bound is loose. That is not wrong in itself, but it limits
  what the number is useful for.
- An **overestimate** means the sim's bound is higher than the hardware can physically reach.

---

## Executive summary

### What the report ratios show today

These are **descriptive ranges over a selected, correlated sample** (several points share one host, engine build or
page), not calibrated error bars. Explanations of the gaps below are hypotheses unless a probe or a measurement is
cited.

- **Dense-model decode on a single device: observed ratio ≈ 0.55–0.86.**
  - Consumer and workstation dGPUs, at contexts up to 33k: 0.56–0.82, median 0.73 (n = 14).
  - Apple, dense MLX/llama.cpp: 0.64–0.86. Mac pairs and quads over TB5 RDMA: 0.65–0.85 (n = 3).
  - Strix Halo, dense: 0.75–0.85. DGX Spark, dense: 0.55–0.81. Most fresh Spark rows were simulated at context 1
    because their prompt length is unknown, so those ratios are lower bounds (see §2).
  - In this sample, good engines reached about 0.7 × the sim. On Strix Halo and the M3 Ultra, the dense ratios sit
    close to the measured STREAM-type bandwidth fraction (F2); that is consistent with, not proof of, bandwidth being
    the main residual.
- **Where the bound is far from observed rates, or not comparable:**
  - Fine-grained MoE and small models at batch 1: 0.41–0.73 on Strix Halo, 0.12–0.47 on Apple MoE.
  - Datacenter GPUs at batch 1: 0.15–0.42 (hypothesis: fixed per-op costs dominate a short step, F1).
  - Multi-GPU over PCIe: median 0.31.
  - Prefill is a ceiling only: llama.cpp reached 0.1–0.38, vLLM and MLX 0.5–0.8. Short-prompt TTFT rows say nothing.
  - **CPU + GPU expert offload**: llama.cpp computes on the CPU, a mode the sim does not support (F3), so those
    reports are mismatched comparisons, in both directions.
  - **SSD offload**: F-W12/F-W13 inputs corrected in F13; both decode rows now asserted (see §2).

The theoretical headline stays the datasheet peak. Sustained-rate scenarios (F2, F4) and the per-op floor (F1) are
proposed as **labelled additional figures**, not as a replacement bound.

### Top findings, ranked by impact on the headline numbers

| #   | finding                                                                                                                                                                     | direction                                                                   | magnitude                                                                                                                                                                                                                                                                                                                                                        | confidence                                                |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 1   | **F3** Unsupported execution mode: llama.cpp `--cpu-moe` computes the offloaded experts on the CPU; the sim only models streaming them over PCIe                            | sim below such reports (a mode mismatch, not a transfer-mode formula error) | gpt-oss-120b on a 4090 + DDR5-6000, all experts in host RAM: transfer mode 16.3 tok/s @ 64; a CPU-compute probe bound 42.6 tok/s @ 64 (×2.6). Report ÷ sim in `reports.ts`: 1.13–3.65 (four rows, two of low reliability)                                                                                                                                        | high (mode); medium (magnitude)                           |
| 2   | **F5** Balanced routing reads `min(E, T·k)` experts at batch > 1 (**fixed**: now the iid-uniform expectation)                                                               | pessimistic at batch 4–32                                                   | reading the iid-uniform _expected_ unique count instead: +4% to +40% (Qwen3-30B-A3B @ 2k), +4% to +51% (gpt-oss-120b on a Spark @ 2k); kink at the saturation point. A **plausible partial contributor** to `web-04` (ratio 2.03), not a full explanation: after the fix 128.9 → 191.1 tok/s @ 4112 (ratio 1.37), still below the reported 262.2 tok/s @ 4112    | high (direction); medium (size)                           |
| 3   | **Confirmed preset bug**: Arc B580 catalog (`src/sim/hw/presets/catalog.ts:424`) offers PCIe 5.0 x16 for a PCIe 4.0 x8 card; Arc Pro B60 (`:423`) offers x16 for an x8 card | overestimate (comm)                                                         | link 4× too fast (B60: 2×). A 2-GPU 100 MB all-reduce takes 34% too little time via host bounce, and a quarter of the right time via direct P2P. The tok/s effect scales with the exposed comm share                                                                                                                                                             | certain                                                   |
| 4   | **F2** Decode uses the datasheet memory bandwidth (correct for the theoretical bound)                                                                                       | sustained scenario lower                                                    | measured STREAM / clpeak fractions (workload-specific): H100 91–93% (−7% to −9%), Strix Halo 83% (−17%), M3 Ultra 78% (−22%), Spark 73–90% (−10% to −27%)                                                                                                                                                                                                        | high (measurements); scenario, not a proven tighter bound |
| 5   | **F4** Matmul peaks are vendor (mostly boost-clock) figures; tile/wave quantisation is the only derate                                                                      | sustained scenario lower (prefill)                                          | independently supported clock/power limits: Strix Halo sustained ≈ 2.0 of 2.9 GHz (−31%), GB10 throttling (size varies by OEM). Library-efficiency measurements, separately: Spark BF16 77–99 of 125 TF, H100 MAMF 80–86%, MI300X 49–58%                                                                                                                         | high (Halo clock); medium (rest)                          |
| 6 | **SSD-offload inputs** F-W12 / F-W13 — resolved F13 | corrected ratios 0.321 / 0.479 @612 | checkpoint, module layout, table placement and measured/derived cache hits corrected; defaults unchanged | high (input correction) |
| 7   | **F1** No per-op latency term in decode: tiny ops cost only bytes ÷ bandwidth                                                                                               | sensitivity, possibly an overestimate                                       | a floor of 0.6–1 µs per IR op would cost −2.5% to −4% (Llama-3-8B, 4090) and −12% to −19% (Qwen3-30B-A3B, 4090). The hardware floor is **unmeasured**: IR ops are not all dependent DRAM round trips, and the 78% megakernel figure is one fused implementation. Engine dispatch (1.3–2.9 µs per kernel) is out of scope                                         | low (hardware size); high (dispatch is large for engines) |
| 8   | **F6** No softmax/SFU (exp) cost in attention                                                                                                                               | overestimate (long-context datacenter prefill)                              | B200: −4% / −10% / −16% for Llama-3-8B fp8 @ pp8k/32k/128k, up to −48% for gpt-oss (head dim 64) @ pp128k with overlapped exp. H100 0% overlapped. Decode unaffected                                                                                                                                                                                             | medium-high                                               |
| 9   | **F7** Prefill assumes ideal comm/compute overlap; small all-reduce α                                                                                                       | overestimate                                                                | prefill under `overlap: 'none'`: −40% (4×4090 TP4, Llama-3-70B), −31.5% (Spark pair), −24% (2×3090 MoE). Decode: the measured 6.1–13.5 µs PCIe all-reduce includes kernel launch; the hardware share (≈ 2–3 µs) would cost a few percent                                                                                                                         | medium                                                    |
| 10  | **F8** PCIe ceiling excludes TLP/DLLP protocol overhead                                                                                                                     | overestimate (offload, host bounce)                                         | −9% (0.91 payload efficiency) to −17% (26 GB/s measured) on transfer-bound offload decode                                                                                                                                                                                                                                                                        | high                                                      |
| 11  | **F9** Unified-memory fit ignores the macOS GPU wired limit (~75% of RAM by default)                                                                                        | optimistic fit                                                              | binary: e.g. Llama-3-70B 8-bit on an M4 Max 128 GB needs 96.5 GB @ 64k; the sim says it fits, macOS refuses it by default                                                                                                                                                                                                                                        | high (macOS)                                              |
| 12  | **Acceptance-default provenance**                                                                                                                                           | overestimate risk (spec-dec)                                                | DSpark prose α 0.76 gives E[τ] 3.70 @ k 7; a Qwen3.8 retrain measures τ 3.01 (+23%, one target). DFlash2 agentic is DFlash v1 data marked as measured                                                                                                                                                                                                            | high (provenance); medium (size)                          |
| 13  | **F10** iid acceptance extrapolated across k                                                                                                                                | either direction                                                            | **illustrative**: under a hypothetical decay curve (0.92, −0.06 per position), an α fitted at k = 7 is +18% at k = 15 and −9% at k = 3. No measured per-position data was used                                                                                                                                                                                   | low (size)                                                |
| 14  | **TB4 TCP** 5 GB/s is the cable rate; the cited test negotiated 20 Gb/s and reached 15.9–18.8 Gb/s                                                                          | overestimate (comm)                                                         | link bandwidth up to 2× too high                                                                                                                                                                                                                                                                                                                                 | medium                                                    |

### How to read the findings

- **Two rows break the comparison, not the formulas.** F3 is a missing execution mode. F5 is a pessimistic
  approximation. The ratios above 1 in the data are one of these, the unresolved `lm-35` row, or `lm-36`
  (1.079, inside the test's 10% tolerance).
- **The sustained scenarios are not a tighter proven bound.** F2 and F4 measurements are workload-specific. They are
  useful as a labelled "sustained" figure next to the peak.
- **The decode overestimates may compound.** A probe derate of Qwen3-30B-A3B Q4 on a 4090 (90% bandwidth plus a 1.5–3 µs
  dispatch term per op) moves the sim from 502.6 to 229–304 tok/s @ 64, against reports of 211–226 tok/s @ 64. That
  term is mostly engine dispatch, so this is a plausibility check, not a decomposition.

---

## §1 Preset and knob audit

Coverage: 29 devices, 23 links, 9 generated host-PCIe links, 13 host-RAM presets, 8 SSD presets, 48 topology instances, all
29 setup-catalog entries, all published GeForce unlock modifiers, all 20 spec-dec acceptance entries, and the engine
defaults. Result: **303 ok / 293 suspicious / 3 wrong** (599 rows). The full table, with every row's evidence and
suggested value, is in [presets-audit.md](presets-audit.md).

> **Caveat from the auditor.**
>
> - _Suspicious_ means: estimated, transferred between platforms, not independently recoverable, or a loose modelling
>   assumption. It does **not** mean a proven bug.
> - Rows are table rows, not independent defects. One template issue, such as "capacity is decimal GB", repeats across
>   every device.
> - No 2× dense/sparse or bandwidth-direction error was found in any GPU peak table. The PCIe catalog assignments
>   below are the only definite numeric mistakes.

### Wrong rows, and suspicious rows that move headline numbers

The "#" column is the row number in the full table in [presets-audit.md](presets-audit.md).

| #       | preset/knob                                                            | value                                                                                    | verdict    | evidence                                                                                                                                                                                                                                                                                                                                                                                                          | suggested value                                                                                                                 |
| ------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 576     | Arc B580 setup catalog (`src/sim/hw/presets/catalog.ts:424`)           | default PCIe 5.0 x16 host bounce (63.0 GB/s); also PCIe 5.0 x8 host, PCIe 5.0 x16 direct | **wrong**  | [Intel B580 spec](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html): PCIe 4.0 ×8 = 15.75 GB/s. The device preset itself records Gen4 x8 (`src/sim/hw/presets/intel.ts:450`), but collective paths do not clamp the link to `hostInterface`. Probe (2 GPUs, 100 MB all-reduce): host bounce 4.19 ms against 6.37 ms correct; direct 1.59 ms against 6.35 ms | Default PCIe 4.0 x8 host bounce; optional PCIe 4.0 x8 direct. Add a test that every catalog link ≤ the device's `hostInterface` |
| 575     | Arc Pro B60 setup catalog (`catalog.ts:423`)                           | offers PCIe 5.0 x16 direct (63.0 GB/s)                                                   | **wrong**  | [Intel B60 spec](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html): PCIe 5.0 ×8 electrical = 31.5 GB/s. The default x8 is correct                                                                                                                                                                                                                       | Remove the x16 option                                                                                                           |
| 542     | `dflash2/agentic` (`src/sim/engine/specdec.ts:139`)                    | α 0.65, `interpolated: false`                                                            | **fixed**  | [DDTree Table 1](https://arxiv.org/html/2604.12989): SWE-bench Lite τ 2.82 @ k 15 is Qwen3-8B with **DFlash v1**, not DFlash 2. The α inversion is fine; the attribution is not                                                                                                                                                                                                                                   | `interpolated: true`, labelled as a borrowed v1 proxy. No source gives a better point value                                     |
| 513     | memory bandwidth utilisation (`src/sim/engine/opTime.ts:59`)           | 1.0 of peak                                                                              | suspicious | Strix Halo ≈ 212 of 256 GB/s ([llm-tracker](https://llm-tracker.info/AMD-Strix-Halo-%28Ryzen-AI-Max%2B-395%29-GPU-Performance)); Spark 200–247 of 273; M3 Ultra ≈ 635 of 819; H100 STREAM 91–93%                                                                                                                                                                                                                  | Keep the peak mode; add a sourced sustained figure per preset (F2). No universal factor                                         |
| 515     | on-device small-op floor (`opTime.ts:66`)                              | 0 s                                                                                      | suspicious | No source establishes a zero hardware minimum, and none establishes its size. The launch-overhead exclusion is not a measurement of memory latency                                                                                                                                                                                                                                                                | Optional sensitivity setting (F1); keep 0 as the default until a hardware floor is measured                                     |
| 514     | compute efficiency assumptions (`src/sim/hw/efficiency.ts:53`)         | tile/wave only; vendor peak clock; 1 CTA per unit                                        | suspicious | Measured large-GEMM rates are 49–86% of peak, mixing clock/power limits with library efficiency (F4)                                                                                                                                                                                                                                                                                                              | Optional sourced "sustained clocks" `clockScale` where a clock limit is documented (F4)                                         |
| 509     | overlap default (`src/sim/engine/simulate.ts:55`)                      | `ideal`                                                                                  | suspicious | No engine overlaps comm across layers by default. `none` costs −24% to −40% in TP prefill (F7)                                                                                                                                                                                                                                                                                                                    | Show `none` beside `ideal` when exposed comm is large                                                                           |
| 347–355 | PCIe host links (`src/sim/hw/presets/links.ts:277`)                    | line rate × 128/130; α 1 µs                                                              | suspicious | The bandwidth arithmetic is right but excludes TLP/DLLP framing. Measured pinned copy on Gen4 x16: 25–26.4 GB/s of 31.5 (F8). α 1 µs is a Gen5 P2P-write proxy                                                                                                                                                                                                                                                    | × 256/280 payload efficiency, or a measured "achievable" figure                                                                 |
| 269 ff. | PCIe host bounce (`links.ts:182`)                                      | 96 GB/s shared host DRAM; +12 µs                                                         | suspicious | [tinygrad P2P matrices](https://github.com/tinygrad/open-gpu-kernel-modules/issues/29): one 5090 measurement. 96 GB/s assumes dual-channel DDR5-6000                                                                                                                                                                                                                                                              | Use the selected host RAM; give a range                                                                                         |
| 284 ff. | PCIe / NVLink / IB per-message α                                       | 0.8–1.5 µs                                                                               | suspicious | Proxies transferred across generations. Measured small all-reduce on PCIe: one-shot 6.1 µs, NCCL 13.5 µs, both including kernel launch (F7)                                                                                                                                                                                                                                                                       | A sourced hardware-only floor (≈ 2–3 µs for PCIe P2P), not the launch-inclusive figure                                          |
| 516     | default prefetch accuracy (`src/sim/engine/transfer.ts:30`)            | 0.8                                                                                      | suspicious | [DAOP](https://arxiv.org/html/2501.10375): 84% on Mixtral. Qwen2 ≈ 0.67. Model-specific. Effect is ±4% when transfer-bound (F13), up to 1.67× when transfer ≈ compute                                                                                                                                                                                                                                             | Show 0 / 0.8 / 1 sensitivity                                                                                                    |
| 519     | fine-grained hit skew (`transfer.ts:58`)                               | γ 0.2, h = f^γ                                                                           | suspicious | Developer measurements only (llama.cpp PRs 27861, 29887); gpt-oss reported flat. At f = 0.25: h 0.758 against 0.25 uniform, **3.1× fewer misses**                                                                                                                                                                                                                                                                 | Keep opt-in and low-confidence; prefer model-specific curves                                                                    |
| 421     | Samsung 9100 PRO outstanding (`src/sim/hw/presets/storage.ts:668`)     | 8192 (QD256 × 32 threads)                                                                | suspicious | That is the vendor's rating condition, not an inference queue depth. Large sequential expert reads make QD irrelevant in the probe (F13)                                                                                                                                                                                                                                                                          | Separate deployment QD from rated QD                                                                                            |
| 523     | default host RAM (`storage.ts:580`)                                    | DDR5-6000 2ch, 96 GB/s                                                                   | suspicious | A desktop prior, not representative of DDR4 or server hosts                                                                                                                                                                                                                                                                                                                                                       | Require or clearly show the host preset with offload                                                                            |
| 329     | Thunderbolt 4 TCP bandwidth (`links.ts:645`)                           | 5 GB/s                                                                                   | suspicious | [Thunderbolt networking test](https://sebastianalexis.com/blog/thunderbolt-networking-is-underused/): negotiated 20 Gb/s, measured 15.9–18.8 Gb/s                                                                                                                                                                                                                                                                 | Add a negotiated 2.5 GB/s variant; verify Mac↔Mac                                                                              |
| 543     | `dspark/prose` (`specdec.ts:142`)                                      | α 0.76                                                                                   | suspicious | [DSpark Table 1](https://arxiv.org/html/2607.05147), non-thinking Qwen3-4/8/14B, τ 3.64–3.72. The Qwen3.8 thinking retrain gives τ 3.01. Preset gives E[τ] 3.70 @ k 7: +23%                                                                                                                                                                                                                                       | Store target, drafter, k, temperature with each value                                                                           |
| 529     | `mtp_multi_step/code` (`specdec.ts:122`)                               | α 0.82                                                                                   | suspicious | Mixes greedy llama.cpp PR results with T = 1 model cards                                                                                                                                                                                                                                                                                                                                                          | As above                                                                                                                        |
| 552     | single-step MTP tail (`specdec.ts:167`)                                | 0.30, 0.10, 0.05…                                                                        | suspicious | [FastMTP](https://arxiv.org/html/2509.18362): vanilla 70/11/2% against fine-tuned 81/56/36%                                                                                                                                                                                                                                                                                                                       | Separate vanilla and fine-tuned scenarios                                                                                       |
| 192     | DGX Spark matrix peaks (`src/sim/hw/presets/nvidia-datacenter.ts:158`) | FP4 500 dense / 1000 sparse; BF16 125                                                    | suspicious | 1 PFLOP FP4 is the sparse figure (correct). BF16 125 is derived; measured 77–99 TF (F4)                                                                                                                                                                                                                                                                                                                           | Publish an interval; add a sustained scenario                                                                                   |
| 96 ff.  | Apple M3/M4 GPU peaks (`src/sim/hw/presets/apple.ts:436` etc.)         | e.g. M3 Ultra 80c 28.6 TF fp16                                                           | suspicious | Clocks are extrapolated from other SKUs; Apple publishes no GPU FLOPS                                                                                                                                                                                                                                                                                                                                             | Microbenchmark per SKU                                                                                                          |
| 34 ff.  | Apple M5 GPU Neural Accelerator peaks (`apple.ts:490` etc.)            | e.g. M5 Ultra 80c 132.7 TF fp16                                                          | suspicious | Community-derived per-core rate; Ultra clock assumed from M5 Max. It is the default engine, which overestimates engines without NA kernels, such as llama.cpp Metal (F4)                                                                                                                                                                                                                                          | Keep as an estimate; flag the engine choice in the UI                                                                           |
| 2 ff.   | device capacity (every dGPU)                                           | marketed GB read as 10⁹ B                                                                | suspicious | Physical GDDR/HBM is binary (≈ 7.4% more bytes). Conservative for fit; no fixed-batch tok/s effect                                                                                                                                                                                                                                                                                                                | Binary module bytes, with a separate usable reserve (F9, F11)                                                                   |

**Status of the acceptance rows** (fix rank 5): each acceptance preset in `src/sim/engine/specdec.ts` now records
every τ it is fitted to, with source, method version, target, drafter, k and temperature, and whether it is measured,
borrowed or carried over ([simulation-model.md §13 "Acceptance defaults"](../simulation-model.md#acceptance-defaults)).

- **542** fixed: `dflash2/agentic` is `basis: 'borrowed'` (DFlash v1, fitted at k 15), shown as "from DFlash v1"; α
  0.65 kept, as no DFlash 2 agentic measurement exists.
- **543** label fixed: `dspark/prose` is fitted to the DSpark paper's non-thinking Qwen3-4/8/14B rows (k 7) only;
  Inco's k 15 retrain (.796) and the community Qwen3.8-27B thinking retrain (τ 3.01) are cited, not fitted, so k 15 is flagged. α 0.76 stays: it
  is the method authors' controlled value at the default k, and the cited reports straddle it (.68–.80).
- **529** label fixed: `mtp_multi_step/code` is `basis: 'pooled'` (fitted at no k), noted as mixing the greedy
  llama.cpp run (α_iid .88) with T = 1 sources (.78); α 0.82 kept. `mtp_multi_step/reasoning` 0.70 → 0.68 (PR 27210 τ 2.47 at k 3).

The auditor's notable **non-findings**:

- The RTX 5090 INT8 838 / 1676 TOPS is published.
- The B300 dense/sparse FP4 14 / 18 PFLOP/s is a vendor figure, not a 2:1 typo.
- Spark's 273 GB/s and CX-7 25 GB/s one way are correct.
- MI355X FP6 = FP4 is valid.
- All 13 DDR channel calculations check out.
- The Samsung 9100 PRO QD256 × 32 rating condition is real.
- TB5 RDMA (7 µs) against TCP (45 µs) is consistent. The earlier inversion has not reappeared.

---

## §2 Real-world reports against the sim

### Sources and conventions

- **Data.** 105 reports: the 45 earlier entries in `src/sim/__sanity__/reports.ts` and 60 fresh ones.
  - The fresh entries come from 52 source rows of a 106-row collection.
    - Sources: localmaxxing.com (all 13,112 approved speed tests were pulled; previously used run IDs excluded), the llama.cpp, MLX
      and exo discussions, vendor blogs (NVIDIA, Apple ML research), r/LocalLLaMA, and community pages.
  - 54 collected rows were not encoded:
    - no device preset (e.g. RTX 5080, RTX A6000, M1/M4 Pro)
    - no fixture for the model
    - an unsupported method (EAGLE3, n-gram)
    - unknown quantisation
    - an unsupported topology (TP 3, eight switched Sparks)
- **Each entry records** the engine, quant, batch, context and source URL.
- **The metric** is the ratio report ÷ sim, per metric, for prefill and decode.
  - The decode context is the reported depth plus half the generated length (the harness midpoint).
  - The sim uses the most favourable honest settings, as described in `docs/sanity-check.md`.
- **The test.** `reports.test.ts` asserts `simulated ≥ 0.9 × reported`, i.e. a ratio ≤ 1.11. Cases that fail it are
  listed as exceptions with a reason.
- **181 metrics:**
  - 158 asserted
  - 6 known exceptions
  - 17 excluded (fresh short-prompt TTFT prefill rows, and one series of a low-reliability host)
  - The distributions use the 164 asserted and exception metrics.
- **Caveat on independence.** Source series are correlated, not independent samples. Several points share one host, one
  engine build, or one web page.
- **Unknown prompt lengths in decode.** 11 fresh decode rows (`F-W30`–`F-W33`, `F-W37`–`F-W39`, `F-W41`, `F-W42`,
  `F-LM31`, `F-LM34`) do not state a prompt length. They are simulated at a 1-token prompt, the most favourable context
  for the sim, so **their ratios are lower bounds**. 9 of the 11 fresh Spark decode metrics are of this kind, including
  every multi-node Spark row.
- **Classes are exclusive**, assigned in this order: spec-dec, then SSD, then DDR offload, then device/topology. "Spark"
  covers ×1, ×2 and ×4.

### Ratio distribution per setup class

All reports (earlier + fresh):

| class              | phase   | n   | min   | p25   | median | p75   | max   |
| ------------------ | ------- | --- | ----- | ----- | ------ | ----- | ----- |
| single GPU         | decode  | 22  | 0.145 | 0.455 | 0.716  | 0.743 | 0.822 |
| single GPU         | prefill | 16  | 0.008 | 0.122 | 0.154  | 0.285 | 0.760 |
| multi-GPU PCIe     | decode  | 10  | 0.204 | 0.263 | 0.311  | 0.505 | 0.686 |
| multi-GPU PCIe     | prefill | 6   | 0.063 | 0.080 | 0.147  | 0.233 | 0.482 |
| single Mac         | decode  | 11  | 0.115 | 0.314 | 0.422  | 0.708 | 0.855 |
| single Mac         | prefill | 8   | 0.155 | 0.483 | 0.653  | 0.745 | 0.795 |
| Mac TB5 cluster    | decode  | 3   | 0.645 | 0.714 | 0.783  | 0.814 | 0.845 |
| Mac TB5 cluster    | prefill | 3   | 0.462 | 0.484 | 0.506  | 0.635 | 0.765 |
| Spark              | decode  | 17  | 0.419 | 0.577 | 0.721  | 0.777 | 2.034 |
| Spark              | prefill | 6   | 0.105 | 0.190 | 0.271  | 0.332 | 0.354 |
| Strix Halo         | decode  | 10  | 0.411 | 0.543 | 0.712  | 0.742 | 0.845 |
| Strix Halo         | prefill | 7   | 0.114 | 0.198 | 0.215  | 0.228 | 0.243 |
| DDR expert offload | decode  | 11  | 0.065 | 0.079 | 0.173  | 1.140 | 3.648 |
| DDR expert offload | prefill | 8   | 0.045 | 0.063 | 0.086  | 0.144 | 0.544 |
| SSD offload        | decode  | 5   | 0.321 | 0.423 | 0.479  | 0.485 | 0.489 |
| SSD offload        | prefill | 3   | 0.153 | 0.387 | 0.625  | 0.810 | 0.995 |
| spec-dec           | decode  | 12  | 0.221 | 0.366 | 0.627  | 0.823 | 1.079 |
| spec-dec           | prefill | 6   | 0.098 | 0.309 | 0.501  | 0.571 | 0.954 |

Fresh reports only:

| class              | phase   | n   | min   | p25   | median | p75   | max   |
| ------------------ | ------- | --- | ----- | ----- | ------ | ----- | ----- |
| single GPU         | decode  | 6   | 0.369 | 0.616 | 0.682  | 0.722 | 0.755 |
| single GPU         | prefill | 3   | 0.277 | 0.292 | 0.307  | 0.343 | 0.378 |
| multi-GPU PCIe     | decode  | 2   | 0.204 | 0.227 | 0.251  | 0.274 | 0.297 |
| multi-GPU PCIe     | prefill | 0   | —     | —     | —      | —     | —     |
| single Mac         | decode  | 5   | 0.200 | 0.254 | 0.777  | 0.793 | 0.855 |
| single Mac         | prefill | 2   | 0.763 | 0.771 | 0.779  | 0.787 | 0.795 |
| Mac TB5 cluster    | decode  | 3   | 0.645 | 0.714 | 0.783  | 0.814 | 0.845 |
| Mac TB5 cluster    | prefill | 3   | 0.462 | 0.484 | 0.506  | 0.635 | 0.765 |
| Spark              | decode  | 11  | 0.419 | 0.558 | 0.593  | 0.749 | 0.812 |
| Spark              | prefill | 1   | 0.332 | 0.332 | 0.332  | 0.332 | 0.332 |
| Strix Halo         | decode  | 6   | 0.411 | 0.666 | 0.712  | 0.731 | 0.845 |
| Strix Halo         | prefill | 3   | 0.191 | 0.203 | 0.215  | 0.225 | 0.235 |
| DDR expert offload | decode  | 9   | 0.065 | 0.074 | 0.147  | 1.131 | 3.648 |
| DDR expert offload | prefill | 7   | 0.045 | 0.061 | 0.068  | 0.175 | 0.544 |
| SSD offload        | decode  | 5   | 0.321 | 0.423 | 0.479  | 0.485 | 0.489 |
| SSD offload        | prefill | 3   | 0.153 | 0.387 | 0.625  | 0.810 | 0.995 |
| spec-dec           | decode  | 9   | 0.221 | 0.367 | 0.534  | 0.819 | 0.841 |
| spec-dec           | prefill | 3   | 0.098 | 0.275 | 0.453  | 0.501 | 0.548 |

### Chart: report ÷ sim per metric

The chart uses a log scale. The dashed line at 1 is the bound. Black ticks are class medians.

```vega-lite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "report ÷ sim per metric (asserted and known exceptions; excluded metrics left out), by setup class, decode and prefill",
  "data": {
    "values": [
      {"id":"lm-01","class":"single GPU","phase":"prefill","ratio":0.23,"context":"2048","set":"earlier","check":"asserted"},
      {"id":"lm-01","class":"single GPU","phase":"decode","ratio":0.746,"context":"2304","set":"earlier","check":"asserted"},
      {"id":"lm-02","class":"single GPU","phase":"prefill","ratio":0.76,"context":"296","set":"earlier","check":"asserted"},
      {"id":"lm-02","class":"single GPU","phase":"decode","ratio":0.803,"context":"552","set":"earlier","check":"asserted"},
      {"id":"lm-04","class":"single GPU","phase":"prefill","ratio":0.133,"context":"8192","set":"earlier","check":"asserted"},
      {"id":"lm-04","class":"single GPU","phase":"decode","ratio":0.727,"context":"256","set":"earlier","check":"asserted"},
      {"id":"lm-05","class":"single GPU","phase":"prefill","ratio":0.196,"context":"512","set":"earlier","check":"asserted"},
      {"id":"lm-05","class":"single GPU","phase":"decode","ratio":0.562,"context":"64","set":"earlier","check":"asserted"},
      {"id":"lm-06","class":"single GPU","phase":"prefill","ratio":0.559,"context":"1037","set":"earlier","check":"asserted"},
      {"id":"lm-06","class":"single GPU","phase":"decode","ratio":0.735,"context":"1549","set":"earlier","check":"asserted"},
      {"id":"lm-07","class":"single GPU","phase":"prefill","ratio":0.142,"context":"347","set":"earlier","check":"asserted"},
      {"id":"lm-07","class":"single GPU","phase":"decode","ratio":0.376,"context":"859","set":"earlier","check":"asserted"},
      {"id":"lm-08","class":"single GPU","phase":"prefill","ratio":0.008,"context":"347","set":"earlier","check":"asserted"},
      {"id":"lm-08","class":"single GPU","phase":"decode","ratio":0.182,"context":"859","set":"earlier","check":"asserted"},
      {"id":"lm-09","class":"spec-dec","phase":"prefill","ratio":0.261,"context":"345","set":"earlier","check":"asserted"},
      {"id":"lm-09","class":"spec-dec","phase":"decode","ratio":0.811,"context":"601","set":"earlier","check":"asserted"},
      {"id":"lm-10","class":"single GPU","phase":"prefill","ratio":0.126,"context":"512","set":"earlier","check":"asserted"},
      {"id":"lm-10","class":"single GPU","phase":"decode","ratio":0.822,"context":"64","set":"earlier","check":"asserted"},
      {"id":"lm-11","class":"multi-GPU PCIe","phase":"prefill","ratio":0.253,"context":"573","set":"earlier","check":"asserted"},
      {"id":"lm-11","class":"multi-GPU PCIe","phase":"decode","ratio":0.571,"context":"829","set":"earlier","check":"asserted"},
      {"id":"lm-13","class":"single GPU","phase":"prefill","ratio":0.166,"context":"512","set":"earlier","check":"asserted"},
      {"id":"lm-13","class":"single GPU","phase":"decode","ratio":0.749,"context":"64","set":"earlier","check":"asserted"},
      {"id":"lm-15","class":"single GPU","phase":"prefill","ratio":0.111,"context":"343","set":"earlier","check":"asserted"},
      {"id":"lm-15","class":"single GPU","phase":"decode","ratio":0.721,"context":"471","set":"earlier","check":"asserted"},
      {"id":"lm-18","class":"single GPU","phase":"prefill","ratio":0.127,"context":"512","set":"earlier","check":"asserted"},
      {"id":"lm-18","class":"single GPU","phase":"decode","ratio":0.728,"context":"64","set":"earlier","check":"asserted"},
      {"id":"lm-20","class":"Strix Halo","phase":"prefill","ratio":0.243,"context":"349","set":"earlier","check":"asserted"},
      {"id":"lm-20","class":"Strix Halo","phase":"decode","ratio":0.814,"context":"477","set":"earlier","check":"asserted"},
      {"id":"lm-22","class":"single Mac","phase":"prefill","ratio":0.739,"context":"75","set":"earlier","check":"asserted"},
      {"id":"lm-22","class":"single Mac","phase":"decode","ratio":0.639,"context":"203","set":"earlier","check":"asserted"},
      {"id":"lm-23","class":"Spark","phase":"prefill","ratio":0.354,"context":"74","set":"earlier","check":"asserted"},
      {"id":"lm-23","class":"Spark","phase":"decode","ratio":0.683,"context":"202","set":"earlier","check":"asserted"},
      {"id":"lm-24","class":"multi-GPU PCIe","phase":"prefill","ratio":0.174,"context":"573","set":"earlier","check":"asserted"},
      {"id":"lm-24","class":"multi-GPU PCIe","phase":"decode","ratio":0.304,"context":"701","set":"earlier","check":"asserted"},
      {"id":"lm-27","class":"Strix Halo","phase":"prefill","ratio":0.204,"context":"345","set":"earlier","check":"asserted"},
      {"id":"lm-27","class":"Strix Halo","phase":"decode","ratio":0.491,"context":"473","set":"earlier","check":"asserted"},
      {"id":"lm-28","class":"single Mac","phase":"prefill","ratio":0.155,"context":"619","set":"earlier","check":"asserted"},
      {"id":"lm-28","class":"single Mac","phase":"decode","ratio":0.115,"context":"747","set":"earlier","check":"asserted"},
      {"id":"lm-32","class":"multi-GPU PCIe","phase":"prefill","ratio":0.12,"context":"278","set":"earlier","check":"asserted"},
      {"id":"lm-32","class":"multi-GPU PCIe","phase":"decode","ratio":0.349,"context":"761","set":"earlier","check":"asserted"},
      {"id":"lm-34","class":"single Mac","phase":"prefill","ratio":0.402,"context":"574","set":"earlier","check":"asserted"},
      {"id":"lm-34","class":"single Mac","phase":"decode","ratio":0.474,"context":"702","set":"earlier","check":"asserted"},
      {"id":"lm-35","class":"Spark","phase":"decode","ratio":1.601,"context":"637","set":"earlier","check":"exception"},
      {"id":"lm-36","class":"spec-dec","phase":"prefill","ratio":0.954,"context":"274","set":"earlier","check":"asserted"},
      {"id":"lm-36","class":"spec-dec","phase":"decode","ratio":1.079,"context":"786","set":"earlier","check":"asserted"},
      {"id":"lm-38","class":"multi-GPU PCIe","phase":"prefill","ratio":0.067,"context":"367","set":"earlier","check":"asserted"},
      {"id":"lm-38","class":"multi-GPU PCIe","phase":"decode","ratio":0.318,"context":"621","set":"earlier","check":"asserted"},
      {"id":"lm-39","class":"spec-dec","phase":"prefill","ratio":0.578,"context":"512","set":"earlier","check":"asserted"},
      {"id":"lm-39","class":"spec-dec","phase":"decode","ratio":0.355,"context":"1024","set":"earlier","check":"asserted"},
      {"id":"lm-42","class":"multi-GPU PCIe","phase":"prefill","ratio":0.482,"context":"345","set":"earlier","check":"asserted"},
      {"id":"lm-42","class":"multi-GPU PCIe","phase":"decode","ratio":0.557,"context":"473","set":"earlier","check":"asserted"},
      {"id":"web-01","class":"Spark","phase":"prefill","ratio":0.33,"context":"2048","set":"earlier","check":"asserted"},
      {"id":"web-01","class":"Spark","phase":"decode","ratio":0.768,"context":"16","set":"earlier","check":"asserted"},
      {"id":"web-02","class":"Spark","phase":"prefill","ratio":0.211,"context":"2048","set":"earlier","check":"asserted"},
      {"id":"web-02","class":"Spark","phase":"decode","ratio":0.749,"context":"32784","set":"earlier","check":"asserted"},
      {"id":"web-04","class":"Spark","phase":"prefill","ratio":0.105,"context":"4096","set":"earlier","check":"asserted"},
      {"id":"web-04","class":"Spark","phase":"decode","ratio":2.034,"context":"4112","set":"earlier","check":"exception"},
      {"id":"web-07","class":"Spark","phase":"prefill","ratio":0.183,"context":"2048","set":"earlier","check":"asserted"},
      {"id":"web-07","class":"Spark","phase":"decode","ratio":0.726,"context":"16","set":"earlier","check":"asserted"},
      {"id":"web-12","class":"single Mac","phase":"prefill","ratio":0.51,"context":"2048","set":"earlier","check":"asserted"},
      {"id":"web-12","class":"single Mac","phase":"decode","ratio":0.42,"context":"16","set":"earlier","check":"asserted"},
      {"id":"web-14","class":"single Mac","phase":"prefill","ratio":0.62,"context":"2048","set":"earlier","check":"asserted"},
      {"id":"web-14","class":"single Mac","phase":"decode","ratio":0.374,"context":"64","set":"earlier","check":"asserted"},
      {"id":"web-26","class":"single Mac","phase":"prefill","ratio":0.685,"context":"2048","set":"earlier","check":"asserted"},
      {"id":"web-26","class":"single Mac","phase":"decode","ratio":0.422,"context":"2112","set":"earlier","check":"asserted"},
      {"id":"web-29","class":"Strix Halo","phase":"prefill","ratio":0.114,"context":"512","set":"earlier","check":"asserted"},
      {"id":"web-29","class":"Strix Halo","phase":"decode","ratio":0.505,"context":"64","set":"earlier","check":"asserted"},
      {"id":"web-31","class":"Strix Halo","phase":"prefill","ratio":0.222,"context":"512","set":"earlier","check":"asserted"},
      {"id":"web-31","class":"Strix Halo","phase":"decode","ratio":0.745,"context":"64","set":"earlier","check":"asserted"},
      {"id":"web-36","class":"multi-GPU PCIe","phase":"prefill","ratio":0.063,"context":"50000","set":"earlier","check":"asserted"},
      {"id":"web-36","class":"multi-GPU PCIe","phase":"decode","ratio":0.252,"context":"50064","set":"earlier","check":"asserted"},
      {"id":"web-41","class":"multi-GPU PCIe","phase":"decode","ratio":0.243,"context":"768","set":"earlier","check":"asserted"},
      {"id":"web-42","class":"multi-GPU PCIe","phase":"decode","ratio":0.686,"context":"1088","set":"earlier","check":"asserted"},
      {"id":"web-44","class":"single GPU","phase":"decode","ratio":0.419,"context":"768","set":"earlier","check":"asserted"},
      {"id":"web-45","class":"single GPU","phase":"decode","ratio":0.653,"context":"768","set":"earlier","check":"asserted"},
      {"id":"web-48","class":"single GPU","phase":"decode","ratio":0.767,"context":"768","set":"earlier","check":"asserted"},
      {"id":"web-50","class":"single GPU","phase":"prefill","ratio":0.041,"context":"512","set":"earlier","check":"asserted"},
      {"id":"web-50","class":"single GPU","phase":"decode","ratio":0.145,"context":"64","set":"earlier","check":"asserted"},
      {"id":"web-51","class":"single GPU","phase":"prefill","ratio":0.057,"context":"512","set":"earlier","check":"asserted"},
      {"id":"web-51","class":"single GPU","phase":"decode","ratio":0.291,"context":"64","set":"earlier","check":"asserted"},
      {"id":"web-52","class":"DDR expert offload","phase":"decode","ratio":0.367,"context":"257","set":"earlier","check":"asserted"},
      {"id":"web-53","class":"DDR expert offload","phase":"prefill","ratio":0.104,"context":"4096","set":"earlier","check":"asserted"},
      {"id":"web-53","class":"DDR expert offload","phase":"decode","ratio":1.938,"context":"512","set":"earlier","check":"exception"},
      {"id":"F-LM01","class":"single GPU","phase":"decode","ratio":0.755,"context":"487","set":"fresh","check":"asserted"},
      {"id":"F-LM02","class":"single GPU","phase":"prefill","ratio":0.277,"context":"8192","set":"fresh","check":"asserted"},
      {"id":"F-LM02","class":"single GPU","phase":"decode","ratio":0.726,"context":"8448","set":"fresh","check":"asserted"},
      {"id":"F-LM03","class":"single GPU","phase":"prefill","ratio":0.307,"context":"32763","set":"fresh","check":"asserted"},
      {"id":"F-LM03","class":"single GPU","phase":"decode","ratio":0.653,"context":"33019","set":"fresh","check":"asserted"},
      {"id":"F-LM04","class":"single GPU","phase":"prefill","ratio":0.378,"context":"261095","set":"fresh","check":"asserted"},
      {"id":"F-LM04","class":"single GPU","phase":"decode","ratio":0.369,"context":"261351","set":"fresh","check":"asserted"},
      {"id":"F-LM05","class":"single GPU","phase":"decode","ratio":0.604,"context":"522","set":"fresh","check":"asserted"},
      {"id":"F-LM06","class":"single GPU","phase":"decode","ratio":0.711,"context":"522","set":"fresh","check":"asserted"},
      {"id":"F-LM10","class":"multi-GPU PCIe","phase":"decode","ratio":0.297,"context":"590","set":"fresh","check":"asserted"},
      {"id":"F-LM11","class":"spec-dec","phase":"decode","ratio":0.534,"context":"560","set":"fresh","check":"asserted"},
      {"id":"F-LM12","class":"spec-dec","phase":"prefill","ratio":0.548,"context":"17201","set":"fresh","check":"asserted"},
      {"id":"F-LM12","class":"spec-dec","phase":"decode","ratio":0.721,"context":"17457","set":"fresh","check":"asserted"},
      {"id":"F-LM13","class":"spec-dec","phase":"decode","ratio":0.819,"context":"230","set":"fresh","check":"asserted"},
      {"id":"F-LM13-pp2520","class":"spec-dec","phase":"prefill","ratio":0.098,"context":"2520","set":"fresh","check":"asserted"},
      {"id":"F-LM17","class":"spec-dec","phase":"decode","ratio":0.367,"context":"476","set":"fresh","check":"asserted"},
      {"id":"F-LM18","class":"multi-GPU PCIe","phase":"decode","ratio":0.204,"context":"472","set":"fresh","check":"asserted"},
      {"id":"F-LM21","class":"single Mac","phase":"decode","ratio":0.777,"context":"506","set":"fresh","check":"asserted"},
      {"id":"F-LM25","class":"single Mac","phase":"decode","ratio":0.2,"context":"187","set":"fresh","check":"asserted"},
      {"id":"F-LM26","class":"single Mac","phase":"decode","ratio":0.254,"context":"142","set":"fresh","check":"asserted"},
      {"id":"F-LM27","class":"spec-dec","phase":"decode","ratio":0.528,"context":"816","set":"fresh","check":"asserted"},
      {"id":"F-LM28","class":"spec-dec","phase":"decode","ratio":0.841,"context":"715","set":"fresh","check":"asserted"},
      {"id":"F-LM31","class":"Spark","phase":"decode","ratio":0.812,"context":"257","set":"fresh","check":"asserted"},
      {"id":"F-LM35","class":"Strix Halo","phase":"decode","ratio":0.727,"context":"463","set":"fresh","check":"asserted"},
      {"id":"F-LM36","class":"Strix Halo","phase":"decode","ratio":0.732,"context":"462","set":"fresh","check":"asserted"},
      {"id":"F-LM37","class":"Strix Halo","phase":"decode","ratio":0.698,"context":"1433","set":"fresh","check":"asserted"},
      {"id":"F-LM39","class":"spec-dec","phase":"decode","ratio":0.835,"context":"477","set":"fresh","check":"asserted"},
      {"id":"F-LM53","class":"spec-dec","phase":"decode","ratio":0.221,"context":"874","set":"fresh","check":"asserted"},
      {"id":"F-LM54","class":"spec-dec","phase":"prefill","ratio":0.453,"context":"620","set":"fresh","check":"asserted"},
      {"id":"F-LM54","class":"spec-dec","phase":"decode","ratio":0.365,"context":"876","set":"fresh","check":"asserted"},
      {"id":"F-W01-pp2048","class":"DDR expert offload","phase":"prefill","ratio":0.239,"context":"2048","set":"fresh","check":"asserted"},
      {"id":"F-W01-pp2048","class":"DDR expert offload","phase":"decode","ratio":1.15,"context":"64","set":"fresh","check":"exception"},
      {"id":"F-W01-pp8192","class":"DDR expert offload","phase":"prefill","ratio":0.112,"context":"8192","set":"fresh","check":"asserted"},
      {"id":"F-W01-pp16384","class":"DDR expert offload","phase":"prefill","ratio":0.058,"context":"16384","set":"fresh","check":"asserted"},
      {"id":"F-W01-pp32768","class":"DDR expert offload","phase":"prefill","ratio":0.068,"context":"32768","set":"fresh","check":"asserted"},
      {"id":"F-W07-p85","class":"DDR expert offload","phase":"decode","ratio":0.083,"context":"149","set":"fresh","check":"asserted"},
      {"id":"F-W07-p500","class":"DDR expert offload","phase":"decode","ratio":0.072,"context":"564","set":"fresh","check":"asserted"},
      {"id":"F-W07-p1400","class":"DDR expert offload","phase":"decode","ratio":0.074,"context":"1464","set":"fresh","check":"asserted"},
      {"id":"F-W07-p5500","class":"DDR expert offload","phase":"prefill","ratio":0.065,"context":"5500","set":"fresh","check":"asserted"},
      {"id":"F-W07-p5500","class":"DDR expert offload","phase":"decode","ratio":0.065,"context":"5564","set":"fresh","check":"asserted"},
      {"id":"F-W08","class":"DDR expert offload","phase":"decode","ratio":1.131,"context":"135","set":"fresh","check":"exception"},
      {"id":"F-W09-p87","class":"DDR expert offload","phase":"decode","ratio":0.173,"context":"151","set":"fresh","check":"asserted"},
      {"id":"F-W09-p5650","class":"DDR expert offload","phase":"prefill","ratio":0.045,"context":"5650","set":"fresh","check":"asserted"},
      {"id":"F-W09-p5650","class":"DDR expert offload","phase":"decode","ratio":0.147,"context":"5714","set":"fresh","check":"asserted"},
      {"id":"F-W10","class":"DDR expert offload","phase":"prefill","ratio":0.544,"context":"2048","set":"fresh","check":"asserted"},
      {"id":"F-W10","class":"DDR expert offload","phase":"decode","ratio":3.648,"context":"64","set":"fresh","check":"exception"},
      {"id":"F-W12","class":"SSD offload","phase":"prefill","ratio":0.625,"context":"512","set":"fresh","check":"asserted"},
      {"id":"F-W12","class":"SSD offload","phase":"decode","ratio":0.321,"context":"612","set":"fresh","check":"asserted"},
      {"id":"F-W13","class":"SSD offload","phase":"prefill","ratio":0.995,"context":"512","set":"fresh","check":"asserted"},
      {"id":"F-W13","class":"SSD offload","phase":"decode","ratio":0.479,"context":"612","set":"fresh","check":"asserted"},
      {"id":"F-W19","class":"single Mac","phase":"prefill","ratio":0.795,"context":"8192","set":"fresh","check":"asserted"},
      {"id":"F-W19","class":"single Mac","phase":"decode","ratio":0.793,"context":"8256","set":"fresh","check":"asserted"},
      {"id":"F-W20","class":"single Mac","phase":"prefill","ratio":0.763,"context":"8192","set":"fresh","check":"asserted"},
      {"id":"F-W20","class":"single Mac","phase":"decode","ratio":0.855,"context":"8256","set":"fresh","check":"asserted"},
      {"id":"F-W21","class":"Mac TB5 cluster","phase":"prefill","ratio":0.765,"context":"8192","set":"fresh","check":"asserted"},
      {"id":"F-W21","class":"Mac TB5 cluster","phase":"decode","ratio":0.845,"context":"8256","set":"fresh","check":"asserted"},
      {"id":"F-W22","class":"Mac TB5 cluster","phase":"prefill","ratio":0.506,"context":"8192","set":"fresh","check":"asserted"},
      {"id":"F-W22","class":"Mac TB5 cluster","phase":"decode","ratio":0.783,"context":"8256","set":"fresh","check":"asserted"},
      {"id":"F-W23","class":"Mac TB5 cluster","phase":"prefill","ratio":0.462,"context":"8192","set":"fresh","check":"asserted"},
      {"id":"F-W23","class":"Mac TB5 cluster","phase":"decode","ratio":0.645,"context":"8256","set":"fresh","check":"asserted"},
      {"id":"F-W30","class":"Spark","phase":"decode","ratio":0.778,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W31","class":"Spark","phase":"decode","ratio":0.721,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W32","class":"Spark","phase":"decode","ratio":0.577,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W33","class":"Spark","phase":"decode","ratio":0.625,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W37","class":"Spark","phase":"decode","ratio":0.554,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W38","class":"Spark","phase":"decode","ratio":0.563,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W39","class":"Spark","phase":"decode","ratio":0.531,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W40","class":"Spark","phase":"prefill","ratio":0.332,"context":"131072","set":"fresh","check":"asserted"},
      {"id":"F-W40","class":"Spark","phase":"decode","ratio":0.777,"context":"131584","set":"fresh","check":"asserted"},
      {"id":"F-W41","class":"Spark","phase":"decode","ratio":0.593,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W42","class":"Spark","phase":"decode","ratio":0.419,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-W45","class":"Strix Halo","phase":"prefill","ratio":0.215,"context":"512","set":"fresh","check":"asserted"},
      {"id":"F-W45","class":"Strix Halo","phase":"decode","ratio":0.656,"context":"64","set":"fresh","check":"asserted"},
      {"id":"F-W46","class":"Strix Halo","phase":"prefill","ratio":0.191,"context":"512","set":"fresh","check":"asserted"},
      {"id":"F-W46","class":"Strix Halo","phase":"decode","ratio":0.845,"context":"64","set":"fresh","check":"asserted"},
      {"id":"F-W47","class":"Strix Halo","phase":"prefill","ratio":0.235,"context":"512","set":"fresh","check":"asserted"},
      {"id":"F-W47","class":"Strix Halo","phase":"decode","ratio":0.411,"context":"64","set":"fresh","check":"asserted"},
      {"id":"F-LM07","class":"SSD offload","phase":"prefill","ratio":0.153,"context":"262004","set":"fresh","check":"asserted"},
      {"id":"F-LM07","class":"SSD offload","phase":"decode","ratio":0.489,"context":"262068","set":"fresh","check":"asserted"},
      {"id":"F-LM34","class":"SSD offload","phase":"decode","ratio":0.485,"context":"65","set":"fresh","check":"asserted"},
      {"id":"F-LM49","class":"SSD offload","phase":"decode","ratio":0.423,"context":"790","set":"fresh","check":"asserted"}
    ]
  },
  "transform": [{ "calculate": "random()", "as": "jitter" }],
  "facet": { "row": { "field": "phase", "type": "nominal", "title": null, "sort": ["decode", "prefill"] } },
  "spec": {
    "width": 600,
    "height": 300,
    "layer": [
      {
        "mark": { "type": "point", "filled": true, "size": 45, "opacity": 0.75 },
        "encoding": {
          "y": {
            "field": "class",
            "type": "nominal",
            "title": null,
            "sort": ["single GPU", "multi-GPU PCIe", "single Mac", "Mac TB5 cluster", "Spark", "Strix Halo", "DDR expert offload", "SSD offload", "spec-dec"]
          },
          "yOffset": { "field": "jitter", "type": "quantitative", "scale": { "domain": [-0.6, 1.6] } },
          "x": {
            "field": "ratio",
            "type": "quantitative",
            "title": "report ÷ sim (log scale; > 1 = sim below reality)",
            "scale": { "type": "log", "domain": [0.005, 6] }
          },
          "color": { "field": "check", "type": "nominal", "title": "check", "scale": { "domain": ["asserted", "exception"], "range": ["#4c78a8", "#e45756"] } },
          "shape": { "field": "set", "type": "nominal", "title": "source", "scale": { "domain": ["earlier", "fresh"], "range": ["circle", "triangle-up"] } },
          "tooltip": [
            { "field": "id", "title": "report" },
            { "field": "class" },
            { "field": "phase" },
            { "field": "ratio", "title": "report ÷ sim", "format": ".3f" },
            { "field": "context", "title": "context (tokens)" },
            { "field": "check" }
          ]
        }
      },
      {
        "mark": { "type": "tick", "color": "black", "thickness": 2, "size": 18 },
        "encoding": {
          "y": { "field": "class", "type": "nominal", "sort": ["single GPU", "multi-GPU PCIe", "single Mac", "Mac TB5 cluster", "Spark", "Strix Halo", "DDR expert offload", "SSD offload", "spec-dec"] },
          "x": { "aggregate": "median", "field": "ratio", "type": "quantitative" },
          "tooltip": [{ "aggregate": "median", "field": "ratio", "title": "median report ÷ sim", "format": ".3f" }]
        }
      },
      { "mark": { "type": "rule", "strokeDash": [4, 4], "color": "#888" }, "encoding": { "x": { "datum": 1 } } }
    ]
  }
}
```

### The gap, class by class

The ranges below are **observed**. The attributions are **hypotheses**: each names the effect that would explain the gap,
split into **engine** (out of scope by the project's rules) and **hardware/math** (cross-referenced to §3). Only where a
probe or a measurement is cited is the size of an effect estimated, and none of them has been validated end to end
against these rows. Ratios are quoted at the reported context.

- **Single GPU, decode.** Two populations.
  - **Consumer and workstation dGPUs, dense 27–32B** (n = 14, ≤ 33k context): 0.56–0.82, median 0.73.
    - Hypothesis: a sustained-bandwidth scenario (F2, ≈ 0.9 on dGPUs) would account for about a third of the gap; the
      rest is engine (dequant kernels, launch gaps, sampling).
    - This is the class where the sim is closest to observed rates.
  - **Datacenter GPUs at batch 1:**
    - MI300X with an 8B and a 70B model under llama.cpp: 0.145 and 0.291.
    - B200 with a 27B model: 0.18 under llama.cpp, 0.38 under vLLM.
    - H100 with a 32B model under vLLM: 0.42.
    - Hypothesis: the step is short (the sim gives 475.6 tok/s @ 859 for a 27B Q4 on a B200, ≈ 2.1 ms per token), so
      fixed per-kernel costs of 1–2 µs over ~1000 ops are the same size as the whole step (F1, mostly engine dispatch).
      **Not probed on these devices.**
  - At 261k context: 0.37 (`F-LM04`, 16.5 tok/s @ 261k). Hypothesis: engine attention kernels at extreme depth; the KV
    read is costed.
- **Single GPU, prefill.**
  - llama.cpp: 0.11–0.38. vLLM: 0.56–0.76. Hypothesis: the spread is engine (MMQ / dequant GEMM paths); the
    clock/power part (F4) is −0% to −12% on a 4090 at 450 W.
  - The earlier rows with ~350-token prompts (`lm-08` at 0.008, `lm-07` at 0.14 on a B200) are TTFT measurements dominated
    by fixed costs. They are still asserted, since they pass, but **they are not evidence about compute**. Do not read the
    class median 0.154 as a hardware gap.
- **Multi-GPU PCIe, decode: 0.20–0.69, median 0.31.** The lowest dense-device class.
  - Hypotheses:
    - Hardware: the hardware share of the small all-reduce latency (≈ 2–3 µs against the modelled ≈ 2 µs per all-reduce,
      F7). The measured 6.1–13.5 µs includes kernel launch, which is engine.
    - Engine: llama.cpp's layer split runs the GPUs one after another; vLLM TP pays a collective launch per layer; small
      or MoE models spread over 4–8 GPUs leave each GPU a very short step, where per-kernel costs dominate (F1).
  - Fresh rows `F-LM18` (0.204: 4× B65, vLLM, MoE) and `F-LM10` (0.297: 2×3090, gpt-oss-20b, llama.cpp) are of that
    last kind.
- **Multi-GPU PCIe, prefill: 0.06–0.48.** Hypothesis: F7's ideal-overlap assumption (−24% to −40% in the probes when
  TP prefill is comm-heavy), plus llama.cpp's short-prompt TTFT rows.
- **Single Mac, decode: bimodal.**
  - Dense: MLX 0.78–0.86, llama.cpp 0.78, Ollama 0.64. These sit close to the measured M3 Ultra bandwidth fraction (≈ 0.78,
    F2), which is consistent with, not proof of, bandwidth as the main residual.
  - MoE: 0.115–0.474.
    - Hypothesis: per-kernel dispatch (Metal ≈ 2.75 µs per dependent kernel, engine) plus F2. A probe derate of
      Qwen3-235B on an M3 Ultra with both gives 40.5 tok/s @ 4k, against a report of 25.6 tok/s @ 4k; the ≈ 0.63
      residual would be MLX gather and dequant (engine).
    - `lm-28` (0.115: Qwen3.8-Flash-Next MLX 4-bit, 14.8 tok/s @ 747) is far below even that and likely engine-specific.
- **Single Mac, prefill: 0.16–0.80.** MLX and Metal at 8k reach 0.76–0.80. Low values are short prompts and MoE.
  - On the M5 the default engine is the GPU Neural Accelerators. That is an overestimate for engines without NA kernels
    (F4).
- **Mac TB5 cluster: decode 0.65–0.85, prefill 0.46–0.77.** Three points, all dense Qwen3.8-27B BF16 under MLX JACCL
  (RDMA), with different parallelism:
  - `F-W21`: 2 nodes, **pipeline** parallel: 0.845 (decode at batch 1 gets no PP speed-up, and the sim claims none).
  - `F-W22`: 2 nodes, tensor parallel: 0.783. `F-W23`: 4 nodes, tensor parallel: 0.645.
  - Hypothesis: the TP 2 → 4 drop fits latency-bound TP decode (α per all-reduce, which the sim models) plus engine
    collective costs. The residual is F2 plus engine.
- **Spark, decode.**
  - Single Spark, dense: 0.68–0.81. Single Spark, MoE (gpt-oss-120b, GLM-4.5-Air): 0.59–0.77.
  - ×2 and ×4: 0.42–0.72. **Every multi-node fresh Spark row is simulated at a 1-token context** because its prompt
    length is unknown, so these ratios are lower bounds (9 of the 11 fresh Spark decode metrics).
  - Hypothesis: F2 (sustained 73–90% of 273 GB/s) covers most of the single-node gap; multi-node adds F7 and engine
    collectives.
  - Two exceptions above 1:
    - `web-04` (2.03, batch 32; 1.37 after the F5 fix): F5 is a partial contributor, not a full explanation (see the exceptions table).
    - `lm-35` (1.60): unresolved checkpoint layout.
- **Spark, prefill: 0.10–0.35, uniformly low.**
  - Hypothesis: GB10 clock throttling under sustained load is independently reported (F4). Library efficiency
    (measured BF16 77–99 of 125 TF) is a separate, engine-side factor. A ≈ 0.4 gap remains after both, attributed to
    the engine (llama.cpp on early Spark builds).
- **Strix Halo, decode.**
  - Dense: 0.75–0.85, close to the measured ≈ 0.83 bandwidth fraction (F2).
  - MoE: 0.41–0.73. Hypothesis: per-kernel costs at batch 1 (F1) plus Vulkan/ROCm kernel maturity.
- **Strix Halo, prefill: 0.11–0.24, the most consistently low class.**
  - Hypothesis, in two parts:
    - Clock: the sustained GPU clock is ≈ 2.0 of 2.9 GHz (×0.69). This is a documented power limit, the kind a
      sustained-clocks scenario would include (F4).
    - Library: hipBLASLt reaches 36.9 of 59.4 TF and rocBLAS about 5 TF. That is library efficiency, engine side.
  - After the clock limit alone the ratio would be ≈ 0.16–0.35; the remainder is engine.
- **DDR expert offload: not comparable as encoded.**
  - **Above 1 (four exceptions):** llama.cpp / ik_llama.cpp compute the offloaded experts on the CPU, an execution mode the
    sim does not support (F3). These are **mismatched comparisons**, not evidence that the transfer-mode cost formula is
    wrong.
    - `F-W01-pp2048`: 1.15, high reliability.
    - `web-53`: 1.94.
    - `F-W10`: 3.65, low reliability.
    - `F-W08`: 1.13, low reliability, and its whole-layer count looks mislabeled.
  - **Below 0.1:** the `F-W07` series, 1.4–1.8 tok/s against 21.7 tok/s simulated @ 149–5564.
    - It comes from one EPYC/DDR4 host that other DDR4 users beat by ~10×. Likely a misconfiguration (NUMA, threads).
    - **Do not read it as "DDR offload loses 90%"**: one host dominates the low tail.
  - `F-W09` (0.15–0.17) gives no host details.
  - Prefill (0.045–0.24 at pp2k–32k):
    - The sim streams each expert once per prompt.
    - **Hypothesis:** llama.cpp moves the weights per micro-batch (ubatch), so PCIe traffic scales with
      prompt ÷ ubatch. That is an engine choice, not hardware.
- **SSD offload.**
  - Three resident-weight/small-SSD rows now have ratios 0.42–0.49 after the merged review fixes.
  - **Resolved F13 (2026-11-09): input issue, not an SSD-bandwidth formula bug.** Corrected F-W12/F-W13 decode
    is 31.7 / 38.9 tok/s @612 vs reported 10.18 / 18.66 (ratios 0.321 / 0.479); both now asserted.
    The checkpoint is `antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q4.gguf`, 518,596,067,328 B.
    Routed experts use Q4_K 4.5 bpw; attention/shared/head Q8_0 8.5 bpw; token_embd F16.
    Engram tables are NVMe-resident 264-B interleaved rows, not device-resident.
  - Published cache data replaces uniform h=f: F-W12 h≈0.933 is derived from device bytes (OS-cache ambiguity);
    F-W13 h=0.9491 is measured by 12.216 misses / 240 lookups. One prompt after prefill seeding, not a new default.
    Sources and the full before/after table: [sanity-check F13 correction](../sanity-check.md).
  - Remaining gap is software: decode uses only 25–35% of drive bandwidth; fork changes cache/scheduling as well.
    Keep datasheet SSD rate for these bound checks: corrected F-W13 prefill 31.15 tok/s @512 vs 30.98 reported
    would fail under the optional 0.84 sustained scenario. Do not tune h or SSD fractions to match throughput.
- **Spec-dec, decode: 0.22–0.84, plus `lm-36` at 1.079.**
  - Hypothesis: the base-decode gap (F1, F2) plus verify-kernel efficiency.
  - Where the report gives an acceptance, it is used. Otherwise the preset α is used, and its provenance issues
    (§1, F10) add uncertainty in the overestimate direction.
  - The MoE verify was ≈ 19% too pessimistic on expert bytes under iid-uniform routing (F5, now fixed), which pushed the other way.

### Ratios above 1, and reliability caveats

| report         | class              | ratio | reported → simulated                 | reason (from `reports.ts`)                                                                                                                                                                                                                                                                    | finding                             |
| -------------- | ------------------ | ----- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `F-W10`        | DDR expert offload | 3.648 | 26.4 → 7.2 tok/s @ 64                | exception. EPYC 12-channel DDR5 computes the experts on the CPU. **Low-reliability** secondary page                                                                                                                                                                                           | F3                                  |
| `web-04`       | Spark              | 2.034 | 262.2 → 128.9 tok/s @ 4112, batch 32 | exception. Balanced routing reads all 128 experts per layer; the iid-uniform expected unique count is ≈ 82. **After the F5 fix**: 191.1 tok/s @ 4112, ratio **1.372**, still an exception (262.2 needs ≤ ≈ 55 of 128 experts per layer, uniform touches 81.7), so F5 is a partial explanation | F5 (partial; fixed, exception kept) |
| `web-53`       | DDR expert offload | 1.938 | 10 → 5.2 tok/s @ 512                 | exception. ik_llama.cpp CPU-computes the experts of layers 10–93 from DDR5                                                                                                                                                                                                                    | F3                                  |
| `lm-35`        | Spark ×4           | 1.601 | 74.6 → 46.6 tok/s @ 637              | exception, **unresolved**. Local GLM-5.3-Flash NVFP4 checkpoint of unknown layout. An all-linear NVFP4 layout would give a ratio of 0.84                                                                                                                                                      | open question                       |
| `F-W01-pp2048` | DDR expert offload | 1.150 | 26.9 → 23.4 tok/s @ 64               | exception. llama.cpp computes the offloaded experts on DDR5-5200. High reliability                                                                                                                                                                                                            | F3                                  |
| `F-W08`        | DDR expert offload | 1.131 | 0.9 → 0.8 tok/s @ 135                | exception. CPU whole-layer compute on DDR4. **Low reliability**; the source claims 32 layers for a 36-layer model                                                                                                                                                                             | F3                                  |
| `lm-36`        | spec-dec, Spark ×4 | 1.079 | 76 → 70.4 tok/s @ 786                | **asserted**: within the 10% margin. DeepSeek-V4.1-Flash, DSpark at the reported τ 2.85. F5's pessimistic MoE verify (6 tokens read `min(E, 6k)` = 36 experts) was a candidate; **after the F5 fix** the verify reads U(6) ≈ 34.0: 72.2 tok/s @ 786, ratio **1.053**, still asserted          | partly F5 (fixed); rest unexplained |

Reliability caveats:

- **Low-reliability DDR series.**
  - The `F-W07` / `F-W08` host (one web page) and `F-W10` (an SEO-style guide) dominate the DDR class's tails at both ends.
  - Fresh DDR medians (decode 0.147, prefill 0.068) are therefore not a hardware statement.
- **Short-prompt TTFT prefill.**
  - LocalMaxxing prefill is the client's prompt tokens ÷ TTFT.
  - With 230–350-token prompts it understates prefill 5–10×. The fresh rows of that kind are excluded (with the
    low-reliability host series, 17 excluded metrics in total).
  - The earlier rows of that kind remain asserted. They pass, but they confound the single-GPU and multi-GPU prefill medians
    (0.154, 0.147).
- **Unknown prompt lengths.** The NVIDIA multi-Spark prefill numbers (`F-W30`–`F-W33`) give no prompt length and were
  dropped. Their decode rows, and those of `F-W37`–`F-W39`, `F-W41`, `F-W42`, `F-LM31` and `F-LM34`, are simulated at a
  1-token context, so their ratios are lower bounds.
- **Re-encoded rows.** `F-LM13`'s prefill is now `F-LM13-pp2520` at its 2520-token prompt (0.098). `F-LM28` now uses
  the single-step MTP tail [0.85, 0.30, 0.10] instead of iid 0.85: 78.35 tok/s reported against 93.1 simulated @ 715
  (0.841, was 0.565). Acceptance modelling alone moved this row by half.
- **Corrected encoding.** `F-LM54` prefill was first encoded per-sequence. A batch-8 / TTFT cross-check supports an
  aggregate reading, which gives 0.453, now asserted.
  - The aggregation is inferred; the raw payload is not available.
- **`lm-36` at 1.079** passes only because of the 0.9 tolerance.

---

## §3 Flaws: what the sim leaves out or idealises

### Probe baselines

The sim is unmodified. Decode is batch 1. tg128 runs use the harness midpoint (@ 64).

| config                                  | prefill              | decode                           |
| --------------------------------------- | -------------------- | -------------------------------- |
| Llama-3-8B Q4_K_M, RTX 4090             | 10 830 tok/s @ pp512 | 215.3 tok/s @ 64                 |
| Llama-3.2-1B Q4_K_M, RTX 4090           | 71 172 tok/s @ pp512 | 1 243 tok/s @ 64                 |
| Qwen3-30B-A3B Q4_K_M, RTX 4090          | 16 564 tok/s @ pp512 | 502.6 tok/s @ 64                 |
| gpt-oss-120b MXFP4, DGX Spark           | 5 904 tok/s @ pp2048 | 75.5 tok/s @ 64                  |
| gpt-oss-120b MXFP4, Strix Halo (395)    | 5 024 tok/s @ pp2048 | 70.6 tok/s @ 64                  |
| Qwen3-235B-A22B MLX 4-bit, M3 Ultra 80c | 650 tok/s @ pp512    | 64.5 tok/s @ 64; 61.0 tok/s @ 4k |

### F1. No per-op latency term in decode (a sensitivity, not an established flaw)

- **Code.**
  - `src/sim/engine/opTime.ts`: `tensorOpTime` (≈ :143–178) and `vectorOpTime` (≈ :205–228) cost
    `max(F / peak, bytes / memBW)`, with no latency term.
  - The stage time is the plain sum of op times (`src/sim/engine/pipeline.ts`).
  - `docs/simulation-model.md` §9 excludes engine kernel-launch overhead.
- **Probe, decode @ 64:**
  - Llama-3-8B on a 4090: 355 dependent ops, 226 of them under 2 µs.
  - Qwen3-30B-A3B on a 4090: 723 ops, 530 under 2 µs (95 µs of a 1.99 ms step).
  - Effect of a floor:

  | model / device       | hardware floor 0.6 µs | 1 µs  | dispatch floor 1 µs | 2 µs | 3 µs     | 5 µs |
  | -------------------- | --------------------- | ----- | ------------------- | ---- | -------- | ---- |
  | Llama-3-8B, 4090     | −2.5%                 | −4.3% | −7%                 | −13% | −19%     | −28% |
  | Llama-3.2-1B, 4090   | −7.4%                 | −12%  | −18%                | −31% | −40%     | −53% |
  | Qwen3-30B-A3B, 4090  | −12%                  | −19%  | −27%                | −42% | **−52%** | −65% |
  | gpt-oss-120b, Spark  | −1.1%                 | −2.1% | −4%                 |      | −10%     |      |
  | Qwen3-235B, M3 Ultra | −3%                   | −5.1% | −8%                 |      | −22%     | −31% |
  - "Hardware floor" is `Σ max(t_op, L)` over the IR ops, with L ≈ the DRAM load-to-use latency. "Dispatch floor" adds
    `N_ops · τ`. Both are sensitivity models: IR ops are not all dependent DRAM round trips (engines fuse elementwise ops
    and overlap independent ones), so the op count overstates the dependency chain.
  - Combined with 90% bandwidth, Qwen3-30B-A3B gives 304 tok/s @ 64 at τ = 1.5 µs and 229 tok/s @ 64 at τ = 3 µs, against
    a report of 211–226 tok/s @ 64. That τ is mostly engine dispatch.

- **Classification.**
  - **Hardware part: plausible, unmeasured.** Little's law says a kernel cannot ramp to full bandwidth on a few KB, and a
    true dependency chain serialises. But no measurement isolates a per-dependent-op hardware floor. The 78%-of-bandwidth
    megakernel result is one fused implementation, not a hardware limit. Treat 0.5–1 µs as an assumption with an
    unmeasured lower bound.
  - **Dispatch part: engine.** CPU launch, command-processor and graph-node dispatch (1.3–2.9 µs) is engine/runtime
    overhead, outside the project's stated exceptions (spec-dec, offload). It is large in practice for small and
    fine-grained-MoE models at batch 1, but out of scope unless the user wants it modelled.
- **Repro.**
  - Hazy Research, ["Look Ma, No Bubbles!"](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles): launch ≈ 2.1 µs
    with streams, 1.3 µs with CUDA graphs. The megakernel reaches 78% of bandwidth; vLLM and SGLang reach ≤ 50%.
  - NVIDIA, [CUDA graphs](https://developer.nvidia.com/blog/cuda-graphs/).
  - [mpk-apple notes](https://github.com/jiazhihao/mpk-apple/blob/main/docs/research/apple-inference-systems.md): MLX pays
    2.75 µs per tiny dependent kernel.
- **Fix.**
  - (a) An optional per-device `dependentOpLatencySec` sensitivity setting, default 0, applied as `max(t, floor)` to
    ops on the dependency chain. Promote it to a default only after a microbenchmark measures it. Size S–M.
  - (b) A "typical dispatch floor" overlay: engine overhead, so out of scope unless the user asks for it. Size M.
- **Confidence.** High that per-kernel costs are a large real gap for small and MoE models at batch 1 (mostly engine).
  Low on the size of any hardware-only floor.

### F2. Decode uses the datasheet memory bandwidth (a sustained scenario is missing)

- **Code.**
  - `src/sim/engine/opTime.ts:55–60`: `engineMemBandwidthBytesPerSec = min(device peak, engine cap)`.
  - `:159`: `memorySec = bytes / memBW`.
  - Presets hold the vendor peak. No sustained fraction exists anywhere.
- **Probe.** Decode is fully bandwidth bound, so the effect is linear: ×0.9 → −10%, ×0.8 → −20%.
  - Qwen3-235B on the M3 Ultra moves −9.4% / −19.2% / −29% at ×0.9 / ×0.8 / ×0.7: 18% of its time is compute bound.
  - MoE prefill is bandwidth-sensitive too: gpt-oss-120b on a Spark moves −18% @ pp2048 at ×0.8.

  | device     | measured                                  | fraction | decode effect |
  | ---------- | ----------------------------------------- | -------- | ------------- |
  | H100       | STREAM 3.06–3.13 of 3.35 TB/s             | 91–93%   | −7% to −9%    |
  | DGX Spark  | clpeak 234–247, triad 200–213 of 273 GB/s | 73–90%   | −10% to −27%  |
  | Strix Halo | `rocm_bandwidth_test` ≈ 212 of 256 GB/s   | 83%      | −17%          |
  | M3 Ultra   | ≈ 635 of 819 GB/s                         | 78%      | −22%          |

- **Classification. A missing scenario, not a wrong bound.** The datasheet peak is the right theoretical ceiling.
  Refresh, bank conflicts and read/write turnaround are memory-system properties, but STREAM, clpeak and triad are
  specific access patterns, so their fractions are workload-specific "sustained" figures, not a proven tighter bound.
  - Consistent observation: dense-model ratios on Strix Halo (0.75–0.85) and the M3 Ultra (0.79) sit close to the
    measured fractions.
- **Repro.**
  - [Kempner STREAM on H100](https://github.com/KempnerInstitute/nvidia-hpc-benchmarks/blob/main/p4_nvidia_stream/README.md)
  - [Spark bandwidth measured](https://gpuwerk.com/dgx-spark/guides/dgx-spark-memory-bandwidth-measured/)
  - [Strix Halo](https://llm-tracker.info/AMD-Strix-Halo-%28Ryzen-AI-Max%2B-395%29-GPU-Performance)
  - [M3 Ultra](https://www.creativeworksofknowledge.com/en/cwk-quests/apple-silicon-quest/memory/fleet-table/)
- **Fix.** Add an optional, sourced `memory.sustainedBandwidthBytesPerSec` per preset, shown as a labelled sustained
  figure next to the peak headline. Size M; most of it is preset research.
- **Confidence.** High on the measurements; the choice of benchmark per preset is a judgement.

### F3. Unsupported CPU-compute offload mode (reports run in that mode are mismatched comparisons)

- **Code.**
  - `src/sim/engine/types.ts:88–92`: `mode: 'cpu_compute'` is "a designed hook, rejected by validation".
  - `src/sim/engine/transfer.ts:212` `transferTime`: every missed expert crosses PCIe.
  - `src/sim/hw/presets/links.ts:122–125`: the PCIe ceiling.
- **Probe.** gpt-oss-120b MXFP4 on a 4090, every routed expert in host RAM, decode @ 64:
  - Transfer mode gives 16.3 tok/s @ 64, whatever the DRAM: DDR5-6000 and DDR4-3200 give identical results, because
    PCIe 4.0 x16 binds. 1.90 GB per step at 31.5 GB/s = 60.5 ms.
  - A CPU-compute bound gives 42.6 tok/s @ 64 on DDR5-6000 (×2.61) and 24.5 tok/s @ 64 on DDR4-3200.
    - At 80% of DRAM bandwidth: 35 and 20 tok/s @ 64.
  - Reports on a 4090 + DDR5 are 23–40 tok/s (context not stated), i.e. ≈ 1.5–2.5 × the transfer-mode figure.
  - In `reports.ts`: `F-W01-pp2048` 1.15, `F-W08` 1.13, `web-53` 1.94, `F-W10` 3.65.
- **Classification. An unsupported execution mode.** The transfer-mode cost formula is not shown to be wrong: it bounds a
  different execution (weights streamed to the GPU). Reports where the CPU computes the experts are mismatched
  comparisons. This is the dominant consumer offload deployment, so the missing mode matters.
  - Transfer mode remains the right bound for prefill-sized batches, where llama.cpp also streams weights to the GPU.
- **Repro.**
  - [r/LocalLLaMA 1obmhh1](https://www.reddit.com/r/LocalLLaMA/comments/1obmhh1) (`--cpu-moe`, AM5 + 4090, ≈ 28 tok/s,
    context not stated)
  - [1ndit0a](https://www.reddit.com/r/LocalLLaMA/comments/1ndit0a) (25–40 tok/s, partial offload, context not stated)
  - [1r9mcjw](https://www.reddit.com/r/LocalLLaMA/comments/1r9mcjw) (23–25 tok/s, context not stated)
  - [llama.cpp discussion 15396](https://github.com/ggml-org/llama.cpp/discussions/15396) (`F-W01`)
- **Fix.**
  - Implement `cpu_compute`: the CPU share costs `max(bytes / B_ram, F / CPU_peak)`, plus a hidden-state round trip over
    PCIe per layer.
  - `HostComputeSpec` already exists (`src/sim/hw/types.ts`). Size M–L.
  - Each of the four exception rows would then need re-encoding in that mode and validating on its own; that they would
    pass is untested.
- **Confidence.** High that the mode is missing; medium on the probe's magnitude (sourced CPU peaks are needed).

### F4. Peak vs achievable matmul (sustained clocks, power limits)

- **Code.**
  - `src/sim/hw/efficiency.ts:53–102`: MMA padding and wave quantisation only.
  - Peaks are vendor figures, mostly at the boost clock. Exceptions: the Apple and Strix Halo rates are derived from
    community clock estimates, and DGX Spark's non-FP4 rates are derived from the FP4 figure. See
    [presets-audit.md](presets-audit.md).
  - §9 excludes throttling "beyond an explicit clockScale".
- **Probe.** Large prefill GEMMs already reach a model efficiency of 0.93–0.99. Measured large-GEMM rates mix two
  different things, separated here:

  | device     | independently supported clock / power limit          | library-efficiency measurement            | prefill effect if applied (compute bound) |
  | ---------- | ---------------------------------------------------- | ----------------------------------------- | ----------------------------------------- |
  | Strix Halo | sustained GPU clock ≈ 2.0 vs 2.9 GHz                 | hipBLASLt 36.9 of 59.4 TF                 | clock −31%; measured rate −38%            |
  | DGX Spark  | GB10 throttling under sustained load (varies by OEM) | bf16 77–99 of 125 TF                      | −21% to −38% (measured rate)              |
  | RTX 4090   | power-limited at 450 W                               | cuBLAS fp16, fp32 acc.: 145–163 of 165 TF | −0% to −12%                               |
  | H100 SXM   | none documented here                                 | MAMF bf16 80–86% of peak                  | −15% to −20% (measured rate)              |
  | MI300X     | none documented here                                 | MAMF 49–58%                               | −40% to −50% (measured rate)              |

- **Dtype paths.**
  - Fallbacks are conservative and warn (e.g. nvfp4 → fp8 on a 4090).
  - The M5's default engine is the GPU Neural Accelerators (66.4 TF fp16 on an M5 Max). That is right for MLX with NA
    kernels and an unflagged overestimate for llama.cpp Metal.
  - The harness's `accumulate: 'fp32'` halves the GeForce fp16 rate. llama.cpp's int8 MMQ path could therefore _exceed_ the
    sim's prefill. No asserted prefill ratio > 0.76 was seen.
- **Classification.** Two parts. Documented clock and power limits (Strix Halo's sustained clock, GB10 throttling) are
  hardware, and a labelled sustained-clocks scenario could include them. Library efficiency (hipBLASLt, rocBLAS, MAMF
  shortfalls) is engine side. Neither makes the peak wrong as a theoretical bound.
- **Repro.**
  - [MAMF table](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md)
  - [4090 half rate](https://github.com/juliensimon/geforce-half-rate)
  - [Spark TFLOPS](https://gpuwerk.com/dgx-spark/guides/dgx-spark-tflops-measured/)
  - [GB10 throttling by OEM](https://gpuwerk.com/dgx-spark/guides/gb10-sustained-performance-by-oem/)
  - [gfx1151 WMMA](https://www.reddit.com/r/ROCm/comments/1wmkzwn/gfx1151_strix_halo_low_wmma_performance/)
- **Fix.** An optional, sourced "sustained clocks" `clockScale` per preset, only where a clock or power limit is
  documented (the mechanism exists in `src/sim/hw/modifiers.ts`): Strix Halo ≈ 0.69 first. Spark only once a sustained
  clock is sourced. Do not encode library efficiency as a clock. Size S–M.
- **Confidence.** High for the Strix Halo clock; medium for GB10; library figures are engine-specific.

### F5. Balanced MoE routing at batch > 1 (pessimistic)

- **Status: fixed** (task `review-fixes-moe-expected-unique-experts`). The grouped expert GEMMs now read the weights
  of `U` experts (`GemmOp.distinctWeightMatrices`, `lower/moe.ts: moeRouting` → `neededExperts` →
  `engine/transfer.ts: uniqueExperts`, the one helper shared with offload); the compute keeps the balanced grid
  `a = min(E_loc, A_max)`, `mE = ceil(A_max/a)`. Documented as the iid-uniform expectation, not a bound
  (`docs/simulation-model.md` §5 "Expected distinct experts"). After (re-run probes, before → after, aggregate
  decode @ 2k): Qwen3-30B-A3B Q4_K_M on a 4090 b4 660 → 706, b8 712 → 837, b16 738 → 1034, b32 1273 → 1396,
  b64 1995 → 2013 tok/s @ 2048 (the b16 → b32 step ×1.73 → ×1.35); gpt-oss-120b on a Spark b4 113.6 → 117.9,
  b8 124.3 → 136.4, b16 130.4 → 160.1, b32 133.7 → 201.9, b64 256.5 → 290.6 tok/s @ 2048. The Qwen3-30B-A3B
  8-token verify reads 7.09 GB of expert weights instead of 8.79 (−19.3 %). End to end: `web-04` 128.9 → 191.1
  tok/s @ 4112 (ratio 2.034 → 1.372, still an exception: real routing must touch ≤ ≈ 55 of 128 experts per
  layer, uniform touches 81.7); `lm-36` 70.4 → 72.2 tok/s @ 786 (ratio 1.079 → 1.053, still asserted).

- **Code.**
  - `src/sim/lower/moe.ts` `moeRouting`: `a = min(E_loc, ceil(T·k / ep))` active experts, whose weights the grouped
    GEMMs read.
  - The expected-unique formula `U = E·(1 − (1 − k/E)^T)` exists in `src/sim/engine/transfer.ts`, but it is used only for
    offload.
- **Probe.** Each cell: sim aggregate rate → the rate when only the iid-uniform expected unique experts are read.

  | batch | Qwen3-30B-A3B Q4, 4090 (128 experts, top-8), tok/s @ 2k | gpt-oss-120b, Spark (top-4), tok/s @ 2k |
  | ----- | ------------------------------------------------------- | --------------------------------------- |
  | 4     | 656 → 702 (+7%)                                         | 113 → 117 (+4%)                         |
  | 8     | 707 → 832 (+18%)                                        | 124 → 136 (+10%)                        |
  | 16    | 731 → 1 023 (**+40%**)                                  | 130 → 159 (+23%)                        |
  | 32    | 1 242 → 1 359 (+10%)                                    | 133 → 201 (**+51%**)                    |
  | 64    | 1 892 → 1 908 (+1%)                                     | 255 → 288 (+13%)                        |
  - The sim's curve is flat from b8 to b16, then jumps ×1.7 at b32, an artefact of `a` saturating at E.
  - MoE spec-dec verify at k = 7 reads 64 experts against 51.6 expected: −19% expert bytes.
  - Field data: `web-04` (ratio 2.03). F5 is a plausible partial contributor only: +51% on 128.9 tok/s @ 4112 gives ≈ 195
    tok/s @ 4112, still below the reported 262.2 tok/s @ 4112. No end-to-end probe of that row was run.

- **Classification.** A deliberate, pessimistic approximation that can put the sim below a real run.
- **What `U(T)` is.** `U = E·(1 − (1 − k/E)^T)` is the **expectation under iid-uniform routing**, not a bound. Real
  routing is skewed and temporally correlated and can touch fewer experts; a batch can also touch more than the mean.
  Using `U(T)` makes the figure an expected estimate. A strict optimistic bound would use the fewest experts a batch can
  touch (k, if every token picks the same k), which is far too loose to be useful.
- **Fix.** Use `U(T)` (`U_dev` at ep > 1) for the expert weight bytes, labelled as the iid-uniform expectation; keep
  `mE = ceil(A / U)` for the compute shape. Optionally expose a routing-skew factor. Size S–M; it changes many fixtures.
- **Confidence.** High on the direction; medium on the size for real routing.

### F6. Softmax / SFU (exp) throughput is excluded

- **Code.**
  - `src/sim/engine/opTime.ts`: attention is a batched GEMM at the matmul peak.
  - `src/sim/ir/metrics.ts`: FLOP = QK + PV only.
  - §9 lists the SFU as excluded.
- **Probe.** One exp per (query, key, head), at SMs × 16/clk × boost (an estimate), plus ~5 FP32 ops per score.

  | config                     | overlapped @ pp8k | pp32k | pp128k | serial @ pp8k | pp32k        | pp128k       |
  | -------------------------- | ----------------- | ----- | ------ | ------------- | ------------ | ------------ |
  | Llama-3-8B fp8, B200       | −4%               | −10%  | −16%   | −19%          | −38%         | −50%         |
  | gpt-oss-120b (hd 64), B200 | −12%              | −31%  | −48%   |               |              | −61%         |
  | Llama-3-8B bf16/fp8, H100  | 0%                | 0%    | 0%     | −7% to −10%   | −18% to −23% | −29% to −33% |
  | Llama-3-8B fp16, 4090      |                   |       |        |               |              | ≤ −5%        |

  Decode is unaffected: it is bound by KV bandwidth, which is costed.

- **Classification. Hardware**, for datacenter long-context prefill: the tensor-to-SFU ratio is 256:1 on the H100 and wider on
  Blackwell.
- **Repro.**
  - [FlashAttention-3](https://arxiv.org/abs/2407.08608)
  - [FlashAttention-4](https://arxiv.org/abs/2603.05451): exp is the main forward bottleneck on the B200.
- **Fix.** Add `ComputeEngine.sfuOpsPerSec` and cost attention as `max(t_mma, t_exp + t_vec)`, the overlapped bound. Size
  S–M.
- **Confidence.** Medium-high; the SFU rates are estimates.
- **Status. Done** (task review-fixes-softmax-sfu-exp-cost-in-att). `ComputeEngine.sfuOpsPerSec` on every preset
  (sourced or estimated per family, `src/sim/hw/presets/sfu.ts`); attention costs `max(t_mma, t_exp + t_vec)`
  (`src/sim/engine/opTime.ts`, `docs/simulation-model.md` §7 "Attention softmax"). Measured, prefill tok/s before →
  after (default options): B200 Llama-3-8B fp8 204,388 → 196,178 @ pp8k (− 4.0 %), 129,382 → 116,359 @ pp32k
  (− 10.1 %), 52,151 → 44,092 @ pp128k (− 15.5 %); B200 gpt-oss-120b 73,922 → 38,360 @ pp128k (− 48.1 %); H100
  Llama-3-8B bf16 and fp8 unchanged at all three; serial sensitivity H100 bf16 − 6.7 / − 17.6 / − 29.3 %, B200 Llama
  − 19.4 / − 37.8 / − 49.5 %. Decode unchanged. FMA-emulated exp (FA4) is a documented limitation.

### F7. Small-message all-reduce α; ideal comm overlap in prefill

- **Code.**
  - `src/sim/hw/presets/links.ts:155,229`: PCIe `latencySec: 1e-6`.
  - Ring `2(n−1)α` (`docs/simulation-model.md` §6).
  - `src/sim/engine/pipeline.ts`: `'ideal'` = `lat + max(compute, bw)`, the default (`src/sim/engine/simulate.ts:55`).
- **Probe.**
  - **2×3090, TP2, Llama-3-8B Q8_0, P2P.** Decode 226.8 tok/s @ 64. α = 1.98 µs per all-reduce, 66 all-reduces = 131 µs of a
    4.41 ms step.
    - Re-costed at the measured 6.1 µs (one-shot): −6%. At 13.5 µs (NCCL): −15%. Both measurements include kernel
      launch (engine); the hardware-only share is estimated at ≈ 2–3 µs, close to the modelled 1.98 µs.
  - **The same box without P2P.** 25.8 µs per all-reduce: 167 tok/s @ 64.
  - **4×4090 TP4, Llama-3-70B Q4.** Prefill is comm bound (β 675 ms against compute 469 ms); `overlap: 'none'` gives
    **−40%**.
  - **Other `overlap: 'none'` prefill effects.** Spark pair: −31.5%. 2×3090, TP2, Qwen3-30B-A3B: −24%.
  - **PP at batch 1** is modelled correctly, with no phantom pipelining.
- **Classification.**
  - The α floor: the hardware part (a PCIe write, a flag and a fence) is ≈ 2–3 µs, an estimate. The rest of the measured
    6.1–13.5 µs is the collective's kernel launch (engine).
  - The ideal prefill overlap is a deliberate idealisation that vLLM, SGLang and llama.cpp do not achieve by default.
- **Repro.**
  - [PCIe one-shot all-reduce](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md)
  - [NVLink vs PCIe](https://blog.kkkzoz.top/posts/ml-systems/nvlink-vs-pcie-inference/)
- **Fix.**
  - (a) A sourced hardware-only PCIe P2P α (≈ 2–3 µs if confirmed). Size S. Effect on decode: a few percent.
  - (b) Show prefill with `'none'` beside `'ideal'` when exposed β ÷ compute > 0.2. Size S.
- **Confidence.** Medium.
- **Status (α part, fix 9a).** Not changed, by decision: no citable measurement isolates the hardware-only share of a
  small PCIe all-reduce. The one-shot 6.1 µs and NCCL 13.5 µs include the kernel launch and a system-scope barrier, and
  the ≈ 2–3 µs above is an estimate. The only hardware-only figure found is the patched P2P write latency
  (0.49–0.59 µs), which the 1 µs α already rounds up. Recorded by the α in `src/sim/hw/presets/links.ts`
  (`PCIE_ALPHA_NOT_RAISED`) and in `docs/simulation-model.md` §4. Fix 9b (show `overlap: 'none'` beside `'ideal'`)
  is out of scope of this change and still open.

### F8. PCIe ceiling excludes protocol overhead

- **Code.** `src/sim/hw/presets/links.ts:122–125`: `GT/s × lanes × 128/130 / 8` = 31.5 GB/s for Gen4 x16, "before packet
  overhead".
- **Probe.** Transfer-bound offload decode scales 1:1 with the link. gpt-oss-120b on a 4090, all experts in host RAM: 16.3 →
  13.5 tok/s @ 64 at the measured 26 GB/s (**−17%**).
- **Classification. Hardware**: TLP headers, ACK/FC DLLPs and the maximum payload size.
- **Repro.**
  - [NVIDIA forum](https://forums.developer.nvidia.com/t/blackwell-integer/320578?page=3) (26.0 GB/s on a 4090)
  - [vLLM offload bench](https://github.com/Pinned-Memory/vLLM-offload-bench/blob/main/RESULTS.md) (26.2 GB/s)
- **Fix.** × 256/(256 + 24) ≈ 0.91 payload efficiency, or a measured achievable figure. Size S.
- **Confidence.** High.
- **Status (fix 4): done.** `PCIE_PAYLOAD_EFFICIENCY` = 256 / (256 + 24) = 0.914 in `src/sim/hw/presets/links.ts`,
  applied once in `pcieBandwidthPerDirectionBytesPerSec` to every PCIe link: host bounce, direct P2P, the host link for
  offload, inter-node PCIe staging and the NVMe slot ceiling. The source is [Intel AN 855](https://cdrdv2-public.intel.com/666287/an855-683504-666287.pdf):
  6 dwords of TLP overhead without ECRC; the 256 B maximum payload size is assumed. DLLPs are left out, so it stays an
  upper bound above the measured 26.0–26.2 GB/s on Gen4 x16. The probe above moved 16.35 → 14.96 tok/s @ 64 (−8.5%).

### F9. Unified memory: no GPU-usable capacity limit

- **Code.**
  - `src/sim/engine/simulate.ts:466–469` only warns that "the usable fraction is OS-dependent and not modeled".
  - `src/sim/engine/memory.ts` compares the total against the full capacity.
- **Probe.**
  - Llama-3-70B MLX 8-bit on an M4 Max 128 GB @ 64k needs 96.5 GB. The sim says it fits. The default macOS working set
    is ≈ 96 GB, so it fails by default.
  - Qwen3-235B at 3.9 bpw @ 8k needs 118 GB (92%). That needs a raised `iogpu.wired_limit_mb`.
  - `maxSequencesPerReplica` is inflated the same way.
- **Classification. Flaw**: a platform limit, not an engine one.
- **Repro.**
  - [MLX `set_wired_limit`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
  - [Apple tech talk](https://developer.apple.com/videos/play/tech-talks/10580/)
- **Fix.** Add a per-device default GPU-usable fraction with a "raised limit" override, and report "fits by default / fits
  with override". Size S–M.
- **Confidence.** High for macOS, medium for Spark (119.7 GiB visible to CUDA) and Strix Halo (GTT configurable).
- **Status. Fixed** (rank 8). `MemorySpec.gpuUsableFraction` per preset (Apple 0.75, DGX Spark 119.7/128 = 0.935, Strix
  Halo 0.5 = the Linux GTT default, each cited), `HardwareStack.gpuUsableFraction` override (Hardware → Advanced →
  Memory fit → GPU memory limit, URL `gpu`); the fit checks `f · capacity` and warns when a stage fits only with a raised
  limit (docs/simulation-model.md §8 "GPU-usable capacity"). Probes re-run, before → after (rates unchanged):
  - Llama-3-70B MLX 8-bit, M4 Max 128 GB @ 64k: 96.5 GB fits of 128 GB (max 2 sequences) → does not fit 96.0 GB (max 0);
    fits at a raised 85% (108.8 GB, max 1).
  - Qwen3-235B 3.9 bpw @ 8k: 116.3 GB fits (max 8) → does not fit 96.0 GB; fits at 95% (121.6 GB, max 4).

### F10. Spec-dec: iid acceptance extrapolated across k

- **Code.** `src/sim/engine/specdec.ts` `expectedTokens`, iid `(1 − α^{k+1}) / (1 − α)`. The presets are fitted at k = 7 or 15;
  the UI allows k from 1 to 16.
- **Probe.**
  - **Illustrative only.** Under a hypothetical position-decaying acceptance (0.92, then −0.06 per position), an iid α
    fitted at k = 7 gives E[τ] 5.36 at k = 15 against 4.53 (+18%), and 3.05 against 3.34 at k = 3 (−9%). The decay curve
    is assumed, not measured.
  - The cost side is sound. The dense verify costs +0.3% to +0.8% over decode, with KV growth included. The MoE verify
    costs +139% (k = 3) and +325% (k = 7), which is real but was ≈ 19% too pessimistic (F5, now fixed).
- **Classification.** A math-model mismatch whose size depends on the real per-position curve. No external measurement
  was made.
- **Fix.** Store per-position vectors where the sources give them; otherwise warn when k differs from the fitted k. Size S.
- **Confidence.** Medium.
- **Status.** Fixed (warning). Each acceptance preset records the k of its measurements; the UI marks a preset α
  used at another k with "⚠ fitted at k = …" (`specdec.ts: presetDraftLengthExtrapolation`). The i.i.d. model is
  unchanged, so the probe numbers stand. Per-position curves exist for some sources (DFlash 2 blog Figure 5, MATH-500, Qwen3.5-4B; DSpark Figure 2) and are recorded in the docs, but not used: the presets stay i.i.d.

### F11. Memory fit: prefill activation peak (pessimistic); runtime reserve (optimistic)

- **Code.** `src/sim/engine/memory.ts` `stageActivationPeakBytes`: the largest single op over both phases, which scales with
  the prompt length.
- **Probe.**
  - Llama-3-8B Q4 on a 4090 with a 128k prompt: activations of 11.3 GB give "does not fit" (33.4 of 24 GB). The same KV at
    decode fits (22.2 GB), and engines chunk the prefill.
  - The CUDA context and buffers (~0.5–1.5 GB) are not counted.
  - KV per layer type is correct, including gpt-oss SWA.
- **Classification.** A modelling artefact (fit only; the rate is unaffected), plus an engine reserve.
- **Fix.** Compute the fit's activation peak on a bounded prefill chunk, and add an optional `runtimeReserveBytes`. Size S.
- **Confidence.** High.
- **Status. Fixed** (rank 12). The fit's activation peak is the decode step's or the LAST prefill chunk's
  (`Workload.prefillChunkTokens`, default 8192, sourced from SGLang / vLLM defaults; `memory.ts: prefillChunkShape`); an
  optional `HardwareStack.runtimeReserveBytes` (default 0: no vendor source; Hardware → Advanced → Memory fit, URL `rr`)
  enters every stage's total (docs/simulation-model.md §8 "Activation peak", "Runtime reserve"). Probe re-run, before →
  after (rates unchanged, 3348 tok/s @ pp131072 and 215.6 tok/s @ 1): Llama-3-8B Q4_K_M on a 4090 with a 128k prompt,
  activations 11.27 → 0.70 GB, 33.41 GB does not fit → 22.84 GB fits; with a 1 GB reserve 23.84 GB fits, with 1.5 GB
  24.34 GB does not.

### F12. EP: the busiest device gets the mean load

- **Status: fixed** (task `review-fixes-moe-expected-unique-experts`). `ParallelConfig.epImbalance` ρ (max/mean
  assignments per device, default `DEFAULT_EP_IMBALANCE` = 1.2 at ep > 1, 1 at ep = 1; engine-level, no UI) gives the
  busiest device `A_max = min(T_g·min(k, E_loc), max(A, ceil(ρ·T_g·k/ep)))` expert rows (ρ on the unrounded mean,
  capped by what the group's `T_g` tokens can route to one device, so sparse decode never gets work no token
  carries; FLOP, activations, expert-output norm, mode A combine; the prefill numbers below are the same under the
  earlier `ceil(ρ·A)`, whose mean there is whole and whose cap does not bind); the expert
  weights read stay at the mean share's `U` (hot experts draw more tokens, not more distinct experts). Default
  sources: TensorRT-LLM wide-EP blog Table 1 (DeepSeek-R1, EPLB with redundant slots 1.11 at EP32 / 1.12 at EP36,
  2.56 without), SGLang large-scale EP blog (balancedness ≈ 0.68 at 8 GPUs without EPLB). After (Qwen3-235B-A22B FP8,
  8× H100, prefill 64×4096, re-run): mode B tp 1 dp 8 ep 8 282,617 → 256,117 tok/s @ 4096 (−9.4 %; ρ 1.5: −20.3 %);
  tp 2 dp 4 ep 8 −6.9 % (ρ 1.5: −18.1 %); mode A tp 8 ep 8 unchanged at 165,859 (TP all-reduce bound). The
  review's −4.8 % / −11.3 % probe used a different split; it was not re-identified (`docs/simulation-model.md` §5
  "EP load imbalance").

- **Code.** `src/sim/lower/moe.ts` `moeRouting`: `A = ceil(T·k / ep)`.
- **Probe.** Qwen3-235B FP8 on 8×H100, ep 8, prefill 64×4096: 142k tok/s @ pp4096. A max/mean load of 1.2 gives −4.8%; 1.5
  gives −11.3%.
- **Classification.** A deliberate idealisation, partly mitigated by engines ([EPLB](https://github.com/deepseek-ai/EPLB)).
- **Fix.** An optional imbalance factor. Size S.
- **Confidence.** Medium.

### F13. Offload knobs barely move transfer-bound results — resolved (opt-in)

- **Code.** `src/sim/engine/transfer.ts:30` (prefetch 0.8), the pipeline's overlap term, and `transferTime`.
- **Probe.**
  - Prefetch accuracy 0 / 0.8 / 1: 15.6 / 16.3 / 16.5 tok/s @ 64 (±4%).
  - NVMe queue depth makes no difference for Qwen3-235B on a 4090 with experts on a 990 Pro: 0.96 tok/s @ 64 at any depth.
    Each expert is one large sequential read, so the result is bandwidth bound.
  - No double counting.
- **Classification.** Fine as an upper bound. The original unmeasured estimate was 70–85% of the SSD's sequential
  rate; the implemented scenario uses per-drive measurements instead (design doc §2.2–2.3).
- **Implemented.** `OffloadConfig.ssdReadRate` opts into sourced low-QD fractions; absent/datasheet stays byte-identical. Per-preset decode changes range −16% to −49%, not uniformly the estimated −15% to −30%. Tiny SSD rows remain unchanged. F-W12/F-W13 exceptions resolved as input issues (§2).
- **Confidence.** Medium.

**Reconstructed probes (2026-11-09).** Original probe configs were not retained. Qwen3-30B-A3B Q8_0 on a
4090, all routed experts in default DDR5-6000 ×2 RAM (f=0, uniform), prefetch 0 / 0.8 / 1 gives
15.518 / 16.178 / 16.352 tok/s @64 on the original branch. After rebasing the merged F8 PCIe payload fix,
the same probe gives 14.251 / 14.806 / 14.952 tok/s @64, identical in datasheet and sustained mode (RAM is unchanged by F13).
Qwen3-235B-A22B with 4.5-bpw weights (fp16 embedding/head), one 4090, experts on each SSD, f=0, uniform, default prefetch 0.8:
The SSD table is unchanged by the rebase; omitted and explicit datasheet results remain byte-identical.

| SSD | Before = after datasheet | After sustained | Sustained/datasheet |
| --- | --- | --- | --- |
| 990 PRO | 0.927060 | 0.600264 | 0.647 |
| SN850X | 0.908161 | 0.679559 | 0.748 |
| 9100 PRO | 1.830071 | 1.322379 | 0.723 |
| T705 | 1.793439 | 1.163716 | 0.649 |
| Generic Gen4 | 0.871082 | 0.581400 | 0.667 |
| Apple M3/M4 Max | 0.795772 | 0.407299 | 0.512 |
| Apple M5 Pro/Max | 1.617307 | 1.361659 | 0.842 |
| DGX Spark | 1.515868 | 0.989223 | 0.653 |

All rates are decode tok/s @64, identical at deployment QD1 / QD8 / default. Workload: batch 1, prompt 128,
generated 128, context starts empty (midpoint 64); `UPPER_BOUND_OPTIONS`, fp16 activations, fp32 accumulate.
The 990 PRO reconstruction is 3.4% below the review's 0.96; compare changes against the reproduced baseline,
not a guessed exact original config. Corrected SSD sanity rows and their sources are in [sanity-check](../sanity-check.md).

### F14. Apple GEMV path pads M = 1 to the 8×8×8 MMA

- **Code.** `src/sim/hw/presets/apple.ts` `simdTiling`.
- **Probe.** Qwen3-235B on the M3 Ultra: 0.51 ms of compute-bound excess in a 15.5 ms step (−3.3%), mostly the router.
- **Classification.** A minor underestimate. MLX `qmv` uses SIMD FMA at M = 1.
- **Fix.** A GEMV path with no MMA padding for m ≤ 4. Size S.
- **Confidence.** Medium.
- **Status. Done** (task review-fixes-softmax-sfu-exp-cost-in-att). `ComputeEngine.gemvMaxRows` = 1 on the Apple
  engines (MLX `gemv` / `qmv` and llama.cpp `mul_mv` both run m = 1 on the vector ALUs; `docs/simulation-model.md` §4
  "GEMV path"). Qwen3-235B-A22B MLX 4-bit decode, M3 Ultra 80c: 63.80 → 65.74 tok/s @ 1k (+ 3.0 %); M4 Max 40c
  41.64 → 43.83 (+ 5.3 %); M5 Max 40c 49.10 → 49.28 (+ 0.4 %). Fine-grained 4–5 bpw MoE gains more (lm-28 129 →
  151 tok/s @ 747).

### Confirmed preset and catalog bugs (from §1)

- **Arc B580 catalog** (`src/sim/hw/presets/catalog.ts:424`). Every offered local link is faster than the card's PCIe 4.0 x8:
  - the default PCIe 5.0 x16 host bounce: 4×
  - PCIe 5.0 x8: 2×
  - PCIe 5.0 x16 direct: 4×

  The device preset is right (`src/sim/hw/presets/intel.ts:450`). Nothing clamps a catalog link to `hostInterface`. The
  automatic offload host link is unaffected.

  **Status: fixed (fix 1).** The catalog offers the B580 PCIe 4.0 x8 host bounce (default) and PCIe 4.0 x8 direct.
  `catalog.test.ts` now checks that no offered PCIe link is wider or newer than the device's `hostInterface`, for
  every device; it found only the B580 and B60 rows. The probe now gives 6.97 ms (host bounce) and 6.95 ms (direct),
  including the payload efficiency (6.37 / 6.35 ms without it).

- **Arc Pro B60 catalog** (`catalog.ts:423`) offers PCIe 5.0 x16 direct for a Gen5 x8 card (2×). The default is correct. **Fixed:** x8 only now.
- **DFlash2 agentic acceptance** (`src/sim/engine/specdec.ts:139`): DFlash v1 data, marked as measured. **Fixed**: now
  `basis: 'borrowed'`, shown as "from DFlash v1".
- **Thunderbolt 4 TCP** (`src/sim/hw/presets/links.ts:645`, suspicious, not proven): the cable rate rather than a
  negotiated host-to-host path. Up to 2×.
  **Status (fix 11): done, as a variant.** The sources do not support lowering the default. Two M1 Macs report a
  bonded 40 Gb/s link ("Mac to Mac: 40 Gb/s x1, link width 0x2", [Bergeron](https://chrisbergeron.github.io/2021/07/25/Ultra-fast-Thunderbolt-NAS-with-Apple-M1-and-Linux/)),
  so `TB4_TCP` stays at 5 GB/s, the hardware ceiling. Both 20 Gb/s tests are Mac ↔ Linux: sysfs 20.0 Gb/s (Bergeron)
  and "a 20 Gigabit/s link" (Alexis). `TB4_TCP_20G` (2.5 GB/s) is that case, offered as the M3 Max's second network.
  A 2-node 100 MB all-reduce takes 20.09 ms on `TB4_TCP` and 40.09 ms on `TB4_TCP_20G`.

### Items checked and found sound

- PP bubble at batch 1: no pipelining gain is claimed (`cycle = Σ t_s`).
- Causal prefill attention pairs are exact (`src/sim/ir/metrics.ts`).
- Decode KV reads grow with context: attention is 79% of the step @ 128k on a 4090.
- Peak fallbacks only move to wider, slower formats, and always warn.
- Dense spec-dec verify at k + 1 includes the KV growth and the draft cost.
- Embedding and LM head sit on the correct PP stages. SWA KV is `min(ctx, W)`.
- Elementwise fusion defaults to `'none'`, which is conservative: gpt-oss prefill on a Spark carries 12.8 GB of
  elementwise traffic, ≈ 14% of its time.
- GPU peak tables: no dense/sparse or one-way/two-way bandwidth mix-up found. The GeForce FP32-accumulate half rate is
  modelled.
- Spark 273 GB/s and CX-7 25 GB/s per direction are correct. TB5 RDMA and TCP latencies are consistent, with no inversion.
- DDR channel arithmetic: all 13 presets are correct.
- No double counting in offload. The DMA write into device memory is not charged, which is negligible on dGPUs.

---

## Proposed fix tasks, ranked

Ranked by (impact on trustworthiness) ÷ effort. Confirmed data errors and approximations that can put the sim below a
real run come first. Sustained scenarios and sensitivity settings are additions next to the unchanged peak headline.

| rank | title                                                                                                                                           | scope (files)                                                                                                                                       | expected effect on headline numbers                                                                                                                                                                                                        | risk                                                                                | size         |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------ |
| 1    | Fix the Arc B580 / B60 catalog links; test catalog links ≤ device `hostInterface`                                                               | `src/sim/hw/presets/catalog.ts`, a catalog test                                                                                                     | B580 multi-GPU comm 1.5–4× slower (correct); B60 x16 option removed. tok/s drop scales with the comm share                                                                                                                                 | low                                                                                 | XS–S         |
| 2    | **Done.** Expected-unique experts for MoE weight bytes at batch > 1 (F5)                                                                        | `src/sim/lower/moe.ts`, `src/sim/engine/transfer.ts` (`uniqueExperts`), `docs/simulation-model.md` §5, fixtures                                     | MoE decode at b4–b32 +4% to +51% (iid-uniform expectation, labelled as such); removes the b16→b32 kink; MoE spec-dec verify −19% expert bytes. Measured end to end: `web-04` ratio 2.03 → 1.37 (still an exception), `lm-36` 1.079 → 1.053 | medium: many fixtures move; an expectation, not a bound                             | S–M          |
| 3    | Implement the `cpu_compute` offload mode (F3)                                                                                                   | `src/sim/engine/types.ts`, `transfer.ts`, `pipeline.ts`, validation, `src/sim/hw/types.ts` `HostComputeSpec`, host CPU presets, UI knob, design doc | DDR-offload decode in that mode ×1.5–2.6 over transfer mode (probe). The four exception rows would be re-encoded in that mode; each needs validation                                                                                       | medium: needs sourced CPU peaks and a CPU/GPU overlap rule                          | M–L          |
| 4    | PCIe payload efficiency in link bandwidth (F8)                                                                                                  | `src/sim/hw/presets/links.ts`, docs                                                                                                                 | offload and host-bounce transfer-bound rates −9% (0.91 factor) to −17% (26 GB/s measured)                                                                                                                                                  | low                                                                                 | S            |
| 5    | Done: fix acceptance provenance; warn on k extrapolation (§1, F10)                                                                              | `src/sim/engine/specdec.ts`, UI warning                                                                                                             | labels only; a warning where k differs from the fitted k (illustrative size: +18% at k = 15, −9% at k = 3 under an assumed decay curve)                                                                                                    | low                                                                                 | XS–S         |
| 6    | Optional sustained-bandwidth scenario per preset (F2)                                                                                           | `src/sim/hw/types.ts`, every `src/sim/hw/presets/*.ts`, `opTime.ts`, UI, docs                                                                       | a labelled sustained figure next to the unchanged peak headline: decode −7% (H100) to −22% (M3 Ultra) / −27% (Spark low end)                                                                                                               | low code risk; benchmark choice per preset is a judgement                           | M            |
| 7    | Sustained-clock `clockScale` option where a clock/power limit is documented (F4)                                                                | `src/sim/hw/modifiers.ts`, presets (Strix Halo first; Spark once sourced)                                                                           | compute-bound prefill −31% on Strix Halo when selected; peak headline unchanged                                                                                                                                                            | low                                                                                 | S            |
| 8    | GPU-usable capacity on unified memory: fit by default vs with override (F9)                                                                     | `src/sim/hw/types.ts`, Apple/Spark/Halo presets, `src/sim/engine/memory.ts`, UI                                                                     | fit verdicts and max batch on Macs (≈ 75% default); no tok/s change                                                                                                                                                                        | low                                                                                 | S–M          |
| 9    | Hardware-only PCIe α if sourced; show prefill with `overlap: 'none'` when comm is exposed (F7)                                                  | `links.ts`, `simulate.ts` / UI                                                                                                                      | TP decode over PCIe a few percent; prefill second figure −24% to −40%                                                                                                                                                                      | low–medium (α source)                                                               | S            |
| 10   | **Done.** SFU / exp cost in attention (F6)                                                                                                      | `src/sim/hw/types.ts` (`sfuOpsPerSec`), `opTime.ts`, datacenter presets                                                                             | long-context datacenter prefill −4% to −48% (B200); H100 overlapped 0%                                                                                                                                                                     | medium: SFU rates partly estimated                                                  | S–M          |
| 11   | Thunderbolt 4 negotiated-path variant                                                                                                           | `links.ts`, catalog                                                                                                                                 | TB4 TCP link 5 → 2.5 GB/s where negotiated at 20 Gb/s                                                                                                                                                                                      | low                                                                                 | XS           |
| 12   | Chunked-prefill activation peak for fit; optional runtime reserve (F11)                                                                         | `src/sim/engine/memory.ts`                                                                                                                          | fit verdicts at long prompts (less pessimistic) and near-full dGPUs (more realistic)                                                                                                                                                       | low                                                                                 | S            |
| 13   | Optional per-op latency sensitivity setting (F1a), default 0                                                                                    | `src/sim/hw/types.ts`, `opTime.ts`, docs §7/§9                                                                                                      | none by default; at 0.6–1 µs: decode −1% to −4% (large dense) to −12% to −19% (Qwen3-30B-A3B on a 4090)                                                                                                                                    | medium: the hardware floor is unmeasured; needs a microbenchmark before any default | S–M          |
| 14 | **Done:** opt-in sustained NVMe fraction (F13); F-W12/F-W13 re-encoded and asserted | `storage.ts`, `transfer.ts`, `sustained-read.test.ts`; see §2 | default unchanged; sustained SSD-bound decode −16% to −49% | low-medium (scenario, not proven bound) | S + research |
| 15   | **Done.** EP imbalance factor (F12), default 1.2                                                                                                | `src/sim/lower/moe.ts`                                                                                                                              | EP prefill −5% to −11% at max/mean 1.2–1.5 (re-run on Qwen3-235B FP8, 8× H100: −6.9% to −9.4% at 1.2; 0% where the TP all-reduce binds)                                                                                                    | low                                                                                 | S            |
| 16   | **Done.** Apple GEMV path without MMA padding (F14)                                                                                             | `src/sim/hw/presets/apple.ts`, `src/sim/hw/efficiency.ts`                                                                                           | Apple MoE decode +≈ 3%                                                                                                                                                                                                                     | low                                                                                 | S            |
| 17   | Dispatch-floor overlay (F1b): engine overhead, out of scope unless the user wants it                                                            | `simulate.ts` / result, UI                                                                                                                          | informational only: −13% to −52% on small and fine-grained MoE at batch 1; headline unchanged                                                                                                                                              | medium: outside the project's stated exceptions                                     | M            |

### Fix status

| rank | status                                                                                    |
| ---- | ----------------------------------------------------------------------------------------- |
| 1    | done: catalog links clamped to `hostInterface` by a test (B580, B60 fixed)                |
| 2    | done: expected distinct experts for the expert weights (F5 status)                        |
| 4    | done: PCIe payload efficiency 256/280 (F8 status)                                         |
| 9    | α: unchanged, no hardware-only measurement (F7 status); the overlap display is still open |
| 11   | done: `TB4_TCP_20G` variant; default kept at 5 GB/s (Thunderbolt 4 status above)          |
| 14 | done: opt-in sustained NVMe rate; corrected F-W12/F-W13 both asserted (F13 status) |
| 15   | done: EP load imbalance ρ, default 1.2 (F12 status)                                       |

---

## Open questions and uncertainty

- **What the headline is.** The datasheet peak stays the theoretical headline. Open: whether the sustained scenarios
  (F2, F4) are shown next to it by default, and which benchmark defines "sustained" per preset.
- **F1's hardware floor.** No measurement isolates a per-dependent-op hardware floor; the 0.5–1 µs figure is an
  assumption, and IR op counts overstate the dependency chain. The dispatch part (1.3–2.9 µs) is better measured but is
  engine overhead.
- **Datacenter batch-1 gaps** (0.15–0.42) are attributed to F1 by estimate only. No probe was run on B200 or MI300X.
- **SSD rows resolved (F13).** Checkpoint/layout/table placement and prompt-specific custom hits are now encoded; see §2 and [sanity-check](../sanity-check.md). Unpublished internal SSD capacity and upstream OS-cache contribution remain caveats, not unexplained exceptions.
- **The sample.** The ratio ranges are descriptive over a selected, correlated sample; 11 fresh decode rows are lower
  bounds (context 1). Class attributions in §2 are hypotheses.
- **The DDR offload class** is dominated by a few correlated, partly low-reliability sources:
  - `F-W07`, `F-W08` and `F-W10` are low reliability.
  - `F-W09` gives no host details.
  - The CPU-compute magnitude (F3) rests on the probe and Reddit reports more than on `reports.ts`.
- **The llama.cpp offload prefill gap** (0.045–0.24) is explained by ubatch re-streaming as a hypothesis only.
- **Unresolved rows.**
  - `lm-35` (1.60): checkpoint layout unknown.
  - `lm-36` (1.079; 1.053 after the F5 fix): the rest is unexplained.
  - `web-04` (2.03; 1.37 after the F5 fix): F5 explains part of it.
- **Acceptance defaults** transfer across targets, temperatures and thinking modes. No universal measured α exists.
  - The +23% DSpark example is one target; the error could be smaller or larger elsewhere.
- **Compute peaks.** Apple M3/M4 GPU clocks and M5 Ultra NA rates are extrapolated. Spark BF16 125 TF is derived. SFU rates
  (F6) are sourced for NVIDIA H100/H200, B200, B300 and GeForce (CUDA guide, Blackwell Ultra blog) and the 2× of
  AMD CDNA4; the rest (GB10, AMD CDNA3 / RDNA / MI455X, Intel, Apple) are estimates (`src/sim/hw/presets/sfu.ts`).
- **Capacity units.** Decimal GB for binary modules is conservative by ≈ 7.4%. It interacts with F9/F11 in fit verdicts.
