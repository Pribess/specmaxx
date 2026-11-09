# Sanity check: simulated maxima vs community benchmarks

The simulator gives **theoretical maxima** (only hardware and math limits, see
[simulation-model.md §1](simulation-model.md#1-scope-and-conventions)). A real measurement must therefore come
out below it: for every compared metric we require

```
simulated ≥ 0.9 × reported            (ratio = reported / simulated ≤ 1.11)
```

A report above that bound means the simulator underestimates (a wrong preset number, a wrong op cost, or a missing
hardware feature), unless the case is explained and listed as a known exception. A very low ratio is not a failure,
but it is examined too (an engine far from the hardware, or a report artefact).

Code: [`src/sim/__sanity__/`](../src/sim/__sanity__/)

| File | What |
|---|---|
| `harness.ts` | `simulateReport(input)`: model + weight quant + hardware (+ unlocks, + host link / RAM / SSD) + workload (+ deployment knobs: offload, speculative decoding) → `simulate()`; quant presets (GGUF, MLX, FP8, NVFP4, …); `simpleDecoder` for hand-written specs |
| `models.ts` | hand-written specs of sibling models the HF loader does not map (gpt-oss, GLM-4.5-Air, Qwen3-30B-A3B, Llama-2-7B, Llama-3.x-70B, QwQ-32B, Qwen3-Next-80B-A3B), dims from each repo's config.json |
| `reports.ts` | the 108 reports as typed data (the 45 described here, 60 added by the [practicality review](review/practicality-review.md), 3 CMP 170HX unlock stages), each with its harness input and its check mode |
| `reports.test.ts` | the assertion, the known-exception check, a weight-file-size cross-check, and the printed ratio table |
| `configs.testutil.ts` | loads HF configs from `__sanity__/configs/` or the shared fixtures |

Run: `bunx vitest run src/sim/__sanity__` (the comparison table and the ratio summary are printed).

## Method

### Settings: the most favourable honest upper bound

Nothing below was tuned to fit a report. Every choice is the one that gives the simulator the best case that is
still physically honest for the engine that ran.

| Setting | Choice | Why |
|---|---|---|
| Elementwise fusion | `'ideal'` (`UPPER_BOUND_OPTIONS`) | Real engines fuse norms, activations and residual adds into GEMM epilogues. The simulator's default `'none'` would make prefill lower than a fused kernel can reach ([§7 "Per-op time"](simulation-model.md#per-op-time)). |
| Comm / compute overlap, all-reduce | defaults (`'ideal'`, `'best'`) | Already the favourable case. |
| GEMM input, llama.cpp GGUF | `int8` (`LLAMACPP_MMQ`) on CUDA, HIP, Vulkan and SYCL | MMQ quantizes the activations to q8_1 and runs int8 MMA / dot products. The Vulkan and SYCL integer-dot paths do the same. On every device used here the int8 peak is ≥ the fp16 peak, so int8 also bounds the dequantize-to-fp16 path a backend may pick instead. Weight-only fp16 would put the RTX 4090 pp512 ceiling for an 8B Q4_K_M model at 10.7k tok/s, which the community beats. |
| GEMM input, Metal / MLX | the activation dtype (fp16 / bf16) | llama.cpp Metal and MLX dequantize inside fp16 / bf16 matmuls. |
| GEMM input, vLLM / SGLang | the checkpoint's own format | Native W8A8 FP8, W4A4 NVFP4 and W4A16 come from the quant fixture. On Ampere (no fp8 / fp4 tensor cores) the run is weight-only (Marlin): `weightOnly: true`, or the `fp8-wo` preset. |
| Accumulator | fp32 (the default) | No engine here asks for fp16-accumulate GEMMs. Marlin and MMQ accumulate in fp32 / int32. |
| Prefill metric | `perSequence` (prompt / TTFT) for batch 1, `aggregate` for batched reports | With a pipeline (layer split), `prefill.tokensPerSec` assumes a stream of batches, about pp × the per-sequence rate (see "Sim quirks"). |
| Decode metric | per stream at batch 1, aggregate when the report gives an aggregate | — |
| Decode context | `contextDepth + ceil(genLen / 2)` | The midpoint of the generated run. Decode time is about affine in the context, so this is the run's average rate. `contextDepth` includes the prompt wherever generation follows it (servers, vLLM, `mlx_lm.benchmark`, llama-batched-bench); only llama-bench's `tg` test starts from an empty cache (plus `-d`). The test checks this for every row. |
| Prefill at depth (`pp @ dN`) | simulated from an empty cache | `simulate` has no prefill at depth. Depth 0 is faster, so it is still an upper bound. |
| Unknown details | resolved in the simulator's favour | P2P driver when not stated, the faster of the stated link widths, the lower end of a bits-per-weight range, the largest memory option. |
| Unlocks | only when the report states them | lm-01 / lm-02: the run notes say "Zagitta/gpu-unlock", so the 3090 `tensor-rate` axis is on. |
| Speculative decoding, offload | simulated with the same knobs on (`input.deployment`, [simulation-model.md §12–§13](simulation-model.md#12-offloading)), then asserted like every other row | A spec-dec or offload run is bounded by the spec-dec / offload bound given its parameters, not by the plain autoregressive one. Parameters come from the report where it states them (lm-36 mean accepted length 2.85; web-52 cache size and 89 % hit rate); otherwise the method's prose acceptance preset (lm-09 MTP α 0.75, lm-39 DFlash 2 α 0.79). |
| Memory fit | not asserted (`SanityResult.fits`); the simulator defaults: GPU-usable share on unified memory, an 8192-token prefill chunk, no runtime reserve ([§8](simulation-model.md#gpu-usable-capacity)) | The check bounds rates, which these knobs never change (re-run 2026-11-09: every simulated rate identical). Fit verdicts moved: lm-27 (Strix Halo, 65.4 GB of the 64 GB default GTT share) and F-LM49 (218.7 GB of the 192 GB macOS share of 256 GB) no longer fit by default, which only says those runs raised the GPU limit (lm-27: a 96 GB VGM record); F-LM04 (RTX 4090, 22.4 GB with a chunked prefill instead of 48.8 GB) now fits. |

### Weight formats

GGUF and MLX have no exact dtype, so `bpwFormat(bpw)` stores them as `int4` (below 8 bpw) or `int8` with the rest
of the bits in `extraBitsPerElem`. Byte counts are exact; compute stays honest because these are weight-only schemes
(the GEMM runs in the activation or int8 format, [§2 "GEMM input class"](simulation-model.md#dtypes-math-and-accumulate)).

| Preset / format | bits / weight | Embeddings / LM head |
|---|---|---|
| `q4_0` | 4.5 | 4.5 / Q6_K 6.5625 |
| `q4_k_m` | 4.85 (Q4_K 4.5 with some Q6_K; ±0.1 by model) | Q4_K 4.5 / Q6_K 6.5625 |
| `q5_k_m` | ≈ 5.7 | 5.5 / 6.5625 |
| `q6_k` / `q8_0` | 6.5625 / 8.5 | same |
| UD / IQ GGUFs (`ggufBpw`) | the researcher's range, lower end: UD-Q4_K_XL 4.9, IQ4_XS 4.25, UD-IQ3_XXS 3.1, UD-Q2_K_XL 2.9 | same |
| `mlx-4bit` / `mlx-8bit` | 4.5 / 8.5 (affine g64: 16-bit scale + bias per 64) | quantized too |
| MLX 4-bit g32 | 5.0 | quantized too |
| gpt-oss GGUF | MXFP4 routed experts (4.25), everything else ≈ Q8_0 (8.5) | Q8_0 |
| `fp8`, `nvfp4`, `int4-g128`, native | from the dtype table / quant fixture | unquantized unless the repo says otherwise |

Cross-check: for the 6 reports that give a file size, the simulated resident weight bytes are 0.964–1.010 × the
file (the test requires ±8 %): lm-10 0.964, web-01 0.997, web-07 1.000, web-14 0.997, web-50 1.009, web-51 1.010.

Below 4 bpw (UD-IQ3_XXS, UD-Q2_K_XL), `extraBitsPerElem` is negative. Nothing validates it and the byte counts come
out right, but it is a workaround (follow-up 5).

### Data

- **LocalMaxxing** (`lm-NN` = the row in the researchers' curated table): records from
  `https://localmaxxing.com/api/speed-tests`, snapshot 2026-11-07/08. **Prefill there is promptTokens / client
  TTFT** over the ~220–350-token canonical prompt, with HTTP, tokenization and the first decode step included, so
  very low prefill ratios are expected. These rows test decode much better than prefill.
- **Web** (`web-NN` = the row in the web table): the llama.cpp DGX Spark benchmark file and GitHub discussions,
  r/LocalLLaMA, the mlx-lm BENCHMARKS.md, blogs and gists. llama-bench `ppN` / `tgN` are separate tests from an
  empty cache.
- Selection: every usable tier-A/B LocalMaxxing row (one or two per setup), the most informative tier-C rows (Apple,
  Spark, multi-GPU, the only DeepSeek and Inkling data), and web rows covering multi-GPU (3090 PCIe with and without
  P2P, 4090 ×6, 5090 ×2), Apple (M3 Ultra, M4 Max, M5 Max), DGX Spark (single and 4-node), Strix Halo, MI300X and
  H100, plus two expert-offload reports (web-52, web-53). **45 reports** (26 LocalMaxxing + 19 web), 38 prefill and
  45 decode numbers, **37 distinct (model, device) pairs** over 14 models and 16 device presets. Three LocalMaxxing
  rows ran with speculative decoding (lm-09, lm-36, lm-39) and two web rows with experts in host RAM.
- **Later additions.** The [practicality review](review/practicality-review.md) (§2) added 60 fresh reports in nine
  setup classes (single GPU, multi-GPU PCIe, single Mac, Mac TB5 clusters, DGX Spark ×1/2/4, Strix Halo, DDR and SSD
  expert offload, speculative decoding). `reports.ts` now holds **105 reports** and 181 metrics: 158 asserted, 6 known
  exceptions and 17 excluded. The tables and distributions below cover the original 45; the per-class distributions,
  the chart and the exceptions for all 105 are in that review's §2. Three more rows since (F-W48–F-W50, six asserted
  metrics): Llama-2-7B Q4_0 on the CMP 170HX, locked with the stock build, locked with the DP2A / `--fmad=false`
  build (`compute-path: locked-workarounds`) and unlocked by cmpunlocker (`compute-path: cmpunlocker`); report ÷ sim
  0.10–0.66.

- **F13 SSD correction (2026-11-09):** F-W12/F-W13 are now asserted, not exceptions. The checkpoint is
  `antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q4.gguf`, 518,596,067,328 B (HF file listing/header).
  Routed experts are Q4_K (4.5 bpw), attention/shared/head Q8_0 (8.5 bpw), embedding F16; Engram tables live
  on NVMe as 264-B interleaved rows. Custom hits are 0.933 (F-W12, derived from device bytes; OS-cache ambiguity)
  and 0.9491 (F-W13, 12.216 misses per 240 lookups, engine counter). Both describe one prompt after prefill seeding,
  not general routing defaults. This was an **input issue** (uniform h=f and tables on-device), not an SSD-bandwidth
  formula bug. Remaining decode overhead uses only 25–35% of the drive's bandwidth.
  Sources: [ds4 bd66c402](https://github.com/antirez/ds4/tree/bd66c402),
  [argodrive ladder](https://github.com/argonautlabsai/ds4-argodrive/tree/v41-stack-20260927/argodrive/candidates/2026-09-27-ladder),
  [checkpoint](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf).
  Keep these bound checks at datasheet SSD rate: F-W13 prefill is 31.15 tok/s @512 vs reported 30.98;
  the optional 0.84 sustained scenario would fail the bound check.

  | Row | Phase/context | Before sim (report/sim) | Corrected sim (report/sim) |
  | --- | --- | --- | --- |
  | F-W12 | prefill @512 | 28.9 (0.622) | 28.7 (0.625) |
  | F-W12 | decode @612 | 3.5 (2.879) | 31.7 (0.321) |
  | F-W13 | prefill @512 | 31.3 (0.989) | 31.1 (0.995) |
  | F-W13 | decode @612 | 3.8 (4.874) | 38.9 (0.479) |

  After rebasing the merged review fixes, decode is 31.667 / 38.920 tok/s @612 (previous F13 branch: 31.616 / 38.597).
  Both still pass without exceptions; the original before-column is retained as a historical baseline.
  Rates are tokens/s. Exact weight-file bytes are documented, not used as a resident-weight assertion:
  the harness's file-size check compares only device weights, excluding streamed tiers and including cache copies.
- Siblings (not shortlist models) are hand-written specs (`models.ts`) whose dims come from the real HF config.json;
  approximations (gpt-oss attention sinks and expert biases, GLM partial RoPE, the Qwen3-Next shared-expert gate) are
  listed on each spec. No 2-node DGX Spark report exists; the 4-node ones are used.

## Comparison table

Ratio = reported / simulated (≤ 1.11 passes). Bound = the simulator's dominant bound in prefill / decode
(c compute, m memory, t offload transfer). `agg` = aggregate over the batch. "B / p / g / d" = batch / prompt /
generated / depth.

| Report | Tier | Model | Quant | Hardware | Engine | B / p / g / d | pp rep | pp sim | pp ratio | tg rep | tg sim | tg ratio | Bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [lm-01](https://localmaxxing.com/en/models/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF?run=cmw13j1dv0aeioa01vc2babep) | A | Qwen3.8-27B | Q4_K_M, KV q4_0 | 1× RTX 3090 (gpu-unlock) | llama.cpp b12030 CUDA | 1/2048/512/2048 | 1284 | 5582 | 0.23 | 43.2 | 57.9 | 0.75 | c/m |
| [lm-02](https://localmaxxing.com/en/models/RedHatAI/Qwen3.8-27B-INT4?run=cmvu9h7cs0hbqs6017yhvimid) | A | Qwen3.8-27B | INT4 W4A16 g128 | 1× RTX 3090 (gpu-unlock) | vLLM 0.32.0 | 1/296/512/296 | 1938 | 2550 | 0.76 | 48.6 | 60.5 | 0.80 | c/m |
| [lm-04](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmszmr3yy0e75ms01z8t64io4) | B | Qwen3.8-27B | UD-Q4_K_XL, KV q8_0 | 1× RTX 4090 | llama.cpp b8101 CUDA (bench) | 1/8192/512/0 | 1559 | 11690 | 0.13 | 45.7 | 62.9 | 0.73 | c/m |
| [lm-05](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmsxftgtn0b99ms01ubcci7my) | B | Qwen3.8-27B | UD-Q4_K_XL | 1× RTX 5090 | llama.cpp b10470 CUDA (bench) | 1/512/128/0 | 3097 | 15818 | 0.20 | 62.8 | 112 | 0.56 | c/m |
| [lm-06](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-NVFP4?run=cmsxzbo3t0cnbms016n4fdk9x) | B | Qwen3.8-27B | NVFP4, KV fp8 | 1× RTX 5090 | vLLM 0.27.1 | 1/1037/1024/1037 | 9145 | 16365 | 0.56 | 71.1 | 96.8 | 0.73 | c/m |
| [lm-07](https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B-FP8?run=cmtpbv6tt02hin701q3p40kpp) | A | Qwen3.8-27B | FP8 | 1× B200 | vLLM 0.28.0 | 1/347/1024/347 | 9352 | 65698 | 0.14 | 106.2 | 282 | 0.38 | c/m |
| [lm-08](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmtpbzgly02htn701f9arg2ui) | A | Qwen3.8-27B | UD-Q4_K_M | 1× B200 | llama.cpp 74a7c89 CUDA | 1/347/1024/347 | 549 | 68218 | **0.008** | 86.8 | 476 | 0.18 | c/m |
| [lm-09](https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B-FP8?run=cmtrfjhhv00dcpa013a8uyn0j) | C | Qwen3.8-27B | FP8 | 1× H200 SXM | vLLM 0.24.1 (Pareton, MTP 8 drafts; sim k 8, α 0.75) | 1/345/512/345 | 7748 | 29706 | 0.26 | 274.9 | 339 | 0.81 | c/m |
| [lm-10](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmtm14eqs00wgoe01mmqnbok9) | B | Qwen3.8-27B | Q4_0 | 1× R9700 | llama.cpp b10711 Vulkan (bench) | 1/512/128/0 | 932 | 7403 | 0.13 | 34.9 | 42.5 | 0.82 | c/m |
| [lm-11](https://localmaxxing.com/en/models/amd/Qwen3.8-27B-Quark-AWQ-MXFP4?run=cmtx6ay860ai3ps011oye4gtx) | B | Qwen3.8-27B | MXFP4 W4A8, KV fp8 | 2× R9700 TP2 | vllm-radiance 1.0.16 ROCm | 1/573/512/573 | 3487 | 13767 | 0.25 | 45.2 | 79.2 | 0.57 | c/m |
| [lm-13](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmsxtoccy0cjlms01lnbw0pzu) | B | Qwen3.8-27B | Q6_K | 1× Arc Pro B60 | llama.cpp SYCL/Vulkan (bench) | 1/512/128/0 | 643 | 3877 | 0.17 | 16.0 | 21.4 | 0.75 | c/m |
| [lm-15](https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B?run=cmvrc4jz300j0s601ppkoyo08) | A | Qwen3.8-27B | Q8_0 | 1× Arc Pro B65 | llama.cpp b11913 | 1/343/256/343 | 419 | 3765 | 0.11 | 15.9 | 22.0 | 0.72 | c/m |
| [lm-18](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmsxndz0o0bxsms01v7pv9o2d) | B | Qwen3.8-27B | Q6_K | 1× Arc Pro B70 | llama.cpp SYCL/Vulkan (bench) | 1/512/128/0 | 905 | 7121 | 0.13 | 20.7 | 28.5 | 0.73 | c/m |
| [lm-20](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmvpm4oa7028vng011l44ex55) | A | Qwen3.8-27B | UD-Q4_K_M | Ryzen AI Max+ 395 | llama.cpp b11901 Vulkan | 1/349/256/349 | 262 | 1077 | 0.24 | 12.9 | 15.8 | 0.81 | c/m |
| [lm-22](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-GGUF?run=cmtmy9ugn004ppb01hljz9pcq) | C | Qwen3.8-27B | Q4_K_M | Apple M5 Max 40c | Ollama Metal | 1/75/256/75 | 882 | 1194 | 0.74 | 24.3 | 38.0 | 0.64 | c/m |
| [lm-23](https://localmaxxing.com/en/models/unsloth/Qwen3.8-27B-NVFP4?run=cmt5s1z0103m7pu01km9fczpa) | C | Qwen3.8-27B | NVFP4 | 1× DGX Spark | vLLM | 1/74/256/74 | 374 | 1055 | 0.35 | 10.1 | 14.8 | 0.68 | m/m |
| [lm-24](https://localmaxxing.com/en/models/Qwen/Qwen3.8-Flash-Next?run=cmw1704i40ajcoa01fwx8w8m9) | B | Qwen3.8-Flash-Next | UD-IQ3_XXS | 3× R9700 layer split | llama.cpp rdna-boosts ROCm | 1/573/256/573 | 1013 | 5812 | 0.17 | 47.5 | 156 | 0.30 | m/m |
| [lm-27](https://localmaxxing.com/en/models/unsloth/Qwen3.8-Flash-Next-GGUF?run=cmvpm4rw10292ng01k4wnsipr) | A | Qwen3.8-Flash-Next | UD-Q2_K_XL | Ryzen AI Max+ 395 | llama.cpp b11901 Vulkan | 1/345/256/345 | 298 | 1463 | 0.20 | 32.0 | 65.2 | 0.49 | m/m |
| [lm-28](https://localmaxxing.com/en/models/Qwen/Qwen3.8-Flash-Next?run=cmtexccjm00bgn0011lvaqdyi) | C | Qwen3.8-Flash-Next | MLX 4-bit g32 | Apple M3 Ultra 80c | oMLX 0.6.3 | 1/619/256/619 | 310 | 1999 | 0.15 | 14.8 | 151 | **0.098** | c/m |
| [lm-32](https://localmaxxing.com/en/models/nvidia/GLM-5.3-Flash-FP4-Local?run=cmvfggz3n014srf01ulz9qf41) | A | GLM-5.3-Flash | NVFP4 + FP8 (Marlin), KV fp8 | 8× RTX 3090 TP8/EP8 | vLLM 0.28.1 SM86 fork | 1/278/966/278 | 721 | 5920 | 0.12 | 81.3 | 233 | 0.35 | m/m |
| [lm-34](https://localmaxxing.com/en/models/zai-org/GLM-5.3-Flash?run=cmtipc711066gp401ddtg1b3p) | C | GLM-5.3-Flash | MLX 4-bit | Apple M3 Ultra 80c | oMLX 0.6.4 | 1/574/256/574 | 345 | 857 | 0.40 | 32.9 | 82.5 | 0.40 | c/m |
| [lm-35](https://localmaxxing.com/en/models/zai-org/GLM-5.3-Flash?run=cmum3byy10j02lq01b6821y8t) | B | GLM-5.3-Flash | NVFP4 (local checkpoint; simulated as the nvidia NVFP4 layout), KV fp8 | 4× DGX Spark TP4 | vLLM 0.30.1 Ray | 1/253/768/253 | — | — | — | 74.6 | 46.6 | **1.60 (exception, unresolved)** | –/m |
| [lm-36](https://localmaxxing.com/en/models/deepseek-ai/DeepSeek-V4.1-Flash?run=cmuhen2200drdlq01fa2qr70h) | C | DeepSeek-V4.1-Flash | FP8 + MXFP4 | 4× DGX Spark TP4 | SGLang (DSpark; sim k 5, E[τ] 2.85) | 1/274/1024/274 | 888 | 942 | **0.94** | 76.0 | 72.2 | **1.05** | m/m |
| [lm-38](https://localmaxxing.com/en/models/unsloth/Inkling-Small-GGUF?run=cms90mh5r0045pm01lf9n6cre) | B | Inkling-Small | UD-IQ4_XS | 6× RTX 3090 layer split | llama.cpp inkling-exp CUDA | 1/367/508/367 | 148 | 2222 | **0.067** | 47.4 | 149 | 0.32 | m/m |
| [lm-39](https://localmaxxing.com/en/models/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead?run=cmtx753t70aiups01lmudzstu) | C | Qwen3.8-27B | NVFP4 | 2× RTX 5090 TP2, P2P | SGLang (DFlash 2; sim k 7, α 0.79) | 1/512/1024/512 | 17563 | 30381 | 0.58 | 227.3 | 640 | 0.36 | c/m |
| [lm-42](https://localmaxxing.com/en/models/Qwen/Qwen3.8-27B?run=cmvjl3vpc03vcrf01vc9hojva) | C | Qwen3.8-27B | BF16 | 4× Arc Pro B65 TP4 | vLLM 0.32.0rc1 XPU | 16/345/256/345 | 239 | 494 | 0.48 | 372 agg | 660 | 0.56 | c/m |
| [web-01](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md) | — | gpt-oss-120b | MXFP4 GGUF | 1× DGX Spark | llama.cpp b7941 CUDA | 1/2048/32/0 | 2444 | 7416 | 0.33 | 58.7 | 76.5 | 0.77 | m/m |
| [web-02](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md) | — | gpt-oss-120b | MXFP4 GGUF | 1× DGX Spark | llama.cpp b7941 CUDA | 1/2048/32/32768 | 1567 | 7416 | 0.21 | 42.8 | 57.1 | 0.75 | m/m |
| [web-04](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md) | — | gpt-oss-120b | MXFP4 GGUF | 1× DGX Spark | llama-batched-bench B=32 | 32/4096/32/4096 | 2411 agg | 22916 | 0.11 | 262 agg | 191 | **1.37 (exception)** | c/m |
| [web-07](https://www.reddit.com/r/LocalLLaMA/comments/1o7k7zz/dgx_spark_compiled_llamacpp_benchmarks_compared/) | — | GLM-4.5-Air | Q4_K (5.45 bpw) | 1× DGX Spark | llama.cpp CUDA (Oct 2025) | 1/2048/32/0 | 841 | 4609 | 0.18 | 22.6 | 31.1 | 0.73 | m/m |
| [web-12](https://www.reddit.com/r/LocalLLaMA/comments/1o7k7zz/dgx_spark_compiled_llamacpp_benchmarks_compared/) | — | gpt-oss-120b | MXFP4 GGUF | Apple M4 Max 40c | llama.cpp Metal | 1/2048/32/0 | 871 | 1707 | 0.51 | 62.9 | 153 | 0.41 | c/m |
| [web-14](https://github.com/ggml-org/llama.cpp/discussions/15396) | — | gpt-oss-20b | MXFP4 GGUF | Apple M3 Ultra 80c | llama.cpp b6310 Metal | 1/2048/128/0 | 2816 | 4542 | 0.62 | 115.5 | 318 | 0.36 | c/m |
| [web-26](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md) | — | Qwen3-30B-A3B | MLX 4-bit | Apple M4 Max 40c 64 GB | mlx-lm 0.28.2 | 1/2048/128/2048 | 1754 | 2561 | 0.68 | 113.3 | 281 | 0.40 | c/m |
| [web-29](https://www.reddit.com/r/LocalLLaMA/comments/1m6b151/updated_strix_halo_ryzen_ai_max_395_llm_benchmark/) | — | Qwen3-30B-A3B | UD-Q4_K_XL | Ryzen AI Max+ 395 | llama.cpp b5863 Vulkan | 1/512/128/0 | 605 | 5311 | 0.11 | 66.3 | 131 | 0.51 | m/m |
| [web-31](https://www.reddit.com/r/LocalLLaMA/comments/1m6b151/updated_strix_halo_ryzen_ai_max_395_llm_benchmark/) | — | Llama-3.x-70B | i1-Q4_K_M | Ryzen AI Max+ 395 | llama.cpp b5863 HIP / Vulkan | 1/512/128/0 | 94.7 | 426 | 0.22 | 4.5 | 6.0 | 0.75 | c/m |
| [web-36](https://www.reddit.com/r/LocalLLaMA/comments/1x023z4/qwen38flashnext_on_6x3090_6x4090_without_nvlink/) | — | Qwen3.8-Flash-Next | UD-Q4_K_XL, KV q8_0 | 6× RTX 4090 layer split | llama.cpp df03399b8 stock | 1/50000/128/50000 | 1728 | 27140 | **0.064** | 46.2 | 184 | 0.25 | c/m |
| [web-41](https://www.reddit.com/r/LocalLLaMA/comments/1r66jyp/vllm_maximum_performance_on_multi3090/) | — | Qwen3-Coder-Next (80B-A3B) | FP8 (Marlin) | 4× RTX 3090 TP4, P2P | vLLM | 1/512/512/512 | — | — | — | 144.6 | 595 | 0.24 | –/m |
| [web-42](https://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html) | — | QwQ-32B | FP8 (Marlin) | 4× RTX 3090 TP4, no P2P | vLLM | 1/1024/128/1024 | — | — | — | 44.0 | 64.3 | 0.68 | –/m |
| [web-44](https://www.reddit.com/r/LocalLLaMA/comments/1jobe0u/benchmark_dualgpu_boosts_speed_despire_all_common/) | — | QwQ-32B | AWQ 4-bit | 1× H100 SXM | vLLM | 1/512/512/512 | — | — | — | 78.0 | 186 | 0.42 | –/m |
| [web-45](https://www.reddit.com/r/LocalLLaMA/comments/1jobe0u/benchmark_dualgpu_boosts_speed_despire_all_common/) | — | QwQ-32B | AWQ 4-bit | 1× RTX 5090 | vLLM | 1/512/512/512 | — | — | — | 65.0 | 99.6 | 0.65 | –/m |
| [web-48](https://www.reddit.com/r/LocalLLaMA/comments/1jobe0u/benchmark_dualgpu_boosts_speed_despire_all_common/) | — | QwQ-32B | AWQ 4-bit | 1× RTX 4090 | vLLM | 1/512/512/512 | — | — | — | 43.0 | 56.0 | 0.77 | –/m |
| [web-50](https://github.com/ggml-org/llama.cpp/discussions/14640) | — | Qwen3-8B (R1-0528 distill) | Q4_K_M | 1× MI300X | llama.cpp b5873–5898 HIP | 1/512/128/0 | 5840 | 142677 | **0.041** | 161.5 | 1116 | 0.14 | c/m |
| [web-51](https://github.com/ggml-org/llama.cpp/discussions/14640) | — | Llama-3.x-70B (R1 distill) | Q4_K_M | 1× MI300X | llama.cpp b5873–5898 HIP | 1/512/128/0 | 950 | 16745 | **0.057** | 36.3 | 125 | 0.29 | c/m |
| [web-52](https://github.com/ggml-org/llama.cpp/pull/29887) | — | Qwen3.8-Flash-Next | Q4_0 | 1× RTX 5090 on PCIe 4.0 x16, experts in host RAM (cache f 0.28, h 0.89) | llama.cpp PR 29887 `--moe-cache-mib 19000` | 1/1/512/1 | — | — | — | 67.8 | 171 | 0.40 | –/t |
| [web-53](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF) | — | Qwen3-235B-A22B | mix-IQ3_K (3.9 bpw) | 1× RTX 3090 Ti (3090 preset), experts of layers 10–93 in DDR5-6400 | ik_llama.cpp sweep-bench | 1/4096/1024/0 | 140 | 1227 | 0.11 | 10.0 | 4.7 | **2.12 (exception)** | t/t |

## Ratio distribution

This section covers the original 45 reports. For all 105, per setup class, see
[the practicality review §2](review/practicality-review.md).

Every metric is included: the three known exceptions (web-04, lm-35, web-53 decode) too. The spec-dec decodes
(lm-09, lm-36, lm-39) and the offload rows are compared with the spec-dec / offload bound.

| Phase | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| prefill | 38 | 0.008 | 0.123 | 0.200 | 0.390 | 0.942 |
| decode | 45 | 0.098 | 0.376 | 0.639 | 0.749 | 2.117 |

| Ratio bucket | [0, 0.1) | [0.1, 0.25) | [0.25, 0.5) | [0.5, 0.75) | [0.75, 0.9) | [0.9, 1) | ≥ 1 |
|---|---|---|---|---|---|---|---|
| prefill | 5 | 19 | 6 | 6 | 1 | 1 | 0 |
| decode | 1 | 3 | 14 | 17 | 6 | 0 | 4 |

What this shows:

- **Decode** clusters at 0.6–0.82 for clean, bandwidth-bound llama.cpp / vLLM single-GPU runs, which is what
  70–85 % of the memory bandwidth looks like in practice: lm-01, lm-02, lm-10, lm-20, web-31 and web-48 sit at
  0.75–0.82 on NVIDIA and AMD GPUs and APUs. (The researchers' Arc B70 Q6_K vs Q8_0 pair on one box scales with
  bytes too, at ≈ 75 % of 608 GB/s.) No asserted plain-decode ratio is above 0.82, so no single-device bandwidth preset
  looks too low; the plain-decode exceptions are a batched MoE run and a multi-node run with an unknown checkpoint.
- **Speculative decoding** (simulated with the knob on): lm-09 (MTP, 8 drafts) 0.81 and lm-39 (DFlash 2, 7 drafts)
  0.36 sit well inside the bound; lm-36 (DSpark on 4× Spark) is at 1.05, which passes but is flagged below.
- **Expert offload**: web-52 (all experts in host RAM behind a GPU cache) is 0.40 of the transfer-bound decode at the
  default prefetch accuracy p = 0.8 (0.35 with the former perfect prefetch; 0.37 / 0.32 before the PCIe payload efficiency 256/280); web-53
  computes its experts on the CPU, a mode the simulator does not cost, and is a known exception.
- **Prefill** is mostly 0.1–0.25. That is expected for LocalMaxxing, whose prefill is the client TTFT over a
  ~300-token prompt, and for llama.cpp on large GPUs. The highest clean prefill ratios are compute-bound Apple runs
  (0.62–0.74) and the vLLM INT4 3090 run (0.76).

## Flagged cases

### Reported above 0.9 × simulated (possible underestimate)

| Report | Ratio | Cause | Verdict |
|---|---|---|---|
| web-04 decode (gpt-oss-120b, Spark, B=32 aggregate) | 1.37 | **Simulator modelling.** The sim reads the experts expected under iid-uniform routing ([§5 "Expected distinct experts"](simulation-model.md#expected-distinct-experts)): 32 × top-4 touch 128·(1 − (1 − 4/128)³²) ≈ 81.7 of 128 experts per layer, about 46 GB/step including the KV, or 191 tok/s (before that fix it read all 128, 63 GB/step, 129 tok/s, ratio 2.03). The measured 262 tok/s needs ≤ 33 GB/step at 273 GB/s, ≤ ≈ 55 experts per layer, so real routing is more concentrated than uniform. Not an artefact: llama-batched-bench has fed random tokens since llama.cpp #15398 (Aug 2025), so the 32 sequences do not share their routing. | **Known exception**, asserted to still fail. Uniform routing is an expectation, not a bound: for batched MoE decode the simulator is not an upper bound when real routing is skewed. |
| lm-36 prefill (DeepSeek-V4.1-Flash, 4× Spark) | 0.94 | Passes, but at the bound. Prefill of 274 tokens is memory-bound: it reads nearly every routed expert (U(274) ≈ 255.6 of 256 expected under uniform routing; ≈ 80 GB per device at 273 GB/s), and TTFT is 309 ms. Reaching 94 % of LPDDR5X peak is implausible, so the likely cause is on the report side: SGLang's radix prefix cache is on by default and LocalMaxxing reuses its canonical prompt, so part of the prompt was probably cached. Skewed real routing touching fewer experts would also explain it. (Simulated with the DSpark draft-cache prefill on: 942 tok/s, 930 before the expected-unique fix.) | Asserted (passes); watch. |
| lm-36 decode (DeepSeek-V4.1-Flash, 4× Spark, DSpark) | 1.05 | Passes (≤ 1.11), close to the bound. Simulated with the in-checkpoint DSpark drafter, k 5, at the reported mean accepted length 2.85: 72.2 tok/s (70.4 before the expected-unique fix), a speedup of only ×1.17 over the 61.5 tok/s plain decode, because the 6-token verify of this MoE reads U(6) = 256·(1 − (1 − 6/256)⁶) ≈ 34.0 routed experts per layer (min(E, 6·6) = 36 before; 5.7× a plain step), and the drafter's own 3 MoE layers add more. Consecutive real tokens share experts, more than uniform routing assumes, so routing locality (as in web-04) is the likely cause. | Asserted (passes); watch. |
| lm-35 decode (GLM-5.3-Flash, 4× Spark TP4) | 1.60 | **Unresolved.** The run used a local checkpoint (`/models/glm-5.3-flash-nvfp4`); the record says only "NVFP4 4-bit weights" and gives no module layout or measured bpw (the researchers' ≈ 4.5 bpw is a format estimate). It is simulated with the closest published checkpoint, `nvidia/GLM-5.3-Flash-NVFP4`, which keeps attention, KDA and shared experts in BF16: ≈ 22 GB read per token over the 4 devices → 46.6 tok/s, below the reported 74.6. Possible causes: (a) the local checkpoint quantizes more modules than the published one; for scale, an all-linear NVFP4 layout would simulate to 88.8 tok/s (ratio 0.84), but nothing in the record shows that layout, so it is a scenario, not the mapping; (b) the simulator underestimates this setup (TP4 over RoCE, KDA decode). Not decided without the checkpoint's quant config. | **Known exception**, asserted to still fail; kept as an open flag. |
| web-53 decode (Qwen3-235B-A22B, 3090 Ti + DDR5-6400, experts on CPU) | 2.12 | **Mode not modelled.** ik_llama.cpp computes the experts of layers 10–93 on the CPU, reading them from DDR5 at 102.4 GB/s. The simulator's transfer mode streams them to the GPU over PCIe 4.0 x16 (28.8 GB/s payload: 31.5 GB/s line rate × 256/280) instead: ≈ 6 GB per token → 4.7 tok/s (5.2 and a ratio of 1.94 before the payload efficiency). The CPU-compute mode that would bound this run is a designed hook that validation rejects (design doc §2.8). Prefill (4096-token ubatch, transfer-bound either way) is asserted: 0.11. | **Known exception**, asserted to still fail. |

Two more observations, consistent with the bound but worth knowing:

- **lm-02 confirms the RTX 3090 tensor-rate unlock.** vLLM Marlin (fp16 MMA, fp32 accumulate) prefills
  296 tokens of a 27B model in 153 ms, about 104 TFLOP/s. The stock fp16 / fp32-accumulate peak is 71 TFLOP/s
  (GA102 whitepaper), so the stock preset would give ratio 1.5. With the unlock the run notes state (142 TFLOP/s),
  the ratio is 0.76.
- **Apple decode GEMMs run on the GEMV path.** Decode GEMMs have one token row. The shape-efficiency model pads
  that row to the tensor tile ([§4 "Shape efficiency"](simulation-model.md#shape-efficiency)), which on the M3/M4 SIMD
  engines made some decode GEMMs compute-bound (web-12 decode was "compute"). Real Metal kernels run them as GEMVs
  (MLX `gemv` / `qmv`, llama.cpp `mul_mv`), and the simulator now does too for m ≤ 1 ([§4 "GEMV
  path"](simulation-model.md#gemv-path)): web-12 150 → 153 tok/s (now memory-bound), web-14 309 → 318, web-26 269 →
  281. Fine-grained 4–5 bpw MoE models gain most, because their expert GEMVs had the most padded compute: lm-28
  129 → 151 tok/s, lm-34 69.5 → 82.5.

### Ratio below 0.1 (gross overestimate or report artefact)

| Report | Ratio | Likely cause |
|---|---|---|
| lm-08 prefill (B200, llama.cpp) | 0.008 | **Report / engine.** 347-token prompt measured as client TTFT (632 ms) on a GPU whose compute finishes it in about 5 ms. The submitter calls the run kernel-launch-bound on sm_100 for this hybrid Gated-DeltaNet model (decode 0.18 too). vLLM on the same B200 reaches 0.14. Not a sim issue. |
| web-50 / web-51 prefill (MI300X, llama.cpp HIP) | 0.041 / 0.057 | **Engine.** llama.cpp's ROCm kernels of July 2025 are not tuned for CDNA3 (the post says vLLM / ATOM are much faster). The simulator uses the int8 MFMA peak (2.6 POP/s), which is right for an upper bound. Decode is also low (0.14 / 0.29) on a 5.3 TB/s part, which points to engine overhead, not to a wrong preset. |
| lm-38 prefill (Inkling-Small, 6× 3090 layer split) | 0.067 | **Report / engine.** Custom `inkling-exp` branch, client TTFT on 367 tokens (2.5 s), no flags or notes. Decode is 0.32. |
| web-36 prefill (Flash-Next, 6× 4090, 50k prompt, stock llama.cpp) | 0.063 | **Engine.** The poster's patched fork prefills the same 50k session at 7050 tok/s (ratio 0.26); stock upstream loses most of it to the indexer top-k path, by the poster's own analysis. |
| lm-28 decode (Flash-Next, M3 Ultra, oMLX MLX 4-bit) | 0.098 | **Engine.** A 7 B-active MoE at 14.8 tok/s on 819 GB/s: engine / MoE-gather-bound per the researchers. It was 0.115 until the GEMV path (F14) removed the padded MMA compute of its expert GEMVs. |

Others near the threshold, noted but not flagged: lm-15 prefill 0.11 and web-29 prefill 0.11 (TTFT and early Strix
Halo builds).

## Fixes made

- **Practicality-review fixes F6 and F14** ([the review](review/practicality-review.md)) changed the model: attention
  now pays its softmax (exp on the SFU, overlapped with the MMAs, [§7 "Attention softmax"](simulation-model.md#attention-softmax)),
  and Apple decode GEMMs of one token row take the GEMV path. Only Apple rows moved (the table above: lm-28, lm-34,
  web-12, web-14, web-26 decode; fresh rows F-LM25/26/27/53/54 decode by +0.1 % to +25 % (F-LM54 249 → 312 tok/s),
  F-W13 decode 4.874 → 4.873 before the F13 input correction above, F-LM53 prefill +0.01 %); the datacenter prompts here are too short for the softmax to
  bind. Every check keeps
  its verdict.
- **No report was traced to a wrong hardware number.** The
  underestimates are the MoE routing limit (web-04, a modelling choice), lm-35, whose checkpoint layout is unknown
  (unresolved, see above), and web-53, which uses an offload mode the simulator does not cost. The six file-size
  cross-checks put the simulated weight bytes within −3.6 % to +1.0 % of the GGUF files, so the bpw mapping holds.
- **Speculative decoding and offload are now simulated.** lm-09, lm-36 and lm-39 decode were excluded (printed only)
  before the deployment knobs existed; they now run with spec-dec on and are asserted. Their prefill now includes the
  drafter's cache fill (lm-09 31,655 → 29,706 tok/s, lm-39 31,336 → 30,381). web-52 and web-53 are new offload rows
  (design doc §8.2 S-1). The harness takes `deployment` and `hardware.host`; `reports.test.ts` checks that the
  spec-dec rows run at the intended E[τ] and that the offload rows stream experts.
- `reports.ts` web-26: `mlx_lm.benchmark` generates from the cache of its 2048-token prompt (v0.28.2
  `benchmark.py`), so decode is now simulated at context 2048 + 64 instead of 64 (decode ratio 0.38 → 0.42). A
  regression test checks it, and a test now checks every row's decode context against its recorded depth (only
  llama-bench rows may decode below the prompt length). No other row had the same mistake: every server, vLLM,
  MLX and batched-bench row already decoded after its prompt.
- **Expected distinct experts and EP imbalance** (practicality review F5, F12). The expert GEMMs read the experts
  expected under iid-uniform routing, `E·(1 − (1 − k/E)^T)`, instead of min(E, T·k) ([§5 "Expected distinct
  experts"](simulation-model.md#expected-distinct-experts)), and at ep > 1 the busiest device carries 1.2× the mean
  expert rows ([§5 "EP load imbalance"](simulation-model.md#ep-load-imbalance)). web-04 decode 129 → 191 tok/s
  (ratio 2.03 → 1.37, still an exception), lm-36 decode 70.4 → 72.2 (1.08 → 1.05) and prefill 930 → 942
  (0.95 → 0.94). Batch-1 decode without spec-dec is unchanged (U(1) = k); short memory-bound MoE prefills rise
  0.02–1.2 % (U a little below E; lm-36 the most). Of the newer rows, F-LM17, F-LM27, F-LM28 and F-LM49 decode rise 0.7–3 %
  (batched or spec-dec MoE), and lm-32 (EP8) prefill falls 0.2 % (ρ).
- **Whole-layer offload removed** (`OffloadConfig.denseLayers`; [§12 "Placement"](simulation-model.md#placement)).
  F-W08 (GPT-OSS-120B, llama.cpp with 13 of the layers on an RTX 3090, the rest on the CPU) was its only user. It is
  re-encoded as the routed experts of layers 0–22 in host RAM and its decode check, a known exception before, is now
  **excluded**: llama.cpp computes those layers on the CPU, a mode the simulator does not cost (CPU-compute, review F3),
  and expert streaming is a different setup, so the comparison would mean nothing. No other row used whole-layer
  offload; every other row is unchanged.

## Follow-ups (proposed, not done)

1. ~~**MoE routing locality for batched decode**~~ (done for the expected value: see "Fixes made"). The expert
   weights now follow the iid-uniform expectation `E·(1 − (1 − k/E)^T)`. A skewed-routing option (fewer distinct
   experts than uniform) would still be needed to assert web-04: uniform routing gives 191 tok/s, below the
   reported 262.
2. ~~**GEMV path for decode**~~ Done (F14): GEMMs with m ≤ `gemvMaxRows` run at the vector peak without MMA padding
   on the Apple engines ([§4 "GEMV path"](simulation-model.md#gemv-path)).
3. **Prefill at depth** (`workload.prefillDepth`). `simulate` always prefills from an empty cache. llama-bench
   `pp @ dN` and LocalMaxxing depth rows can only be checked against the depth-0 bound.
4. **fp32-accumulate default on GeForce.** It halves the fp16 tensor rate on GeForce parts (§2). It is right for the
   paths used here (Marlin and MMQ accumulate in fp32 / int32), but a kernel that accumulates in fp16 would beat the
   bound. `numerics.accumulate: 'fp16'` exists per report; the UI should say which engines can use it.
5. **Sub-4-bpw weight formats.** IQ1–IQ3 / Q2_K / Q3_K GGUFs are stored as `int4` with a negative
   `extraBitsPerElem` (lm-24 3.1 bpw, lm-27 2.9 bpw). Byte counts are right, but a real `bitsPerWeight` storage format
   (or `int2` / `int3` dtypes) would be cleaner, and validation should reject negative extras otherwise.

## Sim quirks noticed (not bugs, but they affect comparisons)

- With pp > 1 at batch 1, `prefill.tokensPerSec` assumes a stream of batches (≈ pp × the per-sequence rate);
  layer-split llama.cpp prefill of one prompt must be compared with `prefillPerSequenceTokensPerSec`.
- `hgxCluster` puts the IB link name in the topology id even for one node, and `DGX_SPARK_PAIR`'s id repeats
  "dgx-spark"; topology preset ids are long, so `reports.ts` uses the builders.
- `memCapacityBytes` must be one of the preset's options: the 96 GB Strix Halo records use the 128 GB option (same
  bandwidth).

## Data-quality notes (from the researchers' files)

- **Speculative decoding and MTP flags are unreliable** on LocalMaxxing: `specDecoding = false` on more than 500 runs
  whose command line or notes show MTP / DFlash / ngram drafts; conversely `true` sometimes marks an always-on
  engine. Rows were filtered on the command text, and the three spec rows kept here are simulated with spec-dec on.
- **CPU-offload flags are unreliable**: hundreds of Flash-Next / GLM-5.3-Flash runs spill experts to system RAM
  without saying so; all of them are excluded.
- **Hardware mislabeling**: DGX Spark clusters are recorded as one GB10 with 128–512 GB (lm-35 says "128 GB" but runs
  on 4 nodes); "R9700 × 3" records that used 2 GPUs; "B65 × 3" that is data parallel; Apple records carry no GPU-core
  count (inferred here from the RAM size, which is only sold with one GPU configuration).
- **LocalMaxxing prefill is client-side TTFT** over short prompts; `prefillTokens` is mostly 0 / null.
- **Same config, very different decode**: R9700 27B llama.cpp records span 1.2–52.7 tok/s (the top ones exceed the
  640 GB/s weight-streaming bound); Strix Halo Q8_0 7.8–14.9 tok/s (the top value exceeds 256 GB/s / 28.7 GB).
  Only rows with notes that explain them were used.
- **Intel Arc llama.cpp decode is not bandwidth-bound on the B65 rig** (flat across Q4–Q8), but scales with bytes on
  the B70 box: the simulator is a bound for both, tighter for the B70.
- **Web rows**: web-12 is second-hand (a nitter post quoted on reddit) and noisy; web-31 mixes backends (pp from HIP
  rocWMMA, tg from another run); web-41 is one vLLM log line; web-42/44/45/48 do not state prompt or output lengths
  (vLLM `benchmark_serving` defaults 1024 / 128 or 512 / 512 were assumed; they only move the decode context).
- **Coverage holes**: no LocalMaxxing run on H100, B300, MI3xx/MI4xx, RX 9070 XT, Arc B580, M4 Max, M3 Ultra 60c or
  M5 Ultra; no non-spec DeepSeek-V4.1-Flash decode anywhere; no 2-node DGX Spark report; MiniMax-M3 has a single
  eval-suite-average record (not used).

## Sources that could not be fetched directly

- **reddit.com** blocks the extraction service; r/LocalLLaMA threads were read through a redlib mirror (same post
  text; URLs here point at reddit).
- **localmaxxing.com** HTML returns 403 to the fetcher; the JSON API (`/api/speed-tests?hfId=…`) works, and the run
  links here are the site's `?run=` model-page URLs from that API.
- The hardware-corner article tables (M5 Max gpt-oss-120b, RTX PRO 6000) and the gpt-oss guide rows beyond the
  truncated page text were not used where the numbers could not be seen verbatim.
