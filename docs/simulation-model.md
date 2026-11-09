# specmaxx simulation model

This document is the design reference for `src/sim/**`. Every section names the functions that implement it
(`file: function`), and the code points back here with `// See docs/simulation-model.md §N "Section"` comments.
Formulas in the code (`// Formula: …`) use the same notation as this document.

Contents:

1. [Scope and conventions](#1-scope-and-conventions)
2. [Model IR](#2-model-ir)
3. [Op catalogue](#3-op-catalogue)
4. [Hardware](#4-hardware)
5. [Parallelism](#5-parallelism)
6. [Collectives](#6-collectives)
7. [Per-op time, stages, pipeline](#7-per-op-time-stages-pipeline)
8. [Memory fit](#8-memory-fit)
9. [Deliberately excluded, and approximations](#9-deliberately-excluded-and-approximations)
10. [Worked example: REF_DENSE_8B on one RTX 5090](#10-worked-example-ref_dense_8b-on-one-rtx-5090)
11. [Extending](#11-extending)
12. [Offloading](#12-offloading)
13. [Speculative decoding](#13-speculative-decoding)

---

## 1. Scope and conventions

specmaxx estimates the **theoretical maximum** prefill and decode tokens/s of a model on a hardware stack:
what the hardware and the math allow, not what a particular inference engine achieves. Concretely:

- Every op runs at the roofline `max(compute, memory)` of the device it runs on, with compulsory memory traffic
  only (each operand read once, each result written once) and a deterministic tensor-shape efficiency
  (MMA padding + wave quantization, §4 "Shape efficiency").
- Work is costed **per device, on the critical path**. Sharding is balanced, so all devices of a pipeline
  stage are symmetric and one device stands for all of them.
- Communication uses the alpha-beta model (§6). Under the default overlap assumption, bandwidth terms hide
  behind compute and latency terms do not (§7 "Stage time and overlap").
- No inference-engine knobs (§9). The one engine-overhead exception, the engine dispatch overlay, is opt-in and a
  labelled figure next to the headline, never in it (§7 "Engine dispatch overlay").
- Speculative decoding and weight / expert / lookup-table offloading are **opt-in deployment-scenario knobs, off by
  default** (`SimInput.deployment`; [§12](#12-offloading), [§13](#13-speculative-decoding)). Off, the results are
  bit-identical to the model without them (`engine/identity.test.ts`). On, a result is still the hardware/math upper
  bound given the knob's stated parameters (hit ratio, acceptance, placement, queue depth, overlap).

**Units: SI decimal.** 1 GB = 1e9 B, 1 TB/s = 1e12 B/s, 1 TFLOP/s = 1e12 FLOP/s (`src/sim/hw/units.ts`:
`tflops`, `gbPerSec`, `tbPerSec`, `gb`, `mb`, `usec`). Times are seconds, rates tokens/s, sizes bytes or
elements as the field name says.

| Quantity                                                     | Convention                                                                                                                                                                        |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Matmul peak (`MatmulPeak.flopsPerSec`)                       | per device; keyed by (input format, accumulator, sparsity); FLOP/s (OP/s for integers)                                                                                            |
| 2:4 sparse peak                                              | dense-equivalent FLOP/s (vendor convention); op FLOP counts are never halved                                                                                                      |
| Memory bandwidth (`MemorySpec.bandwidthBytesPerSec`)         | per device, read + write combined, as vendors quote it                                                                                                                            |
| Link bandwidth (`LinkSpec.bandwidthPerDirectionBytesPerSec`) | per device, **per direction** (unidirectional injection rate), never the bidirectional sum                                                                                        |
| Link latency (`latencySec`)                                  | α per message / algorithm step                                                                                                                                                    |
| `CollectiveOp.messageBytes`                                  | per device: all_reduce = the full buffer; all_gather = the full OUTPUT; reduce_scatter = the full INPUT; all_to_all = the total sent incl. the self share; send_recv = bytes sent |
| Op bytes (`ByteBreakdown`)                                   | per device, per microbatch step                                                                                                                                                   |
| `tokensPerSec`                                               | aggregate over DP replicas; `perSequenceTokensPerSec` is what one sequence sees                                                                                                   |

---

## 2. Model IR

A model is described by an **architecture spec** (`src/sim/ir/model.ts`: `ModelSpec`, `LayerSpec`,
`BlockSpec`), with logical (unsharded) dimensions and no tokens or parallelism. Layers are heterogeneous: each
`LayerSpec` is its own ordered list of blocks (`norm`, `residual`, `residual_mix`, `gqa_attention`, `mla_attention`,
`csa_attention`, `linear_attention`, `mlp`, `moe`, `hc_pre`, `hc_post`, `ngram_embedding`, `short_conv`), which
covers hybrid linear/full attention stacks, dense first layers before MoE, and alternating sliding-window layers.
The builders (`src/sim/ir/builders.ts`: `decoderLayer`, `decoderModel`, `repeatLayers`, `hyperCollapse`) produce
pre-norm decoder layers from `MixerParams` + `FfnParams` (+ optional `residual`, `ngram`, `shortConv`):

- plain residual: `[ngram] → norm → mixer → [short_conv] → [post-norm] → residual → norm → ffn → [short_conv] →
[post-norm] → residual`;
- hyper-connections (`residual.kind: 'hyper'`, N streams): each `norm … residual` pair becomes
  `hc_pre → [norm if subNorms] … hc_post`. DeepSeek-V4.1 and GLM-5.3-Flash keep the sub-norms, Qwen4Exp does not.
- Attention Residuals (`residual.kind: 'attn_res'`, Kimi K3): `[residual_mix] → norm → mixer → residual →
residual_mix → norm → ffn → residual`, and `attnResCollapse` as `beforeHead` (§5 "Attention Residuals").
- `decoderModel({ beforeHead })` appends blocks (the final hyper-connection collapse) to the LAST layer, so they are
  lowered and pipeline-staged as part of it. `finalNorm: 'none'` (Qwen4Exp) has no final norm.
- `ModelSpec.residualStreams` N (default 1): each token carries `N·hidden` residual elements; the PP activation
  send is `T·N·h·bpe(act)` (`lower.ts: lowerModel`).
- `singleBlockLayer({ hidden, mixer } | { hidden, ffn })` builds a layer with ONE sublayer,
  `norm → mixer|ffn → residual` (Nemotron-H, whose `hybrid_override_pattern` makes each layer a Mamba-2,
  attention or MoE block).

**Cross-layer sharing** names an earlier layer by index: `MlaAttentionBlock.kvCacheFromLayer` (Mistral Large 4:
read another layer's latent; no `kv_a`, no cache), `CsaGlobalSpec.kvCacheFromLayer` (DeepSeek-V4.1: read a
source's compressed KV; no compressor), `SparseSelection {kind: 'reuse'}` (GLM-5.3, Mistral, DeepSeek "Reuse": take
the nearest earlier indexer's top-k; no indexer ops, no key cache, no indexer weights) and `IndexerSpec.ownsKeys:
false` (DeepSeek "Reindex": read an earlier indexer's keys). `src/sim/ir/validateModel.ts: modelSpecErrors` checks
that each reference points to a compatible earlier layer, that `gateRank` is set iff the variant is `kda`, and that
hyper-connection and n-gram widths equal `residualStreams·hidden`.

New block kinds (lowered in `src/sim/lower/{csa,hyperConnection,ngram,shortConv}.ts`, §5):

- `csa_attention` (DeepSeek-V4, V4.1): one cached `headDim` vector per token used as K and V; every layer attends
  its own `windowTokens` raw window, plus (with `global`) entries of a KV compressed `compressRatio`:1, either the
  `topK` an indexer keeps or, with `CsaSelection {kind: 'all'}` (DeepSeek-V4 r = 128 layers), every visible entry;
  low-rank q (`qLoraRank`) and grouped low-rank output (`outGroups` × `outLoraRank`). `CsaGlobalSpec.compressorOverlap`
  (DeepSeek-V4 r = 4) pools overlapping windows (coff = 2, below).
- `hc_pre` / `hc_post`: hyper-connections over N streams (Qwen "Gated Residual", DeepSeek/GLM mHC). `hc_pre`
  collapses the streams through its `projections` (and an optional γ over N·D, Sinkhorn iterations); `hc_post`
  writes the sublayer output back, mixing the streams (`mixesStreams`, mHC) or not (Qwen).
- `ngram_embedding` (Qwen PLE, DeepSeek Engram): a hashed lookup table of `tableRows × rowDim` (memory
  capacity), `rowsPerToken` rows gathered per token, then `projections`, a gate and an optional dilated conv,
  written into the `outputDim = N·D` residual.
- `short_conv` (Inkling): `x + conv(x)`, a depthwise causal conv of `kernel` taps on a branch output.
- `residual_mix` (Kimi K3 Attention Residuals): a softmax-weighted mix of `candidates` residual vectors per token
  (the running sum and the earlier block summaries), `2·dim` params (score norm γ and `dim → 1` projection).
- New fields: `IndexerSpec.{keyPoolTokens, keyPoolGate, keyPoolOverlap, keyInputDim, ownsKeys, maxScoredKeys, headWeights}` (GLM
  k-pool, Qwen QSA, MiniMax MSA, DeepSeek CSA indexers); `GqaAttentionBlock.{relativeBias, kvConvKernel}`
  (Inkling); `LinearAttentionBlock.variant 'kda'` with `gateRank` (GLM-5.3-Flash Kimi Delta Attention) and
  `fullRankOutputGate` (Kimi K3: one `h → vH·dv` output gate instead of the rank-r pair); `MlaAttentionBlock.outputGate`
  (Kimi K3: a sigmoid gate from its own `h → H·dv` projection); `MoeBlock.latentNorm` (Kimi K3 latent MoE: an
  RMSNorm over l before the `l → h` projection, see `latentDim` and §5 "Latent MoE"); `FfnActivation 'situ'`
  (Kimi K3's tanh-bounded SiLU).
- Mamba-2 fields (Nemotron-H; §5 "Mamba-2"): `LinearAttentionBlock.convBias` (conv1d bias of C elements, default
  false) and `chunkTokens` (chunk length Q of the chunked prefill form, default 64, any variant). Activation
  `'relu2'` (`relu(x)²`, ungated only; `MlpBlock`, `MoeBlock`) for Nemotron-H's non-gated experts.
  `modelSpecErrors` checks that `chunkTokens` is a positive integer and that `relu2` is ungated.

**Lowering** (§5, `src/sim/lower/lower.ts: lowerModel`) turns spec + workload step + quantization +
parallelism into the ops ONE device runs per stage, with concrete shapes and dtypes (§3).

### Dtypes, math and accumulate

`src/sim/ir/dtype.ts`: `DTYPE_INFO`, `bitsPerElement`, `bytesPerElement`, `resolveGemmMath`,
`resolveAccumulate`.

- Storage bits per element: `bits = elementBits + blockScaleBitsPerElem + extraBitsPerElem`;
  `bpe = bits / 8`. Block scales: MX formats 8 bit per 32 elements (0.25 bit/elem); nvfp4 8 bit per 16 (0.5).
  `extraBitsPerElem` carries group-quantization overhead outside the dtype, e.g. GPTQ int4 g128 with a 16-bit
  scale and 4-bit zero: (16 + 4)/128 = 0.15625. Two more optional fields cover fractional-bit trellis formats
  (EXL3, "Quant overheads" below): `valueBitsPerElem` replaces the dtype's `elementBits` (e.g. 2.5), and
  `perDimBits` is stored once per row and once per column of each PHYSICAL matrix, so a `rows × cols` matrix takes
  `matrixStoredBytes = rows·cols·bits/8 + perDimBits·(rows + cols)/8` (`dtype.ts`). A fused GEMM (`qkv_proj`,
  `gate_up`, KDA `lin.qkv`, `in_proj`) lists its physical matrices in `GemmOp.outParts`; each part carries its own
  vectors, so `sideVectorBytes = perDimBits·(rows + P·cols)/8` with `P` the number of non-empty parts. Empty
  matrices (no shared experts, rank-0 gates) store nothing. An op that runs only a slice of one stored matrix sets `GemmOp.sideVectorElems`, its explicit share of the
  side-vector entries: the MLA absorbed `q_absorb` (`Hl·dn + rkv`) and `v_absorb` (`Hl·dv`) split the stored `kv_b`
  (`Hl·(dn+dv)` rows, `rkv` columns), so decode and prefill (MHA `kv_b`) charge identical bytes.
- **GEMM input class** (`resolveGemmMath`): the operand with more element bits wins (int4 × bf16 runs at the
  bf16 rate, because weight-only quantization dequantizes); on a tie the activation's format wins.
- **Accumulator** (`resolveAccumulate`): integer inputs → `int32`; bf16 / tf32 / fp32 → always `fp32`;
  fp16 / fp8 / fp6 / fp4 → the requested `QuantConfig.accumulate` (**default `fp32`**). This default matters on
  GeForce parts, where fp32 accumulate runs at half the fp16-accumulate rate.
- `QuantConfig` fields: `weights` (linear layers), `activations`, `kvCache` (KV, MLA latent, CSA window),
  `recurrentState` (default fp32), `embeddings` / `lmHead` (default `highPrecision`), `highPrecision` (norms,
  router, biases, conv taps; default bf16), `attentionMath`, `accumulate`, `weightSparsity`, and:
  - `modules` (per `WeightModule` class): `{ weights, gemmInput? }`. Classes: `attention` (GQA q/k/v/o, MLA q path,
    `g_proj` and o\*proj, CSA q path, window `wkv` and output), `attention_kv_down` (MLA `kv_a`, CSA compressor),
    `attention_kv_up` (MLA `kv_b` and its absorbed slices `q_absorb`, `v_absorb`), `linear_attention`, `indexer`
    (query `wq`/`wq_b`), `indexer_key` (`wk`, k-pool gate), `indexer_head_weights` (`weights_proj`), `dense_mlp`,
    `shared_experts`, `routed_experts`, `moe_latent` (latent-MoE projections `h → l`, `l → h`: Nemotron
    `fc1/fc2_latent_proj`, Kimi K3 `routed_expert_{down,up}_proj`), `residual_mixer`, `ngram`.
  - Resolution (`WEIGHT_MODULE_FALLBACK`): an absent class uses its parent's entry if listed
    (`attention_kv_down`, `attention_kv_up` → `attention`; `indexer_key`, `indexer_head_weights` → `indexer`;
    `moe_latent` → `shared_experts`), else `{ weights }`; `residual_mixer` defaults to `{ dtype: highPrecision }`.
    This expresses e.g. GLM-5.3-Flash FP8 (`kv_b_proj` BF16, other MLA projections FP8), Mistral Large 4 FP8
    (`kv_a`, `kv_b`, indexer `wk`, `weights_proj` BF16) and Kimi K3 (MXFP4 routed experts, everything else BF16).
  - `gemmInput` (W8A8, W4A4) defaults to `activations` and is the GEMM's input dtype, so
    `math = resolveGemmMath(w, gemmInput)`. `weightSparsity` applies to every class but `residual_mixer`; the router
    and the LM head are not modules (`src/sim/lower/common.ts: moduleQuant`, `linearGemm({ module | format })`).
  - `lmHeadGemmInput`: the LM-head GEMM input dtype (default `activations`; NVFP4 W4A4 heads set `nvfp4`).
  - `lookupTables` (n-gram tables; default the `embeddings` format), `compressedKvCache` (CSA global KV; default
    `kvCache`), `indexerKeyCache` (indexer keys; default `kvCache`).

### Quant overheads

The Hugging Face quantization readers (`src/sim/models/hf/quant/`) turn each checkpoint's scale layout into a
`TensorFormat`. MX and NV formats carry their block scale inside the dtype (`ir/dtype.ts: DTYPE_INFO`); every other
scale or zero point outside the dtype goes into `extraBitsPerElem`, amortized per element
(`quant/common.ts: blockScaleBits, groupScaleBits, dtypeNameBits`):

- 2-D block scale, one `s`-bit scale per `r × c` block: `extra = s/(r·c)` (`blockScaleBits`).
- group scale along the input dim, one `s`-bit scale and one `z`-bit zero point (asymmetric only) per `g`
  elements: `extra = (s + z)/g` (`groupScaleBits`).
- Per-channel and per-tensor scales are ≈ 0 and are left out.

| Scheme                                                                                                            | Storage dtype                                            | Overhead (bits/element)                                          | Reader                                                                           |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| fp8 block 128×128, fp32 scales (DeepSeek-V3 / GLM / Qwen FP8, ModelOpt `FP8_PB_WO`, `FP8_BLOCK_SCALES`)           | `fp8_e4m3`                                               | 32/16384 ≈ 0.00195                                               | `fp8.ts`, `modelopt.ts`, `compressedTensors.ts`, `quark.ts`                      |
| fp8 block 32×32, ue8m0 scales (DeepSeek-V4.1, Quark `float8_e8m0fnu`)                                             | `fp8_e4m3`                                               | 8/1024 ≈ 0.0078                                                  | `fp8.ts`, `quark.ts`                                                             |
| fp8 per-tensor / per-channel (`FP8`, `float-quantized` channel)                                                   | `fp8_e4m3`                                               | ≈ 0                                                              | `modelopt.ts`, `compressedTensors.ts`                                            |
| mxfp8 (block 1×32 e8m0, `mxfp8-quantized`, ModelOpt `MXFP8`)                                                      | `mxfp8`                                                  | in the dtype: 8/32 = 0.25                                        | `fp8.ts`, `compressedTensors.ts`, `modelopt.ts`, `mlx.ts`                        |
| mxfp4 (e2m1, g32 e8m0; Quark fp4 g32 e8m0, `mxfp4-pack-quantized`, DeepSeek `expert_dtype: fp4`)                  | `mxfp4`                                                  | in the dtype: 8/32 = 0.25                                        | `fp8.ts`, `compressedTensors.ts`, `modelopt.ts`, `quark.ts`, `gptq.ts`, `mlx.ts` |
| nvfp4 (e2m1, g16 e4m3 + an fp32 per-tensor `weight_scale_2`)                                                      | `nvfp4`                                                  | in the dtype: 8/16 = 0.5; the per-tensor scale ≈ 0               | `modelopt.ts`, `compressedTensors.ts`, `quark.ts`, `gptq.ts`, `mlx.ts`           |
| GPTQ / AWQ / AutoRound int`b` g`g` (fp16 scale + packed `b`-bit `qzeros`, also when symmetric)                    | `int4` / `int8`                                          | (16 + b)/g: g128 0.15625, g32 0.625; `group_size -1` ≈ 0         | `gptq.ts`                                                                        |
| compressed-tensors int`b` group (`scale_dtype`, default bf16; zero point `zp_dtype` only when `symmetric: false`) | `int4` / `int8`                                          | (16 + z)/g: sym g128 0.125, asym g32 int8 zp 0.75                | `compressedTensors.ts`                                                           |
| ModelOpt `INT4_AWQ` / `W4A16_AWQ` / `W4A8_AWQ`                                                                    | `int4`                                                   | 16/g                                                             | `modelopt.ts`                                                                    |
| Quark int4 / uint4 per_group g                                                                                    | `int4`                                                   | (16 + z)/g, z = 4 when asymmetric or uint                        | `quark.ts`                                                                       |
| bitsandbytes NF4 / FP4, absmax per 64 (codes stored as `int4`)                                                    | `int4`                                                   | fp32 absmax 32/64 = 0.5; double quant 8/64 + 32/(64·256) ≈ 0.127 | `bnb.ts`                                                                         |
| bitsandbytes int8 (LLM.int8(), per-row fp32 scale; fp16 outlier columns not modeled)                              | `int8`                                                   | ≈ 0                                                              | `bnb.ts`                                                                         |
| MLX affine `b`-bit g`g` (16-bit scale + 16-bit bias per group)                                                    | `int4` / `int8` (widths without a dtype round up, noted) | 32/g: g64 0.5                                                    | `mlx.ts`                                                                         |

#### EXL3 (ExLlamaV3, fractional bits per weight)

`src/sim/ir/exl3.ts: exl3Format`, reader `quant/exl3.ts` (`quant_method: "exl3"`). Format facts, from exllamav3's
`LinearEXL3` (`exllamav3/modules/quant/exl3.py`), its converter docs (`doc/convert.md`) and real EXL3 configs
(fixtures in `src/sim/models/__fixtures__/quant/`):

- Per quantized linear the checkpoint stores `trellis` (int16 `[in/16, out/16, 16·K]`: a 16×16 tile takes `16·K`
  words = `256·K` bits, so exactly `K` bits per weight; `K` is an integer 1–8, or a half-integer with the `mul1`
  codebook), `suh` (`in` fp16) and `svh` (`out` fp16) Hadamard sign/scale vectors, and one codebook scalar
  (`mul1` / `mcg`, ≈ 0). A checkpoint's "2.5 bpw" is the AVERAGE of `K` over its tensors; the converter's
  per-tensor allocation (and `--recipe` files) are not in `config.json`, so the simulation applies the average to
  every linear layer.
- `config.json` `quantization_config`: `{ quant_method: "exl3", version, bits, head_bits, calibration, out_scales,
codebook, mtp_bits, vision_bits, original_quantization_config? }`. The average key is `bits` (turboderp,
  converter `-b`), `bpw` or `avg_bits` (community mixed-precision optimizers); `head_bits` (converter default 6;
  16 = unquantized) is the LM head; `mtp_bits` / `vision_bits` concern parts outside the IR;
  `original_quantization_config` describes the source checkpoint and is ignored. A community `scope:
"…routed_experts_only"` quantizes only the routed experts. turboderp publishes one repo per model with a branch
  per bitrate (`SC_4.00bpw_H5_V6`, `sc_3.10bpw`), each with its own `config.json`.
- Scope: attention, MLP, shared and routed experts, indexer and linear-attention projections at `bits`; LM head at
  `head_bits`; embeddings, routers, norms, hyper-connection mixers and n-gram lookup tables stay unquantized
  (`--embed_bits` default 16).

Weight bytes, with `bpw` = average bits per weight and `n × k` the matrix shape:

```
bytes(n×k matrix) = n·k·bpw/8 + 16·(n + k)/8        (trellis + fp16 suh/svh;  bpw clamped to [1, 8])
bytes(fused GEMM) = n·k·bpw/8 + 16·(n + P·k)/8     (P physical matrices share the k input features; GemmOp.outParts)
```

so 2.5 bpw costs `params × 2.5 / 8` bytes plus the side vectors (< 1 % for 4096-wide matrices); `head_bits` 16 is
plain fp16. Decode in the memory-bound regime therefore scales as ≈ 1/bpw (`ir/exl3.test.ts`).

**Compute dtype.** The kernels decode the trellis to fp16 inside the tensor-core matmul (small row counts), or
reconstruct an fp16 weight and call a normal fp16 GEMM (above ~144 rows). Either way the GEMM is fp16 × fp16 with
the accumulate of `QuantConfig.accumulate`. The format is stored as `{ dtype: 'fp16', valueBitsPerElem, perDimBits }`,
so `resolveGemmMath` (more element bits wins, tie → activations) gives the activation dtype's peak for 16-bit
activations and fp16 for fp8 activations. The reader sets `activations: 'fp16'` (exllamav3 runs in fp16). The
trellis decode and the Hadamard rotations are NOT charged as FLOPs: they are integer/add-only work that overlaps the
matmul, and this model counts weight bytes and GEMM FLOPs only (dequantization is free compute, like every
weight-only scheme above). Where that overlap fails (very low bpw on a compute-limited GPU) the real kernel is
slower than the theoretical maximum.

Weight-only schemes leave `gemmInput` unset (the GEMM runs at the activations' format, §2 "GEMM input class");
schemes that also quantize activations (`input_activations`, W8A8 / W4A4 algorithms, AutoRound `act_bits` < 16)
set `ModuleQuant.gemmInput`.

### Parameter counts

`src/sim/ir/params.ts`: `blockParams`, `activeBlockParams`, `countParams`, `activeParamsPerToken`. Logical
totals (unsharded), used to sanity-check mappers against published sizes.

- gqa: `h·(heads + 2·kvHeads)·hd` (+ `h·heads·hd` with an output gate) + `heads·hd·h` (o) + biases + `2·hd`
  (q/k norm) + indexer; + `h·heads·relDim + relDim·extent` with `relativeBias`; + `2·kvHeads·hd·K` with
  `kvConvKernel`.
- mla: q path `h·rq + rq + rq·H·(dn+dr)` (or `h·H·(dn+dr)` when rq = 0), kv `h·(rkv+dr) + rkv + rkv·H·(dn+dv)`,
  o `H·dv·h`, + `h·H·dv` with `outputGate` (`g_proj`), + indexer (q input rq or h). With `kvCacheFromLayer` the
  `h·(rkv+dr) + rkv` term is dropped. Kimi K3: 144,115,712 + 88,080,384 (gate) per layer.
- csa: q `h·rq + rq + rq·H·hd`, kv `h·hd + hd`, out `G·R_o·(H·hd/G) + G·R_o·h`; compressor (owner only)
  `2·h·coff·hd + hd` for r > 1 (coff = 2 with `compressorOverlap`, else 1), `h·hd + hd` for r = 1; + the global
  selection's indexer (q input rq; 0 for `reuse` and `all`). DeepSeek-V4.1: 126.6 M per layer without compressor and
  indexer. The DeepSeek-V4 compressor APE (`r·coff·hd`, fp32) is not counted (`params.ts: blockParams`).
- indexer: `qIn·H_I·hd_I` + [ownsKeys] `kIn·coff·hd_I + 2·hd_I` (key norm γ, β) + [keyPoolGate]
  `h·coff·hd_I + p·coff·hd_I` + [headWeights] `h·H_I`, coff = 2 with `keyPoolOverlap`, else 1. A `'reuse'`
  selection has 0.
- linear_attention: in_proj `h·(2·qkH·dk + vH·dv·(gate?2:1) + 2·vH)`, conv `(2·qkH·dk + vH·dv)·K`, `2·vH`
  (A_log, dt_bias), `dv` (out norm), out_proj `vH·dv·h`; + `(2·qkH·dk + vH·dv)` with `convBias`.
- mamba2 (G = qkH B/C groups, H = vH heads, N = dk state size, P = dv head dim; `params.ts: blockParams`,
  per-variant constants `linearVariantConstants`): in_proj `h·(2·G·N + 2·H·P + H)` (z, x, B, C and ONE dt per
  head), conv `(2·G·N + H·P)·K` + [convBias] `2·G·N + H·P`, `3·H` (A_log, D, dt_bias), `H·P` (grouped out-norm
  γ), out_proj `H·P·h`. Nemotron-3.5-Lightning (h 2688, H 64, P 64, G 8, N 128, K 4, bias): 38,742,208 per
  layer; Super (h 4096, H 128): 109,635,968 (both equal the checkpoint headers without the 2688 / 4096 pre-norm).
- kda: `h·(2·qkH·dk + vH·dv)` (q, k, v) + `h·vH` (β) + `h·r + r·vH·dk` (forget gate) + output gate `h·r + r·vH·dv`
  (or `h·vH·dv` with `fullRankOutputGate`) + conv `(2·qkH·dk + vH·dv)·K` + `vH` (A_log) + `vH·dk` (dt_bias) + `dv`
  (o_norm) + `vH·dv·h` (o_proj). GLM-5.3-Flash: q, k, v, f, g, b projections 104.07 M per layer; Kimi K3 (full-rank
  gate) 443,740,384 per layer (`A_log` is stored as [128] for 96 heads: 32 not counted).
- mlp `(gated?3:2)·h·f`; moe `E·(gated?3:2)·l·fe + E·h + [latent] 2·h·l + [latentNorm] l + S·(gated?3:2)·h·fs` with
  `l = latentDim ?? h` (`params.ts: moeParams`; the router and the shared experts read h, the latent terms exist
  only with `latentDim`; Kimi K3: 896·3·3584·3072 + 896·7168 + 2·7168·3584 + 3584 + 3·7168·6144 = 29,784,935,936
  per layer); active MoE uses `k` instead of `E` for the routed experts.
- residual_mix: `2·dim` (Kimi K3: 4·h per layer for the two mixes, 2·h for the final one; layer 0 keeps an unused
  mixer-side pair, 2·h, that the IR does not count).
- hc_pre: [normGamma] `N·D` + `Σ in·out` + `vectorParams`; hc_post 0. Qwen3.8-Flash-Next 6,604,800 per sublayer
  (634 M in 48 layers, final collapse 6.56 M); DeepSeek-V4.1 `24·20480 + 27` per sublayer (39.3 M); GLM-5.3-Flash
  0.79 M per layer.
- ngram_embedding: `tableRows·rowDim + Σ in·out + [conv] outputDim·K + vectorParams`. **Active** params drop the
  table and add the gathered rows `rowsPerToken·rowDim`: the table counts toward weight memory and total
  params, not active params.
- short_conv: `dim·K`.
- Final norm: `h` (rmsnorm), `2·h` (layernorm), 0 (`'none'`).

### First-class vs expressed

| Feature                                                                          | How it is expressed                                                                                                                   | New op kind?       |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Sliding window                                                                   | `KvSelection {window}` on `AttentionOp`                                                                                               | no                 |
| Top-k sparse attention                                                           | `KvSelection {topk}` + `IndexerOp` (no V, own key cache, emits indices)                                                               | **yes: indexer**   |
| MLA                                                                              | GEMMs (incl. per-head absorb GEMMs) + `AttentionOp` with `kvHeads = 1`, qk ≠ v dims, latent `kvRead`/`kvWrite`; block `mla_attention` | no                 |
| GatedDeltaNet / Mamba2 / linear attention                                        | `RecurrentOp` (fixed-size state, chunked vs step) + `causal_conv1d`, `gated_rmsnorm`; block `linear_attention`                        | **yes: recurrent** |
| Gated attention output                                                           | `sigmoid_mul` + a wider q projection                                                                                                  | no                 |
| MoE shared experts, dense first layers                                           | `MoeBlock.sharedExperts`; heterogeneous `layers[]`                                                                                    | no                 |
| Latent MoE (experts at l < h; Muse Spark, Nemotron)                              | `MoeBlock.latentDim` → replicated `moe.latent_in` / `moe.latent_out` GEMMs (module `moe_latent`), experts and a2a at l (§5)           | no                 |
| Per-expert output norm (Muse Spark `post_expert_norm`)                           | `MoeBlock.expertOutputNorm` → `rmsnorm`; with etp > 1 a reduce-scatter + sum-of-squares all-reduce (§5)                               | no                 |
| Attention sinks (gpt-oss)                                                        | ignored (negligible cost)                                                                                                             | no                 |
| DeepSeek-V4.1 CSA (K = V, window + compressed top-k, grouped low-rank output)    | block `csa_attention`; one `AttentionOp` with `secondaryKv` (§5 "CSA")                                                                | no                 |
| Hyper-connections (Qwen Gated Residual, mHC)                                     | blocks `hc_pre` / `hc_post` → GEMMs + `hc_collapse`, `hc_expand`, `sinkhorn`; `ModelSpec.residualStreams`                             | no                 |
| n-gram / Engram tables                                                           | block `ngram_embedding` → `EmbeddingOp` gather + GEMMs + `ngram_gate`; `QuantConfig.lookupTables`                                     | no                 |
| Short convs on branch outputs / K, V (Inkling)                                   | block `short_conv`; `GqaAttentionBlock.kvConvKernel` → `causal_conv1d`                                                                | no                 |
| Relative-position bias (Inkling)                                                 | `GqaAttentionBlock.relativeBias` → two GEMMs                                                                                          | no                 |
| KDA (low-rank gates, per-channel decay)                                          | `LinearAttentionBlock.variant 'kda'` + `gateRank` (+ `fullRankOutputGate`) → GEMMs + `RecurrentOp` (`gateElemsPerTokenPerHead`)       | no                 |
| MLA sigmoid output gate (Kimi K3)                                                | `MlaAttentionBlock.outputGate` → `mla.g_proj` GEMM + `sigmoid_mul`                                                                    | no                 |
| Latent MoE with a latent norm (Kimi K3)                                          | `MoeBlock.latentNorm` → `moe.latent_all_reduce` + `moe.latent_norm` before a row-parallel `moe.latent_out` (§5 "Latent MoE")          | no                 |
| Attention Residuals (Kimi K3)                                                    | block `residual_mix` → one `residual_mix` elementwise op per sublayer; `attnResCollapse` before the head                              | no                 |
| Pooled / gated / foreign-key / bounded indexers                                  | `IndexerSpec` optional fields → `IndexerOp.{keyPoolTokens, maxScoredKeys, keyWriteElemsPerToken, headWeights}` + `kv_pool`            | no                 |
| Shared top-k (indexer reuse)                                                     | `SparseSelection {kind: 'reuse'}`: `KvSelection {topk}`, no `IndexerOp`                                                               | no                 |
| Cross-layer KV sharing                                                           | `MlaAttentionBlock.kvCacheFromLayer`, `CsaGlobalSpec.kvCacheFromLayer`: no producer GEMMs, `kvWrite` 0                                | no                 |
| Per-module weight formats (NVFP4 experts, FP8 attention, BF16 hyper-connections) | `QuantConfig.modules` per `WeightModule`, `linearGemm({ module })`                                                                    | no                 |

---

## 3. Op catalogue

`src/sim/ir/ops.ts` defines the ops; `src/sim/ir/metrics.ts`: `opFlops`, `opBytes` count them. An op is the
work of one device on its stage's critical path for one microbatch step. Notation: `bpe(f)` bytes per element,
`past = kvLen − qLen`, `T = seqs·qLen`. FLOP counts are dense-equivalent.

### GEMM

`metrics.ts: gemmFlops, gemmBytes, gemmWeightBytes`. `out[b] (m×n) = in[b] (m×k) · W[b] (k×n)`; batch entries
have distinct weights (grouped experts, MLA per-head absorb). `b_w = GemmOp.distinctWeightMatrices ?? b` is the
number of weight matrices read: a MoE grouped GEMM computes on a grid of b experts but reads only the `U ≤ b`
expected distinct ones (§5 "Expected distinct experts"); every other GEMM has `b_w = b`.

- `F = 2·b·m·n·k`
- weights `b_w·(n·k·bpe(w) + perDimBits·(n + P·k)/8)` (`perDimBits` = 0 except EXL3; `P` = fused physical matrices, `GemmOp.outParts`); with 2:4 sparsity `b_w·n·k·(bits/2 + 1)/8` (half the values plus a 2-bit index per kept
  value); bias `+ b_w·n·bpe(bias)`
- read `b·m·k·bpe(in)`, write `b·m·n·bpe(out)`

### Attended pairs

`metrics.ts: attendedPairs`. (query, key) pairs per sequence per head. `cap` = window size or top-k (∞ for full),
`r` = tokens per key (1 for plain tokens; r for compressed CSA entries or pooled indexer keys). A query at absolute
position `p` sees `p/r` keys (continuous: against `floor(p/r)` this overcounts by fewer than `qLen/2` pairs, §9).

- non-causal: `qLen·min(cap, ceil(kvLen/r))`
- causal, full: `(qLen·past + qLen(qLen+1)/2)/r` — **the causal triangle is counted exactly, diagonal included**
  (masked tiles are skipped); e.g. q = kv = 4096, r = 1 → 8,390,656 pairs.
- causal, capped: `j0 = clamp(cap·r − past, 0, qLen)`; pairs = `(j0·past + j0(j0+1)/2)/r + (qLen − j0)·cap`.
- r = 1 is exactly the token formula (tested in `metrics.test.ts`).

`metrics.ts: kvTokensRead` (distinct keys touched per sequence): full `ceil(kvLen/r)`; window
`min(ceil(kvLen/r), W + qLen − 1)`; top-k `min(ceil(kvLen/r), topK·qLen)` (worst case: disjoint selections).

### Attention

`metrics.ts: attentionFlops, attentionBytes, attentionPairs`. Fused QKᵀ → softmax → ·V (FlashAttention-style:
scores never reach memory). An optional **secondary key set** (`AttentionOp.secondaryKv: AttentionKvSource`, CSA's
compressed global entries next to its raw window) shares the same softmax: Q is read once and O written once. Its
`kvSelection` is `topk` (an indexer's entries) or `full` (CSA `{kind: 'all'}`: every visible entry, `p/r` at
position p).

- `pairs = attendedPairs(primary) + attendedPairs(secondary, r)` (`attentionPairs`, also used by the shape model)
- `F = 2·seqs·heads·pairs·(qkHeadDim + vHeadDim)`: tensor FLOP of Q·Kᵀ and P·V only
- softmax scores `S = seqs·heads·pairs` (`attentionScores`): the (query, key, head) triples the kernel computes, so
  the causal triangle (≈ half the square at prefill), window and top-k caps and compressed entries count as for the
  FLOP, and decode is `heads·kvLen` per sequence. The softmax is not in `F`, but it is in the op's time: one exp
  and `SOFTMAX_VECTOR_FLOP_PER_SCORE` = 5 vector FLOP per score, overlapped with the MMAs (§7 "Attention
  softmax"). Linear attention (`recurrent` ops) has no softmax.
- read Q `seqs·qLen·heads·qk·bpe(act)`, write O `seqs·qLen·heads·v·bpe(act)`
- kvRead `seqs·kvTokensRead·kvRead.elemsPerToken·bpe(kvRead)`; kvWrite `seqs·w·kvWrite.elemsPerToken·bpe(kvWrite)` with
  `w = kvWriteTokens ?? qLen` (`attentionBytes`; a block drafter attends over its new rows but keeps only the m_ctx
  context rows, §13 "Draft ops")
- secondary: kvRead `seqs·kvTokensRead(sel, r)·elemsPerEntry·bpe`; kvWrite `seqs·qLen·elemsPerNewToken·bpe` (the
  owner of a compressed cache writes `entryElems/r` per token, a consumer 0). Lowering it as two attention ops
  would add a second Q read, an O round trip and a merge, which in batched decode cost as much as the compressed
  KV read itself.

### Indexer

`metrics.ts: indexerFlops, indexerBytes, indexerPairs`. DeepSeek DSA "lightning indexer":
`score(q, s) = Σ_h w_h·ReLU(q_h·k_s)`, top-k per query. It scores every visible indexer key, each pooling
`p = keyPoolTokens` tokens (GLM k-pool, Qwen QSA, DeepSeek compressed entries), at most `maxScoredKeys` per query
(DeepSeek's hierarchical candidate pool).

- `pairs_I = attendedPairs(qLen, kvLen, causal, cap = maxScoredKeys ?? ∞, r = p)`; `F = 2·seqs·heads·pairs_I·headDim`
- read q `seqs·qLen·heads·headDim·bpe(act)` + head weights `seqs·qLen·heads·bpe(act)` (only with `headWeights`)
- key cache read `seqs·min(ceil(kvLen/p), (maxScoredKeys ?? ∞)·qLen)·keyElemsPerKey·bpe(key)`; write
  `seqs·qLen·keyWriteElemsPerToken·bpe(key)` (an owner writes `hd_I/p` per token; a layer reading another's keys 0)
- indices `seqs·qLen·topK·4`, `topK` in indexer keys

### Recurrent

`metrics.ts: recurrentFlops, recurrentBytes`; `lower/linearAttention.ts: stateFlopsPerElemPerToken`.
State per sequence = `heads·dk·dv` elements.

- `F = seqs·qLen·heads·( c_s·dk·dv + (chunked ? (C+1)·(dk+dv) : 0) )`; the second term is causal attention
  inside a chunk of C tokens ((C+1)/2 keys per query on average, 2 FLOP per element over dk + dv).
- Mamba2 (chunked SSD, Mamba-2 paper §6; §5 "Mamba-2"): with T = seqs·qLen, H = heads, G = qkHeads (B/C
  groups), N = dk, P = dv, Q = chunkLen,
  `F = T·( H·c_s·N·P + (chunked ? (Q+1)·(G·N + H·P) : 0) )`. The scores C·Bᵀ inside a chunk are shared by
  the heads of a group, so their `(Q+1)·N` term is counted once per **group**; `(L∘CBᵀ)·X` (`(Q+1)·P`) is per
  head. Step (decode): `F = T·H·5·N·P` (`h ← exp(dt·A)·h + (dt·x)⊗B`: decay 1, outer product 1, add 1;
  `y = C·h`: 2). Nemotron-3.5-Lightning layer at tp 1 (H 64, G 8, N 128, P 64): decode 2,621,440 FLOP per
  token; prefill qLen 4096, Q 128: 4096·(64·4·8192 + 129·(8·128 + 64·64)) = 11,295,260,672 FLOP.
- `c_s` (documented approximations): chunked 4 for all variants (inter-chunk output 2 + state update 2);
  step: GatedDeltaNet and KDA 7 (Sᵀk 2 + delta outer product 2 + decay 1 + output 2; KDA's per-channel decay is
  still 1 FLOP per state element), Mamba2 5, linear 4.
- state read `hasPriorState ? seqs·heads·dk·dv·bpe(state) : 0`, state write
  `(stateSnapshots ?? 1)·seqs·heads·dk·dv·bpe(state)` (a speculative verify writes k + 1 snapshots, §13 "Verify");
  chunked mode keeps intermediate chunk states on chip.
- read q, k `2·T·qkHeads·dk·bpe(act)`, v `T·heads·dv·bpe(act)`, gates `T·heads·g·bpe(act)` with
  `g = gateElemsPerTokenPerHead` (2 for GatedDeltaNet / Mamba2; KDA `dk + 1`: per-channel decay plus β); write o
  `T·heads·dv·bpe(act)`.

### Elementwise

`metrics.ts: opFlops (elementwise), elementwiseBytes`; `lower/common.ts: ELEMENTWISE_FLOPS_PER_ELEM`,
`SHAPE_DEPENDENT_FNS` (set by their lowering).
`F = elems·flopsPerElem`; bytes are the op's explicit fields (activation read/write, weights, state, and
`kvWriteBytes` for a write-only cache op, counted as KV writes).

| fn             | FLOP / elem                                                      | fn                         | FLOP / elem                                      |
| -------------- | ---------------------------------------------------------------- | -------------------------- | ------------------------------------------------ |
| rmsnorm        | 4                                                                | layernorm                  | 8                                                |
| residual_add   | 1                                                                | rope                       | 3                                                |
| qk_norm        | 4                                                                | silu                       | 4                                                |
| silu_mul       | 5                                                                | gelu_mul                   | 9                                                |
| gelu           | 8                                                                | relu                       | 1                                                |
| relu_mul       | 2                                                                | sigmoid_mul                | 5                                                |
| gated_rmsnorm  | 9                                                                | causal_conv1d              | 2·kernelWidth                                    |
| router_topk    | 5 per logit                                                      | moe_combine                | 2                                                |
| hc_collapse    | 2·N per output elem                                              | hc_expand                  | 2N + 2 (mixing), else 2                          |
| sinkhorn       | 4·iters per T·N² elem                                            | ngram_gate                 | 8 (two norm statistics, dot, sigmoid, fused add) |
| kv_pool        | 4 (softmax weight, multiply, add)                                | causal_conv1d (+ residual) | 2·K + 1                                          |
| kv_cache_write | 1 (cast to the KV format; §13 "Draft-cache prefill")             | candidate_score            | 2 (DFlash 2 selector multiply-add)               |
| situ           | 9 (β·tanh(x/β) 5, sigmoid 3, multiply)                           | situ_mul                   | 15 (situ + tanh bound on `up` 5 + multiply)      |
| residual_mix   | 6·C per output elem over C candidates (§5 "Attention Residuals") | relu2                      | 2 (max, square; ungated only)                    |

The DFlash 2 dynamic convs (§13 "Draft ops") are `causal_conv1d` ops with an explicit `flopsPerElem = 4·K` (base
and dynamic taps, applied in both the prepare and the finish step).

These only feed the vector time; elementwise ops are memory-bound in practice.

### Embedding

`metrics.ts: embeddingBytes`. `F = 0`; weights `tokens·hidden·bpe(w)` (gathered rows, conservative); write
`tokens·hidden·bpe(out)`.

### Collective

No FLOP, no memory traffic (`opBytes` returns zeros); its time is `messageBytes` through the alpha-beta model
(§6).

### Transfer

`ir/ops.ts: TransferOp` (deployment knob `offload`, §12). A read of off-device weight bytes in front of the op
that needs them: physical read classes `reads [{ count, bytesEach }]` (expected counts, real) and the `usefulBytes`
that cross the host link. No FLOP and no device-memory traffic (`opBytes` returns zeros; the DMA write into device
memory is a documented approximation); its time is the tier/link model of §12 "Transfer path". Its
`prefetchedFraction` q in [0, 1] is the share of that time that may hide behind the stage's compute (§12 "Expert
prefetch").

---

## 4. Hardware

Schema: `src/sim/hw/types.ts` (`DeviceSpec`, `ComputeEngine`, `MatmulPeak`, `GemmTiling`, `MemorySpec`,
`LinkSpec`, `TopologySpec`, `HardwareStack`). Presets: `src/sim/hw/presets/` (registry `index.ts`:
`DEVICE_PRESETS`, `DEVICE_PRESET_GROUPS`, `DEVICE_UNLOCKS`, `deviceUnlocks`, `singleDevice`; the vendor files,
`LINK_PRESETS` and `TOPOLOGY_PRESETS`: see "Preset library" below). Unlock axes: `src/sim/hw/unlocks.ts`.

- A device has ≥ 1 **compute engine** (GPU tensor cores, NPU, TPU MXU); a run uses exactly one
  (`HardwareStack.engineId`, default `engines[0]`). An engine lists matmul peaks, vector (non-tensor) peaks,
  compute units (the wave granularity) and tiling parameters, and may cap the memory bandwidth it can reach
  (`memBandwidthCapBytesPerSec`, e.g. an NPU on a narrower fabric). It also gives its SFU exp2 rate
  (`sfuOpsPerSec`, "SFU rate" below) and, where kernels run GEMMs of a few rows on the vector ALUs, a GEMV row
  limit (`gemvMaxRows`, "GEMV path" below).
- **Memory** is `dedicated` (GDDR / HBM) or `unified` (one pool shared with the CPU and OS: Apple M-series, DGX
  Spark, Strix Halo). `capacityOptionsBytes` lists SKU configurations (`[0]` default;
  `HardwareStack.memCapacityBytes` must be one of them).
- `l2Bytes` is recorded but not used by the v1 model.

### Peak lookup and fallback

`src/sim/hw/peaks.ts: matmulPeakFlopsPerSec, nextFallbackInput, vectorPeakFlopsPerSec`.

1. The exact (input, accumulate, sparsity) entry.
2. If `fp16` accumulate is missing, the `fp32`-accumulate entry (datacenter parts list a single rate).
3. If `2:4` is missing, the dense entry.
4. Otherwise walk the input chain fp4 → fp6 → fp8 → fp16; int4 → int8 → fp16; bf16 ↔ fp16; tf32 → fp32 (visited
   set; crossing integer ↔ float switches the accumulator to int32 / fp32) and repeat 1–3.
5. End of chain: the engine's vector fp32 peak.

Every non-exact hit pushes one warning naming the requested and the used key (they appear in
`SimResult.warnings`). Each step moves to a format at least as wide, so a fallback never overstates the rate.
Vector peak: the bf16 / fp16 entry for those dtypes, else fp32. Validation requires every float input with a
peak to list an fp32-accumulate entry.

**V100 format fallback.** Volta has only dense FP16 tensor cores (FP16 or FP32 accumulation): no BF16,
TF32, FP8, INT8/INT4 tensor or 2:4 path. BF16 GEMMs/attention therefore use FP16/FP32 at 125 TFLOP/s
(SXM2) or 112 TFLOP/s (PCIe), with a warning explaining the cast (real engines need, e.g., vLLM
`--dtype=half`). The simulation deduplicates identical peak warnings. FP8/INT8 math also falls back to
FP16 tensor math (integer accumulation switches to FP32); narrow weights with wider activations already use
the wider math format. This models cast/dequantize-to-FP16 computation, not native low-bit support or its
conversion overhead. Storage and KV bytes keep the selected dtype; FP16 range/overflow is not modeled.
BF16 vector work without a BF16 vector entry still uses FP32, not FP16, per the generic vector rule.
Sources: [Volta whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf),
[A100 whitepaper, V100 comparison](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf).

### Shape efficiency

`src/sim/hw/efficiency.ts: gemmComputeSec`.

```
P = computeUnits · ctasPerUnit                        (concurrent CTA slots)
for (A, B) in {(m, n), (n, m)}                        (swap-AB: the small token dim may sit on the MMA N side)
  A' = roundUp(A, mmaM); B' = roundUp(B, mmaN); K' = roundUp(k, mmaK)
  for t in ctaTiles:
    tm = min(t.m, A'); tn = min(t.n, B')              (kernels shrink tiles to small dims)
    tiles = batch · ceil(A'/tm) · ceil(B'/tn);  waves = ceil(tiles / P)
    sec = waves · 2·tm·tn·K' / (peak / P)             (full padded tiles at the per-slot peak)
computeSec = min over all;  efficiency = (2·batch·m·n·k / peak) / computeSec
```

Non-GEMM tensor ops use the same function as an equivalent batched GEMM (`src/sim/engine/opTime.ts:
attentionShape, indexerShape, recurrentChunkShape`), and their compute time is `F / (peak · efficiency)`:

- attention: `batch = seqs·kvHeads, m = qLen·heads/kvHeads` (GQA rows packed), `n = ceil(pairs/qLen), k = qkHeadDim`
- indexer: `batch = seqs, m = qLen·heads, n = ceil(pairs_full/qLen), k = headDim`
- recurrent chunked: `batch = seqs·heads·ceil(qLen/chunkLen), m = chunkLen, n = valueDim, k = keyDim`
- recurrent step: the vector peak of `stateDtype`, no efficiency term

Worked examples on the synthetic GPU (100 units, tile 128×128, MMA 64×8×16; `hw/efficiency.test.ts`):

- **M = 1** (decode GEMM, N 12800, K 1024): unswapped, the row pads to 64 of a 128-row tile → efficiency 1/64.
  Swapped, the token sits on the N side, padded to 8, tile 128×8 → 100 tiles, one wave, efficiency 1/8 = 0.125.
  Such GEMMs are memory-bound anyway.
- **Wave quantization** (128 × 12928 × 1024): 101 tiles on 100 slots → 2 waves → efficiency 101/200 = 0.505.
- On the RTX 5090 (170 SMs), the REF_DENSE_8B prefill GEMMs at 4096 tokens reach 0.96–1.0 (§10).

Not modeled: split-K / stream-K (pessimistic for few-tile shapes, which are memory-bound anyway) and
shared-memory bandwidth penalties of small tiles.

### GEMV path

`src/sim/hw/efficiency.ts: gemvComputeSec`; `src/sim/engine/opTime.ts: gemmOpTime`; `ComputeEngine.gemvMaxRows`.

A GEMM with very few token rows does not need the MMA unit: GEMV kernels compute each output as a SIMD dot product
on the vector ALUs, so the token row is not padded to the MMA shape and no tile or wave quantization applies. An
engine that sets `gemvMaxRows` gets that path for a GEMM with `m ≤ gemvMaxRows`, dense weights and fp16 / bf16 /
fp32 math (int8 / fp8 / fp4 math has no vector peak and keeps the MMA path):

```
t_gemv    = F / vectorPeak(math)               (like an elementwise op: every lane busy, no efficiency term)
t_compute = min(t_mma, t_gemv)                 (t_mma: the shape-efficiency time above)
```

When the GEMV path wins, the op reports `computePath: 'gemv'`, `efficiency 1` and the vector peak as
`peakFlopsPerSec`; the memory term is unchanged. The Apple engines set `gemvMaxRows = 1` (`presets/apple.ts`): MLX
(`matmul.cpp` `gemv`, `quantized.cpp` `qmv`, commit fc344c98) and llama.cpp Metal (`ggml_metal_mul_mv_mma_rows_min`,
commit a0ad695) both run m = 1 as a vector GEMV for every weight format; larger limits are format- and
chip-specific (MLX affine 4-bit up to m 3 on M3/M4), so 1 is the conservative choice. On M3/M4 the SIMD FMA rate
equals the `simdgroup_matrix` rate, so this removes the MMA padding only (the 8 × 8 tile made an m = 1 GEMM run at
1/8 of the peak). On M5 the Neural Accelerator engine falls back to the SIMD ALUs for these GEMMs (its vector peaks
are the SIMD engine's): NAX at m = 1 would run 1024/8 = 128 FLOP/core/clock, below the SIMD fp16 512. Example
(`opTime.test.ts`): a MoE router GEMM (m 1, n 128, k 4096, bf16) on the M3 Ultra becomes memory-bound.
Other engines leave `gemvMaxRows` absent, the MMA path only; their m = 1 GEMMs are memory-bound either way.

### SFU rate

`ComputeEngine.sfuOpsPerSec`; per-family rates and their provenance in `src/sim/hw/presets/sfu.ts` (`SfuRate`,
`engineSfu`, `nvidiaSfu`). The special-function units (NVIDIA MUFU, AMD transcendental unit, Intel extended math,
Apple's SIMD transcendental path) compute the softmax exponential, one FP32 `exp2` per attention score (§7
"Attention softmax"). Per device, in results/s:

```
sfuOpsPerSec = compute units · exp2 results per clock per unit · clock     (the clock of the vector FP32 peak)
```

| Family (presets)                                                                               | exp2 / clock / unit | Status                                                                           |
| ----------------------------------------------------------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NVIDIA Volta, Ampere, Ada, Hopper, B200, GeForce Blackwell (V100, A100, RTX 3090 / 4090 / 5090, H100, H200, B200) | 16 / SM             | sourced: CUDA guide throughput table (cc 7.0, 8.0, 8.6, 8.9, 9.0, 10.0, 12.0). GA100/GV100 have 64 FP32 lanes/SM, so their clock uses 128 FP32 FLOP/clk/SM, not 256. |
| NVIDIA GB10 (DGX Spark)                                                                        | 16 / SM             | estimate: cc 12.1 is not in the table; the cc 12.0 rate                          |
| NVIDIA Blackwell Ultra (B300)                                                                  | 32 / SM             | sourced: Blackwell Ultra blog ("SFU throughput doubled"), FA4 paper              |
| AMD CDNA3 (MI300X, MI325X)                                                                     | 16 / CU             | estimate: quarter-rate transcendentals (AMD RDNA deck "¼ rate (like GCN)", LLVM) |
| AMD CDNA4 (MI355X)                                                                             | 32 / CU             | sourced 2×: CDNA 4 whitepaper "transcendental rates increased by 2x"             |
| AMD MI455X                                                                                     | 64 / unit           | estimate, no source: CDNA4's ratio to vector FP32 (the upper-bound rule)         |
| AMD RDNA 3.5 / 4 (Strix Halo, R9700, RX 9070 XT)                                               | 16 / CU             | estimate: one quarter-rate unit per SIMD32                                       |
| Intel Xe2 (Arc Pro B70 / B65 / B60, B580)                                                      | 32 / Xe core        | estimate, no source: the upper end of 16–32                                      |
| Apple M3 / M4 / M5 GPU (both M5 engines)                                                       | 32 / core           | estimate: philipturner/metal-benchmarks M1 measurement, assumed for M3+          |

The vendors' "TeraExponentials/s" figures (Hopper 4.5, Blackwell 5, Blackwell Ultra 10.7) are full-die numbers
(≈ 144 / 160 SMs); the presets use the per-SKU derivation instead (H100 4.19e12, B200 4.69e12, B300 9.38e12
exp2/s). An estimate takes the upper end of its plausible range, because a rate that is too low could push the
bound below a real kernel. `hw/presets.test.ts` checks that every engine sets a covered rate between 1/64 and 1/2
of its vector FP32 FLOP/s (at most one exp per FP32 lane per clock), and that B300 = 2 × B200. Absent on an engine
= the softmax is not costed (the synthetic test devices).

### Estimated fields

Every numeric leaf of a preset is covered by a `sources[].fields` path (a prefix covers its subtree) or an
`estimated[]` entry with a reason (`src/sim/hw/validate.ts: validateDevice, validateLink, validateTopology,
numericLeafPaths, isPathCovered`; enforced by `hw/presets.test.ts`). Vendors do not publish CTA tile shapes or
CTAs per SM, so those are `estimated` (typical CUTLASS tiles, one CTA per SM). Validation also checks finite,
non-negative numbers, sparse ≥ dense for the same key, an fp32-accumulate entry per float input, tiles that are
multiples of the MMA shape, non-empty capacity options, unique engine ids, `hostBounce` present iff
`p2p = 'host_bounce'`, and that every rate the engine divides by is > 0 (memory bandwidth, matmul and vector
peaks, an engine's memory cap, link and host-bounce bandwidth) with a vector fp32 peak present.
`validateInput` (`src/sim/engine/validate.ts`) runs these on the **effective** device (after modifiers, or the
unmodified base) and topology of every simulation, so an invalid or unsourced spec never reaches the cost code.

### Device variants and modifiers

`src/sim/hw/modifiers.ts: applyDeviceModifiers, applyLinkModifiers, resolveDeviceVariant,
deviceModifierErrors, linkModifierErrors, describeDeviceModifier, describeLinkModifier`;
`src/sim/hw/validate.ts: validateDeviceModifiers, validateLinkModifiers`.

A variant (VRAM mod, tensor-rate unlock, power limit, P2P driver) is **a base spec plus an ordered list of typed
partial overrides**, not a copied spec. Device and link modifiers are separate unions, because P2P is a
property of the link.

| Kind                 | Effect                                                                                                                           | Paths its provenance covers                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `memoryCapacity`     | replaces `memory.capacityOptionsBytes`                                                                                           | `memory.capacityOptionsBytes`                                                          |
| `memoryBandwidth`    | replaces `memory.bandwidthBytesPerSec`                                                                                           | `memory.bandwidthBytesPerSec`                                                          |
| `peakScale`          | multiplies every matmul peak matching ALL given filters (engineId, input, accumulate, sparsity) by `factor`; must match ≥ 1 peak | `engines.<i>.matmulPeaks`                                                              |
| `clockScale`         | multiplies all matmul AND vector peaks and the SFU rate of an engine (default all) by `factor`; memory bandwidth unchanged       | `engines.<i>.matmulPeaks`, `engines.<i>.vectorFlopsPerSec`, `engines.<i>.sfuOpsPerSec` |
| `vectorScale`        | multiplies the vector peaks of an engine (default all) by `factor`: only the `dtype` entry when given, else every listed one; must match ≥ 1 | `engines.<i>.vectorFlopsPerSec.<dtype>` (each scaled entry)                            |
| `computeUnits`       | sets `computeUnits` of an engine (default all) to N and multiplies its matmul and vector peaks and SFU rate by N / units (per-unit rate and clock kept) | `engines.<i>.computeUnits`, `engines.<i>.matmulPeaks`, `engines.<i>.vectorFlopsPerSec`, `engines.<i>.sfuOpsPerSec` |
| `hostInterface`      | replaces `hostInterface.generation` and / or `.lanes`, each given field alone (Gen2 from one axis + x16 from another = Gen2 x16); needs a `hostInterface` | `hostInterface.generation`, `hostInterface.lanes` (the given ones)                     |
| link `p2p`           | sets `p2p`; `'direct'` drops `hostBounce`, `'host_bounce'` requires it                                                           | `p2p`, `hostBounce`                                                                    |
| link `linkBandwidth` | replaces `bandwidthPerDirectionBytesPerSec`                                                                                      | `bandwidthPerDirectionBytesPerSec`                                                     |

- Applied **in list order**; scales compose multiplicatively (`clockScale 0.8` then `peakScale ×2` on bf16 =
  209.5 · 0.8 · 2 = 335.2 TFLOP/s).
- Pure: the base is never mutated (explicit spreads; `structuredClone` is not in the ES2022 lib).
- **Provenance**: every modifier carries a `note` and `sources` (or an `estimatedReason`); applying appends them
  to the spec's `sources` / `estimated` and records the list in `appliedModifiers`.
- Validation: `factor > 0`, the engineId exists, a `peakScale` / `vectorScale` filter matches something (so a typo
  cannot silently do nothing), a `computeUnits` count is an integer ≥ 1, a `hostInterface` modifier gives ≥ 1 valid
  field (generation 1–5, lanes 4 / 8 / 16) of a device that has a host interface, provenance present, and the result
  passes `validateDevice`.
- The modified `hostInterface` is the one every consumer reads: the engine's default host link for offloading
  (`engine/host.ts: resolveHost`) and inter-node host staging (`engine/staging.ts`) take the effective device after
  the modifiers, and the UI builds its host link and its "(device)" label from the resolved device.
- `simulate` applies `HardwareStack.deviceModifiers` and `linkModifiers` **once, up front**
  (`src/sim/engine/simulate.ts: effectiveHardware`); all cost code sees plain specs. Each applied modifier
  becomes a `variant: …` line in `SimResult.warnings`, and `SimResult.effectiveDevice` is the modified spec.

### Preset library

`src/sim/hw/presets/index.ts: DEVICE_PRESET_GROUPS, DEVICE_PRESETS, DEVICE_UNLOCKS, deviceUnlocks, singleDevice`;
checked by `src/sim/hw/presets.test.ts` (every preset) and one `<vendor>.test.ts` per file (vendor ratios and
spot values against the datasheet).

| File                           | Covers                                                                                                                                                                          | Export                                                                                                                   |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `presets/nvidia.ts`            | GeForce RTX 3090, 4090, 5090 and their unlock axes                                                                                                                              | `NVIDIA_CONSUMER_DEVICE_PRESETS`, `RTX_3090_UNLOCKS`, `RTX_4090_UNLOCKS`, `RTX_5090_UNLOCKS`                             |
| `presets/nvidia-cmp.ts`        | CMP 170HX 8 GB (GA100 mining card, no vendor datasheet: review / community sources, peaks derived from A100 per-SM rates) and its unlock axes                                    | `NVIDIA_CMP_DEVICE_PRESETS`, `CMP_170HX_UNLOCKS`                                                                         |
| `presets/nvidia-pro.ts` | RTX PRO Blackwell 6000 Workstation / Max-Q, 5000 (48/72GB), 4000 SFF; sustained-clock options | `NVIDIA_PRO_DEVICE_PRESETS`, `RTX_PRO_6000_WS_UNLOCKS`, `RTX_PRO_6000_MAX_Q_UNLOCKS` |
| `presets/nvidia-datacenter.ts` | DGX Spark (GB10), H100 SXM, H200 SXM, B200, B300, A100 SXM4 / PCIe 40/80GB, V100 SXM2 / PCIe 16/32GB                                                                            | `NVIDIA_DATACENTER_DEVICE_PRESETS`                                                                                       |
| `presets/apple.ts`             | M5 Ultra, M5 Max, M4 Max, M3 Ultra, M3 Max; binned GPU-core counts as separate SKUs                                                                                             | `APPLE_DEVICE_PRESETS`                                                                                                   |
| `presets/amd.ts`               | Radeon AI PRO R9700, RX 9070 XT, Ryzen AI Max+ 395 (Strix Halo iGPU), Instinct MI300X / MI325X / MI355X / MI455X                                                                | `AMD_DEVICE_PRESETS`                                                                                                     |
| `presets/intel.ts`             | Arc Pro B70, B65, B60, Arc B580                                                                                                                                                 | `INTEL_DEVICE_PRESETS`                                                                                                   |
| `presets/links.ts` | PCIe Gen3/4/5, NVLink bridge / NVLink 2/3/4/5, InfiniBand HDR200/NDR400/XDR800, ConnectX-7, xGMI, Thunderbolt, USB4, Ethernet | `LINK_PRESETS`, `HOST_BOUNCE_PCIE_LINKS` |
| `presets/topologies.ts` | PCIe boxes, SoC clusters (with fabric shape), HGX A100/H100/B200/B300 nodes, A100 bridge pairs, V100 4/8-GPU NVLink boards and Gen3 PCIe boxes, MI300X xGMI node | `TOPOLOGY_PRESETS`, `pcieBox`, `socCluster`, `hgxCluster`, `nodeTopology`, `fabricIdSuffix` |
| `presets/catalog.ts`           | The setup catalog: what each device composes into (GPUs per machine, machines, local and network links, cabling), validation, `composeTopology`, SKU labels, the popular setups | `setupOptionsFor`, `allowedCablings`, `defaultCabling`, `setupErrors`, `composeTopology`, `setupLabel`, `POPULAR_SETUPS` |

`DEVICE_PRESET_GROUPS` is the display order: consumer / prosumer first (GeForce, NVIDIA RTX PRO, Apple, AMD Radeon and Strix
Halo, Intel Arc, DGX Spark), datacenter last; `DEVICE_PRESETS` is the groups flattened. Within a group: newest
generation first, then by tier (Ultra before Max, more cores first).

Conventions every preset follows (each file's header states its own traps):

- **Source kinds** (`SourceRef.kind`): `'vendor'` (absent = vendor; datasheet, whitepaper, ISA manual, official
  spec page), `'review'` (independent technical reviews and databases: Chips and Cheese, TechPowerUp, ...),
  `'community'` (forum, GitHub, seller or user measurements). Every source has an `https://` URL and a title,
  and its `note` quotes the table or sentence the numbers come from.
- **Estimated fields**: numbers no vendor publishes (CTA tiles, CTAs per unit, Apple GPU FLOPS derived as cores ×
  FLOP/clk × a community-measured clock; the M3 Max, a MacBook Pro-only chip with Thunderbolt 4, uses the same
  1.398 GHz M2 Max DVFS clock estimate as the M3 Ultra; Intel FP16 XMX from per-core rates, dense rates derived as ½ a sparse
  headline) are in `estimated[]` with the derivation as the reason, or in a source whose note shows the
  arithmetic. Peaks are per device at the clock the header names (vendor boost / max clock unless stated).
- **Single-issue vs dual-issue vector**: AMD's RDNA "FP32 Vector" headline counts dual issue (VOPD / wave64),
  which compilers rarely emit for FMA; `vectorFlopsPerSec.fp32` is the single-issue rate (half the headline).
  CDNA, NVIDIA and Intel have no such caveat.
- **Sparse-headline trap**: NVIDIA datacenter and DGX Spark headlines are 2:4 sparse ("1 PFLOP FP4" on DGX
  Spark is sparse; dense is half). Sparse figures go only in `'2:4'` entries; dense entries hold the printed
  dense figure, else ½ the sparse one (estimated). AMD's "4:2" structured sparsity is the same pattern and is
  recorded as `'2:4'`.
- **GeForce fp32-accumulate half rate**: on GeForce, fp16 / fp8 matmuls with fp32 accumulate run at half the
  fp16-accumulate rate, and bf16 / tf32 accumulate only in fp32, so GeForce lists both accumulators.
  RTX PRO, datacenter parts (including A100/V100), DGX Spark, AMD and Intel run both accumulators at one rate
  and list only fp32-accumulate entries (peak lookup step 2 maps an fp16-accumulate request onto them).
  RTX PRO 5000 / 4000 SFF full-rate FP32 accumulation is an estimate; the 6000 WS / Max-Q rates are vendor-cited.
- Formats a vendor gives no rate for are omitted (the lookup falls back to a wider format and warns), not
  invented. Capacities are recorded as quoted (SI bytes), unified-memory devices list their SKU options.

### Unlock axes

`src/sim/hw/unlocks.ts: UnlockAxis, UnlockOption, UnlockSelection, resolveUnlocks, allUnlockSelections,
unlockAxesErrors`; per device in `src/sim/hw/presets/index.ts: DEVICE_UNLOCKS` (keyed by `DeviceSpec.id`).

An **unlock axis** is one independent way a base device is modified in practice. Its options are mutually
exclusive; "stock" is always implied and not listed. Each option is just lists of the typed modifiers above
(`deviceModifiers`, and `intraNodeLinkModifiers` for the topology's intra-node link), so the base preset is
never copied and every number keeps its provenance.

- UI: one control per axis: a **checkbox** when the axis has one option, a **radio group** (stock + options) when
  it has several. Axes combine freely.
- `resolveUnlocks(axes, selection)` concatenates the chosen options' modifiers **in axis order** (not selection
  order), ready for `HardwareStack.deviceModifiers` / `linkModifiers.intraNode`; an unknown axis or option id
  throws (a UI bug).
- `allUnlockSelections(axes)` enumerates all ∏ (options + 1) selections. `hw/presets.test.ts` checks, for every
  device with unlocks, `unlockAxesErrors == []` and that EVERY selection passes `validateDeviceModifiers` and
  (link modifiers) `validateLinkModifiers` on each stock host-bounce PCIe link (`HOST_BOUNCE_PCIE_LINKS`).

Axes per consumer card (`src/sim/hw/presets/nvidia.ts`):

| Card     | Axis (`id`)        | Option: modifiers                                                                                                                                                                                                  | Source class                                  |
| -------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| RTX 3090 | `tensor-rate`      | `full-rate`: `peakScale ×2` on fp16 / bf16 / tf32 with fp32 accumulate (dense and 2:4)                                                                                                                             | community (Zagitta gpu-unlock, listed stable) |
| RTX 3090 | `pcie-p2p`         | `direct`: link `p2p → 'direct'` (bandwidth and latency kept)                                                                                                                                                       | community (patched open kernel driver)        |
| RTX 4090 | `tensor-rate`      | as the 3090                                                                                                                                                                                                        | community (listed experimental)               |
| RTX 4090 | `vram-mod`         | `48gb`: `memoryCapacity [48 GB]` (bandwidth unchanged: unverified)                                                                                                                                                 | review + community (teardown, GPU Lab)        |
| RTX 4090 | `pcie-p2p`         | as the 3090                                                                                                                                                                                                        | community (tinygrad driver)                   |
| RTX 4090 | `memory-overclock` | `1545mhz`: `memoryBandwidth` 1545 MHz × 16 × 384 bit / 8 = 1186.56 GB/s (theoretical bus peak)                                                                                                                     | review (TechPowerUp OC table)                 |
| RTX 5090 | `pcie-p2p`         | as the 3090                                                                                                                                                                                                        | community (tinygrad driver)                   |
| RTX 5090 | `power-clock`      | `575w`: `clockScale` 2768 / 2407 ≈ 1.150; `400w`: `clockScale` 2465 / 2407 ≈ 1.024 (measured sustained SM clocks under an AI load, relative to the 2407 MHz boost clock the vendor peaks assume; memory unchanged) | community (measured SM clocks)                |

Rejected as UI choices (no capacity test or debunked; listed with links at the end of `nvidia.ts`): 3090 48 GB,
4090 96 GB, 5090 64 / 96 / 128 GB, a Blackwell tensor unlock. The 3090 NVLink bridge is a topology
(`RTX_3090_NVLINK_2`), not an unlock.

Axes of the NVIDIA CMP 170HX (`src/sim/hw/presets/nvidia-cmp.ts`; stock = the card as shipped: FP32 FMA, FP64 and
every tensor instruction fused to 1/32 issue rate, 8 GB, PCIe Gen1 x4). Stock peaks are the A100's per-SM rates
× 70 SMs × 1.41 GHz, ÷ 32 where fused (FP16 tensor 6.32 with FP16 and with FP32 accumulate, BF16 6.32, INT8
12.63 TOP/s, FP32 0.395, HFMA2 50.5 TFLOP/s). The axes are listed in stage order (firmware, driver, board), which is
the order their modifiers apply; every option is a complete stage, so every combination is physically possible
(`nvidia-cmp.test.ts` checks the prerequisites of all 24):

| Axis (`id`)          | Option: modifiers                                                                                                                                                                                                   | Source class                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `vbios-6d`           | `6d`: `memoryBandwidth` 1728 MHz × 2 × 4096 bit / 8 = 1769.47 GB/s, `clockScale` 1470 / 1410                                                                                                                        | community (PixelML, vBIOS flash guide)            |
| `compute-path`       | `locked-workarounds`: `vectorScale` fp32 ×16 (FADD + FMUL, 6.32 TFLOP/s); `peakScale` fp16 / **fp16-accumulate** dense ×8, 2:4 ×4 (HFMA2, 50.5); `peakScale` int8 dense ×2 (DP2A, 25.3 TOP/s, estimated)              | community (arabel1a, llama.cpp #24616)            |
|                      | `cmpunlocker`: `peakScale` all ×32; `vectorScale` fp32 and fp64 ×32; `memoryCapacity` [64 GB, 32 GB]; `hostInterface` generation 2                                                                                 | community (cmpunlocker, Consensus wiki, PixelML)  |
|                      | `cmpunlocker-hbm-1890`: the same (once) plus `memoryBandwidth` 1890 MHz × 2 × 4096 / 8 = 1935.36 GB/s (the HBM PLL overclock needs cmpunlocker's PLM patch, so it exists only here; it overrides 6D's 1769)            | community (170tune, wanghar)                      |
|                      | `cmpunlocker-74sm`, `cmpunlocker-74sm-hbm-1890`: the two above plus `computeUnits` 74 (cmpunlocker PR #58 re-enables 4 fused-off SMs: every compute rate × 74/70, 74-SM waves; part of the patched driver, so only here) | community (cmpunlocker PR #58)                    |
| `pcie-capacitor-mod` | `x16`: `hostInterface` lanes 16                                                                                                                                                                                     | community + review (Consensus wiki, niconiconi)   |

The two PCIe unlocks compose through two `hostInterface` modifiers: Gen1 x4 → Gen2 x4 (cmpunlocker), Gen1 x16
(capacitor mod), Gen2 x16 (both). The device-to-device links follow: the setup catalog offers a PCIe local link only
while it fits the EFFECTIVE host interface (`catalog.ts: LocalLinkOption.pcie, localLinkFits, allowedLocalLinks`;
`SetupChoice.hostInterface`), so a stock card offers PCIe 1.0 x4 only and both unlocks offer all four, fastest first
(the default). The UI normalizes a chosen link the unlocks no longer allow (reducer `setUnlock`, URL decoding) and
never simulates one (`src/ui/state/hardware.ts: effectiveHostInterface, normalizeLocalLink, resolveLinks`).

Peak choice for FP16 GEMMs: the table lists both accumulators (equal on GA100). The default fp32-accumulate request
resolves to the fp32-accumulate entry, which `locked-workarounds` leaves at 6.32 TFLOP/s (the fused tensor rate,
equal to an FP32-accumulating CUDA-core kernel's FMUL + FADD rate); only an fp16-accumulate request reaches the half2
rate, because HFMA2 accumulates in half. llama.cpp GGUF prefill (MMQ) uses the int8 entry. Not modelled (the preset's
JSDoc says why):
PCIe Gen3 (10 GB SKU only, after a permanent fuse burn), the x8 half capacitor mod, the static-BAR1 P2P patches, the
10 GB SKU, power-cap tuning. Other devices have no unlock axes.

### Links and topologies

`src/sim/hw/presets/links.ts: LINK_PRESETS, HOST_BOUNCE_PCIE_LINKS`;
`src/sim/hw/presets/topologies.ts: TOPOLOGY_PRESETS, pcieBox, socCluster, hgxCluster, nodeTopology, linkPresetId`;
costed in §6
(`src/sim/engine/collectives.ts`).

- **Per direction, per device**: `bandwidthPerDirectionBytesPerSec` is what one device can inject in one
  direction, never a bidirectional sum (vendor "900 GB/s NVLink" bidirectional → 450 GB/s).
- **Theoretical bandwidth, hardware-floor latency**: bandwidth is the line-rate ceiling (PCIe: the payload ceiling,
  below); `latencySec` is a one-way hardware / verbs-level floor (P2P write, `ib_write_lat`), never an NCCL / RCCL
  software floor. Latencies no vendor publishes are `estimated`.
- **PCIe payload efficiency** (`src/sim/hw/presets/links.ts: PCIE_PAYLOAD_EFFICIENCY,
pcieBandwidthPerDirectionBytesPerSec, pcieLineRateBytesPerSec`): every PCIe link (host bounce, direct P2P, the
  device's host link for offload streaming, inter-node PCIe staging, the NVMe slot ceiling) carries payload at

  ```
  R_line = GT/s per lane × 1e9 × lanes × code / 8              (code 128/130 on Gen3–Gen5, 8/10 on Gen1 / Gen2)
  η_PCIe = MPS / (MPS + overhead) = 256 / (256 + 24) = 0.914    (TLP framing, Intel AN 855)
  B      = R_line × η_PCIe                                      (Gen5 x16 57.61, Gen4 x16 28.81, Gen4 x8 14.40 GB/s)
  ```

  Gen1 2.5 and Gen2 5 GT/s per lane use 8b/10b (PCI-SIG), so Gen2 x16 = 7.31 and Gen1 x4 = 0.914 GB/s (the CMP
  170HX links `CMP_PCIE_LINKS` and its host link). `overhead` = 24 B per TLP without ECRC (STP token with the
  sequence number, a 4-dword header for 64-bit addresses, LCRC); `MPS` = 256 B maximum payload size, an assumed
  platform setting (128 B would give 0.842). DLLPs
  (ACK, flow control) and SKP ordered sets are left out, so `B` stays an upper bound: measured pinned copies on
  Gen4 x16 reach 24.8–26.2 GB/s against 28.81. Only PCIe: NVLink, InfiniBand, Ethernet, Thunderbolt and USB4 keep
  their vendor data rates. Source and review: docs/review/practicality-review.md F8.

- **One α definition for every inter-node transport** (`src/sim/hw/types.ts: LinkSpec.latencySec`, the header of
  `src/sim/hw/presets/links.ts`): the one-way small-message floor as a collective library sees it = **half the
  average (or median) round trip of a tuned small-message ping-pong** over that transport (`ib_write_lat` /
  `ibv_uc_pingpong` for RDMA, a 1-byte TCP ping-pong for TCP). An ICMP ping is not used (untuned ICMP on Strix Halo
  USB4 gave 0.119 ms minimum vs a 14.1 µs tuned TCP minimum, ≈ 8× too high), nor the minimum sample (on Strix Halo
  USB4 the minima would even invert RDMA vs TCP: 7.39 vs 7.07 µs, while the averages are 8.33 vs 8.7 µs). The
  per-step time of a small all-reduce is used only as an upper bound when no ping-pong exists (TB4 / TB5 TCP).
- **Transport** (`LinkSpec.transport`, `'rdma' | 'tcp'`; `LINK_MEDIA`, `links.test.ts`): every inter-node preset says
  how the collective library moves bytes, intra-node presets leave it absent. `'tcp'` means kernel sockets, so it
  needs `gpuDirectRdma: false` (`validateLink`) and pays the socket copy (§6 "TCP socket copy"). **RDMA ≤ TCP on one
  medium**: RDMA is zero-copy and kernel-bypass (the NIC moves the registered buffer itself), while TCP adds the
  kernel's syscalls, copies and protocol processing on top of the same wire, so on the same ports and cable RDMA's
  `latencySec` can never exceed TCP's, and neither can its `forwardingLatencySec` (a user-space verbs re-post vs a
  kernel receive and send). `LINK_MEDIA` lists the presets per physical medium as data; a test checks every
  RDMA / TCP pair of a medium and that every preset with a `transport` is listed exactly once, so new presets join.

  | medium               | `'rdma'`                   | `'tcp'`                               | α RDMA / TCP             |
  | -------------------- | -------------------------- | ------------------------------------- | ------------------------ |
  | Thunderbolt 5        | `TB5_RDMA`                 | `TB5_TCP`                             | 7 / 45 µs                |
  | Thunderbolt 4        | —                          | `TB4_TCP`, `TB4_TCP_20G`              | — / 45 µs                |
  | Strix Halo USB4      | `USB4_HALO_RDMA`           | `USB4_HALO_TCP`                       | 8 / 9 µs                 |
  | DGX Spark ConnectX-7 | `SPARK_CONNECTX_7`         | — (no Spark TCP preset)               | 1.5 / — µs               |
  | InfiniBand NDR / XDR | `IB_NDR_400`, `IB_XDR_800` | —                                     | 1 / — µs                 |
  | 5GbE / 10GbE         | —                          | `ETHERNET_5G_TCP`, `ETHERNET_10G_TCP` | — / 10 µs (direct cable) |

- **Host bounce**: stock GeForce PCIe links are `p2p: 'host_bounce'` (`HOST_BOUNCE_PCIE_LINKS`, Gen4 / Gen5 × x8 /
  x16), with `hostBounce.hostMemBandwidthBytesPerSec` an assumed dual-channel DDR5-6000 host (96 GB/s,
  estimated: replace for the actual host) and `extraLatencySec` 12 µs (community P2P matrices). The
  `pcie-p2p` unlock switches the link to `'direct'`; Intel Arc Pro boxes use the `_DIRECT` links (stock driver).
  The cost is §6 "Host bounce".
- **PCIe α = 1 µs** (every PCIe preset, estimated): the patched Gen5 x16 P2P write latency, 0.49–0.59 µs (tinygrad
  P2P matrices), rounded up; a 2-GPU ring all-reduce pays 2α. Not raised to the measured small-message PCIe
  all-reduce (one-shot 6.1 µs, NCCL 13.5 µs at 1 KB on 4× RTX PRO 6000, local-inference-lab rtx6kpro): those include
  the collective's kernel launch and a system-scope barrier, and no measurement found isolates the hardware share
  (the review's ≈ 2–3 µs is an estimate). See `PCIE_ALPHA_NOT_RAISED` in `links.ts` and practicality-review F7.
- **GPUDirect RDMA capability** (`LinkSpec.gpuDirectRdma`, `DeviceSpec.gpuDirectRdma`; `src/sim/engine/staging.ts:
needsHostStaging`; device citations in `src/sim/hw/presets/gpudirect.ts`). Whether an inter-node transport can move
  a message straight between the NIC and an accelerator's dedicated memory. A dedicated-memory device on an
  inter-node path stages every message through host memory when the link or the device lacks it (§6 "Inter-node host
  staging"). TCP never has it: NCCL's socket transport takes host pointers only (`net_socket.cc:57`, `ptrSupport =
NCCL_PTR_HOST`), and so do Gloo (`cuda_allreduce_ring.cc:82,102`) and MLX. On RDMA, both ends must support it.
  Absent = capable, the pre-staging convention, so every earlier result is unchanged; every inter-node link preset
  and every dedicated device preset sets the field (a test), and unified-memory devices must leave it absent
  (`validateDevice`).

  | preset                                                                       | `gpuDirectRdma` | source                                                                                                  |
  | ---------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
  | `ETHERNET_5G_TCP`, `ETHERNET_10G_TCP`, `TB4_TCP`, `TB5_TCP`, `USB4_HALO_TCP` | false           | NCCL `net_socket.cc:57` (commit 3e46c089f4ae); Gloo `cuda_allreduce_ring.cc:82,102`                     |
  | `TB5_RDMA`, `USB4_HALO_RDMA`                                                 | false           | RDMA into host / unified memory only (Apple TN3205; Strix Halo notes); no effect: SoCs only             |
  | `SPARK_CONNECTX_7`                                                           | false           | NVIDIA forum 352051/2: "GPUDirect RDMA technology is not supported" on Spark; no effect: unified memory |
  | `IB_NDR_400`, `IB_XDR_800`                                                   | true            | CUDA GPUDirect RDMA doc (https://docs.nvidia.com/cuda/gpudirect-rdma/index.html); ROCm PeerDirect       |
  | RTX 5090 / 4090 / 3090                                                       | false           | NVIDIA forums 154509/5 ("RDMA is not supported on geforce"), 157984/2                                   |
  | H100 SXM, H200 SXM, B200, B300                                               | true            | CUDA GPUDirect RDMA doc                                                                                 |
  | MI300X / MI325X / MI355X / MI455X                                            | true            | ROCm "GPU-enabled MPI" (PeerDirect), https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html    |
  | R9700, RX 9070 XT, Arc Pro B70 / B65 / B60, Arc B580                         | false           | estimated: no RDMA source found; moot on TCP, where they stage anyway                                   |
  | DGX Spark, Apple M-series, Ryzen AI Max+ 395                                 | absent          | unified memory: the NIC reads the shared pool                                                           |

  PXB caveat: NCCL uses GPUDirect by default only when the GPU and the NIC sit under the same PCIe switch
  (`src/graph/paths.cc:511-536`). HGX nodes meet this (DGX B200: 8 ConnectX-7, one per GPU); a workstation NIC on
  another root port would not, which no preset models.

- **Link latencies, RDMA and TCP, one method** (`src/sim/hw/presets/links.ts`, `ui/components/linkHelp.tsx`). Each is
  the one-way floor by the definition above, not what a socket collective library achieves:

  | link                 | old    | new                | evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
  | -------------------- | ------ | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `TB5_RDMA`           | 50 µs  | **7 µs**, measured | `ibv_uc_pingpong` M4 Pro ↔ M3 Ultra over TB5: "1000 iters in 0.01 seconds = 13.72 usec/iter" (one iteration = one round trip ⇒ 6.86 µs one way; Anemll, https://x.com/anemll/status/1993182652204187929). Cross-check: JACCL 4-way all-reduce ≈ 9–10 µs at 1–2 KB, one mesh step for small messages (MLX PR 2808), an upper bound. The former 50 µs was Jeff Geerling's "memory access latency … <50us", read off the all-reduce plot: a collective upper bound, not a link α. (An earlier note attributed the ping-pong to MLX PR #3900; its source is the Anemll post.) |
  | `TB4_TCP`, `TB5_TCP` | 45 µs  | 45 µs, estimated   | No tuned TCP ping-pong over a Thunderbolt bridge found. MLX PR 2808 (https://github.com/ml-explore/mlx/pull/2808): a 4-Mac TCP-ring all-reduce takes 290–300 µs flat over 1–32 KB, 2(N−1) = 6 steps of ≈ 48 µs including thread hand-offs and the reduction, an upper bound; 45 µs sits just below it. The 0.142 ms ICMP RTT (MacBook Air ↔ Linux PC, untuned) is context only.                                                                                                                                                                                           |
  | `USB4_HALO_RDMA`     | 8 µs   | 8 µs, measured     | Level1Techs usb4_rdma `ib_write_lat` 2 B, n = 10000: average 8.33 µs (minimum 7.39, 99 % 9.98). Hellas: ≈ 7 µs at 64 B. The verbs are emulated over the same USB4 DMA rings as TCP, so RDMA ≈ TCP is plausible here.                                                                                                                                                                                                                                                                                                                                                       |
  | `USB4_HALO_TCP`      | 10 µs  | **9 µs**, measured | Level1Techs (post 3), tuned 1-byte TCP ping-pong: RTT average Base 20.25 / Partial 17.35 / Full 19.56 µs (minimum 15.15 / 14.13 / 17.41 µs); α = the best-tuned average / 2 = 8.7 µs, rounded to 9 µs.                                                                                                                                                                                                                                                                                                                                                                     |
  | `SPARK_CONNECTX_7`   | 1.5 µs | 1.5 µs, measured   | `ib_write_lat` 2 B, t_typical (median) 1.45 µs (alessandrosangiorgi.net). No Spark TCP preset, so no pair.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  | `ETHERNET_10G_TCP`   | 10 µs  | 10 µs, estimated   | No 10GbE desktop ping-pong found: carried over from the 5GbE direct measurement. Cross-checks: 10GBASE-T PHY 2–4 µs; IX (OSDI'14) Linux 10GbE one-way 24 µs through a switch with epoll; Cloudflare's tuned switched UDP echo average 30.7 µs RTT.                                                                                                                                                                                                                                                                                                                         |
  | `ETHERNET_5G_TCP`    | 10 µs  | 10 µs              | Level1Techs (Framework Desktop built-in 5GbE, tuned 1-byte ping-pong): direct RTT average 20.8–22.8 µs (minimum 19.9 µs). Through a 10G switch the same port measured average 31.8–40.3 µs (minimum 29.4–32.3): a switch adds ≈ 5 µs one way; α is the direct-cable figure.                                                                                                                                                                                                                                                                                                |

  What a collective library over sockets measures in practice, for scale (the Network link's "?" says so too): NCCL
  on sockets 57–64 µs for a small all-reduce between two nodes of 4 GH200 over Slingshot TCP
  (https://github.com/eth-cscs/cray-network-stack/issues/6), 118–130 µs for 2 × H100 over AWS ENA
  (https://github.com/dstackai/dstack/issues/3069); MLX's TCP ring 290–300 µs for 4 Macs over Thunderbolt. That is
  5–10× the floor: software, which the bound leaves out.

- **A100 / V100 interconnects** (`links.ts`, `topologies.ts`): HGX A100 uses NVLink3/NVSwitch
  (600 GB/s bidirectional = 300 GB/s per direction per GPU); clusters default to one HDR200 NIC per GPU
  (25 GB/s per direction), not NDR400, which exceeds the A100 Gen4 host path. PCIe A100's three-bridge
  NVLink connection is offered for exactly two matching-capacity cards. V100 PCIe uses Gen3 x16 and has no
  NVLink (DGX Station's special SKU is not modeled).
- **V100 NVLink2 is not a switch.** `NVLINK_2_HCM_8` represents DGX-1's eight-GPU hybrid cube-mesh:
  six links × 25 GB/s = 150 GB/s aggregate injection per direction, but only four directly connected peers.
  The uniform intra-node link model is optimistic for subgroups (a quad is at most about 100 GB/s,
  a pair 25–50 GB/s) and send/recv/all-to-all to non-neighbours, which require explicit relay or PCIe.
  A distinct `NVLINK_2_MESH_4` describes four-GPU fully connected NV2 boards: 150 GB/s aggregate,
  50 GB/s per peer; TP2 subgroups are optimistic there too. The catalog restricts these links to eight
  and four devices respectively and offers a 50 GB/s NV2 pair option for two selected GPUs.
  A per-group link matrix would be needed to remove this approximation; no V100 inter-node setup is offered.
  Sources: [Volta whitepaper Figure 12](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf),
  [four-GPU board matrix](https://github.com/NVIDIA/nccl-tests/issues/251).
- **xGMI aggregate**: `XGMI_MI300X_8_GPU` (used by `MI300X_XGMI_8`) is 7 peers × 64 GB/s = 448 GB/s per
  direction, the full-mesh injection ceiling, because the collective formulas treat B as a device's aggregate
  injection; it is optimistic for subgroups and send/recv. `XGMI_MI300X_PER_PEER` (64 GB/s) is the one-peer link.
- Topologies: `pcieBox` (one node, N devices, one PCIe link each; shared switch / root-complex uplinks not
  modeled), `socCluster` (N single-SoC machines, one link budget per device, no port aggregation; Ethernet
  assumes a non-oversubscribed switch), `hgxCluster` (8-GPU NVLink nodes, one full-rate NIC per GPU), and the
  general `nodeTopology` (`devicesPerNode × nodes`, a link only at the levels that exist; the UI composes it from
  device and node counts and link presets). Device and node counts are user configuration and cite no source.
  Builders take the inter-node fabric shape (`nodeTopology({ fabric })`, `socCluster(device, n, link, fabric?)`): it
  is kept only with an inter-node link and omitted when it is the default (a non-blocking switch), so every
  pre-fabric topology is unchanged; otherwise the id ends in `-mesh` / `-mesh2` / `-ring` / `-chain` and the name
  carries the shape. Named Spark presets: `DGX_SPARK_PAIR` (one cable), `DGX_SPARK_3` (the triangle: a full mesh),
  `DGX_SPARK_4` (200GbE switch) and `DGX_SPARK_4_RING` (switchless ring), the last two citing NVIDIA's
  connect-multiple-Sparks playbook; `R9700_PCIE_4` (four R9700s, PCIe 5.0 x16 per AMD's page, host bounce: no
  cited ROCm Radeon P2P). Thunderbolt clusters are meshes; four Strix Halo boxes over their two USB4 ports are a
  ring (the direct all-to-all is no longer assumed there).
- **Setup catalog** (`src/sim/hw/presets/catalog.ts`): the UI composes hardware as device × GPUs per machine ×
  machines × link × cabling instead of picking from a flat list. `setupOptionsFor(deviceId)` gives, for every device
  preset, its SKU-exact short name, the counts offered, the local links it has (`[0]` = default) and the networks
  the machine has (`NetworkOption`: the per-port link, its USABLE `portsPerNode`, whether a switch exists, how many
  cables a pair can stripe, and an optional vendor limit on switchless size), each port count citing the machine's
  spec page. Usable ports: DGX Spark 2 QSFP (sharing the 25 GB/s cap; a switch exists; direct cabling up to 4);
  M3 Ultra Mac Studio 5 of 6 TB5 ports for RDMA (the port next to Ethernet has none: exo README) and 6 for TCP;
  M4 Max Studio 3 of 4 (same chassis rule); M5 Max Studio 3 (ESTIMATED: the rule's status on the 2026 Studio is
  unknown); M5 Ultra Studio 6 (INFERRED from Apple's 4-node mesh with two cables per pair); M3 Max MacBook Pro 3
  TB4 ports, TCP only (no TB5, so no RDMA); Strix Halo 2 USB4 ports, plus 5GbE / 10GbE through a switch. A
  `CablingId` (`switched | mesh | mesh2 | ring | chain`, the URL's `fab=`) maps 1:1 to a `FabricShape` (`fabricOf`,
  `cablingOf`). `allowedCablings(net, nodes, devicesPerNode)` applies `fabricShapeErrors` with the ports and switch:
  a switch only where one exists (Thunderbolt and USB4 have none), a mesh while (nodes − 1) · cables ≤ ports, `mesh2`
  only on TB5 RDMA, a ring from 4 machines (at 3 it is the mesh), a chain from 3, a switch only for machines with
  several GPUs, and nothing switchless beyond the vendor's direct-cabling limit; `defaultCabling` is switch, else
  mesh, else ring, else chain, so the catalog defaults equal the pre-catalog UI defaults for every device but one
  (`catalog.test.ts` pins them; the Arc B580, a PCIe 4.0 x8 card, now defaults to PCIe 4.0 x8 host bounce instead of
  the old, wrong PCIe 5.0 x16), no PCIe link offered to a card is wider or of a later generation than its
  `hostInterface` (`catalog.test.ts` checks every device; B580: Gen4 x8 host bounce / direct, Arc Pro B60: Gen5 x8
  only), and two Thunderbolt or USB4 machines are a one-cable mesh, which costs exactly a
  switch. `setupErrors` rejects impossible choices (a 5-Mac mesh on an M4 Max, a 7-Mac mesh on an M3 Ultra, two
  cables per pair on 4 M3 Ultras, a ring of 2, a switch on Thunderbolt, an 8-Spark ring); `composeTopology` builds
  the `TopologySpec` (`interNodeFabric` omitted for a switch); `setupLabel` names the exact SKU everywhere:
  `'M5 Ultra 80-core GPU, 256GB ×2 · TB5 mesh'`, `'DGX Spark 128GB ×4 · ConnectX-7 ring'`,
  `'RTX 5090 32GB ×2 · PCIe 5.0 x16'`.
- **Popular setups** (`POPULAR_SETUPS`, ≤ 10, grouped by vendor, each citing who runs it): RTX 5090 32GB ×2 PCIe 5.0
  x16; RTX 3090 24GB ×2 NVLink bridge; DGX Spark 128GB ×2 (one cable), ×3 (triangle) and ×4 ring (NVIDIA playbooks,
  SparkRing) and ×4 switch (MikroTik CRS804/812); M3 Ultra 80-core 512GB ×4 TB5 mesh (Geerling, exo #1050);
  M5 Ultra 80-core 256GB ×2 TB5 (the user's example; weak source); M5 Ultra 80-core 512GB ×4 TB5 mesh with two
  cables per pair (Apple ML Research, 2026-10-15); Strix Halo 395 128GB ×4 USB4 ring (Level1Techs). Picking one
  fills in device, memory, counts, link and cabling; the URL stores those components, never a setup id.
- **Fabric link fields, values**. `forwardingLatencySec` (α_f) is RELAY HANDLING ONLY: the engine charges α for every
  hop a message crosses (`lat(h) = h·α + (h−1)·α_f`, §6), so α_f holds only what the relaying host adds between the
  incoming completion and the outgoing send, never the extra hop's wire time. Every α_f is ESTIMATED (no vendor
  publishes one), with the derivation in the preset. `SPARK_CONNECTX_7`: `allPortsCapPerDirectionBytesPerSec` = 25 GB/s
  (both QSFP ports share two PCIe Gen5 x4 links: NVIDIA's clustering doc; 185–190 Gb/s measured over both, STH; "two
  cables will not increase performance", NVIDIA playbook) and α_f = 4.5 µs = (16.9 − 10.9 µs) measured 2-hop vs 1-hop
  10 KiB one-shot all-reduce on a host-relayed 4-Spark ring − 1.5 µs for the second hop's α (the relay host's
  completion poll, re-post and PCIe round trip; RoCE cannot route through a Spark in hardware), so a 2-hop path costs
  2 × 1.5 + 4.5 = 7.5 µs (+6 µs, the measurement) and a 3-hop one 13.5 µs. `TB5_RDMA`: no cap (one Thunderbolt
  controller per port, TN3205; 1 → 4 cables scale ≈ linearly to ~36.5 GB/s, MLX PR #3900; the SoC ceiling for 5–6
  concurrent ports is unmeasured) and α_f = 3 µs, the host turnaround of a user-space verbs relay (JACCL's ≈ 9.5 µs
  small all-reduce step minus the 7 µs one-way α; TN3205: "the application is responsible for forwarding"): 2 hops =
  2 × 7 + 3 = 17 µs and 3 hops = 27 µs. `TB5_TCP` (the Thunderbolt bridge on TB5 ports at the 80 Gb/s = 10 GB/s
  per-direction line rate, Intel's TB5 brief; not an achieved goodput, none is cleanly measured) and `TB4_TCP`
  (40 Gb/s = 5 GB/s, the M3 Max: the bonded link two Macs report, "Mac to Mac: 40 Gb/s x1, link width 0x2",
  Bergeron; `TB4_TCP_20G` = 2.5 GB/s is the same bridge when a host pair links one unbonded 20 Gb/s lane, as both
  cited Mac ↔ Linux tests did: sysfs 20.0 Gb/s and "a 20 Gigabit/s link", 15.9–18.8 Gb/s TCP measured; offered as
  the M3 Max's second network): α 45 µs (estimated, ≤ the ≈ 48 µs MLX TCP-ring step; see "Link latencies" above) and
  α_f = 20 µs, the user-space receive-to-send turnaround, about a third of the ~45–48 µs one-way TCP time that is
  itself almost all software (MLX TCP ring ~48 µs per step; TN3205: the bridge forwards in software); 2 hops
  2 × 45 + 20 = 110 µs, 3 hops 3 × 45 + 2 × 20 = 175 µs. `USB4_HALO_RDMA` α_f = 3 µs (as TB5 RDMA; 2 hops 19 µs,
  3 hops 30 µs), `USB4_HALO_TCP` 5 µs (the kernel's forwarding turnaround on a tuned stack, about half the 9 µs
  one-way; 23 / 37 µs). On each medium RDMA's α_f ≤ TCP's (3 ≤ 20 and 3 ≤ 5 µs; `LINK_MEDIA`). `presets.test.ts`
  pins these hop examples. No link was renamed; `LEGACY_LINK_IDS` (empty) is where a rename would be recorded for old
  URLs, and a TB5 Mac URL naming the TB4 bridge still resolves to that link.
- **Link ids**: `LinkSpec` has no id; `linkPresetId(link)` slugs the name (`'PCIe 5.0 x16 (host bounce)'` →
  `'pcie-5-0-x16-host-bounce'`), unique over `LINK_PRESETS`. Builder ids and the UI's URL use it.
- **Multi-port links and fabric shapes** (`src/sim/hw/types.ts: LinkSpec, FabricShape, TopologySpec.interNodeFabric`):
  on a switchless inter-node shape (full mesh, ring, chain; §6 "Fabric shapes") `bandwidthPerDirectionBytesPerSec`
  is ONE port (one cable). Two optional link fields serve those shapes only, so no switched result depends on them:
  `allPortsCapPerDirectionBytesPerSec` (C, B/s: the most one node can send, and separately receive, over all its
  ports at once, relayed bytes included, e.g. a NIC's host PCIe attachment; absent = the ports add up; must be ≥ the
  per-port bandwidth) and `forwardingLatencySec` (α_f, s: the extra latency of a node relaying a message to the next
  hop; absent = 0). Like every number, both are sourced or `estimated`. `TopologySpec.interNodeFabric` absent means
  a non-blocking switch, the convention of every topology before fabric shapes; shapes other than `'switched'` need
  one device per node (`validateTopology`).

### Host and storage tiers

`src/sim/hw/types.ts: HostMemorySpec, NvmeSpec, StorageTierSpec, HostSpec, HostInterfaceSpec, HostComputeSpec`;
`src/sim/hw/presets/storage.ts: HOST_RAM_PRESETS, SSD_PRESETS`; `src/sim/hw/presets/links.ts: hostPcieLink,
pcieBandwidthPerDirectionBytesPerSec`; `src/sim/hw/validate.ts: validateStorageTier, validateHost`. Read only when a
deployment offloads weights off the device (design: `docs/design/offload-specdec-latency.md` §2.2); with nothing
offloaded no result depends on them.

- **One host per node** (`HardwareStack.host?: { link?, ram?, ssd? }`): `ram` (host DRAM, `kind: 'host_ram'`) and
  `ssd` (NVMe, `kind: 'nvme'`) are shared by every device of the node; `link` is per device, per direction (a
  `LinkSpec`, `p2p: 'direct'`: the device DMAs host memory itself). On unified memory host RAM is the device pool
  (no separate tier, no link); only the SSD is a tier.
- **Host RAM**: `bandwidthBytesPerSec` = MT/s × 8 B × channels, the theoretical read + write peak of the node
  (`ddr5BandwidthBytesPerSec`; DDR5-6000 × 2 = 96 GB/s, the same host the host-bounce PCIe links assume);
  `latencySec` the unloaded load-to-use latency (Chips and Cheese measurements, else estimated);
  `readGranularityBytes` one 64 B cache line; `maxOutstanding` the memory-level parallelism of random reads (64,
  estimated). Capacities are configuration choices, always `estimated`.
- **NVMe**: `seqReadBandwidthBytesPerSec` the datasheet sequential read; `randomReadLatencySec` the QD1 4 KiB
  latency (1 / QD1 IOPS when only that is published); `randomRead?: { iops, queueDepth, threads }` the datasheet
  maximum and its condition (optional: absent drops the IOPS term); `maxOutstanding` = queueDepth × threads of
  that condition (or an estimate); `readGranularityBytes` 4096 (one logical block); `linkBandwidthBytesPerSec?`
  the slot ceiling when it binds below the drive (Gen4 x4 = 7.20 GB/s payload). Datasheet ceilings, not achieved rates.
  Apple SSDs have no datasheet: their numbers rest on crowd measurements (`kind: 'community'`).
  `sustainedRead?: { fraction, blockBytes, queueDepth, threads }` records low-QD / same-run deep-queue large-read
  bandwidth (dimensionless), measured block size (bytes), and measurement queue/thread counts. It affects only
  the opt-in SSD read-rate scenario (§12); absent data warns and falls back to the datasheet rate.
- **Host link**: `hostPcieLink(generation, lanes)` = GT/s × lanes × code ÷ 8 × 256/280 per direction (Gen1 2.5 and
  Gen2 5 GT/s with code 8/10; Gen3 8, Gen4 16, Gen5 32 GT/s with 128/130; the payload efficiency above), the same
  formula as the device-to-device PCIe links (Gen5 x16 = 57.614 GB/s), α = 1 µs estimated.
  `DeviceSpec.hostInterface?: { generation, lanes }` is the card's cited edge connector (electrical lanes: Arc Pro
  B60 is x16 mechanical, x8 electrical; CMP 170HX Gen1 x4 as shipped); the engine builds the default host link from
  it, after the device modifiers, so a `hostInterface` unlock (CMP Gen2 patch, capacitor mod) changes the link the
  offload streams over. SXM / OAM modules and SoCs carry none. The UI's host-link override lists Gen3–Gen5 only.
- **Defaults**: `DDR5_6000_2CH` (host RAM) and `NVME_990_PRO_2TB` (SSD), used with a warning when the stack names
  no tier.
- **CPU compute** (`HostComputeSpec`): a hook for a future CPU-compute offload mode, not costed; the mode is
  rejected by input validation.

---

## 5. Parallelism

`src/sim/lower/` (`lower.ts: lowerModel, lowerBlock`; one file per block kind; shared rules in `common.ts`).
`ParallelConfig { tp, pp, dp, ep }`, world = `tp·pp·dp` (ep groups live inside it).
The divisibility rules (heads, KV heads, widths over tp; experts and MoE modes over ep) are checked by lowering
each block once for a one-token step: `src/sim/engine/validate.ts: blockShardingErrors` (called by
`validateInput`; exported so the UI can attribute TP / EP errors to their fields).
The UI fixes dp = 1 (`src/ui/state/parallelism.ts: effectiveParallel`), so only MoE mode A is reachable there: valid
ep = the divisors of tp that divide E (and pass placement), the device count is `tp·pp`, and a dense model runs with
ep = 1. Under mode A EP adds no all-to-all ([MoE modes A/B](#moe-modes-ab); `src/sim/engine/ep.test.ts`).

### TP sharding

`lower/common.ts: shardHeads, shardWidth, tpAllReduce, linearGemm`. Query heads and intermediate widths split
evenly: `Hl = heads/tp`, `f/tp` (the divisibility is validated). Column-parallel projections (qkv, gate_up,
in_proj) then row-parallel ones (o_proj, down, out_proj), followed by one `all_reduce(tp)` of `T·h·reduceBpe`
per block. Every linear GEMM uses `math = resolveGemmMath(w, act)` and `accumulate = resolveAccumulate(math,
quant.accumulate ?? 'fp32')`. Row-parallel biases are counted on every rank.

GQA (`lower/gqa.ts: lowerGqa`): `qkv_proj n = (Hl·(gate?2:1) + 2·KVl)·hd, k = h` → [qk_norm] → rope → [indexer]
→ attention (`kvRead = kvWrite = {kvCache, 2·KVl·hd}`, selection from slidingWindow / sparse) → [sigmoid_mul]
→ `o_proj n = h, k = Hl·hd` → all_reduce. MLP (`lower/mlp.ts: lowerMlp, lowerDenseFfn`):
`gate_up n = (gated?2:1)·f/tp` → activation → `down k = f/tp` → all_reduce.

### KV-head replication

`lower/common.ts: shardKvHeads`. `KVl = kvHeads ≥ tp ? kvHeads/tp : 1` (requires `kvHeads % tp == 0` or
`tp % kvHeads == 0`): with fewer KV heads than ranks each rank holds one replicated head. The same rule applies
to `qkHeads` of linear attention. It does NOT apply to linear-attention `vHeads` (see "Linear-attention
sharding").

### Reductions dtype

`lower/common.ts: reduceBpe, reduceDtype`. All-reduce, reduce-scatter and the MoE combine carry
`reduceBpe = max(bpe(act), 2)` (at least 16-bit, so fp8 activations reduce in bf16). Gathers, dispatch and
send/recv carry `bpe(act)`.

### Norms and residuals

`lower/norm.ts: lowerNorm, lowerResidual`; `lower/common.ts: normOp`. Norm: read and write `T·dim·bpe(act)`,
gamma `dim·bpe(hp)` (×2 for layernorm); replicated on every TP rank (no sequence parallelism). Residual: read
`2·T·dim`, write `T·dim`.

### Embedding and LM head

`lower/embedding.ts: lowerEmbedding, embeddingResidentWeightBytes`; `lower/head.ts: lowerHead, headTokens,
lmHeadResidentWeightBytes`. Embedding (stage 0) is vocab-parallel: `ceil(vocab/tp)` rows per rank, then an
`all_reduce` of `T·h`. Head (last stage): final norm, then `lm_head m = Tout, n = ceil(vocab/tp), k = h`.
**Prefill computes logits only for the last position** (`Tout = seqs`); decode `Tout = seqs·qLen`.
No full-logits gather: when tp > 1, cross-shard sampling adds one `all_gather(tp)` after `lm_head`,
with full output `S = Tout·tp·8 B` (one fp32 score + uint32 token ID per rank and output position).
Both phases and every draft head use `lower/head.ts: samplingExchangeOps` (`lowerHead` and
`lower/draft.ts: sharedLmHeadOps`; audit G1). Block-drafter prefill has no head, hence no exchange.

### MLA forms

`lower/mla.ts: lowerMla, mlaForm, mlaSequenceStateBytes`. `Hl = heads/tp`.

- Common: rq > 0: `q_a (n = rq, k = h, replicated)` → rmsnorm(T·rq) → `q_b (n = Hl·(dn+dr), k = rq)`; else
  `q (n = Hl·(dn+dr), k = h)`. Then `kv_a (n = rkv+dr, k = h, replicated)` → rmsnorm(T·rkv) → rope over
  `T·(Hl·dr + dr)` → [indexer].
- **Absorbed** (decode, and prefill when sparse): `q_absorb (batch Hl, m T, n rkv, k dn)` → attention
  (`heads Hl, kvHeads 1, qk rkv+dr, v rkv, kvRead = kvWrite = {kvCache, rkv+dr}`) → `v_absorb (batch Hl, n dv, k rkv)`.
- **MHA** (dense prefill): `kv_b (m = seqs·kvLen, n = Hl·(dn+dv), k = rkv)` reading the latent → attention
  (`kvHeads Hl, qk dn+dr, v dv, kvRead {act, Hl·(dn+dv)+dr}, kvWrite {kvCache, rkv+dr}`).
- Then `o_proj (n = h, k = Hl·dv)` → all_reduce. The latent cache is replicated on every TP rank.
- Why per phase: per (query, key, head) the MHA form costs `2·(dn+dr+dv)` FLOP against `2·(2·rkv+dr)` absorbed,
  so MHA is cheaper for long dense prefill; decode is memory-bound and reads only the latent. This is a
  documented choice, not a knob.

### Sparse attention indexer

`lower/indexer.ts: lowerSparseSelection, sparseSelectionStateBytes, indexerKeyStateBytes, indexerSelection`. Ops:
`idx.q (n = H_I·hd_I, k = qIn)` (qIn = h, or rq for MLA with q compression and for CSA) → if the layer owns its
keys: `idx.k (n = coff·hd_I, k = keyInputDim ?? h)` → key LayerNorm over `T·hd_I` → with `keyPoolGate`:
`idx.pool_gate (n = coff·hd_I, k = h)` → `kv_pool` (over `T·coff·hd_I`, write `T·hd_I/p`, APE `p·coff·hd_I`) →
with `headWeights`: `idx.w (n = H_I, k = h)` →
the `indexer` op (key cache `QuantConfig.indexerKeyCache ?? kvCache`, `hd_I` elems per key, `p` tokens per key,
`hd_I/p` written per token by an owner, 0 by a layer with `ownsKeys: false`; `topK_I = ceil(topK·t/p)` indexer
keys, t = tokens per main-attention key: 1, or r for CSA). Replicated on every TP rank. The main attention gets
`kvSelection {topk, topK}`. Module formats: `idx.q` `indexer`, `idx.k` / `idx.pool_gate` `indexer_key`, `idx.w`
`indexer_head_weights`. Sequence state: the owned keys `ceil(ctx/p)·hd_I·bpe(key)`.
coff = 2 with `keyPoolOverlap` (DeepSeek-V4 indexer: its own compressor over h with overlapping windows, as in
§5 "CSA" step 3), else 1.

### Top-k reuse

A `SparseSelection {kind: 'reuse', topK}` (GLM-5.3 / Mistral Large 4 'shared' indexers, DeepSeek-V4.1 "Reuse")
emits no ops and keeps no keys: the attention gets `kvSelection {topk, topK}` from the nearest earlier indexer
(MLA stays in the absorbed form in prefill, as with an own indexer).

### KV sharing

`lower/mla.ts`. An MLA block with `kvCacheFromLayer` reads an earlier layer's latent (Mistral Large 4
`kv_source_layers`): no `kv_a`, no `kv_a_norm`, rope only on q (`T·Hl·dr`), `kvWrite` 0 elements and no
sequence state of its own (except its own indexer keys); `kv_b` / the absorb GEMMs are kept. A CSA global branch
with `kvCacheFromLayer` (DeepSeek-V4.1 consumers) runs no compressor and writes no compressed entries. MLA with
`qkRopeDim` 0 (NoPE) emits no rope op.

### CSA

`lower/csa.ts: lowerCsa, csaSequenceStateBytes, compressedKvFormat, compressorCoff, csaEntrySelection`. `Hl = H/tp`,
`Gl = G/tp` (tp must divide both);
the one K = V vector per token, the compressor, the indexer and all caches are replicated (one KV head).

1. `csa.q_a (n rq, k h)` → rmsnorm(T·rq) → `csa.q_b (n Hl·hd, k rq)` → rope over `T·Hl·dr`.
2. `csa.wkv (n hd, k h)` → rmsnorm(T·hd) → rope over `T·dr`.
3. Owner of the compressed cache: r > 1: `csa.wkv_c` + `csa.wgate_c` (n coff·hd, k h, module `attention_kv_down`)
   → `kv_pool` (over `T·coff·hd`: read `2·T·coff·hd`, write `T/r·hd`) → rmsnorm and rope over `T/r` entries;
   r = 1: `csa.wkv_c` → rmsnorm. coff = 2 with `compressorOverlap` (DeepSeek-V4 `Compressor.overlap`, r = 4: one
   hd-wide half for the previous window, one for the current, so each entry pools 2r tokens), else 1.
4. The selection: indexer ops (qIn = rq, `p = r`; keys from the hd-wide latent in V4.1, from h through the
   indexer's own overlapping compressor in V4), nothing for `reuse`, nothing for `all`.
5. One `AttentionOp`: `kvHeads 1`, `qk = v = hd`, `kvSelection {window, W}`, `kvRead = kvWrite = {kvCache, hd}`
   (K = V, so one vector), and `secondaryKv {r, topk(topK entries) or full (all), kvRead {compressedKvCache, hd},
kvWrite {compressedKvCache, owner ? hd/r : 0}}` (§3 "Attention"; `csaEntrySelection`).
6. Inverse rope over `T·Hl·dr` → `csa.wo_a` grouped (batch Gl, m T, n R_o, k H·hd/G) → `csa.wo_b` (n h, k Gl·R_o)
   → all_reduce(tp).

Sequence state: `min(ctx, W)·hd·bpe(kv)` + owner `ceil(ctx/r)·hd·bpe(compressedKv)` + compressor tail
`2·(coff·r)·(coff·hd)·4` (fp32 kv and score state, r > 1) + owned indexer keys `ceil(ctx/r)·hd_I·bpe(key)` (none
for `reuse` and `all`).

### KDA

`lower/linearAttention.ts` (variant `'kda'`, GLM-5.3-Flash, Kimi K3). `lin.qkv (n 2·qkL·dk + vL·dv + vL, k h)` (q, k,
v and β) → `lin.f_a (n r, k h, replicated)` → `lin.f_b (n vL·dk, k r)` → output gate: `lin.g_a (n r, k h)` → `lin.g_b
(n vL·dv, k r)`, or with `fullRankOutputGate` (Kimi K3 `use_full_rank_gate`) the single `lin.g (n vL·dv, k h)` →
conv, recurrent (`variant 'kda'`, gates `dk + 1` per token and head: per-channel decay plus β), gated_rmsnorm,
out_proj, all_reduce. Sharding as for GatedDeltaNet (`vL = vHeads/tp` must divide).

### Mamba-2

`lower/linearAttention.ts: lowerLinearAttention` (variant `'mamba2'`, Nemotron-H), `ir/params.ts:
linearVariantConstants`, `ir/metrics.ts: recurrentFlops`. Mapping: `qkHeads` = B/C groups G (`n_groups`),
`vHeads` = SSM heads H (`mamba_num_heads`), `keyDim` = state size N (`ssm_state_size`), `valueDim` = head dim P
(`mamba_head_dim`), `convKernel` K, `convBias` (`use_conv_bias`), `chunkTokens` Q (`chunk_size`), `outputGate`
true. The inner width is `H·P` (not `expand·h`). Same op sequence and sharding as GatedDeltaNet ("Linear-attention
sharding"); only shapes and counts differ, per device (`qkL`, `vL = H/tp`, `C = 2·qkL·N + vL·P`):

1. `lin.in_proj (m T, n 2·qkL·N + 2·vL·P + vL, k h)`: outputs z, x, B, C and ONE dt per head
   (`linearVariantConstants('mamba2').gateOutputsPerHead = 1`; GatedDeltaNet has 2, a and b).
2. `lin.conv1d` over x, B, C: `T·C` elements, taps `C·K·bpe(hp)` + [convBias] `C·bpe(hp)`, state
   `seqs·C·(K−1)·bpe(act)`.
3. `lin.recurrent` (`variant 'mamba2'`): decode step `c_s = 5`; prefill chunked SSD with
   `chunkLen = min(chunkTokens ?? 64, qLen)` and the per-group C·Bᵀ term (§3 "Recurrent"). Gates read: 2 per token
   and head (dt and its decay).
4. `lin.out_norm` gated_rmsnorm `norm(y·silu(z))` over `T·vL·P`, grouped (group size `H·P/G`); γ holds one value
   per channel, `vL·P·bpe(hp)` per device (GatedDeltaNet: one `dv`-wide γ shared by all heads).
5. `lin.out_proj (m T, n h, k vL·P)` → all_reduce(tp) of `T·h`.

Per-head vectors A_log, D, dt_bias (`3·H` params) are not streamed by any op (negligible). Sequence state per
layer `vL·N·P·bpe(state) + C·(K−1)·bpe(act)`: Nemotron-3.5-Lightning at tp 1, fp32 state, bf16 activations,
`64·128·64·4 + 6144·3·2 = 2,134,016` B; Super `4,194,304 + 61,440 = 4,255,744` B. At tp 8 Lightning has
`vL 8, qkL 1, C 768`; past `tp > G` the groups replicate (KV-head rule).

### MLA output gate

`lower/mla.ts` (`MlaAttentionBlock.outputGate`, Kimi K3 `mla_use_output_gate`; `modeling_kimi_linear.py:396-399,
467-469`). After the q path: `mla.g_proj (m T, n Hl·dv, k h, module attention)`, sharded with the heads; after the
attention core (either form): `attn.output_gate` = `sigmoid_mul` over `T·Hl·dv` (read the attention output and the
gate, write the gated output), then `o_proj`. Params `h·H·dv`. Unlike the GQA gate (a wider q projection), MLA's
gate is its own projection, since q is low-rank.

### Attention Residuals

`lower/norm.ts: lowerResidualMix` (block `residual_mix`, Kimi K3 `attn_res_block_size` B; `modeling_kimi_linear.py:
975-1047, 1075-1088`). The stack keeps ONE running residual sum and stores a summary of it at the first layer of
each block of B layers (`i % B == 0`). Before each sublayer the input is `Softmax_j(s_j) · v_j` over the C candidate
vectors `v_j` (the running sum and the stored summaries), with `s_j = ⟨RMSNorm(v_j), γ ⊙ w⟩` (one shared `h`-wide
score weight per mix: the norm's γ and the `h → 1` projection). Layer i mixes `C = 1 + ceil(i/B)` candidates before
its mixer (none at layer 0) and `C = 2 + floor(i/B)` before its FFN (`builders.ts: attnResSummariesBefore`); the final
mix before the final norm has `C = 1 + ceil(n/B)` (`attnResCollapse`, Kimi K3: 9). The add after each sublayer stays
a plain `residual_add`. One elementwise op per mix: `elems = T·h`, `6·C` FLOP per element (per candidate: RMS
statistic 2, normalize-and-score 2, weighted sum 2), read `C·T·h·bpe(act)`, write `T·h·bpe(act)`, weights
`2·h·bpe(hp)`. Replicated on every TP rank, like norms; no sequence state (the summaries live only within one
forward pass, like activations). A PP boundary carries the running sum AND the stored summaries (§5 "PP split").

### Hyper-connections

`lower/hyperConnection.ts: lowerHcPre, lowerHcPost`. Replicated on every TP rank, like norms; no sequence state.

- `hc_pre`: norm over `T·N·D` (with γ: read and write, γ `N·D·bpe(hp)`; mHC statistics only: read) → each
  projection as a GEMM (m T, module `residual_mixer`) → `sinkhorn` (`T·N²` elements × `4·iters`) if iters > 0 →
  `hc_collapse` (read `T·N·D`, write `T·D`, `2N` FLOP per output; carries the `vectorParams` weights).
- `hc_post`: `hc_expand` over `T·N·D` (read `T·N·D + T·D`, write `T·N·D`; `2N + 2` FLOP when mixing, else 2).
- The PP send carries `T·residualStreams·h` (§5 "PP split").

### n-gram embeddings

`lower/ngram.ts: lowerNgram, ngramResidentWeightBytes, ngramSequenceStateBytes, lookupTableFormat`.

1. `EmbeddingOp` gather of `T·rowsPerToken` rows of `rowDim` (`vocabShard = ceil(rows/tp)`, format
   `lookupTables ?? embeddings`), then all_reduce(tp) of `T·g·d` when tp > 1.
2. Each projection as a GEMM (m T, module `ngram`, replicated).
3. `ngram_gate` over `T·O`: read the projection outputs and the stream, write the stream.
4. With `conv`: dilated `causal_conv1d` over `T·O`, state `(K−1)·dilation·O·bpe(act)`.

Resident weights have their own formula (a step reads only the gathered rows): `ceil(rows/tp)·rowDim·bpe(lookup)` +
projections + vectors + conv taps.

### Short convs

`lower/shortConv.ts`, `lower/common.ts: causalConvOp`. `short_conv`: `causal_conv1d` over `T·dim` with the
residual add fused (`2·K + 1` FLOP per element), taps `dim·K·bpe(hp)`, state `(K−1)·dim·bpe(act)`. GQA
`kvConvKernel`: `attn.kv_conv` over `T·2·KVl·hd` after `qkv_proj`, with its residual add fused like Inkling's
`k_sconv` / `v_sconv` (`2·K + 1` FLOP per element), state `(K−1)·2·KVl·hd·bpe(act)`.

### Relative-position bias

`lower/gqa.ts` (Inkling). `attn.rel_proj (m T, n Hl·relDim, k h, module attention)` → `attn.rel_table (m T·Hl,
n extent, k relDim, weights in highPrecision, replicated)`. The bias gather and add inside the attention kernel
are vector work, not costed (§9; the softmax itself is, §7 "Attention softmax").

### Module formats

`lower/common.ts: moduleQuant, linearGemm({ module | format })`. Every linear GEMM names its `WeightModule`
(§2): its weight format, GEMM input dtype (math) and sparsity come from `QuantConfig.modules` with the fallback
rule. The router (`{ weights: highPrecision }`) and the LM head (`{ weights: lmHead, gemmInput: lmHeadGemmInput }`)
pass an explicit format and are always dense.

### Linear-attention sharding

`lower/linearAttention.ts: lowerLinearAttention, linearAttentionSequenceStateBytes,
stateFlopsPerElemPerToken`. Per device `qkL` by the replication rule and **`vL = vHeads/tp`, which must divide**
(`vHeads % tp == 0`, else a validation error): each value head's output goes through the row-parallel `out_proj`
and the TP all-reduce, so a value head replicated on r ranks would be summed r times (e.g. vHeads 2, dv 64 at
tp 4: four ranks projecting 64 features each, 256 summed features for 128 logical ones). A replicated q/k head
is only an input shared by the local value heads, like a GQA KV head. Conv channels
`C = 2·qkL·dk + vL·dv`. `in_proj n = 2·qkL·dk + vL·dv·(gate?2:1) + g·vL` (`g` gate outputs per head: 2,
Mamba2 1) → `causal_conv1d` over `T·C`
(`2·K` FLOP/elem; rolling state `seqs·C·(K−1)·bpe(act)` read if a prior state exists, and written) →
`recurrent` (prefill: chunked, `chunkLen = min(chunkTokens ?? 64, qLen)`, `hasPriorState = kvLen > qLen`; decode:
step) →
gated_rmsnorm (or rmsnorm) over `T·vL·dv` → `out_proj k = vL·dv` → all_reduce. Sequence state
`vL·dk·dv·bpe(state) + C·(K−1)·bpe(act)`, independent of context.

### MoE modes A/B

`lower/moe.ts: lowerMoe, moeLayout, moeRouting, moeParallelErrors, moeResidentWeightBytes, moeExpertWidth,
moeLatentWeightBytes, expertOutputShards`. Routing is balanced in load and uniform in expert choice ("Expected
distinct experts" and "EP load imbalance" below). Valid iff `E % ep == 0` and (`tp % ep == 0` → **mode A**, or
`ep % tp == 0` and `(tp·dp) % ep == 0` → **mode B**). `Eloc = E/ep`, `etp = max(1, tp/ep)`, `p = max(1, ep/tp)`.
`l = latentDim ?? h` is the routed experts' width ("Latent MoE" below).

- **Mode A** (includes ep = 1): every TP rank has all T tokens. `A = ceil(T·k/ep)` assignments per device,
  `A_max` on the busiest one ("EP load imbalance" below; = A at ep = 1), a compute grid of `a = min(Eloc, A_max)` experts with
  `mE = ceil(A_max/a)` tokens each, and the weights of `U ≤ a` expected distinct experts. Router (hp, replicated) →
  top-k → [`latent_in`] → grouped `gate_up (batch a, weights U, m mE, n (gated?2:1)·fe/etp, k l)` → act →
  `down (n l)` → combine (`a·mE·l`, writes `T·l·reduceBpe`) → [`latent_all_reduce` → `latent_norm`] →
  [`latent_out`] → shared experts as one dense MLP of `S·fs` sharded by tp → **one** all_reduce(tp) of `T·h`.
  No token moves: the expert outputs are partial sums of the layer output and ride that same all-reduce, so at
  dp = 1 (the UI) ep = tp has the same collectives as ep = 1 (except the expert-TP pair of a split expert-output
  norm, below, which etp = 1 removes). What EP changes is per device: `A_max` instead of an even fe/tp slice of
  every row (FLOP ×≈ρ in prefill), activations of its own `a·mE` rows only, and whole experts in decode
  (`src/sim/engine/ep.test.ts`).
- **Mode B**: the EP group spans p replicas; each device routes its slice `Ts = ceil(T/tp)`; `A = Ts·k`,
  `A_max` as in mode A, `etp = 1`. Router → top-k → [`latent_in`] → `all_to_all(ep)` dispatch `Ts·k·l·bpe(act)` →
  experts (grid and weights as in mode A) → `all_to_all(ep)` combine `Ts·k·l·reduceBpe` → combine (`Ts·k·l`) →
  [`latent_norm`] → [`latent_out`] → shared experts (unsharded) on Ts → `all_gather(tp)` of `T·h·bpe(act)` when
  tp > 1 (replaces the all-reduce).
- **Latent MoE** (`MoeBlock.latentDim` = l, Muse Spark `moe_hidden_size`, Nemotron `moe_latent_size`, Kimi K3
  `routed_expert_hidden_size`): `latent_in (m Ts, n l, k h)` after the top-k and `latent_out (m Ts, n h, k l)` after
  the combine (Ts = T in mode A), module `moe_latent`. Both are **replicated** (except a split expert-output norm,
  below, and `latentNorm`, "Latent MoE" below): every TP rank computes them whole and stores both matrices,
  `2·h·l·bpe(w_latent)` per device per layer at any tp (vLLM `ReplicatedLinear`; Muse Spark bf16: 8.3 GB over its
  62 layers). In mode A `latent_out` maps each rank's `T·l` partial sum to a `T·h` partial sum, valid by linearity
  (`Σ_r W·p_r = W·Σ_r p_r`), so the one all-reduce stays at h; TP-sharding them would need an extra collective
  at width l. The router reads h; shared experts stay at h. Without `latentDim`, `l = h` and both GEMMs are absent:
  every op is the same as before.
- **Expert-output norm** (`MoeBlock.expertOutputNorm`, Muse Spark `post_expert_norm`): an RMSNorm on each
  expert's l-wide down output before the weighted combine, over the `R = a·mE` token-expert rows of a device.
  With etp = 1 (mode B, mode A with ep = tp) it is one `moe.expert_norm` (rmsnorm over `R·l`) after `down` and no
  extra communication. With TP-sharded experts (mode A, etp > 1) each rank's down output is a partial sum over
  its `fe/etp` slice, and the RMS needs the reduced row, so with `s = etp` (`expertOutputShards`, else s = 1):

  ```
  reduce_scatter(etp, s ranks) S = R·l·reduceBpe          moe.expert_out_reduce_scatter  (each rank keeps l/s)
  all_reduce(etp, s ranks)     S = R·4  (fp32 Σx² per row) moe.expert_norm_sumsq_all_reduce
  rmsnorm                      elems = R·l/s
  combine                      elems = R·l/s, writes T·l/s·reduceBpe
  latent_out (k-split)         m T, n h, k l/s  → a T·h partial sum (W_out row-split by s)
  all_reduce(tp)               S = T·h·reduceBpe  (the existing one, unchanged)
  ```

  `Σ_r W_out[slice_r]·c_r` over all tp ranks is the full `W_out·c`, so the single closing all-reduce still
  finishes the block (without `latentDim`, each rank's `h/s` column slice of the zero-filled `T·h` buffer is
  summed the same way). Both collectives run on the expert-TP group (`group 'etp'`: the s ranks of a TP group
  holding the same experts, strided by ep; its links are in "Rank layout" below). This is the minimal communication: the reduce-scatter moves half the bytes of an all-reduce of `R·l` (ring:
  `(s−1)/s·S` vs `2·(s−1)/s·S`), and the sum-of-squares exchange is one scalar per row. Absent or false: no norm
  op and every op as before.

- Resident: `Eloc·(gated?3:2)·l·fe/etp·bpe(w)` + router `E·h·bpe(hp)` + [latent] `W_latent = (1 + 1/s)·h·l·bpe(w_latent)`
  [+ `l·bpe(hp)` with `latentNorm`] (`moeLatentWeightBytes`, summed per GEMM like the ops; `s = latentOutShards`:
  `expertOutputShards`, or with `latentNorm` tp in mode A and 1 in mode B) + [expertOutputNorm] `(l/s)·bpe(hp)`
  (the γ slice the norm reads) + shared (`/tp` in mode A).

### Expected distinct experts

`lower/moe.ts: moeRouting, neededExperts`; `engine/transfer.ts: uniqueExperts, uniqueLocalExperts` (the one
rule, shared with the offload transfers of §12 "Expert hit ratio"); `ir/metrics.ts: gemmBytes`
(`GemmOp.distinctWeightMatrices`). The routed-expert GEMMs read the weights of the experts a step is expected to
touch when every token picks its k experts independently and uniformly:

```
U     = E·(1 − (1 − k/E)^T)              ep = 1: expected distinct experts of T routed tokens
U_dev = E_loc·(1 − (1 − 1/E_loc)^A)       ep > 1: per device, over its balanced share A (with replacement)
weights read   = U·W_expert                gate_up and down: distinctWeightMatrices = U
compute (grid) = a experts × mE tokens     a = min(Eloc, A_max), mE = ceil(A_max/a)  (unchanged by U)
```

- `U(1) = k` exactly (batch-1 decode is unchanged), `U ≤ min(E, T·k)` always, and `U → E` in prefill (T in the
  thousands). `uniqueExperts` evaluates `E − (E − k)·(1 − k/E)^(T−1)`, the same value with no rounding at T = 1.
- **An expectation under iid-uniform routing, not a bound.** Real routing is skewed and temporally correlated
  (consecutive tokens, a speculative verify block) and can touch fewer experts; one batch can also touch more than
  the mean. A strict optimistic bound (k experts, every token picking the same ones) is far too loose to be useful.
  The earlier rule, `min(E, T·k)`, is the worst case for bytes: it put batched MoE decode below real runs (web-04)
  and made a kink where T·k reaches E.
- The compute keeps the balanced grid: FLOP, tiles and activations of `a·mE` rows. Only the weight bytes use U, so
  a U a hair below E in prefill (`E − 1e-12`) does not move a tile boundary.
- The speculative verify (§13) routes `T = b·(k+1)` tokens with the same rule.

Effect (sim aggregate decode, before → after, `docs/review/practicality-review.md` F5):

| batch | Qwen3-30B-A3B Q4_K_M, 1× RTX 4090 (E 128, top-8) | gpt-oss-120b MXFP4, 1× DGX Spark (E 128, top-4) |
| ----- | ------------------------------------------------ | ----------------------------------------------- |
| 4     | 660.2 → 705.8 tok/s @ 2048 (+6.9 %)              | 113.6 → 117.9 tok/s @ 2048 (+3.8 %)             |
| 8     | 711.7 → 837.3 tok/s @ 2048 (+17.6 %)             | 124.3 → 136.4 tok/s @ 2048 (+9.7 %)             |
| 16    | 737.9 → 1033.9 tok/s @ 2048 (+40.1 %)            | 130.4 → 160.1 tok/s @ 2048 (+22.8 %)            |
| 32    | 1273.2 → 1396.0 tok/s @ 2048 (+9.6 %)            | 133.7 → 201.9 tok/s @ 2048 (+51.0 %)            |
| 64    | 1995.1 → 2012.7 tok/s @ 2048 (+0.9 %)            | 256.5 → 290.6 tok/s @ 2048 (+13.3 %)            |

The Qwen3 b16 → b32 step was ×1.73 (a saturating at E); it is ×1.35 now. A Qwen3-30B-A3B verify of 8 tokens (k 7)
reads `U(8) = 51.6` experts per layer instead of 64 (−19.3 % expert bytes).

### EP load imbalance

`lower/moe.ts: epImbalance, DEFAULT_EP_IMBALANCE, moeRouting`; `ParallelConfig.epImbalance` (engine-level; the UI
sets ep, not ρ). The step waits for the busiest device of the EP group, which gets more than the mean share:

```
ρ     = max/mean (token, expert) assignments per device     1 at ep = 1; default 1.2 at ep > 1
T_g   = mode A ? T : ep·Ts                                  distinct tokens routed through the EP group
A_max = min(T_g·min(k, Eloc), max(A, ceil(ρ·T_g·k/ep)))  the busiest device's assignments (critical path)
```

- ρ multiplies the **unrounded** mean `T_g·k/ep`, not the already-ceiled share A, so a fractional mean is not
  rounded up twice. The result is never below A (ρ ≥ 1) and never above `T_g·min(k, Eloc)`, the most one device
  can receive: each token picks k distinct experts, at most `min(k, Eloc)` of them on one device (this also
  implies `≤ T_g·k`). A is always within the cap, since `T_g·k/ep ≤ T_g·min(k, Eloc)` for `k ≤ E`. Example:
  E 8, top-2, T 1, ep 8 (mode A): mean 0.25, `ceil(1.2·0.25) = 1`, cap `1·min(2, 1) = 1`, so A_max = 1 (the plain
  `ceil(ρ·A)` would give 2 assignments to a device that one token can reach only once). In prefill the cap never
  binds and `T_g·k/ep` is usually whole, so `A_max = ceil(ρ·A)` there.

- **Scaled by ρ** (through `A_max`): the expert grid rows `a·mE`, i.e. the grouped GEMMs' FLOP and activations,
  `moe.act`, the expert-output norm and its etp collectives, and the mode A combine.
- **Not scaled**: the expert weights read (`U` comes from the mean share A: the imbalance comes from hot experts
  drawing more tokens, so the busiest device runs more rows through the same experts; in prefill `U = Eloc`
  anyway), the router and top-k, the shared experts, the dispatch and combine all-to-alls (each device sends its
  own `Ts·k` rows; the busiest device's larger receive is not modelled) and the TP all-reduce / all-gather.
- **Default 1.2** (`DEFAULT_EP_IMBALANCE`, sources in its comment). Measured max/mean token load per EP rank,
  DeepSeek-V3/R1: 1.11 at EP32 and 1.12 at EP36 with an expert load balancer
  ([EPLB](https://github.com/deepseek-ai/EPLB)) and redundant expert slots, 2.56 at EP32 without one
  ([TensorRT-LLM wide-EP blog, Table 1](https://nvidia.github.io/TensorRT-LLM/1.0.0rc3/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html));
  ≈ 1.47 at 8 GPUs without one ([SGLang large-scale EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/),
  balancedness ≈ 0.68). 1.2 is the balanced case rounded up, because the 1.11 needs redundant expert copies the
  model does not store. An estimate of the routing, not a hardware constant; set `epImbalance: 1` for the
  perfectly balanced bound.
- Effect (`docs/review/practicality-review.md` F12): Qwen3-235B-A22B FP8 on 8× H100 (HGX), batch 64, prefill
  64×4096: mode B tp 1 dp 8 ep 8 282,617 → 256,117 tok/s @ 4096 (−9.4 %; ρ 1.5: 225,145, −20.3 %); tp 2 dp 4
  ep 8 274,549 → 255,701 (−6.9 %); mode A tp 8 ep 8 unchanged at 165,859 (that prefill is bound by the TP
  all-reduce, which hides the extra expert compute). ρ alone moves decode by under 0.1 %.

### Latent MoE

`lower/moe.ts` (`MoeBlock.latentNorm`, Kimi K3 `routed_expert_hidden_size` l = 3584, `latent_moe_use_norm`;
`modeling_kimi_linear.py:762-831`). A latent MoE (`latentDim`, above) with an RMSNorm over l between the combine
and `latent_out` (`latentOutShards`; not combinable with `expertOutputNorm`, `modelSpecErrors` rejects both).
`latent_in` is as above: mode A replicated on every TP rank (like MLA's `q_a`; every rank needs the full latent for
its experts), mode B on the device's own slice Ts. Replication costs each rank the full `h·l` weight read per step
(Kimi K3 at tp 8: 4.1 GB more per step than a column-parallel down-projection followed by an all-gather of `T·l`;
that alternative trades the bytes for 92 small collectives and is not modeled). After the combine:

- Mode A: `moe.latent_all_reduce` of `T·l·reduceBpe` when tp > 1 (each rank holds only its experts' and
  fe-shard's partial sums, and the norm is nonlinear, so the linearity rule above no longer applies); then
  `moe.latent_norm` (rmsnorm over `T·l`, γ `l·bpe(hp)`); then `moe.latent_out (m T, n h, k l/tp)`: row-parallel,
  so its partial sums ride the one `moe.tp_all_reduce` with the shared experts.
- Mode B: `moe.latent_norm` over `Ts·l` (the rows are complete after the combine a2a), then
  `moe.latent_out (m Ts, n h, k l)` unsharded.
- Resident: `h·l·bpe(w_latent)` (`latent_in`) + `h·l/(mode A ? tp : 1)·bpe(w_latent)` (`latent_out`) +
  `l·bpe(hp)` (norm).
- Module class `moe_latent` (falls back to `shared_experts`). Kimi K3 stores these BF16 next to MXFP4 experts; K3's
  compressed-tensors `ignore` list misses them, so their probes carry `nativeHighPrecision` (`quant/moduleClass.ts`).

### PP split

`lower/lower.ts: lowerModel, stageLayerRange, ppBoundaryResidualVectors`. Stage s holds layers
`[floor(s·L/pp), floor((s+1)·L/pp))`; stage 0 adds the embedding, the last stage the head. Every stage but the
last sends `S = T·N_live·h·bpe(act)` to the next (`send_recv`, group pp), where `N_live` is the residual
vectors per token that the layers after the boundary still read (`ppBoundaryResidualVectors(model, i)`, i the next
stage's first layer):

- plain residual: `N_live = 1`; hyper-connections: `N_live = residualStreams` N;
- Attention Residuals (§5 "Attention Residuals", block size B): the running sum plus the `ceil(i/B)` summaries
  stored before layer i, `N_live = 1 + ceil(i/B)`, read off layer i's pre-mixer `residual_mix` (whose `C` is
  exactly those vectors). Kimi K3 at pp 2: i = 46, `N_live = 1 + 4 = 5`, `S/T = 5·7168·2 B = 71,680 B` per token
  in BF16 (a plain residual would send 14,336 B).

A cross-layer source (shared KV cache, reused top-k,
shared indexer keys) on an earlier stage than its consumer produces a `simulate` warning
(`engine/simulate.ts: crossStageWarnings`); its transfer is not costed (§9).

Decode with pp > 1 additionally returns sampled uint32 IDs from the last stage to stage 0:
`S_return = b_mb·qLen·4 B = b_mb·(k+1)·4 B` (plain decode k = 0). One `send_recv`,
not one per intermediate stage; prefill has none (`lower/lower.ts: sendToFirst`, audit G2).
`engine/placement.ts: ppSendPath(..., destinationStage = 0)` resolves the actual return endpoints,
including node crossings and ring/chain hop counts.

### DP

DP replicas are independent and not lowered separately: `B_r = batch/dp` (must divide), aggregate tok/s = dp ×
the per-replica rate, per-sequence rates unchanged (§7 "Pipeline and DP").

### Rank layout

`src/sim/engine/placement.ts: groupSpan, groupSpanErrors, collectivePath, ppBoundaryIsInterNode, ppSendPath,
checkPlacement`. `rank = tpIdx + tp·(dpIdx + dp·ppIdx)`; ranks fill node 0 first (`node = floor(rank/dpn)`),
so TP and EP groups are contiguous. A group of n is intra-node if `n ≤ dpn` and `dpn % n == 0`, hierarchical
(n1 = dpn, n2 = n/dpn) if `n % dpn == 0`, otherwise an error. A PP boundary is inter-node if any rank r and its
partner `r + tp·dp` sit in different nodes. `tp·pp·dp ≤ dpn·nodes`; unused devices are a warning; a missing
required link is an error.

On an inter-node fabric (§6 "Fabric shapes"), node i of a ring or chain is cabled to node i ± 1 in this order.
`collectivePath` gives a flat inter-node group its view of the fabric (`groupFabric`: a ring the group does not
cover is a chain of the group's nodes), and a hierarchical group an oversubscribed switch (the only shape with
dpn > 1). `ppSendHops`: an inter-node PP send crosses `h = max_r nodeHops(node(r), node(r + tp·dp))` links, 1 on a
switch or mesh; with dpn = 1 on a ring or chain, h = 1 for pure PP (stages on neighbours) and h = tp·dp otherwise
(tp 2 × pp 2 on a ring of 4: h = 2). A path carries `fabric` / `sendHops` only when they differ from the
non-blocking switch / 1, so pre-fabric topologies get exactly their old paths.

**Expert-TP group** (`placement.ts: expertTpGroupSpan, expertTpCollectivePath`; group `'etp'`, used only by the MoE
expert-output norm with etp > 1). EP groups are `ep` consecutive ranks, so in mode A the rank with EP index j holds
the same experts as `j + i·ep`, i < etp = tp/ep: a group strided by ep inside the contiguous TP group. Its span:

```
tp ≤ dpn                 n1 = etp,     n2 = 1                    intra-node
tp > dpn, ep ≤ dpn       n1 = dpn/ep,  n2 = tp/dpn (adjacent)    hierarchical (flat inter-node when ep = dpn)
tp > dpn, ep > dpn       n1 = 1,       n2 = etp, ep/dpn apart    flat inter-node
```

2 × 8 H100 at tp 16: ep 2 → 4 × 2 hierarchical, ep 4 → 2 × 2, ep 8 → one per node. The path is built by
`collectivePath` from this span (inter-node staging included). Members on non-adjacent nodes are modeled on a
switched or full-mesh fabric (one hop between any two nodes); on a ring or chain they are a validation error
(`etp: …`), since the hop counts there assume adjacent nodes. A model without expert-TP collectives builds no `etp`
path, so its results are unchanged.

---

## 6. Collectives

`src/sim/engine/collectives.ts`. S = messageBytes per device (§1 conventions), n group size, B per-direction
bandwidth, α latency. Each result splits α terms (`latencySec`) from β terms (`bandwidthSec`).
`collectiveTime` dispatches on kind and path; a flat path with a `fabric` goes to `fabricCollectiveTime`
("Fabric shapes" below).

### Flat collectives

`ringAllReduceSec, treeAllReduceSec, allReduceSec, ringAllGatherSec, ringReduceScatterSec, allToAllSec,
sendRecvSec`.

| Collective                                                | Time                                                            |
| --------------------------------------------------------- | --------------------------------------------------------------- |
| ring all-reduce                                           | `2(n−1)·α + 2(n−1)/n·S/B`                                       |
| tree all-reduce                                           | `2⌈log2 n⌉·α + 2·S/B`                                           |
| all-reduce `'best'` (per-op candidate)                    | min(ring, tree); stage-aware selection under ideal overlap (§7) |
| ring all-gather (S = output) / reduce-scatter (S = input) | `(n−1)·α + (n−1)/n·S/B`                                         |
| all-to-all (direct, concurrent)                           | `α + (n−1)/n·S/B`                                               |
| send/recv                                                 | `α + S/B`                                                       |

### Hierarchical collectives

`hierarchicalAllReduceSec, hierarchicalAllGatherSec, hierarchicalReduceScatterSec, hierarchicalAllToAllSec`.
A group of n1 devices per node × n2 nodes:

- all-reduce = `RS_intra(S, n1) + AR_inter(S/n1, n2, algorithm) + AG_intra(S, n1)`
- all-gather = `AG_inter(S/n1, n2) + AG_intra(S, n1)`; reduce-scatter = `RS_intra(S, n1) + RS_inter(S/n1, n2)`
- all-to-all = `max(α_intra,eff, α_inter,eff) + max((n1−1)/n·S/B_intra, (n−n1)/n·S/B_inter)`
  (audit G3: concurrent messages retain the larger effective latency floor, including host bounce).

### Host bounce

`effectiveLink`. A `'host_bounce'` link (e.g. GeForce without P2P) stages every byte through host memory once
written, once read: `B_eff = min(B, hostMemBW / (2·nActive))`, `α_eff = α + extraLatencySec`, with
`nActive = n` for flat collectives; PP send/recv uses `min(devicesPerNode, tp·dp)` for both forward
activations and sampled-token returns (`placement.ts: ppSendPath`, `CollectivePath.sendActiveDevices`, audit G8).
Standalone send/recv paths default to one active device. Two-level collectives use the group's devices per node
at both levels.

Flat host-bounce all-to-all without an explicit fabric uses `n−1` serialized peer rounds:
`latency = (n−1)·α_eff`, `bandwidth = (n−1)/n·S/B_eff` (`collectives.ts: collectiveTime`, audit G4).
Ring/chain paths already use `lat(maxHops)` below; no second hop multiplier is added.
PCIe switch / shared-uplink topologies are not modeled.

Also not modeled: NVLS / SHARP in-switch reduction, the memory traffic of collectives, and contention between
communication and compute for memory.

### Inter-node host staging

`src/sim/engine/staging.ts: needsHostStaging, resolveHostStaging`; `collectives.ts: effectiveLink, stagedStageRates,
stagedLink, stagingBandwidthCapBytesPerSec, collectiveTime`; `fabricCollectives.ts: fabricLinkParams,
fabricCollectiveTime, fabricAllReduceSec`; `placement.ts: collectivePath, ppSendPath, hasInterNodeTraffic`. A device
with dedicated memory cannot hand device memory to a NIC without GPUDirect RDMA, so each inter-node message is staged
through host memory over PCIe. This section is that PCIe part; the TCP socket copy (next section) rides the same
pipeline (`HostStaging.socketCopy`) and applies to every device over TCP, unified memory included.

**When.** `staged ⇔ device.memory.kind = 'dedicated' ∧ nodes > 1 ∧ (interNode.gpuDirectRdma = false ∨
device.gpuDirectRdma = false)` (§4 "Links and topologies" for the fields and the capability table). So every
dedicated GPU over TCP stages; on InfiniBand only a GPU without GPUDirect (a GeForce card) does. It does not apply
to unified memory (Mac, DGX Spark, Strix Halo: the NIC reads the shared pool), to GPUDirect paths, or to intra-node
links (those are "Host bounce" above). Only inter-node paths carry it (`CollectivePath.interStaging`: flat
`level: 'inter'`, the inter phase of hierarchical collectives, PP sends whose boundary crosses nodes). One info
warning per run, when some message crosses nodes.

**Mechanism.** NCCL's network transport without GDR keeps its send and receive buffers in pinned host memory mapped
into the GPU (`src/transport/net.cc:936`, `:1121`, `ncclCudaHostCalloc`; commit 3e46c089f4ae): the GPU kernel writes
the message and a flag into host memory, a CPU proxy thread makes the socket call, and the receiving GPU polls and
reads host memory over PCIe. No `cudaMemcpy`, no stream sync. A message larger than one buffer step pipelines GPU
copy, wire and GPU copy; one within a step pays them one after another.

**Parameters** (`HostStaging`, no new free number; `B_pcie`, `α_pcie` in `HostStaging.pcie`, absent when nothing
PCIe-stages):

| symbol   | meaning                                    | value                                                                                                   |
| -------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `B_pcie` | the device's PCIe host link, per direction | `hardware.host.link ?? hostPcieLink(device.hostInterface)`, else PCIe 5.0 x16 with a warning            |
| `α_pcie` | one PCIe crossing                          | that link's `latencySec`, 1 µs (estimated)                                                              |
| `B_host` | node DRAM, read + write combined           | `hardware.host.ram ?? DEFAULT_HOST_RAM` (DDR5-6000 2ch, 96 GB/s); the pool peak on unified memory       |
| `c`      | one pipeline step of the staging buffer    | `HOST_STAGING_STEP_BYTES` = 4 MiB / 8 = 512 KiB (NCCL `NCCL_BUFFSIZE` default 4 MiB / `NCCL_STEPS` = 8) |
| `n_stg`  | devices of one node staging at once        | `min(devicesPerNode, tp·dp)`                                                                            |

**Formula** (`stagedStageRates`, `stagedLink`, applied inside `effectiveLink` after the host-bounce lines), in α-β
terms, here with PCIe staging only (`socketCopy` false; the general form with the copy is in "TCP socket copy", and
with no copy it computes exactly these operations). A staged message crosses three stages: GPU → pinned host buffer
(PCIe), the wire, host buffer → GPU (PCIe). In a collective every node sends and receives at once, so each stage's
rate, per device and per direction, is:

```
r_pcie = min(B_pcie, B_host / n_stg)            GPU ↔ host buffer: 1 DRAM access per byte per device
r_net  = min(B, B_host / (2·n_stg))             wire: NIC read at the sender + NIC write at the receiver
B_hj   = B_host / (4·n_stg)                     all stages at once (the pipelined steady state): 4 accesses
B_eff  = min(r_pcie, r_net, B_hj) = min(B, B_pcie, B_host / (4·n_stg))          β rate

s = min(m, c),  k = m / s steps                  m = the message of one step
t_pipe = Σ_i s/r_i + (k − 1)·s/B_eff = m/B_eff + (2·s/r_pcie + s/r_net − s/B_eff)
α_eff = α + 2·α_pcie + (2·s/r_pcie + s/r_net − s/B_eff)
```

- `2·α_pcie` = α_stage = 2 µs (estimated): one PCIe write plus flag at the sender, one PCIe poll / read at the
  receiver. Sources: GDRCopy README (https://github.com/NVIDIA/gdrcopy), `gdrcopy_pplat` CPU↔GPU mapped-memory round
  trip 1.09 µs; Neugebauer et al., SIGCOMM'18 pcie-bench, 64 B DMA read median 547 ns. For contrast, the Gloo path
  (`cudaMemcpy` + sync, `cuda_allreduce_ring.cc`) costs "6-7us" per copy (GDRCopy README): engine overhead here.
- Host memory: the PCIe copies take one DRAM access per byte per device, the wire two (NIC read at the sender, NIC
  write at the receiver), and the steady state of all three stages at once four. Each cap enters its own stage rate,
  and the four-access share bounds the pipelined β.
- The fill / drain term is one step through every stage except the one that sets B_eff (it is ≥ 0, since
  Σ 1/r_i ≥ 1/B_eff), so nothing is counted twice when PCIe or host memory binds. A small message (`m ≤ c`, one
  step) pays the serial sum `m/r_pcie + m/r_net + m/r_pcie`. NIC-bound with host memory not binding, the fill is
  `2·s/B_pcie`. The fill is serial with the step chain, so it goes into α (latency), never β.
- Worked checks (`staging.test.ts`): a 50 GB/s NIC, α 1 µs, PCIe 3.0 x16, host memory not binding: a 4 KiB send
  costs 3.65067 µs (serial); an 8 MiB send costs `3 µs + m/B_pcie + c/B + c/B_pcie` = 632.28576 µs (B_pcie =
  14.404 GB/s, Gen3 x16 payload). Host-bound (50 GB/s NIC, PCIe 5.0 x16, 96 GB/s, n_stg 1): β = m / 24 GB/s, fill =
  `2c/57.614 GB/s + c/48 GB/s − c/24 GB/s`.
- `m` per step, chosen per algorithm before the ring / tree / best comparison, so each candidate gets its own staged
  α (`collectiveTime`, `fabricCollectiveTime`):
  - flat: ring all-reduce, all-gather, reduce-scatter and all-to-all `S/n`; tree all-reduce `S`; send/recv `S`;
  - hierarchical inter phase: ring `(S/n1)/n2 = S/n`; tree `S/n1`;
  - fabric shapes: ring, mesh-direct, all-gather, reduce-scatter and all-to-all `S/N`; tree `S`; send `S` (the
    2-node shortcut likewise).

  Example: S = 64 KiB, n = 4, PCIe 5.0 x16, 10GbE: an explicit tree costs `α = 4·(10 + 2 + 2·S/B_pcie) µs` =
  57.10 µs. At n = 4, `best` switches from tree to ring at `S* = 24 µs / (5/B_pcie + 0.5/B)` ≈ 49 KB.

- Staging is not host bounce: the flat all-to-all keeps one α (no `(n−1)·α` rule). On an oversubscribed
  hierarchical switch, staging applies after the `B/o` reduction (its caps are the device's own).
- Fabric shapes (`fabricLinkParams`): every hop gets the staging α, and `B_L = min(B_L, B_stg)`, `C = min(C,
B_stg)`, with `B_stg = stagingBandwidthCapBytesPerSec = min(B_pcie, B_host/((2·a + 2)·n_stg))` (a = 1 here, so
  `B_host/(4·n_stg)`; the staging bandwidth is per device, whatever the port count). A relay already holds the bytes
  in host memory, so charging the staging α per hop over-counts relays: a documented approximation; no
  dedicated-memory preset uses a ring or chain.
- The α terms land in `CollectiveTime.latencySec` (→ `PhaseResult.comm.latencySec`), the β terms in `bandwidthSec`,
  so the UI's comm latency / bandwidth split shows staging with no UI change.
- On TCP the bandwidth terms never bind (PCIe 5.0 x16 57.6 GB/s, a 96 GB/s host's share 12 GB/s for one device once
  the socket copy joins, vs 1.25 GB/s for 10GbE); the 2 µs per message is what counts.
- **Known inconsistency**: intra-node host bounce keeps 12 µs (the `cudaMemcpyPeer` driver path, community P2P
  matrices), while inter-node staging uses the 2 µs mapped-memory floor that NCCL's SHM and NET transports reach.
  Revisiting the intra-node value is a follow-up.

Example (`src/ui/state/urlScenario.test.ts`): 2 machines × 1 RTX 5090 over 10GbE TCP, TP 2, Qwen3.8-Flash-Next
nvfp4, experts offloaded with f 0.7. Decode @ 8k: comm α 1.970 → 2.380 ms per token (≈ 205 inter-node message steps ×
(2 µs + a fill of ≈ 0.04–0.08 µs)), 193.0 → 178.8 tok/s @ 8k; the TCP latency recheck leaves 10GbE at 10 µs (no
change); the default prefetch accuracy (§12 "Expert prefetch") then takes it to 162.0 tok/s @ 8k (exposed transfer
0.31 → 0.89 ms); the TCP socket copy (next section) adds 2.380 → 2.402 ms of α (≈ 0.11 µs per step): 161.4 tok/s
@ 8k. Prefill @ 8k: 2489.7 → 2487.0 tok/s, unchanged by prefetch, 2483.5 tok/s with the copy.

### TCP socket copy

`src/sim/engine/staging.ts: needsSocketCopy, resolveHostStaging`; `collectives.ts: stagingCopyAccessesPerByte,
stagedStageRates, stagedLink, stagingBandwidthCapBytesPerSec`; `fabricCollectives.ts: fabricLinkParams`; constant
`src/sim/hw/presets/links.ts: SOCKET_COPY_STEP_BYTES`; tests `src/sim/engine/socketCopy.test.ts`.

**When.** `socketCopy ⇔ nodes > 1 ∧ interNode.transport = 'tcp'` (§4 "Links and topologies", `LinkSpec.transport`),
for every device: unified memory (Mac, Strix Halo, DGX Spark over TCP) and dedicated GPUs alike. RDMA links
(`transport: 'rdma'`), single-node topologies and intra-node paths never pay it; with neither the copy nor PCIe staging
the path carries no `interStaging`, exactly as before. No warning is added (the result barely moves).

**Mechanism.** A socket transport hands the kernel a user buffer: `send()` copies it into kernel socket buffers
(skbs), `recv()` copies it out into the receiver's user buffer, each a CPU memcpy in host DRAM (a read and a write per
byte). RDMA is zero-copy: the NIC reads and writes the registered buffer itself. NCCL's socket transport calls plain
`send()` / `recv()` with `MSG_DONTWAIT | MSG_NOSIGNAL` (`src/os/linux.cc:351-355`, commit 3e46c089f4ae), so it does not
use `MSG_ZEROCOPY`; Gloo and MLX's ring (macOS has no `MSG_ZEROCOPY`) are plain sockets too. The kernel copies and
transmits a large send in TCP segments of up to 64 KiB (Linux `GSO_LEGACY_MAX_SIZE`), so the copies pipeline with
the wire in steps.

**Parameters** (`HostStaging`):

| symbol   | meaning                                 | value                                                                                                                                               |
| -------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `B_host` | the DRAM the copy runs in, read + write | unified memory: the pool peak `device.memory.bandwidthBytesPerSec`; dedicated: `hardware.host.ram ?? DEFAULT_HOST_RAM`                              |
| `c`      | one pipeline step                       | `SOCKET_COPY_STEP_BYTES` = 64 KiB (Linux `include/linux/netdevice.h:2333`, v6.12; ESTIMATED for macOS) without PCIe staging; with it NCCL's 512 KiB |
| `n_stg`  | devices of one node copying at once     | `min(devicesPerNode, tp·dp)`                                                                                                                        |

**Formula** (the general `stagedLink`; with `socketCopy` false it is the PCIe-only formula above, the same IEEE
operations, pinned with `Object.is` in `socketCopy.test.ts`). A message crosses up to five stages: PCIe copy-out,
socket copy-in (user → kernel), the wire, socket copy-out (kernel → user), PCIe copy-in. An absent stage has rate ∞,
so its term is 0:

```
a      = (pcie ? 1 : 0) + (socketCopy ? 2 : 0)   DRAM accesses per byte of ONE end's copies (DMA 1; memcpy = read + write 2)
r_pcie = min(B_pcie, B_host / n_stg)            (∞ without PCIe staging)
r_sock = B_host / (2·n_stg)                     (∞ without the socket copy)
r_net  = min(B, B_host / (2·n_stg))             NIC read at the sender + NIC write at the receiver
B_hj   = B_host / ((2·a + 2)·n_stg)             all stages at once: 2 ends × a + 2 NIC accesses
B_eff  = min(r_pcie, r_sock, r_net, B_hj) = min(B, B_pcie, B_hj)                     β rate

s      = min(m, c)
α_eff  = α + 2·α_pcie + (2·s/r_pcie + 2·s/r_sock + s/r_net − s/B_eff)            (α_pcie = 0 without PCIe)
B_stg  = min(B_pcie, B_hj)                       stagingBandwidthCapBytesPerSec (fabric C cap; B_pcie = ∞ without PCIe)
```

| case                             | a   | B_hj               | what the copy adds                                                                                                                  |
| -------------------------------- | --- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| PCIe staging only (RDMA, no GDR) | 1   | `B_host/(4·n_stg)` | nothing: bit-identical to before                                                                                                    |
| unified memory over TCP          | 2   | `B_mem/(6·n_stg)`  | small m, wire-bound: `Δα = 2·m/r_sock = 4·n_stg·m/B_mem`; large: β = `m/min(B, B_mem/(6·n_stg))` plus a fill ≤ `2·c/r_sock`         |
| dedicated GPU over TCP           | 3   | `B_host/(8·n_stg)` | the PCIe staging plus `2·s/r_sock` of fill per step (step c stays NCCL's 512 KiB: conservative, the kernel pipelines within a step) |

- **Copies per message step**: two (the sender's copy-in, the receiver's copy-out), on every algorithm step, with the
  per-algorithm step message `m` of "Inter-node host staging" (ring `S/n`, tree `S`, send `S`, hierarchical inter
  phase `S/n` or `S/n1`, fabric shapes as there).
- **No fixed latency.** The measured TCP α (half a tuned 1-byte ping-pong, §4) already holds the syscalls and a
  1-byte copy, so the copy adds only its size-dependent part `2·s/r_sock` (the latency share) plus the DRAM cap on β.
  For scale, the whole kernel copy is ≈ 20 % of a 3.4 µs Linux UDP echo on 10GbE (Arrakis, OSDI'14, Table 1: copy in
  0.24 + out 0.44 µs; https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-peter_simon.pdf), and it
  matters mostly for long flows (Cai et al., SIGCOMM'21, https://www.cs.cornell.edu/~ragarwal/pubs/network-stack.pdf).
- **Dedicated GPUs over TCP are not double-counted.** PCIe staging models the GPU ↔ pinned host buffer copy and the
  NIC's DMA of that buffer; NCCL's socket proxy then calls `send()` / `recv()` on the pinned buffer and the kernel
  copies it into and out of skbs. That is a third, physically separate copy, so dedicated + TCP = PCIe copy + socket
  copy at each end. A dedicated GPU on RDMA without GPUDirect (a GeForce on InfiniBand) keeps the PCIe copy only.
- **Magnitude: small on the presets.** B_mem is 0.4–1.2 TB/s on Macs and 256 GB/s on Strix Halo, so `r_sock ≥ 128
GB/s` and `B_mem/6 ≥ 42 GB/s`, far above the 2.5–10 GB/s links: a 4 KiB decode step gains 0.02–0.06 µs and β never
  binds. The RDMA vs TCP gap on one cable comes from α (§4 "Link latencies", TB5 7 vs 45 µs), not from the copy.
- Worked checks (`socketCopy.test.ts`): unified TCP, n 1, B 10 GB/s, B_mem 800 GB/s: a 4 KiB step gains 8192 B / 400
  GB/s = 0.02048 µs with β unchanged; an 8 MiB step has β = 838.8608 µs and a fill of 2·64 KiB / 400 GB/s = 0.32768 µs.
  Host-bound (B_mem 30 GB/s): r_sock 15, r_net 10, B_hj 5 GB/s, β(8 MiB) = 1677.7216 µs, fill = 64 KiB · (2/15 + 1/10 −
  1/5) / (1 GB/s) = 2.18453 µs. Dedicated 10GbE on PCIe 5.0 x16, 96 GB/s: a 4 KiB step gains 8192 / 48 GB/s =
  0.170667 µs, β unchanged (B_hj = 12 GB/s).

Example (`src/ui/state/rdmaTcpScenario.test.ts`): DeepSeek-V4.1-Flash on 2 × M5 Ultra (64-core GPU), TP 2, lookup
tables on SSD, over Thunderbolt 5. Decode @ 8k with DSpark (code): TB5 TCP 150.12 → 150.02 tok/s with the copy (comm α
8.190 → 8.208 ms per token); without spec-dec 73.68 → 73.66 tok/s (7.515 → 7.518 ms). TB5 RDMA (no copy) after the α
fix: 201.8 tok/s @ 8k with DSpark (comm α 9.10 → 1.27 ms) and 138.4 tok/s @ 8k without (8.35 → 1.17 ms), so RDMA now
decodes faster than TCP, as it must. (The rows without spec-dec include the Apple GEMV path of §4; before it they
were 66.80 and 116.0 tok/s.)

### Fabric shapes

`src/sim/engine/fabricCollectives.ts: fabricCollectiveTime, fabricLinkParams, pathLatencySec,
fabricRingAllReduceSec, fabricTreeAllReduceSec, meshDirectAllReduceSec, fabricAllReduceSec,
fabricAllGatherOrReduceScatterSec, chainReduceScatterSec, chainAllGatherSec, chainAllReduceSec, fabricAllToAllSec,
fabricSendRecvSec`; geometry in `src/sim/hw/fabric.ts: nodeHops, maxHops, portsNeeded, treeStepHops, groupFabric,
fabricShapeErrors`. Inter-node only (intra-node links stay switched).

Shapes (`FabricShape`): **switched** (one port per node into a switch, oversubscription o ≥ 1; absent shape = o 1),
**full_mesh** (every pair cabled with k = `cablesPerPair` cables, striped; N = 2 is "direct cable(s)"), **ring**
(N ≥ 3, node i to i ± 1; at N = 3 the same cabling as the mesh), **chain** (0 – 1 – … – N−1). Every shape but switched
needs one device per node, so a node is a device and N = the group's size.

Notation: B_L = per-port per-direction bandwidth, C = `allPortsCapPerDirectionBytesPerSec` (∞ when absent), α =
per-hop latency, α_f = `forwardingLatencySec` (0 when absent), m = S/N. Named bandwidths: B_P = k·B_L (one mesh
pair), B_one = min(B_P, C) (one flow), B_two = min(2·B_P, C) (a node streaming on two ports), B_mesh = min((N−1)·B_P,
C) (a mesh node sending to all peers); k = 1 except on the mesh.

Two rules give every formula:

- **Path latency** of an h-hop message: `lat(h) = h·α + (h−1)·α_f` (exactly α for h = 1). Algorithm steps between
  neighbours (ring and chain RS / AG) are one hop each; tree steps and all-to-all flows to farther nodes are relayed.
- **Bandwidth term**: `β = max(L_link / B_L, E_node / C)`, L_link = bytes on the busiest link direction, E_node = the
  busiest node's egress, its own bytes plus those it relays (ingress is symmetric in every pattern here). Relays go
  through the host (no switching inside these NICs), so relayed bytes count against C.

| collective         | switched, o > 1                | full_mesh (k), N ≥ 3           | ring                                   | chain                                                                                |
| ------------------ | ------------------------------ | ------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------ |
| ring-class AR      | `2(N−1)α + 2(N−1)/N·S/(B_L/o)` | `2(N−1)α + 2(N−1)/N·S/B_two`   | `2(N−1)α + 2(N−1)/N·S/min(2B_L, C)`    | `RS_chain + AG_chain`                                                                |
| tree AR            | `2⌈log2 N⌉α + 2S/(B_L/o)`      | `2⌈log2 N⌉α + 2S/B_one`        | `2·Σ_k lat(h_k) + 2S/B_one`            | same, chain hops                                                                     |
| direct AR          | —                              | `2α + 2(N−1)/N·S/B_mesh`       | —                                      | —                                                                                    |
| AR `'best'`        | min(ring, tree)                | min(ring, direct, tree)        | min(ring, tree)                        | min(ring, tree)                                                                      |
| AG / RS            | `(N−1)α + (N−1)/N·S/(B_L/o)`   | direct: `α + (N−1)/N·S/B_mesh` | `(N−1)α + (N−1)/N·S/min(2B_L, C)`      | RS: `(N−1)α + (N−1)/N·S/min(B_L, C)`; AG: `(N−1)α + max((N−1)/N·S/B_L, (N+1)/N·S/C)` |
| all-to-all         | `α + (N−1)/N·S/(B_L/o)`        | `α + (N−1)/N·S/B_mesh`         | `lat(⌊N/2⌋) + ⌊N²/4⌋/N·S/min(2B_L, C)` | `lat(N−1) + max(⌊N²/4⌋/N·S/B_L, E_N/N·S/C)`                                          |
| send/recv (h hops) | `α + S/(B_L/o)`                | `α + S/B_one`                  | `lat(h) + h·S/B_one`                   | `lat(h) + h·S/B_one`                                                                 |

h_k = `treeStepHops` (binomial tree, step-k partners 2^k apart): ring `min(2^k, N−2^k)`, chain `2^k`.
E_N = `(N−1) + 2⌊(N−1)²/4⌋`. **Two nodes** on a mesh or chain are one (striped) link: the switched formulas at
B_one, so one cable with C ≥ B_L is bit-identical to the switch. A ring needs N ≥ 3. Oversubscription scales only B,
never α; on a hierarchical path it divides the inter-node bandwidth.

Derivations (backed by a brute-force routing test, `fabricCollectives.test.ts`):

- **Bidirectional ring** (ring-class AR, AG, RS on a ring or a mesh of N ≥ 3, which contains a Hamiltonian cycle): the
  buffer is split in halves that run the ring algorithm in opposite directions. Per step a node sends S/(2N) on each of
  two ports, so `β_step = max(S/(2N)/B_P, (S/N)/C) = (S/N)/min(2B_P, C)`. With C = B_L (DGX Spark: both ports share
  ~25 GB/s) this is exactly the switched ring at B_L: a ring of Sparks runs bandwidth-bound all-reduce at the switch's
  speed.
- **Mesh direct**: RS, AG and all-to-all as one-shot all-pairs exchanges, m per pair and (N−1)·m per node, so
  `β = max(m/B_P, (N−1)·m/C) = (N−1)/N·S/B_mesh` with one α. For N ≥ 3 it beats the ring on both terms.
- **Chain reduce-scatter** (linear): chunk c is reduced at node c, partials flowing toward c from both sides. Link
  j→j+1 carries the partials of chunks j+1 … N−1, at most (N−1)·m; node j sends (N−1−j)·m right and j·m left,
  (N−1)·m in all. N−1 neighbour steps.
- **Chain all-gather** (linear): each chunk travels from its node to both ends. The end links carry (N−1)·m, but an
  inner node sends its own chunk both ways and forwards every other chunk once: (N+1)·m. That holds for ANY all-gather
  on a chain: there are N(N−1) chunk receptions, all neighbour transfers; an end node can only ever send its own chunk
  to its one neighbour (m), so the N−2 inner nodes send the rest, (N+1)·m each on average. So AG ≠ RS on a chain when
  C < (N+1)/(N−1)·B_L. Chain AR = linear RS + linear AG; the folded ring is worse on both terms.
- **Ring all-to-all**: every node sends m to every other node on the shorter way round, a distance-N/2 flow (even N)
  half each way. The clockwise link crossings sum to N·Σ d over the distances, so every link direction carries
  `L = ⌊N²/4⌋/2·m` (N²/8·m for even N, (N²−1)/8·m for odd), and each node's egress is 2L over its two ports:
  `β = max(L/B_L, 2L/C) = ⌊N²/4⌋·m/min(2B_L, C)`. At N = 3 this equals the k = 1 mesh (same cabling).
- **Chain all-to-all**: link j→j+1 carries the flows from {0 … j} to {j+1 … N−1}, (j+1)(N−1−j)·m ≤ ⌊N²/4⌋·m;
  inner node j sends its own (N−1)·m plus relays 2j(N−1−j)·m, at most E_N·m.
- **All-to-all latency**: all flows run at once, so α is the farthest destination's `lat(maxHops)`: ⌊N/2⌋ hops on
  a ring, N−1 on a chain.
- **Tree**: step k's message crosses h_k links (relayed, pipelined), one flow per step at B_one; the β term is the
  switched tree's `2S/B` at B_one.
- **PP send**: the boundary's flows (rank r → r + tp·dp) all run the same way, so on a ring or chain the busiest
  link and the busiest relay each carry h of them: `β = h·S/B_one`.
- **Subgroups**: a TP / EP group covering part of a ring is a chain of its nodes; concurrent groups use disjoint
  segments.

Worked examples (`fabricCollectives.test.ts`): B_L = 1e10 B/s, α = 2 µs, α_f = 1 µs, S = 8e6 B (S/B_L = 800 µs),
N = 4; each cell α + β in µs.

| collective | switched  | switched o=2 | ring C=∞  | ring C=B_L             | chain C=∞ | chain C=B_L | mesh k=1 C=∞ | mesh k=1 C=B_L | mesh k=2 C=∞ |
| ---------- | --------- | ------------ | --------- | ---------------------- | --------- | ----------- | ------------ | -------------- | ------------ |
| ring AR    | 12 + 1200 | 12 + 2400    | 12 + 600  | 12 + 1200 (= switched) | 12 + 1200 | 12 + 1600   | 12 + 600     | 12 + 1200      | 12 + 300     |
| tree AR    | 8 + 1600  | 8 + 3200     | 14 + 1600 | 14 + 1600              | 14 + 1600 | 14 + 1600   | 8 + 1600     | 8 + 1600       | 8 + 800      |
| direct AR  | —         | —            | —         | —                      | —         | —           | 4 + 400      | 4 + 1200       | 4 + 200      |
| AG         | 6 + 600   | 6 + 1200     | 6 + 300   | 6 + 600                | 6 + 600   | 6 + 1000    | 2 + 200      | 2 + 600        | 2 + 100      |
| RS         | 6 + 600   | 6 + 1200     | 6 + 300   | 6 + 600                | 6 + 600   | 6 + 600     | 2 + 200      | 2 + 600        | 2 + 100      |
| all-to-all | 2 + 600   | 2 + 1200     | 5 + 400   | 5 + 800                | 8 + 800   | 8 + 1400    | 2 + 200      | 2 + 600        | 2 + 100      |
| send h=1   | 2 + 800   | 2 + 1600     | 2 + 800   | 2 + 800                | 2 + 800   | 2 + 800     | 2 + 800      | 2 + 800        | 2 + 400      |
| send h=2   | —         | —            | 5 + 1600  | 5 + 1600               | 5 + 1600  | 5 + 1600    | —            | —              | —            |

At a decode-sized S = 1e4 B, `'best'` all-reduce picks the tree on the switch (8 + 2 = 10 µs), the ring algorithm on
the ring (12 + 1.5 = 13.5 µs with C = B_L; the tree's 2-hop step makes it 14 + 2) and direct on the k = 1 mesh (4 +
0.5 = 4.5 µs). 4 × DGX Spark (B_L = C = 25 GB/s, α 1.5 µs, α_f 5 µs in `simulate.test.ts`; the preset's estimate is 4.5 µs), S = 8e6 B: ring AR 9 + 480 on both
fabrics (bit-identical); tree 6 + 640 switched vs 19 + 640 on the ring; all-to-all 1.5 + 240 vs 8 + 320; at decode
size switched takes the tree (6.8 µs), the ring the ring algorithm (9.6 µs): decode is slower on the ring, prefill
equal.

---

## 7. Per-op time, stages, pipeline

### Per-op time

`src/sim/engine/opTime.ts: opTime, roofline, engineMemBandwidthBytesPerSec, elementwiseBytesUnderFusion,
attentionSoftmaxSec`.
One exhaustive switch on `op.kind`. `memBW = min(memory.bandwidthBytesPerSec, engine.memBandwidthCapBytesPerSec ?? ∞)`.

| Op                                               | Compute term                                                                                                                                                                                          | Memory term                                                                                                                                       |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| indexer, recurrent (chunked)                     | `t_mma = F / (peak·efficiency)`; peak from §4 "Peak lookup and fallback" for the op's (math, accumulate, sparsity); efficiency from §4 "Shape efficiency" with the tiling of the format actually used | `bytes/memBW`                                                                                                                                     |
| gemm                                             | `t_mma`; with `m ≤ gemvMaxRows` (dense, fp16 / bf16 / fp32 math) `min(t_mma, F / vectorPeak(math))` (§4 "GEMV path")                                                                                  | `bytes/memBW`                                                                                                                                     |
| attention                                        | `max(t_mma, t_exp + t_vec)` with `t_exp = S / sfuOpsPerSec`, `t_vec = 5·S / vectorPeak(fp32)` ("Attention softmax" below)                                                                             | `bytes/memBW`                                                                                                                                     |
| recurrent (step), elementwise                    | `F / vectorPeak(dtype)` (state dtype for the step)                                                                                                                                                    | `bytes/memBW`                                                                                                                                     |
| embedding                                        | 0                                                                                                                                                                                                     | `bytes/memBW`                                                                                                                                     |
| collective                                       | §6 on the group's path (TP group, EP group, or this stage's PP boundary)                                                                                                                              | —                                                                                                                                                 |
| transfer (off-device read, deployment `offload`) | —                                                                                                                                                                                                     | the tier/link model of §12 (latency `transferLatencySec` + service `transferBandwidthSec`); `bound = 'transfer'`. No FLOP, no device-memory bytes |

`t = max(compute, memory)`; the larger term is the op's `bound` (a tie counts as compute). A tensor op whose
lookup falls back to the vector peak keeps the efficiency term (conservative).

**Elementwise fusion** (`SimOptions.elementwiseFusion`, default **`'none'`**): `'none'` charges every
elementwise op its activation read and write; `'ideal'` sets them to 0 (fused into a neighbouring GEMM),
keeping weight and state bytes. The default is the conservative one; for REF_DENSE_8B prefill the difference is
≈ 5% (§10).

### Per-op latency floor

`src/sim/engine/opTime.ts: opTime, withLatencyFloor`; `SimOptions.perOpLatencyFloorSec`. UI: Hardware → Advanced →
"Per-op latency floor µs" (URL `lf`). Tests: `engine/whatIf.test.ts`.

A sensitivity setting for review F1 (`docs/review/practicality-review.md`): no op finishes faster than a fixed latency
L. A small kernel cannot ramp up to full bandwidth on a few KB, and a dependent op pays at least the DRAM load-to-use
latency before its first bytes arrive. Default **L = 0**: off, and the op contexts carry no floor at all, so results are
bit-identical.

```
t_op = max(t_op, L)          every IR op: on-device ops, collectives (incl. PP sends), off-device transfers
Δ    = L − t_op  (when > 0)   booked where the stage puts the op on its critical path:
  on-device op:   sec = L                       → compute_s
  collective:     α′ = α + Δ, β unchanged       → lat_s (never hidden); α′ + β = L
  transfer:       α_xfer′ = α_xfer + Δ          → xfer_s
```

- **max, not additive.** A large op hides its first-byte latency: once it keeps enough bytes in flight (Little's law),
  the latency overlaps the stream, and its roofline time stands. Only ops shorter than L are raised. An additive
  `t_roofline + L` would also charge the large ops that hide it, double-counting. The max form is the review's
  "hardware floor" `Σ max(t_op, L)`; the additive form `t_op + τ` belongs to host-side kernel launches, which run
  before the kernel and are not hidden ("Engine dispatch overlay" below).
- **Every IR op**, collectives and transfers included: a 100 µs floor lifts each TP all-reduce to 100 µs too. A
  collective's lift goes to its latency term α, which is on the dependency chain under both overlap modes ("Stage
  time and overlap"), so it is never hidden behind compute; its bandwidth term β is link occupancy and stays as it
  is (so the pipeline's link bottleneck `βp2p` is unchanged). A transfer's lift goes to its latency term likewise.
  Below the floor, the op's own α is already part of its time, so the lift tops it up rather than adding to it.
- A floored op keeps its own `bound`; `OpResult.latencyFloorSec = L` marks it.
- Every op counts as on the dependency chain (the stage time sums them, "Stage time and overlap"). Engines
  fuse elementwise ops and overlap independent ones, so the IR op count overstates the real chain (review F1).
- **No default value until a microbenchmark exists.** No measurement isolates a per-dependent-op hardware floor; the
  review's 0.5–1 µs is an assumption. The setting stays a what-if at 0 until a microbenchmark measures it.
- Review probes (decode @ 64, batch 1; `whatIf.test.ts`), L = 0.6 / 1 µs: Llama-3-8B Q4_K_M on an RTX 4090 −2.5% /
  −4.3%; Qwen3-30B-A3B Q4_K_M on an RTX 4090 −12.0% / −18.9%; gpt-oss-120b on a DGX Spark −1.1% / −2.1% (the
  review: −2.5 / −4.3%, −12 / −19%, −1.1 / −2.1%).

### Attention softmax

`src/sim/engine/opTime.ts: attentionOpTime, attentionSoftmaxSec`; `src/sim/ir/metrics.ts: attentionScores,
SOFTMAX_VECTOR_FLOP_PER_SCORE`; rates `ComputeEngine.sfuOpsPerSec` (§4 "SFU rate"). Tests:
`engine/opTime.test.ts`, `engine/softmaxCost.test.ts`, `ir/metrics.test.ts`.

Fused attention computes, per score, Q·Kᵀ and P·V on the tensor cores and the softmax in between: one
exponential on the special-function units and a few FP32 vector ops (running max, scale-and-subtract into the exp
argument, row sum, the convert of P to the MMA format, the online-softmax rescale of the output and the sum,
amortised). With `S = seqs·heads·pairs` scores (§3 "Attention"):

```
t_mma     = F / (peak · efficiency)                 (the tensor roofline, as for any tensor op)
t_exp     = S / sfuOpsPerSec                        (one exp2; the e^x → 2^(x·log2 e) scale folds into the FMA)
t_vec     = 5 · S / vectorPeak(fp32)                (5 FLOP per score at the vector FP32 FLOP/s peak)
t_compute = max(t_mma, t_exp + t_vec)               (softmax overlapped with the MMAs)
t         = max(t_compute, t_memory)
```

- **Overlap.** FlashAttention-3 (Hopper) and -4 (Blackwell) ping-pong warpgroups so one tile's softmax runs while
  another tile's MMAs do, so the headline is the overlapped bound. Inside the softmax, the SFU and the FMA pipes are
  added (conservative: they are separate pipes, but the vector ops depend on the exp). `Math.max` returns `t_mma`
  itself when the softmax does not bind, so such ops are bit-identical to the model before the softmax term.
- **The 5 FLOP** count each op as 1 FLOP against a peak that counts an FMA as 2, i.e. half a lane-cycle per op: the
  rate of packed FP32×2 issue (Blackwell `f32x2`, CDNA packed FP32), optimistic elsewhere, which keeps the bound an
  upper bound. With 5 full lane-ops (10 FLOP) B200 Llama-3-8B fp8 would read ≈ − 9 % instead of − 4 % @ pp8k
  (analytic: attention × 1.52 instead of × 1.23 on a 20 % attention share).
- **Which ops.** Causal masking, sliding windows, top-k (DSA) and compressed (CSA) entries enter through `S`, the
  scores the kernel computes (≈ half the square for a causal prefill). The indexer's ReLU scores and linear-attention
  (`recurrent`) ops have no softmax term. Decode stays bound by the KV read: per score it reads `kvRead` bytes of
  K and V, far more time than one exp (`softmaxCost.test.ts` asserts, for every preset engine and a GQA, a head-dim-64
  fp8-KV and an absorbed-MLA decode at 4k and 128k, seqs 1 and 64, that the op's time and bound do not change).
- **Where it binds.** The ratio of tensor to SFU throughput: per score `t_mma = 4·hd / peak`, so the softmax binds
  when `peak / sfuOpsPerSec > 4·hd / (1 + 5·sfuOpsPerSec / vectorPeak)`. H100 bf16 (989 TFLOP/s, 4.19e12 exp2/s) at
  hd 128 has t_mma 5.18e-13 s against a softmax of 3.13e-13 s per score: overlapped, it never binds. B200 (2.25
  PFLOP/s, 4.69e12 exp2/s) at hd 128: 2.28e-13 against 2.80e-13 (× 1.23 on attention); at hd 64 × 2.46. The ratio
  `peak / sfuOpsPerSec` (tensor FLOP per exp2) is 236 on H100 and 480 on B200, but 128 on MI300X and MI355X, which
  therefore stay MMA-bound even at hd 64; consumer GPUs, DGX Spark, Intel and Apple (ratios ≤ 68) never bind.

Probe (the practicality review's F6 table, reproduced; single device, batch 1, `elementwiseFusion 'none'`, bf16
attention; before = the same device without `sfuOpsPerSec`):

| Model, device                       | pp8k                              | pp32k                        | pp128k                     | serial @ pp8k     | pp32k               | pp128k              |
| ----------------------------------- | --------------------------------- | ---------------------------- | -------------------------- | ----------------- | ------------------- | ------------------- |
| Llama-3-8B fp8, B200                | 204,388 → 196,178 tok/s (− 4.0 %) | 129,382 → 116,359 (− 10.1 %) | 52,151 → 44,092 (− 15.5 %) | − 19.4 %          | − 37.8 %            | − 49.5 %            |
| gpt-oss-120b (hd 64) mxfp4, B200    | 181,385 → 159,000 (− 12.3 %)      | 140,981 → 97,825 (− 30.6 %)  | 73,922 → 38,360 (− 48.1 %) | − 19.5 %          | − 42.7 %            | − 61.0 %            |
| Llama-3-8B bf16, H100               | 54,493 (0 %)                      | 40,520 (0 %)                 | 19,732 (0 %)               | − 6.7 %           | − 17.6 %            | − 29.3 %            |
| Llama-3-8B fp8, H100                | 89,290 (0 %)                      | 56,805 (0 %)                 | 22,924 (0 %)               | − 10.5 %          | − 23.0 %            | − 32.5 %            |
| gpt-oss-120b mxfp4, B300            | − 4.7 %                           | − 13.6 %                     | 73,922 → 55,507 (− 24.9 %) | − 12.9 %          | − 31.6 %            | − 49.2 %            |
| gpt-oss-120b mxfp4, MI355X / MI455X | 0 %                               | 0 %                          | 0 %                        | − 7.2 % / − 7.6 % | − 19.6 % / − 20.3 % | − 33.9 % / − 34.4 % |

All rates are prefill tok/s at the stated prompt length. Decode on B200 and H100 is unchanged at 8k and 128k. The
**serial** columns are a documented sensitivity only, not a setting: `T_serial = T + Σ_attention [max(t_memory,
t_mma + t_softmax) − t]`, the bound if the softmax did not overlap the MMAs at all.

**Cross-check against real kernels** (attention-only bound `F / t` for a causal 16k prefill, batch 2, 16 × hd 128
or 32 × hd 64 heads; `softmaxCost.test.ts`). The bound must stay above published forward throughput:

| Device, attention        | bound (TFLOP/s) | published                    |                |
| ------------------------ | --------------- | ---------------------------- | -------------- |
| H100, bf16, hd 64 / 128  | 817 / 982       | FlashAttention-3: 497 / 648  | above          |
| H100, fp8, hd 64 / 128   | 817 / 1634      | FlashAttention-3: 613 / 1008 | above          |
| B200, bf16, hd 128 (32k) | 1829            | FlashAttention-4: 1613       | above, by 13 % |

**Known limitation (not in the headline): exp emulated on the FMA pipes.** FlashAttention-4 computes 10–25 % of
its exponentials with a polynomial on the FMA units instead of the SFU, and an FP4-attention preprint (arXiv
2609.04105, trading accuracy) reports 2998 TFLOP/s on GB200 at hd 128, above this MUFU-only bound (1829 on B200).
The headline costs every exp on the SFU, as the CUDA guide rates it; a kernel that moves exps to the FMA pipes can
exceed it on Blackwell. A sensitivity would add the FMA pipes' spare throughput to the SFU rate (an exp2
polynomial costs ~10 FMA, so each 1 % of FMA-pipe time buys ~1/10 of an exp per lane-cycle).

### Stage time and overlap

`src/sim/engine/pipeline.ts: stageResult, layerResult`. Per stage and microbatch: `compute_s` = Σ op times of every
op that is neither a collective nor an off-device transfer, `lat_s` = Σ collective α terms, `bw_s` = Σ collective β
terms, `p2p_s` = the send to the next stage, `xfer_s` = Σ transfer op times (0 unless something is offloaded).

First the core time `t_core,s`:

- `overlap = 'none'`: `t_core,s = compute_s + lat_s + bw_s + p2p_s`
- `overlap = 'ideal'` (**default**): `t_core,s = lat_s + max(compute_s, bw_s)`. β can hide behind independent work
  (other token or column chunks); α is on the dependency chain. The PP send is not in `t_s`; it lengthens the
  latency path and, as a link resource, bounds the cycle (below).
- `exposedCommSec = t_core,s − compute_s`.

Then the off-device transfers, by their prefetched share (§12 "Expert prefetch"). Each transfer op carries
`q = prefetchedFraction`, set by lowering from the prefetch accuracy p. With `core_s = max(compute_s, bw_s)` under
`overlap 'ideal'` and `compute_s + bw_s + p2p_s` under `'none'`:

```
pbar_s = Σ_ops q·t_xfer / xfer_s                          time-weighted mean of q over the stage's transfer ops
t_s    = lat_s + core_s + xfer_s − pbar_s·min(core_s, xfer_s)
```

- `pbar_s = 1` (p = 1, perfect prediction): `t_s = lat_s + max(core_s, xfer_s)`, the former `offloadOverlap
'ideal'`.
- `pbar_s = 0` (p = 0, no prefetching): `t_s = t_core,s + xfer_s`, the former `'none'`.
- Always `lat_s + max(core_s, xfer_s) ≤ t_s ≤ t_core,s + xfer_s`.
- `exposedTransferSec = t_s − t_core,s`. With `xfer_s = 0`, `t_s = t_core,s` exactly. The code computes `xfer_s = 0`
  and both endpoints with their own (former) expressions, so those results are bit-identical to before.

Under `overlap 'ideal'` and `collectiveAlgorithm 'best'`, `engine/simulate.ts: simulatePhase` compares
three assignments per stage: (a) each all-reduce's min-total choice (including mesh-direct where applicable),
(b) all ring, and (c) all tree. It picks the minimum exact core score `Σα + max(compute_s, Σβ)`;
ties preserve the original assignment. With β hidden this prefers exposed α, rather than total α+β
(audit G7). Explicit ring/tree choices and `overlap 'none'` are unchanged. This is a **bounded heuristic**,
not a search over every mixed ring/tree assignment. It cannot worsen the candidate core time; it does not
optimize algorithm assignment against the independent offload-transfer overlap floor.

### Prefill without overlap

`src/sim/engine/simulate.ts: simulate` → `SimResult.prefillWithoutOverlap`. UI: a compact second figure next to the
prefill headline ("Without overlap: 2,530" under "4,210 tok/s @ 8k"). Tests: `engine/whatIf.test.ts`.

The default `overlap 'ideal'` hides every collective's bandwidth term behind compute, which vLLM, SGLang and llama.cpp
do not achieve by default (review F7). When the run uses `'ideal'` and its prefill has inter-device communication
(`comm.latencySec + comm.bandwidthSec > 0`: a TP/EP collective or a PP send over a link), the engine also costs the
same lowered prefill under `overlap 'none'` (`t_core,s = compute_s + lat_s + bw_s + p2p_s`, above) and reports its
rates, a labelled what-if; the headline stays `'ideal'`. Absent on one device, and when the run already uses `'none'`
(then the headline is that figure). It equals a separate run with `overlap: 'none'`, bit for bit.

Review probes (`whatIf.test.ts`; the review's exact workloads are not recorded, and the PCIe payload efficiency and
the softmax cost landed since): 4× RTX 4090 TP4 (direct P2P), Llama-3-70B Q4_K_M, prefill @ 8k: −38.3% (review
−40%); 2 × DGX Spark TP2, Llama-3-70B NVFP4, @ 2k: −31.5% (review −31.5%); 2× RTX 3090 TP2 (direct P2P),
Qwen3-30B-A3B Q4_K_M with fp16 GEMMs, @ 512: −21.5% (review −24%).

### Comm latency vs bandwidth split

`src/sim/engine/pipeline.ts: phaseComm, phaseTransfer` → `PhaseResult.comm`, `PhaseResult.transfer`. Display only:
sums of numbers the engine already computes, so no rate depends on them (design doc §4.2).

- `comm.latencySec = Σ_s lat_s + Σ α_forward + α_return` (return 0 for pp1 or prefill).
- `comm.bandwidthSec = Σ_s bw_s + Σ β_forward + β_return`.
- `comm.exposedSec = Σ_s exposedComm_s + Σ p2p` under `overlap 'ideal'` (the sends are on the latency path, not
  in `t_s`); under `'none'` each forward send is already inside `t_s` and is not added again.
  Return α+β is added in both modes (never inside a stage). `PhaseResult.loopBack` exposes its op cost.
- `transfer = { sec: Σ_s xfer_s, exposedSec: Σ_s exposedTransfer_s, bytes: Σ usefulBytes, requests: Σ reads.count }`
  for one microbatch through every stage; absent when no stage has a transfer op.
- Example (`pipeline.test.ts`, design doc §8.3): REF_DENSE_8B on `SYNTH_CLUSTER`, decode @ 1024. tp 4: 73 tree
  all-reduces of `2·⌈log2 4⌉·1 µs` plus sampling AG give `latencySec = 295 µs`,
  `bandwidthSec = 11.96056 µs`. tp 2 pp 2 (b 2): `147 µs` in stages (146 AR + 1 AG)
  - forward α 1 µs + return α 1 µs = `149 µs`; forward β 0.08192 µs and return β 0.00004 µs
    both enter `bandwidthSec`.

### Pipeline and DP

`src/sim/engine/pipeline.ts: microbatching, phaseResult`; `src/sim/engine/simulate.ts: simulate`.

- `B_r = batch/dp`, `m = min(pp, B_r)` microbatches of `b_mb = ceil(B_r/m)` sequences. Each phase is lowered
  once with `seqs = b_mb`. `Σ p2p` below is the sends' full time (α + β) under `'ideal'` and 0 under `'none'`
  (already in t_s).
- **Resources.** Each stage and each PP boundary link is a resource every microbatch occupies once per cycle:
  stage s for `t_s`, the link after stage s for the send's bandwidth time `βp2p_s = bytes/B` (α is latency,
  not occupancy). Overlapping a send with compute does not let the link carry more bytes than its bandwidth, so
  the bottleneck per microbatch is `b = max(max_s t_s, max_s βp2p_s, β_return)`.
  The decode return is another link resource; prefill has β_return = 0.
  (Under `'none'`, t_s ≥ the forward βp2p_s already.)
- **Decode**: `cycle = max(m·b, Σ_s t_s + Σ p2p + return_p2p)` (the sampled IDs must return to stage 0).
  `return_p2p = α_return + β_return` in both overlap modes; it can be hidden by the `m·b` branch,
  so adding it need not increase cycle time. `tokensPerSec = dp·B_r/cycle`, `perSequenceTokensPerSec = 1/cycle`,
  `latency = cycle`.
- **Prefill** (a steady stream of replica batches; the bottleneck resource bounds it): `cycle = m·b`,
  `tokensPerSec = dp·B_r·promptLen/cycle`, `latency (TTFT of a replica batch) = Σ t_s + Σ p2p + (m−1)·b`,
  `perSequenceTokensPerSec = promptLen/latency`.
- Example (`pipeline.test.ts`): TINY_DENSE, pp 2, batch 2, prompt 128 over a 1 B/s link: each microbatch sends
  128·256·2 = 65,536 B, so `cycle = 2·65,536 s = 131,072 s` and TTFT ≈ 131,072 s; decode sends 512 B per
  microbatch → `cycle = 1,024 s`.
- **Context of each rate** (`src/sim/engine/simulate.ts: simulate`, `types.ts: Workload`). Every rate is quoted
  at the context it applies to ("33.9 tok/s @ 8k"):
  - **Prefill @ P** (`promptLen = P`): the whole P-token prompt from an _empty_ KV cache, so `past = 0` and
    `qLen = kvLen = P`. Prefill tok/s @ P = `P / latency` per sequence (`dp·B_r·P/cycle` aggregate).
  - **Decode @ N** (`decodeContextLen = N`): one decode step with N tokens _already_ in the KV cache. The new
    token's query also sees its own key, so `qLen = 1` and `kvLen = past + qLen = N + 1`. Attention FLOP
    (`2·seqs·heads·(N + 1)·(qk + v)`) and KV reads (`seqs·(N + 1)·elemsPerToken·bpe(kv)`) grow linearly in N
    (full attention; a window or top-k selection caps both, §3 "Attended pairs").
    Decode tok/s @ N = `1/cycle` per sequence (`dp·B_r/cycle` aggregate). Test: `simulate.test.ts`.

### Engine dispatch overlay

`src/sim/engine/dispatch.ts: DEFAULT_DISPATCH_SEC_PER_KERNEL, stageKernelCount, withDispatch`;
`SimInput.overlays.engineDispatch` → `SimResult.decodeWithDispatch`. UI: Hardware → Advanced → "Engine dispatch
overlay" (off by default, URL `ed=1`), shown as "With engine dispatch: …" under the decode headline. Tests:
`engine/whatIf.test.ts`.

**Engine overhead, informational only.** This is the project's one engine-overhead exception (§9), and it is opt-in:
it never replaces the headline, which stays the theoretical maximum. Every launched kernel costs the host (CPU launch,
command processor, graph-node dispatch) a fixed time τ before the device runs it.

```
n_s  = Σ over the stage's layers of the IR ops        kernels per stage per microbatch step
t_s' = t_s + n_s·τ                                    then the pipeline rule ("Pipeline and DP") on t_s'
```

- **Kernel count = IR ops, no fusion.** One kernel per IR op of the decode step (GEMM, attention, norm, activation,
  collective, transfer, …; the draft layers too with speculative decoding). Engines fuse elementwise ops and capture
  CUDA graphs, so a real engine launches fewer kernels; this is the unfused count. The PP sends and the sampled-token
  return are not counted.
- **τ = 2 µs** by default, inside the measured 1.3–2.9 µs of review F1 (`docs/review/practicality-review.md`):
  Hazy Research, ["Look Ma, No Bubbles!"](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), launch
  ≈ 2.1 µs with CUDA streams and ≈ 1.3 µs with CUDA graphs;
  [mpk-apple notes](https://github.com/jiazhihao/mpk-apple/blob/main/docs/research/apple-inference-systems.md), MLX
  ≈ 2.75 µs per tiny dependent kernel. `EngineDispatchOverlay.perKernelSec` overrides it.
- **Additive**, unlike the per-op latency floor: it is the review's "dispatch floor" `N_ops·τ`. It fits batch 1,
  where most decode kernels are shorter than τ and the device waits for each launch. Where kernels are longer than τ
  (large batches), an engine's asynchronous launches overlap them, so the overlay overstates the cost there.
- Decode only. Prefill runs the same number of kernels per step for many more tokens, so its dispatch share is small.
- Review probes (decode @ 64, batch 1, τ = 1 / 2 / 3 µs; `whatIf.test.ts`): Llama-3-8B Q4_K_M on an RTX 4090 (355
  kernels): −7.1% / −13.3% / −18.7%; Qwen3-30B-A3B Q4_K_M on an RTX 4090 (723): −26.7% / −42.1% / −52.2%;
  gpt-oss-120b on a DGX Spark (507): −3.7% / −7.1% / −10.3%. The review: −7 / −13 / −19%, −27 / −42 / −52%,
  −4 / — / −10%.

---

## 8. Memory fit

### Memory fit

`src/sim/engine/memory.ts: memoryReport, stageWeightBytes, stageSequenceStateBytesPerSeq,
stageActivationPeakBytes`; `lower/lower.ts: blockResidentWeightBytes, blockSequenceStateBytes`; the per-block
helpers `gqaSequenceStateBytes`, `mlaSequenceStateBytes`, `csaSequenceStateBytes`, `sparseSelectionStateBytes`,
`linearAttentionSequenceStateBytes`, `ngramSequenceStateBytes`, `shortConvSequenceStateBytes`,
`moeResidentWeightBytes`, `ngramResidentWeightBytes`, `embeddingResidentWeightBytes`, `lmHeadResidentWeightBytes`.

Per device of stage s:

- `W_s` = Σ resident block weights (non-MoE blocks: the weight bytes of a one-token lowering, so resident and
  streamed weights agree by construction; MoE: all local experts; n-gram: the table shard `ceil(rows/tp)·d·bpe(lookup)`
  plus its projections and vectors) + the embedding shard on stage 0
  (`ceil(vocab/tp)·h·bpe(emb)`) + the LM-head shard and final norm on the last stage. A tied head shares the
  embedding allocation only with pp = 1 and an identical stored format (dtype and `extraBitsPerElem`,
  `memory.ts: sameStoredFormat`); a tied head stored differently (e.g. bf16 embeddings, int4 head) is a second
  copy and counted in full.
- `state_s` per sequence at `ctx = max(promptLen, decodeContextLen + 1)` (the cache after prefill, or after the
  decode step appended its token): GQA `min(ctx, W)·2·KVl·hd·bpe(kv)`
  (a sliding window keeps W tokens) + K/V conv state `(K−1)·2·KVl·hd·bpe(act)`; MLA latent `ctx·(rkv+dr)·bpe(kv)`
  (0 with `kvCacheFromLayer`); CSA `min(ctx, W)·hd·bpe(kv)` + owner `ceil(ctx/r)·hd·bpe(ckv) + [r > 1] 2·(coff·r)·(coff·hd)·4`;
  owned indexer keys `ceil(ctx/p)·hd_I·bpe(key)` (0 for `reuse` and `ownsKeys: false`); linear attention
  `vL·dk·dv·bpe(state) + C·(K−1)·bpe(act)`; short conv `(K−1)·dim·bpe(act)`; n-gram conv
  `(K−1)·dilation·O·bpe(act)`; hyper-connections 0.
- `act_s` = the largest activation read + write of any single op of the stage over the decode step and the LAST
  prefill chunk (see "Activation peak" below): the transient inputs and outputs of the op running.
- `reserve` = the runtime reserve per device (`HardwareStack.runtimeReserveBytes`, default 0; see "Runtime
  reserve").
- `total_s = W_s + B_r·state_s + act_s + reserve (+ staging_s)`; `fits = total_s ≤ cap`, with the GPU-usable
  capacity `cap = f·capacity` (see "GPU-usable capacity"; f = 1 on dedicated memory) and capacity =
  `memCapacityBytes ?? capacityOptionsBytes[0]` of the modified device. With weights offloaded, `W_s` keeps only the
  resident part, `staging_s` is the streaming buffer, and the off-device tiers are checked too (§12 "Memory tiers").
- `maxSequencesPerReplica = floor(min_s (cap − W_s − act_s − reserve − staging_s)/state_s)` (Infinity when no
  stage keeps per-sequence state). The activation peak is the one at the evaluated batch, so this is an estimate.

### Activation peak

`memory.ts: prefillChunkShape, DEFAULT_PREFILL_CHUNK_TOKENS`; `simulate.ts` step 6.

Engines prefill long prompts in chunks of C tokens per step (vLLM `max_num_batched_tokens`, SGLang
`--chunked-prefill-size`), so the prompt's activations never exist all at once. The fit lowers one more prefill step,
the one that holds the most tokens under a budget of C per microbatch of `b_mb` prompts of P tokens:

- `b_mb·P ≤ C`: the whole microbatch (the step the rate is costed on; no extra lowering).
- `P ≤ C < b_mb·P`: `floor(C/P)` whole prompts, `qLen = kvLen = P`.
- `C < P`: the LAST chunk of one prompt, `qLen = C`, `kvLen = P` (it attends to the whole prompt).

Every op's activations grow with the step's tokens `seqs·qLen`, so this step bounds all chunks. Its KV reads are
cache state, already counted in `state_s`. C = `Workload.prefillChunkTokens`, default 8192 (SGLang's default and
vLLM's for its `LLM` class and H100-class API servers; smaller budgets only lower the peak). Only the fit uses it:
prefill tok/s @ P is still the whole prompt in one pass (§7). With `prefillChunkTokens ≥ b_mb·P` the fit is the
one-pass peak of before.

### GPU-usable capacity

`memory.ts: resolveGpuUsableFraction, unifiedMemoryWarnings`; `MemorySpec.gpuUsableFraction`,
`HardwareStack.gpuUsableFraction`.

On `kind: 'unified'` devices the OS lets the GPU allocate only part of the pool. The fit checks against
`cap = f·capacity`, with `f = HardwareStack.gpuUsableFraction ?? MemorySpec.gpuUsableFraction ?? 1` (the override is
a raised OS limit; dedicated memory always has f = 1). The preset defaults, each cited in its preset:

| device                                     | f                   | what it is                                                                                                   |
| ------------------------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| Apple M3/M4/M5 Max and Ultra (all ≥ 36 GB) | 0.75                | macOS `recommendedMaxWorkingSetSize` (Apple: 48 GB of 64 GB); raised with `sudo sysctl iogpu.wired_limit_mb` |
| DGX Spark                                  | 119.7 / 128 = 0.935 | what CUDA reports as total global memory on DGX OS                                                           |
| Ryzen AI Max+ 395 (Strix Halo)             | 0.5                 | Linux default GTT limit (ROCm guide); `ttm.pages_limit` raises it to ~120 GB of 128 GB                       |

`f` is a ratio of usable to installed bytes, so it holds whatever unit the capacity is recorded in. `DeviceMemory`
reports both `capacityBytes` (= cap, what the check uses) and `physicalCapacityBytes`; `simulate` warns with the
share used and, for a stage above cap but within the installed memory, that it fits only with a raised limit. Rates
do not depend on f.

Host RAM IS the device pool on unified memory, so it is never a tier: no `host_ram` entry appears in
`MemoryReport.tiers` (`stageTierBreakdown` throws if a group would land there), and pinned-host table scales (§12)
are device bytes in `W_s`. Only `nvme` is a tier.

### Runtime reserve

`HardwareStack.runtimeReserveBytes`, per device, default 0. Framework and driver memory besides the model (CUDA
context, allocator pools, GEMM and attention workspaces; commonly 0.5–1.5 GB on a CUDA GPU) is added to every
stage's total. No vendor publishes it, and it depends on the engine and driver, not the hardware, so the bound keeps
0 unless the user sets it (Hardware → Advanced → Memory fit).

---

## 9. Deliberately excluded, and approximations

Excluded on purpose (only hardware and math limits are modeled):

- **Inference-engine knobs**: paged attention, continuous batching, chunked prefill (the rate; the memory fit's
  activation peak assumes a chunk, §8 "Activation peak"), scheduler overhead,
  kernel launch overhead of a specific engine, CUDA graphs. The exception: speculative decoding and weight / expert /
  lookup-table offloading are **opt-in deployment-scenario knobs, off by default** ([§12](#12-offloading),
  [§13](#13-speculative-decoding); design `docs/design/offload-specdec-latency.md`). Off, results are bit-identical;
  on, they are the hardware/math bound given the stated acceptance, hit ratio and placement. Draft trees, the DSpark
  confidence scheduler, predictor accuracy and the CPU-compute offload mode (a hook validation rejects) are not
  modelled. The **engine dispatch overlay** (§7) is the one engine-overhead exception: opt-in only, off by default, a
  per-kernel launch cost added to decode and shown as a labelled second figure; the headline never includes it.
- Dequantization compute of weight-only quantized GEMMs (they run at the wider format's peak, §2).
- Split-K / stream-K; L2 / cache reuse beyond compulsory traffic.
- Clock throttling beyond an explicit `clockScale` modifier.
- NVLS / SHARP in-switch reduction; PCIe switch / shared-uplink contention; the memory traffic of collectives
  and comm/compute contention for memory.
- Sequence parallelism; full-logits gather (small sampling exchange is included); attention sinks; skewed or
  correlated MoE routing (expert weights follow the iid-uniform expectation, §5 "Expected distinct experts"); the
  framework / driver memory reserve (0 unless set, §8 "Runtime reserve").

Documented approximations (each isolated in one constant or rule):

- Attention softmax (§7 "Attention softmax"): `SOFTMAX_VECTOR_FLOP_PER_SCORE` = 5 FLOP per score at the vector FP32
  FLOP/s peak (the FlashAttention O-rescale is folded into it); the SFU and FMA parts in series, the whole softmax
  overlapped with the MMAs (the serial variant is a documented sensitivity, not a setting); every exp on the SFU
  (FlashAttention-4's polynomial exp on the FMA pipes is not credited, so such kernels can exceed the Blackwell
  bound); estimated SFU rates take the upper end (§4 "SFU rate"). Not costed: the indexer's ReLU and head weighting
  (vector work next to its GEMM), and the linear-attention decay-gate exps (≈ chunkLen/2 per token per head, next
  to the `(C+1)·(dk+dv)` FLOP of the chunk).
- GEMV path (§4 "GEMV path"): `gemvMaxRows = 1` on Apple only (kernel dispatch, conservative); a GEMV runs at the
  full vector peak of its math format.
- Recurrent `c_s` constants (chunked 4; step GDN 7, Mamba2 5, linear 4) and the chunk length
  `min(chunkTokens ?? 64, qLen)`.
- linear-attention `in_proj` carries `2·vH` gate outputs for GatedDeltaNet and linear, `vH` (dt) for Mamba2 (KDA:
  β only, its decay comes from the low-rank forget gate).
- Mamba2 chunked SSD FLOPs leave out the state passing between chunks (`2·N·P/Q` per token and head: `2/Q` FLOP
  per state element next to `c_s = 4`, under 0.4 % at Q = 128), the decay mask `exp(segsum)` (`Q+1` per token and
  head), the dt softplus and the D skip (vector work, a few FLOP per element).
- MLA form by phase (absorbed for decode and sparse prefill, MHA for dense prefill).
- The indexer is replicated on every TP rank; its key norm is counted as a LayerNorm.
- Kimi K3: the `mla_use_nope` RoPE slice is cached but not rotated; the IR keeps it as `qkRopeDim`, so one small
  `rope` elementwise op per MLA layer is an over-count. The `residual_mix` FLOP constant `6·C`, the `situ`
  activation constants, and the latent all-reduce emitted also without a latent norm (§5 "Latent MoE").
- conv1d and recurrent state reads are 0 on the first prefill step (no prior state).
- A sliding-window cache holds `min(ctx, W)` tokens.
- Row-parallel biases are counted on every TP rank.
- Reductions travel as bf16 when activations are narrower than 16 bits.
- Prefill computes LM-head logits for one position per sequence (`Tout = seqs`).
- Balanced MoE routing on the busiest device (ceilings), with the EP load imbalance ρ (`DEFAULT_EP_IMBALANCE`
  1.2, §5 "EP load imbalance"); the expert weights read are the expected distinct experts under iid-uniform
  routing, an expectation rather than a bound (§5 "Expected distinct experts").
- Elementwise FLOP-per-element table (§3) and the fp32-accumulate default.
- The activation peak in the memory fit is the largest single op.
- Fabric shapes (§6 "Fabric shapes"): a TP / EP group covering part of a ring is a chain (the long way round is
  ignored); PP flows on a ring are not split over both directions (conservative by at most 2× on that send's β);
  relays go through the host and count against the all-ports cap C (no in-NIC switching is assumed: L3-routed RoCE
  between Sparks is reported not to work and mesh forwarding is unmeasured); tree steps are assumed not to share
  links; mesh striping over k cables is ideal; every shape but switched needs one device per node; intra-node links
  stay switched (`XGMI_MI300X_8_GPU` encodes its full mesh as aggregate injection under the switched formulas, which
  equals the mesh's direct β but keeps the ring's 2(n−1)α; re-modelling it is a follow-up).
- TCP socket copy (§6 "TCP socket copy"): the CPU memcpy runs at the DRAM peak (B_host; the CPU's own copy rate is
  not datasheet-published, so this is optimistic); `MSG_ZEROCOPY` / `TCP_ZEROCOPY_RECEIVE` are not assumed (NCCL's
  socket transport, Gloo and MLX's ring do not use them by default); a dedicated GPU over TCP keeps NCCL's 512 KiB
  step for the copy stages too (conservative: the kernel pipelines within a step); a TCP relay forwarded by the
  kernel's IP stack is charged the copy fill per hop although it never copies to user space (≤ 2·s/r_sock per relay);
  the NIC's DRAM accesses are charged only on staged paths (TCP or PCIe staging), never on zero-copy RDMA; the copy
  traffic does not contend with compute for unified memory (the existing exclusion of collective memory traffic).
- Offload (§12): the DMA write of streamed bytes into device memory is not charged to its bandwidth; expected
  misses and requests are real numbers (no rounding; the staging buffer rounds up); the prefetch window is the whole
  stage's core time and wrong guesses cost only otherwise idle link time (§12 "Expert prefetch"); the host link
  takes no link modifiers.
- Speculative decoding (§13): `m_ctx = ⌈E[τ]⌉` context rows; every MTP pass reads `N + k + 1` keys; rejected KV
  rows and tentative MTP rows are overwritten, not charged back; the selector's and Markov head's sequential walks
  are not costed; the DeepSeek-V4.1 MoE drafter adds its context rows to the compressed-KV GEMM and runs its
  attention non-causally, without the context rows' kv norm / RoPE.

Shortlist-model approximations (sub-percent unless noted):

- DeepSeek-V4.1 prefill runs all layers on all N prompt tokens, as the reference `model.py` does (no "Bounded
  Replay" of the decoder on the last window only).
- Attention sinks, partial-RoPE dims (rope is costed over the RoPE dims the IR names: `qkRopeDim`, CSA `ropeDim`;
  GQA rope over the whole head), `swigluoai` costed as gated silu, the shared-expert sigmoid gate, the Inkling
  router's E+S rows and shared-expert sink, final-logit softcap, the embedding norm, ssmax / log-scaled q, LeRoPE.
- Compressed / pooled keys: a query sees `p/r` keys, continuous (`< qLen/2` pairs over `floor(p/r)`); the
  compressor and its norm run on `T/r` entries per step, also continuous in decode (it pools once every r steps).
- The indexer key GEMM runs on all T tokens even when its keys come from pooled latents (DeepSeek: 512×128).
- The CSA compressor weights are read in their stored (module) format; the reference code promotes r > 1
  compressor weights to fp32 at run time, an implementation detail.
- DeepSeek-V4: the compressor APE (`r·coff·hd` fp32 per owner layer, 1.40 M params in V4-Flash) is not counted
  in the params or weight reads (the indexer's pooled-key APE is); the final `hc_head` projection (N·D → N mixing
  weights before the head, 65.5 k params) is not modeled (the streams collapse as in V4.1); the hash-routing table
  `tid2eid` (`num_hash_layers · vocab · k` int64 expert ids) is not counted, and the router GEMM still runs in
  those layers (`Gate.forward` computes the weights); the indexer compressor's fp32 pooling state is not counted
  as sequence state.
- DeepSeek-V4 indexer keys go through the generic indexer path: the key norm (`idx.k_norm`) runs on the `T·hd_I`
  pre-pool elements, while model.py's indexer `Compressor` normalizes the `T/r` pooled keys after pooling, so the norm
  is overstated by at most `(r−1)/r·T·hd_I` elements per indexer layer. The key RoPE and the Hadamard rotation
  (`rotate_activation`) of the indexer compressor are not emitted, as for the generic indexer. Both are elementwise
  and negligible next to the GEMM and attention bytes.
- The hyper-connection mixing weights the projections produce are consumed on chip (only the streams are read by
  `hc_collapse`); the embedding broadcast into N streams is ignored (< 0.5%).
- n-gram tables count toward weight memory and total params, not active params; the last n−1 token ids the hashes
  need are not counted as state.
- KDA `A_log` (vH) and `dt_bias` (vH·dk) are not modeled as weight reads; per-channel decay costs like GDN's.
- The relative-position bias gather and add inside attention are excluded (vector work; the softmax is costed, §7).
- A cross-layer source on another pipeline stage: the consumer's stage is assumed to hold the data; no transfer is
  costed and the data is not counted twice in memory (a `simulate` warning names each case).

---

## 10. Worked example: REF_DENSE_8B on one RTX 5090

REF_DENSE_8B has Qwen3-8B's shape (h 4096, 36 layers, 32 heads, 8 KV heads, hd 128, ffn 12288, vocab 151936,
8,190,735,360 params). bf16 weights, activations and KV; fp32 accumulate; batch 1. RTX 5090: bf16 / fp32-acc
209.5 TFLOP/s dense, 1.792 TB/s (whitepaper Table 3). Tests: `src/sim/engine/e2e.test.ts` (T9).

**Decode at a 1024-token context** (1024 tokens cached + the new one: `kvLen = 1025`). Bytes that must stream per
token: GEMM weights `2·(36·192,937,984 + 622,329,856) = 15,136,194,560 B` (linear weights per layer: qkv
25,165,824 + o 16,777,216 + MLP 150,994,944; plus the LM head) and KV
`36 layers·1025 tokens·(2·8·128·2 B) = 151,142,400 B`, total 15,287,336,960 B → bound
`1.792e12 / 15.287e9 = 117.2 tok/s`. The simulator gives **117.1 tok/s** (norms,
activations and the embedding row add 0.1%); every op is memory-bound.

**Prefill of 4096 tokens.** Tensor FLOP: linear GEMMs `2·6,945,767,424·4096 = 56,899,726,737,408`, attention
`36·2·32·8,390,656·256 = 4,949,010,284,544` (exact causal pairs 4096·4097/2), LM head for the last position
`2·622,329,856`; total 61,849,981,681,664 FLOP → `0.29523 s` at 209.5 TFLOP/s → bound **13,874 tok/s**.
The simulator gives **12,892 tok/s** with `elementwiseFusion 'none'` (0.93 of the bound: GEMM shape efficiency
0.96–1.0 and ≈ 29 GB of elementwise traffic ≈ 16 ms) and **13,575 tok/s** with `'ideal'` (0.98).
The attention softmax (§7) does not bind here: per score 6.55e12 exp2/s and 104.8 TFLOP/s vector give 2.0e-13 s
against the MMA's 512 FLOP / 209.5 TFLOP/s = 2.4e-12 s, so these numbers are unchanged by it.

**Memory**: weights 16,381,470,720 B + KV at 4096 tokens 603,979,776 B + activation peak 301,989,888 B
(the prefill `silu_mul`) = 17.29e9 B ≤ 32e9 B: fits.

The estimator's default state (`src/ui/state/appState.ts`, rendered by `src/ui/App.tsx`) is Qwen/Qwen3.8-27B
(bundled from its Hub fixture, `src/ui/state/bundledModel.ts`; 26.9e9 params, bf16) on one RTX 5090, TP 1,
batch 1, prefill length 8192 and decode context 8192. The simulator gives **3,953 tok/s @ 8k** prefill and
**34.4 tok/s @ 8k** decode, and the memory check says it does **not fit**: weights 53.79e9 B + state 0.69e9 B +
activation peak 0.86e9 B = 55.34e9 B > 32e9 B (the "RTX 5090 ×2 (PCIe)" setup with TP 2 holds it). The UI lets you
change the model and hardware; the About page renders this document.

---

## 11. Extending

These checklists are binding for later tasks. TypeScript's exhaustive `never` checks flag the missing `case`s.

**New op kind**

1. `src/sim/ir/ops.ts`: an interface with JSDoc (meaning, units) + an `Op` union member.
2. `src/sim/ir/metrics.ts`: a `case` in `opFlops` and in `opBytes`.
3. `src/sim/engine/opTime.ts`: a `case` in `opTime` (peak choice + roofline; a GEMM-equivalent shape if it runs on
   tensor cores).
4. `src/sim/lower/<block>.ts`: emit it.
5. This document, §3: a subsection with the formulas and code links (and §7 if its time rule is new).
6. `src/sim/ir/metrics.test.ts`: hand-derived numbers.

**New block kind**

1. `src/sim/ir/model.ts`: a `BlockSpec` union member + interface.
2. `src/sim/ir/builders.ts`: a `MixerParams` / `FfnParams` member + its mapping.
3. `src/sim/ir/params.ts`: a `case` in `blockParams` (and `activeBlockParams`).
4. `src/sim/lower/<kind>.ts` + the dispatch in `lower/lower.ts: lowerBlock`.
5. `lower/lower.ts`: `case`s in `blockResidentWeightBytes` and `blockSequenceStateBytes`.
6. This document, §2 and §5.
7. A lowering test.

**New modifier kind**

1. `src/sim/hw/modifiers.ts`: a `DeviceModifier` or `LinkModifier` union member with JSDoc (units).
2. A `case` in `apply*Modifiers` (incl. the provenance paths) and in `deviceModifierErrors` /
   `linkModifierErrors` (and `describe*Modifier`).
3. This document, §4 "Device variants and modifiers" (the table).
4. A test in `src/sim/hw/modifiers.test.ts`.

**New device preset**

1. The vendor file in `src/sim/hw/presets/`: an exported `DeviceSpec` const with a JSDoc header (what it is, the
   clock its peaks assume, footnote traps), every number in `sources[].fields` or `estimated[]`; add it to the
   file's `<VENDOR>_DEVICE_PRESETS` array.
2. `src/sim/hw/presets/index.ts`: add it to a `DEVICE_PRESET_GROUPS` group (the coverage test fails otherwise);
   unlock axes, if any, as an exported `UnlockAxis[]` in the vendor file and a `DEVICE_UNLOCKS` entry.
3. Spot values and vendor ratios in the vendor's `<vendor>.test.ts`; `hw/presets.test.ts` covers validation,
   every unlock selection and a `simulate()` smoke run automatically.
4. This document, §4 "Preset library" (and "Unlock axes" for new axes).

## 12. Offloading

Deployment knob `SimInput.deployment.offload` (design: `docs/design/offload-specdec-latency.md` §2). Off by
default: absent, `{}` or `lookupTables: { tier: 'device' }` emit no transfer op, add no memory, never read
`hardware.host`, and give bit-identical results (`engine/identity.test.ts`). With it on, the result is still the
hardware/math bound given the stated hit ratio, queue depth and prefetch accuracy: only bytes crossing the host link
and requests paying a latency are added. Worked examples W1, W3, W4 are tests in `engine/e2e-offload.test.ts`.

### Placement

`src/sim/engine/host.ts: resolveOffload`; `lower/offload.ts: expertPlacementAt, denseLayerPlacementAt,
tablePlacementAt`; `lower/moe.ts: expertFetchOp`; `lower/lower.ts: denseLayerFetchOp, layerWeightBytesOnDevice`;
`lower/ngram.ts: tableFetchOp`. Three weight groups, each lowered to one `TransferOp` (§3 op catalogue: `kind
'transfer'`) in front of the block that consumes it, **only when it has bytes to move**:

| group                                 | selector                             | stays on device                                                   | `reads` / `usefulBytes` per step       |
| ------------------------------------- | ------------------------------------ | ----------------------------------------------------------------- | -------------------------------------- |
| `routed_experts` (`moe.expert_fetch`) | MoE layers in `layers` (default all) | `C = f·E_loc` experts, router, latent projections, shared experts | `[{ M, W_expert }]` / `M·W_expert`     |
| `dense_layer` (`layer.fetch`)         | layers `[from, to)`                  | nothing of the layer                                              | `[{ 1, W_layer }]` / `W_layer`         |
| `lookup_table` (`<block>.row_fetch`)  | every n-gram table                   | projections, gate vectors, conv taps                              | per scale layout, below / `R·rowBytes` |

- `W_expert = gemmWeightBytes(gated ? 3 : 2, l, fe/etp, w_routed)`, `l = latentDim ?? h` (`moe.ts: expertWeightBytes`); the grouped
  expert GEMMs keep their `a` active experts, so the device-memory bytes of the step are unchanged.
- `W_layer` = Σ of the layer's resident block weights with everything on the device (all experts, the whole table
  shard). A layer in the `denseLayers` range gets no expert or table transfer on top (it is streamed whole).
- On unified memory host RAM is the device pool: a `'host_ram'` placement is dropped with a note (no op, the
  weights stay counted on the device); only `'nvme'` is a tier there. An NVMe table with `scaleLayout 'pinned_host'`
  keeps its scales in that pool too: they are device bytes (§12 "Memory tiers"). `pinned_host` stays the default for
  `nvme` on unified memory: the rates are unchanged (the scale read is not costed), only the device bytes grow.
- Draft layers (spec-dec) are lowered without a layer index and get no transfer op.

### Storage tiers

`src/sim/engine/host.ts: resolveHost, defaultSsdFor, transferPath`. The tiers and the host link of §4 "Host and
storage tiers", resolved only when something is off-device; each default warns:

- dedicated memory: `host.link ?? hostPcieLink(device.hostInterface)` (validation rejects a device with neither);
  `host.ram ?? DDR5_6000_2CH` (a tier, and the staging hop of NVMe reads);
- `host.ssd ??` the device's own drive on unified systems (Apple M5 → `apple-ssd-m5-pro-max`, other Apple →
  `apple-ssd-m3-m4-max-1tb`, DGX Spark → `dgx-spark-ssd-4tb`, Ryzen AI Max → `nvme-gen4-x4-generic`), else
  `NVME_990_PRO_2TB`. An SSD without `randomRead` warns (no IOPS term).

### Transfer path

`src/sim/engine/transfer.ts: transferTime, tierLatencySec, tierBandwidthBytesPerSec, resolveQueueDepth`;
`engine/opTime.ts` (`case 'transfer'`). One rule for every group, per device and step:

```
R          = Σ_i count_i                                      read requests (expected, real)
bytes_tier = Σ_i count_i·ceil(bytes_i / gran)·gran            every read rounded up to the tier's granularity
α_path     = α_tier + α_link                                  latencies add (α_link = 0 on unified memory)
B_i        = NVMe sustained and bytes_i ≥ blockBytes: min(f_sus·B_seq, B_slot); otherwise B_tier
tier       = Σ_i count_i·ceil(bytes_i/gran)·gran / (B_i/n_stream)
service    = max( (R − 1)·α_tier / QD,                        Little's law, QD outstanding requests
                  R / (IOPS / n_stream),                      SSD request rate (rated SSDs only)
                  tier,                                      tier bandwidth by read class
                  usefulBytes / B_link,                       this device's link (not shared)
                  [NVMe, dedicated] bytes_tier / (B_ram / (2·n_stream)) )   host RAM staging: written and read once
t_path     = α_path + service
```

`α_tier` = host RAM `latencySec` or the SSD's `randomReadLatencySec`; `B_tier` = host RAM `bandwidthBytesPerSec`
or `min(seqRead, linkBandwidth)`; `QD = min(queueDepth ?? maxOutstanding, maxOutstanding)`. Hops are pipelined
across requests and chunks, so the throughput terms take the max. The op's `transferLatencySec = α_path`,
`transferBandwidthSec = service`, `sec = t_path`, `bound = 'transfer'`. Documented approximation: the DMA write
into device memory is not charged to the device's memory bandwidth (≤ `B_link/memBW`, ≈ 3.5% on an RTX 5090).

**SSD read rate** (`OffloadConfig.ssdReadRate`, resolved in `engine/host.ts: transferPath`):
absent or `'datasheet'` preserves the original result bit for bit. `'sustained'` applies the preset's fraction
before the slot cap, to individual reads at least `blockBytes`, whether random experts or streamed whole layers.
Small table rows, host RAM, IOPS, Little's law, links and staging are unchanged. Measurement QD/threads describe
the fraction; they do not replace deployment QD. This is a measured scenario, not a tighter proven bound.
The presets use 1 MiB QD1 / QD8 ratios; they are pessimistic for 10–20 MB expert reads
(2 MB QD3 reaches 91–95% in CDRLabs' 990 PRO/T705/9100 PRO tests). No thermal or engine-overhead model is added.

**Sharing** (`stageNStream, nStream, streamSlots, stageDevicesOnNode`): a tier is shared by its concurrent
consumers on the node. `consumers(s)` = devices of stage s on the node (rank layout of §5) if stage s has a
transfer op from the tier in this phase, else 0; `slots = p > 0 ? |S| : min(m, |S|)` (p the prefetch accuracy)
over the streaming stages S; `n_stream = max(consumers(this stage), Σ of the slots smallest consumers)` (the
optimistic schedule). A stage spanning several nodes takes the largest of its nodes' values.

### Expert hit ratio

`src/sim/engine/transfer.ts: uniqueExperts, uniqueLocalExperts, hitRatio, expertHitsMisses`;
`lower/moe.ts: neededExperts`. U is the same expected-distinct count the expert GEMMs read their weights for (§5
"Expected distinct experts"), so the fetched misses are a subset of the experts the GEMMs touch.

```
U     = E·(1 − (1 − k/E)^T)                    ep = 1: expected distinct experts of T routed tokens
U_dev = E_loc·(1 − (1 − 1/E_loc)^A)           ep > 1: per device, A assignments (with replacement; warning)
C     = f·E_loc
hits  = clamp(U·h, max(0, U + C − E_loc), min(U, C))
M     = U − hits                               expected missed experts (no rounding)
```

`h = hitRatio(model, f)`: `uniform` f (default), `skewed_classic` f^0.55, `skewed_fine_grained` f^0.2, `custom`
h. f = 1 ⇒ M = 0 exactly (no op); f = 0 ⇒ M = U. Decode `T = b_mb`, prefill `T = b_mb·P` (so prefill streams
every non-resident expert once per step). Skewed models with `0 < C < k` warn (an LRU cache thrashes there).

### Expert prefetch

`src/sim/lower/offload.ts: prefetchedFraction`; `engine/pipeline.ts: stageResult`; `engine/transfer.ts:
DEFAULT_PREFETCH_ACCURACY`. Deployment knob `OffloadConfig.prefetchAccuracy` = **p**, dimensionless, in [0, 1]:
the predictor's measured recall one layer ahead, i.e. the share of the needed experts it picks (from the previous
layer's hidden state, with a predicted set the size of the real one). It replaces the former
`SimOptions.offloadOverlap`.

Each transfer op gets a prefetched share **q** (`TransferOp.prefetchedFraction`):

```
q = 0                                           p = 0, every group  (no prefetching at all)
q = max(p, M/(E_loc − C))                        routed experts, p > 0
q = 1                                           dense layers and table rows, p > 0
```

The stage formula of §7 "Stage time and overlap" then gives
`t_s = lat_s + core_s + xfer_s − pbar_s·min(core_s, xfer_s)`, pbar_s being the time-weighted mean of q.
`PhaseResult.transfer` sums the stages (`pipeline.ts: phaseTransfer`).

- **Physical reading.** During the window (the stage's core time) the link copies predicted experts, for
  `min(core_s, xfer_s)` of copy time: it cannot copy longer than the window, nor more than the misses need. A share
  p of those copies are experts the step really needs, so `pbar_s·min(core_s, xfer_s)` of the transfer is hidden.
  Wrong guesses use link time inside the window that would otherwise sit idle, so they cost nothing extra. The rest of
  the misses, `xfer_s − pbar_s·min(core_s, xfer_s)`, are copied after the router, on the critical path. Never
  issuing a prefetch longer than the window avoids HOBBIT's non-cancellable-copy slowdown. Bounds:
  `lat_s + max(core_s, xfer_s) ≤ t_s ≤ t_core,s + xfer_s`.
- **q for experts = max(p, M/(E_loc − C)).** The engine takes the better of two strategies: prefetch its
  predictions, of which a share p are needed; or, when nearly every non-resident expert is needed, copy them
  unconditionally, of which a share `M/(E_loc − C)` are needed. At batch 1 `M ≪ E_loc − C`, so q = p exactly, the
  measured recall (128 experts, top-8, nothing resident: 8/128 = 0.0625 < p; Mixtral top-2 of 8: 0.25 < 0.84). In
  prefill `M = E_loc − C`, so q = 1: every expert is needed and prediction is trivially exact (HOBBIT arXiv
  2411.01433 §5; Madan et al. arXiv 2603.19289 §5.2). In between, e.g. 32 decode tokens on 128 experts top-8, the
  unconditional copy wins once `M/(E_loc − C)` passes p (0.873 there). Taking the better strategy is the upper
  bound; no chance overlap of wrong guesses is added to the recall.
- **Endpoint rule.** Dense layers and table rows are deterministic transfers: every streamed layer is needed, and
  table rows follow from the token ids. An engine that prefetches at all prefetches these, so q = 1 for any p > 0.
  p = 0 means no prefetching at all, so q = 0 everywhere. Both former modes stay reachable bit for bit: **p = 1 ≡
  `'ideal'`** and **p = 0 ≡ `'none'`** (`pipeline.test.ts`, `e2e-offload.test.ts`). The price is a jump at p = 0:
  p → 0⁺ still prefetches dense layers and tables.
- **Window.** The prefetch window is the whole stage's core time, not one layer's: the same relaxation as the
  former `'ideal'`. A per-layer form would be more pessimistic at p = 1 whenever layers differ, and would no longer
  reproduce `'ideal'`.
- **Default p = 0.8 (estimated).** The middle of what a training-free, exact-output predictor with the same set
  size k achieves one layer ahead (the next layer's gate applied to the current hidden state):
  - Mixtral 8×7B: 0.84 (DAOP arXiv 2501.10375 Fig. 5), ≈0.9 (AdapMoE arXiv 2408.10284 §6), ≈0.85 (Eliseev & Mazur
    arXiv 2312.17238 Fig. 2);
  - Qwen1.5-MoE: 0.79 (Fate arXiv 2502.12224 §4.2);
  - Qwen2-57B-A14B: 0.67 (ProMoE arXiv 2410.22134 §4.2);
  - Qwen3-30B-A3B: ≈0.9 beyond the first two layers (Madan et al. arXiv 2603.19289 §3.3).

  Learned predictors reach 0.85–0.95 (ProMoE 84.7%; SeqMoE arXiv 2609.12978 §9.4, 86–91% fetching k + 3), and
  0.97–0.99 only by fetching extra experts. Papers define accuracy differently (top-1, recall at k).

- p > 0 also lets every streaming stage prefetch at once (`slots` in "Transfer path") and double-buffers an op with
  q > 0 ("Memory tiers").
- UI: Hardware → Advanced → "Prefetch accuracy %"; URL key `opf`. An old URL's `oo=ideal` / `oo=none` maps to
  `opf=1` / `opf=0` when something is offloaded.

**Is the former `'ideal'` default realistic?** Not quite. What real systems do:

- **llama.cpp** does not predict experts in decode. `--n-cpu-moe` / `-cmoe` computes host-resident experts on the
  CPU (`common/arg.cpp`); host weights are copied to the GPU only for batches of ≥ 32 tokens
  (`GGML_OP_OFFLOAD_MIN_BATCH`). The GPU expert cache (https://github.com/ggml-org/llama.cpp/pull/29887) uploads
  misses on demand, after the router. A one-split-ahead prefetch
  (https://github.com/ggml-org/llama.cpp/pull/28414) was prefill-only and closed unmerged ("decode is flat").
- **ktransformers** (SOSP'25) and **Fiddler** (ICLR'25, arXiv 2402.07033) compute cold experts on the CPU, so only
  activations cross PCIe (the designed-but-rejected `mode: 'cpu_compute'` here).
- **Prefetching systems** (Mixtral-offloading arXiv 2312.17238 §3.2, MoE-Infinity arXiv 2401.14361, ProMoE,
  HOBBIT, AdapMoE, DAOP, Fate, ExpertFlow arXiv 2410.17954, SeqMoE) predict 0.67–0.9 of the experts one layer ahead
  with the same k. Pre-gated MoE (ISCA'24, arXiv 2308.12066) knows the routing a layer ahead only by retraining the
  gate, so it does not apply to off-the-shelf checkpoints.

How much transfer time is hidden: little when the copy dominates, whatever the accuracy, because one layer's compute
is all a prefetch can hide behind. Madan et al. (Qwen3-30B-A3B, A6000, PCIe 4.0): the copy is 84–88% of the token
time and prefetching saves 9–14% (5–8% on A100 / GH200), matching their Σ min(t_copy, t_compute) bound. SeqMoE
(RTX 4090): one layer's compute covers 1.39 of the 8 experts needed. HOBBIT (Mixtral, RTX 4090): 1.01× in FP16.
MoE-Infinity reports that inaccurate prefetching can be slower than loading on demand. The `min(core_s, xfer_s)`
term already caps the gain there, so p = 0.8 instead of 1 costs a few percent. It matters more when compute is at
least as long as the copy: many experts resident, or a fast link (GH200 C2C, unified memory).

### Memory tiers

`src/sim/engine/memory.ts: stageWeightBytes, stageStagingBytes, stageTierBytes, tierMemory, stageLookupTableBytes`;
`lower/moe.ts: moeResidentWeightBytes, moeLocalExpertBytes`; `lower/ngram.ts: ngramResidentWeightBytes,
ngramTableShardBytes`.

```
W_s        = resident weights: offloaded MoE layer C·W_expert (+ router, latent projections, shared); dense-layer range 0;
             off-device table: the non-table part  + [unified ∧ pinned_host] ceil(rows/tp)·scaleBytes
             (lower/ngram.ts: ngramResidentWeightBytes, pinnedScaleBytes)
staging_s  = max_{phase, transfer op} (q > 0 ? 2 : 1)·buffer          (double buffer for a prefetched op),
             buffer = experts / layers: Σ ceil(count)·bytes;  tables: usefulBytes
total_s    = W_s + B_r·state_s + act_s + staging_s ≤ capacity
tier bytes = max over nodes of Σ_{devices of the node} (full groups: E_loc·W_expert per offloaded layer, W_layer,
             table shards; NVMe tables with scaleLayout 'pinned_host' put ceil(rows/tp)·scaleBytes on host RAM —
             dedicated memory only; unified memory has no host_ram tier, the scales are in W_s)
fits       = ∀ s total_s ≤ capacity  ∧  ∀ tiers bytes ≤ capacityBytes
```

`MemoryReport.tiers` lists each tier with bytes; `DeviceMemory.lookupTableBytes` reports the on-device table
shards (included in `weightBytes`). DP replicas on one node are counted once each (no shared host copy).

### n-gram tables off-device

`lower/ngram.ts: tableFetchOp, tableRowBytes`. A row of `d` elements: `valueBytes = d·(elementBits +
extraBits)/8`, `scaleBytes = d·blockScaleBits/8`, `rowBytes = d·bpe = valueBytes + scaleBytes` (Engram mxfp8
256 + 8 = 264 B; PLE bf16 320 + 0). Rows per device `R = T·g/tp` (the table is row-parallel). Reads by
`scaleLayout` (default `'separate'` on host RAM, `'pinned_host'` on NVMe): `interleaved` `[{ R, rowBytes }]`;
`separate` `[{ R, valueBytes }, { R, scaleBytes }]`; `pinned_host` `[{ R, valueBytes }]` (the scales live in host
RAM, counted in its capacity; their read is not costed; on unified memory they stay in the unified pool and count in
the device's `W_s` instead). A table without scales reads `[{ R, rowBytes }]` in any
layout. `usefulBytes = R·rowBytes` whatever the layout. The existing gather op stays (device-memory side).

## 13. Speculative decoding

Deployment knob `SimInput.deployment.speculative` (design: `docs/design/offload-specdec-latency.md` §3). Off by
default: absent or `{}` lowers the plain decode (`qLen 1`), appends no `draft.*` layer and gives bit-identical
results. With it on, the result is the hardware/math bound given the stated acceptance: the engine adds the GEMM,
attention and state work the draft and verify passes physically need, nothing else (no scheduler, tree, or
confidence policy). Rejected with pp > 1 (`validate.ts: SPEC_DEC_PP_ERROR`). Works with ep > 1: the drafter's MoE
layers lower with the same parallel config, and `deploymentErrors` checks their expert count against ep like the
target's (it may differ: DeepSeek-V4.1 DSpark has 128 experts against 384; `engine/ep.test.ts`). Worked example W2
and U-8, U-14, U-18 are tests in `engine/e2e-specdec.test.ts`.

### Draft spec

`src/sim/ir/model.ts: DraftSpec, MtpDraftSpec, BlockDraftSpec`; `models/hf/drafter.ts` (a drafter repo's
`config.json` → `BlockDraftSpec`); the HF family mappers keep their MTP module as `model.draft` (never counted in
`countParams` or the fit unless spec-dec is on). `method 'mtp'` uses `model.draft`; `'dflash2'` / `'dspark'` use
`SpecDecodeConfig.drafter.spec` (a repo's config or the labelled hypothetical). `engine/specdec.ts:
resolveSpeculative` builds the `SpeculativeLowering` (`verifyTokens = k + 1`, `contextRows = m_ctx`,
`stateSnapshots = k + 1`) and warns when a single-step MTP head (`maxReliableDepth`) is drafted past its depth.

### Verify

`src/sim/engine/simulate.ts` (the verify context); `lower/common.ts: stateSnapshots`; `lower/linearAttention.ts`.
The decode step is the target lowered with `qLen = k + 1`, `kvLen = N + k + 1` (every GEMM `m = b·(k+1)`, the
head over every position, attention causal over `past = N`, k + 1 KV rows written; rejected rows are overwritten
next step and not charged back). MoE routing and the offload path use `T = b·(k+1)` with no new rule. Recurrent and
conv state ops in `'step'` mode write `k + 1` per-position snapshots (`RecurrentOp.stateSnapshots`,
`stateWriteBytes·(k+1)`; the prior state is read once) so the state can roll back to the accepted prefix.

### Draft ops

`src/sim/lower/draft.ts: lowerDraftDecode`; appended to the LAST stage after the head as layers `draft.<i>`
(`simulate.ts: withDraftLayers`), so stage sums, overlap and collectives apply unchanged. `m_ctx = ⌈E[τ]⌉`
(`specdec.ts: draftContextRows`) newly committed tokens per sequence feed the drafter.

- **MTP**: a catch-up pass of `qLen = m_ctx` (shared `embed`, `enorm`, `hnorm`, `eh_proj` (m b·m_ctx, n h, k 2h),
  the module's layer at `kvLen N + k + 1`, `final_norm` and the shared `lm_head` over the last position, m b), then
  `k − 1` one-row passes. The shared embedding and head are read per pass.
- **Block drafter** (DFlash 2, DSpark): one parallel pass: `fc` + `hidden_norm` over the m_ctx context features,
  the block embedding, per layer the attention over `min(N, W) + m_ctx + B` keys keeping only the m_ctx context rows
  (`AttentionOp.kvWriteTokens`), K/V projections as one GEMM over context + block rows, the MLP, DFlash 2's two
  dynamic convs; then the shared head over the k draft positions and DFlash 2's candidate selector or DSpark's
  Markov head. Sequential walks (selector, Markov sampling) are latency, not costed.

### Draft-cache prefill

`src/sim/lower/draft.ts: lowerDraftPrefill`, appended to the prefill's last stage. MTP: the module's ordinary
forward over the prompt (writing its KV) and one head row per sequence. Block drafter: per layer only the context
K/V projections over `b·P` rows, their q/k-norm and RoPE, and a write-only cache op (`ElementwiseOp fn
'kv_cache_write'`, `kvWriteBytes = b·min(P, W)·2·KV·hd·bpe(kv)`); plus `fc` and `hidden_norm` once. No attention,
MLP, head or selector in prefill.

### Accepted tokens

`src/sim/engine/specdec.ts: expectedTokens, presetAcceptance, ACCEPTANCE_PRESETS`; `engine/pipeline.ts:
phaseResult` (`PhaseShape.tokensPerStep`); `simulate.ts: speculativeSummary`.

```
i.i.d.         E[τ] = (1 − α^(k+1)) / (1 − α)                α ∈ [0, 1)
per-position   E[τ] = 1 + Σ_{i=1..k} Π_{j=1..i} α_j            α_j ∈ [0, 1]; a short vector repeats its last value
cycle          = the decode cycle rule of §7 over the verify stages with draft.* appended
tokens/s       = dp · B_r · E[τ] / cycle  @ N;   per sequence E[τ] / cycle;   latency (per token) = cycle / E[τ]
speedup        = tokens/s / tokens/s of the same input with speculative decoding off
```

`PhaseResult.speculative = { draftSec, verifySec, expectedTokensPerStep, speedup }`: `draftSec` / `verifySec` are
the Σ op times of the `draft.*` layers and of every other layer (without overlap). With `tokensPerStep` absent the
rate expressions are today's, bit for bit.

### Acceptance defaults

`src/sim/engine/specdec.ts: ACCEPTANCE_PRESETS, AcceptancePreset, AcceptanceMeasurement, acceptanceRuns,
iidAlphaForTokens, FIT_TOLERANCE_ALPHA, fittedDraftLengths, pooledDraftLengths, presetDraftLengthExtrapolation`; the UI
(`ui/components/SpecDecodeSection.tsx`) fills α from the preset unless the user types one. Each reported τ is stored
with its source URL and location, the method version actually measured, the target (and thinking mode), the drafter
where it is not the method's own release, the benchmark, k and temperature. A preset's `measurements` are the runs
behind α; `otherReports` are cited for context only and never count as a fitted k.

```
run            = one source table at one k; τ_run = mean of its τ over the use case's benchmarks and targets
α_iid(τ, k)    = the α with (1 − α^(k+1)) / (1 − α) = τ                   bisection; E[τ] rises with α
reproduces     |α − α_iid(τ_run, k_run)| ≤ 0.006                      α quoted to 0.01, τ to 0.01
measured       α reproduces every run in `measurements`; no run in `otherReports` reproduces it
borrowed       the same, with runs of another version of the method (a proxy)
pooled         α lies strictly between its runs and reproduces none of them
carried_over   α = α(other use case) + offset                       no source measured this method × use case
fitted k       = { k_run } of the reproduced runs (carried_over: the source use case's; pooled: none)
UI             k ∉ fitted k ⇒ "⚠ fitted at k = …";  pooled ⇒ "pooled from k = …" (its runs' k), "⚠ pooled from …" at any other k
```

The i.i.d. model carries one α across every k, but real per-position acceptance can fall (or rise) along the draft:
an α fitted at k = 7 to an assumed curve `0.92 − 0.06·(j − 1)` gives E[τ] 5.36 at k = 15 against the curve's 4.53
(+18%), and 3.05 against 3.34 at k = 3 (−9%) (practicality review F10; illustrative). So the UI marks the preset α
when k is none of its fitted k (`presetDraftLengthExtrapolation`): a cited report at that k does not clear it (DSpark
prose 0.76 at k = 15 gives E[τ] 4.12, Inco's k 15 retrain measures 4.77). A pooled preset follows the same rule with
its pooled runs' k: at any other k it is extrapolated too. No mark for a user α, or for a single-step MTP head past
its depth, where the collapsing tail replaces the i.i.d. α and the engine warns. The mark can appear at the default k:
MTP multi-step prose and math (default k 3, fitted at 15 and 7), DeepSeek-V4.1's in-checkpoint DSpark (k capped at 5
by `blockSize − 1`), DFlash 2 agentic (fitted at 15). Labels: "prose" is MT-Bench, "code" HumanEval + MBPP, "math"
GSM8K + MATH-500. Runs are given as τ (α_iid).

| method / use case                          | α    | basis                       | fitted k     | runs α is fitted to (or pooled from)                                                                                                  | cited, not fitted                                                                                                                                               |
| ------------------------------------------ | ---- | --------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MTP multi-step / prose                     | 0.75 | measured                    | 15           | blog Qwen3.5-4B MT-Bench 3.90 (.746)                                                                                                  | card 3.74 (.763); [llama.cpp PR 27210](https://github.com/ggml-org/llama.cpp/pull/27210#issuecomment-5385811671) C1 greedy k 3, τ = 1 + 3 × 0.557 = 2.67 (.735) |
| MTP multi-step / code                      | 0.82 | pooled, **mixed T**         | — (3, 7, 15) | card HE/MBPP 3.95 (.782); blog 4.50 (.782); PR 27210 greedy k 3 3.35 (.883)                                                           | —                                                                                                                                                               |
| MTP multi-step / math                      | 0.85 | measured                    | 7            | card GSM8K/MATH 4.87 (.851)                                                                                                           | blog 4.91 (.802)                                                                                                                                                |
| MTP multi-step / reasoning                 | 0.68 | measured (greedy)           | 3            | PR 27210 C1, thinking on, 2.47 (.684)                                                                                                 | —                                                                                                                                                               |
| MTP multi-step / agentic                   | 0.75 | carried over (prose)        | 15           | —                                                                                                                                     | —                                                                                                                                                               |
| MTP single-step / prose                    | 0.85 | measured                    | 1            | [DeepSeek-V3 §5.4.3](https://arxiv.org/abs/2412.19437), all topics, 85–90% → 1.85 (.85)                                               | [FastMTP Table 1](https://arxiv.org/abs/2509.18362) MiMo-RL-7B vanilla greedy MT 1.72 (.72)                                                                     |
| MTP single-step / code, reasoning, agentic | 0.85 | carried over (prose)        | 1            | —                                                                                                                                     | code: FastMTP 1.73                                                                                                                                              |
| MTP single-step / math                     | 0.87 | carried over (prose + 0.02) | 1            | —                                                                                                                                     | FastMTP 1.83                                                                                                                                                    |
| DFlash 2 / prose                           | 0.79 | measured                    | 7            | card 4.10 (.795)                                                                                                                      | [Muse card](https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2) 4.42 (.778); blog 5.20 (.815)                                                               |
| DFlash 2 / code                            | 0.83 | measured                    | 7, 15        | card HE/MBPP 4.59 (.832); Muse 5.48 (.826)                                                                                            | blog 5.85 (.839)                                                                                                                                                |
| DFlash 2 / math                            | 0.87 | pooled                      | — (7, 15)    | card GSM8K/MATH 5.37 (.882); Muse 6.57 (.862); blog 6.48 (.859)                                                                       | —                                                                                                                                                               |
| DFlash 2 / reasoning                       | 0.81 | carried over (prose + 0.02) | 7            | —                                                                                                                                     | —                                                                                                                                                               |
| DFlash 2 / agentic                         | 0.65 | **borrowed: DFlash v1**     | 15           | [DDTree Table 1](https://arxiv.org/abs/2604.12989) Qwen3-8B, z-lab DFlash v1, SWE-bench Lite, T 1, 2.82 (.646)                        | no DFlash 2 agentic data                                                                                                                                        |
| DSpark / prose                             | 0.76 | measured                    | 7            | [DSpark Table 1](https://arxiv.org/abs/2607.05147), DeepSeek-trained, non-thinking Qwen3-4B/8B/14B, T 1: 3.64/3.72/3.70 → 3.69 (.758) | blog Inco retrain 4.77 (.796); community Qwen3.8-27B thinking k 7 3.01 (.684), Muse k 15 3.59 (.723)                                                            |
| DSpark / code                              | 0.84 | pooled                      | — (7, 15)    | paper HE/MBPP 5.31 (.879); blog Inco retrain 5.38 (.822)                                                                              | community k 7 3.41 (.730), k 15 4.18 (.764)                                                                                                                     |
| DSpark / math                              | 0.88 | pooled                      | — (7, 15)    | paper GSM8K/MATH 5.95 (.913); blog Inco retrain 5.95 (.843)                                                                           | community k 7 4.14 (.798), k 15 5.23 (.816)                                                                                                                     |
| DSpark / reasoning                         | 0.78 | carried over (prose + 0.02) | 7            | —                                                                                                                                     | —                                                                                                                                                               |
| DSpark / agentic                           | 0.76 | carried over (prose)        | 7            | —                                                                                                                                     | —                                                                                                                                                               |

"card" is the [Qwen3.8-27B DFlash 2 card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (thinking, xhigh effort,
k 7, T 1), "Muse" the Muse-Glimmer-30B card (high reasoning strength, k 15, T 1), "blog" the
[DFlash 2 blog](https://inco.ai/blog/dflash2/)'s matched Qwen3.5-4B comparison, Table 3 (thinking enabled, T 1.0, top-p
0.95, top-k 20, presence penalty 1.5, k 15). DSpark prose 0.76 is fitted to the paper alone (k = 7, non-thinking Qwen3)
and stays: it is the method authors' controlled measurement at the default k, and the other reports straddle it (Inco's
retrain higher, the community thinking retrain lower: E[τ](0.76, 7) = 3.70 is +23% over its 3.01).

**Per-position data.** The blog's Figure 5 gives per-position conditional acceptance on MATH-500 (Qwen3.5-4B, same
sampling, 15 positions) for MTP (84.6% falling to about 78%), DFlash v1 (88.4%, then 78–81%), DSpark (87.2% to 79.9%)
and DFlash 2 (flat near 85–86%); their products give E[τ] 4.87 / 5.14 / 5.92 / 6.46 against Table 3's 5.04 / 5.42 /
6.20 / 6.76 (pooled per-position rates vs per-request means). The blog's selector table gives DFlash Recall@1 per position for k = 7, and
DSpark Figure 2 per-position curves for Qwen3-4B. These are recorded, not used: the presets stay i.i.d., and
`expectedTokens` accepts a `per_position` vector for a user who wants one. The α depends on target, temperature and
thinking mode; no universal measured α exists.

### Draft memory

`src/sim/lower/draft.ts: draftResidentWeightBytes, draftSequenceStateBytes`; `engine/memory.ts: memoryReport,
stageSnapshotStateBytesPerSeq`.

```
context            = max(P, N + k + 1)                         the verify appends k + 1 rows
draft_s            = [last stage] W_draft + B_r · S_draft(context)
                     W_draft: module layers / drafter layers, eh_proj or fc, norms, conv kernels, selector codebooks,
                     Markov W1/W2; the shared embedding and LM head are not counted again (pp = 1)
                     S_draft: MTP the module's layer state over the full context; block drafter L·min(ctx, W)·2·KV·hd·bpe(kv)
specTransient_s    = B_r · (k + 1) · snap_s                    snap_s = recurrent + conv state of the stage's
                                                               linear-attention, short-conv, n-gram and GQA-conv blocks
total_s            = W_s + B_r·state_s + act_s + staging_s + draft_s + specTransient_s
maxSequences       = floor(min_s (capacity − W_s − act_s − staging_s − W_draft) / (state_s + S_draft + (k+1)·snap_s))
```

### Spec-dec × offload

The verify's `T = b·(k+1)` gives `U(b·(k+1))` needed experts per offloaded layer (E 128, k_moe 8: `U(2) = 15.5`
at k = 1, 1.94× W1's decode misses at f = .5). The drafter's own MoE layers follow the routed-expert placement only
with `includeDraft: true`: they are lowered with the placement's range widened to every drafter layer (`draft.ts:
draftContext`), get `draft.moe.expert_fetch` ops on the last stage, keep `C = f·E_loc` experts resident in
`draft_s`, and add their full expert set to the tier (`memory.ts: stageTierBytes`).
