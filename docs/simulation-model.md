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
- No inference-engine knobs (§9).

**Units: SI decimal.** 1 GB = 1e9 B, 1 TB/s = 1e12 B/s, 1 TFLOP/s = 1e12 FLOP/s (`src/sim/hw/units.ts`:
`tflops`, `gbPerSec`, `tbPerSec`, `gb`, `mb`, `usec`). Times are seconds, rates tokens/s, sizes bytes or
elements as the field name says.

| Quantity | Convention |
|---|---|
| Matmul peak (`MatmulPeak.flopsPerSec`) | per device; keyed by (input format, accumulator, sparsity); FLOP/s (OP/s for integers) |
| 2:4 sparse peak | dense-equivalent FLOP/s (vendor convention); op FLOP counts are never halved |
| Memory bandwidth (`MemorySpec.bandwidthBytesPerSec`) | per device, read + write combined, as vendors quote it |
| Link bandwidth (`LinkSpec.bandwidthPerDirectionBytesPerSec`) | per device, **per direction** (unidirectional injection rate), never the bidirectional sum |
| Link latency (`latencySec`) | α per message / algorithm step |
| `CollectiveOp.messageBytes` | per device: all_reduce = the full buffer; all_gather = the full OUTPUT; reduce_scatter = the full INPUT; all_to_all = the total sent incl. the self share; send_recv = bytes sent |
| Op bytes (`ByteBreakdown`) | per device, per microbatch step |
| `tokensPerSec` | aggregate over DP replicas; `perSequenceTokensPerSec` is what one sequence sees |

---

## 2. Model IR

A model is described by an **architecture spec** (`src/sim/ir/model.ts`: `ModelSpec`, `LayerSpec`,
`BlockSpec`), with logical (unsharded) dimensions and no tokens or parallelism. Layers are heterogeneous: each
`LayerSpec` is its own ordered list of blocks (`norm`, `residual`, `gqa_attention`, `mla_attention`,
`csa_attention`, `linear_attention`, `mlp`, `moe`, `hc_pre`, `hc_post`, `ngram_embedding`, `short_conv`), which
covers hybrid linear/full attention stacks, dense first layers before MoE, and alternating sliding-window layers.
The builders (`src/sim/ir/builders.ts`: `decoderLayer`, `decoderModel`, `repeatLayers`, `hyperCollapse`) produce
pre-norm decoder layers from `MixerParams` + `FfnParams` (+ optional `residual`, `ngram`, `shortConv`):

- plain residual: `[ngram] → norm → mixer → [short_conv] → [post-norm] → residual → norm → ffn → [short_conv] →
  [post-norm] → residual`;
- hyper-connections (`residual.kind: 'hyper'`, N streams): each `norm … residual` pair becomes
  `hc_pre → [norm if subNorms] … hc_post`. DeepSeek-V4.1 and GLM-5.3-Flash keep the sub-norms, Qwen4Exp does not.
- `decoderModel({ beforeHead })` appends blocks (the final hyper-connection collapse) to the LAST layer, so they are
  lowered and pipeline-staged as part of it. `finalNorm: 'none'` (Qwen4Exp) has no final norm.
- `ModelSpec.residualStreams` N (default 1): each token carries `N·hidden` residual elements; the PP activation
  send is `T·N·h·bpe(act)` (`lower.ts: lowerModel`).

**Cross-layer sharing** names an earlier layer by index: `MlaAttentionBlock.kvCacheFromLayer` (Mistral Large 4:
read another layer's latent; no `kv_a`, no cache), `CsaGlobalSpec.kvCacheFromLayer` (DeepSeek-V4.1: read a
source's compressed KV; no compressor), `SparseSelection {kind: 'reuse'}` (GLM-5.3, Mistral, DeepSeek "Reuse": take
the nearest earlier indexer's top-k; no indexer ops, no key cache, no indexer weights) and `IndexerSpec.ownsKeys:
false` (DeepSeek "Reindex": read an earlier indexer's keys). `src/sim/ir/validateModel.ts: modelSpecErrors` checks
that each reference points to a compatible earlier layer, that `gateRank` is set iff the variant is `kda`, and that
hyper-connection and n-gram widths equal `residualStreams·hidden`.

New block kinds (lowered in `src/sim/lower/{csa,hyperConnection,ngram,shortConv}.ts`, §5):

- `csa_attention` (DeepSeek-V4.1): one cached `headDim` vector per token used as K and V; every layer attends its
  own `windowTokens` raw window, plus (with `global`) `topK` entries of a KV compressed `compressRatio`:1; low-rank
  q (`qLoraRank`) and grouped low-rank output (`outGroups` × `outLoraRank`).
- `hc_pre` / `hc_post`: hyper-connections over N streams (Qwen "Gated Residual", DeepSeek/GLM mHC). `hc_pre`
  collapses the streams through its `projections` (and an optional γ over N·D, Sinkhorn iterations); `hc_post`
  writes the sublayer output back, mixing the streams (`mixesStreams`, mHC) or not (Qwen).
- `ngram_embedding` (Qwen PLE, DeepSeek Engram): a hashed lookup table of `tableRows × rowDim` (memory
  capacity), `rowsPerToken` rows gathered per token, then `projections`, a gate and an optional dilated conv,
  written into the `outputDim = N·D` residual.
- `short_conv` (Inkling): `x + conv(x)`, a depthwise causal conv of `kernel` taps on a branch output.
- New fields: `IndexerSpec.{keyPoolTokens, keyPoolGate, keyInputDim, ownsKeys, maxScoredKeys, headWeights}` (GLM
  k-pool, Qwen QSA, MiniMax MSA, DeepSeek CSA indexers); `GqaAttentionBlock.{relativeBias, kvConvKernel}`
  (Inkling); `LinearAttentionBlock.variant 'kda'` with `gateRank` (GLM-5.3-Flash Kimi Delta Attention).

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
  - `modules` (per `WeightModule` class): `{ weights, gemmInput? }`. Classes: `attention` (GQA q/k/v/o, MLA q path
    and o_proj, CSA q path, window `wkv` and output), `attention_kv_down` (MLA `kv_a`, CSA compressor),
    `attention_kv_up` (MLA `kv_b` and its absorbed slices `q_absorb`, `v_absorb`), `linear_attention`, `indexer`
    (query `wq`/`wq_b`), `indexer_key` (`wk`, k-pool gate), `indexer_head_weights` (`weights_proj`), `dense_mlp`,
    `shared_experts`, `routed_experts`, `residual_mixer`, `ngram`.
  - Resolution (`WEIGHT_MODULE_FALLBACK`): an absent class uses its parent's entry if listed
    (`attention_kv_down`, `attention_kv_up` → `attention`; `indexer_key`, `indexer_head_weights` → `indexer`), else
    `{ weights }`; `residual_mixer` defaults to `{ dtype: highPrecision }`. This expresses e.g. GLM-5.3-Flash FP8
    (`kv_b_proj` BF16, other MLA projections FP8) and Mistral Large 4 FP8 (`kv_a`, `kv_b`, indexer `wk`,
    `weights_proj` BF16).
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

| Scheme | Storage dtype | Overhead (bits/element) | Reader |
|---|---|---|---|
| fp8 block 128×128, fp32 scales (DeepSeek-V3 / GLM / Qwen FP8, ModelOpt `FP8_PB_WO`, `FP8_BLOCK_SCALES`) | `fp8_e4m3` | 32/16384 ≈ 0.00195 | `fp8.ts`, `modelopt.ts`, `compressedTensors.ts`, `quark.ts` |
| fp8 block 32×32, ue8m0 scales (DeepSeek-V4.1, Quark `float8_e8m0fnu`) | `fp8_e4m3` | 8/1024 ≈ 0.0078 | `fp8.ts`, `quark.ts` |
| fp8 per-tensor / per-channel (`FP8`, `float-quantized` channel) | `fp8_e4m3` | ≈ 0 | `modelopt.ts`, `compressedTensors.ts` |
| mxfp8 (block 1×32 e8m0, `mxfp8-quantized`, ModelOpt `MXFP8`) | `mxfp8` | in the dtype: 8/32 = 0.25 | `fp8.ts`, `compressedTensors.ts`, `modelopt.ts`, `mlx.ts` |
| mxfp4 (e2m1, g32 e8m0; Quark fp4 g32 e8m0, `mxfp4-pack-quantized`, DeepSeek `expert_dtype: fp4`) | `mxfp4` | in the dtype: 8/32 = 0.25 | `fp8.ts`, `compressedTensors.ts`, `modelopt.ts`, `quark.ts`, `gptq.ts`, `mlx.ts` |
| nvfp4 (e2m1, g16 e4m3 + an fp32 per-tensor `weight_scale_2`) | `nvfp4` | in the dtype: 8/16 = 0.5; the per-tensor scale ≈ 0 | `modelopt.ts`, `compressedTensors.ts`, `quark.ts`, `gptq.ts`, `mlx.ts` |
| GPTQ / AWQ / AutoRound int`b` g`g` (fp16 scale + packed `b`-bit `qzeros`, also when symmetric) | `int4` / `int8` | (16 + b)/g: g128 0.15625, g32 0.625; `group_size -1` ≈ 0 | `gptq.ts` |
| compressed-tensors int`b` group (`scale_dtype`, default bf16; zero point `zp_dtype` only when `symmetric: false`) | `int4` / `int8` | (16 + z)/g: sym g128 0.125, asym g32 int8 zp 0.75 | `compressedTensors.ts` |
| ModelOpt `INT4_AWQ` / `W4A16_AWQ` / `W4A8_AWQ` | `int4` | 16/g | `modelopt.ts` |
| Quark int4 / uint4 per_group g | `int4` | (16 + z)/g, z = 4 when asymmetric or uint | `quark.ts` |
| bitsandbytes NF4 / FP4, absmax per 64 (codes stored as `int4`) | `int4` | fp32 absmax 32/64 = 0.5; double quant 8/64 + 32/(64·256) ≈ 0.127 | `bnb.ts` |
| bitsandbytes int8 (LLM.int8(), per-row fp32 scale; fp16 outlier columns not modeled) | `int8` | ≈ 0 | `bnb.ts` |
| MLX affine `b`-bit g`g` (16-bit scale + 16-bit bias per group) | `int4` / `int8` (widths without a dtype round up, noted) | 32/g: g64 0.5 | `mlx.ts` |

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
  o `H·dv·h`, + indexer (q input rq or h). With `kvCacheFromLayer` the `h·(rkv+dr) + rkv` term is dropped.
- csa: q `h·rq + rq + rq·H·hd`, kv `h·hd + hd`, out `G·R_o·(H·hd/G) + G·R_o·h`; compressor (owner only)
  `2·h·hd + hd` for r > 1, `h·hd + hd` for r = 1; + the global selection's indexer (q input rq). DeepSeek-V4.1:
  126.6 M per layer without compressor and indexer.
- indexer: `qIn·H_I·hd_I` + [ownsKeys] `kIn·hd_I + 2·hd_I` (key norm γ, β) + [keyPoolGate] `h·hd_I + p·hd_I`
  + [headWeights] `h·H_I`. A `'reuse'` selection has 0.
- linear_attention: in_proj `h·(2·qkH·dk + vH·dv·(gate?2:1) + 2·vH)`, conv `(2·qkH·dk + vH·dv)·K`, `2·vH`
  (A_log, dt_bias), `dv` (out norm), out_proj `vH·dv·h`.
- kda: `h·(2·qkH·dk + vH·dv)` (q, k, v) + `h·vH` (β) + `h·r + r·vH·dk` (forget gate) + `h·r + r·vH·dv` (output
  gate) + conv `(2·qkH·dk + vH·dv)·K` + `vH` (A_log) + `vH·dk` (dt_bias) + `dv` (o_norm) + `vH·dv·h` (o_proj).
  GLM-5.3-Flash: q, k, v, f, g, b projections 104.07 M per layer.
- mlp `(gated?3:2)·h·f`; moe `E·(gated?3:2)·h·fe + E·h + S·(gated?3:2)·h·fs`; active MoE uses `k` instead of `E`
  for the routed experts.
- hc_pre: [normGamma] `N·D` + `Σ in·out` + `vectorParams`; hc_post 0. Qwen3.8-Flash-Next 6,604,800 per sublayer
  (634 M in 48 layers, final collapse 6.56 M); DeepSeek-V4.1 `24·20480 + 27` per sublayer (39.3 M); GLM-5.3-Flash
  0.79 M per layer.
- ngram_embedding: `tableRows·rowDim + Σ in·out + [conv] outputDim·K + vectorParams`. **Active** params drop the
  table and add the gathered rows `rowsPerToken·rowDim`: the table counts toward weight memory and total
  params, not active params.
- short_conv: `dim·K`.
- Final norm: `h` (rmsnorm), `2·h` (layernorm), 0 (`'none'`).

### First-class vs expressed

| Feature | How it is expressed | New op kind? |
|---|---|---|
| Sliding window | `KvSelection {window}` on `AttentionOp` | no |
| Top-k sparse attention | `KvSelection {topk}` + `IndexerOp` (no V, own key cache, emits indices) | **yes: indexer** |
| MLA | GEMMs (incl. per-head absorb GEMMs) + `AttentionOp` with `kvHeads = 1`, qk ≠ v dims, latent `kvRead`/`kvWrite`; block `mla_attention` | no |
| GatedDeltaNet / Mamba2 / linear attention | `RecurrentOp` (fixed-size state, chunked vs step) + `causal_conv1d`, `gated_rmsnorm`; block `linear_attention` | **yes: recurrent** |
| Gated attention output | `sigmoid_mul` + a wider q projection | no |
| MoE shared experts, dense first layers | `MoeBlock.sharedExperts`; heterogeneous `layers[]` | no |
| Attention sinks (gpt-oss) | ignored (negligible cost) | no |
| DeepSeek-V4.1 CSA (K = V, window + compressed top-k, grouped low-rank output) | block `csa_attention`; one `AttentionOp` with `secondaryKv` (§5 "CSA") | no |
| Hyper-connections (Qwen Gated Residual, mHC) | blocks `hc_pre` / `hc_post` → GEMMs + `hc_collapse`, `hc_expand`, `sinkhorn`; `ModelSpec.residualStreams` | no |
| n-gram / Engram tables | block `ngram_embedding` → `EmbeddingOp` gather + GEMMs + `ngram_gate`; `QuantConfig.lookupTables` | no |
| Short convs on branch outputs / K, V (Inkling) | block `short_conv`; `GqaAttentionBlock.kvConvKernel` → `causal_conv1d` | no |
| Relative-position bias (Inkling) | `GqaAttentionBlock.relativeBias` → two GEMMs | no |
| KDA (low-rank gates, per-channel decay) | `LinearAttentionBlock.variant 'kda'` + `gateRank` → GEMMs + `RecurrentOp` (`gateElemsPerTokenPerHead`) | no |
| Pooled / gated / foreign-key / bounded indexers | `IndexerSpec` optional fields → `IndexerOp.{keyPoolTokens, maxScoredKeys, keyWriteElemsPerToken, headWeights}` + `kv_pool` | no |
| Shared top-k (indexer reuse) | `SparseSelection {kind: 'reuse'}`: `KvSelection {topk}`, no `IndexerOp` | no |
| Cross-layer KV sharing | `MlaAttentionBlock.kvCacheFromLayer`, `CsaGlobalSpec.kvCacheFromLayer`: no producer GEMMs, `kvWrite` 0 | no |
| Per-module weight formats (NVFP4 experts, FP8 attention, BF16 hyper-connections) | `QuantConfig.modules` per `WeightModule`, `linearGemm({ module })` | no |

---

## 3. Op catalogue

`src/sim/ir/ops.ts` defines the ops; `src/sim/ir/metrics.ts`: `opFlops`, `opBytes` count them. An op is the
work of one device on its stage's critical path for one microbatch step. Notation: `bpe(f)` bytes per element,
`past = kvLen − qLen`, `T = seqs·qLen`. FLOP counts are dense-equivalent.

### GEMM

`metrics.ts: gemmFlops, gemmBytes, gemmWeightBytes`. `out[b] (m×n) = in[b] (m×k) · W[b] (k×n)`; batch entries
have distinct weights (grouped experts, MLA per-head absorb).

- `F = 2·b·m·n·k`
- weights `b·(n·k·bpe(w) + perDimBits·(n + P·k)/8)` (`perDimBits` = 0 except EXL3; `P` = fused physical matrices, `GemmOp.outParts`); with 2:4 sparsity `b·n·k·(bits/2 + 1)/8` (half the values plus a 2-bit index per kept
  value); bias `+ b·n·bpe(bias)`
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
compressed global entries next to its raw window) shares the same softmax: Q is read once and O written once.

- `pairs = attendedPairs(primary) + attendedPairs(secondary, r)` (`attentionPairs`, also used by the shape model)
- `F = 2·seqs·heads·pairs·(qkHeadDim + vHeadDim)` (softmax not counted, §9)
- read Q `seqs·qLen·heads·qk·bpe(act)`, write O `seqs·qLen·heads·v·bpe(act)`
- kvRead `seqs·kvTokensRead·kvRead.elemsPerToken·bpe(kvRead)`; kvWrite `seqs·qLen·kvWrite.elemsPerToken·bpe(kvWrite)`
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
- `c_s` (documented approximations): chunked 4 for all variants (inter-chunk output 2 + state update 2);
  step: GatedDeltaNet and KDA 7 (Sᵀk 2 + delta outer product 2 + decay 1 + output 2; KDA's per-channel decay is
  still 1 FLOP per state element), Mamba2 5, linear 4.
- state read `hasPriorState ? seqs·heads·dk·dv·bpe(state) : 0`, state write `seqs·heads·dk·dv·bpe(state)`;
  chunked mode keeps intermediate chunk states on chip.
- read q, k `2·T·qkHeads·dk·bpe(act)`, v `T·heads·dv·bpe(act)`, gates `T·heads·g·bpe(act)` with
  `g = gateElemsPerTokenPerHead` (2 for GatedDeltaNet / Mamba2; KDA `dk + 1`: per-channel decay plus β); write o
  `T·heads·dv·bpe(act)`.

### Elementwise

`metrics.ts: opFlops (elementwise), elementwiseBytes`; `lower/common.ts: ELEMENTWISE_FLOPS_PER_ELEM`,
`SHAPE_DEPENDENT_FNS` (set by their lowering).
`F = elems·flopsPerElem`; bytes are the op's explicit fields (activation read/write, weights, state).

| fn | FLOP / elem | fn | FLOP / elem |
|---|---|---|---|
| rmsnorm | 4 | layernorm | 8 |
| residual_add | 1 | rope | 3 |
| qk_norm | 4 | silu | 4 |
| silu_mul | 5 | gelu_mul | 9 |
| gelu | 8 | relu | 1 |
| relu_mul | 2 | sigmoid_mul | 5 |
| gated_rmsnorm | 9 | causal_conv1d | 2·kernelWidth |
| router_topk | 5 per logit | moe_combine | 2 |
| hc_collapse | 2·N per output elem | hc_expand | 2N + 2 (mixing), else 2 |
| sinkhorn | 4·iters per T·N² elem | ngram_gate | 8 (two norm statistics, dot, sigmoid, fused add) |
| kv_pool | 4 (softmax weight, multiply, add) | causal_conv1d (+ residual) | 2·K + 1 |

These only feed the vector time; elementwise ops are memory-bound in practice.

### Embedding

`metrics.ts: embeddingBytes`. `F = 0`; weights `tokens·hidden·bpe(w)` (gathered rows, conservative); write
`tokens·hidden·bpe(out)`.

### Collective

No FLOP, no memory traffic (`opBytes` returns zeros); its time is `messageBytes` through the alpha-beta model
(§6).

---

## 4. Hardware

Schema: `src/sim/hw/types.ts` (`DeviceSpec`, `ComputeEngine`, `MatmulPeak`, `GemmTiling`, `MemorySpec`,
`LinkSpec`, `TopologySpec`, `HardwareStack`). Presets: `src/sim/hw/presets/` (registry `index.ts`:
`DEVICE_PRESETS`, `DEVICE_PRESET_GROUPS`, `DEVICE_UNLOCKS`, `deviceUnlocks`, `singleDevice`; the vendor files,
`LINK_PRESETS` and `TOPOLOGY_PRESETS`: see "Preset library" below). Unlock axes: `src/sim/hw/unlocks.ts`.

- A device has ≥ 1 **compute engine** (GPU tensor cores, NPU, TPU MXU); a run uses exactly one
  (`HardwareStack.engineId`, default `engines[0]`). An engine lists matmul peaks, vector (non-tensor) peaks,
  compute units (the wave granularity) and tiling parameters, and may cap the memory bandwidth it can reach
  (`memBandwidthCapBytesPerSec`, e.g. an NPU on a narrower fabric).
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

| Kind | Effect | Paths its provenance covers |
|---|---|---|
| `memoryCapacity` | replaces `memory.capacityOptionsBytes` | `memory.capacityOptionsBytes` |
| `memoryBandwidth` | replaces `memory.bandwidthBytesPerSec` | `memory.bandwidthBytesPerSec` |
| `peakScale` | multiplies every matmul peak matching ALL given filters (engineId, input, accumulate, sparsity) by `factor`; must match ≥ 1 peak | `engines.<i>.matmulPeaks` |
| `clockScale` | multiplies all matmul AND vector peaks of an engine (default all) by `factor`; memory bandwidth unchanged | `engines.<i>.matmulPeaks`, `engines.<i>.vectorFlopsPerSec` |
| link `p2p` | sets `p2p`; `'direct'` drops `hostBounce`, `'host_bounce'` requires it | `p2p`, `hostBounce` |
| link `linkBandwidth` | replaces `bandwidthPerDirectionBytesPerSec` | `bandwidthPerDirectionBytesPerSec` |

- Applied **in list order**; scales compose multiplicatively (`clockScale 0.8` then `peakScale ×2` on bf16 =
  209.5 · 0.8 · 2 = 335.2 TFLOP/s).
- Pure: the base is never mutated (explicit spreads; `structuredClone` is not in the ES2022 lib).
- **Provenance**: every modifier carries a `note` and `sources` (or an `estimatedReason`); applying appends them
  to the spec's `sources` / `estimated` and records the list in `appliedModifiers`.
- Validation: `factor > 0`, the engineId exists, a `peakScale` filter matches something (so a typo cannot
  silently do nothing), provenance present, and the result passes `validateDevice`.
- `simulate` applies `HardwareStack.deviceModifiers` and `linkModifiers` **once, up front**
  (`src/sim/engine/simulate.ts: effectiveHardware`); all cost code sees plain specs. Each applied modifier
  becomes a `variant: …` line in `SimResult.warnings`, and `SimResult.effectiveDevice` is the modified spec.

### Preset library

`src/sim/hw/presets/index.ts: DEVICE_PRESET_GROUPS, DEVICE_PRESETS, DEVICE_UNLOCKS, deviceUnlocks, singleDevice`;
checked by `src/sim/hw/presets.test.ts` (every preset) and one `<vendor>.test.ts` per file (vendor ratios and
spot values against the datasheet).

| File | Covers | Export |
|---|---|---|
| `presets/nvidia.ts` | GeForce RTX 3090, 4090, 5090 and their unlock axes | `NVIDIA_CONSUMER_DEVICE_PRESETS`, `RTX_3090_UNLOCKS`, `RTX_4090_UNLOCKS`, `RTX_5090_UNLOCKS` |
| `presets/nvidia-datacenter.ts` | DGX Spark (GB10), H100 SXM, H200 SXM, B200, B300 | `NVIDIA_DATACENTER_DEVICE_PRESETS` |
| `presets/apple.ts` | M5 Ultra, M5 Max, M4 Max, M3 Ultra, M3 Max; binned GPU-core counts as separate SKUs | `APPLE_DEVICE_PRESETS` |
| `presets/amd.ts` | Radeon AI PRO R9700, RX 9070 XT, Ryzen AI Max+ 395 (Strix Halo iGPU), Instinct MI300X / MI325X / MI355X / MI455X | `AMD_DEVICE_PRESETS` |
| `presets/intel.ts` | Arc Pro B70, B65, B60, Arc B580 | `INTEL_DEVICE_PRESETS` |
| `presets/links.ts` | PCIe, NVLink bridge / NVLink 4 / 5, InfiniBand, ConnectX-7, xGMI, Thunderbolt, USB4, Ethernet | `LINK_PRESETS`, `HOST_BOUNCE_PCIE_LINKS` |
| `presets/topologies.ts` | PCIe boxes, SoC clusters (with the inter-node fabric shape), HGX nodes, MI300X xGMI node | `TOPOLOGY_PRESETS`, `pcieBox`, `socCluster`, `hgxCluster`, `nodeTopology`, `fabricIdSuffix` |
| `presets/catalog.ts` | The setup catalog: what each device composes into (GPUs per machine, machines, local and network links, cabling), validation, `composeTopology`, SKU labels, the popular setups | `setupOptionsFor`, `allowedCablings`, `defaultCabling`, `setupErrors`, `composeTopology`, `setupLabel`, `POPULAR_SETUPS` |

`DEVICE_PRESET_GROUPS` is the display order: consumer / prosumer first (GeForce, Apple, AMD Radeon and Strix
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
  Datacenter parts, DGX Spark, AMD and Intel run both accumulators at one rate and list only fp32-accumulate
  entries (peak lookup step 2 maps an fp16-accumulate request onto them).
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

| Card | Axis (`id`) | Option: modifiers | Source class |
|---|---|---|---|
| RTX 3090 | `tensor-rate` | `full-rate`: `peakScale ×2` on fp16 / bf16 / tf32 with fp32 accumulate (dense and 2:4) | community (Zagitta gpu-unlock, listed stable) |
| RTX 3090 | `pcie-p2p` | `direct`: link `p2p → 'direct'` (bandwidth and latency kept) | community (patched open kernel driver) |
| RTX 4090 | `tensor-rate` | as the 3090 | community (listed experimental) |
| RTX 4090 | `vram-mod` | `48gb`: `memoryCapacity [48 GB]` (bandwidth unchanged: unverified) | review + community (teardown, GPU Lab) |
| RTX 4090 | `pcie-p2p` | as the 3090 | community (tinygrad driver) |
| RTX 4090 | `memory-overclock` | `1545mhz`: `memoryBandwidth` 1545 MHz × 16 × 384 bit / 8 = 1186.56 GB/s (theoretical bus peak) | review (TechPowerUp OC table) |
| RTX 5090 | `pcie-p2p` | as the 3090 | community (tinygrad driver) |
| RTX 5090 | `power-clock` | `575w`: `clockScale` 2768 / 2407 ≈ 1.150; `400w`: `clockScale` 2465 / 2407 ≈ 1.024 (measured sustained SM clocks under an AI load, relative to the 2407 MHz boost clock the vendor peaks assume; memory unchanged) | community (measured SM clocks) |

Rejected as UI choices (no capacity test or debunked; listed with links at the end of `nvidia.ts`): 3090 48 GB,
4090 96 GB, 5090 64 / 96 / 128 GB, a Blackwell tensor unlock. The 3090 NVLink bridge is a topology
(`RTX_3090_NVLINK_2`), not an unlock. Other vendors' devices have no unlock axes.

### Links and topologies

`src/sim/hw/presets/links.ts: LINK_PRESETS, HOST_BOUNCE_PCIE_LINKS`;
`src/sim/hw/presets/topologies.ts: TOPOLOGY_PRESETS, pcieBox, socCluster, hgxCluster, nodeTopology, linkPresetId`;
costed in §6
(`src/sim/engine/collectives.ts`).

- **Per direction, per device**: `bandwidthPerDirectionBytesPerSec` is what one device can inject in one
  direction, never a bidirectional sum (vendor "900 GB/s NVLink" bidirectional → 450 GB/s).
- **Theoretical bandwidth, hardware-floor latency**: bandwidth is the line-rate ceiling (PCIe: GT/s × lanes ×
  128/130 ÷ 8, before packet overhead); `latencySec` is a one-way hardware / verbs-level floor (P2P write,
  `ib_write_lat`, the transport's measured minimum for TCP / RDMA over Thunderbolt and USB4), never an NCCL /
  RCCL software floor. Latencies no vendor publishes are `estimated`.
- **Host bounce**: stock GeForce PCIe links are `p2p: 'host_bounce'` (`HOST_BOUNCE_PCIE_LINKS`, Gen4 / Gen5 × x8 /
  x16), with `hostBounce.hostMemBandwidthBytesPerSec` an assumed dual-channel DDR5-6000 host (96 GB/s,
  estimated: replace for the actual host) and `extraLatencySec` 12 µs (community P2P matrices). The
  `pcie-p2p` unlock switches the link to `'direct'`; Intel Arc Pro boxes use the `_DIRECT` links (stock driver).
  The cost is §6 "Host bounce".
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
  mesh, else ring, else chain, so the catalog defaults equal the pre-catalog UI defaults for every device
  (`catalog.test.ts` pins them) and two Thunderbolt or USB4 machines are a one-cable mesh, which costs exactly a
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
  concurrent ports is unmeasured) and α_f = 3 µs, the host turnaround of a user-space verbs relay (JACCL's 9–12 µs
  small 4-way all-reduce minus its ~7 µs one-way transfer, half the 14 µs `ibv_uc_pingpong` round trip; TN3205: "the
  application is responsible for forwarding"): physically 2 hops = 2 × 7 + 3 = 17 µs and 3 hops = 27 µs; with this
  preset's 50 µs `latencySec`, kept in this task although ~7 µs was measured (see the latency-audit follow-up), 103
  and 156 µs. `TB5_TCP` (new: the Thunderbolt bridge on TB5 ports at the 80 Gb/s = 10 GB/s per-direction line rate,
  Intel's TB5 brief; not an achieved goodput, none is cleanly measured) and `TB4_TCP` (40 Gb/s, the M3 Max): α 70 µs
  (half the bridge's 142 µs ICMP RTT) and α_f = 20 µs, the user-space receive-to-send turnaround, about a third of the
  ~48–70 µs one-way TCP time that is itself almost all software (MLX TCP ring ~48 µs per step; TN3205: the bridge
  forwards in software); 2 hops 160 µs, 3 hops 250 µs. `USB4_HALO_RDMA` α_f = 3 µs (as TB5 RDMA; 2 hops 19 µs, 3 hops
  30 µs), `USB4_HALO_TCP` 5 µs (the kernel's forwarding turnaround on a tuned stack, half the 10 µs one-way; 25 /
  40 µs). `presets.test.ts` pins these hop examples. No link was renamed; `LEGACY_LINK_IDS` (empty) is where a rename
  would be recorded for old URLs, and a TB5 Mac URL naming the TB4 bridge still resolves to that link.
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

---

## 5. Parallelism

`src/sim/lower/` (`lower.ts: lowerModel, lowerBlock`; one file per block kind; shared rules in `common.ts`).
`ParallelConfig { tp, pp, dp, ep }`, world = `tp·pp·dp` (ep groups live inside it).
The divisibility rules (heads, KV heads, widths over tp; experts and MoE modes over ep) are checked by lowering
each block once for a one-token step: `src/sim/engine/validate.ts: blockShardingErrors` (called by
`validateInput`; exported so the UI can attribute TP / EP errors to their fields).

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
**Prefill computes logits only for the last position** (`Tout = seqs`); decode `Tout = seqs·qLen`. No logits
gather: cross-shard sampling exchanges O(Tout·tp) scalars, treated as free.

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
keys: `idx.k (n = hd_I, k = keyInputDim ?? h)` → key LayerNorm over `T·hd_I` → with `keyPoolGate`:
`idx.pool_gate (n = hd_I, k = h)` → `kv_pool` (APE `p·hd_I`) → with `headWeights`: `idx.w (n = H_I, k = h)` →
the `indexer` op (key cache `QuantConfig.indexerKeyCache ?? kvCache`, `hd_I` elems per key, `p` tokens per key,
`hd_I/p` written per token by an owner, 0 by a layer with `ownsKeys: false`; `topK_I = ceil(topK·t/p)` indexer
keys, t = tokens per main-attention key: 1, or r for CSA). Replicated on every TP rank. The main attention gets
`kvSelection {topk, topK}`. Module formats: `idx.q` `indexer`, `idx.k` / `idx.pool_gate` `indexer_key`, `idx.w`
`indexer_head_weights`. Sequence state: the owned keys `ceil(ctx/p)·hd_I·bpe(key)`.

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

`lower/csa.ts: lowerCsa, csaSequenceStateBytes, compressedKvFormat`. `Hl = H/tp`, `Gl = G/tp` (tp must divide both);
the one K = V vector per token, the compressor, the indexer and all caches are replicated (one KV head).

1. `csa.q_a (n rq, k h)` → rmsnorm(T·rq) → `csa.q_b (n Hl·hd, k rq)` → rope over `T·Hl·dr`.
2. `csa.wkv (n hd, k h)` → rmsnorm(T·hd) → rope over `T·dr`.
3. Owner of the compressed cache: r > 1: `csa.wkv_c` + `csa.wgate_c` (n hd, k h, module `attention_kv_down`) →
   `kv_pool` (read `2·T·hd`, write `T/r·hd`) → rmsnorm and rope over `T/r` entries; r = 1: `csa.wkv_c` → rmsnorm.
4. The selection: indexer ops (qIn = rq, keys from the hd-wide latent, `p = r`), or nothing for `reuse`.
5. One `AttentionOp`: `kvHeads 1`, `qk = v = hd`, `kvSelection {window, W}`, `kvRead = kvWrite = {kvCache, hd}`
   (K = V, so one vector), and `secondaryKv {r, topk(topK entries), kvRead {compressedKvCache, hd},
   kvWrite {compressedKvCache, owner ? hd/r : 0}}` (§3 "Attention").
6. Inverse rope over `T·Hl·dr` → `csa.wo_a` grouped (batch Gl, m T, n R_o, k H·hd/G) → `csa.wo_b` (n h, k Gl·R_o)
   → all_reduce(tp).

Sequence state: `min(ctx, W)·hd·bpe(kv)` + owner `ceil(ctx/r)·hd·bpe(compressedKv)` + compressor tail `2·r·hd·4`
(fp32 kv and score state, r > 1) + owned indexer keys `ceil(ctx/r)·hd_I·bpe(key)`.

### KDA

`lower/linearAttention.ts` (variant `'kda'`, GLM-5.3-Flash). `lin.qkv (n 2·qkL·dk + vL·dv + vL, k h)` (q, k, v and
β) → `lin.f_a (n r, k h, replicated)` → `lin.f_b (n vL·dk, k r)` → `lin.g_a (n r, k h)` → `lin.g_b (n vL·dv, k r)` →
conv, recurrent (`variant 'kda'`, gates `dk + 1` per token and head: per-channel decay plus β), gated_rmsnorm,
out_proj, all_reduce. Sharding as for GatedDeltaNet (`vL = vHeads/tp` must divide).

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
are vector work, excluded like softmax (§9).

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
`C = 2·qkL·dk + vL·dv`. `in_proj n = 2·qkL·dk + vL·dv·(gate?2:1) + 2·vL` → `causal_conv1d` over `T·C`
(`2·K` FLOP/elem; rolling state `seqs·C·(K−1)·bpe(act)` read if a prior state exists, and written) →
`recurrent` (prefill: chunked, `chunkLen = min(64, qLen)`, `hasPriorState = kvLen > qLen`; decode: step) →
gated_rmsnorm (or rmsnorm) over `T·vL·dv` → `out_proj k = vL·dv` → all_reduce. Sequence state
`vL·dk·dv·bpe(state) + C·(K−1)·bpe(act)`, independent of context.

### MoE modes A/B

`lower/moe.ts: lowerMoe, moeLayout, moeRouting, moeParallelErrors, moeResidentWeightBytes`. Balanced
("uniform") routing. Valid iff `E % ep == 0` and (`tp % ep == 0` → **mode A**, or `ep % tp == 0` and
`(tp·dp) % ep == 0` → **mode B**). `Eloc = E/ep`, `etp = max(1, tp/ep)`, `p = max(1, ep/tp)`.

- **Mode A** (includes ep = 1): every TP rank has all T tokens. `A = ceil(T·k/ep)` assignments on the busiest
  device, `a = min(Eloc, A)` active experts, `mE = ceil(A/a)` tokens each. Router (hp, replicated) → top-k →
  grouped `gate_up (batch a, m mE, n (gated?2:1)·fe/etp)` → act → `down` → combine (`a·mE·h`) → shared experts
  as one dense MLP of `S·fs` sharded by tp → **one** all_reduce(tp).
- **Mode B**: the EP group spans p replicas; each device routes its slice `Ts = ceil(T/tp)`; `A = Ts·k`,
  `etp = 1`. Router → top-k → `all_to_all(ep)` dispatch `Ts·k·h·bpe(act)` → experts → `all_to_all(ep)` combine
  `Ts·k·h·reduceBpe` → combine → shared experts (unsharded) on Ts → `all_gather(tp)` of `T·h·bpe(act)` when
  tp > 1 (replaces the all-reduce).
- Resident: `Eloc·(gated?3:2)·h·fe/etp·bpe(w)` + router + shared (`/tp` in mode A).

### PP split

`lower/lower.ts: lowerModel, stageLayerRange`. Stage s holds layers `[floor(s·L/pp), floor((s+1)·L/pp))`;
stage 0 adds the embedding, the last stage the head. Every stage but the last sends `T·N·h·bpe(act)` to the next
(`send_recv`, group pp; N = `residualStreams`, default 1). A cross-layer source (shared KV cache, reused top-k,
shared indexer keys) on an earlier stage than its consumer produces a `simulate` warning
(`engine/simulate.ts: crossStageWarnings`); its transfer is not costed (§9).

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

---

## 6. Collectives

`src/sim/engine/collectives.ts`. S = messageBytes per device (§1 conventions), n group size, B per-direction
bandwidth, α latency. Each result splits α terms (`latencySec`) from β terms (`bandwidthSec`).
`collectiveTime` dispatches on kind and path; a flat path with a `fabric` goes to `fabricCollectiveTime`
("Fabric shapes" below).

### Flat collectives

`ringAllReduceSec, treeAllReduceSec, allReduceSec, ringAllGatherSec, ringReduceScatterSec, allToAllSec,
sendRecvSec`.

| Collective | Time |
|---|---|
| ring all-reduce | `2(n−1)·α + 2(n−1)/n·S/B` |
| tree all-reduce | `2⌈log2 n⌉·α + 2·S/B` |
| all-reduce `'best'` (default) | min(ring, tree) |
| ring all-gather (S = output) / reduce-scatter (S = input) | `(n−1)·α + (n−1)/n·S/B` |
| all-to-all (direct, concurrent) | `α + (n−1)/n·S/B` |
| send/recv | `α + S/B` |

### Hierarchical collectives

`hierarchicalAllReduceSec, hierarchicalAllGatherSec, hierarchicalReduceScatterSec, hierarchicalAllToAllSec`.
A group of n1 devices per node × n2 nodes:

- all-reduce = `RS_intra(S, n1) + AR_inter(S/n1, n2, algorithm) + AG_intra(S, n1)`
- all-gather = `AG_inter(S/n1, n2) + AG_intra(S, n1)`; reduce-scatter = `RS_intra(S, n1) + RS_inter(S/n1, n2)`
- all-to-all = `α_inter + max((n1−1)/n·S/B_intra, (n−n1)/n·S/B_inter)`

### Host bounce

`effectiveLink`. A `'host_bounce'` link (e.g. GeForce without P2P) stages every byte through host memory once
written, once read: `B_eff = min(B, hostMemBW / (2·nActive))`, `α_eff = α + extraLatencySec`, with
`nActive = n` for collectives and 1 for send/recv. In two-level collectives, the devices sharing host memory are
the group's devices per node, at both levels. PCIe switch / shared-uplink topologies are not modeled.

Also not modeled: NVLS / SHARP in-switch reduction, the memory traffic of collectives, and contention between
communication and compute for memory.

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

| collective | switched, o > 1 | full_mesh (k), N ≥ 3 | ring | chain |
|---|---|---|---|---|
| ring-class AR | `2(N−1)α + 2(N−1)/N·S/(B_L/o)` | `2(N−1)α + 2(N−1)/N·S/B_two` | `2(N−1)α + 2(N−1)/N·S/min(2B_L, C)` | `RS_chain + AG_chain` |
| tree AR | `2⌈log2 N⌉α + 2S/(B_L/o)` | `2⌈log2 N⌉α + 2S/B_one` | `2·Σ_k lat(h_k) + 2S/B_one` | same, chain hops |
| direct AR | — | `2α + 2(N−1)/N·S/B_mesh` | — | — |
| AR `'best'` | min(ring, tree) | min(ring, direct, tree) | min(ring, tree) | min(ring, tree) |
| AG / RS | `(N−1)α + (N−1)/N·S/(B_L/o)` | direct: `α + (N−1)/N·S/B_mesh` | `(N−1)α + (N−1)/N·S/min(2B_L, C)` | RS: `(N−1)α + (N−1)/N·S/min(B_L, C)`; AG: `(N−1)α + max((N−1)/N·S/B_L, (N+1)/N·S/C)` |
| all-to-all | `α + (N−1)/N·S/(B_L/o)` | `α + (N−1)/N·S/B_mesh` | `lat(⌊N/2⌋) + ⌊N²/4⌋/N·S/min(2B_L, C)` | `lat(N−1) + max(⌊N²/4⌋/N·S/B_L, E_N/N·S/C)` |
| send/recv (h hops) | `α + S/(B_L/o)` | `α + S/B_one` | `lat(h) + h·S/B_one` | `lat(h) + h·S/B_one` |

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

| collective | switched | switched o=2 | ring C=∞ | ring C=B_L | chain C=∞ | chain C=B_L | mesh k=1 C=∞ | mesh k=1 C=B_L | mesh k=2 C=∞ |
|---|---|---|---|---|---|---|---|---|---|
| ring AR | 12 + 1200 | 12 + 2400 | 12 + 600 | 12 + 1200 (= switched) | 12 + 1200 | 12 + 1600 | 12 + 600 | 12 + 1200 | 12 + 300 |
| tree AR | 8 + 1600 | 8 + 3200 | 14 + 1600 | 14 + 1600 | 14 + 1600 | 14 + 1600 | 8 + 1600 | 8 + 1600 | 8 + 800 |
| direct AR | — | — | — | — | — | — | 4 + 400 | 4 + 1200 | 4 + 200 |
| AG | 6 + 600 | 6 + 1200 | 6 + 300 | 6 + 600 | 6 + 600 | 6 + 1000 | 2 + 200 | 2 + 600 | 2 + 100 |
| RS | 6 + 600 | 6 + 1200 | 6 + 300 | 6 + 600 | 6 + 600 | 6 + 600 | 2 + 200 | 2 + 600 | 2 + 100 |
| all-to-all | 2 + 600 | 2 + 1200 | 5 + 400 | 5 + 800 | 8 + 800 | 8 + 1400 | 2 + 200 | 2 + 600 | 2 + 100 |
| send h=1 | 2 + 800 | 2 + 1600 | 2 + 800 | 2 + 800 | 2 + 800 | 2 + 800 | 2 + 800 | 2 + 800 | 2 + 400 |
| send h=2 | — | — | 5 + 1600 | 5 + 1600 | 5 + 1600 | 5 + 1600 | — | — | — |

At a decode-sized S = 1e4 B, `'best'` all-reduce picks the tree on the switch (8 + 2 = 10 µs), the ring algorithm on
the ring (12 + 1.5 = 13.5 µs with C = B_L; the tree's 2-hop step makes it 14 + 2) and direct on the k = 1 mesh (4 +
0.5 = 4.5 µs). 4 × DGX Spark (B_L = C = 25 GB/s, α 1.5 µs, α_f 5 µs in `simulate.test.ts`; the preset's estimate is 4.5 µs), S = 8e6 B: ring AR 9 + 480 on both
fabrics (bit-identical); tree 6 + 640 switched vs 19 + 640 on the ring; all-to-all 1.5 + 240 vs 8 + 320; at decode
size switched takes the tree (6.8 µs), the ring the ring algorithm (9.6 µs): decode is slower on the ring, prefill
equal.

---

## 7. Per-op time, stages, pipeline

### Per-op time

`src/sim/engine/opTime.ts: opTime, roofline, engineMemBandwidthBytesPerSec, elementwiseBytesUnderFusion`.
One exhaustive switch on `op.kind`. `memBW = min(memory.bandwidthBytesPerSec, engine.memBandwidthCapBytesPerSec ?? ∞)`.

| Op | Compute term | Memory term |
|---|---|---|
| gemm, attention, indexer, recurrent (chunked) | `F / (peak·efficiency)`; peak from §4 "Peak lookup and fallback" for the op's (math, accumulate, sparsity); efficiency from §4 "Shape efficiency" with the tiling of the format actually used | `bytes/memBW` |
| recurrent (step), elementwise | `F / vectorPeak(dtype)` (state dtype for the step) | `bytes/memBW` |
| embedding | 0 | `bytes/memBW` |
| collective | §6 on the group's path (TP group, EP group, or this stage's PP boundary) | — |

`t = max(compute, memory)`; the larger term is the op's `bound` (a tie counts as compute). A tensor op whose
lookup falls back to the vector peak keeps the efficiency term (conservative).

**Elementwise fusion** (`SimOptions.elementwiseFusion`, default **`'none'`**): `'none'` charges every
elementwise op its activation read and write; `'ideal'` sets them to 0 (fused into a neighbouring GEMM),
keeping weight and state bytes. The default is the conservative one; for REF_DENSE_8B prefill the difference is
≈ 5% (§10).

### Stage time and overlap

`src/sim/engine/pipeline.ts: stageResult, layerResult`. Per stage and microbatch: `compute_s` = Σ non-collective
op times, `lat_s` = Σ collective α terms, `bw_s` = Σ collective β terms, `p2p_s` = the send to the next stage.

- `overlap = 'none'`: `t_s = compute_s + lat_s + bw_s + p2p_s`
- `overlap = 'ideal'` (**default**): `t_s = lat_s + max(compute_s, bw_s)`. β can hide behind independent work
  (other token or column chunks); α is on the dependency chain. The PP send is not in `t_s`; it lengthens the
  latency path and, as a link resource, bounds the cycle (below).
- `exposedCommSec = t_s − compute_s`.

### Pipeline and DP

`src/sim/engine/pipeline.ts: microbatching, phaseResult`; `src/sim/engine/simulate.ts: simulate`.

- `B_r = batch/dp`, `m = min(pp, B_r)` microbatches of `b_mb = ceil(B_r/m)` sequences. Each phase is lowered
  once with `seqs = b_mb`. `Σ p2p` below is the sends' full time (α + β) under `'ideal'` and 0 under `'none'`
  (already in t_s).
- **Resources.** Each stage and each PP boundary link is a resource every microbatch occupies once per cycle:
  stage s for `t_s`, the link after stage s for the send's bandwidth time `βp2p_s = bytes/B` (α is latency,
  not occupancy). Overlapping a send with compute does not let the link carry more bytes than its bandwidth, so
  the bottleneck per microbatch is `b = max(max_s t_s, max_s βp2p_s)`. (Under `'none'`, t_s ≥ βp2p_s already.)
- **Decode**: `cycle = max(m·b, Σ_s t_s + Σ p2p)` (a microbatch cannot start its next token before the
  previous one left the last stage). `tokensPerSec = dp·B_r/cycle`, `perSequenceTokensPerSec = 1/cycle`,
  `latency = cycle`.
- **Prefill** (a steady stream of replica batches; the bottleneck resource bounds it): `cycle = m·b`,
  `tokensPerSec = dp·B_r·promptLen/cycle`, `latency (TTFT of a replica batch) = Σ t_s + Σ p2p + (m−1)·b`,
  `perSequenceTokensPerSec = promptLen/latency`.
- Example (`pipeline.test.ts`): TINY_DENSE, pp 2, batch 2, prompt 128 over a 1 B/s link: each microbatch sends
  128·256·2 = 65,536 B, so `cycle = 2·65,536 s = 131,072 s` and TTFT ≈ 131,072 s; decode sends 512 B per
  microbatch → `cycle = 1,024 s`.
- **Context of each rate** (`src/sim/engine/simulate.ts: simulate`, `types.ts: Workload`). Every rate is quoted
  at the context it applies to ("33.9 tok/s @ 8k"):
  - **Prefill @ P** (`promptLen = P`): the whole P-token prompt from an *empty* KV cache, so `past = 0` and
    `qLen = kvLen = P`. Prefill tok/s @ P = `P / latency` per sequence (`dp·B_r·P/cycle` aggregate).
  - **Decode @ N** (`decodeContextLen = N`): one decode step with N tokens *already* in the KV cache. The new
    token's query also sees its own key, so `qLen = 1` and `kvLen = past + qLen = N + 1`. Attention FLOP
    (`2·seqs·heads·(N + 1)·(qk + v)`) and KV reads (`seqs·(N + 1)·elemsPerToken·bpe(kv)`) grow linearly in N
    (full attention; a window or top-k selection caps both, §3 "Attended pairs").
    Decode tok/s @ N = `1/cycle` per sequence (`dp·B_r/cycle` aggregate). Test: `simulate.test.ts`.

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
  (0 with `kvCacheFromLayer`); CSA `min(ctx, W)·hd·bpe(kv)` + owner `ceil(ctx/r)·hd·bpe(ckv) + [r > 1] 2·r·hd·4`;
  owned indexer keys `ceil(ctx/p)·hd_I·bpe(key)` (0 for `reuse` and `ownsKeys: false`); linear attention
  `vL·dk·dv·bpe(state) + C·(K−1)·bpe(act)`; short conv `(K−1)·dim·bpe(act)`; n-gram conv
  `(K−1)·dilation·O·bpe(act)`; hyper-connections 0.
- `act_s` = the largest activation read + write of any single op of the stage over both phases (the transient
  inputs and outputs of the op running). It scales with `b_mb·promptLen` in prefill.
- `total_s = W_s + B_r·state_s + act_s`; `fits = total_s ≤ capacity`, capacity =
  `memCapacityBytes ?? capacityOptionsBytes[0]` of the modified device.
- `maxSequencesPerReplica = floor(min_s (capacity − W_s − act_s)/state_s)` (Infinity when no stage keeps
  per-sequence state). The activation peak is the one at the evaluated batch, so this is an estimate.

### Unified memory

On `kind: 'unified'` devices the capacity is shared with the CPU and OS; the usable fraction is OS-dependent and
not modeled, and `simulate` adds a warning saying so.

---

## 9. Deliberately excluded, and approximations

Excluded on purpose (only hardware and math limits are modeled):

- **Inference-engine knobs**: paged attention, continuous batching, chunked prefill, scheduler overhead,
  kernel launch overhead of a specific engine, CUDA graphs. Speculative decoding and weight/expert/lookup-table
  offloading are opt-in deployment-scenario knobs (off by default, identical results when off); their design is
  `docs/design/offload-specdec-latency.md`.
- Softmax / SFU throughput (it can bind for head_dim ≤ 64).
- Dequantization compute of weight-only quantized GEMMs (they run at the wider format's peak, §2).
- Split-K / stream-K; L2 / cache reuse beyond compulsory traffic.
- Clock throttling beyond an explicit `clockScale` modifier.
- NVLS / SHARP in-switch reduction; PCIe switch / shared-uplink contention; the memory traffic of collectives
  and comm/compute contention for memory.
- Sequence parallelism; logits gather; attention sinks; expected-value MoE routing
  (`E·(1−(1−k/E)^T)` active experts is a documented future option); the OS memory reservation.

Documented approximations (each isolated in one constant or rule):

- Recurrent `c_s` constants (chunked 4; step GDN 7, Mamba2 5, linear 4) and the chunk length `min(64, qLen)`.
- linear-attention `in_proj` carries `2·vH` gate outputs for GatedDeltaNet, Mamba2 and linear (KDA: β only, its
  decay comes from the low-rank forget gate).
- MLA form by phase (absorbed for decode and sparse prefill, MHA for dense prefill).
- The indexer is replicated on every TP rank; its key norm is counted as a LayerNorm.
- conv1d and recurrent state reads are 0 on the first prefill step (no prior state).
- A sliding-window cache holds `min(ctx, W)` tokens.
- Row-parallel biases are counted on every TP rank.
- Reductions travel as bf16 when activations are narrower than 16 bits.
- Prefill computes LM-head logits for one position per sequence (`Tout = seqs`).
- Balanced MoE routing on the busiest device (ceilings).
- Elementwise FLOP-per-element table (§3) and the fp32-accumulate default.
- The activation peak in the memory fit is the largest single op.
- Fabric shapes (§6 "Fabric shapes"): a TP / EP group covering part of a ring is a chain (the long way round is
  ignored); PP flows on a ring are not split over both directions (conservative by at most 2× on that send's β);
  relays go through the host and count against the all-ports cap C (no in-NIC switching is assumed: L3-routed RoCE
  between Sparks is reported not to work and mesh forwarding is unmeasured); tree steps are assumed not to share
  links; mesh striping over k cables is ideal; every shape but switched needs one device per node; intra-node links
  stay switched (`XGMI_MI300X_8_GPU` encodes its full mesh as aggregate injection under the switched formulas, which
  equals the mesh's direct β but keeps the ring's 2(n−1)α; re-modelling it is a follow-up).

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
- The hyper-connection mixing weights the projections produce are consumed on chip (only the streams are read by
  `hc_collapse`); the embedding broadcast into N streams is ignored (< 0.5%).
- n-gram tables count toward weight memory and total params, not active params; the last n−1 token ids the hashes
  need are not counted as state.
- KDA `A_log` (vH) and `dt_bias` (vH·dk) are not modeled as weight reads; per-channel decay costs like GDN's.
- The relative-position bias gather and add inside attention are excluded, like softmax.
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
