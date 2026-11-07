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
`linear_attention`, `mlp`, `moe`), which covers hybrid linear/full attention stacks, dense first layers before
MoE, and alternating sliding-window layers. The builders (`src/sim/ir/builders.ts`: `decoderLayer`,
`decoderModel`, `repeatLayers`) produce pre-norm decoder layers from `MixerParams` + `FfnParams`.

**Lowering** (§5, `src/sim/lower/lower.ts: lowerModel`) turns spec + workload step + quantization +
parallelism into the ops ONE device runs per stage, with concrete shapes and dtypes (§3).

### Dtypes, math and accumulate

`src/sim/ir/dtype.ts`: `DTYPE_INFO`, `bitsPerElement`, `bytesPerElement`, `resolveGemmMath`,
`resolveAccumulate`.

- Storage bits per element: `bits = elementBits + blockScaleBitsPerElem + extraBitsPerElem`;
  `bpe = bits / 8`. Block scales: MX formats 8 bit per 32 elements (0.25 bit/elem); nvfp4 8 bit per 16 (0.5).
  `extraBitsPerElem` carries group-quantization overhead outside the dtype, e.g. GPTQ int4 g128 with a 16-bit
  scale and 4-bit zero: (16 + 4)/128 = 0.15625.
- **GEMM input class** (`resolveGemmMath`): the operand with more element bits wins (int4 × bf16 runs at the
  bf16 rate, because weight-only quantization dequantizes); on a tie the activation's format wins.
- **Accumulator** (`resolveAccumulate`): integer inputs → `int32`; bf16 / tf32 / fp32 → always `fp32`;
  fp16 / fp8 / fp6 / fp4 → the requested `QuantConfig.accumulate` (**default `fp32`**). This default matters on
  GeForce parts, where fp32 accumulate runs at half the fp16-accumulate rate.
- `QuantConfig` fields: `weights` (linear layers), `activations`, `kvCache` (KV, MLA latent and indexer keys),
  `recurrentState` (default fp32), `embeddings` / `lmHead` (default `highPrecision`), `highPrecision` (norms,
  router, biases, conv taps; default bf16), `attentionMath`, `accumulate`, `weightSparsity`.

### Parameter counts

`src/sim/ir/params.ts`: `blockParams`, `activeBlockParams`, `countParams`, `activeParamsPerToken`. Logical
totals (unsharded), used to sanity-check mappers against published sizes.

- gqa: `h·(heads + 2·kvHeads)·hd` (+ `h·heads·hd` with an output gate) + `heads·hd·h` (o) + biases + `2·hd`
  (q/k norm) + indexer.
- mla: q path `h·rq + rq + rq·H·(dn+dr)` (or `h·H·(dn+dr)` when rq = 0), kv `h·(rkv+dr) + rkv + rkv·H·(dn+dv)`,
  o `H·dv·h`, + indexer (q input rq or h).
- indexer: `qIn·heads_I·hd_I + h·hd_I + h·heads_I + 2·hd_I` (key LayerNorm γ, β).
- linear_attention: in_proj `h·(2·qkH·dk + vH·dv·(gate?2:1) + 2·vH)`, conv `(2·qkH·dk + vH·dv)·K`, `2·vH`
  (A_log, dt_bias), `dv` (out norm), out_proj `vH·dv·h`.
- mlp `(gated?3:2)·h·f`; moe `E·(gated?3:2)·h·fe + E·h + S·(gated?3:2)·h·fs`; active MoE uses `k` instead of `E`
  for the routed experts.

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

---

## 3. Op catalogue

`src/sim/ir/ops.ts` defines the ops; `src/sim/ir/metrics.ts`: `opFlops`, `opBytes` count them. An op is the
work of one device on its stage's critical path for one microbatch step. Notation: `bpe(f)` bytes per element,
`past = kvLen − qLen`, `T = seqs·qLen`. FLOP counts are dense-equivalent.

### GEMM

`metrics.ts: gemmFlops, gemmBytes, gemmWeightBytes`. `out[b] (m×n) = in[b] (m×k) · W[b] (k×n)`; batch entries
have distinct weights (grouped experts, MLA per-head absorb).

- `F = 2·b·m·n·k`
- weights `b·n·k·bpe(w)`; with 2:4 sparsity `b·n·k·(bits/2 + 1)/8` (half the values plus a 2-bit index per kept
  value); bias `+ b·n·bpe(bias)`
- read `b·m·k·bpe(in)`, write `b·m·n·bpe(out)`

### Attended pairs

`metrics.ts: attendedPairs`. (query, key) pairs per sequence per head. `cap` = window size or top-k (∞ for full).

- non-causal: `qLen·min(cap, kvLen)`
- causal, full: `qLen·past + qLen(qLen+1)/2` — **the causal triangle is counted exactly, diagonal included**
  (masked tiles are skipped); e.g. q = kv = 4096 → 8,390,656 pairs.
- causal, capped: `j0 = clamp(cap − past, 0, qLen)`; pairs = `j0·past + j0(j0+1)/2 + (qLen − j0)·cap`.

`metrics.ts: kvTokensRead` (distinct keys touched per sequence): full `kvLen`; window `min(kvLen, W + qLen − 1)`;
top-k `min(kvLen, topK·qLen)` (worst case: disjoint selections).

### Attention

`metrics.ts: attentionFlops, attentionBytes`. Fused QKᵀ → softmax → ·V (FlashAttention-style: scores never
reach memory).

- `F = 2·seqs·heads·pairs·(qkHeadDim + vHeadDim)` (softmax not counted, §9)
- read Q `seqs·qLen·heads·qk·bpe(act)`, write O `seqs·qLen·heads·v·bpe(act)`
- kvRead `seqs·kvTokensRead·kvRead.elemsPerToken·bpe(kvRead)`; kvWrite `seqs·qLen·kvWrite.elemsPerToken·bpe(kvWrite)`

### Indexer

`metrics.ts: indexerFlops, indexerBytes`. DeepSeek DSA "lightning indexer": `score(q, s) = Σ_h w_h·ReLU(q_h·k_s)`,
top-k per query. It scores every visible key.

- `F = 2·seqs·heads·pairs_full·headDim`
- read q `seqs·qLen·heads·headDim·bpe(act)` + head weights `seqs·qLen·heads·bpe(act)`
- key cache read `seqs·kvLen·keyElems·bpe(key)`, write `seqs·qLen·keyElems·bpe(key)`; indices `seqs·qLen·topK·4`

### Recurrent

`metrics.ts: recurrentFlops, recurrentBytes`; `lower/linearAttention.ts: stateFlopsPerElemPerToken`.
State per sequence = `heads·dk·dv` elements.

- `F = seqs·qLen·heads·( c_s·dk·dv + (chunked ? (C+1)·(dk+dv) : 0) )`; the second term is causal attention
  inside a chunk of C tokens ((C+1)/2 keys per query on average, 2 FLOP per element over dk + dv).
- `c_s` (documented approximations): chunked 4 for all variants (inter-chunk output 2 + state update 2);
  step: GatedDeltaNet 7 (Sᵀk 2 + delta outer product 2 + decay 1 + output 2), Mamba2 5, linear 4.
- state read `hasPriorState ? seqs·heads·dk·dv·bpe(state) : 0`, state write `seqs·heads·dk·dv·bpe(state)`;
  chunked mode keeps intermediate chunk states on chip.
- read q, k `2·T·qkHeads·dk·bpe(act)`, v `T·heads·dv·bpe(act)`, gates `2·T·heads·bpe(act)`; write o `T·heads·dv·bpe(act)`.

### Elementwise

`metrics.ts: opFlops (elementwise), elementwiseBytes`; `lower/common.ts: ELEMENTWISE_FLOPS_PER_ELEM`.
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
`LinkSpec`, `TopologySpec`, `HardwareStack`). Presets: `src/sim/hw/presets/` (`RTX_5090`, `singleDevice`).

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

---

## 5. Parallelism

`src/sim/lower/` (`lower.ts: lowerModel, lowerBlock`; one file per block kind; shared rules in `common.ts`).
`ParallelConfig { tp, pp, dp, ep }`, world = `tp·pp·dp` (ep groups live inside it).

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

`lower/indexer.ts: lowerIndexer, indexerSequenceStateBytes, indexerSelection`. Ops: `idx.q (n = heads_I·hd_I,
k = qIn)` (qIn = h, or rq for MLA with q compression) → `idx.k (n = hd_I, k = h)` → key LayerNorm over `T·hd_I`
→ `idx.w (n = heads_I, k = h)` → the `indexer` op (key cache `keyDtype ?? kvCache`, `hd_I` elems per token).
Replicated on every TP rank. The main attention gets `kvSelection {topk, topK}`.

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
stage 0 adds the embedding, the last stage the head. Every stage but the last sends `T·h·bpe(act)` to the next
(`send_recv`, group pp).

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

---

## 6. Collectives

`src/sim/engine/collectives.ts`. S = messageBytes per device (§1 conventions), n group size, B per-direction
bandwidth, α latency. Each result splits α terms (`latencySec`) from β terms (`bandwidthSec`).
`collectiveTime` dispatches on kind and path.

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
- Decode is evaluated at one step with `kvLen = decodeContextLen` (the KV length including the new token).

---

## 8. Memory fit

### Memory fit

`src/sim/engine/memory.ts: memoryReport, stageWeightBytes, stageSequenceStateBytesPerSeq,
stageActivationPeakBytes`; `lower/lower.ts: blockResidentWeightBytes, blockSequenceStateBytes`; the per-block
helpers `gqaSequenceStateBytes`, `mlaSequenceStateBytes`, `indexerSequenceStateBytes`,
`linearAttentionSequenceStateBytes`, `moeResidentWeightBytes`, `embeddingResidentWeightBytes`,
`lmHeadResidentWeightBytes`.

Per device of stage s:

- `W_s` = Σ resident block weights (non-MoE blocks: the weight bytes of a one-token lowering, so resident and
  streamed weights agree by construction; MoE: all local experts) + the embedding shard on stage 0
  (`ceil(vocab/tp)·h·bpe(emb)`) + the LM-head shard and final norm on the last stage. A tied head shares the
  embedding allocation only with pp = 1 and an identical stored format (dtype and `extraBitsPerElem`,
  `memory.ts: sameStoredFormat`); a tied head stored differently (e.g. bf16 embeddings, int4 head) is a second
  copy and counted in full.
- `state_s` per sequence at `ctx = max(promptLen, decodeContextLen)`: GQA `min(ctx, W)·2·KVl·hd·bpe(kv)`
  (a sliding window keeps W tokens); MLA latent `ctx·(rkv+dr)·bpe(kv)`; indexer keys `ctx·hd_I·bpe(key)`;
  linear attention `vL·dk·dv·bpe(state) + C·(K−1)·bpe(act)`.
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
  speculative decoding, kernel launch overhead of a specific engine, CUDA graphs.
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
- linear-attention `in_proj` carries `2·vH` gate outputs for every variant.
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

---

## 10. Worked example: REF_DENSE_8B on one RTX 5090

REF_DENSE_8B has Qwen3-8B's shape (h 4096, 36 layers, 32 heads, 8 KV heads, hd 128, ffn 12288, vocab 151936,
8,190,735,360 params). bf16 weights, activations and KV; fp32 accumulate; batch 1. RTX 5090: bf16 / fp32-acc
209.5 TFLOP/s dense, 1.792 TB/s (whitepaper Table 3). Tests: `src/sim/engine/e2e.test.ts` (T9).

**Decode at a 1024-token context.** Bytes that must stream per token: GEMM weights
`2·(36·192,937,984 + 622,329,856) = 15,136,194,560 B` (linear weights per layer: qkv 25,165,824 + o 16,777,216 +
MLP 150,994,944; plus the LM head) and KV `36 layers·1024 tokens·(2·8·128·2 B) = 150,994,944 B`, total
15,287,189,504 B → bound `1.792e12 / 15.287e9 = 117.2 tok/s`. The simulator gives **117.1 tok/s** (norms,
activations and the embedding row add 0.1%); every op is memory-bound.

**Prefill of 4096 tokens.** Tensor FLOP: linear GEMMs `2·6,945,767,424·4096 = 56,899,726,737,408`, attention
`36·2·32·8,390,656·256 = 4,949,010,284,544` (exact causal pairs 4096·4097/2), LM head for the last position
`2·622,329,856`; total 61,849,981,681,664 FLOP → `0.29523 s` at 209.5 TFLOP/s → bound **13,874 tok/s**.
The simulator gives **12,892 tok/s** with `elementwiseFusion 'none'` (0.93 of the bound: GEMM shape efficiency
0.96–1.0 and ≈ 29 GB of elementwise traffic ≈ 16 ms) and **13,575 tok/s** with `'ideal'` (0.98).

**Memory**: weights 16,381,470,720 B + KV at 4096 tokens 603,979,776 B + activation peak 301,989,888 B
(the prefill `silu_mul`) = 17.29e9 B ≤ 32e9 B: fits.

The demo line in `src/App.tsx` runs the same model with prompt 1024 and decode context 1024 (≈ 12,800 prefill
tok/s, 117.1 decode tok/s).

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
