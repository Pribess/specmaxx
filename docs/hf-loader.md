# Hugging Face loader: model ID → config.json → model IR

The loader turns a Hugging Face model ID into the simulator's model IR (`ModelSpec`, docs/simulation-model.md §2)
and default numeric formats (`QuantConfig`). It runs in the browser; there is no backend.

```
loadHfModel(id, { revision?, token? })                       src/hf/load.ts
 ├─ fetchHfRepoFiles: api + config.json + sidecars → HfRepoFiles   src/hf/fetch.ts (network, cache, token)
 └─ mapHfModel(files) → HfMapResult                              src/sim/models/hf/index.ts (pure, no fetch)
     1. unwrapTextConfig    text_config / multimodal wrapper     textConfig.ts
     2. dispatchFamily      architectures[0] / model_type        dispatch.ts, families.ts
     3. family.build        config fields → ModelSpec            families/*.ts
     4. deriveQuantDefaults quantization metadata → QuantConfig  quant/*.ts
        mergeQuantHints     + the family's architecture-native formats
     5. checkParams         IR + excluded parts vs safetensors   paramCheck.ts
```

`mapHfModel` never throws for bad input. It returns `ok` (model, family, the detected `modelType` and
`architecture` (`architectures[0]`), quant defaults, excluded parts, param check, notes), `unsupported` (unknown
`model_type` / architecture, with the supported list) or `invalid_config`
(path-qualified errors such as `text_config.hidden_size: missing number`).

## 1. Fetching (`src/hf/`)

- **Files.** `GET https://huggingface.co/api/models/<id>[/revision/<rev>]` for the metadata (`safetensors`
  parameter counts by dtype, tags, `cardData.base_model`, `siblings`), then
  `https://huggingface.co/<id>/resolve/<rev>/config.json`. Both allow CORS; `resolve` answers with a 307 to
  `/api/resolve-cache/…`, which `fetch` follows. The sidecars `hf_quant_config.json` (ModelOpt),
  `quantization_config.json` (some AutoRound / Quark exports) and `params.json` (Mistral consolidated format)
  are fetched only when `siblings` lists them, so there is no 404 probing.
- **Repos without config.json.** Mistral NVFP4 repos ship only `params.json`. When `config.json` is 404 and the
  repo has a `params.json` and a `cardData.base_model`, the base model's `config.json` is used. The result records
  that repo in `configFrom`, as data only: no note, and nothing in the UI shows it. The quant defaults still come
  from the repo's own `params.json`.
- **Gated repos with a mirror.** Gated repos with a known ungated mirror (`src/hf/mirrors.ts`, e.g.
  `meta-models/Muse-Spark-Open-816B-A42B` → `unsloth/Muse-Spark-Open-816B-A42B`) fall back to that mirror for
  `config.json`, the sidecars and the `/api/models` metadata. This happens only when an official request answers
  401, no token is set and the revision is the default branch. The result keeps the official ID, so the model is
  named after it, and records the mirror in `configFrom` / `metadataFrom`. Those are data only: no note, and
  nothing in the UI mentions the mirror. If any mirror request fails, the original 401 (the token prompt) is
  returned. With a token, the official repo is read as before, and a gated repo without a mirror still answers 401.
- **Revision.** Optional, default `main`; it goes into both URLs and the cache key.
- **Token and 401.** Gated and private repos answer 401. The error (`unauthorized`) says so and asks for a token.
  An optional user token is kept in `localStorage` under `specmaxx:hfToken` (`token.ts`) and sent as
  `Authorization: Bearer <token>`. It is never logged and never part of a cache key. 403 (`forbidden`) means the
  token lacks access or the gate terms are not accepted. The other errors are `invalid_id`, `not_found`,
  `rate_limited` (429), `http`, `network` (offline or CORS) and `bad_json` (`errors.ts`).
- **Caching.** An in-memory `Map` in front of `sessionStorage`, keyed `specmaxx:hf:<id>@<revision>:<file>`
  (`cache.ts`). Response texts are cached, and so is "absent" for an optional file's 404. Errors (401, 403, network)
  are never cached, so a retry after adding a token works. A full or unavailable `sessionStorage` is tolerated.
- Everything takes an injectable `fetchImpl` and `Storage`; the tests use recorded responses and no network.

## 2. Language model only: `text_config` unwrap

Multimodal checkpoints nest the language model under `text_config` (NVIDIA's Nemotron-H omni wrappers, e.g.
Nemotron-3.5-Super-VL: `llm_config`; `text_config` wins when both exist) and keep `architectures`, `model_type` and
`vision_config` (sometimes also `quantization_config`) at the top. The simulator models the language model only;
vision and audio towers are excluded and their parameters are estimated for the param check
(`estimateVisionParams`, or a family's own estimate). The result's notes say "language model only; vision/audio
towers excluded". Flat (text-only) configs are used as they are. `vision_config.model_type` is never used for
dispatch (Qwen4Exp's equals the top-level type).

## 3. Dispatch and families

Keys are tried in this order; the first one a family claims wins: top-level `architectures[0]`, top-level
`model_type`, `text_config.architectures[0]`, `text_config.model_type`. Architectures come first because they
name the model class; model types second because text-only extractions often drop `architectures`.

| Family (`families/…`)                                                          | `model_type`                                             | `architectures[0]`                                                      | Shortlist models                                                               |
| ------------------------------------------------------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `qwen` (`qwen.ts`)                                                             | `qwen3_5`, `qwen3_5_text`, `qwen4_exp`, `qwen4_exp_text` | `Qwen3_5ForConditionalGeneration`, `Qwen4ExpForConditionalGeneration`   | Qwen3.8-27B, Qwen3.8-Flash-Next                                                |
| `deepseek_v41` (`deepseekV41.ts`)                                              | `deepseek_v41`, `deepseek_v41_text`                      | `DeepseekV41ForCausalLM`                                                | DeepSeek-V4.1-Flash (and `-1017`)                                              |
| `deepseek_v4` (`deepseekV41.ts`, `deepseekV4Family`: same builder, V4 profile) | `deepseek_v4`                                            | `DeepseekV4ForCausalLM`                                                 | DeepSeek-V4-Flash-0731                                                         |
| `glm5_next` (`glm5Next.ts`)                                                    | `glm5_next`, `glm5_next_text`                            | `Glm5NextForConditionalGeneration`                                      | GLM-5.3-Flash                                                                  |
| `glm_moe_dsa` (`glmMoeDsa.ts`)                                                 | `glm_moe_dsa`                                            | `GlmMoeDsaForCausalLM`                                                  | GLM-5.3 (also GLM-5)                                                           |
| `mistral_large4` (`mistralLarge4.ts`)                                          | `mistral_large4`, `mistral_large4_text`                  | `MistralLarge4ForConditionalGeneration`                                 | Mistral Large 4                                                                |
| `inkling` (`inkling.ts`)                                                       | `inkling_mm_model`, `inkling_text`                       | `InklingForConditionalGeneration`                                       | Inkling-Small                                                                  |
| `minimax` (`minimax.ts`)                                                       | `minimax_m3_vl`, `minimax_m3_vl_text`                    | `MiniMaxM3SparseForConditionalGeneration`, `MiniMaxM3SparseForCausalLM` | MiniMax-M3 (and M3.1-Flash)                                                    |
| `nemotron_h` (`nemotronH.ts`)                                                  | `nemotron_h`, `nemotron_h_omni`                          | `NemotronHForCausalLM`, `NemotronH_Omni_Reasoning_V3`                   | Nemotron 3.5 Lightning, Nemotron 3.5 Super VL                                  |
| `muse_spark` (`museSpark.ts`)                                                  | `muse_spark`, `muse_spark_text`                          | `MuseSparkForConditionalGeneration`                                     | Muse Spark (`meta-models/Muse-Spark-Open-816B-A42B`, read through `unsloth/…`) |

The MLA/DSA families share their readers in `families/mlaCommon.ts` (MLA dims from `qk_nope_head_dim` /
`qk_rope_head_dim` / `v_head_dim` — never `head_dim` or `num_key_value_heads`, which are traps in GLM and Mistral
configs —, the DSA indexer, the `indexer_types` / `mlp_layer_types` schedules with the transformers fallbacks, MoE,
mHC hyper-connections, MTP estimates).

**Nemotron-H.** Each decoder layer holds ONE block (`norm → block → residual`, `singleBlockLayer`), chosen by
`layers_block_type` (or the legacy `hybrid_override_pattern` string, `nemotronHPattern.ts`): a Mamba-2 mixer
(`linear_attention` variant `mamba2`: `n_groups` B/C groups, `mamba_num_heads` × `mamba_head_dim`, `ssm_state_size`,
`conv_kernel`, conv bias, `chunk_size`), full attention WITHOUT RoPE (the forward applies none), an MoE with non-gated
relu² experts, or a dense relu² MLP. Super's routed experts run at a latent width (`moe_latent_size`, down/up
projections around them, kept in BF16). The omni wrapper (`llm_config`) is simulated as its language model only.
MTP (one attention + MoE module) is the drafter (`mtp_layers_block_type`), costed only with speculative decoding on.
The SSM state format comes from `mamba_ssm_cache_dtype` (fp32; the model card serves fp16).

**Rules every family follows.**

- The mapping reads config fields, never model names, so sibling checkpoints of the same architecture (BF16
  twins, later revisions, fine-tunes) map too. Per-layer roles come from config lists (`layer_types`,
  `mlp_layer_types`, `indexer_types`, `compress_ratios`, `kv_source_layer_ids`, `kv_source_layers`,
  `no_rope_layers`, …), with the fallbacks transformers uses when a list is absent.
- Vision or audio towers are excluded from the IR. MTP / next-n / draft modules are kept as `model.draft`
  (§7 "Drafters") but are not part of the model either: `countParams`, the memory fit and every op list ignore them
  unless speculative decoding is on. For the param check both are estimated (`mtpParamsEstimate`,
  `visionParamsEstimate`, `excludedParts`).
- n-gram / Engram lookup tables are in the IR: they count toward total parameters and weight memory, not toward
  active parameters (a token reads only its own rows).
- Sub-percent extras that the IR does not express (attention sinks, router biases, logit softcap, …) are listed in
  the family's notes and in docs/simulation-model.md §9.
- The built model passes `modelSpecErrors` (`ir/validateModel.ts`); a config whose cross-layer references point
  nowhere (e.g. a shared indexer before any full one) is `invalid_config`.

**Adding a family.**

1. Write `src/sim/models/hf/families/<name>.ts` exporting a `FamilyEntry`: its `family` name, the `modelTypes` and
   `architectures` it claims, and `build(text, top, files)`. Read fields with `createReader` (`reader.ts`): it
   never throws, collects path-qualified errors, and the builder returns them all as `invalid_config`.
2. Build layers with the builder API (`ir/builders.ts`: `decoderLayer`, `decoderModel`, `repeatLayers`); if a block
   kind is missing, add it to `src/sim/ir/` and its lowering first (docs/simulation-model.md §11).
3. Return the excluded-part estimates and, if the architecture fixes some formats natively (e.g. DeepSeek's cache
   formats), `quantHints`.
4. List the entry in `families.ts` (a model type or architecture may appear in only one entry; a test checks it).
5. Add the repo's `config.json` and trimmed `api.json` to `src/sim/models/__fixtures__/` (see its README) and a
   test: total and active parameters against the safetensors headers, the layer-type mix, sibling checkpoints.
   `shortlist.e2e.test.ts` shows how to run it through `simulate`.

## 4. Quant defaults (`quant/`)

`deriveQuantDefaults` reads the repo's quantization metadata and returns a `QuantConfig` plus a scheme label, its
source and notes. The defaults are plain data; the UI may override any of them. Source, first found wins:
`params.json` `quantization_config` (Mistral consolidated exports), config.json `quantization_config` (top level,
else `text_config` / `llm_config`, dispatched on `quant_method`: `fp8`, `modelopt`, `modelopt_mixed`, `compressed-tensors`, `quark`, `gptq`, `awq`,
`auto-round`, `bitsandbytes`, `exl3`; a `bits` + `group_size` object without `quant_method` is MLX), `hf_quant_config.json`,
`quantization_config.json`, the MLX `quantization` key, `params.json` `quantization` (native Mistral fp8), and
finally the model's `torch_dtype` / `dtype` (unquantized).

- **Per module class.** Checkpoints quantize module classes differently (NVFP4 routed experts with FP8 attention
  and BF16 hyper-connections; BF16 `kv_b` with FP8 q/kv_a/o). Each reader is asked about a fixed set of probes
  (one synthetic module name per class, layer and projection, in the checkpoint's own naming; `moduleClass.ts`)
  and `assemble.ts` takes the majority per `WeightModule` class into `QuantConfig.modules`, reporting splits and
  unmatched ignore-list entries as notes.
- **ModelOpt `modelopt_mixed`.** NVIDIA's Nemotron-3.5 NVFP4 exports say `quant_method: "modelopt_mixed"` with
  `quant_algo: MIXED_PRECISION` and a per-module `quantized_layers` list (`readModeloptMixed`, the same schema as
  ModelOpt MIXED_PRECISION; `config_groups` is a compressed-tensors view of the same list and is not read).
  Nemotron-H probes use the `backbone.layers.<i>.mixer.*` naming (`language_model.` prefixed in the VL wrapper),
  one sublayer per layer from `layers_block_type`. Super-VL gives FP8 W8A8 Mamba in/out projections, attention
  and LM head, NVFP4 W4A4 routed and shared experts, BF16 latent projections (`moe_latent`), router and
  embeddings, and an fp8 KV cache.
- **Scale overheads.** MX and NV formats carry their block scale in the dtype; every other scale or zero point is
  amortized into `extraBitsPerElem` (block scale `s/(r·c)`, group scale `(s + z)/g`). The full table of schemes,
  overheads and readers is in docs/simulation-model.md §2 "Quant overheads".
- **EXL3.** `quant_method: "exl3"` (`quant/exl3.ts`) gives fractional bits per weight: `bits` (or `bpw` /
  `avg_bits`) for the linear layers, `head_bits` for the LM head. `QuantDefaults.exl3` carries both numbers so the
  UI's bits-per-weight inputs start from the checkpoint. Format and weight-byte formula: docs/simulation-model.md
  §2 "EXL3".
- **Activations and caches.** W8A8 / W4A4 schemes set `ModuleQuant.gemmInput`; weight-only schemes leave it unset.
  `kv_cache_scheme`, `kv_cache_quant_algo` and Mistral's `dsa.fp8_kv_cache` / `dsa.fp8_indexer` set the KV and
  indexer-key caches.
- **Family hints vs explicit metadata.** A family may return `quantHints`: formats the architecture fixes
  natively, e.g. DeepSeek-V4.1's mxfp8 window KV, nvfp4 compressed KV and mxfp4 indexer keys (`model.py`).
  `mergeQuantHints` applies a hint only to a field the repo's metadata does not set explicitly
  (`QuantDefaults.explicit`); an explicit `kv_cache_scheme` therefore wins over the hint, with a note.

## 5. Parameter cross-check (`paramCheck.ts`)

Expected = the Hub's `safetensors.total` (sum of the safetensors header element counts; it includes MTP layers and
vision/audio towers). Actual = `countParams(model) + mtpParamsEstimate + visionParamsEstimate`. Rules, in order:

1. No `safetensors` in the api info (Mistral consolidated repos) → `unavailable`.
2. **Packed quantized storage → `packed`.** The Hub counts tensor elements: a packed int4 word (`U8`, `U16`, `U32`,
   `I16`, `I32`) counts as one element although it holds 2 or 8 weights, so `safetensors.total` of a GPTQ / AWQ /
   NVFP4 / MXFP4 / MLX repo is not a parameter count (the same architecture reports 50B to 321B depending on the
   repo). `I8` counts as packed too unless `quant_method` is DeepSeek's native `fp8`, whose fp4 experts are stored
   as one `I8` per weight and whose total matches. The note suggests comparing against the base model
   (`cardData.base_model`).
3. **Per-dtype sum above the total.** When `Σ safetensors.parameters` exceeds `safetensors.total` by more than
   1e-4 (relative), the sum is the expected count, with a note. The total can come from the index's
   `total_parameters`, which leaves out MTP tensors (Nemotron-3.5-Lightning: total 31,577,937,344, per-dtype sum
   32,913,266,240), while the per-dtype counts are read from the headers. A sum below the total is not used. For
   every other shortlist repo the two are equal, so their results are unchanged.
4. Within ±2% → `ok`, else `mismatch` (a warning in the notes, not an error).

## 6. The shortlist

All 11 base repos are public except Muse Spark: `meta-models/Muse-Spark-Open-816B-A42B` is gated and is read through
its ungated mirror `unsloth/Muse-Spark-Open-816B-A42B` without a token (§1), so its fixture has the mirror's ID.
Parameters are the IR's (`countParams`, text model incl. n-gram tables, MTP and vision excluded) and active per token
incl. embedding and head (`activeParamsPerToken`); the param check compares against `safetensors.total` with the
excluded estimates added.

| Shortlist name                    | HF ID                                                                    | Family           | IR total / active                                           | Param check                               | Native format                                        |
| --------------------------------- | ------------------------------------------------------------------------ | ---------------- | ----------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| Qwen3.8-27B                       | `Qwen/Qwen3.8-27B`                                                       | `qwen`           | 26.90B / 26.90B                                             | ok                                        | BF16                                                 |
| Qwen3.8-Flash-Next                | `Qwen/Qwen3.8-Flash-Next`                                                | `qwen`           | 176.94B / 7.31B                                             | ok                                        | BF16                                                 |
| DeepSeek-V4.1-Flash               | `deepseek-ai/DeepSeek-V4.1-Flash` (`-1017` identical)                    | `deepseek_v41`   | 748.49B / 16.79B                                            | ok                                        | FP8 32×32 ue8m0, experts FP4                         |
| DeepSeek-V4-Flash-0731            | `deepseek-ai/DeepSeek-V4-Flash-0731`                                     | `deepseek_v4`    | 284.33B (304.18B incl. MTP) / 13.27B active excl. embedding | ok                                        | FP8 128×128 ue8m0, experts FP4                       |
| GLM-5.3 Flash                     | `zai-org/GLM-5.3-Flash` (`-BF16` twin)                                   | `glm5_next`      | 313.33B / 17.38B                                            | ok                                        | FP8 128×128                                          |
| GLM-5.3                           | `zai-org/GLM-5.3` (`-BF16` twin)                                         | `glm_moe_dsa`    | 743.38B / 41.25B                                            | ok                                        | FP8 128×128                                          |
| Mistral Large 4                   | `mistralai/Mistral-Large-4-1T-A52B`                                      | `mistral_large4` | 1049.36B / 52.34B                                           | unavailable (no safetensors totals)       | FP8 128×128                                          |
| Inkling 2 Small (stand-in, below) | `thinkingmachines/Inkling-Small`                                         | `inkling`        | 263.65B / 12.00B                                            | ok                                        | BF16                                                 |
| MiniMax M3                        | `MiniMaxAI/MiniMax-M3` (`M3.1-Flash` same text dims)                     | `minimax`        | 426.17B / 25.96B                                            | ok                                        | BF16                                                 |
| Nemotron 3.5 Lightning            | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`                      | `nemotron_h`     | 31.58B / 3.58B                                              | ok (against the per-dtype sum, §5; −9e-8) | BF16                                                 |
| Nemotron 3.5 Super VL             | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`                    | `nemotron_h`     | 120.67B / 12.77B                                            | packed (U8 words)                         | FP8 Mamba/attention/head, NVFP4 experts, BF16 latent |
| Muse Spark                        | `meta-models/Muse-Spark-Open-816B-A42B` (fixture: the `unsloth/` mirror) | `muse_spark`     | 819.99B / 46.09B                                            | ok                                        | BF16 (router F32)                                    |

**Inkling 2 Small.** No repo of that name exists. Thinking Machines publishes only `thinkingmachines/Inkling` and
`thinkingmachines/Inkling-Small`; `NarsAI/Inkling-2` is a byte-identical re-upload of the big Inkling. The loader is
tested on `thinkingmachines/Inkling-Small` as the closest match; any repo with the same `model_type` maps the same
way.

**Feature matrix** (from the fetched configs, reference code and safetensors headers):

| Feature                             | Q-Flash-Next          | Q-27B                 | DSV4.1-Flash                                       | GLM-5.3-Flash  | GLM-5.3        | Mistral L4                                       | Inkling-Small  | MiniMax-M3                     | Nemotron-Lightning               | Nemotron-Super                          | Muse Spark                 |
| ----------------------------------- | --------------------- | --------------------- | -------------------------------------------------- | -------------- | -------------- | ------------------------------------------------ | -------------- | ------------------------------ | -------------------------------- | --------------------------------------- | -------------------------- |
| GQA full attention                  | 1 of 4 layers, gated  | 1 of 4 layers, gated  | –                                                  | –              | –              | –                                                | ✓              | ✓                              | 6 of 52 layers, no RoPE          | 8 of 88 layers, no RoPE                 | 16 of 62, NoPE, gated      |
| Linear attention                    | GatedDeltaNet, 3 of 4 | GatedDeltaNet, 3 of 4 | –                                                  | KDA, 3 of 4    | –              | –                                                | –              | –                              | Mamba-2 (SSM), 23 of 52          | Mamba-2 (SSM), 40 of 88                 | –                          |
| MLA                                 | –                     | –                     | CSA (one K=V vector, q-LoRA, grouped low-rank out) | NoPE           | ✓              | no q-LoRA, NoPE every 4th                        | –              | –                              | –                                | –                                       | –                          |
| Sparse top-k (indexer)              | QSA, keys pooled 4:1  | –                     | compressed + hierarchical                          | DSA, k-pool 4  | DSA            | DSA                                              | –              | block top-k                    | –                                | –                                       | –                          |
| Top-k reused across layers          | –                     | –                     | 30 of 40                                           | –              | 57 of 78       | 28 of 72                                         | –              | –                              | –                                | –                                       | –                          |
| Cross-layer KV sharing              | –                     | –                     | 4 source layers                                    | –              | –              | 9 layers                                         | –              | –                              | –                                | –                                       | –                          |
| Sliding window                      | –                     | –                     | 128 + global top-k                                 | –              | –              | –                                                | 35 of 42 (512) | –                              | –                                | –                                       | 46 of 62 (2048), gated     |
| KV compression                      | –                     | –                     | r = 1 or 2                                         | –              | –              | –                                                | –              | –                              | –                                | –                                       | –                          |
| Hyper-connections                   | 4 streams, low-rank   | –                     | mHC, 4 streams                                     | mHC, 4 streams | –              | –                                                | –              | –                              | –                                | –                                       | – (learned residual gates) |
| n-gram / Engram tables              | 51B, 1 layer          | –                     | 196B, 2 layers                                     | –              | –              | –                                                | –              | –                              | –                                | –                                       | –                          |
| Relative-position bias, short convs | –                     | –                     | –                                                  | –              | –              | –                                                | ✓              | –                              | –                                | –                                       | –                          |
| MoE experts / top-k / shared        | 512 / 10 / 1          | dense                 | 384 / 6 / 1                                        | 288 / 8 / 1    | 256 / 8 / 1    | 128 / 4 / 1                                      | 256 / 6 / 2    | 128 / 4 / 1                    | 128 / 6 / 1 (relu², ungated)     | 512 / 22 / 1 (relu², latent width 1024) | 256 / 8 / 0, latent 4096   |
| Dense first layers                  | –                     | all                   | –                                                  | 3              | 3              | 1                                                | 2              | 3                              | –                                | –                                       | –                          |
| MTP / draft (`model.draft`, §7)     | 1, multi-step         | 1, multi-step         | DSpark block drafter, 3 MoE layers                 | 1, single-step | 1, single-step | none (3-layer DSpark in `params.json`, deferred) | 8, single-step | none (config only, no weights) | 1, single-step (attention + MoE) | 1, single-step (attention + MoE)        | none                       |
| Vision / audio (excluded)           | ✓                     | ✓                     | ✓                                                  | ✓              | –              | ✓                                                | vision + audio | ✓                              | –                                | vision (RADIO ViT + mlp1)               | vision + audio             |

**DeepSeek-V4.1 prefill.** The reference `inference/model.py` runs all 40 layers on all N prompt tokens, and so does
the simulator. The tech report's "Decoder SWA Bounded Replay" (the decoder half re-runs only the last window of the
prompt), which its "8B active in prefill" figure assumes, is a deployment optimization and is left out with the
other inference-engine knobs (docs/simulation-model.md §9).

**DeepSeek-V4 (Flash-0731).** Same builder as V4.1. With no `kv_source_layer_ids` / `index_source_layer_ids` in
the config, every r > 0 layer owns its compressor, r = 4 layers (21) run an indexer whose keys come from its own
overlapping compressor over the hidden state, and r = 128 layers (20) attend every compressed entry
(`CsaSelection {kind: 'all'}`). Layers 0-2 route experts by token id (`tid2eid`); the router GEMM still runs. No
Engram tables, so no lookup-table offload. Prefill runs all 43 layers on all prompt tokens. The checkpoint has 3
DSpark stages although `num_nextn_predict_layers` is 1 (`compress_ratios` has 43 + 3 entries, inference/config.json
`n_mtp_layers` 3, and every safetensors dtype bucket reconciles only with 46 blocks); the loader takes
max(`num_nextn_predict_layers`, `len(compress_ratios)` − `num_hidden_layers`) and notes the disagreement. Native
caches (model.py QAT): window and compressed KV fp8 with bf16 RoPE dims and a ue8m0 scale per 64 (583 B per
512-element vector), indexer keys mxfp4. Tech report: 284B total / 13B activated.

**Tests.** `src/sim/models/hf/families/*.test.ts` check each family against the safetensors headers (total and
active parameters, per-layer counts, layer roles, sibling checkpoints); `quant/*.test.ts` the derived defaults of
every fixture repo; `shortlist.e2e.test.ts` runs every shortlist fixture through `mapHfModel` and `simulate`
(prefill and decode, 1 and 8 RTX 5090s) and checks the per-token KV growth against hand derivations (Qwen3.8-27B
65,536 B, DeepSeek-V4.1-Flash 890 B as its README states, GLM-5.3 78 latents + 21 indexer keys). `museSpark.test.ts`
also decodes Muse Spark @ 8k on 4 × M3 Ultra 512 GB (TB5 mesh, pp = 4) and checks it fits and reaches ≥ 0.8× of the
memory-bandwidth bound (≈ 9.1 tok/s @ 8k).

## 7. Drafters

Speculative decoding (docs/simulation-model.md §13) needs a drafter. Two sources, both mapped to the IR's
`DraftSpec` (`src/sim/ir/model.ts`) and never part of the target model's counts.

**The checkpoint's own MTP module** (`src/sim/models/hf/drafter.ts: mtpDraft, withDraft`). A family that already
builds the MTP decoder layer for its param estimate keeps it as `model.draft = { kind: 'mtp', layers, hiddenProj:
true }`, one layer per module named `mtp.<j>` (modules run as a chain: pass p uses module min(p, L)).
`maxReliableDepth` is set for single-step heads (DeepSeek-V3-style `num_nextn_predict_layers` modules, one draft
position each) and absent for heads trained for several positions (Qwen3.5+/3.8 `mtp_num_hidden_layers`).

| Family                     | Config field                                                                                                           | Draft                                                                                           | Reliable depth   |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------- |
| `qwen`                     | `mtp_num_hidden_layers`                                                                                                | MTP, the target's full-attention layer shape                                                    | any (multi-step) |
| `glm_moe_dsa`, `glm5_next` | `num_nextn_predict_layers`                                                                                             | MTP                                                                                             | module count     |
| `inkling`                  | `mtp_config.num_nextn_predict_layers` (dense MLPs; `local_layer_ids` windowed)                                         | MTP                                                                                             | module count     |
| `deepseek_v41`             | `num_nextn_predict_layers` + `dspark_*`                                                                                | DSpark block drafter (below), the `mtp.*` stages as `moeLayers`                                 | —                |
| `deepseek_v4`              | max(`num_nextn_predict_layers`, `len(compress_ratios)` − `num_hidden_layers`) + `dspark_*` (V4-Flash-0731: 1 vs 3 → 3) | DSpark block drafter, as V4.1 (main MoE 256 / top-6 in the stages)                              | —                |
| `mistral_large4`           | `params.json` `mtp`                                                                                                    | none: deferred (`params.json` lacks the drafter's head counts and dims; its head type is `rnn`) | —                |
| `minimax`                  | MTP config fields                                                                                                      | none: the released checkpoint has no MTP tensors                                                | —                |
| `muse_spark`               | –                                                                                                                      | none: no MTP fields or tensors                                                                  | —                |

**Block drafters** (DFlash 2, DSpark; `drafter.ts: parseDrafterConfig`; fetched by `src/hf/loadDrafter.ts`). A
separate repo's `config.json` → `BlockDraftSpec`:

| `BlockDraftSpec`                                                                         | config.json                                                                                                                                                          | Note                                                                                                                                         |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`                                                                                 | `architectures[0]`: `DFlash2DraftModel` → `dflash2`, `Qwen3DSparkModel` → `dspark`                                                                                   | anything else is `invalid_config`                                                                                                            |
| `hidden`, `vocab`, `layers`, `heads`, `kvHeads`, `headDim`, `intermediate`, `activation` | `hidden_size`, `vocab_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` (default hidden/heads), `intermediate_size`, `hidden_act` | gated MLP                                                                                                                                    |
| `windowTokens`                                                                           | `sliding_window` when every `layer_types` entry is `sliding_attention`                                                                                               | mixed layer types are rejected                                                                                                               |
| `targetLayerIds`                                                                         | `dflash_config.target_layer_ids`, else top-level, else DFlash's `build_target_layer_ids(L, n)` = `round(1 + i·(L − 4)/(n − 1))` (half-to-even)                       | zero-based target layers                                                                                                                     |
| `blockSize`                                                                              | DFlash: `block_size` (default 16); DSpark: `block_size + 1`                                                                                                          | DFlash's block counts the anchor (k ≤ block_size − 1); DeepSpec's counts draft positions (k ≤ block_size). Qwen3-8B DSpark: blockSize 8, k 7 |
| `conv`, `selector`                                                                       | DFlash 2 `conv_kernel_size`, `conv_group_size`, `selector_top_k`, `selector_rank`                                                                                    |                                                                                                                                              |
| `markovRank`                                                                             | DSpark `markov_rank`                                                                                                                                                 |                                                                                                                                              |

DeepSeek-V4.1-Flash's in-checkpoint DSpark (`deepseekV41.ts: dsparkDraft`) maps the same way from `dspark_block_size`
(blockSize = it + 1: 6, default k 5), `dspark_target_layer_ids`, `dspark_markov_rank` and the CSA `sliding_window`,
with the `mtp.*` stages as `moeLayers`. `KNOWN_DRAFTERS` lists the released target → drafter repos the UI offers
(`Qwen/Qwen3.8-27B` → `incoai/Qwen3.8-27B-DFlash2`, `Qwen/Qwen3-8B` → `deepseek-ai/dspark_qwen3_8b_block7`);
`hypotheticalDrafter` is the labelled stand-in for any other target (5 layers, 32 / 8 heads, hd 128, window 2048,
the target's h, V and FFN width). Fixtures: `src/sim/models/hf/__fixtures__/`.

**Layer indexing of the n-gram tables** (checked against the modelling code; design doc §2.10). Qwen
`ple_layer_ids` is **one-based** (`configuration_qwen4_exp.py`: "One-indexed decoder layer ids";
`modeling_qwen4_exp.py` `config.ple_layer_ids.index(layer_idx + 1)`): config `[2]` is layer index 1, the only layer
with `layers.1.ple.*` tensors (`qwen.ts` tests `pleLayers.includes(i + 1)`). DeepSeek `engram_layer_ids` `[1, 14]` is
**zero-based** (`inference/model.py`; SGLang `deepseek_v4.py`): tensors `layers.1.engram.*` and `layers.14.engram.*`
(`deepseekV41.ts` uses the ids as indices).
