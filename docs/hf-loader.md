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

`mapHfModel` never throws for bad input. It returns `ok` (model, family, quant defaults, excluded parts, param check,
notes), `unsupported` (unknown `model_type` / architecture, with the supported list) or `invalid_config`
(path-qualified errors such as `text_config.hidden_size: missing number`).

## 1. Fetching (`src/hf/`)

- **Files.** `GET https://huggingface.co/api/models/<id>[/revision/<rev>]` for the metadata (`safetensors`
  parameter counts by dtype, tags, `cardData.base_model`, `siblings`), then
  `https://huggingface.co/<id>/resolve/<rev>/config.json`. Both allow CORS; `resolve` answers with a 307 to
  `/api/resolve-cache/…`, which `fetch` follows. The sidecars `hf_quant_config.json` (ModelOpt),
  `quantization_config.json` (some AutoRound / Quark exports) and `params.json` (Mistral consolidated format)
  are fetched only when `siblings` lists them, so there is no 404 probing.
- **Repos without config.json.** Mistral NVFP4 repos ship only `params.json`. When `config.json` is 404 and the
  repo has a `params.json` and a `cardData.base_model`, the base model's `config.json` is used (`configFrom`, noted
  in the result). The quant defaults still come from the repo's own `params.json`.
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

Multimodal checkpoints nest the language model under `text_config` and keep `architectures`, `model_type` and
`vision_config` (sometimes also `quantization_config`) at the top. The simulator models the language model only;
vision and audio towers are excluded and their parameters are estimated for the param check
(`estimateVisionParams`, or a family's own estimate). The result's notes say "language model only; vision/audio
towers excluded". Flat (text-only) configs are used as they are. `vision_config.model_type` is never used for
dispatch (Qwen4Exp's equals the top-level type).

## 3. Dispatch and families

Keys are tried in this order; the first one a family claims wins: top-level `architectures[0]`, top-level
`model_type`, `text_config.architectures[0]`, `text_config.model_type`. Architectures come first because they
name the model class; model types second because text-only extractions often drop `architectures`.

| Family (`families/…`) | `model_type` | `architectures[0]` | Shortlist models |
|---|---|---|---|
| `qwen` (`qwen.ts`) | `qwen3_5`, `qwen3_5_text`, `qwen4_exp`, `qwen4_exp_text` | `Qwen3_5ForConditionalGeneration`, `Qwen4ExpForConditionalGeneration` | Qwen3.8-27B, Qwen3.8-Flash-Next |
| `deepseek_v41` (`deepseekV41.ts`) | `deepseek_v41`, `deepseek_v41_text` | `DeepseekV41ForCausalLM` | DeepSeek-V4.1-Flash (and `-1017`) |
| `glm5_next` (`glm5Next.ts`) | `glm5_next`, `glm5_next_text` | `Glm5NextForConditionalGeneration` | GLM-5.3-Flash |
| `glm_moe_dsa` (`glmMoeDsa.ts`) | `glm_moe_dsa` | `GlmMoeDsaForCausalLM` | GLM-5.3 (also GLM-5) |
| `mistral_large4` (`mistralLarge4.ts`) | `mistral_large4`, `mistral_large4_text` | `MistralLarge4ForConditionalGeneration` | Mistral Large 4 |
| `inkling` (`inkling.ts`) | `inkling_mm_model`, `inkling_text` | `InklingForConditionalGeneration` | Inkling-Small |
| `minimax` (`minimax.ts`) | `minimax_m3_vl`, `minimax_m3_vl_text` | `MiniMaxM3SparseForConditionalGeneration`, `MiniMaxM3SparseForCausalLM` | MiniMax-M3 (and M3.1-Flash) |

The MLA/DSA families share their readers in `families/mlaCommon.ts` (MLA dims from `qk_nope_head_dim` /
`qk_rope_head_dim` / `v_head_dim` — never `head_dim` or `num_key_value_heads`, which are traps in GLM and Mistral
configs —, the DSA indexer, the `indexer_types` / `mlp_layer_types` schedules with the transformers fallbacks, MoE,
mHC hyper-connections, MTP estimates).

**Rules every family follows.**
- The mapping reads config fields, never model names, so sibling checkpoints of the same architecture (BF16
  twins, later revisions, fine-tunes) map too. Per-layer roles come from config lists (`layer_types`,
  `mlp_layer_types`, `indexer_types`, `compress_ratios`, `kv_source_layer_ids`, `kv_source_layers`,
  `no_rope_layers`, …), with the fallbacks transformers uses when a list is absent.
- MTP / next-n / draft layers and vision or audio towers are excluded from the IR; their parameters are estimated
  (`mtpParamsEstimate`, `visionParamsEstimate`, `excludedParts`) only for the param check.
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
else `text_config`, dispatched on `quant_method`: `fp8`, `modelopt`, `compressed-tensors`, `quark`, `gptq`, `awq`,
`auto-round`, `bitsandbytes`; a `bits` + `group_size` object without `quant_method` is MLX), `hf_quant_config.json`,
`quantization_config.json`, the MLX `quantization` key, `params.json` `quantization` (native Mistral fp8), and
finally the model's `torch_dtype` / `dtype` (unquantized).

- **Per module class.** Checkpoints quantize module classes differently (NVFP4 routed experts with FP8 attention
  and BF16 hyper-connections; BF16 `kv_b` with FP8 q/kv_a/o). Each reader is asked about a fixed set of probes
  (one synthetic module name per class, layer and projection, in the checkpoint's own naming; `moduleClass.ts`)
  and `assemble.ts` takes the majority per `WeightModule` class into `QuantConfig.modules`, reporting splits and
  unmatched ignore-list entries as notes.
- **Scale overheads.** MX and NV formats carry their block scale in the dtype; every other scale or zero point is
  amortized into `extraBitsPerElem` (block scale `s/(r·c)`, group scale `(s + z)/g`). The full table of schemes,
  overheads and readers is in docs/simulation-model.md §2 "Quant overheads".
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
3. Within ±2% → `ok`, else `mismatch` (a warning in the notes, not an error).

## 6. The shortlist

All 8 base repos are public (not gated). Parameters are the IR's (`countParams`, text model incl. n-gram tables,
MTP and vision excluded) and active per token incl. embedding and head (`activeParamsPerToken`); the param check
compares against `safetensors.total` with the excluded estimates added.

| Shortlist name | HF ID | Family | IR total / active | Param check | Native format |
|---|---|---|---|---|---|
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` | `qwen` | 26.90B / 26.90B | ok | BF16 |
| Qwen3.8-Flash-Next | `Qwen/Qwen3.8-Flash-Next` | `qwen` | 176.94B / 7.31B | ok | BF16 |
| DeepSeek-V4.1-Flash | `deepseek-ai/DeepSeek-V4.1-Flash` (`-1017` identical) | `deepseek_v41` | 748.49B / 16.79B | ok | FP8 32×32 ue8m0, experts FP4 |
| GLM-5.3 Flash | `zai-org/GLM-5.3-Flash` (`-BF16` twin) | `glm5_next` | 313.33B / 17.38B | ok | FP8 128×128 |
| GLM-5.3 | `zai-org/GLM-5.3` (`-BF16` twin) | `glm_moe_dsa` | 743.38B / 41.25B | ok | FP8 128×128 |
| Mistral Large 4 | `mistralai/Mistral-Large-4-1T-A52B` | `mistral_large4` | 1049.36B / 52.34B | unavailable (no safetensors totals) | FP8 128×128 |
| Inkling 2 Small (stand-in, below) | `thinkingmachines/Inkling-Small` | `inkling` | 263.65B / 12.00B | ok | BF16 |
| MiniMax M3 | `MiniMaxAI/MiniMax-M3` (`M3.1-Flash` same text dims) | `minimax` | 426.17B / 25.96B | ok | BF16 |

**Inkling 2 Small.** No repo of that name exists. Thinking Machines publishes only `thinkingmachines/Inkling` and
`thinkingmachines/Inkling-Small`; `NarsAI/Inkling-2` is a byte-identical re-upload of the big Inkling. The loader is
tested on `thinkingmachines/Inkling-Small` as the closest match; any repo with the same `model_type` maps the same
way.

**Feature matrix** (from the fetched configs, reference code and safetensors headers):

| Feature | Q-Flash-Next | Q-27B | DSV4.1-Flash | GLM-5.3-Flash | GLM-5.3 | Mistral L4 | Inkling-Small | MiniMax-M3 |
|---|---|---|---|---|---|---|---|---|
| GQA full attention | 1 of 4 layers, gated | 1 of 4 layers, gated | – | – | – | – | ✓ | ✓ |
| Linear attention | GatedDeltaNet, 3 of 4 | GatedDeltaNet, 3 of 4 | – | KDA, 3 of 4 | – | – | – | – |
| MLA | – | – | CSA (one K=V vector, q-LoRA, grouped low-rank out) | NoPE | ✓ | no q-LoRA, NoPE every 4th | – | – |
| Sparse top-k (indexer) | QSA, keys pooled 4:1 | – | compressed + hierarchical | DSA, k-pool 4 | DSA | DSA | – | block top-k |
| Top-k reused across layers | – | – | 30 of 40 | – | 57 of 78 | 28 of 72 | – | – |
| Cross-layer KV sharing | – | – | 4 source layers | – | – | 9 layers | – | – |
| Sliding window | – | – | 128 + global top-k | – | – | – | 35 of 42 (512) | – |
| KV compression | – | – | r = 1 or 2 | – | – | – | – | – |
| Hyper-connections | 4 streams, low-rank | – | mHC, 4 streams | mHC, 4 streams | – | – | – | – |
| n-gram / Engram tables | 51B, 1 layer | – | 196B, 2 layers | – | – | – | – | – |
| Relative-position bias, short convs | – | – | – | – | – | – | ✓ | – |
| MoE experts / top-k / shared | 512 / 10 / 1 | dense | 384 / 6 / 1 | 288 / 8 / 1 | 256 / 8 / 1 | 128 / 4 / 1 | 256 / 6 / 2 | 128 / 4 / 1 |
| Dense first layers | – | all | – | 3 | 3 | 1 | 2 | 3 |
| MTP (excluded) | 1 | 1 | 3 (DSpark) | 1 | 1 | 3-layer draft in `params.json` | 8 | config only, no weights |
| Vision / audio (excluded) | ✓ | ✓ | ✓ | ✓ | – | ✓ | vision + audio | ✓ |

**DeepSeek-V4.1 prefill.** The reference `inference/model.py` runs all 40 layers on all N prompt tokens, and so does
the simulator. The tech report's "Decoder SWA Bounded Replay" (the decoder half re-runs only the last window of the
prompt), which its "8B active in prefill" figure assumes, is a deployment optimization and is left out with the
other inference-engine knobs (docs/simulation-model.md §9).

**Tests.** `src/sim/models/hf/families/*.test.ts` check each family against the safetensors headers (total and
active parameters, per-layer counts, layer roles, sibling checkpoints); `quant/*.test.ts` the derived defaults of
every fixture repo; `shortlist.e2e.test.ts` runs every shortlist fixture through `mapHfModel` and `simulate`
(prefill and decode, 1 and 8 RTX 5090s) and checks the per-token KV growth against hand derivations (Qwen3.8-27B
65,536 B, DeepSeek-V4.1-Flash 890 B as its README states, GLM-5.3 78 latents + 21 indexer keys).
