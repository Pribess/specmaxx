# Hugging Face fixtures

Real Hub files for the model shortlist, used by the `src/sim/models/hf/` tests (no network in tests).
Load them with `fixtures.testutil.ts` (`fixtureRepoFiles('Qwen/Qwen3.8-27B')`), never with a JSON
`import`: the loader reads raw text and `JSON.parse`s it, so TypeScript does not infer types for large files.

## Naming

`<org>__<repo>.<file>.json`, with `/` in the repo ID written as `__`:

| `<file>` | Source | Kept |
|---|---|---|
| `config` | `https://huggingface.co/<id>/resolve/main/config.json` | verbatim |
| `api` | `https://huggingface.co/api/models/<id>` | trimmed (below) |
| `params` | `params.json` (Mistral consolidated format) | verbatim |
| `hf_quant_config` | `hf_quant_config.json` (NVIDIA ModelOpt) | verbatim |
| `quantization_config` | `quantization_config.json` (AutoRound / Quark exports) | verbatim |

Quantized-repo fixtures (work package C) go in `quant/` with the same naming.

## Trimming rules for `*.api.json`

Keep only what the mapping reads:

- `id`, `sha`, `gated`, `private`, `tags`
- `cardData.base_model` (and nothing else of `cardData`)
- `safetensors.parameters` (counts by dtype) and `safetensors.total`
- `siblings`, filtered to `config.json`, `params.json`, `hf_quant_config.json`, `quantization_config.json`
  (the fetched responses had no `siblings`, so none of these fixtures carries it)

Everything else (`config`, `spaces`, `widgetData`, downloads, likes, dates, …) is dropped.
The trimming was done once by a throwaway script outside the repo; to refresh a fixture, fetch both URLs
again, keep `config.json` as is and apply the rules above.

## Base-model fixtures

Copied from the research of the HF-loader task (base repos only; quantized repos are C's, in `quant/`).

| Repo | Files | Notes |
|---|---|---|
| `Qwen/Qwen3.8-27B` | config, api | `qwen3_5`, dense hybrid GDN + gated attention |
| `Qwen/Qwen3.8-Flash-Next` | config, api | `qwen4_exp`, MoE + QSA + PLE/n-gram |
| `deepseek-ai/DeepSeek-V4.1-Flash` | config, api | `deepseek_v41`; native fp8 + fp4 experts (`I8` unpacked) |
| `deepseek-ai/DeepSeek-V4.1-Flash-1017` | config, api | sibling; must build identically |
| `zai-org/GLM-5.3-Flash` | config, api | `glm5_next`; native fp8 (large `quantization_config`) |
| `zai-org/GLM-5.3-Flash-BF16` | config, api | BF16 sibling |
| `zai-org/GLM-5.3` | config, api | `glm_moe_dsa`, flat config; native fp8 |
| `zai-org/GLM-5.3-BF16` | config, api | BF16 sibling |
| `mistralai/Mistral-Large-4-1T-A52B` | config, api, params | `mistral_large4`; no `safetensors` in api (consolidated) |
| `thinkingmachines/Inkling-Small` | config, api | `inkling_mm_model`; `text_config` without `model_type` |
| `MiniMaxAI/MiniMax-M3` | config, api | `minimax_m3_vl`; `text_config` without `model_type` |
| `MiniMaxAI/MiniMax-M3.1-Flash` | config, api | sibling (all layers sparse); `mxfp8` with `U8` storage, so the param check says `packed` |
