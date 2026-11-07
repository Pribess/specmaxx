# Quantized-repo fixtures (work package C)

Real Hub files of quantized variants of the shortlist models, used by `src/sim/models/hf/quant/*.test.ts`.
Naming and loading as in `../README.md` (`fixtureRepoFiles(id, { dir: 'quant' })`; a repo without its own
config falls back to the base fixture in `../`, e.g. `configFrom: 'mistralai/Mistral-Large-4-1T-A52B'`).
Copied from the HF-loader research (`config.json`, `hf_quant_config.json`, `params.json` as fetched); no
`api.json` files (the quant derivation does not read them).

| Repo | Files | Scheme |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | config | native fp8, block 128×128, `modules_to_not_convert` |
| `nvidia/Qwen3.8-27B-NVFP4` | config | ModelOpt MIXED_PRECISION (FP8 attention/GDN, NVFP4 MLP + lm_head) |
| `unsloth/Qwen3.8-27B-NVFP4` | config | compressed-tensors mixed-precision (NVFP4 MLP, FP8 per-channel attention, layers 56–63 MLP FP8) |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | config | compressed-tensors pack-quantized int4 asym g32, int8 zp |
| `RedHatAI/Qwen3.8-27B-INT4` | config | compressed-tensors pack-quantized int4 sym g128, fp8 `kv_cache_scheme` |
| `RedHatAI/Qwen3.8-27B-MXFP4` | config | compressed-tensors mxfp4-pack-quantized W4A4 |
| `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` | config | Quark fp4 g32 e8m0 (MXFP4 W4A4) |
| `sokada4/Qwen3.8-27B-GPTQ-Int4` | config | GPTQ 4-bit g128 sym |
| `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` | config | bitsandbytes nf4 + double quant, fp16 compute |
| `Qwen/Qwen3.8-Flash-Next-FP8` | config | native fp8, `modules_to_convert` (PLE n-gram table) |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | config, hf_quant_config | ModelOpt MIXED_PRECISION (NVFP4 experts, FP8 n-gram table) |
| `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` | config (trimmed) | AutoRound int4 g128 sym, `auto_round:auto_gptq`, `extra_config` |
| `btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit` | config | GPTQ 4-bit g32 |
| `nvidia/DeepSeek-V4.1-Flash-NVFP4` | config | `quant_method: fp8` but ModelOpt: NVFP4 experts over native fp8 32×32 ue8m0 |
| `amd/DeepSeek-V4.1-Flash-Quark-MXFP4` | config (trimmed) | Quark MXFP4 + fp8 32×32 `layer_quant_config` + `exclude` |
| `intel-ai/DeepSeek-V4.1-Flash-MXFP4-Mixed-CT-AutoRound` | config (trimmed) | compressed-tensors mixed: mxfp4 experts, mxfp8 rest |
| `INCModel3/DeepSeek-V4.1-Flash-W4A16-Engram-AutoRound` | config | AutoRound int4 g32 |
| `MiniMaxAI/MiniMax-M3-MXFP8` | config | `quant_method: mxfp8`, block [1, 32] |
| `cyankiwi/MiniMax-M3-AWQ-INT4` | config | compressed-tensors int4 asym g32 (regex ignores) |
| `EmbeddedLLM/MiniMax-M3-FP8-dynamic` | config | compressed-tensors float-quantized fp8 per-channel, per-token |
| `nvidia/GLM-5.3-Flash-FP4-Local` | config | ModelOpt MIXED_PRECISION (NVFP4, MXFP8, FP8_PB_WO, W4A16_NVFP4) |
| `nvidia/GLM-5.3-Flash-NVFP4` | config | ModelOpt NVFP4 + fp8 `kv_cache_scheme` |
| `thinkingmachines/Inkling-Small-NVFP4` | config, hf_quant_config | config without `quantization_config`; ModelOpt NVFP4 in hf_quant_config.json |
| `mlx-community/Inkling-Small-mlx-4bit` | config | MLX `quantization` affine 4-bit g64, `recipe: experts_only` |
| `mlx-community/Inkling-Small-mxfp4` | config | MLX mode mxfp4 |
| `mlx-community/Mistral-Large-4-1T-A52B-Text-4bit` | config | MLX affine 4-bit g64 |
| `mistralai/Mistral-Large-4-1T-A52B-NVFP4` | params | no config.json; ModelOpt MIXED_PRECISION in params.json |
| `amd/Mistral-Large-4-1T-A52B-Quark-MXFP4` | config (trimmed) | Quark MXFP4 + fp8 128×128 `layer_quant_config` + `exclude` |

## Trimming

Files over ~150 KB were trimmed by a throwaway script outside the repo, keeping what the derivation reads; each
trimmed config carries a `quantization_config._fixture_note` saying what was removed:

- AutoRound (`Intel/…`): `extra_config` entries for `mtp.*` removed (MTP is not in the IR).
- Quark (`amd/DeepSeek…`, `amd/Mistral…`): each quant spec reduced to `dtype, is_dynamic, qscheme, ch_axis,
  group_size, block_size, symmetric, scale_type, scale_format, zero_point_type` (null keys dropped); vision
  entries of `exclude` / `layer_quant_config` removed. Written as compact JSON.
- compressed-tensors (`intel-ai/DeepSeek…`): the 47k explicit expert targets of `group_0` reduced to layers
  0–3, experts 0–1; the `RoutedExperts` class target, which covers all experts, is kept. Vision ignore entries
  removed.
