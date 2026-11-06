# Ragbot Sizing Matrix — Open-Weights Models on Apple Silicon

This document maps each open-weights model in Ragbot's `engines.yaml` to the Mac hardware that runs it, with memory footprints and quantization guidance. It answers one question: *will this model run usefully on my Mac, and at what precision?*

The lineup and every size below were verified on November 6, 2026 against the Ollama library tag pages and the model cards (sources in [`model-catalog-2026-11-06.yaml`](model-catalog-2026-11-06.yaml)). "Default tag" is the published Ollama download size of the exact tag `engines.yaml` names (Q4_K_M for most models); FP16 and Q8 are derived from parameter counts (about 2 and 1.08 bytes per parameter). Ollama on Apple Silicon is the assumed runtime.

---

## How to read the matrix

### Quantization quality vs. memory

- **FP16 / BF16** — the training precision. No quality loss. ~2 bytes per parameter.
- **Q8_0** — 8-bit. Quality loss below the noise floor for most workloads. ~1 byte per parameter.
- **Q4_K_M** — 4-bit, the default Ollama tag for most models. Noticeable but acceptable loss for chat, retrieval-augmented Q&A and summarization; some loss on code, math and long-context reasoning. ~0.5-0.6 bytes per parameter.

Two caveats:

1. **MoE routing is more quantization-sensitive than dense layers.** The router picks experts from weight distributions; lower precision makes it pick wrong experts more often. Moving from Q4 to Q8 helps the MoE models here (Gemma 4 26B, Nemotron 3.5 Super, Qwen3.8 Flash-Next) more than the dense ones.
2. **Unified memory is shared.** macOS and your other apps need their share. Each hardware section below states the LLM budget it assumes, roughly 70-90% of unified memory depending on size.

### Verdicts

Each (hardware, model) cell is computed, not estimated by hand:

- **comfortable (P)** — the model at precision P, plus 25% headroom for an 8K-token KV cache, fits the LLM budget. The highest precision that qualifies is shown.
- **tight (default tag)** — the Q4 download fits the budget, but with less than 25% headroom. Workable on a Mac dedicated to the model.
- **won't fit** — the Q4 download alone exceeds the budget.

### KV cache memory

The KV cache grows linearly with context length. As a rough guide at FP16 KV: 8K context 0.5-2 GB, 32K 2-8 GB, 128K 8-32 GB, 1M 64-256 GB. The verdicts assume an 8K working context; long-context work needs the headroom, or a quantized KV cache (below).

### Memory bandwidth and decode speed

Decode speed on Apple Silicon is bound by memory bandwidth: `tokens/sec ≈ bandwidth_GB_per_s / bytes read per token`. A dense model reads all its weights per token; an MoE model reads only its active experts, which is why Gemma 4 26B (3.8B active) and Qwen3.8 Flash-Next (6B active) decode far faster than their total size suggests, while still needing every weight in memory.

| Chip | Unified memory bandwidth |
|---|---|
| M4 | 120 GB/s |
| M4 Pro | 273 GB/s |
| M4 Max | 546 GB/s |
| M5 Max | 614 GB/s |
| M3 Ultra | 800 GB/s |

---

## Per-hardware sections

### Mac mini

#### Mac mini 16 GB (M4)

- **Unified memory:** 16 GB · **LLM budget assumed:** ~10 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | tight (default tag) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (default tag) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | won't fit |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | won't fit |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | won't fit |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | won't fit |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | won't fit |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | won't fit |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | won't fit |

#### Mac mini 32 GB (M4 Pro)

- **Unified memory:** 32 GB · **LLM budget assumed:** ~24 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (FP16) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (Q8) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | comfortable (default tag) |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | comfortable (default tag) |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | comfortable (default tag) |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | comfortable (default tag) |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | won't fit |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | won't fit |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | won't fit |

#### Mac mini 64 GB (M4 Pro)

- **Unified memory:** 64 GB · **LLM budget assumed:** ~56 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (FP16) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (FP16) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | comfortable (Q8) |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | comfortable (Q8) |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | comfortable (Q8) |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | comfortable (Q8) |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | won't fit |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | won't fit |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | won't fit |

### MacBook Air

#### MacBook Air 24 GB (M5)

- **Unified memory:** 24 GB · **LLM budget assumed:** ~16 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (Q8) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (default tag) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | won't fit |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | won't fit |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | won't fit |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | won't fit |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | won't fit |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | won't fit |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | won't fit |

#### MacBook Air 32 GB (M5)

- **Unified memory:** 32 GB · **LLM budget assumed:** ~24 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (FP16) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (Q8) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | comfortable (default tag) |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | comfortable (default tag) |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | comfortable (default tag) |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | comfortable (default tag) |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | won't fit |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | won't fit |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | won't fit |

### MacBook Pro

#### MacBook Pro 48 GB (M4 Pro)

- **Unified memory:** 48 GB · **LLM budget assumed:** ~38 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (FP16) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (FP16) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | comfortable (Q8) |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | comfortable (Q8) |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | comfortable (default tag) |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | comfortable (default tag) |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | won't fit |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | won't fit |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | won't fit |

#### MacBook Pro 128 GB (M5 Max)

- **Unified memory:** 128 GB · **LLM budget assumed:** ~110 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (FP16) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (FP16) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | comfortable (FP16) |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | comfortable (FP16) |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | comfortable (FP16) |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | comfortable (FP16) |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | comfortable (default tag) |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | tight (default tag) |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | won't fit |

### Mac Studio

#### Mac Studio 192 GB (M3 Ultra)

- **Unified memory:** 192 GB · **LLM budget assumed:** ~175 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (FP16) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (FP16) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | comfortable (FP16) |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | comfortable (FP16) |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | comfortable (FP16) |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | comfortable (FP16) |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | comfortable (Q8) |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | comfortable (Q8) |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | comfortable (Q8) |

#### Mac Studio 256 GB (M3 Ultra)

- **Unified memory:** 256 GB · **LLM budget assumed:** ~240 GB

| Model | Params | FP16 | Q8 | Default tag | Verdict |
|---|---|---|---|---|---|
| Gemma 4 E4B | 8B | 16 GB | 9 GB | 9.6 GB | comfortable (FP16) |
| Gemma 4 12B | 11.95B | 24 GB | 13 GB | 8 GB | comfortable (FP16) |
| Gemma 4 26B MoE | 25.2B (3.8B active) | 50 GB | 27 GB | 17 GB | comfortable (FP16) |
| Qwen3.8 27B | 27B | 54 GB | 29 GB | 18 GB | comfortable (FP16) |
| Muse Glimmer 30B | 29.6B | 59 GB | 32 GB | 18 GB | comfortable (FP16) |
| Gemma 4 31B | 30.7B | 61 GB | 33 GB | 19 GB | comfortable (FP16) |
| Mistral Medium 3.5 128B | 128B | 256 GB | 138 GB | 80 GB | comfortable (Q8) |
| Nemotron 3.5 Super 120B-A12B | 120B (12B active) | 240 GB | 130 GB | 90 GB | comfortable (Q8) |
| Qwen3.8 Flash-Next 125B-A6B (preview) | 125B (6B active) | 250 GB | 135 GB | 120 GB | comfortable (Q8) |

---

## Per-model sections

### Gemma 4 E4B

- **Ollama tag:** `gemma4:e4b` · **Tier:** bulk · **License:** Apache 2.0
- **Parameters:** 8B · **Context:** 131,072 tokens
- **Memory:** default tag 9.6 GB (`ollama list`), Q8 ~9 GB, FP16 ~16 GB

4.5B effective parameters (8B including embeddings). The default tag is 9.6 GB; Ollama also lists smaller E4B tags from 6.6 GB, which make it comfortable on a 16 GB Mac.

### Gemma 4 12B

- **Ollama tag:** `gemma4:12b` · **Tier:** bulk · **License:** Apache 2.0
- **Parameters:** 11.95B · **Context:** 262,144 tokens
- **Memory:** default tag 8 GB (Ollama tags (7.7-8.0 GB)), Q8 ~13 GB, FP16 ~24 GB

Gemma 4 12B Unified (2026-06-03). Dense, 256K context; a strong small-Mac default.

### Gemma 4 26B MoE

- **Ollama tag:** `gemma4:26b` · **Tier:** routine · **License:** Apache 2.0
- **Parameters:** 25.2B (3.8B active) · **Context:** 262,144 tokens
- **Memory:** default tag 17 GB (`ollama list`), Q8 ~27 GB, FP16 ~50 GB

MoE with ~3.8B active parameters, so it decodes much faster than a dense 26B while needing all 26B of weights in memory.

### Qwen3.8 27B

- **Ollama tag:** `qwen3.8:27b` · **Tier:** routine · **License:** Apache 2.0
- **Parameters:** 27B · **Context:** 262,144 tokens
- **Memory:** default tag 18 GB (Ollama tags), Q8 ~29 GB, FP16 ~54 GB

Dense with vision (2026-08). Thinks by default; hybrid thinking can be turned off per request. 56 GB as `-mlx-bf16`.

### Muse Glimmer 30B

- **Ollama tag:** `muse-glimmer:30b` · **Tier:** routine · **License:** Apache 2.0
- **Parameters:** 29.6B · **Context:** 131,072 tokens
- **Memory:** default tag 18 GB (Ollama tags), Q8 ~32 GB, FP16 ~59 GB

Meta's open-weight successor to Llama 4 (2026-08). Dense, ~1.8B of the parameters are a vision encoder. Reasoning strength is set in the system prompt.

### Gemma 4 31B

- **Ollama tag:** `gemma4:31b` · **Tier:** judgment · **License:** Apache 2.0
- **Parameters:** 30.7B · **Context:** 262,144 tokens
- **Memory:** default tag 19 GB (`ollama list`), Q8 ~33 GB, FP16 ~61 GB

Dense; Ragbot's default local model.

### Mistral Medium 3.5 128B

- **Ollama tag:** `mistral-medium-3.5:128b` · **Tier:** judgment · **License:** Modified MIT
- **Parameters:** 128B · **Context:** 262,144 tokens
- **Memory:** default tag 80 GB (Ollama tags (Q8 138 GB)), Q8 ~138 GB, FP16 ~256 GB

Dense with vision. No MLX tags on Ollama, so it runs through the GGUF path.

### Nemotron 3.5 Super 120B-A12B

- **Ollama tag:** `nemotron-3.5-super:120b-a12b` · **Tier:** judgment · **License:** OpenMDW-1.1
- **Parameters:** 120B (12B active) · **Context:** 1,048,576 tokens
- **Memory:** default tag 90 GB (Ollama tags (MLX nvfp4 80 GB)), Q8 ~130 GB, FP16 ~240 GB

NVIDIA hybrid Mamba-2/Transformer MoE with vision (2026-10), 1M context, thinking on/off.

### Qwen3.8 Flash-Next 125B-A6B (preview)

- **Ollama tag:** `qwen3.8-flash-next:125b-a6b-q4_K_M` · **Tier:** judgment · **License:** qwen-community-1.0
- **Parameters:** 125B (6B active) · **Context:** 262,144 tokens
- **Memory:** default tag 120 GB (Ollama tags (MLX 105 GB)), Q8 ~135 GB, FP16 ~250 GB

Preview of the Qwen4 architecture (2026-08): 125B total / 6B active plus a 51B n-gram embedding. Thinking on by default.

---

## Choosing by memory

The smallest Mac on which each model is *comfortable* at its default tag or better:

| Model | Smallest comfortable Mac in this matrix |
|---|---|
| Gemma 4 E4B | MacBook Air 24 GB (M5) |
| Gemma 4 12B | Mac mini 16 GB (M4) |
| Gemma 4 26B MoE | Mac mini 32 GB (M4 Pro) |
| Qwen3.8 27B | Mac mini 32 GB (M4 Pro) |
| Muse Glimmer 30B | Mac mini 32 GB (M4 Pro) |
| Gemma 4 31B | Mac mini 32 GB (M4 Pro) |
| Mistral Medium 3.5 128B | MacBook Pro 128 GB (M5 Max) |
| Nemotron 3.5 Super 120B-A12B | Mac Studio 192 GB (M3 Ultra) |
| Qwen3.8 Flash-Next 125B-A6B (preview) | Mac Studio 192 GB (M3 Ultra) |

Reasonable starting points by memory: 16 GB — Gemma 4 E4B or 12B; 24-32 GB — Qwen3.8 27B, Gemma 4 26B MoE or Muse Glimmer 30B at their default Q4 tags; 48-64 GB — Gemma 4 31B at Q8 on 64 GB; 128 GB — Mistral Medium 3.5 comfortably, Nemotron 3.5 Super when the Mac is dedicated to it; 192 GB and up — every model, including Qwen3.8 Flash-Next.

---

## Cross-cutting topics

### Ollama, MLX and GGUF

Ragbot's local path runs through Ollama (the `ollama_chat/` LiteLLM prefix). On Apple Silicon, Ollama's MLX backend loads MLX-format weights directly; most current models publish `-mlx`, `-mlx-bf16`, `-nvfp4` and `-mxfp8` tag variants alongside the default GGUF tag (Mistral Medium 3.5 has no MLX tags). The `ollama_tag` in `engines.yaml` names the default tag; pull a variant and select it by name if you prefer it.

### Thinking modes on local models

Qwen3.8 27B, Qwen3.8 Flash-Next and Nemotron 3.5 Super declare hybrid thinking in `engines.yaml`. Ragbot sends no thinking parameter by default, which leaves each model's own default (Qwen3.8 thinks by default); an explicit effort reaches Ollama as its `think` option. Gemma 4 thinking is enabled through its chat template, and Muse Glimmer's reasoning strength is set in the system prompt, so neither takes a thinking parameter.

### KV cache quantization

For contexts above ~32K tokens the KV cache often dominates memory. Ollama can store it at Q8 (`OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`), halving its size with negligible quality loss for most work, or at Q4 for retrieval-heavy contexts read once.

### Keeping models warm

The verdicts assume one model loaded at a time. To keep two warm, subtract both footprints from the budget. Ollama unloads idle models after five minutes; `OLLAMA_KEEP_ALIVE=-1` keeps them resident.

### Cloud-only open-weight models

Some current open-weight releases (DeepSeek V4 and V4.1, Mistral Large 4, Muse Spark Open) are available on Ollama only as `:cloud` tags, which run on Ollama's servers. Ragbot treats its Ollama provider as local for workspace routing, so these are deliberately absent from `engines.yaml`.

## Keeping this matrix current

`engines.yaml` is the single source of truth for which models Ragbot exposes. When the local lineup changes, regenerate the tables from verified Ollama download sizes and parameter counts in the same pull request, and record the sources in the dated `docs/model-catalog-*.yaml`.
