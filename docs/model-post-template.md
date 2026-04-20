---
title: Running [MODEL NAME] locally on [VRAM]GB VRAM
description: End-to-end [MODEL NAME] setup on llama.cpp with throughput notes.
date: YYYY-MM-DD
updated: YYYY-MM-DD
authored_by: human           # or: ai-assisted
draft: true
tags:
  - AI
  - Self-Host
pinned: false
---

<!-- One or two sentences on what makes this model worth running locally. -->

This post covers my setup running [MODEL NAME] on [VRAM]GB VRAM using [mainline / fork] llama.cpp, with real-world throughput numbers.

## TL;DR

- **Model**: `[HF_REPO/MODEL_FILE]` (`[QUANT]`).
- **Stack**: [mainline / PR#XXXXX] `llama.cpp`.
- **Best synthetic bench**: `pp512=X`, `tg128=X`.
- **Server-realistic throughput**: `~X tok/s` @ [CTX]k context.
- **Key note**: <!-- any important gotcha, memory note, or stability caveat -->

## Why [MODEL NAME]?

<!-- Briefly: what's the size/architecture tradeoff? Why this quant? Why now? -->

## Build

<!-- Use this block for MAINLINE llama.cpp. If a custom PR is needed, add a note. -->

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89   # 86=RTX30, 75=RTX20, 61=GTX10
cmake --build . --config Release \
  --target llama-server llama-bench --parallel
```

<!--
If a custom PR or fork was needed, describe it here:

  git fetch origin pull/XXXXX/head:model-pr
  git merge model-pr

Note any compile-time issues and their fixes.
-->

> **Easier path**: [carteakey/l3ms](https://github.com/carteakey/l3ms) ships `maintenance/build-llama-cpp.sh` (or a model-specific build helper) that wraps the above with auto-detection of CUDA arch and OS deps.

## Download

```bash
huggingface-cli download [HF_REPO] \
  --include '[GLOB_PATTERN]' \
  --local-dir ~/models/[HF_REPO]
```

<!-- List additional files if needed (e.g. mmproj for vision models): -->
<!--
huggingface-cli download [HF_REPO] \
  --include '[MMPROJ_GLOB]' \
  --local-dir ~/models/[HF_REPO]
-->

> **Easier path**: `model_downloader/models_config.json` in [carteakey/l3ms](https://github.com/carteakey/l3ms) has a named profile for this model usable from the TUI or CLI downloader.

## Run

```bash
llama-server \
  -m ~/models/[HF_REPO]/[MODEL_FILE] \
  --host 0.0.0.0 --port 8001 \
  --ctx-size [CTX_SIZE] \
  --n-gpu-layers 99 \
  --fit on --fit-target [FIT_TARGET] --fit-ctx [FIT_CTX] \
  --temp [TEMP] --top-p [TOP_P] --top-k [TOP_K] \
  --repeat-penalty [REPEAT_PENALTY] \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --batch-size [BATCH_SIZE] --ubatch-size [UBATCH_SIZE] \
  --threads 10 --threads-batch 12 \
  --no-mmap --mlock \
  --parallel 1 --prio 2 --no-warmup --jinja
```

<!-- Add --mmproj flag for vision models:
  --mmproj ~/models/[HF_REPO]/[MMPROJ_FILE] \
-->

<!-- Sampling rationale: where did these defaults come from? (model card, testing, etc.) -->

> **Easier path**: `run-models/run-llama-cpp-[model-slug].sh` in [carteakey/l3ms](https://github.com/carteakey/l3ms) has all of the above pre-set with env-var overrides.

## Bench

```bash
llama-bench \
  -m ~/models/[HF_REPO]/[MODEL_FILE] \
  -p 512 -n 128 -r 3 \
  --n-gpu-layers 99 \
  --flash-attn 1 --no-mmap \
  -ctk q8_0 -ctv q8_0 \
  --batch-size [BATCH_SIZE] --ubatch-size [UBATCH_SIZE] \
  --threads 10
```

### Synthetic bench results

| Config | pp512 (tok/s) | tg128 (tok/s) | pp512+tg128 (tok/s) |
| --- | ---: | ---: | ---: |
| [Strategy 1] | X | X | X |
| [Strategy 2] | X | X | X |

### Server-realistic throughput

| Mode | Context | Throughput |
| --- | ---: | ---: |
| Text | [CTX]k | **~X tok/s** |

<!-- Add vision row if applicable:
| Vision | [CTX]k | **~X tok/s** |
-->

## Notes

<!-- Any gotchas, failure modes, memory tuning tips, or qualitative impressions. -->

<!--
Common sections to add:
- Vision/mmproj memory notes
- Sampling parameter rationale
- Fit-based placement tuning
- Cultural/qualitative model impressions
-->

## Changelog

| Date | Note |
| --- | --- |
| YYYY-MM-DD | Initial post. |

## References

- [Model card / HuggingFace page][link]
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [l3ms - homelab LLM toolkit with scripts for this model](https://github.com/carteakey/l3ms)
