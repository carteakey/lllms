---
title: Running Sarvam locally (30B now, 105B next)
description: A practical path to get Sarvam running in llama.cpp, plus notes on what broke and what felt new
date: 2026-03-15
authored_by: human
updated: 2026-03-15
tags:
  - AI
  - llama.cpp
---

## TL;DR

- `sarvam-30B-Q6_K.gguf` is running locally with my current setup.
- The most reliable build path was: upstream `llama.cpp` + Sarvam PR `#20275` merged, then local build.
- I added a `sarvam-105B` download profile first so I can start collecting files while I keep testing.
- Biggest gotcha was API drift in `llama_hparams` after upstream changes.

## Build

Sarvam support isn't in mainline llama.cpp yet. The working path is to apply PR `#20275` on top of upstream and build from source.

```bash
# Clone upstream
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Fetch and merge the Sarvam PR
git fetch origin pull/20275/head:sarvam-pr
git merge sarvam-pr

# Build with CUDA
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89   # adjust for your GPU
cmake --build . --config Release \
  --target llama-server llama-bench --parallel
```

> **Easier path**: [carteakey/l3ms](https://github.com/carteakey/l3ms) ships `maintenance/build-sarvam-llama-cpp.sh` which wraps the PR fetch + build in one command. Set `SARVAM_PR_NUMBER` to override the default (`20275`).

## Model downloads

For 30B:

```bash
huggingface-cli download Sumitc13/sarvam-30b-GGUF \
  --include '*sarvam-30B-Q6_K.gguf*' \
  --local-dir ~/models/Sumitc13/sarvam-30b-GGUF
```

For 105B (start collecting early while you keep testing 30B):

```bash
huggingface-cli download limegreenpeper1/sarvam-105B-GGUF \
  --include '*Q4_K_M*' \
  --local-dir ~/models/limegreenpeper1/sarvam-105B-GGUF
```

> Both of these are also in `model_downloader/models_config.json` in [carteakey/l3ms](https://github.com/carteakey/l3ms) as named profiles for the TUI downloader.

## Run

```bash
llama-server \
  -m ~/models/Sumitc13/sarvam-30b-GGUF/sarvam-30B-Q6_K.gguf \
  --host 0.0.0.0 --port 8001 \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --fit on --fit-target 512 --fit-ctx 4096 \
  --temp 1.0 --top-p 1.0 --top-k 20 \
  --repeat-penalty 1.0 \
  --batch-size 2048 --ubatch-size 512 \
  --threads 10 --threads-batch 12 \
  --flash-attn on \
  --no-mmap --mlock \
  --prio 2 --jinja
```

Sampling defaults follow the Sarvam card recommendation for coding/knowledge: `temp=1.0`, `top_p=1.0`.

> **Easier path**: `run-models/run-llama-cpp-sarvam-30b.sh` in [carteakey/l3ms](https://github.com/carteakey/l3ms) has all of the above pre-set with env-var overrides.

## Bench

```bash
llama-bench \
  -m ~/models/Sumitc13/sarvam-30b-GGUF/sarvam-30B-Q6_K.gguf \
  -p 512 -n 128 -r 3 \
  --n-gpu-layers 99 \
  --flash-attn 1 --no-mmap \
  --batch-size 2048 --ubatch-size 512
```

## What broke (and why)

After merging PR `#20275` on top of newer upstream, I hit compile errors in `src/models/sarvam-moe.cpp` around:

- `n_embd_head_v`
- `n_embd_head_k`

This was due to API drift tied to upstream commit `59db9a357` (`#20301`), which moved head-dim handling toward accessor-style usage in this path. The fix was small, but it took a bit to realize the error was merge-context drift, not a fundamental Sarvam issue.

## A note on "culturally different" model behavior

After more trials, one thing stood out: Sarvam often feels culturally different in tone and framing, especially when discussing philosophy or social reasoning. I'm not claiming that as a formal benchmark result yet, but it has been a consistent qualitative impression for me. More broadly, I think LLMs are cultural snapshots of the data and reinforcement choices that shaped them. I want to write a separate post just on that idea once I gather better side-by-side examples.

## Changelog

| Date | Note |
| --- | --- |
| 2026-03-15 | Initial post — 30B setup, PR build path, API drift fix, 105B download. |

## References

- Sarvam-105B model card: `sarvamai/sarvam-105b`
- Sarvam-30B GGUF: `Sumitc13/sarvam-30b-GGUF`
- Sarvam-105B GGUF: `limegreenpeper1/sarvam-105B-GGUF`
- llama.cpp Sarvam PR: `#20275`
- upstream API-change commit: `59db9a357` (`#20301`)
- [l3ms - homelab LLM toolkit with scripts for this model](https://github.com/carteakey/l3ms)
