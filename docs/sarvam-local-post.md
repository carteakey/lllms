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
- The most reliable build path was: upstream `llama.cpp` + Sarvam PR merge + local build.
- I added a `sarvam-105B` download profile first so I can start collecting files while I keep testing.
- Biggest gotcha was API drift in `llama_hparams` after upstream changes.

## What I actually ran

I now have a local flow in `l3ms` that keeps this simple:

```bash
cd ~/repos/l3ms
./maintenance/build-sarvam-llama-cpp.sh
./run-models/run-llama-cpp-sarvam-30b.sh
./bench-models/bench-llama-cpp-sarvam-30b.sh
```

`build-sarvam-llama-cpp.sh` wraps my PR test builder and targets Sarvam PR `#20275` by default, so I can reproduce the same build context without re-remembering flags.

## Model downloads

For 30B, I used:

```bash
./model_downloader/download_hf_model.py \
  --repo-id Sumitc13/sarvam-30b-GGUF \
  --allow-patterns '*sarvam-30B-Q6_K.gguf*' \
  --local-dir /mnt/lab/models/Sumitc13/sarvam-30b-GGUF \
  --max-workers 2
```

I also added a 105B profile in `model_downloader/models_config.json`:

- `repo_id`: `limegreenpeper1/sarvam-105B-GGUF`
- default pattern: `*Q4_K_M*`
- local dir: `/mnt/lab/models/limegreenpeper1/sarvam-105B-GGUF`

One-shot 105B download command:

```bash
./model_downloader/download_hf_model.py \
  --repo-id limegreenpeper1/sarvam-105B-GGUF \
  --allow-patterns '*Q4_K_M*' \
  --local-dir /mnt/lab/models/limegreenpeper1/sarvam-105B-GGUF \
  --max-workers 2
```

## What broke (and why)

After merging PR `#20275` on top of newer upstream, I hit compile errors in `src/models/sarvam-moe.cpp` around:

- `n_embd_head_v`
- `n_embd_head_k`

This was due to API drift tied to upstream commit `59db9a357` (`#20301`), which moved head-dim handling toward accessor-style usage in this path. The fix was small, but it took a bit to realize the error was merge-context drift, not a fundamental Sarvam issue.

## A note on “culturally different” model behavior

After more trials, one thing stood out: Sarvam often feels culturally different in tone and framing, especially when discussing philosophy or social reasoning. I’m not claiming that as a formal benchmark result yet, but it has been a consistent qualitative impression for me. More broadly, I think LLMs are cultural snapshots of the data and reinforcement choices that shaped them. I want to write a separate post just on that idea once I gather better side-by-side examples.

## References

- Sarvam-105B model card: `sarvamai/sarvam-105b`
- Sarvam-30B GGUF: `Sumitc13/sarvam-30b-GGUF`
- Sarvam-105B GGUF: `limegreenpeper1/sarvam-105B-GGUF`
- llama.cpp Sarvam PR: `#20275`
- upstream API-change commit: `59db9a357` (`#20301`)
