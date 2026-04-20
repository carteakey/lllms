---
title: Running Qwen3.6-35B-A3B locally with text + vision
description: Practical Qwen3.6 setup on 12 GB VRAM using llama.cpp, llama-swap, and l3ms scripts.
date: 2026-04-17
updated: 2026-04-17
authored_by: ai-assisted
draft: true
tags:
  - AI
  - Self-Host
pinned: false
---

## TL;DR

- **Model**: `unsloth/Qwen3.6-35B-A3B-GGUF` (`UD-Q5_K_XL`) + `mmproj-F16.gguf`.
- **Stack**: mainline `llama.cpp` + `llama-swap` + `l3ms` run/bench scripts.
- **Best text bench on this machine**: **pp512=970.77**, **tg128=52.33** (fit winner).
- **Serve IDs**: `qwen3-6-35b-a3b` (text), `qwen3-6-35b-a3b-vision` (vision).
- **Vision-safe defaults (12 GB class GPUs)**: `FIT_TARGET=2048`, `BATCH_SIZE=256`, `GGML_CUDA_GRAPH_OPT=0`.

If you already run llama-swap as your front door, Qwen3.6 plugs in cleanly and gives a strong text baseline with optional multimodal mode.

## What I added in l3ms

Qwen3.6 now has a complete local workflow:

- Direct run helper: `bench-models/run-llama-cpp-qwen3-6-35b-a3b.sh`
- Vision wrapper: `bench-models/run-llama-cpp-qwen3-6-35b-a3b-vision.sh`
- Bench suite:
  - `bench-models/bench-llama-cpp-qwen3-6-35b-a3b.sh`
  - `bench-models/bench-llama-cpp-qwen3-6-35b-a3b-strategies.sh`
  - `bench-models/bench-llama-cpp-qwen3-6-35b-a3b-fit.sh`
- llama-swap entries:
  - `qwen3-6-35b-a3b`
  - `qwen3-6-35b-a3b-vision`

## Download

`models_config.json` now includes Qwen3.6 patterns for both text weights and projector:

- `*UD-Q5_K_XL*`
- `*mmproj-F16*`

One-shot direct download example:

```bash
cd ~/repos/l3ms
./model_downloader/download_hf_model.py \
  --repo-id unsloth/Qwen3.6-35B-A3B-GGUF \
  --allow-patterns '*UD-Q5_K_XL*' '*mmproj-F16*' \
  --local-dir /mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF \
  --max-workers 2
```

## Text benchmark outcomes

Hardware profile used for these runs: RTX 4070 12 GB + 64 GB DDR5 host RAM.

| Strategy | pp512 (tok/s) | tg128 (tok/s) |
| --- | ---: | ---: |
| baseline / all-cpu experts | 654.16 | 41.10 |
| partial-cpu | 746.36 | 44.35 |
| up-down-cpu | 865.26 | 48.95 |
| fit (winner) | **970.77** | **52.33** |
| up-cpu | OOM | OOM |

The fit-derived split was the clear winner and became the default serving profile in llama-swap.

## Serving with llama-swap

Text:

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-6-35b-a3b",
    "messages":[{"role":"user","content":"Give me a 5-line summary of sparse MoE tradeoffs."}]
  }'
```

Vision (OpenAI-format multimodal input):

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-6-35b-a3b-vision",
    "messages":[
      {"role":"user","content":[
        {"type":"text","text":"What is in this image?"},
        {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"}}
      ]}
    ]
  }'
```

> If your service still listens on `:8080`, just swap the port in both examples.

## Vision notes that matter

Qwen3.6 vision is stable on 12 GB VRAM with conservative headroom:

- `FIT_TARGET=2048`
- `BATCH_SIZE=256`
- `UBATCH_SIZE=512`
- `GGML_CUDA_GRAPH_OPT=0`

Trying to push `FIT_TARGET` too low while keeping high context can trigger OOM during longer sessions.

## References

- [Qwen3.6-35B-A3B GGUF (unsloth)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [l3ms](https://github.com/carteakey/l3ms)
