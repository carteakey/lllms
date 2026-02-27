#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
HF_MODEL="${HF_MODEL:-unsloth/Qwen3-30B-A3B-GGUF:q4_k_m}"

exec "${LLAMA_SERVER}" \
  -hf "${HF_MODEL}" \
  --n-gpu-layers 0 \
  --jinja \
  --reasoning-format deepseek \
  --flash-attn \
  -sm row \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0 \
  --ctx-size 40960 \
  --n-predict 32768 \
  --no-context-shift \
  --port 8080 \
  --host 0.0.0.0 \
  --metrics \
  --alias "Qwen3-30B (CPU Only)"
