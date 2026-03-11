#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${REPO_DIR}/chat-template.jinja}"
MODEL="${MODEL:-/mnt/lab//models/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf}"

if [ ! -x "${LLAMA_SERVER}" ]; then
  echo "llama-server not found/executable: ${LLAMA_SERVER}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

exec "${LLAMA_SERVER}" \
  -m "${MODEL}" \
  --n-cpu-moe 4 \
  --ctx-size 32000 \
  --n-gpu-layers 99 \
  --temp 1.0 \
  --min-p 0.0 \
  --top-p 1.0 \
  --top-k 20 \
  --flash-attn \
  --jinja \
  --reasoning-format none \
  --chat-template-file "${CHAT_TEMPLATE}" \
  --chat-template-kwargs '{"reasoning_effort": "high"}' \
  --host 0.0.0.0 \
  --port 8081 \
