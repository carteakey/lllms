#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${REPO_DIR}/chat-template.jinja}"
CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/home/kchauhan/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf}"

if [ ! -x "${LLAMA_SERVER}" ]; then
  echo "llama-server not found/executable: ${LLAMA_SERVER}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

cmd=(
  "${LLAMA_SERVER}"
  -m "${MODEL}"
  --alias "ggml-org/gpt-oss-120b"
  --fit on
  --fit-ctx 32678
  --fit-target 512
  --no-mmap
  --threads 10
  --threads-batch 12
  --flash-attn on
  --prio 2
  --no-warmup
  --temp 1.0
  --min-p 0.0
  --top-p 1.0
  --jinja
  --reasoning-format none
  --chat-template-kwargs '{"reasoning_effort":"high"}'
  --chat-template-file "${CHAT_TEMPLATE}"
  --host 0.0.0.0
  --port 8001
  --api-key "dummy"
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
