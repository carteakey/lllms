#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

#export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
#export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/run/media/kchauhan/Windows/models/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-IQ4_KSS.gguf}"


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
  --alias "unsloth/Qwen3.5-122B-A10B-thinking-coding"
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --host 0.0.0.0
  --port 8001
  --jinja
  -ctk q8_0
  -ctv q8_0
  --flash-attn on
  --ctx-size 65536
  --fit on
  --fit-ctx 65536
  --fit-target 512
  --no-mmap
  --threads 10
  --threads-batch 12
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

# --mmproj /run/media/kchauhan/Windows/models/unsloth/Qwen3.5-122B-A10B-GGUF/MXFP4_MOE/mmproj-F16.gguf \
exec "${cmd[@]}"
