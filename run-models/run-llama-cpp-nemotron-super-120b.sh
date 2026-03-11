#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/mnt/lab//models/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q3_K_XL-00001-of-00003.gguf}"

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
  --alias "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B (UD-Q3_K_XL)"
  --seed 3407
  --temp 0.6
  --top-p 0.95
  --top-k 40
  --min-p 0.05
  --repeat-penalty 1.0
  --host 0.0.0.0
  --port 8001
  --jinja
  --ctx-size 32768
  --fit on
  --fit-ctx 32768
  --fit-target 512
  --no-mmap
  --threads 10
  --threads-batch 12
  --flash-attn on
  --batch-size 2048
  --ubatch-size 512
  --prio 2
  --no-warmup
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
