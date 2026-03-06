#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-bench}"
MODEL="${MODEL:-/run/media/kchauhan/Windows/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf}"
CPU_RANGE="${CPU_RANGE:-0-11}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-10}"
MMP="${MMP:-0}"

if [ ! -x "${LLAMA_BENCH}" ]; then
  echo "llama-bench not found/executable: ${LLAMA_BENCH}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

cmd=(
  "${LLAMA_BENCH}"
  -m "${MODEL}"
  --n-gpu-layers "${N_GPU_LAYERS}"
  --threads "${THREADS}"
  -pg "${TASKS}"
  -fa 1
  -mmp "${MMP}"
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
