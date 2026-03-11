#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-bench}"
MODEL="${MODEL:-/mnt/lab//models/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-MXFP4_MOE.gguf}"
CPU_RANGE="${CPU_RANGE:-0-11}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-10}"
N_CPU_MOE="${N_CPU_MOE:-26}"
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
  --n-cpu-moe "${N_CPU_MOE}"
  --threads "${THREADS}"
  -pg "${TASKS}"
  -fa 1
  -mmp "${MMP}"
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
