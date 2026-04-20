#!/usr/bin/env bash
# Qwen3.6-35B-A3B (UD-Q5_K_XL) — direct llama-server run helper
#
# Used for local serve tuning while keeping llama-swap as the primary runtime.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CPU_RANGE="${CPU_RANGE:-0-11}"
PORT="${PORT:-8002}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf}"
ALIAS="${ALIAS:-unsloth/Qwen3.6-35B-A3B (UD-Q5_K_XL)}"
USE_VISION="${USE_VISION:-0}"
MMPROJ="${MMPROJ:-/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf}"

CTX_SIZE="${CTX_SIZE:-65536}"
FIT_CTX="${FIT_CTX:-65536}"
FIT_TARGET="${FIT_TARGET:-512}"
PARALLEL="${PARALLEL:-1}"

THREADS="${THREADS:-10}"
THREADS_BATCH="${THREADS_BATCH:-12}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

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
  --alias "${ALIAS}"
  --seed 3407
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00
  --presence-penalty 1.5 --repeat-penalty 1.0
  --host 0.0.0.0 --port "${PORT}"
  --jinja
  --ctx-size "${CTX_SIZE}"
  --fit on --fit-ctx "${FIT_CTX}" --fit-target "${FIT_TARGET}"
  --parallel "${PARALLEL}"
  -ctk "${CACHE_TYPE_K}" -ctv "${CACHE_TYPE_V}"
  --no-mmap --mlock
  --threads "${THREADS}" --threads-batch "${THREADS_BATCH}"
  --flash-attn on
  --batch-size "${BATCH_SIZE}" --ubatch-size "${UBATCH_SIZE}"
  --prio 2 --no-warmup
)

if [ "${USE_VISION}" = "1" ]; then
  if [ ! -f "${MMPROJ}" ]; then
    echo "mmproj file not found: ${MMPROJ}" >&2
    exit 1
  fi
  cmd+=(--mmproj "${MMPROJ}")
fi

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
