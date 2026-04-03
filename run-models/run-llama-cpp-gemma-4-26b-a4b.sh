#!/usr/bin/env bash
# Gemma-4-26B-A4B-it (UD-Q5_K_XL) — llama-server run script
#
# Defaults align with Gemma 4 guidance:
#   temp=1.0, top-p=0.95, top-k=64
# Practical serving defaults:
#   128k context for text mode + q8_0 KV + fit-based placement.
# Base script is text-first; use run-llama-cpp-gemma-4-26b-a4b-vision.sh
# (or USE_VISION=1) to enable mmproj.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/home/kchauhan/models/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf}"
MMPROJ="${MMPROJ:-/home/kchauhan/models/unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
CTX_SIZE="${CTX_SIZE:-131072}"
FIT_CTX="${FIT_CTX:-131072}"
FIT_TARGET="${FIT_TARGET:-512}"
PARALLEL="${PARALLEL:-1}"

THREADS="${THREADS:-10}"
THREADS_BATCH="${THREADS_BATCH:-12}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
MIN_P="${MIN_P:-0.0}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.0}"

USE_VISION="${USE_VISION:-0}"

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
  --alias "unsloth/gemma-4-26B-A4B-it (UD-Q5_K_XL)"
  --seed 3407
  --temp "${TEMP}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --min-p "${MIN_P}"
  --repeat-penalty "${REPEAT_PENALTY}"
  --host "${HOST}"
  --port "${PORT}"
  --jinja
  --ctx-size "${CTX_SIZE}"
  --fit on
  --fit-ctx "${FIT_CTX}"
  --fit-target "${FIT_TARGET}"
  --parallel "${PARALLEL}"
  -ctk q8_0
  -ctv q8_0
  --no-mmap
  --mlock
  --threads "${THREADS}"
  --threads-batch "${THREADS_BATCH}"
  --flash-attn on
  --batch-size "${BATCH_SIZE}"
  --ubatch-size "${UBATCH_SIZE}"
  --prio 2
  --no-warmup
)

if [ "${USE_VISION}" = "1" ]; then
  if [ ! -f "${MMPROJ}" ]; then
    echo "vision enabled but mmproj not found: ${MMPROJ}" >&2
    echo "Set USE_VISION=0 for text-only mode, or set MMPROJ to a valid path." >&2
    exit 1
  fi
  cmd+=(--mmproj "${MMPROJ}")
fi

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
