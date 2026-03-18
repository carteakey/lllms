#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp-pr-test-20275/build/bin/llama-server}"
MODEL="${MODEL:-/mnt/lab/models/Sumitc13/sarvam-30b-GGUF/sarvam-30B-Q6_K.gguf}"

# Sarvam card defaults (coding/knowledge): temp=1.0, top_p=1.0
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
CTX_SIZE="${CTX_SIZE:-4096}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-4096}"
THREADS="${THREADS:-10}"
THREADS_BATCH="${THREADS_BATCH:-12}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.0}"
N_PREDICT="${N_PREDICT:-512}"

if [ ! -x "${LLAMA_SERVER}" ]; then
  echo "llama-server not found/executable: ${LLAMA_SERVER}" >&2
  echo "Build it with: ./maintenance/build-sarvam-llama-cpp.sh" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

exec "${LLAMA_SERVER}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL}" \
  --ctx-size "${CTX_SIZE}" \
  --n-gpu-layers "${N_GPU_LAYERS}" \
  --fit on \
  --fit-target "${FIT_TARGET}" \
  --fit-ctx "${FIT_CTX}" \
  --n-predict "${N_PREDICT}" \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --threads "${THREADS}" \
  --threads-batch "${THREADS_BATCH}" \
  --temp "${TEMP}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --min-p "${MIN_P}" \
  --repeat-penalty "${REPEAT_PENALTY}" \
  --flash-attn on \
  --jinja \
  --no-mmap \
  --mlock \
  --prio 2
