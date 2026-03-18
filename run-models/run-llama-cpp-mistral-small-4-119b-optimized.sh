#!/usr/bin/env bash
# run-llama-cpp-mistral-small-4-119b-optimized.sh
# ------------------------------------------------
# Throughput-oriented preset for Mistral Small 4 119B IQ4_XS.
# Start with this after you validate stability with the standard script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

if [ -z "${LLAMA_SERVER:-}" ]; then
  for candidate in \
    "${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server" \
    "${REPO_DIR}/vendor/llama.cpp/llama-server" \
    "${REPO_DIR}/vendor-forks/llama.cpp-copilot/build/bin/llama-server" \
    "${REPO_DIR}/vendor-forks/llama.cpp-copilot/llama-server" \
    "${REPO_DIR}/vendor/llama.cpp/build-cublas/bin/llama-server" \
    "${REPO_DIR}/vendor-forks/llama.cpp-copilot/build-cublas/bin/llama-server"
  do
    if [ -x "${candidate}" ]; then
      LLAMA_SERVER="${candidate}"
      break
    fi
  done
  LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
fi

CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/Mistral-Small-4-119B-2603-GGUF/UD-IQ4_XS/Mistral-Small-4-119B-2603-UD-IQ4_XS-00001-of-00003.gguf}"
MMPROJ="${MMPROJ:-/mnt/lab/models/unsloth/Mistral-Small-4-119B-2603-GGUF/mmproj-F16.gguf}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
CTX_SIZE="${CTX_SIZE:-32768}"

THREADS="${THREADS:-10}"
THREADS_BATCH="${THREADS_BATCH:-12}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

N_GPU_LAYERS="${N_GPU_LAYERS:-9}"
# Bench-derived best tg on this host (N_GPU_LAYERS=9): keep gate experts on GPU,
# spill up/down experts to CPU for stable 15+ t/s decode on IQ4_XS.
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-.ffn_(up|down)_(ch|)exps=CPU}"

# Mistral-Small-4 guidance: 0.0-0.7 for non-reasoning, ~0.7 for high reasoning effort.
TEMP="${TEMP:-0.15}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-40}"
MIN_P="${MIN_P:-0.0}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.0}"

if [ ! -x "${LLAMA_SERVER}" ]; then
  echo "llama-server not found/executable: ${LLAMA_SERVER}" >&2
  echo "Set LLAMA_SERVER to a newer llama.cpp build with mistral4 support." >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

cmd=(
  "${LLAMA_SERVER}"
  -m "${MODEL}"
  --alias "unsloth/Mistral-Small-4-119B-2603 (optimized)"
  --fit off
  -ngl "${N_GPU_LAYERS}"
  --override-tensor "${OVERRIDE_TENSOR}"
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
  --parallel 1
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

if [ -f "${MMPROJ}" ]; then
  cmd+=(--mmproj "${MMPROJ}")
fi

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
