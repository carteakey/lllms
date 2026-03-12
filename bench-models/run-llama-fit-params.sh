#!/usr/bin/env bash
# Shared llama-fit-params runner. Set env vars before exec'ing this script.
#
# Runs llama-fit-params and prints the fitted CLI args to stdout.
# Useful standalone to inspect what fit would choose, or pipe into run-llama-fit-bench.sh.
#
# Required (no default):
#   MODEL             - path to .gguf model file
#
# Optional (llama-fit-params defaults shown):
#   LLAMA_FIT         - path to llama-fit-params binary (default: ../vendor/llama.cpp/build/bin/llama-fit-params)
#
#   --- fit parameters ---
#   FIT_TARGET        - MiB of free VRAM margin to leave on each GPU (default: 1024)
#   FIT_CTX           - minimum context size fit is allowed to reduce to (default: 4096)
#
#   --- model/runtime parameters (passed to fit-params for accurate memory projection) ---
#   BATCH_SIZE        - logical batch size (default: 2048)
#   UBATCH_SIZE       - physical batch size (default: 512)
#   THREADS           - CPU threads (default: unset, uses llama-fit-params default)
#   FA                - flash attention 0|1 (default: unset)
#   N_GPU_LAYERS      - layers offloaded to GPU; leave unset to let fit decide (default: unset)
#   TENSOR_SPLIT      - tensor split ratios e.g. "3,1"; leave unset to let fit decide (default: unset)
#   N_CPU_MOE         - MoE expert layers kept on CPU (default: unset, flag omitted)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_FIT="${LLAMA_FIT:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-fit-params}"

# --- validation ---

if [ -z "${MODEL:-}" ]; then
  echo "MODEL is not set" >&2
  exit 1
fi

if [ ! -x "${LLAMA_FIT}" ]; then
  echo "llama-fit-params not found/executable: ${LLAMA_FIT}" >&2
  echo "Rebuild with: cmake --build . --target llama-fit-params" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

# --- build command ---

cmd=("${LLAMA_FIT}" -m "${MODEL}")

[ -n "${FIT_TARGET:-}"    ] && cmd+=(-fitt "${FIT_TARGET}")
[ -n "${FIT_CTX:-}"       ] && cmd+=(-fitc "${FIT_CTX}")
[ -n "${BATCH_SIZE:-}"    ] && cmd+=(-b    "${BATCH_SIZE}")
[ -n "${UBATCH_SIZE:-}"   ] && cmd+=(-ub   "${UBATCH_SIZE}")
[ -n "${THREADS:-}"       ] && cmd+=(-t    "${THREADS}")
[ -n "${FA:-}"            ] && cmd+=(-fa   "${FA}")
[ -n "${N_GPU_LAYERS:-}"  ] && cmd+=(-ngl  "${N_GPU_LAYERS}")
[ -n "${TENSOR_SPLIT:-}"  ] && cmd+=(-ts   "${TENSOR_SPLIT}")
[ -n "${N_CPU_MOE:-}"     ] && cmd+=(-ncmoe "${N_CPU_MOE}")

# --- launch ---
# stderr (fit progress/diagnostics) goes to the terminal; stdout (fitted args) is captured by the caller.

exec "${cmd[@]}"
