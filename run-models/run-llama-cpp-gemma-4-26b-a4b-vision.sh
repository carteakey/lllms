#!/usr/bin/env bash
# Gemma-4-26B-A4B-it (UD-Q5_K_XL) — vision preset
# Forces mmproj-enabled launch and delegates to the base Gemma run script.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export USE_VISION="${USE_VISION:-1}"
export MMPROJ="${MMPROJ:-/home/kchauhan/models/unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf}"
export CTX_SIZE="${CTX_SIZE:-128000}"
export FIT_CTX="${FIT_CTX:-128000}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export UBATCH_SIZE="${UBATCH_SIZE:-512}"
export FIT_TARGET=2048
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-0}"

exec "${SCRIPT_DIR}/run-llama-cpp-gemma-4-26b-a4b.sh" "$@"
