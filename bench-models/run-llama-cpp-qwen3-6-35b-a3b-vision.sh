#!/usr/bin/env bash
# Qwen3.6-35B-A3B (UD-Q5_K_XL) — vision preset (mmproj enabled)
#
# Wrapper over run-llama-cpp-qwen3-6-35b-a3b.sh with vision-safe defaults for 12 GB VRAM.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export USE_VISION="${USE_VISION:-1}"
export MMPROJ="${MMPROJ:-/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf}"
export ALIAS="${ALIAS:-unsloth/Qwen3.6-35B-A3B (UD-Q5_K_XL, vision/mmproj-F16)}"
export PORT="${PORT:-8003}"

# Vision profile: keep larger VRAM headroom and smaller prefill batches.
export CTX_SIZE="${CTX_SIZE:-65536}"
export FIT_CTX="${FIT_CTX:-65536}"
export FIT_TARGET="${FIT_TARGET:-2048}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export UBATCH_SIZE="${UBATCH_SIZE:-512}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-0}"

exec "${SCRIPT_DIR}/run-llama-cpp-qwen3-6-35b-a3b.sh" "$@"
