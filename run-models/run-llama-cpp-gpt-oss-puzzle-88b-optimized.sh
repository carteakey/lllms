#!/usr/bin/env bash
# run-llama-cpp-gpt-oss-puzzle-88b-optimized.sh
# ----------------------------------------------
# Optimized run script for gpt-oss-puzzle-88b based on benchmark winner.
#
# Winning pattern (llama-bench):
#   -ngl 37
#   -ot "blk.4 ffn_down to CPU, blk.5+ ffn_(up|down|gate_up|gate)_(ch|)exps to CPU"
#   q8_0 KV, parallel 1
#
# Bench snapshot:
#   baseline (ncmoe=48, ngl=99):  pp ~433, tg ~20.7
#   fit/optimized split (ngl=37): pp ~549, tg ~27.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp-pr-test-21032/build/bin/llama-server}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${REPO_DIR}/chat-template.jinja}"
CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/home/kchauhan/models/SamPurkis/gpt-oss-puzzle-88B-GGUF/gpt-oss-puzzle-88B.MXFP4_MOE.gguf}"

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
  --alias "SamPurkis/gpt-oss-puzzle-88B"

  # placement (bench-derived static split)
  -ngl 37
  --override-tensor "blk\.4\.ffn_down.*=CPU,blk\.(5|[6-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
  --no-mmap

  # kv + context
  -ctk q8_0
  -ctv q8_0
  --ctx-size 32768
  --parallel 1

  # compute
  --flash-attn on
  --batch-size 2048
  --ubatch-size 512
  --threads 10
  --threads-batch 12
  --prio 2
  --no-warmup

  # sampling + serving
  --temp 1.0
  --min-p 0.0
  --top-p 1.0
  --jinja
  --reasoning-format none
  --chat-template-kwargs '{"reasoning_effort":"high"}'
  --chat-template-file "${CHAT_TEMPLATE}"
  --host 0.0.0.0
  --port 8001
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
