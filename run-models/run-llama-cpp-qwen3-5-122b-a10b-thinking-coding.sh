#!/usr/bin/env bash
# Qwen3.5-122B-A10B — llama-server run script
#
# Best -ot placement from bench (2025-03, RTX 4070 12 GB / 96 GB DDR5):
#   partial-cpu: blk 0-2 expert tensors on GPU, blk 3-47 on CPU.
#   pp=284 t/s, tg=9.8 t/s (512pp+128tg).
#
#   tg is gated by active parameter count: this model activates 10B params/token
#   vs 3B for Qwen3-Coder-Next — ~3x more compute per decode step, full stop.
#   No -ot strategy or backend flag changes that. Run this model for quality and
#   thinking budget, not throughput.
#
#   Pattern covers BOTH routed experts (_exps) and shared experts (_shexp)
#   via the (ch|) group — omitting shared experts causes CUDA OOM.
#   See docs/bench-runbook.md §8 for full results.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

#export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
#export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CPU_RANGE="${CPU_RANGE:-0-11}"
# MODEL="${MODEL:-/mnt/lab//models/unsloth/Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-IQ4_KSS.gguf}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.5-122B-A10B-GGUF/UD-IQ4_XS/Qwen3.5-122B-A10B-UD-IQ4_XS-00001-of-00003.gguf}"

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
  --alias "unsloth/Qwen3.5-122B-A10B-thinking-coding"
  --temp 1.0
  --top-p 0.95
  --top-k 20
  --min-p 0.0
  --host 0.0.0.0
  --port 8001
  --jinja
  -ctk q8_0
  -ctv q8_0
  --flash-attn on
  --ctx-size 65536
  --no-mmap
  --override-tensor "blk\.(3|[4-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU"
  --threads 10
  --threads-batch 12
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

# --mmproj /mnt/lab//models/unsloth/Qwen3.5-122B-A10B-GGUF/MXFP4_MOE/mmproj-F16.gguf \
exec "${cmd[@]}"
