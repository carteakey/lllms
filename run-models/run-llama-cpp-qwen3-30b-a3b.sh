#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-/home/kchauhan/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf}"

if [ ! -x "${LLAMA_SERVER}" ]; then
  echo "llama-server not found/executable: ${LLAMA_SERVER}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

exec "${LLAMA_SERVER}" \
  --host 127.0.0.1 \
  --port 9045 \
  --model "${MODEL}" \
  --n-gpu-layers 99 \
  --flash-attn \
  --slots \
  --metrics \
  --ubatch-size 512 \
  --batch-size 512 \
  --presence-penalty 1.5 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --no-context-shift \
  --ctx-size 32768 \
  --n-predict 32768 \
  --temp 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0 \
  --repeat-penalty 1.1 \
  --jinja \
  --reasoning-format deepseek \
  --threads 5 \
  --threads-http 5 \
  --cache-reuse 256 \
  --override-tensor 'blk\.([0-9]*[02468])\.ffn_.*_exps\.=CPU' \
  --no-mmap
