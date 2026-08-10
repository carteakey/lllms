#!/usr/bin/env bash
# L3MS benchmark template. Copy this file into bench-models/ and set MODEL.
set -euo pipefail

SCRIPT_DIR="\$(cd -- "\$(dirname -- "\${BASH_SOURCE[0]}")" && pwd)"
ROOT="\$(cd -- "\${SCRIPT_DIR}/.." && pwd)"

MODEL="\${MODEL:-/path/to/model.gguf}"
MODEL_KEY="\${MODEL_KEY:-replace-me}"
STRATEGY="\${STRATEGY:-baseline}"
LLAMA_BENCH="\${LLAMA_BENCH:-\${ROOT}/vendor/llama.cpp/build/bin/llama-bench}"

if [[ ! -f "\${MODEL}" ]]; then
  echo "MODEL is not a file: \${MODEL}" >&2
  exit 1
fi

MODEL="\${MODEL}" \
MODEL_KEY="\${MODEL_KEY}" \
STRATEGY="\${STRATEGY}" \
LLAMA_BENCH="\${LLAMA_BENCH}" \
  "\${SCRIPT_DIR}/run-llama-bench.sh"
