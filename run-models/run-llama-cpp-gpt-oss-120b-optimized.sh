#!/usr/bin/env bash
# run-llama-cpp-gpt-oss-120b-optimized.sh
# ----------------------------------------
# Optimized run script for gpt-oss-120b (mxfp4) based on bench results.
#
# Key differences from run-llama-cpp-gpt-oss-120b.sh:
#
#   --fit removed         Replaced with explicit -ngl 37 + --override-tensor.
#                         fit adds ~4s startup latency and occasionally lands a
#                         slightly different placement run-to-run. Static -ot is
#                         deterministic and starts faster.
#
#   -ngl 37               Bench-confirmed optimal: blk 0-4 fully on GPU (~10.5 GB
#                         VRAM), blk 5-36 experts on CPU (~50 GB RAM).
#                         DO NOT raise ngl without monitoring RAM — all-CPU loads
#                         ~60 GB and hard-crashes a 64 GB system.
#
#   --override-tensor     fit-params recommended pattern: blk 5+ experts to CPU.
#                         Covers both routed (_exps) and shared (_shexp) via (ch|).
#
#   --parallel 1          Single inference slot. Multi-slot (default=4) multiplies
#                         KV cache VRAM by n_parallel, eating into the headroom
#                         that fit was accounting for. Single slot = stable, more
#                         VRAM headroom, and this is a single-user homelab server.
#
#   --ctx-size 32768      Explicit 32k context. Matches fit-params FIT_CTX used
#                         during bench. Keeps KV cache predictable in VRAM.
#                         llama.cpp pre-allocates the full KV cache at startup —
#                         context growth mid-session does NOT expand VRAM usage.
#                         The 12 MiB VRAM free at startup is post-KV-allocation,
#                         so long contexts won't OOM. The risk is other processes
#                         (compositor, browser) grabbing that headroom externally.
#                         If you see CUDA OOM on load, reduce --ctx-size to 16384
#                         or switch to the fit-based script with --fit-target 1024.
#
#   cuBLAS binary tested  GGML_CUDA_FORCE_CUBLAS=ON build was benchmarked and was
#                         ~45 t/s pp slower with no tg gain. Default binary wins.
#                         See docs/bench-runbook.md §8 for full results.
#
# Bench results (512pp+128tg, default build, partial-cpu -ot, ngl=37):
#   pp = 427.92 t/s   tg = 23.36 t/s
# Server observed (single slot, short prompt):
#   pp ~45 t/s        tg ~27-30 t/s  (server overhead + KV checkpoint amortised)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${REPO_DIR}/chat-template.jinja}"
CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/mnt/lab/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf}"

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
  --alias "ggml-org/gpt-oss-120b"

  # --- placement (bench-derived, static) ---
  # 12 MiB VRAM free at startup is tight but safe — KV cache is pre-allocated
  # in full at load time, so long contexts don't grow VRAM usage mid-session.
  # If external processes cause OOM, reduce --ctx-size to 16384 as first step.
  -ngl 37
  --override-tensor "blk\.(5|[6-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU"
  --no-mmap

  # --- KV cache quantization ---
  # q8_0 is lossless for most purposes and halves KV cache VRAM vs f16.
  # Bench result: pp=429 t/s, tg=23.7 t/s — within noise of f16 baseline.
  # At 32k context this saves ~570 MiB VRAM, giving more headroom on 12 GB.
  -ctk q8_0
  -ctv q8_0

  # --- context ---
  --ctx-size 32768
  --parallel 1

  # --- compute ---
  --flash-attn on
  --batch-size 2048
  --ubatch-size 512
  --threads 10
  --threads-batch 12

  # --- sampling ---
  --temp 1.0
  --min-p 0.0
  --top-p 1.0

  # --- serving ---
  --host 0.0.0.0
  --port 8001
  --no-warmup
  --jinja
  --reasoning-format none
  --chat-template-kwargs '{"reasoning_effort":"high"}'
  --chat-template-file "${CHAT_TEMPLATE}"
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
