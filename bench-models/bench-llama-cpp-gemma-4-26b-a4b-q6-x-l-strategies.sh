#!/usr/bin/env bash
# Gemma-4-26B-A4B-it (UD-Q6_K_XL) — manual CPU/GPU offload strategy experiments
#
# Strategy overview:
#   none         no tensor override (default)
#   late-cpu     move FFN tensors to CPU from blk 20 onward
#   all-ffn-cpu  move all FFN tensors to CPU (lowest VRAM)
#
# Usage:
#   ./bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-strategies.sh
#   STRATEGY=late-cpu ./bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-strategies.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

MODEL="${MODEL:-/mnt/lab/models/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

STRATEGY="${STRATEGY:-none}"

if [ -z "${OVERRIDE_TENSOR:-}" ]; then
  case "${STRATEGY}" in
    none)
      OVERRIDE_TENSOR=""
      ;;
    late-cpu)
      OVERRIDE_TENSOR="blk\.(2[0-9]|[3-9][0-9]|[0-9][0-9][0-9])\.ffn_.*=CPU"
      ;;
    all-ffn-cpu)
      OVERRIDE_TENSOR=".ffn_.*=CPU"
      ;;
    *)
      echo "Unknown STRATEGY '${STRATEGY}'." >&2
      echo "Valid: none | late-cpu | all-ffn-cpu" >&2
      exit 1
      ;;
  esac
fi

export MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V OVERRIDE_TENSOR

echo "# model    : ${MODEL}"
echo "# strategy : ${STRATEGY}"
echo "# override : ${OVERRIDE_TENSOR:-<none>}"
echo "# tasks    : ${TASKS}"
echo "# ngl      : ${N_GPU_LAYERS}"
echo "# threads  : ${THREADS} (pinned ${CPU_RANGE})"
echo "# fa       : ${FA}  mmp: ${MMP}"
echo "# batch    : ${BATCH_SIZE}  ubatch: ${UBATCH_SIZE}"
echo

exec "${SCRIPT_DIR}/run-llama-bench.sh"
