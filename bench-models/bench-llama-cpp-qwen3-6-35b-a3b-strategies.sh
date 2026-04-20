#!/usr/bin/env bash
# Qwen3.6-35B-A3B (UD-Q5_K_XL) — manual MoE offload strategy experiments
#
# Model: 40 layers, 256 experts (8 active/tok), ~25 GB, RTX 4070 12 GB
#
# Strategy overview (least → most VRAM):
#
#   all-cpu-moe   All MoE expert tensors on CPU; only attention/shared tensors on GPU
#                 -ot ".ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
#
#   partial-cpu   Keep early layers on GPU, push layer 4+ experts to CPU
#                 -ot "blk\.(4|[5-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
#
#   up-down-cpu   Gate experts on GPU; up+down experts on CPU
#                 -ot ".ffn_(up|down)_(ch|)exps=CPU"
#
#   up-cpu        Only up experts on CPU; gate+down on GPU
#                 -ot ".ffn_up_(ch|)exps=CPU"
#
#   none          No override; use N_CPU_MOE integer path if set
#
# `(ch|)exps` matches both routed experts (`_exps`) and shared experts (`_shexp`)
# when present in tensor names.

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf}"
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

STRATEGY="${STRATEGY:-all-cpu-moe}"

if [ -z "${OVERRIDE_TENSOR:-}" ]; then
  case "${STRATEGY}" in
    all-cpu-moe)
      OVERRIDE_TENSOR=".ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
      ;;
    partial-cpu)
      OVERRIDE_TENSOR="blk\.(4|[5-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU"
      ;;
    up-down-cpu)
      OVERRIDE_TENSOR=".ffn_(up|down)_(ch|)exps=CPU"
      ;;
    up-cpu)
      OVERRIDE_TENSOR=".ffn_up_(ch|)exps=CPU"
      ;;
    none)
      OVERRIDE_TENSOR=""
      ;;
    *)
      echo "Unknown STRATEGY '${STRATEGY}'." >&2
      echo "Valid: all-cpu-moe | partial-cpu | up-down-cpu | up-cpu | none" >&2
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
echo

exec "${SCRIPT_DIR}/run-llama-bench.sh"
