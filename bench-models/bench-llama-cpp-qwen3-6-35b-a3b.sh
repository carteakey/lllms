#!/usr/bin/env bash
# Qwen3.6-35B-A3B (UD-Q5_K_XL) — default llama.cpp bench
#
# Architecture:
#   40 transformer blocks, 256 experts per MoE layer, 8 active/token
#   ~25 GB on disk (single GGUF), RTX 4070 12 GB
#
# Safe baseline for 12 GB VRAM:
#   - keep all experts on CPU via OVERRIDE_TENSOR
#   - keep attention/non-expert tensors on GPU with ngl=99

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

export MODEL TASKS N_GPU_LAYERS OVERRIDE_TENSOR THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V

exec "$(dirname -- "$0")/run-llama-bench.sh"
