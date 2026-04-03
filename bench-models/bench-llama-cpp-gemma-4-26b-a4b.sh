#!/usr/bin/env bash
# Gemma-4-26B-A4B-it (UD-Q5_K_XL) — default llama.cpp bench
#
# Uses Gemma-recommended sampling and q8_0 KV cache defaults.

MODEL="${MODEL:-/home/kchauhan/models/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf}"
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

export MODEL TASKS N_GPU_LAYERS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V

exec "$(dirname -- "$0")/run-llama-bench.sh"
