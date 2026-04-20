#!/usr/bin/env bash
# Qwen3.6-35B-A3B (UD-Q5_K_XL) — automatic fit-based bench
#
# Delegates to run-llama-fit-bench.sh (fit -> bench).

MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-65536}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"

export MODEL TASKS THREADS CPU_RANGE FA MMP FIT_TARGET FIT_CTX BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V

exec "$(dirname -- "$0")/run-llama-fit-bench.sh"
