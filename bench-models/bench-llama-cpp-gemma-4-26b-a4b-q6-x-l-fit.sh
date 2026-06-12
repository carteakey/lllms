#!/usr/bin/env bash
# Gemma-4-26B-A4B-it (UD-Q6_K_XL) — automatic fit-based bench
# Delegates to run-llama-fit-bench.sh; set env vars to override defaults.

MODEL="${MODEL:-/mnt/lab/models/unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf}"
TASKS="${TASKS:-512,128}"
THREADS="${THREADS:-10}"
CPU_RANGE="${CPU_RANGE:-0-11}"
FA="${FA:-1}"
MMP="${MMP:-0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
FIT_TARGET="${FIT_TARGET:-512}"
FIT_CTX="${FIT_CTX:-32768}"

export MODEL TASKS THREADS CPU_RANGE FA MMP BATCH_SIZE UBATCH_SIZE CACHE_TYPE_K CACHE_TYPE_V FIT_TARGET FIT_CTX

exec "$(dirname -- "$0")/run-llama-fit-bench.sh"
