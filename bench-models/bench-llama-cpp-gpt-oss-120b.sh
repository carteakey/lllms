#!/usr/bin/env bash
MODEL="${MODEL:-/mnt/lab/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf}"
N_CPU_MOE="${N_CPU_MOE:-31}"
export MODEL N_CPU_MOE
exec "$(dirname -- "$0")/run-llama-bench.sh"
