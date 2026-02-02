# Environment ----------------------------------------------------
export LLAMA_SET_ROWS=1
# 1 row per thread → better CPU cache locality
# export GGML_CUDA_ENABLE_UNIFIED_MEMORY=0
# export GGML_VK_ALLOW_SYSMEM_FALLBACK=0

sudo sysctl vm.swappiness=0

LLAMA_CPP_CUDA_PATH=./vendor/llama.cpp/build/bin/llama-server
LLAMA_CPP_VULKAN_PATH=./vendor/llama.cpp/build-vulkan/bin/llama-server

# Model path -----------------------------------------------------
MODEL="/home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
# ----------------------------------------------------------------
# Server command (llama.cpp `llama-server`)
# ----------------------------------------------------------------
taskset -c 0-11 $LLAMA_CPP_CUDA_PATH \
-m "$MODEL" \
--n-cpu-moe 32 \
--n-gpu-layers 99 \
--ctx-size 24576 \
--no-warmup \
-b 2048 \
-ub 2048 \
--temp 1.0 \
--top-k 100 \
--min-p 0.0 \
--top-p 1.0 \
-fa on \
--jinja \
--no-mmap \
--mlock \
--threads 10 \
--threads-batch 10 \
--reasoning-format none \
--chat-template-kwargs '{"reasoning_effort":"high"}' \
--chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
--host 0.0.0.0 --port 8502 \
--api-key "dummy"

# --threads 10 \
# --threads-batch 10 \
# --mlock \
# endor/llama.cpp/build/bin/llama-server \
# -m "$MODEL" \
# --n-cpu-moe 32 \
# --n-gpu-layers 99 \
# --ctx-size 24576 \
# --no-warmup \
# -b 2048 \
# -ub 2048 \
# --threads 11 \
# --threads-batch 11 \
# --temp 1.0 \
# --top-k 100 \
# --min-p 0.0 \
# --top-p 1.0 \
# --mlock \
# -fa on \
# --jinja \
# --reasoning-format none \
# --chat-template-kwargs '{"reasoning_effort":"high"}' \
# --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
# --host 0.0.0.0 --port 8502 \
# --api-key "dummy"

# prompt eval time =   40244.92 ms /  5506 tokens (    7.31 ms per token,   136.81 tokens per second)
#        eval time =   92791.30 ms /   964 tokens (   96.26 ms per token,    10.39 tokens per second)
#       total time =  133036.22 ms /  6470 tokens

# taskset -c 0-11 ./vendor/llama.cpp/build/bin/llama-server \
# -m "$MODEL" \
# --n-cpu-moe 31 \
# --n-gpu-layers 99 \
# --ctx-size 24576 \
# --no-mmap \
# --no-warmup \
# -b 2048 \
# -ub 2048 \
# --threads 11 \
# --threads-batch 11 \
# --cpu-range 0-11 \
# --cpu-strict 1 \
# --temp 1.0 \
# --top-k 100 \
# --min-p 0.0 \
# --top-p 1.0 \
# -fa on \
# --jinja \
# --reasoning-format none \
# --chat-template-kwargs '{"reasoning_effort":"high"}' \
# --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
# --host 0.0.0.0 --port 8502 \
# --api-key "dummy"

    # --cpu-range 0-5 \
    # --cpu-strict 1 \
    # --swa-full \
    #     --cache-type-k q8_0 \
    #     --cache-type-v q4_0 \
    # --cache-reuse 0 \
    #
