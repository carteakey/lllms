# Environment ----------------------------------------------------
export LLAMA_SET_ROWS=1
# 1 row per thread → better CPU cache locality
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export GGML_VK_ALLOW_SYSMEM_FALLBACK=0

# Model path -----------------------------------------------------
MODEL="/home/carteakey/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf"

# ----------------------------------------------------------------
# Server command (llama.cpp `llama-server`)
# ----------------------------------------------------------------
./vendor/llama.cpp/build/bin/llama-server \
    -m "$MODEL" \
    # This model has 36 layers. Keep first 31 MoE layers on CPU (5 stay on GPU)
    --n-cpu-moe 31 \
    # Offload the rest to the GPU
    --n-gpu-layers 99 \
    # 24 K context (fits comfortably in 12 GB VRAM)
    --ctx-size 24576 \
    # Faster prompt processing when the whole model fits in RAM
    --no-mmap \
    # Skip the 1‑pass warm‑up
    --no-warmup \
    -b 2048 \
    -ub 2048 \
    # Strangely, more threads help in my case. YMMV
    --threads 14 \
    # Pin threads to the 6 performance cores only
    --cpu-range 0-5 \
    # Strictly enforce the above
    --cpu-strict 1 \
    --temp 1.0 \
    # Limit choices to the top‑100 tokens (speed boost)
    --top-k 100 \
    --min-p 0.0 \
    --top-p 1.0 \
    # Enable flash‑attention (CUDA kernels)
    -fa \
    # For tool-calling
    --jinja \
    --reasoning-format none \
    # Proper way to select reasoning
    --chat-template-kwargs '{"reasoning_effort":"high"}' \
    --chat-template-file /home/carteakey/lllms/chat-template.jinja \
    --host 0.0.0.0 --port 8502 \
    --api-key "dummy"
```


    # --cpu-range 0-5 \
    # --cpu-strict 1 \
    # --swa-full \
    #     --cache-type-k q8_0 \
    #     --cache-type-v q4_0 \
    # --cache-reuse 0 \
    #
