CTX_SIZE=${CTX_SIZE:-131072}         # big KV caches eat VRAM – 8k is plenty
N_GPU_LAYERS=${N_GPU_LAYERS:-99}     # off-load as many layers as fit
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-2048}     # micro-batch size

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}

THREADS=${THREADS:-$(nproc)}
THREADS_BATCH=${THREADS_BATCH:-$(( THREADS - 2 ))}
THREADS_HTTP=${THREADS_HTTP:-4}
N_CPU_MOE=${N_CPU_MOE:-2}
LLAMA_PARALLEL=${LLAMA_PARALLEL:-1}  # 1 copy avoids extra VRAMCTX_SIZE=${CTX_SIZE:-131072}         # big KV caches eat VRAM – 8k is plenty
N_GPU_LAYERS=${N_GPU_LAYERS:-99}     # off-load as many layers as fit
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-2048}     # micro-batch size


HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8080}


THREADS=${THREADS:-$(nproc)}
THREADS_BATCH=${THREADS_BATCH:-$(( THREADS - 2 ))}
THREADS_HTTP=${THREADS_HTTP:-4}
N_CPU_MOE=${N_CPU_MOE:-2}
LLAMA_PARALLEL=${LLAMA_PARALLEL:-1}  # 1 copy avoids extra VRAM


exec "$LLAMA_DIR/build/bin/llama-server" \
  --model "$MODEL_PATH" \
  --alias "${MODEL_NAME,,}" \
  --threads "$THREADS" \
  --threads-batch "$THREADS_BATCH" \
  --threads-http "$THREADS_HTTP" \
  --ctx-size "$CTX_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --n-cpu-moe "$N_CPU_MOE" \
  --host "$HOST" \
  --port "$PORT" \
  --parallel "$LLAMA_PARALLEL" \
  --flash-attn \
  --jinja \
  --tensor-split 0.33,0.33,0.34 \
  --verbose \
  --timeout 600 \
  --chat-template-kwargs '{"reasoning_effort": "low"}' \
  --temp 1.0 \
  --min-p 0.0 \
  --top-p 1.0 \
  --top-k 0.0
