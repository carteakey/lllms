export LLAMA_SET_ROWS=1
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
export GGML_VK_ALLOW_SYSMEM_FALLBACK=0

# numactl -C0,1,2,3,4,5 ./vendor/llama.cpp/build/bin/llama-bench \
#     -m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
#     --n-cpu-moe 31 \
#     --n-gpu-layers 99 \
#     -t 6 \
#     --numa numactl

./vendor/llama.cpp/build/bin/llama-bench \
-m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
--n-cpu-moe 31 \
--n-gpu-layers 99 \
--threads 6 \
-ub 2048 \
-b 2048 \
--cache-type-k q8_0 \
--cache-type-v q4_0 \
--mmap 0
#     --temp 1.0 \
#     --top-k 100.0 \
#     --min-p 0.0 \
#     --top-p 1.0 \
# -v


# ./vendor/llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
#     --n-cpu-moe 31 \
#     --ctx-size 24576 \
#     --n-gpu-layers 999 \
#     --no-mmap \
#     --no-warmup \
#     -ub 2048 -b 2048 \
#     --cache-type-k q8_0 \
#     --cache-type-v q4_0 \
#     --threads 14 \
#     --cpu-range 0-5 \
#     --cpu-strict 1 \
#     --temp 1.0 \
#     --top-k 100.0 \
#     --min-p 0.0 \
#     --top-p 1.0 \
#     -fa on\
#     --jinja \
#     --reasoning-format none \
#     --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
#     --chat-template-kwargs "{\"reasoning_effort\": \"high\"}" \
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \
    # --swa-full \
    # --threads 6 \
    # --cpu-range 0-5 \
    # --cpu-strict 1 \
# ./vendor/llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
#     --n-cpu-moe 31 \
#     --ctx-size 16384 \
#     --n-gpu-layers 99 \
#     --threads 6 \
#     --cpu-range 0-5 \
#     --cpu-strict 1 \
#     --swa-full \
#     --no-warmup \
#     --temp 1.0 \
#     --top-k 100.0 \
#     --min-p 0.0 \
#     --top-p 1.0 \
#     -fa on\
#     --jinja \
#     --reasoning-format none \
#     --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
#     --chat-template-kwargs "{\"reasoning_effort\": \"medium\"}" \
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \

    # --cache-reuse 0 \
    # -ub 2048 -b 2048 \
    # --no-mmap \
    # --cache-type-k q8_0 \
    # --cache-type-v q4_0 \


# ./vendor/llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-IQ4_NL.gguf \
#     --n-cpu-moe 28 \
#     --ctx-size 32684 \
#     --n-gpu-layers 99 \
#     --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --presence-penalty 1.0 \
#     -fa \
#     --jinja \
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \


# ./vendor/llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf \
#     --n-cpu-moe 4 \
#     --ctx-size 32000 \
#     --n-gpu-layers 99 \
#     --temp 1.0 \
#     --min-p 0.0 \
#     --top-p 1.0 \
#     --top-k 20.0 \
#     -fa \
#     --jinja \
#     --reasoning-format none \
#     --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
#     --chat-template-kwargs "{\"reasoning_effort\": \"high\"}" \
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \

# ./vendor/llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/unsloth/gpt-oss-120b-GGUF/Q2_K_L/gpt-oss-120b-Q2_K_L-00001-of-00002.gguf \
#     --n-cpu-moe 31 \
#     --ctx-size 16384 \
#     --n-gpu-layers 99 \
#     --temp 1.0 \
#     --min-p 0.0 \
#     --top-p 1.0 \
#     --cache-type-k q8_0 \
#     --cache-type-v q4_0 \
#     --top-k 0.0 \
#     -fa \
#     --jinja \
#     --reasoning-format none \
#     --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
#     --chat-template-kwargs "{\"reasoning_effort\": \"high\"}" \
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \

# ./vendor/llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
#     --n-cpu-moe 31 \
#     --ctx-size 16384 \
#     --n-gpu-layers 99 \
#     --temp 1.0 \
#     --min-p 0.0 \
#     --top-p 1.0 \
#     --cache-type-k q8_0 \
#     --cache-type-v q4_0 \
#     --top-k 0.0 \
#     -fa \
#     --jinja \
#     --reasoning-format none \
#     --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
#     --chat-template-kwargs "{\"reasoning_effort\": \"high\"}" \
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \

# ./llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
#     --n-cpu-moe 31 \
#     --ctx-size 16384 \
#     --n-gpu-layers 99 \
#     --temp 1.0 \
#     --min-p 0.0 \
#     --top-p 1.0 \
#     --cache-type-k q8_0 \
#     --cache-type-v q4_0 \
#     --top-k 0.0 \
#     -fa \
#     --jinja \
#     --reasoning-format none \
#     --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
#     --chat-template-kwargs "{\"reasoning_effort\": \"high\"}"
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \

# ./llama.cpp/build/bin/llama-server \
#     -m models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
#     --jinja \
#     -fa \
#     -ngl 99 \
#     -ot ".ffn_.*_exps.=CPU" \
#     # -ot "blk.(1[8-9]|[2-4][0-9]).ffn_.*._exps.=CPU" -ub 768 -b 4096 -c 40960 \
#     # --threads -1 \
#     --ctx-size 16384 \
#     --temp 0.7 \
#     --min-p 0.0 \
#     --top-p 0.8 \
#     --top-k 20 \
#     --host 0.0.0.0 --port 8502 --api-key "dummy"


# ./llama.cpp/build/bin/llama-server \
#     -m models/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf \
#     --jinja \
#     -fa \
#     -ngl 99 \
#     -ot ".ffn_.*_exps.=CPU" \
#     --threads -1 \
#     --ctx-size 16384 \
#     --temp 1.0 \
#     --top-p 1.0 \
#     --top-k 0 \
#     --host 0.0.0.0 --port 8502 --api-key "dummy"



# ./llama.cpp/build/bin/llama-bench \
#     -m models/ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf \
#     -ot ".ffn_.*_exps.=CPU" \
#     -fa \
#     --jinja

# ./llama.cpp/build/bin/llama-bench \
#     -m models/Qwen3-Coder-30B-A3B-Instruct-IQ4_KSS.gguf \
#     -ot ".ffn_.*_exps.=CPU" \
#     # -fa
