export LLAMA_SET_ROWS=1
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

# LLAMA_SET_ROWS=1 ./llama-server --api-key 1 -a qwen3 -m ~/dev/Qwen3-Coder-30B-A3B-Instruct-IQ4_NL.gguf -ngl 999 -ot "blk.(1[8-9]|[2-4][0-9]).ffn_.*._exps.=CPU" -ub 768 -b 4096 -c 40960 -ctk q5_1 -ctv q5_1 -fa
#this model has 36 MOE blocks. So cpu-moe 36 means all moe are running on the CPU. You can adjust this to move some MOE to the GPU, but it doesn't even make things that much faster.
#everything else on the GPU, about 8GB
#max context (128k), flash attention
    # --n-cpu-moe 31 \
    # -ot ".ffn_(up|down)_exps.=CPU" \
    #
#
./llama.cpp/build/bin/llama-server  \
    -m /home/kchauhan/Desktop/repos/lllms/models/ggml-org/gpt-oss-120b-GGUF/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
    --n-cpu-moe 31 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    --temp 1.0 \
    --min-p 0.0 \
    --top-p 1.0 \
    --cache-type-k q8_0 \
    --cache-type-v q4_0 \
    --top-k 0.0 \
    -fa \
    --jinja \
    --reasoning-format none \
    --chat-template-file /home/kchauhan/Desktop/repos/lllms/chat-template.jinja \
    --chat-template-kwargs "{\"reasoning_effort\": \"high\"}"
    --host 0.0.0.0 --port 8502 --api-key "dummy" \

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
