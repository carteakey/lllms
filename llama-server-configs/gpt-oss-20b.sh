

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
