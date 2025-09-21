& "C:\llama-cpp\llama-server.exe" `
--host [127.0.0.1](http://127.0.0.1) --port 9045 `
--model "C:\llama-cpp\models\Qwen3-30B-A3B.Q8_0.gguf" `
--n-gpu-layers 99 --flash-attn --slots --metrics `
--ubatch-size 512 --batch-size 512 `
--presence-penalty 1.5 `
--cache-type-k q8_0 --cache-type-v q8_0 `
--no-context-shift --ctx-size 32768 --n-predict 32768 `
--temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 `
--repeat-penalty 1.1 --jinja --reasoning-format deepseek `
--threads 5 --threads-http 5 --cache-reuse 256 `
--override-tensor 'blk\.([0-9]*[02468])\.ffn_.*_exps\.=CPU' `
--no-mmap


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


# ./vendor/llama.cpp/build/bin/llama-server  \
#     -m /home/kchauhan/Desktop/repos/lllms/models/qwen/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-IQ4_NL.gguf \
#     --n-cpu-moe 28 \
#     --ctx-size 32684 \
#     --n-gpu-layers 99 \
#     --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --presence-penalty 1.0 \
#     -fa \
#     --jinja \
#     --host 0.0.0.0 --port 8502 --api-key "dummy" \
