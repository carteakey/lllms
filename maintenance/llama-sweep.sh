model=/mnt/lab//models/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf
/home/kchauhan/repos/l3ms/vendor/ik_llama.cpp/build/bin/llama-sweep-bench \
--model "$model" \
-ctk q8_0 -ctv q8_0 \
-c 69632 \
-ub 1024 -b 2048 \
--merge-qkv \
-ngl 99 \
--n-cpu-moe 40 \
--threads 1 \
--warmup-batch \
-n 128
