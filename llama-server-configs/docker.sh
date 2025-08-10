docker run -d --gpus all \
  --name llamacpp-chatgpt120 \
  --restart unless-stopped \
  -p 8080:8080 \
  -v /home/infantryman/llamacpp:/models \
  llamacpp-server-cuda:latest \
  --model /models/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
  --alias chatgpt \
  --host 0.0.0.0 \
  --port 8080 \
  --jinja \
  --ctx-size 32768 \
  --n-cpu-moe 19 \
  --flash-attn \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --n-gpu-layers 999
```
