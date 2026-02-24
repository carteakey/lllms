# Local LLMS (lllms)

Script-first tooling for running local large language models (LLMs), including
model downloads and llama.cpp workflows for run, bench, and maintenance tasks.

## Project Layout

- `model_downloader/`: Hugging Face downloader + model config
- `run-models/`: one `run-llama-cpp-*.sh` server script per model
- `bench-models/`: one `bench-llama-cpp-*.sh` benchmark script per model
- `maintenance/`: system/build scripts (`install-cuda.sh`, `build-llama-cpp*.sh`)
- `vendor/llama.cpp/`: llama.cpp source checkout/build target

## Downloader CLI

Use the downloader directly with config file support, safe resume behavior, and
worker throttling.

```bash
python3 model_downloader/download_hf_model.py --config model_downloader/models_config.json --slow
```

Download a single model with explicit throttling:

```bash
python3 model_downloader/download_hf_model.py \
  --repo-id ggml-org/gpt-oss-20b-GGUF \
  --allow-patterns '*Q8_0*' \
  --max-workers 2
```

## Run And Bench

Run a model server:

```bash
bash run-models/run-llama-cpp-gpt-oss-20b.sh
```

Run a benchmark:

```bash
bash bench-models/bench-llama-cpp-gpt-oss-20b.sh
```

## Maintenance

Build llama.cpp with CUDA:

```bash
bash maintenance/build-llama-cpp.sh
```

Install CUDA dependencies:

```bash
bash maintenance/install-cuda.sh
```
