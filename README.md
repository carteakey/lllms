# Local LLMS (lllms)

Script-first tooling for running local large language models (LLMs), including
model downloads and llama.cpp workflows for run, bench, and maintenance tasks.

## TUI

`L3MS` is the terminal UI for backend-first workflows.

Install:

```bash
python3 -m pip install -r requirements-tui.txt
```

Start:

```bash
python3 l3ms.py
```

Current TUI scope in this feature:

- Download tab with config path loader
- Model config editor (add/apply/delete)
- Config validation and save with versioned backups in `.toolkit/download_config_versions/`
- Download actions for selected model or all enabled models
- Keyboard-first shortcuts for Download workflow (`F1`, `Alt+T`, `Alt+I`, `Alt+O`, `Alt+W`, `Alt+V`, `Alt+D`, `Alt+E`)
- Run Models tab with run/bench script inventory, filter, start/stop, live output, and extra args
- Run Models script editor with per-script version snapshots in `.toolkit/script_versions/`
- Keyboard-first shortcuts for Run workflow (`F2`, `Ctrl+F`, `Ctrl+J`, `Ctrl+U`, `Ctrl+M`, `Ctrl+R`, `Ctrl+S`, `Ctrl+P`)
- Placeholder tabs for `Maintenance`, `Settings`, and `Jobs`

Roadmap note:

- `L3MS` is intentionally Python-first for fast iteration.
- Plan: port `L3MS` to Rust after feature scope stabilizes.

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
