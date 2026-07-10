# L3MS

`L3MS` (Local Large Language Model System) is a keyboard-first terminal toolkit for homelab LLM workflows.

## App Description

L3MS is built for developers who want script-first control with better ergonomics:

- Manage model download configs with validation and version history
- Run and bench llama.cpp models from curated script folders
- Browse local GGUF inventories with size, quantization, and metadata details
- Edit run/bench scripts in-place with snapshot restore
- Track live process output and runtime resource usage while a model is running
- Use either a full TUI or interactive CLI commands (`--run`, `--bench`)

## Public Dashboard

The [L3MS profile dashboard](https://l3ms.carteakey.dev) publishes the active
12 GB homelab configurations, measured generation throughput, task-level MTP
comparisons, and the reasons older profiles were retired.

Open any served profile to get a portable `llama-server` command generated
from `llama-swap.yaml`. Public commands replace local model paths with shell
variables and bind to `127.0.0.1` by default.

## Install

```bash
python3 -m pip install -r requirements-tui.txt
```

## Start TUI

```bash
python3 l3ms.py
```

Show quick-start instructions without opening TUI:

```bash
python3 l3ms.py --quickstart
```

## Interactive CLI Modes

List available scripts:

```bash
python3 l3ms.py --list all
```

Interactively run a model script:

```bash
python3 l3ms.py --run
```

Interactively run a bench script:

```bash
python3 l3ms.py --bench
```

Filter script picker and pass extra args:

```bash
python3 l3ms.py --run qwen --extra "--ctx-size 32768"
```

## TUI Scope

- Workbench shell:
  - persistent command bar with context-specific shortcut hints
  - richer `Ctrl+P` command palette with shortcut column and token filtering
- `Workbench` tab:
  - default first screen for quick llama-swap model loading
  - live model table from the configured llama-swap endpoint
  - fast actions for load, unload, chat, bench, browser, downloads, and jobs
  - delegates model loads to Model Ops so job history and resource telemetry stay consistent
- `Download` tab:
  - config load/save/validate/restore
  - model row add/apply/delete
  - download selected or enabled models
  - config snapshots in `.toolkit/download_config_versions/`
- `Model Ops` tab:
  - llama-swap run mode and bench script mode
  - live run logs + start/stop
  - current running model + resource telemetry (CPU/RAM/GPU when available)
  - script editor and per-script snapshots in `.toolkit/script_versions/` for bench mode
- `Model Browser` tab:
  - scan any local directory for `.gguf` files
  - inspect size, quantization, params, architecture, and modified timestamp in a table
  - filter/sort results and inspect per-file metadata details
- Additional tabs: `Chat`, `Maintenance`, `Jobs`

## Keyboard-first Controls

Global:

- `Ctrl+P`: command palette
- `?`: key binding help
- `F1`: Workbench tab
- `F2`: Model Ops tab
- `F3`: Chat tab
- `F4`: Model Browser tab
- `F5`: Download tab
- `F6`: Jobs tab
- `F7`: Maintenance tab
- `Alt+1..Alt+7`: tab fallback when F-keys are unreliable
- `Alt+←` / `Alt+→`: previous / next tab

Workbench:

- `Ctrl+R` / `Enter`: load selected llama-swap model
- `Ctrl+S`: unload selected llama-swap model
- `Ctrl+F`: focus model filter
- `Ctrl+J`: focus model table
- `Ctrl+L`: clear workbench log
- `F3`: chat with the loaded model endpoint
- `F2`: open full Model Ops

Download (active only on Download tab):

- `Alt+T`: focus models table
- `Alt+I`: focus model editor
- `Alt+O`: load config
- `Alt+W`: save config
- `Alt+V`: validate config
- `Alt+N`: add model
- `Alt+A`: apply model edit
- `Alt+K`: delete selected model
- `Alt+D`: download selected model
- `Alt+E`: download enabled models
- `Alt+Y`: clear download log

Model Ops (active only on Model Ops tab):

- `Ctrl+F`: focus model/script filter
- `Ctrl+J`: focus model/script table
- `Ctrl+U`: focus detail/script editor
- `Ctrl+M`: toggle run/bench mode
- `Ctrl+R`: load selected llama-swap model or run selected bench script
- `Ctrl+S`: unload selected llama-swap model or stop running bench
- `Alt+P`: save edited bench script snapshot
- `Ctrl+L`: clear run log

Model Browser (active only on Model Browser tab):

- `Alt+R`: scan selected GGUF directory
- `Alt+G`: focus directory path input
- `Alt+J`: focus GGUF table

## Project Layout

- `model_downloader/`: Hugging Face downloader + model config
- `llama-swap.yaml`: single source of truth for servable models (see `docs/llama-swap-runbook.md`)
- `bench-models/`: one `bench-llama-cpp-*.sh` script per model
- `maintenance/`: system/build scripts
- `maintenance/systemd/`: user service units (including `llama-swap.service` and `nanobot-gateway.service`)
- `l3ms/`: TUI app + stores
- `l3ms.py`: launcher (TUI and CLI modes)
- `docs/llama-swap-runbook.md`: install, start/stop, curl, add-a-model
- `docs/model-onboarding-playbook.md`: end-to-end checklist for adding new model families

## Serving

Models are served by [llama-swap](https://github.com/mostlygeek/llama-swap) on
a single OpenAI-compatible endpoint (`http://<host>:8080`). The daemon
hot-swaps models on demand and exposes every entry in `llama-swap.yaml`
under `/v1/models`. See `docs/llama-swap-runbook.md`.

## Downloader CLI (direct)

Run with config file (downloads all enabled models):

```bash
./model_downloader/download_hf_model.py --config model_downloader/models_config.json
```

Pull updates for already-downloaded models (skips models with no local files):

```bash
./model_downloader/download_hf_model.py --config model_downloader/models_config.json --update
```

Single model with pattern filter:

```bash
./model_downloader/download_hf_model.py --repo-id Qwen/Qwen3-32B-GGUF --allow-patterns "*Q6_K*"
```

Throttle concurrency (useful on metered connections):

```bash
./model_downloader/download_hf_model.py --config model_downloader/models_config.json --slow
./model_downloader/download_hf_model.py --config model_downloader/models_config.json --max-workers 2
```

| Flag | Short | Description |
|------|-------|-------------|
| `--config` | `-c` | Path to JSON config file |
| `--repo-id` | `-r` | Single repo to download |
| `--allow-patterns` | `-a` | File glob patterns to include |
| `--ignore-patterns` | `-i` | File glob patterns to exclude |
| `--local-dir` | `-d` | Override local destination directory |
| `--revision` | | Specific branch/tag/commit to pin |
| `--update` | `-u` | Sync updates for models already on disk (skips new ones) |
| `--force-download` | | Re-download all files even if already present |
| `--slow` | | Throttle to `max_workers=4` |
| `--max-workers` | | Explicit worker count |
| `--base-models-dir` | | Override base directory for auto-organized downloads |

> **Note**: Run the script directly (`./model_downloader/download_hf_model.py`) rather than via `python3` to ensure the correct venv Python is used.

## Dashboard Development

The served-profile data and commands in `docs/generated-models.js` are
generated from `llama-swap.yaml`. Presentation-only values such as measured
throughput and benchmark comparisons live in `docs/dashboard-meta.json`.

Regenerate after changing either source:

```bash
python3 docs/generate_dashboard_data.py
```

Preview locally:

```bash
python3 -m http.server 8080 -d docs
```

Do not edit `docs/generated-models.js` by hand.

## Versioning

This project uses semantic versioning. See [CHANGELOG.md](CHANGELOG.md).

## Roadmap Note

L3MS is intentionally Python-first for fast iteration.
Plan: port L3MS to Rust once feature scope stabilizes.
