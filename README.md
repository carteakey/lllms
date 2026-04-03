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

## Install

```bash
python3 -m pip install -r requirements-tui.txt
```

## Start TUI

```bash
python3 l3ms.py
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

- `Download` tab:
  - config load/save/validate/restore
  - model row add/apply/delete
  - download selected or enabled models
  - config snapshots in `.toolkit/download_config_versions/`
- `Model Ops` tab:
  - run/bench script inventory and filtering
  - live run logs + start/stop
  - current running model + resource telemetry (CPU/RAM/GPU when available)
  - script editor and per-script snapshots in `.toolkit/script_versions/`
- `Model Browser` tab:
  - scan any local directory for `.gguf` files
  - inspect size, quantization, params, architecture, and modified timestamp in a table
  - filter/sort results and inspect per-file metadata details
- Additional tabs: `Chat`, `Maintenance`, `Settings`, `Jobs`

## Keyboard-first Controls

Global:

- `F1`: Download tab
- `F2`: Model Ops tab
- `F3`: Chat tab
- `F4`: Maintenance tab
- `F5`: Settings tab
- `F6`: Jobs tab
- `F7`: Model Browser tab

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

- `Ctrl+F`: focus script filter
- `Ctrl+J`: focus script table
- `Ctrl+U`: focus script editor
- `Ctrl+M`: toggle run/bench mode
- `Ctrl+R`: run selected script
- `Ctrl+S`: stop running script
- `Alt+P`: save edited script snapshot
- `Ctrl+L`: clear run log

Model Browser (active only on Model Browser tab):

- `Alt+R`: scan selected GGUF directory
- `Alt+G`: focus directory path input
- `Alt+J`: focus GGUF table

## Project Layout

- `model_downloader/`: Hugging Face downloader + model config
- `run-models/`: one `run-llama-cpp-*.sh` script per model
- `bench-models/`: one `bench-llama-cpp-*.sh` script per model
- `maintenance/`: system/build scripts
- `maintenance/systemd/`: user service units (optional startup services)
- `l3ms/`: TUI app + stores
- `l3ms.py`: launcher (TUI and CLI modes)
- `docs/model-onboarding-playbook.md`: end-to-end checklist for adding new model families

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

## Versioning

This project uses semantic versioning. See [CHANGELOG.md](CHANGELOG.md).

## Roadmap Note

L3MS is intentionally Python-first for fast iteration.
Plan: port L3MS to Rust once feature scope stabilizes.
