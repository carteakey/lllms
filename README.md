# L3MS

`L3MS` (Local Large Language Model System) is a keyboard-first terminal toolkit for homelab LLM workflows.

## App Description

L3MS is built for developers who want script-first control with better ergonomics:

- Manage model download configs with validation and version history
- Run and bench llama.cpp models from curated script folders
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
- Placeholder tabs: `Maintenance`, `Settings`, `Jobs`

## Keyboard-first Controls

Global:

- `F1`: Download tab
- `F2`: Model Ops tab
- `F3`: Maintenance tab
- `F4`: Settings tab
- `F5`: Jobs tab

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
- `Ctrl+P`: save edited script snapshot
- `Ctrl+L`: clear run log

## Project Layout

- `model_downloader/`: Hugging Face downloader + model config
- `run-models/`: one `run-llama-cpp-*.sh` script per model
- `bench-models/`: one `bench-llama-cpp-*.sh` script per model
- `maintenance/`: system/build scripts
- `l3ms/`: TUI app + stores
- `l3ms.py`: launcher (TUI and CLI modes)

## Downloader CLI (direct)

```bash
python3 model_downloader/download_hf_model.py --config model_downloader/models_config.json --slow
```

## Versioning

This project uses semantic versioning. See [CHANGELOG.md](CHANGELOG.md).

## Roadmap Note

L3MS is intentionally Python-first for fast iteration.
Plan: port L3MS to Rust once feature scope stabilizes.
