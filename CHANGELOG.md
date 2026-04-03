# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- **Gemma-4-26B-A4B workflow support**:
  - `run-models/run-llama-cpp-gemma-4-26b-a4b.sh` run script targeting mainline `vendor/llama.cpp/build/bin/llama-server`
  - `run-models/run-llama-cpp-gemma-4-26b-a4b-vision.sh` dedicated vision preset wiring `mmproj-BF16.gguf`
  - default contexts now aligned to this local profile: text `128k`, vision `64k`
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b.sh` baseline bench script
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-strategies.sh` strategy sweep bench script
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-fit.sh` fit-based bench script
  - `model_downloader/models_config.json` profile for `unsloth/gemma-4-26B-A4B-it-GGUF` (`UD-Q5_K_XL` + `mmproj-BF16`)
  - `docs/bench-runbook.md` quickstart section for Gemma-4-26B-A4B on mainline llama.cpp
- **Gemma vision user service support**:
  - `maintenance/systemd/gemma-vision.service` user-level systemd unit for always-on startup
  - `maintenance/setup-gemma-vision-service.sh` helper for `install/start/stop/restart/enable/disable/status/logs`
  - `docs/bench-runbook.md` usage section for managing Gemma vision as a startup service
- **gpt-oss-puzzle-88B workflow support**:
  - `maintenance/build-gpt-oss-puzzle-llama-cpp.sh` wrapper for upstream PR merge build flow (defaults to PR `#21032` via `llama-test-pr.sh`)
  - `run-models/run-llama-cpp-gpt-oss-puzzle-88b.sh` run script targeting puzzle-compatible llama.cpp build output
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b.sh` baseline bench script
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-strategies.sh` strategy sweep bench script (defaulted to fit-shaped partial split, `ngl=37`, semicolon-delimited `-ot` patterns)
  - `bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-fit.sh` fit-based bench script
  - `model_downloader/models_config.json` profile for `SamPurkis/gpt-oss-puzzle-88B-GGUF` (`*MXFP4_MOE*` primary pattern)
  - `docs/bench-runbook.md` quickstart section + recorded benchmark results (`pp/tg`) for baseline, all-cpu-moe, and fit/fit-shaped partial split
- **Model onboarding playbook**:
  - new `docs/model-onboarding-playbook.md` documenting end-to-end model-family onboarding (build wrapper, downloader profile, run/bench scripts, docs/changelog, validation, and targeted download flow)
- **GGUF Model Browser tab**:
  - New `Model Browser` tab to scan any local directory for `.gguf` files (recursive or top-level)
  - Sortable/filterable table with quantization, size, parameter count, architecture, and modified time
  - Lightweight GGUF header parser for per-file metadata (model name, architecture, tokenizer, tensor count)
  - New keyboard actions: `Alt+R` scan, `Alt+G` focus path, `Alt+J` focus table, and `F7` tab switch
- **Sarvam 30B workflow support**:
  - `maintenance/llama-test-pr.sh` now defaults to PR-specific vendor folders: `vendor/llama.cpp-pr-test-<joined-prs>`
  - `maintenance/build-sarvam-llama-cpp.sh` is now a thin wrapper over `maintenance/llama-test-pr.sh` with default `SARVAM_PR_NUMBER=20275`
  - `run-models/run-llama-cpp-sarvam-30b.sh` updated with cleaner server flags and Sarvam-aligned sampling defaults (`temp=1.0`, `top_p=1.0`, `top_k=20`)
  - `bench-models/bench-llama-cpp-sarvam-30b.sh` for repeatable `llama-bench` runs against the Sarvam build
  - `bench-models/bench-llama-cpp-sarvam-30b-fit.sh` for automatic fit-based benching
  - `bench-models/fit-params-sarvam-30b.sh` for printing fitted `-ngl/-ts/-ot` placement args
  - `model_downloader/models_config.json` entries for `Sumitc13/sarvam-30b-GGUF` (Q6_K) and `limegreenpeper1/sarvam-105B-GGUF` (Q4_K_M default)
  - `docs/sarvam-local-post.md`: first natural-language draft post for running Sarvam locally (30B workflow + 105B download profile)
- **gpt-oss-120b bench suite**: `bench-llama-cpp-gpt-oss-120b.sh`, `bench-llama-cpp-gpt-oss-120b-strategies.sh`, `bench-llama-cpp-gpt-oss-120b-fit.sh`, `bench-ik-llama-cpp-gpt-oss-120b.sh` — full runbook coverage for gpt-oss-120b mxfp4 on 64 GB RAM systems
- **Qwen3.5-122B-A10B bench suite**: `bench-llama-cpp-qwen3-5-122b-a10b.sh`, `bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh`, `bench-llama-cpp-qwen3-5-122b-a10b-fit.sh`, `bench-ik-llama-cpp-qwen3-5-122b-a10b.sh` — full bench coverage; documents shared-expert `(ch|)exps` pattern gotcha
- **`run-llama-cpp-gpt-oss-120b-optimized.sh`**: optimized server run script with static `-ngl 37 --override-tensor` (fit-derived), `--parallel 1`, explicit `--ctx-size 32768`; drops `--fit` startup overhead; confirmed +540 MiB more model weight on GPU and 28 t/s tg vs 27 t/s with original script
- **`maintenance/build-llama-cpp-cublas.sh`**: builds llama.cpp with `GGML_CUDA_FORCE_CUBLAS=ON` + `GGML_CUDA_FORCE_DMMV=OFF` into a separate `build-cublas/` dir; tested against gpt-oss-120b, found slower than default build (GGML MMQ mxfp4 kernel wins at decode-batch sizes)
- **`docs/bench-runbook.md §8`**: bench results for Qwen3.5-122B-A10B and gpt-oss-120b; documents shared-expert OOM root cause, RAM ceiling constraints, cuBLAS/ik_llama findings, static-ot vs fit VRAM breakdown comparison, and active-parameter tg scaling table
- **`2025-09-21-optimizing-gpt-oss-120b-local-inference.md`**: updated TL;DR tg to 28 t/s, pp to 420+; added `llama-fit-params` workflow section; expanded `--override-tensor` section with shared-expert gotcha and RAM ceiling warning; updated run script to static placement + `--parallel 1`; added l3ms repo link; closed out cuBLAS and ik_llama experiments
- Zellij-style `?` help overlay: footer now shows only tab-switch keys (`F1–F7`), `q`, `?`, and `Ctrl+P`; all `Ctrl+*`/`Alt+*` shortcuts moved to a `HelpScreen` modal grouped by context (Global / Jobs / Run / Chat / Download)
- **Command palette** (`Ctrl+P`): fuzzy-filtered `CommandPaletteScreen` modal lists every app action; type to narrow, Enter to run, Esc to cancel
- **Jobs tab stop + retry**: `■ Stop Running` and `↺ Retry Selected` buttons added to Jobs panel; running job shown with `▶` indicator; `s` / `r` key shortcuts when Jobs tab is active; `StopRequest` / `RetryRequest` messages routed through `L3MSApp` to `RunPanel`; `script_path` and `mode` now persisted in job history for reliable retry
- **Chat history persistence**: `save_chat` now writes both `.md` (human-readable) and `.json` (machine-loadable) to `~/.l3ms/chats/`; new `Sessions` / `Load` buttons open `ChatHistoryScreen` modal to browse and restore saved sessions
- **Graceful shutdown** (`action_quit`): on `q`, all running subprocesses are `terminate()`d and async resource/task loops are cancelled before exit — no more orphaned `llama-server` processes on quit
- `run-llama-cpp-nemotron-super-120b.sh`: run script for NVIDIA Nemotron 3 Super 120B-A12B UD-Q3_K_XL (latent-MoE, 12B active params, port 8001, ctx 32768)
- `NVIDIA-Nemotron-3-Super-120B-A12B-GGUF` UD-Q3_K_XL entry added to `models_config.json` (~62.6 GB, 3 split shards, enabled)
- **Mistral Small 4 (119B) script set**:
  - `run-models/run-llama-cpp-mistral-small-4-119b.sh` standard fit-based server script (safe defaults)
  - `run-models/run-llama-cpp-mistral-small-4-119b-optimized.sh` throughput-oriented static-placement script (`-ngl`, `--override-tensor`, `q8_0` KV, `--parallel 1`)
  - `run-models/run-llama-cpp-mistral-small-4-119b-optimized-no-vision.sh` explicit non-vision optimized preset (same tuned defaults as current non-vision path)
  - `run-models/run-llama-cpp-mistral-small-4-119b-optimized-vision.sh` explicit vision optimized preset with `--mmproj` and one fewer GPU layer by default (`-ngl 8`)
  - `bench-models/bench-llama-cpp-mistral-small-4-119b-strategies.sh` strategy sweep script that compares tg across offload presets and reports the best strategy
- **Bench + run script expansion**:
  - Added model-specific strategy/fit benches for Nemotron 120B, Qwen3.5-122B-A10B, Sarvam 30B, and gpt-oss-120B under `bench-models/`
  - Added optimized launch presets `run-llama-cpp-qwen3-coder-next-optimized.sh` and `run-llama-cpp-mistral-small-4-119b-optimized-no-vision-thinking.sh`
  - Added `preflight-check.sh` and maintenance helpers (`llama-sweep.sh`, `llama-test-pr.sh`) for repeatable local validation workflows
- **Docs additions**:
  - Added `docs/vibe_configuration.md` and expanded bench/run workflow guidance for current local model ops

### Fixed
- `MarkupError` crash in `refresh_disk_space`: paths like `/home/user` inside `[…]` were parsed as Rich closing tags; fixed by passing the full `[path]` token through `markup_escape()`
- **Maintenance tab output capture**: `run_script` no longer `await`s its own task — script output now streams live to `maint_log` without blocking the event loop
- `on_run_panel_job_started` and `on_run_panel_job_finished` now forward `mode` correctly to `JobsPanel`
- `ctrl+p` remapped from `run_save_script` (moved to `alt+p`) to `show_command_palette` for consistency with editor convention

- Launcher CLI interactive modes: `--run`, `--bench`, `--list`, `--extra`
- Model Ops runtime status with current model and resource telemetry
- Keyboard-first action scoping to active tab to prevent unwanted tab switching
- Project docs set: `TODO.md`, `AGENTS.md`, and app description updates
- New Qwen3.5 run config scripts for four modes: thinking-general, thinking-coding, instruct-general, instruct-reasoning

### Changed

- Renamed tab label from `Run Models` to `Model Ops`
- Updated docs branding to `L3MS`

### Fixed

- Global shortcut actions no longer force-switch to the Run tab

## [0.4.0] - 2026-02-24

### Added

- Keyboard-first Download controls and shortcuts
- Run script editor with save/restore snapshots in `.toolkit/script_versions/`

## [0.3.0] - 2026-02-24

### Added

- Keyboard-first Run tab with run/bench mode, script filter, start/stop, and live logs

## [0.2.0] - 2026-02-24

### Added

- Renamed toolkit TUI to `L3MS`

## [0.1.0] - 2026-02-24

### Added

- Initial Textual TUI with Download config editor/updater
- Config validation and snapshot history for model config files
