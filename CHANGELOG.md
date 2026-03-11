# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Zellij-style `?` help overlay: footer now shows only tab-switch keys (`F1–F6`), `q`, `?`, and `Ctrl+P`; all `Ctrl+*`/`Alt+*` shortcuts moved to a `HelpScreen` modal grouped by context (Global / Jobs / Run / Chat / Download)
- **Command palette** (`Ctrl+P`): fuzzy-filtered `CommandPaletteScreen` modal lists every app action; type to narrow, Enter to run, Esc to cancel
- **Jobs tab stop + retry**: `■ Stop Running` and `↺ Retry Selected` buttons added to Jobs panel; running job shown with `▶` indicator; `s` / `r` key shortcuts when Jobs tab is active; `StopRequest` / `RetryRequest` messages routed through `L3MSApp` to `RunPanel`; `script_path` and `mode` now persisted in job history for reliable retry
- **Chat history persistence**: `save_chat` now writes both `.md` (human-readable) and `.json` (machine-loadable) to `~/.l3ms/chats/`; new `Sessions` / `Load` buttons open `ChatHistoryScreen` modal to browse and restore saved sessions
- **Graceful shutdown** (`action_quit`): on `q`, all running subprocesses are `terminate()`d and async resource/task loops are cancelled before exit — no more orphaned `llama-server` processes on quit
- `run-llama-cpp-nemotron-super-120b.sh`: run script for NVIDIA Nemotron 3 Super 120B-A12B UD-Q3_K_XL (latent-MoE, 12B active params, port 8001, ctx 32768)
- `NVIDIA-Nemotron-3-Super-120B-A12B-GGUF` UD-Q3_K_XL entry added to `models_config.json` (~62.6 GB, 3 split shards, enabled)

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
