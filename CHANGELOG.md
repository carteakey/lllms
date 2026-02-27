# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added

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
