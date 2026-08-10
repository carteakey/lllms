# AGENTS

Agent and contributor guide for `L3MS`.

## Purpose

Build a keyboard-first, script-first homelab LLM toolkit with strong operational ergonomics.

## Engineering Rules

- Prefer deterministic script orchestration over hidden automation.
- Serving is declared in `llama-swap.yaml`; benching is declared in
  `bench-models/*.sh`. Both are editable text, not hard-parameterized UI
  forms. When serving and benching flags drift, update them together.
- Treat keyboard control as first-class; mouse workflows are optional.
- Preserve existing files by default for downloads unless explicitly overridden.
- Keep version snapshots for both model configs and scripts before writes.

## Structure

- `src/app.rs`: Ratatui layout, keybindings, workflows, and process supervision
- `src/cli.rs`: launcher + interactive CLI (`--run`, `--bench`, `--list`)
- `src/llama_swap.rs`: authenticated llama-swap HTTP boundary
- `src/config_store.rs`: download config CRUD + validation + snapshots
- `src/script_store.rs`: script CRUD + snapshots
- `l3ms/` and `l3ms.py`: legacy Python TUI retained during parity work
- `model_downloader/`: Python downloader compatibility boundary

## Versioning

Use semantic versioning (`MAJOR.MINOR.PATCH`):

- `MAJOR`: breaking workflow or API changes
- `MINOR`: new features (tabs, actions, commands)
- `PATCH`: bug fixes, UX polish, non-breaking improvements

Update `CHANGELOG.md` on every user-visible change.

## Release Intent

The Rust port is active under `CAR-97` and starts at version `0.7.0`.
Retain the legacy Python TUI and downloader until their Rust replacement or
compatibility boundary has explicit parity coverage.

## Planning Authority

Linear is authoritative for committed L3MS work. The repository and Markdown
remain authoritative for code and canonical documentation. The single Linear
project is `L3MS` with milestone `Operations backlog completion — 2026-08-10`.
The reconciled TODO groups are tracked by `CAR-140` through `CAR-146`, while
`CAR-97` remains the parent Rust-port history and is attached to the same
project/milestone. `TODO.md` is reserved for speculative or research notes
that have not been intentionally committed.
