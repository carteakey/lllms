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

## llama-swap Operations

- Config hot-reload: `kill -HUP $(pgrep '[l]lama-swap')`. New/edited
  `llama-swap.yaml` entries are picked up without restarting the router
  (verified: no dropped connections, running models keep serving).
- Full-VRAM experiments (bench ladders, OOM hunting, fitting runs): pause the
  router and its spawned servers with `systemctl --user stop llama-swap.service`,
  run the experiment, then `systemctl --user start llama-swap.service`. A 12 GB
  card cannot hold two loaded models; llama-swap models claim the whole card.
- Requesting a model through the router swaps out whatever is loaded (server
  for the previous model is killed, new one spawned with its declared flags).
  `globalTTL: 600` unloads idle models after 10 minutes.
- New model entries: snapshot the yaml first
  (`llama-swap.yaml.bak-YYYYmmdd-HHMMSS`), add a `${...}_server` macro for any
  non-default binary under `vendor/`, then SIGHUP and smoke-test with a
  `max_tokens: 6` chat request before considering it wired.

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
