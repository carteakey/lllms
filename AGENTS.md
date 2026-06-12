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

- `l3ms/app.py`: TUI layout, keybindings, workflows
- `l3ms/config_store.py`: download config CRUD + validation + snapshots
- `l3ms/script_store.py`: script CRUD + snapshots
- `l3ms.py`: launcher + interactive CLI (`--run`, `--bench`, `--list`)

## Versioning

Use semantic versioning (`MAJOR.MINOR.PATCH`):

- `MAJOR`: breaking workflow or API changes
- `MINOR`: new features (tabs, actions, commands)
- `PATCH`: bug fixes, UX polish, non-breaking improvements

Update `CHANGELOG.md` on every user-visible change.

## Release Intent

Current implementation is Python-first for speed.
Future intent is a Rust port once workflows are stable.
