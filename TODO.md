# TODO

## 🔴 Critical

- **Fix hardcoded shebang**: `model_downloader/download_hf_model.py` uses a
  local venv path — shipping should use `#!/usr/bin/env python3` with a proper
  install step (`pip install -e .`) so the right interpreter is always on `$PATH`
  *(note: intentionally kept for now to pin the correct HF-aware venv)*

## 🟡 High Priority

- **Benchmark result browser**: UI to view and compare results from
  `bench-results/`; parse timing/throughput from existing `.md` files and
  display in a sortable table — closes the run → bench → compare workflow loop
- **Run presets**: Named arg bundles per script (e.g. "fast", "low-vram",
  "debug") stored alongside the script; a preset selector next to
  `#run_extra_args` injects the bundle at launch time
- **Script templates**: "New script from template" action in the palette or Run
  tab; pick a base model family, fill in a few fields, get a ready-to-run `.sh`
  — removes the copy-and-edit friction for new models
- **Settings tab** *(currently placeholder)*: minimal config surface —
  default port, base models dir, binary path, theme; persisted to
  `~/.l3ms/settings.json`
- **Fix bare `except Exception:` patterns**: 20+ instances across `app.py`
  silently swallow errors; replace with specific exception types and route
  messages to the relevant `activity_log` / `RichLog`

## 🟡 Medium Priority

- **Implement script lint checks**: run `shellcheck` (if available) on save;
  show warnings inline in the script editor status line before allowing run
- **Fix datetime timezone consistency**: mix of naive `datetime.now()` and
  `datetime.now(timezone.utc)` across the codebase — standardise on UTC-aware
- **Add pagination**: large model lists (>100 entries) cause `DataTable`
  rendering lag; add virtual scrolling or page controls to the Download tab
- **Kill detected server action**: `action_quit` only cleans up processes
  started by the current session; add a "Kill running server" action to the
  Chat tab's Detect flow for servers started externally
- **Run telemetry improvements**: VRAM per process, disk I/O, network I/O
  alongside the existing CPU/RAM resource loop
- **Download progress estimates**: show ETA / remaining size based on
  `huggingface_hub` progress callbacks

## 🟢 Low Priority / Enhancements

- **Profile import/export**: save/load full homelab configs (models + scripts +
  settings) as a single portable bundle
- **Chat system prompt library**: save/load named system prompts; picker in
  the Chat params row
- **Bench result diffing**: side-by-side comparison of two bench runs in the
  result browser

## Repo / Platform

- Rename GitHub repository to `l3ms` (external GitHub action required)
- Keep TUI stable while preparing Rust port plan

## Long-term

- Port TUI core to Rust after workflow stabilizes
- Keep Python downloader compatibility layer during migration
- Consider WebAssembly port for browser-based access

## ✅ Done

- ~~Fix `preserve_existing` field loss~~ — removed from schema, no longer needed
- ~~Add `action_quit` method~~ — implemented with graceful subprocess + task cleanup
- ~~Add missing Maintenance tab output capture~~ — `run_script` no longer awaits
  its own task; output streams live
- ~~Add Jobs tab stop/retry functionality~~ — `■ Stop` / `↺ Retry` buttons,
  `script_path` + `mode` persisted in history, `s` / `r` shortcuts
- ~~Add proper task cleanup~~ — `action_quit` terminates processes and cancels
  resource/chat/run async tasks
- ~~Add command palette~~ — `Ctrl+P` `CommandPaletteScreen` with fuzzy filter
  over all app actions
- ~~Chat history persistence~~ — `save_chat` writes `.md` + `.json`;
  `Sessions` / `Load` open `ChatHistoryScreen` to restore sessions
- ~~Zellij-style footer~~ — only `F1–F6`, `q`, `?`, `Ctrl+P` visible; all
  `Ctrl+*` / `Alt+*` shortcuts in `?` help overlay grouped by context