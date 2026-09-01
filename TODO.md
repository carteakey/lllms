# TODO

All intentionally committed TODO items were reconciled into the single Linear
`L3MS` project and its `Operations backlog completion — 2026-08-10` milestone
on 2026-08-10. See CAR-140 through CAR-146 for the seven grouped issues and
their acceptance criteria. Linear is the source of truth for committed work;
this file is reserved for ideas and research that have not been intentionally
committed.

## Speculative / research

There are no uncommitted ideas at this time. Add future possibilities here
only until they are explicitly chosen and promoted to Linear.

## ✅ Done

- ~~Generated public profile dashboard~~ — served profiles and portable
  `llama-server` commands now derive from `llama-swap.yaml`; benchmark metadata
  stays separate, rankings are deterministic, and mobile uses readable cards.
- ~~TUI: teach Model Ops about llama-swap~~ — Run mode now reads from
  `/v1/models`; Start/Stop call `/models/load` and `/models/unload`;
  editor shows model state + curl snippets. Bench mode unchanged.
- ~~Chat tab default port~~ — already `8080` (checked post-migration,
  TODO was stale)
- ~~Retire `gemma-vision.service`~~ — moved unit + helper to
  `maintenance/systemd/archive/` and `maintenance/archive/`; llama-swap
  preloads `gemma-4-26b-a4b-vision` instead
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
