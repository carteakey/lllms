# L3MS Architecture

L3MS is a keyboard-first terminal application around editable homelab LLM
configuration and scripts. The Rust port is being developed incrementally so
the operational files remain useful throughout the migration.

## Sources of truth

- `llama-swap.yaml` declares models that can be served.
- `bench-models/bench-*.sh` declares benchmark entry points.
- `model_downloader/models_config.json` declares model downloads.
- The repository is authoritative for code and public documentation.
- Linear issue `CAR-97` is authoritative for the committed Rust-port work.

The terminal application orchestrates these files; it does not replace them
with an internal database or generated UI-only configuration.

## Rust application

The `l3ms` crate builds one binary and keeps the implementation in a small
module tree:

- `src/cli.rs` owns argument parsing, model/script pickers, repository
  discovery, and headless command execution.
- `src/llama_swap.rs` owns the authenticated llama-swap HTTP boundary.
- `src/chat.rs` owns bounded server-sent-event parsing and streaming chat
  requests.
- `src/commands.rs` owns typed command metadata, contextual help, and palette
  search.
- `src/config_store.rs` owns typed download configuration, normalization,
  validation, atomic writes, and configuration snapshots.
- `src/download_editor.rs` owns download-editor selection, validation, dirty
  state, and snapshot save/reload/restore operations independently of the UI.
- `src/gguf.rs` owns bounded GGUF v2/v3 metadata parsing and safe directory
  inventory.
- `src/job_history.rs` owns the persisted job lifecycle and safe reconstruction
  of retryable script jobs.
- `src/chat_history.rs` adapts the TUI chat transcript to the compatible
  JSON/Markdown session store.
- `src/script_store.rs` owns safe repository-contained script access,
  command construction, atomic writes, mode preservation, and snapshots.
- `src/script_editor.rs` owns reusable bench and maintenance editor state,
  including dirty tracking and snapshot save/reload/restore operations.
- `src/state_store.rs` owns the bounded legacy-compatible job and chat-session
  file formats.
- `src/telemetry.rs` owns process-tree CPU/RAM and optional NVIDIA memory
  sampling.
- `src/app.rs` owns the Ratatui event loop, shared model selection, background
  operations, supervised child processes, and the seven top-level views.

The TUI keeps network requests and subprocess output off the rendering thread.
Background delivery is bounded, process lines are size-capped, and each tick
handles a bounded number of events so continuous output cannot starve keyboard
input. On Unix, scripts run in their own process group with stdin detached;
shutdown sends `TERM`, waits briefly, then escalates to `KILL` if required.

## Runtime boundaries

llama-swap remains the serving daemon. The Rust client reads:

- `LLAMA_SWAP_URL`, defaulting to `http://localhost:8080`
- `LLAMA_SWAP_API_KEY`, when bearer authentication is enabled

The client uses `/v1/models`, `/models/load`, and `/models/unload`. Non-success
HTTP responses are errors; they are not recorded as successful model loads.

The existing Python downloader remains a compatibility child process. L3MS
executes `model_downloader/download_hf_model.py` directly so its pinned virtual
environment shebang continues to apply. Downloader behavior is not being
rewritten as part of the first Rust slice.

## Persistence and safety

Download configuration snapshots remain under
`.toolkit/download_config_versions/`. Script snapshots remain under
`.toolkit/script_versions/`. Context-aware store APIs take the runtime
repository and version roots explicitly, which keeps installed binaries and
alternate checkouts from writing snapshots into the build checkout.

Writes use a temporary file plus rename, snapshot names are unique even during
rapid saves, restore names cannot traverse out of their version directory, and
script paths must resolve inside the selected repository. Existing Unix script
permissions are preserved.

The Rust TUI uses bounded, atomic `jobs.json` persistence and compatible
JSON/Markdown chat sessions under `L3MS_DATA_DIR` (defaulting to `~/.l3ms`).
Job startup reconciles stale running entries, the UI keeps the newest 200 rows,
and retry reconstruction is limited to contained bench or maintenance scripts.
Malformed top-level job state is never overwritten until an explicit clear.

## Migration boundary

The Rust binary currently provides the headless llama-swap run/list workflow,
benchmark execution, typed stores, and a functional seven-view TUI. The TUI
now includes streamed chat with request parameters, an executable searchable
command palette, context-derived help, supervised processes, and live
CPU/RAM/GPU telemetry. Jobs and chat sessions persist through their compatible
legacy formats. The GGUF browser now uses the bounded metadata scanner and
supports recursive inventory, filtering, deterministic sorting, file details,
and per-file parse warnings.

The Python TUI remains available during the parity period for its richer script
and download editor surfaces, server detection, explicit model selection,
response cancellation, and remaining browser/editor controls. Reusable Rust
editor state now covers safe selection, validation, dirty tracking, and snapshot
save/reload/restore, but those engines are not yet wired into the TUI. The
Python TUI should only be retired after the corresponding Rust workflows have
compatibility coverage and live llama-swap smoke verification.

Development is paused at this coherent checkpoint. Linear issue `CAR-97`
contains the authoritative ordered resume checklist; this document records the
implemented boundary rather than maintaining a second backlog.
