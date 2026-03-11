# TODO

## 🔴 Critical

- **Fix `preserve_existing` field loss**: Config files have `preserve_existing` but it's filtered out by `MODEL_KEYS` in `config_store.py` - causes data loss on config reload
- **Fix hardcoded shebang**: `model_downloader/download_hf_model.py` uses `#!/home/kchauhan/repos/l3ms/.venv/bin/python3` - replace with `#!/usr/bin/env python3` and add `pip install -e .` setup
- **Add `action_quit` method**: Currently uses Textual default `q` binding; add explicit handler for consistency with other action patterns

## 🟡 High Priority

- **Add `preserve_existing` to MODEL_KEYS**: Include in `l3ms/config_store.py` to preserve existing files on download
- **Add missing Maintenance tab output capture**: Currently runs scripts but doesn't capture/parse output properly
- **Add Jobs tab stop/retry functionality**: History exists but cannot stop running jobs or retry failed ones
- **Fix bare `except Exception:` patterns**: 20+ instances across `app.py` - should catch specific exceptions and log properly
- **Add proper task cleanup**: Resource loops and async tasks should be cancelled on app shutdown

## 🟡 Medium Priority

- **Add command palette**: Keyboard-first discoverability for all actions (Ctrl+P or F12)
- **Implement script lint checks**: Shellcheck-style validation before save/run to catch syntax errors
- **Add run presets**: Named arg bundles per script (e.g., "fast", "low-vram", "debug")
- **Fix datetime timezone consistency**: Mix of naive `datetime.now()` and `datetime.now(timezone.utc)`
- **Add pagination**: Large model lists (>100) cause table rendering lag

## 🟢 Low Priority / Enhancements

- **Benchmark result browser**: UI to view and compare results from `bench-results/`
- **Profile import/export**: Save/load homelab configurations
- **Script templating helpers**: Fast model onboarding with templates
- **Chat history persistence**: Save/load chats from disk (not just export)
- **Run telemetry improvements**: VRAM per process, disk I/O, network I/O
- **Download progress estimates**: Show remaining time based on previous runs

## Repo / Platform

- Rename GitHub repository to `l3ms` (external GitHub action required)
- Keep TUI stable while preparing Rust port plan
- **Shipping**: fix Python interpreter resolution — `download_hf_model.py` shebang is currently hardcoded to a local venv path; shipping should use `#!/usr/bin/env python3` with a proper install step (`pip install -e .` or a setup script) so the right interpreter is always on `$PATH`

## Long-term

- Port TUI core to Rust after workflow stabilizes
- Keep Python downloader compatibility layer during migration
- Consider WebAssembly port for browser-based access
