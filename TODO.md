# TODO

## Near-term

- Implement Maintenance tab actions and output capture
- Implement Jobs tab with persisted history, live tails, and stop/retry
- Add richer run telemetry (VRAM split by process tree, disk I/O, net I/O)
- Add run presets (named arg bundles) per script
- Add safer script lint checks before save/run (shellcheck-style warnings)

## Mid-term

- Add benchmark result browser and compare view from `bench-results/`
- Add command palette for all actions (keyboard-first discoverability)
- Add profile import/export for homelab setups
- Add optional script templating helpers for fast model onboarding

## Repo / Platform

- Rename GitHub repository to `l3ms` (external GitHub action required)
- Keep TUI stable while preparing Rust port plan

## Long-term

- Port TUI core to Rust after workflow stabilizes
- Keep Python downloader compatibility layer during migration
