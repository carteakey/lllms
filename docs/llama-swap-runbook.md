# llama-swap runbook

Single-port OpenAI-compatible endpoint that hot-swaps between every model in
`llama-swap.yaml`. Replaces the 28 per-model `run-models/*.sh` scripts.

## Install

Download a release binary into `~/bin/`:

```bash
mkdir -p ~/bin
# pick the linux_amd64 tarball from https://github.com/mostlygeek/llama-swap/releases
curl -L -o /tmp/llama-swap.tar.gz \
  "https://github.com/mostlygeek/llama-swap/releases/latest/download/llama-swap_linux_amd64.tar.gz"
tar -xzf /tmp/llama-swap.tar.gz -C ~/bin llama-swap
chmod +x ~/bin/llama-swap
~/bin/llama-swap --version
```

`L3MS_ROOT` must resolve to this repo's root at launch time. On the serving
host that's typically `/home/kchauhan/repos/l3ms`. Export it in your shell rc
or rely on the systemd unit (which sets it explicitly).

## Validate config

```bash
cd ~/repos/l3ms
L3MS_ROOT=$(pwd) ~/bin/llama-swap -config ./llama-swap.yaml -watch
```

`-watch` does a dry load and surfaces macro/env errors. Kill with Ctrl-C once
you see "listening on …".

## Run as a systemd user service

The unit uses `%h` (home) and three env vars so it runs unmodified on any
account. Defaults:

- `L3MS_ROOT=$HOME/repos/l3ms`
- `LLAMA_SWAP_BIN=$HOME/bin/llama-swap`
- `LLAMA_SWAP_LISTEN=:8080`

```bash
mkdir -p ~/.config/systemd/user
cp maintenance/systemd/llama-swap.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-swap.service
journalctl --user -fu llama-swap
```

Override any of the defaults with a drop-in (no fork of the unit file):

```bash
systemctl --user edit llama-swap.service
# [Service]
# Environment=L3MS_ROOT=/srv/l3ms
# Environment=LLAMA_SWAP_BIN=/usr/local/bin/llama-swap
# Environment=LLAMA_SWAP_LISTEN=:9090
```

Stop / restart:

```bash
systemctl --user stop    llama-swap.service
systemctl --user restart llama-swap.service
```

## Using it

List every model (including aliases):

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

Chat with the production default (warm after startup preload):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role":"user","content":"say hi"}]
  }'
```

Switch reasoning effort on gpt-oss without restarting (aliases are created
automatically from `setParamsByID` keys):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-120b:low", "messages":[...]}'
```

Hot-swap to another model (evicts the current one, loads the new one):

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next", "messages":[...]}'
```

Manually load / unload:

```bash
curl -X POST http://localhost:8080/models/load   -d '{"model":"qwen3-coder-next"}'
curl -X POST http://localhost:8080/models/unload -d '{"model":"qwen3-coder-next"}'
```

## Adding a new model

1. Append a new entry under `models:` in `llama-swap.yaml`. Minimum fields:
   `cmd:` (using `${PORT}` for the upstream port) and a unique key.
2. Reuse the existing macros (`${llama_server}`, `${chat_template}`,
   `${cpu_range}`) so per-host paths stay in one place.
3. Add `env:` if the model needs `LLAMA_SET_ROWS` / `GGML_CUDA_GRAPH_OPT`
   overrides.
4. Restart: `systemctl --user restart llama-swap.service`.
5. Verify: `curl -s http://localhost:8080/v1/models | jq '.data[].id'`.

## Ports

- `8080` — llama-swap listener (client-facing OpenAI endpoint)
- `10001+` — per-model upstream ports (auto-assigned via `startPort`)

If you previously had the L3MS TUI Chat tab pointed at `http://<host>:8001`,
update it to `http://<host>:8080/v1`.

## Troubleshooting

- **"env macro not set: L3MS_ROOT"** — export `L3MS_ROOT` or start via
  systemd (the unit sets it).
- **"model file not found"** — the per-model path in `cmd:` is absolute; a
  missing mount (`/mnt/lab/models`) is the usual cause.
- **ik_llama / puzzle / sarvam builds missing** — those models reference
  `${ik_server}`, `${puzzle_server}`, `${sarvam_server}`. Build them via
  `maintenance/build-ik-llama-cpp.sh`, `maintenance/build-gpt-oss-puzzle-llama-cpp.sh`,
  `maintenance/build-sarvam-llama-cpp.sh` respectively.
- **Model stays loaded forever** — `globalTTL: 600` unloads after 10 min
  idle. Pin with `ttl: 0` per-model; evict now via `/models/unload`.
- **Concurrent models** — the default is single-model swap (LRU). To allow
  concurrent small+vision models, add a `matrix:` block (see
  `configuration.md`).
