# /llama-swap

Manage the llama-swap service: restart, validate config, hot-swap models, check status.
See `docs/llama-swap-runbook.md` for full reference.

## Usage

```
/llama-swap <action> [<model-key>]
```

**Actions:**
- `status` — show service status + currently loaded models
- `restart` — restart the service and verify
- `validate` — dry-load config without starting the server
- `list` — list all available model IDs (including aliases)
- `load <model-key>` — manually preload a model
- `unload <model-key>` — manually unload a model
- `swap <model-key>` — hot-swap to a different model via a chat request
- `logs` — tail the service logs

## Service management

### Status
```bash
systemctl --user status llama-swap.service
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

### Restart
```bash
systemctl --user restart llama-swap.service
journalctl --user -fu llama-swap   # follow logs
```

### Validate config (dry run — no server started)
```bash
cd ~/repos/l3ms
L3MS_ROOT=$(pwd) ~/bin/llama-swap -config ./llama-swap.yaml -watch-config &
sleep 3
curl -s http://localhost:8080/v1/models | jq '.data[].id'
kill %1
```
Kill with Ctrl-C once you see "listening on …"

### Tail logs
```bash
journalctl --user -fu llama-swap
```

## Model operations

### List all models (including aliases)
```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

### Hot-swap to a model (sends an inference request which triggers the swap)
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-key>","messages":[{"role":"user","content":"say hi"}]}'
```
Note: llama-swap evicts the current model and loads the new one before responding.

### Manually preload a model
```bash
curl -X POST http://localhost:8080/models/load -d '{"model":"<model-key>"}'
```

### Manually unload a model (frees VRAM/RAM immediately)
```bash
curl -X POST http://localhost:8080/models/unload -d '{"model":"<model-key>"}'
```

### Switch reasoning effort (for models with setParamsByID aliases)
```bash
# High reasoning effort (default for gpt-oss-120b)
curl -s http://localhost:8080/v1/chat/completions \
  -d '{"model":"gpt-oss-120b:high", "messages":[...]}'

# Medium effort
curl -s http://localhost:8080/v1/chat/completions \
  -d '{"model":"gpt-oss-120b:med", "messages":[...]}'

# Low effort (faster, less thinking)
curl -s http://localhost:8080/v1/chat/completions \
  -d '{"model":"gpt-oss-120b:low", "messages":[...]}'
```

## Port reference

| Port | Purpose |
|------|---------|
| 8080 | llama-swap listener (client-facing OpenAI-compatible endpoint) |
| 10001+ | per-model upstream ports (auto-assigned by llama-swap) |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `env macro not set: L3MS_ROOT` | L3MS_ROOT not exported | Export it or start via systemd (unit sets it) |
| `model file not found` | `/mnt/lab/models` not mounted | Check NFS/mount: `ls /mnt/lab/models` |
| `ik_llama / puzzle / sarvam build missing` | Missing binary | Build via `maintenance/build-*.sh` |
| Model stays loaded forever | TTL not firing | `globalTTL: 600` should unload after 10 min idle; check `ttl: 0` override per-model |
| OOM during model swap | Too many concurrent models | Default is single-model LRU swap; use `/models/unload` before loading another |

## Adding a new model (quick version)

For the full flow, use `/new-model-config`. For a quick YAML-only add:
1. Append entry under `models:` in `llama-swap.yaml`
2. Reuse macros: `${llama_server}`, `${chat_template}`, `${cpu_range}`
3. Include `--port ${PORT} --host 0.0.0.0` (required)
4. `systemctl --user restart llama-swap.service`
5. `curl -s http://localhost:8080/v1/models | jq '.data[].id'`

## Drop-in env overrides (no unit file fork needed)
```bash
systemctl --user edit llama-swap.service
# Add:
# [Service]
# Environment=L3MS_ROOT=/srv/l3ms
# Environment=LLAMA_SWAP_BIN=/usr/local/bin/llama-swap
# Environment=LLAMA_SWAP_LISTEN=:9090
```
