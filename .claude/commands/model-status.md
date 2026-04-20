# /model-status

Show the current state of all configured models: serving status, bench state, and optimization level.
Useful for a quick overview of what's running, what's been benched, and what needs attention.

## Usage

```
/model-status [<model-key>]
```

**Arguments:**
- `<model-key>` — optional; if provided, show detailed status for one model only

## Workflow

### Step 1 — Check llama-swap service

```bash
systemctl --user status llama-swap.service --no-pager
curl -s http://localhost:8080/v1/models | jq '[.data[].id]'
```

### Step 2 — Parse llama-swap.yaml for model inventory

Read `llama-swap.yaml` and for each non-`unlisted` model entry, collect:
- Model key
- `name` field
- Whether `--fit` is still in use (not yet optimized) vs hardcoded `-ngl` + `--override-tensor`
- Whether bench scripts exist: `bench-models/bench-llama-cpp-<key>.sh`
- Whether a JSONL results file exists: `bench-models/results/<key>.jsonl`

### Step 3 — Output status table

Check JSONL files for the latest bench result per model:
```bash
# Best tg across all models from JSONL records
cat bench-models/results/*.jsonl 2>/dev/null | \
  python3 -c "
import sys, json
rows = list(map(json.loads, sys.stdin))
rows.sort(key=lambda r: r.get('tg_ts', 0), reverse=True)
seen = set()
for r in rows:
    k = r['model_key']
    if k not in seen:
        seen.add(k)
        print(f'{k:35}  strategy={r.get(\"strategy\",\"?\"):15}  pp={r.get(\"pp_ts\",\"?\")}  tg={r.get(\"tg_ts\",\"?\")}  [{r[\"ts\"][:10]}]')
"
```

Format as a markdown table:

| Model Key | Type | Serving | Fit? | JSONL Results | Best tg (t/s) | Notes |
|-----------|------|---------|------|---------------|---------------|-------|
| `gpt-oss-120b` | MoE | ✅ live | ❌ static -ot | ✅ exists | 28.0 | production |
| `qwen3-coder-next` | MoE | ✅ live | ❌ static -ot | ✅ exists | 39.6 | production |
| `gemma-4-26b-a4b` | MoE | ✅ live | ✅ --fit | ✅ exists | 47.9 | needs optimize |
| `new-model` | ? | ❌ not loaded | ✅ --fit | ❌ missing | — | onboarding |


**Fit? column:**
- `✅ --fit` = still using `--fit on` auto-placement (less deterministic, startup overhead)
- `❌ static -ot` = using bench-derived `-ngl` + `--override-tensor` (production)

### Step 4 — Identify action items

Print a prioritized action list:
1. Models using `--fit` that have bench data → suggest `/optimize-model <key>`
2. Models missing bench scripts → suggest `/new-model-config <key>` or `/bench-model <key>`
3. Models with bench scripts but no logged results → suggest `/bench-model <key>`
4. Models listed as `unlisted: true` → note as legacy/regression variants

### Model optimization levels

| Level | Criteria | Next action |
|-------|----------|-------------|
| 🔴 Unbenched | No bench scripts, no results | `/new-model-config` or `/bench-model` |
| 🟡 Benched, not optimized | Has bench results, still using `--fit` | `/optimize-model` |
| 🟢 Optimized | Static `-ngl` + `-ot` from bench data | `/bench-model` to verify with new llama.cpp builds |
| ⚫ Legacy | `unlisted: true` | Keep for regression; remove if unneeded |

## Detailed model status (single model)

When a `<model-key>` is provided, show:

```
Model: <model-key>
Name: <name from llama-swap.yaml>
Status: loaded | not loaded | service down
Placement: --fit on | static -ngl=<N> -ot=<pattern>
Bench scripts: bench-llama-cpp-<key>.sh | strategies | fit
Bench logs: <most recent log timestamp or "none">
Bench results in runbook: §8 section exists | not yet documented
VRAM env: GGML_CUDA_GRAPH_OPT=<0|1>
Context: <ctx_size>
KV cache: <ctk>/<ctv>
Vision: yes (mmproj) | no
```

## Quick health check commands

```bash
# Is llama-swap up?
systemctl --user is-active llama-swap.service

# What models are available?
curl -s http://localhost:8080/v1/models | jq '.data[].id'

# What models have been benched recently?
ls -lt bench-models/logs/ | head -20

# What models are in llama-swap.yaml (non-unlisted)?
grep -E '^\s+"[^"]+":' llama-swap.yaml | grep -v unlisted
```
