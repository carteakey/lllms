# /add-bench-result

Record benchmark results into `docs/bench-results.md` and update CHANGELOG.md.
Results are now **also** stored as structured JSONL in `bench-models/results/<model-key>.jsonl`.
The JSONL file is written automatically by the runner scripts — this skill handles the human-readable runbook update.

## Usage

```
/add-bench-result <model-key> [--log <log-file>] [--from-jsonl]
```

**Arguments:**
- `<model-key>` — model slug matching the bench script suffix (e.g. `qwen3-6-35b-a3b`)
- `--log <log-file>` — specific log file from `bench-models/logs/` to parse (default: most recent)
- `--from-jsonl` — read from `bench-models/results/<model-key>.jsonl` instead of raw logs

## JSONL result file

Every bench run now **automatically** writes a JSONL record to `bench-models/results/<model-key>.jsonl`.
These files are git-tracked and contain all the structured data you need for the runbook table.

### Record schema

```json
{
  "ts":             "2026-04-19T20:00:00Z",
  "model_key":      "qwen3-6-35b-a3b",
  "model":          "/mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF/...",
  "backend":        "llama.cpp",
  "strategy":       "fit",
  "ngl":            41,
  "n_cpu_moe":      null,
  "override_tensor": "blk.13 down + blk.14-40 ...",
  "fit_ctx":        65536,
  "fit_target":     512,
  "ctx":            "512,128",
  "ctk":            "q8_0",
  "ctv":            "q8_0",
  "threads":        10,
  "repetitions":    1,
  "pp_tokens":      512,
  "pp_ts":          970.77,
  "pp_std":         3.1,
  "tg_tokens":      128,
  "tg_ts":          52.33,
  "tg_std":         0.2,
  "git_sha":        "d9f6201",
  "llama_version":  "b5000",
  "log_file":       "bench-models/logs/2026-04-19_..._fit.log",
  "notes":          ""
}
```

### Manually log a result from an existing log file

```bash
MODEL_KEY=qwen3-6-35b-a3b \
STRATEGY=fit \
LOG_FILE=bench-models/logs/2026-04-19_..._fit.log \
  bench-models/log-result.sh
```

### Query JSONL results

```bash
# All results for a model, sorted by tg_ts descending
cat bench-models/results/qwen3-6-35b-a3b.jsonl | \
  python3 -c "import sys,json; rows=list(map(json.loads,sys.stdin)); \
  rows.sort(key=lambda r: r.get('tg_ts',0), reverse=True); \
  [print(f'{r[\"ts\"][:10]}  strategy={r.get(\"strategy\",\"?\"):20}  pp={r.get(\"pp_ts\",\"?\")}  tg={r.get(\"tg_ts\",\"?\")}') for r in rows]"

# Best tg across ALL models
cat bench-models/results/*.jsonl 2>/dev/null | \
  python3 -c "import sys,json; rows=list(map(json.loads,sys.stdin)); \
  rows.sort(key=lambda r: r.get('tg_ts',0), reverse=True); \
  [print(f'{r[\"model_key\"]:30}  strategy={r.get(\"strategy\",\"?\"):20}  tg={r.get(\"tg_ts\",\"?\")}') for r in rows[:20]]"

# Show pp and tg for a specific strategy
grep '"strategy": "fit"' bench-models/results/qwen3-6-35b-a3b.jsonl | python3 -c \
  "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"
```

## Workflow

### Step 1 — Check the JSONL results file

```bash
# See all recorded runs for a model
cat bench-models/results/<model-key>.jsonl | python3 -m json.tool --no-ensure-ascii | head -80

# Or quick summary
cat bench-models/results/<model-key>.jsonl | \
  python3 -c "import sys,json; [print(f'{r[\"ts\"][:10]}  {r.get(\"strategy\",\"?\"):20}  pp={r.get(\"pp_ts\",\"?\")}  tg={r.get(\"tg_ts\",\"?\")}') for r in map(json.loads, sys.stdin)]"
```

### Step 2 — Find the best results

From the JSONL, identify the winning strategy per metric:
- Highest `tg_ts` = best token generation config
- Highest `pp_ts` = best prompt processing config

Note the associated `ngl`, `override_tensor`, `ctk`/`ctv`, `fit_ctx`, `fit_target` for the runbook.

### Step 3 — Read existing results section for the model

Read `docs/bench-results.md` to find any existing section. If none exists, create it.

### Step 4 — Update bench-results.md

**Format for each model's results:**

```markdown
### <Display Name> <Quant> — <GPU> / <CPU> / <RAM>

**Architecture:** <N> blocks, <N> experts per MoE layer, <N> active/token, <size> on disk.
<Key architecture notes: shared experts? DeltaNet? Non-standard naming?>

**Results (<pp_tokens>pp + <tg_tokens>tg, <threads> threads, FA=1, no-mmap, <ctk> KV):**

| Strategy | ngl | flags | pp (t/s) | tg (t/s) | Notes |
|----------|-----|-------|----------|----------|-------|
| baseline / `all-cpu-moe` | 99 | ... | <pp> | <tg> | safe baseline |
| **fit-params auto** | **<ngl>** | `...` | **<pp>** | **<tg>** | **winner** |

**Winner:** <which strategy> — <brief reason>

**Recommended serving configuration (`llama-swap.yaml` model: `<model-key>`):**

| Setting | Value | Reason |
|---------|-------|--------|
| `-ngl` | <value> | fit-params derived for <ctx> ctx + q8_0 + <target> MiB margin |
| `--override-tensor` | `<ot-pattern>` | Static placement |
| `-ctk / -ctv` | `q8_0` | KV at <ctx> ≈ <size> GB |
| `--ctx-size` | <ctx> | <reason> |
| `GGML_CUDA_GRAPH_OPT` | <0 or 1> | <reason> |
| `--parallel` | 1 | Single-user homelab |

**To reproduce:**
```sh
./bench-models/bench-llama-cpp-<model-key>.sh
./bench-models/bench-llama-cpp-<model-key>-fit.sh
```
```

### Step 5 — Update CHANGELOG.md

Under `## [Unreleased]` → `### Added` or `### Changed`:
```markdown
- `bench-models/results/<model-key>.jsonl`: bench results record (pp=<pp>, tg=<tg> t/s)
- `docs/bench-results.md`: results section for <model-key>
```

### Gotcha checklist to document

- [ ] Shared experts: does this model have `_shexp` tensors? (use `(ch|)exps` pattern)
- [ ] Non-standard expert naming: `gate_up` vs `gate` (Qwen3.6-35B-A3B needs `gate_up` included)
- [ ] N_CPU_MOE instability: was the integer `-ncmoe` path unstable?
- [ ] Vision OOM profile: what `FIT_TARGET` was needed for stable vision inference?
- [ ] GGML_CUDA_GRAPH_OPT: should it be 0 or 1?
