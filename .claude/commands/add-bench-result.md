# /add-bench-result

Record benchmark results into `docs/bench-runbook.md` §8 and update CHANGELOG.md.
Read existing results tables before writing to maintain consistent formatting.

## Usage

```
/add-bench-result <model-key> [--log <log-file>]
```

**Arguments:**
- `<model-key>` — model slug matching the bench script suffix (e.g. `qwen3-6-35b-a3b`)
- `--log <log-file>` — specific log file from `bench-models/logs/` to parse (default: most recent)

## Workflow

### Step 1 — Find the bench log

```bash
ls -lt bench-models/logs/ | grep <model-key> | head -10
```

Read the most recent log file(s) for the model. They are named:
```
YYYY-MM-DD_HH-MM-SS_<model-key>.log
YYYY-MM-DD_HH-MM-SS_<model-key>_fit.log
YYYY-MM-DD_HH-MM-SS_<model-key>_strategies.log
```

### Step 2 — Extract the key numbers

From the llama-bench Markdown output table, extract:
- `pp N` — prompt processing tokens/s (higher is better)
- `tg N` — token generation tokens/s (higher is better)
- The configuration used (`-ngl`, `N_CPU_MOE`, `STRATEGY`, `CACHE_TYPE_K`, etc.)

Example bench output format:
```
| model | size | params | backend | ngl | threads | n_kv | fmoe | test | t/s |
|-------|------|--------|---------|-----|---------|------|------|------|-----|
| ...   | ...  | ...    | CUDA    | 49  | 10      | ...  | 0    | pp 512 | 502.34 ± 3.1 |
| ...   | ...  | ...    | CUDA    | 49  | 10      | ...  | 0    | tg 128 | 39.62 ± 0.2  |
```

### Step 3 — Read existing §8 section for the model

Read `docs/bench-runbook.md` to find the existing section. If none exists, create it.

### Step 4 — Update bench-runbook.md §8

**Format for each model's results:**

```markdown
### <Display Name> <Quant> — <GPU> / <CPU> / <RAM>

**Architecture:** <N> blocks, <N> experts per MoE layer, <N> active/token, <size> on disk.
<Key architecture notes: shared experts? DeltaNet? No shared expert tensors?>

**Results (<pp_ctx>pp + <tg_ctx>tg, <threads> threads, FA=<fa>, no-mmap, q8_0 KV, <reps> repetitions):**

| Strategy | ngl | flags | pp (t/s) | tg (t/s) | Notes |
|----------|-----|-------|----------|----------|-------|
| baseline / `all-cpu-moe` | 99 | ... | <pp> | <tg> | safe baseline |
| **fit-params auto** | **<ngl>** | `blk.<N> + blk.<N+1>-<N_last>` | **<pp>** | **<tg>** | **winner** |

**Winner:** <which strategy> — <brief reason>

**Recommended serving configuration (`llama-swap.yaml` model: `<model-key>`):**

| Setting | Value | Reason |
|---------|-------|--------|
| `-ngl` | <value> | fit-params derived for <ctx> ctx + q8_0 + <target> MiB margin |
| `--override-tensor` | `<ot-pattern>` | Static; <MiB> MiB margin |
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
- `docs/bench-runbook.md`: §8 results for <model-key> — best pp=<pp>, tg=<tg> t/s
```

### Guidelines for recording results

- **Bold the winner row** in the results table
- Include the bench context (`512pp + 128tg` unless different) in the section header
- Note KV cache type used during bench (f16 or q8_0) — it matters for VRAM headroom
- For MoE models, note whether `N_CPU_MOE` or `-ot` regex path was used
- Clearly separate bench tg (short context) from server tg (real serving context) with a note like:
  > These numbers use bench-only 512-token context. Server tg at 64k ctx with q8_0 KV will differ.
- If there is a model-specific gotcha (shared experts, `gate_up` naming, unstable integer path), document it prominently

### Gotcha checklist to document

- [ ] Shared experts: does this model have `_shexp` tensors? (Qwen3.5-122B does; use `(ch|)exps` pattern)
- [ ] Non-standard expert naming: `gate_up` vs `gate` (Qwen3.6-35B-A3B needs `gate_up` included)
- [ ] N_CPU_MOE instability: was the integer `-ncmoe` path unstable? (use `-ot` or `--fit` instead)
- [ ] Vision OOM profile: what `FIT_TARGET` was needed for stable vision inference?
- [ ] GGML_CUDA_GRAPH_OPT: should it be 0 or 1? (0 if context depth varies)
