# /new-model-config

Complete model onboarding for a new llama.cpp model family in L3MS.
Follow the 8-step process from `docs/model-onboarding-playbook.md` in order.
Read `llama-swap.yaml` to understand existing patterns and macros before writing anything.

## Usage

```
/new-model-config <model-family> <hf-repo-id> <quant-pattern> [--vision] [--pr-build <pr-number>]
```

**Arguments:**
- `<model-family>` — short slug used in file names (e.g. `gemma-4-26b-a4b`, `qwen3-6-35b-a3b`)
- `<hf-repo-id>` — HuggingFace repo (e.g. `unsloth/gemma-4-26B-A4B-it-GGUF`)
- `<quant-pattern>` — allow-patterns glob (e.g. `*UD-Q5_K_XL*`)
- `--vision` — include mmproj profile in llama-swap.yaml
- `--pr-build <pr-number>` — model requires a PR-merge test build, not upstream

## Step-by-step workflow

### Step 1 — Confirm binary source
- If `--pr-build N` is given:
  - Check if `maintenance/build-<model-family>-llama-cpp.sh` exists
  - If not, create it using an existing `maintenance/build-*.sh` as template
  - The wrapper must set `PR_NUM=<N>` and delegate to `maintenance/llama-test-pr.sh`
- Otherwise, confirm `vendor/llama.cpp/build/bin/llama-server` exists

### Step 2 — Add downloader profile
Read `model_downloader/models_config.json`, then append a disabled-by-default entry:
```json
{
  "name": "<model-family>",
  "enabled": false,
  "repo_id": "<hf-repo-id>",
  "local_dir": "/mnt/lab/models/<hf-repo-id>",
  "allow_patterns": ["<quant-pattern>"],
  "force_download": false,
  "description": "<model-family>: <short description>. Check compatibility requirements."
}
```
- Keep `force_download: false`
- Use `max_workers: 2` if it is a multi-file model (sharded GGUF)

### Step 3 — Add llama-swap.yaml entry
Read current `llama-swap.yaml` and append under `models:`.

**Conventions:**
- Use the existing macros: `${llama_server}`, `${ik_server}`, `${puzzle_server}`, `${sarvam_server}`, `${chat_template}`, `${cpu_range}`
- If `--pr-build N`, use the matching server macro (e.g. `${puzzle_server}`) or create a new macro
- Always include `--port ${PORT} --host 0.0.0.0`
- Always include `taskset -c ${cpu_range}` unless it is a CPU-only model
- Start with `--fit on --fit-ctx <ctx> --fit-target 512` for a new model (safer than hardcoded `-ot`)
- Safe defaults: `-ctk q8_0 -ctv q8_0`, `--parallel 1`, `--flash-attn on`, `--no-mmap`, `--prio 2`, `--no-warmup`
- For reasoning models (gpt-oss, Mistral): add `filters.setParamsByID` block with `:high`/`:med`/`:low` aliases
- If `--vision`: add a second `<model-family>-vision` entry with `--mmproj`, `FIT_TARGET=2048`, `BATCH_SIZE=256`/`UBATCH_SIZE=512`, `GGML_CUDA_GRAPH_OPT=0`
- Add `env:` block with `LLAMA_SET_ROWS=1` and `GGML_CUDA_GRAPH_OPT=1` (or `0` if vision/unstable graph recapture)

**Template (MoE model, no PR build, no vision):**
```yaml
"<model-family>":
  name: "<Display name> (<quant>)"
  description: "New model — bench not yet run. Use --fit until optimized."
  env:
    - "LLAMA_SET_ROWS=1"
    - "GGML_CUDA_GRAPH_OPT=1"
  cmd: |
    taskset -c ${cpu_range} ${llama_server}
    -m /mnt/lab/models/<hf-repo-id>/<model-file>.gguf
    --alias "<hf-repo-id>"
    --fit on --fit-ctx 65536 --fit-target 512
    --no-mmap --mlock
    -ctk q8_0 -ctv q8_0
    --ctx-size 65536 --parallel 1
    --threads 10 --threads-batch 12
    --flash-attn on
    --batch-size 1024 --ubatch-size 512
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0
    --host 0.0.0.0 --port ${PORT}
    --jinja --prio 2 --no-warmup
```

### Step 4 — Create bench scripts
Create the three bench scripts in `bench-models/`:
- `bench-llama-cpp-<model-family>.sh` — copy closest existing script and update `MODEL=`, `N_CPU_MOE=`, `THREADS=`
- `bench-llama-cpp-<model-family>-strategies.sh` — strategy sweep (`all-cpu-moe`, `partial-cpu`, `up-down-cpu`, `up-cpu`)
- `bench-llama-cpp-<model-family>-fit.sh` — fit-params auto-placement

For PR-build models, override runner binaries:
```bash
LLAMA_BENCH="${LLAMA_BENCH:-${REPO_DIR}/vendor/llama.cpp-pr-test-<N>/build/bin/llama-bench}"
LLAMA_FIT="${LLAMA_FIT:-${REPO_DIR}/vendor/llama.cpp-pr-test-<N>/build/bin/llama-fit-params}"
```

Validate syntax: `bash -n bench-models/bench-llama-cpp-<model-family>.sh`

### Step 5 — Update docs
In `docs/bench-runbook.md` + `docs/bench-results.md`:
- Add model to the reference hardware table in §1 (with placeholder values: `TBD`)
- Add a bench results section in `docs/bench-results.md` with architecture notes

### Step 6 — Update CHANGELOG.md
Under `## [Unreleased]` → `### Added`:
```markdown
- `<model-family>`: llama-swap.yaml entry, bench scripts, downloader config
- `docs/bench-results.md`: stub for <model-family> results
```

### Step 7 — Validate wiring
```bash
# Syntax check bench scripts
bash -n bench-models/bench-llama-cpp-<model-family>.sh
bash -n bench-models/bench-llama-cpp-<model-family>-strategies.sh
bash -n bench-models/bench-llama-cpp-<model-family>-fit.sh

# Validate llama-swap config
cd ~/repos/l3ms
L3MS_ROOT=$(pwd) ~/bin/llama-swap -config ./llama-swap.yaml -watch-config &
sleep 3
curl -s http://localhost:8080/v1/models | jq '.data[] | select(.id | contains("<model-family>"))'
kill %1
```

### Step 8 — Download model (print command, do not execute)
Print the targeted download command for the user to run manually:
```bash
./model_downloader/download_hf_model.py \
  --repo-id <hf-repo-id> \
  --allow-patterns '<quant-pattern>' \
  --local-dir /mnt/lab/models/<hf-repo-id> \
  --max-workers 2
```

## After onboarding: next steps

1. Run `/preflight` to verify system state
2. Run `/bench-model <model-family>` to get baseline numbers
3. Run `/optimize-model <model-family>` once bench data is collected
4. Restart llama-swap: `systemctl --user restart llama-swap.service`
