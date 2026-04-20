#!/usr/bin/env bash
# log-result.sh — parse a llama-bench log and append a JSONL record.
#
# Called automatically by run-llama-bench.sh, run-ik-llama-bench.sh, and
# run-llama-fit-bench.sh after each run. Can also be called manually:
#
#   MODEL_KEY=qwen3-coder-next LOG_FILE=bench-models/logs/2026-04-19_….log \
#     bench-models/log-result.sh
#
# Required:
#   LOG_FILE   - path to the llama-bench log file to parse
#   MODEL_KEY  - short slug for the results file (e.g. qwen3-coder-next)
#                Falls back to the model filename stem if unset.
#
# Context env vars (captured from the calling runner — all optional):
#   MODEL, BACKEND, STRATEGY, N_GPU_LAYERS, N_CPU_MOE, OVERRIDE_TENSOR,
#   FIT_CTX, FIT_TARGET, CACHE_TYPE_K, CACHE_TYPE_V, THREADS, REPETITIONS,
#   TASKS, NOTES
#
# Output:
#   bench-models/results/<MODEL_KEY>.jsonl  — one JSON object per run appended

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

LOG_FILE="${LOG_FILE:-}"
if [ -z "${LOG_FILE}" ] || [ ! -f "${LOG_FILE}" ]; then
  echo "log-result: LOG_FILE not set or not found (${LOG_FILE:-<unset>})" >&2
  exit 1
fi

# Determine model key: explicit > MODEL env basename > log file basename
_model_stem="$(basename "${LOG_FILE}" .log | sed 's/^[0-9_-]*_//')"
MODEL_KEY="${MODEL_KEY:-${_model_stem}}"
RESULTS_FILE="${RESULTS_DIR}/${MODEL_KEY}.jsonl"

# ---------------------------------------------------------------------------
# Parse llama-bench markdown table output.
# The table rows look like:
#   | ... | CUDA | 49 | 10 | ... | pp 512 | 502.34 ± 3.10 |
#   | ... | CUDA | 49 | 10 | ... | tg 128 |  39.62 ± 0.20 |
# We extract pp_ts, pp_std, tg_ts, tg_std from the last pp/tg pair found.
# ---------------------------------------------------------------------------

_parsed=$(python3 - "${LOG_FILE}" <<'PYEOF'
import re, sys, json

log = open(sys.argv[1]).read()

# Match markdown table data rows (skip header/separator)
row_re = re.compile(r'^\|[^|]+\|[^|]+\|[^|]+\|[^|]+\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(pp|tg)\s+(\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|', re.MULTILINE)

results = {"pp_tokens": None, "pp_ts": None, "pp_std": None,
           "tg_tokens": None, "tg_ts": None, "tg_std": None}

for m in row_re.finditer(log):
    kind, tokens, ts, std = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
    results[f"{kind}_tokens"] = tokens
    results[f"{kind}_ts"]     = ts
    results[f"{kind}_std"]    = std

# Also try the simpler jsonl output format (-o jsonl) if md parse found nothing
if results["pp_ts"] is None:
    jl_re = re.compile(r'^\{.*"test".*\}', re.MULTILINE)
    for line in jl_re.findall(log):
        try:
            obj = json.loads(line)
            test = obj.get("test", "")
            ts   = obj.get("avg_ts", obj.get("t_avg", None))
            std  = obj.get("std_ts", obj.get("t_std", None))
            if test.startswith("pp") and ts:
                results["pp_tokens"] = int(test.split()[1]) if len(test.split()) > 1 else None
                results["pp_ts"]  = ts
                results["pp_std"] = std
            elif test.startswith("tg") and ts:
                results["tg_tokens"] = int(test.split()[1]) if len(test.split()) > 1 else None
                results["tg_ts"]  = ts
                results["tg_std"] = std
        except Exception:
            pass

print(json.dumps(results))
PYEOF
)

# ---------------------------------------------------------------------------
# Build the JSONL record
# ---------------------------------------------------------------------------

_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
_git_sha="$(git -C "${SCRIPT_DIR}/.." rev-parse --short HEAD 2>/dev/null || echo '')"
_llama_bin="${LLAMA_BENCH:-${SCRIPT_DIR}/../vendor/llama.cpp/build/bin/llama-bench}"
_llama_ver="$("${_llama_bin}" --version 2>/dev/null | head -1 || echo '')"

python3 - <<PYEOF
import json, os, sys

parsed = json.loads("""${_parsed}""")

record = {
    "ts":            "${_ts}",
    "model_key":     "${MODEL_KEY}",
    "model":         "${MODEL:-}",
    "backend":       "${BACKEND:-llama.cpp}",
    "strategy":      "${STRATEGY:-}",
    "ngl":           int("${N_GPU_LAYERS:-99}") if "${N_GPU_LAYERS:-}" else None,
    "n_cpu_moe":     int("${N_CPU_MOE:-}") if "${N_CPU_MOE:-}" else None,
    "override_tensor": "${OVERRIDE_TENSOR:-}",
    "fit_ctx":       int("${FIT_CTX:-}") if "${FIT_CTX:-}" else None,
    "fit_target":    int("${FIT_TARGET:-}") if "${FIT_TARGET:-}" else None,
    "ctx":           "${TASKS:-${N_PROMPT:-512},${N_GEN:-128}}",
    "ctk":           "${CACHE_TYPE_K:-f16}",
    "ctv":           "${CACHE_TYPE_V:-f16}",
    "threads":       int("${THREADS:-6}") if "${THREADS:-}" else None,
    "repetitions":   int("${REPETITIONS:-}") if "${REPETITIONS:-}" else None,
    "pp_tokens":     parsed.get("pp_tokens"),
    "pp_ts":         parsed.get("pp_ts"),
    "pp_std":        parsed.get("pp_std"),
    "tg_tokens":     parsed.get("tg_tokens"),
    "tg_ts":         parsed.get("tg_ts"),
    "tg_std":        parsed.get("tg_std"),
    "git_sha":       "${_git_sha}",
    "llama_version": "${_llama_ver}",
    "log_file":      "${LOG_FILE}",
    "notes":         "${NOTES:-}",
}

# Drop None values to keep records compact
record = {k: v for k, v in record.items() if v not in (None, "")}

results_file = "${RESULTS_FILE}"
with open(results_file, "a") as f:
    f.write(json.dumps(record) + "\n")

pp = record.get("pp_ts"); tg = record.get("tg_ts")
print(f"log-result: wrote to {results_file}")
if pp or tg:
    print(f"  pp={pp} ± {record.get('pp_std','?')} t/s   tg={tg} ± {record.get('tg_std','?')} t/s")
else:
    print("  warning: no pp/tg values parsed from log — check log format")
PYEOF
