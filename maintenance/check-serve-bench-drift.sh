#!/usr/bin/env bash
# Compare the tracked serving flags with benchmark entry points.
#
# This is intentionally a read-only check. It reports drift and never rewrites
# llama-swap.yaml or bench scripts.
set -euo pipefail

if [[ $# -gt 0 ]]; then
  ROOT="$1"
else
  ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
fi
SERVE="$ROOT/llama-swap.yaml"
BENCH_DIR="$ROOT/bench-models"

if [[ ! -f "$SERVE" || ! -d "$BENCH_DIR" ]]; then
  echo "serve/bench audit: expected llama-swap.yaml and bench-models/ under $ROOT" >&2
  exit 2
fi

serve_flags="$(
  grep -Eo -- '(^|[[:space:]])(--override-tensor|-ot|-ngl)([[:space:]]|$)' "$SERVE" \
    | sed -E 's/^[[:space:]]*//; s/[[:space:]]+$//' \
    | sed 's/^-ot$/--override-tensor/' \
    | sort -u
)"
bench_flags="$(
  grep -RhoE -- '(^|[[:space:]])(--override-tensor|-ot|-ngl)([[:space:]]|$)' \
    "$BENCH_DIR"/bench-*.sh 2>/dev/null \
    | sed -E 's/^[[:space:]]*//; s/[[:space:]]+$//' \
    | sed 's/^-ot$/--override-tensor/' \
    | sort -u || true
)"

missing="$(comm -23 <(printf '%s\n' "$serve_flags") <(printf '%s\n' "$bench_flags") || true)"
extra="$(comm -13 <(printf '%s\n' "$serve_flags") <(printf '%s\n' "$bench_flags") || true)"

if [[ -n "$missing" || -n "$extra" ]]; then
  [[ -n "$missing" ]] && printf 'missing in bench scripts: %s\n' "$missing"
  [[ -n "$extra" ]] && printf 'only in bench scripts: %s\n' "$extra"
  exit 1
fi

printf 'serve/bench flags aligned: %s\n' "$(printf '%s' "$serve_flags" | tr '\n' ', ')"
