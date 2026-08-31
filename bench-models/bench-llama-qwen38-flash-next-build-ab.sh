#!/usr/bin/env bash
# Qwen3.8-Flash-Next — binary A/B: production build vs PR-stack build
#
# Question: do the stacked PRs (#28023 indexer heads, #28068 GDN l2norm,
# #27941 follow-up fixes) on current master beat the production
# pr-test-27742 branch build on the SAME AtomicChat AD-4.27bpw quant?
#
# Arms (override with ARMS="prod stack"):
#   prod  : vendor/llama.cpp-pr-test-27742 (production binary, fit64 flags)
#   stack : vendor/llama.cpp-pr-test-28023-28068-27941 (master + PR stack)
#
# Each arm: fresh load, 1 warmup probe, 2 measured pp probes (unique prefix
# forces full re-prefill), 2 tg probes (64 tok), plus VRAM/RAM and a
# short-answer sanity dump for a quick eyeball quality check.
# Requires llama-swap stopped (full-VRAM experiment):
#   systemctl --user stop llama-swap.service
# set -euo pipefail

REPO="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$HOME/models/qwen38-flash-next/AD-4.27bpw-Q4_K_M-M64/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
PORT=8022
THP="$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
FILLER="$(python3 -c "print('The quick brown fox jumps over the lazy dog while engineers profile memory bandwidth on hybrid MoE inference. ' * 30)")"

declare -A BINS=(
  [prod]="$REPO/vendor/llama.cpp-master/build/bin/llama-server"
  [stack]="$REPO/vendor/llama.cpp-pr-test-28023-28068-27941/build/bin/llama-server-exp"
)
# both master and the exp stack are post-rename: --tensor-read-lazy -> --lazy-mode
declare -A LAZY=(
  [prod]="--lazy-mode on"
  [stack]="--lazy-mode on"
)

probe_pp() {
  # unique prefix per call forces a full re-prefill (otherwise KV prefix reuse
  # makes prompt_n collapse and the pp number meaningless)
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"[$(date +%s%N)] Summarize in one sentence: $FILLER\"}],\"max_tokens\":8,\"temperature\":0}" \
  | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); n=t.get('prompt_n',0); print(f'{n/max(t.get(\"prompt_ms\",1),1)*1000:.0f} t/s over n={n}', end=' ')"
}

probe_tg() {
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"[$(date +%s%N)] Count from 1 to 40, one number per line.\"}],\"max_tokens\":64,\"temperature\":0}" \
  | python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); n=t.get('predicted_n',0); print(f'{n/max(t.get(\"predicted_ms\",1),1)*1000:.1f} t/s over n={n}', end=' ')"
}

probe_quality() {
  # eyeball check: reasoning truncation, formatting, factual coherence
  curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Name the four inner planets of the solar system in order, one per line, no commentary."}],"max_tokens":96,"temperature":0}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip()[:220].replace(chr(10),' | '))"
}

run_arm() { # run_arm <label>
  local label="$1" BIN="${BINS[$1]}"
  local LOG="/tmp/opencode/build-ab-$label.log"
  [ -x "$BIN" ] || { echo "== arm: $label — binary missing, skipped ($BIN)"; return 1; }
  echo "== arm: $label ($(git -C "$(dirname "$(dirname "$(dirname "$BIN")")")" log --oneline -1 2>/dev/null | cut -c1-40))  [THP: $THP]"
  nohup taskset -c 0-11 "$BIN" -m "$MODEL" --alias "build-ab-$label" \
    --fit on --fit-target 512 \
    -c 65536 --parallel 1 -b 4096 -ub 1024 \
    -fa on --jinja -ctk q8_0 -ctv q8_0 \
    -t 10 --threads-batch 12 --prio 2 \
    ${LAZY[$label]} \
    --spec-type ngram-mod \
    --spec-ngram-mod-n-match 60 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 24 \
    --no-warmup \
    --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
  local SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  local ok=""
  for i in $(seq 1 300); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    grep -qiE 'out of memory|cudaMalloc failed|std::bad_alloc|failed to allocate|error: unknown argument' "$LOG" && { echo "   LOAD ERROR"; break; }
    sleep 2
  done
  [ -n "$ok" ] || { echo "   server failed:"; tail -3 "$LOG"; kill $SRV 2>/dev/null || true; trap - EXIT; return 1; }
  echo -n "   warmup: " ; probe_tg; echo
  echo -n "   pp:     " ; probe_pp; probe_pp; echo
  echo -n "   tg:     " ; probe_tg; probe_tg; echo
  echo "   sanity: $(probe_quality)"
  echo "   VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  echo "   RAM avail: $(free -g | awk '/^Mem:/{print $7}') GiB"
  kill $SRV 2>/dev/null || true
  wait $SRV 2>/dev/null || true
  trap - EXIT
  sleep 3
}

for arm in ${ARMS:-prod stack}; do
  run_arm "$arm" || true
done
echo "done."
