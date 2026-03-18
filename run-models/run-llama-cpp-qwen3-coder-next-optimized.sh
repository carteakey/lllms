#!/usr/bin/env bash
# run-llama-cpp-qwen3-coder-next-optimized.sh
# --------------------------------------------
# Optimized run script for Qwen3-Coder-Next UD-Q4_K_XL based on bench results.
#
# Model: Qwen3-Coder-Next 80B.A3B, ~47 GB on disk, 512 experts, 10 active/tok
# Hardware target: RTX 4070 12 GB + 64–96 GB DDR5 RAM
#
# Key differences from run-llama-cpp-qwen3-coder-next.sh:
#
#   --fit removed         Replaced with explicit -ngl 49 + --override-tensor.
#                         fit adds startup latency and can vary placement run-to-run.
#                         Static -ot is deterministic and starts faster.
#
#   -ngl 49               Bench-confirmed optimal via llama-fit-params (FIT_CTX=65536,
#                         CACHE_TYPE_K=q8_0, FIT_TARGET=512). Keeps blk 0-6 fully on
#                         GPU + blk 7 attn+up on GPU (only ffn_down spills). Blk 8-48
#                         experts go to CPU. Measured: pp=502 t/s, tg=39.62 t/s.
#                         (FIT_TARGET=128 gives pp=511 but risks mid-session CUDA OOM;
#                          see §9 of bench-runbook.md for explanation.)
#
#   --override-tensor     fit-params derived pattern (FIT_CTX=65536 + q8_0 + 512 MiB).
#                         Spills blk 7 ffn_down + all blk 8-48 expert tensors
#                         (up/down/gate + shared via (ch|)) to CPU. Keeps blk 0-6
#                         expert tensors and all attention/norms/embedding on GPU.
#
#   -ctk q8_0 / -ctv q8_0
#                         KV cache quantization. q8_0 KV at 64k ctx ≈ 2 GB vs f16 ≈ 4 GB.
#                         The saved ~2 GB allowed fit to push further than 64k+f16.
#                         Fit progression at 64k (FIT_TARGET=512):
#                           f16 KV:   pp=497, tg=39.60  (blk 7-gate+blk 8-48 on CPU)
#                           q8_0 KV:  pp=502, tg=39.62  (blk 7-down+blk 8-48 on CPU)
#                         q8_0 is lossless for most purposes and saves significant VRAM.
#
#   --parallel 1          Single inference slot. Default multi-slot multiplies KV
#                         cache VRAM by n_parallel — on a 12 GB card with 131k ctx
#                         this would OOM immediately. Single user homelab = 1 slot.
#
#   --ctx-size 65536      64k context. Reduced from 131k to free VRAM for two additional
#                         GPU layers (blk 6-7 expert tensors now fully on GPU).
#                         q8_0 KV at 64k ≈ 2 GB — much smaller than 131k (≈ 4 GB q8_0
#                         or ≈ 8 GB f16). The 2 GB savings vs 64k+f16 unlocked blk 8.
#                         64k is ample for most coding sessions and agent loops.
#                         For 131k context, switch to --fit on --fit-ctx 131072 (reverts
#                         to blk 6-48 CPU placement, pp≈476, tg≈38).
#
#   --no-mmap / --mlock   Required for hybrid CPU+GPU inference. mmap causes
#                         non-deterministic VRAM usage and expert tensor thrashing.
#                         mlock keeps expert weights pinned in RAM for stable latency.
#                         Requires sufficient locked memory limit (ulimit -l or
#                         /etc/security/limits.conf). See bench-runbook.md §6.
#
# Bench summary (512pp+128tg, 10 threads, FA=1, no-mmap):
#   N_CPU_MOE=40, f16 KV:            pp=451 t/s  tg=39.5 t/s  ← bench artefact
#   N_CPU_MOE=40, q8_0 KV:           pp=455 t/s  tg=36.6 t/s
#   fit ngl=49, 131k ctx, f16 KV:    pp=476 t/s  tg=38.2 t/s
#   fit ngl=49, 64k ctx, f16 KV:     pp=497 t/s  tg=39.60 t/s  (FIT_TARGET=128)
#   fit ngl=49, 64k ctx, q8_0 KV:    pp=511 t/s  tg=39.93 t/s  (FIT_TARGET=128, OOM risk)
#   fit ngl=49, 64k ctx, q8_0 KV:    pp=502 t/s  tg=39.62 t/s  ← this script (FIT_TARGET=512)
#
# ⚠ The 39.5 t/s tg for N_CPU_MOE=40 is a bench artefact (512-token context, no KV
#   pressure). At 64k server context, q8_0 KV ≈ 2 GB — leaving ~10 GB for weights
#   and attention. Realistic server tg at 64k is ~39–40 t/s, close to bench.
#
#   The previous script used --fit on --fit-ctx 131072 (131k context, blk 6-48 CPU).
#   This script uses 64k context + static -ot (blk 9-48 CPU) for deterministic startup
#   and 2 additional GPU layers, giving +35 pp and +1.7 tg vs the 131k fit.
#
# See docs/bench-runbook.md §8 for full bench tables and methodology.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export LLAMA_SET_ROWS="${LLAMA_SET_ROWS:-1}"
# GGML_CUDA_GRAPH_OPT: disabled by default — CUDA graph re-capture at new context depths
# can trigger cuMemCreate with a 1 GiB chunk request, which OOMs with tight VRAM headroom.
# Re-enable only if you have more headroom (FIT_TARGET >= 1200 MiB) or benchmarks show benefit.
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-0}"

LLAMA_SERVER="${LLAMA_SERVER:-${REPO_DIR}/vendor/llama.cpp/build/bin/llama-server}"
CPU_RANGE="${CPU_RANGE:-0-11}"
MODEL="${MODEL:-/mnt/lab/models/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf}"

if [ ! -x "${LLAMA_SERVER}" ]; then
  echo "llama-server not found/executable: ${LLAMA_SERVER}" >&2
  exit 1
fi

if [ ! -f "${MODEL}" ]; then
  echo "model file not found: ${MODEL}" >&2
  exit 1
fi

cmd=(
  "${LLAMA_SERVER}"
  -m "${MODEL}"
  --alias "unsloth/Qwen3-Coder-Next"

  # --- placement (bench-derived, static) ---
  # fit-params result (64k ctx + q8_0 KV + 512 MiB margin): ngl=49, blk 7 ffn_down +
  # blk 8-48 experts to CPU. Blk 0-6 fully on GPU, blk 7 attn+up on GPU.
  # 512 MiB margin (vs 128 MiB) leaves ~500 MB for VMM pool growth mid-session.
  # Do NOT raise --parallel or --ctx-size without checking startup VRAM log.
  -ngl 49
  --override-tensor "blk\.7\.ffn_down.*=CPU,blk\.([89]|[1-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU"
  --no-mmap
  --mlock

  # --- KV cache quantization ---
  # q8_0 at 64k ctx ≈ 2 GB (vs f16 ≈ 4 GB). The 2 GB savings shifted the fit
  # CPU boundary from blk.8 to blk.9, keeping one more full GPU layer.
  -ctk q8_0
  -ctv q8_0

  # --- context + concurrency ---
  --ctx-size 65536
  --parallel 1

  # --- compute ---
  --threads 10
  --threads-batch 12
  --flash-attn on
  --batch-size 2048
  --ubatch-size 512

  # --- sampling ---
  --seed 3407
  --temp 1.0
  --top-p 0.95
  --min-p 0.01
  --top-k 40
  --repeat-penalty 1

  # --- serving ---
  --host 0.0.0.0
  --port 8001
  --jinja
  --prio 2
  --no-warmup
)

if command -v taskset >/dev/null 2>&1 && [ -n "${CPU_RANGE}" ]; then
  exec taskset -c "${CPU_RANGE}" "${cmd[@]}"
fi

exec "${cmd[@]}"
