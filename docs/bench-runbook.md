# Bench Runbook: Hybrid CPU+GPU MoE Inference

Practical guide for benchmarking large MoE models (Qwen3, DeepSeek, etc.) on a
consumer GPU + large-RAM host. Covers both llama.cpp and ik_llama.cpp, ordered
from "run this first" to "try only if you need more".

---

## 1. Overview

**Goal:** Find the fastest stable configuration for a model that does not fit
entirely in GPU VRAM. The typical setup is a single consumer GPU (8–16 GB VRAM)
paired with 64–128 GB system RAM. Expert tensors spill to CPU; attention and
shared weights stay on GPU.

**Two numbers matter:**

| Metric | What it measures | When it dominates |
|--------|-----------------|-------------------|
| `pp` (prompt processing) | Tokens/s filling the KV cache | RAG, long system prompts, first-token latency |
| `tg` (token generation) | Tokens/s during autoregressive decode | Everything the user sees streaming |

For coding agents running multi-turn long sessions, **tg is the bottleneck**.
Optimise pp only if first-token latency is visibly painful.

**Reference hardware (these scripts were tuned on):**

- GPU: RTX 4070 12 GB
- RAM: 96 GB DDR5
- CPU: AMD Ryzen 9 7900 (12-core, single socket)
- Model: Qwen3-Coder-Next UD-Q4\_K\_XL (~47 GB on disk, 512 experts, 10 active/tok)

---

## 2. Prerequisites

### Binaries

| Binary | Repo | Default path |
|--------|------|-------------|
| `llama-bench` | `llama.cpp` | `vendor/llama.cpp/build/bin/llama-bench` |
| `llama-fit-params` | `llama.cpp` | `vendor/llama.cpp/build/bin/llama-fit-params` |
| `llama-bench` (ik) | `ik_llama.cpp` | `vendor/ik_llama.cpp/build/bin/llama-bench` |

Override any path via the matching env var (`LLAMA_BENCH=`, `LLAMA_FIT=`,
`IK_BENCH=`).

### Build (llama.cpp)

```sh
cd vendor/llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-bench llama-fit-params -j$(nproc)
```

### Build (ik_llama.cpp)

```sh
cd vendor/ik_llama.cpp
cmake -B build -DGGML_CUDA=ON -DIK_LLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-bench -j$(nproc)
```

Confirm both are executable before running any bench:

```sh
vendor/llama.cpp/build/bin/llama-bench --version
vendor/ik_llama.cpp/build/bin/llama-bench --version
```

---

## 3. Script Layout

```
bench-models/
  run-llama-bench.sh            # shared llama.cpp runner
  run-ik-llama-bench.sh         # shared ik_llama.cpp runner
  run-llama-fit-bench.sh        # two-stage fit→bench runner (llama.cpp only)
  run-llama-fit-params.sh       # fit-params standalone (print fitted args)

  bench-llama-cpp-<model>.sh              # default bench for a model
  bench-llama-cpp-<model>-strategies.sh  # -ot strategy experiments
  bench-llama-cpp-<model>-fit.sh         # fit-based auto placement

  bench-ik-llama-cpp-<model>.sh              # ik_llama default bench
  bench-ik-llama-cpp-<model>-strategies.sh  # ik_llama MoE flag experiments
```

### Runner delegation

Every model script sets env vars and then `exec`s a runner:

```
bench-llama-cpp-qwen3-coder-next.sh
  └─ exec run-llama-bench.sh

bench-llama-cpp-qwen3-coder-next-strategies.sh
  └─ exec run-llama-bench.sh          (with OVERRIDE_TENSOR set)

bench-llama-cpp-qwen3-coder-next-fit.sh
  └─ exec run-llama-fit-bench.sh      (runs fit-params, then bench)

bench-ik-llama-cpp-qwen3-coder-next.sh
  └─ exec run-ik-llama-bench.sh

bench-ik-llama-cpp-qwen3-coder-next-strategies.sh
  └─ exec run-ik-llama-bench.sh       (with FUSED_MOE/MERGE_*/GER flags)
```

### Key env var overrides (both runners)

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL` | _(model-script default)_ | Path to `.gguf` |
| `TASKS` | `512,128` | `pp,tg` passed as `-pg`; overrides `N_PROMPT`/`N_GEN` |
| `N_GPU_LAYERS` | `99` | Set lower to spill non-MoE layers too |
| `N_CPU_MOE` | `40` | Integer MoE-layer CPU count (`-ncmoe` / `--n-cpu-moe`) |
| `THREADS` | `10` | CPU threads for generation |
| `CPU_RANGE` | `0-11` | `taskset -c` affinity; set `""` to disable |
| `FA` | `1` | Flash attention |
| `MMP` | `0` | mmap (0 = load into RAM; recommended for hybrid) |
| `CACHE_TYPE_K` | _(f16)_ | KV cache K dtype |
| `CACHE_TYPE_V` | _(f16)_ | KV cache V dtype |
| `OVERRIDE_TENSOR` | _(unset)_ | Custom `-ot` regex; disables `N_CPU_MOE` path |
| `REPETITIONS` | _(runner default)_ | Repetitions per test |
| `OUTPUT_FMT` | _(runner default)_ | `md` \| `csv` \| `json` \| `jsonl` |

#### ik_llama-specific extras

| Var | Flag | Default in ik_llama | Effect |
|-----|------|-------------------|--------|
| `FUSED_MOE` | `-fmoe` | **1** (on) | Fused MoE expert kernel — major tg win |
| `MERGE_UP_GATE` | `-muge` | 0 | Repack up+gate into one matrix; +tg, -pp, +~27 GB RAM |
| `MERGE_QKV` | `-mqkv` | 0 | Merge Q/K/V projections; small tg improvement |
| `GROUPED_ROUTING` | `-ger` | 0 | Group expert routing for cache locality |
| `ROPE_CACHE` | `-rcache` | 0 | Cache RoPE; may help at long context |

---

## 4. Experiment Sequence

Run experiments in this order. Stop when results are good enough.

### 4a. Baseline — default N_CPU_MOE

```sh
./bench-llama-cpp-qwen3-coder-next.sh
```

What it tests: the integer `-ncmoe` path with 40 MoE layers on CPU.
What to look for: establishes your floor. Note `pp` and `tg` t/s.

Try adjusting `N_CPU_MOE` up/down to find the VRAM sweet spot:

```sh
N_CPU_MOE=30 ./bench-llama-cpp-qwen3-coder-next.sh   # more on GPU
N_CPU_MOE=48 ./bench-llama-cpp-qwen3-coder-next.sh   # all on CPU
```

---

### 4b. Manual -ot regex strategies (llama.cpp)

```sh
STRATEGY=all-cpu-moe  ./bench-llama-cpp-qwen3-coder-next-strategies.sh
STRATEGY=partial-cpu  ./bench-llama-cpp-qwen3-coder-next-strategies.sh
STRATEGY=up-down-cpu  ./bench-llama-cpp-qwen3-coder-next-strategies.sh
STRATEGY=up-cpu       ./bench-llama-cpp-qwen3-coder-next-strategies.sh
```

| Strategy | VRAM use | pp | tg | Notes |
|----------|----------|----|----|-------|
| `all-cpu-moe` | Lowest | Baseline | Baseline | Safe starting point |
| `partial-cpu` | Medium | +pp | ~same tg | Layers 0–5 expert on GPU |
| `up-down-cpu` | Higher | ++ pp | +tg | Gate experts stay on GPU |
| `up-cpu` | Highest | +++ pp | ++ tg | Only up-proj on CPU |

**OOM risk:** `up-down-cpu` and `up-cpu` load 512 experts × multiple
projections onto 12 GB VRAM. On a 12 GB card with a large model they will
OOM. If you see a CUDA OOM, fall back to `all-cpu-moe` or use `partial-cpu`
with a higher layer cutoff.

Custom regex: set `OVERRIDE_TENSOR` directly to bypass the preset:

```sh
OVERRIDE_TENSOR=".ffn_gate_exps.=CPU" \
  ./bench-llama-cpp-qwen3-coder-next-strategies.sh
```

---

### 4c. llama-fit-params auto-placement

```sh
./bench-llama-cpp-qwen3-coder-next-fit.sh
```

`llama-fit-params` probes free VRAM and computes optimal `-ngl`/`-ot`/`-ts`
automatically, leaving `FIT_TARGET` MiB headroom.

Useful knobs:

```sh
FIT_TARGET=2048 ./bench-llama-cpp-qwen3-coder-next-fit.sh   # leave 2 GB free
FIT_CTX=8192    ./bench-llama-cpp-qwen3-coder-next-fit.sh   # min context 8k
```

Use the standalone runner to inspect what fit would choose without running
bench:

```sh
MODEL=/path/to/model.gguf ./run-llama-fit-params.sh
```

Best used when you do not want to hand-tune `-ot` patterns, or when switching
between models frequently.

---

### 4d. Poll level (llama.cpp only)

```sh
POLL=0   ./bench-llama-cpp-qwen3-coder-next.sh
POLL=50  ./bench-llama-cpp-qwen3-coder-next.sh
POLL=100 ./bench-llama-cpp-qwen3-coder-next.sh
```

`--poll` controls CPU spin-wait aggressiveness (0 = sleep/yield, 100 = busy
spin). **On hybrid CPU+GPU inference the effect is typically flat** — the GPU
kernel and PCIe transfer dominate, not CPU polling latency. Run the sweep once
to confirm; if differences are within noise, leave at default (50) or set to 0
to reduce CPU load.

---

### 4e. KV cache quantization

```sh
CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 ./bench-llama-cpp-qwen3-coder-next.sh
CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 ./bench-llama-cpp-qwen3-coder-next.sh
```

At the default bench context (`512pp + 128tg = 640 tokens`) the KV cache is
tiny — impact on t/s will be near zero. KV quant matters when:

- Context is 8k+ tokens (KV cache grows large, VRAM pressure increases).
- You are simultaneously running multiple inference sessions.

`q8_0` is lossless for most purposes. `q4_0` saves more VRAM but may degrade
quality on long contexts. Bench at your real working context size to measure
the actual tradeoff.

---

### 4f. ik_llama.cpp — fused-moe (default on)

```sh
./bench-ik-llama-cpp-qwen3-coder-next.sh
```

`FUSED_MOE` is on by default in ik_llama. This is the single largest tg
improvement ik_llama offers over stock llama.cpp — expect a meaningful uplift
(model-dependent, often +10–30% tg). Compare directly against the llama.cpp
baseline from step 4a.

To explicitly test with fused-moe off:

```sh
FUSED_MOE=0 ./bench-ik-llama-cpp-qwen3-coder-next.sh
```

---

### 4g. ik_llama — merge-qkv

```sh
STRATEGY=fused-mqkv ./bench-ik-llama-cpp-qwen3-coder-next-strategies.sh
```

Merges Q, K, V projection weight matrices into a single matrix. Small tg
improvement, no significant RAM penalty. Safe to enable alongside fused-moe.

---

### 4h. ik_llama — merge-up-gate-experts

```sh
STRATEGY=fused-muge ./bench-ik-llama-cpp-qwen3-coder-next-strategies.sh
```

Repacks each expert's up and gate projection into a single weight matrix,
enabling a more efficient combined kernel.

**Trade-offs:**

| | Effect |
|-|--------|
| tg | +0.5–1.0 t/s improvement |
| pp | Significant regression (−200–300 t/s) |
| RAM | ~+27 GB extra system RAM required |
| Load time | Noticeably longer (repack happens at startup) |

**Only use if:** tg is the sole metric that matters, you have ≥75 GB RAM free
after the model loads, and you can tolerate slow pp. Do not enable alongside
heavy prompt workloads.

---

### 4i. ik_llama — grouped-expert-routing

```sh
STRATEGY=fused-ger ./bench-ik-llama-cpp-qwen3-coder-next-strategies.sh
```

Sorts token-to-expert assignments to improve memory access locality during
expert computation. Effect varies by model and batch size. Run the sweep and
compare against plain `fused`:

```sh
STRATEGY=fused     ./bench-ik-llama-cpp-qwen3-coder-next-strategies.sh
STRATEGY=fused-ger ./bench-ik-llama-cpp-qwen3-coder-next-strategies.sh
```

---

### 4j. NUMA modes

```sh
NUMA=distribute ./bench-llama-cpp-qwen3-coder-next.sh
NUMA=isolate    ./bench-llama-cpp-qwen3-coder-next.sh
```

On a **single-socket** system (one physical CPU), NUMA modes typically provide
no benefit — there is only one NUMA node. Confirm with `numactl --hardware`.

Useful only on dual-socket or multi-die systems where memory locality between
sockets matters. On a single-socket Ryzen, `taskset -c 0-11` (via `CPU_RANGE`)
is sufficient affinity control.

---

### 4k. Speculative decoding

Not yet scripted. Future work. Draft flag: `-d <draft-depth>` in
`run-llama-bench.sh` via `N_DEPTH`. Requires a small draft model of the same
architecture. Potential 2–4× tg uplift when draft acceptance rate is high.

---

## 5. Reading Results

llama-bench emits a result table (default: Markdown) after each run:

```
| model | size | params | backend | ... | test | t/s |
|-------|------|--------|---------|-----|------|-----|
| ...   | ...  | ...    | CUDA    | ... | pp 512 | 1234.56 ± 5.6 |
| ...   | ...  | ...    | CUDA    | ... | tg 128 | 12.34 ± 0.1  |
```

| Column | Meaning |
|--------|---------|
| `pp N` | Prompt processing N tokens — **higher is better** |
| `tg N` | Token generation N tokens — **higher is better** |
| `t/s` | Tokens per second (mean ± std over repetitions) |

**Context size in the bench** is `N_PROMPT + N_GEN`. The default `TASKS=512,128`
tests 512-token prompt processing and 128-token generation. This is a short
context — KV cache fits easily, and VRAM pressure is minimal. To stress VRAM
and long-context performance, use `TASKS=4096,512` or higher.

**For coding agents:** tg dominates user experience. A 12 t/s tg with 600 pp
is better than 8 t/s tg with 900 pp for interactive use. Optimise accordingly.

---

## 6. Known Constraints

### Why up-cpu and up-down-cpu OOM on 12 GB

Qwen3-Coder-Next has 512 experts per MoE layer across 48 layers. Each expert
projection tensor is small, but 512 × 3 projections (gate, up, down) × 48
layers adds up quickly. Keeping gate+down on GPU while only spilling up to CPU
still requires loading hundreds of expert weight matrices into VRAM. On 12 GB
this exhausts VRAM during prefill. The `all-cpu-moe` strategy avoids this by
routing all expert tensors to CPU.

### Why merge-up-gate nearly doubles RAM

`MERGE_UP_GATE` concatenates up-proj and gate-proj weights for every expert at
load time, creating new combined tensors stored in system RAM. The original
tensors remain in memory too until they are evicted (implementation-dependent).
Peak RAM during load can be ~2× the expert weight size, and the steady-state
footprint remains higher due to the merged tensors. For a ~47 GB model with
~60% of weight in expert projections, this adds ~27–30 GB.

### Why pp regresses with merge-up-gate

The fused up+gate kernel processes one token's expert set at a time and is
optimised for tg (low-batch sequential decode). Prompt processing is high-batch
parallel computation that benefits from the original separate matrix layouts.
The merged kernel loses this batching advantage, causing pp regression.

### mmap=0 is recommended for hybrid inference

With `MMP=0` (default in these scripts), the entire model is loaded into RAM
before inference starts. This avoids page-fault latency during generation when
expert tensors are accessed non-sequentially. With mmap enabled (`MMP=1`), cold
expert access causes OS page faults that add jitter to tg and can reduce
throughput significantly.

---

## 7. Adding a New Model

### Step 1 — Create the default bench script

Copy the closest existing script as a template:

```sh
cp bench-models/bench-llama-cpp-qwen3-coder-next.sh \
   bench-models/bench-llama-cpp-<new-model>.sh
```

### Step 2 — Set model path and key params

Edit the new script. At minimum update:

```sh
MODEL="${MODEL:-/path/to/your/model.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"     # start at 99; lower if OOM
N_CPU_MOE="${N_CPU_MOE:-<n>}"          # set to total MoE layer count first
THREADS="${THREADS:-10}"
```

To find the MoE layer count:

```sh
# llama-bench -m model.gguf --verbose prints layer metadata
# or inspect the GGUF header:
python3 -c "
import struct, sys
# quick gguf key scan
with open('model.gguf','rb') as f:
    print(f.read(4096))  # scan for 'expert_count' key
"
```

### Step 3 — Run the baseline

```sh
./bench-models/bench-llama-cpp-<new-model>.sh
```

If it OOMs, reduce `N_GPU_LAYERS` or switch to `N_CPU_MOE` equal to total
layer count (all experts on CPU).

### Step 4 — Run fit and strategies

Once baseline is stable, run fit-params to get a better placement:

```sh
# Create a fit script:
cp bench-models/bench-llama-cpp-qwen3-coder-next-fit.sh \
   bench-models/bench-llama-cpp-<new-model>-fit.sh
# Update MODEL path, then:
./bench-models/bench-llama-cpp-<new-model>-fit.sh
```

Then create the strategies variant and work through the experiment sequence in
§4, starting from §4a.

---

## Quick Reference

```sh
# Fastest path to a working bench:
./bench-models/bench-llama-cpp-<model>.sh               # baseline
./bench-models/bench-llama-cpp-<model>-fit.sh           # auto placement
./bench-models/bench-ik-llama-cpp-<model>.sh            # ik_llama fused-moe

# Override any param inline:
TASKS=4096,256 THREADS=12 ./bench-models/bench-llama-cpp-<model>.sh

# ik_llama strategy sweep:
for s in fused fused-ger fused-mqkv; do
  STRATEGY=$s ./bench-models/bench-ik-llama-cpp-<model>-strategies.sh
done
```
```

Now let me create the file properly using the edit_file tool since it needs to be saved: