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

### Evidence required for a published profile

Save one record from the same run with: PP, TG, tested context, cold/warm cache
state, speculative draft acceptance when applicable, the exact llama.cpp
commit, and the exact benchmark command. Do not fill a missing field from a
different run just because the model and quant match. Historical dashboard
profiles show `not recorded` where the old scripts did not preserve this data.

External hardware results use `docs/community-runs.schema.json`. They stay in
the dashboard's separate community view and never enter the local RTX 4070
ranking.

**Reference hardware (these scripts were tuned on):**

- GPU: RTX 4070 12 GB
- RAM: 64 GB DDR5-5867 (4×16 GB)
- CPU: Intel i5-12600K (10-core/16-thread, 6 P-cores + 4 E-cores, Alder Lake, single socket)
- Model: Qwen3-Coder-Next UD-Q4\_K\_XL (~47 GB on disk, 512 experts, 10 active/tok)

**Benched models:**

| Model | Quant | Disk | Experts | Active/tok | Best tg (bench) | Best pp |
|-------|-------|------|---------|------------|---------|---------|
| Qwen3-Coder-Next (80B.A3B) | UD-Q4\_K\_XL | ~47 GB | 512 | **3B** | 40.6 t/s¹ | **511 t/s²** |
| Qwen3.6-35B-A3B | UD-Q5\_K\_XL | ~25 GB | 256 | **8B** | **52.3 t/s³** | **971 t/s³** |
| Qwen3.5-122B-A10B | UD-IQ4\_XS | ~56 GB | 256 | **10B** | 9.8 t/s | 284 t/s |
| Gemma-4-26B-A4B | Q6\_X\_L | TBD | TBD | TBD | TBD | TBD |

¹ N\_CPU\_MOE=40, f16 KV, **512-token bench context only** — bench does not pre-allocate full ctx KV.
  At 64k server context with q8\_0 KV the realistic tg is ~39–40 t/s (fit ngl=49 placement).
² fit-params ngl=49, 64k ctx floor + q8\_0 KV; blk 0–7 fully on GPU, blk 8 partial, blk 9–48 on CPU
³ fit-params (`FIT_CTX=65536`, `FIT_TARGET=512`) with `-ngl 41` and fit-shaped expert offload (`gate_up` included).

Qwen3.6-35B-A3B is currently the fastest decode path in this runbook on a 12 GB card.
Qwen3.5-122B remains much slower at tg because it activates more parameters per token,
so no offload pattern can close that architectural gap.

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

### System state check (run before every bench)

Bench tg is CPU-bound (expert compute). If the CPU is in a low-power state, tg
will read 27–30 t/s instead of the expected ~39 t/s — a 30% drop with no other
symptoms (pp is GPU-bound and unaffected, so it will look fine).

```sh
# 1. Check governor — must be "performance" on all cores
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# expected: performance

# 2. Check EPP — must be "performance" (intel_pstate active mode)
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
# expected: performance

# 3. Check actual CPU frequency — P-cores should be near max boost
grep "cpu MHz" /proc/cpuinfo | sort -rn | head -4
# expected: 4500–4900 MHz on i5-12600K
```

If governor or EPP is not `performance`, fix before running:

```sh
sudo cpupower frequency-set -g performance
# or via tuned-ppd (CachyOS recommended over power-profiles-daemon):
sudo tuned-adm profile throughput-performance
```

**Why this happens on CachyOS:** `intel_pstate` defaults to `schedutil` or
`powersave` EPP, optimised for responsiveness. CachyOS recommends `tuned-ppd`
for persistent power profile management. See:
https://wiki.cachyos.org/configuration/general_system_tweaks/

**Intel Alder Lake specific:** even with governor=performance, EPP matters.
`balance_performance` EPP caps boost duration on the P-cores under sustained
load. Always verify EPP reads `performance`, not just the governor.

### Tg variability — ROOT CAUSE FOUND AND FIXED

**Root cause:** `power-profiles-daemon` (KDE default) was setting a non-performance power profile on some boots, subtly degrading CPU/HWP state in a way that all sysfs checks (`governor`, `EPP`, `scaling_max_freq`, `cpu MHz`) still showed "performance" — yet tg ran at 32–35 t/s instead of ~39–40 t/s.

**Fix:** Replace `power-profiles-daemon` with `tuned-ppd` (CachyOS recommended):
```sh
sudo pacman -S tuned-ppd        # removes power-profiles-daemon automatically
sudo systemctl enable --now tuned
sudo tuned-adm profile throughput-performance
```
After reboot: tg = **40.60 t/s** with zero preflight checks needed.

### Tg variability checklist (intermittent 33 t/s vs expected ~39 t/s) — archived

Symptom: tg varies between boots with no obvious cause. pp unaffected (GPU-bound).

| # | Check | How to verify | Status |
|---|-------|--------------|--------|
| 1 | CPU governor = performance | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | ✅ ruled out |
| 2 | EPP = performance | `cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference` | ✅ ruled out |
| 3 | CPU frequency boosting (4.5–4.9 GHz) | `grep "cpu MHz" /proc/cpuinfo \| sort -rn \| head -4` | ✅ ruled out |
| 4 | RAPL power limits (PL1=125W, PL2=241W) | `sudo cat /sys/class/powercap/intel-rapl/intel-rapl:0/constraint_{0,1}_power_limit_uw` | ✅ ruled out |
| 5 | GPU VRAM pressure | `nvidia-smi \| grep MiB` — must be near-empty before bench | ✅ ruled out |
| 6 | Thermal throttling | `cat /sys/class/thermal/thermal_zone*/temp` — expect <70°C | ✅ ruled out |
| 7 | LTS kernel version | `uname -r` — same kernel both good and bad runs | ✅ ruled out |
| 8 | llama.cpp binary | build hash shown in bench output — same both sessions | ✅ ruled out |
| 9 | Background CPU load (Zed etc.) | `ps aux --sort=-%cpu \| head -8` | ✅ ruled out |
| 10 | cgroup CPU quota | `cat /sys/fs/cgroup/$(cat /proc/self/cgroup \| grep -o '[^:]*$' \| head -1)/cpu.max` — expect `max 100000` | ✅ ruled out |
| 11 | RAM speed | `sudo dmidecode -t memory \| grep "Configured Memory Speed"` — expect 5867 MT/s | ✅ ruled out |
| 12 | Deep C-states (C6/C7/C8/C10 wake latency) | `sudo cpupower idle-info` — disable with `sudo cpupower idle-set -D 1` then bench | ✅ ruled out — C10 (680µs), C8 (280µs), C6 (220µs) all disabled; tg unchanged |
| 13 | tuned-ppd throughput profile | `sudo pacman -S tuned && sudo tuned-adm profile throughput-performance` | ⬜ not yet tested |
| 14 | Intel HWP hardware response | May not follow SW governor correctly on some boots; `cat /sys/devices/system/cpu/intel_pstate/hwp_dynamic_boost` | ⬜ hwp_dynamic_boost=0 confirmed; HWP MSR state unverified (needs `sudo rdmsr -a 0x774`) |
| 15 | PCIe link speed | `nvidia-smi -q \| grep -A 3 "PCIe Generation"` — must be Gen 4 during bench | ✅ ruled out — Gen 4 confirmed live during tg phase |
| 16 | GPU SM clock drop during tg | `watch -n 0.2 "nvidia-smi -q -d CLOCK"` during bench tg phase | ✅ ruled out — 2520 MHz during tg (normal; GPU scales back due to low utilisation) |
| 17 | RAM speed on slow boot | `sudo dmidecode -t memory \| grep "Configured Memory Speed"` — run on a confirmed-slow boot | ✅ ruled out — 5867 MT/s confirmed on slow boot |
| 18 | Transparent Huge Pages | `cat /sys/kernel/mm/transparent_hugepage/enabled` | ✅ ruled out — [always] mode active |
| 19 | HWP MSR hardware EPP (bits 31–24 of 0x774) | `sudo modprobe msr && sudo rdmsr -a 0x774` — 0x00=performance, 0x80=balanced | ⬜ not yet verified — sysfs shows performance but HW may differ |
| 20 | power-profiles-daemon interfering | `systemctl status power-profiles-daemon` + `powerprofilesctl get` | ✅ **ROOT CAUSE** — replaced with tuned-ppd; tg jumped to 40.60 t/s on clean reboot |
| 21 | vm.swappiness=150 (ZRAM) pushing expert weights into ZRAM | `sysctl vm.swappiness` — CachyOS default is 150 with ZRAM | ⬜ confirmed 150; unlikely with 58 GiB free but worth checking on memory-pressured boots |
| 22 | energy_perf_bias per-core | `cat /sys/devices/system/cpu/cpu*/power/energy_perf_bias` — expect all 0 | ✅ ruled out — all P-cores at 0 (performance) |

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

  logs/                         # timestamped bench output (gitignored)
    YYYY-MM-DD_HH-MM-SS_<model>.log
    YYYY-MM-DD_HH-MM-SS_<model>_fit.log
```

### Runner delegation

Every model script sets env vars and then delegates to a runner. Runners write
output to both stdout and `logs/` (via `tee`):

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
sockets matters. On a single-socket system, `taskset -c 0-11` (via `CPU_RANGE`)
is sufficient affinity control.

**Intel Alder Lake note:** the i5-12600K has 6 P-cores (CPUs 0–11 with SMT) and
4 E-cores (CPUs 12–15, no SMT). Always keep `CPU_RANGE=0-11` to pin expert
compute to P-cores only — E-cores are significantly slower for GEMM workloads
and will drag down tg if included.

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

## 8. Bench Results

### Gemma-4-26B-A4B UD-Q6\_K\_XL — benchmark stub (results pending)

**Architecture notes:** Unsloth refreshed Gemma 4 26B-A4B GGUF variants; this
profile targets the `UD-Q6_K_XL` quant specifically for validation against the
existing `UD-Q5_K_XL` baseline.

**Artifacts added for this profile:**

- `llama-swap.yaml` model ID: `gemma-4-26b-a4b-q6-k-xl`
- Bench scripts:
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l.sh`
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-strategies.sh`
  - `bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-fit.sh`

**Runbook command set (fill pp/tg after run):**

```sh
./bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l.sh
./bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-strategies.sh
./bench-models/bench-llama-cpp-gemma-4-26b-a4b-q6-x-l-fit.sh
```

### Qwen3-Coder-Next UD-Q4\_K\_XL — RTX 4070 12 GB / Intel i5-12600K / 64 GB DDR5

**Architecture:** 48 blocks, 512 routed experts per MoE layer, 10 active/token, ~47 GB on disk.
Model activates only ~3B parameters per token (80B total, sparse MoE), which is why tg is
exceptionally fast for an 80B-class model on consumer hardware.

**No shared expert tensors.** Unlike Qwen3.5-122B, this model only uses `_exps` tensors —
no `_shexp`. The `-ot` patterns below do not need the `(ch|)` group, though it is harmless
to include it for forward-compat.

**fit-params result (FIT\_CTX=131072, FIT\_TARGET=128 MiB, f16 KV):**
```
-c 131072 -ngl 49 -ot "blk\.5\.ffn_down.*=CPU,blk\.6\.ffn_(up|down|gate)_(ch|)exps=CPU,...,blk\.48\.ffn_(up|down|gate)_(ch|)exps=CPU"
```
Placement: blk 0–4 fully on GPU, blk 5 attention on GPU (only blk 5 ffn_down spills),
blk 6–48 expert tensors to CPU. pp=476, tg=38.2.

**fit-params result (FIT\_CTX=65536, CACHE\_TYPE\_K=q8\_0, FIT\_TARGET=128 MiB):**
```
-c 65536 -ngl 49 -ot "blk\.8\.ffn_(gate|down).*=CPU,blk\.9\.ffn_(up|down|gate)_(ch|)exps=CPU,...,blk\.48\.ffn_(up|down|gate)_(ch|)exps=CPU"
```
Placement: blk 0–7 fully on GPU, blk 8 partially on GPU (gate+down spill only),
blk 9–48 expert tensors to CPU. pp=511, tg=39.93. **This is the optimized script config.**
Smaller ctx floor (64k vs 131k) freed VRAM for 2 more GPU layers; q8\_0 KV (~2 GB vs ~4 GB
at 64k f16) freed one additional layer beyond the 64k+f16 result.

**Strategy sweep (512pp + 128tg, 10 threads, FA=1, no-mmap):**

| Strategy | flags | KV | pp (t/s) | tg (t/s) | Notes |
|----------|-------|----|----------|----------|-------|
| **default (N\_CPU\_MOE=40)** | `-ncmoe 40 -ngl 99` | f16 | 451–456 | **40.6** | best bench tg — requires performance governor+EPP |
| `all-cpu-moe` via `-ot` | `-ot ".ffn_.*_exps.=CPU" -ngl 99` | f16 | 410 | 33.5 | all experts on CPU; pp hurt, tg hurt vs N\_CPU\_MOE |
| fit-params (131k ctx) | `-ngl 49 -ot blk5-down+blk6-48` | f16 | 476 | 38.2 | best pp at 131k |
| fit-params (64k ctx) | `-ngl 49 -ot blk7-gate+blk8-48` | f16 | 497 | 39.60 | +2 GPU layers vs 131k fit |
| fit-params (64k ctx, 128 MiB margin) | `-ngl 49 -ot blk8-gate+blk9-48` | q8\_0 | 511 | 39.93 | best bench pp — OOM risk at long prompts (VMM pool) |
| **fit-params (64k ctx, 512 MiB margin)** | **`-ngl 49 -ot blk7-down+blk8-48`** | **q8\_0** | **502** | **39.62** | **production config — stable at 44k+ token prompts** |

**Both 39.5 and 38.2 are real bench numbers** — both measured at 512-token context where
KV VRAM is negligible. The 1.3 t/s gap between N\_CPU\_MOE=40 and fit-params is real but
small. The 39.5 requires a clean performance power state (governor + EPP); under powersave
or balance\_performance it drops to 27–33 t/s. At 131k server context the KV cache alone
consumes ~4 GB (q8\_0) or ~8 GB (f16), which squeezes attention VRAM — realistic server
tg is ~36–38 t/s with fit placement and q8\_0 KV.

**Winner for pp:** fit-params (`-ngl 49` + per-block `-ot`). Physically drops unneeded GPU
allocations above ngl=49 while keeping blk 0–4 fully on GPU.

**N\_CPU\_MOE=40 vs fit-params — why bench tg differs from server tg:**
`-ncmoe 40` with `ngl=99` leaves the full 12 GB VRAM free for attention at bench context.
At 131k context with f16 KV, ~8 GB is consumed by the KV cache, leaving only ~4 GB for
attention — forcing fit-derived placement (ngl=49) as the only viable option. This is why
`--fit on` in the run script and fit-params in the bench script must both use
`--fit-ctx 131072` / `FIT_CTX=131072` to get a placement valid under real serving load.

---

**KV cache quantization sweep (512pp + 128tg, N\_CPU\_MOE=40, ngl=99):**

> ⚠️ These numbers use bench-only 512-token context. The tg improvement from q8\_0 is
> real even in bench, but the absolute tg figures are higher than you will see at 131k
> server context (where KV VRAM pressure dominates regardless of quant type).

| KV type | pp (t/s) | tg (t/s) | VRAM saved vs f16 at 131k ctx | Notes |
|---------|----------|----------|-------------------------------|-------|
| `f16` | 455.75 | 33.47 | — | bench baseline (measured under degraded power state) |
| **`q8_0`** | **454.67** | **36.57** | **~4 GB** | **recommended — +3.1 tg vs f16, free win** |
| `q4_0` | 454.34 | 36.85 | ~6 GB | marginal further tg gain, lower fidelity |

**q8\_0 is a strong win at bench context and essential at 131k server context.** At bench
(512 tokens) it frees just enough VRAM to give +3.1 t/s tg with zero pp cost. At 131k
context, f16 KV consumes ~8 GB of the 12 GB VRAM — leaving only ~4 GB for model weights
and attention, which is not enough for ngl=49. q8\_0 halves that to ~4 GB, giving fit the
room it needs to keep blk 0–4 on GPU. **Without q8\_0 (or similar), 131k context + ngl=49
may fail to load entirely.** Always use `-ctk q8_0 -ctv q8_0` for this model at 131k.

**q4\_0** saves another ~2 GB vs q8\_0 and gains a marginal further 0.3 t/s tg in bench,
but the quality tradeoff is more visible at q4 precision. Not recommended for coding tasks.

---

**Recommended serving configuration (`llama-swap.yaml` model: `qwen3-coder-next`):**

| Setting | Value | Reason |
|---------|-------|--------|
| `-ngl` | 49 | fit-params derived for 64k ctx + q8\_0 + 512 MiB margin; blk 0–6 fully on GPU |
| `--override-tensor` | `blk7-down + blk8-48 experts → CPU` | Static; 512 MiB margin leaves room for VMM pool growth vs 128 MiB (OOM at 44k+ prompts) |
| `-ctk / -ctv` | `q8_0` | KV at 64k ≈ 2 GB (vs ~4 GB f16); smaller KV freed 2 extra GPU layers vs 131k fit |
| `--ctx-size` | 65536 | 64k; ample for coding sessions; 2 extra GPU layers vs 131k fit |
| `GGML_CUDA_GRAPH_OPT` | 0 | Disabled — graph re-capture at new context depths triggers VMM pool growth; see §9 |
| `--parallel` | 1 | Single-user homelab; each extra slot multiplies KV VRAM by n\_parallel |

**Bench vs server tg summary:**

| Config | ctx | KV | pp (t/s) | tg (t/s) | Notes |
|--------|-----|----|----------|----------|-------|
| N\_CPU\_MOE=40 | 512 bench | f16 | 451 | **40.6** | requires performance governor+EPP |
| N\_CPU\_MOE=40 | 512 bench | q8\_0 | 455 | 36.6 | +0 pp cost at bench context |
| fit ngl=49, 131k ctx | 512 bench | f16 | 476 | 38.2 | best pp at 131k |
| fit ngl=49, 64k ctx | 512 bench | f16 | 497 | 39.60 | +2 GPU layers vs 131k |
| fit ngl=49, 64k ctx, 128 MiB | 512 bench | q8\_0 | 511 | 39.93 | OOM risk at long prompts |
| **fit ngl=49, 64k ctx, 512 MiB** | 512 bench | q8\_0 | **502** | **39.62** | **this script** |
| fit ngl=49, 64k ctx, 512 MiB | **64k server** | q8\_0 | — | **~39–40** | KV ≈ 2 GB, stable at 44k+ tokens |

**To reproduce:**

```sh
# Baseline (N_CPU_MOE=40, f16 KV)
./bench-models/bench-llama-cpp-qwen3-coder-next.sh

# fit-params placement
./bench-models/bench-llama-cpp-qwen3-coder-next-fit.sh

# KV quant comparison
CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 ./bench-models/bench-llama-cpp-qwen3-coder-next.sh

# Optimized server path via llama-swap
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"say hi"}]}'
```

---

### Qwen3.6-35B-A3B UD-Q5\_K\_XL — RTX 4070 12 GB / Intel i5-12600K / 64 GB DDR5

**Architecture:** 40 blocks, 256 experts per MoE layer, 8 active/token, 24.76 GiB on disk.

**Two model-specific gotchas:**

1. Include `gate_up` in expert patterns (`ffn_(up|down|gate_up|gate)_...`) or
   the offload is incomplete and memory pressure rises unexpectedly.
2. The integer `-ncmoe` path was not stable for this build/quant profile on
   12 GB VRAM. Explicit `-ot` patterns or fit-derived placement were reliable.

**Results (512pp + 128tg, 10 threads, FA=1, no-mmap, q8\_0 KV, 1 repetition):**

| Strategy | ngl | flags | pp (t/s) | tg (t/s) | Notes |
|----------|-----|-------|----------|----------|-------|
| baseline / `all-cpu-moe` | 99 | `.ffn_(up\|down\|gate_up\|gate)_(ch\|)exps=CPU` | 654.16 | 41.10 | safe baseline |
| `partial-cpu` | 99 | `blk\.(4\|...)\.ffn_(up\|down\|gate_up\|gate)_(ch\|)exps=CPU` | 746.36 | 44.35 | keep early blocks fully on GPU |
| `up-down-cpu` | 99 | `.ffn_(up\|down)_(ch\|)exps=CPU` | 865.26 | 48.95 | gate experts on GPU |
| **fit-params auto** | **41** | `blk.13 down + blk.14-40 (up/down/gate_up/gate)` | **970.77** | **52.33** | **winner** |
| `up-cpu` | 99 | `.ffn_up_(ch\|)exps=CPU` | OOM | — | fails to load on 12 GB |

**Winner:** fit-derived placement (`bench-llama-cpp-qwen3-6-35b-a3b-fit.sh`).
Fit chose `-ngl 41` with a split that keeps early blocks dense on GPU and moves
late-block experts to CPU. This was the best pp/tg combination in this sweep.

**Vision profile added:** use `qwen3-6-35b-a3b-vision` in `llama-swap.yaml` or
`bench-models/run-llama-cpp-qwen3-6-35b-a3b-vision.sh` for direct runs. The
vision profile keeps `GGML_CUDA_GRAPH_OPT=0`, `FIT_TARGET=2048`,
`BATCH_SIZE=256`, `UBATCH_SIZE=512` as safer defaults for 12 GB VRAM.

**To reproduce best result:**

```sh
REPETITIONS=1 ./bench-models/bench-llama-cpp-qwen3-6-35b-a3b-fit.sh
```

---

### Qwen3.5-122B-A10B UD-IQ4\_XS — RTX 4070 12 GB / Intel i5-12600K / 64 GB DDR5

**Architecture:** 48 blocks, 256 routed experts + 1 shared expert per layer, 8 active/token, 56 GB on disk.

**Key gotcha — shared experts must be included in `-ot` patterns.**
This model uses two expert tensor families:
- `ffn_{up,down,gate}_exps` — routed experts (256 per layer)
- `ffn_{up,down,gate}_shexp` — shared expert (always active, 1 per layer)

Simple patterns like `.ffn_.*_exps.=CPU` only match routed experts. The shared
expert tensors stay on GPU and silently consume VRAM, causing CUDA OOM on any
strategy that tries to keep attention on GPU. The correct pattern form is
`ffn_(up|down|gate)_(ch|)exps=CPU` — the `(ch|)` captures both `_exps` (routed)
and `_shexp` (shared). This was confirmed by `llama-fit-params` output.

**Results (512pp + 128tg, 10 threads, FA=1, no-mmap):**

| Strategy | Backend | flags | pp (t/s) | tg (t/s) | Notes |
|----------|---------|-------|----------|----------|-------|
| `N_CPU_MOE=48` | llama.cpp | `-ncmoe 48` | 267.85 | 9.30 | baseline, all experts on CPU |
| `all-cpu-moe` | llama.cpp | `.ffn_(up\|down\|gate)_(ch\|)exps=CPU` | 267.76 | 9.47 | same placement via `-ot` |
| **`partial-cpu`** | **llama.cpp** | `blk\.(3\|[4-9]\|[0-9][0-9]+)\.ffn_(up\|down\|gate)_(ch\|)exps=CPU` | **284.41** | **9.84** | **best — blk 0-2 on GPU** |
| fit-params auto | llama.cpp | `-ngl 49` + per-layer `-ot` from blk 3 | 283.93 | 9.55 | matches partial-cpu |
| `up-down-cpu` | llama.cpp | `.ffn_(up\|down)_(ch\|)exps=CPU` | OOM | — | gate experts on GPU too much |
| ik_llama fused-moe + partial-cpu `-ot` | ik_llama | `-fmoe 1` + same `-ot` | ~55 | 8.60 | pp collapses, tg worse |
| ik_llama no fused-moe + partial-cpu `-ot` | ik_llama | `-fmoe 0` + same `-ot` | ~63 | **10.44** | pp still crushed by graph compile |

**Winner: llama.cpp `partial-cpu`** — keep blk 0–2 expert tensors on GPU, send blk 3–47 to CPU.

**Active parameter reality check:**

| Model | Total params | Active/tok | tg (best) |
|-------|-------------|------------|-----------|
| Qwen3-Coder-Next | 80B | **3B** | 39.5 t/s |
| Qwen3.5-122B-A10B | 122B | **10B** | 9.8 t/s |

The 122B model's low tg is not a tuning failure — it activates ~3× more parameters
per token than Qwen3-Coder-Next. That is simply more compute per decode step, and
no `-ot` strategy or backend flag changes it. If tg matters most, Qwen3-Coder-Next
is the practical choice on this hardware. Run 122B when output quality or the larger
thinking budget justifies the slower pace.

**To reproduce best result:**

```sh
STRATEGY=partial-cpu ./bench-models/bench-llama-cpp-qwen3-5-122b-a10b-strategies.sh
```

Or inline:

```sh
OVERRIDE_TENSOR="blk\.(3|[4-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU" \
  ./bench-models/bench-llama-cpp-qwen3-5-122b-a10b.sh
```

---

### gpt-oss-120b mxfp4 — RTX 4070 12 GB / Intel i5-12600K / 64 GB DDR5

**Architecture:** 36 blocks, 128 experts per MoE layer, 4 active/token, 60 GB on disk (mxfp4 quant).

**Critical constraint — 64 GB RAM ceiling.**
The model is 60 GB on disk. With all experts on CPU (`N_CPU_MOE=36`) the full
weight set lands in RAM and will OOM/hard-crash a 64 GB system. The fix is to
use `-ngl 37` with a `partial-cpu` `-ot` pattern, which keeps blk 0–4 fully in
VRAM (~10.5 GB used) and spills blk 5–36 experts to CPU (~50 GB RAM). This was
derived from `llama-fit-params` and confirmed stable.

**No shared experts.** Unlike Qwen3.5-122B, this model uses pure `_exps` tensors
only — no `_shexp`. The `(ch|)` group in patterns is not needed here, though
fit-params emits it anyway (harmless).

**Results (512pp + 128tg, 10 threads, FA=1, no-mmap, LLAMA_SET_ROWS=1, GGML_CUDA_GRAPH_OPT=1):**

> Historical note: `LLAMA_SET_ROWS=1` was present for these runs, but current upstream llama.cpp does not read that environment variable. Treat it as a no-op when interpreting or reproducing the result.

| Strategy | binary | ngl | pp (t/s) | tg (t/s) | RAM est. | Notes |
|----------|--------|-----|----------|----------|----------|-------|
| `N_CPU_MOE=36` | default | 99 | — | — | ~60 GB | **CRASHES 64 GB SYSTEM — do not use** |
| **`partial-cpu` blk 5+ on CPU** | **default** | **37** | **427.92** | **23.36** | **~50 GB** | **🏆 winner** |
| fit-params auto (`FIT_CTX=32768`) | default | 37 | 421.34 | 22.93 | ~50 GB | same placement, within noise |
| `up-down-cpu` | default | 99 | OOM | — | — | mxfp4 matrices too wide for 12 GB VRAM |
| `partial-cpu` + `FORCE_CUBLAS` build | build-cublas | 37 | 383.31 | 23.32 | ~50 GB | pp worse — GGML MMQ beats cuBLAS here |
| fit-params + `FORCE_CUBLAS` build | build-cublas | 37 | 378.20 | 23.03 | ~50 GB | fit lands same ngl=37 — cuBLAS still loses |

**Winner: default build + `partial-cpu` + `-ngl 37`.**

**cuBLAS build (`GGML_CUDA_FORCE_CUBLAS=ON` + `GGML_CUDA_FORCE_DMMV=OFF`) was tested** via
a separate `build-cublas/` binary (see `maintenance/build-llama-cpp-cublas.sh`), both with
a static `-ot` and with fit-params selecting the placement independently. Both landed the
same `-ngl 37` — fit confirms there is no better VRAM placement available — and both were
slower than the default build by ~45 t/s pp with no tg gain. The default GGML MMQ kernels
have a native `mxfp4` path (`mmq-instance-mxfp4.cu`) that beats cuBLAS dispatch at this
batch size. cuBLAS may be worth retesting if more layers can be pushed to GPU (e.g. 24 GB card).

**ik_llama.cpp was also tested** (fused-moe off, same `-ot`): pp collapsed to ~98 t/s,
tg dropped to 20.3 t/s. CUDA graph compilation overhead dominates. Not competitive here.

**cuBLAS exploration is closed.** Default build is definitively the best binary for this model on this hardware.

**Static `-ot` vs `--fit` in the server — memory breakdown comparison:**

The optimized llama-swap preset (`gpt-oss-120b` in `llama-swap.yaml`) replaces
`--fit` with a static `--override-tensor` and sets `--parallel 1`. The VRAM
breakdown confirms the win:

| | VRAM model | VRAM free | Host RAM | tg observed |
|---|---|---|---|---|
| `--fit` + `--parallel 4` (original) | 9157 MiB | 540 MiB | 51358 MiB | 27.45 t/s |
| static `-ot` + `--parallel 1` (optimized) | 9697 MiB | 12 MiB | 50818 MiB | 28.46 t/s |

+540 MiB more model weight on GPU, ~500 MB less RAM used. The static placement packs the GPU
tighter because it has no fit conservatism margin — every available MiB goes to weights.
`--parallel 4` was reserving KV cache headroom for 4 concurrent slots that are never used on
a single-user homelab server; dropping to `--parallel 1` reclaims that VRAM for the model.

The 12 MiB VRAM free is tight but confirmed stable. Do not raise `--parallel` or `--ctx-size`
without checking the memory breakdown at startup.

**Active parameter reality check:**

| Model | Total | Active/tok | tg (best) |
|-------|-------|------------|-----------|
| Qwen3-Coder-Next | 80B | 3B | 39.5 t/s |
| gpt-oss-120b | 120B | **4B** | **23.4 t/s** |
| Qwen3.5-122B-A10B | 122B | 10B | 9.8 t/s |

gpt-oss sits between the two Qwen models — 4B active is faster than 10B but
slower than 3B. The mxfp4 quant is unusual (wider matrices than standard int4)
which likely explains why `up-down-cpu` OOMs despite having only 128 experts.

**To reproduce best result:**

```sh
bash bench-models/bench-llama-cpp-gpt-oss-120b.sh
```

Or inline:

```sh
OVERRIDE_TENSOR="blk\.(5|[6-9]|[0-9][0-9]|[0-9][0-9][0-9])\.ffn_(up|down|gate)_(ch|)exps=CPU" \
  N_GPU_LAYERS=37 bash bench-models/bench-llama-cpp-gpt-oss-120b.sh
```

---

## Quick Reference

```sh
# Fastest path to a working bench:
./bench-models/bench-llama-cpp-<model>.sh               # baseline
./bench-models/bench-llama-cpp-<model>-fit.sh           # auto placement (recommended first)
./bench-models/bench-ik-llama-cpp-<model>.sh            # ik_llama fused-moe

# Override any param inline:
TASKS=4096,256 THREADS=12 ./bench-models/bench-llama-cpp-<model>.sh

# KV cache quant sweep (compare f16 / q8_0 / q4_0):
CACHE_TYPE_K=f16  CACHE_TYPE_V=f16  ./bench-models/bench-llama-cpp-<model>.sh
CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 ./bench-models/bench-llama-cpp-<model>.sh
CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 ./bench-models/bench-llama-cpp-<model>.sh

# ik_llama strategy sweep:
for s in fused fused-ger fused-mqkv; do
  STRATEGY=$s ./bench-models/bench-ik-llama-cpp-<model>-strategies.sh
done
```

### Qwen3.6-35B-A3B quickstart

```sh
# 1) Build/update mainline llama.cpp
./maintenance/build-llama-cpp.sh

# 2) Download UD-Q5_K_XL + vision projector
./model_downloader/download_hf_model.py \
  --repo-id unsloth/Qwen3.6-35B-A3B-GGUF \
  --allow-patterns '*UD-Q5_K_XL*' '*mmproj-F16*' \
  --local-dir /mnt/lab/models/unsloth/Qwen3.6-35B-A3B-GGUF \
  --max-workers 2

# 3) Serve text / bench
# Serving via llama-swap model IDs: qwen3-6-35b-a3b (text), qwen3-6-35b-a3b-vision (vision)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-6-35b-a3b","messages":[{"role":"user","content":"hi"}]}'

./bench-models/bench-llama-cpp-qwen3-6-35b-a3b.sh
STRATEGY=up-down-cpu ./bench-models/bench-llama-cpp-qwen3-6-35b-a3b-strategies.sh
./bench-models/bench-llama-cpp-qwen3-6-35b-a3b-fit.sh

# 4) Vision serve (llama-swap)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-6-35b-a3b-vision",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Describe this image briefly."},
      {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/640px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}
    ]}]
  }'

# Optional direct serve helpers (outside llama-swap)
PORT=8002 ./bench-models/run-llama-cpp-qwen3-6-35b-a3b.sh
PORT=8003 ./bench-models/run-llama-cpp-qwen3-6-35b-a3b-vision.sh
```

### gpt-oss-puzzle-88B quickstart

```sh
# 1) Build puzzle-compatible llama.cpp from upstream PR merge flow
./maintenance/build-gpt-oss-puzzle-llama-cpp.sh

# 2) Download MXFP4_MOE quant
./model_downloader/download_hf_model.py \
  --repo-id SamPurkis/gpt-oss-puzzle-88B-GGUF \
  --allow-patterns '*MXFP4_MOE*' \
  --local-dir /home/kchauhan/models/SamPurkis/gpt-oss-puzzle-88B-GGUF \
  --max-workers 2

# 3) Serve / bench (llama-swap model ID: gpt-oss-puzzle-88b)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss-puzzle-88b","messages":[{"role":"user","content":"hi"}]}'
./bench-models/bench-llama-cpp-gpt-oss-puzzle-88b.sh
./bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-fit.sh
```

Notes:

- These puzzle scripts default to `vendor/llama.cpp-pr-test-21032/build/bin/*`.
- Strategy sweeps are available via:

```sh
STRATEGY=partial-cpu ./bench-models/bench-llama-cpp-gpt-oss-puzzle-88b-strategies.sh
```

Puzzle benchmark notes (RTX 4070, PR #21032 build):

- Baseline (`N_CPU_MOE=48`, `ngl=99`): `pp ~432.8`, `tg ~20.7`
- Strategy `all-cpu-moe`: `pp ~431.5`, `tg ~23.9`
- Best observed: fit/fit-shaped partial split at `ngl=37`:
  - fit auto: `pp ~549.5`, `tg ~27.0`
  - partial-cpu (semicolons in `OVERRIDE_TENSOR`): `pp ~548.7`, `tg ~27.0`

Like gpt-oss-120b, this model benefits from lowering `ngl` and using explicit
expert offload patterns rather than high-`ngl` defaults.

### Gemma-4-26B-A4B quickstart

```sh
# 1) Build/update mainline llama.cpp
./maintenance/build-llama-cpp.sh

# 2) Download Gemma 4 26B-A4B UD-Q5_K_XL + mmproj
./model_downloader/download_hf_model.py \
  --repo-id unsloth/gemma-4-26B-A4B-it-GGUF \
  --allow-patterns '*gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf*' '*mmproj-BF16.gguf*' \
  --local-dir /home/kchauhan/models/unsloth/gemma-4-26B-A4B-it-GGUF \
  --max-workers 2

# 3) Serve / bench
# Serving is handled by llama-swap (model IDs: gemma-4-26b-a4b, gemma-4-26b-a4b-vision)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-26b-a4b-vision","messages":[{"role":"user","content":"hi"}]}'
./bench-models/bench-llama-cpp-gemma-4-26b-a4b.sh
./bench-models/bench-llama-cpp-gemma-4-26b-a4b-fit.sh
```

Notes:

- The llama-swap `gemma-4-26b-a4b` entry defaults to text, 131k ctx.
- The `gemma-4-26b-a4b-vision` entry wires `mmproj-BF16.gguf`, 128k ctx.
- `gemma-4-26b-a4b-vision` is the llama-swap preload default — it's warm at
  service startup. See `docs/llama-swap-runbook.md`.
- Gemma defaults are `temp=1.0`, `top-p=0.95`, `top-k=64`.

### Per-model winner cheatsheet

| Model | Best bench config | pp (t/s) | tg (t/s) | Llama-swap model ID |
|-------|------------------|----------|----------|---------------------|
| **Qwen3.6-35B-A3B** | fit ngl=41, 64k ctx, q8_0 KV, fit-shaped `-ot` | **970.8** | **52.3** | `qwen3-6-35b-a3b` |
| **Qwen3.6-35B-A3B (vision)** | 64k ctx, q8_0 KV, `FIT_TARGET=2048`, batch 256 | _(serve profile)_ | _(serve profile)_ | `qwen3-6-35b-a3b-vision` |
| **Qwen3-Coder-Next** | fit ngl=49, 64k ctx, q8_0 KV, 512 MiB margin | **502** | **~39–40** | `qwen3-coder-next` |
| **gpt-oss-120b** | static -ot ngl=37, q8_0 KV | 428 | 23.4 | `gpt-oss-120b` |
| **Qwen3.5-122B** | partial-cpu blk 3+, f16 KV | 284 | 9.8 | `qwen3-5-122b-thinking-coding` |
| **Gemma-4-26B-A4B** | fit, 32k ctx, q8_0 KV | _(run bench)_ | _(run bench)_ | `gemma-4-26b-a4b` |

### KV cache quant — universal recommendation

¹ Bench shows 40.6 t/s (N\_CPU\_MOE=40, tiny bench KV). Optimized script uses 64k ctx + q8\_0 KV — server-realistic tg is ~39–40 t/s (KV footprint only ~2 GB at 64k, minimal VRAM pressure).

| KV type | pp delta | tg delta | VRAM saved (131k ctx) | Verdict |
|---------|----------|----------|-----------------------|---------|
| `f16` | baseline | baseline | — | baseline only |
| **`q8_0`** | **≈0** | **+3 t/s** | **~4 GB** | **use this** |
| `q4_0` | ≈0 | +3.4 t/s | ~6 GB | marginal gain, lower fidelity |

`q8_0` is a free win on every model tested: near-zero performance cost, halves KV VRAM,
measurably improves tg by freeing VRAM pressure on attention layers. Always set
`-ctk q8_0 -ctv q8_0` in run scripts.

### Common troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| CUDA OOM at startup | ctx-size too large, parallel > 1, or too many layers on GPU | Reduce `--ctx-size`, set `--parallel 1`, or lower `-ngl` |
| RAM OOM / hard crash | Too many experts on CPU (e.g. N_CPU_MOE = total blocks) | Use fit or partial-cpu -ot; never send all experts to CPU on 64 GB |
| CUDA OOM mid-prompt (cuMemCreate) | GGML VMM pool can't grow — see §9 | Increase `FIT_TARGET` to ≥1200 MiB, or set `GGML_CUDA_GRAPH_OPT=0` |
| pp collapses with ik_llama | CUDA graph compilation overhead | Pass `-fmoe 0` or switch back to llama.cpp |
| `_shexp` OOM with Qwen3.5-122B | Regex missing shared expert tensors | Use `(ch|)exps` not just `_exps` in all -ot patterns |
| Slow tg despite GPU offload | High active-param model (e.g. 122B 10B active) | This is architecture, not a tuning failure — switch to Qwen3-Coder-Next for speed |
| tg varies 27–40 t/s between boots | CPU power profile not set to performance | `echo performance \| sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` and `sudo powerprofilesctl set performance`. On CachyOS/KDE, `power-profiles-daemon` may reset EPP at login — make it persistent or run before bench. |

---

## 9. GGML CUDA Memory Pools

Understanding this section prevents mysterious mid-session OOMs that don't appear at startup.

### How GGML allocates VRAM

GGML manages GPU memory in two pool implementations:

**VMM pool** (`ggml_cuda_pool_vmm`) — default on Turing+ hardware (RTX 20 series and newer):
- Uses CUDA Virtual Memory Management (`cuMemCreate` / `cuMemMap`).
- Reserves virtual address space in **1 GiB chunks**. Physical GPU memory is committed lazily on first access.
- When the pool needs to grow, it calls `cuMemCreate(&handle, 1_GiB, ...)`. If less than 1 GiB of free physical VRAM exists, this fails with `CUDA error: out of memory` — even if the actual allocation needed was only a few MB.
- Error site: `ggml-cuda.cu:489` function `alloc`.

**Legacy pool** (`ggml_cuda_pool_leg`) — fallback on older hardware:
- Uses plain `cudaMalloc`. Allocates only what's requested. No 1 GiB growth steps. Slower for many small allocs but never fails trying to reserve 1 GiB when you only needed 20 MB.

### Why fit's headroom margin matters for server use

`llama-bench` runs at a short context (512+128 = 640 tokens). At bench time, the VMM pool commits only the physical pages needed for that tiny context. All remaining VRAM stays uncommitted/free — `nvidia-smi` shows it as free.

`llama-fit-params` measures free VRAM at this same near-zero-load state and leaves `FIT_TARGET` MiB as margin. With `FIT_TARGET=128`, only 128 MiB of VRAM is intentionally reserved.

At server runtime with a growing conversation:
1. The KV cache fills with real tokens — new physical pages are committed lazily as the KV depth increases.
2. Scratch buffers for compute graphs may grow when a new context depth is hit for the first time (especially with `GGML_CUDA_GRAPH_OPT=1`, which recaptures CUDA graphs at new depths).
3. When any of these triggers a VMM pool growth attempt → `cuMemCreate(1 GiB)` → fails if < 1 GiB is physically free.

### Observed VRAM at runtime (RTX 4070, Q3CN, FIT_TARGET=512)

```
nvidia-smi at idle (server loaded, no active context):
  llama-server: 11182 MiB / 12282 MiB total (nvidia-smi)
  CUDA usable:  11875 MiB → free: 11875 - 11182 = 693 MiB physical
```

The 693 MiB free at idle is more than the 512 MiB FIT_TARGET because at idle the KV cache virtual space is reserved but physical pages haven't been committed yet. As context fills toward 64k, physical free shrinks toward ~500 MiB. The 693→500 MiB delta (~190 MiB) matches the KV cache for the first large prompt (~35k tokens × 2.2 KB/tok × 12 GQA layers ≈ ~140 MiB).

### Observed failure pattern

| Event | Tokens | Physical free | Status |
|-------|--------|--------------|--------|
| Server start | 0 | ~693 MiB | ✅ loads fine |
| First long prompt | ~35k | ~550 MiB | ✅ KV lazy commit, pool doesn't need to grow |
| Second longer prompt | ~44k | ~500 MiB | ❌ (old 128 MiB config) pool tries to grow → `cuMemCreate(1 GiB)` → OOM |
| Second longer prompt | ~44k | ~500 MiB | ✅ (current 512 MiB config + graph opt off) no growth needed |

The server process stays alive but the slot is in an error state when OOM occurs.

### Fixes

**Option 1 — Disable CUDA graph opt (recommended first try):**
```sh
# In llama-swap.yaml, set model "qwen3-coder-next" env:
#   - "GGML_CUDA_GRAPH_OPT=0"
systemctl --user restart llama-swap.service
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"say hi"}]}'
```
CUDA graph re-capture at new context depths is a significant VMM consumer. Disabling it removes this pressure at the cost of a small tg regression (typically < 1 t/s).

**Option 2 — Increase FIT_TARGET above 1 GiB:**
```sh
CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 FIT_CTX=65536 FIT_TARGET=1536 \
  ./bench-models/bench-llama-cpp-qwen3-coder-next-fit.sh
```
Leaving ≥ 1200 MiB free ensures the VMM pool can always grow by one chunk. This typically costs 1–2 GPU layers (pp regression of ~14–30 t/s) but makes the server stable at any prompt depth up to ctx-size.

**Option 3 — Reduce batch size:**
```sh
--batch-size 1024 --ubatch-size 256
```
Smaller batches need smaller scratch buffers, reducing peak VMM pressure during prefill. May slightly reduce pp throughput.

### Rule of thumb

> If `FIT_TARGET < 1200 MiB`, set `GGML_CUDA_GRAPH_OPT=0`. The VMM pool grows in 1 GiB steps — any smaller headroom is a latent OOM waiting for a long enough prompt.
```
