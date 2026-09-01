# Bench Results: Hybrid CPU+GPU MoE Inference

Measured per-model results and quickstarts on the reference hardware
(RTX 4070 12 GB / Intel i5-12600K / 64 GB DDR5-5867).

Methodology, experiment sequence, script layout, and prerequisites live in
`docs/bench-runbook.md`. Troubleshooting forensics (tg variability, GGML CUDA
memory pools) live in `docs/bench-troubleshooting.md`.

To add a new model's results here, follow §7 of the runbook and
`.claude/commands/add-bench-result.md`.

---

## 1. Per-model results

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
| `GGML_CUDA_GRAPH_OPT` | 0 | Disabled — graph re-capture at new context depths triggers VMM pool growth; see `bench-troubleshooting.md` (GGML CUDA memory pools) |
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

### Gemma 4 26B QAT (UD-Q4_K_XL) + MTP drafter — RTX 4070 12 GB / Intel i5-12600K / 64 GB DDR5

**Results (128pp + 128tg, 8 threads, no-mmap + mlock, f16 KV, spec-draft-mtp):**

| Strategy | flags | pp (t/s) | tg (t/s) | Notes |
|----------|-------|----------|----------|-------|
| fit (no MTP) | `--fit on` | 1323.8 (512pp) | 45.03 (128tg) | prior run, MTP off |
| **spec-draft-mtp + drafter** | `--spec-draft-model mtp-…it.gguf --spec-type draft-mtp --spec-draft-n-max 2` | 67.1 | **69.1** | **~1.5x tg over no-MTP fit** |

**Winner:** external MTP drafter (`mtp-gemma-4-26B-A4B-it.gguf`). Placement (`-ngl 31` +
`-ot` for blk.13 gate tensors and blk.14-30 ffn exps) derived standalone via
`llama-fit-params` — `--fit` with an external draft model loops in build 571d0d5
(`Gemma4Assistant requires ctx_other` warning). Note pp dropped because this run
fit for 131072 ctx / f16 KV; tg is the comparable metric.

**To reproduce:**
```sh
./bench-models/bench-llama-gemma-26b-qat-mtp.sh
```

### Qwen3.6-35B-A3B UD-Q6_K + MTP spec-draft — RTX 4070 12 GB / Intel i5-12600K / 64 GB DDR5

**Results (128pp + 128tg, 8 threads, no-mmap + mlock, q8_0 KV, spec-draft-mtp):**

| Strategy | flags | pp (t/s) | tg (t/s) | Notes |
|----------|-------|----------|----------|-------|
| fit (no MTP) | `--fit on` | 638.5 (512pp) | 26.92 (128tg) | prior run, MTP off |
| **spec-draft-mtp baseline** | `--spec-type draft-mtp --spec-draft-p-min 0.75 --spec-draft-n-max 2` | **72.7** | **51.3** | **~1.9x tg over no-MTP fit** |

**Winner:** self-drafted MTP. Qwen3.6's MTP head is built into the model (no external
draft .gguf needed). Matches the Reddit baseline approach; the Q4_K_XL variant claim
of ~89-110 t/s remains unverified (Q4_K_XL not downloaded).

**To reproduce:**
```sh
./bench-models/bench-llama-qwen3-6-reddit-baseline-Q6K-MTP.sh
```

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

## 2. Per-model quickstarts

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

---

## 3. Per-model winner cheatsheet

| Model | Best bench config | pp (t/s) | tg (t/s) | Llama-swap model ID |
|-------|------------------|----------|----------|---------------------|
| **Qwen3.6-35B-A3B** | fit ngl=41, 64k ctx, q8_0 KV, fit-shaped `-ot` | **970.8** | **52.3** | `qwen3-6-35b-a3b` |
| **Qwen3.6-35B-A3B (vision)** | 64k ctx, q8_0 KV, `FIT_TARGET=2048`, batch 256 | _(serve profile)_ | _(serve profile)_ | `qwen3-6-35b-a3b-vision` |
| **Qwen3-Coder-Next** | fit ngl=49, 64k ctx, q8_0 KV, 512 MiB margin | **502** | **~39–40** | `qwen3-coder-next` |
| **gpt-oss-120b** | static -ot ngl=37, q8_0 KV | 428 | 23.4 | `gpt-oss-120b` |
| **Qwen3.5-122B** | partial-cpu blk 3+, f16 KV | 284 | 9.8 | `qwen3-5-122b-thinking-coding` |
| **Gemma-4-26B-A4B** | fit, 32k ctx, q8_0 KV | _(run bench)_ | _(run bench)_ | `gemma-4-26b-a4b` |

## 4. KV cache quant — universal recommendation

¹ Bench shows 40.6 t/s (N\_CPU\_MOE=40, tiny bench KV). Optimized script uses 64k ctx + q8\_0 KV — server-realistic tg is ~39–40 t/s (KV footprint only ~2 GB at 64k, minimal VRAM pressure).

| KV type | pp delta | tg delta | VRAM saved (131k ctx) | Verdict |
|---------|----------|----------|-----------------------|---------|
| `f16` | baseline | baseline | — | baseline only |
| **`q8_0`** | **≈0** | **+3 t/s** | **~4 GB** | **use this** |
| `q4_0` | ≈0 | +3.4 t/s | ~6 GB | marginal gain, lower fidelity |

`q8_0` is a free win on every model tested: near-zero performance cost, halves KV VRAM,
measurably improves tg by freeing VRAM pressure on attention layers. Always set
`-ctk q8_0 -ctv q8_0` in run scripts.
